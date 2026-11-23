"""Resumable CPU cleanup evaluation of a separately trained GPU checkpoint."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from training_data.planar_488.cleanup_decoder import CleanupPredictions, TesseractCleanup
from training_data.planar_488.cleanup_decoder import decoder_identity, logical_failures
from training_data.planar_488.evaluation import classification_counts, correction_effect
from training_data.planar_488.evaluation import count_true, detector_event_counts
from training_data.planar_488.evaluation import wilson_interval
from training_data.planar_488.jobs.configuration import BEST_CHECKPOINT_NAME, CASES
from training_data.planar_488.jobs.configuration import PilotConfiguration, make_batch
from training_data.planar_488.jobs.configuration import read_training_report
from training_data.planar_488.jobs.runtime import STOPPED_EARLY_EXIT_CODE, RunBudget
from training_data.planar_488.jobs.runtime import environment_record, require_cpu_allocation
from training_data.planar_488.learning import load_model_weights, predict_probabilities
from training_data.planar_488.learning import probability_loss
from training_data.planar_488.model import build_predecoder
from training_data.planar_488.optimization import normalized_configuration
from training_data.planar_488.policy import Threshold, threshold_groups
from training_data.planar_488.validation import abstains
from training_data.reports import file_hash, recorded_attempt, text_hash, write_report


# Paired shots between saved prefixes; a lost allocation repeats at most this many.
PROGRESS_SAVE_INTERVAL = 16
EVALUATION_INTERPRETATION = (
    "Descriptive pilot counts; independent trials and uncertainty are needed "
    "for an improvement claim."
)


@dataclass
class CaseProgress:
    """Completed paired shots and decoder outputs; unfinished suffixes are never scored."""

    completed_shots: int
    baseline_logical: np.ndarray
    combined_logical: np.ndarray
    baseline_unreliable: np.ndarray
    combined_unreliable: np.ndarray
    baseline_seconds: float = 0.0
    combined_seconds: float = 0.0
    baseline_shot_seconds: np.ndarray | None = None
    combined_shot_seconds: np.ndarray | None = None


def load_case_progress(path: Path, batch) -> CaseProgress:
    """Require regenerated samples to match a saved prefix before resuming."""
    shot_count, observable_count = batch.trajectory.logical_flips.shape
    baseline_shot_seconds = np.full(shot_count, np.nan)
    combined_shot_seconds = np.full(shot_count, np.nan)
    if not path.exists():
        logical_shape = (shot_count, observable_count)
        baseline_logical = np.zeros(logical_shape, dtype=bool)
        combined_logical = np.zeros(logical_shape, dtype=bool)
        baseline_unreliable = np.zeros(shot_count, dtype=bool)
        combined_unreliable = np.zeros(shot_count, dtype=bool)
        return CaseProgress(
            completed_shots=0,
            baseline_logical=baseline_logical,
            combined_logical=combined_logical,
            baseline_unreliable=baseline_unreliable,
            combined_unreliable=combined_unreliable,
            baseline_shot_seconds=baseline_shot_seconds,
            combined_shot_seconds=combined_shot_seconds,
        )
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["detectors"], batch.trajectory.detector_events)
        np.testing.assert_array_equal(arrays["observable_flips"], batch.trajectory.logical_flips)
        completed_value = arrays["completed_shots"]
        completed_shots = completed_value.item()
        if not 0 <= completed_shots <= shot_count:
            raise ValueError("Saved evaluation prefix has an invalid shot count")
        baseline_seconds_value = arrays["baseline_seconds"]
        combined_seconds_value = arrays["combined_seconds"]
        baseline_seconds = baseline_seconds_value.item()
        combined_seconds = combined_seconds_value.item()
        if "baseline_shot_seconds" in arrays:
            baseline_shot_seconds = arrays["baseline_shot_seconds"]
            combined_shot_seconds = arrays["combined_shot_seconds"]
            expected_shape = (shot_count,)
            baseline_shape_valid = baseline_shot_seconds.shape == expected_shape
            combined_shape_valid = combined_shot_seconds.shape == expected_shape
            if not (baseline_shape_valid and combined_shape_valid):
                raise ValueError("Saved per-shot timing arrays have an invalid shape")
        return CaseProgress(
            completed_shots=completed_shots,
            baseline_logical=arrays["baseline_logical"],
            combined_logical=arrays["combined_logical"],
            baseline_unreliable=arrays["baseline_unreliable"],
            combined_unreliable=arrays["combined_unreliable"],
            baseline_seconds=baseline_seconds,
            combined_seconds=combined_seconds,
            baseline_shot_seconds=baseline_shot_seconds,
            combined_shot_seconds=combined_shot_seconds,
        )


def save_case_progress(path: Path, progress: CaseProgress, batch, effect) -> None:
    """Atomically preserve both arms of every completed paired test shot."""
    temporary_path = path.with_suffix(".partial")
    residual_detectors = batch.trajectory.detector_events ^ effect.detectors
    timing_arrays = {}
    if progress.baseline_shot_seconds is not None:
        timing_arrays["baseline_shot_seconds"] = progress.baseline_shot_seconds
        timing_arrays["combined_shot_seconds"] = progress.combined_shot_seconds
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            completed_shots=progress.completed_shots,
            detectors=batch.trajectory.detector_events,
            observable_flips=batch.trajectory.logical_flips,
            residual_detectors=residual_detectors,
            predecoder_logical=effect.logical,
            baseline_logical=progress.baseline_logical,
            combined_logical=progress.combined_logical,
            baseline_unreliable=progress.baseline_unreliable,
            combined_unreliable=progress.combined_unreliable,
            baseline_seconds=progress.baseline_seconds,
            combined_seconds=progress.combined_seconds,
            **timing_arrays,
        )
    temporary_path.replace(path)


def failure_counts(progress, batch, effect):
    """Score only completed paired shots and retain decoder uncertainty as failures."""
    completed = progress.completed_shots
    baseline_logical = progress.baseline_logical[:completed]
    baseline_unreliable = progress.baseline_unreliable[:completed]
    combined_logical = progress.combined_logical[:completed]
    combined_unreliable = progress.combined_unreliable[:completed]
    baseline = CleanupPredictions(logical=baseline_logical, unreliable=baseline_unreliable)
    combined = CleanupPredictions(logical=combined_logical, unreliable=combined_unreliable)
    actual_logical = batch.trajectory.logical_flips[:completed]
    predecoder_logical = effect.logical[:completed]
    residual_logical = actual_logical ^ predecoder_logical
    baseline_failed = logical_failures(baseline, actual_logical)
    combined_failed = logical_failures(combined, residual_logical)
    baseline_count = count_true(baseline_failed)
    combined_count = count_true(combined_failed)
    baseline_succeeded = ~baseline_failed
    combined_succeeded = ~combined_failed
    improved = baseline_failed & combined_succeeded
    worsened = baseline_succeeded & combined_failed
    return baseline_count, combined_count, improved, worsened


def case_metrics(batch, probabilities, threshold, progress, effect, cleanup, seed) -> dict:
    """Report prediction quality and paired decoding outcomes on the completed prefix."""
    completed = progress.completed_shots
    baseline_count, combined_count, improved, worsened = failure_counts(progress, batch, effect)
    baseline_interval = None
    combined_interval = None
    if completed:
        baseline_interval = wilson_interval(baseline_count, completed)
        combined_interval = wilson_interval(combined_count, completed)
    selected_counts = classification_counts(batch, probabilities, threshold, completed)
    half_counts = classification_counts(batch, probabilities, .5, completed)
    improved_count = count_true(improved)
    worsened_count = count_true(worsened)
    baseline_unreliable = progress.baseline_unreliable[:completed]
    combined_unreliable = progress.combined_unreliable[:completed]
    baseline_unreliable_count = count_true(baseline_unreliable)
    combined_unreliable_count = count_true(combined_unreliable)
    loss = probability_loss(batch, probabilities)
    planned_shots = len(batch.inputs)
    status = "partial"
    if completed == planned_shots:
        status = "completed"
    reuses_baseline = abstains(threshold)
    circuit_hash = text_hash(batch.circuit)
    dem_hash = text_hash(cleanup.dem)
    metrics = {
        "basis": batch.basis,
        "start_phase": batch.start_phase,
        "seed": seed,
        "shots": completed,
        "planned_shots": planned_shots,
        "status": status,
        "prediction_loss_all_planned_shots": loss,
        "heads_at_selected_threshold": selected_counts,
        "heads_at_half": half_counts,
        "baseline_failures": baseline_count,
        "combined_failures": combined_count,
        "baseline_block_error_interval_95": baseline_interval,
        "combined_block_error_interval_95": combined_interval,
        "paired_improved_shots": improved_count,
        "paired_worsened_shots": worsened_count,
        "baseline_unreliable": baseline_unreliable_count,
        "combined_unreliable": combined_unreliable_count,
        "baseline_cleanup_seconds": progress.baseline_seconds,
        "combined_cleanup_seconds": progress.combined_seconds,
        "combined_reuses_baseline": reuses_baseline,
        "cleanup_version": cleanup.version,
        "circuit_sha256": circuit_hash,
        "dem_sha256": dem_hash,
    }
    event_counts = detector_event_counts(batch, effect, completed)
    metrics.update(event_counts)
    return metrics


@dataclass
class EvaluationSession:
    configuration: PilotConfiguration
    model: torch.nn.Module
    threshold: Threshold
    output: Path
    report: dict

    @property
    def report_path(self) -> Path:
        return self.output / "report.json"


def evaluate_case(session: EvaluationSession, case_index: int, budget: RunBudget) -> bool:
    """Check the budget between paired shots, saving prefixes for exact continuation."""
    configuration = session.configuration
    threshold = session.threshold
    seed = configuration.test_seed + case_index
    batch = make_batch(configuration, case_index, configuration.test_shots, seed)
    device = torch.device("cpu")
    inference_started = time.monotonic()
    probabilities = predict_probabilities(session.model, batch, device, configuration.batch_size)
    inference_seconds = time.monotonic() - inference_started
    effect = correction_effect(batch, probabilities, threshold)
    predecoder_seconds = time.monotonic() - inference_started
    original_detectors = batch.trajectory.detector_events
    residual_detectors = original_detectors ^ effect.detectors
    cleanup = TesseractCleanup(batch.circuit)
    name = f"test_{batch.basis}_phase{batch.start_phase}.npz"
    samples_path = session.output / name
    progress = load_case_progress(samples_path, batch)

    def publish_progress() -> dict:
        """Save the shot prefix before the report, so a report never scores unsaved shots."""
        save_case_progress(samples_path, progress, batch, effect)
        metrics = case_metrics(batch, probabilities, threshold, progress, effect, cleanup, seed)
        metrics["inference_device"] = "cpu"
        metrics["inference_seconds_all_planned_shots"] = inference_seconds
        metrics["predecoder_seconds_all_planned_shots"] = predecoder_seconds
        session.report["cases"][case_index] = metrics
        write_report(session.report_path, session.report)
        return metrics

    while progress.completed_shots < configuration.test_shots:
        if budget.expired():
            break
        shot_index = progress.completed_shots
        original_shot = original_detectors[shot_index]
        started = time.monotonic()
        baseline = cleanup.decode_one(original_shot)
        baseline_elapsed = time.monotonic() - started
        progress.baseline_seconds += baseline_elapsed
        progress.baseline_shot_seconds[shot_index] = baseline_elapsed
        progress.baseline_logical[shot_index] = baseline.logical
        progress.baseline_unreliable[shot_index] = baseline.unreliable
        combined_elapsed = 0.0
        if abstains(threshold):
            combined = baseline
        else:
            residual_shot = residual_detectors[shot_index]
            started = time.monotonic()
            combined = cleanup.decode_one(residual_shot)
            combined_elapsed = time.monotonic() - started
            progress.combined_seconds += combined_elapsed
        # Abstention reuses the baseline; zero records the absence of a second decode.
        progress.combined_shot_seconds[shot_index] = combined_elapsed
        progress.combined_logical[shot_index] = combined.logical
        progress.combined_unreliable[shot_index] = combined.unreliable
        progress.completed_shots += 1
        if progress.completed_shots % PROGRESS_SAVE_INTERVAL == 0:
            publish_progress()
    metrics = publish_progress()
    completed_shots = progress.completed_shots
    baseline_failures = metrics["baseline_failures"]
    combined_failures = metrics["combined_failures"]
    summary = (
        f"Evaluated {batch.basis} phase {batch.start_phase}: shots={completed_shots}, "
        f"baseline_failures={baseline_failures}, combined_failures={combined_failures}"
    )
    print(summary, flush=True)
    return completed_shots == configuration.test_shots


def evaluation_summary(cases, threshold):
    """Pool equally sized final cases without interpreting a small difference as improvement."""
    shots = 0
    baseline_failures = 0
    combined_failures = 0
    improved = 0
    worsened = 0
    for case in cases:
        shots += case["shots"]
        baseline_failures += case["baseline_failures"]
        combined_failures += case["combined_failures"]
        improved += case["paired_improved_shots"]
        worsened += case["paired_worsened_shots"]
    baseline_rate = baseline_failures / shots
    combined_rate = combined_failures / shots
    corrections_abstained = abstains(threshold)
    return {
        "test_shots": shots,
        "baseline_failures": baseline_failures,
        "combined_failures": combined_failures,
        "baseline_block_error_rate": baseline_rate,
        "combined_block_error_rate": combined_rate,
        "paired_improved_shots": improved,
        "paired_worsened_shots": worsened,
        "corrections_abstained": corrections_abstained,
        "logical_error_improvement_established": False,
        "interpretation": EVALUATION_INTERPRETATION,
    }


def prepare_evaluation(training_output: Path, output: Path, resume: bool,
                       policy_path: Path | None = None) -> EvaluationSession:
    """Load a selected model and reject incompatible evaluation resumes."""
    training_report = read_training_report(training_output)
    if training_report.get("status") != "trained":
        raise ValueError("Training must finish and select a threshold before CPU evaluation")
    saved_configuration = training_report["configuration"]
    configuration = PilotConfiguration.from_record(saved_configuration)
    device = torch.device("cpu")
    model = build_predecoder(
        configuration.distance,
        configuration.rounds,
        configuration.model_profile,
    )
    checkpoint_path = training_output / BEST_CHECKPOINT_NAME
    load_model_weights(checkpoint_path, model, saved_configuration, device)
    threshold = training_report["selected_threshold"]
    checkpoint_hash = file_hash(checkpoint_path)
    policy = None
    if policy_path is not None:
        policy = json.loads(policy_path.read_text())
        if policy.get("status") != "completed":
            raise ValueError("Cleanup-validation policy must be completed")
        if policy["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Cleanup-validation policy belongs to a different checkpoint")
        policy_configuration = normalized_configuration(policy["configuration"])
        evaluation_configuration = normalized_configuration(saved_configuration)
        if policy_configuration != evaluation_configuration:
            raise ValueError("Cleanup-validation policy has a different sampling configuration")
        threshold = policy["selected_threshold"]
        threshold_groups(threshold)
        validation_end = policy["seed"] + len(CASES) - 1
        test_end = configuration.test_seed + len(CASES) - 1
        if policy["seed"] <= test_end and configuration.test_seed <= validation_end:
            raise ValueError("Correction policy used the final test seed range")
    cleanup_identity = decoder_identity()
    if policy is not None and policy["decoder"] != cleanup_identity:
        raise ValueError("Correction policy used different cleanup decoder settings")
    experiment = {
        "training_schema": training_report["training_schema"],
        "configuration": saved_configuration,
        "checkpoint_sha256": checkpoint_hash,
        "threshold": threshold,
        "cleanup": cleanup_identity,
    }
    if policy is not None:
        experiment["cleanup_validation_policy"] = {
            "sha256": file_hash(policy_path),
            "seed": policy["seed"],
            "shots_per_case": policy["shots_per_case"],
        }
    report_path = output / "report.json"
    if report_path.exists():
        if not resume:
            raise ValueError("Evaluation output exists; use --resume or a fresh directory")
        report_text = report_path.read_text()
        report = json.loads(report_text)
        if report["experiment"] != experiment:
            raise ValueError(
                "Evaluation resume differs in decoder, checkpoint, threshold, "
                "or sampling configuration"
            )
    else:
        case_count = len(CASES)
        unscored_cases = [None] * case_count
        report = {"status": "preparing", "experiment": experiment, "cases": unscored_cases}
    output.mkdir(parents=True, exist_ok=True)
    return EvaluationSession(
        configuration=configuration,
        model=model,
        threshold=threshold,
        output=output,
        report=report,
    )


def run_evaluation(training_output: Path, output: Path, seconds: float, resume=False,
                   *, policy_path: Path | None = None) -> bool:
    """Evaluate on CPU and return whether all planned paired shots completed."""
    session = prepare_evaluation(training_output, output, resume, policy_path)
    budget = RunBudget(seconds)
    with recorded_attempt(session.report_path, session.report), budget.handle_signals():
        session.report["environment"] = environment_record()
        session.report["status"] = "evaluating"
        write_report(session.report_path, session.report)
        case_count = len(CASES)
        for case_index in range(case_count):
            previous = session.report["cases"][case_index]
            already_completed = previous is not None and previous["status"] == "completed"
            if already_completed:
                continue
            if budget.expired():
                session.report["status"] = "interrupted"
                return False
            completed = evaluate_case(session, case_index, budget)
            if not completed:
                session.report["status"] = "interrupted"
                return False
        cases = session.report["cases"]
        session.report["summary"] = evaluation_summary(cases, session.threshold)
        session.report["status"] = "completed"
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-seconds", type=int, default=1500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policy", type=Path)
    arguments = parser.parse_args(argv)
    require_cpu_allocation()
    if arguments.evaluation_seconds < 1:
        parser.error("Evaluation time must be positive")
    completed = run_evaluation(
        arguments.training_output,
        arguments.output,
        arguments.evaluation_seconds,
        arguments.resume,
        policy_path=arguments.policy,
    )
    if not completed:
        raise SystemExit(STOPPED_EARLY_EXIT_CODE)


if __name__ == "__main__":
    main()
