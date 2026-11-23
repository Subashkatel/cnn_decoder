"""Fresh-data GPU training, kept separate from CPU cleanup-decoder evaluation."""

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch

from training_data.planar_488.generator import generate_batch
from training_data.planar_488.label_protocol import SCHEMA_BY_PROTOCOL, schema_for_protocol
from training_data.planar_488.evaluation import classification_counts, correction_effect
from training_data.planar_488.evaluation import detector_event_counts
from training_data.planar_488.jobs.configuration import BEST_CHECKPOINT_NAME, CASES
from training_data.planar_488.jobs.configuration import TRAINING_REPORT_NAME, PilotConfiguration
from training_data.planar_488.jobs.configuration import make_batch, read_training_report
from training_data.planar_488.jobs.runtime import STOPPED_EARLY_EXIT_CODE, RunBudget
from training_data.planar_488.jobs.runtime import compare_backends
from training_data.planar_488.jobs.runtime import configure_precision, environment_record
from training_data.planar_488.jobs.runtime import require_cuda_allocation
from training_data.planar_488.learning import estimate_head_priors
from training_data.planar_488.learning import load_checkpoint, probability_loss
from training_data.planar_488.learning import save_checkpoint, train_one_step
from training_data.planar_488.model import MODEL_PROFILES, build_predecoder, geometry_coverage
from training_data.planar_488.model import initialize_output_bias
from training_data.planar_488.optimization import build_optimizer, learning_rate_for_step
from training_data.planar_488.optimization import set_learning_rate, training_probability
from training_data.planar_488.optimization import update_averaged_model
from training_data.planar_488.validation import choose_threshold, validate_model
from training_data.reports import file_hash, recorded_attempt, write_report


# Seed of the tiny distance-3 batch used for the CPU/GPU check before every training job.
BACKEND_CHECK_SEED = 203011120000
# Shots per case in the training-only batches that estimate each head's prevalence.
CALIBRATION_SHOTS = 64
# Steps between progress lines that are not also validation steps.
PROGRESS_INTERVAL = 25
# Fixed-step snapshots use the same validation samples and selection rule.
MILESTONE_INTERVAL = 4000


@dataclass
class LearningSession:
    """The model, optimizer, validation set, and durable state of one training run."""

    configuration: PilotConfiguration
    device: torch.device
    output: Path
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    validation_batches: list
    report: dict
    averaged_model: torch.nn.Module | None = None

    @property
    def report_path(self) -> Path:
        return self.output / TRAINING_REPORT_NAME

    @property
    def checkpoint_configuration(self) -> dict:
        return asdict(self.configuration)

    @property
    def inference_model(self):
        """EMA is used for selection and evaluation while raw weights remain resumable."""
        if self.averaged_model is not None:
            return self.averaged_model
        return self.model


def save_session_checkpoint(session, path, step):
    save_checkpoint(
        path, session.model, session.optimizer, step, session.checkpoint_configuration,
        inference_model=session.averaged_model,
    )


def restore_session_checkpoint(session, path):
    return load_checkpoint(
        path, session.model, session.optimizer, session.checkpoint_configuration, session.device,
        inference_model=session.averaged_model,
    )


def verify_pilot_backend(configuration, device) -> dict:
    """Compare the configured architecture on CPU and GPU with a tiny batch before training."""
    distance = 3
    rounds = 2
    batch = generate_batch(
        distance=distance,
        rounds=rounds,
        basis="Z",
        start_phase=1,
        probability=.001,
        shots=2,
        seed=BACKEND_CHECK_SEED,
        target_protocol=configuration.target_protocol,
    )
    cpu_model = build_predecoder(distance, rounds, configuration.model_profile)
    gpu_model = copy.deepcopy(cpu_model)
    gpu_model = gpu_model.to(device)
    return compare_backends(cpu_model, gpu_model, batch, device)


def validation_and_priors(configuration):
    """Keep calibration and validation samples in separate, recorded seed ranges."""
    validation_batches = []
    calibration_batches = []
    case_count = len(CASES)
    for case_index in range(case_count):
        validation_seed = configuration.validation_seed + case_index
        validation_shots = configuration.validation_shots
        validation_batch = make_batch(configuration, case_index, validation_shots, validation_seed)
        validation_batches.append(validation_batch)
        calibration_seed = configuration.calibration_seed + case_index
        calibration_batch = make_batch(
            configuration, case_index, CALIBRATION_SHOTS, calibration_seed,
            probability=training_probability(configuration),
        )
        calibration_batches.append(calibration_batch)
    priors = estimate_head_priors(calibration_batches)
    return validation_batches, priors


def prepare_session(configuration, device, output, report, resume) -> LearningSession:
    """Check geometry and initialize output biases from training-only prevalence."""
    validation_batches, priors = validation_and_priors(configuration)
    model = build_predecoder(
        configuration.distance,
        configuration.rounds,
        configuration.model_profile,
    )
    example = validation_batches[0]
    coverage = geometry_coverage(example.code, example.layout, model)
    report["geometry_coverage"] = coverage
    if not coverage["whole_faces_covered"]:
        raise RuntimeError("The model's receptive field does not span a whole face")
    if not coverage["incident_centers_covered"]:
        raise RuntimeError("The model's receptive field does not reach every face anchor")
    initialize_output_bias(model, priors)
    model = model.to(device)
    optimizer = build_optimizer(model, configuration)
    averaged_model = None
    if configuration.ema_decay > 0:
        averaged_model = copy.deepcopy(model)
        averaged_model.requires_grad_(False)
    session = LearningSession(
        configuration=configuration,
        device=device,
        output=output,
        model=model,
        optimizer=optimizer,
        validation_batches=validation_batches,
        report=report,
        averaged_model=averaged_model,
    )
    report["completed_steps"] = 0
    if resume is not None:
        restored_step = restore_session_checkpoint(session, resume)
        report["completed_steps"] = restored_step
    report["training_head_priors"] = priors.tolist()
    constant_losses = []
    channel_priors = priors[None, :, None, None, None]
    for batch in validation_batches:
        constant_predictions = np.broadcast_to(channel_priors, batch.labels.shape)
        constant_loss = probability_loss(batch, constant_predictions)
        constant_losses.append(constant_loss)
    mean_constant_loss = np.mean(constant_losses)
    report["constant_prior_validation_loss"] = float(mean_constant_loss)
    initial_loss, _ = validate_model(
        session.inference_model, validation_batches, device, configuration.batch_size,
    )
    report.setdefault("initial_validation_loss", initial_loss)
    report.setdefault("best_validation_loss", initial_loss)
    best_path = output / BEST_CHECKPOINT_NAME
    if not best_path.exists():
        completed_steps = report["completed_steps"]
        save_session_checkpoint(session, best_path, completed_steps)
    return session


def publish_milestone(session, step: int, validation_loss: float, probabilities_by_case) -> None:
    """Save the exact current model for a separate CPU evaluation.

    Threshold selection uses validation only. This publication does not sample
    data or advance Torch RNG state, preserving the controlled training stream.
    """
    configuration = session.configuration
    threshold, candidates = choose_threshold(session.validation_batches, probabilities_by_case)
    validation_cases = []
    for case_index, batch in enumerate(session.validation_batches):
        probabilities = probabilities_by_case[case_index]
        effect = correction_effect(batch, probabilities, threshold)
        event_counts = detector_event_counts(batch, effect)
        selected_counts = classification_counts(batch, probabilities, threshold)
        half_counts = classification_counts(batch, probabilities, .5)
        validation_seed = configuration.validation_seed + case_index
        shot_count = len(batch.inputs)
        metrics = {
            "basis": batch.basis,
            "start_phase": batch.start_phase,
            "seed": validation_seed,
            "shots": shot_count,
            "heads_at_selected_threshold": selected_counts,
            "heads_at_half": half_counts,
        }
        metrics.update(event_counts)
        validation_cases.append(metrics)
    milestone = session.output / "milestones" / f"step_{step:06d}"
    milestone.mkdir(parents=True, exist_ok=True)
    checkpoint_path = milestone / BEST_CHECKPOINT_NAME
    save_session_checkpoint(session, checkpoint_path, step)
    checkpoint_hash = file_hash(checkpoint_path)
    parent_output = session.output.resolve()
    parent_output_name = str(parent_output)
    report = {
        "status": "trained",
        "training_schema": schema_for_protocol(configuration.target_protocol),
        "configuration": session.checkpoint_configuration,
        "completed_steps": step,
        "training_shots": step * configuration.batch_size,
        "checkpoint_selection": "fixed_training_step",
        "parent_training_output": parent_output_name,
        "checkpoint_sha256": checkpoint_hash,
        "selected_validation_loss": validation_loss,
        "selected_threshold": threshold,
        "validation_threshold_candidates": candidates,
        "validation_cases": validation_cases,
        "evaluation_status": "awaiting_separate_cpu_job",
    }
    report_path = milestone / TRAINING_REPORT_NAME
    write_report(report_path, report)
    milestones = session.report.setdefault("milestones", {})
    milestones[str(step)] = str(milestone)


def checkpoint_validation(
    session: LearningSession, step: int, training_loss: float, elapsed: float,
) -> None:
    """Publish losses and both best/latest full training checkpoints together."""
    configuration = session.configuration
    held_out_loss, probabilities = validate_model(
        session.inference_model, session.validation_batches, session.device, configuration.batch_size,
    )
    report = session.report
    history_entry = {
        "step": step,
        "training_loss": training_loss,
        "validation_loss": held_out_loss,
        "elapsed_training_seconds": elapsed,
        "learning_rate": session.optimizer.param_groups[0]["lr"],
    }
    report["history"].append(history_entry)
    if held_out_loss < report["best_validation_loss"]:
        report["best_validation_loss"] = held_out_loss
        report["best_step"] = step
        best_path = session.output / BEST_CHECKPOINT_NAME
        save_session_checkpoint(session, best_path, step)
    last_path = session.output / "last.pt"
    save_session_checkpoint(session, last_path, step)
    if not report.get("checkpoint_reload_checked"):
        restored_step = restore_session_checkpoint(session, last_path)
        if restored_step != step:
            raise RuntimeError("Checkpoint step did not reload")
        report["checkpoint_reload_checked"] = True
    report["completed_steps"] = step
    report["training_shots"] = step * configuration.batch_size
    if step % MILESTONE_INTERVAL == 0 or step == configuration.steps:
        publish_milestone(session, step, held_out_loss, probabilities)
    write_report(session.report_path, report)
    print(f"Step {step}: train={training_loss:.6f}, validation={held_out_loss:.6f}", flush=True)


def fit_model(session: LearningSession, budget: RunBudget) -> None:
    """Use one fresh physical batch per optimizer step, with visible early progress."""
    configuration = session.configuration
    starting_step = session.report["completed_steps"]
    training_started = time.monotonic()
    last_step = starting_step
    training_loss = None
    generation_seconds = 0.0
    optimizer_seconds = 0.0
    case_count = len(CASES)
    first_step = starting_step + 1
    stop_step = configuration.steps + 1
    for step in range(first_step, stop_step):
        if budget.expired():
            break
        case_index = (step - 1) % case_count
        seed = configuration.training_seed + step
        generation_started = time.monotonic()
        batch = make_batch(
            configuration, case_index, configuration.batch_size, seed,
            probability=training_probability(configuration),
        )
        generation_elapsed = time.monotonic() - generation_started
        generation_seconds += generation_elapsed
        if step == 1:
            sample_path = session.output / "first_training_batch.npz"
            np.savez_compressed(
                sample_path,
                trainX=batch.inputs,
                trainY=batch.labels,
                target_mask=batch.target_mask,
            )
        optimizer_started = time.monotonic()
        rate = learning_rate_for_step(configuration, step)
        set_learning_rate(session.optimizer, rate)
        training_loss = train_one_step(
            session.model, session.optimizer, batch, session.device,
            gradient_clip=configuration.gradient_clip,
        )
        if session.averaged_model is not None:
            update_averaged_model(session.averaged_model, session.model, configuration.ema_decay)
        optimizer_elapsed = time.monotonic() - optimizer_started
        optimizer_seconds += optimizer_elapsed
        last_step = step
        elapsed = time.monotonic() - training_started
        is_first_step = step == 1
        at_validation_interval = step % configuration.validation_interval == 0
        at_milestone = step % MILESTONE_INTERVAL == 0
        validate_now = is_first_step or at_validation_interval or at_milestone
        reached_step_limit = step == configuration.steps
        out_of_time = budget.expired()
        final_step = reached_step_limit or out_of_time
        if validate_now or final_step:
            checkpoint_validation(session, step, training_loss, elapsed)
        elif step % PROGRESS_INTERVAL == 0:
            print(f"Step {step}: train={training_loss:.6f}, elapsed={elapsed:.1f}s", flush=True)
    saved_step = session.report["completed_steps"]
    if last_step > saved_step:
        elapsed = time.monotonic() - training_started
        checkpoint_validation(session, last_step, training_loss, elapsed)
    if last_step == configuration.steps:
        session.report["training_stop_reason"] = "step_limit"
    else:
        session.report["training_stop_reason"] = "time_budget_or_signal"
    attempted_steps = last_step - starting_step
    generated_shots = attempted_steps * configuration.batch_size
    work_seconds = generation_seconds + optimizer_seconds
    throughput = None
    if work_seconds:
        throughput = generated_shots / work_seconds
    session.report["timing"] = {
        "fresh_shots_this_attempt": generated_shots,
        "generation_seconds": generation_seconds,
        "optimizer_seconds_including_transfers": optimizer_seconds,
        "shots_per_second_excluding_validation_and_checkpoints": throughput,
    }


def finalize_training(session: LearningSession) -> None:
    """Choose predictions on validation only; CPU evaluation consumes the saved best model."""
    model = session.inference_model
    device = session.device
    validation_batches = session.validation_batches
    best_path = session.output / BEST_CHECKPOINT_NAME
    restore_session_checkpoint(session, best_path)
    batch_size = session.configuration.batch_size
    loss, probabilities = validate_model(model, validation_batches, device, batch_size)
    threshold, candidates = choose_threshold(validation_batches, probabilities)
    report = session.report
    report["selected_validation_loss"] = loss
    report["selected_threshold"] = threshold
    report["validation_threshold_candidates"] = candidates
    report["status"] = "trained"
    report["evaluation_status"] = "awaiting_separate_cpu_job"
    constant_prior_loss = report["constant_prior_validation_loss"]
    report["loss_better_than_constant_prior"] = loss < constant_prior_loss
    write_report(session.report_path, report)


def initial_report(configuration, output, resume):
    """Start a new report, or continue a saved one only when resuming."""
    report_path = output / TRAINING_REPORT_NAME
    if report_path.exists():
        if resume is None:
            raise ValueError("Output exists; use a new directory or --resume")
        return read_training_report(output)
    return {
        "status": "preparing",
        "training_schema": schema_for_protocol(configuration.target_protocol),
        "configuration": asdict(configuration),
        "history": [],
        "scope": "GPU training only; cleanup evaluation runs in a separate CPU allocation",
    }


def run_pilot(configuration, output: Path, resume: Path | None) -> bool:
    """Train inside one GPU allocation and publish a resumable, versioned result."""
    device = require_cuda_allocation()
    output.mkdir(parents=True, exist_ok=True)
    report = initial_report(configuration, output, resume)
    report_path = output / TRAINING_REPORT_NAME
    budget = RunBudget(configuration.training_seconds)
    with recorded_attempt(report_path, report), budget.handle_signals():
        try:
            configure_precision()
            report["environment"] = environment_record(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.manual_seed(configuration.torch_seed)
            report["backend_comparison"] = verify_pilot_backend(configuration, device)
            configure_precision(tf32=configuration.tf32)
            report["training_precision"] = {
                "parameter_dtype": "float32",
                "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            }
            torch.manual_seed(configuration.torch_seed)
            session = prepare_session(configuration, device, output, report, resume)
            report["status"] = "training"
            write_report(report_path, report)
            fit_model(session, budget)
            if session.report["completed_steps"] == 0:
                report["status"] = "budget_exhausted_before_training"
                return False
            if budget.stop_requested:
                report["status"] = "interrupted_after_checkpoint"
                return False
            finalize_training(session)
            return True
        finally:
            peak_memory = torch.cuda.max_memory_allocated(device)
            report["peak_cuda_memory_bytes"] = peak_memory


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    defaults = PilotConfiguration()
    parser.add_argument("--distance", type=int, default=defaults.distance)
    parser.add_argument("--rounds", type=int, default=defaults.rounds)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--training-seconds", type=int, default=defaults.training_seconds)
    parser.add_argument("--validation-interval", type=int, default=defaults.validation_interval)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--validation-shots", type=int, default=defaults.validation_shots)
    parser.add_argument("--test-shots", type=int, default=defaults.test_shots)
    parser.add_argument("--training-seed", type=int, default=defaults.training_seed)
    parser.add_argument("--validation-seed", type=int, default=defaults.validation_seed)
    parser.add_argument("--calibration-seed", type=int, default=defaults.calibration_seed)
    parser.add_argument("--test-seed", type=int, default=defaults.test_seed)
    parser.add_argument("--torch-seed", type=int, default=defaults.torch_seed)
    parser.add_argument("--model-profile", choices=tuple(MODEL_PROFILES),
                        default=defaults.model_profile)
    parser.add_argument("--probability", type=float, default=defaults.probability)
    parser.add_argument("--training-probability", type=float, default=defaults.training_probability)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--optimizer", choices=("adamw", "lion"), default=defaults.optimizer)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--learning-rate-schedule", choices=("constant", "warmup_then_decay"),
                        default=defaults.learning_rate_schedule)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--ema-decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--gradient-clip", type=float, default=defaults.gradient_clip)
    parser.add_argument("--tf32", action="store_true", default=defaults.tf32)
    parser.add_argument("--target-protocol", choices=tuple(SCHEMA_BY_PROTOCOL),
                        default=defaults.target_protocol)
    arguments = parser.parse_args(argv)
    configuration = PilotConfiguration(
        distance=arguments.distance,
        rounds=arguments.rounds,
        steps=arguments.steps,
        training_seconds=arguments.training_seconds,
        validation_interval=arguments.validation_interval,
        batch_size=arguments.batch_size,
        validation_shots=arguments.validation_shots,
        test_shots=arguments.test_shots,
        training_seed=arguments.training_seed,
        validation_seed=arguments.validation_seed,
        calibration_seed=arguments.calibration_seed,
        test_seed=arguments.test_seed,
        torch_seed=arguments.torch_seed,
        model_profile=arguments.model_profile,
        probability=arguments.probability,
        training_probability=arguments.training_probability,
        learning_rate=arguments.learning_rate,
        optimizer=arguments.optimizer,
        weight_decay=arguments.weight_decay,
        learning_rate_schedule=arguments.learning_rate_schedule,
        warmup_steps=arguments.warmup_steps,
        ema_decay=arguments.ema_decay,
        gradient_clip=arguments.gradient_clip,
        tf32=arguments.tf32,
        target_protocol=arguments.target_protocol,
    )
    positive_values = (
        configuration.rounds,
        configuration.steps,
        configuration.training_seconds,
        configuration.validation_interval,
        configuration.batch_size,
        configuration.validation_shots,
        configuration.test_shots,
    )
    if min(positive_values) < 1:
        parser.error("All training sizes and time limits must be positive")
    completed = run_pilot(configuration, arguments.output, arguments.resume)
    if not completed:
        raise SystemExit(STOPPED_EARLY_EXIT_CODE)


if __name__ == "__main__":
    main()
