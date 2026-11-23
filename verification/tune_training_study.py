"""Screen correction thresholds on validation shots with bounded native cleanup.

Deadlines are reported separately from logical failures. This screening chooses
a frozen policy; uncensored held-out evaluation is still required afterward.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import socket
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from training_data.detectors import detector_columns
from training_data.planar_488.cleanup_decoder import decoder_identity
from training_data.planar_488.evaluation import correction_effect
from training_data.planar_488.jobs.configuration import CASES, PilotConfiguration
from training_data.planar_488.jobs.configuration import make_batch, read_training_report
from training_data.planar_488.jobs.runtime import require_cpu_allocation
from training_data.planar_488.learning import load_model_weights, predict_probabilities
from training_data.planar_488.model import build_predecoder
from training_data.planar_488.policy import abstains, threshold_groups, threshold_key
from training_data.reports import file_hash, write_report
from verification.bounded_cleanup import BoundedCleanup


def validate_tuning_seed(configuration, seed):
    """Training and final-test samples must never enter correction-policy selection."""
    last_seed = seed + len(CASES) - 1
    if not 0 <= seed <= last_seed < 2 ** 64:
        raise ValueError("Validation seed range must fit unsigned 64-bit integers")
    excluded = (
        (configuration.training_seed + 1, configuration.training_seed + configuration.steps),
        (configuration.calibration_seed, configuration.calibration_seed + len(CASES) - 1),
        (configuration.test_seed, configuration.test_seed + len(CASES) - 1),
    )
    for first_excluded, last_excluded in excluded:
        overlaps = seed <= last_excluded and first_excluded <= last_seed
        if overlaps:
            raise ValueError("Cleanup validation overlaps training, calibration, or final test seeds")


def score_decode(decoded, expected, timeout):
    """Keep censored searches unknown rather than claiming a wrong logical bit."""
    measurement = dict(decoded)
    measurement["timeout"] = decoded["status"] == "timeout"
    measurement["failed"] = False
    if measurement["timeout"]:
        measurement["capped_seconds"] = timeout
        return measurement
    disagrees = decoded["logical"] != expected.tolist()
    measurement["failed"] = bool(disagrees or decoded["unreliable"])
    measurement["capped_seconds"] = min(timeout, decoded["seconds"])
    return measurement


def threshold_summary(threshold, cases):
    """Reject new known failures or deadlines missed on shots baseline completed."""
    summary = {
        "threshold": threshold, "shots": 0, "logical_failures": 0,
        "timeouts": 0, "new_failures": 0, "new_timeouts": 0,
        "capped_cleanup_seconds": 0.0, "original_face_events": 0,
        "residual_face_events": 0,
    }
    key = threshold_key(threshold)
    for case in cases:
        policy = case["thresholds"][key]
        summary["original_face_events"] += case["original_face_events"]
        summary["residual_face_events"] += policy["residual_face_events"]
        for baseline, corrected in zip(case["baseline"], policy["shots"]):
            summary["shots"] += 1
            summary["logical_failures"] += int(corrected["failed"])
            summary["timeouts"] += int(corrected["timeout"])
            summary["capped_cleanup_seconds"] += corrected["capped_seconds"]
            if corrected["failed"] and not baseline["failed"]:
                summary["new_failures"] += 1
            if corrected["timeout"] and not baseline["timeout"]:
                summary["new_timeouts"] += 1
    original_events = summary["original_face_events"]
    summary["face_reduction_fraction"] = None
    if original_events:
        summary["face_reduction_fraction"] = 1 - summary["residual_face_events"] / original_events
    return summary


def select_policy(summaries):
    """Abstention remains the fallback when a correction policy fails screening."""
    baseline = next(summary for summary in summaries if abstains(summary["threshold"]))
    selected = baseline
    for summary in summaries:
        summary["eligible"] = (
            summary["new_failures"] == 0
            and summary["new_timeouts"] == 0
            and summary["logical_failures"] <= baseline["logical_failures"]
            and summary["timeouts"] <= baseline["timeouts"]
            and summary["capped_cleanup_seconds"] <= baseline["capped_cleanup_seconds"]
        )
        if not summary["eligible"]:
            continue
        ranking = (summary["logical_failures"], summary["timeouts"],
                   summary["capped_cleanup_seconds"])
        selected_ranking = (selected["logical_failures"], selected["timeouts"],
                            selected["capped_cleanup_seconds"])
        if ranking < selected_ranking:
            selected = summary
    return selected


def evaluate_validation_case(configuration, case_index, model, arguments):
    seed = arguments.seed + case_index
    batch = make_batch(configuration, case_index, arguments.shots, seed)
    probabilities = predict_probabilities(model, batch, torch.device("cpu"), 16)
    original = batch.trajectory.detector_events
    truth = batch.trajectory.logical_flips
    face_columns = detector_columns(batch.descriptors, "face")
    case = {
        "basis": batch.basis, "phase": batch.start_phase, "seed": seed,
        "original_face_events": int(original[:, face_columns].sum()),
        "baseline": [], "thresholds": {},
    }
    cleanup = BoundedCleanup(batch.circuit, arguments.timeout, arguments.cleanup)
    try:
        for syndrome, expected in zip(original, truth):
            decoded = cleanup.decode(syndrome)
            case["baseline"].append(score_decode(decoded, expected, arguments.timeout))
        for threshold in arguments.thresholds:
            effect = correction_effect(batch, probabilities, threshold)
            residual = original ^ effect.detectors
            expected_logical = truth ^ effect.logical
            policy = {"residual_face_events": int(residual[:, face_columns].sum()), "shots": []}
            for shot_index in range(arguments.shots):
                unchanged_detectors = np.array_equal(residual[shot_index], original[shot_index])
                unchanged_logical = not np.any(effect.logical[shot_index])
                if unchanged_detectors and unchanged_logical:
                    scored = dict(case["baseline"][shot_index])
                    scored["reused_baseline"] = True
                else:
                    decoded = cleanup.decode(residual[shot_index])
                    scored = score_decode(decoded, expected_logical[shot_index], arguments.timeout)
                policy["shots"].append(scored)
            key = threshold_key(threshold)
            case["thresholds"][key] = policy
            print(batch.basis, batch.start_phase, "finished threshold", threshold, flush=True)
    finally:
        cleanup.close()
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261125000000)
    parser.add_argument("--timeout", type=float, default=3)
    parser.add_argument("--cleanup", choices=("tesseract", "bp_lsd"), default="tesseract")
    threshold_arguments = parser.add_mutually_exclusive_group()
    threshold_arguments.add_argument("--thresholds", type=float, nargs="+",
                                     default=[.5, .7, .8, .9, .95, .99, 1.0])
    threshold_arguments.add_argument(
        "--threshold-pair", dest="threshold_pairs", type=float, nargs=2, action="append",
        metavar=("SPATIAL", "TEMPORAL"),
        help="Select separate cutoffs; repeat this option and include the pair 1 1 for abstention",
    )
    arguments = parser.parse_args()
    if arguments.shots < 1 or arguments.timeout <= 0:
        parser.error("Shots and timeout must be positive")
    if arguments.threshold_pairs is not None:
        arguments.thresholds = []
        for spatial, temporal in arguments.threshold_pairs:
            arguments.thresholds.append({"spatial": spatial, "temporal": temporal})
    for threshold in arguments.thresholds:
        threshold_groups(threshold)
    has_abstention = any(abstains(threshold) for threshold in arguments.thresholds)
    if not has_abstention:
        parser.error("Include threshold 1.0 or the pair 1 1 as the abstention reference")
    require_cpu_allocation()
    torch.set_num_threads(1)
    training_report = read_training_report(arguments.training_output)
    configuration = PilotConfiguration.from_record(training_report["configuration"])
    validate_tuning_seed(configuration, arguments.seed)
    model = build_predecoder(configuration.distance, configuration.rounds, configuration.model_profile)
    checkpoint = arguments.training_output / "best.pt"
    load_model_weights(checkpoint, model, training_report["configuration"], torch.device("cpu"))
    report = {
        "status": "running", "purpose": "Bounded cleanup validation, not a final latency or LER claim",
        "configuration": asdict(configuration), "checkpoint_sha256": file_hash(checkpoint),
        "training_output": str(arguments.training_output), "host": socket.gethostname(),
        "decoder": decoder_identity(arguments.cleanup), "seed": arguments.seed,
        "shots_per_case": arguments.shots,
        "timeout_seconds": arguments.timeout, "cases": [],
        "threshold_values": arguments.thresholds,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for case_index in range(len(CASES)):
        case = evaluate_validation_case(configuration, case_index, model, arguments)
        report["cases"].append(case)
        write_report(arguments.output, report)
    summaries = []
    for threshold in arguments.thresholds:
        summaries.append(threshold_summary(threshold, report["cases"]))
    selected = select_policy(summaries)
    report["threshold_summaries"] = summaries
    report["selected_threshold"] = selected["threshold"]
    report["selection"] = selected
    report["status"] = "completed"
    write_report(arguments.output, report)
    print("Selected cleanup-validation policy:", selected, flush=True)


if __name__ == "__main__":
    main()
