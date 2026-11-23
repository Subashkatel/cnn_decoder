"""Prepare validation and test residuals using GPU neural inference."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from training_data.planar_488.comparison import CORRECTION_POLICIES
from training_data.planar_488.evaluation import correction_effect
from training_data.planar_488.jobs.configuration import CASES, PilotConfiguration
from training_data.planar_488.jobs.configuration import make_batch, read_training_report
from training_data.planar_488.jobs.configuration import validate_tuning_seed
from training_data.planar_488.jobs.configuration import validate_fresh_test_seed
from training_data.planar_488.jobs.runtime import configure_precision, environment_record
from training_data.planar_488.jobs.runtime import require_cuda_allocation
from training_data.planar_488.learning import load_model_weights, predict_probabilities
from training_data.planar_488.model import build_predecoder
from training_data.reports import file_hash, recorded_attempt, write_report


def prepare_case(model, configuration, device, split, case_index, shots, seed, output):
    """Measure warmed inference with transfers, then save every policy's residuals."""
    batch = make_batch(configuration, case_index, shots, seed)
    predict_probabilities(model, batch, device, configuration.batch_size)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    probabilities = predict_probabilities(model, batch, device, configuration.batch_size)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    arrays = {
        "original_detectors": batch.trajectory.detector_events,
        "original_logical": batch.trajectory.logical_flips,
    }
    policy_costs = {}
    for policy_name, threshold in CORRECTION_POLICIES.items():
        started = time.perf_counter()
        effect = correction_effect(batch, probabilities, threshold)
        arrays["detectors__" + policy_name] = arrays["original_detectors"] ^ effect.detectors
        arrays["logical__" + policy_name] = arrays["original_logical"] ^ effect.logical
        conversion_seconds = time.perf_counter() - started
        policy_costs[policy_name] = inference_seconds + conversion_seconds
    name = f"{split}_{batch.basis}_phase{batch.start_phase}"
    circuit_path = output / (name + ".stim")
    circuit_path.write_text(str(batch.circuit))
    arrays_path = output / (name + ".npz")
    np.savez_compressed(arrays_path, **arrays)
    return {
        "basis": batch.basis, "start_phase": batch.start_phase, "seed": seed,
        "shots": shots, "arrays": arrays_path.name, "circuit": circuit_path.name,
        "arrays_sha256": file_hash(arrays_path), "circuit_sha256": file_hash(circuit_path),
        "gpu_inference_seconds": inference_seconds,
        "policy_predecoder_seconds": policy_costs,
    }


def prepare_checkpoint(checkpoint, output, arguments, device):
    """All checkpoint choices use separate selection and final-test seed streams."""
    training_report = read_training_report(checkpoint)
    settings = training_report["configuration"]
    configuration = PilotConfiguration.from_record(settings)
    validate_tuning_seed(configuration, arguments.selection_seed)
    test_seed = configuration.test_seed
    if arguments.test_seed is not None:
        test_seed = arguments.test_seed
        validate_fresh_test_seed(configuration, test_seed, arguments.selection_seed)
    configure_precision(configuration.tf32)
    model = build_predecoder(configuration.distance, configuration.rounds, configuration.model_profile)
    weights = checkpoint / "best.pt"
    load_model_weights(weights, model, settings, torch.device("cpu"))
    model = model.to(device)
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "name": checkpoint.name,
        "configuration": settings,
        "checkpoint_sha256": file_hash(weights),
        "training_step": training_report["completed_steps"],
        "evaluation_test_seed": test_seed,
        "validation": [], "test": [],
    }
    splits = (
        ("validation", arguments.validation_shots, arguments.selection_seed),
        ("test", arguments.test_shots, test_seed),
    )
    for split, shots, seed_base in splits:
        for case_index in range(len(CASES)):
            case = prepare_case(
                model, configuration, device, split, case_index,
                shots, seed_base + case_index, output,
            )
            record[split].append(case)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[4000, 8000, 12000])
    parser.add_argument("--validation-shots", type=int, default=8)
    parser.add_argument("--test-shots", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=203011110000)
    parser.add_argument("--test-seed", type=int,
                        help="Use a fresh test stream, separate from training and validation")
    arguments = parser.parse_args()
    if min(arguments.validation_shots, arguments.test_shots, *arguments.steps) < 1:
        parser.error("Shot counts and checkpoint steps must be positive")
    device = require_cuda_allocation()
    torch.set_num_threads(1)
    arguments.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "preparing", "environment": environment_record(device),
        "policies": CORRECTION_POLICIES, "checkpoints": [],
        "timing_scope": "Warmed GPU inference including transfers and CPU correction conversion; excludes sampling, file I/O, and model setup",
        "test_policy": "Decode test shots only after selecting checkpoint and thresholds on validation",
    }
    report_path = arguments.output / "prepared.json"
    with recorded_attempt(report_path, report):
        for step in arguments.steps:
            name = f"step_{step:06d}"
            checkpoint = arguments.training_output / "milestones" / name
            record = prepare_checkpoint(checkpoint, arguments.output / name, arguments, device)
            if report["checkpoints"]:
                reference = report["checkpoints"][0]["configuration"]
                if record["configuration"] != reference:
                    raise ValueError("Checkpoint comparison requires identical physical and training settings")
            report["checkpoints"].append(record)
            write_report(report_path, report)
            print("Prepared GPU predictions:", name, flush=True)
        report["status"] = "prepared"


if __name__ == "__main__":
    main()
