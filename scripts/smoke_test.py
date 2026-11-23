"""Check bundled sources, data export, a tiny optimizer update, and both decoders."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from training_data.planar_488.cleanup_decoder import build_cleanup, logical_failures
from training_data.planar_488.evaluation import correction_effect
from training_data.planar_488.generator import generate_dataset
from training_data.planar_488.jobs.configuration import PilotConfiguration, make_batch
from training_data.planar_488.jobs.runtime import require_cuda_allocation
from training_data.planar_488.learning import load_checkpoint, predict_probabilities
from training_data.planar_488.learning import save_checkpoint, train_one_step
from training_data.planar_488.model import build_predecoder
from training_data.planar_488.optimization import build_optimizer
from training_data.reports import write_report


def train_and_reload(batch, configuration, output, device):
    """Exercise the real optimizer and resume contract using a small model."""
    model = build_predecoder(configuration.distance, configuration.rounds, "inspection")
    model = model.to(device)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = build_optimizer(model, configuration)
    losses = []
    for _ in range(configuration.steps):
        loss = train_one_step(model, optimizer, batch, device)
        losses.append(loss)
    changed = []
    for initial, parameter in zip(initial_parameters, model.parameters()):
        changed.append(not torch.equal(initial, parameter))
    if not any(changed):
        raise RuntimeError("The optimizer did not update model weights")
    checkpoint = output / "smoke.pt"
    settings = asdict(configuration)
    save_checkpoint(checkpoint, model, optimizer, configuration.steps, settings)
    restored = build_predecoder(configuration.distance, configuration.rounds, "inspection")
    restored = restored.to(device)
    restored_optimizer = build_optimizer(restored, configuration)
    step = load_checkpoint(checkpoint, restored, restored_optimizer, settings, device)
    if step != configuration.steps:
        raise RuntimeError("Checkpoint did not restore the training step")
    expected = predict_probabilities(model, batch, device, configuration.batch_size)
    actual = predict_probabilities(restored, batch, device, configuration.batch_size)
    np.testing.assert_array_equal(actual, expected)
    return restored, losses


def check_cleanup(batch, model, device):
    """Test the learned residual path and the exact-target correction oracle."""
    probabilities = predict_probabilities(model, batch, device, 8)
    learned_effect = correction_effect(batch, probabilities, .99)
    oracle_effect = correction_effect(batch, batch.labels, .5)
    results = {}
    for name in ("bp_lsd", "tesseract"):
        cleanup = build_cleanup(batch.circuit, name)
        original = batch.trajectory.detector_events
        actual_logical = batch.trajectory.logical_flips
        baseline = cleanup.decode(original)
        residual = original ^ learned_effect.detectors
        expected = actual_logical ^ learned_effect.logical
        combined = cleanup.decode(residual)
        oracle_residual = original ^ oracle_effect.detectors
        oracle = cleanup.decode(oracle_residual)
        oracle_expected = actual_logical ^ oracle_effect.logical
        if logical_failures(oracle, oracle_expected).any():
            raise RuntimeError(f"{name} failed the exact-target correction oracle")
        if baseline.unreliable.any() or combined.unreliable.any():
            raise RuntimeError(f"{name} could not explain a smoke-test syndrome")
        baseline_failures = logical_failures(baseline, actual_logical)
        combined_failures = logical_failures(combined, expected)
        results[name] = {
            "shots": len(original),
            "baseline_logical_failures": int(baseline_failures.sum()),
            "learned_logical_failures": int(combined_failures.sum()),
            "oracle_logical_failures": 0,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / ".runtime/smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(73001)
    device = torch.device("cpu")
    if arguments.device == "cuda":
        device = require_cuda_allocation()
    arguments.output.mkdir(parents=True, exist_ok=True)
    configuration = PilotConfiguration(
        distance=3, rounds=3, batch_size=8, steps=2,
        validation_shots=8, test_shots=8, model_profile="inspection", optimizer="lion",
        probability=.005,
    )
    exported = generate_dataset(
        distance=3, rounds=3, basis="Z", start_phase=0,
        probability=.005, shots=8, seed=73002,
        output=arguments.output / "data",
    )
    batch = make_batch(configuration, 0, 8, 73003)
    model, losses = train_and_reload(batch, configuration, arguments.output, device)
    cleanup = check_cleanup(batch, model, device)
    report = {
        "status": "passed", "device": str(device), "training_steps": 2,
        "training_losses": losses, "exported_input_shape": exported["input_shape"],
        "checkpoint_reload": "identical predictions", "cleanup": cleanup,
        "scope": "Installation and pipeline check; no decoding improvement claim",
    }
    write_report(arguments.output / "report.json", report)
    print("PASS: data export, optimizer update, checkpoint reload, BP+LSD, and Tesseract.")
    print("This small check does not measure decoding performance.")


if __name__ == "__main__":
    main()
