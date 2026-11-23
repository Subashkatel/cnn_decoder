"""Slurm-only CUDA validation, with CPU comparisons and checkpoint reload checks."""

import argparse
import copy
from pathlib import Path

import torch

from training_data.planar_488.generator import TRAINING_SCHEMA, generate_batch
from training_data.planar_488.jobs.runtime import compare_backends, configure_precision
from training_data.planar_488.jobs.runtime import environment_record, require_cuda_allocation
from training_data.planar_488.learning import batch_tensors, require_finite, train_one_step
from training_data.planar_488.model import MODEL_PROFILES, build_predecoder
from training_data.planar_488.model import masked_binary_loss
from training_data.reports import recorded_attempt, write_report


# Each case uses this seed plus twice its index for training and the next seed for validation.
SEED_BASE = 20261210
CHECK_ROUNDS = 3
CHECK_PROBABILITY = .005
CHECK_LEARNING_RATE = .003


def train_steps(model, batch, device, steps: int) -> list[float]:
    """Run the real training step briefly, without interpreting loss as decoding accuracy."""
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=CHECK_LEARNING_RATE)
    losses = []
    for _ in range(steps):
        loss = train_one_step(model, optimizer, batch, device)
        losses.append(loss)
    return losses


def validation_loss(model, tensors) -> float:
    inputs, labels, mask = tensors
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = masked_binary_loss(logits, labels, mask)
    require_finite(loss, "validation loss")
    return loss.item()


def verify_checkpoint(model, batch, tensors, device, path: Path, profile: str) -> None:
    """Reload the trained weights and require identical predictions on CUDA."""
    state = model.state_dict()
    cpu_state = {}
    for name, tensor in state.items():
        cpu_state[name] = tensor.cpu()
    torch.save(cpu_state, path)
    boundary_count = batch.inputs.shape[2]
    rounds = boundary_count - 1
    restored_model = build_predecoder(batch.distance, rounds, profile)
    restored_state = torch.load(path, map_location="cpu", weights_only=True)
    restored_model.load_state_dict(restored_state)
    restored_model = restored_model.to(device)
    restored_model.eval()
    model.eval()
    inputs = tensors[0]
    with torch.no_grad():
        expected_logits = model(inputs)
        restored_logits = restored_model(inputs)
    torch.testing.assert_close(expected_logits, restored_logits, rtol=0, atol=0)


def check_batch(distance, basis, phase, shots, seed):
    return generate_batch(
        distance=distance,
        rounds=CHECK_ROUNDS,
        basis=basis,
        start_phase=phase,
        probability=CHECK_PROBABILITY,
        shots=shots,
        seed=seed,
    )


def verify_case(distance, basis, phase, shots, steps, seed, device, output, profile) -> dict:
    """Compare backends, train briefly on CUDA, and reload the checkpoint for one case."""
    training_batch = check_batch(distance, basis, phase, shots, seed)
    validation_seed = seed + 1
    validation_batch = check_batch(distance, basis, phase, shots, validation_seed)
    torch.manual_seed(seed)
    cpu_model = build_predecoder(distance, CHECK_ROUNDS, profile)
    gpu_model = copy.deepcopy(cpu_model)
    gpu_model = gpu_model.to(device)
    comparison = compare_backends(cpu_model, gpu_model, training_batch, device)

    model_state = gpu_model.state_dict()
    initial_state = copy.deepcopy(model_state)
    losses = train_steps(gpu_model, training_batch, device, steps)
    updated = False
    updated_state = gpu_model.state_dict()
    for name, tensor in updated_state.items():
        unchanged = torch.equal(initial_state[name], tensor)
        if not unchanged:
            updated = True
    if not updated:
        raise RuntimeError("CUDA optimizer did not update model weights")
    validation_tensors = batch_tensors(validation_batch, device)
    held_batch_loss = validation_loss(gpu_model, validation_tensors)
    checkpoint_name = f"d{distance}_{basis}_phase{phase}.pt"
    checkpoint_path = output / checkpoint_name
    verify_checkpoint(
        gpu_model, training_batch, validation_tensors, device, checkpoint_path, profile,
    )
    input_shape = list(training_batch.inputs.shape)
    return {
        "distance": distance,
        "basis": basis,
        "start_phase": phase,
        "shots_per_batch": shots,
        "rounds": CHECK_ROUNDS,
        "noise_probability": CHECK_PROBABILITY,
        "training_seed": seed,
        "validation_seed": validation_seed,
        "input_shape": input_shape,
        "model_profile": profile,
        "cpu_gpu_comparison": comparison,
        "optimizer_steps": steps,
        "training_losses": losses,
        "separate_seed_batch_loss": held_batch_loss,
        "weights_updated": updated,
        "checkpoint_reload_passed": True,
        "checkpoint": checkpoint_name,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shots", type=int, default=8)
    profile_names = list(MODEL_PROFILES)
    parser.add_argument("--model-profile", choices=profile_names, default="inspection")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    smallest_size = min(arguments.steps, arguments.shots)
    if smallest_size < 1:
        parser.error("Use positive steps and shots")
    arguments.output.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output / "report.json"
    report = {
        "status": "running",
        "cases": [],
        "training_schema": TRAINING_SCHEMA,
        "scope": "CUDA correctness and training-interface checks only",
    }
    with recorded_attempt(report_path, report):
        device = require_cuda_allocation()
        configure_precision()
        report["environment"] = environment_record(device)
        torch.cuda.reset_peak_memory_stats(device)
        write_report(report_path, report)
        for distance in (3, 7):
            for basis in "XZ":
                for phase in (0, 1):
                    case_index = len(report["cases"])
                    seed_offset = 2 * case_index
                    seed = SEED_BASE + seed_offset
                    result = verify_case(
                        distance, basis, phase, arguments.shots, arguments.steps,
                        seed, device, arguments.output, arguments.model_profile,
                    )
                    report["cases"].append(result)
                    write_report(report_path, report)
                    print("Passed CUDA case", distance, basis, phase, flush=True)
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
        report["peak_cuda_memory_bytes"] = peak_memory
        report["status"] = "passed"


if __name__ == "__main__":
    main()
