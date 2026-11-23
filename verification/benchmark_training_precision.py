"""Measure fresh-data training throughput and TF32 numerical error on one MIG GPU."""

import argparse
import copy
import math
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training_data.planar_488.jobs.configuration import PilotConfiguration, make_batch
from training_data.planar_488.jobs.configuration import read_training_report
from training_data.planar_488.jobs.runtime import configure_precision, environment_record
from training_data.planar_488.jobs.runtime import evaluate_gradients, require_cuda_allocation
from training_data.planar_488.learning import batch_tensors, load_model_weights, train_one_step
from training_data.planar_488.model import build_predecoder
from training_data.planar_488.optimization import build_optimizer, training_probability
from training_data.planar_488.optimization import update_averaged_model
from training_data.reports import file_hash, recorded_attempt, write_report


BENCHMARK_SEED = 20261128000000


def set_tf32(enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def numerical_comparison(base_model, configuration, device):
    """Compare full-size outputs and gradients on identical input and weights."""
    batch = make_batch(configuration, 0, 128, BENCHMARK_SEED,
                       probability=training_probability(configuration))
    tensors = batch_tensors(batch, device)
    evaluated = []
    for enabled in (False, True):
        set_tf32(enabled)
        model = copy.deepcopy(base_model).to(device)
        evaluated.append(evaluate_gradients(model, tensors))
        del model
    reference_logits, reference_loss, reference_gradients = evaluated[0]
    candidate_logits, candidate_loss, candidate_gradients = evaluated[1]
    logit_difference = reference_logits - candidate_logits
    maximum_logit_error = logit_difference.abs().max().item()
    reference_norm_squared = 0.0
    difference_norm_squared = 0.0
    for reference, candidate in zip(reference_gradients, candidate_gradients):
        difference = reference - candidate
        reference_norm_squared += reference.square().sum().item()
        difference_norm_squared += difference.square().sum().item()
    gradient_relative_error = math.sqrt(difference_norm_squared / reference_norm_squared)
    loss_difference = abs(reference_loss.item() - candidate_loss.item())
    loss_relative_error = loss_difference / reference_loss.item()
    passed = (maximum_logit_error < .02 and gradient_relative_error < .01
              and loss_relative_error < .005)
    return {
        "maximum_logit_absolute_error": maximum_logit_error,
        "gradient_l2_relative_error": gradient_relative_error,
        "loss_relative_error": loss_relative_error,
        "tolerances": {"logit_absolute": .02, "gradient_l2_relative": .01,
                       "loss_relative": .005},
        "passed": passed,
    }


def benchmark_case(base_model, configuration, device, enabled, batch_size, repetitions):
    """Include fresh sampling, transfers, optimization and EMA in each timed step."""
    set_tf32(enabled)
    model = copy.deepcopy(base_model).to(device)
    averaged = copy.deepcopy(model)
    averaged.requires_grad_(False)
    optimizer = build_optimizer(model, configuration)
    elapsed_steps = []
    generation_times = []
    warmup_steps = 5
    for index in range(warmup_steps + repetitions):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        batch = make_batch(configuration, index % 4, batch_size, BENCHMARK_SEED + index + 1,
                           probability=training_probability(configuration))
        generated = time.perf_counter()
        train_one_step(model, optimizer, batch, device, gradient_clip=configuration.gradient_clip)
        update_averaged_model(averaged, model, configuration.ema_decay)
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        if index >= warmup_steps:
            elapsed_steps.append(finished - started)
            generation_times.append(generated - started)
        if index == warmup_steps - 1:
            torch.cuda.reset_peak_memory_stats(device)
    total_seconds = sum(elapsed_steps)
    return {
        "tf32": enabled, "batch_size": batch_size, "repetitions": repetitions,
        "median_step_seconds": statistics.median(elapsed_steps),
        "mean_step_seconds": statistics.mean(elapsed_steps),
        "generation_seconds": sum(generation_times),
        "total_seconds": total_seconds,
        "fresh_shots_per_second": batch_size * repetitions / total_seconds,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(device),
        "step_seconds": elapsed_steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    arguments = parser.parse_args()
    device = require_cuda_allocation()
    configure_precision()
    training_report = read_training_report(arguments.training_output)
    configuration = PilotConfiguration.from_record(training_report["configuration"])
    model = build_predecoder(configuration.distance, configuration.rounds, configuration.model_profile)
    checkpoint = arguments.training_output / "best.pt"
    load_model_weights(checkpoint, model, training_report["configuration"], torch.device("cpu"))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "environment": environment_record(device),
              "checkpoint_sha256": file_hash(checkpoint), "cases": []}
    with recorded_attempt(arguments.output, report):
        comparison = numerical_comparison(model, configuration, device)
        report["numerical_comparison"] = comparison
        if not comparison["passed"]:
            raise RuntimeError("TF32 numerical comparison exceeded the recorded tolerances")
        for enabled, batch_size in ((False, 128), (True, 128), (True, 256), (True, 512)):
            torch.cuda.empty_cache()
            try:
                measured = benchmark_case(model, configuration, device, enabled,
                                          batch_size, arguments.repetitions)
            except torch.cuda.OutOfMemoryError:
                measured = {"tf32": enabled, "batch_size": batch_size, "status": "out_of_memory"}
            report["cases"].append(measured)
            write_report(arguments.output, report)
            print(measured, flush=True)
        report["status"] = "completed"


if __name__ == "__main__":
    main()
