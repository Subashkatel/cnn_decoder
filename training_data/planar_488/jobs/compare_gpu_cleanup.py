"""Select GPU cleanup and neural policy on validation, then decode separate test shots."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from training_data.planar_488.comparison import paired_changes, read_prepared_case
from training_data.planar_488.comparison import select_candidate
from training_data.planar_488.evaluation import wilson_interval
from training_data.planar_488.nvidia_cleanup import NvidiaCleanup, PRESETS
from training_data.reports import file_hash, recorded_attempt, write_report


def decode_batch(cleanup, detectors, expected):
    """Measure warmed batch time, including transfers and parity verification."""
    chunk_size = 64
    cleanup.decode(detectors[:chunk_size])
    cleanup.cuda.deviceSynchronize()
    logical = np.empty_like(expected, dtype=bool)
    unreliable = np.zeros(len(detectors), dtype=bool)
    started = time.perf_counter()
    for first_shot in range(0, len(detectors), chunk_size):
        last_shot = min(first_shot + chunk_size, len(detectors))
        decoded = cleanup.decode(detectors[first_shot:last_shot])
        logical[first_shot:last_shot] = decoded.logical
        unreliable[first_shot:last_shot] = decoded.unreliable
    cleanup.cuda.deviceSynchronize()
    seconds = time.perf_counter() - started
    wrong = np.any(logical != expected, axis=1)
    failed = wrong | unreliable
    samples = []
    for index in range(len(detectors)):
        samples.append({
            "failed": bool(failed[index]), "wrong_logical": bool(wrong[index]),
            "unreliable": bool(unreliable[index]),
        })
    return {"decode_seconds": seconds, "samples": samples}


def summarize(cases, overhead):
    samples = []
    seconds = 0.0
    for case in cases:
        samples.extend(case["samples"])
        seconds += case["decode_seconds"]
    shots = len(samples)
    failures = sum(sample["failed"] for sample in samples)
    wrong = sum(sample["wrong_logical"] for sample in samples)
    unreliable = sum(sample["unreliable"] for sample in samples)
    interval = wilson_interval(failures, shots)
    return {
        "shots": shots, "failures": failures, "wrong_logical": wrong,
        "unreliable": unreliable, "block_failure_rate": failures / shots,
        "failure_interval_95": interval, "cleanup_seconds": seconds,
        "predecoder_seconds": overhead, "pipeline_seconds": seconds + overhead,
        "amortized_seconds_per_shot": (seconds + overhead) / shots,
    }


def validate_prepared(prepared):
    """A shared physical cohort is required when ranking different checkpoints."""
    reference = prepared["checkpoints"][0]["configuration"]
    for checkpoint in prepared["checkpoints"]:
        if checkpoint["configuration"] != reference:
            raise ValueError("GPU comparison requires the same physical and sampling settings")


def select_neural_candidate(candidates, preset):
    """Measure the best safe neural policy even when raw cleanup is recommended."""
    neural = []
    for candidate in candidates:
        if candidate["preset"] != preset or candidate["policy"] == "abstain":
            continue
        if candidate["paired_changes"]["worsened"] == 0:
            neural.append(candidate)
    if not neural:
        return None
    return select_candidate(neural)


def select_on_validation(prepared_directory, prepared, report, report_path, presets):
    """Reuse each GPU decoder for all policies of one physical case."""
    checkpoints = prepared["checkpoints"]
    for preset in presets:
        baseline = []
        candidate_cases = {}
        overhead = {}
        for checkpoint in checkpoints:
            for policy in prepared["policies"]:
                key = (checkpoint["name"], policy)
                candidate_cases[key] = []
                overhead[key] = 0.0
        for case_index, reference in enumerate(checkpoints[0]["validation"]):
            circuit, arrays = read_prepared_case(prepared_directory, checkpoints[0], reference)
            cleanup = NvidiaCleanup(circuit, preset, batch_size=64)
            raw = decode_batch(cleanup, arrays["original_detectors"], arrays["original_logical"])
            baseline.append(raw)
            for checkpoint in checkpoints:
                case = checkpoint["validation"][case_index]
                _, candidate_arrays = read_prepared_case(prepared_directory, checkpoint, case)
                np.testing.assert_array_equal(candidate_arrays["original_detectors"], arrays["original_detectors"])
                np.testing.assert_array_equal(candidate_arrays["original_logical"], arrays["original_logical"])
                for policy in prepared["policies"]:
                    key = (checkpoint["name"], policy)
                    if policy == "abstain":
                        measured = raw
                    else:
                        measured = decode_batch(
                            cleanup, candidate_arrays["detectors__" + policy],
                            candidate_arrays["logical__" + policy],
                        )
                        overhead[key] += case["policy_predecoder_seconds"][policy]
                    candidate_cases[key].append(measured)
            del cleanup
        for (checkpoint_name, policy), cases in candidate_cases.items():
            candidate = {
                "checkpoint": checkpoint_name, "policy": policy, "preset": preset,
                "summary": summarize(cases, overhead[(checkpoint_name, policy)]),
                "paired_changes": paired_changes(baseline, cases),
            }
            report["candidates"].append(candidate)
        report["validation_baselines"][preset] = summarize(baseline, 0.0)
        write_report(report_path, report)
        print("Finished GPU validation:", preset, flush=True)
    report["selection"] = select_candidate(report["candidates"])
    selected_preset = report["selection"]["preset"]
    report["neural_selection"] = select_neural_candidate(report["candidates"], selected_preset)
    write_report(report_path, report)


def evaluate_choice(prepared_directory, prepared, selection):
    checkpoint = next(item for item in prepared["checkpoints"] if item["name"] == selection["checkpoint"])
    original_cases = []
    corrected_cases = []
    overhead = 0.0
    for case in checkpoint["test"]:
        circuit, arrays = read_prepared_case(prepared_directory, checkpoint, case)
        cleanup = NvidiaCleanup(circuit, selection["preset"], batch_size=64)
        original = decode_batch(cleanup, arrays["original_detectors"], arrays["original_logical"])
        original_cases.append(original)
        if selection["policy"] == "abstain":
            corrected_cases.append(original)
        else:
            policy = selection["policy"]
            corrected = decode_batch(cleanup, arrays["detectors__" + policy], arrays["logical__" + policy])
            corrected_cases.append(corrected)
            overhead += case["policy_predecoder_seconds"][policy]
        del cleanup
    summary = {
        "gpu_cleanup_alone": summarize(original_cases, 0.0),
        "neural_plus_gpu_cleanup": summarize(corrected_cases, overhead),
        "paired_changes": paired_changes(original_cases, corrected_cases),
        "neural_policy_abstains": selection["policy"] == "abstain",
    }
    return {
        "selection": selection, "summary": summary,
        "baseline_cases": original_cases, "combined_cases": corrected_cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--presets", choices=PRESETS, nargs="+", default=list(PRESETS))
    arguments = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit GPU cleanup through Slurm")
    import cupy

    prepared_path = arguments.prepared / "prepared.json"
    prepared_text = prepared_path.read_text()
    prepared = json.loads(prepared_text)
    if prepared["status"] != "prepared":
        raise ValueError("GPU neural preparation must complete before cleanup")
    validate_prepared(prepared)
    arguments.output.mkdir(parents=True, exist_ok=False)
    properties = cupy.cuda.runtime.getDeviceProperties(0)
    report = {
        "status": "validating", "prepared_sha256": file_hash(prepared_path),
        "job_id": os.environ["SLURM_JOB_ID"], "gpu": properties["name"].decode(),
        "decoder_version": importlib.metadata.version("cudaq-qec-cu12"),
        "candidates": [], "validation_baselines": {},
        "timing_scope": "Warmed batched GPU decoding including transfers and parity checks; neural times are the actual cached GPU inference and correction conversion times. Excludes decoder setup and simulated data generation.",
        "statistical_scope": "Checkpoint, policy, and GPU preset chosen on validation. Final test shots are separate. Small-sample results do not establish a threshold or single-shot real-time latency.",
    }
    report_path = arguments.output / "comparison.json"
    with recorded_attempt(report_path, report):
        select_on_validation(arguments.prepared, prepared, report, report_path, arguments.presets)
        selected = evaluate_choice(arguments.prepared, prepared, report["selection"])
        report["test"] = selected
        report["summary"] = selected["summary"]
        if report["neural_selection"] is not None:
            report["neural_comparison"] = evaluate_choice(
                arguments.prepared, prepared, report["neural_selection"],
            )
        report["status"] = "completed"
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
