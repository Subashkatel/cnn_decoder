"""Select on validation, then compare GPU pre-decoding plus BP+LSD with Tesseract."""

import argparse
import json
from pathlib import Path

import numpy as np

from training_data.planar_488.cleanup_decoder import decoder_identity
from training_data.planar_488.comparison import decode_shots, paired_changes
from training_data.planar_488.comparison import select_candidate, summarize_samples
from training_data.planar_488.comparison import read_prepared_case as read_case
from training_data.planar_488.jobs.runtime import RunBudget, environment_record
from training_data.planar_488.jobs.runtime import require_cpu_allocation
from training_data.reports import file_hash, recorded_attempt, write_report


def baseline_cases(prepared, checkpoint, split, decoder, arguments, budget):
    results = []
    for case in checkpoint[split]:
        circuit, arrays = read_case(prepared, checkpoint, case)
        result = decode_shots(
            circuit, arrays["original_detectors"], arrays["original_logical"],
            decoder, arguments.timeout, budget,
        )
        results.append(result)
        print(split, decoder, case["basis"], case["start_phase"], "completed", flush=True)
    return results


def corrected_cases(prepared, checkpoint, split, policy, references, arguments, budget):
    results = []
    overhead = 0.0
    for case, reference in zip(checkpoint[split], references, strict=True):
        circuit, arrays = read_case(prepared, checkpoint, case)
        detectors = arrays["detectors__" + policy]
        expected = arrays["logical__" + policy]
        if split == "validation":
            same_detectors = np.array_equal(detectors, arrays["original_detectors"])
            same_logical = np.array_equal(expected, arrays["original_logical"])
            if same_detectors and same_logical:
                results.append(reference)
                overhead += case["policy_predecoder_seconds"][policy]
                continue
        result = decode_shots(
            circuit, detectors, expected, "bp_lsd", arguments.timeout, budget,
        )
        results.append(result)
        overhead += case["policy_predecoder_seconds"][policy]
    return results, overhead


def evaluate_validation(prepared, report, arguments, budget, report_path):
    """Choose checkpoint and policy without reading any final-test decode result."""
    checkpoints = prepared["checkpoints"]
    if "validation_baseline" not in report:
        baseline = baseline_cases(arguments.prepared, checkpoints[0], "validation", "bp_lsd", arguments, budget)
        report["validation_baseline"] = baseline
        write_report(report_path, report)
    baseline = report["validation_baseline"]
    completed = {(item["checkpoint"], item["policy"]) for item in report["candidates"]}
    for checkpoint in checkpoints:
        for policy in prepared["policies"]:
            key = (checkpoint["name"], policy)
            if key in completed:
                continue
            if policy == "abstain":
                results = baseline
                overhead = 0.0
            else:
                results, overhead = corrected_cases(
                    arguments.prepared, checkpoint, "validation", policy,
                    baseline, arguments, budget,
                )
            summary = summarize_samples(results, overhead)
            candidate = {
                "checkpoint": checkpoint["name"], "policy": policy,
                "summary": summary, "paired_changes": paired_changes(baseline, results),
            }
            report["candidates"].append(candidate)
            write_report(report_path, report)
            print(checkpoint["name"], policy, summary["failures"], "validation failures", flush=True)
    report["selection"] = select_candidate(report["candidates"])
    report["status"] = "selected_on_validation"
    write_report(report_path, report)


def evaluate_test(prepared, report, arguments, budget, report_path):
    selection = report["selection"]
    checkpoint = next(item for item in prepared["checkpoints"] if item["name"] == selection["checkpoint"])
    for decoder in ("tesseract", "bp_lsd"):
        if decoder not in report["test_baselines"]:
            results = baseline_cases(arguments.prepared, checkpoint, "test", decoder, arguments, budget)
            report["test_baselines"][decoder] = results
            write_report(report_path, report)
    baseline = report["test_baselines"]["bp_lsd"]
    if "selected_test" not in report:
        if selection["policy"] == "abstain":
            results = baseline
            overhead = 0.0
        else:
            results, overhead = corrected_cases(
                arguments.prepared, checkpoint, "test", selection["policy"],
                baseline, arguments, budget,
            )
        report["selected_test"] = {"cases": results, "predecoder_seconds": overhead}
    tesseract = summarize_samples(report["test_baselines"]["tesseract"])
    bp_lsd = summarize_samples(baseline)
    combined = summarize_samples(
        report["selected_test"]["cases"], report["selected_test"]["predecoder_seconds"],
    )
    report["summary"] = {
        "tesseract": tesseract, "bp_lsd": bp_lsd, "neural_plus_bp_lsd": combined,
        "neural_policy_abstains": selection["policy"] == "abstain",
        "hybrid_speedup_over_tesseract": tesseract["pipeline_seconds"] / combined["pipeline_seconds"],
        "bp_lsd_speedup_over_tesseract": tesseract["pipeline_seconds"] / bp_lsd["pipeline_seconds"],
        "hybrid_vs_tesseract": paired_changes(report["test_baselines"]["tesseract"], report["selected_test"]["cases"]),
        "hybrid_vs_bp_lsd": paired_changes(baseline, report["selected_test"]["cases"]),
    }
    report["status"] = "completed"
    write_report(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seconds", type=float, default=2400.0)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.timeout, arguments.seconds) <= 0:
        parser.error("Time budgets must be positive")
    require_cpu_allocation()
    prepared_path = arguments.prepared / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    if prepared["status"] != "prepared":
        raise ValueError("GPU preparation must complete before decoding")
    arguments.output.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output / "comparison.json"
    identity = {
        "prepared_sha256": file_hash(prepared_path), "timeout_seconds": arguments.timeout,
        "decoders": [decoder_identity("tesseract"), decoder_identity("bp_lsd")],
    }
    if report_path.exists():
        if not arguments.resume:
            raise ValueError("Comparison exists; use --resume or a fresh directory")
        report = json.loads(report_path.read_text())
        if report["identity"] != identity:
            raise ValueError("Resume changes prepared data, decoder settings, or deadlines")
    else:
        report = {
            "status": "validating", "identity": identity, "candidates": [], "test_baselines": {},
            "environment": environment_record(),
            "timing_scope": "One CPU cleanup worker plus measured GPU inference and transfers; decoder setup and simulated data generation are reported separately or excluded",
            "statistical_scope": "Small paired pilot; no threshold or decoding advantage is established by one run",
        }
    budget = RunBudget(arguments.seconds)
    with recorded_attempt(report_path, report), budget.handle_signals():
        try:
            evaluate_validation(prepared, report, arguments, budget, report_path)
            evaluate_test(prepared, report, arguments, budget, report_path)
        except TimeoutError:
            report["status"] = "interrupted"
            raise SystemExit(75)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
