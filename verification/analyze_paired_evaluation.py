"""Summarize a completed paired test without hiding slow cleanup shots."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from training_data.reports import file_hash, write_report


@dataclass
class CaseRuntime:
    """Aligned baseline/residual durations in seconds and full-case CPU overhead."""

    baseline_shot_seconds: np.ndarray
    combined_shot_seconds: np.ndarray
    predecoder_seconds: float


def runtime_intervals(cases, repetitions=5000, seed=20261133000000):
    """Resample paired shots within each basis/phase, keeping measured CPU overhead fixed."""
    random = np.random.default_rng(seed)
    baseline_totals = np.zeros(repetitions)
    combined_totals = np.zeros(repetitions)
    overhead = 0.0
    for case in cases:
        shot_count = len(case.baseline_shot_seconds)
        indices = random.integers(shot_count, size=(repetitions, shot_count))
        baseline_samples = case.baseline_shot_seconds[indices]
        combined_samples = case.combined_shot_seconds[indices]
        baseline_totals += baseline_samples.sum(axis=1)
        combined_totals += combined_samples.sum(axis=1)
        overhead += case.predecoder_seconds
    cleanup_ratios = baseline_totals / combined_totals
    pipeline_totals = combined_totals + overhead
    pipeline_ratios = baseline_totals / pipeline_totals
    cleanup_interval = np.percentile(cleanup_ratios, [2.5, 97.5])
    pipeline_interval = np.percentile(pipeline_ratios, [2.5, 97.5])
    return {
        "method": "Percentile paired bootstrap, stratified by memory basis and starting phase",
        "repetitions": repetitions,
        "seed": seed,
        "cleanup_speedup_interval_95": cleanup_interval.tolist(),
        "cpu_pipeline_speedup_interval_95": pipeline_interval.tolist(),
        "scope": "Shot variation in this pilot; excludes hardware, training-seed and model-selection uncertainty",
    }


def read_case_runtime(output, case):
    """Reject incomplete or unknown timing records rather than imputing latencies."""
    name = f"test_{case['basis']}_phase{case['start_phase']}.npz"
    with np.load(output / name, allow_pickle=False) as saved:
        completed = saved["completed_shots"].item()
        if completed != case["planned_shots"]:
            raise ValueError("Runtime analysis requires a complete paired case")
        baseline = saved["baseline_shot_seconds"][:completed]
        combined = saved["combined_shot_seconds"][:completed]
        for times in (baseline, combined):
            finite = np.isfinite(times).all()
            positive = (times > 0).all()
            if not (finite and positive):
                raise ValueError("Missing, reused or invalid per-shot timings cannot support a latency comparison")
        if not np.isclose(baseline.sum(), case["baseline_cleanup_seconds"]):
            raise ValueError("Per-shot baseline times disagree with the case total")
        if not np.isclose(combined.sum(), case["combined_cleanup_seconds"]):
            raise ValueError("Per-shot residual times disagree with the case total")
        return CaseRuntime(
            baseline_shot_seconds=baseline,
            combined_shot_seconds=combined,
            predecoder_seconds=case["predecoder_seconds_all_planned_shots"],
        )


def latency_quantiles(times):
    """Describe the observed shot distribution, including its largest measured tail."""
    quantiles = np.quantile(times, [.5, .95, .99])
    maximum = times.max()
    return {
        "median_seconds": float(quantiles[0]),
        "p95_seconds": float(quantiles[1]),
        "p99_seconds": float(quantiles[2]),
        "maximum_seconds": float(maximum),
    }


def fraction_removed(original, residual):
    """A reduction fraction is undefined when no original events occurred."""
    if original == 0:
        return None
    return 1 - residual / original


def summarize_runtime(runtimes):
    """Separate cleanup speed from CPU decoding time including neural corrections."""
    baseline_cases = [case.baseline_shot_seconds for case in runtimes]
    combined_cases = [case.combined_shot_seconds for case in runtimes]
    baseline = np.concatenate(baseline_cases)
    combined = np.concatenate(combined_cases)
    overhead = sum(case.predecoder_seconds for case in runtimes)
    baseline_total = baseline.sum()
    combined_total = combined.sum()
    baseline_seconds = float(baseline_total)
    combined_seconds = float(combined_total)
    baseline_quantiles = latency_quantiles(baseline)
    combined_quantiles = latency_quantiles(combined)
    intervals = runtime_intervals(runtimes)
    return {
        "baseline_cleanup_seconds": baseline_seconds,
        "residual_cleanup_seconds": combined_seconds,
        "cpu_predecoder_seconds": overhead,
        "cpu_pipeline_seconds": combined_seconds + overhead,
        "cleanup_speedup": baseline_seconds / combined_seconds,
        "cpu_pipeline_speedup": baseline_seconds / (combined_seconds + overhead),
        "baseline_latency": baseline_quantiles,
        "residual_latency": combined_quantiles,
        "shot_sampling_uncertainty": intervals,
        "scope": "Measured CPU neural inference, correction conversion and cleanup; excludes sampling and common DEM setup",
    }


def analyze_evaluation(output):
    """Bind exact event counts, logical outcomes and runtime estimates to a saved test."""
    report_path = output / "report.json"
    report_text = report_path.read_text()
    report = json.loads(report_text)
    if report["status"] != "completed":
        raise ValueError("Finish evaluation before analyzing its performance")
    cases = report["cases"]
    original_faces = sum(case["original_face_events"] for case in cases)
    residual_faces = sum(case["residual_face_events"] for case in cases)
    original_flags = sum(case["original_flag_events"] for case in cases)
    residual_flags = sum(case["residual_flag_events"] for case in cases)
    original_events = original_faces + original_flags
    residual_events = residual_faces + residual_flags
    evaluation_hash = file_hash(report_path)
    source_path = Path(__file__)
    source_hash = file_hash(source_path)
    face_reduction = fraction_removed(original_faces, residual_faces)
    all_reduction = fraction_removed(original_events, residual_events)
    analysis = {
        "evaluation_report_sha256": evaluation_hash,
        "analysis_source_sha256": source_hash,
        "experiment": report["experiment"],
        "logical_outcomes": report["summary"],
        "test_seeds": [case["seed"] for case in cases],
        "original_face_events": original_faces,
        "residual_face_events": residual_faces,
        "face_event_reduction_fraction": face_reduction,
        "original_flag_events": original_flags,
        "residual_flag_events": residual_flags,
        "all_detector_event_reduction_fraction": all_reduction,
    }
    if report["summary"]["corrections_abstained"]:
        analysis["runtime"] = {"comparison": "Abstention reuses baseline; no independent hybrid timing"}
        return analysis
    runtimes = [read_case_runtime(output, case) for case in cases]
    analysis["runtime"] = summarize_runtime(runtimes)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    arguments = parser.parse_args()
    analysis = analyze_evaluation(arguments.evaluation_output)
    output = arguments.evaluation_output / "analysis.json"
    write_report(output, analysis)
    formatted = json.dumps(analysis, indent=2)
    print(formatted)


if __name__ == "__main__":
    main()
