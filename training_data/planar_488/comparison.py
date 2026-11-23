"""Shared policy selection and paired decoding metrics for the training loop."""

import time

import numpy as np
import stim

from training_data.planar_488.evaluation import wilson_interval
from training_data.planar_488.bounded_cleanup import BoundedCleanup
from training_data.reports import file_hash


CORRECTION_POLICIES = {
    "abstain": {"spatial": 1.0, "temporal": 1.0},
    "temporal_050": {"spatial": 1.0, "temporal": .5},
    "temporal_070": {"spatial": 1.0, "temporal": .7},
    "temporal_090": {"spatial": 1.0, "temporal": .9},
    "both_050": {"spatial": .5, "temporal": .5},
    "spatial_090_temporal_070": {"spatial": .9, "temporal": .7},
    "spatial_095_temporal_090": {"spatial": .95, "temporal": .9},
    "spatial_099_temporal_090": {"spatial": .99, "temporal": .9},
    "spatial_099_temporal_095": {"spatial": .99, "temporal": .95},
}


def read_prepared_case(prepared_directory, checkpoint, case):
    """Load the exact circuit and residual arrays produced by GPU inference."""
    directory = prepared_directory / checkpoint["name"]
    circuit_path = directory / case["circuit"]
    arrays_path = directory / case["arrays"]
    if file_hash(circuit_path) != case["circuit_sha256"]:
        raise ValueError("Prepared circuit changed after GPU inference")
    if file_hash(arrays_path) != case["arrays_sha256"]:
        raise ValueError("Prepared detector arrays changed after GPU inference")
    circuit_text = circuit_path.read_text()
    circuit = stim.Circuit(circuit_text)
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    return circuit, arrays


def decode_shots(circuit, detectors, expected_logical, decoder_name, timeout, budget):
    """Keep deadline misses as failures and include IPC in measured decoder time."""
    cleanup = BoundedCleanup(circuit, timeout, decoder_name)
    setup_started = time.perf_counter()
    cleanup.start()
    setup_seconds = time.perf_counter() - setup_started
    samples = []
    try:
        for syndrome, expected in zip(detectors, expected_logical):
            if budget.expired():
                raise TimeoutError("Comparison worker exhausted its job budget")
            started = time.perf_counter()
            decoded = cleanup.decode(syndrome)
            wall_seconds = time.perf_counter() - started
            timed_out = decoded["status"] == "timeout"
            wrong_logical = False
            unreliable = False
            if not timed_out:
                wrong_logical = decoded["logical"] != expected.tolist()
                unreliable = decoded["unreliable"]
            failed = timed_out or wrong_logical or unreliable
            samples.append({
                "failed": bool(failed), "wrong_logical": bool(wrong_logical),
                "timeout": timed_out, "unreliable": bool(unreliable),
                "seconds": wall_seconds,
            })
    finally:
        cleanup.close()
    return {"setup_seconds": setup_seconds, "samples": samples}


def summarize_samples(cases, overhead_seconds=0.0):
    """Report a block failure rate under the stated per-search deadline."""
    samples = []
    setup_seconds = 0.0
    for case in cases:
        samples.extend(case["samples"])
        setup_seconds += case["setup_seconds"]
    shot_count = len(samples)
    if not shot_count:
        raise ValueError("A comparison needs at least one decoded shot")
    failures = sum(sample["failed"] for sample in samples)
    timeouts = sum(sample["timeout"] for sample in samples)
    wrong_logical = sum(sample["wrong_logical"] for sample in samples)
    unreliable = sum(sample["unreliable"] for sample in samples)
    seconds = np.asarray([sample["seconds"] for sample in samples])
    cleanup_seconds = float(seconds.sum())
    pipeline_seconds = cleanup_seconds + overhead_seconds
    quantiles = np.quantile(seconds, [.5, .95, .99])
    return {
        "shots": shot_count, "failures": failures,
        "wrong_logical": wrong_logical, "timeouts": timeouts, "unreliable": unreliable,
        "deadline_failure_rate": failures / shot_count,
        "failure_interval_95": wilson_interval(failures, shot_count),
        "cleanup_seconds": cleanup_seconds,
        "predecoder_seconds": overhead_seconds,
        "pipeline_seconds": pipeline_seconds,
        "pipeline_seconds_per_shot": pipeline_seconds / shot_count,
        "cleanup_p50_seconds": float(quantiles[0]),
        "cleanup_p95_seconds": float(quantiles[1]),
        "cleanup_p99_seconds": float(quantiles[2]),
        "decoder_setup_seconds": setup_seconds,
        "uncensored_logical_error_rate": failures / shot_count if timeouts == 0 else None,
    }


def paired_changes(reference_cases, candidate_cases):
    """Compare corresponding physical shots, preserving each case's ordering."""
    improved = 0
    worsened = 0
    for reference_case, candidate_case in zip(reference_cases, candidate_cases, strict=True):
        reference = reference_case["samples"]
        candidate = candidate_case["samples"]
        for original, corrected in zip(reference, candidate, strict=True):
            improved += int(original["failed"] and not corrected["failed"])
            worsened += int(corrected["failed"] and not original["failed"])
    return {"improved": improved, "worsened": worsened}


def select_candidate(candidates):
    """Abstention wins unless a policy preserves paired validation accuracy."""
    eligible = []
    for candidate in candidates:
        if candidate["paired_changes"]["worsened"] == 0:
            eligible.append(candidate)
    if not eligible:
        raise ValueError("Validation must include an abstention reference")
    return min(
        eligible,
        key=lambda candidate: (
            candidate["summary"]["failures"],
            candidate["summary"]["pipeline_seconds"],
        ),
    )
