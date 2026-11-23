"""Policy selection protects logical outcomes and includes neural overhead."""

import os
import signal
import subprocess
import sys
import unittest
from unittest.mock import Mock

import numpy as np

from training_data.planar_488.comparison import paired_changes, select_candidate
from training_data.planar_488.comparison import summarize_samples
from training_data.paths import PROJECT_ROOT
from training_data.planar_488.jobs.compare_gpu_cleanup import select_neural_candidate
from training_data.planar_488.jobs.compare_gpu_cleanup import decode_batch
from training_data.planar_488.cleanup_decoder import CleanupPredictions


def case_samples(failures, seconds, timeouts=None):
    if timeouts is None:
        timeouts = [False] * len(failures)
    samples = []
    for failed, elapsed, timed_out in zip(failures, seconds, timeouts, strict=True):
        samples.append({
            "failed": failed, "wrong_logical": failed and not timed_out,
            "unreliable": False, "timeout": timed_out, "seconds": elapsed,
        })
    return {"setup_seconds": 2.0, "samples": samples}


class ComparisonTests(unittest.TestCase):
    def test_large_gpu_batches_preserve_shot_order_with_bounded_chunks(self):
        values = np.arange(137) % 2
        detectors = values[:, None]
        expected = detectors.astype(bool)
        expected[[65, 129]] ^= True

        def decode_chunk(chunk):
            self.assertLessEqual(len(chunk), 64)
            logical = chunk.astype(bool)
            unreliable = np.zeros(len(chunk), dtype=bool)
            return CleanupPredictions(logical=logical, unreliable=unreliable)

        cleanup = Mock()
        cleanup.decode.side_effect = decode_chunk
        result = decode_batch(cleanup, detectors, expected)
        self.assertEqual(len(result["samples"]), 137)
        failures = []
        for index, sample in enumerate(result["samples"]):
            if sample["failed"]:
                failures.append(index)
        self.assertEqual(failures, [65, 129])

    def test_neural_comparison_is_available_when_deployment_abstains(self):
        baseline = {
            "preset": "gpu", "policy": "abstain", "paired_changes": {"worsened": 0},
            "summary": {"failures": 0, "pipeline_seconds": 1.0},
        }
        neural = {
            "preset": "gpu", "policy": "conservative", "paired_changes": {"worsened": 0},
            "summary": {"failures": 0, "pipeline_seconds": 2.0},
        }
        self.assertEqual(select_candidate([baseline, neural])["policy"], "abstain")
        selected = select_neural_candidate([baseline, neural], "gpu")
        self.assertEqual(selected["policy"], "conservative")

    def test_worker_shutdown_is_bounded_when_parent_handles_sigterm(self):
        script = '''
import signal
import stim
from training_data.planar_488.bounded_cleanup import BoundedCleanup

signal.signal(signal.SIGTERM, lambda signum, frame: None)
circuit = stim.Circuit("R 0\\nM 0\\nDETECTOR rec[-1]")
cleanup = BoundedCleanup(circuit, 2, "bp_lsd")
cleanup.start()
cleanup.close()
'''
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", script], cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            _, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            self.fail("Decoder worker shutdown hung under the parent's SIGTERM handler")
        self.assertEqual(process.returncode, 0, stderr)

    def test_fast_policy_cannot_hide_a_new_logical_failure(self):
        baseline = {
            "policy": "abstain", "paired_changes": {"worsened": 0},
            "summary": {"failures": 1, "pipeline_seconds": 10.0},
        }
        unsafe = {
            "policy": "aggressive", "paired_changes": {"worsened": 1},
            "summary": {"failures": 1, "pipeline_seconds": 1.0},
        }
        selected = select_candidate([baseline, unsafe])
        self.assertEqual(selected["policy"], "abstain")

    def test_summary_includes_neural_time_and_does_not_drop_timeouts(self):
        case = case_samples([False, True], [.1, 3.0], [False, True])
        summary = summarize_samples([case], overhead_seconds=.2)
        self.assertEqual(summary["shots"], 2)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["timeouts"], 1)
        self.assertAlmostEqual(summary["pipeline_seconds"], 3.3)
        self.assertIsNone(summary["uncensored_logical_error_rate"])

    def test_paired_changes_detect_swapped_failures(self):
        baseline = case_samples([True, False], [1.0, 1.0])
        candidate = case_samples([False, True], [.5, .5])
        changes = paired_changes([baseline], [candidate])
        self.assertEqual(changes, {"improved": 1, "worsened": 1})
