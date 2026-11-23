"""Policy selection protects logical outcomes and includes neural overhead."""

import unittest

from training_data.planar_488.comparison import paired_changes, select_candidate
from training_data.planar_488.comparison import summarize_samples


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
