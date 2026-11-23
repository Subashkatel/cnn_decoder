"""Paired bootstrap must preserve each shot's baseline/residual relationship."""

import unittest

import numpy as np

from verification.analyze_paired_evaluation import CaseRuntime, runtime_intervals


class RuntimeAnalysisTests(unittest.TestCase):
    def test_exact_per_shot_speedup_survives_stratified_resampling(self):
        cases = []
        for baseline_times, overhead in [([1., 2., 10.], .1), ([3., 20.], .2)]:
            baseline = np.array(baseline_times)
            combined = baseline / 2
            case = CaseRuntime(
                baseline_shot_seconds=baseline,
                combined_shot_seconds=combined,
                predecoder_seconds=overhead,
            )
            cases.append(case)
        intervals = runtime_intervals(cases, repetitions=1000, seed=91)
        np.testing.assert_allclose(intervals["cleanup_speedup_interval_95"], [2, 2])
        lower, upper = intervals["cpu_pipeline_speedup_interval_95"]
        self.assertLess(lower, upper)
        self.assertLess(upper, 2)


if __name__ == "__main__":
    unittest.main()
