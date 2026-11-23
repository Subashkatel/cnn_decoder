"""Retain measured per-shot cleanup times across interrupted evaluations."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from training_data.planar_488.evaluation import correction_effect
from training_data.planar_488.generator import generate_batch
from training_data.planar_488.jobs.evaluate import load_case_progress, save_case_progress


class EvaluationTimingTests(unittest.TestCase):
    def test_timing_prefix_survives_resume_without_scoring_the_unfinished_suffix(self):
        batch = generate_batch(
            distance=3, rounds=2, basis="Z", start_phase=0,
            probability=.001, shots=4, seed=202611310001,
        )
        effect = correction_effect(batch, batch.labels, .5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.npz"
            progress = load_case_progress(path, batch)
            self.assertTrue(np.isnan(progress.baseline_shot_seconds).all())
            self.assertTrue(np.isnan(progress.combined_shot_seconds).all())
            progress.completed_shots = 2
            progress.baseline_shot_seconds[:2] = [.5, 1.0]
            progress.combined_shot_seconds[:2] = [.25, .5]
            progress.baseline_seconds = 1.5
            progress.combined_seconds = .75
            save_case_progress(path, progress, batch, effect)
            restored = load_case_progress(path, batch)
            self.assertEqual(restored.completed_shots, 2)
            np.testing.assert_array_equal(restored.baseline_shot_seconds[:2], [.5, 1.0])
            np.testing.assert_array_equal(restored.combined_shot_seconds[:2], [.25, .5])
            self.assertTrue(np.isnan(restored.baseline_shot_seconds[2:]).all())
            self.assertTrue(np.isnan(restored.combined_shot_seconds[2:]).all())
            self.assertEqual(restored.baseline_seconds, 1.5)
            self.assertEqual(restored.combined_seconds, .75)

            # A legacy prefix keeps its totals; missing per-shot timings stay unknown.
            with np.load(path) as saved:
                legacy = {name: saved[name] for name in saved.files
                          if name not in ("baseline_shot_seconds", "combined_shot_seconds")}
            np.savez_compressed(path, **legacy)
            restored = load_case_progress(path, batch)
            self.assertEqual(restored.completed_shots, 2)
            self.assertEqual(restored.baseline_seconds, 1.5)
            self.assertEqual(restored.combined_seconds, .75)
            self.assertTrue(np.isnan(restored.baseline_shot_seconds).all())
            self.assertTrue(np.isnan(restored.combined_shot_seconds).all())


if __name__ == "__main__":
    unittest.main()
