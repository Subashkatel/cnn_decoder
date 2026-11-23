"""Spatial and temporal cutoffs must select the intended physical corrections."""

import unittest

import numpy as np

from training_data.planar_488.channels import TargetChannel
from training_data.planar_488.evaluation import classification_counts, correction_effect
from training_data.planar_488.generator import generate_batch
from training_data.planar_488.validation import abstains
from verification.tune_training_study import select_policy


class CorrectionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = generate_batch(
            distance=3, rounds=2, basis="Z", start_phase=0,
            probability=.01, shots=8, seed=202611340100,
        )

    def test_equal_group_cutoffs_match_the_existing_scalar_policy(self):
        random = np.random.default_rng(91)
        probabilities = random.random(self.batch.labels.shape)
        shared = correction_effect(self.batch, probabilities, .7)
        groups = {"spatial": .7, "temporal": .7}
        separated = correction_effect(self.batch, probabilities, groups)
        np.testing.assert_array_equal(separated.detectors, shared.detectors)
        np.testing.assert_array_equal(separated.logical, shared.logical)
        shared_counts = classification_counts(self.batch, probabilities, .7)
        group_counts = classification_counts(self.batch, probabilities, groups)
        self.assertEqual(group_counts, shared_counts)

    def test_group_abstention_matches_explicitly_disabling_the_named_heads(self):
        labels = self.batch.labels
        temporal_labels = labels.copy()
        temporal_labels[:, TargetChannel.Z_ERROR_DIFFERENCE] = 0
        temporal_labels[:, TargetChannel.X_ERROR_DIFFERENCE] = 0
        expected = correction_effect(self.batch, temporal_labels, .5)
        temporal = correction_effect(self.batch, labels, {"spatial": 1., "temporal": .5})
        np.testing.assert_array_equal(temporal.detectors, expected.detectors)
        self.assertFalse(temporal.logical.any())
        spatial_labels = labels.copy()
        spatial_labels[:, TargetChannel.X_TEMPORAL_RESIDUAL] = 0
        spatial_labels[:, TargetChannel.Z_TEMPORAL_RESIDUAL] = 0
        expected = correction_effect(self.batch, spatial_labels, .5)
        spatial = correction_effect(self.batch, labels, {"spatial": .5, "temporal": 1.})
        np.testing.assert_array_equal(spatial.detectors, expected.detectors)
        np.testing.assert_array_equal(spatial.logical, expected.logical)
        complete = correction_effect(self.batch, labels, .5)
        np.testing.assert_array_equal(spatial.detectors ^ temporal.detectors, complete.detectors)
        self.assertTrue(abstains({"spatial": 1., "temporal": 1.}))
        self.assertFalse(abstains({"spatial": 1., "temporal": .5}))

    def test_float32_cutoff_boundary_matches_scalar_comparison(self):
        probabilities = np.full(self.batch.labels.shape, .8, dtype=np.float32)
        shared = classification_counts(self.batch, probabilities, .8)
        separated = classification_counts(self.batch, probabilities, {"spatial": .8, "temporal": .8})
        for head in shared:
            self.assertEqual(head["true_positive"], 0)
            self.assertEqual(head["false_positive"], 0)
        self.assertEqual(separated, shared)

    def test_unknown_or_invalid_group_cutoffs_are_rejected(self):
        policies = [
            {"spatial": .5},
            {"spatial": 0., "temporal": .5},
            {"spatial": .5, "temporal": float("nan")},
        ]
        for policy in policies:
            with self.assertRaises(ValueError):
                correction_effect(self.batch, self.batch.labels, policy)

    def test_cleanup_selection_recognizes_group_abstention_as_its_fallback(self):
        helpful = {
            "threshold": {"spatial": .7, "temporal": .5},
            "capped_cleanup_seconds": 1., "logical_failures": 0,
            "timeouts": 0, "new_failures": 0, "new_timeouts": 0,
        }
        baseline = dict(helpful)
        baseline["threshold"] = {"spatial": 1., "temporal": 1.}
        baseline["capped_cleanup_seconds"] = 2.
        selected = select_policy([helpful, baseline])
        self.assertEqual(selected["threshold"], helpful["threshold"])


if __name__ == "__main__":
    unittest.main()
