"""A fast but harmful policy must not pass cleanup validation or leak test samples."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training_data.planar_488.cleanup_decoder import decoder_identity
from training_data.planar_488.generator import TRAINING_SCHEMA
from training_data.planar_488.jobs.configuration import PilotConfiguration
from training_data.planar_488.jobs.configuration import validate_fresh_test_seed
from training_data.planar_488.jobs.evaluate import prepare_evaluation
from training_data.planar_488.learning import save_checkpoint
from training_data.planar_488.model import build_predecoder
from training_data.planar_488.optimization import build_optimizer
from training_data.reports import file_hash, write_report
from verification.tune_training_study import score_decode, select_policy, validate_tuning_seed


def policy_summary(threshold, seconds, *, failures=0, timeouts=0, new_failures=0, new_timeouts=0):
    return {
        "threshold": threshold, "capped_cleanup_seconds": seconds,
        "logical_failures": failures, "timeouts": timeouts,
        "new_failures": new_failures, "new_timeouts": new_timeouts,
    }


class PlanarTuningTests(unittest.TestCase):
    def test_extended_test_stream_cannot_reuse_training_or_validation(self):
        configuration = PilotConfiguration()
        selection_seed = configuration.test_seed + 10000
        forbidden = [
            configuration.training_seed + 1,
            configuration.validation_seed - 1,
            configuration.calibration_seed,
            configuration.test_seed,
            selection_seed + 1,
        ]
        for seed in forbidden:
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    validate_fresh_test_seed(configuration, seed, selection_seed)
        validate_fresh_test_seed(configuration, selection_seed + 10000, selection_seed)

    def test_timeout_is_unknown_not_a_claimed_logical_failure(self):
        decoded = {"status": "timeout", "seconds_lower_bound": 3}
        scored = score_decode(decoded, np.array([False]), 3)
        self.assertTrue(scored["timeout"])
        self.assertFalse(scored["failed"])
        self.assertEqual(scored["capped_seconds"], 3)

    def test_fast_policy_with_a_new_failure_is_rejected(self):
        harmful = policy_summary(.5, .1, failures=1, new_failures=1)
        helpful = policy_summary(.9, 1)
        baseline = policy_summary(1, 2)
        selected = select_policy([harmful, helpful, baseline])
        self.assertEqual(selected["threshold"], .9)
        self.assertFalse(harmful["eligible"])

    def test_new_deadline_misses_fall_back_to_abstention(self):
        harmful = policy_summary(.5, 3, timeouts=1, new_timeouts=1)
        baseline = policy_summary(1, 1)
        selected = select_policy([harmful, baseline])
        self.assertEqual(selected["threshold"], 1)

    def test_training_and_test_seed_ranges_cannot_enter_tuning(self):
        configuration = PilotConfiguration()
        for seed in (configuration.training_seed + 1, configuration.test_seed,
                     configuration.calibration_seed):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                validate_tuning_seed(configuration, seed)
        validate_tuning_seed(configuration, 20261125000000)

    def test_final_evaluation_loads_only_a_policy_for_its_exact_checkpoint(self):
        configuration = PilotConfiguration(distance=3, rounds=2, model_profile="inspection")
        model = build_predecoder(3, 2, "inspection")
        optimizer = build_optimizer(model, configuration)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            save_checkpoint(checkpoint, model, optimizer, 1, asdict(configuration))
            training_report = {
                "status": "trained", "training_schema": TRAINING_SCHEMA,
                "configuration": asdict(configuration), "selected_threshold": .9,
            }
            write_report(root / "report.json", training_report)
            policy = {
                "status": "completed", "checkpoint_sha256": file_hash(checkpoint),
                "configuration": asdict(configuration), "selected_threshold": .5,
                "seed": 20261125000000, "shots_per_case": 8, "decoder": decoder_identity(),
            }
            policy_path = root / "policy.json"
            write_report(policy_path, policy)
            session = prepare_evaluation(root, root / "evaluation", False, policy_path)
            self.assertEqual(session.threshold, .5)
            policy["selected_threshold"] = {"spatial": .7, "temporal": .5}
            write_report(policy_path, policy)
            session = prepare_evaluation(root, root / "group_evaluation", False, policy_path)
            self.assertEqual(session.threshold, policy["selected_threshold"])
            policy["checkpoint_sha256"] = "wrong checkpoint"
            write_report(policy_path, policy)
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                prepare_evaluation(root, root / "different", False, policy_path)
            policy["checkpoint_sha256"] = file_hash(checkpoint)
            policy["seed"] = configuration.test_seed
            write_report(policy_path, policy)
            with self.assertRaisesRegex(ValueError, "final test"):
                prepare_evaluation(root, root / "different", False, policy_path)


if __name__ == "__main__":
    unittest.main()
