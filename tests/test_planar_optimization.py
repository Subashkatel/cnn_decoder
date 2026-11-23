"""Training recipes preserve inference weights, strict configuration, and exact resume."""

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from training_data.planar_488.dependencies import load_nvidia_module
from training_data.planar_488.jobs.configuration import PilotConfiguration, make_batch
from training_data.planar_488.jobs.runtime import RunBudget, configure_precision
from training_data.planar_488.jobs import train as train_job
from training_data.planar_488.jobs.train import finalize_training, fit_model
from training_data.planar_488.jobs.train import initial_report, prepare_session
from training_data.planar_488.learning import load_checkpoint, load_model_weights
from training_data.planar_488.learning import save_checkpoint, train_one_step
from training_data.planar_488.model import build_predecoder
from training_data.planar_488.optimization import TrainingOptions, build_optimizer
from training_data.planar_488.optimization import learning_rate_for_step, set_learning_rate
from training_data.planar_488.optimization import training_probability, update_averaged_model
from training_data.planar_488.validation import validate_model


class PlanarOptimizationTests(unittest.TestCase):
    def test_tf32_training_is_enabled_after_the_strict_backend_check(self):
        configuration = PilotConfiguration(
            distance=3, rounds=2, steps=1, batch_size=2, validation_shots=2,
            model_profile="inspection", tf32=True,
        )

        def check_reference_precision(options, device):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            return {"scope": "CPU orchestration test"}

        try:
            with tempfile.TemporaryDirectory() as directory:
                with (
                    patch.object(train_job, "require_cuda_allocation", return_value=torch.device("cpu")),
                    patch.object(train_job, "environment_record", return_value={"scope": "CPU test"}),
                    patch.object(train_job, "verify_pilot_backend", side_effect=check_reference_precision),
                    patch("torch.cuda.reset_peak_memory_stats"),
                    patch("torch.cuda.max_memory_allocated", return_value=0),
                ):
                    self.assertTrue(train_job.run_pilot(configuration, Path(directory), None))
                report_text = (Path(directory) / "report.json").read_text()
                report = json.loads(report_text)
                self.assertTrue(report["training_precision"]["allow_tf32_cudnn"])
        finally:
            configure_precision()

    def test_schedule_matches_upstream_at_warmup_and_decay_boundaries(self):
        configuration = PilotConfiguration(
            steps=20, warmup_steps=2, learning_rate_schedule="warmup_then_decay",
        )
        model = torch.nn.Linear(1, 1)
        optimizer = build_optimizer(model, configuration)
        scheduler_options = SimpleNamespace(
            type="warmup_then_decay", warmup_steps=2,
            milestones=[.25, .5, 1.0], gamma=.7,
        )
        upstream_configuration = SimpleNamespace(lr_scheduler=scheduler_options)
        upstream = load_nvidia_module("training.optimizers")
        scheduler = upstream.get_lr_scheduler(upstream_configuration, optimizer, 20)
        for step in range(1, 22):
            expected_rate = optimizer.param_groups[0]["lr"]
            actual_rate = learning_rate_for_step(configuration, step)
            self.assertAlmostEqual(actual_rate, expected_rate)
            optimizer.step()
            scheduler.step()

    def test_old_default_checkpoints_load_but_recipe_changes_are_rejected(self):
        configuration = PilotConfiguration(target_protocol="trajectory")
        recorded = asdict(configuration)
        new_options = asdict(TrainingOptions())
        for name in new_options:
            recorded.pop(name)
        model = torch.nn.Linear(2, 1)
        optimizer = build_optimizer(model, configuration)
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            save_checkpoint(checkpoint, model, optimizer, 0, recorded)
            load_model_weights(checkpoint, model, asdict(configuration), device)
            changed_recipes = (
                replace(configuration, optimizer="lion"),
                replace(configuration, training_probability=.003),
                replace(configuration, gradient_clip=1),
                replace(configuration, tf32=True),
                replace(configuration, steps=4001),
            )
            for changed in changed_recipes:
                with self.assertRaisesRegex(ValueError, "configuration"):
                    load_model_weights(checkpoint, model, asdict(changed), device)

    def test_higher_training_noise_does_not_change_validation_noise(self):
        configuration = PilotConfiguration(
            distance=3, rounds=2, probability=0, training_probability=.03,
        )
        validation = make_batch(configuration, 0, 32, 202611270001)
        training = make_batch(
            configuration, 0, 32, 202611270002,
            probability=training_probability(configuration),
        )
        self.assertFalse(np.any(validation.trajectory.detector_events))
        self.assertTrue(np.any(training.trajectory.detector_events))

    def test_lion_ema_resume_reproduces_next_update_and_inference_uses_ema(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        configuration = PilotConfiguration(
            distance=3, rounds=2, steps=8, model_profile="inspection",
            optimizer="lion", learning_rate_schedule="warmup_then_decay",
            warmup_steps=2, ema_decay=.9, gradient_clip=1,
        )
        device = torch.device("cpu")
        batch = make_batch(configuration, 0, 2, 202611270003)
        model = build_predecoder(3, 2, "inspection")
        averaged = copy.deepcopy(model)
        averaged.requires_grad_(False)
        optimizer = build_optimizer(model, configuration)

        def update_pair(raw_model, averaged_model, step_optimizer, step):
            rate = learning_rate_for_step(configuration, step)
            set_learning_rate(step_optimizer, rate)
            train_one_step(raw_model, step_optimizer, batch, device, gradient_clip=1)
            update_averaged_model(averaged_model, raw_model, .9)

        update_pair(model, averaged, optimizer, 1)
        update_pair(model, averaged, optimizer, 2)
        saved_average = copy.deepcopy(averaged.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_checkpoint(checkpoint, model, optimizer, 2, asdict(configuration),
                            inference_model=averaged)
            update_pair(model, averaged, optimizer, 3)
            expected_raw = copy.deepcopy(model.state_dict())
            expected_average = copy.deepcopy(averaged.state_dict())
            restored = build_predecoder(3, 2, "inspection")
            restored_average = copy.deepcopy(restored)
            restored_optimizer = build_optimizer(restored, configuration)
            restored_step = load_checkpoint(
                checkpoint, restored, restored_optimizer, asdict(configuration), device,
                inference_model=restored_average,
            )
            self.assertEqual(restored_step, 2)
            update_pair(restored, restored_average, restored_optimizer, 3)
            for name, weights in restored.state_dict().items():
                torch.testing.assert_close(weights, expected_raw[name], rtol=0, atol=0)
            for name, weights in restored_average.state_dict().items():
                torch.testing.assert_close(weights, expected_average[name], rtol=0, atol=0)
            inference = build_predecoder(3, 2, "inspection")
            load_model_weights(checkpoint, inference, asdict(configuration), device)
            for name, weights in inference.state_dict().items():
                torch.testing.assert_close(weights, saved_average[name], rtol=0, atol=0)

    def test_trainer_publishes_ema_weights_that_reproduce_reported_validation_loss(self):
        configuration = PilotConfiguration(
            distance=3, rounds=2, steps=4, batch_size=2, validation_shots=2,
            model_profile="inspection", optimizer="lion", ema_decay=.0001,
            learning_rate_schedule="warmup_then_decay", warmup_steps=1,
            validation_interval=3, training_probability=.003,
        )
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = initial_report(configuration, output, None)
            session = prepare_session(configuration, device, output, report, None)
            fit_model(session, RunBudget(60))
            finalize_training(session)
            inference = build_predecoder(3, 2, "inspection")
            checkpoint = output / "best.pt"
            load_model_weights(checkpoint, inference, asdict(configuration), device)
            actual_loss, _ = validate_model(inference, session.validation_batches, device, 2)
            self.assertAlmostEqual(actual_loss, report["selected_validation_loss"])
            self.assertEqual(report["completed_steps"], 4)
            milestone = output / "milestones/step_000004/best.pt"
            self.assertTrue(milestone.is_file())


if __name__ == "__main__":
    unittest.main()
