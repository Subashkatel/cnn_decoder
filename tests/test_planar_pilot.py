"""Contracts for applying neural heads, cleanup, and exact training continuation."""

from pathlib import Path
import json
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import stim
import torch

from training_data.detectors import detector_columns
from training_data.planar_488.channels import TargetChannel
from training_data.planar_488.cleanup_decoder import TesseractCleanup, logical_failures
from training_data.planar_488.code import build_code, build_circuit
from training_data.planar_488.evaluation import classification_counts, correction_effect
from training_data.planar_488.evaluation import wilson_interval
from training_data.planar_488.generator import generate_batch
from training_data.planar_488.generator import TRAINING_SCHEMA
from training_data.planar_488.jobs import train as train_job
from training_data.planar_488.jobs.configuration import CASES, PilotConfiguration
from training_data.planar_488.jobs.evaluate import run_evaluation
from training_data.planar_488.jobs.train import run_pilot
from training_data.planar_488.learning import load_checkpoint, save_checkpoint, train_one_step
from training_data.planar_488.model import build_predecoder, geometry_coverage
from training_data.planar_488.tensors import build_layout
from training_data.planar_488.validation import choose_threshold


DECODER_OUTPUTS = (
    "baseline_logical",
    "combined_logical",
    "baseline_unreliable",
    "combined_unreliable",
)


def error_column(instruction, detector_count: int) -> int:
    """One DEM error as a bit mask over detectors, followed by logical observables."""
    column = 0
    for target in instruction.targets_copy():
        if target.is_relative_detector_id():
            position = target.val
        elif target.is_logical_observable_id():
            position = detector_count + target.val
        else:
            continue
        position_bit = 1 << position
        column ^= position_bit
    return column


def add_to_basis(pivots: dict, column: int) -> None:
    """Reduce a column against a GF(2) basis, keeping it when it is independent."""
    while column:
        pivot = column.bit_length() - 1
        if pivot not in pivots:
            pivots[pivot] = column
            return
        column ^= pivots[pivot]


def read_report(directory: Path) -> dict:
    report_path = directory / "report.json"
    report_text = report_path.read_text()
    return json.loads(report_text)


class CorrectionApplicationTests(unittest.TestCase):
    def test_native_dem_spans_residual_detector_and_logical_patterns(self):
        code = build_code(7)
        for basis in "XZ":
            for phase in (0, 1):
                circuit = build_circuit(code, 7, basis, phase, .001)
                dem = circuit.detector_error_model(approximate_disjoint_errors=True)
                pivots = {}
                for instruction in dem.flattened():
                    if instruction.type != "error":
                        continue
                    column = error_column(instruction, dem.num_detectors)
                    add_to_basis(pivots, column)
                rank = len(pivots)
                expected_rank = dem.num_detectors + dem.num_observables
                self.assertEqual(rank, expected_rank)

    def test_oracle_cancels_faces_and_logical_but_preserves_flags(self):
        for distance in (3, 7):
            for basis in "XZ":
                for phase in (0, 1):
                    seed = 202611110000 + distance * 10 + phase
                    batch = generate_batch(
                        distance=distance,
                        rounds=3,
                        basis=basis,
                        start_phase=phase,
                        probability=.01,
                        shots=8,
                        seed=seed,
                    )
                    effect = correction_effect(batch, batch.labels, .5)
                    original = batch.trajectory.detector_events
                    residual = original ^ effect.detectors
                    face_columns = detector_columns(batch.descriptors, "face")
                    flag_columns = detector_columns(batch.descriptors, "record_constraint")
                    face_residuals = residual[:, face_columns]
                    any_face_residual = face_residuals.any()
                    self.assertFalse(any_face_residual)
                    original_flags = original[:, flag_columns]
                    residual_flags = residual[:, flag_columns]
                    np.testing.assert_array_equal(residual_flags, original_flags)
                    logical_residual = batch.trajectory.logical_flips ^ effect.logical
                    any_logical_residual = logical_residual.any()
                    self.assertFalse(any_logical_residual)

    def test_logical_operator_changes_logical_without_changing_detector_effect(self):
        batch = generate_batch(
            distance=3,
            rounds=2,
            basis="Z",
            start_phase=0,
            probability=0,
            shots=1,
            seed=202611110100,
        )
        probabilities = np.zeros_like(batch.labels)
        for row, column in batch.layout.data_sites:
            probabilities[:, TargetChannel.X_ERROR_DIFFERENCE, -1, row, column] = 1
        effect = correction_effect(batch, probabilities, .5)
        any_detector_effect = effect.detectors.any()
        every_logical_flipped = effect.logical.all()
        self.assertFalse(any_detector_effect)
        self.assertTrue(every_logical_flipped)

    def test_invalid_targets_and_abstention_cannot_modify_the_stream(self):
        batch = generate_batch(
            distance=3,
            rounds=2,
            basis="X",
            start_phase=1,
            probability=.001,
            shots=2,
            seed=202611110101,
        )
        padding = ~batch.target_mask
        probabilities = np.zeros_like(batch.labels)
        probabilities[:, padding] = 1
        effect = correction_effect(batch, probabilities, .5)
        any_detector_effect = effect.detectors.any()
        any_logical_effect = effect.logical.any()
        self.assertFalse(any_detector_effect)
        self.assertFalse(any_logical_effect)
        probabilities[:] = 1
        abstention = correction_effect(batch, probabilities, 1.0)
        any_abstention_detector = abstention.detectors.any()
        any_abstention_logical = abstention.logical.any()
        self.assertFalse(any_abstention_detector)
        self.assertFalse(any_abstention_logical)

    def test_threshold_selection_keeps_oracle_and_rejects_all_positive(self):
        batch = generate_batch(
            distance=3,
            rounds=3,
            basis="Z",
            start_phase=0,
            probability=.02,
            shots=16,
            seed=202611110102,
        )
        threshold, _ = choose_threshold([batch], [batch.labels])
        self.assertLess(threshold, 1.0)
        wrong_predictions = np.ones_like(batch.labels)
        threshold, _ = choose_threshold([batch], [wrong_predictions])
        self.assertEqual(threshold, 1.0)
        counts = classification_counts(batch, batch.labels, .5)
        for channel in counts:
            self.assertEqual(channel["false_positive"], 0)
            self.assertEqual(channel["false_negative"], 0)

    def test_compact_grid_covers_octagons_with_small_kernels(self):
        code = build_code(7)
        layout = build_layout(code)
        old_model = build_predecoder(7, 7, "paper_cnn1")
        old_coverage = geometry_coverage(code, layout, old_model)
        self.assertTrue(old_coverage["whole_faces_covered"])
        model = build_predecoder(7, 7, "planar_pilot")
        coverage = geometry_coverage(code, layout, model)
        self.assertTrue(coverage["whole_faces_covered"])
        self.assertTrue(coverage["incident_centers_covered"])
        self.assertEqual(layout.shape, (7, 7))
        distinct_face_sites = set(layout.face_sites)
        distinct_face_count = len(distinct_face_sites)
        self.assertEqual(distinct_face_count, code.face_count)
        for face, anchor in zip(code.geometry["plaquettes"], layout.face_anchors):
            self.assertIn(anchor, face["qubits"])


class CleanupAndCheckpointTests(unittest.TestCase):
    def test_training_command_preserves_explicit_independent_seed_streams(self):
        arguments = [
            "--output", "/tmp/planar488_seed_argument_check",
            "--training-seed", "20261116000000",
            "--validation-seed", "20261117000000",
            "--calibration-seed", "20261118000000",
            "--test-seed", "20261119000000",
            "--torch-seed", "20261116",
        ]
        with patch("training_data.planar_488.jobs.train.run_pilot", return_value=True) as run_training:
            train_job.main(arguments)
        configuration = run_training.call_args.args[0]
        self.assertEqual(configuration.training_seed, 20261116000000)
        self.assertEqual(configuration.validation_seed, 20261117000000)
        self.assertEqual(configuration.calibration_seed, 20261118000000)
        self.assertEqual(configuration.test_seed, 20261119000000)
        self.assertEqual(configuration.torch_seed, 20261116)

    def test_training_and_validation_seed_ranges_cannot_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            PilotConfiguration(
                training_seed=100,
                validation_seed=101,
                calibration_seed=10000,
                test_seed=20000,
            )

    def test_cpu_orchestration_runs_fresh_batches_checkpointing_and_cleanup(self):
        configuration = PilotConfiguration(
            distance=3,
            rounds=2,
            probability=.01,
            steps=4,
            batch_size=2,
            validation_shots=2,
            test_shots=4,
            validation_interval=3,
            training_seconds=1000,
            training_seed=202611130000,
            validation_seed=202611140000,
            calibration_seed=202611150000,
            test_seed=202611160000,
        )
        # Only hardware probes are replaced; sampling, model, optimizer and cleanup run.
        cuda_context = {"initialized": False}

        def inspect_environment(device):
            cuda_context["initialized"] = True
            return {"scope": "CPU test"}

        def reset_memory_counter(device):
            self.assertTrue(cuda_context["initialized"], "CUDA context is not initialized")

        cpu_device = torch.device("cpu")
        cpu_comparison = {"scope": "CPU test"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.object(train_job, "require_cuda_allocation", return_value=cpu_device),
                patch.object(train_job, "environment_record", side_effect=inspect_environment),
                patch.object(train_job, "verify_pilot_backend", return_value=cpu_comparison),
                patch.object(train_job, "MILESTONE_INTERVAL", 2),
                patch("torch.cuda.reset_peak_memory_stats", side_effect=reset_memory_counter),
                patch("torch.cuda.max_memory_allocated", return_value=0),
            ):
                run_pilot(configuration, output, None)
            report = read_report(output)
            self.assertEqual(report["status"], "trained")
            self.assertEqual(report["completed_steps"], 4)
            self.assertTrue(report["checkpoint_reload_checked"])
            last_checkpoint = output / "last.pt"
            last_checkpoint_saved = last_checkpoint.is_file()
            self.assertTrue(last_checkpoint_saved)
            for step in (2, 4):
                milestone = output / "milestones" / f"step_{step:06d}"
                milestone_report = read_report(milestone)
                self.assertEqual(milestone_report["status"], "trained")
                self.assertEqual(milestone_report["completed_steps"], step)
                self.assertEqual(milestone_report["checkpoint_selection"], "fixed_training_step")
                self.assertEqual(len(milestone_report["validation_cases"]), len(CASES))
            last_state = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
            milestone_checkpoint = milestone / "best.pt"
            milestone_state = torch.load(milestone_checkpoint, map_location="cpu", weights_only=True)
            for name, weights in last_state["model"].items():
                torch.testing.assert_close(milestone_state["model"][name], weights, rtol=0, atol=0)
            torch.testing.assert_close(milestone_state["torch_rng"], last_state["torch_rng"], rtol=0, atol=0)
            milestone_evaluation = output / "milestone_evaluation"
            completed = run_evaluation(milestone, milestone_evaluation, seconds=1000)
            self.assertTrue(completed)
            evaluation_output = output / "evaluation"
            completed = run_evaluation(output, evaluation_output, seconds=1000)
            self.assertTrue(completed)
            evaluation_report = read_report(evaluation_output)
            self.assertEqual(evaluation_report["status"], "completed")
            self.assertEqual(evaluation_report["summary"]["test_shots"], 16)
            decoder = evaluation_report["experiment"]["cleanup"]
            self.assertEqual(decoder["name"], "tesseract")
            case_samples = evaluation_output / "test_Z_phase1.npz"
            case_samples_saved = case_samples.is_file()
            self.assertTrue(case_samples_saved)
            with np.load(case_samples) as saved:
                for arm in ("baseline", "combined"):
                    shot_times = saved[f"{arm}_shot_seconds"]
                    self.assertTrue(np.isfinite(shot_times).all())
                    self.assertTrue((shot_times >= 0).all())
                    self.assertAlmostEqual(shot_times.sum(), saved[f"{arm}_seconds"].item())
            last_case = evaluation_report["cases"][-1]
            self.assertEqual(last_case["inference_device"], "cpu")
            self.assertGreater(last_case["inference_seconds_all_planned_shots"], 0)

            interrupted_output = output / "interrupted_evaluation"
            original_decode = TesseractCleanup.decode_one
            decoded_calls = 0

            def interrupt_during_case(cleanup, syndrome):
                nonlocal decoded_calls
                decoded = original_decode(cleanup, syndrome)
                decoded_calls += 1
                if decoded_calls == 3:
                    signal.raise_signal(signal.SIGUSR1)
                return decoded

            with patch.object(TesseractCleanup, "decode_one", new=interrupt_during_case):
                completed = run_evaluation(output, interrupted_output, seconds=1000)
            self.assertFalse(completed)
            partial = read_report(interrupted_output)
            self.assertEqual(partial["status"], "interrupted")
            self.assertGreater(partial["cases"][0]["shots"], 0)
            self.assertLess(partial["cases"][0]["shots"], configuration.test_shots)
            prefix_shots = partial["cases"][0]["shots"]
            prefix_path = interrupted_output / "test_X_phase0.npz"
            with np.load(prefix_path) as saved:
                prefix_times = saved["baseline_shot_seconds"][:prefix_shots].copy()
            completed = run_evaluation(output, interrupted_output, seconds=1000, resume=True)
            self.assertTrue(completed)
            with np.load(prefix_path) as resumed:
                np.testing.assert_array_equal(resumed["baseline_shot_seconds"][:prefix_shots], prefix_times)
            for basis, phase in CASES:
                case_name = f"test_{basis}_phase{phase}.npz"
                expected_path = evaluation_output / case_name
                resumed_path = interrupted_output / case_name
                with np.load(expected_path) as expected:
                    with np.load(resumed_path) as resumed:
                        for name in DECODER_OUTPUTS:
                            np.testing.assert_array_equal(expected[name], resumed[name])
            evaluation_report["experiment"]["cleanup"]["pqlimit"] = 100000
            changed_report = json.dumps(evaluation_report)
            evaluation_report_path = evaluation_output / "report.json"
            evaluation_report_path.write_text(changed_report)
            with self.assertRaisesRegex(ValueError, "decoder"):
                run_evaluation(output, evaluation_output, seconds=1000, resume=True)

    def test_cleanup_predicts_known_logical_flip_and_counts_unreliable_as_failure(self):
        circuit_text = (
            "R 0\n"
            "X_ERROR(0.1) 0\n"
            "M 0\n"
            "DETECTOR rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]"
        )
        circuit = stim.Circuit(circuit_text)
        cleanup = TesseractCleanup(circuit)
        syndromes = np.array([[False], [True], [False], [True]])
        predictions = cleanup.decode(syndromes)
        np.testing.assert_array_equal(predictions.logical, syndromes)
        any_unreliable = predictions.unreliable.any()
        self.assertFalse(any_unreliable)
        failures = logical_failures(predictions, syndromes)
        any_failure = failures.any()
        self.assertFalse(any_failure)
        predictions.unreliable[0] = True
        failures = logical_failures(predictions, syndromes)
        self.assertTrue(failures[0])

    def test_confident_cleanup_errors_explain_native_syndromes(self):
        code = build_code(3)
        circuit = build_circuit(code, 3, "Z", 0, .003)
        sampler = circuit.compile_detector_sampler(seed=20261108)
        detectors, _ = sampler.sample(64, separate_observables=True)
        cleanup = TesseractCleanup(circuit)
        errors = cleanup.decoder.errors
        for syndrome in detectors:
            decoded = cleanup.decode_one(syndrome)
            self.assertFalse(decoded.unreliable)
            explained_detectors = np.zeros_like(syndrome)
            explained_logical = np.zeros_like(decoded.logical)
            for error_index in cleanup.decoder.predicted_errors_buffer:
                symptom = errors[error_index].symptom
                detector_indices = list(symptom.detectors)
                observable_indices = list(symptom.observables)
                explained_detectors[detector_indices] ^= True
                explained_logical[observable_indices] ^= True
            np.testing.assert_array_equal(explained_detectors, syndrome)
            np.testing.assert_array_equal(explained_logical, decoded.logical)

    def test_checkpoint_continues_optimizer_rng_and_next_step_exactly(self):
        torch.manual_seed(20261111)
        torch.set_num_threads(1)
        device = torch.device("cpu")
        batch = generate_batch(
            distance=3,
            rounds=2,
            basis="Z",
            start_phase=1,
            probability=.01,
            shots=2,
            seed=202611110103,
        )
        configuration = {
            "distance": 3,
            "rounds": 2,
            "profile": "inspection",
            "target_protocol": batch.target_protocol,
        }
        model = build_predecoder(3, 2)
        model_parameters = model.parameters()
        optimizer = torch.optim.AdamW(model_parameters, lr=.001)
        train_one_step(model, optimizer, batch, device)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            save_checkpoint(path, model, optimizer, 1, configuration)
            expected_random = torch.rand(4)
            expected_loss = train_one_step(model, optimizer, batch, device)
            restored = build_predecoder(3, 2)
            restored_parameters = restored.parameters()
            restored_optimizer = torch.optim.AdamW(restored_parameters, lr=.001)
            step = load_checkpoint(path, restored, restored_optimizer, configuration, device)
            self.assertEqual(step, 1)
            actual_random = torch.rand(4)
            torch.testing.assert_close(actual_random, expected_random, rtol=0, atol=0)
            actual_loss = train_one_step(restored, restored_optimizer, batch, device)
            self.assertEqual(actual_loss, expected_loss)
            expected_parameters = model.parameters()
            actual_parameters = restored.parameters()
            for expected, actual in zip(expected_parameters, actual_parameters):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            different_configuration = dict(configuration)
            different_configuration["profile"] = "different"
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_checkpoint(path, restored, restored_optimizer, different_configuration, device)
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["cuda_rng"] = torch.get_rng_state()
            torch.save(checkpoint, path)
            with patch("torch.cuda.set_rng_state") as restore_cuda:
                load_checkpoint(path, restored, restored_optimizer, configuration, device)
                restore_cuda.assert_not_called()
            checkpoint["training_schema"] = "planar_488_local_targets_v2"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "schema"):
                load_checkpoint(path, restored, restored_optimizer, configuration, device)
            checkpoint["training_schema"] = TRAINING_SCHEMA
            checkpoint["embedding"] = "physical_coordinates_v1"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "embedding"):
                load_checkpoint(path, restored, restored_optimizer, configuration, device)

    def test_zero_observed_failures_still_have_nonzero_uncertainty(self):
        lower, upper = wilson_interval(0, 100)
        self.assertLess(lower, 1e-15)
        self.assertGreater(upper, .03)
        self.assertLess(upper, .04)


if __name__ == "__main__":
    unittest.main()
