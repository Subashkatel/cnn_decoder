"""Scientific contracts for the native 4.8.8 correction-target adaptation."""

import copy
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import stim
import torch

from training_data.detectors import describe_detectors, structure_detectors
from training_data.planar_488.code import build_code, build_circuit
from training_data.planar_488.generator import generate_batch, generate_dataset
from training_data.planar_488.learning import train_one_step
from training_data.planar_488.model import build_predecoder, masked_binary_loss
from training_data.planar_488.targets import make_targets, reduce_spatial_targets, verify_targets
from training_data.planar_488.trajectory import sample_trajectory


def insert_fault(instructions, location: int, pauli: str, qubit: int) -> stim.Circuit:
    """Copy a noiseless circuit with one certain Pauli fault before one instruction."""
    faulty = stim.Circuit()
    for index, instruction in enumerate(instructions):
        if index == location:
            faulty.append(f"{pauli}_ERROR", [qubit], 1)
        faulty.append(instruction)
    return faulty


def tableau_detection(circuit: stim.Circuit):
    """Detector events and logical flips from an independent tableau simulation."""
    tableau = stim.TableauSimulator(seed=91)
    tableau.do(circuit)
    measurement_record = tableau.current_measurement_record()
    record_array = np.asarray(measurement_record, dtype=bool)
    records = record_array[None, :]
    converter = circuit.compile_m2d_converter()
    return converter.convert(measurements=records, separate_observables=True)


class PlanarTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        for distance in (3, 7):
            for basis in "XZ":
                for phase in (0, 1):
                    batch = generate_batch(
                        distance=distance,
                        rounds=3,
                        basis=basis,
                        start_phase=phase,
                        probability=.02,
                        shots=16,
                        seed=20261108,
                    )
                    cls.cases.append((distance, basis, phase, batch))

    def test_grid_preserves_every_face_flag_and_correction(self):
        for distance, basis, phase, batch in self.cases:
            with self.subTest(distance=distance, basis=basis, phase=phase):
                structured = batch.structured
                face_events = structured.face_detection_events
                flag_events = structured.flag_detection_events
                for face_index, (row, column) in enumerate(batch.layout.face_sites):
                    event_inputs = batch.inputs[:, :2, :, row, column]
                    expected_events = face_events[:, :, :, face_index]
                    np.testing.assert_array_equal(event_inputs, expected_events)
                    temporal_labels = batch.labels[:, 2:, :, row, column]
                    face_temporal = batch.corrections.temporal[:, :, :, face_index]
                    expected_temporal = face_temporal.transpose(0, 2, 1)
                    np.testing.assert_array_equal(temporal_labels, expected_temporal)
                for round_index in range(3):
                    cycle_phase = (phase + round_index) % 2
                    phase_flag_sites = batch.layout.flag_sites[cycle_phase]
                    for flag_index, (row, column) in enumerate(phase_flag_sites):
                        flag_inputs = batch.inputs[:, 4, round_index, row, column]
                        expected_flags = flag_events[:, round_index, flag_index]
                        np.testing.assert_array_equal(flag_inputs, expected_flags)
                        flag_presence = batch.inputs[:, 5, round_index, row, column]
                        flag_always_present = np.all(flag_presence)
                        self.assertTrue(flag_always_present)
                for qubit, (row, column) in enumerate(batch.layout.data_sites):
                    spatial_labels = batch.labels[:, :2, :, row, column]
                    # Label heads list Z before X; correction components list X before Z.
                    qubit_spatial = batch.corrections.spatial[:, :, ::-1, qubit]
                    expected_spatial = qubit_spatial.transpose(0, 2, 1)
                    np.testing.assert_array_equal(spatial_labels, expected_spatial)

    def test_noiseless_labels_and_events_are_zero_for_both_endings(self):
        for basis in "XZ":
            for phase in (0, 1):
                for rounds in (1, 2):
                    with self.subTest(basis=basis, phase=phase, rounds=rounds):
                        batch = generate_batch(
                            distance=3,
                            rounds=rounds,
                            basis=basis,
                            start_phase=phase,
                            probability=0,
                            shots=8,
                            seed=20261109,
                        )
                        face_presence = batch.inputs[:, 2:4]
                        flag_presence = batch.inputs[:, 5]
                        any_label = np.any(batch.labels)
                        any_frame = np.any(batch.trajectory.frames)
                        any_detector_event = np.any(batch.trajectory.detector_events)
                        any_logical_flip = np.any(batch.trajectory.logical_flips)
                        any_face_present = np.any(face_presence)
                        any_flag_present = np.any(flag_presence)
                        self.assertFalse(any_label)
                        self.assertFalse(any_frame)
                        self.assertFalse(any_detector_event)
                        self.assertFalse(any_logical_flip)
                        self.assertTrue(any_face_present)
                        self.assertTrue(any_flag_present)

    def test_presence_channels_encode_face_weight_over_eight(self):
        for _, _, _, batch in self.cases:
            face_weights = batch.code.checks.sum(axis=1)
            presence = batch.structured.face_present
            for face_index, (row, column) in enumerate(batch.layout.face_sites):
                face_presence = presence[:, :, face_index]
                weight_fraction = face_weights[face_index] / 8.0
                expected = face_presence * weight_fraction
                for shot in batch.inputs:
                    shot_presence = shot[2:4, :, row, column]
                    np.testing.assert_array_equal(shot_presence, expected)

    def test_exhaustive_d3_spatial_errors_preserve_syndrome_and_logical_class(self):
        code = build_code(3)
        data_count = code.data_count
        pattern_count = 2 ** data_count
        errors = np.zeros((pattern_count, 1, 2, data_count), dtype=bool)
        for mask in range(pattern_count):
            # The Z component runs through the patterns in reverse order.
            complement = pattern_count - 1 - mask
            for qubit in range(data_count):
                shifted_mask = mask >> qubit
                selected_bit = shifted_mask & 1
                selected = bool(selected_bit)
                errors[mask, 0, 0, qubit] = selected
                errors[complement, 0, 1, qubit] = selected
        reduced = reduce_spatial_targets(errors, code.checks)
        difference = errors ^ reduced
        integer_difference = difference.astype(np.uint8)
        syndrome_overlap = integer_difference @ code.checks.T
        logical_overlap = integer_difference @ code.logical_support
        syndrome_change = syndrome_overlap % 2
        logical_change = logical_overlap % 2
        any_syndrome_change = np.any(syndrome_change)
        any_logical_change = np.any(logical_change)
        self.assertFalse(any_syndrome_change)
        self.assertFalse(any_logical_change)
        reduced_weights = reduced.sum(axis=-1)
        error_weights = errors.sum(axis=-1)
        weight_not_increased = reduced_weights <= error_weights
        never_heavier = np.all(weight_not_increased)
        self.assertTrue(never_heavier)
        reduced_again = reduce_spatial_targets(reduced, code.checks)
        np.testing.assert_array_equal(reduced_again, reduced)

    def test_logical_corruption_is_rejected_even_when_syndrome_is_unchanged(self):
        distance, basis, _, batch = self.cases[0]
        code = build_code(distance)
        corrupted = copy.deepcopy(batch.corrections)
        logical_operator = code.logical_support.astype(bool)
        corrupted.spatial[0, 0, 0] ^= logical_operator
        with self.assertRaisesRegex(ValueError, "logical class"):
            verify_targets(
                code,
                batch.trajectory,
                corrupted,
                batch.structured.face_detection_events,
                batch.structured.face_present,
                basis,
            )

    def test_stabilizer_only_labels_reduce_to_identity(self):
        code = build_code(7)
        stabilizer_rows = code.checks[:, None, None, :]
        errors = stabilizer_rows.astype(bool)
        reduced = reduce_spatial_targets(errors, code.checks)
        any_correction = np.any(reduced)
        self.assertFalse(any_correction, "A stabilizer needs no physical correction")

    def test_corrupted_temporal_label_is_rejected(self):
        distance, basis, _, batch = self.cases[0]
        code = build_code(distance)
        corrupted = copy.deepcopy(batch.corrections)
        corrupted.temporal[0, 0, 0, 0] ^= True
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            verify_targets(
                code,
                batch.trajectory,
                corrupted,
                batch.structured.face_detection_events,
                batch.structured.face_present,
                basis,
            )

    def test_deterministic_faults_agree_with_tableau_measurement_sampling(self):
        code = build_code(3)
        fault_qubits = (0, code.data_count)
        fault_cases_checked = 0
        for basis in "XZ":
            for phase in (0, 1):
                circuit = build_circuit(code, 1, basis, phase, 0)
                flattened = circuit.flattened()
                instructions = list(flattened)
                descriptors = describe_detectors(code.geometry, code.cycle_maps, 1, basis, phase)
                # Faults go just after the first CZ instruction and just before each M.
                locations = []
                for index, instruction in enumerate(instructions):
                    first_cz = instruction.name == "CZ" and not locations
                    if first_cz:
                        after_first_cz = index + 1
                        locations.append(after_first_cz)
                    if instruction.name == "M":
                        locations.append(index)
                fault_cases = itertools.product(locations, fault_qubits, "XYZ")
                for location, qubit, pauli in fault_cases:
                    faulty = insert_fault(instructions, location, pauli, qubit)
                    trajectory = sample_trajectory(code, faulty, 1, basis, phase, 1, 47)
                    expected_events, expected_logicals = tableau_detection(faulty)
                    np.testing.assert_array_equal(trajectory.detector_events, expected_events)
                    np.testing.assert_array_equal(trajectory.logical_flips, expected_logicals)
                    structured = structure_detectors(
                        expected_events, descriptors, 1, code.face_count, 1,
                    )
                    make_targets(
                        code,
                        trajectory,
                        structured.face_detection_events,
                        structured.face_present,
                        basis,
                    )
                    fault_cases_checked += 1
        self.assertEqual(fault_cases_checked, 72)

    def test_masked_loss_ignores_padding_and_has_zero_padding_gradients(self):
        batch = self.cases[0][3]
        labels = torch.from_numpy(batch.labels)
        mask = torch.from_numpy(batch.target_mask)
        logits = torch.zeros_like(labels, requires_grad=True)
        loss = masked_binary_loss(logits, labels, mask)
        expanded_mask = mask.expand_as(labels)
        padding = ~expanded_mask
        detached_logits = logits.detach()
        changed_logits = detached_logits.clone()
        changed_logits[padding] = 100
        changed_loss = masked_binary_loss(changed_logits, labels, mask)
        torch.testing.assert_close(loss, changed_loss)
        loss.backward()
        padding_gradients = logits.grad[padding]
        target_gradients = logits.grad[expanded_mask]
        any_padding_gradient = torch.any(padding_gradients)
        any_target_gradient = torch.any(target_gradients)
        self.assertFalse(any_padding_gradient)
        self.assertTrue(any_target_gradient)

    def test_original_nvidia_model_accepts_batch_and_updates_weights(self):
        batch = self.cases[0][3]
        boundary_count = batch.inputs.shape[2]
        rounds = boundary_count - 1
        cpu_device = torch.device("cpu")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261110)
            model = build_predecoder(batch.distance, rounds)
            model_state = model.state_dict()
            initial_state = copy.deepcopy(model_state)
            parameters = model.parameters()
            optimizer = torch.optim.SGD(parameters, lr=0.01)
            loss = train_one_step(model, optimizer, batch, cpu_device)
        loss_is_finite = np.isfinite(loss)
        self.assertTrue(loss_is_finite)
        gradient_total = 0.0
        for parameter in model.parameters():
            gradient = parameter.grad
            self.assertIsNotNone(gradient)
            absolute_gradient = gradient.abs()
            gradient_sum = absolute_gradient.sum()
            gradient_value = gradient_sum.item()
            gradient_total += gradient_value
        self.assertGreater(gradient_total, 0)
        updated_state = model.state_dict()
        changed = False
        for name, tensor in updated_state.items():
            unchanged = torch.equal(initial_state[name], tensor)
            if not unchanged:
                changed = True
        self.assertTrue(changed)

    def test_export_contains_training_tensors_and_auditable_intermediates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = generate_dataset(
                distance=3,
                rounds=2,
                basis="Z",
                start_phase=1,
                probability=.001,
                shots=4,
                seed=20261111,
                output=output,
            )
            archive_path = output / "planar_488_training_d3_r2_Z_phase1.npz"
            with np.load(archive_path) as arrays:
                input_shape = list(arrays["trainX"].shape)
                target_shape = list(arrays["trainY"].shape)
                self.assertEqual(input_shape, summary["input_shape"])
                self.assertEqual(target_shape, summary["target_shape"])
                self.assertEqual(arrays["target_mask"].dtype, bool)
                self.assertIn("corrected_frames_xz", arrays.files)
                self.assertIn("temporal_xz", arrays.files)

    def test_invalid_parameters_do_not_create_output(self):
        valid_options = {
            "distance": 3,
            "rounds": 3,
            "basis": "Z",
            "start_phase": 0,
            "probability": .001,
            "shots": 4,
            "seed": 10,
        }
        invalid_changes = (
            {"shots": 0},
            {"seed": -1},
            {"distance": 5},
            {"rounds": 0},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            for invalid in invalid_changes:
                options = valid_options | invalid
                with self.assertRaises(ValueError):
                    generate_dataset(output=output, **options)
                output_created = output.exists()
                self.assertFalse(output_created)
