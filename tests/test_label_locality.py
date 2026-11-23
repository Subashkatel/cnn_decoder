"""Fault-locality contracts in addition to detector and logical parity checks."""

import unittest

import numpy as np
import stim

from training_data.detectors import describe_detectors, detector_columns, structure_detectors
from training_data.planar_488.channels import TargetChannel
from training_data.planar_488.code import build_code, build_circuit
from training_data.planar_488.evaluation import correction_effect
from training_data.planar_488.generator import generate_batch
from training_data.planar_488.targets import make_targets
from training_data.planar_488.trajectory import sample_trajectory


NOISE_GATES = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}
FAULT_SEED = 203011170000


def inject_one_fault(circuit, data_count, *, round_index, record_index=None,
                     data_qubit=None, data_gate="X_ERROR"):
    """Keep one deterministic reset-wait or ancilla-measurement fault."""
    fault_circuit = stim.Circuit()
    reset_round = -1
    measured_round = 0
    inserted = False
    for instruction in circuit.flattened():
        if instruction.name in NOISE_GATES:
            continue
        targets = instruction.targets_copy()
        ancilla_instruction = False
        if targets:
            first_qubit = targets[0].value
            ancilla_instruction = first_qubit >= data_count
        if instruction.name == "M" and ancilla_instruction:
            requested_record = record_index is not None and measured_round == round_index
            if requested_record:
                ancilla_qubit = targets[record_index].value
                fault_circuit.append("X_ERROR", [ancilla_qubit], 1.0)
                inserted = True
            measured_round += 1
        fault_circuit.append(instruction)
        if instruction.name == "R" and ancilla_instruction:
            reset_round += 1
            requested_reset = data_qubit is not None and reset_round == round_index
            if requested_reset:
                fault_circuit.append(data_gate, [data_qubit], 1.0)
                inserted = True
    if not inserted:
        raise AssertionError("Requested fault location does not exist")
    return fault_circuit


def sample_fault_case(circuit, code, basis, phase, rounds):
    descriptors = describe_detectors(code.geometry, code.cycle_maps, rounds, basis, phase)
    trajectory = sample_trajectory(code, circuit, rounds, basis, phase, 1, FAULT_SEED)
    constraints = code.cycle_maps[0]["record_constraints"]
    constraint_count = len(constraints)
    structured = structure_detectors(
        trajectory.detector_events, descriptors, rounds, code.face_count, constraint_count,
    )
    corrections = make_targets(
        code, trajectory, structured.face_detection_events, structured.face_present, basis,
    )
    return trajectory, structured, corrections


class LabelLocalityTests(unittest.TestCase):
    def assert_labels_remove_face_events(self, batch):
        """Applying the stored labels must clear every face detection event."""
        effect = correction_effect(batch, batch.labels, .5)
        residual = batch.trajectory.detector_events ^ effect.detectors
        face_columns = detector_columns(batch.descriptors, "face")
        face_residuals = residual[:, face_columns]
        any_face_residual = face_residuals.any()
        self.assertFalse(any_face_residual)
        return effect

    def test_reset_wait_fault_uses_the_boundary_that_detects_it(self):
        code = build_code(7)
        circuit = build_circuit(code, 7, "Z", 1, .001)
        faulty = inject_one_fault(circuit, code.data_count, round_index=2, data_qubit=15)
        _, structured, corrections = sample_fault_case(faulty, code, "Z", 1, 7)
        events = structured.face_detection_events
        z_check_events = events[0, 1]
        detection_positions = np.nonzero(z_check_events)
        detection_boundaries = np.unique(detection_positions[0])
        x_error_differences = corrections.spatial[0, :, 0]
        spatial_positions = np.nonzero(x_error_differences)
        spatial_boundaries = np.unique(spatial_positions[0])
        np.testing.assert_array_equal(spatial_boundaries, detection_boundaries)
        any_temporal = corrections.temporal.any()
        self.assertFalse(any_temporal)

    def test_one_record_flip_does_not_make_later_temporal_chains(self):
        for distance in (3, 7):
            code = build_code(distance)
            for basis in "XZ":
                for phase in (0, 1):
                    circuit = build_circuit(code, 7, basis, phase, .001)
                    for fault_round in (0, 1, 6):
                        for record_index in range(code.ancilla_count):
                            faulty = inject_one_fault(
                                circuit,
                                code.data_count,
                                round_index=fault_round,
                                record_index=record_index,
                            )
                            _, _, corrections = sample_fault_case(faulty, code, basis, phase, 7)
                            other_rounds = corrections.temporal.copy()
                            other_rounds[:, fault_round] = False
                            any_other_round = other_rounds.any()
                            self.assertFalse(any_other_round)

    def test_reset_wait_faults_are_local_in_both_phases_and_pauli_frames(self):
        for distance in (3, 7):
            code = build_code(distance)
            for basis in "XZ":
                for phase in (0, 1):
                    circuit = build_circuit(code, 3, basis, phase, .001)
                    for data_gate in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
                        for qubit_index in range(code.data_count):
                            faulty = inject_one_fault(
                                circuit,
                                code.data_count,
                                round_index=1,
                                data_qubit=qubit_index,
                                data_gate=data_gate,
                            )
                            _, _, corrections = sample_fault_case(faulty, code, basis, phase, 3)
                            first_shot_spatial = corrections.spatial[0]
                            positions = np.nonzero(first_shot_spatial)
                            boundaries = np.unique(positions[0])
                            np.testing.assert_array_equal(boundaries, [1])
                            any_temporal = corrections.temporal.any()
                            self.assertFalse(any_temporal)

    def test_noisy_short_memories_preserve_the_masked_correction_contract(self):
        for distance in (3, 7):
            for rounds in (1, 2):
                for basis in "XZ":
                    for phase in (0, 1):
                        seed = 203011190000 + phase
                        batch = generate_batch(
                            distance=distance,
                            rounds=rounds,
                            basis=basis,
                            start_phase=phase,
                            probability=.01,
                            shots=8,
                            seed=seed,
                        )
                        effect = self.assert_labels_remove_face_events(batch)
                        actual_logical = batch.trajectory.logical_flips
                        np.testing.assert_array_equal(effect.logical, actual_logical)

    def test_opening_temporal_interval_is_kept_when_its_other_endpoint_exists(self):
        batch = generate_batch(
            distance=3,
            rounds=3,
            basis="Z",
            start_phase=0,
            probability=.01,
            shots=8,
            seed=203011170001,
        )
        # A Z-basis memory has no X checks at boundary 0, but each interval's right end exists.
        x_temporal_mask = batch.target_mask[TargetChannel.X_TEMPORAL_RESIDUAL]
        for row, column in batch.layout.face_sites:
            opening_interval_defined = x_temporal_mask[0, row, column]
            self.assertTrue(opening_interval_defined)
        self.assert_labels_remove_face_events(batch)

    def test_commuting_endpoint_errors_are_not_supervised(self):
        invisible_heads = (
            ("Z", TargetChannel.Z_ERROR_DIFFERENCE),
            ("X", TargetChannel.X_ERROR_DIFFERENCE),
        )
        for basis, invisible_head in invisible_heads:
            batch = generate_batch(
                distance=3,
                rounds=3,
                basis=basis,
                start_phase=1,
                probability=.01,
                shots=8,
                seed=203011170002,
            )
            head_mask = batch.target_mask[invisible_head]
            for row, column in batch.layout.data_sites:
                first_boundary_defined = head_mask[0, row, column]
                last_boundary_defined = head_mask[-1, row, column]
                self.assertFalse(first_boundary_defined)
                self.assertFalse(last_boundary_defined)


if __name__ == "__main__":
    unittest.main()
