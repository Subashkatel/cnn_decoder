"""Jointly sample circuit faults, detector events, and corrected data frames."""

from dataclasses import dataclass

import numpy as np
import stim

from training_data.detectors import record_indices
from training_data.planar_488.channels import PAULI_ORDER
from training_data.planar_488.code import PlanarCode


@dataclass
class Trajectory:
    """Audit physical frames and fresh error contributions as (shot, boundary, X/Z, data)."""

    frames: np.ndarray
    measurement_flips: np.ndarray
    detector_events: np.ndarray
    logical_flips: np.ndarray
    opening_temporal_reference: np.ndarray
    independent_spatial_errors: np.ndarray | None = None


def make_error_simulator(qubit_count, shots, seed):
    """Fix the unobservable stabilizer gauge identically in both coupled streams."""
    return stim.FlipSimulator(
        batch_size=shots, num_qubits=qubit_count, seed=seed,
        disable_stabilizer_randomization=True,
    )


@dataclass
class IndependentFaultTracker:
    """Use identical fault draws while clearing historical data frames in this replica."""

    simulator: stim.FlipSimulator
    data_count: int
    byproducts: np.ndarray
    boundaries: list[np.ndarray]

    @classmethod
    def create(cls, qubit_count, data_count, shots, seed):
        simulator = make_error_simulator(qubit_count, shots, seed)
        byproducts = np.zeros((shots, 2, data_count), dtype=bool)
        return cls(simulator=simulator, data_count=data_count,
                   byproducts=byproducts, boundaries=[])

    def capture_frame(self, hadamard_qubits):
        frame = corrected_frame(self.simulator, self.data_count, hadamard_qubits, self.byproducts)
        self.boundaries.append(frame)

    def capture_and_reset_boundary(self, hadamard_qubits):
        """Clear only the auxiliary data frame; keep fresh ancilla reset faults."""
        self.capture_frame(hadamard_qubits)
        x_flips, z_flips, *_ = self.simulator.to_numpy(output_xs=True, output_zs=True)
        # Deterministic p=1 broadcasts clear frames without advancing the noise RNG.
        self.simulator.broadcast_pauli_errors(pauli="X", mask=x_flips[:self.data_count])
        self.simulator.broadcast_pauli_errors(pauli="Z", mask=z_flips[:self.data_count])
        self.byproducts.fill(False)

    def accumulate_round_byproducts(self, first_record, last_record, phase_map):
        records = self.simulator.get_measurement_flips()
        shot_records = records.T
        round_records = shot_records[:, first_record:last_record]
        accumulate_byproducts(self.byproducts, round_records, phase_map)


def record_parity(records: np.ndarray, mask: int) -> np.ndarray:
    """Evaluate one record-map parity in every shot, including an empty support."""
    positions = record_indices(mask)
    selected_records = records[:, positions]
    return np.logical_xor.reduce(selected_records, axis=1)


def corrected_frame(
    simulator: stim.FlipSimulator,
    data_count: int,
    hadamard_qubits: list[int],
    accumulated_byproducts: np.ndarray,
) -> np.ndarray:
    """Undo physical H coordinates, then remove known record-dependent byproducts."""
    x_flips, z_flips, *_ = simulator.to_numpy(
        transpose=True,
        output_xs=True,
        output_zs=True,
    )
    data_x = x_flips[:, :data_count]
    data_z = z_flips[:, :data_count]
    frame = np.stack([data_x, data_z], axis=1)
    swapped_components = frame[:, ::-1, hadamard_qubits]
    frame[:, :, hadamard_qubits] = swapped_components
    return frame ^ accumulated_byproducts


def accumulate_byproducts(
    accumulated: np.ndarray,
    round_records: np.ndarray,
    phase_map: dict,
) -> None:
    """The instrument's Pauli maps are linear functions of local record flips."""
    component_names = ("pauli_frame_x", "pauli_frame_z")
    for component_index, map_name in enumerate(component_names):
        record_masks = phase_map[map_name]
        for qubit_index, record_mask in enumerate(record_masks):
            byproduct = record_parity(round_records, record_mask)
            accumulated[:, component_index, qubit_index] ^= byproduct


def sample_trajectory(
    code: PlanarCode,
    circuit: stim.Circuit,
    rounds: int,
    basis: str,
    start_phase: int,
    shots: int,
    seed: int,
    *, independent_rounds: bool = True,
) -> Trajectory:
    """Track joint physical noise and independent interval fault contributions.

    Stabilizer randomization is disabled to fix the unobservable Pauli-frame
    gauge. These records are noise flips, not complete physical measurement
    outcomes. Independent checks compare their deterministic parities with Stim.
    """
    if shots < 1:
        raise ValueError("Use a positive number of shots")
    if not 0 <= seed < 2**64:
        raise ValueError("Seed must fit Stim's unsigned 64-bit range")

    simulator = make_error_simulator(circuit.num_qubits, shots, seed)
    independent = None
    if independent_rounds:
        independent = IndependentFaultTracker.create(
            qubit_count=circuit.num_qubits, data_count=code.data_count, shots=shots, seed=seed,
        )
    frame_shape = (shots, 2, code.data_count)
    accumulated_byproducts = np.zeros(frame_shape, dtype=bool)
    boundaries = []
    measurement_count = 0
    completed_rounds = 0
    reset_count = 0
    waiting_for_reset_tick = False
    final_boundary_captured = False

    for instruction in circuit.flattened():
        instruction_name = instruction.name
        instruction_targets = instruction.targets_copy()

        # Reset-wait data noise belongs to the input boundary it can trigger.
        if instruction_name == "TICK" and waiting_for_reset_tick:
            round_index = reset_count - 1
            cycle_phase = (start_phase + round_index) % 2
            phase_map = code.cycle_maps[cycle_phase]
            input_h_frame = phase_map["input_h_frame"]
            boundary_frame = corrected_frame(
                simulator, code.data_count, input_h_frame, accumulated_byproducts,
            )
            boundaries.append(boundary_frame)
            if independent is not None:
                independent.capture_and_reset_boundary(input_h_frame)
            waiting_for_reset_tick = False

        detector_during_rounds = instruction_name == "DETECTOR" and completed_rounds < rounds
        if detector_during_rounds:
            expected_measurements = (completed_rounds + 1) * code.ancilla_count
            if measurement_count == expected_measurements:
                cycle_phase = (start_phase + completed_rounds) % 2
                phase_map = code.cycle_maps[cycle_phase]
                measurement_records = simulator.get_measurement_flips()
                shot_records = measurement_records.T
                round_start = completed_rounds * code.ancilla_count
                round_records = shot_records[:, round_start:expected_measurements]
                accumulate_byproducts(accumulated_byproducts, round_records, phase_map)
                if independent is not None:
                    independent.accumulate_round_byproducts(round_start, expected_measurements, phase_map)
                completed_rounds += 1

        if instruction_name == "M" and instruction_targets:
            first_qubit = instruction_targets[0].value
            if first_qubit < code.data_count:
                readout_h_frame = []
                if basis == "X":
                    data_indices = range(code.data_count)
                    readout_h_frame = list(data_indices)
                final_frame = corrected_frame(
                    simulator, code.data_count, readout_h_frame, accumulated_byproducts,
                )
                boundaries.append(final_frame)
                if independent is not None:
                    independent.capture_frame(readout_h_frame)
                final_boundary_captured = True

        simulator.do(instruction)
        if independent is not None:
            independent.simulator.do(instruction)
        if instruction_name == "R" and instruction_targets:
            first_qubit = instruction_targets[0].value
            if first_qubit >= code.data_count:
                reset_count += 1
                waiting_for_reset_tick = True
        if instruction_name == "M":
            measurement_count += len(instruction_targets)

    if completed_rounds != rounds:
        raise ValueError("Circuit does not have the expected number of native memory rounds")
    if not final_boundary_captured:
        raise ValueError("Circuit has no final data-measurement boundary")
    expected_boundary_count = rounds + 1
    if len(boundaries) != expected_boundary_count:
        raise ValueError("Missing input or output frame boundary")

    frames = np.stack(boundaries, axis=1)
    measurement_records = simulator.get_measurement_flips()
    measurement_flips = measurement_records.T
    opening_reference = opening_temporal_reference(code, frames, measurement_flips, start_phase)
    detector_records = simulator.get_detector_flips()
    observable_records = simulator.get_observable_flips()
    independent_errors = None
    if independent is not None:
        independent_errors = np.stack(independent.boundaries, axis=1)
    return Trajectory(
        frames=frames,
        measurement_flips=measurement_flips,
        detector_events=detector_records.T,
        logical_flips=observable_records.T,
        opening_temporal_reference=opening_reference,
        independent_spatial_errors=independent_errors,
    )


def opening_temporal_reference(code, frames, measurement_flips, start_phase):
    """Compare first-cycle input-record signs with the actual input-frame syndrome.

    This fixes the unmeasured opening gauge using that cycle's faults, without
    a running parity chain or a choice based on future rounds.
    """
    opening_frame = frames[:, 0, ::-1]
    integer_frame = opening_frame.astype(np.uint8)
    overlap_counts = integer_frame @ code.checks.T
    input_syndromes = overlap_counts % 2
    first_records = measurement_flips[:, :code.ancilla_count]
    phase_map = code.cycle_maps[start_phase]
    face_maps = phase_map["input_face_record_maps"]
    record_signs = np.zeros_like(input_syndromes, dtype=bool)
    for check_axis, check_basis in enumerate(PAULI_ORDER):
        for face_index, record_mask in enumerate(face_maps[check_basis]):
            parity = record_parity(first_records, record_mask)
            record_signs[:, check_axis, face_index] = parity
    boolean_syndromes = input_syndromes.astype(bool)
    return record_signs ^ boolean_syndromes
