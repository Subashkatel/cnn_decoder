"""Interpret and preserve detector columns from Noah's native circuit maps."""

from dataclasses import dataclass

import numpy as np
import stim

from training_data.planar_488.channels import check_axis


@dataclass
class StructuredDetectors:
    """Face events, boundary presence, and flags in the native circuit order."""

    face_detection_events: np.ndarray
    face_present: np.ndarray
    flag_detection_events: np.ndarray

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Expose named arrays at the NPZ serialization boundary without copying."""
        return {
            "face_detection_events": self.face_detection_events,
            "face_present": self.face_present,
            "flag_detection_events": self.flag_detection_events,
        }


def record_indices(mask: int) -> list[int]:
    """Return the zero-based positions selected by a nonnegative record mask."""
    measurement_positions = []
    number_of_bits = mask.bit_length()
    for bit_index in range(number_of_bits):
        shifted_mask = mask >> bit_index

        current_bit = shifted_mask & 1
        if current_bit == 1:
            measurement_positions.append(bit_index)

    return measurement_positions


def detector_columns(descriptors: list[dict], kind: str) -> list[int]:
    """Columns of every detector of one kind: "face", or "record_constraint" for a flag."""
    columns = []
    for detector_index, descriptor in enumerate(descriptors):
        if descriptor["kind"] == kind:
            columns.append(detector_index)
    return columns


def describe_detectors(
    geometry: dict,
    cycle_maps: list[dict],
    rounds: int,
    basis: str,
    start_phase: int,
) -> list[dict]:
    """Describe detectors in the native triangle memory builder declaration order."""
    descriptors = []
    faces = geometry["plaquettes"]
    face_count = len(faces)

    ancilla_coordinates = geometry["ancilla_coords"]
    ancilla_count = len(ancilla_coordinates)

    for round_index in range(rounds):
        phase_number = start_phase + round_index
        cycle_phase = phase_number % 2

        measurement_offset = round_index * ancilla_count

        phase_maps = cycle_maps[cycle_phase]
        record_constraints = phase_maps["record_constraints"]

        for constraint_index, mask in enumerate(record_constraints):
            local_positions = record_indices(mask)

            global_mask = mask << measurement_offset
            global_positions = record_indices(global_mask)

            descriptor = {
                "kind": "record_constraint",
                "round": round_index,
                "phase": cycle_phase,
                "constraint": constraint_index,
                "local_record_support": local_positions,
                "measurement_support": global_positions,
            }

            descriptors.append(descriptor)

        if round_index == 0:
            check_bases = basis
        else:
            check_bases = "XZ"

        input_face_maps = phase_maps["input_face_record_maps"]

        for face_index in range(face_count):
            for check_basis in check_bases:
                basis_input_maps = input_face_maps[check_basis]
                current_mask = basis_input_maps[face_index]
                mask = current_mask << measurement_offset

                if round_index > 0:
                    previous_phase = 1 - cycle_phase
                    previous_phase_maps = cycle_maps[previous_phase]

                    output_face_maps = previous_phase_maps["output_face_record_maps"]
                    basis_output_maps = output_face_maps[check_basis]
                    previous_mask = basis_output_maps[face_index]

                    previous_mask_offset = measurement_offset - ancilla_count
                    shifted_previous_mask = previous_mask << previous_mask_offset
                    mask = mask ^ shifted_previous_mask

                measurement_positions = record_indices(mask)
                descriptor = {
                    "kind": "face",
                    "round": round_index,
                    "phase": cycle_phase,
                    "face": face_index,
                    "basis": check_basis,
                    "measurement_support": measurement_positions,
                }

                descriptors.append(descriptor)

    last_round_index = rounds - 1
    final_phase_number = start_phase + last_round_index
    final_phase = final_phase_number % 2

    final_phase_maps = cycle_maps[final_phase]
    final_output_maps = final_phase_maps["output_face_record_maps"]
    final_basis_maps = final_output_maps[basis]

    last_round_offset = last_round_index * ancilla_count
    data_measurement_offset = rounds * ancilla_count

    for face_index in range(face_count):
        output_mask = final_basis_maps[face_index]
        mask = output_mask << last_round_offset

        face = faces[face_index]
        data_qubits = face["qubits"]

        for qubit_index in data_qubits:
            measurement_index = data_measurement_offset + qubit_index
            measurement_mask = 1 << measurement_index
            mask = mask ^ measurement_mask

        measurement_positions = record_indices(mask)
        descriptor = {
            "kind": "face",
            "round": rounds,
            "phase": final_phase,
            "face": face_index,
            "basis": basis,
            "measurement_support": measurement_positions,
        }
        descriptors.append(descriptor)

    return descriptors


def measurement_parities(
    circuit: stim.Circuit,
    samples: np.ndarray,
    reference: np.ndarray,
    descriptors: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate detector events and logical flips for Noah's native circuits.

    Samples have shape (shots, measurements). The noiseless reference has
    shape (measurements,). Parities relative to that reference match Stim's
    detector-event and logical-observable-flip conventions.
    """
    detector_columns = []

    shot_count = len(samples)
    observable_count = circuit.num_observables
    observable_shape = (shot_count, observable_count)
    observable_values = np.zeros(observable_shape, dtype=bool)

    measurement_count = 0
    flattened_circuit = circuit.flattened()

    for instruction in flattened_circuit:
        instruction_name = instruction.name

        # Noah's native circuits record measurements using M instructions.
        if instruction_name == "M":
            measurement_targets = instruction.targets_copy()
            new_measurement_count = len(measurement_targets)
            measurement_count = measurement_count + new_measurement_count
            continue

        if instruction_name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            continue

        measurement_positions = []
        record_targets = instruction.targets_copy()

        for target in record_targets:
            if not target.is_measurement_record_target:
                raise ValueError("Expected measurement-record parity annotations")

            # Stim rec[-k] targets count backward from the current record end.
            relative_position = target.value
            measurement_position = measurement_count + relative_position
            measurement_positions.append(measurement_position)

        selected_measurements = samples[:, measurement_positions]
        selected_reference = reference[measurement_positions]
        measurement_differences = selected_measurements ^ selected_reference
        parity_values = np.logical_xor.reduce(measurement_differences, axis=1)

        if instruction_name == "DETECTOR":
            detector_index = len(detector_columns)
            descriptor = descriptors[detector_index]

            expected_positions = descriptor["measurement_support"]
            sorted_positions = sorted(measurement_positions)

            if sorted_positions != expected_positions:
                raise ValueError(
                    "Face/flag description differs from the circuit's record parity"
                )

            detector_columns.append(parity_values)
        else:
            instruction_arguments = instruction.gate_args_copy()
            observable_argument = instruction_arguments[0]
            observable_index = int(observable_argument)

            # Several annotations can contribute to the same logical observable.
            previous_values = observable_values[:, observable_index]
            updated_values = previous_values ^ parity_values
            observable_values[:, observable_index] = updated_values

    sample_measurement_count = samples.shape[1]

    if measurement_count != sample_measurement_count:
        raise ValueError("The record parser did not account for every measurement")

    detector_values = np.stack(detector_columns, axis=1)

    return detector_values, observable_values


def structure_detectors(
    detectors: np.ndarray,
    descriptors: list[dict],
    rounds: int,
    face_count: int,
    constraint_count: int,
) -> StructuredDetectors:
    """Arrange detector columns by face, check basis, and measurement round.

    Faces have shape (shots, X/Z, rounds+1, face), with X at index 0.
    Flags have shape (shots, rounds, constraint). The presence mask records
    which face checks exist, even when every shot has a zero event value.
    """
    shot_count = len(detectors)
    boundary_count = rounds + 1

    face_shape = (shot_count, 2, boundary_count, face_count)
    face_values = np.zeros(face_shape, dtype=bool)

    presence_shape = (2, boundary_count, face_count)
    face_present = np.zeros(presence_shape, dtype=bool)

    flag_shape = (shot_count, rounds, constraint_count)
    flag_values = np.zeros(flag_shape, dtype=bool)

    reconstructed_detectors = np.zeros_like(detectors)
    structured_columns = []

    for detector_index, descriptor in enumerate(descriptors):
        round_index = descriptor["round"]
        detector_kind = descriptor["kind"]

        if detector_kind == "face":
            check_basis = descriptor["basis"]
            basis_index = check_axis(check_basis)

            face_index = descriptor["face"]
            destination_column = face_values[:, basis_index, round_index, face_index]
            face_present[basis_index, round_index, face_index] = True
        else:
            constraint_index = descriptor["constraint"]
            destination_column = flag_values[:, round_index, constraint_index]

        source_column = detectors[:, detector_index]
        destination_column[:] = source_column

        # Keep views so later writes to the same slot remain visible here.
        structured_columns.append(destination_column)

    # Reconstruct after all writes to catch collisions that lose detector data.
    for detector_index, structured_column in enumerate(structured_columns):
        reconstructed_detectors[:, detector_index] = structured_column

    np.testing.assert_array_equal(reconstructed_detectors, detectors)

    return StructuredDetectors(
        face_detection_events=face_values,
        face_present=face_present,
        flag_detection_events=flag_values,
    )
