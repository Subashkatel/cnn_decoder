"""The physical code, its CSS algebra, and native circuit construction."""

from dataclasses import dataclass

import numpy as np
import stim

from training_data.planar_488.dependencies import load_noah_builder


@dataclass
class PlanarCode:
    """Data indices and record maps refer to the unchanged native source order."""

    distance: int
    geometry: dict
    cycles: list[stim.Circuit]
    cycle_maps: list[dict]
    checks: np.ndarray
    logical_support: np.ndarray

    @property
    def data_count(self) -> int:
        return self.checks.shape[1]

    @property
    def face_count(self) -> int:
        return self.checks.shape[0]

    @property
    def ancilla_count(self) -> int:
        coordinates = self.geometry["ancilla_coords"]
        return len(coordinates)


def require_planar_distance(distance) -> None:
    """Accept only the triangular 4.8.8 family d = 4m - 1: 3, 7, 11, ..."""
    if type(distance) is not int:
        raise ValueError("Planar 4.8.8 distance must be an integer")
    distance_remainder = distance % 4
    in_family = distance >= 3 and distance_remainder == 3
    if not in_family:
        raise ValueError("Planar 4.8.8 distance must be 3, 7, 11, ...")


def build_code(distance: int) -> PlanarCode:
    """Check the one-logical-qubit CSS algebra before generating labels."""
    require_planar_distance(distance)
    noah_builder = load_noah_builder()
    geometry, cycles, cycle_maps = noah_builder._triangle_components(distance, "native-cz")
    faces = geometry["plaquettes"]
    data_coordinates = geometry["qubit_coords"]
    data_count = len(data_coordinates)
    face_count = len(faces)
    checks = np.zeros((face_count, data_count), dtype=np.uint8)
    for face_index, face in enumerate(faces):
        data_support = face["qubits"]
        checks[face_index, data_support] = 1

    logical_support = np.zeros(data_count, dtype=np.uint8)
    logical_qubits = cycle_maps[0]["logical_support"]
    logical_support[logical_qubits] = 1
    stabilizer_overlaps = checks @ checks.T
    stabilizer_parities = stabilizer_overlaps % 2
    logical_overlaps = checks @ logical_support
    logical_parities = logical_overlaps % 2
    logical_weight = logical_support.sum()
    if np.any(stabilizer_parities):
        raise ValueError("CSS checks do not commute")
    if np.any(logical_parities):
        raise ValueError("Logical representatives do not commute with the checks")
    if logical_weight % 2 != 1:
        raise ValueError("Logical X and Z must anticommute")

    for phase_map in cycle_maps:
        verify_instrument_maps(phase_map, faces, logical_qubits)

    return PlanarCode(distance, geometry, cycles, cycle_maps, checks, logical_support)


def verify_instrument_maps(phase_map: dict, faces: list[dict], logical_qubits: list[int]) -> None:
    """Prove record byproducts account for every face and logical sign change."""
    components = (("X", "pauli_frame_z"), ("Z", "pauli_frame_x"))
    for check_basis, component_name in components:
        pauli_masks = phase_map[component_name]
        input_face_maps = phase_map["input_face_record_maps"]
        input_maps = input_face_maps[check_basis]
        output_face_maps = phase_map["output_face_record_maps"]
        output_maps = output_face_maps[check_basis]
        for face_index, face in enumerate(faces):
            byproduct_mask = 0
            for qubit_index in face["qubits"]:
                byproduct_mask ^= pauli_masks[qubit_index]
            expected_mask = input_maps[face_index] ^ output_maps[face_index]
            if byproduct_mask != expected_mask:
                raise ValueError("Record byproduct does not preserve a face instrument flow")
        logical_mask = 0
        for qubit_index in logical_qubits:
            logical_mask ^= pauli_masks[qubit_index]
        logical_sign_maps = phase_map["logical_choi_signs"]
        basis_signs = logical_sign_maps[check_basis]
        expected_logical_mask = basis_signs[0]
        if logical_mask != expected_logical_mask:
            raise ValueError("Record byproduct does not preserve the logical instrument flow")


def build_circuit(
    code: PlanarCode,
    rounds: int,
    basis: str,
    start_phase: int,
    probability: float,
) -> stim.Circuit:
    """Use the source's exact native-CZ SI1000 memory construction."""
    noah_builder = load_noah_builder()
    return noah_builder.build_memory(
        "planar_488",
        code.distance,
        basis,
        rounds=rounds,
        start_phase=start_phase,
        gate_set="native-cz",
        p=probability,
    )
