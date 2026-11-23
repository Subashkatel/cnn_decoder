"""Compact data-site embedding with explicit face anchors and correction masks."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from training_data.detectors import StructuredDetectors, record_indices
from training_data.planar_488.channels import FACE_CHANNELS, PAULI_ORDER, SPATIAL_CHANNELS
from training_data.planar_488.channels import InputChannel, TargetChannel
from training_data.planar_488.code import PlanarCode
from training_data.planar_488.targets import CorrectionTargets, defined_spatial_targets
from training_data.planar_488.targets import defined_temporal_targets


EMBEDDING = "compact_data_anchors_v1"
# Octagons are the heaviest 4.8.8 faces; presence inputs divide face weight by this.
MAXIMUM_FACE_WEIGHT = 8


@dataclass
class GridLayout:
    """Every face and flag is assigned to a distinct nearby data site per channel."""

    shape: tuple[int, int]
    data_sites: list[tuple[int, int]]
    face_sites: list[tuple[int, int]]
    flag_sites: list[list[tuple[int, int]]]
    face_anchors: list[int]
    flag_anchors: list[list[int]]


def distinct_anchors(candidate_lists: list[list[int]]) -> list[int]:
    """Find a deterministic matching, restricted to each object's nearby candidates."""
    object_count = len(candidate_lists)
    site_owners = {}
    object_anchors = {}

    def assign(object_index, visited_sites):
        for data_index in candidate_lists[object_index]:
            if data_index in visited_sites:
                continue
            visited_sites.add(data_index)
            previous_owner = site_owners.get(data_index)
            available = previous_owner is None
            if not available:
                available = assign(previous_owner, visited_sites)
            if available:
                site_owners[data_index] = object_index
                object_anchors[object_index] = data_index
                return True
        return False

    for object_index in range(object_count):
        visited_sites = set()
        if not assign(object_index, visited_sites):
            raise ValueError("Objects collide: no distinct nearby data anchors exist")
    anchors = []
    for object_index in range(object_count):
        anchors.append(object_anchors[object_index])
    return anchors


def nearby_candidates(center, data_sites, support) -> list[int]:
    """Order candidate data indices by compact-grid distance, then native index."""
    center_row, center_column = center
    distances = []
    for data_index in support:
        row, column = data_sites[data_index]
        row_distance = row - center_row
        column_distance = column - center_column
        squared_distance = row_distance ** 2 + column_distance ** 2
        distances.append((squared_distance, data_index))
    distances.sort()
    candidates = []
    for _, data_index in distances:
        candidates.append(data_index)
    return candidates


def build_layout(code: PlanarCode) -> GridLayout:
    """Compress empty physical rows and map face centers onto supporting data sites.

    Native data rows are two physical units apart. Compressing that spacing
    produces a d-by-d grid without rounding or moving any data qubit relative
    to other data qubits. Face centers rank anchors, never enlarge the grid.
    """
    coordinates = code.geometry["qubit_coords"]
    x_coordinates = []
    y_coordinates = []
    for x_coordinate, y_coordinate in coordinates:
        x_coordinates.append(x_coordinate)
        y_coordinates.append(y_coordinate)
    x_origin = min(x_coordinates)
    y_origin = min(y_coordinates)
    data_sites = []
    for x_coordinate, y_coordinate in coordinates:
        physical_row = y_coordinate - y_origin
        if physical_row % 2:
            raise ValueError("Native data rows do not have the expected spacing")
        row = physical_row // 2
        column = x_coordinate - x_origin
        data_sites.append((row, column))
    unique_data_sites = set(data_sites)
    unique_data_count = len(unique_data_sites)
    if unique_data_count != code.data_count:
        raise ValueError("Two data qubits collide in the compact grid")

    face_candidates = []
    for face in code.geometry["plaquettes"]:
        support = face["qubits"]
        row_total = 0
        column_total = 0
        for data_index in support:
            row, column = data_sites[data_index]
            row_total += row
            column_total += column
        face_weight = len(support)
        center_row = Fraction(row_total, face_weight)
        center_column = Fraction(column_total, face_weight)
        candidates = nearby_candidates((center_row, center_column), data_sites, support)
        face_candidates.append(candidates)
    face_anchors = distinct_anchors(face_candidates)
    face_sites = []
    for data_index in face_anchors:
        face_sites.append(data_sites[data_index])

    flag_anchors = []
    flag_sites = []
    for cycle, phase_map in zip(code.cycles, code.cycle_maps):
        measurement_order = []
        for instruction in cycle:
            if instruction.name == "M":
                for target in instruction.targets_copy():
                    measurement_order.append(target.value)
        flag_candidates = []
        for constraint in phase_map["record_constraints"]:
            record_support = record_indices(constraint)
            if len(record_support) != 1:
                raise ValueError("Each native flag must have one physical measurement record")
            record_index = record_support[0]
            ancilla_qubit = measurement_order[record_index]
            ancilla_index = ancilla_qubit - code.data_count
            ancilla_coordinates = code.geometry["ancilla_coords"]
            x_coordinate, y_coordinate = ancilla_coordinates[ancilla_index]
            row_offset = y_coordinate - y_origin
            column_offset = x_coordinate - x_origin
            center_row = Fraction(row_offset, 2)
            center_column = Fraction(column_offset)
            data_indices = range(code.data_count)
            candidates = nearby_candidates((center_row, center_column), data_sites, data_indices)
            flag_candidates.append(candidates)
        phase_anchors = distinct_anchors(flag_candidates)
        phase_sites = []
        for data_index in phase_anchors:
            phase_sites.append(data_sites[data_index])
        flag_anchors.append(phase_anchors)
        flag_sites.append(phase_sites)

    rows, columns = zip(*data_sites)
    height = max(rows) + 1
    width = max(columns) + 1
    return GridLayout(
        shape=(height, width),
        data_sites=data_sites,
        face_sites=face_sites,
        flag_sites=flag_sites,
        face_anchors=face_anchors,
        flag_anchors=flag_anchors,
    )


def encode_tensors(
    layout: GridLayout,
    structured: StructuredDetectors,
    targets: CorrectionTargets,
    basis: str,
    start_phase: int,
    face_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode named heads, normalizing face presence by MAXIMUM_FACE_WEIGHT.

    The mask places the defined targets from targets.py and excludes grid padding.
    """
    face_events = structured.face_detection_events
    face_present = structured.face_present
    flag_events = structured.flag_detection_events
    shots, _, boundaries, _ = face_events.shape
    height, width = layout.shape
    input_channel_count = len(InputChannel)
    target_channel_count = len(TargetChannel)
    inputs = np.zeros((shots, input_channel_count, boundaries, height, width), dtype=np.float32)
    labels = np.zeros((shots, target_channel_count, boundaries, height, width), dtype=np.float32)
    target_mask = np.zeros((target_channel_count, boundaries, height, width), dtype=bool)
    temporal_defined = defined_temporal_targets(face_present)
    spatial_defined = defined_spatial_targets(basis, boundaries)

    for face_index, (row, column) in enumerate(layout.face_sites):
        normalized_weight = face_weights[face_index] / MAXIMUM_FACE_WEIGHT
        for check_axis, event_head, presence_head, target_head in FACE_CHANNELS:
            inputs[:, event_head, :, row, column] = face_events[:, check_axis, :, face_index]
            presence = face_present[check_axis, :, face_index]
            weighted_presence = presence * normalized_weight
            inputs[:, presence_head, :, row, column] = weighted_presence
            labels[:, target_head, :, row, column] = targets.temporal[:, :, check_axis, face_index]
            target_mask[target_head, :, row, column] = temporal_defined[check_axis, :, face_index]

    for qubit_index, (row, column) in enumerate(layout.data_sites):
        for component, target_head in SPATIAL_CHANNELS:
            labels[:, target_head, :, row, column] = targets.spatial[:, :, component, qubit_index]
            target_mask[target_head, :, row, column] = spatial_defined[:, component]

    rounds = boundaries - 1
    for round_index in range(rounds):
        phase = (start_phase + round_index) % 2
        for flag_index, (row, column) in enumerate(layout.flag_sites[phase]):
            flag_column = flag_events[:, round_index, flag_index]
            inputs[:, InputChannel.FLAG_EVENT, round_index, row, column] = flag_column
            inputs[:, InputChannel.FLAG_PRESENT, round_index, row, column] = 1
    return inputs, labels, target_mask


def decode_corrections(layout: GridLayout, heads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read selected heads back into label space, inverting the placement in encode_tensors.

    Heads have axes (shot, target channel, boundary, row, column). The returned spatial
    array has axes (shot, boundary, X/Z, data) and the temporal array ends in face.
    """
    shots, _, boundaries, _, _ = heads.shape
    component_count = len(PAULI_ORDER)
    data_count = len(layout.data_sites)
    face_count = len(layout.face_sites)
    spatial = np.zeros((shots, boundaries, component_count, data_count), dtype=bool)
    temporal = np.zeros((shots, boundaries, component_count, face_count), dtype=bool)
    for qubit_index, (row, column) in enumerate(layout.data_sites):
        for component, target_head in SPATIAL_CHANNELS:
            spatial[:, :, component, qubit_index] = heads[:, target_head, :, row, column]
    for face_index, (row, column) in enumerate(layout.face_sites):
        for check_axis, _, _, target_head in FACE_CHANNELS:
            temporal[:, :, check_axis, face_index] = heads[:, target_head, :, row, column]
    return spatial, temporal
