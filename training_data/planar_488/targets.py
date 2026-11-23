"""Spatial stabilizer reduction and temporal labels with an exact parity contract."""

from dataclasses import dataclass

import numpy as np

from training_data.planar_488.channels import PAULI_ORDER, check_axis, error_component
from training_data.planar_488.channels import flipping_component
from training_data.planar_488.code import PlanarCode
from training_data.planar_488.label_protocol import DEFAULT_TARGET_PROTOCOL, ROUND_LOCAL_PROTOCOL
from training_data.planar_488.label_protocol import schema_for_protocol
from training_data.planar_488.trajectory import Trajectory


@dataclass
class CorrectionTargets:
    """Spatial axes are (shot, boundary, X/Z, data); temporal axes end in face."""

    raw_spatial: np.ndarray
    spatial: np.ndarray
    temporal: np.ndarray


def frame_differences(frames: np.ndarray) -> np.ndarray:
    """Use zero before the first boundary, so XOR over time gives the final frame."""
    differences = frames.copy()
    earlier_frames = frames[:, :-1]
    differences[:, 1:] ^= earlier_frames
    return differences


def face_syndromes(spatial: np.ndarray, checks: np.ndarray) -> np.ndarray:
    """X face checks detect Z errors; Z face checks detect X errors."""
    check_components = spatial[:, :, ::-1]
    integer_components = check_components.astype(np.uint8)
    overlap_counts = integer_components @ checks.T
    syndrome_values = overlap_counts % 2
    return syndrome_values.astype(bool)


def reduce_spatial_targets(spatial: np.ndarray, checks: np.ndarray) -> np.ndarray:
    """Reduce weight using complete face stabilizers of any even weight.

    Equal-weight moves must decrease the bit string lexicographically. Each
    accepted move decreases (weight, bit string), so the loop terminates.
    Weight-reducing moves run before equal-weight moves. Exact row-space
    membership then clears any stabilizer-only remainder. Other cosets still
    use local reduction, not exhaustive minimum-weight canonicalization.
    """
    reduced = spatial.copy()
    changed = True
    while changed:
        weight_changed = apply_stabilizer_pass_in_place(reduced, checks, reduce_weight=True)
        tie_changed = apply_stabilizer_pass_in_place(reduced, checks, reduce_weight=False)
        changed = weight_changed or tie_changed
    return clear_stabilizer_only_targets(reduced, checks)


def apply_stabilizer_pass_in_place(errors, checks, *, reduce_weight: bool) -> bool:
    """Apply one ordered pass of either weight reduction or tie fixing."""
    changed = False
    for check in checks:
        stabilizer_mask = check.astype(bool)
        support = np.flatnonzero(check)
        face_weight = len(support)
        selected_values = errors[..., support]
        overlap_weights = selected_values.sum(axis=-1)
        doubled_overlap = 2 * overlap_weights
        if reduce_weight:
            should_toggle = doubled_overlap > face_weight
        else:
            same_weight = doubled_overlap == face_weight
            first_qubit = support[0]
            first_bit_present = errors[..., first_qubit]
            should_toggle = same_weight & first_bit_present
        if np.any(should_toggle):
            toggles = should_toggle[..., None] & stabilizer_mask
            errors ^= toggles
            changed = True
    return changed


def clear_stabilizer_only_targets(errors: np.ndarray, checks: np.ndarray) -> np.ndarray:
    """Map the identity coset to zero using exact binary Gaussian elimination."""
    pivot_rows = {}
    for check in checks:
        row = check.astype(bool)
        support = np.flatnonzero(row)
        while support.size:
            pivot = int(support[0])
            if pivot not in pivot_rows:
                pivot_rows[pivot] = row
                break
            row ^= pivot_rows[pivot]
            support = np.flatnonzero(row)

    remainder = errors.copy()
    pivot_entries = pivot_rows.items()
    ordered_pivots = sorted(pivot_entries)
    for pivot, row in ordered_pivots:
        pivot_values = remainder[..., pivot]
        toggles = pivot_values[..., None] & row
        remainder ^= toggles
    nonzero_remainder = np.any(remainder, axis=-1)
    stabilizer_only = ~nonzero_remainder
    cleared_errors = errors.copy()
    cleared_errors[stabilizer_only] = False
    return cleared_errors


def make_targets(
    code: PlanarCode,
    trajectory: Trajectory,
    face_events: np.ndarray,
    face_present: np.ndarray,
    basis: str,
    target_protocol: str = DEFAULT_TARGET_PROTOCOL,
) -> CorrectionTargets:
    """Solve D[t] = H e[t] XOR m[t] XOR m[t-1] on every present face.

    Face arrays use (shot, X/Z, boundary, face). The measured opening boundary
    uses the prepared reference. The other opening reference is the first
    cycle's record sign relative to its input frame. No later rounds choose it.
    The measured final temporal boundary must close independently of this solve.
    """
    schema_for_protocol(target_protocol)
    physical_differences = frame_differences(trajectory.frames)
    raw_spatial = physical_differences
    if target_protocol == ROUND_LOCAL_PROTOCOL:
        raw_spatial = trajectory.independent_spatial_errors
        if raw_spatial is None:
            raise ValueError("Round-local targets require independently propagated faults")
        verify_spatial_equivalence(code, physical_differences, raw_spatial, "Round-local propagation")
    spatial = reduce_spatial_targets(raw_spatial, code.checks)
    syndromes = face_syndromes(spatial, code.checks)
    ordered_events = face_events.transpose(0, 2, 1, 3)
    ordered_presence = face_present.transpose(1, 0, 2)
    residuals = ordered_events ^ syndromes
    residuals &= ordered_presence
    temporal = np.logical_xor.accumulate(residuals, axis=1)

    measured_axis = check_axis(basis)
    unmeasured_axis = 1 - measured_axis
    opening_reference = trajectory.opening_temporal_reference[:, unmeasured_axis]
    temporal[:, :-1, unmeasured_axis] ^= opening_reference[:, None, :]
    final_residual = temporal[:, -1, measured_axis]
    if np.any(final_residual):
        raise ValueError("Final temporal boundary does not close in the memory basis")

    # The other basis has an unobserved final boundary. Fix that gauge to zero.
    temporal[:, -1] = False
    targets = CorrectionTargets(raw_spatial, spatial, temporal)
    verify_targets(code, trajectory, targets, face_events, face_present, basis)
    return targets


def verify_spatial_equivalence(code, first, second, description):
    """Different representatives must preserve every boundary syndrome and logical class."""
    equivalence = first ^ second
    integer_equivalence = equivalence.astype(np.uint8)
    syndrome_counts = integer_equivalence @ code.checks.T
    syndrome_changes = syndrome_counts % 2
    logical_counts = integer_equivalence @ code.logical_support
    logical_changes = logical_counts % 2
    if np.any(syndrome_changes):
        raise ValueError(f"{description} changed a syndrome")
    if np.any(logical_changes):
        raise ValueError(f"{description} changed a logical class")


def verify_targets(
    code: PlanarCode,
    trajectory: Trajectory,
    targets: CorrectionTargets,
    face_events: np.ndarray,
    face_present: np.ndarray,
    basis: str,
) -> None:
    """Check local equivalence, face reconstruction, and the final logical label."""
    verify_spatial_equivalence(code, targets.raw_spatial, targets.spatial, "Spatial reduction")

    temporal_differences = frame_differences(targets.temporal)
    syndromes = face_syndromes(targets.spatial, code.checks)
    reconstructed_events = syndromes ^ temporal_differences
    reconstructed_faces = reconstructed_events.transpose(0, 2, 1, 3)
    face_mismatch = reconstructed_faces ^ face_events
    present_mismatch = face_mismatch & face_present
    if np.any(present_mismatch):
        raise ValueError("Targets do not reconstruct every present face detector")

    final_frame = np.logical_xor.reduce(targets.spatial, axis=1)
    measured_component = flipping_component(basis)
    measured_errors = final_frame[:, measured_component]
    integer_errors = measured_errors.astype(np.uint8)
    logical_overlaps = integer_errors @ code.logical_support
    predicted_logical = logical_overlaps % 2
    expected_logical = trajectory.logical_flips[:, 0]
    np.testing.assert_array_equal(predicted_logical, expected_logical)


def defined_temporal_targets(face_present: np.ndarray) -> np.ndarray:
    """Interval t spans boundaries t and t+1 and is defined when either endpoint face exists.

    Axes are (X/Z, boundary, face), as in face_present. The final boundary starts no
    interval. The unprepared opening interval stays defined because masking it would
    remove a necessary correction.
    """
    defined = np.zeros_like(face_present)
    left_endpoints = face_present[:, :-1]
    right_endpoints = face_present[:, 1:]
    defined[:, :-1] = left_endpoints | right_endpoints
    return defined


def defined_spatial_targets(basis: str, boundary_count: int) -> np.ndarray:
    """Axes are (boundary, X/Z error component).

    Errors of the memory basis's own type commute with its preparation and readout,
    so they change neither detectors nor the logical label at the first and last
    boundaries. Those targets are undefined there.
    """
    component_count = len(PAULI_ORDER)
    defined = np.ones((boundary_count, component_count), dtype=bool)
    commuting_component = error_component(basis)
    defined[0, commuting_component] = False
    defined[-1, commuting_component] = False
    return defined
