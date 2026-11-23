"""Apply the four correction heads in the native detector and logical order."""

from dataclasses import dataclass
import math

import numpy as np

from training_data.detectors import detector_columns
from training_data.planar_488.channels import TARGET_CHANNELS, check_axis, flipping_component
from training_data.planar_488.policy import Threshold, select_corrections
from training_data.planar_488.targets import face_syndromes, frame_differences
from training_data.planar_488.tensors import decode_corrections


@dataclass
class CorrectionEffect:
    """Parities to XOR into the observed detector stream and logical label."""

    detectors: np.ndarray
    logical: np.ndarray


def correction_effect(batch, probabilities: np.ndarray, threshold: Threshold) -> CorrectionEffect:
    """Convert masked heads to face parities; flags have no correction head."""
    selected = select_corrections(batch, probabilities, threshold)
    spatial, temporal = decode_corrections(batch.layout, selected)

    code = batch.code
    spatial_syndromes = face_syndromes(spatial, code.checks)
    temporal_syndromes = frame_differences(temporal)
    face_effects = spatial_syndromes ^ temporal_syndromes
    detector_effects = np.zeros_like(batch.trajectory.detector_events)
    for detector_index, descriptor in enumerate(batch.descriptors):
        if descriptor["kind"] != "face":
            continue
        check_basis = descriptor["basis"]
        face_axis = check_axis(check_basis)
        boundary = descriptor["round"]
        face_index = descriptor["face"]
        detector_effects[:, detector_index] = face_effects[:, boundary, face_axis, face_index]

    final_frame = np.logical_xor.reduce(spatial, axis=1)
    measured_component = flipping_component(batch.basis)
    measured_errors = final_frame[:, measured_component]
    integer_errors = measured_errors.astype(np.uint8)
    overlap_counts = integer_errors @ code.logical_support
    logical_parities = overlap_counts % 2
    logical_columns = logical_parities[:, None]
    logical_effects = logical_columns.astype(bool)
    return CorrectionEffect(detectors=detector_effects, logical=logical_effects)


def count_true(values: np.ndarray) -> int:
    """Count true entries as a plain integer, which JSON reports accept."""
    count = np.count_nonzero(values)
    return int(count)


def detector_event_counts(batch, effect: CorrectionEffect, shot_count=None) -> dict[str, int]:
    """Count face events before and after corrections, requiring flag events to be unchanged."""
    original = batch.trajectory.detector_events[:shot_count]
    corrections = effect.detectors[:shot_count]
    residual = original ^ corrections
    face_columns = detector_columns(batch.descriptors, "face")
    flag_columns = detector_columns(batch.descriptors, "record_constraint")
    original_flags = original[:, flag_columns]
    residual_flags = residual[:, flag_columns]
    np.testing.assert_array_equal(original_flags, residual_flags)
    original_faces = original[:, face_columns]
    residual_faces = residual[:, face_columns]
    original_face_count = count_true(original_faces)
    residual_face_count = count_true(residual_faces)
    original_flag_count = count_true(original_flags)
    residual_flag_count = count_true(residual_flags)
    return {
        "original_face_events": original_face_count,
        "residual_face_events": residual_face_count,
        "original_flag_events": original_flag_count,
        "residual_flag_events": residual_flag_count,
    }


def classification_counts(batch, probabilities, threshold: Threshold, shot_count=None) -> list[dict]:
    """Count positive predictions only at defined targets, separately per head."""
    selected_labels = batch.labels[:shot_count]
    predictions = select_corrections(batch, probabilities, threshold)
    predictions = predictions[:shot_count]
    labels = selected_labels.astype(bool)
    reports = []
    for channel_index, name in enumerate(TARGET_CHANNELS):
        channel_mask = batch.target_mask[channel_index]
        selected_labels = labels[:, channel_index, channel_mask]
        selected_predictions = predictions[:, channel_index, channel_mask]
        negative_labels = ~selected_labels
        negative_predictions = ~selected_predictions
        true_positive_sites = selected_labels & selected_predictions
        false_positive_sites = negative_labels & selected_predictions
        false_negative_sites = selected_labels & negative_predictions
        true_positive = count_true(true_positive_sites)
        false_positive = count_true(false_positive_sites)
        false_negative = count_true(false_negative_sites)
        predicted_positive = true_positive + false_positive
        actual_positive = true_positive + false_negative
        precision = None
        recall = None
        if predicted_positive:
            precision = true_positive / predicted_positive
        if actual_positive:
            recall = true_positive / actual_positive
        positive_label_count = count_true(selected_labels)
        channel_report = {
            "channel": name,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "positive_labels": positive_label_count,
            "valid_positions": selected_labels.size,
            "precision": precision,
            "recall": recall,
        }
        reports.append(channel_report)
    return reports


def verify_encoded_targets(batch) -> None:
    """Verify the actual masked model heads, including opening temporal intervals."""
    effect = correction_effect(batch, batch.labels, .5)
    residual = batch.trajectory.detector_events ^ effect.detectors
    face_columns = detector_columns(batch.descriptors, "face")
    flag_columns = detector_columns(batch.descriptors, "record_constraint")
    face_residuals = residual[:, face_columns]
    if np.any(face_residuals):
        raise ValueError("Masked correction heads do not reconstruct a face detector")
    flag_effects = effect.detectors[:, flag_columns]
    if np.any(flag_effects):
        raise ValueError("Correction heads modify a flag detector")
    np.testing.assert_array_equal(effect.logical, batch.trajectory.logical_flips)


def wilson_interval(failures: int, shots: int) -> list[float]:
    """Two-sided 95% Wilson interval for the block-failure probability."""
    if shots < 1:
        raise ValueError("A Wilson interval needs at least one shot")
    failures_in_range = 0 <= failures <= shots
    if not failures_in_range:
        raise ValueError("Failures must lie between zero and the shot count")
    normal_quantile = 1.959963984540054
    probability = failures / shots
    squared_quantile = normal_quantile ** 2
    denominator = 1 + squared_quantile / shots
    center = (probability + squared_quantile / (2 * shots)) / denominator
    variance = probability * (1 - probability) / shots
    variance += squared_quantile / (4 * shots ** 2)
    standard_error = math.sqrt(variance)
    radius = normal_quantile * standard_error / denominator
    lower_bound = center - radius
    upper_bound = center + radius
    clipped_lower_bound = max(0.0, lower_bound)
    clipped_upper_bound = min(1.0, upper_bound)
    return [clipped_lower_bound, clipped_upper_bound]
