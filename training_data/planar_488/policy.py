"""Select spatial and temporal corrections without changing their physical effects."""

import json
from numbers import Real

import numpy as np

from training_data.planar_488.channels import TargetChannel


Threshold = float | dict[str, float]
ABSTENTION_THRESHOLD = 1.0
CHANNEL_GROUPS = {
    TargetChannel.Z_ERROR_DIFFERENCE: "spatial",
    TargetChannel.X_ERROR_DIFFERENCE: "spatial",
    TargetChannel.X_TEMPORAL_RESIDUAL: "temporal",
    TargetChannel.Z_TEMPORAL_RESIDUAL: "temporal",
}


def threshold_groups(threshold: Threshold) -> dict[str, float]:
    """Keep legacy scalar cutoffs and validate an explicit spatial/temporal pair."""
    if isinstance(threshold, Real):
        groups = {"spatial": threshold, "temporal": threshold}
    elif isinstance(threshold, dict) and set(threshold) == {"spatial", "temporal"}:
        groups = threshold
    else:
        raise ValueError("Correction policy needs a scalar or spatial/temporal thresholds")
    normalized = {}
    for name, value in groups.items():
        if not isinstance(value, Real) or not 0 < value <= 1:
            raise ValueError("Correction threshold must be in (0, 1]")
        normalized[name] = float(value)
    return normalized


def abstains(threshold: Threshold) -> bool:
    """A policy abstains completely only when both correction groups are disabled."""
    groups = threshold_groups(threshold)
    return groups["spatial"] == 1 and groups["temporal"] == 1


def threshold_key(threshold: Threshold) -> str:
    """Preserve scalar report keys and give pairs an order-independent JSON key."""
    if isinstance(threshold, Real):
        return str(threshold)
    groups = threshold_groups(threshold)
    return json.dumps(groups, sort_keys=True, separators=(",", ":"))


def select_corrections(batch, probabilities, threshold: Threshold):
    """Threshold in (shot, channel, boundary, row, column) order and apply the mask."""
    if probabilities.shape != batch.labels.shape:
        raise ValueError("Correction probabilities and batch targets have different shapes")
    finite_probabilities = np.isfinite(probabilities)
    if not np.all(finite_probabilities):
        raise ValueError("Correction probabilities contain nonfinite values")
    groups = threshold_groups(threshold)
    cutoffs = threshold
    if not isinstance(threshold, Real):
        values = []
        for channel in TargetChannel:
            group = CHANNEL_GROUPS[channel]
            values.append(groups[group])
        # Preserve NumPy's scalar promotion at float32 probability boundaries.
        threshold_dtype = np.result_type(probabilities, *values)
        channel_thresholds = np.array(values, dtype=threshold_dtype)
        channel_count = len(values)
        threshold_shape = (1, channel_count, 1, 1, 1)
        cutoffs = channel_thresholds.reshape(threshold_shape)
    selected = probabilities > cutoffs
    selected &= batch.target_mask
    return selected
