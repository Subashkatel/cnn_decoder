"""NVIDIA Conv3D model profiles, masked loss, output priors, and receptive-field coverage."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as functional

from training_data.planar_488.channels import InputChannel, TargetChannel
from training_data.planar_488.dependencies import load_nvidia_predecoder


INPUT_COUNT = len(InputChannel)
HEAD_COUNT = len(TargetChannel)


@dataclass(frozen=True)
class ModelProfile:
    """Convolution widths and kernel sizes per layer; the last width is the head count."""

    filters: tuple[int, ...]
    kernels: tuple[int, ...]
    activation: str
    dropout: float


MODEL_PROFILES = {
    "inspection": ModelProfile(
        filters=(8, HEAD_COUNT),
        kernels=(3, 3),
        activation="relu",
        dropout=0.0,
    ),
    # NVIDIA paper, Table II model 1 and Table III activation/dropout.
    # Six inputs preserve Noah's flags; this changes the first-layer weights.
    "paper_cnn1": ModelProfile(
        filters=(128, 128, 128, HEAD_COUNT),
        kernels=(3, 3, 3, 3),
        activation="gelu",
        dropout=.01,
    ),
    # Compact data anchors let four small convolutions cover an octagon.
    "planar_pilot": ModelProfile(
        filters=(64, 64, 64, HEAD_COUNT),
        kernels=(3, 3, 3, 3),
        activation="gelu",
        dropout=.01,
    ),
    "compact": ModelProfile(
        filters=(32, 32, 32, HEAD_COUNT),
        kernels=(3, 3, 3, 3),
        activation="gelu",
        dropout=.01,
    ),
}


def build_predecoder(distance: int, rounds: int, profile: str = "inspection") -> torch.nn.Module:
    """Use the original configurable Conv3D architecture with six input channels."""
    if profile not in MODEL_PROFILES:
        profile_names = ", ".join(MODEL_PROFILES)
        raise ValueError(f"Model profile must be one of: {profile_names}")
    selected_profile = MODEL_PROFILES[profile]
    filters = list(selected_profile.filters)
    kernels = list(selected_profile.kernels)
    model_configuration = SimpleNamespace(
        dropout_p=selected_profile.dropout,
        activation=selected_profile.activation,
        num_filters=filters,
        kernel_size=kernels,
        input_channels=INPUT_COUNT,
        out_channels=HEAD_COUNT,
    )
    boundary_count = rounds + 1
    configuration = SimpleNamespace(
        distance=distance,
        n_rounds=boundary_count,
        model=model_configuration,
    )
    predecoder_class = load_nvidia_predecoder()
    return predecoder_class(configuration)


def initialize_output_bias(model: torch.nn.Module, head_priors: np.ndarray) -> None:
    """Start every head at its training-only prevalence, the constant-prediction baseline.

    NVIDIA's model ends in the convolution net[-1], whose bias is each head's prior logit.
    """
    prior_odds = head_priors / (1 - head_priors)
    prior_logits = np.log(prior_odds)
    prior_tensor = torch.from_numpy(prior_logits)
    last_convolution = model.net[-1]
    with torch.no_grad():
        last_convolution.bias.copy_(prior_tensor)


def masked_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Average only over defined data and face targets, excluding padding and gauges."""
    element_losses = functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    expanded_mask = target_mask.expand_as(logits)
    mask_weights = expanded_mask.to(logits.dtype)
    valid_count = mask_weights.sum()
    if valid_count.item() == 0:
        raise ValueError("The training batch has no valid target positions")
    weighted_losses = element_losses * mask_weights
    loss_total = weighted_losses.sum()
    return loss_total / valid_count


def geometry_coverage(code, layout, model) -> dict:
    """Check whole-face spans and incident face anchors on the compact grid.

    This is a necessary geometry check, not a proof that every trajectory label
    is inferable from a local window. Record-byproduct locality remains empirical.
    """
    receptive_radius = 0
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv3d):
            spatial_width = layer.kernel_size[-1]
            receptive_radius += spatial_width // 2
    maximum_face_span = 0
    maximum_center_distance = 0
    faces = code.geometry["plaquettes"]
    for face, center in zip(faces, layout.face_sites):
        sites = []
        for qubit_index in face["qubits"]:
            site = layout.data_sites[qubit_index]
            sites.append(site)
            for coordinate, center_coordinate in zip(site, center):
                center_offset = coordinate - center_coordinate
                separation = abs(center_offset)
                maximum_center_distance = max(maximum_center_distance, separation)
        for first_site in sites:
            for second_site in sites:
                for first_coordinate, second_coordinate in zip(first_site, second_site):
                    site_offset = first_coordinate - second_coordinate
                    span = abs(site_offset)
                    maximum_face_span = max(maximum_face_span, span)
    return {
        "grid_shape": list(layout.shape),
        "receptive_radius": receptive_radius,
        "maximum_face_span": maximum_face_span,
        "maximum_incident_center_distance": maximum_center_distance,
        "whole_faces_covered": receptive_radius >= maximum_face_span,
        "incident_centers_covered": receptive_radius >= maximum_center_distance,
    }
