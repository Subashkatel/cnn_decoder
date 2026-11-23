"""Readable optimizer, prediction, and resumable checkpoint operations."""

from pathlib import Path

import numpy as np
import torch

from training_data.planar_488.label_protocol import schema_for_configuration
from training_data.planar_488.model import HEAD_COUNT, masked_binary_loss
from training_data.planar_488.optimization import normalized_configuration
from training_data.planar_488.tensors import EMBEDDING


# Clipping keeps both log terms finite when a stored probability saturates.
PROBABILITY_FLOOR = 1e-7

def batch_tensors(batch, device):
    """Move a batch's inputs, labels, and shared validity mask to one device."""
    input_tensor = torch.from_numpy(batch.inputs)
    label_tensor = torch.from_numpy(batch.labels)
    mask_tensor = torch.from_numpy(batch.target_mask)
    inputs = input_tensor.to(device)
    labels = label_tensor.to(device)
    mask = mask_tensor.to(device)
    return inputs, labels, mask


def require_finite(values: torch.Tensor, description: str) -> None:
    """Stop before a nonfinite loss or gradient reaches an update or a report."""
    finite_values = torch.isfinite(values)
    all_finite = torch.all(finite_values)
    if not all_finite.item():
        raise RuntimeError(f"Nonfinite {description}")


def train_one_step(model, optimizer, batch, device, gradient_clip: float = 5.0) -> float:
    """Update weights from one fresh batch and reject nonfinite loss or gradients."""
    model.train()
    inputs, labels, mask = batch_tensors(batch, device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = masked_binary_loss(logits, labels, mask)
    require_finite(loss, "training loss")
    loss.backward()
    parameters = model.parameters()
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=gradient_clip)
    require_finite(gradient_norm, "training gradient")
    optimizer.step()
    return loss.item()


def predict_probabilities(model, batch, device, batch_size: int) -> np.ndarray:
    """Bound GPU memory while retaining the original shot order."""
    model.eval()
    probabilities = np.empty_like(batch.labels)
    shot_count = len(batch.inputs)
    with torch.no_grad():
        for start_index in range(0, shot_count, batch_size):
            stop_index = start_index + batch_size
            end_index = min(stop_index, shot_count)
            input_array = batch.inputs[start_index:end_index]
            input_tensor = torch.from_numpy(input_array)
            device_inputs = input_tensor.to(device)
            logits = model(device_inputs)
            device_probabilities = torch.sigmoid(logits)
            cpu_probabilities = device_probabilities.cpu()
            probabilities[start_index:end_index] = cpu_probabilities.numpy()
    finite_probabilities = np.isfinite(probabilities)
    if not np.all(finite_probabilities):
        raise RuntimeError("Nonfinite validation predictions")
    return probabilities


def probability_loss(batch, probabilities: np.ndarray) -> float:
    """Unweighted BCE on valid sites, including both positive and negative labels.

    This scores stored probabilities, such as validation predictions or constant
    priors. Training uses masked_binary_loss on logits instead.
    """
    probability_ceiling = 1 - PROBABILITY_FLOOR
    clipped = np.clip(probabilities, PROBABILITY_FLOOR, probability_ceiling)
    negated = -clipped
    log_probability = np.log(clipped)
    log_complement = np.log1p(negated)
    positive_loss = -batch.labels * log_probability
    negative_loss = -(1 - batch.labels) * log_complement
    losses = positive_loss + negative_loss
    valid_losses = losses[:, batch.target_mask]
    mean_loss = valid_losses.mean()
    return float(mean_loss)


def estimate_head_priors(batches) -> np.ndarray:
    """Use training-only prevalence for a constant-probability loss baseline."""
    positive_counts = np.ones(HEAD_COUNT, dtype=np.float64)
    total_counts = np.full(HEAD_COUNT, 2, dtype=np.float64)
    for batch in batches:
        for channel_index in range(HEAD_COUNT):
            channel_mask = batch.target_mask[channel_index]
            valid_labels = batch.labels[:, channel_index, channel_mask]
            positive_counts[channel_index] += valid_labels.sum()
            total_counts[channel_index] += valid_labels.size
    return positive_counts / total_counts


def save_checkpoint(path: Path, model, optimizer, step: int, configuration: dict,
                    *, inference_model=None) -> None:
    """Save optimizer and RNG state as well as weights, using an atomic rename."""
    checkpoint = {
        "training_schema": schema_for_configuration(configuration),
        "embedding": EMBEDDING,
        "configuration": configuration,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }
    if inference_model is not None:
        checkpoint["inference_model"] = inference_model.state_dict()
    model_parameters = model.parameters()
    parameter = next(model_parameters)
    if parameter.is_cuda:
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state(parameter.device)
    temporary_path = path.with_suffix(".partial")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def read_checkpoint(path: Path, configuration: dict, device) -> dict:
    """Load a checkpoint, rejecting one from different labels, embedding, or run settings."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("training_schema") != schema_for_configuration(configuration):
        raise ValueError("Checkpoint training schema differs from the current labels")
    if checkpoint.get("embedding") != EMBEDDING:
        raise ValueError("Checkpoint grid embedding differs from the current tensors")
    saved_configuration = normalized_configuration(checkpoint["configuration"])
    requested_configuration = normalized_configuration(configuration)
    if saved_configuration != requested_configuration:
        raise ValueError("Checkpoint configuration differs from this training run")
    if saved_configuration["ema_decay"] > 0 and "inference_model" not in checkpoint:
        raise ValueError("EMA checkpoint is missing its inference weights")
    return checkpoint


def load_model_weights(path: Path, model, configuration: dict, device) -> None:
    """Load only the weights, for evaluation that never resumes optimization."""
    checkpoint = read_checkpoint(path, configuration, device)
    inference_weights = checkpoint.get("inference_model", checkpoint["model"])
    model.load_state_dict(inference_weights)


def load_checkpoint(path: Path, model, optimizer, configuration: dict, device,
                    *, inference_model=None) -> int:
    """Continue the exact sampling step, optimizer state, and RNG stream of a training run."""
    checkpoint = read_checkpoint(path, configuration, device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if inference_model is not None:
        inference_model.load_state_dict(checkpoint["inference_model"])
    cpu_rng = checkpoint["torch_rng"].cpu()
    torch.set_rng_state(cpu_rng)
    restore_cuda_rng = "cuda_rng" in checkpoint and device.type == "cuda"
    if restore_cuda_rng:
        cuda_rng = checkpoint["cuda_rng"].cpu()
        torch.cuda.set_rng_state(cuda_rng, device)
    return int(checkpoint["step"])
