"""Recorded training options and NVIDIA's optimizer recipe for controlled studies."""

from dataclasses import asdict, dataclass
import math

import torch

from training_data.planar_488.dependencies import load_nvidia_module
from training_data.planar_488.label_protocol import DEFAULT_TARGET_PROTOCOL
from training_data.planar_488.label_protocol import protocol_from_record, schema_for_protocol


@dataclass(frozen=True)
class TrainingOptions:
    """Keep AdamW defaults while new runs use independent round-local labels."""

    optimizer: str = "adamw"
    weight_decay: float | None = None
    learning_rate_schedule: str = "constant"
    warmup_steps: int = 100
    ema_decay: float = 0.0
    gradient_clip: float = 5.0
    training_probability: float | None = None
    tf32: bool = False
    target_protocol: str = DEFAULT_TARGET_PROTOCOL


def validate_training_options(configuration) -> None:
    """Reject invalid recipes before a GPU allocation does any training."""
    if configuration.optimizer not in ("adamw", "lion"):
        raise ValueError("Optimizer must be adamw or lion")
    schema_for_protocol(configuration.target_protocol)
    if not isinstance(configuration.tf32, bool):
        raise ValueError("TF32 must be explicitly enabled or disabled")
    if configuration.learning_rate_schedule not in ("constant", "warmup_then_decay"):
        raise ValueError("Unknown learning rate schedule")
    if configuration.warmup_steps < 0:
        raise ValueError("Warmup steps cannot be negative")
    if not math.isfinite(configuration.learning_rate) or configuration.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not 0 <= configuration.ema_decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    if not math.isfinite(configuration.gradient_clip) or configuration.gradient_clip <= 0:
        raise ValueError("Gradient clipping must be finite and positive")
    if configuration.weight_decay is not None:
        if not math.isfinite(configuration.weight_decay) or configuration.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
    for probability in (configuration.probability, configuration.training_probability):
        if probability is not None and not 0 <= probability <= 1:
            raise ValueError("Noise probabilities must be in [0, 1]")


def normalized_configuration(configuration: dict) -> dict:
    """Fill optimizer defaults and retain legacy label semantics for older records."""
    normalized = asdict(TrainingOptions())
    normalized["target_protocol"] = protocol_from_record(configuration)
    normalized.update(configuration)
    return normalized


def training_probability(configuration) -> float:
    """Keep training noise distinct from validation and final evaluation noise."""
    if configuration.training_probability is None:
        return configuration.probability
    return configuration.training_probability


def build_optimizer(model, configuration):
    """Reuse NVIDIA's unchanged Lion implementation; preserve AdamW as the reference."""
    weight_decay = configuration.weight_decay
    if weight_decay is None:
        weight_decay = .01 if configuration.optimizer == "adamw" else 1e-7
    parameters = model.parameters()
    if configuration.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=configuration.learning_rate, weight_decay=weight_decay)
    upstream = load_nvidia_module("training.optimizers")
    return upstream.Lion(
        parameters, lr=configuration.learning_rate, betas=(.9, .95), weight_decay=weight_decay,
    )


def learning_rate_for_step(configuration, step: int) -> float:
    """Match zero-based warmup and fractional decays without extra resume state."""
    rate = configuration.learning_rate
    if configuration.learning_rate_schedule == "constant":
        return rate
    optimizer_step = step - 1
    if optimizer_step < configuration.warmup_steps:
        denominator = max(1, configuration.warmup_steps)
        return rate * optimizer_step / denominator
    for fraction in (.25, .5, 1.0):
        boundary = int(configuration.steps * fraction)
        if optimizer_step >= boundary:
            rate *= .7
    return rate


def set_learning_rate(optimizer, rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = rate


@torch.no_grad()
def update_averaged_model(averaged_model, model, decay: float) -> None:
    """Use NVIDIA's EMA convention: old * decay + new * (1 - decay)."""
    for averaged_parameter, parameter in zip(averaged_model.parameters(), model.parameters()):
        averaged_parameter.mul_(decay)
        averaged_parameter.add_(parameter, alpha=1 - decay)
    for averaged_buffer, buffer in zip(averaged_model.buffers(), model.buffers()):
        averaged_buffer.copy_(buffer)
