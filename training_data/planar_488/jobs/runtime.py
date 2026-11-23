"""Shared Slurm/CUDA requirements, CPU/GPU agreement, provenance, and job time budgets."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import stim
import torch

from training_data.paths import PACKAGE_ROOT, PROJECT_ROOT
from training_data.planar_488.dependencies import NVIDIA_PREDECODER_SOURCE, NOAH_SOURCE_DIRECTORY
from training_data.planar_488.learning import batch_tensors
from training_data.planar_488.model import masked_binary_loss
from training_data.reports import file_hash, source_tree_hash


LOGIT_TOLERANCES = {"rtol": 1e-4, "atol": 1e-5}
GRADIENT_TOLERANCES = {"rtol": 1e-3, "atol": 1e-5}
# sysexits EX_TEMPFAIL: the job stopped early and can continue with --resume.
STOPPED_EARLY_EXIT_CODE = 75


def require_cuda_allocation() -> torch.device:
    """Reject missing allocations and CPU fallback before starting a GPU workload."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA work must run inside a Slurm GPU allocation")
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated job has no usable CUDA device; CPU fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This worker requires exactly one allocated visible CUDA device")
    return torch.device("cuda:0")


def require_cpu_allocation() -> None:
    """Reject runs outside Slurm and CPU work placed inside a GPU allocation."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Submit cleanup evaluation using the CPU Slurm script")
    allocated_gpus = os.environ.get("SLURM_JOB_GPUS")
    if allocated_gpus:
        raise RuntimeError("Cleanup evaluation must not hold a GPU allocation")


def evaluate_gradients(model, tensors):
    inputs, labels, mask = tensors
    # Disable dropout for comparisons across backends. Gradients remain enabled.
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = masked_binary_loss(logits, labels, mask)
    loss.backward()
    gradients = []
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError("A trainable parameter has no gradient")
        detached_gradient = gradient.detach()
        cpu_gradient = detached_gradient.cpu()
        gradients.append(cpu_gradient)
    detached_logits = logits.detach()
    cpu_logits = detached_logits.cpu()
    detached_loss = loss.detach()
    cpu_loss = detached_loss.cpu()
    return cpu_logits, cpu_loss, gradients


def compare_results(cpu_results, gpu_results) -> dict:
    """Use explicit FP32 tolerances for outputs, masked loss, and all gradients."""
    cpu_logits, cpu_loss, cpu_gradients = cpu_results
    gpu_logits, gpu_loss, gpu_gradients = gpu_results
    torch.testing.assert_close(cpu_logits, gpu_logits, **LOGIT_TOLERANCES)
    torch.testing.assert_close(cpu_loss, gpu_loss, **LOGIT_TOLERANCES)
    cpu_gradient_count = len(cpu_gradients)
    gpu_gradient_count = len(gpu_gradients)
    if cpu_gradient_count != gpu_gradient_count:
        raise RuntimeError("CPU and GPU parameter counts differ")
    maximum_gradient_difference = 0.0
    for cpu_gradient, gpu_gradient in zip(cpu_gradients, gpu_gradients):
        torch.testing.assert_close(cpu_gradient, gpu_gradient, **GRADIENT_TOLERANCES)
        difference = cpu_gradient - gpu_gradient
        absolute_difference = difference.abs()
        maximum_difference = absolute_difference.max()
        difference_value = maximum_difference.item()
        maximum_gradient_difference = max(maximum_gradient_difference, difference_value)
    logit_difference = cpu_logits - gpu_logits
    absolute_logits = logit_difference.abs()
    maximum_logit_tensor = absolute_logits.max()
    maximum_logit_difference = maximum_logit_tensor.item()
    return {
        "max_logit_absolute_difference": maximum_logit_difference,
        "max_gradient_absolute_difference": maximum_gradient_difference,
        "logit_and_loss_tolerances": LOGIT_TOLERANCES,
        "gradient_tolerances": GRADIENT_TOLERANCES,
    }


def compare_backends(cpu_model, gpu_model, batch, device) -> dict:
    """Require matching logits, masked loss, and gradients from identical CPU and GPU weights."""
    cpu_device = torch.device("cpu")
    cpu_tensors = batch_tensors(batch, cpu_device)
    gpu_tensors = batch_tensors(batch, device)
    cpu_results = evaluate_gradients(cpu_model, cpu_tensors)
    gpu_results = evaluate_gradients(gpu_model, gpu_tensors)
    return compare_results(cpu_results, gpu_results)


def configure_precision(tf32: bool = False) -> None:
    """Use FP32 reference arithmetic unless a recorded training recipe enables TF32."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_record(device=None) -> dict:
    """Record versions, source hashes, and the effective compute-node allocation.

    Inspecting CUDA device properties initializes its context before callers
    reset memory counters. CPU evaluation never probes for a GPU.
    """
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    source_hashes = {}
    source_candidates = PACKAGE_ROOT.rglob("*.py")
    source_paths = sorted(source_candidates)
    for source_path in source_paths:
        relative_path = source_path.relative_to(PACKAGE_ROOT)
        relative_name = str(relative_path)
        source_hashes[relative_name] = file_hash(source_path)
    record = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_architecture": platform.machine(),
        "numpy": np.__version__,
        "stim": stim.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "code_revision": revision.stdout.strip(),
        "source_sha256": source_hashes,
        "noah_source_sha256": source_tree_hash(NOAH_SOURCE_DIRECTORY),
        "nvidia_model_sha256": file_hash(NVIDIA_PREDECODER_SOURCE),
    }
    snapshot_path = PROJECT_ROOT / "source_snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        record["live_repository_revision"] = record["code_revision"]
        record["code_revision"] = snapshot["revision"]
        record["source_snapshot"] = snapshot
    if record["job_id"]:
        allocation = subprocess.run(
            ["scontrol", "-o", "show", "job", record["job_id"]],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        record["effective_slurm_allocation"] = allocation.stdout.strip()
    if device is not None:
        properties = torch.cuda.get_device_properties(device)
        gpu_uuid = getattr(properties, "uuid", "unavailable")
        record.update({
            "gpu_name": properties.name,
            "gpu_uuid": str(gpu_uuid),
            "gpu_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "allocated_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "cudnn": torch.backends.cudnn.version(),
        })
    return record


@dataclass
class RunBudget:
    """Stop between bounded work units, including each cleanup-decoding shot."""

    seconds: float
    started: float = field(default_factory=time.monotonic)
    stop_requested: bool = False

    def expired(self) -> bool:
        elapsed = time.monotonic() - self.started
        out_of_time = elapsed >= self.seconds
        return self.stop_requested or out_of_time

    def request_stop(self, signum, frame) -> None:
        self.stop_requested = True

    @contextmanager
    def handle_signals(self):
        """Restore handlers after a run, including notebook and test invocations."""
        previous_handlers = {}
        for signal_number in (signal.SIGUSR1, signal.SIGTERM):
            previous_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, self.request_stop)
        try:
            yield self
        finally:
            for signal_number, handler in previous_handlers.items():
                signal.signal(signal_number, handler)
