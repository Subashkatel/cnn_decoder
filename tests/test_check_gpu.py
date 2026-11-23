"""GPU harness guards and comparison checks, runnable without GPU access."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from training_data.planar_488.jobs.check_gpu import verify_checkpoint
from training_data.planar_488.jobs.runtime import compare_results, require_cuda_allocation
from training_data.planar_488.model import build_predecoder
from training_data.planar_488.nvidia_cleanup import NvidiaCleanup


class GpuValidationTests(unittest.TestCase):
    def test_gpu_cleanup_rejects_missing_allocation_before_optional_imports(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm GPU allocation"):
                NvidiaCleanup(None)

    def test_no_slurm_allocation_is_rejected_before_cuda_probe(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("torch.cuda.is_available") as available:
                with self.assertRaisesRegex(RuntimeError, "Slurm"):
                    require_cuda_allocation()
                available.assert_not_called()

    def test_missing_gpu_is_rejected_instead_of_falling_back_to_cpu(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "test"}, clear=True):
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "fallback is forbidden"):
                    require_cuda_allocation()

    def test_cuda_check_uses_one_visible_allocated_device(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "test"}, clear=True):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=2):
                    with self.assertRaisesRegex(RuntimeError, "exactly one"):
                        require_cuda_allocation()
                with patch("torch.cuda.device_count", return_value=1):
                    device = require_cuda_allocation()
                    device_name = str(device)
                    self.assertEqual(device_name, "cuda:0")

    def test_backend_comparator_accepts_agreement_and_rejects_wrong_gradients(self):
        logits = torch.tensor([.2, -.4])
        loss = torch.tensor(.3)
        gradient = torch.tensor([1., -2.])
        reference = (logits, loss, [gradient])
        result = compare_results(reference, reference)
        self.assertEqual(result["max_logit_absolute_difference"], 0)
        wrong_gradient = gradient + .01
        changed = (logits, loss, [wrong_gradient])
        with self.assertRaises(AssertionError):
            compare_results(reference, changed)

    def test_paper_profile_runs_and_reloads_the_same_checkpoint(self):
        model = build_predecoder(3, 1, "paper_cnn1")
        inputs = torch.ones((1, 6, 2, 3, 3))
        labels = torch.zeros((1, 4, 2, 3, 3))
        mask = torch.ones((4, 2, 3, 3), dtype=bool)
        zero_inputs = np.zeros(inputs.shape)
        batch = SimpleNamespace(inputs=zero_inputs, distance=3)
        tensors = (inputs, labels, mask)
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            verify_checkpoint(model, batch, tensors, device, path, "paper_cnn1")
        model.eval()
        outputs = model(inputs)
        self.assertEqual(outputs.shape, labels.shape)

    def test_unknown_model_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            build_predecoder(3, 1, "unknown")
