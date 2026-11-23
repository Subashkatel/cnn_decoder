"""Optional NVIDIA GPU cleanup on the full native detector-error matrix."""

import importlib.metadata
import os

import numpy as np

from training_data.planar_488.bp_cleanup import fault_matrices
from training_data.planar_488.cleanup_decoder import CleanupPredictions, CleanupShot


PRESETS = ("bp60-osd0", "relay-bp", "bp10-minllr-jobs")


class NvidiaCleanup:
    """Decode batches on one allocated GPU and verify detector parity."""

    def __init__(self, circuit, preset="bp10-minllr-jobs", batch_size=64):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("NVIDIA cleanup must run inside a Slurm GPU allocation")
        if preset not in PRESETS:
            raise ValueError(f"Unknown NVIDIA decoder preset: {preset}")
        import cupy
        import cudaq_qec as qec
        from cudaq_qec.plugins import nv_qldpc

        if cupy.cuda.runtime.getDeviceCount() != 1:
            raise RuntimeError("NVIDIA cleanup requires exactly one allocated visible GPU")
        self.cuda = cupy.cuda.runtime
        self.dem = circuit.detector_error_model(approximate_disjoint_errors=True)
        self.matrices = fault_matrices(self.dem)
        self.settings = nv_qldpc.preset(preset)
        self.preset = preset
        self.version = importlib.metadata.version("cudaq-qec-cu12")
        detector_matrix = self.matrices.detectors.tocsr()
        self.decoder = qec.get_decoder(
            "nv-qldpc-decoder", detector_matrix,
            error_rate_vec=self.matrices.probabilities,
            bp_batch_size=batch_size,
            **self.settings,
        )
        self.cuda.deviceSynchronize()

    def decode(self, detectors: np.ndarray) -> CleanupPredictions:
        """Include result transfers and full syndrome verification in this API."""
        syndromes = np.asarray(detectors, dtype=np.float64, order="C")
        if syndromes.ndim != 2 or syndromes.shape[1] != self.dem.num_detectors:
            raise ValueError("Detector batch width differs from the native DEM")
        if not np.all((syndromes == 0) | (syndromes == 1)):
            raise ValueError("Detector entries must be binary")
        decoded = self.decoder.decode_batch(syndromes)
        estimates = np.asarray(decoded.result)
        self.cuda.deviceSynchronize()
        expected_shape = (len(detectors), self.matrices.detectors.shape[1])
        if estimates.shape != expected_shape:
            raise RuntimeError("GPU decoder returned an unexpected correction shape")
        estimated_faults = (estimates > .5).astype(np.uint8)
        reconstructed = self.matrices.detectors @ estimated_faults.T
        reconstructed %= 2
        valid = np.all(reconstructed.T == detectors, axis=1)
        logical = self.matrices.logical @ estimated_faults.T
        logical %= 2
        # The plugin's convergence flag describes BP, including when OSD repaired it.
        # A parity-valid OSD correction remains a valid cleanup result.
        return CleanupPredictions(logical=logical.T.astype(bool), unreliable=~valid)

    def decode_one(self, syndrome: np.ndarray) -> CleanupShot:
        detectors = np.asarray(syndrome)[None, :]
        decoded = self.decode(detectors)
        return CleanupShot(logical=decoded.logical[0], unreliable=bool(decoded.unreliable[0]))
