"""BP+LSD cleanup on the full native detector model, including flags."""

from dataclasses import dataclass
import importlib.metadata

import numpy as np
from scipy.sparse import csc_matrix

from training_data.planar_488.cleanup_decoder import CleanupPredictions, CleanupShot


BP_LSD_CONFIGURATION = {
    "max_iter": 30,
    "bp_method": "minimum_sum",
    "ms_scaling_factor": .625,
    "schedule": "serial",
    "omp_thread_count": 1,
    "lsd_method": "LSD_0",
    "lsd_order": 0,
}


@dataclass
class FaultMatrices:
    """Each column is one DEM fault, without merging different logical effects."""

    detectors: csc_matrix
    logical: csc_matrix
    probabilities: np.ndarray


def binary_matrix(rows, columns, shape):
    """Repeated detector or logical targets cancel under XOR."""
    entries = np.ones(len(rows), dtype=np.uint8)
    matrix = csc_matrix((entries, (rows, columns)), shape=shape)
    matrix.sum_duplicates()
    matrix.data %= 2
    matrix.eliminate_zeros()
    return matrix


def fault_matrices(model) -> FaultMatrices:
    """Keep every hyperedge and flag, retaining the DEM's disjoint-error approximation."""
    detector_rows = []
    detector_columns = []
    logical_rows = []
    logical_columns = []
    probabilities = []
    for instruction in model.flattened():
        if instruction.type in {"detector", "logical_observable"}:
            continue
        if instruction.type != "error":
            raise ValueError(f"Unexpected flattened DEM instruction: {instruction.type}")
        fault_index = len(probabilities)
        probability = instruction.args_copy()[0]
        probabilities.append(probability)
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detector_rows.append(target.val)
                detector_columns.append(fault_index)
            elif target.is_logical_observable_id():
                logical_rows.append(target.val)
                logical_columns.append(fault_index)
            elif not target.is_separator():
                raise ValueError(f"Unexpected DEM error target: {target}")
    fault_count = len(probabilities)
    detector_shape = (model.num_detectors, fault_count)
    logical_shape = (model.num_observables, fault_count)
    detectors = binary_matrix(detector_rows, detector_columns, detector_shape)
    logical = binary_matrix(logical_rows, logical_columns, logical_shape)
    return FaultMatrices(
        detectors=detectors,
        logical=logical,
        probabilities=np.asarray(probabilities),
    )


def decoder_identity() -> dict:
    """Record the package and exact untuned BP+LSD settings used for cleanup."""
    version = importlib.metadata.version("ldpc")
    return {"name": "bp_lsd", "version": version, **BP_LSD_CONFIGURATION}


class BpLsdCleanup:
    """Predict logical flips and reject corrections that fail detector parity."""

    def __init__(self, circuit):
        from ldpc.bplsd_decoder import BpLsdDecoder

        self.version = importlib.metadata.version("ldpc")
        self.dem = circuit.detector_error_model(approximate_disjoint_errors=True)
        self.matrices = fault_matrices(self.dem)
        self.decoder = None
        if self.matrices.probabilities.size:
            error_channel = self.matrices.probabilities.tolist()
            self.decoder = BpLsdDecoder(
                self.matrices.detectors,
                error_channel=error_channel,
                **BP_LSD_CONFIGURATION,
            )

    def decode_one(self, syndrome: np.ndarray) -> CleanupShot:
        """Logical agreement is evaluated separately from satisfying the syndrome."""
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape != (self.dem.num_detectors,):
            raise ValueError("Syndrome length differs from the detector model")
        if np.any(syndrome > 1):
            raise ValueError("Syndrome entries must be binary")
        if self.decoder is None:
            logical = np.zeros(self.dem.num_observables, dtype=bool)
            return CleanupShot(logical=logical, unreliable=bool(syndrome.any()))
        estimated_faults = self.decoder.decode(syndrome)
        reconstructed = self.matrices.detectors @ estimated_faults
        reconstructed %= 2
        syndrome_matches = np.array_equal(reconstructed, syndrome)
        predicted_logical = self.matrices.logical @ estimated_faults
        predicted_logical %= 2
        logical = predicted_logical.astype(bool)
        return CleanupShot(logical=logical, unreliable=not syndrome_matches)

    def decode(self, detectors: np.ndarray) -> CleanupPredictions:
        """Retain every shot, including any invalid syndrome reconstruction."""
        shape = (len(detectors), self.dem.num_observables)
        logical = np.zeros(shape, dtype=bool)
        unreliable = np.zeros(len(detectors), dtype=bool)
        for shot_index, syndrome in enumerate(detectors):
            decoded = self.decode_one(syndrome)
            logical[shot_index] = decoded.logical
            unreliable[shot_index] = decoded.unreliable
        return CleanupPredictions(logical=logical, unreliable=unreliable)
