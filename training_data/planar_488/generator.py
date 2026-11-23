"""Compose physical sampling, correction labels, tensor encoding, and export."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import platform

import numpy as np
import stim

from training_data.detectors import StructuredDetectors, describe_detectors, structure_detectors
from training_data.reports import source_tree_hash, text_hash, write_report
from training_data.planar_488.dependencies import NOAH_SOURCE_DIRECTORY
from training_data.planar_488.channels import INPUT_CHANNELS, TARGET_CHANNELS
from training_data.planar_488.code import PlanarCode, build_code, build_circuit
from training_data.planar_488.evaluation import verify_encoded_targets
from training_data.planar_488.label_protocol import DEFAULT_TARGET_PROTOCOL, ROUND_LOCAL_PROTOCOL
from training_data.planar_488.label_protocol import schema_for_protocol
from training_data.planar_488.targets import CorrectionTargets, make_targets
from training_data.planar_488.tensors import EMBEDDING, GridLayout
from training_data.planar_488.tensors import build_layout, encode_tensors
from training_data.planar_488.trajectory import Trajectory, sample_trajectory


TRAINING_SCHEMA = schema_for_protocol(DEFAULT_TARGET_PROTOCOL)


@dataclass
class TrainingBatch:
    """Model tensors and auditable physical arrays for one native circuit."""

    inputs: np.ndarray
    labels: np.ndarray
    target_mask: np.ndarray
    trajectory: Trajectory
    corrections: CorrectionTargets
    structured: StructuredDetectors
    descriptors: list[dict]
    layout: GridLayout
    code: PlanarCode
    circuit: stim.Circuit
    basis: str
    start_phase: int
    target_protocol: str = DEFAULT_TARGET_PROTOCOL

    @property
    def distance(self) -> int:
        return self.code.distance


@dataclass
class TrainingCase:
    """Reusable circuit, geometry, and parity converters; shots remain freshly sampled."""

    code: PlanarCode
    circuit: stim.Circuit
    descriptors: list[dict]
    layout: GridLayout
    reference: np.ndarray
    converter: stim.CompiledMeasurementsToDetectionEventsConverter
    face_weights: np.ndarray
    basis: str
    start_phase: int
    rounds: int

    def sample(self, shots: int, seed: int,
               target_protocol: str = DEFAULT_TARGET_PROTOCOL) -> TrainingBatch:
        """Sample a new correlated trajectory and verify its actual masked targets."""
        schema_for_protocol(target_protocol)
        trajectory = sample_trajectory(
            self.code, self.circuit, self.rounds, self.basis, self.start_phase, shots, seed,
            independent_rounds=target_protocol == ROUND_LOCAL_PROTOCOL,
        )
        records = trajectory.measurement_flips ^ self.reference
        detector_events, logical_flips = self.converter.convert(
            measurements=records, separate_observables=True,
        )
        np.testing.assert_array_equal(detector_events, trajectory.detector_events)
        np.testing.assert_array_equal(logical_flips, trajectory.logical_flips)
        constraints = self.code.cycle_maps[0]["record_constraints"]
        constraint_count = len(constraints)
        structured = structure_detectors(
            detector_events, self.descriptors, self.rounds, self.code.face_count, constraint_count,
        )
        corrections = make_targets(
            self.code,
            trajectory,
            structured.face_detection_events,
            structured.face_present,
            self.basis,
            target_protocol=target_protocol,
        )
        inputs, labels, mask = encode_tensors(
            self.layout, structured, corrections, self.basis, self.start_phase, self.face_weights,
        )
        batch = TrainingBatch(
            inputs=inputs, labels=labels, target_mask=mask,
            trajectory=trajectory, corrections=corrections, structured=structured,
            descriptors=self.descriptors, layout=self.layout,
            code=self.code, circuit=self.circuit, basis=self.basis, start_phase=self.start_phase,
            target_protocol=target_protocol,
        )
        verify_encoded_targets(batch)
        return batch


@lru_cache(maxsize=32)
def build_training_case(distance, rounds, basis, start_phase, probability) -> TrainingCase:
    """Cache only deterministic setup, never sampled shots or random-number state."""
    code = build_code(distance)
    circuit = build_circuit(code, rounds, basis, start_phase, probability)
    descriptors = describe_detectors(code.geometry, code.cycle_maps, rounds, basis, start_phase)
    layout = build_layout(code)
    clean_circuit = circuit.without_noise()
    reference = clean_circuit.reference_sample()
    converter = circuit.compile_m2d_converter()
    face_weights = code.checks.sum(axis=1)
    return TrainingCase(
        code=code, circuit=circuit, descriptors=descriptors, layout=layout,
        reference=reference, converter=converter, face_weights=face_weights,
        basis=basis, start_phase=start_phase, rounds=rounds,
    )


def generate_batch(
    *, distance: int, rounds: int, basis: str, start_phase: int,
    probability: float, shots: int, seed: int,
    target_protocol: str = DEFAULT_TARGET_PROTOCOL,
) -> TrainingBatch:
    """Generate one fresh batch using reusable deterministic circuit setup."""
    case = build_training_case(distance, rounds, basis, start_phase, probability)
    return case.sample(shots, seed, target_protocol)


def generate_dataset(
    *,
    distance: int,
    rounds: int,
    basis: str,
    start_phase: int,
    probability: float,
    shots: int,
    seed: int,
    output: Path,
    target_protocol: str = DEFAULT_TARGET_PROTOCOL,
) -> dict:
    """Export one inspectable case with its tensors, intermediates, and conventions."""
    batch = generate_batch(
        distance=distance,
        rounds=rounds,
        basis=basis,
        start_phase=start_phase,
        probability=probability,
        shots=shots,
        seed=seed,
        target_protocol=target_protocol,
    )
    source_hash = source_tree_hash(NOAH_SOURCE_DIRECTORY)
    layout = batch.layout
    trajectory = batch.trajectory
    corrections = batch.corrections
    circuit_hash = text_hash(batch.circuit)
    python_version = platform.python_version()
    summary = {
        "schema": schema_for_protocol(target_protocol),
        "target_protocol": target_protocol,
        "distance": distance,
        "rounds": rounds,
        "basis": basis,
        "start_phase": start_phase,
        "noise_probability": probability,
        "shots": shots,
        "seed": seed,
        "noise": "source native-CZ SI1000, sampled jointly by Stim FlipSimulator",
        "frame_gauge": "stabilizer randomization disabled; H frames and record byproducts removed",
        "frame_boundaries": "after each ancilla reset moment; final frame before data measurement",
        "opening_reference": "first-cycle input-record signs XOR corrected input-frame syndrome",
        "presence_encoding": "face weight divided by eight; binary flag presence",
        "mask_policy": "observed temporal intervals; no commuting endpoint spatial supervision",
        "source_tree_sha256": source_hash,
        "source_status": (
            "external Noah source installation; identified by source_tree_sha256"
        ),
        "circuit_sha256": circuit_hash,
        "versions": {
            "python": python_version,
            "numpy": np.__version__,
            "stim": stim.__version__,
        },
        "input_channels": INPUT_CHANNELS,
        "target_channels": TARGET_CHANNELS,
        "axes": ["shot", "channel", "boundary", "row", "column"],
        "input_shape": list(batch.inputs.shape),
        "target_shape": list(batch.labels.shape),
        "spatial_reduction": (
            "weight-first local face reduction; exact clearing of stabilizer-only targets"
        ),
        "temporal_targets": "adapted residuals satisfying D[t] = H e[t] XOR m[t] XOR m[t-1]",
        "flag_policy": "preserved as inputs; no flag correction head",
        "grid": {
            "shape": layout.shape,
            "embedding": EMBEDDING,
            "face_data_anchors": layout.face_anchors,
            "flag_data_anchors_by_phase": layout.flag_anchors,
            "data_sites": layout.data_sites,
            "face_sites": layout.face_sites,
            "flag_sites_by_phase": layout.flag_sites,
        },
        "checks_passed": [
            "CSS stabilizer and logical algebra",
            "record parities agree with the Stim measurement-to-detector converter",
            "spatial reduction preserves every syndrome and logical class",
            "spatial and temporal targets reconstruct every present face detector",
            "temporal targets close at the measured final boundary",
            "target logical class agrees with the sampled final logical flip",
            "masked model heads reconstruct every face detector and final logical flip",
        ],
        "scope": (
            "adapted 4.8.8 training data; not the original NVIDIA four-input checkpoint format"
        ),
    }
    prefix = f"planar_488_training_d{distance}_r{rounds}_{basis}_phase{start_phase}"
    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / f"{prefix}.npz"
    sample_arrays = {
        "trainX": batch.inputs, "trainY": batch.labels, "target_mask": batch.target_mask,
        "corrected_frames_xz": trajectory.frames,
        "opening_temporal_reference": trajectory.opening_temporal_reference,
        "measurement_flips": trajectory.measurement_flips,
        "detectors": trajectory.detector_events, "observable_flips": trajectory.logical_flips,
        "raw_spatial_xz": corrections.raw_spatial, "spatial_xz": corrections.spatial,
        "temporal_xz": corrections.temporal,
        "face_detection_events": batch.structured.face_detection_events,
        "face_present": batch.structured.face_present,
        "flag_detection_events": batch.structured.flag_detection_events,
    }
    if trajectory.independent_spatial_errors is not None:
        sample_arrays["independent_spatial_xz"] = trajectory.independent_spatial_errors
    np.savez_compressed(samples_path, **sample_arrays)
    summary_path = output / f"{prefix}.json"
    write_report(summary_path, summary)
    return summary
