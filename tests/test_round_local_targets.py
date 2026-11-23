"""New fault targets must not depend on older faults outside the CNN's view."""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import stim
import torch

from training_data.detectors import structure_detectors
from training_data.planar_488.generator import generate_batch
from training_data.planar_488.jobs.configuration import PilotConfiguration
from training_data.planar_488.learning import load_model_weights, save_checkpoint
from training_data.planar_488.model import build_predecoder
from training_data.planar_488.optimization import build_optimizer
from training_data.planar_488.targets import make_targets
from training_data.planar_488.tensors import encode_tensors
from training_data.planar_488.trajectory import sample_trajectory


def input_boundary_indices(instructions, data_count):
    boundaries = []
    waiting_for_tick = False
    for index, instruction in enumerate(instructions):
        targets = instruction.targets_copy()
        if instruction.name == "R" and targets and targets[0].value >= data_count:
            waiting_for_tick = True
        elif instruction.name == "TICK" and waiting_for_tick:
            boundaries.append(index)
            waiting_for_tick = False
    return boundaries


def inject_canonical_x(instructions, base, faults):
    """Inject known X components in the canonical input H frame at named boundaries."""
    boundaries = input_boundary_indices(instructions, base.code.data_count)
    circuit = stim.Circuit()
    for index, instruction in enumerate(instructions):
        for boundary, qubit in faults:
            if index != boundaries[boundary]:
                continue
            phase = boundary % 2
            hadamards = base.code.cycle_maps[phase]["input_h_frame"]
            pauli = "Z" if qubit in hadamards else "X"
            circuit.append(pauli + "_ERROR", [qubit], 1)
        circuit.append(instruction)
    return circuit


def encoded_fault_case(base, circuit, protocol):
    arguments = (base.code, circuit, 7, "Z", 0, 1, 202611300101)
    trajectory = sample_trajectory(*arguments, independent_rounds=protocol == "round_local")
    constraints = base.code.cycle_maps[0]["record_constraints"]
    structured = structure_detectors(trajectory.detector_events, base.descriptors, 7,
                                      base.code.face_count, len(constraints))
    arguments = (base.code, trajectory, structured.face_detection_events,
                 structured.face_present, "Z")
    corrections = make_targets(*arguments, target_protocol=protocol)
    weights = base.code.checks.sum(axis=1)
    inputs, labels, _ = encode_tensors(base.layout, structured, corrections, "Z", 0, weights)
    return inputs, labels


class RoundLocalTargetTests(unittest.TestCase):
    def test_hidden_early_fault_cannot_change_a_late_local_target(self):
        torch.set_num_threads(1)
        base = generate_batch(distance=7, rounds=7, basis="Z", start_phase=0,
                               probability=0, shots=1, seed=202611300100)
        instructions = list(base.circuit.flattened())
        late_faults = [(5, 27), (5, 0)]
        late_circuit = inject_canonical_x(instructions, base, late_faults)
        both_circuit = inject_canonical_x(instructions, base, [(0, 0), *late_faults])
        legacy_late_inputs, legacy_late_labels = encoded_fault_case(base, late_circuit, "trajectory")
        legacy_both_inputs, legacy_both_labels = encoded_fault_case(base, both_circuit, "trajectory")
        np.testing.assert_array_equal(legacy_late_inputs[:, :, 1:], legacy_both_inputs[:, :, 1:])
        self.assertFalse(np.array_equal(legacy_late_labels[:, :2, 5], legacy_both_labels[:, :2, 5]))
        model = build_predecoder(7, 7, "paper_cnn1")
        model.eval()
        combined_inputs = np.concatenate([legacy_late_inputs, legacy_both_inputs])
        with torch.no_grad():
            logits = model(torch.from_numpy(combined_inputs))
        torch.testing.assert_close(logits[0, :, 5], logits[1, :, 5], rtol=0, atol=0)
        local_inputs, local_labels = encoded_fault_case(base, late_circuit, "round_local")
        both_inputs, both_labels = encoded_fault_case(base, both_circuit, "round_local")
        np.testing.assert_array_equal(local_inputs, legacy_late_inputs)
        np.testing.assert_array_equal(both_inputs, legacy_both_inputs)
        np.testing.assert_array_equal(local_labels[:, :2, 5], both_labels[:, :2, 5])

    def test_protocol_changes_labels_without_changing_the_physical_samples(self):
        for distance in (3, 7):
            for basis in "XZ":
                for phase in (0, 1):
                    arguments = dict(distance=distance, rounds=3, basis=basis, start_phase=phase,
                                     probability=.003, shots=8, seed=202611300200)
                    legacy = generate_batch(**arguments, target_protocol="trajectory")
                    local = generate_batch(**arguments, target_protocol="round_local")
                    np.testing.assert_array_equal(local.inputs, legacy.inputs)
                    np.testing.assert_array_equal(local.trajectory.frames, legacy.trajectory.frames)
                    np.testing.assert_array_equal(local.trajectory.measurement_flips,
                                                  legacy.trajectory.measurement_flips)
                    np.testing.assert_array_equal(local.trajectory.logical_flips,
                                                  legacy.trajectory.logical_flips)
                    np.testing.assert_array_equal(local.corrections.temporal, legacy.corrections.temporal)

    def test_checkpoint_schema_rejects_a_different_label_protocol(self):
        configuration = PilotConfiguration(target_protocol="round_local")
        model = torch.nn.Linear(2, 1)
        optimizer = build_optimizer(model, configuration)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_checkpoint(checkpoint, model, optimizer, 1, asdict(configuration))
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["training_schema"], "planar_488_local_targets_v4")
            load_model_weights(checkpoint, model, asdict(configuration), torch.device("cpu"))
            changed = replace(configuration, target_protocol="trajectory")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_model_weights(checkpoint, model, asdict(changed), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
