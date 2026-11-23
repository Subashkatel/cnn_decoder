"""Contracts for measurement parities and lossless detector inputs."""

import copy
import unittest

import numpy as np

from training_data import detectors as detector_helpers
from training_data.planar_488.code import build_code, build_circuit


def sample_case(distance, basis, phase, seed):
    """Use Stim as the parity oracle without loading any saved dataset."""
    rounds = distance
    code = build_code(distance)
    circuit = build_circuit(code, rounds, basis, phase, .001)
    sampler = circuit.compile_sampler(seed=seed)
    measurements = sampler.sample(shots=32)
    reference = circuit.reference_sample()
    converter = circuit.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=measurements,
        separate_observables=True,
    )
    descriptors = detector_helpers.describe_detectors(
        code.geometry, code.cycle_maps, rounds, basis, phase,
    )
    constraint_count = len(code.cycle_maps[0]["record_constraints"])
    return {
        "name": f"distance_{distance}_{basis}_phase_{phase}",
        "arrays": {
            "measurements": measurements,
            "reference_measurements": reference,
            "detectors": detectors,
            "observable_flips": observables,
        },
        "circuit": circuit,
        "maps": {"detectors": descriptors},
        "summary": {
            "rounds": rounds,
            "face_count": code.face_count,
            "record_constraints_per_round": constraint_count,
        },
    }


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        for distance in (3, 7):
            for basis in "XZ":
                for phase in (0, 1):
                    seed = 202611230100 + len(cls.cases)
                    case = sample_case(distance, basis, phase, seed)
                    cls.cases.append(case)

    def test_measurement_parities_match_stim_detectors_and_labels(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                arrays = case["arrays"]
                detectors, observables = detector_helpers.measurement_parities(
                    case["circuit"],
                    arrays["measurements"],
                    arrays["reference_measurements"],
                    case["maps"]["detectors"],
                )
                np.testing.assert_array_equal(detectors, arrays["detectors"])
                np.testing.assert_array_equal(observables, arrays["observable_flips"])

    def test_face_and_flag_inputs_use_explicit_named_slots(self):
        descriptors = [
            {"kind": "face", "basis": "X", "round": 0, "face": 0},
            {"kind": "face", "basis": "Z", "round": 1, "face": 1},
            {"kind": "record_constraint", "round": 0, "constraint": 0},
        ]
        events = np.array([[True, False, True], [False, True, False]])
        structured = detector_helpers.structure_detectors(
            events, descriptors, rounds=1, face_count=2, constraint_count=1,
        )
        expected_faces = np.zeros((2, 2, 2, 2), dtype=bool)
        expected_faces[0, 0, 0, 0] = True
        expected_faces[1, 1, 1, 1] = True
        expected_presence = np.zeros((2, 2, 2), dtype=bool)
        expected_presence[0, 0, 0] = True
        expected_presence[1, 1, 1] = True
        expected_flags = np.array([[[True]], [[False]]])
        np.testing.assert_array_equal(structured.face_detection_events, expected_faces)
        np.testing.assert_array_equal(structured.face_present, expected_presence)
        np.testing.assert_array_equal(structured.flag_detection_events, expected_flags)

    def test_present_checks_are_retained_when_every_event_is_zero(self):
        case = self.cases[0]
        summary = case["summary"]
        zero_detectors = np.zeros_like(case["arrays"]["detectors"])
        structured = detector_helpers.structure_detectors(
            zero_detectors,
            case["maps"]["detectors"],
            summary["rounds"],
            summary["face_count"],
            summary["record_constraints_per_round"],
        )
        original = detector_helpers.structure_detectors(
            case["arrays"]["detectors"],
            case["maps"]["detectors"],
            summary["rounds"],
            summary["face_count"],
            summary["record_constraints_per_round"],
        )
        np.testing.assert_array_equal(structured.face_present, original.face_present)
        any_face_event = np.any(structured.face_detection_events)
        any_flag_event = np.any(structured.flag_detection_events)
        self.assertFalse(any_face_event)
        self.assertFalse(any_flag_event)

    def test_wrong_record_support_is_rejected(self):
        case = self.cases[0]
        descriptors = copy.deepcopy(case["maps"]["detectors"])
        descriptors[0]["measurement_support"] = [0]
        self.assertNotEqual(
            descriptors[0]["measurement_support"],
            case["maps"]["detectors"][0]["measurement_support"],
        )
        with self.assertRaises(ValueError):
            detector_helpers.measurement_parities(
                case["circuit"],
                case["arrays"]["measurements"],
                case["arrays"]["reference_measurements"],
                descriptors,
            )

    def test_colliding_face_slots_cannot_silently_drop_a_column(self):
        descriptor = {"kind": "face", "basis": "X", "round": 0, "face": 0}
        # Two detector columns claim the same face slot.
        descriptors = [descriptor, descriptor]
        different_columns = np.array([[True, False], [False, True]])
        with self.assertRaises(AssertionError):
            detector_helpers.structure_detectors(
                different_columns,
                descriptors,
                rounds=1,
                face_count=1,
                constraint_count=0,
            )
