"""Native hyperedges and logical labels survive matrix-based cleanup."""

import unittest

import numpy as np
import stim

from training_data.planar_488.bp_cleanup import BpLsdCleanup, fault_matrices
from training_data.planar_488.code import build_code, build_circuit


class BpCleanupTests(unittest.TestCase):
    def test_equal_detector_support_with_different_logical_support_stays_distinct(self):
        model = stim.DetectorErrorModel("""
            error(0.1) D0 D1 D2
            error(0.2) D0 D1 D2 L0
        """)
        matrices = fault_matrices(model)
        np.testing.assert_array_equal(matrices.detectors.toarray(), [[1, 1]] * 3)
        np.testing.assert_array_equal(matrices.logical.toarray(), [[0, 1]])
        np.testing.assert_array_equal(matrices.probabilities, [.1, .2])

    def test_native_fault_matrices_reproduce_stim_fault_records(self):
        code = build_code(7)
        for basis in "XZ":
            for phase in (0, 1):
                with self.subTest(basis=basis, phase=phase):
                    circuit = build_circuit(code, 3, basis, phase, .01)
                    model = circuit.detector_error_model(approximate_disjoint_errors=True)
                    matrices = fault_matrices(model)
                    sampler = model.compile_sampler(seed=12007)
                    detectors, logical, faults = sampler.sample(16, return_errors=True)
                    reconstructed_detectors = matrices.detectors @ faults.T
                    reconstructed_logical = matrices.logical @ faults.T
                    np.testing.assert_array_equal(reconstructed_detectors.T % 2, detectors)
                    np.testing.assert_array_equal(reconstructed_logical.T % 2, logical)
                    self.assertEqual(matrices.detectors.shape[0], circuit.num_detectors)
                    degrees = matrices.detectors.sum(axis=0)
                    self.assertGreater(degrees.max(), 2)

    def test_decoder_returns_the_unique_logical_prediction(self):
        circuit = stim.Circuit("""
            R 0
            X_ERROR(0.1) 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """)
        cleanup = BpLsdCleanup(circuit)
        detectors = np.asarray([[0], [1]], dtype=np.uint8)
        decoded = cleanup.decode(detectors)
        np.testing.assert_array_equal(decoded.logical, [[0], [1]])
        self.assertFalse(decoded.unreliable.any())

    def test_noiseless_model_rejects_an_impossible_syndrome(self):
        circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]")
        cleanup = BpLsdCleanup(circuit)
        valid = cleanup.decode_one(np.asarray([0], dtype=np.uint8))
        invalid = cleanup.decode_one(np.asarray([1], dtype=np.uint8))
        self.assertFalse(valid.unreliable)
        self.assertTrue(invalid.unreliable)
