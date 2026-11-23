"""Independent fixtures transcribed from Shutty arXiv:2609.21376v3, pp. 9 and 17."""

import json
import unittest
from collections import Counter

from training_data.paths import NOAH_SOURCE_ROOT
from training_data.planar_488.cleanup_decoder import PUBLISHED_CONFIGURATION, detector_orders
from training_data.planar_488.code import build_code, build_circuit
from training_data.planar_488.dependencies import load_tesseract


# Table 5 uses explanatory CNOT record indices. Appendix E.2 specifies the
# native phase-1 permutation and conversion to the all-data logical support.
FACE_SUPPORTS = [
    [0, 3, 4, 5], [0, 1, 5, 6], [1, 2, 6, 7], [3, 4, 10, 11],
    [4, 5, 6, 7, 11, 12, 13, 14], [8, 9, 15, 16],
    [9, 10, 11, 12, 16, 17, 18, 19], [12, 13, 19, 20],
    [13, 14, 20, 21], [15, 16, 17, 22], [17, 18, 22, 23],
    [18, 19, 20, 21, 23, 24, 25, 26], [24, 25, 27, 28],
    [25, 26, 28, 29], [27, 28, 29, 30],
]
PHASE0_X_INPUT = [
    [0], [2], [4], [7], [8], [12, 13], [13], [15],
    [17], [19], [21], [23], [27], [28], [31],
]
PHASE0_X_OUTPUT = [
    [7], [2, 8], [4], [0, 7, 13], [2, 8, 15], [12, 13, 19],
    [7, 13, 21], [8, 15, 23], [17], [12], [13, 21],
    [15, 23, 27], [23, 27], [28, 31], [27, 31],
]
PHASE0_Z = [[1], [3], [5], [6], [9], [11], [14], [16], [18], [20], [22], [24], [26], [29], [32]]
PHASE1_X = [[0], [2], [4], [7], [8], [12], [13], [15], [17], [19], [21], [23], [27], [28], [31]]
PHASE1_Z_INPUT = [
    [1], [3], [5], [6], [9], [11, 14], [14], [16],
    [18], [20], [22], [24], [26], [29], [32],
]
PHASE1_Z_OUTPUT = [
    [6], [3, 9], [5], [1, 6, 14], [3, 9, 16], [11, 14, 20],
    [6, 14, 22], [9, 16, 24], [18], [11], [14, 22],
    [16, 24, 26], [24, 26], [29, 32], [26, 32],
]
PHASE1_RECORD_SWAPS = {8: 9, 9: 8, 23: 24, 24: 23}


def native_mask(records, phase):
    mask = 0
    for record in records:
        native_record = record
        if phase == 1:
            native_record = PHASE1_RECORD_SWAPS.get(record, record)
        record_bit = 1 << native_record
        mask ^= record_bit
    return mask


def native_masks(rows, phase):
    masks = []
    for row in rows:
        mask = native_mask(row, phase)
        masks.append(mask)
    return masks


def target_qubits(instruction) -> list[int]:
    targets = instruction.targets_copy()
    qubits = []
    for target in targets:
        qubits.append(target.value)
    return qubits


def qubit_pairs(qubits: list[int]) -> list[tuple[int, int]]:
    """Pairs of a two-qubit instruction, each sorted so that pair order does not matter."""
    first_qubits = qubits[::2]
    second_qubits = qubits[1::2]
    pairs = []
    for first, second in zip(first_qubits, second_qubits):
        low, high = sorted((first, second))
        pairs.append((low, high))
    return pairs


def first_argument(instruction) -> float:
    arguments = instruction.gate_args_copy()
    return arguments[0]


class NoahPaperContractTests(unittest.TestCase):
    def test_published_face_sign_maps_match_native_source_after_record_conversion(self):
        code = build_code(7)
        actual_supports = []
        for face in code.geometry["plaquettes"]:
            actual_supports.append(face["qubits"])
        self.assertEqual(actual_supports, FACE_SUPPORTS)
        tables = [
            (0, "X", PHASE0_X_INPUT, PHASE0_X_OUTPUT),
            (0, "Z", PHASE0_Z, PHASE0_Z),
            (1, "X", PHASE1_X, PHASE1_X),
            (1, "Z", PHASE1_Z_INPUT, PHASE1_Z_OUTPUT),
        ]
        for phase, basis, input_rows, output_rows in tables:
            with self.subTest(phase=phase, basis=basis):
                mapping = code.cycle_maps[phase]
                expected_input = native_masks(input_rows, phase)
                expected_output = native_masks(output_rows, phase)
                self.assertEqual(mapping["input_face_record_maps"][basis], expected_input)
                self.assertEqual(mapping["output_face_record_maps"][basis], expected_output)

    def test_published_logical_signs_convert_to_the_all_data_representative(self):
        code = build_code(7)
        logical_tables = (
            (0, "X", [0, 2, 12], PHASE0_X_INPUT, PHASE0_X_OUTPUT),
            (1, "Z", [1, 3, 11], PHASE1_Z_INPUT, PHASE1_Z_OUTPUT),
        )
        for phase, basis, logical_records, input_rows, output_rows in logical_tables:
            expected = native_mask(logical_records, phase)
            for face_index in (4, 9, 11, 14):
                input_row = input_rows[face_index]
                output_row = output_rows[face_index]
                input_mask = native_mask(input_row, phase)
                output_mask = native_mask(output_row, phase)
                expected ^= input_mask
                expected ^= output_mask
            self.assertEqual(code.cycle_maps[phase]["logical_choi_signs"][basis][0], expected)
        data_qubit_range = range(code.data_count)
        all_data_qubits = list(data_qubit_range)
        self.assertEqual(code.cycle_maps[0]["input_h_frame"], all_data_qubits)
        self.assertEqual(code.cycle_maps[0]["output_h_frame"], [])
        self.assertEqual(code.cycle_maps[1]["input_h_frame"], [])
        self.assertEqual(code.cycle_maps[1]["output_h_frame"], all_data_qubits)
        expected_constraints = [
            1 << 30,
            1 << 25,
            1 << 10,
        ]
        self.assertEqual(code.cycle_maps[0]["record_constraints"], expected_constraints)

    def test_published_resources_and_native_moment_counts(self):
        for distance in (3, 7, 11, 15):
            with self.subTest(distance=distance):
                code = build_code(distance)
                expected_data = (distance**2 + 2 * distance - 1) // 2
                expected_total = (2 * distance**2 + 5 * distance - 5) // 2
                total_qubits = code.data_count + code.ancilla_count
                self.assertEqual(code.data_count, expected_data)
                self.assertEqual(total_qubits, expected_total)
                size_parameter = (distance + 1) // 4
                expected_cz = 60 * size_parameter**2 - 20 * size_parameter - 5
                moments_per_cycle = 24
                if distance == 3:
                    moments_per_cycle = 23
                for cycle in code.cycles:
                    cz_gates = 0
                    for instruction in cycle:
                        if instruction.name == "CZ":
                            cz_qubits = target_qubits(instruction)
                            pair_count = len(cz_qubits) // 2
                            cz_gates += pair_count
                    self.assertEqual(cycle.num_ticks, moments_per_cycle)
                    self.assertEqual(cz_gates, expected_cz)
                memory = build_circuit(code, 3, "Z", 0, .001)
                expected_memory_moments = 3 * moments_per_cycle + 3
                self.assertEqual(memory.num_ticks, expected_memory_moments)

    def test_cleanup_uses_the_published_tesseract_configuration(self):
        record_path = NOAH_SOURCE_ROOT / "data" / "decoder_configs.json"
        records_text = record_path.read_text()
        records = json.loads(records_text)
        published = records["tesseract_63fbe2f8fd7d7d6f"]
        for key, value in PUBLISHED_CONFIGURATION.items():
            self.assertEqual(value, published[key])
        code = build_code(3)
        circuit = build_circuit(code, 1, "Z", 0, .001)
        dem = circuit.detector_error_model(approximate_disjoint_errors=True)
        tesseract_package = load_tesseract()
        orders = detector_orders(tesseract_package.utils, dem)
        order_count = len(orders)
        self.assertEqual(order_count, published["total_orders"])

    def test_si1000_gate_error_probabilities_match_appendix_a(self):
        code = build_code(3)
        probability = .001
        gate_probability = probability / 10
        reset_probability = 2 * probability
        collapse_wait_probability = 2 * probability
        measurement_probability = 5 * probability
        circuit = build_circuit(code, 1, "Z", 0, probability)
        flattened = circuit.flattened()
        instructions = list(flattened)
        expected_followers = {
            "R": ("X_ERROR", reset_probability),
            "H": ("DEPOLARIZE1", gate_probability),
            "CZ": ("DEPOLARIZE2", probability),
        }
        for index, instruction in enumerate(instructions):
            name = instruction.name
            if name in expected_followers:
                expected_name, expected_probability = expected_followers[name]
                following = instructions[index + 1]
                following_arguments = following.gate_args_copy()
                self.assertEqual(following.name, expected_name)
                self.assertEqual(following_arguments, [expected_probability])
                gate_qubits = target_qubits(instruction)
                noisy_qubits = target_qubits(following)
                for qubit in gate_qubits:
                    self.assertIn(qubit, noisy_qubits)
            if name == "M":
                preceding = instructions[index - 1]
                preceding_arguments = preceding.gate_args_copy()
                preceding_targets = preceding.targets_copy()
                measured_targets = instruction.targets_copy()
                self.assertEqual(preceding.name, "X_ERROR")
                self.assertEqual(preceding_arguments, [measurement_probability])
                self.assertEqual(preceding_targets, measured_targets)
            if name == "DEPOLARIZE1":
                depolarizing_probability = first_argument(instruction)
                allowed_probabilities = (gate_probability, collapse_wait_probability)
                self.assertIn(depolarizing_probability, allowed_probabilities)

    def test_si1000_noise_covers_every_gate_idle_and_collapse_wait(self):
        code = build_code(3)
        probability = .001
        idle_probability = probability / 10
        collapse_wait_probability = 2 * probability
        gate_noise = {
            "R": ("X_ERROR", 2 * probability),
            "M": ("X_ERROR", 5 * probability),
            "H": ("DEPOLARIZE1", probability / 10),
        }
        circuit = build_circuit(code, 1, "Z", 0, probability)
        moments = []
        current_moment = []
        for instruction in circuit.flattened():
            if instruction.name == "TICK":
                moments.append(current_moment)
                current_moment = []
            else:
                current_moment.append(instruction)
        qubit_range = range(circuit.num_qubits)
        all_qubits = set(qubit_range)
        for moment_index, moment in enumerate(moments):
            active = set()
            collapsed = set()
            expected = Counter()
            actual = Counter()
            for instruction in moment:
                name = instruction.name
                qubits = target_qubits(instruction)
                if name in ("X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"):
                    error_probability = first_argument(instruction)
                    if name == "DEPOLARIZE2":
                        for pair in qubit_pairs(qubits):
                            actual[(name, error_probability, pair)] += 1
                    else:
                        for qubit in qubits:
                            actual[(name, error_probability, (qubit,))] += 1
                if name not in ("R", "M", "H", "CZ"):
                    continue
                active.update(qubits)
                if name in ("R", "M"):
                    collapsed.update(qubits)
                if name == "CZ":
                    for pair in qubit_pairs(qubits):
                        expected[("DEPOLARIZE2", probability, pair)] += 1
                else:
                    channel, error_probability = gate_noise[name]
                    for qubit in qubits:
                        expected[(channel, error_probability, (qubit,))] += 1
            idle_qubits = all_qubits - active
            for qubit in idle_qubits:
                expected[("DEPOLARIZE1", idle_probability, (qubit,))] += 1
            if collapsed:
                waiting_qubits = all_qubits - collapsed
                for qubit in waiting_qubits:
                    expected[("DEPOLARIZE1", collapse_wait_probability, (qubit,))] += 1
            self.assertEqual(actual, expected, f"Noise differs in moment {moment_index}")
