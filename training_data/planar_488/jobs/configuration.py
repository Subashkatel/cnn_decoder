"""Settings shared by the training and evaluation jobs, and the training output they exchange."""

from dataclasses import dataclass
import json
from pathlib import Path

from training_data.planar_488.generator import generate_batch
from training_data.planar_488.label_protocol import protocol_from_record, schema_for_configuration
from training_data.planar_488.optimization import TrainingOptions, validate_training_options


CASES = (("X", 0), ("X", 1), ("Z", 0), ("Z", 1))
TRAINING_REPORT_NAME = "report.json"
BEST_CHECKPOINT_NAME = "best.pt"
# Stim and NumPy seeds are unsigned 64-bit integers.
SEED_LIMIT = 2 ** 64


@dataclass(frozen=True)
class PilotConfiguration(TrainingOptions):
    distance: int = 7
    rounds: int = 7
    probability: float = .001
    batch_size: int = 128
    validation_shots: int = 128
    test_shots: int = 512
    steps: int = 4000
    validation_interval: int = 100
    training_seconds: int = 1500
    model_profile: str = "planar_pilot"
    learning_rate: float = .0003
    training_seed: int = 203011070000
    validation_seed: int = 203011080000
    calibration_seed: int = 203011090000
    test_seed: int = 203011100000
    torch_seed: int = 20301107

    @classmethod
    def from_record(cls, recorded):
        """Keep old serialized runs on their original label convention."""
        options = dict(recorded)
        options["target_protocol"] = protocol_from_record(recorded)
        return cls(**options)

    def __post_init__(self):
        """Prevent seed-stream overlap when a larger step budget is requested."""
        validate_training_options(self)
        case_count = len(CASES)
        last_case_offset = case_count - 1
        first_training_seed = self.training_seed + 1
        last_training_seed = self.training_seed + self.steps
        last_validation_seed = self.validation_seed + last_case_offset
        last_calibration_seed = self.calibration_seed + last_case_offset
        last_test_seed = self.test_seed + last_case_offset
        sample_ranges = [
            (first_training_seed, last_training_seed),
            (self.validation_seed, last_validation_seed),
            (self.calibration_seed, last_calibration_seed),
            (self.test_seed, last_test_seed),
        ]
        sample_ranges.sort()
        previous_end = -1
        for first_seed, last_seed in sample_ranges:
            if first_seed <= previous_end:
                raise ValueError("Training, calibration, validation, and test seed ranges overlap")
            fits_unsigned_64 = 0 <= first_seed and last_seed < SEED_LIMIT
            if not fits_unsigned_64:
                raise ValueError("Seed ranges must fit unsigned 64-bit values")
            previous_end = last_seed


def make_batch(configuration: PilotConfiguration, case_index: int, shots: int, seed: int,
               probability: float | None = None):
    basis, phase = CASES[case_index]
    if probability is None:
        probability = configuration.probability
    return generate_batch(
        distance=configuration.distance,
        rounds=configuration.rounds,
        basis=basis,
        start_phase=phase,
        probability=probability,
        shots=shots,
        seed=seed,
        target_protocol=configuration.target_protocol,
    )


def validate_tuning_seed(configuration, seed):
    """Keep policy-selection samples separate from training, calibration, and test."""
    last_seed = seed + len(CASES) - 1
    if not 0 <= seed <= last_seed < SEED_LIMIT:
        raise ValueError("Validation seed range must fit unsigned 64-bit integers")
    excluded = (
        (configuration.training_seed + 1, configuration.training_seed + configuration.steps),
        (configuration.calibration_seed, configuration.calibration_seed + len(CASES) - 1),
        (configuration.test_seed, configuration.test_seed + len(CASES) - 1),
    )
    for first_excluded, last_excluded in excluded:
        overlaps = seed <= last_excluded and first_excluded <= last_seed
        if overlaps:
            raise ValueError("Cleanup validation overlaps training, calibration, or final test seeds")


def read_training_report(training_output: Path) -> dict:
    """Read a saved training report, rejecting one written for different labels."""
    report_path = training_output / TRAINING_REPORT_NAME
    report_text = report_path.read_text()
    report = json.loads(report_text)
    saved_schema = report.get("training_schema")
    expected_schema = schema_for_configuration(report["configuration"])
    if saved_schema != expected_schema:
        raise ValueError("Training report uses a different label schema")
    return report
