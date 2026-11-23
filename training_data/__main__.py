"""Command interface for exporting one inspectable planar 4.8.8 training case."""

import argparse
import json
from pathlib import Path

from training_data.paths import EXAMPLES_ROOT
from training_data.planar_488.generator import generate_dataset
from training_data.planar_488.label_protocol import DEFAULT_TARGET_PROTOCOL, SCHEMA_BY_PROTOCOL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    training_parser = subcommands.add_parser(
        "noah-training", help="Generate verified 4.8.8 local targets and model tensors",
    )
    training_parser.add_argument("--distance", type=int, default=3)
    training_parser.add_argument("--rounds", type=int, default=3)
    training_parser.add_argument("--basis", choices=["X", "Z"], default="Z")
    training_parser.add_argument("--start-phase", type=int, choices=[0, 1], default=0)
    training_parser.add_argument("--shots", type=int, default=32)
    training_parser.add_argument("--p", type=float, default=.001)
    training_parser.add_argument("--seed", type=int, default=20261107)
    training_parser.add_argument("--target-protocol", choices=tuple(SCHEMA_BY_PROTOCOL),
                                 default=DEFAULT_TARGET_PROTOCOL)
    training_output = EXAMPLES_ROOT / "planar_488_training"
    training_parser.add_argument("--output", type=Path, default=training_output)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        report = generate_dataset(
            distance=arguments.distance,
            rounds=arguments.rounds,
            basis=arguments.basis,
            start_phase=arguments.start_phase,
            probability=arguments.p,
            shots=arguments.shots,
            seed=arguments.seed,
            output=arguments.output,
            target_protocol=arguments.target_protocol,
        )
    except ValueError as error:
        error_message = str(error)
        parser.error(error_message)
    report_json = json.dumps(report, indent=2)
    print(report_json)


if __name__ == "__main__":
    main()
