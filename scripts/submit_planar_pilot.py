"""Submit GPU training and dependent CPU evaluation from a Della login terminal."""

import json
from pathlib import Path
import subprocess


_script_path = Path(__file__)
_resolved_script_path = _script_path.resolve()
PROJECT = _resolved_script_path.parents[1]


def submit_job(arguments) -> str:
    command = ["sbatch", "--parsable", *arguments]
    receipt = subprocess.run(
        command,
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=True,
    )
    response = receipt.stdout.strip()
    # sbatch --parsable prints the job ID, followed by ";cluster" on multi-cluster systems.
    job_id, _, _ = response.partition(";")
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected Slurm submission receipt: {response}")
    return job_id


def main():
    training_id = submit_job(["scripts/train_planar_pilot.sbatch"])
    training_directory = PROJECT / "verification" / "pilot_runs" / training_id
    print("Submitted GPU training", training_id, flush=True)
    evaluation_arguments = [
        f"--dependency=afterok:{training_id}",
        "--kill-on-invalid-dep=yes",
        "scripts/evaluate_planar_pilot.sbatch",
        str(training_directory),
    ]
    try:
        evaluation_id = submit_job(evaluation_arguments)
    except Exception:
        print(
            "CPU evaluation submission failed. The GPU training job is already submitted.",
            flush=True,
        )
        print(
            "Submit only evaluation after inspecting that job; do not duplicate GPU training.",
            flush=True,
        )
        raise
    receipt = {
        "training_job": training_id,
        "evaluation_job": evaluation_id,
        "training_output": str(training_directory),
    }
    receipt_path = PROJECT / "verification" / f"submission-{training_id}.json"
    serialized = json.dumps(receipt, indent=2)
    receipt_text = serialized + "\n"
    receipt_path.write_text(receipt_text)
    print("Submitted dependent CPU evaluation", evaluation_id, flush=True)


if __name__ == "__main__":
    main()
