"""Freeze source, then submit compact GPU training and dependent CPU comparison."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_environment():
    """Reject an incomplete install before reserving cluster resources."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Activate the Python 3.11 .venv created by scripts/setup.sh")
    version = importlib.metadata.version("torch")
    if version.endswith("+cpu"):
        raise RuntimeError("GPU training requires scripts/setup.sh --cuda")
    for package in ("numpy", "stim", "scipy", "ldpc", "sinter", "pymatching"):
        importlib.metadata.version(package)
    subprocess.run(
        [sys.executable, "-B", "-c",
         "from training_data.planar_488.dependencies import load_tesseract; load_tesseract()"],
        cwd=PROJECT_ROOT, check=True,
    )


def freeze_source(destination):
    """Copy tracked working files so later edits cannot change a queued experiment."""
    names = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=PROJECT_ROOT,
    )
    file_hashes = {}
    destination.mkdir(parents=True, exist_ok=False)
    for encoded_name in names.split(b"\0"):
        if not encoded_name:
            continue
        name = os.fsdecode(encoded_name)
        source = PROJECT_ROOT / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        contents = target.read_bytes()
        file_hashes[name] = hashlib.sha256(contents).hexdigest()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    )
    snapshot = {"revision": revision.strip(), "files_sha256": file_hashes}
    manifest = destination / "source_snapshot.json"
    manifest.write_text(json.dumps(snapshot, indent=2) + "\n")
    decoder_runtime = PROJECT_ROOT / ".runtime/tesseract"
    if not decoder_runtime.is_dir():
        raise FileNotFoundError("Run scripts/setup.sh before submitting the comparison")
    local_runtime = destination / ".runtime"
    local_runtime.mkdir()
    (local_runtime / "tesseract").symlink_to(decoder_runtime, target_is_directory=True)
    return snapshot


def submit(arguments, environment):
    receipt = subprocess.check_output(
        ["sbatch", "--parsable", *arguments], cwd=PROJECT_ROOT,
        env=environment, text=True,
    )
    job_id = receipt.strip().partition(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected Slurm job receipt: {receipt}")
    return job_id


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / ".runtime/experiments" / timestamp)
    parser.add_argument("--seed-base", type=int, default=secrets.randbits(40))
    arguments = parser.parse_args()
    if not 0 <= arguments.seed_base < 2 ** 64 - 500000:
        parser.error("Seed base must leave room for independent sample streams")
    check_environment()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    snapshot = freeze_source(source)
    environment = dict(os.environ)
    environment["TRAINING_PYTHON"] = sys.executable
    training_output = output / "training"
    prepared_output = output / "prepared"
    comparison_output = output / "comparison"
    receipt = {
        "source_revision": snapshot["revision"], "seed_base": arguments.seed_base,
        "source": str(source), "training_output": str(training_output),
        "prepared_output": str(prepared_output), "comparison_output": str(comparison_output),
        "python": sys.executable,
    }
    receipt_path = output / "submission.json"
    training_id = submit([
        "--output=" + str(output / "training-%j.out"),
        "--error=" + str(output / "training-%j.err"),
        str(source / "scripts/train_compare.sbatch"),
        str(source), str(training_output), str(prepared_output), str(arguments.seed_base),
    ], environment)
    receipt["training_job"] = training_id
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("GPU training submitted:", training_id, flush=True)
    comparison_arguments = [
        "--dependency=afterok:" + training_id, "--kill-on-invalid-dep=yes",
        "--output=" + str(output / "comparison-%j.out"),
        "--error=" + str(output / "comparison-%j.err"),
        str(source / "scripts/compare_cleanup.sbatch"),
        str(source), str(prepared_output), str(comparison_output),
    ]
    try:
        comparison_id = submit(comparison_arguments, environment)
    except subprocess.CalledProcessError:
        print("GPU training is already submitted as", training_id, file=sys.stderr)
        print("Submit only the missing CPU comparison; keep the receipt:", receipt_path, file=sys.stderr)
        raise
    receipt["comparison_job"] = comparison_id
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Dependent one-CPU comparison submitted:", comparison_id, flush=True)
    print("Run directory:", output, flush=True)


if __name__ == "__main__":
    main()
