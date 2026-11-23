"""Atomic experiment reports, recorded job attempts, and reproducible content hashes."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import traceback


def write_report(path: Path, report: dict) -> None:
    """Publish a complete JSON report so readers never observe a partial write."""
    serialized = json.dumps(report, indent=2, allow_nan=False)
    report_text = serialized + "\n"
    temporary_path = path.with_suffix(".partial")
    temporary_path.write_text(report_text)
    temporary_path.replace(path)


@contextmanager
def recorded_attempt(path: Path, report: dict):
    """Publish the report however a job attempt ends, with its duration and any traceback."""
    started = time.monotonic()
    try:
        yield
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        elapsed = time.monotonic() - started
        report["elapsed_seconds"] = elapsed
        write_report(path, report)


def text_hash(value) -> str:
    serialized = str(value)
    encoded = serialized.encode()
    digest = hashlib.sha256(encoded)
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    content = path.read_bytes()
    digest = hashlib.sha256(content)
    return digest.hexdigest()


def source_tree_hash(directory: Path) -> str:
    """Hash source paths and contents without requiring a bundled archive.

    Bytecode caches are machine-generated and do not identify the source.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Source directory is missing: {directory}")
    digest = hashlib.sha256()
    source_paths = sorted(directory.rglob("*"))
    for source_path in source_paths:
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(directory)
        if "__pycache__" in relative_path.parts:
            continue
        if source_path.suffix in (".pyc", ".pyo"):
            continue
        source_name = relative_path.as_posix()
        encoded_name = source_name.encode("utf-8")
        content_hash = file_hash(source_path)
        encoded_hash = content_hash.encode("ascii")
        digest.update(encoded_name)
        digest.update(b"\0")
        digest.update(encoded_hash)
        digest.update(b"\0")
    return digest.hexdigest()
