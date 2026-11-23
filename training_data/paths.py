"""Project locations and the bundled upstream source boundary."""

from pathlib import Path
import sys


_module_path = Path(__file__)
_resolved_module_path = _module_path.resolve()
PACKAGE_ROOT = _resolved_module_path.parent
PROJECT_ROOT = _resolved_module_path.parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / ".runtime" / "examples"
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
NVIDIA_SOURCE_ROOT = THIRD_PARTY_ROOT / "nvidia"
NOAH_SOURCE_ROOT = THIRD_PARTY_ROOT / "noah"


def add_source_directory(directory: Path) -> None:
    """Expose a source snapshot without installing or changing the snapshot."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Source directory is missing: {directory}")
    source_path = str(directory)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
