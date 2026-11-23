"""Load bundled circuit and model sources and the installed cleanup decoder."""

from functools import lru_cache
import importlib
from pathlib import Path
import sys

from training_data.paths import NOAH_SOURCE_ROOT, NVIDIA_SOURCE_ROOT, PROJECT_ROOT
from training_data.paths import add_source_directory


NOAH_SOURCE_DIRECTORY = NOAH_SOURCE_ROOT / "src"
NVIDIA_SOURCE_DIRECTORY = NVIDIA_SOURCE_ROOT / "code"
NVIDIA_PREDECODER_SOURCE = NVIDIA_SOURCE_DIRECTORY / "model" / "predecoder.py"
TESSERACT_ROOT = PROJECT_ROOT / ".runtime" / "tesseract"


def load_noah_builder():
    """Return Noah's unchanged circuit builder from this checkout."""
    add_source_directory(NOAH_SOURCE_DIRECTORY)
    return importlib.import_module("denser_codes")


def load_nvidia_module(name: str):
    """Return a bundled NVIDIA module, such as model.predecoder or training.optimizers."""
    add_source_directory(NVIDIA_SOURCE_DIRECTORY)
    return importlib.import_module(name)


def load_nvidia_predecoder():
    """Return NVIDIA's configurable Conv3D pre-decoder class from this checkout."""
    predecoder_module = load_nvidia_module("model.predecoder")
    return predecoder_module.PreDecoderModelMemory_v1


@lru_cache(maxsize=1)
def load_tesseract():
    """Return the package with its decoder and ordering modules, preferring the isolated wheel."""
    binary = TESSERACT_ROOT / "tesseract_decoder.so"
    if binary.is_file():
        runtime_path = str(TESSERACT_ROOT)
        if runtime_path not in sys.path:
            sys.path.insert(0, runtime_path)
        loaded_module = sys.modules.get("tesseract_decoder")
        if loaded_module is not None:
            module_path = Path(loaded_module.__file__)
            if module_path != binary:
                raise RuntimeError("Restart Python to load the project-local Tesseract package")
    try:
        return importlib.import_module("tesseract_decoder")
    except ImportError as error:
        message = "Tesseract is unavailable or incompatible. Run scripts/setup_tesseract.sh."
        raise RuntimeError(message) from error
