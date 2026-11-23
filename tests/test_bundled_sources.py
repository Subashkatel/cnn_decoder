"""Required upstream implementations travel with every source checkout."""

from pathlib import Path
import hashlib
import tomllib
import unittest

from training_data.paths import PROJECT_ROOT
from training_data.planar_488.dependencies import load_noah_builder, load_nvidia_module


class BundledSourceTests(unittest.TestCase):
    def test_circuit_and_model_sources_are_bundled(self):
        source_root = PROJECT_ROOT / "third_party"
        modules = [
            load_noah_builder(),
            load_nvidia_module("model.predecoder"),
            load_nvidia_module("training.optimizers"),
        ]
        for module in modules:
            with self.subTest(module=module.__name__):
                path = Path(module.__file__).resolve()
                self.assertTrue(path.is_relative_to(source_root), str(path))

    def test_upstream_sources_match_the_recorded_originals(self):
        source_root = PROJECT_ROOT / "third_party"
        manifest_path = source_root / "sources.toml"
        manifest = tomllib.loads(manifest_path.read_text())
        for record in manifest["files"]:
            with self.subTest(path=record["path"]):
                path = source_root / record["path"]
                contents = path.read_bytes()
                digest = hashlib.sha256(contents).hexdigest()
                self.assertEqual(digest, record["sha256"])
