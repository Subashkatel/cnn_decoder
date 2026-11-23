"""Exercise command selection and reject invalid runs before creating outputs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from training_data.paths import PROJECT_ROOT


class CommandTests(unittest.TestCase):
    def test_gpu_cleanup_help_does_not_import_optional_gpu_libraries(self):
        script = '''
import runpy
import sys
sys.modules['cupy'] = None
sys.modules['cudaq_qec'] = None
sys.argv = ['compare_gpu_cleanup', '--help']
runpy.run_module('training_data.planar_488.jobs.compare_gpu_cleanup', run_name='__main__')
'''
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout)

    def run_command(self, arguments, block_torch=False):
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if block_torch:
            script = (
                "import sys; sys.modules['torch'] = None; "
                "from training_data.__main__ import main; "
                f"main({arguments!r})"
            )
            command = [sys.executable, "-B", "-c", script]
        else:
            command = [sys.executable, "-B", "-m", "training_data", *arguments]
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_help_does_not_require_torch(self):
        for arguments in (["--help"], ["noah-training", "--help"]):
            with self.subTest(arguments=arguments):
                completed = self.run_command(arguments, block_torch=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout)

    def test_noah_training_generation_does_not_require_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["noah-training", "--rounds", "1", "--shots", "1", "--output", directory]
            completed = self.run_command(arguments, block_torch=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["input_shape"][1], 6)
            self.assertEqual(summary["target_shape"][1], 4)
