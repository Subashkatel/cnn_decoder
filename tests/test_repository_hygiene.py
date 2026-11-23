"""The published history contains source, documentation, and empty-output notebooks."""

from pathlib import Path
import json
import subprocess
import unittest

from training_data.paths import PROJECT_ROOT


ROOT_SOURCE_FILES = {
    ".gitignore", "README.md", "requirements.txt", "detector_playground.ipynb",
    "LICENSE", "NOTICE", "CONTRIBUTING.md",
}
SOURCE_DIRECTORIES = {"training_data", "tests", "scripts", "verification"}
SOURCE_EXTENSIONS = {".py", ".md", ".sh", ".sbatch"}


def git_lines(*arguments):
    """Inspect the public branch's ancestry rather than local private backup refs."""
    output = subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True,
    )
    return output.splitlines()


def is_public_source(name):
    if name in ROOT_SOURCE_FILES:
        return True
    path = Path(name)
    if path.parts[0] not in SOURCE_DIRECTORIES:
        return False
    return path.suffix in SOURCE_EXTENSIONS


class RepositoryHygieneTests(unittest.TestCase):
    def test_public_ancestry_contains_only_source_files(self):
        rejected = set()
        head_names = git_lines("ls-tree", "--full-tree", "-r", "--name-only", "HEAD")
        self.assertIn("training_data/planar_488/generator.py", head_names)
        for revision in git_lines("rev-list", "HEAD"):
            names = git_lines("ls-tree", "--full-tree", "-r", "--name-only", revision)
            for name in names:
                if not is_public_source(name):
                    rejected.add(name)
        self.assertEqual(sorted(rejected), [], "Generated data or internal artifacts are in public history")

    def test_notebook_has_no_saved_outputs(self):
        path = PROJECT_ROOT / "detector_playground.ipynb"
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
