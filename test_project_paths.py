"""Check relocated paths and support for a flat repository checkout."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import project_paths


class ProjectPathTests(unittest.TestCase):
    def test_legacy_windows_dataset_path_is_relocated(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "Dataset" / "hrsid_yolo"
            with patch.object(project_paths, "DATASET_ROOT", dataset):
                resolved = project_paths.resolve_saved_path(
                    r"D:\CAPSTONE_HYPOTHESIS\hrsid_yolo\test\images\ship.png")
            self.assertEqual(resolved, dataset / "test" / "images" / "ship.png")

    def test_existing_custom_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom.yaml"
            custom.write_text("names: [ship]", encoding="utf-8")
            self.assertEqual(project_paths.resolve_saved_path(custom), custom)

    def test_root_is_independent_of_working_directory_and_code_layout(self):
        source = Path(project_paths.__file__).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for folder in (root, root / "useful_ahh"):
                folder.mkdir(exist_ok=True)
                module_path = folder / "project_paths.py"
                module_path.write_text(source, encoding="utf-8")
                spec = importlib.util.spec_from_file_location("paths_under_test", module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertEqual(module.ROOT, root)
                self.assertEqual(module.DATASET_ROOT, root / "Dataset" / "hrsid_yolo")
                self.assertEqual(module.RUNS_ROOT, root / "useless_ahh" / "runs")


if __name__ == "__main__":
    unittest.main()
