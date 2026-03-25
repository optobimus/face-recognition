import tempfile
import unittest
from pathlib import Path

from facerec.model_io import load_model


class TestModelIO(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_model_raises_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_model(self.root / "missing_model.npz")
