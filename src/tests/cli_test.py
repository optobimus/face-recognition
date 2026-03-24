import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from facerec.cli import main


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_root = self.root / "orl"
        self._create_identity("s1", count=3, base_value=30)
        self._create_identity("s2", count=3, base_value=220)
        self.model_path = self.root / "model.npz"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_identity(self, label: str, count: int, base_value: int):
        identity_dir = self.dataset_root / label
        identity_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, count + 1):
            image = Image.new("L", (10, 10), color=min(base_value + i, 255))
            image.save(identity_dir / f"{i}.pgm")

    def test_train_command_creates_model_file(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "train",
                    "--dataset-root",
                    str(self.dataset_root),
                    "--model-out",
                    str(self.model_path),
                    "--n-components",
                    "2",
                    "--train-per-identity",
                    "2",
                    "--image-width",
                    "8",
                    "--image-height",
                    "8",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertTrue(self.model_path.exists())
        self.assertIn("trained_samples=", stdout.getvalue())

    def test_predict_command_returns_label_and_distance(self):
        main(
            [
                "train",
                "--dataset-root",
                str(self.dataset_root),
                "--model-out",
                str(self.model_path),
                "--n-components",
                "2",
                "--train-per-identity",
                "2",
                "--image-width",
                "8",
                "--image-height",
                "8",
            ]
        )
        query_image = self.dataset_root / "s1" / "1.pgm"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "predict",
                    "--model",
                    str(self.model_path),
                    "--image",
                    str(query_image),
                ]
            )
        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("label=", output)
        self.assertIn("distance=", output)
