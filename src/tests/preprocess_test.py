import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from facerec.preprocess import preprocess_image


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "face.pgm"
        image = Image.new("L", (16, 16), color=128)
        image.save(self.image_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preprocess_image_returns_flattened_vector(self):
        vector = preprocess_image(self.image_path, image_size=(8, 8))
        self.assertEqual(vector.shape, (64,))

    def test_preprocess_image_normalizes_to_zero_one_range(self):
        vector = preprocess_image(self.image_path, image_size=(8, 8))
        self.assertTrue(np.all(vector >= 0.0))
        self.assertTrue(np.all(vector <= 1.0))

    def test_preprocess_image_is_deterministic(self):
        vector_a = preprocess_image(self.image_path, image_size=(8, 8))
        vector_b = preprocess_image(self.image_path, image_size=(8, 8))
        self.assertTrue(np.array_equal(vector_a, vector_b))

    def test_preprocess_image_raises_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            preprocess_image(self.root / "missing.pgm")
