import tempfile
import unittest
from pathlib import Path

from PIL import Image

from facerec.data import discover_orl_images, split_per_identity


class TestData(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self._create_identity("s1", count=4, base_value=20)
        self._create_identity("s2", count=4, base_value=40)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_identity(self, label: str, count: int, base_value: int):
        identity_dir = self.root / label
        identity_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, count + 1):
            image = Image.new("L", (8, 8), color=base_value + i)
            image.save(identity_dir / f"{i}.pgm")

    def test_discover_orl_images_finds_all_images(self):
        items = discover_orl_images(self.root)
        self.assertEqual(len(items), 8)
        labels = sorted(label for _, label in items)
        self.assertEqual(labels.count("s1"), 4)
        self.assertEqual(labels.count("s2"), 4)

    def test_discover_orl_images_raises_for_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            discover_orl_images(self.root / "missing")

    def test_split_per_identity_returns_expected_counts(self):
        items = discover_orl_images(self.root)
        train, test = split_per_identity(items, train_per_identity=2, seed=42)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 4)
        train_labels = sorted(label for _, label in train)
        test_labels = sorted(label for _, label in test)
        self.assertEqual(train_labels, ["s1", "s1", "s2", "s2"])
        self.assertEqual(test_labels, ["s1", "s1", "s2", "s2"])

    def test_split_per_identity_is_deterministic_for_same_seed(self):
        items = discover_orl_images(self.root)
        train_a, test_a = split_per_identity(items, train_per_identity=2, seed=7)
        train_b, test_b = split_per_identity(items, train_per_identity=2, seed=7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(test_a, test_b)

    def test_split_per_identity_raises_when_train_size_too_large(self):
        items = discover_orl_images(self.root)
        with self.assertRaises(ValueError):
            split_per_identity(items, train_per_identity=4, seed=1)
