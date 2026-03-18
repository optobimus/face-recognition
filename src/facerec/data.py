"""Dataset indexing and deterministic train/test splitting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

ImageItem = tuple[Path, str]
SUPPORTED_IMAGE_EXTENSIONS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp"}


def discover_orl_images(dataset_root: Path) -> list[ImageItem]:
    """Discover ORL-style labeled images as (path, label) tuples."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    items: list[ImageItem] = []
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = identity_dir.name
        image_files = sorted(
            (
                p
                for p in identity_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=lambda p: (p.stem, p.name),
        )
        for image_path in image_files:
            items.append((image_path, label))

    if not items:
        raise ValueError(f"No supported image files found under {root}")
    return items


def split_per_identity(
    items: list[ImageItem], train_per_identity: int, seed: int
) -> tuple[list[ImageItem], list[ImageItem]]:
    """Split items per identity into train and test sets"""
    if train_per_identity < 1:
        raise ValueError("train_per_identity must be at least 1")
    if not items:
        raise ValueError("Cannot split an empty item list")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path, label in items:
        grouped[label].append(path)

    rng = random.Random(seed)
    train: list[ImageItem] = []
    test: list[ImageItem] = []

    for label in sorted(grouped.keys()):
        paths = sorted(grouped[label], key=lambda p: (p.stem, p.name))
        if len(paths) <= train_per_identity:
            raise ValueError(
                f"Label '{label}' has {len(paths)} images, which is not enough for "
                f"train_per_identity={train_per_identity} with a non-empty test split"
            )
        shuffled = paths[:]
        rng.shuffle(shuffled)
        for path in shuffled[:train_per_identity]:
            train.append((path, label))
        for path in shuffled[train_per_identity:]:
            test.append((path, label))

    return train, test
