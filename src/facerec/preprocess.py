"""Image preprocessing utilities for face recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

DEFAULT_IMAGE_SIZE: Final[tuple[int, int]] = (64, 64)


def preprocess_image(
    image_path: Path, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Load, grayscale, resize, normalize, and flatten an image."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    with Image.open(path) as img:
        gray = img.convert("L")
        resized = gray.resize(image_size, Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)

    normalized = array / 255.0
    return normalized.reshape(-1)
