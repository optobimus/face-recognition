"""Face recognition package."""

from facerec.data import discover_orl_images, split_per_identity
from facerec.preprocess import preprocess_image

__all__ = ["discover_orl_images", "split_per_identity", "preprocess_image"]
