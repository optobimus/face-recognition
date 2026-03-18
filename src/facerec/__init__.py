"""Face recognition package."""

from facerec.data import discover_orl_images, split_per_identity
from facerec.pca import PCAState, fit_pca, transform_pca
from facerec.preprocess import preprocess_image

__all__ = [
    "discover_orl_images",
    "split_per_identity",
    "preprocess_image",
    "PCAState",
    "fit_pca",
    "transform_pca",
]
