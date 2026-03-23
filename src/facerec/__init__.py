"""Face recognition package."""

from facerec.data import discover_orl_images, split_per_identity
from facerec.knn import predict_nearest_neighbor
from facerec.pca import PCAState, fit_pca, transform_pca
from facerec.preprocess import preprocess_image

__all__ = [
    "discover_orl_images",
    "split_per_identity",
    "preprocess_image",
    "predict_nearest_neighbor",
    "PCAState",
    "fit_pca",
    "transform_pca",
]
