"""Face recognition package."""

from facerec.data import discover_orl_images, split_per_identity
from facerec.knn import predict_nearest_neighbor
from facerec.model_io import load_model, save_model
from facerec.pca import PCAState, fit_pca, transform_pca
from facerec.pipeline import (
    EigenfaceModel,
    embed_vectors,
    predict_from_vector,
    train_model_from_vectors,
)
from facerec.preprocess import preprocess_image

__all__ = [
    "discover_orl_images",
    "split_per_identity",
    "preprocess_image",
    "predict_nearest_neighbor",
    "save_model",
    "load_model",
    "PCAState",
    "fit_pca",
    "transform_pca",
    "EigenfaceModel",
    "train_model_from_vectors",
    "embed_vectors",
    "predict_from_vector",
]
