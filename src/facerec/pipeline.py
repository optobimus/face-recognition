"""Training and inference pipeline for Eigenfaces + nearest neighbor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from facerec.knn import MetricName, predict_nearest_neighbor
from facerec.pca import PCAState, fit_pca, transform_pca


@dataclass(frozen=True)
class EigenfaceModel:
    pca: PCAState
    gallery_embeddings: np.ndarray
    gallery_labels: np.ndarray
    metric: MetricName
    image_size: tuple[int, int]


def train_model_from_vectors(
    x: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    metric: MetricName = "euclidean",
    image_size: tuple[int, int] = (64, 64),
) -> EigenfaceModel:
    """Train Eigenfaces model from feature vectors and labels."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (n_samples, n_features)")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if x.shape[0] != labels.shape[0]:
        raise ValueError("x sample count and labels length must match")

    pca_state = fit_pca(x, n_components=n_components)
    gallery_embeddings = transform_pca(x, pca_state)

    return EigenfaceModel(
        pca=pca_state,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=labels.astype(str),
        metric=metric,
        image_size=image_size,
    )


def embed_vectors(model: EigenfaceModel, x: np.ndarray) -> np.ndarray:
    """Project input vectors into the trained PCA space."""
    return transform_pca(x, model.pca)


def predict_from_vector(
    model: EigenfaceModel, vector: np.ndarray
) -> tuple[str, float, int]:
    """Predict nearest label for one input vector."""
    if vector.ndim != 1:
        raise ValueError("vector must be a 1D array")

    embedded = embed_vectors(model, vector.reshape(1, -1))
    return predict_nearest_neighbor(
        embedded[0], model.gallery_embeddings, model.gallery_labels, metric=model.metric
    )
