"""Nearest-neighbor classification utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np

MetricName = Literal["euclidean", "cosine"]


def euclidean_distances(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances from query to each gallery vector."""
    diff = gallery - query
    return np.linalg.norm(diff, axis=1)


def cosine_distances(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Compute cosine distances from query to each gallery vector."""
    query_norm = np.linalg.norm(query)
    gallery_norms = np.linalg.norm(gallery, axis=1)
    denom = np.maximum(query_norm * gallery_norms, 1e-12)
    cosine_sim = (gallery @ query) / denom
    return 1.0 - cosine_sim


def predict_nearest_neighbor(
    query: np.ndarray,
    gallery: np.ndarray,
    labels: np.ndarray,
    metric: MetricName = "euclidean",
) -> tuple[str, float, int]:
    """Predict the nearest label for a query embedding."""
    if query.ndim != 1:
        raise ValueError("query must be a 1D vector")
    if gallery.ndim != 2:
        raise ValueError("gallery must be a 2D array")
    if gallery.shape[0] == 0:
        raise ValueError("gallery must contain at least one sample")
    if gallery.shape[1] != query.shape[0]:
        raise ValueError("query and gallery feature dimensions must match")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if labels.shape[0] != gallery.shape[0]:
        raise ValueError("labels length must match gallery sample count")

    if metric == "euclidean":
        distances = euclidean_distances(query, gallery)
    elif metric == "cosine":
        distances = cosine_distances(query, gallery)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    best_index = int(np.argmin(distances))
    return str(labels[best_index]), float(distances[best_index]), best_index
