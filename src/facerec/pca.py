"""PCA utilities for Eigenfaces-style dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAState:
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray


def fit_pca(x: np.ndarray, n_components: int) -> PCAState:
    """Fit PCA with SVD and return the fitted state"""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (n_samples, n_features)")
    n_samples, n_features = x.shape
    if n_samples < 2:
        raise ValueError("At least 2 samples are required to fit PCA")
    max_components = min(n_samples, n_features)
    if n_components < 1 or n_components > max_components:
        raise ValueError(
            f"n_components must be in [1, {max_components}], got {n_components}"
        )

    mean = x.mean(axis=0, dtype=np.float64)
    centered = x - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    explained_variance = (singular_values**2) / (n_samples - 1)

    return PCAState(
        mean=mean,
        components=components,
        singular_values=singular_values[:n_components],
        explained_variance=explained_variance[:n_components],
    )


def transform_pca(x: np.ndarray, state: PCAState) -> np.ndarray:
    """Project samples into the PCA component space."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (n_samples, n_features)")
    if x.shape[1] != state.mean.shape[0]:
        raise ValueError(
            "x has incompatible feature dimension for the provided PCA state"
        )
    centered = x - state.mean
    return centered @ state.components.T
