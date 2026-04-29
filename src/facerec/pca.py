"""PCA utilities for Eigenfaces-style dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from facerec.matrix_ops import (
    center_matrix,
    column_means,
    sample_covariance_matrix,
    scale_vector,
    top_k_eigenpairs_symmetric,
    transpose_matrix_vector_multiply,
    vector_dot,
    vector_l2_norm,
)


@dataclass(frozen=True)
class PCAState:
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray


def fit_pca(x: np.ndarray, n_components: int) -> PCAState:
    """Fit PCA with covariance eigendecomposition and return the fitted state."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (n_samples, n_features)")

    n_samples, n_features = x.shape
    if n_samples < 2:
        raise ValueError("At least 2 samples are required to fit PCA")

    max_components = min(n_samples - 1, n_features)
    if n_components < 1 or n_components > max_components:
        raise ValueError(
            f"n_components must be in [1, {max_components}], got {n_components}"
        )

    mean = column_means(x)
    centered = center_matrix(x, mean)
    sample_cov = sample_covariance_matrix(centered, denominator=n_samples - 1)
    explained_variance, sample_eigenvectors = top_k_eigenpairs_symmetric(
        sample_cov, n_components
    )

    singular_values = [0.0 for _ in range(n_components)]
    components = [[0.0 for _ in range(n_features)] for _ in range(n_components)]
    for idx in range(n_components):
        eig = max(float(explained_variance[idx]), 0.0)
        singular_values[idx] = math.sqrt(eig * (n_samples - 1))
        raw_component = transpose_matrix_vector_multiply(
            centered, sample_eigenvectors[idx]
        )

        if singular_values[idx] >= 1e-12:
            component = scale_vector(raw_component, 1.0 / singular_values[idx])
        else:
            component = raw_component

        component_norm = vector_l2_norm(component)
        if component_norm >= 1e-12:
            component = scale_vector(component, 1.0 / component_norm)

        components[idx] = component

    return PCAState(
        mean=np.asarray(mean, dtype=np.float64),
        components=np.asarray(components, dtype=np.float64),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        explained_variance=np.asarray(explained_variance, dtype=np.float64),
    )


def transform_pca(x: np.ndarray, state: PCAState) -> np.ndarray:
    """Project samples into the PCA component space."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (n_samples, n_features)")

    if x.shape[1] != state.mean.shape[0]:
        raise ValueError(
            "x has incompatible feature dimension for the provided PCA state"
        )

    centered = center_matrix(x, state.mean)
    n_samples = len(centered)
    components = state.components.tolist()
    n_components = len(components)

    projected = [[0.0 for _ in range(n_components)] for _ in range(n_samples)]
    for row in range(n_samples):
        for comp_idx in range(n_components):
            projected[row][comp_idx] = vector_dot(
                centered[row], components[comp_idx]
            )
    return np.asarray(projected, dtype=np.float64)
