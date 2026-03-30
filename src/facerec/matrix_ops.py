"""Basic matrix and vector operations implemented without linear algebra libraries."""

from __future__ import annotations

import math

import numpy as np


def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product of two equal-length vectors."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("vector_dot expects 1D vectors")
    if a.shape[0] != b.shape[0]:
        raise ValueError("vector_dot expects equal-length vectors")
    total = 0.0
    for idx in range(a.shape[0]):
        total += float(a[idx]) * float(b[idx])
    return total


def vector_l2_norm(a: np.ndarray) -> float:
    """Compute L2 norm of a vector."""
    return math.sqrt(vector_dot(a, a))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two equal-length vectors."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("euclidean_distance expects 1D vectors")
    if a.shape[0] != b.shape[0]:
        raise ValueError("euclidean_distance expects equal-length vectors")
    squared_sum = 0.0
    for idx in range(a.shape[0]):
        diff = float(a[idx]) - float(b[idx])
        squared_sum += diff * diff
    return math.sqrt(squared_sum)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    denom = max(vector_l2_norm(a) * vector_l2_norm(b), 1e-12)
    cosine_sim = vector_dot(a, b) / denom
    return 1.0 - cosine_sim


def argmin_index(values: np.ndarray) -> int:
    """Return the index of the smallest value in a non-empty 1D array."""
    if values.ndim != 1:
        raise ValueError("argmin_index expects a 1D array")
    if values.shape[0] == 0:
        raise ValueError("argmin_index expects a non-empty array")
    best_idx = 0
    best_value = float(values[0])
    for idx in range(1, values.shape[0]):
        current = float(values[idx])
        if current < best_value:
            best_value = current
            best_idx = idx
    return best_idx
