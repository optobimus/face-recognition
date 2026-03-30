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


def scale_vector(vector: np.ndarray, scalar: float) -> np.ndarray:
    """Return a scaled copy of a 1D vector."""
    if vector.ndim != 1:
        raise ValueError("scale_vector expects a 1D vector")
    result = np.zeros(vector.shape[0], dtype=np.float64)
    for idx in range(vector.shape[0]):
        result[idx] = float(vector[idx]) * scalar
    return result


def column_means(matrix: np.ndarray) -> np.ndarray:
    """Compute mean value for each column in a 2D matrix."""
    if matrix.ndim != 2:
        raise ValueError("column_means expects a 2D matrix")
    n_rows, n_cols = matrix.shape
    means = np.zeros(n_cols, dtype=np.float64)
    for col in range(n_cols):
        total = 0.0
        for row in range(n_rows):
            total += float(matrix[row, col])
        means[col] = total / n_rows
    return means


def center_matrix(matrix: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Center each row of a matrix by subtracting column means."""
    if matrix.ndim != 2:
        raise ValueError("center_matrix expects a 2D matrix")
    if means.ndim != 1:
        raise ValueError("center_matrix expects a 1D means vector")
    n_rows, n_cols = matrix.shape
    if means.shape[0] != n_cols:
        raise ValueError("means length must match matrix column count")

    centered = np.zeros((n_rows, n_cols), dtype=np.float64)
    for row in range(n_rows):
        for col in range(n_cols):
            centered[row, col] = float(matrix[row, col]) - float(means[col])
    return centered


def covariance_matrix(centered: np.ndarray, denominator: int) -> np.ndarray:
    """Compute covariance matrix from centered samples."""
    if centered.ndim != 2:
        raise ValueError("covariance_matrix expects a 2D matrix")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    n_rows, n_cols = centered.shape
    cov = np.zeros((n_cols, n_cols), dtype=np.float64)
    for i in range(n_cols):
        for j in range(i, n_cols):
            total = 0.0
            for row in range(n_rows):
                total += float(centered[row, i]) * float(centered[row, j])
            value = total / denominator
            cov[i, j] = value
            cov[j, i] = value
    return cov


def matrix_vector_multiply(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Multiply a 2D matrix by a 1D vector."""
    if matrix.ndim != 2:
        raise ValueError("matrix_vector_multiply expects a 2D matrix")
    if vector.ndim != 1:
        raise ValueError("matrix_vector_multiply expects a 1D vector")
    n_rows, n_cols = matrix.shape
    if n_cols != vector.shape[0]:
        raise ValueError("matrix column count must match vector length")

    result = np.zeros(n_rows, dtype=np.float64)
    for row in range(n_rows):
        total = 0.0
        for col in range(n_cols):
            total += float(matrix[row, col]) * float(vector[col])
        result[row] = total
    return result


def power_iteration_symmetric(
    matrix: np.ndarray,
    init_vector: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> tuple[float, np.ndarray]:
    """Approximate dominant eigenpair of a symmetric matrix."""
    if matrix.ndim != 2:
        raise ValueError("power_iteration_symmetric expects a 2D matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("power_iteration_symmetric expects a square matrix")
    if init_vector.ndim != 1:
        raise ValueError("init_vector must be 1D")
    if init_vector.shape[0] != matrix.shape[0]:
        raise ValueError("init_vector length must match matrix size")

    vector = np.array(init_vector, dtype=np.float64)
    norm = vector_l2_norm(vector)
    if norm < 1e-12:
        vector = np.ones(matrix.shape[0], dtype=np.float64)
        norm = vector_l2_norm(vector)
    vector = scale_vector(vector, 1.0 / norm)

    for _ in range(max_iter):
        multiplied = matrix_vector_multiply(matrix, vector)
        multiplied_norm = vector_l2_norm(multiplied)
        if multiplied_norm < 1e-12:
            break
        next_vector = scale_vector(multiplied, 1.0 / multiplied_norm)
        delta = euclidean_distance(next_vector, vector)
        vector = next_vector
        if delta < tol:
            break

    rayleigh_num = vector_dot(vector, matrix_vector_multiply(matrix, vector))
    rayleigh_den = max(vector_dot(vector, vector), 1e-12)
    eigenvalue = rayleigh_num / rayleigh_den
    return eigenvalue, vector


def deflate_symmetric(
    matrix: np.ndarray, eigenvalue: float, eigenvector: np.ndarray
) -> np.ndarray:
    """Deflate symmetric matrix by one eigenpair."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("deflate_symmetric expects a square matrix")
    if eigenvector.ndim != 1 or eigenvector.shape[0] != matrix.shape[0]:
        raise ValueError("eigenvector length must match matrix size")

    result = np.array(matrix, dtype=np.float64, copy=True)
    size = matrix.shape[0]
    for i in range(size):
        for j in range(size):
            result[i, j] -= eigenvalue * float(eigenvector[i]) * float(eigenvector[j])
    return result


def top_k_eigenpairs_symmetric(
    matrix: np.ndarray,
    k: int,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-k eigenpairs of a symmetric matrix by power iteration + deflation."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("top_k_eigenpairs_symmetric expects a square matrix")
    size = matrix.shape[0]
    if k < 1 or k > size:
        raise ValueError(f"k must be in [1, {size}]")

    working = np.array(matrix, dtype=np.float64, copy=True)
    eigenvalues = np.zeros(k, dtype=np.float64)
    eigenvectors = np.zeros((k, size), dtype=np.float64)

    for idx in range(k):
        init = np.ones(size, dtype=np.float64)
        init[idx % size] = 2.0
        value, vector = power_iteration_symmetric(
            working, init_vector=init, max_iter=max_iter, tol=tol
        )
        if value < 0.0 and abs(value) < 1e-10:
            value = 0.0
        eigenvalues[idx] = value
        eigenvectors[idx] = vector
        working = deflate_symmetric(working, value, vector)

    return eigenvalues, eigenvectors
