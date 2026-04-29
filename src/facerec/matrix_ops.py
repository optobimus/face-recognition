"""Basic matrix and vector operations implemented without linear algebra libraries."""

from __future__ import annotations

import math
from typing import Any


def _is_iterable_value(value: Any) -> bool:
    return hasattr(value, "__iter__") and not isinstance(value, (str, bytes))


def _to_vector(values: Any, name: str) -> list[float]:
    if not _is_iterable_value(values):
        raise ValueError(f"{name} expects 1D vectors")

    raw = values.tolist() if hasattr(values, "tolist") else list(values)
    if len(raw) == 0:
        return []
    if _is_iterable_value(raw[0]):
        raise ValueError(f"{name} expects 1D vectors")
    return [float(item) for item in raw]


def _to_matrix(values: Any, name: str) -> list[list[float]]:
    if not _is_iterable_value(values):
        raise ValueError(f"{name} expects a 2D matrix")

    raw_rows = values.tolist() if hasattr(values, "tolist") else list(values)
    if len(raw_rows) == 0:
        raise ValueError(f"{name} expects a 2D matrix")
    if not _is_iterable_value(raw_rows[0]):
        raise ValueError(f"{name} expects a 2D matrix")

    matrix: list[list[float]] = []
    expected_len: int | None = None
    for raw_row in raw_rows:
        if not _is_iterable_value(raw_row):
            raise ValueError(f"{name} expects a 2D matrix")

        row = raw_row.tolist() if hasattr(raw_row, "tolist") else list(raw_row)
        if expected_len is None:
            expected_len = len(row)
        elif expected_len != len(row):
            raise ValueError("matrix rows must have equal length")

        converted_row: list[float] = []
        for item in row:
            if _is_iterable_value(item):
                raise ValueError(f"{name} expects a 2D matrix")
            converted_row.append(float(item))
        matrix.append(converted_row)
    return matrix


def vector_dot(a: Any, b: Any) -> float:
    """Compute dot product of two equal-length vectors."""
    vector_a = _to_vector(a, "vector_dot")
    vector_b = _to_vector(b, "vector_dot")

    if len(vector_a) != len(vector_b):
        raise ValueError("vector_dot expects equal-length vectors")

    total = 0.0
    for value_a, value_b in zip(vector_a, vector_b):
        total += value_a * value_b
    return total


def vector_l2_norm(a: Any) -> float:
    """Compute L2 norm of a vector."""
    return math.sqrt(vector_dot(a, a))


def euclidean_distance(a: Any, b: Any) -> float:
    """Compute Euclidean distance between two equal-length vectors."""
    vector_a = _to_vector(a, "euclidean_distance")
    vector_b = _to_vector(b, "euclidean_distance")

    if len(vector_a) != len(vector_b):
        raise ValueError("euclidean_distance expects equal-length vectors")

    squared_sum = 0.0
    for value_a, value_b in zip(vector_a, vector_b):
        diff = value_a - value_b
        squared_sum += diff * diff
    return math.sqrt(squared_sum)


def cosine_distance(a: Any, b: Any) -> float:
    """Compute cosine distance between two vectors."""
    denom = max(vector_l2_norm(a) * vector_l2_norm(b), 1e-12)
    cosine_sim = vector_dot(a, b) / denom
    return 1.0 - cosine_sim


def argmin_index(values: Any) -> int:
    """Return the index of the smallest value in a non-empty 1D array."""
    vector = _to_vector(values, "argmin_index")

    if len(vector) == 0:
        raise ValueError("argmin_index expects a non-empty array")

    best_idx = 0
    best_value = vector[0]
    for idx in range(1, len(vector)):
        current = vector[idx]
        if current < best_value:
            best_value = current
            best_idx = idx
    return best_idx


def scale_vector(vector: Any, scalar: float) -> list[float]:
    """Return a scaled copy of a 1D vector."""
    data = _to_vector(vector, "scale_vector")
    return [value * scalar for value in data]


def orthogonalize_vector(vector: Any, basis: Any) -> list[float]:
    """Orthogonalize a vector against a basis using Gram-Schmidt."""
    result = _to_vector(vector, "orthogonalize_vector")
    basis_vectors = _to_matrix(basis, "orthogonalize_vector")

    if len(basis_vectors) == 0:
        return result
    if any(len(basis_vector) != len(result) for basis_vector in basis_vectors):
        raise ValueError("basis vector length must match target vector length")

    for basis_vector in basis_vectors:
        basis_norm_sq = max(vector_dot(basis_vector, basis_vector), 1e-12)
        projection_scale = vector_dot(result, basis_vector) / basis_norm_sq
        for idx, value in enumerate(result):
            result[idx] = value - projection_scale * basis_vector[idx]
    return result


def _optional_basis_vectors(values: Any | None, name: str) -> list[list[float]]:
    if values is None:
        return []

    raw_rows = values.tolist() if hasattr(values, "tolist") else list(values)
    if len(raw_rows) == 0:
        return []

    return _to_matrix(raw_rows, name)


def _validated_orthogonal_basis(values: Any | None, size: int) -> list[list[float]]:
    basis_vectors = _optional_basis_vectors(values, "orthogonal_basis")
    if any(len(basis_vector) != size for basis_vector in basis_vectors):
        raise ValueError("basis vector length must match matrix size")
    return basis_vectors


def _normalized_orthogonal_vector(
    vector: Any,
    basis_vectors: list[list[float]],
) -> list[float]:
    candidate = _to_vector(vector, "init_vector")

    if basis_vectors:
        candidate = orthogonalize_vector(candidate, basis_vectors)

    norm = vector_l2_norm(candidate)
    if norm >= 1e-12:
        return scale_vector(candidate, 1.0 / norm)

    for pivot_idx in range(len(candidate)):
        unit_vector = [0.0 for _ in range(len(candidate))]
        unit_vector[pivot_idx] = 1.0

        if basis_vectors:
            unit_vector = orthogonalize_vector(unit_vector, basis_vectors)

        unit_norm = vector_l2_norm(unit_vector)
        if unit_norm >= 1e-12:
            return scale_vector(unit_vector, 1.0 / unit_norm)

    raise ValueError("could not construct a non-zero vector orthogonal to the basis")


def _power_iteration_step(
    matrix: list[list[float]],
    vector: list[float],
    basis_vectors: list[list[float]],
) -> list[float] | None:
    multiplied = matrix_vector_multiply(matrix, vector)

    if basis_vectors:
        multiplied = orthogonalize_vector(multiplied, basis_vectors)

    multiplied_norm = vector_l2_norm(multiplied)
    if multiplied_norm < 1e-12:
        return None

    return scale_vector(multiplied, 1.0 / multiplied_norm)


def column_means(matrix: Any) -> list[float]:
    """Compute mean value for each column in a 2D matrix."""
    data = _to_matrix(matrix, "column_means")
    n_rows = len(data)
    n_cols = len(data[0])

    means = [0.0 for _ in range(n_cols)]
    for col in range(n_cols):
        total = 0.0
        for row in range(n_rows):
            total += data[row][col]
        means[col] = total / n_rows
    return means


def center_matrix(matrix: Any, means: Any) -> list[list[float]]:
    """Center each row of a matrix by subtracting column means."""
    data = _to_matrix(matrix, "center_matrix")
    mean_vector = _to_vector(means, "center_matrix")
    n_rows = len(data)
    n_cols = len(data[0])

    if len(mean_vector) != n_cols:
        raise ValueError("means length must match matrix column count")

    centered: list[list[float]] = []
    for row in range(n_rows):
        centered_row = [0.0 for _ in range(n_cols)]
        for col in range(n_cols):
            centered_row[col] = data[row][col] - mean_vector[col]
        centered.append(centered_row)
    return centered


def covariance_matrix(centered: Any, denominator: int) -> list[list[float]]:
    """Compute covariance matrix from centered samples."""
    data = _to_matrix(centered, "covariance_matrix")

    if denominator <= 0:
        raise ValueError("denominator must be positive")

    n_rows = len(data)
    n_cols = len(data[0])
    cov = [[0.0 for _ in range(n_cols)] for _ in range(n_cols)]
    for i in range(n_cols):
        for j in range(i, n_cols):
            total = 0.0
            for row in range(n_rows):
                total += data[row][i] * data[row][j]
            value = total / denominator
            cov[i][j] = value
            cov[j][i] = value
    return cov


def sample_covariance_matrix(centered: Any, denominator: int) -> list[list[float]]:
    """Compute sample-space covariance matrix from centered samples."""
    data = _to_matrix(centered, "sample_covariance_matrix")

    if denominator <= 0:
        raise ValueError("denominator must be positive")

    n_rows = len(data)
    n_cols = len(data[0])
    cov = [[0.0 for _ in range(n_rows)] for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(i, n_rows):
            total = 0.0
            for col in range(n_cols):
                total += data[i][col] * data[j][col]
            value = total / denominator
            cov[i][j] = value
            cov[j][i] = value
    return cov


def matrix_vector_multiply(matrix: Any, vector: Any) -> list[float]:
    """Multiply a 2D matrix by a 1D vector."""
    data = _to_matrix(matrix, "matrix_vector_multiply")
    vector_data = _to_vector(vector, "matrix_vector_multiply")
    n_rows = len(data)
    n_cols = len(data[0])

    if n_cols != len(vector_data):
        raise ValueError("matrix column count must match vector length")

    result = [0.0 for _ in range(n_rows)]
    for row in range(n_rows):
        total = 0.0
        for col in range(n_cols):
            total += data[row][col] * vector_data[col]
        result[row] = total
    return result


def transpose_matrix_vector_multiply(matrix: Any, vector: Any) -> list[float]:
    """Multiply a transposed 2D matrix by a 1D vector."""
    data = _to_matrix(matrix, "transpose_matrix_vector_multiply")
    vector_data = _to_vector(vector, "transpose_matrix_vector_multiply")
    n_rows = len(data)
    n_cols = len(data[0])

    if n_rows != len(vector_data):
        raise ValueError("matrix row count must match vector length")

    result = [0.0 for _ in range(n_cols)]
    for col in range(n_cols):
        total = 0.0
        for row in range(n_rows):
            total += data[row][col] * vector_data[row]
        result[col] = total
    return result


def power_iteration_symmetric(
    matrix: Any,
    init_vector: Any,
    orthogonal_basis: Any | None = None,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> tuple[float, list[float]]:
    """Approximate dominant eigenpair of a symmetric matrix."""
    data = _to_matrix(matrix, "power_iteration_symmetric")
    size = len(data)

    if any(len(row) != size for row in data):
        raise ValueError("power_iteration_symmetric expects a square matrix")

    basis_vectors = _validated_orthogonal_basis(orthogonal_basis, size)
    init_data = _to_vector(init_vector, "init_vector")
    if len(init_data) != size:
        raise ValueError("init_vector length must match matrix size")

    vector = _normalized_orthogonal_vector(init_data, basis_vectors)

    for _ in range(max_iter):
        next_vector = _power_iteration_step(data, vector, basis_vectors)
        if next_vector is None:
            break

        delta = euclidean_distance(next_vector, vector)
        vector = next_vector
        if delta < tol:
            break

    rayleigh_num = vector_dot(vector, matrix_vector_multiply(data, vector))
    rayleigh_den = max(vector_dot(vector, vector), 1e-12)
    eigenvalue = rayleigh_num / rayleigh_den
    return eigenvalue, vector


def deflate_symmetric(matrix: Any, eigenvalue: float, eigenvector: Any) -> list[list[float]]:
    """Deflate symmetric matrix by one eigenpair."""
    data = _to_matrix(matrix, "deflate_symmetric")
    size = len(data)

    if any(len(row) != size for row in data):
        raise ValueError("deflate_symmetric expects a square matrix")

    vector = _to_vector(eigenvector, "eigenvector")
    if len(vector) != size:
        raise ValueError("eigenvector length must match matrix size")

    result = [row[:] for row in data]
    for i in range(size):
        for j in range(size):
            result[i][j] -= eigenvalue * vector[i] * vector[j]
    return result


def top_k_eigenpairs_symmetric(
    matrix: Any,
    k: int,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> tuple[list[float], list[list[float]]]:
    """Compute top-k eigenpairs of a symmetric matrix by power iteration + deflation."""
    data = _to_matrix(matrix, "top_k_eigenpairs_symmetric")
    size = len(data)

    if any(len(row) != size for row in data):
        raise ValueError("top_k_eigenpairs_symmetric expects a square matrix")
    if k < 1 or k > size:
        raise ValueError(f"k must be in [1, {size}]")

    working = [row[:] for row in data]
    eigenvalues = [0.0 for _ in range(k)]
    eigenvectors: list[list[float]] = []

    for idx in range(k):
        init = [1.0 for _ in range(size)]
        init[idx % size] = 2.0
        value, vector = power_iteration_symmetric(
            working,
            init_vector=init,
            orthogonal_basis=eigenvectors,
            max_iter=max_iter,
            tol=tol,
        )
        if value < 0.0 and abs(value) < 1e-10:
            value = 0.0
        eigenvalues[idx] = value
        eigenvectors.append(vector)
        working = deflate_symmetric(working, value, vector)

    return eigenvalues, eigenvectors
