import unittest

import numpy as np

from facerec.matrix_ops import (
    argmin_index,
    center_matrix,
    column_means,
    cosine_distance,
    covariance_matrix,
    deflate_symmetric,
    euclidean_distance,
    matrix_vector_multiply,
    power_iteration_symmetric,
    scale_vector,
    top_k_eigenpairs_symmetric,
    vector_dot,
    vector_l2_norm,
)


class TestMatrixOps(unittest.TestCase):
    def test_vector_dot_and_l2_norm(self):
        a = np.array([3.0, 4.0], dtype=np.float64)
        b = np.array([2.0, 5.0], dtype=np.float64)
        self.assertAlmostEqual(vector_dot(a, b), 26.0)
        self.assertAlmostEqual(vector_l2_norm(a), 5.0)

    def test_scale_vector(self):
        vector = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        scaled = scale_vector(vector, 0.5)
        self.assertTrue(np.array_equal(scaled, np.array([0.5, -1.0, 1.5])))

    def test_euclidean_and_cosine_distance(self):
        a = np.array([1.0, 0.0], dtype=np.float64)
        b = np.array([4.0, 0.0], dtype=np.float64)
        c = np.array([0.0, 2.0], dtype=np.float64)
        self.assertAlmostEqual(euclidean_distance(a, b), 3.0)
        self.assertAlmostEqual(cosine_distance(a, b), 0.0)
        self.assertAlmostEqual(cosine_distance(a, c), 1.0)

    def test_euclidean_distance_raises_for_invalid_input(self):
        with self.assertRaises(ValueError):
            euclidean_distance(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            euclidean_distance(np.array([1.0]), np.array([1.0, 2.0]))

    def test_argmin_index(self):
        values = np.array([3.0, -2.0, -1.0, 4.0], dtype=np.float64)
        self.assertEqual(argmin_index(values), 1)

    def test_column_means_and_center_matrix(self):
        matrix = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float64)
        means = column_means(matrix)
        centered = center_matrix(matrix, means)
        self.assertTrue(np.array_equal(means, np.array([2.0, 4.0])))
        self.assertTrue(
            np.array_equal(centered, np.array([[-1.0, -1.0], [1.0, 1.0]]))
        )

    def test_covariance_matrix(self):
        centered = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float64)
        cov = covariance_matrix(centered, denominator=1)
        expected = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
        self.assertTrue(np.array_equal(cov, expected))

    def test_matrix_vector_multiply(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        vector = np.array([5.0, 6.0], dtype=np.float64)
        result = matrix_vector_multiply(matrix, vector)
        self.assertTrue(np.array_equal(result, np.array([17.0, 39.0])))

    def test_power_iteration_symmetric(self):
        matrix = np.array([[4.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        init = np.array([1.0, 1.0], dtype=np.float64)
        eigenvalue, eigenvector = power_iteration_symmetric(matrix, init)
        self.assertAlmostEqual(eigenvalue, 4.0, places=6)
        self.assertAlmostEqual(vector_l2_norm(eigenvector), 1.0, places=6)

    def test_top_k_eigenpairs_symmetric(self):
        matrix = np.array(
            [[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        eigenvalues, eigenvectors = top_k_eigenpairs_symmetric(matrix, k=2)
        self.assertEqual(len(eigenvalues), 2)
        self.assertEqual(len(eigenvectors), 2)
        self.assertEqual(len(eigenvectors[0]), 3)
        sorted_values = sorted([float(v) for v in eigenvalues], reverse=True)
        self.assertAlmostEqual(sorted_values[0], 5.0, places=4)
        self.assertAlmostEqual(sorted_values[1], 3.0, places=4)

    def test_vector_dot_raises_for_invalid_input(self):
        with self.assertRaises(ValueError):
            vector_dot(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            vector_dot(np.array([1.0]), np.array([1.0, 2.0]))

    def test_argmin_index_raises_for_invalid_input(self):
        with self.assertRaises(ValueError):
            argmin_index(np.array([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            argmin_index(np.array([], dtype=np.float64))

    def test_scale_vector_raises_for_invalid_input(self):
        with self.assertRaises(ValueError):
            scale_vector(np.array([[1.0, 2.0]]), 2.0)

    def test_column_means_raises_for_invalid_input(self):
        with self.assertRaises(ValueError):
            column_means(np.array([1.0, 2.0]))

    def test_center_matrix_raises_for_invalid_input(self):
        matrix = np.array([[1.0, 2.0]], dtype=np.float64)
        means = np.array([1.0, 2.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            center_matrix(np.array([1.0, 2.0]), means)
        with self.assertRaises(ValueError):
            center_matrix(matrix, np.array([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            center_matrix(matrix, np.array([1.0]))

    def test_covariance_matrix_raises_for_invalid_input(self):
        centered = np.array([[1.0, 2.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            covariance_matrix(np.array([1.0, 2.0]), denominator=1)
        with self.assertRaises(ValueError):
            covariance_matrix(centered, denominator=0)

    def test_matrix_vector_multiply_raises_for_invalid_input(self):
        matrix = np.array([[1.0, 2.0]], dtype=np.float64)
        vector = np.array([1.0, 2.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            matrix_vector_multiply(np.array([1.0, 2.0]), vector)
        with self.assertRaises(ValueError):
            matrix_vector_multiply(matrix, np.array([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            matrix_vector_multiply(matrix, np.array([1.0]))

    def test_power_iteration_symmetric_raises_for_invalid_input(self):
        square = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            power_iteration_symmetric(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            power_iteration_symmetric(np.array([[1.0, 2.0, 3.0]]), np.array([1.0]))
        with self.assertRaises(ValueError):
            power_iteration_symmetric(square, np.array([[1.0, 1.0]]))
        with self.assertRaises(ValueError):
            power_iteration_symmetric(square, np.array([1.0]))

    def test_power_iteration_handles_zero_matrix(self):
        matrix = np.zeros((2, 2), dtype=np.float64)
        init = np.array([0.0, 0.0], dtype=np.float64)
        eigenvalue, eigenvector = power_iteration_symmetric(matrix, init)
        self.assertAlmostEqual(eigenvalue, 0.0)
        self.assertAlmostEqual(vector_l2_norm(eigenvector), 1.0, places=6)

    def test_power_iteration_with_zero_iterations(self):
        matrix = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        init = np.array([1.0, 0.0], dtype=np.float64)
        eigenvalue, eigenvector = power_iteration_symmetric(matrix, init, max_iter=0)
        self.assertAlmostEqual(eigenvalue, 2.0, places=6)
        self.assertAlmostEqual(vector_l2_norm(eigenvector), 1.0, places=6)

    def test_deflate_symmetric_raises_for_invalid_input(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        vector = np.array([1.0, 0.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            deflate_symmetric(np.array([1.0, 2.0]), 1.0, vector)
        with self.assertRaises(ValueError):
            deflate_symmetric(matrix, 1.0, np.array([[1.0, 0.0]]))
        with self.assertRaises(ValueError):
            deflate_symmetric(matrix, 1.0, np.array([1.0]))

    def test_top_k_eigenpairs_symmetric_raises_for_invalid_input(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            top_k_eigenpairs_symmetric(np.array([1.0, 2.0]), k=1)
        with self.assertRaises(ValueError):
            top_k_eigenpairs_symmetric(matrix, k=0)
        with self.assertRaises(ValueError):
            top_k_eigenpairs_symmetric(matrix, k=3)

    def test_top_k_eigenpairs_clamps_tiny_negative(self):
        matrix = np.array([[-1e-12]], dtype=np.float64)
        eigenvalues, _ = top_k_eigenpairs_symmetric(matrix, k=1)
        self.assertAlmostEqual(float(eigenvalues[0]), 0.0, places=12)
