import unittest

import numpy as np

from facerec.pca import fit_pca, transform_pca


class TestPCA(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 3.0],
                [1.0, 4.0, 2.0],
                [3.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )

    def test_fit_pca_returns_expected_shapes(self):
        state = fit_pca(self.X, n_components=2)
        self.assertEqual(state.mean.shape, (3,))
        self.assertEqual(state.components.shape, (2, 3))
        self.assertEqual(state.singular_values.shape, (2,))
        self.assertEqual(state.explained_variance.shape, (2,))

    def test_transform_pca_returns_expected_shape(self):
        state = fit_pca(self.X, n_components=2)
        projected = transform_pca(self.X, state)
        self.assertEqual(projected.shape, (4, 2))

    def test_fit_pca_raises_for_invalid_component_count(self):
        with self.assertRaises(ValueError):
            fit_pca(self.X, n_components=0)
        with self.assertRaises(ValueError):
            fit_pca(self.X, n_components=5)

    def test_fit_pca_raises_for_too_few_samples(self):
        X_single = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            fit_pca(X_single, n_components=1)

    def test_transform_pca_raises_for_feature_mismatch(self):
        state = fit_pca(self.X, n_components=2)
        bad = np.array([[1.0, 2.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            transform_pca(bad, state)

    def test_transform_pca_is_deterministic(self):
        state = fit_pca(self.X, n_components=2)
        a = transform_pca(self.X, state)
        b = transform_pca(self.X, state)
        self.assertTrue(np.array_equal(a, b))

    def test_fit_pca_raises_for_non_matrix_input(self):
        bad = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            fit_pca(bad, n_components=1)

    def test_transform_pca_raises_for_non_matrix_input(self):
        state = fit_pca(self.X, n_components=2)
        bad = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            transform_pca(bad, state)
