import unittest

import numpy as np

from facerec.knn import predict_nearest_neighbor


class TestKNN(unittest.TestCase):
    def setUp(self):
        self.gallery = np.array(
            [[0.0, 0.0], [1.0, 1.0], [5.0, 4.0], [0.5, 0.4]],
            dtype=np.float64,
        )
        self.labels = np.array(["a", "b", "c", "d"])

    def test_predict_nearest_neighbor_with_euclidean_metric(self):
        query = np.array([0.45, 0.35], dtype=np.float64)
        label, distance, index = predict_nearest_neighbor(
            query, self.gallery, self.labels, metric="euclidean"
        )
        self.assertEqual(label, "d")
        self.assertEqual(index, 3)
        self.assertTrue(distance >= 0.0)

    def test_predict_nearest_neighbor_with_cosine_metric(self):
        query = np.array([4.8, 3.7], dtype=np.float64)
        label, distance, index = predict_nearest_neighbor(
            query, self.gallery, self.labels, metric="cosine"
        )
        self.assertEqual(label, "c")
        self.assertEqual(index, 2)
        self.assertTrue(distance >= 0.0)

    def test_predict_nearest_neighbor_raises_for_empty_gallery(self):
        query = np.array([1.0, 2.0], dtype=np.float64)
        gallery = np.empty((0, 2), dtype=np.float64)
        labels = np.array([], dtype=str)
        with self.assertRaises(ValueError):
            predict_nearest_neighbor(query, gallery, labels)

    def test_predict_nearest_neighbor_raises_for_feature_mismatch(self):
        query = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            predict_nearest_neighbor(query, self.gallery, self.labels)

    def test_predict_nearest_neighbor_raises_for_label_mismatch(self):
        query = np.array([1.0, 2.0], dtype=np.float64)
        labels = np.array(["a", "b"])
        with self.assertRaises(ValueError):
            predict_nearest_neighbor(query, self.gallery, labels)

    def test_predict_nearest_neighbor_raises_for_unsupported_metric(self):
        query = np.array([0.0, 0.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            predict_nearest_neighbor(query, self.gallery, self.labels, metric="l1")
