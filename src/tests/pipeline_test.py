import unittest

import numpy as np

from facerec.pipeline import embed_vectors, predict_from_vector, train_model_from_vectors


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.2, 0.0, 0.1],
                [4.9, 5.1, 4.8],
                [5.2, 5.0, 5.1],
            ],
            dtype=np.float64,
        )
        self.labels = np.array(["alice", "alice", "bob", "bob"])

    def test_train_model_from_vectors_builds_expected_shapes(self):
        model = train_model_from_vectors(self.X, self.labels, n_components=2)
        self.assertEqual(model.gallery_embeddings.shape, (4, 2))
        self.assertEqual(model.gallery_labels.shape, (4,))
        self.assertEqual(model.metric, "euclidean")

    def test_embed_vectors_projects_to_model_space(self):
        model = train_model_from_vectors(self.X, self.labels, n_components=2)
        embedded = embed_vectors(model, self.X[:2])
        self.assertEqual(embedded.shape, (2, 2))

    def test_predict_from_vector_returns_expected_label(self):
        model = train_model_from_vectors(self.X, self.labels, n_components=2)
        query = np.array([5.1, 4.9, 5.0], dtype=np.float64)
        label, distance, index = predict_from_vector(model, query)
        self.assertEqual(label, "bob")
        self.assertTrue(distance >= 0.0)
        self.assertTrue(index in (2, 3))

    def test_train_model_from_vectors_raises_for_label_mismatch(self):
        bad_labels = np.array(["alice", "alice", "bob"])
        with self.assertRaises(ValueError):
            train_model_from_vectors(self.X, bad_labels, n_components=2)

    def test_predict_from_vector_raises_for_wrong_shape(self):
        model = train_model_from_vectors(self.X, self.labels, n_components=2)
        bad = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            predict_from_vector(model, bad)
