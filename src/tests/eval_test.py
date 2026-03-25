import unittest

from facerec.eval import build_confusion_matrix, evaluate_predictions


class TestEval(unittest.TestCase):
    def test_build_confusion_matrix_counts_correctly(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "a"]
        matrix = build_confusion_matrix(y_true, y_pred)
        self.assertEqual(matrix["a"]["a"], 1)
        self.assertEqual(matrix["a"]["b"], 1)
        self.assertEqual(matrix["b"]["a"], 1)
        self.assertEqual(matrix["b"]["b"], 1)

    def test_evaluate_predictions_computes_accuracy(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "b"]
        summary = evaluate_predictions(y_true, y_pred)
        self.assertEqual(summary.total, 4)
        self.assertEqual(summary.correct, 3)
        self.assertAlmostEqual(summary.accuracy, 0.75)

    def test_evaluate_predictions_raises_for_empty_input(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([], [])
