"""Evaluation utilities for face recognition predictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    correct: int
    accuracy: float
    confusion_matrix: dict[str, dict[str, int]]


def build_confusion_matrix(
    y_true: list[str], y_pred: list[str]
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix as nested dictionaries."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    labels = sorted(set(y_true) | set(y_pred))
    matrix: dict[str, dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in labels} for true_label in labels
    }
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[true_label][pred_label] += 1
    return matrix


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> EvaluationSummary:
    """Compute basic evaluation metrics from labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty predictions")

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred, strict=True))
    total = len(y_true)
    accuracy = correct / total
    confusion = build_confusion_matrix(y_true, y_pred)
    return EvaluationSummary(
        total=total,
        correct=correct,
        accuracy=accuracy,
        confusion_matrix=confusion,
    )
