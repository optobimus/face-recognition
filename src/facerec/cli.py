"""Command-line interface for training and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from facerec.data import discover_orl_images, split_per_identity
from facerec.eval import evaluate_predictions
from facerec.model_io import load_model, save_model
from facerec.pipeline import predict_from_vector, train_model_from_vectors
from facerec.preprocess import preprocess_image


def _build_vectors(items: list[tuple[Path, str]], image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for path, label in items:
        vectors.append(preprocess_image(path, image_size=image_size))
        labels.append(label)
    return np.stack(vectors), np.array(labels, dtype=str)


def _train_command(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    model_out = Path(args.model_out)
    image_size = (int(args.image_width), int(args.image_height))

    items = discover_orl_images(dataset_root)
    train_items, _ = split_per_identity(
        items,
        train_per_identity=int(args.train_per_identity),
        seed=int(args.seed),
    )
    X_train, y_train = _build_vectors(train_items, image_size=image_size)
    model = train_model_from_vectors(
        X_train,
        y_train,
        n_components=int(args.n_components),
        metric=args.metric,
        image_size=image_size,
    )
    save_model(model, model_out)
    print(
        f"trained_samples={X_train.shape[0]} components={args.n_components} model={model_out}"
    )
    return 0


def _predict_command(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    vector = preprocess_image(Path(args.image), image_size=model.image_size)
    label, distance, _ = predict_from_vector(model, vector)
    print(f"label={label} distance={distance:.6f}")
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    dataset_root = Path(args.dataset_root)
    report_out = Path(args.report_out)

    items = discover_orl_images(dataset_root)
    _, test_items = split_per_identity(
        items,
        train_per_identity=int(args.train_per_identity),
        seed=int(args.seed),
    )
    if len(test_items) == 0:
        raise ValueError("Test split is empty for the provided configuration")

    y_true: list[str] = []
    y_pred: list[str] = []
    for image_path, label in test_items:
        vector = preprocess_image(image_path, image_size=model.image_size)
        pred_label, _, _ = predict_from_vector(model, vector)
        y_true.append(label)
        y_pred.append(pred_label)

    summary = evaluate_predictions(y_true, y_pred)
    report = {
        "total": summary.total,
        "correct": summary.correct,
        "accuracy": summary.accuracy,
        "confusion_matrix": summary.confusion_matrix,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"evaluated_samples={summary.total} accuracy={summary.accuracy:.6f} report={report_out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facerec")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-root", required=True)
    train_parser.add_argument("--model-out", required=True)
    train_parser.add_argument("--n-components", type=int, default=20)
    train_parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    train_parser.add_argument("--train-per-identity", type=int, default=6)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--image-width", type=int, default=64)
    train_parser.add_argument("--image-height", type=int, default=64)
    train_parser.set_defaults(handler=_train_command)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--image", required=True)
    predict_parser.set_defaults(handler=_predict_command)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--dataset-root", required=True)
    evaluate_parser.add_argument("--report-out", required=True)
    evaluate_parser.add_argument("--train-per-identity", type=int, default=6)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.set_defaults(handler=_evaluate_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
