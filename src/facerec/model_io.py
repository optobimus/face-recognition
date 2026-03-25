"""Model persistence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from facerec.knn import MetricName
from facerec.pca import PCAState
from facerec.pipeline import EigenfaceModel


def save_model(model: EigenfaceModel, model_path: Path) -> None:
    """Save an Eigenface model to a compressed NumPy file."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=model.pca.mean,
        components=model.pca.components,
        singular_values=model.pca.singular_values,
        explained_variance=model.pca.explained_variance,
        gallery_embeddings=model.gallery_embeddings,
        gallery_labels=model.gallery_labels.astype(str),
        metric=np.array(model.metric),
        image_size=np.array(model.image_size, dtype=np.int64),
    )


def load_model(model_path: Path) -> EigenfaceModel:
    """Load an Eigenface model from a compressed NumPy file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        gallery_labels_array = np.asarray(data["gallery_labels"], dtype=str)
        image_size_values = np.asarray(data["image_size"], dtype=np.int64).tolist()
        metric_value = cast(MetricName, str(data["metric"]))
        pca = PCAState(
            mean=data["mean"],
            components=data["components"],
            singular_values=data["singular_values"],
            explained_variance=data["explained_variance"],
        )
        image_size = (int(image_size_values[0]), int(image_size_values[1]))
        return EigenfaceModel(
            pca=pca,
            gallery_embeddings=data["gallery_embeddings"],
            gallery_labels=gallery_labels_array,
            metric=metric_value,
            image_size=image_size,
        )
