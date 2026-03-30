"""Integration test for the end-to-end CLI flow."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image


def _create_identity(dataset_root: Path, label: str, count: int, base_value: int) -> None:
    identity_dir = dataset_root / label
    identity_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        image = Image.new("L", (16, 16), color=min(base_value + idx, 255))
        image.save(identity_dir / f"{idx}.pgm")


def _build_env() -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing}" if existing else src_path
    )
    return env


def _run_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "facerec.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def main() -> int:
    env = _build_env()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        dataset_root = root / "orl"
        model_path = root / "model.npz"
        report_path = root / "eval.json"

        _create_identity(dataset_root, "s1", count=4, base_value=20)
        _create_identity(dataset_root, "s2", count=4, base_value=220)

        train_result = _run_cli(
            [
                "train",
                "--dataset-root",
                str(dataset_root),
                "--model-out",
                str(model_path),
                "--n-components",
                "2",
                "--train-per-identity",
                "2",
                "--image-width",
                "8",
                "--image-height",
                "8",
                "--seed",
                "42",
            ],
            env=env,
        )
        if train_result.returncode != 0:
            raise AssertionError(f"train failed: {train_result.stderr}")
        if not model_path.exists():
            raise AssertionError("train did not create model file")

        predict_result = _run_cli(
            ["predict", "--model", str(model_path), "--image", str(dataset_root / "s1" / "1.pgm")],
            env=env,
        )
        if predict_result.returncode != 0:
            raise AssertionError(f"predict failed: {predict_result.stderr}")
        if "label=" not in predict_result.stdout or "distance=" not in predict_result.stdout:
            raise AssertionError("predict output missing expected fields")

        evaluate_result = _run_cli(
            [
                "evaluate",
                "--model",
                str(model_path),
                "--dataset-root",
                str(dataset_root),
                "--report-out",
                str(report_path),
                "--train-per-identity",
                "2",
                "--seed",
                "42",
            ],
            env=env,
        )
        if evaluate_result.returncode != 0:
            raise AssertionError(f"evaluate failed: {evaluate_result.stderr}")
        if not report_path.exists():
            raise AssertionError("evaluate did not create report file")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        if "accuracy" not in report or "confusion_matrix" not in report:
            raise AssertionError("report missing required fields")
        if report["total"] <= 0:
            raise AssertionError("report total must be positive")

    print("integration_cli_flow: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
