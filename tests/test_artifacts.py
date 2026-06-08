import json
from pathlib import Path

from vib_project.artifacts import write_experiment_artifacts


def test_write_experiment_artifacts(tmp_path: Path):
    metrics = {
        "dataset": "mnist",
        "model": "vib",
        "beta": 0.001,
        "train_accuracy": 0.9,
        "test_accuracy": 0.8,
        "train_test_gap": 0.1,
        "average_kl": 2.0,
        "noise_accuracy": {"0.0": 0.8, "0.1": 0.7},
    }
    write_experiment_artifacts(tmp_path, "mnist_vib_beta_0_001", metrics)
    metrics_path = tmp_path / "mnist_vib_beta_0_001" / "metrics.json"
    index_path = tmp_path / "index.json"
    assert metrics_path.exists()
    assert index_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert saved["average_kl"] == 2.0
    assert index["experiments"][0]["id"] == "mnist_vib_beta_0_001"


def test_write_projection_and_confusion_artifacts(tmp_path: Path):
    metrics = {
        "dataset": "mnist",
        "model": "vib",
        "beta": 0.001,
        "train_accuracy": 0.9,
        "test_accuracy": 0.8,
        "train_test_gap": 0.1,
        "average_kl": 2.0,
        "noise_accuracy": {"0.0": 0.8},
        "latent_pca": [{"x": 0.1, "y": 0.2, "label": 3}],
        "confusion_matrix": [[1, 0], [0, 1]],
    }
    write_experiment_artifacts(tmp_path, "mnist_vib_beta_0_001", metrics)
    assert (tmp_path / "mnist_vib_beta_0_001" / "latent_pca.json").exists()
    assert (tmp_path / "mnist_vib_beta_0_001" / "confusion_matrix.json").exists()
