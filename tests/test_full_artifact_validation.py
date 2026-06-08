import json
from pathlib import Path

from scripts.validate_full_artifacts import validate_artifact_root


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_artifact_root_accepts_complete_experiment(tmp_path: Path):
    _write_json(
        tmp_path / "index.json",
        {
            "datasets": ["mnist"],
            "experiments": [
                {"id": "mnist_vib_beta_0_001", "dataset": "mnist", "model": "vib", "beta": 0.001, "path": "mnist_vib_beta_0_001"}
            ],
        },
    )
    exp = tmp_path / "mnist_vib_beta_0_001"
    _write_json(exp / "metrics.json", {"dataset": "mnist", "model": "vib", "beta": 0.001, "test_accuracy": 0.9, "average_kl": 2.0, "noise_accuracy": {"0.0": 0.9}})
    _write_json(exp / "curves.json", {"noise": [{"sigma": 0.0, "accuracy": 0.9}]})
    _write_json(exp / "latent_pca.json", {"points": [{"x": 0.0, "y": 0.0, "label": 1}]})
    _write_json(exp / "confusion_matrix.json", {"matrix": [[1]]})

    report = validate_artifact_root(tmp_path)

    assert report["experiment_count"] == 1
    assert report["missing"] == []


def test_validate_artifact_root_reports_missing_files(tmp_path: Path):
    _write_json(
        tmp_path / "index.json",
        {
            "datasets": ["mnist"],
            "experiments": [
                {"id": "mnist_vib_beta_0_001", "dataset": "mnist", "model": "vib", "beta": 0.001, "path": "mnist_vib_beta_0_001"}
            ],
        },
    )
    _write_json(tmp_path / "mnist_vib_beta_0_001" / "metrics.json", {"test_accuracy": 0.9})

    report = validate_artifact_root(tmp_path)

    assert "mnist_vib_beta_0_001/curves.json" in report["missing"]
    assert "mnist_vib_beta_0_001/latent_pca.json" in report["missing"]
    assert "mnist_vib_beta_0_001/confusion_matrix.json" in report["missing"]
