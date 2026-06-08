import json
from pathlib import Path

from scripts.validate_phase_artifacts import validate_phase_artifact_root


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_phase_artifact_root_accepts_complete_experiment(tmp_path: Path):
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
    _write_json(exp / "metrics.json", {"dataset": "mnist", "test_accuracy": 0.9})
    _write_json(exp / "robustness.json", {"rows": []})
    _write_json(exp / "latent_geometry.json", {"fisher_ratio": 1.0})
    _write_json(exp / "phase_indicators.json", {"phase_label": "useful-compression"})

    report = validate_phase_artifact_root(tmp_path)

    assert report["experiment_count"] == 1
    assert report["missing"] == []


def test_validate_phase_artifact_root_reports_missing_files(tmp_path: Path):
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

    report = validate_phase_artifact_root(tmp_path)

    assert "mnist_vib_beta_0_001/robustness.json" in report["missing"]
    assert "mnist_vib_beta_0_001/latent_geometry.json" in report["missing"]
    assert "mnist_vib_beta_0_001/phase_indicators.json" in report["missing"]
