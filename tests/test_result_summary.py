import json
from pathlib import Path

from scripts.summarize_results import build_metric_rows, build_robustness_rows, derive_key_findings


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_artifacts(root: Path):
    _write_json(
        root / "index.json",
        {
            "datasets": ["mnist"],
            "experiments": [
                {"id": "mnist_cnn_baseline", "dataset": "mnist", "model": "cnn", "beta": 0.0, "path": "mnist_cnn_baseline"},
                {"id": "mnist_vib_beta_0_01", "dataset": "mnist", "model": "vib", "beta": 0.01, "path": "mnist_vib_beta_0_01"},
            ],
        },
    )
    _write_json(root / "mnist_cnn_baseline" / "metrics.json", {"dataset": "mnist", "model": "cnn", "beta": 0.0, "test_accuracy": 0.88, "average_kl": 0.0, "train_test_gap": 0.02, "noise_accuracy": {"0.0": 0.88, "0.4": 0.50}})
    _write_json(root / "mnist_vib_beta_0_01" / "metrics.json", {"dataset": "mnist", "model": "vib", "beta": 0.01, "test_accuracy": 0.91, "average_kl": 3.0, "train_test_gap": 0.01, "noise_accuracy": {"0.0": 0.91, "0.4": 0.62}})


def test_build_metric_rows(tmp_path: Path):
    _make_artifacts(tmp_path)

    rows = build_metric_rows(tmp_path)

    assert rows[0]["experiment_id"] == "mnist_cnn_baseline"
    assert rows[1]["beta"] == 0.01
    assert rows[1]["test_accuracy"] == 0.91


def test_build_robustness_rows(tmp_path: Path):
    _make_artifacts(tmp_path)

    rows = build_robustness_rows(tmp_path)

    assert {row["sigma"] for row in rows} == {0.0, 0.4}
    assert any(row["experiment_id"] == "mnist_vib_beta_0_01" and row["accuracy"] == 0.62 for row in rows)


def test_derive_key_findings_identifies_best_accuracy(tmp_path: Path):
    _make_artifacts(tmp_path)

    findings = derive_key_findings(build_metric_rows(tmp_path))

    assert "mnist" in findings
    assert findings["mnist"]["best_clean_experiment"] == "mnist_vib_beta_0_01"
