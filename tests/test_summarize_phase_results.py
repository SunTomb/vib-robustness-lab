import json
from pathlib import Path

from scripts.summarize_phase_results import build_phase_summary, refresh_phase_indicators, write_phase_result_exports


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_experiment(
    root: Path,
    experiment_id: str,
    beta: float,
    clean: float,
    gaussian_auc: float,
    phase_label: str,
    model: str = "vib",
    average_kl: float = 1.0,
):
    exp = root / experiment_id
    _write_json(exp / "metrics.json", {"dataset": "mnist", "model": model, "beta": beta, "test_accuracy": clean, "average_kl": average_kl})
    _write_json(exp / "robustness.json", {"rows": [{"corruption": "gaussian", "severity": 0.0, "accuracy": gaussian_auc}]})
    _write_json(exp / "latent_geometry.json", {"fisher_ratio": 2.0})
    _write_json(exp / "phase_indicators.json", {"robustness_auc": {"gaussian": gaussian_auc}, "phase_label": phase_label})


def test_build_phase_summary_identifies_best_clean_and_robust(tmp_path: Path):
    _write_json(
        tmp_path / "index.json",
        {
            "datasets": ["mnist"],
            "experiments": [
                {"id": "mnist_vib_beta_0_001", "dataset": "mnist", "model": "vib", "beta": 0.001, "path": "mnist_vib_beta_0_001"},
                {"id": "mnist_vib_beta_0_01", "dataset": "mnist", "model": "vib", "beta": 0.01, "path": "mnist_vib_beta_0_01"},
            ],
        },
    )
    _write_experiment(tmp_path, "mnist_vib_beta_0_001", 0.001, clean=0.88, gaussian_auc=0.86, phase_label="robustness-specialized")
    _write_experiment(tmp_path, "mnist_vib_beta_0_01", 0.01, clean=0.91, gaussian_auc=0.80, phase_label="useful-compression")

    summary = build_phase_summary(tmp_path)

    assert summary["datasets"]["mnist"]["best_clean_experiment"] == "mnist_vib_beta_0_01"
    assert summary["datasets"]["mnist"]["best_robust_by_corruption"]["gaussian"] == "mnist_vib_beta_0_001"


def test_refresh_phase_indicators_uses_dataset_baselines(tmp_path: Path):
    _write_json(
        tmp_path / "index.json",
        {
            "datasets": ["mnist"],
            "experiments": [
                {"id": "mnist_cnn_baseline", "dataset": "mnist", "model": "cnn", "beta": 0, "path": "mnist_cnn_baseline"},
                {"id": "mnist_vib_beta_0", "dataset": "mnist", "model": "vib", "beta": 0, "path": "mnist_vib_beta_0"},
                {"id": "mnist_vib_beta_0_1", "dataset": "mnist", "model": "vib", "beta": 0.1, "path": "mnist_vib_beta_0_1"},
            ],
        },
    )
    _write_experiment(tmp_path, "mnist_cnn_baseline", 0, clean=0.8, gaussian_auc=0.4, phase_label="unstable", model="cnn", average_kl=0.0)
    _write_experiment(tmp_path, "mnist_vib_beta_0", 0, clean=0.9, gaussian_auc=0.72, phase_label="unstable", average_kl=10.0)
    _write_experiment(tmp_path, "mnist_vib_beta_0_1", 0.1, clean=0.86, gaussian_auc=0.602, phase_label="unstable", average_kl=1.0)

    refresh_phase_indicators(tmp_path)

    phase = json.loads((tmp_path / "mnist_vib_beta_0_1" / "phase_indicators.json").read_text(encoding="utf-8"))
    assert phase["compression_benefit_index"]["gaussian"] == 0.2
    assert phase["kl_collapse_score"] == 0.1


def test_write_phase_result_exports_writes_artifact_summary(tmp_path: Path):
    artifact_root = tmp_path / "phase"
    output_root = tmp_path / "phase_results"
    _write_json(
        artifact_root / "index.json",
        {
            "datasets": ["mnist"],
            "experiments": [
                {"id": "mnist_cnn_baseline", "dataset": "mnist", "model": "cnn", "beta": 0, "path": "mnist_cnn_baseline"},
            ],
        },
    )
    _write_experiment(artifact_root, "mnist_cnn_baseline", 0, clean=0.8, gaussian_auc=0.4, phase_label="unstable", model="cnn", average_kl=0.0)

    write_phase_result_exports(artifact_root, output_root)

    assert (artifact_root / "summary.json").exists()
    assert (output_root / "phase_summary.json").exists()
