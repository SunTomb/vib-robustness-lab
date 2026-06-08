from pathlib import Path

from vib_project.config import ExperimentConfig
from vib_project.phase_runner import run_phase_experiment


def test_run_phase_experiment_writes_required_payload(tmp_path: Path):
    config = ExperimentConfig(
        dataset="mnist",
        model="vib",
        beta=0.001,
        epochs=1,
        batch_size=32,
        latent_dim=8,
        learning_rate=1e-3,
        seed=5,
        output_dir=tmp_path,
        limit_train=64,
        limit_test=32,
    )

    payload = run_phase_experiment(
        config,
        corruptions=["gaussian", "contrast"],
        severities=[0.0, 0.2],
        max_diagnostic_points=32,
    )

    assert payload["metrics"]["dataset"] == "mnist"
    assert len(payload["robustness"]["rows"]) == 4
    assert "fisher_ratio" in payload["latent_geometry"]
    assert "phase_label" in payload["phase_indicators"]
