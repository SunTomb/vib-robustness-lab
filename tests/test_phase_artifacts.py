from pathlib import Path

from vib_project.phase_artifacts import write_phase_experiment, write_phase_index


def test_write_phase_experiment_files(tmp_path: Path):
    payload = {
        "metrics": {"dataset": "mnist", "model": "vib", "beta": 0.001, "test_accuracy": 0.9, "average_kl": 2.0},
        "robustness": {"rows": [{"corruption": "gaussian", "severity": 0.0, "accuracy": 0.9}]},
        "latent_geometry": {"intra_class_variance": 1.0, "inter_class_distance": 2.0, "fisher_ratio": 2.0, "per_class": []},
        "phase_indicators": {"phase_label": "useful-compression", "over_compression_flag": False},
        "latent_drift": {"rows": []},
    }

    write_phase_experiment(tmp_path, "mnist_vib_beta_0_001", payload)

    exp = tmp_path / "mnist_vib_beta_0_001"
    assert (exp / "metrics.json").exists()
    assert (exp / "robustness.json").exists()
    assert (exp / "latent_geometry.json").exists()
    assert (exp / "phase_indicators.json").exists()
    assert (exp / "latent_drift.json").exists()


def test_write_phase_index(tmp_path: Path):
    write_phase_index(
        tmp_path,
        datasets=["mnist"],
        experiments=[{"id": "mnist_vib_beta_0_001", "dataset": "mnist", "model": "vib", "beta": 0.001, "path": "mnist_vib_beta_0_001"}],
    )
    assert (tmp_path / "index.json").exists()
