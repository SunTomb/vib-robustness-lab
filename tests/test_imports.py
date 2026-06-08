from pathlib import Path

from vib_project.config import ExperimentConfig


def test_experiment_id_for_vib():
    config = ExperimentConfig(
        dataset="mnist",
        model="vib",
        beta=0.001,
        epochs=1,
        batch_size=32,
        latent_dim=16,
        learning_rate=1e-3,
        seed=7,
        output_dir=Path("artifacts/dev"),
    )
    assert config.experiment_id == "mnist_vib_beta_0_001"


def test_experiment_id_for_cnn():
    config = ExperimentConfig(
        dataset="fashion_mnist",
        model="cnn",
        beta=0.0,
        epochs=1,
        batch_size=32,
        latent_dim=16,
        learning_rate=1e-3,
        seed=7,
        output_dir=Path("artifacts/dev"),
    )
    assert config.experiment_id == "fashion_mnist_cnn_baseline"
