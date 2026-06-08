from pathlib import Path

import torch

from vib_project.config import ExperimentConfig
from vib_project.train import _configure_torch_backend, run_training


def test_disable_cudnn_env_configures_backend(monkeypatch):
    original = torch.backends.cudnn.enabled
    try:
        torch.backends.cudnn.enabled = True
        monkeypatch.setenv("VIB_DISABLE_CUDNN", "1")
        _configure_torch_backend()
        assert torch.backends.cudnn.enabled is False
    finally:
        torch.backends.cudnn.enabled = original


def test_tiny_training_outputs_metrics(tmp_path: Path):
    config = ExperimentConfig(
        dataset="mnist",
        model="vib",
        beta=0.001,
        epochs=1,
        batch_size=64,
        latent_dim=8,
        learning_rate=1e-3,
        seed=3,
        output_dir=tmp_path,
        limit_train=128,
        limit_test=64,
    )
    result = run_training(config)
    assert result["dataset"] == "mnist"
    assert result["model"] == "vib"
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["average_kl"] >= 0.0
