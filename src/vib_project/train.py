import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from vib_project.config import ExperimentConfig
from vib_project.data import build_dataset
from vib_project.evaluate import collect_outputs, evaluate_model
from vib_project.losses import vib_loss
from vib_project.models import CNNClassifier, VIBClassifier
from vib_project.projections import confusion_matrix_payload, pca_projection


def _configure_torch_backend() -> None:
    if os.environ.get("VIB_DISABLE_CUDNN") == "1":
        torch.backends.cudnn.enabled = False


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(config: ExperimentConfig, input_channels: int, image_size: int, num_classes: int):
    if config.model == "cnn":
        return CNNClassifier(input_channels, image_size, config.latent_dim, num_classes)
    if config.model == "vib":
        return VIBClassifier(input_channels, image_size, config.latent_dim, num_classes)
    raise ValueError(f"Unsupported model: {config.model}")


def run_training(config: ExperimentConfig) -> dict[str, object]:
    _configure_torch_backend()
    _seed_everything(config.seed)
    device = _device()
    bundle = build_dataset(
        config.dataset,
        batch_size=config.batch_size,
        limit_train=config.limit_train,
        limit_test=config.limit_test,
    )
    model = _build_model(config, bundle.input_channels, bundle.image_size, bundle.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for _epoch in range(config.epochs):
        model.train()
        for images, labels in bundle.train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if config.model == "vib":
                output = model(images)
                loss, _components = vib_loss(output.logits, labels, output.mu, output.logvar, beta=config.beta)
            else:
                logits, _latent = model(images)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

    train_eval = evaluate_model(model, bundle.train_loader, device, config.model)
    test_eval = evaluate_model(model, bundle.test_loader, device, config.model)
    noise_accuracy: dict[str, float] = {}
    for sigma in [0.0, 0.1, 0.2, 0.3, 0.4]:
        noisy = evaluate_model(model, bundle.test_loader, device, config.model, noise_sigma=sigma)
        noise_accuracy[f"{sigma:.1f}"] = noisy["accuracy"]

    collected = collect_outputs(model, bundle.test_loader, device, config.model)
    latent_pca = pca_projection(collected["latents"], collected["labels"])
    confusion = confusion_matrix_payload(collected["labels"], collected["predictions"], bundle.num_classes)

    return {
        "dataset": config.dataset,
        "model": config.model,
        "beta": config.beta,
        "train_accuracy": train_eval["accuracy"],
        "test_accuracy": test_eval["accuracy"],
        "train_test_gap": train_eval["accuracy"] - test_eval["accuracy"],
        "average_kl": test_eval["average_kl"],
        "noise_accuracy": noise_accuracy,
        "latent_pca": latent_pca,
        "confusion_matrix": confusion,
    }
