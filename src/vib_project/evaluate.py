from collections.abc import Iterable

import torch
from torch import nn

from vib_project.corruption import apply_corruption
from vib_project.losses import kl_normal_standard
from vib_project.noise import add_gaussian_noise


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    batches: Iterable,
    device: torch.device,
    model_name: str,
    noise_sigma: float = 0.0,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    kl_sum = 0.0
    batch_count = 0

    for images, labels in batches:
        images = images.to(device)
        labels = labels.to(device)
        if noise_sigma > 0.0:
            images = add_gaussian_noise(images, sigma=noise_sigma)

        if model_name == "vib":
            output = model(images)
            logits = output.logits
            kl_sum += float(kl_normal_standard(output.mu, output.logvar).detach().cpu())
        else:
            logits, _ = model(images)

        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
        batch_count += 1

    accuracy = correct / total if total else 0.0
    average_kl = kl_sum / batch_count if batch_count and model_name == "vib" else 0.0
    return {"accuracy": accuracy, "average_kl": average_kl}


@torch.no_grad()
def collect_outputs(model: nn.Module, batches: Iterable, device: torch.device, model_name: str) -> dict[str, object]:
    import numpy as np

    model.eval()
    latents = []
    labels_all = []
    predictions_all = []
    for images, labels in batches:
        images = images.to(device)
        labels = labels.to(device)
        if model_name == "vib":
            output = model(images)
            logits = output.logits
            latent = output.mu
        else:
            logits, latent = model(images)
        predictions = logits.argmax(dim=1)
        latents.append(latent.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        predictions_all.append(predictions.detach().cpu().numpy())
    return {
        "latents": np.concatenate(latents, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "predictions": np.concatenate(predictions_all, axis=0),
    }


def _confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.max(dim=1).values


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    return -(probabilities * log_probabilities).sum(dim=1)


@torch.no_grad()
def evaluate_corruption_grid(
    model: nn.Module,
    batches: Iterable,
    device: torch.device,
    model_name: str,
    corruptions: list[str],
    severities: list[float],
) -> list[dict[str, float | str]]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    for corruption in corruptions:
        for severity in severities:
            correct = 0
            total = 0
            confidence_sum = 0.0
            entropy_sum = 0.0
            for batch_index, (images, labels) in enumerate(batches):
                images = images.to(device)
                labels = labels.to(device)
                corrupted = apply_corruption(images, corruption, severity, seed=batch_index)
                if model_name == "vib":
                    output = model(corrupted)
                    logits = output.logits
                else:
                    logits, _latent = model(corrupted)
                predictions = logits.argmax(dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
                confidence_sum += float(_confidence_from_logits(logits).sum().detach().cpu())
                entropy_sum += float(_entropy_from_logits(logits).sum().detach().cpu())
            rows.append(
                {
                    "corruption": corruption,
                    "severity": float(severity),
                    "accuracy": correct / total if total else 0.0,
                    "mean_confidence": confidence_sum / total if total else 0.0,
                    "mean_entropy": entropy_sum / total if total else 0.0,
                }
            )
    return rows
