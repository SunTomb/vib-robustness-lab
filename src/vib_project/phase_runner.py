from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vib_project.config import ExperimentConfig
from vib_project.corruption import apply_corruption
from vib_project.data import build_dataset
from vib_project.evaluate import collect_outputs, evaluate_corruption_grid, evaluate_model
from vib_project.latent_diagnostics import compute_latent_drift, compute_latent_geometry
from vib_project.losses import vib_loss
from vib_project.phase_metrics import build_phase_indicators, compute_normalized_robustness_auc, compute_robustness_auc
from vib_project.projections import confusion_matrix_payload, pca_projection
from vib_project.train import _build_model, _configure_torch_backend, _device, _seed_everything


def _train_model(config: ExperimentConfig):
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
    return model, bundle, device


def _collect_latents_with_corruption(model, batches, device: torch.device, model_name: str, corruption: str, severity: float, limit: int):
    latents = []
    labels_all = []
    seen = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(batches):
            images = images.to(device)
            labels = labels.to(device)
            images = apply_corruption(images, corruption, severity, seed=batch_index)
            if model_name == "vib":
                output = model(images)
                latent = output.mu
            else:
                _logits, latent = model(images)
            remaining = limit - seen
            if remaining <= 0:
                break
            latents.append(latent[:remaining].detach().cpu().numpy())
            labels_all.append(labels[:remaining].detach().cpu().numpy())
            seen += min(remaining, labels.numel())
    return np.concatenate(latents, axis=0), np.concatenate(labels_all, axis=0)


def run_phase_experiment(
    config: ExperimentConfig,
    corruptions: list[str],
    severities: list[float],
    max_diagnostic_points: int = 1000,
    baseline_normalized_auc: dict[str, float | None] | None = None,
    beta0_kl: float | None = None,
    best_accuracy: float | None = None,
) -> dict[str, Any]:
    model, bundle, device = _train_model(config)
    train_eval = evaluate_model(model, bundle.train_loader, device, config.model)
    test_eval = evaluate_model(model, bundle.test_loader, device, config.model)
    robustness_rows = evaluate_corruption_grid(model, bundle.test_loader, device, config.model, corruptions, severities)
    collected = collect_outputs(model, bundle.test_loader, device, config.model)
    latent_pca = pca_projection(collected["latents"], collected["labels"])
    confusion = confusion_matrix_payload(collected["labels"], collected["predictions"], bundle.num_classes)
    diagnostic_latents = collected["latents"][:max_diagnostic_points]
    diagnostic_labels = collected["labels"][:max_diagnostic_points]
    latent_geometry = compute_latent_geometry(diagnostic_latents, diagnostic_labels)

    drift_rows = []
    clean_latents = diagnostic_latents
    for corruption in corruptions:
        for severity in severities:
            if severity == 0.0:
                continue
            corrupted_latents, corrupted_labels = _collect_latents_with_corruption(
                model, bundle.test_loader, device, config.model, corruption, severity, max_diagnostic_points
            )
            count = min(len(clean_latents), len(corrupted_latents), len(diagnostic_labels), len(corrupted_labels))
            drift_rows.extend(
                compute_latent_drift(
                    clean_latents[:count],
                    corrupted_latents[:count],
                    diagnostic_labels[:count],
                    corruption,
                    severity,
                )
            )

    metrics = {
        "dataset": config.dataset,
        "model": config.model,
        "beta": config.beta,
        "train_accuracy": train_eval["accuracy"],
        "test_accuracy": test_eval["accuracy"],
        "train_test_gap": train_eval["accuracy"] - test_eval["accuracy"],
        "average_kl": test_eval["average_kl"],
    }
    robustness_auc = compute_robustness_auc(robustness_rows)
    default_baseline = baseline_normalized_auc or compute_normalized_robustness_auc(robustness_auc, test_eval["accuracy"])
    phase_indicators = build_phase_indicators(
        robustness_rows,
        clean_accuracy=test_eval["accuracy"],
        baseline_normalized_auc=default_baseline,
        average_kl=test_eval["average_kl"],
        beta0_kl=beta0_kl if beta0_kl is not None else max(test_eval["average_kl"], 1e-12),
        best_accuracy=best_accuracy if best_accuracy is not None else test_eval["accuracy"],
    )
    return {
        "metrics": metrics,
        "robustness": {"dataset": config.dataset, "experiment_id": config.experiment_id, "rows": robustness_rows},
        "latent_geometry": latent_geometry,
        "latent_drift": {"rows": drift_rows},
        "phase_indicators": phase_indicators,
        "latent_pca": {"points": latent_pca},
        "confusion_matrix": {"matrix": confusion},
    }
