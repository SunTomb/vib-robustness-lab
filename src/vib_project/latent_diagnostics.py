from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def _float(value: float | np.floating) -> float:
    return float(value)


def compute_latent_geometry(latents: np.ndarray, labels: np.ndarray, max_silhouette_points: int = 1000) -> dict[str, object]:
    unique_labels = np.unique(labels)
    centers: dict[int, np.ndarray] = {}
    per_class: list[dict[str, float | int]] = []
    intra_values: list[float] = []

    for label in unique_labels:
        class_latents = latents[labels == label]
        center = class_latents.mean(axis=0)
        centers[int(label)] = center
        distances = np.linalg.norm(class_latents - center, axis=1)
        intra_variance = float(distances.mean()) if len(distances) else 0.0
        intra_values.append(intra_variance)
        per_class.append(
            {
                "label": int(label),
                "intra_variance": intra_variance,
                "center_norm": _float(np.linalg.norm(center)),
            }
        )

    intra_class_variance = float(np.mean(intra_values)) if intra_values else 0.0
    inter_class_distance: float | None = None
    fisher_ratio: float | None = None
    if len(unique_labels) >= 2:
        distances = []
        labels_list = [int(label) for label in unique_labels]
        for index, left in enumerate(labels_list):
            for right in labels_list[index + 1:]:
                distances.append(float(np.linalg.norm(centers[left] - centers[right])))
        inter_class_distance = float(np.mean(distances)) if distances else None
        if inter_class_distance is not None:
            fisher_ratio = inter_class_distance / max(intra_class_variance, 1e-12)

    silhouette: float | None = None
    if len(unique_labels) >= 2 and len(latents) > 2:
        capped_latents = latents[:max_silhouette_points]
        capped_labels = labels[:max_silhouette_points]
        if len(np.unique(capped_labels)) >= 2:
            silhouette = float(silhouette_score(capped_latents, capped_labels))

    return {
        "intra_class_variance": intra_class_variance,
        "inter_class_distance": inter_class_distance,
        "fisher_ratio": fisher_ratio,
        "silhouette_score": silhouette,
        "per_class": per_class,
    }


def compute_latent_drift(
    clean_latents: np.ndarray,
    corrupted_latents: np.ndarray,
    labels: np.ndarray,
    corruption: str,
    severity: float,
) -> list[dict[str, float | int | str]]:
    distances = np.linalg.norm(clean_latents - corrupted_latents, axis=1)
    rows: list[dict[str, float | int | str]] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        label_distances = distances[labels == label]
        rows.append(
            {
                "corruption": corruption,
                "severity": float(severity),
                "label": label,
                "mean_drift": float(label_distances.mean()) if len(label_distances) else 0.0,
                "median_drift": float(np.median(label_distances)) if len(label_distances) else 0.0,
            }
        )
    return rows
