from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_PHASE_FILES = ["metrics.json", "robustness.json", "latent_geometry.json", "phase_indicators.json"]
OPTIONAL_PHASE_FILES = ["latent_drift.json", "latent_pca.json", "confusion_matrix.json"]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_phase_experiment(root: Path, experiment_id: str, payload: dict[str, Any]) -> None:
    experiment_dir = root / experiment_id
    for key in ["metrics", "robustness", "latent_geometry", "phase_indicators", "latent_drift", "latent_pca", "confusion_matrix"]:
        if key in payload:
            _write_json(experiment_dir / f"{key}.json", payload[key])


def write_phase_index(root: Path, datasets: list[str], experiments: list[dict[str, Any]]) -> None:
    _write_json(root / "index.json", {"datasets": datasets, "experiments": experiments})


def write_phase_summary(root: Path, summary: dict[str, Any]) -> None:
    _write_json(root / "summary.json", summary)
