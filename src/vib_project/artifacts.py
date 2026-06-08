import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_experiment_artifacts(output_dir: Path, experiment_id: str, metrics: dict[str, Any]) -> None:
    experiment_dir = output_dir / experiment_id
    _write_json(experiment_dir / "metrics.json", metrics)
    curves = {
        "noise": [
            {"sigma": float(sigma), "accuracy": accuracy}
            for sigma, accuracy in metrics.get("noise_accuracy", {}).items()
        ]
    }
    _write_json(experiment_dir / "curves.json", curves)
    if "latent_pca" in metrics:
        _write_json(experiment_dir / "latent_pca.json", {"points": metrics["latent_pca"]})
    if "confusion_matrix" in metrics:
        _write_json(experiment_dir / "confusion_matrix.json", {"matrix": metrics["confusion_matrix"]})

    index_path = output_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"datasets": [], "experiments": []}

    dataset = str(metrics["dataset"])
    if dataset not in index["datasets"]:
        index["datasets"].append(dataset)

    entry = {
        "id": experiment_id,
        "dataset": dataset,
        "model": metrics["model"],
        "beta": metrics["beta"],
        "path": experiment_id,
    }
    index["experiments"] = [item for item in index["experiments"] if item["id"] != experiment_id]
    index["experiments"].append(entry)
    index["experiments"].sort(key=lambda item: item["id"])
    _write_json(index_path, index)
