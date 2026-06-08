import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


def _read_json(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing artifact: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_app(
    artifact_dir: str | os.PathLike = "artifacts/full",
    phase_artifact_dir: str | os.PathLike | None = None,
) -> FastAPI:
    root = Path(artifact_dir)
    phase_root = Path(phase_artifact_dir or os.environ.get("VIB_PHASE_ARTIFACT_DIR", "artifacts/phase"))
    app = FastAPI(title="VIB Robust Generalization API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True, "artifact_dir": str(root)}

    @app.get("/api/index")
    def index():
        return _read_json(root / "index.json")

    @app.get("/api/experiments/{experiment_id}")
    def experiment(experiment_id: str):
        index_payload = _read_json(root / "index.json")
        matches = [item for item in index_payload["experiments"] if item["id"] == experiment_id]
        if not matches:
            raise HTTPException(status_code=404, detail="Unknown experiment")
        experiment_dir = root / matches[0]["path"]
        return {
            "metadata": matches[0],
            "metrics": _read_json(experiment_dir / "metrics.json"),
            "curves": _read_json(experiment_dir / "curves.json"),
            "latent_pca": _read_json(experiment_dir / "latent_pca.json"),
        }

    @app.get("/api/phase/index")
    def phase_index():
        return _read_json(phase_root / "index.json")

    @app.get("/api/phase/summary")
    def phase_summary():
        return _read_json(phase_root / "summary.json")

    @app.get("/api/phase/experiments/{experiment_id}")
    def phase_experiment(experiment_id: str):
        index_payload = _read_json(phase_root / "index.json")
        matches = [item for item in index_payload["experiments"] if item["id"] == experiment_id]
        if not matches:
            raise HTTPException(status_code=404, detail="Unknown phase experiment")
        experiment_dir = phase_root / matches[0]["path"]
        payload = {
            "metadata": matches[0],
            "metrics": _read_json(experiment_dir / "metrics.json"),
            "robustness": _read_json(experiment_dir / "robustness.json"),
            "latent_geometry": _read_json(experiment_dir / "latent_geometry.json"),
            "phase_indicators": _read_json(experiment_dir / "phase_indicators.json"),
        }
        latent_drift_path = experiment_dir / "latent_drift.json"
        if latent_drift_path.exists():
            payload["latent_drift"] = _read_json(latent_drift_path)
        latent_pca_path = experiment_dir / "latent_pca.json"
        if latent_pca_path.exists():
            payload["latent_pca"] = _read_json(latent_pca_path)
        confusion_path = experiment_dir / "confusion_matrix.json"
        if confusion_path.exists():
            payload["confusion_matrix"] = _read_json(confusion_path)
        return payload

    @app.get("/api/phase/diagnostics")
    def phase_diagnostics():
        diagnostics_dir = phase_root / "diagnostics"
        return {
            "sample_trajectories": _read_json(diagnostics_dir / "sample_trajectories.json") if (diagnostics_dir / "sample_trajectories.json").exists() else {"samples": []},
            "confusion_flows": _read_json(diagnostics_dir / "confusion_flows.json") if (diagnostics_dir / "confusion_flows.json").exists() else {"flows": []},
            "class_geometry": _read_json(diagnostics_dir / "class_geometry.json") if (diagnostics_dir / "class_geometry.json").exists() else {"classes": []},
        }

    @app.get("/api/phase/adaptive/{experiment_id}")
    def phase_adaptive(experiment_id: str):
        adaptive_dir = phase_root / "adaptive" / experiment_id
        return {
            "metrics": _read_json(adaptive_dir / "metrics.json"),
            "adaptive_curves": _read_json(adaptive_dir / "adaptive_curves.json"),
            "robustness": _read_json(adaptive_dir / "robustness.json"),
        }

    return app


app = create_app(os.environ.get("VIB_ARTIFACT_DIR", "artifacts/full"))
