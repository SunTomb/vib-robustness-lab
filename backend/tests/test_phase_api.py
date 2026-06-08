from fastapi.testclient import TestClient

from backend.app import create_app


def test_phase_index_endpoint_reads_sample_phase_artifacts():
    app = create_app(artifact_dir="artifacts/sample", phase_artifact_dir="artifacts/sample_phase")
    client = TestClient(app)
    response = client.get("/api/phase/index")
    assert response.status_code == 200
    assert response.json()["datasets"] == ["mnist"]


def test_phase_summary_endpoint_reads_sample_summary():
    app = create_app(artifact_dir="artifacts/sample", phase_artifact_dir="artifacts/sample_phase")
    client = TestClient(app)
    response = client.get("/api/phase/summary")
    assert response.status_code == 200
    assert response.json()["datasets"]["mnist"]["best_clean_experiment"] == "mnist_vib_beta_0_001"


def test_phase_experiment_endpoint_reads_detail_files():
    app = create_app(artifact_dir="artifacts/sample", phase_artifact_dir="artifacts/sample_phase")
    client = TestClient(app)
    response = client.get("/api/phase/experiments/mnist_vib_beta_0_001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["dataset"] == "mnist"
    assert payload["phase_indicators"]["phase_label"] == "useful-compression"
    assert payload["latent_drift"]["rows"][0]["mean_drift"] == 0.4
