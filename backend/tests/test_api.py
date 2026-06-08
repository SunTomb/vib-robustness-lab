from fastapi.testclient import TestClient

from backend.app import create_app


def test_index_endpoint_reads_sample_artifacts():
    client = TestClient(create_app("artifacts/sample"))
    response = client.get("/api/index")
    assert response.status_code == 200
    payload = response.json()
    assert payload["datasets"] == ["mnist"]
    assert payload["experiments"][0]["id"] == "mnist_vib_beta_0_001"


def test_experiment_endpoint_reads_metrics_and_curves():
    client = TestClient(create_app("artifacts/sample"))
    response = client.get("/api/experiments/mnist_vib_beta_0_001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["average_kl"] == 3.7
    assert payload["curves"]["noise"][1]["sigma"] == 0.1
    assert len(payload["latent_pca"]["points"]) == 3
