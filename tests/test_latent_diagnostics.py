import numpy as np

from vib_project.latent_diagnostics import compute_latent_drift, compute_latent_geometry


def test_compute_latent_geometry_returns_separation_metrics():
    latents = np.array([
        [0.0, 0.0], [0.1, 0.0],
        [5.0, 5.0], [5.1, 5.0],
    ])
    labels = np.array([0, 0, 1, 1])

    geometry = compute_latent_geometry(latents, labels)

    assert geometry["intra_class_variance"] > 0.0
    assert geometry["inter_class_distance"] > geometry["intra_class_variance"]
    assert geometry["fisher_ratio"] > 1.0
    assert len(geometry["per_class"]) == 2


def test_compute_latent_drift_groups_by_label():
    clean = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])
    corrupted = np.array([[1.0, 0.0], [2.0, 1.0], [6.0, 3.0]])
    labels = np.array([0, 0, 1])

    rows = compute_latent_drift(clean, corrupted, labels, corruption="gaussian", severity=0.3)

    assert rows[0]["corruption"] == "gaussian"
    assert rows[0]["severity"] == 0.3
    assert {row["label"] for row in rows} == {0, 1}
    assert all(row["mean_drift"] >= 0.0 for row in rows)
