import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def pca_projection(latents: np.ndarray, labels: np.ndarray, max_points: int = 1000) -> list[dict[str, float | int]]:
    if len(latents) == 0:
        return []
    count = min(max_points, len(latents))
    selected_latents = latents[:count]
    selected_labels = labels[:count]
    coords = PCA(n_components=2, random_state=7).fit_transform(selected_latents)
    return [
        {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "label": int(selected_labels[i])}
        for i in range(count)
    ]


def confusion_matrix_payload(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> list[list[int]]:
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    return matrix.astype(int).tolist()
