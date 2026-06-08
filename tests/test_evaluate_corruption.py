import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vib_project.evaluate import evaluate_corruption_grid


class ConstantModel(nn.Module):
    def forward(self, images):
        logits = torch.zeros(images.shape[0], 3, device=images.device)
        logits[:, 1] = 3.0
        latent = torch.ones(images.shape[0], 4, device=images.device)
        return logits, latent


def test_evaluate_corruption_grid_returns_rows():
    images = torch.rand(5, 1, 8, 8)
    labels = torch.ones(5, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    rows = evaluate_corruption_grid(
        ConstantModel(),
        loader,
        torch.device("cpu"),
        model_name="cnn",
        corruptions=["gaussian", "contrast"],
        severities=[0.0, 0.2],
    )

    assert len(rows) == 4
    assert rows[0]["corruption"] == "gaussian"
    assert rows[0]["severity"] == 0.0
    assert rows[0]["accuracy"] == 1.0
    assert 0.0 <= rows[0]["mean_confidence"] <= 1.0
    assert rows[0]["mean_entropy"] >= 0.0
