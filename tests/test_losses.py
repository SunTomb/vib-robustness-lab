import torch

from vib_project.losses import kl_normal_standard, vib_loss


def test_kl_zero_for_standard_normal_parameters():
    mu = torch.zeros(3, 5)
    logvar = torch.zeros(3, 5)
    kl = kl_normal_standard(mu, logvar)
    assert torch.isclose(kl, torch.tensor(0.0))


def test_kl_is_non_negative():
    mu = torch.ones(3, 5)
    logvar = torch.zeros(3, 5)
    kl = kl_normal_standard(mu, logvar)
    assert kl.item() >= 0.0


def test_vib_loss_returns_components():
    logits = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 2, 3])
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    total, components = vib_loss(logits, labels, mu, logvar, beta=0.01)
    assert total.item() > 0.0
    assert set(components) == {"ce", "kl", "total"}
