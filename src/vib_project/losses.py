import torch
import torch.nn.functional as F


def kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_sample.mean()


def vib_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(logits, labels)
    kl = kl_normal_standard(mu, logvar)
    total = ce + beta * kl
    return total, {"ce": float(ce.detach().cpu()), "kl": float(kl.detach().cpu()), "total": float(total.detach().cpu())}
