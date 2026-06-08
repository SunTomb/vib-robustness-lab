from __future__ import annotations


def beta_warmup_value(epoch: int, warmup_epochs: int, beta_target: float) -> float:
    if warmup_epochs <= 0:
        return beta_target
    progress = min(max(epoch, 0) / warmup_epochs, 1.0)
    return beta_target * progress


def update_beta_for_target_kl(
    beta: float,
    kl: float,
    target_low: float,
    target_high: float,
    beta_min: float = 1e-6,
    beta_max: float = 1.0,
    factor: float = 1.1,
) -> float:
    next_beta = beta
    if kl > target_high:
        next_beta = beta * factor
    elif kl < target_low:
        next_beta = beta / factor
    return min(max(next_beta, beta_min), beta_max)
