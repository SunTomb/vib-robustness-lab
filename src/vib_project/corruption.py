from __future__ import annotations

import torch
import torch.nn.functional as F

CORRUPTIONS = ["gaussian", "salt_pepper", "blur", "contrast"]


def list_corruptions() -> list[str]:
    return list(CORRUPTIONS)


def corruption_parameter(name: str, severity: float) -> float:
    if severity < 0.0 or severity > 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    if name == "gaussian":
        return severity
    if name == "salt_pepper":
        return severity * 0.4
    if name == "blur":
        return severity
    if name == "contrast":
        return 1.0 - severity
    raise ValueError(f"Unsupported corruption: {name}")


def _generator_for(images: torch.Tensor, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=images.device)
    generator.manual_seed(seed)
    return generator


def _gaussian(images: torch.Tensor, severity: float, seed: int | None) -> torch.Tensor:
    sigma = corruption_parameter("gaussian", severity)
    generator = _generator_for(images, seed)
    noise = torch.randn(images.shape, generator=generator, device=images.device, dtype=images.dtype) * sigma
    return torch.clamp(images + noise, 0.0, 1.0)


def _salt_pepper(images: torch.Tensor, severity: float, seed: int | None) -> torch.Tensor:
    probability = corruption_parameter("salt_pepper", severity)
    generator = _generator_for(images, seed)
    random = torch.rand(images.shape, generator=generator, device=images.device, dtype=images.dtype)
    corrupted = images.clone()
    corrupted = torch.where(random < probability / 2.0, torch.zeros_like(corrupted), corrupted)
    corrupted = torch.where(random > 1.0 - probability / 2.0, torch.ones_like(corrupted), corrupted)
    return torch.clamp(corrupted, 0.0, 1.0)


def _blur(images: torch.Tensor, severity: float) -> torch.Tensor:
    amount = corruption_parameter("blur", severity)
    kernel_size = 3 if amount <= 0.25 else 5
    padding = kernel_size // 2
    channels = images.shape[1]
    kernel = torch.ones((channels, 1, kernel_size, kernel_size), device=images.device, dtype=images.dtype)
    kernel = kernel / float(kernel_size * kernel_size)
    return torch.clamp(F.conv2d(images, kernel, padding=padding, groups=channels), 0.0, 1.0)


def _contrast(images: torch.Tensor, severity: float) -> torch.Tensor:
    factor = corruption_parameter("contrast", severity)
    return torch.clamp((images - 0.5) * factor + 0.5, 0.0, 1.0)


def apply_corruption(images: torch.Tensor, name: str, severity: float, seed: int | None = None) -> torch.Tensor:
    corruption_parameter(name, severity)
    if severity == 0.0:
        return images.clone()
    if name == "gaussian":
        return _gaussian(images, severity, seed)
    if name == "salt_pepper":
        return _salt_pepper(images, severity, seed)
    if name == "blur":
        return _blur(images, severity)
    if name == "contrast":
        return _contrast(images, severity)
    raise ValueError(f"Unsupported corruption: {name}")
