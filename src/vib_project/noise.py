import torch


def add_gaussian_noise(images: torch.Tensor, sigma: float, seed: int | None = None) -> torch.Tensor:
    if sigma == 0.0:
        return images.clone()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=images.device)
        generator.manual_seed(seed)
    noise = torch.randn(images.shape, generator=generator, device=images.device, dtype=images.dtype) * sigma
    return torch.clamp(images + noise, 0.0, 1.0)
