import torch

from vib_project.noise import add_gaussian_noise


def test_zero_sigma_keeps_image_unchanged():
    images = torch.full((2, 1, 4, 4), 0.5)
    noisy = add_gaussian_noise(images, sigma=0.0, seed=1)
    assert torch.equal(noisy, images)


def test_gaussian_noise_clamps_to_unit_range():
    images = torch.full((4, 1, 8, 8), 0.5)
    noisy = add_gaussian_noise(images, sigma=10.0, seed=1)
    assert torch.all(noisy >= 0.0)
    assert torch.all(noisy <= 1.0)


def test_gaussian_noise_is_reproducible_with_seed():
    images = torch.full((2, 1, 4, 4), 0.5)
    first = add_gaussian_noise(images, sigma=0.2, seed=123)
    second = add_gaussian_noise(images, sigma=0.2, seed=123)
    assert torch.equal(first, second)
