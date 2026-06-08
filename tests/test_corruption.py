import torch

from vib_project.corruption import apply_corruption, corruption_parameter, list_corruptions


def test_list_corruptions_contains_required_names():
    assert list_corruptions() == ["gaussian", "salt_pepper", "blur", "contrast"]


def test_zero_severity_preserves_images_for_all_corruptions():
    images = torch.rand(4, 3, 16, 16)
    for name in list_corruptions():
        corrupted = apply_corruption(images, name, severity=0.0, seed=7)
        assert torch.allclose(corrupted, images)


def test_gaussian_corruption_is_seed_reproducible_and_clamped():
    images = torch.full((2, 1, 8, 8), 0.5)
    first = apply_corruption(images, "gaussian", severity=0.3, seed=11)
    second = apply_corruption(images, "gaussian", severity=0.3, seed=11)
    assert torch.allclose(first, second)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
    assert not torch.allclose(first, images)


def test_salt_pepper_corruption_changes_pixels_and_is_clamped():
    images = torch.full((2, 1, 12, 12), 0.5)
    corrupted = apply_corruption(images, "salt_pepper", severity=0.4, seed=13)
    assert float(corrupted.min()) >= 0.0
    assert float(corrupted.max()) <= 1.0
    assert torch.any(corrupted == 0.0) or torch.any(corrupted == 1.0)


def test_contrast_parameter_mapping():
    assert corruption_parameter("contrast", 0.4) == 0.6


def test_unknown_corruption_raises_value_error():
    images = torch.rand(1, 1, 8, 8)
    try:
        apply_corruption(images, "unknown", severity=0.1)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError")
