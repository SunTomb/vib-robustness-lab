import torch

from vib_project.models import CNNClassifier, VIBClassifier


def test_cnn_classifier_shapes():
    model = CNNClassifier(input_channels=1, image_size=28, latent_dim=16, num_classes=10)
    logits, latent = model(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)
    assert latent.shape == (4, 16)


def test_vib_classifier_shapes():
    model = VIBClassifier(input_channels=1, image_size=28, latent_dim=16, num_classes=10)
    output = model(torch.randn(4, 1, 28, 28))
    assert output.logits.shape == (4, 10)
    assert output.z.shape == (4, 16)
    assert output.mu.shape == (4, 16)
    assert output.logvar.shape == (4, 16)


def test_cifar_cnn_classifier_shapes():
    model = CNNClassifier(input_channels=3, image_size=32, latent_dim=16, num_classes=10)
    logits, latent = model(torch.rand(4, 3, 32, 32))
    assert logits.shape == (4, 10)
    assert latent.shape == (4, 16)


def test_cifar_vib_classifier_shapes():
    model = VIBClassifier(input_channels=3, image_size=32, latent_dim=16, num_classes=10)
    output = model(torch.rand(4, 3, 32, 32))
    assert output.logits.shape == (4, 10)
    assert output.z.shape == (4, 16)
    assert output.mu.shape == (4, 16)
    assert output.logvar.shape == (4, 16)
