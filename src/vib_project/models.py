from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VIBOutput:
    logits: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class ConvEncoder(nn.Module):
    def __init__(self, input_channels: int, image_size: int, latent_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        pooled = image_size // 4
        self.projection = nn.Sequential(
            nn.Linear(64 * pooled * pooled, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class CNNClassifier(nn.Module):
    def __init__(self, input_channels: int, image_size: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.encoder = ConvEncoder(input_channels, image_size, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits, latent


class VIBClassifier(nn.Module):
    def __init__(self, input_channels: int, image_size: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.encoder = ConvEncoder(input_channels, image_size, latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> VIBOutput:
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)
        return VIBOutput(logits=logits, z=z, mu=mu, logvar=logvar)
