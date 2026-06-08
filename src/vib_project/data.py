from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    input_channels: int
    image_size: int
    num_classes: int


def _limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def build_dataset(
    name: str,
    batch_size: int,
    data_dir: str | Path = "data",
    limit_train: int | None = None,
    limit_test: int | None = None,
) -> DatasetBundle:
    transform = transforms.ToTensor()
    root = Path(data_dir)
    normalized = name.lower().replace("-", "_")

    if normalized == "mnist":
        train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        channels, size, classes = 1, 28, 10
    elif normalized in {"fashion_mnist", "fashionmnist"}:
        train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        channels, size, classes = 1, 28, 10
    elif normalized == "cifar10":
        train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        channels, size, classes = 3, 32, 10
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train = _limit_dataset(train, limit_train)
    test = _limit_dataset(test, limit_test)
    return DatasetBundle(
        train_loader=DataLoader(train, batch_size=batch_size, shuffle=True),
        test_loader=DataLoader(test, batch_size=batch_size, shuffle=False),
        input_channels=channels,
        image_size=size,
        num_classes=classes,
    )
