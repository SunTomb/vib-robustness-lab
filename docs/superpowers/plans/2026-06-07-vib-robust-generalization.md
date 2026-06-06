# VIB Robust Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Variational Information Bottleneck experiment suite and interactive web demo for analyzing representation compression and robust generalization on image classification.

**Architecture:** Offline Python experiments produce static artifacts; a FastAPI backend serves those artifacts; a React/Vite frontend visualizes information planes, latent spaces, and robustness curves. Training never runs inside the web demo.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, scikit-learn, FastAPI, pytest, React, TypeScript, Vite, Recharts, Vitest.

---

## File Structure

```text
信息论/大作业/
├── README.md
├── pyproject.toml
├── package.json
├── .gitignore
├── artifacts/
│   ├── sample/index.json
│   └── sample/mnist/vib_beta_0.001/{metrics.json,curves.json,latent_pca.json,confusion_matrix.json}
├── src/vib_project/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── noise.py
│   ├── models.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── artifacts.py
│   ├── projections.py
│   └── cli.py
├── backend/
│   ├── app.py
│   └── tests/test_api.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api.ts
│       ├── types.ts
│       ├── lib/metrics.ts
│       └── components/{MetricCards.tsx,InformationPlane.tsx,RobustnessChart.tsx,LatentPlot.tsx,ExperimentSelector.tsx}
├── tests/
│   ├── test_losses.py
│   ├── test_models.py
│   ├── test_noise.py
│   ├── test_artifacts.py
│   └── test_tiny_training.py
└── report/
    ├── outline.md
    └── figures/
```

Responsibility boundaries:

- `src/vib_project/*`: offline experiment code only.
- `artifacts/*`: generated experiment outputs consumed by backend and frontend.
- `backend/*`: read-only artifact API, no training.
- `frontend/*`: visualization and explanation UI.
- `report/*`: final report outline and exported figures.

---

## Task 1: Project Scaffold and Python Package

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/vib_project/__init__.py`
- Create: `src/vib_project/config.py`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Create README**

Write `README.md`:

```markdown
# VIB Robust Generalization

《信息论》课程大作业：复现变分信息瓶颈（Variational Information Bottleneck, VIB），并分析表示压缩对图像分类鲁棒泛化的影响。

## Core Question

How does the bottleneck coefficient beta control the tradeoff between input information compression, label prediction, and robustness to noisy inputs?

## Components

- Offline experiments: PyTorch training and evaluation.
- Artifacts: JSON metrics, robustness curves, latent projections.
- Backend: FastAPI artifact service.
- Frontend: React interactive information-bottleneck dashboard.

## Minimal Workflow

```bash
python -m pytest
python -m vib_project.cli run --dataset mnist --model vib --beta 0.001 --epochs 1 --limit-train 512 --limit-test 256 --output-dir artifacts/dev
uvicorn backend.app:app --reload
```
```

- [ ] **Step 2: Create gitignore**

Write `.gitignore`:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.pyc
.DS_Store
node_modules/
frontend/dist/
artifacts/dev/
artifacts/full/
checkpoints/
*.pt
*.pth
.env
```

- [ ] **Step 3: Create Python project config**

Write `pyproject.toml`:

```toml
[project]
name = "vib-robust-generalization"
version = "0.1.0"
description = "Variational Information Bottleneck experiments for robust image classification."
requires-python = ">=3.10"
dependencies = [
  "torch>=2.2",
  "torchvision>=0.17",
  "numpy>=1.26",
  "scikit-learn>=1.4",
  "fastapi>=0.111",
  "uvicorn>=0.29",
  "pydantic>=2.7",
  "pytest>=8.2"
]

[project.scripts]
vib-exp = "vib_project.cli:main"

[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests", "backend/tests"]
```

- [ ] **Step 4: Create package init and config**

Write `src/vib_project/__init__.py`:

```python
__all__ = ["__version__"]
__version__ = "0.1.0"
```

Write `src/vib_project/config.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    model: str
    beta: float
    epochs: int
    batch_size: int
    latent_dim: int
    learning_rate: float
    seed: int
    output_dir: Path
    limit_train: int | None = None
    limit_test: int | None = None

    @property
    def experiment_id(self) -> str:
        if self.model == "cnn":
            return f"{self.dataset}_cnn_baseline"
        beta_text = f"{self.beta:g}".replace(".", "_").replace("-", "m")
        return f"{self.dataset}_{self.model}_beta_{beta_text}"
```

- [ ] **Step 5: Write import test**

Write `tests/test_imports.py`:

```python
from pathlib import Path

from vib_project.config import ExperimentConfig


def test_experiment_id_for_vib():
    config = ExperimentConfig(
        dataset="mnist",
        model="vib",
        beta=0.001,
        epochs=1,
        batch_size=32,
        latent_dim=16,
        learning_rate=1e-3,
        seed=7,
        output_dir=Path("artifacts/dev"),
    )
    assert config.experiment_id == "mnist_vib_beta_0_001"


def test_experiment_id_for_cnn():
    config = ExperimentConfig(
        dataset="fashion_mnist",
        model="cnn",
        beta=0.0,
        epochs=1,
        batch_size=32,
        latent_dim=16,
        learning_rate=1e-3,
        seed=7,
        output_dir=Path("artifacts/dev"),
    )
    assert config.experiment_id == "fashion_mnist_cnn_baseline"
```

- [ ] **Step 6: Run scaffold tests**

Run:

```bash
python -m pytest tests/test_imports.py -v
```

Expected: 2 tests pass.

- [ ] **Step 7: Commit**

```bash
git add README.md .gitignore pyproject.toml src/vib_project/__init__.py src/vib_project/config.py tests/test_imports.py
git commit -m "chore: scaffold VIB project"
```

---

## Task 2: Noise and Dataset Utilities

**Files:**
- Create: `src/vib_project/noise.py`
- Create: `src/vib_project/data.py`
- Test: `tests/test_noise.py`

- [ ] **Step 1: Write failing noise tests**

Write `tests/test_noise.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_noise.py -v
```

Expected: FAIL because `vib_project.noise` does not exist.

- [ ] **Step 3: Implement noise utility**

Write `src/vib_project/noise.py`:

```python
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
```

- [ ] **Step 4: Implement dataset loader**

Write `src/vib_project/data.py`:

```python
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
```

- [ ] **Step 5: Run utility tests**

Run:

```bash
python -m pytest tests/test_noise.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/vib_project/noise.py src/vib_project/data.py tests/test_noise.py
git commit -m "feat: add dataset and noise utilities"
```

---

## Task 3: CNN and VIB Models

**Files:**
- Create: `src/vib_project/models.py`
- Create: `src/vib_project/losses.py`
- Test: `tests/test_models.py`
- Test: `tests/test_losses.py`

- [ ] **Step 1: Write model and loss tests**

Write `tests/test_models.py`:

```python
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
```

Write `tests/test_losses.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_models.py tests/test_losses.py -v
```

Expected: FAIL because model and loss modules do not exist.

- [ ] **Step 3: Implement losses**

Write `src/vib_project/losses.py`:

```python
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
```

- [ ] **Step 4: Implement models**

Write `src/vib_project/models.py`:

```python
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
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_models.py tests/test_losses.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/vib_project/models.py src/vib_project/losses.py tests/test_models.py tests/test_losses.py
git commit -m "feat: add CNN and VIB models"
```

---

## Task 4: Training and Evaluation Loop

**Files:**
- Create: `src/vib_project/train.py`
- Create: `src/vib_project/evaluate.py`
- Test: `tests/test_tiny_training.py`

- [ ] **Step 1: Write tiny training test**

Write `tests/test_tiny_training.py`:

```python
from pathlib import Path

from vib_project.config import ExperimentConfig
from vib_project.train import run_training


def test_tiny_training_outputs_metrics(tmp_path: Path):
    config = ExperimentConfig(
        dataset="mnist",
        model="vib",
        beta=0.001,
        epochs=1,
        batch_size=64,
        latent_dim=8,
        learning_rate=1e-3,
        seed=3,
        output_dir=tmp_path,
        limit_train=128,
        limit_test=64,
    )
    result = run_training(config)
    assert result["dataset"] == "mnist"
    assert result["model"] == "vib"
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["average_kl"] >= 0.0
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_tiny_training.py -v
```

Expected: FAIL because `run_training` does not exist.

- [ ] **Step 3: Implement evaluation**

Write `src/vib_project/evaluate.py`:

```python
from collections.abc import Iterable

import torch
from torch import nn

from vib_project.losses import kl_normal_standard
from vib_project.models import CNNClassifier, VIBClassifier
from vib_project.noise import add_gaussian_noise


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    batches: Iterable,
    device: torch.device,
    model_name: str,
    noise_sigma: float = 0.0,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    kl_sum = 0.0
    batch_count = 0

    for images, labels in batches:
        images = images.to(device)
        labels = labels.to(device)
        if noise_sigma > 0.0:
            images = add_gaussian_noise(images, sigma=noise_sigma)

        if model_name == "vib":
            output = model(images)
            logits = output.logits
            kl_sum += float(kl_normal_standard(output.mu, output.logvar).detach().cpu())
        else:
            logits, _ = model(images)

        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
        batch_count += 1

    accuracy = correct / total if total else 0.0
    average_kl = kl_sum / batch_count if batch_count and model_name == "vib" else 0.0
    return {"accuracy": accuracy, "average_kl": average_kl}
```

- [ ] **Step 4: Implement training**

Write `src/vib_project/train.py`:

```python
import random

import numpy as np
import torch
import torch.nn.functional as F

from vib_project.config import ExperimentConfig
from vib_project.data import build_dataset
from vib_project.evaluate import evaluate_model
from vib_project.losses import vib_loss
from vib_project.models import CNNClassifier, VIBClassifier


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(config: ExperimentConfig, input_channels: int, image_size: int, num_classes: int):
    if config.model == "cnn":
        return CNNClassifier(input_channels, image_size, config.latent_dim, num_classes)
    if config.model == "vib":
        return VIBClassifier(input_channels, image_size, config.latent_dim, num_classes)
    raise ValueError(f"Unsupported model: {config.model}")


def run_training(config: ExperimentConfig) -> dict[str, float | str | dict[str, float]]:
    _seed_everything(config.seed)
    device = _device()
    bundle = build_dataset(
        config.dataset,
        batch_size=config.batch_size,
        limit_train=config.limit_train,
        limit_test=config.limit_test,
    )
    model = _build_model(config, bundle.input_channels, bundle.image_size, bundle.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for _epoch in range(config.epochs):
        model.train()
        for images, labels in bundle.train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if config.model == "vib":
                output = model(images)
                loss, _components = vib_loss(output.logits, labels, output.mu, output.logvar, beta=config.beta)
            else:
                logits, _latent = model(images)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

    train_eval = evaluate_model(model, bundle.train_loader, device, config.model)
    test_eval = evaluate_model(model, bundle.test_loader, device, config.model)
    noise_accuracy: dict[str, float] = {}
    for sigma in [0.0, 0.1, 0.2, 0.3, 0.4]:
        noisy = evaluate_model(model, bundle.test_loader, device, config.model, noise_sigma=sigma)
        noise_accuracy[f"{sigma:.1f}"] = noisy["accuracy"]

    return {
        "dataset": config.dataset,
        "model": config.model,
        "beta": config.beta,
        "train_accuracy": train_eval["accuracy"],
        "test_accuracy": test_eval["accuracy"],
        "train_test_gap": train_eval["accuracy"] - test_eval["accuracy"],
        "average_kl": test_eval["average_kl"],
        "noise_accuracy": noise_accuracy,
    }
```

- [ ] **Step 5: Run tiny training test**

Run:

```bash
python -m pytest tests/test_tiny_training.py -v
```

Expected: 1 test passes. First run downloads MNIST.

- [ ] **Step 6: Commit**

```bash
git add src/vib_project/train.py src/vib_project/evaluate.py tests/test_tiny_training.py
git commit -m "feat: add training and evaluation loop"
```

---

## Task 5: Artifact Writing and CLI

**Files:**
- Create: `src/vib_project/artifacts.py`
- Create: `src/vib_project/cli.py`
- Test: `tests/test_artifacts.py`

- [ ] **Step 1: Write artifact test**

Write `tests/test_artifacts.py`:

```python
import json
from pathlib import Path

from vib_project.artifacts import write_experiment_artifacts


def test_write_experiment_artifacts(tmp_path: Path):
    metrics = {
        "dataset": "mnist",
        "model": "vib",
        "beta": 0.001,
        "train_accuracy": 0.9,
        "test_accuracy": 0.8,
        "train_test_gap": 0.1,
        "average_kl": 2.0,
        "noise_accuracy": {"0.0": 0.8, "0.1": 0.7},
    }
    write_experiment_artifacts(tmp_path, "mnist_vib_beta_0_001", metrics)
    metrics_path = tmp_path / "mnist_vib_beta_0_001" / "metrics.json"
    index_path = tmp_path / "index.json"
    assert metrics_path.exists()
    assert index_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert saved["average_kl"] == 2.0
    assert index["experiments"][0]["id"] == "mnist_vib_beta_0_001"
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_artifacts.py -v
```

Expected: FAIL because `artifacts.py` does not exist.

- [ ] **Step 3: Implement artifact writer**

Write `src/vib_project/artifacts.py`:

```python
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_experiment_artifacts(output_dir: Path, experiment_id: str, metrics: dict[str, Any]) -> None:
    experiment_dir = output_dir / experiment_id
    _write_json(experiment_dir / "metrics.json", metrics)
    curves = {
        "noise": [
            {"sigma": float(sigma), "accuracy": accuracy}
            for sigma, accuracy in metrics.get("noise_accuracy", {}).items()
        ]
    }
    _write_json(experiment_dir / "curves.json", curves)

    index_path = output_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"datasets": [], "experiments": []}

    dataset = str(metrics["dataset"])
    if dataset not in index["datasets"]:
        index["datasets"].append(dataset)

    entry = {
        "id": experiment_id,
        "dataset": dataset,
        "model": metrics["model"],
        "beta": metrics["beta"],
        "path": experiment_id,
    }
    index["experiments"] = [item for item in index["experiments"] if item["id"] != experiment_id]
    index["experiments"].append(entry)
    index["experiments"].sort(key=lambda item: item["id"])
    _write_json(index_path, index)
```

- [ ] **Step 4: Implement CLI**

Write `src/vib_project/cli.py`:

```python
import argparse
from pathlib import Path

from vib_project.artifacts import write_experiment_artifacts
from vib_project.config import ExperimentConfig
from vib_project.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VIB robust generalization experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"], required=True)
    run.add_argument("--model", choices=["cnn", "vib"], required=True)
    run.add_argument("--beta", type=float, default=0.0)
    run.add_argument("--epochs", type=int, default=5)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--latent-dim", type=int, default=32)
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--limit-train", type=int, default=None)
    run.add_argument("--limit-test", type=int, default=None)
    run.add_argument("--output-dir", type=Path, default=Path("artifacts/dev"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        dataset=args.dataset,
        model=args.model,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    metrics = run_training(config)
    write_experiment_artifacts(config.output_dir, config.experiment_id, metrics)
    print(f"Wrote artifacts for {config.experiment_id} to {config.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run artifact tests**

Run:

```bash
python -m pytest tests/test_artifacts.py -v
```

Expected: 1 test passes.

- [ ] **Step 6: Run a tiny CLI experiment**

Run:

```bash
python -m vib_project.cli run --dataset mnist --model vib --beta 0.001 --epochs 1 --limit-train 128 --limit-test 64 --output-dir artifacts/dev
```

Expected: command prints `Wrote artifacts for mnist_vib_beta_0_001 to artifacts/dev` and creates `artifacts/dev/index.json`.

- [ ] **Step 7: Commit**

```bash
git add src/vib_project/artifacts.py src/vib_project/cli.py tests/test_artifacts.py
git commit -m "feat: write experiment artifacts from CLI"
```

---

## Task 6: Latent Projection and Confusion Matrix Artifacts

**Files:**
- Create: `src/vib_project/projections.py`
- Modify: `src/vib_project/evaluate.py`
- Modify: `src/vib_project/train.py`
- Modify: `src/vib_project/artifacts.py`
- Test: `tests/test_artifacts.py`

- [ ] **Step 1: Extend artifact test**

Append to `tests/test_artifacts.py`:

```python

def test_write_projection_and_confusion_artifacts(tmp_path: Path):
    metrics = {
        "dataset": "mnist",
        "model": "vib",
        "beta": 0.001,
        "train_accuracy": 0.9,
        "test_accuracy": 0.8,
        "train_test_gap": 0.1,
        "average_kl": 2.0,
        "noise_accuracy": {"0.0": 0.8},
        "latent_pca": [{"x": 0.1, "y": 0.2, "label": 3}],
        "confusion_matrix": [[1, 0], [0, 1]],
    }
    write_experiment_artifacts(tmp_path, "mnist_vib_beta_0_001", metrics)
    assert (tmp_path / "mnist_vib_beta_0_001" / "latent_pca.json").exists()
    assert (tmp_path / "mnist_vib_beta_0_001" / "confusion_matrix.json").exists()
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_artifacts.py::test_write_projection_and_confusion_artifacts -v
```

Expected: FAIL because writer does not emit these files.

- [ ] **Step 3: Implement projection utility**

Write `src/vib_project/projections.py`:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def pca_projection(latents: np.ndarray, labels: np.ndarray, max_points: int = 1000) -> list[dict[str, float | int]]:
    if len(latents) == 0:
        return []
    count = min(max_points, len(latents))
    selected_latents = latents[:count]
    selected_labels = labels[:count]
    coords = PCA(n_components=2, random_state=7).fit_transform(selected_latents)
    return [
        {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "label": int(selected_labels[i])}
        for i in range(count)
    ]


def confusion_matrix_payload(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> list[list[int]]:
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    return matrix.astype(int).tolist()
```

- [ ] **Step 4: Extend evaluation to collect latents and predictions**

Add this function to `src/vib_project/evaluate.py`:

```python
@torch.no_grad()
def collect_outputs(model: nn.Module, batches: Iterable, device: torch.device, model_name: str) -> dict[str, np.ndarray]:
    import numpy as np

    model.eval()
    latents = []
    labels_all = []
    predictions_all = []
    for images, labels in batches:
        images = images.to(device)
        labels = labels.to(device)
        if model_name == "vib":
            output = model(images)
            logits = output.logits
            latent = output.mu
        else:
            logits, latent = model(images)
        predictions = logits.argmax(dim=1)
        latents.append(latent.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        predictions_all.append(predictions.detach().cpu().numpy())
    return {
        "latents": np.concatenate(latents, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "predictions": np.concatenate(predictions_all, axis=0),
    }
```

Also add `import numpy as np` inside the function only, as shown, to keep top-level imports minimal.

- [ ] **Step 5: Extend training result**

Modify `src/vib_project/train.py` imports:

```python
from vib_project.evaluate import collect_outputs, evaluate_model
from vib_project.projections import confusion_matrix_payload, pca_projection
```

Before the final `return`, add:

```python
    collected = collect_outputs(model, bundle.test_loader, device, config.model)
    latent_pca = pca_projection(collected["latents"], collected["labels"])
    confusion = confusion_matrix_payload(collected["labels"], collected["predictions"], bundle.num_classes)
```

Then include these keys in the returned dictionary:

```python
        "latent_pca": latent_pca,
        "confusion_matrix": confusion,
```

- [ ] **Step 6: Extend artifact writer**

In `src/vib_project/artifacts.py`, after writing `curves.json`, add:

```python
    if "latent_pca" in metrics:
        _write_json(experiment_dir / "latent_pca.json", {"points": metrics["latent_pca"]})
    if "confusion_matrix" in metrics:
        _write_json(experiment_dir / "confusion_matrix.json", {"matrix": metrics["confusion_matrix"]})
```

- [ ] **Step 7: Run tests**

Run:

```bash
python -m pytest tests/test_artifacts.py tests/test_tiny_training.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/vib_project/projections.py src/vib_project/evaluate.py src/vib_project/train.py src/vib_project/artifacts.py tests/test_artifacts.py
git commit -m "feat: export latent and confusion artifacts"
```

---

## Task 7: Full Experiment Runner Script

**Files:**
- Create: `scripts/run_core_experiments.sh`
- Create: `scripts/run_tiny_smoke.sh`

- [ ] **Step 1: Create smoke script**

Write `scripts/run_tiny_smoke.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m vib_project.cli run --dataset mnist --model cnn --beta 0 --epochs 1 --limit-train 256 --limit-test 128 --output-dir artifacts/dev
python -m vib_project.cli run --dataset mnist --model vib --beta 0.001 --epochs 1 --limit-train 256 --limit-test 128 --output-dir artifacts/dev
```

- [ ] **Step 2: Create core experiment script**

Write `scripts/run_core_experiments.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="artifacts/full"
EPOCHS="10"
BATCH_SIZE="128"
LATENT_DIM="32"

for DATASET in mnist fashion_mnist; do
  python -m vib_project.cli run --dataset "$DATASET" --model cnn --beta 0 --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --latent-dim "$LATENT_DIM" --output-dir "$OUTPUT_DIR"
  for BETA in 0 0.0001 0.001 0.01 0.1 1; do
    python -m vib_project.cli run --dataset "$DATASET" --model vib --beta "$BETA" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --latent-dim "$LATENT_DIM" --output-dir "$OUTPUT_DIR"
  done
done
```

- [ ] **Step 3: Run smoke script**

Run:

```bash
bash scripts/run_tiny_smoke.sh
```

Expected: creates `artifacts/dev/index.json` with two experiments.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_tiny_smoke.sh scripts/run_core_experiments.sh
git commit -m "chore: add experiment runner scripts"
```

---

## Task 8: FastAPI Artifact Service

**Files:**
- Create: `backend/app.py`
- Create: `backend/tests/test_api.py`
- Create: `artifacts/sample/index.json`
- Create: `artifacts/sample/mnist_vib_beta_0_001/metrics.json`
- Create: `artifacts/sample/mnist_vib_beta_0_001/curves.json`
- Create: `artifacts/sample/mnist_vib_beta_0_001/latent_pca.json`

- [ ] **Step 1: Create sample artifacts**

Write `artifacts/sample/index.json`:

```json
{
  "datasets": ["mnist"],
  "experiments": [
    {
      "id": "mnist_vib_beta_0_001",
      "dataset": "mnist",
      "model": "vib",
      "beta": 0.001,
      "path": "mnist_vib_beta_0_001"
    }
  ]
}
```

Write `artifacts/sample/mnist_vib_beta_0_001/metrics.json`:

```json
{
  "dataset": "mnist",
  "model": "vib",
  "beta": 0.001,
  "train_accuracy": 0.91,
  "test_accuracy": 0.88,
  "train_test_gap": 0.03,
  "average_kl": 3.7,
  "noise_accuracy": {
    "0.0": 0.88,
    "0.1": 0.84,
    "0.2": 0.77
  }
}
```

Write `artifacts/sample/mnist_vib_beta_0_001/curves.json`:

```json
{
  "noise": [
    {"sigma": 0.0, "accuracy": 0.88},
    {"sigma": 0.1, "accuracy": 0.84},
    {"sigma": 0.2, "accuracy": 0.77}
  ]
}
```

Write `artifacts/sample/mnist_vib_beta_0_001/latent_pca.json`:

```json
{
  "points": [
    {"x": -1.0, "y": 0.5, "label": 0},
    {"x": 0.8, "y": -0.2, "label": 1},
    {"x": 1.2, "y": 0.9, "label": 1}
  ]
}
```

- [ ] **Step 2: Write API tests**

Write `backend/tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from backend.app import create_app


def test_index_endpoint_reads_sample_artifacts():
    client = TestClient(create_app("artifacts/sample"))
    response = client.get("/api/index")
    assert response.status_code == 200
    payload = response.json()
    assert payload["datasets"] == ["mnist"]
    assert payload["experiments"][0]["id"] == "mnist_vib_beta_0_001"


def test_experiment_endpoint_reads_metrics_and_curves():
    client = TestClient(create_app("artifacts/sample"))
    response = client.get("/api/experiments/mnist_vib_beta_0_001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["average_kl"] == 3.7
    assert payload["curves"]["noise"][1]["sigma"] == 0.1
    assert len(payload["latent_pca"]["points"]) == 3
```

- [ ] **Step 3: Run API tests and verify failure**

Run:

```bash
python -m pytest backend/tests/test_api.py -v
```

Expected: FAIL because `backend.app` does not exist.

- [ ] **Step 4: Implement FastAPI app**

Write `backend/app.py`:

```python
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


def _read_json(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing artifact: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_app(artifact_dir: str | os.PathLike = "artifacts/full") -> FastAPI:
    root = Path(artifact_dir)
    app = FastAPI(title="VIB Robust Generalization API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True, "artifact_dir": str(root)}

    @app.get("/api/index")
    def index():
        return _read_json(root / "index.json")

    @app.get("/api/experiments/{experiment_id}")
    def experiment(experiment_id: str):
        index_payload = _read_json(root / "index.json")
        matches = [item for item in index_payload["experiments"] if item["id"] == experiment_id]
        if not matches:
            raise HTTPException(status_code=404, detail="Unknown experiment")
        experiment_dir = root / matches[0]["path"]
        return {
            "metadata": matches[0],
            "metrics": _read_json(experiment_dir / "metrics.json"),
            "curves": _read_json(experiment_dir / "curves.json"),
            "latent_pca": _read_json(experiment_dir / "latent_pca.json"),
        }

    return app


app = create_app(os.environ.get("VIB_ARTIFACT_DIR", "artifacts/full"))
```

- [ ] **Step 5: Run API tests**

Run:

```bash
python -m pytest backend/tests/test_api.py -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Start backend manually**

Run:

```bash
VIB_ARTIFACT_DIR=artifacts/sample uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Expected: server starts. Visit `/api/health` from browser or use a non-PowerShell HTTP client.

- [ ] **Step 7: Commit**

```bash
git add backend/app.py backend/tests/test_api.py artifacts/sample
git commit -m "feat: serve experiment artifacts with FastAPI"
```

---

## Task 9: Frontend Scaffold and API Client

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/index.html`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`

- [ ] **Step 1: Create frontend package**

Write `frontend/package.json`:

```json
{
  "name": "vib-dashboard",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 0.0.0.0",
    "build": "tsc && vite build",
    "test": "vitest run"
  },
  "dependencies": {
    "@vitejs/plugin-react": "latest",
    "vite": "latest",
    "typescript": "latest",
    "react": "latest",
    "react-dom": "latest",
    "recharts": "latest"
  },
  "devDependencies": {
    "@types/react": "latest",
    "@types/react-dom": "latest",
    "vitest": "latest"
  }
}
```

- [ ] **Step 2: Create Vite config files**

Write `frontend/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>VIB Robust Generalization</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Write `frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ES2020"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
```

Write `frontend/vite.config.ts`:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000'
    }
  }
})
```

- [ ] **Step 3: Create types and API client**

Write `frontend/src/types.ts`:

```ts
export type ExperimentMetadata = {
  id: string
  dataset: string
  model: 'cnn' | 'vib'
  beta: number
  path: string
}

export type ExperimentIndex = {
  datasets: string[]
  experiments: ExperimentMetadata[]
}

export type Metrics = {
  dataset: string
  model: string
  beta: number
  train_accuracy: number
  test_accuracy: number
  train_test_gap: number
  average_kl: number
  noise_accuracy: Record<string, number>
}

export type Curves = {
  noise: Array<{ sigma: number; accuracy: number }>
}

export type LatentPoint = {
  x: number
  y: number
  label: number
}

export type ExperimentPayload = {
  metadata: ExperimentMetadata
  metrics: Metrics
  curves: Curves
  latent_pca: { points: LatentPoint[] }
}
```

Write `frontend/src/api.ts`:

```ts
import type { ExperimentIndex, ExperimentPayload } from './types'

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function fetchIndex(): Promise<ExperimentIndex> {
  return getJson<ExperimentIndex>('/api/index')
}

export function fetchExperiment(id: string): Promise<ExperimentPayload> {
  return getJson<ExperimentPayload>(`/api/experiments/${id}`)
}
```

- [ ] **Step 4: Create minimal app**

Write `frontend/src/main.tsx`:

```tsx
import React from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import './style.css'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

Write `frontend/src/App.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { fetchExperiment, fetchIndex } from './api'
import type { ExperimentIndex, ExperimentPayload } from './types'

export function App() {
  const [index, setIndex] = useState<ExperimentIndex | null>(null)
  const [experiment, setExperiment] = useState<ExperimentPayload | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchIndex()
      .then(async (payload) => {
        setIndex(payload)
        if (payload.experiments.length > 0) {
          setExperiment(await fetchExperiment(payload.experiments[0].id))
        }
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  if (error) return <main><h1>VIB Dashboard</h1><p>{error}</p></main>
  if (!index || !experiment) return <main><h1>VIB Dashboard</h1><p>Loading artifacts...</p></main>

  return (
    <main>
      <section className="hero">
        <p className="eyebrow">Information Theory × Deep Learning</p>
        <h1>Variational Information Bottleneck</h1>
        <p>
          Explore how beta controls representation compression, prediction accuracy,
          and robustness under noisy image inputs.
        </p>
      </section>
      <section className="panel">
        <h2>Loaded Experiment</h2>
        <p>{experiment.metadata.id}</p>
        <p>Test accuracy: {(experiment.metrics.test_accuracy * 100).toFixed(2)}%</p>
        <p>Average KL proxy: {experiment.metrics.average_kl.toFixed(3)}</p>
      </section>
    </main>
  )
}
```

Write `frontend/src/style.css`:

```css
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0f172a;
  color: #e2e8f0;
}

main {
  width: min(1120px, calc(100vw - 40px));
  margin: 0 auto;
  padding: 48px 0;
}

.hero {
  padding: 40px;
  border-radius: 28px;
  background: linear-gradient(135deg, #1e293b, #312e81);
}

.eyebrow {
  color: #93c5fd;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 13px;
}

.panel {
  margin-top: 24px;
  padding: 24px;
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 20px;
  background: rgba(15, 23, 42, 0.72);
}
```

- [ ] **Step 5: Install and build frontend**

Run:

```bash
cd frontend && npm install && npm run build
```

Expected: TypeScript and Vite build pass.

- [ ] **Step 6: Commit**

```bash
git add frontend
git commit -m "feat: scaffold VIB dashboard frontend"
```

---

## Task 10: Frontend Visualization Components

**Files:**
- Create: `frontend/src/lib/metrics.ts`
- Create: `frontend/src/components/MetricCards.tsx`
- Create: `frontend/src/components/InformationPlane.tsx`
- Create: `frontend/src/components/RobustnessChart.tsx`
- Create: `frontend/src/components/LatentPlot.tsx`
- Create: `frontend/src/components/ExperimentSelector.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/style.css`

- [ ] **Step 1: Create frontend metric helpers**

Write `frontend/src/lib/metrics.ts`:

```ts
import type { ExperimentMetadata, ExperimentPayload } from '../types'

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`
}

export function informationPlanePoints(experiments: ExperimentPayload[]) {
  return experiments.map((experiment) => ({
    id: experiment.metadata.id,
    beta: experiment.metadata.beta,
    kl: experiment.metrics.average_kl,
    accuracy: experiment.metrics.test_accuracy,
    model: experiment.metadata.model,
  }))
}

export function sortExperiments(experiments: ExperimentMetadata[]): ExperimentMetadata[] {
  return [...experiments].sort((a, b) => {
    if (a.dataset !== b.dataset) return a.dataset.localeCompare(b.dataset)
    if (a.model !== b.model) return a.model.localeCompare(b.model)
    return a.beta - b.beta
  })
}
```

- [ ] **Step 2: Create selector component**

Write `frontend/src/components/ExperimentSelector.tsx`:

```tsx
import type { ExperimentMetadata } from '../types'

export function ExperimentSelector({
  experiments,
  selectedId,
  onSelect,
}: {
  experiments: ExperimentMetadata[]
  selectedId: string
  onSelect: (id: string) => void
}) {
  return (
    <label className="selector">
      Experiment
      <select value={selectedId} onChange={(event) => onSelect(event.target.value)}>
        {experiments.map((experiment) => (
          <option key={experiment.id} value={experiment.id}>
            {experiment.dataset} / {experiment.model} / beta={experiment.beta}
          </option>
        ))}
      </select>
    </label>
  )
}
```

- [ ] **Step 3: Create metric cards**

Write `frontend/src/components/MetricCards.tsx`:

```tsx
import { formatPercent } from '../lib/metrics'
import type { Metrics } from '../types'

export function MetricCards({ metrics }: { metrics: Metrics }) {
  return (
    <div className="metric-grid">
      <article className="metric-card">
        <span>Clean Accuracy</span>
        <strong>{formatPercent(metrics.test_accuracy)}</strong>
      </article>
      <article className="metric-card">
        <span>Train-Test Gap</span>
        <strong>{formatPercent(metrics.train_test_gap)}</strong>
      </article>
      <article className="metric-card">
        <span>KL Proxy</span>
        <strong>{metrics.average_kl.toFixed(3)}</strong>
      </article>
      <article className="metric-card">
        <span>Beta</span>
        <strong>{metrics.beta}</strong>
      </article>
    </div>
  )
}
```

- [ ] **Step 4: Create charts**

Write `frontend/src/components/RobustnessChart.tsx`:

```tsx
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Curves } from '../types'

export function RobustnessChart({ curves }: { curves: Curves }) {
  return (
    <section className="panel">
      <h2>Robustness under Gaussian Noise</h2>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={curves.noise}>
          <XAxis dataKey="sigma" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Line type="monotone" dataKey="accuracy" stroke="#60a5fa" strokeWidth={3} dot />
        </LineChart>
      </ResponsiveContainer>
    </section>
  )
}
```

Write `frontend/src/components/LatentPlot.tsx`:

```tsx
import { Scatter, ScatterChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import type { LatentPoint } from '../types'

export function LatentPlot({ points }: { points: LatentPoint[] }) {
  return (
    <section className="panel">
      <h2>Latent PCA Projection</h2>
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart>
          <XAxis dataKey="x" type="number" name="PC1" />
          <YAxis dataKey="y" type="number" name="PC2" />
          <ZAxis dataKey="label" name="label" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter data={points} fill="#a78bfa" />
        </ScatterChart>
      </ResponsiveContainer>
    </section>
  )
}
```

Write `frontend/src/components/InformationPlane.tsx`:

```tsx
import { Scatter, ScatterChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'

export type PlanePoint = {
  id: string
  beta: number
  kl: number
  accuracy: number
  model: string
}

export function InformationPlane({ points }: { points: PlanePoint[] }) {
  return (
    <section className="panel">
      <h2>Information Plane</h2>
      <p className="muted">X-axis is KL proxy for I(X;Z); Y-axis is test accuracy as a prediction proxy.</p>
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart>
          <XAxis dataKey="kl" type="number" name="KL proxy" />
          <YAxis dataKey="accuracy" type="number" name="accuracy" domain={[0, 1]} />
          <ZAxis dataKey="beta" name="beta" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter data={points} fill="#38bdf8" />
        </ScatterChart>
      </ResponsiveContainer>
    </section>
  )
}
```

- [ ] **Step 5: Wire components into App**

Replace `frontend/src/App.tsx` with:

```tsx
import { useEffect, useMemo, useState } from 'react'
import { fetchExperiment, fetchIndex } from './api'
import { ExperimentSelector } from './components/ExperimentSelector'
import { InformationPlane } from './components/InformationPlane'
import { LatentPlot } from './components/LatentPlot'
import { MetricCards } from './components/MetricCards'
import { RobustnessChart } from './components/RobustnessChart'
import { informationPlanePoints, sortExperiments } from './lib/metrics'
import type { ExperimentIndex, ExperimentPayload } from './types'

export function App() {
  const [index, setIndex] = useState<ExperimentIndex | null>(null)
  const [selectedId, setSelectedId] = useState<string>('')
  const [experiments, setExperiments] = useState<Record<string, ExperimentPayload>>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchIndex()
      .then(async (payload) => {
        const sorted = sortExperiments(payload.experiments)
        const firstId = sorted[0]?.id ?? ''
        setIndex({ ...payload, experiments: sorted })
        setSelectedId(firstId)
        if (firstId) {
          const first = await fetchExperiment(firstId)
          setExperiments({ [firstId]: first })
        }
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  useEffect(() => {
    if (!selectedId || experiments[selectedId]) return
    fetchExperiment(selectedId)
      .then((payload) => setExperiments((current) => ({ ...current, [selectedId]: payload })))
      .catch((err: Error) => setError(err.message))
  }, [selectedId, experiments])

  const selected = selectedId ? experiments[selectedId] : null
  const planePoints = useMemo(() => informationPlanePoints(Object.values(experiments)), [experiments])

  if (error) return <main><h1>VIB Dashboard</h1><p>{error}</p></main>
  if (!index || !selected) return <main><h1>VIB Dashboard</h1><p>Loading artifacts...</p></main>

  return (
    <main>
      <section className="hero">
        <p className="eyebrow">Information Theory × Deep Learning</p>
        <h1>Variational Information Bottleneck</h1>
        <p>
          Adjust beta to see how representation compression changes prediction,
          KL proxy, and robustness under noisy image inputs.
        </p>
      </section>
      <section className="panel">
        <ExperimentSelector experiments={index.experiments} selectedId={selectedId} onSelect={setSelectedId} />
      </section>
      <MetricCards metrics={selected.metrics} />
      <InformationPlane points={planePoints} />
      <RobustnessChart curves={selected.curves} />
      <LatentPlot points={selected.latent_pca.points} />
    </main>
  )
}
```

- [ ] **Step 6: Extend CSS**

Append to `frontend/src/style.css`:

```css
.selector {
  display: grid;
  gap: 8px;
  color: #bfdbfe;
}

.selector select {
  max-width: 520px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: #020617;
  color: #e2e8f0;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-top: 24px;
}

.metric-card {
  padding: 20px;
  border-radius: 18px;
  background: rgba(30, 41, 59, 0.86);
  border: 1px solid rgba(148, 163, 184, 0.24);
}

.metric-card span {
  display: block;
  color: #94a3b8;
  font-size: 13px;
}

.metric-card strong {
  display: block;
  margin-top: 10px;
  font-size: 28px;
}

.muted {
  color: #94a3b8;
}

@media (max-width: 800px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
```

- [ ] **Step 7: Build frontend**

Run:

```bash
cd frontend && npm run build
```

Expected: build passes.

- [ ] **Step 8: Commit**

```bash
git add frontend/src
git commit -m "feat: visualize VIB experiment artifacts"
```

---

## Task 11: Report Outline and Figure Export Targets

**Files:**
- Create: `report/outline.md`
- Create: `report/figures/.gitkeep`

- [ ] **Step 1: Create report outline**

Write `report/outline.md`:

```markdown
# 基于变分信息瓶颈的神经网络表示压缩与鲁棒泛化分析

## 摘要

说明项目复现 VIB，研究 beta 对表示压缩、分类准确率和噪声鲁棒性的影响，并实现交互式 Web demo。

## 1 引言

- 信息论关注不确定性、编码、压缩与传输。
- 深度学习表示可以理解为从输入 X 到表示 Z 再到标签 Y 的信息处理过程。
- 本文问题：适度压缩是否能提升鲁棒泛化？

## 2 信息论背景

- 熵 H(X)。
- 互信息 I(X;Z)、I(Z;Y)。
- 信息瓶颈目标 max I(Z;Y) - beta I(X;Z)。
- KL proxy 与变分上界。

## 3 方法

- CNN baseline。
- VIB-CNN。
- Reparameterization trick。
- Loss = CE + beta * KL。

## 4 实验设置

- 数据集：MNIST、Fashion-MNIST，进阶 CIFAR-10。
- beta 网格。
- Gaussian noise sigma 网格。
- 指标：clean accuracy、noisy accuracy、train-test gap、average KL、latent PCA。

## 5 结果与分析

- CNN vs VIB。
- beta vs KL proxy。
- beta vs clean accuracy。
- beta vs noisy accuracy。
- information plane。
- latent space 可视化。

## 6 Web 交互系统

- 系统架构。
- Artifact 驱动设计。
- Dashboard、Information Plane、Latent Space、Robustness Lab。

## 7 局限性

- KL 是 proxy，不是精确互信息。
- MNIST/Fashion-MNIST 简化。
- CIFAR-10 可能需要更强模型。
- Adaptive beta 是扩展方向。

## 8 结论

总结信息瓶颈如何连接表示压缩、任务预测与鲁棒泛化。
```

- [ ] **Step 2: Create figure directory marker**

Write `report/figures/.gitkeep` as an empty file.

- [ ] **Step 3: Commit**

```bash
git add report/outline.md report/figures/.gitkeep
git commit -m "docs: add report outline"
```

---

## Task 12: Verification Checklist

**Files:**
- Modify only if failures reveal defects.

- [ ] **Step 1: Run Python tests**

Run:

```bash
python -m pytest -v
```

Expected: all Python and API tests pass.

- [ ] **Step 2: Run tiny smoke experiment**

Run:

```bash
bash scripts/run_tiny_smoke.sh
```

Expected: `artifacts/dev/index.json` contains CNN and VIB experiments.

- [ ] **Step 3: Run backend against dev artifacts**

Run:

```bash
VIB_ARTIFACT_DIR=artifacts/dev uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Expected: backend starts and `/api/index` returns experiment metadata.

- [ ] **Step 4: Run frontend build**

Run:

```bash
cd frontend && npm run build
```

Expected: TypeScript and Vite build pass.

- [ ] **Step 5: Manual browser verification**

Run backend and frontend dev servers, then verify:

- The dashboard loads experiments.
- Experiment selector changes metrics.
- Metric cards show clean accuracy, train-test gap, KL proxy, beta.
- Information plane displays points.
- Robustness chart displays sigma vs accuracy.
- Latent PCA chart displays points.

- [ ] **Step 6: Commit verification fixes**

If any verification step required fixes:

```bash
git add <changed-files>
git commit -m "fix: stabilize VIB dashboard verification"
```

---

## Extension Task A: Adaptive Beta

Implement only after Tasks 1-12 pass.

**Files:**
- Create: `src/vib_project/adaptive_beta.py`
- Modify: `src/vib_project/config.py`
- Modify: `src/vib_project/train.py`
- Modify: `src/vib_project/cli.py`
- Test: `tests/test_adaptive_beta.py`

- [ ] **Step 1: Write adaptive beta test**

Write `tests/test_adaptive_beta.py`:

```python
from vib_project.adaptive_beta import update_beta


def test_beta_increases_when_kl_above_target():
    new_beta = update_beta(beta=0.01, observed_kl=12.0, target_kl=8.0, step=0.2)
    assert new_beta > 0.01


def test_beta_decreases_when_kl_below_target():
    new_beta = update_beta(beta=0.01, observed_kl=4.0, target_kl=8.0, step=0.2)
    assert new_beta < 0.01


def test_beta_stays_positive():
    new_beta = update_beta(beta=1e-8, observed_kl=0.0, target_kl=8.0, step=0.9)
    assert new_beta > 0.0
```

- [ ] **Step 2: Implement adaptive beta utility**

Write `src/vib_project/adaptive_beta.py`:

```python
def update_beta(beta: float, observed_kl: float, target_kl: float, step: float = 0.1) -> float:
    if observed_kl > target_kl:
        return beta * (1.0 + step)
    return max(beta * (1.0 - step), 1e-8)
```

- [ ] **Step 3: Wire adaptive beta into training**

Add config fields to `ExperimentConfig`:

```python
adaptive_beta: bool = False
target_kl: float = 8.0
```

In `train.py`, keep `current_beta = config.beta` before epochs. Use `current_beta` in `vib_loss`. At the end of each epoch, evaluate average KL and update beta if `config.adaptive_beta` is true.

- [ ] **Step 4: Add CLI flags**

In `cli.py`, add:

```python
run.add_argument("--adaptive-beta", action="store_true")
run.add_argument("--target-kl", type=float, default=8.0)
```

Pass both values into `ExperimentConfig`.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_adaptive_beta.py tests/test_tiny_training.py -v
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/vib_project/adaptive_beta.py src/vib_project/config.py src/vib_project/train.py src/vib_project/cli.py tests/test_adaptive_beta.py
git commit -m "feat: add adaptive beta training option"
```

---

## Extension Task B: CIFAR-10

Implement only after MNIST and Fashion-MNIST results are stable.

**Files:**
- Modify: `scripts/run_core_experiments.sh`
- Modify: `src/vib_project/models.py` only if CIFAR-10 accuracy is too low for meaningful comparison.

- [ ] **Step 1: Add CIFAR-10 to script**

Modify `scripts/run_core_experiments.sh` loop:

```bash
for DATASET in mnist fashion_mnist cifar10; do
```

- [ ] **Step 2: Run CIFAR-10 short experiment**

Run:

```bash
python -m vib_project.cli run --dataset cifar10 --model vib --beta 0.001 --epochs 3 --limit-train 5000 --limit-test 1000 --output-dir artifacts/dev
```

Expected: command completes and writes artifacts. Use results only if curves are interpretable.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_core_experiments.sh
git commit -m "chore: include CIFAR-10 experiment option"
```

---

## Self-Review

Spec coverage:

- VIB baseline: Tasks 3-5.
- CNN baseline: Tasks 3-5.
- MNIST/Fashion-MNIST experiments: Task 7.
- KL proxy and robustness metrics: Tasks 4-6.
- Artifacts: Tasks 5-6.
- FastAPI backend: Task 8.
- React dashboard: Tasks 9-10.
- Report outline: Task 11.
- Verification: Task 12.
- Adaptive beta: Extension Task A.
- CIFAR-10: Extension Task B.

Placeholder scan:

- No unresolved TBD/TODO markers are used.
- Extension tasks are explicitly scoped and executable.

Type consistency:

- Python artifact keys match frontend TypeScript types: `train_accuracy`, `test_accuracy`, `train_test_gap`, `average_kl`, `noise_accuracy`, `curves.noise`, `latent_pca.points`.
- Backend response shape matches `ExperimentPayload`.
- Experiment IDs are generated consistently by `ExperimentConfig.experiment_id` and artifact writer paths.
