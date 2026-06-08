# VIB PhaseLab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or inline execution task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not use PowerShell in this repository. Do not commit unless the user explicitly asks.

**Goal:** Upgrade the existing VIB reproduction/dashboard project into VIB PhaseLab: a robustness phase-transition and representation-diagnosis platform that studies when information bottleneck compression helps, fails, or collapses task-relevant representations.

**Architecture:** Keep the existing `artifacts/full` pipeline intact and add a new `artifacts/phase` artifact root. Offline Python code computes corruptions, robustness AUC, phase labels, latent geometry, and latent drift; FastAPI exposes read-only phase artifacts; React/Vite renders an experimental sandbox with Overview, Phase Map, Compression Lens, and Diagnosis Lab. Adaptive β remains a bonus path after the phase-scan MVP works.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, numpy, scikit-learn, matplotlib, pytest, FastAPI, React, TypeScript, Vite, Recharts.

---

## File Structure

```text
vib-robustness-lab/
├── src/vib_project/
│   ├── corruption.py                  # new: multi-corruption functions and severity mapping
│   ├── phase_metrics.py               # new: robustness AUC, CBI, KL collapse, phase labels
│   ├── latent_diagnostics.py          # new: latent geometry and drift metrics
│   ├── phase_artifacts.py             # new: write/read artifacts/phase contract
│   ├── phase_runner.py                # new: phase experiment orchestration
│   ├── adaptive_beta.py               # new optional: warmup and target-KL schedules
│   ├── data.py                        # modify: CIFAR-10 support is already present, verify transforms and metadata
│   ├── models.py                      # modify: CIFAR-specific encoder or deeper encoder selection
│   ├── evaluate.py                    # modify: confidence/entropy, corruption-aware evaluation
│   ├── train.py                       # modify: return CE/KL/loss components and support phase runner needs
│   └── cli.py                         # modify: add `phase-run` and optional adaptive commands
├── scripts/
│   ├── run_phase_smoke.sh             # new: tiny phase smoke
│   ├── run_phase_experiments.sh        # new: full phase sweep on GPU/NAS
│   ├── validate_phase_artifacts.py     # new: schema/completeness validation
│   └── summarize_phase_results.py      # new: summary tables, phase map data, report figures
├── backend/
│   ├── app.py                         # modify: add /api/phase endpoints
│   └── tests/test_phase_api.py         # new: phase endpoint tests
├── frontend/src/
│   ├── types.ts                       # modify: phase artifact types
│   ├── api.ts                         # modify: phase API client
│   ├── App.tsx                        # modify: add PhaseLab navigation
│   ├── lib/phaseMetrics.ts            # new: lightweight frontend formatting/helpers only
│   └── components/
│       ├── PhaseOverview.tsx          # new
│       ├── PhaseMap.tsx               # new
│       ├── CompressionLens.tsx        # new
│       ├── DiagnosisLab.tsx           # new
│       └── AdaptiveReplay.tsx         # optional new
├── tests/
│   ├── test_corruption.py             # new
│   ├── test_phase_metrics.py          # new
│   ├── test_latent_diagnostics.py     # new
│   ├── test_phase_artifacts.py        # new
│   ├── test_phase_runner_smoke.py     # new
│   └── test_adaptive_beta.py          # optional new
├── report/
│   ├── phase_results/                 # generated phase CSV/Markdown summaries
│   ├── phase_figures/                 # generated phase maps and diagnostic figures
│   └── phaselab_report.md             # new report draft after real phase artifacts exist
└── artifacts/
    └── phase/                         # generated, not hand-edited
```

Boundaries:

- `artifacts/full` remains the completed reproduction evidence and must not be overwritten.
- `artifacts/phase` is generated output and must not be hand-edited.
- Backend and frontend remain artifact-driven; no online training, uploads, accounts, or adversarial attacks.
- Expensive experiments run offline on lab GPU/NAS; local work starts with tiny smoke tests.
- Adaptive β is optional and must not block the phase-scan MVP.

---

## Task 1: Multi-Corruption Library

**Files:**
- Create: `src/vib_project/corruption.py`
- Create: `tests/test_corruption.py`
- Modify later: `src/vib_project/evaluate.py`

Purpose: replace the single Gaussian-only noise path with a deterministic corruption abstraction used by phase evaluation.

- [ ] **Step 1: Write failing corruption tests**

Create `tests/test_corruption.py` with tests for:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_corruption.py -v
```

Expected: FAIL because `vib_project.corruption` does not exist.

- [ ] **Step 3: Implement `corruption.py`**

Implement:

```python
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
```

Then add `apply_corruption(images, name, severity, seed=None)`:

- severity 0 returns `images.clone()`.
- gaussian uses `torch.randn` with optional generator and clamps to `[0, 1]`.
- salt-pepper uses random masks with probability `severity * 0.4`, half to 0 and half to 1.
- blur uses grouped convolution with a small deterministic average kernel. For RGB, use `groups=channels`.
- contrast maps around 0.5: `(images - 0.5) * factor + 0.5`, clamped.

- [ ] **Step 4: Run tests and existing noise tests**

Run:

```bash
python -m pytest tests/test_corruption.py tests/test_noise.py -v
```

Expected: PASS.

---

## Task 2: Corruption-Aware Evaluation Metrics

**Files:**
- Modify: `src/vib_project/evaluate.py`
- Create: `tests/test_evaluate_corruption.py`

Purpose: evaluate accuracy, confidence, entropy, CE, KL, and corruption-specific rows for phase artifacts.

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/test_evaluate_corruption.py`:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vib_project.evaluate import evaluate_corruption_grid


class ConstantModel(nn.Module):
    def forward(self, images):
        logits = torch.zeros(images.shape[0], 3, device=images.device)
        logits[:, 1] = 3.0
        latent = torch.ones(images.shape[0], 4, device=images.device)
        return logits, latent


def test_evaluate_corruption_grid_returns_rows():
    images = torch.rand(5, 1, 8, 8)
    labels = torch.ones(5, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    rows = evaluate_corruption_grid(
        ConstantModel(),
        loader,
        torch.device("cpu"),
        model_name="cnn",
        corruptions=["gaussian", "contrast"],
        severities=[0.0, 0.2],
    )

    assert len(rows) == 4
    assert rows[0]["corruption"] == "gaussian"
    assert rows[0]["severity"] == 0.0
    assert rows[0]["accuracy"] == 1.0
    assert 0.0 <= rows[0]["mean_confidence"] <= 1.0
    assert rows[0]["mean_entropy"] >= 0.0
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_evaluate_corruption.py -v
```

Expected: FAIL because `evaluate_corruption_grid` does not exist.

- [ ] **Step 3: Implement evaluation helpers**

In `src/vib_project/evaluate.py`, add:

- `_entropy_from_logits(logits)`
- `_confidence_from_logits(logits)`
- `evaluate_corruption_grid(model, loader, device, model_name, corruptions, severities)`

Behavior:

- For each corruption/severity, apply `apply_corruption` to each batch.
- For CNN, model returns `(logits, latent)`.
- For VIB, model returns `VIBOutput` with `.logits`, `.mu`, `.logvar`.
- Return rows with `corruption`, `severity`, `accuracy`, `mean_confidence`, `mean_entropy`.
- Include `average_kl` only for VIB rows if convenient; otherwise phase runner can use clean evaluation KL.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
python -m pytest tests/test_evaluate_corruption.py tests/test_tiny_training.py -v
```

Expected: PASS.

---

## Task 3: Phase Metrics and Phase Labels

**Files:**
- Create: `src/vib_project/phase_metrics.py`
- Create: `tests/test_phase_metrics.py`

Purpose: turn robustness rows and metrics into interpretable phase indicators.

- [ ] **Step 1: Write failing phase metric tests**

Create `tests/test_phase_metrics.py`:

```python
from vib_project.phase_metrics import (
    compute_robustness_auc,
    compute_compression_benefit_index,
    compute_kl_collapse_score,
    assign_phase_label,
)


def test_compute_robustness_auc_groups_by_corruption():
    rows = [
        {"corruption": "gaussian", "severity": 0.0, "accuracy": 0.9},
        {"corruption": "gaussian", "severity": 0.2, "accuracy": 0.7},
        {"corruption": "contrast", "severity": 0.0, "accuracy": 0.9},
        {"corruption": "contrast", "severity": 0.2, "accuracy": 0.8},
    ]
    assert compute_robustness_auc(rows) == {"gaussian": 0.8, "contrast": 0.85}


def test_compute_compression_benefit_index():
    vib = {"gaussian": 0.9}
    baseline = {"gaussian": 0.82}
    assert compute_compression_benefit_index(vib, baseline) == {"gaussian": 0.08}


def test_kl_collapse_score_handles_zero_reference():
    assert compute_kl_collapse_score(1.0, 0.0) is None
    assert compute_kl_collapse_score(2.0, 10.0) == 0.2


def test_assign_phase_label_over_compressed():
    label = assign_phase_label(
        average_kl=0.1,
        beta0_kl=10.0,
        test_accuracy=0.70,
        best_accuracy=0.90,
        cbi_by_corruption={"gaussian": -0.1},
    )
    assert label == "over-compressed"


def test_assign_phase_label_useful_compression():
    label = assign_phase_label(
        average_kl=4.0,
        beta0_kl=100.0,
        test_accuracy=0.89,
        best_accuracy=0.90,
        cbi_by_corruption={"gaussian": 0.02},
    )
    assert label == "useful-compression"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_phase_metrics.py -v
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement `phase_metrics.py`**

Implement constants:

```python
OVER_COMPRESSION_ACCURACY_DROP = 0.10
KL_COLLAPSE_THRESHOLD = 0.05
USEFUL_ACCURACY_DROP = 0.02
MEANINGFUL_KL_DROP = 0.5
```

Implement:

- `compute_robustness_auc(rows)` grouped by corruption, mean of accuracy.
- `compute_normalized_robustness_auc(auc, clean_accuracy)`.
- `compute_compression_benefit_index(vib_normalized_auc, baseline_normalized_auc)`.
- `compute_kl_collapse_score(average_kl, beta0_kl)` returning `None` if reference invalid.
- `assign_phase_label(...)` using spec labels.
- `build_phase_indicators(...)` wrapper returning JSON-serializable dict.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_phase_metrics.py -v
```

Expected: PASS.

---

## Task 4: Latent Geometry Diagnostics

**Files:**
- Create: `src/vib_project/latent_diagnostics.py`
- Create: `tests/test_latent_diagnostics.py`

Purpose: compute geometry metrics that explain phase labels.

- [ ] **Step 1: Write failing tests**

Create `tests/test_latent_diagnostics.py`:

```python
import numpy as np

from vib_project.latent_diagnostics import compute_latent_geometry, compute_latent_drift


def test_compute_latent_geometry_returns_separation_metrics():
    latents = np.array([
        [0.0, 0.0], [0.1, 0.0],
        [5.0, 5.0], [5.1, 5.0],
    ])
    labels = np.array([0, 0, 1, 1])

    geometry = compute_latent_geometry(latents, labels)

    assert geometry["intra_class_variance"] > 0.0
    assert geometry["inter_class_distance"] > geometry["intra_class_variance"]
    assert geometry["fisher_ratio"] > 1.0
    assert len(geometry["per_class"]) == 2


def test_compute_latent_drift_groups_by_label():
    clean = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])
    corrupted = np.array([[1.0, 0.0], [2.0, 1.0], [6.0, 3.0]])
    labels = np.array([0, 0, 1])

    rows = compute_latent_drift(clean, corrupted, labels, corruption="gaussian", severity=0.3)

    assert rows[0]["corruption"] == "gaussian"
    assert rows[0]["severity"] == 0.3
    assert {row["label"] for row in rows} == {0, 1}
    assert all(row["mean_drift"] >= 0.0 for row in rows)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_latent_diagnostics.py -v
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement diagnostics**

Implement:

- `compute_latent_geometry(latents: np.ndarray, labels: np.ndarray, max_silhouette_points: int = 1000)`.
- `compute_latent_drift(clean_latents, corrupted_latents, labels, corruption, severity)`.

Rules:

- Convert numpy scalars to Python floats/ints for JSON compatibility.
- If only one class exists, set inter-class distance and silhouette to `None`.
- Use `sklearn.metrics.silhouette_score` only when at least two labels and more than two points exist.
- Fisher ratio is `inter / max(intra, 1e-12)` when inter is available.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_latent_diagnostics.py -v
```

Expected: PASS.

---

## Task 5: CIFAR-10 Model Support

**Files:**
- Modify: `src/vib_project/models.py`
- Modify: `src/vib_project/train.py`
- Create/modify: `tests/test_models.py`

Purpose: support CIFAR-10 with a moderately stronger encoder while keeping CNN/VIB interfaces stable.

- [ ] **Step 1: Add failing model tests**

Append to `tests/test_models.py`:

```python
from vib_project.models import CNNClassifier, VIBClassifier


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
```

If these already pass with the current dynamic encoder, keep this task as verification and add a smaller test for CIFAR-specific training config instead.

- [ ] **Step 2: Run tests**

Run:

```bash
python -m pytest tests/test_models.py -v
```

Expected: either PASS if current encoder already supports CIFAR, or FAIL due shape assumptions.

- [ ] **Step 3: Implement CIFAR encoder if needed**

If tests fail or the current encoder is too weak for CIFAR, add a CIFAR-compatible branch inside `ConvEncoder` or create `CifarConvEncoder`:

- input 3×32×32.
- 3 convolution blocks.
- max pooling or strided conv.
- final flatten -> latent projection.

Keep public classes unchanged:

- `CNNClassifier(input_channels, image_size, latent_dim, num_classes)`.
- `VIBClassifier(input_channels, image_size, latent_dim, num_classes)`.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_models.py tests/test_tiny_training.py -v
```

Expected: PASS.

---

## Task 6: Phase Artifact Writer and Validator

**Files:**
- Create: `src/vib_project/phase_artifacts.py`
- Create: `tests/test_phase_artifacts.py`
- Create: `scripts/validate_phase_artifacts.py`
- Create: `tests/test_validate_phase_artifacts.py`

Purpose: establish `artifacts/phase` as a stable contract before running experiments.

- [ ] **Step 1: Write failing artifact writer tests**

Create `tests/test_phase_artifacts.py`:

```python
from pathlib import Path

from vib_project.phase_artifacts import write_phase_experiment, write_phase_index


def test_write_phase_experiment_files(tmp_path: Path):
    payload = {
        "metrics": {"dataset": "mnist", "model": "vib", "beta": 0.001, "test_accuracy": 0.9, "average_kl": 2.0},
        "robustness": {"rows": [{"corruption": "gaussian", "severity": 0.0, "accuracy": 0.9}]},
        "latent_geometry": {"intra_class_variance": 1.0, "inter_class_distance": 2.0, "fisher_ratio": 2.0, "per_class": []},
        "phase_indicators": {"phase_label": "useful-compression", "over_compression_flag": False},
        "latent_drift": {"rows": []},
    }

    write_phase_experiment(tmp_path, "mnist_vib_beta_0_001", payload)

    exp = tmp_path / "mnist_vib_beta_0_001"
    assert (exp / "metrics.json").exists()
    assert (exp / "robustness.json").exists()
    assert (exp / "latent_geometry.json").exists()
    assert (exp / "phase_indicators.json").exists()
    assert (exp / "latent_drift.json").exists()


def test_write_phase_index(tmp_path: Path):
    write_phase_index(
        tmp_path,
        datasets=["mnist"],
        experiments=[{"id": "mnist_vib_beta_0_001", "dataset": "mnist", "model": "vib", "beta": 0.001, "path": "mnist_vib_beta_0_001"}],
    )
    assert (tmp_path / "index.json").exists()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_phase_artifacts.py -v
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement `phase_artifacts.py`**

Implement:

- `_write_json(path, payload)`.
- `write_phase_experiment(root, experiment_id, payload)`.
- `write_phase_index(root, datasets, experiments)`.
- `write_phase_summary(root, summary)`.

Required experiment files:

- `metrics.json`
- `robustness.json`
- `latent_geometry.json`
- `phase_indicators.json`
- optional `latent_drift.json`
- optional `latent_pca.json`
- optional `confusion_matrix.json`

- [ ] **Step 4: Write validator tests and script**

Create `tests/test_validate_phase_artifacts.py` for a tiny temp root containing one complete experiment. Implement `scripts/validate_phase_artifacts.py` with:

- reads `index.json`.
- checks required files per experiment.
- returns JSON report with `experiment_count`, `missing`, `datasets`.
- exits 1 if missing.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_phase_artifacts.py tests/test_validate_phase_artifacts.py -v
```

Expected: PASS.

---

## Task 7: Phase Runner MVP on Tiny Subsets

**Files:**
- Create: `src/vib_project/phase_runner.py`
- Modify: `src/vib_project/cli.py`
- Create: `tests/test_phase_runner_smoke.py`
- Create: `scripts/run_phase_smoke.sh`

Purpose: produce a tiny `artifacts/phase` root using the new contract before full GPU sweeps.

- [ ] **Step 1: Write failing smoke test**

Create `tests/test_phase_runner_smoke.py`:

```python
from pathlib import Path

from vib_project.config import ExperimentConfig
from vib_project.phase_runner import run_phase_experiment


def test_run_phase_experiment_writes_required_payload(tmp_path: Path):
    config = ExperimentConfig(
        dataset="mnist",
        model="vib",
        beta=0.001,
        epochs=1,
        batch_size=32,
        latent_dim=8,
        learning_rate=1e-3,
        seed=5,
        output_dir=tmp_path,
        limit_train=64,
        limit_test=32,
    )

    payload = run_phase_experiment(
        config,
        corruptions=["gaussian", "contrast"],
        severities=[0.0, 0.2],
        max_diagnostic_points=32,
    )

    assert payload["metrics"]["dataset"] == "mnist"
    assert len(payload["robustness"]["rows"]) == 4
    assert "fisher_ratio" in payload["latent_geometry"]
    assert "phase_label" in payload["phase_indicators"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_phase_runner_smoke.py -v
```

Expected: FAIL because `phase_runner` does not exist.

- [ ] **Step 3: Implement `run_phase_experiment`**

Implementation outline:

1. Reuse existing training loop logic or call `run_training(config)` for baseline metrics.
2. Build dataset/model and collect outputs for diagnostics.
3. Run corruption grid evaluation.
4. Compute latent geometry.
5. Compute latent drift for each corruption/severity using a capped subset.
6. Compute phase indicators with available beta0/baseline references when provided; for a single smoke experiment, use safe defaults and phase label `unstable` or `useful-compression` according to thresholds.
7. Return payload; do not write files inside this function unless caller asks.

If reusing `run_training(config)` is too coarse because it writes old artifacts, extract shared training helpers minimally instead of duplicating full code.

- [ ] **Step 4: Add CLI command**

Add to `src/vib_project/cli.py`:

```bash
python -m vib_project.cli phase-run --dataset mnist --model vib --beta 0.001 --epochs 1 --limit-train 64 --limit-test 32 --output-dir artifacts/phase-smoke
```

The command should:

- run one phase experiment.
- write required phase experiment files.
- update `index.json` for the output root.

- [ ] **Step 5: Add smoke script**

Create `scripts/run_phase_smoke.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m vib_project.cli phase-run --dataset mnist --model cnn --beta 0 --epochs 1 --limit-train 64 --limit-test 32 --output-dir artifacts/phase-smoke
python -m vib_project.cli phase-run --dataset mnist --model vib --beta 0.001 --epochs 1 --limit-train 64 --limit-test 32 --output-dir artifacts/phase-smoke
python scripts/validate_phase_artifacts.py artifacts/phase-smoke
```

- [ ] **Step 6: Run tests and smoke**

Run:

```bash
python -m pytest tests/test_phase_runner_smoke.py -v
bash scripts/run_phase_smoke.sh
```

Expected: PASS and validator reports no missing files.

---

## Task 8: Phase Summary Exporter

**Files:**
- Create: `scripts/summarize_phase_results.py`
- Create: `tests/test_summarize_phase_results.py`
- Create generated: `report/phase_results/*.csv`, `report/phase_results/key_findings.md`

Purpose: derive teacher-facing findings and frontend summary data from `artifacts/phase`.

- [ ] **Step 1: Write failing summary tests**

Create `tests/test_summarize_phase_results.py` with temp phase artifacts and assertions for:

- metric rows are read.
- robustness rows are flattened.
- best clean β is identified.
- best robust β by corruption is identified.
- collapse/phase labels are summarized.

Example test core:

```python
from scripts.summarize_phase_results import build_phase_summary


def test_build_phase_summary_identifies_best_clean_and_robust(tmp_path):
    # create index + two experiment dirs with metrics/robustness/phase_indicators
    summary = build_phase_summary(tmp_path)
    assert summary["datasets"]["mnist"]["best_clean_experiment"] == "mnist_vib_beta_0_01"
    assert summary["datasets"]["mnist"]["best_robust_by_corruption"]["gaussian"] == "mnist_vib_beta_0_001"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_summarize_phase_results.py -v
```

Expected: FAIL because script does not exist.

- [ ] **Step 3: Implement summary exporter**

Implement:

- `build_metric_rows(root)`.
- `build_robustness_rows(root)`.
- `build_phase_rows(root)`.
- `build_phase_summary(root)`.
- CSV writers for metrics, robustness, phase labels.
- Markdown key findings writer.
- JSON summary writer to `artifacts/phase/summary.json` or `report/phase_results/phase_summary.json`.

- [ ] **Step 4: Run tests and smoke summary**

Run:

```bash
python -m pytest tests/test_summarize_phase_results.py -v
python scripts/summarize_phase_results.py --artifact-root artifacts/phase-smoke --output-dir report/phase_results
```

Expected: PASS and summary files are written.

---

## Task 9: Full Phase Experiment Script

**Files:**
- Create: `scripts/run_phase_experiments.sh`
- Create: `scripts/run_phase_experiments_cifar_sparse.sh`
- Modify: `CLAUDE.md` and `README.md` only after script is verified

Purpose: provide reproducible GPU/NAS experiment entrypoints.

- [ ] **Step 1: Create dense phase runner script**

Create `scripts/run_phase_experiments.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/phase}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LATENT_DIM="${LATENT_DIM:-32}"
DATASETS="${DATASETS:-mnist fashion_mnist cifar10}"
BETAS="${BETAS:-0 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1}"

for DATASET in $DATASETS; do
  python -m vib_project.cli phase-run --dataset "$DATASET" --model cnn --beta 0 --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --latent-dim "$LATENT_DIM" --output-dir "$OUTPUT_DIR"
  for BETA in $BETAS; do
    python -m vib_project.cli phase-run --dataset "$DATASET" --model vib --beta "$BETA" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --latent-dim "$LATENT_DIM" --output-dir "$OUTPUT_DIR"
  done
done

python scripts/validate_phase_artifacts.py "$OUTPUT_DIR"
python scripts/summarize_phase_results.py --artifact-root "$OUTPUT_DIR" --output-dir report/phase_results
```

- [ ] **Step 2: Create CIFAR sparse helper**

Create `scripts/run_phase_experiments_cifar_sparse.sh` with `DATASETS=cifar10` and sparse β grid.

- [ ] **Step 3: Add skip-existing behavior**

Before launching a command, check whether the experiment directory contains required files. If complete, print `SKIP <experiment_id>` and continue.

- [ ] **Step 4: Run local dry-run variant**

Use tiny limits via CLI flags or a separate smoke script; do not run full experiments locally.

Run:

```bash
bash scripts/run_phase_smoke.sh
```

Expected: PASS.

---

## Task 10: Backend Phase API

**Files:**
- Modify: `backend/app.py`
- Create: `backend/tests/test_phase_api.py`
- Create sample phase artifacts under `artifacts/sample_phase/`

Purpose: expose phase artifacts to the frontend without online computation.

- [ ] **Step 1: Create sample phase artifacts**

Create minimal `artifacts/sample_phase/index.json`, `summary.json`, and one experiment directory containing:

- `metrics.json`
- `robustness.json`
- `latent_geometry.json`
- `phase_indicators.json`
- `latent_drift.json`

- [ ] **Step 2: Write failing API tests**

Create `backend/tests/test_phase_api.py`:

```python
from backend.app import create_app
from fastapi.testclient import TestClient


def test_phase_index_endpoint_reads_sample_phase_artifacts():
    app = create_app(artifact_dir="artifacts/sample", phase_artifact_dir="artifacts/sample_phase")
    client = TestClient(app)
    response = client.get("/api/phase/index")
    assert response.status_code == 200
    assert response.json()["datasets"] == ["mnist"]


def test_phase_experiment_endpoint_reads_detail_files():
    app = create_app(artifact_dir="artifacts/sample", phase_artifact_dir="artifacts/sample_phase")
    client = TestClient(app)
    response = client.get("/api/phase/experiments/mnist_vib_beta_0_001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["dataset"] == "mnist"
    assert payload["phase_indicators"]["phase_label"] == "useful-compression"
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
python -m pytest backend/tests/test_phase_api.py -v
```

Expected: FAIL because phase endpoints do not exist.

- [ ] **Step 4: Implement phase endpoints**

Modify `create_app` signature:

```python
def create_app(artifact_dir="artifacts/full", phase_artifact_dir: str | os.PathLike | None = None) -> FastAPI:
```

Default phase root:

```python
phase_root = Path(phase_artifact_dir or os.environ.get("VIB_PHASE_ARTIFACT_DIR", "artifacts/phase"))
```

Add:

- `/api/phase/index`
- `/api/phase/summary`
- `/api/phase/experiments/{experiment_id}`
- `/api/phase/diagnostics`
- `/api/phase/adaptive/{experiment_id}`

- [ ] **Step 5: Run backend tests**

Run:

```bash
python -m pytest backend/tests/test_api.py backend/tests/test_phase_api.py -v
```

Expected: PASS.

---

## Task 11: Frontend Phase Types and API Client

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/api.ts`
- Create: `frontend/src/lib/phaseMetrics.ts`
- Create: `frontend/src/lib/phaseMetrics.test.ts`

Purpose: prepare frontend to consume phase artifacts.

- [ ] **Step 1: Add phase type definitions**

Add interfaces:

- `PhaseIndex`
- `PhaseExperimentMetadata`
- `RobustnessRow`
- `LatentGeometry`
- `PhaseIndicators`
- `LatentDriftRow`
- `PhaseExperimentPayload`
- `PhaseSummary`

- [ ] **Step 2: Add API functions**

In `api.ts`, add:

- `fetchPhaseIndex()`
- `fetchPhaseSummary()`
- `fetchPhaseExperiment(id)`
- `fetchPhaseDiagnostics()`

- [ ] **Step 3: Add helper tests**

Create `phaseMetrics.test.ts` for:

- phase label Chinese mapping.
- beta display formatting.
- metric scale formatting.

- [ ] **Step 4: Implement helpers and run frontend tests**

Run:

```bash
npm --prefix frontend test -- src/lib/phaseMetrics.test.ts
npm --prefix frontend run build
```

Expected: PASS/build succeeds.

---

## Task 12: Frontend Overview and Phase Map

**Files:**
- Create: `frontend/src/components/PhaseOverview.tsx`
- Create: `frontend/src/components/PhaseMap.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/style.css`

Purpose: make the first screen communicate research innovation and phase transitions.

- [ ] **Step 1: Build components against sample API data**

Implement `PhaseOverview` props:

```ts
{ summary: PhaseSummary }
```

Display:

- datasets
- best clean β
- best robust β
- collapse β if present
- generated conclusion sentence

Implement `PhaseMap` props:

```ts
{
  experiments: PhaseExperimentMetadata[]
  selectedMetric: 'clean_accuracy' | 'robustness_auc' | 'kl_proxy' | 'fisher_ratio' | 'phase_label'
  onSelectExperiment: (id: string) => void
}
```

- [ ] **Step 2: Wire App data loading**

App should:

- Try to load phase endpoints.
- If phase artifacts are unavailable, show current dashboard with a clear message: `PhaseLab artifacts not generated yet`.
- If available, show PhaseLab navigation.

- [ ] **Step 3: Build frontend**

Run:

```bash
npm --prefix frontend run build
```

Expected: PASS.

- [ ] **Step 4: Browser verify with sample phase artifacts**

Run backend with sample phase root and verify:

- Overview appears.
- Phase Map appears.
- clicking an experiment selects it.
- no console-breaking errors.

---

## Task 13: Frontend Compression Lens and Diagnosis Lab

**Files:**
- Create: `frontend/src/components/CompressionLens.tsx`
- Create: `frontend/src/components/DiagnosisLab.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/style.css`

Purpose: make the app an exploration tool, not just a chart viewer.

- [ ] **Step 1: Implement Compression Lens**

Display linked charts for selected dataset:

- β vs clean accuracy
- β vs KL proxy
- β vs robustness AUC
- β vs Fisher ratio

Use Recharts already in the project.

- [ ] **Step 2: Implement Diagnosis Lab**

Display selected experiment:

- phase label
- latent geometry cards
- drift by corruption/class table or chart
- existing latent PCA if present

- [ ] **Step 3: Build frontend**

Run:

```bash
npm --prefix frontend run build
```

Expected: PASS.

- [ ] **Step 4: Browser verify**

Verify:

- selecting dataset changes Compression Lens.
- selecting experiment updates Diagnosis Lab.
- phase labels are visible in Chinese.

---

## Task 14: Adaptive β Bonus

**Files:**
- Create: `src/vib_project/adaptive_beta.py`
- Create: `tests/test_adaptive_beta.py`
- Modify: `src/vib_project/train.py` only if running adaptive experiments now
- Optional create: `frontend/src/components/AdaptiveReplay.tsx`

Purpose: add the schedule/controller logic, but keep actual adaptive experiments optional.

- [ ] **Step 1: Write failing adaptive tests**

Create `tests/test_adaptive_beta.py`:

```python
from vib_project.adaptive_beta import beta_warmup_value, update_beta_for_target_kl


def test_beta_warmup_value_reaches_target():
    assert beta_warmup_value(epoch=0, warmup_epochs=5, beta_target=0.1) == 0.0
    assert beta_warmup_value(epoch=5, warmup_epochs=5, beta_target=0.1) == 0.1
    assert beta_warmup_value(epoch=10, warmup_epochs=5, beta_target=0.1) == 0.1


def test_update_beta_for_target_kl_clamps():
    assert update_beta_for_target_kl(0.1, kl=30.0, target_low=5.0, target_high=20.0) > 0.1
    assert update_beta_for_target_kl(0.1, kl=1.0, target_low=5.0, target_high=20.0) < 0.1
    assert update_beta_for_target_kl(1.0, kl=30.0, target_low=5.0, target_high=20.0) == 1.0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_adaptive_beta.py -v
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement adaptive helpers**

Implement pure functions only first. Do not modify training until helpers pass.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_adaptive_beta.py -v
```

Expected: PASS.

- [ ] **Step 5: Decide whether to run adaptive experiments**

Only after phase MVP artifacts exist, decide whether to implement training integration and Adaptive Replay. Adaptive remains optional.

---

## Task 15: Phase Report Rewrite

**Files:**
- Create: `report/phaselab_report.md`
- Read: `report/phase_results/*`
- Read: `artifacts/phase/summary.json`

Purpose: rewrite the Chinese report around H1-H4 after real phase experiments exist.

- [ ] **Step 1: Generate phase summaries**

Run:

```bash
python scripts/summarize_phase_results.py --artifact-root artifacts/phase --output-dir report/phase_results
```

Expected: summary outputs exist.

- [ ] **Step 2: Draft report sections**

Create `report/phaselab_report.md` with:

```text
1 引言：为什么 VIB 鲁棒性不是 yes/no 问题
2 理论背景：IB objective、KL proxy、compression-prediction tradeoff
3 研究假设：H1-H4
4 实验系统：phase scan + diagnostic lab
5 鲁棒性相变图谱
6 表示空间诊断
7 Adaptive β 初步探索
8 Web 实验沙盘
9 局限性
10 结论
```

- [ ] **Step 3: Fill with real measured values**

Use only `artifacts/phase`, `report/phase_results`, and generated figures. Do not invent metrics.

- [ ] **Step 4: Verify no placeholders remain**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('report/phaselab_report.md').read_text(encoding='utf-8')
bad = [word for word in ['TO' + 'DO', 'T' + 'BD', '填' + '写'] if word in text]
print({'placeholders': bad})
raise SystemExit(1 if bad else 0)
PY
```

Expected: no placeholders.

---

## Task 16: Final Verification for VIB PhaseLab

**Files:**
- Modify only if verification reveals defects.

- [ ] **Step 1: Run Python tests**

Run:

```bash
python -m pytest -v
```

Expected: all tests pass. If the local Python 3.14 pyarrow/sklearn access-violation diagnostic appears but pytest reports all tests passed, record it as an environment warning, not a project failure.

- [ ] **Step 2: Validate phase artifacts**

Run:

```bash
python scripts/validate_phase_artifacts.py artifacts/phase
python scripts/summarize_phase_results.py --artifact-root artifacts/phase --output-dir report/phase_results
```

Expected: validator reports no missing required files; summary generation succeeds.

- [ ] **Step 3: Build frontend**

Run:

```bash
npm --prefix frontend test -- src/lib/metrics.test.ts
npm --prefix frontend test -- src/lib/phaseMetrics.test.ts
npm --prefix frontend run build
```

Expected: tests pass and build succeeds. Recharts/Vite chunk-size warning is acceptable.

- [ ] **Step 4: Browser verify PhaseLab**

Run backend and frontend:

```bash
VIB_ARTIFACT_DIR=artifacts/full VIB_PHASE_ARTIFACT_DIR=artifacts/phase python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5177
```

Verify:

- Overview shows MNIST, Fashion-MNIST, and CIFAR-10.
- Phase Map can switch metrics.
- Compression Lens updates by dataset/β.
- Diagnosis Lab shows latent geometry and drift.
- If adaptive artifacts exist, Adaptive Replay renders; otherwise it is hidden or marked unavailable.

- [ ] **Step 5: Inspect git status and artifact sizes**

Run:

```bash
git status --short
python - <<'PY'
from pathlib import Path
for path in ['artifacts/phase', 'report/phase_results', 'report/phase_figures']:
    root = Path(path)
    if not root.exists():
        continue
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    print(path, round(size / 1024 / 1024, 2), 'MB')
PY
```

Expected: know artifact sizes before deciding commit/submission policy.

- [ ] **Step 6: Prepare commit recommendation without committing**

Group recommended files:

- Python source and tests.
- Backend phase API and tests.
- Frontend PhaseLab source and tests.
- Scripts.
- Specs/plans/docs.
- Report source and derived phase summaries/figures.
- `artifacts/phase` only if the user explicitly wants generated artifacts committed.

Do not run `git add`, `git commit`, or `git push` unless the user explicitly asks.

---

## Execution Strategy

Recommended order:

1. Tasks 1-4: corruption, corruption evaluation, phase metrics, latent diagnostics.
2. Tasks 5-8: CIFAR support, phase artifact contract, phase runner smoke, phase summary exporter.
3. Task 9: GPU/NAS experiment scripts.
4. Run phase experiments on lab GPU/NAS.
5. Tasks 10-13: backend and frontend PhaseLab UI using sample/real phase artifacts.
6. Task 14: optional adaptive β helper and experiments.
7. Task 15: report rewrite after real phase data exists.
8. Task 16: final verification.

Critical rule: do not begin full GPU sweeps until Tasks 1-8 pass locally with tiny artifacts.

---

## Self-Review

Spec coverage:

- CIFAR-10 support is covered by Task 5 and Task 9.
- Multiple corruptions are covered by Tasks 1-2.
- Dense β scan is covered by Task 9.
- Phase indicators are covered by Task 3 and Task 8.
- Latent geometry and drift are covered by Task 4 and Task 7.
- `artifacts/phase` is covered by Task 6.
- Backend phase endpoints are covered by Task 10.
- Frontend sandbox views are covered by Tasks 11-13.
- Adaptive β bonus is covered by Task 14.
- Report rewrite is covered by Task 15.
- Final verification is covered by Task 16.

Placeholder scan:

- The plan contains no unresolved placeholder markers.
- Optional work is explicitly marked optional and does not block the core PhaseLab acceptance criteria.

Consistency check:

- Artifact file names match the approved spec: `metrics.json`, `robustness.json`, `latent_geometry.json`, `phase_indicators.json`, and `latent_drift.json`.
- Backend endpoint names match the approved spec: `/api/phase/index`, `/api/phase/summary`, `/api/phase/experiments/{experiment_id}`, `/api/phase/diagnostics`, and `/api/phase/adaptive/{experiment_id}`.
- Frontend view names match the approved spec: Overview, Phase Map, Compression Lens, Diagnosis Lab, and optional Adaptive Replay.

Scope check:

- The plan does not include online training, user uploads, adversarial attacks, ResNet benchmark competition, exact mutual-information estimation, LLM/RAG features, cloud deployment, or commits.
