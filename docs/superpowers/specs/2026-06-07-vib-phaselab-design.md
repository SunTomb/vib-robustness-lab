# VIB PhaseLab Design Spec

## 1. Purpose

The current project successfully reproduces the core Variational Information Bottleneck (VIB) idea, trains CNN/VIB-CNN models on MNIST and Fashion-MNIST, exports artifacts, serves them through FastAPI, and visualizes them in a Chinese React dashboard. However, its current framing still reads as a reproduction project plus a result viewer.

VIB PhaseLab upgrades the project into a research-oriented experimental system:

> **VIB PhaseLab: a robustness phase-transition and representation-diagnosis platform for information bottleneck compression.**

The upgraded project studies not only whether VIB is robust, but **when representation compression helps, when it fails, and how the latent space changes across datasets, corruption types, and bottleneck strength**.

The central research question is:

> **When does VIB compression become useful compression, and when does it become destructive over-compression?**

This framing allows both positive and negative results. VIB is not expected to universally beat CNN baselines. Instead, the project should identify the conditions under which β-controlled compression improves clean generalization, improves robustness, has no effect, or collapses task-relevant information.

## 2. Research hypotheses

### H1: Compression and robustness are non-monotonic

As β increases, KL proxy should generally decrease, but robustness should not be assumed to increase monotonically. There may be a middle region where VIB preserves accuracy while reducing input-information retention. At larger β, the model may enter over-compression, where both clean accuracy and noisy robustness degrade.

### H2: The useful β region depends on dataset complexity

MNIST may tolerate stronger compression because the classification structure is simple. Fashion-MNIST and CIFAR-10 require more fine-grained visual information, so the same β may cause earlier task-information loss. CIFAR-10 is included specifically to test whether VIB becomes harder to tune as data complexity increases.

### H3: The best β depends on corruption type

Gaussian noise, salt-and-pepper noise, blur, and contrast corruption damage inputs in different ways. VIB may help when corruption injects local noise, but it may be less useful when corruption destroys task-relevant structure or contrast information.

### H4: Robustness changes should be explainable through latent geometry

If VIB improves robustness, latent space should show better class separation or greater stability under corruption. If VIB over-compresses, latent space should show reduced class separation, low KL proxy, low silhouette score, or collapsed class centers.

## 3. Scope

### Must-have scope

VIB PhaseLab must add:

1. CIFAR-10 support.
2. Multiple corruption types: Gaussian, salt-and-pepper, blur, and contrast.
3. A denser β scan for VIB-CNN.
4. Robustness AUC, normalized robustness AUC, compression benefit index, KL collapse score, and phase labels.
5. Latent geometry metrics: intra-class variance, inter-class distance, Fisher ratio, silhouette score, and class-center norms.
6. Latent drift metrics under corruption.
7. A new `artifacts/phase` artifact root that does not overwrite `artifacts/full`.
8. FastAPI endpoints for phase artifacts.
9. A redesigned React dashboard that behaves as an experimental sandbox rather than a static result viewer.
10. A rewritten Chinese report organized around hypotheses H1-H4.

### Optional scope

VIB PhaseLab may add:

1. Sample trajectory explorer.
2. Confusion flow analysis.
3. β warmup training.
4. Target-KL adaptive β controller.
5. Adaptive β replay panel in the frontend.

### Explicitly out of scope

The upgrade must not add:

1. Online training in the web app.
2. User accounts or upload workflows.
3. Adversarial attack experiments.
4. ResNet-scale benchmark competition as the main story.
5. Exact mutual-information estimation beyond KL proxy.
6. LLM/RAG functionality.

## 4. Experimental design

## 4.1 Datasets

VIB PhaseLab uses three datasets:

- MNIST
- Fashion-MNIST
- CIFAR-10

MNIST and Fashion-MNIST reuse the existing experiment pipeline. CIFAR-10 requires 3-channel input support and a moderately stronger encoder, but the project should keep the model small enough that the focus remains on information bottleneck behavior rather than large-scale architecture engineering.

## 4.2 Models

The required models are:

- CNN baseline
- VIB-CNN with fixed β

CIFAR-10 may use a separate CIFAR encoder, such as `CifarConvEncoder`, with deeper convolution blocks than the MNIST encoder. The VIB head should preserve the existing interface: logits, latent z, mu, and logvar.

## 4.3 β grid

The target dense β grid is:

```text
0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1
```

If GPU time becomes a constraint, CIFAR-10 may use a sparse fallback grid:

```text
0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1
```

The report must state which grid was actually used.

## 4.4 Corruption types and severities

VIB PhaseLab evaluates four corruption types:

1. Gaussian noise
2. Salt-and-pepper noise
3. Blur
4. Contrast

All corruption types use a shared severity grid:

```text
0.0, 0.1, 0.2, 0.3, 0.4
```

Severity maps to corruption-specific parameters:

- Gaussian: `sigma = severity`
- Salt-and-pepper: `probability = severity * 0.4`
- Blur: severity maps to blur kernel/sigma in a deterministic way
- Contrast: `factor = 1 - severity`

Severity `0.0` must preserve clean inputs, so every corruption curve includes a clean reference point.

## 5. Phase indicators

VIB PhaseLab introduces derived indicators that turn raw curves into interpretable phase labels.

### 5.1 Robustness AUC

For each dataset, experiment, and corruption type:

```text
robustness_auc = mean accuracy over all severities
```

This is a simple course-project-friendly AUC proxy because all severity grids are uniform.

### 5.2 Normalized robustness AUC

```text
normalized_robustness_auc = robustness_auc / clean_accuracy
```

This separates absolute accuracy from degradation resistance. A model can have high robustness AUC simply because its clean accuracy is high; normalized AUC asks how much performance it preserves under corruption.

### 5.3 Compression benefit index

For each VIB experiment and corruption type:

```text
CBI = normalized_robustness_auc_vib - normalized_robustness_auc_cnn_baseline
```

Positive CBI means VIB preserves noisy performance better than the CNN baseline after accounting for clean accuracy.

### 5.4 KL collapse score

For each dataset and VIB β:

```text
kl_collapse_score = average_kl_beta / average_kl_beta0
```

If `average_kl_beta0` is zero or unavailable, the score should be null and the phase label should avoid collapse claims for that experiment.

### 5.5 Over-compression flag

An experiment is marked as over-compressed if:

```text
test_accuracy_drop_from_best > 0.10
and kl_collapse_score < 0.05
```

The accuracy drop is measured as an absolute fraction, so 0.10 means 10 percentage points.

### 5.6 Phase labels

Each VIB experiment receives one phase label:

- `under-regularized`: KL remains high and compression is weak.
- `useful-compression`: KL drops meaningfully while clean accuracy remains close to the best model and robustness AUC does not degrade.
- `robustness-specialized`: clean accuracy is not best, but normalized robustness AUC or CBI is strong.
- `over-compressed`: KL collapse and large accuracy loss occur together.
- `unstable`: clean and robustness conclusions conflict across corruption types.

The exact threshold constants must be centralized in one analysis module so they are visible and easy to report.

## 6. Latent diagnostics

## 6.1 Latent geometry

For each experiment, collect latent representations on a fixed evaluation subset and compute:

- `intra_class_variance`: average distance from samples to their class center.
- `inter_class_distance`: average pairwise distance between class centers.
- `fisher_ratio`: inter-class distance divided by intra-class variance.
- `silhouette_score`: computed on a capped subset to avoid excessive cost.
- `class_center_norm`: per-class latent center norm.

These metrics should be written to `latent_geometry.json`.

## 6.2 Latent drift under corruption

For the same sample subset, compute clean latent vectors and corrupted latent vectors. For each corruption, severity, and class:

```text
drift = ||z_clean - z_corrupted||
```

The artifact must aggregate:

- mean drift
- median drift
- per-class drift
- corruption type
- severity
- label

These metrics should be written to `latent_drift.json`.

Interpretation rules:

- Low drift and high accuracy suggest stable and useful representations.
- Low drift and low accuracy suggest collapsed representations.
- High drift and low accuracy suggest corruption-sensitive representations.
- High drift and high accuracy suggest latent movement that preserves class separability.

## 6.3 Confusion flow

Optional confusion flow artifacts track transitions from clean correct predictions to corrupted predictions. They help explain which classes become confused under corruption and whether VIB changes the error structure.

## 6.4 Sample trajectories

Optional sample trajectory artifacts store a small set of representative examples for interactive inspection:

- image id
- true label
- clean prediction
- corrupted predictions
- confidence
- entropy
- clean latent coordinates
- corrupted latent coordinates
- drift distance

Sample trajectory storage should be capped to keep artifacts small.

## 7. Adaptive β bonus experiments

Adaptive β is a bonus and must not become the core deliverable.

## 7.1 β warmup

Warmup increases β from 0 to a target value during early epochs:

```text
beta_t = beta_target * min(1, epoch / warmup_epochs)
```

The goal is to reduce early KL pressure and avoid premature collapse.

## 7.2 Target-KL controller

The target-KL controller adjusts β to keep KL within a target interval:

```text
if kl > target_high:
    beta *= 1.1
elif kl < target_low:
    beta *= 0.9
```

β must be clamped:

```text
beta_min = 1e-6
beta_max = 1
```

## 7.3 Bonus experiment range

Adaptive β should be tested only on representative datasets:

- Fashion-MNIST
- CIFAR-10

The comparison set is:

- fixed β best-clean
- fixed β best-robust
- β warmup
- target-KL controller

The required artifact is `adaptive_curves.json`, containing epoch, β, train KL, validation KL, train accuracy, validation accuracy, and a lightweight robustness proxy.

## 8. Artifact contract

The existing `artifacts/full` root remains unchanged. VIB PhaseLab writes a new root:

```text
artifacts/phase/
  index.json
  summary.json
  diagnostics/
    sample_trajectories.json
    confusion_flows.json
    class_geometry.json
  adaptive/
    <adaptive_experiment_id>/
      metrics.json
      adaptive_curves.json
      robustness.json
  <experiment_id>/
    metrics.json
    robustness.json
    confusion_matrix.json
    latent_pca.json
    latent_geometry.json
    latent_drift.json
    phase_indicators.json
```

## 8.1 `robustness.json`

```json
{
  "dataset": "fashion_mnist",
  "experiment_id": "fashion_mnist_vib_beta_0_01",
  "rows": [
    {
      "corruption": "gaussian",
      "severity": 0.2,
      "accuracy": 0.59,
      "mean_confidence": 0.71,
      "mean_entropy": 0.82
    }
  ]
}
```

## 8.2 `latent_geometry.json`

```json
{
  "intra_class_variance": 1.24,
  "inter_class_distance": 4.88,
  "fisher_ratio": 3.94,
  "silhouette_score": 0.41,
  "per_class": [
    {
      "label": 0,
      "intra_variance": 1.12,
      "center_norm": 3.2
    }
  ]
}
```

## 8.3 `phase_indicators.json`

```json
{
  "robustness_auc": {
    "gaussian": 0.72,
    "salt_pepper": 0.68,
    "blur": 0.81,
    "contrast": 0.77
  },
  "normalized_robustness_auc": {
    "gaussian": 0.84
  },
  "compression_benefit_index": {
    "gaussian": 0.03
  },
  "kl_collapse_score": 0.14,
  "phase_label": "useful-compression",
  "over_compression_flag": false
}
```

## 8.4 `latent_drift.json`

```json
{
  "rows": [
    {
      "corruption": "gaussian",
      "severity": 0.3,
      "label": 7,
      "mean_drift": 2.31,
      "median_drift": 2.02
    }
  ]
}
```

## 9. Backend design

The backend remains read-only and artifact-driven. It must not launch training.

New endpoints:

```text
GET /api/phase/index
GET /api/phase/summary
GET /api/phase/experiments/{experiment_id}
GET /api/phase/diagnostics
GET /api/phase/adaptive/{experiment_id}
```

Backend behavior:

- `VIB_PHASE_ARTIFACT_DIR` configures the phase artifact root and defaults to `artifacts/phase`.
- If phase artifacts are missing, endpoints return clear 404 responses.
- The backend should not compute phase indicators on request; all expensive analysis is done offline.
- Existing `artifacts/full` endpoints remain available for backward compatibility during transition.

## 10. Frontend design

The frontend should evolve from a dashboard into an experimental sandbox.

## 10.1 Overview

Purpose: give the teacher an immediate research-level summary.

Content:

- Dataset cards for MNIST, Fashion-MNIST, CIFAR-10.
- Best clean β.
- Best robust β.
- Collapse β if present.
- One generated conclusion sentence per dataset.

## 10.2 Phase Map

Purpose: visualize robustness phase transitions.

Main view:

- x-axis: β
- y-axis: dataset/corruption combinations
- color: selected metric

Metric toggles:

- clean accuracy
- robustness AUC
- normalized robustness AUC
- KL proxy
- Fisher ratio
- phase label

Interaction:

- hover highlights β across all rows.
- click opens experiment detail.
- phase labels are shown as badges.

## 10.3 Compression Lens

Purpose: inspect one dataset across β.

Linked charts:

- β vs clean accuracy
- β vs KL proxy
- β vs robustness AUC
- β vs Fisher ratio

Interaction:

- hover a β value to highlight the same experiment in all charts.
- show automatic explanation text: useful compression, robustness-specialized, or over-compressed.

## 10.4 Diagnosis Lab

Purpose: explain why a phase appears.

Controls:

- dataset
- experiment / β
- corruption type
- severity
- class label

Displays:

- latent PCA or UMAP
- class geometry metrics
- latent drift by class
- confusion matrix or confusion flow

## 10.5 Sample Explorer

Optional purpose: inspect individual sample trajectories.

Displays:

- clean image
- corrupted variants
- predictions and confidence
- latent trajectory
- drift distance

## 10.6 Adaptive Replay

Optional purpose: replay bonus adaptive β experiments.

Displays:

- β over epochs
- KL over epochs
- train/validation accuracy
- robustness proxy

## 11. Report design

The final report should be rewritten around the research hypotheses rather than around implementation modules.

Recommended structure:

```text
1 引言：为什么 VIB 鲁棒性不是 yes/no 问题
2 理论背景：IB objective、KL proxy、compression-prediction tradeoff
3 研究假设：H1-H4
4 实验系统：phase scan + diagnostic lab
5 鲁棒性相变图谱
   5.1 MNIST
   5.2 Fashion-MNIST
   5.3 CIFAR-10
   5.4 跨 corruption 比较
6 表示空间诊断
   6.1 latent geometry
   6.2 latent drift
   6.3 confusion flow
7 Adaptive β 初步探索
8 Web 实验沙盘
9 局限性
10 结论
```

The main conclusion should be conditional:

> VIB provides a controllable compression mechanism, but robustness gains appear only in limited useful-compression regions. Dataset complexity and corruption type shift these regions. Latent diagnostics help identify whether compression improves representation stability or destroys task information.

## 12. Implementation phases

## Phase 1: Analysis metrics on existing artifacts

Add robustness AUC, normalized AUC, CBI, phase labels, and latent geometry for existing MNIST/Fashion-MNIST artifacts. This validates the indicator design before new GPU experiments.

## Phase 2: Corruption system upgrade

Add salt-and-pepper, blur, and contrast corruptions. Extend evaluation and artifacts to record corruption type, severity, accuracy, confidence, and entropy.

## Phase 3: CIFAR-10 support

Add CIFAR-10 data loading, CIFAR encoder support, tiny CIFAR training smoke tests, and artifact compatibility.

## Phase 4: Phase experiment runner

Add a resumable phase sweep runner for dataset × model × β × corruption evaluation. It should skip existing completed artifacts and write to `artifacts/phase`.

## Phase 5: Phase Explorer frontend

Add Overview, Phase Map, Compression Lens, and Diagnosis Lab views. Keep Sample Explorer and Adaptive Replay optional unless artifacts are available.

## Phase 6: Report rewrite

Regenerate summaries and rewrite the report around H1-H4 using only measured artifacts.

## 13. Risks and controls

### Risk: experiment count becomes too large

Control: run MNIST/Fashion-MNIST dense β first; use sparse CIFAR-10 grid if needed. Keep adaptive β optional.

### Risk: CIFAR-10 VIB underperforms

Control: treat this as an expected possible result. The research question asks when VIB helps or fails, not whether it always wins.

### Risk: frontend becomes too complex

Control: all expensive computation remains offline. Frontend reads summary artifacts and focuses on exploration, not computation.

### Risk: adaptive β does not outperform fixed β

Control: label adaptive β as a preliminary control experiment. Negative results are acceptable if analyzed.

### Risk: artifact schema breaks current dashboard

Control: write VIB PhaseLab output to `artifacts/phase` and keep existing `artifacts/full` endpoints working during transition.

## 14. Acceptance criteria

The VIB PhaseLab upgrade is complete when:

1. `artifacts/phase` contains phase artifacts for MNIST, Fashion-MNIST, and CIFAR-10.
2. At least Gaussian, salt-and-pepper, blur, and contrast corruption evaluations are present.
3. Every phase experiment has `metrics.json`, `robustness.json`, `latent_geometry.json`, and `phase_indicators.json`.
4. VIB experiments have `latent_drift.json` where feasible.
5. Summary artifacts identify best clean β, best robust β, collapse regions, and phase labels.
6. Backend phase endpoints return index, summary, experiment details, diagnostics, and adaptive details when available.
7. Frontend includes Overview, Phase Map, Compression Lens, and Diagnosis Lab.
8. The Chinese report is rewritten around H1-H4 and includes CIFAR-10 and multi-corruption results.
9. Verification commands pass, including Python tests, artifact validation, summary generation, frontend tests, frontend build, and browser verification.
10. The report states explicitly that KL is a proxy / variational upper bound, not exact mutual information.

## 15. Non-goals for implementation planning

The implementation plan should not include commits unless the user explicitly asks. It should not include cloud deployment as a required step. It should not require exact mutual-information estimation or adversarial robustness. It should not convert the app into an online training platform.
