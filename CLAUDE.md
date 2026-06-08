# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Absolute safety instruction

Do not use PowerShell commands or the PowerShell tool in this project. The user is connected through CC Switch and explicitly warned that PowerShell tool use may trigger Claude Code internal stack leakage.

Use dedicated file/search/edit/Git tools whenever possible. If a terminal command is unavoidable, use Bash or another non-PowerShell shell and explain why before running it.

## Project status

This repository is the completed Information Theory course project:

**VIB PhaseLab：表示压缩的鲁棒性相变与诊断系统**

It began as a Variational Information Bottleneck (VIB) reproduction project and was expanded into a larger experimental system that studies when representation compression is useful and when it becomes destructive over-compression.

Final local artifacts and reports are already present:

- `artifacts/full/` — original MNIST/Fashion-MNIST VIB reproduction artifacts.
- `artifacts/phase/` — final PhaseLab artifacts: 3 datasets, 39 experiment configs, 780 robustness rows.
- `report/phaselab_report.md` — final PhaseLab report generated from real artifacts.
- `report/phaselab_report.tex` — LaTeX version of the final individual course report.
- `report/phaselab_report.pdf` — compiled PDF version of the final report.
- `report/results/` and `report/phase_results/` — CSV/JSON summaries and key findings.
- `logs/phase_full_20260608_001753.log` — lab GPU full PhaseLab run log.

Do not invent experiment numbers. Any future report edits must be based on `artifacts/full`, `artifacts/phase`, or regenerated summaries.

## Read first

Before implementation, read:

1. `README.md` — current usage, verification commands, and final artifact overview.
2. `docs/superpowers/specs/2026-06-07-vib-phaselab-design.md` — final approved PhaseLab design.
3. `docs/superpowers/plans/2026-06-07-vib-phaselab.md` — task-level implementation plan.
4. `CLAUDE.md` — this handoff and operating guide.

Older robust-generalization specs/plans may exist as historical context, but PhaseLab is the final project framing.

## Architecture

The project has five cooperating layers:

- `src/vib_project/`: offline experiment package. It owns dataset loading, corruptions, CNN/VIB models, VIB loss, training/evaluation, latent diagnostics, phase metrics, artifact writing, and CLI entrypoints.
- `artifacts/`: static JSON boundary between experiments, backend, frontend, and report. The dashboard must read artifacts; it must not launch training.
- `backend/`: FastAPI read-only artifact API. It exposes original experiment endpoints and PhaseLab endpoints.
- `frontend/`: React + TypeScript + Vite Chinese dashboard. It visualizes original VIB results plus PhaseLab overview, phase map, compression lens, and diagnosis lab.
- `report/`: Chinese reports, CSV/JSON summaries, figures, screenshots, and final submission materials.

Data flow:

`PyTorch offline training -> static JSON artifacts -> FastAPI API -> React dashboard -> report screenshots/figures`.

## Information-theory framing

Keep the work grounded in information theory:

- VIB objective: maximize `I(Z;Y) - beta * I(Z;X)`.
- Training loss: `CE(y_hat, y) + beta * KL(q_phi(z|x) || p(z))`.
- `average_kl` is a KL proxy / variational upper bound for input-information retention, not an exact mutual-information estimate.
- `test_accuracy`, corruption accuracy, and robustness AUC are prediction/robustness proxies, not direct `I(Z;Y)` estimates.
- Phase labels are diagnostic labels for experimental comparison, not theoretical guarantees.

PhaseLab indicators:

- `robustness_auc`: average corruption accuracy across severities per corruption.
- `normalized_robustness_auc`: robustness AUC divided by clean accuracy.
- `compression_benefit_index`: VIB normalized robustness AUC minus same-dataset CNN baseline normalized AUC.
- `kl_collapse_score`: average KL divided by same-dataset VIB β=0 KL.
- `phase_label`: one of `under-regularized`, `useful-compression`, `robustness-specialized`, `over-compressed`, `unstable`.

## Common commands

Run commands from repository root.

Full verification:

```bash
python -m pytest -v
python scripts/validate_full_artifacts.py artifacts/full
python scripts/validate_phase_artifacts.py artifacts/phase
python scripts/summarize_results.py --artifact-root artifacts/full --output-dir report/results
python scripts/summarize_phase_results.py --artifact-root artifacts/phase --output-dir report/phase_results
python scripts/generate_phaselab_report.py --phase-root artifacts/phase --result-root report/phase_results --output report/phaselab_report.md
npm --prefix frontend test -- src/lib/metrics.test.ts src/lib/phaseMetrics.test.ts
npm --prefix frontend run build
```

Expected final verification state:

- Python/backend tests: 52 passed, 1 FastAPI/Starlette warning.
- Phase artifact validation: 3 datasets, 39 experiments, missing `[]`.
- Phase summary: 39 metric rows, 780 robustness rows, 39 phase rows.
- Frontend tests: 6 passed.
- Frontend build: passes with an acceptable Recharts/Vite chunk-size warning.

Tiny smoke experiments:

```bash
bash scripts/run_tiny_smoke.sh
bash scripts/run_phase_smoke.sh
```

Single original experiment:

```bash
python -m vib_project.cli run --dataset mnist --model vib --beta 0.001 --epochs 1 --limit-train 128 --limit-test 64 --output-dir artifacts/dev
```

Single PhaseLab experiment:

```bash
python -m vib_project.cli phase-run \
  --dataset mnist \
  --model vib \
  --beta 0.001 \
  --epochs 1 \
  --limit-train 128 \
  --limit-test 64 \
  --output-dir artifacts/dev-phase \
  --corruptions gaussian salt_pepper blur contrast \
  --severities 0.0 0.1 0.2 0.3 0.4
```

Core experiment scripts:

```bash
bash scripts/run_core_experiments.sh
bash scripts/run_phase_experiments.sh
```

If cuDNN fails on lab GPUs:

```bash
VIB_DISABLE_CUDNN=1 bash scripts/run_phase_experiments.sh
```

## Backend and frontend

Start backend with both artifact roots:

```bash
VIB_ARTIFACT_DIR=artifacts/full VIB_PHASE_ARTIFACT_DIR=artifacts/phase \
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Start frontend:

```bash
npm --prefix frontend run dev -- --host 127.0.0.1
```

If backend port 8000 is occupied, use another port and set the Vite proxy target:

```bash
VIB_ARTIFACT_DIR=artifacts/full VIB_PHASE_ARTIFACT_DIR=artifacts/phase \
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8010

VITE_API_PROXY_TARGET=http://127.0.0.1:8010 \
  npm --prefix frontend run dev -- --host 127.0.0.1
```

When making UI changes, run backend and frontend dev servers and manually verify in a browser. The final browser verification confirmed that PhaseLab overview, phase map, Compression Lens, and Diagnosis Lab render real `artifacts/phase` data.

## Artifact contracts

Original artifacts under `artifacts/full` include:

```text
index.json
<experiment_id>/
  metrics.json
  curves.json
  latent_pca.json
  confusion_matrix.json
```

Important original keys include `train_accuracy`, `test_accuracy`, `train_test_gap`, `average_kl`, `noise_accuracy`, `curves.noise`, and `latent_pca.points`.

PhaseLab artifacts under `artifacts/phase` include:

```text
index.json
summary.json
<experiment_id>/
  metrics.json
  robustness.json
  latent_geometry.json
  latent_drift.json
  phase_indicators.json
  latent_pca.json
  confusion_matrix.json
```

Important PhaseLab keys include:

- `robustness.rows[].corruption`
- `robustness.rows[].severity`
- `robustness.rows[].accuracy`
- `phase_indicators.robustness_auc`
- `phase_indicators.normalized_robustness_auc`
- `phase_indicators.compression_benefit_index`
- `phase_indicators.kl_collapse_score`
- `phase_indicators.phase_label`
- `latent_geometry.fisher_ratio`
- `latent_drift.rows`

`scripts/summarize_phase_results.py` refreshes `phase_indicators.json` using cross-experiment references, then writes both `report/phase_results/phase_summary.json` and `artifacts/phase/summary.json`. Keep both outputs because the report and backend read them from different places.

## Lab GPU cluster and NAS notes

The lab environment has shared GPU servers and NAS workspaces.

Known resources:

- `Sui-3-Wu`: 8 × NVIDIA RTX 3090. It may not have a ready PyTorch/torchvision/sklearn environment.
- `Tang-2-Wu`: A40 GPU node. This project used `Tang-2-Wu` successfully with `tn_env`.
- Small cloud server `root@38.76.179.17` exists for lightweight deployment only; do not build heavy PyTorch/CUDA workflows there.

Final PhaseLab full run used:

```bash
ssh Tang-2-Wu
cd /NAS/yesh/vib-robustness-lab
source /data/wujcan/yesh/miniconda3/etc/profile.d/conda.sh
conda activate tn_env
export VIB_DISABLE_CUDNN=1 PYTHONPATH=src
CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase_experiments.sh
```

The full run produced `logs/phase_full_20260608_001753.log` and then artifacts were synced back locally.

Before installing large packages on shared servers, inspect existing environments. Avoid duplicating PyTorch/CUDA installations. During this project, an attempted `torchvision` install on Sui began pulling a different torch version and was stopped; prefer known-good environments like Tang `tn_env` unless the user asks otherwise.

For long experiments, prefer resumable scripts and short one-shot SSH checks. Long-lived SSH monitors were unstable in this session; one-shot polling plus scheduled wakeups was more reliable.

## Development workflow

- Keep the dashboard artifact-driven. Do not add online training, user accounts, RAG/LLM features, or adversarial attack tooling unless the user explicitly changes scope.
- Prefer changing Python/backend/frontend code to preserve artifact contracts rather than adding frontend-only special cases.
- Do not commit unless the user explicitly asks. When committing, inspect git status and diff first, stage explicit paths, and avoid secrets, virtualenvs, caches, `node_modules`, raw datasets, and unrelated files.
- Generated PhaseLab artifacts are part of the final course deliverable, but be mindful of repository size before pushing.

## Verification expectations

Backend/Python:

- Unit tests should cover KL non-negativity, model shapes, corruption clamping/reproducibility, artifact schema, summary/export behavior, report generation, and API response shape.
- Run `python -m pytest -v` before claiming Python/backend changes are complete.

Frontend:

- Run the targeted Vitest tests and `npm --prefix frontend run build`.
- UI changes require browser verification of the actual dashboard, not just tests.

Experiments/report:

- Final claims must come from real `artifacts/full` or `artifacts/phase` outputs.
- `report/phaselab_report.md` is generated by `scripts/generate_phaselab_report.py`; prefer regenerating it from artifacts rather than editing measured values by hand.

## Communication and operating preferences

- Avoid PowerShell entirely in this repository.
- Prefer dedicated tools for file operations and Git MCP/dedicated Git tools for status/diff/log where possible.
- The user values a project direction that combines credible information-theory experiments, a practical web demo, GPU-backed reproducibility, and enough experimental innovation to feel like a major course project.
- When making trade-offs, choose the path that first preserves a reliable artifact-driven loop and clear report evidence.
