# VIB Project Submission Manifest

## Evidence source

- Full experiment artifacts: `artifacts/full/`
- Metrics table: `report/results/metrics_summary.csv`
- Robustness table: `report/results/robustness_summary.csv`
- Figures: `report/figures/*.svg`
- Dashboard screenshots: `report/figures/dashboard_*.png`
- Final report: `report/final_report.md`

## Experiment provenance

- Datasets: MNIST, Fashion-MNIST
- Models: CNN baseline, VIB-CNN
- β grid: 0, 1e-4, 1e-3, 1e-2, 1e-1, 1
- Noise grid: σ = 0.0, 0.1, 0.2, 0.3, 0.4
- Full artifact count: 14 experiments
- Robustness rows: 70 evaluations
- Full experiment log: `logs/core_experiments_20260607_160359.log`

## Submission policy

- Source code should be committed.
- Report files and derived CSV/SVG/PNG figures should be committed.
- `artifacts/full` is currently ignored by `.gitignore`; include it in the course submission archive or commit it only if the user explicitly chooses that policy.
- Local caches, datasets, virtual environments, `node_modules`, frontend build output, and temporary Claude files should not be submitted.

## Verification commands

```bash
python -m pytest -v
python scripts/validate_full_artifacts.py artifacts/full
python scripts/summarize_results.py --artifact-root artifacts/full --output-dir report/results
npm --prefix frontend test -- src/lib/metrics.test.ts
npm --prefix frontend run build
```
