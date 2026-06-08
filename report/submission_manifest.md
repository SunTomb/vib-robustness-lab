# VIB PhaseLab Submission Manifest

## Evidence source

- Original reproduction artifacts: `artifacts/full/`
- PhaseLab artifacts: `artifacts/phase/`
- Original metrics tables: `report/results/`
- PhaseLab summary tables: `report/phase_results/`
- Figures and screenshots: `report/figures/`
- Final report Markdown: `report/phaselab_report.md`
- Final report LaTeX: `report/phaselab_report.tex`
- Final report PDF: `report/phaselab_report.pdf`

## Experiment provenance

- Datasets: MNIST, Fashion-MNIST, CIFAR-10
- Models: CNN baseline, VIB-CNN
- PhaseLab beta grid: 0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1
- Corruptions: Gaussian, salt-and-pepper, blur, contrast
- Severities: 0.0, 0.1, 0.2, 0.3, 0.4
- PhaseLab artifact count: 39 experiments
- PhaseLab robustness rows: 780 evaluations
- Full PhaseLab experiment log: `logs/phase_full_20260608_001753.log`

## Submission policy

- Source code, reports, figures, and final JSON artifacts are committed.
- `report/final_report.md` has been superseded by `report/phaselab_report.md`, `report/phaselab_report.tex`, and `report/phaselab_report.pdf`.
- Local caches, datasets, virtual environments, `node_modules`, frontend build output, and temporary Claude files should not be submitted.

## Verification commands

```bash
python -m pytest -v
python scripts/validate_full_artifacts.py artifacts/full
python scripts/validate_phase_artifacts.py artifacts/phase
python scripts/summarize_results.py --artifact-root artifacts/full --output-dir report/results
python scripts/summarize_phase_results.py --artifact-root artifacts/phase --output-dir report/phase_results
python scripts/generate_phase_figures.py --result-root report/phase_results --output-dir report/figures
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=report/build report/phaselab_report.tex
npm --prefix frontend test -- src/lib/metrics.test.ts src/lib/phaseMetrics.test.ts
npm --prefix frontend run build
```
