#!/usr/bin/env bash
set -euo pipefail

python -m vib_project.cli phase-run --dataset mnist --model cnn --beta 0 --epochs 1 --limit-train 64 --limit-test 32 --batch-size 32 --latent-dim 8 --output-dir artifacts/phase-smoke --corruptions gaussian contrast --severities 0.0 0.2 --max-diagnostic-points 32
python scripts/validate_phase_artifacts.py artifacts/phase-smoke
python -m vib_project.cli phase-run --dataset mnist --model vib --beta 0.001 --epochs 1 --limit-train 64 --limit-test 32 --batch-size 32 --latent-dim 8 --output-dir artifacts/phase-smoke --corruptions gaussian contrast --severities 0.0 0.2 --max-diagnostic-points 32
python scripts/validate_phase_artifacts.py artifacts/phase-smoke
