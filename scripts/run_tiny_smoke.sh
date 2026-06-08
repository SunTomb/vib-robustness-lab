#!/usr/bin/env bash
set -euo pipefail

python -m vib_project.cli run --dataset mnist --model cnn --beta 0 --epochs 1 --limit-train 256 --limit-test 128 --output-dir artifacts/dev
python -m vib_project.cli run --dataset mnist --model vib --beta 0.001 --epochs 1 --limit-train 256 --limit-test 128 --output-dir artifacts/dev
