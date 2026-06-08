#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-cifar10}" \
BETAS="${BETAS:-0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1}" \
bash scripts/run_phase_experiments.sh
