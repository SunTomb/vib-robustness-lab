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
