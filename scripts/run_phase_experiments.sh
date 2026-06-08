#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/phase}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LATENT_DIM="${LATENT_DIM:-32}"
DATASETS="${DATASETS:-mnist fashion_mnist cifar10}"
BETAS="${BETAS:-0 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1}"
CORRUPTIONS="${CORRUPTIONS:-gaussian salt_pepper blur contrast}"
SEVERITIES="${SEVERITIES:-0.0 0.1 0.2 0.3 0.4}"
MAX_DIAGNOSTIC_POINTS="${MAX_DIAGNOSTIC_POINTS:-1000}"

is_complete() {
  local experiment_id="$1"
  local dir="$OUTPUT_DIR/$experiment_id"
  [[ -f "$dir/metrics.json" && -f "$dir/robustness.json" && -f "$dir/latent_geometry.json" && -f "$dir/phase_indicators.json" ]]
}

run_one() {
  local dataset="$1"
  local model="$2"
  local beta="$3"
  local experiment_id
  experiment_id=$(python - <<PY
from pathlib import Path
from vib_project.config import ExperimentConfig
config = ExperimentConfig(
    dataset="$dataset",
    model="$model",
    beta=float("$beta"),
    epochs=1,
    batch_size=1,
    latent_dim=1,
    learning_rate=1e-3,
    seed=7,
    output_dir=Path("$OUTPUT_DIR"),
)
print(config.experiment_id)
PY
)
  if is_complete "$experiment_id"; then
    echo "SKIP $experiment_id"
    return
  fi
  python -m vib_project.cli phase-run \
    --dataset "$dataset" \
    --model "$model" \
    --beta "$beta" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --latent-dim "$LATENT_DIM" \
    --output-dir "$OUTPUT_DIR" \
    --corruptions $CORRUPTIONS \
    --severities $SEVERITIES \
    --max-diagnostic-points "$MAX_DIAGNOSTIC_POINTS"
}

for DATASET in $DATASETS; do
  run_one "$DATASET" cnn 0
  for BETA in $BETAS; do
    run_one "$DATASET" vib "$BETA"
  done
done

python scripts/validate_phase_artifacts.py "$OUTPUT_DIR"
python scripts/summarize_phase_results.py --artifact-root "$OUTPUT_DIR" --output-dir report/phase_results
