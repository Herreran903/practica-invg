#!/usr/bin/env bash

set -u

# Simple smoke test runner for data generation and training CLIs
# Usage: bash scripts/smoke.sh

# Prefer python3 if available
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

export PYTHONPATH="$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

NAMES=()
CODES=()

run() {
  local name="$1"
  shift
  echo "================================================================"
  echo ">>> $name"
  echo "----------------------------------------------------------------"
  echo "+ $*"
  if eval "$*"; then
    code=0
    echo ">>> RESULT: OK"
  else
    code=$?
    echo ">>> RESULT: FAIL (exit $code)"
  fi
  NAMES+=("$name")
  CODES+=("$code")
  echo
}

# CLI help checks (fast, just parser wiring)
run "CLI help: data_generation jssp_images" "$PY -m src.data_generation.jssp_images.cli -h"
run "CLI help: data_generation jssp_tensors" "$PY -m src.data_generation.jssp_tensors.cli -h"
run "CLI help: data_generation sat_images" "$PY -m src.data_generation.sat_images.cli -h"
run "CLI help: training jssp_images" "$PY -m src.training.jssp_images.cli -h"
run "CLI help: training jssp_tensors" "$PY -m src.training.jssp_tensors.cli -h"

# Dataset detected in repo (adjust if you want to smoke-test a different set)
JSSP_CSV="data/jssp/datasets/jsp_cnn_data_gen_22/ground_truth_jsp_generated_dataset.csv"

if [ -f "$JSSP_CSV" ]; then
  echo "Using dataset CSV: $JSSP_CSV"

  # Convert-only (skip solver runs) – fast
  run "SMOKE: jssp_images data_generation convert-only" \
    "$PY -m src.data_generation.jssp_images.cli --mode academic --skip-solvers --csv \"$JSSP_CSV\""

  run "SMOKE: jssp_tensors data_generation convert-only" \
    "$PY -m src.data_generation.jssp_tensors.cli --mode academic --skip-solvers --csv \"$JSSP_CSV\" --no-standardize"

  # Quick training (tiny epochs/folds to validate pipeline)
  run "SMOKE: training jssp_images (1 epoch, 2 folds)" \
    "$PY -m src.training.jssp_images.cli --csv \"$JSSP_CSV\" --task classification --epochs 1 --folds 2 --batch_size 4"

  run "SMOKE: training jssp_tensors (1 epoch, 2 folds)" \
    "$PY -m src.training.jssp_tensors.cli --csv \"$JSSP_CSV\" --task classification --epochs 1 --folds 2 --batch_size 4"
else
  echo "Dataset CSV not found at: $JSSP_CSV"
  echo "Skipping smoke runs that require data. Adjust JSSP_CSV in scripts/smoke.sh."
fi

echo "================================================================"
echo "SUMMARY"
echo "================================================================"
failures=0
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  code="${CODES[$i]}"
  if [ "$code" -eq 0 ]; then
    printf "  [OK]   %s\n" "$name"
  else
    printf "  [FAIL] %s (exit %s)\n" "$name" "$code"
    failures=$((failures+1))
  fi
done

if [ "$failures" -eq 0 ]; then
  echo "All smoke tests passed."
  exit 0
else
  echo "$failures smoke test(s) failed."
  exit 1
fi