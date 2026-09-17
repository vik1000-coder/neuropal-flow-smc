#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
export PYTHONPATH="$ROOT/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
exec nice -n 19 caffeinate -i "$WORKSPACE/.venv/bin/python" \
  "$ROOT/scripts/run_robustness.py" "$ROOT/configs/robustness_medium.yaml"
