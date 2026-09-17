#!/usr/bin/env bash
set -euo pipefail
PY="${PYTHON:-python}"
$PY -m cProfile -s cumtime -m sid_neuromod.experiments.run_elegans_fit \
  --config configs/elegans_fast_debug.yaml | head -40
