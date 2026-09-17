#!/usr/bin/env bash
# Run the full synthetic validation suite (Gate 3).
set -euo pipefail
PY="${PYTHON:-python}"
echo "== E1: hidden thermostat =="
$PY -m sid_neuromod.experiments.run_synthetic --config configs/synthetic_hidden_thermostat.yaml
echo "== E3: change detection =="
$PY -m sid_neuromod.experiments.run_synthetic --config configs/synthetic_change_detection.yaml
echo "All synthetic experiments complete. See output/synthetic/."
