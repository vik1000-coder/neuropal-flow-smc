#!/usr/bin/env bash
set -euo pipefail
PY="${PYTHON:-python}"
$PY -m sid_neuromod.experiments.run_elegans_fit --config configs/elegans_default.yaml
$PY -m sid_neuromod.experiments.make_report --run-dir output/elegans/default
