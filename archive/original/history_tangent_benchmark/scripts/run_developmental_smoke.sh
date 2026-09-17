#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

"${PYTHON}" -m pytest -q
"${PYTHON}" -m history_tangent_benchmark.cli validate-oracles \
  --config configs/developmental_smoke.yaml
"${PYTHON}" -m history_tangent_benchmark.cli run \
  --config configs/developmental_smoke.yaml
