#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PYTHON="${NEUROMOD_BASE_PYTHON:-$WORKSPACE/.venv/bin/python}"

cd "$ROOT"
uv venv .causal_venv --python "$PYTHON"
uv pip install --python .causal_venv/bin/python -r requirements-causal.txt
PYTHONPATH="$ROOT/src" .causal_venv/bin/python -m neuromod_benchmark.causal_worker --check
