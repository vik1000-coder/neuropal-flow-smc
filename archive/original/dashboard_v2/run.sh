#!/bin/sh
set -eu
dashboard_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$dashboard_root"
if [ ! -x "$dashboard_root/.venv/bin/python" ]; then
  echo "The repository Python environment is missing: $dashboard_root/.venv/bin/python" >&2
  exit 1
fi
exec "$dashboard_root/.venv/bin/python" -m dashboard_v2.server "$@"
