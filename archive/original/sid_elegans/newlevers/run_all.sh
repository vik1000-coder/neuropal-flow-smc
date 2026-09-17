#!/bin/bash
# Overnight driver for the 8 new-lever analyses.
# Design goals: (1) never crash the machine, (2) survive individual-lever failures,
# (3) finish before morning. Runs each lever SEQUENTIALLY (peak memory = one job at a time),
# caps BLAS threads (so no oversubscription on the 10-core box), nice-19 (yields to the user),
# and runs under caffeinate so an idle laptop doesn't sleep mid-run.
#
# Usage:  bash sid_elegans/newlevers/run_all.sh            # full overnight run
#         NL_NSURR=50 NL_NBOOT=50 bash .../run_all.sh       # lighter
#         LEVERS="functional gating" bash .../run_all.sh     # a subset
set -u
ROOT=/Users/vik/Developer/new_sbtg_neuro
PY=$ROOT/.venv/bin/python
LOGDIR=$ROOT/sid_elegans/output/newlevers/logs
mkdir -p "$LOGDIR"

# ---- resource caps (each sequential job uses <=3 BLAS threads; 10 cores => big headroom) ----
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-3}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-3}
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH=.:SBTG

# ---- heavy-count knobs (the scripts read these) ----
export NL_NSURR=${NL_NSURR:-200}
export NL_NBOOT=${NL_NBOOT:-200}
export NL_NPERM=${NL_NPERM:-5000}
export NL_THROTTLE=${NL_THROTTLE:-0.3}
export NEWLEVERS_SMOKE=${NEWLEVERS_SMOKE:-0}   # set =1 externally for a fast end-to-end integration test

# Order: highest-value + cheapest first, slowest last, so an early stop still yields the best
# results. shrinkage (28w ACMMA + w-grid) and ladder (whole-grid max-stat bootstrap) are the two
# heaviest -> last; frozen (neural, cpu) before them.
LEVERS=${LEVERS:-"functional selfgain predlik gating classagg frozen shrinkage ladder"}

MANIFEST="$LOGDIR/manifest.txt"
: > "$MANIFEST"
echo "=== new-levers overnight run started $(date) ===" | tee -a "$MANIFEST"
echo "threads=$OPENBLAS_NUM_THREADS NSURR=$NL_NSURR NBOOT=$NL_NBOOT NPERM=$NL_NPERM throttle=$NL_THROTTLE" | tee -a "$MANIFEST"

run_one () {
  local name=$1
  local script=$ROOT/sid_elegans/newlevers/run_${name}.py
  local log=$LOGDIR/${name}.log
  if [ ! -f "$script" ]; then
    echo "[$name] MISSING SCRIPT ($script) — skipped" | tee -a "$MANIFEST"; return
  fi
  echo "[$name] START $(date '+%H:%M:%S')" | tee -a "$MANIFEST"
  local t0=$(date +%s)
  # per-lever cost caps (subshell so overrides don't leak). ladder's whole-grid max-statistic
  # bootstrap is O(n_boot x lags x sigmas x worms); pin it to the clean 6w config + fewer boots.
  (
    case "$name" in
      ladder) export NL_CONFIG=6w_clean_deconv; export NL_NBOOT=100 ;;
    esac
    nice -n 19 "$PY" "$script" > "$log" 2>&1
  )
  local rc=$?
  local t1=$(date +%s)
  echo "[$name] END   $(date '+%H:%M:%S')  exit=$rc  dur=$(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s  log=$log" | tee -a "$MANIFEST"
}

for L in $LEVERS; do
  run_one "$L"
done

echo "=== all levers done $(date) ===" | tee -a "$MANIFEST"
echo "outputs: $ROOT/sid_elegans/output/newlevers/*.json ; logs: $LOGDIR/" | tee -a "$MANIFEST"
