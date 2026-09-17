# Fresh replication: conditional flow and progressive bridge SMC

**Status: running. No scientific replication verdict has been reached.**

This experiment was requested on 11 September 2026 to independently retrain the selected flow models, resample lag-effect matrices, compare with published SBTG on Cook/Randi, and test numerical and lag reliability without shortcuts.

## Read

- `PROTOCOL.md` — frozen scientific design, claim boundaries and completion criteria.
- `ANALYSIS_IMPLEMENTATION.md` — prospectively fixed evaluation details.
- `status.json` — current worker and progress, updated by the live dispatcher.
- `task_plan.json` — all 332 top-level tasks.
- `HANDOFF.md` — instructions for continued supervision and recovery.
- `REPORT.md` — final report, created only after required runs and analyses finish.

## What is running

Two separate cohorts: the historical 20-row/80-class SBTG dataset for comparator compatibility, and the corrected 17-recording/54-class OH16230 head cohort. Five held-out recording folds and three generator seeds give 30 primary models. No old flow checkpoint is reused.

The full sampling grid covers four source-history lags, six forecast horizons, seven response channels and all five event phases. Independent MC repeats, N4096 direct sampling, endpoint-timing-matched comparisons, 1,440 diagnostic cases, and factual predictive controls test reliability. Validation-triggered longer-training sensitivities preserve the original primary estimates.

The program is dispatched serially under `caffeinate`, using one MPS worker at a time. An app heartbeat supervises the job every 30 minutes and continues through failures and final reporting. CPU/GPU work may take many hours or days; time and memory constraints never silently reduce the scientific grid.

## Preserved provenance

All original workspace files remain untouched. `source/` and `inputs/` are copied, hashed snapshots; `snapshot_manifest.json` authenticates them. `execution_manifest.json` freezes the primary training/sampling workers and `analysis_manifest.json` freezes the new analysis workers. Each completed fit and response archive has a hash-bound receipt. A failed import was documented in `validation/incident_001.json` and safely resumed before fitting that task.

The published SBTG matrices and both atlas inputs remain excluded from model/seed/lag selection. The historical cache matches the released traces after explicit neuron-name alignment, but its donor copying and head/tail pseudo-pairing still limit biological interpretation. The current flow matrix is a finite model-relative contrast; shrinking-gap tests are required before calling it a local derivative approximation.

## Validation completed at launch

- Copied training/sampling source suite: 309 passed, 2 skipped.
- Independent multivariate Gaussian path oracle: prespecified gates passed.
- Standalone replication tests: 11 passed, including archive aggregation, direction, bootstrap and exact max-T checks.

These checks validate tested implementations; they do not establish a positive empirical result. Logs and machine-readable receipts are in this directory and `validation/`.
