# Ongoing replication: operator handoff

User authorized rigorous retraining/resampling for as many local-machine hours as necessary, in this new subdirectory. The task is NOT complete until results, diagnostics, scientific adjudication, and rendered report review finish. Do not replace it with a plan or partial favorable metrics. No agents were authorized; do not spawn them.

## Live state and monitoring

- Workspace `/Users/vik/Developer/new_sbtg_neuro/replication_20260911`.
- Python `/Users/vik/Developer/new_sbtg_neuro/.venv/bin/python` (Python 3.14, Torch 2.13, MPS).
- Read `status.json`, `dispatcher.log`, the status-listed worker log, and `task_plan.json`.
- Durable serial controller `dispatcher.py` launched under `caffeinate -i`. It owns an exclusive `pipeline.lock` and starts exactly one worker at a time. Status refreshed every 20 s. PID is in status; original launch + resume receipts are retained. Do not infer a dead process merely from a long training/sampling step. Verify with `ps` and log activity.
- App heartbeat id `flow-smc-replication`, every 30 minutes in this same task, supervises continuation. Keep unchanged healthy states quiet; meaningful milestones, failures requiring user attention and final findings may notify. Stop/pause heartbeat after genuinely complete delivery.
- Mac M4 has 16 GB RAM; initial free disk approximately 27 GB. Dispatcher preserves 5 GB minimum free space. Never launch a second GPU job or delete research data to make space.

## Design and status at setup

Read `PROTOCOL.md` and `ANALYSIS_IMPLEMENTATION.md` in full. Snapshot source/input copies are byte-pinned. Original parent results are absent, but raw recordings, historical cache, and full published SBTG release were found. Historical cache exactly matches released prepared traces after neuron-name permutation and float32 conversion, including missingness. Do not mistake differing column orders for different data.

Source tests: 309 passed, 2 skipped. Tests rerun in `source/` so imports actually use the snapshot (`snapshot_tests.log`); an earlier parent-cwd run is only ancillary. Independent Gaussian oracle: lags 0/1/4/16, N32/N128/N1024, 16 MC repeats, multivariate anchor/source-window precision solution; prespecified gates passed. Standalone replication tests cover independent bootstrap, max-T, direction, normalization, Gaussian direct estimate, and full archive aggregation; see current test log/receipt.

Thirty new primary fits: both historical80 and clean54, 5 recording folds × 3 seeds. Historical15 fits take roughly 30–50 seconds each. Training and sampling use copied existing algorithms without changing their equations. New independently authored evaluation code checks the estimand, orientation and uncertainty. No pretrained flow checkpoint is used. Published SBTG is the frozen comparator, not retrained here; no atlas score is a selection criterion.

The dispatcher has 332 top-level tasks: 30 training; 120 primary sampling archives; 40 independent MC repeats; 40 N4096 direct archives; 60 endpoint-timing-matched archives; 10 diagnostic jobs; 30 predictive validation jobs; analysis and reporting. Diagnostic jobs each emit many cases: 720 compact archives per cohort (1440 total), including particle, derivative, solver and zero-query controls. All raw paths are resume-validated.

`analyze.py` first runs the prespecified `extensions.py`: fresh max200/patience20 retraining and full-grid sensitivity resampling only where a primary fit exhausted its budget with a recent best epoch. It executes before reference scoring, preserves the primary estimates and records exactly which checkpoints changed. If a max200 extension is itself under-converged, report unresolved convergence; do not silently tune using external scores.

## Important implementation details

- `execution_manifest.json` freezes protocol/settings/source identity and primary worker bytes. Per-fit launch + completion receipts bind full training options/checkpoints; completed fits survive dispatcher restart.
- `analysis_manifest.json` freezes diagnostics/prediction/analysis/report code before response evaluation. If a real bug needs fixing, preserve the old manifest, logs and affected outputs, document a dated implementation amendment, determine affected stages and rerun into a separate lineage or explicitly versioned replacement. Never just rewrite receipts to hide changed results.
- Imports must resolve from `source/`. Do not name a root script `pipeline.py` (shadows SBTG pipeline) or `statistics.py` (shadows Python statistics). Incident 001 records the latter transient collision: the sixth training task failed during import before fitting; renaming fixed it. Completed fits were not changed.
- `train.py` catches source-runner swallowed exceptions by inspecting the final trial record. Summaries are cumulative from completed receipts. Failed trial logs remain.
- Sampling gap normalization has an exact-reproduction floor and an additional strict signed-gap view. Derivative diagnostics never use the .10 floor as a small-gap derivative. W1 is unsigned and is not a signed derivative.
- Flow nominal lag ell with forecast horizon h has last-source-frame-to-endpoint separation ell+h. Matched-time sensitivity uses ell0/7+h1 against published SBTG1/8. Four-frame source averaging and repair conditioning still differ from SBTG.
- Reference metrics use a newly written orientation/alignment/mask/bootstrap implementation. Equal source bootstrap is conditional on fitted matrices, not animal/refit uncertainty. Historical imputation/pseudo-pairing means its recording-row inference is descriptive.
- Joint sign-flip max-T uses all two-sided recording sign orbits and fixed strong all-lag support, streamed to control memory. It may take substantial CPU time; lack of a quick result is not a failure. The new inference family uses baseline and onset-minus-baseline mean/log-SD effects and lag contrasts; do not claim byte-identical E29 active-minus-baseline inference.
- Diagnostics CSV is streamed/compressed, then reduced with categorical `observed=True` groups to avoid a cartesian-product explosion. Check all 720 files per cohort exist.
- Direct N16384 is a qualified numerical reference only where all three repeats are valid with ESS≥12 and normalized SD≤.05; unresolved cells remain unresolved. Never infer a universal SMC advantage from matched particle counts. Actual model evaluations and wall time are recorded.

## Remaining work during long execution

1. Follow actual logs and resolve any failures conservatively. All primary sampling/predictive/analysis paths need real execution; unit tests are not an end-to-end certification.
2. Once first full sample archive exists, check shapes, receipt validation, timing, heldouts and sampler diagnostics; estimate runtime/storage from observed work. Do not inspect atlas correspondence before required raw freeze.
3. If a worker fails, inspect its log; address cause within the replication directory, retain failure receipt, and resume without duplicating live work. Before a restart ensure parent and child are no longer running and the lock is free. A standard restart is `caffeinate -i <absolute-python> <absolute-dispatcher.py>` through a detached subprocess with log append; record a new launch receipt.
4. Finish the analysis including required extensions; build all figures and REPORT.md. `validation/final.json` validates artifact completion and numerical schemas, not positive science. Its visual review is explicitly pending until the agent inspects rendered figures.
5. Inspect all 14 generated PNG figures with image tools, check source tables/numbers and conclusions, and fix/report genuine visualization defects. Report completed / failed / unresolved scientific findings separately; do not call a running study successful.
6. Deliver concise results with links to report, key comparison/lag figures and tables. Preserve negative results, missing-support counts, finite-contrast semantics, historical provenance caveats, and the absence of independent new animals.
7. Stop/pause the heartbeat only after the verified final report is delivered. No goal tool was created; do not invent a completed goal.
