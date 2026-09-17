# Validation — 31 August 2026

Completed against the local source files and the served dashboard at port 18781. This validates the presentation and its source bindings, not the scientific validity of the fitted model.

## Automated checks

`./.venv/bin/python -m pytest dashboard_v2/test_dashboard.py -q`: **15 passed**, final run 3.69 seconds.

- Exact means and both CI endpoints for both estimators at asymmetric source/target pairs, multiple outcomes and contexts, every archived lag/horizon. Checked against canonical NPZ arrays with target-row/source-column indexing.
- Progressive worm means agree with the canonical summary within floating-point tolerance. Direct estimates do not inherit progressive worm values, corrected tests or controls.
- Unsigned W1, untested onset contrasts, self-pair exclusions and unavailable endpoint-sample selections retain their correct interpretation.
- All 24 control outcomes reproduce the saved mean paired excess. Computation uses per-worm absolute observed effects before subtracting that worm's control magnitude.
- Endpoint draws match the archived samples after the correct inverse scaling; the factual marker comes from `cut + horizon`.
- Observed bands reproduce the actual 10th/90th percentiles and median of the 17 traces. Missing observations remain missing; JSON contains null rather than NaN/Infinity.
- Saved support gates remain exact. HTTP checks cover valid JSON, invalid coordinates, duplicate query parameters, unknown timing, denied static paths and rejected nonlocal Host headers.

`./dashboard_v2/run.sh --validate`: all **120 input entries** verified, including the 114 entries in the original ledgers and six additional current-input pins. Original atlas, evidence, calibration and explorer checksums remain unchanged.

`node --check dashboard_v2/app.js`: passed.

Independent source review also compared all ten saved seed/fold checkpoint scalers with the reconstruction; they matched exactly. Control summaries and source-data semantics were reviewed independently from the UI implementation.

## Browser checks

Checked in the Codex in-app browser at its normal 830 px viewport:

- Initial effect curve, exact estimate, source support, corrected evidence and paired-control plot.
- Forecast/history-lag switching; chart point positions use true numeric time spacing.
- Direct-estimator mode and estimator overlay, including explicit unavailable individual-worm/control states.
- W1 onset-minus-baseline displays “onset distance minus baseline distance.”
- All 54 target rows load on expansion; selecting a target updates the pair and values.
- Recorded chemical events, all-worm plots, individual-worm display and corresponding legend.
- Saved predictive histograms, factual outcome marker, and event synchronization between the recorded and predictive views.
- External-reference table loads with the original scope and timing labels.
- Reference AUROC and Spearman plots load from the selected reference, keep distinct panels separate, expose exact point metadata on hover/focus, and show explicit empty states when a metric was not archived.
- No horizontal page overflow at the checked viewport and no warning/error console entries during the checked flows.

Reviewed request versioning and loading states so old requests cannot replace a newer selection. Fixed stale target/prediction states, cleared captions during reloads and kept selectors synchronized before fetching. Narrower responsive layouts are provided in CSS but were not separately browser-tested.

## Loading and limits

The initial static app is approximately **59 KB** total (HTML 9,682 B; JS 36,066 B; CSS 13,024 B), versus approximately 37 MiB for the original embedded explorer HTML. The initial metadata and selected edge response are about 5.5 KB and 21 KB. This is a payload comparison, not a controlled user-perceived speed benchmark.

A local smoke check measured the first recorded-signal load at about 2.1 seconds, including cohort initialization; the initial selected-edge request was about 34 ms. Larger banks are opened on demand and caches are bounded. Startup checksum verification is separate from browser request timing.

No models were retrained or resampled. No new measurement-error model, refit uncertainty, significance tests, chemical comparisons or causal evidence were created. Predictive sample banks are limited to eight baseline coordinates, and individual-worm effect arrays are progressive-only. These limits are visible in the interface rather than filled with inferred evidence.

## Portable bundle

`dashboard_v2/build_portable.py --force` produced `dist/neural_atlas_v2_portable` (551 MiB allocated; 119 verified files) and `dist/neural_atlas_v2_portable.zip` (564,001,259 bytes). The archive SHA-256 is `015f5d9b297b52e2f9bfd93bc53ad30d1ef1e00a3463f01ff7f5bf3406fc8661`.

The standalone `verify_bundle.py` passed: all internal checksums, 54 neurons, 17 worms, atlas/worm arrays, recorded signals, saved samples, controls and 210 reference rows loaded. A standalone server on port 18782 returned HTTP 200 for the page, JavaScript, metadata, selected edge and references; metadata identified portable mode and hid the unavailable original-explorer link.

Source-to-bundle comparison passed for four asymmetric selections spanning outcomes and contexts, both estimators' means/CI/support arrays, progressive worm arrays, every one of the 24 control rows, all three recorded chemicals, both model seeds and both repeats at a saved prediction coordinate, and the full external-reference object. The slim bundle matched exact arrays; only recomputed display quantiles allow `1e-7` tolerance because the exported cohort is float32, like the source cohort.
