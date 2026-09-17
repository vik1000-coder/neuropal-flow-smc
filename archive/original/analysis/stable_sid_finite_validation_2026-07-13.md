# Validation report: frozen stable SID finite panel

## Overall assessment: Share with caveats

The finite panel is complete and suitable for developmental method adjudication. The Python process exited nonzero only after all four atomic artifacts were written and the manifest was marked complete, because a final disk-safety check found 3.87 GiB free versus the configured 4.00 GiB floor. This is a resource-audit event, not an incomplete model run. It must remain recorded in the reproducibility notes.

## Artifact and calculation checks

- `run_manifest.json` reports status `complete`, 212,832 metric rows, 4,704 diagnostic rows, eight generators, three perturbation scales, and clean/noisy laws kept separate.
- `finite_contrast_metrics.csv` contains exactly 212,832 rows; every row has status `ok`; every metric value is finite; the row count matches the manifest.
- The panel contains 784 learned checkpoint cases plus one collapsed oracle case identifier, 14 source names, seven finite methods, and 24 metric identifiers.
- The base manifest and source-tree hashes match the frozen core run. The digest-reconciliation wrapper allowed only pre-audited non-executable report/config files created after the core launch; the original core manifest was not modified.
- Independent SHA-256 hashes were recorded after completion:
  - metrics: `1915718c10b3b5a4a452d5c7d3811c6cafae663d16866e6843440ed6e6f01878`
  - diagnostics: `b9a20e523170d48aa7f91944a16075d01f3931f931e563c121e907e60c012922`
  - manifest: `0f93a34d9ca05de722627c92855d625e0a83ad93e0156826ad3d4a7b51ad0ba6`
- The frozen report's sentence headed `delta=0.1` returns no rows because the actual preregistered scales are 0.05, 0.12, and 0.25. This is a report-template defect only; the full metric file is intact and the adjudication uses 0.12 explicitly.

## Issues and required caveats

1. **Medium -- only two independent systems.** The eight learned cells per generator/model pair cross two generator seeds, two data seeds, and two model seeds. They are not eight independent parameterized systems.
2. **Medium -- route selection is exploratory.** Choosing the lowest NRMSE across 14 sources and several adapters creates winner's bias. Values are developmental medians, not confirmatory error estimates.
3. **Medium -- one adapter seed.** The finite classifier/Riesz training uses adapter seed 7001. Source/data/model variability is crossed, but adapter-initialization variability is not.
4. **Medium -- fixed dictionary limitation.** The response dictionary ends at cubic order. Its G8 channel NRMSE near one is a designed negative control and cannot adjudicate the analytically quartic G8 effect. The separate corrected quartic panel supplies that result.
5. **Low -- duplicated oracle exact rows.** There are 428 exact duplicate CSV rows, all under the collapsed `case_id=oracle`; learned-model rows have no duplicates. They arise because replicate identity is not retained for some deterministic oracle metrics. They do not change reported medians, but oracle `n` and quartiles are not interpreted as independent-system uncertainty.
6. **Low -- finite-sample route variance.** Oracle exact-witness channel NRMSE is nonzero because the typed moment is estimated by finite Monte Carlo. Direct common-random-number sample contrasts can have lower NRMSE than the exact-witness plug-in due to variance reduction; this is not evidence of beating the oracle law.
7. **Low -- output safety exit.** The post-write disk-floor exception means orchestration should distinguish completed artifacts from a failed process exit. Future long runs should reserve the output size above the disk floor or checkpoint intermediate tables.

## Interpretation checks

- `delta` changes the estimand, not merely numerical precision. Improvements at 0.25 are evidence for a stronger supported finite contrast and cannot be described as better recovery of the infinitesimal tangent.
- Clean and noisy channel results target different laws and are not pooled.
- Witness NRMSE and typed-channel NRMSE answer different questions. A method may estimate a selected functional well while failing to reconstruct the full witness.
- The bounded-energy sampler remains excluded from primary sampling/proper-score claims because its current finite-pool SIR draws are dependent.

## Reproducible sources

- Frozen finite artifacts: `history_tangent_benchmark/results/stable_sid_finite_20260713/`
- Digest audit: `analysis/stable_sid_digest_reconciliation_2026-07-13.json`
- Reconciled launcher: `analysis/run_finite_with_digest_reconciliation_2026-07-13.py`
- Adjudication outputs: `analysis/stable_sid_adjudication_2026-07-13-v2/`
