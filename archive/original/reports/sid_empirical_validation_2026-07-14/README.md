# SID empirical validation package

This directory is the human-readable entry point for the 14 July 2026
score-derivative SID validation.

## Primary deliverables

- main.pdf: rendered report with methods, gates, local/finite leaderboards,
  dynamic lag results, robustness slices, failure analysis, decision table, and
  reproducibility ledger.
- main.tex: report source.
- generated_results.tex and tables/: tables generated directly from validated
  seed-level summaries.

## Verification snapshot

- 1,290 completed experimental cases: 300 in Tier 1 and 990 in Tier 2,
  including the separately frozen post-hoc diagnostic.
- 180,480 packaged metric rows: 51,000 in Tier 1 and 129,480 in Tier 2.
- 30 paired top-level DGP seeds in every registered primary cell.
- Zero failed cases, unexpected missing values, nonfinite packaged values, or
  raw-array hash mismatches in either validation report.
- Estimator test suite: 10 passed.
- `main.pdf` is 20 US-letter pages; SHA-256:
  `15cb06760355f9f2af7413c0d5c913c9938ec3229d8d561a2eb4dd44e6dcd2d7`.

## Machine-readable results

Tier 1:

- ../../empirical_sid/runs/sid_tier1_primary_20260714_v4/results/seed_level.parquet
- ../../empirical_sid/runs/sid_tier1_primary_20260714_v4/results/summary.csv
- ../../empirical_sid/runs/sid_tier1_primary_20260714_v4/results/validation.json

Tier 2 and the separately labeled post-hoc diagnostic:

- ../../empirical_sid/runs/sid_tier1_primary_20260714_v4/results/tier2/seed_level.parquet
- ../../empirical_sid/runs/sid_tier1_primary_20260714_v4/results/tier2/summary.csv
- ../../empirical_sid/runs/sid_tier1_primary_20260714_v4/results/tier2/validation.json

Claim-to-evidence map:

- ../../empirical_sid/runs/sid_tier1_primary_20260714_v4/results/claims_registry.csv

Raw arrays, per-case metric CSVs, manifests, hashes, configurations, source
snapshots, preregistration amendments, and the legacy compatibility audit remain
under ../../empirical_sid/runs/ and ../../history_tangent_benchmark/results/.

## Interpretation status

The appropriate label is **diagnostic empirical validation**. The implemented
confirmatory cells use 30 top-level DGP seeds and keep local and finite estimands
separate, but the broad runbook is not universally complete: dense unknown
response geometry, full mechanism-confusion matrices, flexible MDN/flow baselines,
neural score backends, and several ablation/stress ladders remain future work.

The post-hoc fixed-amplitude learning curve is explicitly exploratory and cannot
upgrade a confirmatory recovery or advantage claim.
