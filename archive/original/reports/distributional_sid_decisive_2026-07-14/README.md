# Distributional SID decisive synthetic validation

Primary artifact: `main.pdf`.

The report includes the completed dependence, measurement, staged-geometry,
normalized spline-flow, sampling-budget, and analytic full-covariance MDN
extensions. The final E10 comparison covers seven reference families, ten DGP
seeds, three nuisance/model initializations, three MDN component counts, two
flow depths, and sampling budgets 64/256/1024.

Regenerate figures and tables from frozen run outputs:

```bash
.venv/bin/python reports/distributional_sid_decisive_2026-07-14/make_report.py
```

Then compile `main.tex`. The selected source runs are listed in
`tables/artifact_ledger.csv`; `results_snapshot.json` records the overall
decision and gate matrix. The principal completion tables are
`tables/e3_dependence_summary.csv`, `tables/e8_measurement_summary.csv`,
`tables/e9_geometry_summary.csv`, `tables/e10_primary_nrmse.csv`, and
`tables/e10_paired_ratios.csv`. Runs with suffixes such as `_v1` or `_v2` are
retained packaging/development attempts and are not used in the report.
