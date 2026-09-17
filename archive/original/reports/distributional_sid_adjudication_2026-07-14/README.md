# Distributional SID failure adjudication

The main deliverable is `main.pdf`. It combines the original E0--E11 result
landscape with the post-hoc E6/E8/E9 failure analysis, complete baseline audit,
updated method recommendations, and next experiments.

Key machine-readable artifacts:

- `results_snapshot.json`: quoted headline metrics and revised gate decisions.
- `tables/validation_audit.csv`: row counts, failure/duplicate checks, and
  SHA-256 hashes for every adjudication seed table.
- `tables/baseline_coverage.csv`: same-target and alternative-estimand baseline
  inventory.
- `tables/e6_adjudication_summary.csv`, `e8_adjudication_summary.csv`, and
  `e9_adjudication_summary.csv`: aggregated evidence behind the new figures.
- `tables/revised_gate_matrix.csv`: final E0--E11 synthesis.

Regenerate the figures and tables from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=distributional_sid/src \
  .venv/bin/python \
  reports/distributional_sid_adjudication_2026-07-14/make_report.py
```

Compile the report:

```bash
python3 /Users/vik/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py \
  /Users/vik/Developer/new_sbtg_neuro/reports/distributional_sid_adjudication_2026-07-14/main.tex
```

The added studies are explicitly post-hoc adjudications on unused seeds. They
diagnose which original failures are repairable; they are not relabeled as
preregistered confirmations.
