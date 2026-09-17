# Unified SID synthesis (2026-07-12)

- `main.tex`: standalone LaTeX source.
- `main.pdf`: PDF produced directly by the LaTeX build.
- `/output/pdf/SID_Unified_Synthesis_2026-07-12.pdf`: stable delivery copy.

The report synthesizes the non-latent SID, SBTG, point-history, finite-contrast,
conditional-diffusion, and changepoint work. It is scoped to theory and synthetic
data. The latent mechanistic state-space model and real neural-data performance
are intentionally excluded.

The July 12 expanded run is incorporated in Section 12. Its source artifacts are:

- `neuromod_benchmark/outputs/rethink_suite_20260712_v2/` (24-neuron E1--E5
  run, including the representative neural/score panel; 840 successful fits);
- `neuromod_benchmark/outputs/rethink_suite_20260712_n32_fast/` (32-neuron
  full fast-panel confirmation; 672 successful fits);
- `neuromod_benchmark/outputs/rethink_suite_20260712_sid_stability/` (432 SID
  stability fits plus 18 matched ridge fits).

The principal evidence paths are listed in Appendix A of the report. The three
embedded figures remain sourced from
`analysis/synthetic_methods_adjudication/` so that the LaTeX source stays tied to
the executed adjudication outputs.

To rebuild with the bundled LaTeX skill:

```sh
cd /Users/vik/.codex/plugins/cache/openai-bundled/latex/0.2.4
python3 scripts/compile_latex.py \
  /Users/vik/Developer/new_sbtg_neuro/reports/sid_synthesis_2026-07-12/main.tex
```
