# Stable SID report chart map

## G8 score visibility

- Question: can a denoising noise scale simultaneously expose history-controlled mode weights to an outcome score and preserve the original finite law contrast?
- Takeaway: no single scale does both; response-score visibility has a narrow intermediate-noise peak while Jensen--Shannon information and the history tangent decay.
- Family/type: uncertainty-and-benchmark, two-panel ordered line chart.
- Grain: nine declared noise scales; medians over four independent G8 systems, 5,000 exact samples per control side and system.
- Fields: retained Jensen--Shannon divergence, retained history-tangent RMS, relative outcome-score gap.
- Palette: hard two-root cap (blue/orange) plus line style and marker-shape distinctions; white background and neutral grid.
- Surface/output: static PDF and PNG embedded full-width in the LaTeX technical report.
- Source: `analysis/g8_score_visibility_audit_2026-07-13/noise_summary.csv`.
- QA: inspect exported PNG, then inspect the compiled PDF page in final report context.
