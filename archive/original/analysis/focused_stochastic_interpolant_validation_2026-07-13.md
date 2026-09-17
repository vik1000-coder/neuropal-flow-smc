# Focused stochastic-interpolant validation

Status: **complete**. All 32 expected fits succeeded (2 generators × 2 generator seeds × 2 data seeds × 2 model seeds × 2 schedules).

Protocol controls:

- Exact core train/validation/test seeds and scalers were reused.
- Each interpolant used its matching prefit, frozen heteroscedastic-Gaussian anchor.
- The only fitted-method factor was `beta(t)=t` versus `beta(t)=t^2`.
- Epsilon was fixed at 1.0; 128 Euler–Maruyama steps were primary; 64 and 256 were numerical-resolution checks.
- Training used 8,000 cases, validation used 1,600, and checkpoint selection used only native validation loss.
- All computation used one CPU thread.

Primary results (median over eight generator/data/model cells):

| Law | Schedule | Fair energy | Typed NRMSE (128) | Typed slope (128) |
|---|---:|---:|---:|---:|
| G1 Gaussian location | linear | 1.5128 | 0.3380 | 0.9078 |
| G1 Gaussian location | squared | 1.5129 | 0.3195 | 0.8642 |
| G8 fourth-order path | linear | 0.7167 | 0.9925 | 0.0075 |
| G8 fourth-order path | squared | 0.7102 | 0.9994 | 0.0007 |

Interpretation:

- The sampler is not generally broken: both schedules recover a substantial G1 finite mean effect.
- The G8 failure is stable at 64/128/256 steps. It is not an Euler–Maruyama resolution artifact.
- Squared beta improves G8 fair energy relative to linear beta but makes the fourth-order SID effect even closer to zero. Native law fit and contrast fidelity therefore rank the schedules differently.
- Neither schedule beats the frozen G8 flow-matching fair-energy median (0.7035), and neither restores the missing history-controlled mode mass.
- Claims remain developmental: only two independent generated systems were used, though data and model seeds were crossed.

Artifacts:

- `analysis/focused_stochastic_interpolant_2026-07-13/case_metrics.csv`
- `analysis/focused_stochastic_interpolant_2026-07-13/schedule_summary.csv`
- `analysis/focused_stochastic_interpolant_2026-07-13/run_manifest.json`
