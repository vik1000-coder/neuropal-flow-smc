# Exact bounded-energy reevaluation validation

Status: **complete**. All 64 frozen bounded-energy checkpoints were loaded and reevaluated with exact iid rejection sampling; no case failed.

Validity checks:

- The target is the frozen bounded tilt `q(y|h) exp(B tanh f(y,h)) / Z(h)` with `B=1`.
- Proposals are accepted independently under `log U <= log_tilt - B`.
- The theoretical acceptance lower bound is `exp(-2B)=0.1353`; observed per-case rates fluctuate around or above it, with median rates 0.136–0.431 by generator.
- The largest materialized proposal batch was 4,092, below the 4,096 configured cap.
- Energy scores use 96 held-out histories and 64 iid draws per history, matching the frozen core evaluation size and seeds.

Median exact fair-energy scores:

| Law | Exact bounded energy | Gaussian NLL | Frozen best |
|---|---:|---:|---:|
| G1 | 1.5073 | 1.5065 | 1.5023 |
| G2 | 1.5042 | 1.5017 | 1.4947 |
| G3 | 1.7750 | 1.7981 | 1.7806 |
| G4 | 2.3007 | 2.3344 | 2.2746 |
| G5 | 1.2087 | 1.2130 | 1.1467 |
| G6 | 2.3818 | 2.3725 | 2.3685 |
| G7 | 2.7783 | 2.7559 | 2.7297 |
| G8 | 0.7150 | 0.7162 | 0.7035 |

The frozen-best column is descriptive and is not a common paired winner test. Exact bounded energy is competitive and sometimes improves Gaussian, but does not consistently beat the strongest flexible baseline. On G8 its median corrected quartic NRMSE is 1.0001 and slope is −0.00008, so exact sampling repairs the proper-score contract without repairing the missing history-controlled mass effect.

Artifacts:

- `analysis/exact_bounded_energy_reevaluation_2026-07-13/case_metrics.csv`
- `analysis/exact_bounded_energy_reevaluation_2026-07-13/generator_summary.csv`
- `analysis/exact_bounded_energy_reevaluation_2026-07-13/run_manifest.json`
