# G8 endpoint-expert decomposition validation

Status: **complete**. All 80 planned cells succeeded (4 systems × 2 data seeds × 2 model seeds × 5 backend families).

Design controls:

- Fixed total training budget: 8,000 observations, split as 4,000 per supported endpoint expert.
- Validation: 1,000 observations per endpoint expert.
- The declared ±1 control is removed from expert inputs. At the evaluated endpoints, a deterministic normalized gate selects the relevant expert.
- Both experts use the same architecture, scaler, initialization seed, sample budget, and common sampling seed.
- Primary effect uses 8 baseline histories, 512 draws per expert, the exact quartic readout, and the analytic G8 target.
- Fair energy uses 96 held-out histories per endpoint and 64 samples per history.

Median results over 16 cells per backend:

| Endpoint expert | Fair energy | Quartic NRMSE | Quartic slope |
|---|---:|---:|---:|
| EDM | 0.7036 | 0.6649 | 0.3363 |
| affine flow | 0.7045 | 0.7505 | 0.2509 |
| autoregressive MDN | 0.7142 | 0.8231 | 0.1794 |
| flow matching | 0.7028 | 0.9655 | 0.0346 |
| Gaussian NLL | 0.7180 | 0.9999 | 0.0001 |

Relative to the shared balanced-binary fits, explicit endpoint decomposition materially increases EDM (about 0.013→0.336), affine-flow (0.046→0.251), MDN (0.095→0.179), and flow-matching (0.001→0.035) slopes. However, every full-law expert remains substantially worse than the direct finite classifier and projected empirical contrast. The experiment supports a global-gate component but rejects deterministic endpoint gating as a sufficient repair at this budget.

Limitations:

- The gate is evaluated only at the supported endpoints; interior interpolation is not tested.
- Each expert sees half as many observations as the original shared model, although the total method budget is held fixed.
- The construction is developmental and uses four system seeds, not a confirmatory replication.
