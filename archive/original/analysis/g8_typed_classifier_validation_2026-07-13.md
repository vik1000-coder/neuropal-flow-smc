# Validation report: G8 typed finite classifier

## Overall assessment: Share with caveats

The experiment is valid developmental evidence that a contrast-aligned learner can recover the G8 fourth-order effect that the tested full-law generators miss. It is not confirmatory evidence of generalization beyond the supported binary contrast or beyond four independently parameterized G8 systems.

## Methodology review

- Question and estimand match: balanced classification of responses from control `+1` versus `-1` estimates the bounded central signed-measure witness `(2 eta - 1) / delta`; the reported typed effect is the held-out mixture expectation `E_M[phi W]`.
- Train, validation, and test histories use disjoint deterministic seed streams. Hyperparameter selection and neural early stopping use validation data only.
- The control label is not included in any classifier feature. The full-path model receives the baseline history with its control coordinate removed plus the response; the motif models receive residual motif coordinates; the typed classifier receives only the analytic quartic feature.
- Every split is exactly class balanced. Each test cell contains 10,000 independent draws per side.
- The analytic effect formula agrees with the independent empirical difference: median relative error is 0.70%, maximum 1.90%, and every empirical discrepancy is within 1.39 Monte Carlo standard errors.
- Population-negative-control behavior is reproduced: linear and quadratic motif classifiers have median AUC 0.499 and 0.508 and recover essentially none of the typed effect.
- Positive-control behavior is coherent: the exact Bayes witness has median AUC 0.768 and 0.33% typed-effect error; motif MLP AUC is 0.768 and its median effect error is 1.13%.

## Calculation spot checks

- Eight expected generator/data cells are present and successful: four generator seeds crossed with two data seeds.
- Each cell contains all eight declared estimators. No output row is missing.
- Median errors recomputed directly from the case JSON files agree with `estimator_summary.csv`.
- At a matched evaluation budget of 512 draws per side, plug-in Monte Carlo standard errors are estimated at 1.96% of truth for the motif MLP, 2.36% for the raw path/history MLP, and 2.72% for the typed-quartic classifier. These remain far below the 90--100% attenuation of the tested generative samplers, but are extrapolated standard errors rather than a separate refit.
- Errors for the learned classifiers exceed pure evaluation Monte Carlo error in several cells, so their small residual bias is real and is not described as exact recovery.

## Issues and required caveats

1. **Medium -- developmental system count.** There are eight cells but only four independent generator parameterizations; the two data seeds within a generator are not independent systems.
2. **Medium -- favorable representation.** The motif MLP and typed-quartic classifier use oracle residual centering and/or an analytic motif basis. The raw path-plus-history MLP removes this basis advantage and still performs well (3.05% median error), but its maximum error is 10.1%.
3. **Medium -- supported intervention only.** The primary contrast is the randomized, well-supported `-1` versus `+1` comparison. It does not establish accurate infinitesimal derivatives at control zero or extrapolation outside observed support.
4. **Low -- unequal evaluation draws versus sampler audit.** Classifier effects are integrated with 10,000 held-out observations per side, while the generative readout used 512 generated draws per side per anchor. Plug-in 512-draw uncertainty remains small, but a final paper should run a common-budget evaluation.
5. **Low -- model selection.** Logistic regularization and MLP stopping are chosen on a fixed validation split. A confirmatory run should preregister the grid and add independent systems rather than expand it after seeing results.

## Reproducible sources

- Experiment: `analysis/g8_typed_classifier_benchmark_2026-07-13.py`
- Cell records: `analysis/g8_typed_classifier_benchmark_2026-07-13/cases/`
- Summary: `analysis/g8_typed_classifier_benchmark_2026-07-13/estimator_summary.csv`
- Generator and oracle readout: `history_tangent_benchmark/src/history_tangent_benchmark/dgps.py` and `analysis/g8_corrected_readout_benchmark_2026-07-13.py`
