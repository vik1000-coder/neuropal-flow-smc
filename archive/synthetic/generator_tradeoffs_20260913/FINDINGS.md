# What changed when we changed the generator?

All192 planned fit/evaluation tasks have completed; independent algebra/data/receipt audits passed. This is a new fixed-configuration synthetic comparison. It is not the missing original NeuroPAL study. It uses4 laws,3 independent data realizations/law and2 neural initializations/dataset. Seed ranges and means below are descriptive, not simultaneous confidence statements or optimized-family comparisons.

## Average predictive accuracy by setting

Lowest mean energy configurations and runner-up configurations are shown for every law/region. All candidates see the same data and evaluation histories. Paired differences compare the listed leader with flow over the same three datasets. Selection of the lowest observed mean is descriptive and will be optimistic; no new model selection or fitting follows these rankings.

| law                       | region       | lowest_mean_energy_family   |   energy | runner_up        |   runner_up_energy |   flow_energy |   paired_delta_min |   paired_delta_max |
|:--------------------------|:-------------|:----------------------------|---------:|:-----------------|-------------------:|--------------:|-------------------:|-------------------:|
| bimodal                   | boundary     | mdn4                        | 0.43554  | transformer_mdn4 |           0.435756 |      0.44235  |       -0.0111084   |       -0.0030574   |
| bimodal                   | central      | mdn4                        | 0.57368  | flow             |           0.574439 |      0.574439 |       -0.00323465  |        0.0019624   |
| bimodal                   | extrapolated | gaussian_rank2              | 0.561223 | student_rank2    |           0.563514 |      0.642872 |       -0.140601    |       -0.0264925   |
| heavy_tailed              | boundary     | student_rank2               | 0.262563 | transformer_mdn4 |           0.263663 |      0.271636 |       -0.0109676   |       -0.0075703   |
| heavy_tailed              | central      | flow                        | 0.26102  | student_rank2    |           0.262219 |      0.26102  |        0           |        0           |
| heavy_tailed              | extrapolated | gaussian_rank2              | 0.562452 | transformer_mdn4 |           0.574826 |      0.646145 |       -0.165794    |       -0.0288705   |
| linear_gaussian           | boundary     | ridge                       | 0.280604 | gaussian_rank2   |           0.282028 |      0.282923 |       -0.00257333  |       -0.00191004  |
| linear_gaussian           | central      | ridge                       | 0.279331 | gaussian_rank2   |           0.281471 |      0.28156  |       -0.00250573  |       -0.00205031  |
| linear_gaussian           | extrapolated | ridge                       | 0.282449 | flow             |           0.559098 |      0.559098 |       -0.356747    |       -0.177856    |
| nonlinear_heteroskedastic | boundary     | transformer_mdn4            | 0.318424 | gaussian_rank2   |           0.319023 |      0.325011 |       -0.00941314  |       -0.00393716  |
| nonlinear_heteroskedastic | central      | student_rank2               | 0.282324 | flow             |           0.282373 |      0.282373 |       -0.000114631 |       -7.54231e-06 |
| nonlinear_heteroskedastic | extrapolated | gaussian_rank2              | 0.581545 | gaussian_diag    |           0.582259 |      0.648718 |       -0.166645    |        0.0161457   |

## Conditional-effect accuracy: common and rare queries

This comparison holds the direct reference estimator fixed. Only families whose reference qualifies on all3 datasets and both neural initializations enter each row's ranking. The number of eligible families is shown; rows with few eligible models cannot adjudicate the excluded models. See paired_effect_differences.csv and effects_dataset_means.csv for all estimates, exclusions and shared masks. 'Rare' denotes the predeclared tail-directed soft query; actual support varies with history and is retained in effects_raw.csv.

| law                       | region       | event   |   fully_qualified_families | lowest_mean_error_family   |        error | runner_up      |
|:--------------------------|:-------------|:--------|---------------------------:|:---------------------------|-------------:|:---------------|
| bimodal                   | boundary     | common  |                          9 | ridge                      |   0.0109717  | gaussian_rank2 |
| bimodal                   | boundary     | rare    |                          5 | mdn4                       |   0.0456637  | gaussian_diag  |
| bimodal                   | central      | common  |                          9 | mdn4                       |   0.0579274  | edm            |
| bimodal                   | central      | rare    |                          0 | unresolved                 | nan          | none           |
| bimodal                   | extrapolated | common  |                          3 | transformer_mdn4           |   0.191432   | mdn4           |
| bimodal                   | extrapolated | rare    |                          7 | transformer_mdn4           |   0.038114   | edm            |
| heavy_tailed              | boundary     | common  |                          8 | transformer_mdn4           |   0.117859   | student_rank2  |
| heavy_tailed              | boundary     | rare    |                          4 | ridge                      |   0.530175   | gaussian_rank2 |
| heavy_tailed              | central      | common  |                          9 | student_rank2              |   0.0185083  | edm            |
| heavy_tailed              | central      | rare    |                          0 | unresolved                 | nan          | none           |
| heavy_tailed              | extrapolated | common  |                          0 | unresolved                 | nan          | none           |
| heavy_tailed              | extrapolated | rare    |                          7 | flow                       |   0.19896    | gaussian_rank2 |
| linear_gaussian           | boundary     | common  |                          9 | ridge                      |   0.00455537 | gaussian_rank2 |
| linear_gaussian           | boundary     | rare    |                          8 | ridge                      |   0.0130025  | gaussian_rank2 |
| linear_gaussian           | central      | common  |                          9 | ridge                      |   0.00298396 | student_rank2  |
| linear_gaussian           | central      | rare    |                          0 | unresolved                 | nan          | none           |
| linear_gaussian           | extrapolated | common  |                          0 | unresolved                 | nan          | none           |
| linear_gaussian           | extrapolated | rare    |                          9 | ridge                      |   0.00158413 | flow           |
| nonlinear_heteroskedastic | boundary     | common  |                          9 | gaussian_rank2             |   0.0235396  | ridge          |
| nonlinear_heteroskedastic | boundary     | rare    |                          6 | ridge                      |   0.00964808 | gaussian_rank2 |
| nonlinear_heteroskedastic | central      | common  |                          9 | ridge                      |   0.00550813 | student_rank2  |
| nonlinear_heteroskedastic | central      | rare    |                          0 | unresolved                 | nan          | none           |
| nonlinear_heteroskedastic | extrapolated | common  |                          2 | transformer_mdn4           |   0.149014   | gp_rbf         |
| nonlinear_heteroskedastic | extrapolated | rare    |                          9 | ridge                      |   0.020795   | gaussian_rank2 |

## Calibration and derivative tradeoffs

Coverage targets90%, but width must be considered. Tail error uses true-law finite-bank95th-percentile thresholds, so finite-oracle quantile error remains. These are model-output probabilities, not a certification of epistemic uncertainty.

| family           | region       |   coverage90 |   width90 |   tail_probability_abs_error |
|:-----------------|:-------------|-------------:|----------:|-----------------------------:|
| edm              | boundary     |     0.886728 |  1.01396  |                    0.0232096 |
| edm              | central      |     0.88392  |  0.987676 |                    0.0358385 |
| edm              | extrapolated |     0.612144 |  1.11226  |                    0.238188  |
| flow             | boundary     |     0.885995 |  0.978794 |                    0.059474  |
| flow             | central      |     0.871641 |  0.944154 |                    0.0250136 |
| flow             | extrapolated |     0.780769 |  1.31676  |                    0.400647  |
| gaussian_diag    | boundary     |     0.889047 |  0.984162 |                    0.0536174 |
| gaussian_diag    | central      |     0.896294 |  1.01598  |                    0.0294298 |
| gaussian_diag    | extrapolated |     0.617249 |  1.06148  |                    0.135059  |
| gaussian_rank2   | boundary     |     0.885539 |  0.975472 |                    0.0558173 |
| gaussian_rank2   | central      |     0.905613 |  1.04764  |                    0.0330051 |
| gaussian_rank2   | extrapolated |     0.684575 |  1.16305  |                    0.136114  |
| gp_rbf           | boundary     |     0.870248 |  0.961981 |                    0.0540636 |
| gp_rbf           | central      |     0.851468 |  0.947788 |                    0.0512587 |
| gp_rbf           | extrapolated |     0.594604 |  1.96573  |                    0.0333062 |
| mdn4             | boundary     |     0.889018 |  0.995921 |                    0.024901  |
| mdn4             | central      |     0.886929 |  0.978177 |                    0.0264486 |
| mdn4             | extrapolated |     0.611712 |  1.27191  |                    0.0765299 |
| ridge            | boundary     |     0.892863 |  1.03402  |                    0.072168  |
| ridge            | central      |     0.907159 |  1.03217  |                    0.0318197 |
| ridge            | extrapolated |     0.649699 |  1.03349  |                    0.71556   |
| student_rank2    | boundary     |     0.888034 |  0.980141 |                    0.0539442 |
| student_rank2    | central      |     0.902644 |  1.0591   |                    0.0300727 |
| student_rank2    | extrapolated |     0.694214 |  1.21262  |                    0.115461  |
| transformer_mdn4 | boundary     |     0.89445  |  1.00616  |                    0.0228624 |
| transformer_mdn4 | central      |     0.867343 |  0.934143 |                    0.0261271 |
| transformer_mdn4 | extrapolated |     0.663095 |  1.29138  |                    0.0795492 |

Derivative summaries at step.1 (all steps remain in derivative_summary.csv) concern the conditional mean with respect to one history coordinate. They are not repaired-path effects or physical delays. The model-reference soft effects and these derivatives are distinct estimands.

| family           | region       |   step |   absolute_error |       mc_se |
|:-----------------|:-------------|-------:|-----------------:|------------:|
| edm              | boundary     |    0.1 |        0.179743  | 0.000762154 |
| edm              | central      |    0.1 |        0.123568  | 0.00123446  |
| edm              | extrapolated |    0.1 |        0.466826  | 0.000384955 |
| flow             | boundary     |    0.1 |        0.166709  | 0.00125904  |
| flow             | central      |    0.1 |        0.0705089 | 0.00146782  |
| flow             | extrapolated |    0.1 |        0.457061  | 0.000953338 |
| gaussian_diag    | boundary     |    0.1 |        0.13827   | 0.00101497  |
| gaussian_diag    | central      |    0.1 |        0.0869264 | 0.000679186 |
| gaussian_diag    | extrapolated |    0.1 |        0.575324  | 0.000385599 |
| gaussian_rank2   | boundary     |    0.1 |        0.1236    | 0.00180124  |
| gaussian_rank2   | central      |    0.1 |        0.0845619 | 0.00209825  |
| gaussian_rank2   | extrapolated |    0.1 |        0.559007  | 0.00133457  |
| gp_rbf           | boundary     |    0.1 |        0.184907  | 0.000233875 |
| gp_rbf           | central      |    0.1 |        0.0702042 | 7.92863e-06 |
| gp_rbf           | extrapolated |    0.1 |        0.889196  | 0.00247917  |
| mdn4             | boundary     |    0.1 |        0.125066  | 0.00575549  |
| mdn4             | central      |    0.1 |        0.118128  | 0.00908934  |
| mdn4             | extrapolated |    0.1 |        0.626881  | 0.0050163   |
| ridge            | boundary     |    0.1 |        0.22273   | 2.7239e-15  |
| ridge            | central      |    0.1 |        0.0973357 | 3.6305e-16  |
| ridge            | extrapolated |    0.1 |        0.172343  | 6.68073e-15 |
| student_rank2    | boundary     |    0.1 |        0.11553   | 0.00199697  |
| student_rank2    | central      |    0.1 |        0.0928169 | 0.00226428  |
| student_rank2    | extrapolated |    0.1 |        0.565887  | 0.00150326  |
| transformer_mdn4 | boundary     |    0.1 |        0.107706  | 0.00500273  |
| transformer_mdn4 | central      |    0.1 |        0.130835  | 0.00614883  |
| transformer_mdn4 | extrapolated |    0.1 |        0.562089  | 0.00310638  |

## Numerical cost and solver sensitivity

Times are local CPU single-thread observations while another GPU experiment ran; they are implementation-specific, not hardware-independent speed ratios. Each sample-time entry generates256 three-dimensional draws at one history. GP training includes its9-candidate validation grid; ridge includes3 candidates; neural families use one fixed optimizer configuration and validation stopping. This is matched data, not matched training compute or exhaustive tuning.

| family           |   mean_training_seconds |   max_training_seconds |   fits |   sampling_seconds_per_history |
|:-----------------|------------------------:|-----------------------:|-------:|-------------------------------:|
| edm              |              0.925223   |              2.81451   |     24 |                    0.00189715  |
| flow             |              1.01183    |              1.52331   |     24 |                    0.00184109  |
| gaussian_diag    |              0.355472   |              0.804585  |     24 |                    1.52181e-05 |
| gaussian_rank2   |              0.534217   |              0.948959  |     24 |                    2.32951e-05 |
| gp_rbf           |              0.196425   |              0.205943  |     12 |                    2.86965e-05 |
| mdn4             |              0.855698   |              1.83574   |     24 |                    0.000148482 |
| ridge            |              0.00693639 |              0.0185526 |     12 |                    9.86107e-06 |
| student_rank2    |              0.591156   |              1.12553   |     24 |                    2.71197e-05 |
| transformer_mdn4 |              1.86399    |              3.15573   |     24 |                    0.0015297   |

24-versus48-step mean and quantile changes:

| family   | region       |   mean_change_rmse |   quantile_change_rmse |   seconds24 |   seconds48 |
|:---------|:-------------|-------------------:|-----------------------:|------------:|------------:|
| edm      | boundary     |        0.000395871 |            0.00151725  |   0.0177984 |   0.0358439 |
| edm      | central      |        0.000132757 |            0.00141532  |   0.0179008 |   0.0359877 |
| edm      | extrapolated |        0.000535503 |            0.00168145  |   0.0178731 |   0.0358743 |
| flow     | boundary     |        6.77927e-05 |            0.000143372 |   0.0167375 |   0.0329955 |
| flow     | central      |        5.96055e-05 |            0.00016853  |   0.0165971 |   0.0327136 |
| flow     | extrapolated |        6.12775e-05 |            9.04559e-05 |   0.0165539 |   0.0329389 |

Cross-family disagreement versus average error has descriptive Spearman correlation 0.7186 over 34 fully qualified query groups. Groups share laws, datasets and generators; no independent-row p-value is justified. This is not a trained or externally validated OOD detector.

## What is still missing

Full-cohort NeuroPAL Transformer/GP and other-family refits, deliberate support exclusions for every generator, and generator-specific repaired lag matrices remain missing. The original per-generator OOD tables/checkpoints were not recovered. This supplement answers whether these fitted generators differ on the specified known laws, not which is biologically correct. Kernel choice, capacity, hyperparameter tuning, sample size, output dimensionality, calibration and optimization can change rankings. Consult fit_diagnostics.csv for convergence flags and the source-pinned PROTOCOL.md for exact settings.
