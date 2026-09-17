# Completed synthetic generator comparison

All192 planned fit/evaluation tasks completed. This is a supplementary low-dimensional fixed-configuration experiment, not a reanalysis of missing NeuroPAL tables and not biological replication. 0 neural fits remain boundary-best/nonconverged under the declared limit. Review fit_diagnostics.csv before interpreting architecture differences.

## Predictive comparison

Each entry averages histories, then training initializations within a dataset, then equally across three data seeds and four specified laws. Read law-specific and paired tables; an aggregate winner need not win each mechanism. Error intervals are seed ranges, not confidence intervals. Three data seeds/law do not justify universal superiority claims.

| family           | region       |   energy |   mean_squared_error |   coverage90 |   tail_probability_abs_error |   sampling_seconds_per_history |
|:-----------------|:-------------|---------:|---------------------:|-------------:|-----------------------------:|-------------------------------:|
| edm              | boundary     | 0.342103 |           0.0136799  |     0.886728 |                    0.0232096 |                    0.00189338  |
| edm              | central      | 0.352325 |           0.00406525 |     0.88392  |                    0.0358385 |                    0.00190015  |
| edm              | extrapolated | 0.749112 |           0.298508   |     0.612144 |                    0.238188  |                    0.00189791  |
| flow             | boundary     | 0.33048  |           0.00361474 |     0.885995 |                    0.059474  |                    0.00183227  |
| flow             | central      | 0.349848 |           0.00150768 |     0.871641 |                    0.0250136 |                    0.00185687  |
| flow             | extrapolated | 0.624208 |           0.208997   |     0.780769 |                    0.400647  |                    0.00183414  |
| gaussian_diag    | boundary     | 0.334107 |           0.00256717 |     0.889047 |                    0.0536174 |                    1.49954e-05 |
| gaussian_diag    | central      | 0.362377 |           0.00235506 |     0.896294 |                    0.0294298 |                    1.57175e-05 |
| gaussian_diag    | extrapolated | 0.650981 |           0.234094   |     0.617249 |                    0.135059  |                    1.49413e-05 |
| gaussian_rank2   | boundary     | 0.327953 |           0.00202129 |     0.885539 |                    0.0558173 |                    2.35558e-05 |
| gaussian_rank2   | central      | 0.353093 |           0.00215558 |     0.905613 |                    0.0330051 |                    2.29351e-05 |
| gaussian_rank2   | extrapolated | 0.612267 |           0.206748   |     0.684575 |                    0.136114  |                    2.33944e-05 |
| gp_rbf           | boundary     | 0.33889  |           0.00341882 |     0.870248 |                    0.0540636 |                    2.96301e-05 |
| gp_rbf           | central      | 0.368621 |           0.00314176 |     0.851468 |                    0.0512587 |                    3.16211e-05 |
| gp_rbf           | extrapolated | 1.25036  |           1.10331    |     0.594604 |                    0.0333062 |                    2.48382e-05 |
| mdn4             | boundary     | 0.325826 |           0.00216797 |     0.889018 |                    0.024901  |                    0.000150155 |
| mdn4             | central      | 0.351104 |           0.0028998  |     0.886929 |                    0.0264486 |                    0.000146349 |
| mdn4             | extrapolated | 0.808513 |           0.403121   |     0.611712 |                    0.0765299 |                    0.000148941 |
| ridge            | boundary     | 0.335795 |           0.00538359 |     0.892863 |                    0.072168  |                    9.41002e-06 |
| ridge            | central      | 0.354357 |           0.00191252 |     0.907159 |                    0.0318197 |                    9.89221e-06 |
| ridge            | extrapolated | 1.24591  |           0.903694   |     0.649699 |                    0.71556   |                    1.0281e-05  |
| student_rank2    | boundary     | 0.327141 |           0.00266453 |     0.888034 |                    0.0539442 |                    2.6972e-05  |
| student_rank2    | central      | 0.353531 |           0.00303962 |     0.902644 |                    0.0300727 |                    2.6245e-05  |
| student_rank2    | extrapolated | 0.620814 |           0.215977   |     0.694214 |                    0.115461  |                    2.81421e-05 |
| transformer_mdn4 | boundary     | 0.325751 |           0.0025705  |     0.89445  |                    0.0228624 |                    0.00155057  |
| transformer_mdn4 | central      | 0.353589 |           0.00487782 |     0.867343 |                    0.0261271 |                    0.00150179  |
| transformer_mdn4 | extrapolated | 0.676703 |           0.285757   |     0.663095 |                    0.0795492 |                    0.00153673  |

## Generator error in identical soft-event queries

These are independent direct-reference comparisons, not different sampler competitions. Only cells meeting both oracle/model repeat-SE and ESS gates contribute to qualified error. Both initializations must qualify. Coverage differs; use paired_effect_differences.csv for shared qualified comparisons. Unresolved cells are not successes or zero errors.

| family           | region       | event   |   qualified_error |   qualified_datasets |   total_datasets |
|:-----------------|:-------------|:--------|------------------:|---------------------:|-----------------:|
| edm              | boundary     | common  |         0.137602  |                   12 |               12 |
| edm              | boundary     | rare    |         0.13438   |                    5 |               12 |
| edm              | central      | common  |         0.0324526 |                   12 |               12 |
| edm              | central      | rare    |       nan         |                    0 |               12 |
| edm              | extrapolated | common  |       nan         |                    0 |               12 |
| edm              | extrapolated | rare    |         0.125795  |                   12 |               12 |
| flow             | boundary     | common  |         0.123105  |                   11 |               12 |
| flow             | boundary     | rare    |         0.0996708 |                    8 |               12 |
| flow             | central      | common  |         0.048604  |                   12 |               12 |
| flow             | central      | rare    |       nan         |                    0 |               12 |
| flow             | extrapolated | common  |       nan         |                    0 |               12 |
| flow             | extrapolated | rare    |         0.11027   |                   12 |               12 |
| gaussian_diag    | boundary     | common  |         0.410223  |                   12 |               12 |
| gaussian_diag    | boundary     | rare    |         0.341646  |                   12 |               12 |
| gaussian_diag    | central      | common  |         0.393505  |                   12 |               12 |
| gaussian_diag    | central      | rare    |       nan         |                    0 |               12 |
| gaussian_diag    | extrapolated | common  |         0.431481  |                    3 |               12 |
| gaussian_diag    | extrapolated | rare    |         0.261029  |                   12 |               12 |
| gaussian_rank2   | boundary     | common  |         0.13023   |                   12 |               12 |
| gaussian_rank2   | boundary     | rare    |         0.254548  |                   12 |               12 |
| gaussian_rank2   | central      | common  |         0.0956695 |                   12 |               12 |
| gaussian_rank2   | central      | rare    |       nan         |                    0 |               12 |
| gaussian_rank2   | extrapolated | common  |         0.460003  |                    3 |               12 |
| gaussian_rank2   | extrapolated | rare    |         0.128863  |                   12 |               12 |
| gp_rbf           | boundary     | common  |         0.407303  |                   12 |               12 |
| gp_rbf           | boundary     | rare    |         0.340663  |                   12 |               12 |
| gp_rbf           | central      | common  |         0.393409  |                   12 |               12 |
| gp_rbf           | central      | rare    |       nan         |                    0 |               12 |
| gp_rbf           | extrapolated | common  |         0.536203  |                    6 |               12 |
| gp_rbf           | extrapolated | rare    |         0.259004  |                   11 |               12 |
| mdn4             | boundary     | common  |         0.100331  |                   12 |               12 |
| mdn4             | boundary     | rare    |         0.0534652 |                    9 |               12 |
| mdn4             | central      | common  |         0.037188  |                   12 |               12 |
| mdn4             | central      | rare    |       nan         |                    0 |               12 |
| mdn4             | extrapolated | common  |         0.221739  |                    5 |               12 |
| mdn4             | extrapolated | rare    |         0.107804  |                   11 |               12 |
| ridge            | boundary     | common  |         0.106291  |                   12 |               12 |
| ridge            | boundary     | rare    |         0.246325  |                   12 |               12 |
| ridge            | central      | common  |         0.0923529 |                   12 |               12 |
| ridge            | central      | rare    |       nan         |                    0 |               12 |
| ridge            | extrapolated | common  |       nan         |                    0 |               12 |
| ridge            | extrapolated | rare    |         0.170197  |                   12 |               12 |
| student_rank2    | boundary     | common  |         0.1201    |                   12 |               12 |
| student_rank2    | boundary     | rare    |         0.215363  |                    5 |               12 |
| student_rank2    | central      | common  |         0.0793688 |                   12 |               12 |
| student_rank2    | central      | rare    |       nan         |                    0 |               12 |
| student_rank2    | extrapolated | common  |         0.233212  |                    3 |               12 |
| student_rank2    | extrapolated | rare    |         0.103576  |                   10 |               12 |
| transformer_mdn4 | boundary     | common  |         0.0991918 |                   12 |               12 |
| transformer_mdn4 | boundary     | rare    |         0.0884416 |                    6 |               12 |
| transformer_mdn4 | central      | common  |         0.103344  |                   12 |               12 |
| transformer_mdn4 | central      | rare    |       nan         |                    0 |               12 |
| transformer_mdn4 | extrapolated | common  |         0.170223  |                    6 |               12 |
| transformer_mdn4 | extrapolated | rare    |         0.152426  |                   12 |               12 |

## Missing biological evidence

The original six-family per-generator OOD tables and original neural checkpoints were absent. This run does not recover their ranking. Full NeuroPAL Transformer/GP and all-family support-exclusion refits, long-history encoder comparisons, full repaired-path lag matrices across families, and broader training-budget/hyperparameter sensitivity remain unperformed. Model sampling entropy is not calibrated epistemic uncertainty. This GP assumes independent Gaussian outputs; its behavior does not generalize to correlated or non-Gaussian GPs.

## Auditing

All raw rows, paired differences, checkpoint receipts and qualified denominators are retained. All six figures were visually inspected; see analysis/validation.json for the completed audit. See PROTOCOL.md for the exact estimands and design, TRADEOFFS.md for prior evidence and architecture tradeoffs.
