# Conditional effects with analytic / quadrature references

Exploratory supplement documented in analysis_amendment_002.md. It changes neither trained models nor queries. True effects use closed-form Gaussian/mixture conditioning or one-dimensional Student-t quadrature. Gaussian/Student-t learned effects also use analytic/quadrature expectations; MDN, Transformer, flow and EDM retain their original finite-Monte-Carlo reference and precision gate. This unequal numerical access is explicit: excluded models cannot be declared worse from missing reference precision. Primary Monte Carlo results remain intact.

Below are lowest mean errors among families with qualified learned references for all3 datasets and both neural seeds. Means are descriptive across this fixed design, not proof of optimality. Flow errors are displayed even when its qualified-dataset count is below3; do not treat those unresolved values as reliable comparisons. The full paired table masks differences whenever either family lacks a qualified reference.

| law                       | region       | event   |   qualified_families | lowest_mean_error_family   |      error |   flow_error |   flow_qualified_datasets |
|:--------------------------|:-------------|:--------|---------------------:|:---------------------------|-----------:|-------------:|--------------------------:|
| bimodal                   | boundary     | common  |                    9 | ridge                      | 0.00780323 |    0.15375   |                         3 |
| bimodal                   | boundary     | rare    |                    6 | mdn4                       | 0.0430518  |    0.081778  |                         1 |
| bimodal                   | central      | common  |                    9 | mdn4                       | 0.0581376  |    0.121812  |                         3 |
| bimodal                   | central      | rare    |                    5 | gaussian_diag              | 0.179922   |    0.113476  |                         0 |
| bimodal                   | extrapolated | common  |                    7 | transformer_mdn4           | 0.186457   |    0.31539   |                         0 |
| bimodal                   | extrapolated | rare    |                    9 | transformer_mdn4           | 0.038114   |    0.124097  |                         3 |
| heavy_tailed              | boundary     | common  |                    8 | transformer_mdn4           | 0.120558   |    0.12649   |                         2 |
| heavy_tailed              | boundary     | rare    |                    5 | student_rank2              | 0.204069   |    0.4912    |                         1 |
| heavy_tailed              | central      | common  |                    9 | edm                        | 0.0190494  |    0.0266103 |                         3 |
| heavy_tailed              | central      | rare    |                    5 | student_rank2              | 0.0578698  |    0.542267  |                         0 |
| heavy_tailed              | extrapolated | common  |                    6 | student_rank2              | 0.289924   |    0.500482  |                         0 |
| heavy_tailed              | extrapolated | rare    |                    8 | student_rank2              | 0.157653   |    0.193923  |                         3 |
| linear_gaussian           | boundary     | common  |                    9 | ridge                      | 0.00216752 |    0.0986198 |                         3 |
| linear_gaussian           | boundary     | rare    |                    8 | ridge                      | 0.00216752 |    0.038462  |                         3 |
| linear_gaussian           | central      | common  |                    9 | ridge                      | 0.00216752 |    0.0209671 |                         3 |
| linear_gaussian           | central      | rare    |                    5 | ridge                      | 0.00216752 |    0.0622758 |                         0 |
| linear_gaussian           | extrapolated | common  |                    7 | ridge                      | 0.00216752 |    0.0904741 |                         0 |
| linear_gaussian           | extrapolated | rare    |                    9 | ridge                      | 0.00216752 |    0.0597532 |                         3 |
| nonlinear_heteroskedastic | boundary     | common  |                    9 | ridge                      | 0.013602   |    0.117789  |                         3 |
| nonlinear_heteroskedastic | boundary     | rare    |                    7 | ridge                      | 0.013602   |    0.0314665 |                         3 |
| nonlinear_heteroskedastic | central      | common  |                    9 | ridge                      | 0.00555561 |    0.0231867 |                         3 |
| nonlinear_heteroskedastic | central      | rare    |                    5 | ridge                      | 0.00555561 |    0.0823245 |                         0 |
| nonlinear_heteroskedastic | extrapolated | common  |                    6 | ridge                      | 0.0119078  |    0.121096  |                         0 |
| nonlinear_heteroskedastic | extrapolated | rare    |                    9 | ridge                      | 0.0119078  |    0.059298  |                         3 |

The largest absolute discrepancy between a *qualified* original oracle MC reference and the supplementary truth was 0.023526909069873092 output units. All original discrepancies and gates are retained in analytic_effects_raw.csv.
