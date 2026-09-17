# Does Transformer entropy track uncertainty here?

Exploratory supplement: no weights or primary benchmark settings changed. Mixture-label entropy is a categorical representation statistic, not continuous predictive entropy. Joint predictive entropy estimates -E log p(Y|H) from256 model samples, including output scaling. Either can be poorly associated with error in a misspecified model. Differential entropy is coordinate/unit-dependent; compare regions within each law. Means average neural seeds/history rows within each dataset first. Correlations are descriptive across9 dataset-region groups/law, not independent-row significance tests or validated OOD detection.

| law                       | region       |   mixture_label_entropy |   joint_predictive_entropy |   mean_squared_error |   coverage90 |
|:--------------------------|:-------------|------------------------:|---------------------------:|---------------------:|-------------:|
| bimodal                   | boundary     |                0.511634 |                   0.432748 |           0.00458608 |     0.895173 |
| bimodal                   | central      |                0.656434 |                   0.511584 |           0.0119503  |     0.851219 |
| bimodal                   | extrapolated |                0.478654 |                   0.921768 |           0.269558   |     0.701208 |
| heavy_tailed              | boundary     |                0.543143 |                  -0.469454 |           0.00144688 |     0.894305 |
| heavy_tailed              | central      |                0.648644 |                  -0.569402 |           0.00235792 |     0.872748 |
| heavy_tailed              | extrapolated |                0.567105 |                   0.574924 |           0.197955   |     0.728253 |
| linear_gaussian           | boundary     |                0.619574 |                  -0.196699 |           0.00249659 |     0.893012 |
| linear_gaussian           | central      |                0.681214 |                  -0.276306 |           0.0023191  |     0.874783 |
| linear_gaussian           | extrapolated |                0.642513 |                   0.728419 |           0.465959   |     0.499304 |
| nonlinear_heteroskedastic | boundary     |                0.556297 |                   0.174545 |           0.00175244 |     0.895309 |
| nonlinear_heteroskedastic | central      |                0.623657 |                  -0.239584 |           0.00288398 |     0.870624 |
| nonlinear_heteroskedastic | extrapolated |                0.591572 |                   0.91574  |           0.209558   |     0.723615 |

| law                       | entropy                  |   descriptive_spearman_with_mean_squared_error |   n_dataset_region_groups |
|:--------------------------|:-------------------------|-----------------------------------------------:|--------------------------:|
| bimodal                   | mixture_label_entropy    |                                     -0.283333  |                         9 |
| bimodal                   | joint_predictive_entropy |                                      0.483333  |                         9 |
| heavy_tailed              | mixture_label_entropy    |                                      0.566667  |                         9 |
| heavy_tailed              | joint_predictive_entropy |                                      0.55      |                         9 |
| linear_gaussian           | mixture_label_entropy    |                                      0.0833333 |                         9 |
| linear_gaussian           | joint_predictive_entropy |                                      0.716667  |                         9 |
| nonlinear_heteroskedastic | mixture_label_entropy    |                                     -0.0666667 |                         9 |
| nonlinear_heteroskedastic | joint_predictive_entropy |                                      0.633333  |                         9 |
