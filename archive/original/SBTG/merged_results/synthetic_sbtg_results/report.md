# SBTG Benchmark Report

## Experimental Setup

- Number of neurons: 10
- Number of stimuli: 3
- Seeds: [0, 1]
- Noise levels: ['low', 'high']
- Length types: ['short', 'long']
- Short length T_short: 300
- Long length T_long: 800

### SBTG Hyperparameter Search

- For each dataset family × noise × length, we ran a small grid search over DSM training parameters for SBTG-Classic and SBTG-Structured using a single hyperparameter seed.
- For the best training configuration per variant, we then swept a small grid over HAC and FDR parameters and evaluated F1 across all seeds.
- Best SBTG training + statistical configurations (per family/noise/length) are saved in `best_params.json`.

## Average Metrics by Dataset and Method (Best Stat Config per Method)

### var

| Method               | Best stat cfg (for plots) | Precision | Recall | F1 (mean±std) | ROC AUC (mean±std) | PR AUC (mean±std) |
| -------------------- | ------------------------- | --------- | ------ | ------------- | ------------------ | ----------------- |
| DYNOTEARS            | default                   | 0.148     | 0.143  | 0.141±0.155   | 0.558±0.063        | 0.191±0.107       |
| NOTEARS              | default                   | 0.125     | 0.036  | 0.056±0.103   | 0.516±0.030        | 0.104±0.023       |
| PCMCI+               | default                   | 0.089     | 0.288  | 0.128±0.087   | 0.518±0.087        | 0.099±0.002       |
| SBTG-FeatureBilinear | hac7_alpha010_by          | 0.306     | 0.879  | 0.392±0.200   | 0.719±0.160        | 0.252±0.153       |
| SBTG-Linear          | hac5_alpha010_by          | 0.212     | 0.825  | 0.321±0.212   | 0.707±0.191        | 0.217±0.136       |
| SBTG-Minimal         | hac5_alpha010_by          | 0.051     | 0.429  | 0.091±0.100   | 0.535±0.075        | 0.098±0.016       |
| VAR-LASSO            | default                   | 0.183     | 0.325  | 0.207±0.039   | 0.591±0.038        | 0.170±0.033       |
| VAR-LiNGAM           | default                   | 0.094     | 0.188  | 0.122±0.101   | 0.521±0.052        | 0.104±0.034       |
| VAR-Ridge            | default                   | 0.094     | 1.000  | 0.172±0.030   | 0.461±0.138        | 0.134±0.060       |

### poisson

| Method               | Best stat cfg (for plots) | Precision | Recall | F1 (mean±std) | ROC AUC (mean±std) | PR AUC (mean±std) |
| -------------------- | ------------------------- | --------- | ------ | ------------- | ------------------ | ----------------- |
| DYNOTEARS            | default                   | 0.024     | 0.036  | 0.029±0.053   | 0.478±0.040        | 0.092±0.010       |
| NOTEARS              | default                   | 0.035     | 0.036  | 0.035±0.065   | 0.495±0.032        | 0.094±0.010       |
| PCMCI+               | default                   | 0.136     | 0.349  | 0.193±0.072   | 0.562±0.068        | 0.110±0.022       |
| Poisson-GLM          | default                   | 0.089     | 1.000  | 0.163±0.020   | 0.542±0.078        | 0.240±0.108       |
| SBTG-FeatureBilinear | hac5_alpha010_by          | 0.142     | 0.613  | 0.226±0.088   | 0.622±0.118        | 0.130±0.049       |
| SBTG-Linear          | hac7_alpha010_by          | 0.149     | 0.762  | 0.247±0.117   | 0.663±0.159        | 0.151±0.060       |
| SBTG-Minimal         | hac5_alpha010_by          | 0.115     | 0.796  | 0.200±0.068   | 0.586±0.109        | 0.114±0.043       |
| VAR-LASSO            | default                   | 0.185     | 0.308  | 0.225±0.116   | 0.583±0.084        | 0.155±0.061       |
| VAR-LiNGAM           | default                   | 0.233     | 0.060  | 0.091±0.135   | 0.511±0.052        | 0.122±0.056       |
| VAR-Ridge            | default                   | 0.089     | 1.000  | 0.163±0.020   | 0.525±0.093        | 0.147±0.076       |

### hawkes

| Method               | Best stat cfg (for plots) | Precision | Recall | F1 (mean±std) | ROC AUC (mean±std) | PR AUC (mean±std) |
| -------------------- | ------------------------- | --------- | ------ | ------------- | ------------------ | ----------------- |
| DYNOTEARS            | default                   | 0.062     | 0.018  | 0.028±0.079   | 0.503±0.026        | 0.096±0.020       |
| NOTEARS              | default                   | 0.000     | 0.000  | 0.000±0.000   | 0.499±0.002        | 0.089±0.012       |
| PCMCI+               | default                   | 0.094     | 0.258  | 0.135±0.047   | 0.509±0.051        | 0.093±0.012       |
| Poisson-GLM          | default                   | 0.089     | 1.000  | 0.163±0.020   | 0.521±0.084        | 0.194±0.087       |
| SBTG-FeatureBilinear | hac5_alpha010_by          | 0.132     | 0.560  | 0.200±0.087   | 0.559±0.143        | 0.116±0.048       |
| SBTG-Linear          | hac7_alpha010_by          | 0.119     | 0.649  | 0.200±0.092   | 0.575±0.150        | 0.118±0.046       |
| SBTG-Minimal         | hac7_alpha010_by          | 0.117     | 0.714  | 0.199±0.101   | 0.572±0.159        | 0.119±0.059       |
| VAR-LASSO            | default                   | 0.152     | 0.226  | 0.173±0.092   | 0.551±0.061        | 0.135±0.040       |
| VAR-LiNGAM           | default                   | 0.062     | 0.018  | 0.028±0.079   | 0.502±0.026        | 0.096±0.020       |
| VAR-Ridge            | default                   | 0.089     | 1.000  | 0.163±0.020   | 0.508±0.073        | 0.142±0.036       |

### tanh

| Method               | Best stat cfg (for plots) | Precision | Recall | F1 (mean±std) | ROC AUC (mean±std) | PR AUC (mean±std) |
| -------------------- | ------------------------- | --------- | ------ | ------------- | ------------------ | ----------------- |
| DYNOTEARS            | default                   | 0.269     | 0.198  | 0.225±0.111   | 0.573±0.058        | 0.212±0.092       |
| NOTEARS              | default                   | 0.164     | 0.099  | 0.123±0.159   | 0.538±0.060        | 0.158±0.114       |
| PCMCI+               | default                   | 0.107     | 0.500  | 0.169±0.031   | 0.535±0.046        | 0.097±0.011       |
| SBTG-FeatureBilinear | hac7_alpha010_by          | 0.247     | 0.909  | 0.334±0.169   | 0.700±0.136        | 0.204±0.117       |
| SBTG-Linear          | hac7_alpha010_by          | 0.245     | 0.982  | 0.388±0.078   | 0.827±0.066        | 0.241±0.059       |
| SBTG-Minimal         | hac5_alpha010_by          | 0.125     | 0.931  | 0.220±0.039   | 0.648±0.080        | 0.125±0.024       |
| VAR-LASSO            | default                   | 0.133     | 0.496  | 0.205±0.033   | 0.597±0.046        | 0.238±0.079       |
| VAR-LiNGAM           | default                   | 0.055     | 0.165  | 0.077±0.070   | 0.475±0.035        | 0.092±0.013       |
| VAR-Ridge            | default                   | 0.089     | 1.000  | 0.163±0.020   | 0.562±0.052        | 0.237±0.061       |

## Best SBTG Configurations (Per Dataset Variant)

For each dataset family, noise level, and length type, the table below summarizes the best SBTG training and statistical configurations, along with the resulting F1 mean and standard deviation across seeds.

### var

#### Noise: low

- Length: short

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.269   | 0.381  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.424   | 0.293  |
| SBTG-Minimal         | minimal_tuned                | hac7_alpha010_by | 0.059   | 0.083  |

- Length: long

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.387   | 0.278  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac7_alpha010_by | 0.346   | 0.218  |
| SBTG-Minimal         | minimal_tuned                | hac5_alpha010_by | 0.211   | 0.002  |

#### Noise: high

- Length: short

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac7_alpha010_by | 0.245   | 0.067  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.444   | 0.314  |
| SBTG-Minimal         | minimal_tuned                | hac5_alpha010_by | 0.083   | 0.118  |

- Length: long

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.384   | 0.229  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac7_alpha010_by | 0.365   | 0.167  |
| SBTG-Minimal         | minimal_tuned                | hac7_alpha010_by | 0.117   | 0.165  |

### poisson

#### Noise: low

- Length: short

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.218   | 0.152  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.169   | 0.003  |
| SBTG-Minimal         | minimal_tuned                | hac5_alpha010_by | 0.182   | 0.062  |

- Length: long

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.189   | 0.195  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.150   | 0.011  |
| SBTG-Minimal         | minimal_tuned                | hac7_alpha010_by | 0.158   | 0.018  |

#### Noise: high

- Length: short

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac7_alpha010_by | 0.356   | 0.032  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.358   | 0.024  |
| SBTG-Minimal         | minimal_tuned                | hac5_alpha010_by | 0.208   | 0.020  |

- Length: long

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.226   | 0.006  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac7_alpha010_by | 0.228   | 0.008  |
| SBTG-Minimal         | minimal_tuned                | hac5_alpha010_by | 0.264   | 0.116  |

### hawkes

#### Noise: low

- Length: short

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac7_alpha010_by | 0.146   | 0.071  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac7_alpha010_by | 0.189   | 0.093  |
| SBTG-Minimal         | minimal_tuned                | hac5_alpha010_by | 0.165   | 0.092  |

- Length: long

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.156   | 0.127  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.117   | 0.044  |
| SBTG-Minimal         | minimal_tuned                | hac7_alpha010_by | 0.176   | 0.118  |

#### Noise: high

- Length: short

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac7_alpha010_by | 0.244   | 0.127  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.208   | 0.059  |
| SBTG-Minimal         | minimal_tuned                | hac7_alpha010_by | 0.168   | 0.060  |

- Length: long

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.254   | 0.045  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac7_alpha010_by | 0.299   | 0.086  |
| SBTG-Minimal         | minimal_tuned                | hac7_alpha010_by | 0.297   | 0.145  |

### tanh

#### Noise: low

- Length: short

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.309   | 0.058  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac7_alpha010_by | 0.572   | 0.023  |
| SBTG-Minimal         | minimal_tuned                | hac5_alpha010_by | 0.192   | 0.019  |

- Length: long

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac7_alpha010_by | 0.404   | 0.082  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.377   | 0.002  |
| SBTG-Minimal         | minimal_tuned                | hac5_alpha010_by | 0.189   | 0.013  |

#### Noise: high

- Length: short

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac5_alpha010_by | 0.374   | 0.020  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.226   | 0.011  |
| SBTG-Minimal         | minimal_tuned                | hac7_alpha010_by | 0.253   | 0.019  |

- Length: long

| Method               | Train config                 | Stat config      | F1 mean | F1 std |
| -------------------- | ---------------------------- | ---------------- | ------- | ------ |
| SBTG-Linear          | linear_optuna_best           | hac7_alpha010_by | 0.469   | 0.071  |
| SBTG-FeatureBilinear | feature_bilinear_optuna_best | hac5_alpha010_by | 0.168   | 0.031  |
| SBTG-Minimal         | minimal_tuned                | hac5_alpha010_by | 0.264   | 0.044  |

## Notes

- SBTG-Classic uses a generic MLP-based score network trained via denoising score matching.
- SBTG-Structured enforces an energy-based structure on the score, with a bilinear cross-lag term and neuron-wise scalar energies.
- Mean and energy tests with HAC variance and FDR control are used to threshold scores into directed edges.
- Baselines (VAR-LASSO, VAR-Ridge, VAR-LiNGAM, Poisson-GLM, PCMCI+, DYNOTEARS) provide comparison points across linear and nonlinear, parametric and nonparametric causal discovery methods.
