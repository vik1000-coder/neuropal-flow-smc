# Developer handoff: implementation and testing plan for score-identified neuromodulator dynamics

Use this as a build specification for an agent/developer. The goal is to implement a reproducible Python package that estimates **timescale-resolved distributional interactions** in C. elegans neural activity, detects statistically significant changes in the conditional circuit, and reports what changed in biologically interpretable terms.

The core object is not a VAR coefficient. It is the conditional predictive law

[
\rho_t(y\mid H_t),
]

where (y) is future neural activity and (H_t) is recent neural/behavioral history. The uploaded document defines the central lag-influence score

[
\ell_u(y;h)=\nabla_{x_{t+1-u}}\log \rho(y\mid h),
]

and shows that effects on **mean, variance, tails, modes, or any statistic** can be read out as covariances against this score rather than by differentiating a fitted black-box model.  The document also motivates why this goes beyond VAR/SINDy: VAR, spectral, and regression/SINDy-style approaches identify autocovariances or conditional means, whereas the conditional-law approach targets (\rho(y\mid h)) and can expose mean-blind variance or distributional memory. 

For C. elegans, the biological motivation is strong: recent work describes the neuropeptidergic connectome as a dense, decentralized “wireless” signaling network, and a neural signal-propagation atlas found functional interactions that differ from anatomy-based predictions, including extrasynaptic peptidergic effects. ([Cell][1]) Brain-wide C. elegans recordings also show widespread state- and behavior-dependent encoding, so a method that only estimates fixed mean-drive interactions is too narrow for the neuromodulator question. ([ScienceDirect][2])

---

# 1. Project objective

Build a package tentatively named:

```text
sid_neuromod
```

where `sid` means **score-identified dynamics**.

The package must support four workflows:

1. **Synthetic validation**
   Reproduce the theory’s core examples: mean-blind variance dynamics, measurement noise leakage, hidden oscillatory memory, directed spillover, simultaneous inference, and conditional-law change detection.

2. **C. elegans data analysis**
   Given neural traces, timestamps, neuron IDs, and optional behavior/stimulus covariates, estimate source-target-timescale effects on future neural activity.

3. **Circuit-change detection**
   Given a baseline interval and a monitoring interval, detect when the conditional predictive law changes and identify which channels changed.

4. **Neuromodulator-candidate interpretation**
   Rank slow, broad, distributional effects as candidate neuromodulatory factors, while clearly separating predictive evidence from causal or molecular claims.

The minimum viable scientific claim should be:

> We estimate a timescale-resolved distributional functional connectome. Fast mean-channel effects are treated as VAR-comparable predictive drive. Slow gain, tail, covariance, and state-channel effects are treated as latent neuromodulatory candidates, especially when broad across targets and enriched for known peptide/amine/receptor annotations.

---

# 2. Required conceptual background for the developer

The developer should understand these five ideas before coding.

## 2.1 Conditional law, not conditional mean

For each target neuron (i), fit

[
Y_i(t) \mid H_t \sim \rho_i(\cdot\mid H_t),
]

where

[
Y_i(t)=x_i(t+\Delta)
]

or

[
Y_i(t)=x_i(t+\Delta)-x_i(t).
]

The user’s “circuit” is the full predictive density (\rho_i(y\mid H_t)), not just (\mathbb E[Y_i\mid H_t]). The uploaded document explicitly frames the observed system as a partially observed latent process where all observable questions are functionals of the predictive law (\rho(y\mid h)). 

## 2.2 Lag-influence score

For a source feature at lag/timescale (u),

[
\ell_u(y;h)
===========

\nabla_{x_{t+1-u}}\log \rho(y\mid h).
]

This answers:

> If the past source activity were infinitesimally changed, how would the entire future density tilt?

The score projection onto (y) gives a mean effect; projection onto (y^2) gives variance/gain effects; richer statistics give tails, skewness, or mode changes. 

## 2.3 Covariance readout

For any statistic (\phi(Y)),

[
\frac{\partial}{\partial x_{t+1-u}}
\mathbb E[\phi(Y)\mid H_t]
==========================

\operatorname{Cov}_{\rho(\cdot\mid H_t)}
\left(
\phi(Y),\ell_u(Y;H_t)
\right).
]

This is the implementation workhorse. The document proves this as the covariance readout and notes that mean effects use (\phi(y)=y), variance effects use quadratic statistics, and higher basis expansions decompose influence into mean, scale, skew, tail, and mode-switching channels. 

## 2.4 Exponential filter bank instead of dense lag bins

Implement timescale features

[
f_{j,\tau}(t)
=============

\int_0^\infty \frac1{\tau}e^{-s/\tau}z_j(t-s),ds,
]

where (z_j) is a chosen source signal, usually standardized neural activity, deconvolved activity, derivative activity, positive activity, or behavior-corrected activity.

The uploaded document recommends exponential filters because dense adjacent lags become coefficient-fragile as sampling becomes finer, while fixed filter banks give sampling-rate-stable coefficient readouts. 

## 2.5 Why this beats only VAR/SINDy for neuromodulators

VAR/SINDy estimate conditional mean structure. They are appropriate baselines for fast mean-drive effects, but they can miss slow hidden modulation that changes variance, gain, thresholds, synchrony, or mode probability while leaving the mean unchanged. The document’s stochastic-volatility example is exactly this: the process has zero conditional mean and white autocovariance but a nonzero conditional-variance memory kernel. 

---

# 3. Recommended technical stack

Use Python as the primary implementation language.

Core scientific stack:

```text
numpy
scipy
pandas or polars
scikit-learn
statsmodels
matplotlib
```

Use NumPy for array operations and reproducible random generation; SciPy for linear algebra, optimization, distributions, and signal-processing utilities; pandas or Polars for tabular outputs; scikit-learn for CCA/PCA/baseline preprocessing; and statsmodels for VAR baselines and HAC/Newey-West references. NumPy’s documentation recommends `default_rng` for modern random generation, SciPy provides the core numerical algorithms needed here, pandas/Polars cover tabular data workflows, scikit-learn is the standard general-purpose ML toolkit, and statsmodels exposes HAC/Newey-West and VAR functionality. ([NumPy][3])

Testing and reproducibility:

```text
pytest
hypothesis
ruff
mypy or pyright
pre-commit
```

`pytest` is appropriate because it supports small readable tests and scales to complex test suites. ([pytest][4])

Optional model/back-end stack:

```text
jax or pytorch
mlflow or wandb
zarr or h5py
networkx
```

Use JAX or PyTorch only for the later neural/two-head models; the MVP should be closed-form NumPy/SciPy so the estimator is not confounded by optimizer failures. JAX and PyTorch both provide automatic differentiation and accelerator support; MLflow or W&B can track experiments, metrics, and artifacts. ([JAX Documentation][5])

Baseline stack:

```text
statsmodels VAR
pysindy
sklearn Ridge / ElasticNet
```

Use statsmodels VAR for linear conditional-mean baselines and PySINDy for sparse nonlinear dynamics baselines. ([Statsmodels][6])

---

# 4. Repository structure

Use this layout.

```text
sid_neuromod/
  pyproject.toml
  README.md
  LICENSE
  configs/
    synthetic_hidden_thermostat.yaml
    synthetic_change_detection.yaml
    elegans_default.yaml
    elegans_fast_debug.yaml
  src/
    sid_neuromod/
      __init__.py

      data/
        schema.py
        io.py
        validation.py
        preprocessing.py
        splitting.py

      features/
        filter_bank.py
        history.py
        behavior.py
        scaling.py
        feature_registry.py

      models/
        quadratic_score.py
        gaussian_nll.py
        targetwise.py
        two_head_neural.py
        calibration.py
        density.py

      readouts/
        score.py
        mean_gain_tail.py
        covariance.py
        factorization.py
        enrichment.py

      inference/
        hac.py
        sandwich.py
        simultaneous_bands.py
        bootstrap.py
        multiple_testing.py

      monitoring/
        pit.py
        channels.py
        eprocess.py
        changepoint.py

      baselines/
        var.py
        ridge_ar.py
        sindy.py
        garch_like.py

      synthetic/
        hidden_thermostat.py
        hidden_oscillator.py
        leverage.py
        directed_spillover.py
        measurement_noise.py
        change_scenarios.py
        oracles.py

      experiments/
        run_synthetic.py
        run_elegans_fit.py
        run_elegans_monitoring.py
        run_baselines.py
        make_report.py

      viz/
        heatmaps.py
        timescales.py
        pit_plots.py
        change_plots.py
        reports.py

      utils/
        rng.py
        logging.py
        config.py
        linalg.py
        arrays.py

  tests/
    unit/
      test_filter_bank.py
      test_quadratic_score.py
      test_readouts.py
      test_hac.py
      test_pit.py
      test_eprocess.py
      test_schema.py
    integration/
      test_hidden_thermostat_recovery.py
      test_change_detection.py
      test_measurement_noise.py
      test_simultaneous_bands.py
      test_elegans_smoke.py
    regression/
      expected_metrics/
        hidden_thermostat_seed0.json
        change_detection_seed0.json

  notebooks/
    00_sanity_check_synthetic.ipynb
    01_elegans_pit_calibration.ipynb
    02_readout_interpretation.ipynb

  scripts/
    run_all_synthetic.sh
    run_elegans_default.sh
    profile_fit.sh

  output/
    synthetic/
    elegans/
```

---

# 5. Data contracts

## 5.1 Required input for real neural data

Accept data in `.npz`, `.h5`, `.zarr`, or `.parquet`.

Minimum required fields:

```text
X: float array, shape [T, N]
timestamps_s: float array, shape [T]
neuron_ids: string array, shape [N]
```

Optional fields:

```text
behavior: float array, shape [T, B]
behavior_names: string array, shape [B]
stimulus: float array, shape [T, S]
stimulus_names: string array, shape [S]
worm_id: string array or scalar
condition_id: string array or scalar
is_valid: bool array, shape [T, N] or [T]
cell_annotations: table
connectome_annotations: table
neuropeptide_receptor_annotations: table
```

## 5.2 Required preprocessing invariants

The code must enforce:

```text
timestamps_s are strictly increasing.
X has shape [T, N].
No target sample may use future features.
Train/validation/test splits are contiguous or worm-blocked, not randomly interleaved by default.
Feature scaling is fit on train only.
Target scaling is fit on train only.
Filter states are reset or warmed only using pre-split past, never future data.
```

## 5.3 Output schemas

### Fit metadata

```text
fit_metadata.json
```

Required keys:

```json
{
  "dataset_id": "...",
  "run_id": "...",
  "git_commit": "...",
  "config_hash": "...",
  "random_seed": 0,
  "n_timepoints": 0,
  "n_neurons": 0,
  "prediction_horizon_s": 0.0,
  "timescales_s": [],
  "target_mode": "next_activity or delta_activity",
  "model_class": "quadratic_score",
  "ridge_lambda": 0.0,
  "hac_lags": 0,
  "train_interval": [],
  "calibration_interval": [],
  "test_interval": []
}
```

### Readout table

```text
readouts.parquet
```

Columns:

```text
target_id
source_id
timescale_s
channel
estimate
standard_error
z_score
p_value
q_value
ci_low
ci_high
significant
split_id
condition_id
feature_name
model_class
```

Channels must include at least:

```text
mean
gain_log_variance
variance
tail_high
tail_low
```

Optional later channels:

```text
conditional_covariance
mode_probability
latent_factor
entropy_production
```

### Change-event table

```text
change_events.parquet
```

Columns:

```text
event_id
alarm_time_s
lower_ci_time_s
upper_ci_time_s
channel
target_id
source_id
timescale_s
pre_estimate
post_estimate
delta_estimate
delta_standard_error
delta_z
delta_q_value
interpretation_label
```

### Diagnostics

```text
diagnostics.json
```

Required keys:

```json
{
  "pit_ks_stat": 0.0,
  "pit_ks_pvalue": 0.0,
  "pit_mean": 0.0,
  "pit_variance": 0.0,
  "invalid_variance_fraction": 0.0,
  "heldout_nll": 0.0,
  "baseline_var_heldout_nll": 0.0,
  "baseline_ridge_r2": 0.0,
  "sigma_ledger_max_deviation": 0.0,
  "downsample_readout_correlation": 0.0
}
```

---

# 6. Model hierarchy

Implement the project in three model levels. Do not start with neural networks.

## 6.1 Level 1: target-wise Gaussian conditional density

For each target neuron (i),

[
Y_i(t)\mid H_t
\sim
\mathcal N(\mu_i(t), v_i(t)).
]

Use natural parameters:

[
\eta_{1,i}(t)=w^{(1)\top}_i\psi(t),
]

[
\eta_{2,i}(t)=w^{(2)\top}_i\psi(t),
]

with

[
v_i(t)=-\frac{1}{2\eta_{2,i}(t)},
]

[
\mu_i(t)=-\frac{\eta_{1,i}(t)}{2\eta_{2,i}(t)}.
]

Here (\psi(t)) is the feature vector: intercept, source filter features, own-history features, behavior covariates, stimulus covariates, and optional interactions.

This is the “quadratic-(T)” conditional exponential-family model. The uploaded document gives a closed-form score-matching algorithm for this exact case and uses it throughout the pilot experiments so the theory is tested without optimization confounds. 

## 6.2 Level 2: stable deployment Gaussian model

The natural-parameter model can produce invalid positive (\eta_2) on some held-out histories. Implement a fallback model:

[
\mu_i(t)=a_i^\top \psi(t),
]

[
\log v_i(t)=b_i^\top \psi(t),
]

fit by Gaussian negative log likelihood with ridge/elastic-net regularization.

This model is not the pure closed-form estimator from the theory, but it is numerically stable. Use it only after the closed-form model is implemented and tested. All reports must state which model produced the final readouts.

## 6.3 Level 3: two-head neural model

Only after Levels 1–2 work, implement:

```text
shared or target-specific feature trunk
conditional density head
history-marginal head
centering consistency loss
```

The uploaded document argues for a disentangled/two-head architecture because naive joint score training can allocate capacity to the history marginal and lose the transition signal; the two-head design separates the conditional and marginal pieces. 

---

# 7. Feature engineering

## 7.1 Target definitions

Support both:

```text
target_mode = "next"
Y_i(t) = X_i(t + horizon)
```

and

```text
target_mode = "delta"
Y_i(t) = X_i(t + horizon) - X_i(t)
```

Default for neural data:

```text
target_mode = "delta"
horizon = one imaging frame or 1 second, whichever is scientifically appropriate
```

Run sensitivity analyses over:

```text
horizon_s ∈ {one frame, 1 s, 2 s, 5 s}
```

## 7.2 Source signals

For each neuron (j), allow source signal choices:

```text
raw_zscore
delta
positive_part
negative_part
deconvolved
behavior_residual
```

Default MVP:

```text
raw_zscore
delta
```

Optional later:

```text
deconvolved
positive_part
```

## 7.3 Exponential filters

For each source signal (z_j(t)), compute

[
f_{j,\tau}(t)
=============

\int_0^\infty \frac1{\tau}e^{-s/\tau}z_j(t-s),ds.
]

Use exact zero-order-hold updates for irregular timestamps:

[
f_{j,\tau}(t_{k+1})
===================

e^{-\Delta t_k/\tau} f_{j,\tau}(t_k)
+
\left(1-e^{-\Delta t_k/\tau}\right) z_j(t_k).
]

Default timescale grid:

```text
timescales_s = [0.5, 1, 2, 5, 10, 20, 45, 90, 180, 300]
```

For slow recordings or short experiments, use:

```text
min_timescale_s = 2 * median_frame_interval_s
max_timescale_s = min(0.25 * recording_duration_s, 600 s)
number_of_timescales = 8 to 12
grid = log-spaced
```

## 7.4 Behavior and stimulus covariates

Include behavior covariates in (\psi(t)), not as post-hoc regressions only.

Examples:

```text
velocity
angular_velocity
forward/reversal state
body curvature modes
pumping
egg-laying state
stimulus identity
stimulus derivative
odor/salt/light level
```

Run every real-data analysis in at least two modes:

```text
neural_only
neural_plus_behavior
```

A slow “neuromodulator” effect that disappears after behavior covariates are added should be reported as behavior-mediated or behavior-confounded, not as direct neuromodulation.

## 7.5 Feature normalization

For each split:

```text
fit feature mean and std on train only
apply to calibration/debias/test using train parameters
store scaler in fit artifact
```

Never compute scaling parameters on the full recording before splitting.

---

# 8. Closed-form quadratic score-matching estimator

Implement class:

```python
QuadraticScoreMatcher
```

Required methods:

```python
fit(Y: np.ndarray, Psi: np.ndarray, sigma: float, ridge: float) -> FitResult
predict_params(Psi: np.ndarray) -> eta1, eta2, mu, var
logpdf(Y: np.ndarray, Psi: np.ndarray) -> np.ndarray
cdf(Y: np.ndarray, Psi: np.ndarray) -> np.ndarray
sample(Psi: np.ndarray, n_samples: int, rng) -> np.ndarray
estimating_functions(Y, Psi, sigma) -> np.ndarray
```

## 8.1 Fit algorithm

Inputs:

```text
Y: shape [T]
Psi: shape [T, P]
sigma: denoising corruption level
ridge: nonnegative ridge regularization
```

If (\sigma>0):

```text
draw ζ ~ N(0, I)
Y_tilde = Y + sigma * ζ
g = (Y_tilde - Y) / sigma^2
U = concat(Psi, 2 * Y_tilde[:, None] * Psi)
A = U.T @ U / T
theta = - solve(A + ridge * I, U.T @ g / T)
```

If (\sigma=0):

```text
Y_tilde = Y
U = concat(Psi, 2 * Y[:, None] * Psi)
A = U.T @ U / T
c = concat(zeros(P), 2 * mean(Psi, axis=0))
theta = - solve(A + ridge * I, c)
```

This follows Algorithm 1 in the uploaded document. 

## 8.2 Prediction equations

Split:

```text
theta1 = theta[:P]
theta2 = theta[P:]
```

Then:

[
\eta_1(t)=\theta_1^\top\psi(t),
]

[
\eta_2(t)=\theta_2^\top\psi(t).
]

Effective variance:

[
v_{\mathrm{eff}}(t)=-\frac{1}{2\eta_2(t)}.
]

Corrected variance:

[
v(t)=v_{\mathrm{eff}}(t)-\sigma^2.
]

Mean:

[
\mu(t)=-\frac{\eta_1(t)}{2\eta_2(t)}.
]

Numerical requirements:

```text
if eta2 >= -eta2_min, clamp eta2 to -eta2_min for prediction only
if corrected variance <= var_min, clamp to var_min
record invalid_variance_fraction before clamping
```

Default:

```text
eta2_min = 1e-6
var_min = 1e-6
```

If invalid variance fraction exceeds:

```text
1% warning
5% fail model acceptance for that target
```

## 8.3 Denoising sigma ledger

Run the estimator for:

```text
sigma ∈ {0, 0.25 * sd(Y), 0.5 * sd(Y), 1.0 * sd(Y)}
```

After subtracting (\sigma^2) from variance, gain/variance readouts should be stable. The uploaded document treats this as a cross-(\sigma) consistency diagnostic. 

---

# 9. Readout implementation

Implement module:

```text
readouts/mean_gain_tail.py
```

For each target (i), source (j), and timescale (\tau), compute:

```text
D_mean[i, j, tau]
D_variance[i, j, tau]
D_gain_log_variance[i, j, tau]
D_tail_high[i, j, tau]
D_tail_low[i, j, tau]
```

## 9.1 Mapping from feature index to source-timescale

Maintain a feature registry:

```python
FeatureSpec(
    name="filter",
    source_id="AVA",
    signal_kind="raw_zscore",
    timescale_s=45.0,
    column_index=123
)
```

Every readout must trace back to a feature spec.

## 9.2 Analytic derivatives for natural-parameter Gaussian

Let

[
\eta_1=\theta_1^\top\psi,
\qquad
\eta_2=\theta_2^\top\psi,
\qquad
v_{\mathrm{eff}}=-\frac{1}{2\eta_2},
\qquad
\mu=\eta_1v_{\mathrm{eff}}.
]

For feature coordinate (k),

[
\frac{\partial v_{\mathrm{eff}}}{\partial \psi_k}
=================================================

2v_{\mathrm{eff}}^2\theta_{2,k}.
]

For corrected variance (v=v_{\mathrm{eff}}-\sigma^2),

[
\frac{\partial v}{\partial \psi_k}
==================================

2v_{\mathrm{eff}}^2\theta_{2,k}.
]

Mean derivative:

[
\frac{\partial \mu}{\partial \psi_k}
====================================

v_{\mathrm{eff}}\theta_{1,k}
+
2\eta_1v_{\mathrm{eff}}^2\theta_{2,k}.
]

Log-variance derivative:

[
\frac{\partial \log v}{\partial \psi_k}
=======================================

\frac{1}{v}
\frac{\partial v}{\partial \psi_k}.
]

For a high-tail threshold (q_i^{high}), usually the 90th or 95th percentile of train (Y_i),

[
P(Y_i>q)=1-\Phi(a),
\qquad
a=\frac{q-\mu}{\sqrt v}.
]

Then

[
\frac{\partial P(Y_i>q)}{\partial \psi_k}
=========================================

\phi(a)
\left[
\frac{1}{\sqrt v}
\frac{\partial \mu}{\partial \psi_k}
+
\frac{q-\mu}{2v^{3/2}}
\frac{\partial v}{\partial \psi_k}
\right].
]

Low-tail probability is analogous:

[
P(Y_i<q)=\Phi(a),
]

[
\frac{\partial P(Y_i<q)}{\partial \psi_k}
=========================================

\phi(a)
\left[
-\frac{1}{\sqrt v}
\frac{\partial \mu}{\partial \psi_k}
------------------------------------

\frac{q-\mu}{2v^{3/2}}
\frac{\partial v}{\partial \psi_k}
\right].
]

## 9.3 Averaging over histories

Readouts can be local or averaged.

Support:

```text
readout_mode = "global_average"
readout_mode = "state_conditioned"
readout_mode = "time_local"
```

Default:

[
D^{\phi}_{i\leftarrow j}(\tau)
==============================

\frac{1}{T_{\mathrm{test}}}
\sum_{t\in \mathrm{test}}
\frac{\partial}{\partial f_{j,\tau}(t)}
\mathbb E[\phi(Y_i)\mid H_t].
]

For state-conditioned readouts, average over subsets:

```text
forward locomotion
reversal
roaming
dwelling
stimulus-on
stimulus-off
pre-change
post-change
```

## 9.4 Neuromodulator summary scores

Compute these derived scores.

Fast mean mass:

[
M^{fast}_{i\leftarrow j}
========================

\sum_{\tau\le \tau_0}
\left|D^{mean}_{i\leftarrow j}(\tau)\right|.
]

Slow distributional mass:

[
G^{slow}_{i\leftarrow j}
========================

\sum_{\tau>\tau_0}
\left(
|D^{gain}*{i\leftarrow j}(\tau)|
+
|D^{tail}*{i\leftarrow j}(\tau)|
+
|D^{cov}_{i\leftarrow j}(\tau)|
\right).
]

Default:

```text
tau0 = 5 s or 10 s
```

Candidate neuromodulator index:

[
\mathrm{NMI}_{i\leftarrow j}
============================

\frac{G^{slow}*{i\leftarrow j}}
{G^{slow}*{i\leftarrow j}+M^{fast}_{i\leftarrow j}+\epsilon}.
]

Interpretation:

```text
NMI near 0: fast mean-drive dominated
NMI near 1: slow distributional/gain dominated
```

Do not treat NMI as proof of a specific neuromodulator. It is a ranking statistic.

---

# 10. Inference: HAC covariance and simultaneous bands

Implement:

```text
inference/hac.py
inference/sandwich.py
inference/simultaneous_bands.py
```

The document uses HAC/Newey-West long-run covariances, cross-target covariance sums, and simultaneous sup-(t) bands over directed multilag graphs. 

## 10.1 Per-target sandwich covariance

The estimator solves:

[
A_T\hat\theta+c_T=0.
]

Let (\xi_t(\theta^*)) be the per-sample estimating function.

Estimate long-run covariance:

[
S
=

\Gamma_0
+
\sum_{\ell=1}^{L_{\mathrm{HAC}}}
w_\ell(\Gamma_\ell+\Gamma_\ell^\top),
]

where

[
w_\ell=1-\frac{\ell}{L_{\mathrm{HAC}}+1}.
]

Then

[
\widehat{\operatorname{Cov}}(\hat\theta)
========================================

A_T^{-1} S A_T^{-\top}/T.
]

Default HAC lag:

```text
hac_lags = max(20, ceil(4 * (T / 100) ** (2/9)))
```

Also expose config override.

## 10.2 Readout covariance

For readout vector (r(\theta)),

[
\widehat{\operatorname{Cov}}(r)
===============================

J
\widehat{\operatorname{Cov}}(\hat\theta)
J^\top,
]

where (J=\partial r/\partial \theta).

Implementation priority:

1. central finite-difference Jacobian;
2. analytic Jacobian later for speed;
3. unit tests must compare finite-difference and analytic results when analytic is added.

## 10.3 Cross-target covariance

For targets (i) and (k), estimate cross-HAC covariance:

[
S_{ik}
======

\sum_{\ell=-L}^{L}
w_{|\ell|}
\operatorname{Cov}(\xi^{(i)}*t,\xi^{(k)}*{t-\ell}).
]

Then

[
\widehat{\operatorname{Cov}}(\hat\theta_i,\hat\theta_k)
=======================================================

A_i^{-1}S_{ik}A_k^{-\top}/T.
]

This is required for valid simultaneous bands across multiple target neurons.

## 10.4 Sup-(t) simultaneous bands

Stack readouts into vector (r), covariance (C).

Draw:

[
Z^{(b)}\sim \mathcal N(0,\operatorname{Corr}(C)).
]

Compute:

[
q_{1-\alpha}
============

\operatorname{quantile}_{1-\alpha}
\left(
\max_m |Z_m|
\right).
]

Band:

[
r_m
\pm
q_{1-\alpha}\sqrt{C_{mm}}.
]

Default:

```text
alpha = 0.05
n_mc = 10000
```

Flag significant if the simultaneous band excludes zero.

---

# 11. Sequential monitoring and changepoint detection

Implement:

```text
monitoring/pit.py
monitoring/channels.py
monitoring/eprocess.py
monitoring/changepoint.py
```

The uploaded document’s monitoring layer uses PIT/Rosenblatt residuals, bounded martingale-difference channels, e-processes, dead-band correction for calibration error, and changepoint confidence sets. 

## 11.1 Splits

For monitoring, use four contiguous splits:

```text
train: fit conditional density
calibration: PIT ECDF recalibration
debias: estimate residual channel offsets
stream: monitoring interval
```

Default split fractions:

```text
train = 50%
calibration = 20%
debias = 10%
stream = 20%
```

For multiple worms, use worm-blocked splits when the scientific question is generalization across animals.

## 11.2 PIT residuals

For each target (i),

[
U_{i,t}
=======

\hat F_i(Y_{i,t}\mid H_t).
]

For Gaussian:

[
U_{i,t}
=======

\Phi\left(
\frac{Y_{i,t}-\hat\mu_i(t)}
{\sqrt{\hat v_i(t)}}
\right).
]

Clamp:

```text
U = clip(U, 1e-6, 1 - 1e-6)
```

Recalibrate through empirical CDF from the calibration split using randomized ranks.

## 11.3 Monitoring channels

Implement channel functions:

Mean channel:

[
d^{mean}*{i,t}=2(U*{i,t}-1/2).
]

Dispersion channel:

[
d^{disp}*{i,t}=6U*{i,t}^2-6U_{i,t}+1.
]

Serial channel:

[
d^{serial}*{i,t}=d^{disp}*{i,t}d^{disp}_{i,t-1}.
]

Lag-kernel channel for source filter (g_{j,\tau}(H_t)):

[
d^{lag}_{i,j,\tau,t}
====================

d^{disp}*{i,t}
\cdot
\operatorname{clip}
\left(
\frac{g*{j,\tau}(H_t)-\bar g}{s_g},
-1,
1
\right).
]

Optional group channels:

```text
mean over all targets
mean over neuron class
mean over candidate neuromodulator target set
```

## 11.4 E-process implementation

For a bounded channel (d_t\in[-1,1]), compute cumulative sums over dyadic windows.

Use a fixed grid of betting parameters:

```text
lambda_grid = [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0]
```

For a window of length (n) and sum (S), under zero drift:

[
M_\lambda
=========

\exp(\lambda S-\lambda^2 n/2).
]

Average over (\lambda):

[
M=\frac1{|\Lambda|}\sum_{\lambda\in\Lambda}M_\lambda.
]

This grid-mixture implementation is simple and valid.

## 11.5 Dead-band correction

On the debias split, estimate channel offset (\hat b_c). Define

[
\epsilon_c
==========

\sqrt{\frac{2\log(2K/\delta)}{n_{db}}},
]

where (K) is the number of monitored channels.

For positive (\lambda), use:

[
M_\lambda^+
===========

\exp(\lambda(S-\epsilon n)-\lambda^2n/2).
]

For negative (\lambda), use:

[
M_\lambda^-
===========

\exp(\lambda(S+\epsilon n)-\lambda^2n/2).
]

Or equivalently center by the worst-case drift for each sign.

Default:

```text
delta = 0.01
alpha = 0.01
alarm_threshold = 1 / alpha
```

If using a maximum over channels/windows rather than an average mixture, apply Bonferroni or empirical null calibration. For formal validity, default to an average mixture across channels and windows.

## 11.6 Alarm and localization

An alarm occurs when:

[
M_t \ge 1/\alpha.
]

The upper confidence endpoint is the alarm time, as in the document’s Proposition 13.1 logic. 

After an alarm:

1. define pre-window and post-window;
2. refit the model or reuse baseline with local post-fit;
3. compute pre/post readouts;
4. report:

[
\Delta D^{\phi}_{i\leftarrow j}(\tau)
=====================================

## D^{\phi,post}_{i\leftarrow j}(\tau)

D^{\phi,pre}_{i\leftarrow j}(\tau).
]

The required “what changed?” report is the significant entries in (\Delta D).

---

# 12. Synthetic benchmark suite

Do not analyze real data until the synthetic suite passes.

Implement these experiments.

## 12.1 E1: hidden thermostat / mean-blind variance memory

System:

[
w_{t+1}=aw_t+b\xi_t,
]

[
x_t=\exp(w_t/2)\epsilon_t,
]

with default:

```text
a = 0.9
b = 0.4
T = 200000
```

Expected result:

```text
VAR/Ridge mean R2 near 0
raw autocorrelation near 0
variance/gain readout significant over multiple lags/timescales
estimated kernel strongly correlated with Kalman oracle
```

Acceptance:

```text
mean baseline R2 < 1e-3
variance kernel correlation with oracle > 0.90
at least 8 of 12 lag/timescale variance readouts significant under sup-t band
sigma-ledger max relative deviation < 20%
```

This experiment is essential because it demonstrates the core reason the method exists: dynamics invisible to conditional-mean methods but visible in the conditional law. 

## 12.2 E2: measurement noise and leakage

Implement OU or AR process with additive measurement noise.

Test:

```text
multi-scale noise estimation
linear attenuation correction
nonlinear leakage artifact
```

Acceptance:

```text
estimated measurement noise R within 10% of truth
linear corrected coefficients within 10% of truth
nonlinear noisy variance curve matches leakage prediction correlation > 0.90
```

The document warns that input noise plus nonlinear mean dynamics can manufacture apparent variance dynamics, so real neural analyses must audit this. 

## 12.3 E3: conditional-law change detection

Scenarios:

```text
null: no change
scale change: variance changes, mean unchanged
memory change: latent a changes, marginal variance approximately stable
mean change: simple drift change
```

Acceptance:

```text
null false alarm rate <= alpha + delta + Monte Carlo tolerance
scale change detection power >= 80%
memory change detection power >= 60% in long runs
mean-only CUSUM fails on variance-only/memory-only changes more often than e-process
```

## 12.4 E4: hidden oscillator / memory kernel

System: scalar observed variable coupled to hidden 2D rotation.

Expected:

```text
fitted kernel recovers damped oscillatory shape
ESPRIT/Hankel recovers hidden modulus and frequency approximately
filter-bank readouts stable under downsampling
```

Acceptance:

```text
kernel relative L2 error vs oracle < 0.15
estimated hidden modulus error < 0.10
estimated frequency error < 0.15 rad
```

## 12.5 E5: channel-restricted Hankel rank

Build past/future feature blocks for:

```text
mean channel: raw x
variance channel: z = log(x^2 + eps)
```

Acceptance:

```text
mean channel rank = 0 for hidden thermostat
variance channel rank = number of latent volatility factors
leading CCA variate correlates with Kalman state > 0.90 in one-factor setting
```

The uploaded document reports that the leading variance-channel canonical variate recovers the filtering state in the one-factor case. 

## 12.6 E6: directed spillover and simultaneous bands

Two latent volatility factors with one directional structural coupling.

Expected:

```text
predictive directed kernels can be nonzero in directions that are not direct structural wires
simultaneous bands control family-wise coverage
```

Acceptance:

```text
empirical 95% simultaneous coverage between 90% and 98% over at least 50 seeds
all oracle-nonzero kernels detected at sufficient T
```

The developer must understand that predictive directedness is not the same as structural directedness under partial observation. The document explicitly warns that structural non-influence does not imply predictive non-influence and that structural claims require extra assumptions. 

## 12.7 E7: optional irreversibility

Implement later, not MVP.

Acceptance:

```text
reversible Gaussian control EP near 0
rotating irreversible linear system EP positive
DV certificate lower bound positive on leverage alternative
```

---

# 13. Real C. elegans analysis workflow

## 13.1 Pre-analysis checklist

For each dataset:

```text
Confirm frame rate and timestamp regularity.
Confirm neuron identity confidence.
Confirm missingness pattern.
Confirm whether traces are raw fluorescence, ΔF/F, z-scored, or deconvolved.
Confirm behavior/stimulus covariates.
Confirm experimental condition labels.
Confirm whether multiple worms are available.
Confirm whether optogenetic, mutant, or pharmacological perturbations are present.
```

## 13.2 Preprocessing

Required steps:

```text
remove unusable frames
handle missing neurons
z-score per neuron using train interval only
optionally detrend slow photobleaching
optionally regress out motion artifacts
construct behavior covariates
construct filter-bank features causally
```

Do not interpolate target values across long gaps. For missing source values, either:

```text
drop affected rows
```

or

```text
decay filter state without innovation and include missingness indicator
```

The choice must be logged.

## 13.3 Fit protocol

Default:

```text
for each target neuron:
    build Psi from all source filters + behavior/stimulus + intercept
    fit QuadraticScoreMatcher
    compute PIT on heldout
    compute readouts
    compute HAC/sup-t bands
```

For large (N), full all-to-all features may be too large. Use staged fitting:

### Stage A: screening

For each target, screen source-timescale features by:

```text
univariate correlation with target residual squared
univariate mean prediction
known anatomy/receptor candidates
stability across folds
```

Keep:

```text
top K = 100 to 500 source-timescale features per target
always include own-history features
always include behavior/stimulus covariates
```

### Stage B: final model

Fit target-wise multivariate model on selected features.

### Stage C: sensitivity

Repeat with:

```text
larger K
neural_only vs neural_plus_behavior
raw vs deconvolved if available
different prediction horizons
different timescale grid
```

## 13.4 Baselines

Run:

```text
VAR / ridge autoregression
SINDy
mean-only Gaussian model
variance-only model with no neural source filters
behavior-only model
shuffle controls
```

Required baseline comparisons:

```text
held-out negative log likelihood
PIT calibration
mean R2
variance/gain readout detection
change detection delay/power
support overlap with slow distributional readouts
```

The purpose is not to claim VAR/SINDy are bad. It is to show which effects are mean-drive effects and which are only visible in the conditional law.

---

# 14. Result interpretation rules

## 14.1 Channel interpretation

Use this mapping.

| Channel                  | What it means                                        | Biological interpretation                  |
| ------------------------ | ---------------------------------------------------- | ------------------------------------------ |
| `mean`                   | Source changes target’s future mean/drift            | Fast predictive drive; VAR-comparable      |
| `gain_log_variance`      | Source changes target’s future conditional variance  | Candidate gain/excitability modulation     |
| `tail_high`              | Source changes probability of high target activation | Threshold/burst modulation                 |
| `tail_low`               | Source changes probability of suppression            | Inhibitory threshold/state effect          |
| `conditional_covariance` | Source changes target-target coordination            | Synchrony/network-state modulation         |
| `mode_probability`       | Source changes latent state probability              | Brain-state or behavioral-state modulation |

## 14.2 Timescale interpretation

Use this default language:

```text
tau <= 2 s: very fast predictive interaction
2 s < tau <= 10 s: fast/intermediate circuit interaction
10 s < tau <= 60 s: slow modulatory candidate
tau > 60 s: very slow state/modulatory candidate
```

Do not hard-code “slow = peptide” or “fast = synapse.” C. elegans extrasynaptic/dense-core-vesicle signaling can occur on short timescales in some settings, so timescale is evidence, not proof. ([Nature][7])

## 14.3 Neuromodulator-candidate label

A source-target-timescale factor can be labeled:

```text
candidate neuromodulatory distributional factor
```

only if it satisfies at least three of:

```text
slow timescale mass
distributional channel stronger than mean channel
broad target footprint
state dependence
receptor/neuropeptide/amine annotation enrichment
perturbation sensitivity
improved held-out likelihood or PIT calibration over mean-only baseline
replication across worms
```

It may be labeled:

```text
strong neuromodulator candidate
```

only if perturbation or molecular annotation evidence is present.

## 14.4 Language that must be avoided

Do not write:

```text
Neuron A releases serotonin onto neuron B.
This proves peptide X causes the change.
This is the anatomical circuit.
This is a synapse.
```

Write:

```text
Past activity of source A predicts a slow gain-channel change in target B.
The effect is consistent with neuromodulatory gain control.
The factor is enriched for known peptide/receptor annotations.
Perturbation data support a causal neuromodulatory interpretation.
```

---

# 15. Change-detection analysis plan

For each experiment or recording:

## 15.1 Baseline fit

Choose baseline interval:

```text
pre-stimulus
pre-perturbation
wild-type reference
early recording stable segment
```

Fit (\hat\rho_0(y\mid H_t)).

## 15.2 Monitor

Compute PIT channels over the monitoring interval.

Report:

```text
e-process trace
alarm time
dominant channel
dominant source/target/timescale if lag channel
null-calibrated threshold
dead-band epsilon
```

## 15.3 Localize and explain

After alarm:

```text
pre_window = interval before estimated change
post_window = interval after estimated change
fit/readout pre
fit/readout post
compute delta readouts
apply simultaneous bands
rank significant deltas
```

Required “what changed” table:

| Rank | Target | Source | Timescale | Channel | Pre | Post | Delta | Simultaneous CI | Interpretation |
| ---: | ------ | ------ | --------: | ------- | --: | ---: | ----: | --------------- | -------------- |

Example interpretation:

```text
The alarm was driven by slow gain-channel residuals. Post-change, source AVA showed increased 45–90 s gain influence on a broad set of targets, while fast mean-channel effects were stable. This is consistent with a slow state/gain reconfiguration rather than a new fast mean-drive edge.
```

---

# 16. Quality gates and acceptance criteria

## Gate 1: unit tests pass

Required:

```bash
pytest tests/unit -q
```

Coverage target:

```text
>= 90% for core math modules
>= 80% overall
```

## Gate 2: synthetic smoke tests pass

Required:

```bash
pytest tests/integration -q -m "not slow"
```

Must pass:

```text
hidden thermostat recovery
filter-bank causality/no leakage
quadratic estimator sanity
PIT uniformity
e-process null validity smoke
```

## Gate 3: full synthetic validation

Required before publishing real-data claims:

```bash
bash scripts/run_all_synthetic.sh
```

Minimum acceptance:

```text
E1 hidden thermostat: pass
E2 measurement noise: pass
E3 null false alarm: pass
E4 hidden oscillator: pass
E6 simultaneous coverage: pass
```

## Gate 4: real-data diagnostics

For each real run:

```text
invalid variance fraction < 1% preferred, < 5% maximum
PIT mean in [0.45, 0.55]
PIT variance close to 1/12; tolerance ±25% initially
held-out NLL better than behavior-only model
readouts stable across at least two adjacent timescale grids
no major result depends on a single worm unless reported as exploratory
```

## Gate 5: biological interpretation review

Before reporting neuromodulator claims:

```text
confirm behavior covariate sensitivity
confirm downsampling sensitivity
confirm shuffle controls
confirm source/target identity confidence
confirm annotation enrichment is not circular
confirm predictive-not-structural language
```

---

# 17. Unit test specification

## 17.1 Filter bank tests

### `test_filter_bank_constant_signal`

Input:

```text
z_t = 1 for all t
```

Expected:

```text
filter approaches 1
monotone from initialized 0
final error < exp(-duration/tau) + tolerance
```

### `test_filter_bank_no_future_leakage`

Construct a signal with a jump at (t^*). Verify:

```text
features before t* are unchanged by values after t*
```

### `test_filter_bank_irregular_grid`

Generate irregular timestamps and compare recursive implementation to direct numerical zero-order-hold integration.

Acceptance:

```text
max absolute error < 1e-10 for float64 direct equivalent
```

## 17.2 Quadratic score estimator tests

### `test_quadratic_score_recovers_gaussian_moments`

Simulate:

[
Y=a^\top\psi+\sqrt v\epsilon.
]

Fit model.

Acceptance:

```text
mean prediction R2 > 0.95
median variance estimate within 10%
```

### `test_sigma_ledger`

Fit with multiple (\sigma).

Acceptance:

```text
corrected variance estimates agree within 10%
```

### `test_invalid_variance_warning`

Construct a degenerate fit that produces invalid (\eta_2).

Acceptance:

```text
warning emitted
invalid_variance_fraction recorded
clamped predictions finite
```

## 17.3 Readout tests

### `test_mean_readout_matches_known_linear_coefficient`

For homoscedastic Gaussian linear model:

[
Y=\beta f+\epsilon.
]

Acceptance:

```text
D_mean close to beta
D_gain near 0
```

### `test_gain_readout_matches_known_log_variance_coefficient`

For:

[
Y\sim N(0,\exp(c f)).
]

Acceptance:

```text
D_gain close to c
D_mean near 0
```

### `test_tail_derivative_finite_difference`

Compare analytic high-tail derivative to finite difference.

Acceptance:

```text
relative error < 1e-4
```

## 17.4 HAC/inference tests

### `test_hac_white_noise_matches_sample_covariance`

For iid estimating functions:

```text
HAC with lag 0 equals sample covariance
```

### `test_hac_psd_or_near_psd`

Acceptance:

```text
minimum eigenvalue > -1e-8
```

### `test_simultaneous_band_coverage_independent_gaussian`

Monte Carlo check.

Acceptance:

```text
empirical coverage within ±3% of nominal for simple iid case
```

## 17.5 PIT tests

### `test_pit_uniform_correct_model`

Simulate from fitted Gaussian.

Acceptance:

```text
KS p-value usually > 0.01 across seeds
PIT mean near 0.5
```

### `test_pit_detects_variance_misspecification`

Fit mean-only constant variance to heteroscedastic data.

Acceptance:

```text
dispersion channel mean differs from null
```

## 17.6 E-process tests

### `test_eprocess_supermartingale_smoke`

Under iid bounded zero-mean channels:

```text
false alarm rate <= 2 * alpha over enough seeds
```

Use a slow test marker.

### `test_eprocess_detects_positive_drift`

Under drift:

```text
alarm in majority of runs
median delay finite and reasonable
```

## 17.7 Schema tests

### `test_schema_rejects_bad_timestamps`

Must reject:

```text
nonmonotonic timestamps
duplicate timestamps
NaN timestamps
```

### `test_schema_rejects_shape_mismatch`

Must reject inconsistent shapes.

---

# 18. Integration tests

## 18.1 Hidden thermostat recovery

Command:

```bash
python -m sid_neuromod.experiments.run_synthetic \
  --config configs/synthetic_hidden_thermostat.yaml
```

Expected output:

```text
output/synthetic/hidden_thermostat/readouts.parquet
output/synthetic/hidden_thermostat/diagnostics.json
output/synthetic/hidden_thermostat/figures/kernel_recovery.png
```

Pass criteria:

```text
diagnostics["mean_r2"] < 1e-3
diagnostics["variance_kernel_corr"] > 0.90
diagnostics["n_significant_gain_lags"] >= 8
```

## 18.2 Change detection

Command:

```bash
python -m sid_neuromod.experiments.run_synthetic \
  --config configs/synthetic_change_detection.yaml
```

Pass criteria:

```text
null false alarms <= configured tolerance
scale-change detection power >= 80%
memory-change detection power reported
```

## 18.3 Elegans smoke test

Use a tiny mock C. elegans-like dataset.

Pass criteria:

```text
pipeline completes
outputs schemas valid
no future leakage
figures generated
```

---

# 19. Real-data statistical controls

For each real dataset, run these controls.

## 19.1 Circular source shift

Circularly shift each source neuron independently by a large random offset.

Expected:

```text
slow source-target readouts collapse toward null
significant edges greatly reduced
```

## 19.2 Time reversal

Reverse time and rerun selected analyses.

Expected:

```text
predictive readouts change substantially
irreversibility or direction-sensitive claims should not be symmetric unless system is reversible
```

## 19.3 Behavior-only control

Fit model with behavior/stimulus covariates but no neural source filters.

Expected:

```text
some predictive power remains
claimed neural readouts should improve over this control
```

## 19.4 Mean-only control

Fit model where (v_i(t)) is constant.

Expected:

```text
gain/tail effects disappear by construction
heldout PIT dispersion worsens if distributional dynamics are real
```

## 19.5 Downsampling control

Rerun at:

```text
native frame rate
2x downsample
4x downsample if enough data
```

Expected:

```text
filter-bank readouts in physical timescale units remain similar
dense-lag coefficients are less stable
```

This is one of the central claims of the continuous-time/filter-bank layer. 

## 19.6 Split stability

Run at least:

```text
3 contiguous train/test splits
or leave-one-worm-out splits
```

Report:

```text
readout sign stability
rank correlation of top effects
overlap of significant source-target-timescale effects
```

---

# 20. Real-data figures and reports

The package must produce these automatically.

## 20.1 Calibration figures

```text
PIT histogram per target group
PIT QQ plot
dispersion channel over time
heldout NLL comparison bar plot
invalid variance fraction per target
```

## 20.2 Readout figures

```text
source-target heatmap at selected timescales
timescale spectrum for each source or target
mean vs gain scatter
fast mean mass vs slow distributional mass
top slow-gain sources
top slow-tail sources
```

## 20.3 Change figures

```text
e-process trace with threshold
alarm time overlay on behavior/neural summary
pre/post delta heatmap
top changed effects with simultaneous CIs
```

## 20.4 Neuromodulator candidate report

For each candidate factor:

```text
factor_id
dominant sources
dominant targets
dominant channel
dominant timescale range
behavior dependence
condition dependence
annotation enrichment
perturbation evidence if available
warnings/limitations
```

---

# 21. Factorization of slow distributional effects

Implement after basic readouts work.

Given slow gain tensor:

[
G_{i,j,\tau}=D^{gain}_{i\leftarrow j}(\tau),
]

fit low-rank decomposition:

[
G_{i,j,\tau}
\approx
\sum_{a=1}^{r}
T_{i,a}S_{j,a}K_{a,\tau}.
]

MVP methods:

```text
CP decomposition via tensorly
or matricize [target] x [source,timescale] and use SVD/NMF
```

Rank selection:

```text
cross-validated reconstruction error
permutation null
stability across splits
```

Report:

```text
source loadings
target loadings
timescale kernel
channel composition
annotation enrichment
```

Biological interpretation:

```text
A factor with broad target loading, slow K_tau, strong gain/tail/covariance channel, and receptor/source enrichment is a candidate latent neuromodulatory factor.
```

---

# 22. Annotation enrichment

Optional but important for biological interpretation.

Inputs:

```text
source neuron peptide/amine expression table
target neuron receptor expression table
synaptic/gap-junction connectome table
neuropeptide-receptor connectome table
cell class labels
```

Tests:

```text
Are top slow-gain sources enriched for known peptide/amine neurons?
Are top targets enriched for receptors associated with those sources?
Are slow distributional effects less aligned with wired synapses than fast mean effects?
```

Statistics:

```text
hypergeometric enrichment
permutation over neuron labels
degree-preserving permutation if using networks
Benjamini-Hochberg FDR
```

Report enrichment only if annotation tables are supplied and documented.

---

# 23. Performance and scaling plan

## 23.1 Complexity problem

For (N) neurons and (K) timescales, all source filters create (N\times K) neural features per target. Full target-wise fitting is roughly:

```text
N target fits
P ≈ N*K + behavior + intercept
solve size ≈ 2P x 2P per target
```

For (N=300), (K=10), (P\approx3000), so each target solve can become expensive.

## 23.2 MVP scaling

Support:

```text
N <= 100 full all-to-all
N > 100 screened source-timescale features
```

Implement:

```text
parallel target fitting with joblib or multiprocessing
memory-mapped feature arrays
float32 storage, float64 solves
chunked readout computation
```

## 23.3 Screening strategy

For each target:

```text
always keep own filters
always keep behavior/stimulus covariates
keep known candidate source neurons if annotations provided
add top K features by univariate score
```

Default:

```text
K_screen = 300
```

Log all screened-out features so the analysis is auditable.

---

# 24. Configuration template

Example `configs/elegans_default.yaml`:

```yaml
dataset:
  path: data/elegans/example.zarr
  dataset_id: elegans_example
  time_column: timestamps_s
  activity_key: X
  neuron_ids_key: neuron_ids
  behavior_key: behavior
  stimulus_key: stimulus

splits:
  mode: contiguous
  train_fraction: 0.5
  calibration_fraction: 0.2
  debias_fraction: 0.1
  test_fraction: 0.2

preprocessing:
  activity_transform: zscore_train
  detrend: true
  missing_policy: drop_target_decay_source
  min_valid_fraction_per_neuron: 0.8

target:
  mode: delta
  horizon_s: 1.0

features:
  source_signals:
    - raw_zscore
    - delta
  timescales_s: [0.5, 1, 2, 5, 10, 20, 45, 90, 180, 300]
  include_behavior: true
  include_stimulus: true
  include_own_history: true
  source_screening:
    enabled: true
    max_features_per_target: 300

model:
  class: quadratic_score
  ridge: auto
  sigma_grid: [0.0, 0.25, 0.5, 1.0]
  eta2_min: 1.0e-6
  var_min: 1.0e-6

inference:
  hac_lags: auto
  alpha: 0.05
  simultaneous_family: target_source_timescale_channel
  n_mc_sup_t: 10000
  fdr_method: benjamini_hochberg

monitoring:
  enabled: true
  alpha: 0.01
  delta: 0.01
  channels:
    - mean
    - dispersion
    - serial
    - lag_dispersion
  dyadic_windows: [16, 32, 64, 128, 256, 512, 1024]
  lambda_grid: [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0]

baselines:
  run_var: true
  run_ridge_ar: true
  run_sindy: false

outputs:
  output_dir: output/elegans/default
  save_figures: true
  save_model: true
  save_readouts: true
```

---

# 25. Developer task breakdown

## Milestone 0: infrastructure

Deliverables:

```text
pyproject.toml
package skeleton
config loader
logging
random seed utility
CI workflow
pytest setup
```

Acceptance:

```text
pip install -e . works
pytest empty suite works
ruff/mypy basic checks work
```

## Milestone 1: synthetic systems

Deliverables:

```text
hidden thermostat simulator
hidden oscillator simulator
directed spillover simulator
measurement noise simulator
change scenarios
oracle functions where available
```

Acceptance:

```text
simulators produce deterministic outputs under same seed
basic moment checks pass
```

## Milestone 2: feature bank

Deliverables:

```text
exponential filter bank
feature registry
causal history construction
scaling
missing-data handling
```

Acceptance:

```text
filter-bank unit tests pass
no-future-leakage tests pass
```

## Milestone 3: quadratic score model

Deliverables:

```text
QuadraticScoreMatcher
prediction/cdf/logpdf
sigma ledger
fit artifacts
```

Acceptance:

```text
Gaussian sanity tests pass
sigma-ledger tests pass
hidden thermostat readout begins to recover variance kernel
```

## Milestone 4: readouts

Deliverables:

```text
mean/gain/variance/tail readouts
feature-to-source-timescale mapping
readouts.parquet writer
```

Acceptance:

```text
analytic derivative tests pass
finite-difference derivative tests pass
```

## Milestone 5: inference

Deliverables:

```text
HAC covariance
sandwich covariance
cross-target covariance
sup-t bands
multiple-testing utilities
```

Acceptance:

```text
coverage tests pass
simultaneous bands reproduce synthetic expectations
```

## Milestone 6: monitoring

Deliverables:

```text
PIT transform
ECDF recalibration
channels
dead-band e-process
dyadic windows
change report
```

Acceptance:

```text
null false alarm simulation passes
scale/memory change simulations detect
```

## Milestone 7: baselines

Deliverables:

```text
VAR baseline
ridge autoregression baseline
optional PySINDy baseline
baseline report
```

Acceptance:

```text
hidden thermostat shows mean baselines fail while gain readout succeeds
```

## Milestone 8: real C. elegans pipeline

Deliverables:

```text
data loaders
schema validation
end-to-end fit command
report generation
figures
```

Acceptance:

```text
mock elegans smoke test passes
one real dataset run completes
diagnostics generated
```

## Milestone 9: biological interpretation layer

Deliverables:

```text
factorization
annotation enrichment
neuromodulator candidate report
pre/post change explanation
```

Acceptance:

```text
outputs ranked candidate factors
enrichment uses permutation/FDR
interpretation report follows required language
```

---

# 26. Commands the developer should support

## Run synthetic benchmark

```bash
python -m sid_neuromod.experiments.run_synthetic \
  --config configs/synthetic_hidden_thermostat.yaml
```

## Run all synthetic validations

```bash
bash scripts/run_all_synthetic.sh
```

## Fit C. elegans dataset

```bash
python -m sid_neuromod.experiments.run_elegans_fit \
  --config configs/elegans_default.yaml
```

## Run monitoring

```bash
python -m sid_neuromod.experiments.run_elegans_monitoring \
  --config configs/elegans_default.yaml \
  --baseline-interval 0 1800 \
  --monitor-interval 1800 3600
```

## Generate report

```bash
python -m sid_neuromod.experiments.make_report \
  --run-dir output/elegans/default
```

---

# 27. Report template

The final report should have these sections.

```text
1. Dataset summary
2. Preprocessing and split summary
3. Model and feature configuration
4. Calibration diagnostics
5. Held-out predictive performance
6. Mean-channel functional connectome
7. Gain/tail distributional connectome
8. Timescale decomposition
9. Candidate neuromodulatory factors
10. Change detection results
11. What changed: pre/post readout deltas
12. Baseline comparison with VAR/SINDy
13. Control analyses
14. Limitations and caveats
15. Machine-readable artifact inventory
```

Required caveat section:

```text
These are predictive distributional effects, not direct structural synapses.
Slow gain/tail/covariance effects are candidate neuromodulatory signatures, not molecular identification.
Behavioral covariates and calcium kinetics can explain some slow effects.
Bands cover the projection onto the chosen feature/statistic family; unmodeled channels may remain.
Long-memory and informative sampling are known scope boundaries.
```

The uploaded document states similar limitations: bands cover projections rather than eliminating identification bias, long memory and informative sampling are scope boundaries, errors-in-variables corrections are exact only in linear-Gaussian cases, and predictive directedness must not be read structurally without additional assumptions. 

---

# 28. Minimal MVP definition

The MVP is complete when the developer can run:

```bash
bash scripts/run_all_synthetic.sh
python -m sid_neuromod.experiments.run_elegans_fit --config configs/elegans_default.yaml
```

and obtain:

```text
validated hidden-thermostat recovery
valid filter-bank features
target-wise Gaussian conditional density fits
mean/gain/tail readouts
HAC/sup-t confidence bands
PIT calibration diagnostics
basic e-process monitoring
VAR/ridge baseline comparison
readouts.parquet
diagnostics.json
figures
HTML or Markdown report
```

Do not implement neural networks, tensor factorization, or molecular enrichment before this MVP is reliable.

---

# 29. Scientific success criteria

The implementation is scientifically useful if it can answer these questions from real C. elegans traces:

1. **Which source neurons predict which target neurons over which timescales?**

2. **Are the effects fast mean-drive effects or slow distributional effects?**

3. **Does a source neuron change a target’s gain, tail probability, or state dependence without much mean effect?**

4. **When did the conditional circuit change?**

5. **After the change, which source-target-timescale-channel readouts changed significantly?**

6. **Do slow distributional effects replicate across worms, conditions, or perturbations?**

7. **Are candidate slow factors enriched for known neuromodulatory sources or receptors?**

8. **Do these effects improve held-out conditional-law calibration beyond VAR/SINDy/mean-only baselines?**

That is the development target. The package should make these answers reproducible, statistically calibrated, and difficult to confuse with ordinary conditional-mean regression.

[1]: https://www.cell.com/neuron/fulltext/S0896-6273%2823%2900756-0?utm_source=chatgpt.com "The neuropeptidergic connectome of C. elegans: Neuron"
[2]: https://www.sciencedirect.com/science/article/pii/S0092867423008504?utm_source=chatgpt.com "Brain-wide representations of behavior spanning multiple ..."
[3]: https://numpy.org/doc/stable/reference/random/index.html?utm_source=chatgpt.com "Random sampling — NumPy v2.5 Manual"
[4]: https://docs.pytest.org/?utm_source=chatgpt.com "pytest documentation"
[5]: https://docs.jax.dev/?utm_source=chatgpt.com "JAX: High performance array computing — JAX documentation"
[6]: https://www.statsmodels.org/devel/generated/statsmodels.tsa.vector_ar.var_model.VAR.html?utm_source=chatgpt.com "statsmodels.tsa.vector_ar.var_model.VAR"
[7]: https://www.nature.com/articles/s41586-023-06683-4?utm_source=chatgpt.com "Neural signal propagation atlas of Caenorhabditis elegans"
