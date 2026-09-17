# Metric contract for the frozen suite

This document defines what each benchmark number estimates and what claim it can
support. It is intentionally claim-specific: predictive-law fit, finite-time dynamics,
neuromodulator response, latent tracking, causal response, and external biological
consistency are not interchangeable and are never averaged into one leaderboard. It
becomes frozen only when its source/config digest is recorded for the confirmatory run.

Status labels have a literal meaning:

- **implemented**: emitted by the current benchmark and covered by tests;
- **integrated**: implemented and connected to an eligible model/runner comparison;
- **diagnostic**: implemented, but not sufficient for the associated scientific claim;
- **component-ready**: oracle/metric code and falsification tests exist, but model adapter
  or runner integration is not yet part of a completed comparison;
- **planned**: part of the confirmatory design but not yet emitted; it must not appear in
  a current-results claim.

The corresponding falsification requirements and test map are in
[METRIC_VALIDATION.md](METRIC_VALIDATION.md). Metric names are a strict machine API:
registered families accept only registered terminal statistics or explicit lag/operation
patterns, and reporting rejects a typo instead of assigning it a plausible-looking claim.

## 1. Claim levels

| Level | Estimand | Permitted interpretation |
| --- | --- | --- |
| P1 | \(P(Y_{t+h}\mid H_t,U_t)\) on held-out histories | observed predictive law |
| D1 | finite-time transition functionals | observed-effective dynamics |
| D2 | coherent joint rollout law over multiple future times | recursive path dynamics |
| M1 | derivatives with respect to observed history | reduced-form response channel |
| M2 | derivatives with respect to calibrated modulator concentration, or typed simulator operations | physical neuromodulator response in the represented system |
| L1 | latent trajectory modulo a declared alignment | latent tracking, not unique molecular coordinates |
| C1 | response to a represented intervention | intervention-specific causal response under the simulator/model assumptions |
| C2 | structural mechanism or direct causal graph | requires identification assumptions plus intervention evidence |
| B1 | agreement with anatomy, receptor atlas, or experimental functional response | external biological consistency only |

Synthetic truth licenses D1--C1 statements only for the implemented simulator subclass.
B1 is deliberately outside the synthetic scorecard: a connectome is neither truth for a
conditional stochastic law nor truth for transmitter concentration, dispersion, tail
shape, or receptor occupancy.

## 2. Evaluation distribution and scaling

Let \(\nu\) be the empirical distribution of registered, held-out complete states after
grouping by worm/episode. A pointwise field metric uses the same selected occupied states
for truth and estimate. It therefore estimates error on the tested support, not a global
derivative guarantee and not behavior under arbitrary interventions.

Input and output derivatives are made dimensionless using held-out occupied-state standard
deviations. For example,

\[
J^{M,\mu}_{ik}(s)=\frac{s_{M,k}}{s_{Y,i}}
\frac{\partial \mu_i(s)}{\partial M_k},\qquad
J^{H,v}_{ij}(s)=s_{H,j}\frac{\partial\log v_i(s)}{\partial H_j}.
\]

The scale factors make errors comparable across coordinates; they do not create physical
units when concentration or fluorescence is uncalibrated.

## 3. Primary metric families

| Question | Primary estimand and metric | Status | Claim ceiling |
| --- | --- | --- | --- |
| Is the held-out conditional law good? | per-target NLL and paired \(\Delta\)NLL; fair multivariate energy score when joint samples exist | implemented | P1 |
| Are marginal forecasts calibrated? | marginal PIT CvM/ECE, equal-tailed coverage error, within-episode PIT products | implemented diagnostic | P1 |
| Are first and second transition functionals correct? | oracle mean/log-variance error and linear/quadratic operator error | implemented on complete-state \(h=1\) | D1 |
| Is a fixed upper-tail event correct? | RMSE/bias/correlation for \(P(Y_i>q)\) | implemented on complete-state \(h=1\) | D1 |
| Is standardized shape correct? | RMSE/bias/correlation for \(P(|Z_i|>a)\), \(Z_i=(Y_i-\mu_i)/\sqrt{v_i}\) | implemented on complete-state \(h=1\) | D1 |
| Is a state-dependent history response recovered? | occupied-state nISE/RMSE/cosine for mean, log variance, upper tail, and shape tail | implemented for normalized row-wise models | D1/M1 |
| Is a physical modulator response recovered? | occupied-state nISE/RMSE/cosine for mean, log variance, upper tail, shape tail, covariance, and correlation versus \(M\) | implemented for complete-state normalized row-wise models | M2 |
| Is total modulator-gated neural response recovered? | occupied-state error in \(\partial_M\partial_X\mu\) | implemented | M2 response property |
| Is sparse support localized? | AP, **AP excess over prevalence**, tie-aware precision at true \(k\), AUROC secondary | implemented diagnostic | M1/M2 only when paired with the matching field |
| Is the latent modulator tracked? | validation-fit assignment/alignment; held-out Spearman, nRMSE and \(R^2\) | implemented | L1 |
| Does a predictor transfer to altered arm histories? | population-mean and history-conditional response error on each realized arm history | implemented/integrated | P1 |
| Does an explicit model represent receptor knockout? | population-mean and history-conditional response error after invoking its operation on registered arm histories | implemented/integrated for eligible models; secondary | C1, narrowly defined |
| Is the controlled branch-point response recovered? | common-history lag-one nRMSE, sign, and null leakage | implemented/integrated for eligible, label-anchored models; primary | C1 at lag one only |
| Are recursively sampled path dynamics recovered? | fair block energy score, autocovariance/spectrum, escape/extreme frequency | implemented/integrated for the experimental bridge; generic adapters pending | D2 |
| Does a result agree with anatomy or experimental perturbation maps? | channel-matched enrichment/replication | external analysis | B1 only |

An unavailable metric is recorded as not applicable. A score-only or graph-only method is
not assigned a fabricated NLL, rollout, physical susceptibility, or intervention score.

## 4. Proper-score contract and qualifications

For a loss-oriented strictly proper score \(S\), the true forecast uniquely minimizes
\(E_P S(Q,Y)\) within the score's domain. For the log score,

\[
E_P[-\log q(Y)]-E_P[-\log p(Y)]=\mathrm{KL}(P\Vert Q).
\]

This identification is conditional only for histories represented by the held-out
distribution and only when the forecast is frozen, normalized on the same observed space,
and evaluated under the same dominating measure. Finite-sample risk, optimization error,
model selection, distribution shift, and dependence among frames remain sources of error.
Most importantly, small score regret does not imply recovery of derivatives, latent
coordinates, or causal mechanisms.

Metric-specific qualifications:

- `predictive.nll` is primary only for normalized models. It is a per-target score under
  the model's declared factorization unless a genuine joint log density is supplied; it
  cannot by itself validate cross-target dependence.
- `predictive.energy_score` is the fair off-diagonal ensemble U-statistic and evaluates a
  joint sampled forecast under finite-first-moment conditions. It has Monte Carlo error.
  Its expected true-law optimum depends on the outcome distribution and is not generally
  zero; lower is better only for forecasts evaluated on the same registered problem.
- `predictive.crps_ensemble_fair` evaluates marginal sampled forecasts. It does not test
  their copula.
- `predictive.crps_gaussian_moment` scores the Gaussian distribution with the submitted
  mean and variance. For a Student-t or mixture forecast it is a moment-projection
  diagnostic, not a strictly proper score for that submitted non-Gaussian law.
- PIT and coverage are marginal calibration diagnostics. Uniform pooled PIT is not enough:
  serial and state-conditional calibration can still fail.
- DSM risk is comparable only under the same registered corruption kernel, scale,
  scored variables, and conditioning domain. The conditional neural DSM scores
  outcomes given history; SBTG scores the joint consecutive-state vector. Their risks
  are distinct estimands even under the same kernel and scale. Neither identifies a
  normalized clean law or an M2 mechanism.

## 5. Transition and response-field metrics

For a nonzero oracle field \(J(s)\), the primary relative field error is

\[
\operatorname{nISE}(\widehat J,J;\nu)=
\frac{E_{s\sim\nu}\|\widehat J(s)-J(s)\|_F^2}
     {E_{s\sim\nu}\|J(s)\|_F^2}.
\]

The implementation also reports absolute RMSE/MAE, truth and estimate RMS, field cosine,
and error in the RMS support map. Pointwise evaluation is essential: equal-and-opposite
effects can have zero average while the fitted field is wrong everywhere.

For an exactly zero oracle field, nISE and cosine are undefined. The benchmark records
them as unavailable and reports estimated RMS, maximum absolute magnitude, RMS-map error,
and false support instead. Zero truth must never be stabilized with an arbitrary
denominator and interpreted as a relative recovery score.

Finite differences estimate fitted fields on occupied states. Their interpretation
requires local smoothness, interior support, and step-size stability. The finite set of
states does not establish a global Sobolev bound. Average Jacobians remain descriptive
summaries, not primary recovery metrics.

### 5.1 Tail estimands

The fixed-threshold upper-tail functional is the **raw probability**

\[
T_i^{q}(s)=P(Y_i>q\mid s),
\]

and its channel is \(s_j\,\partial T_i^q/\partial s_j\). It intentionally mixes changes
in location, scale, and shape.

The shape-only tail functional is the **two-sided standardized raw probability**

\[
T_i^{\mathrm{shape}}(s)=
P\!\left(\left|\frac{Y_i-\mu_i(s)}{\sqrt{v_i(s)}}\right|>a\,\middle|\,s\right),
\qquad a=2.
\]

Its channel is \(s_j\,\partial T_i^{\mathrm{shape}}/\partial s_j\). Neither field is a
logit derivative. For a conditionally Gaussian marginal, the standardized probability is
\(2\Phi(-a)\), so its derivative is exactly zero even when mean and variance change. That
makes it a negative control for affine location/scale effects and a positive endpoint for
the matched-tail mixture.

Tail probabilities are evaluated from a model's declared CDF when one is available.
For a sample-only forecast, the fallback is the empirical event frequency from registered
joint draws, with conditional Monte Carlo variance \(p(1-p)/M\). A Gaussian CDF assembled
from mean and variance is only valid for a declared Gaussian forecast and must not replace
the CDF of a non-Gaussian method.

### 5.2 Covariance and correlation

Correlation routing is evaluated through off-diagonal
\(\partial_M\Sigma_{ij}\) and \(\partial_M\rho_{ij}\) state fields. Diagonals are excluded
from the routing summary because \(\rho_{ii}=1\) is structurally constant. Marginal mean,
variance, fixed-tail, and standardized-shape invariance are simultaneous negative controls.

### 5.3 Sparse localization

Truth support is defined after a registered beta-min threshold. Let \(\pi\) be the active
edge prevalence and \(\mathrm{AP}\) average precision. The reported improvement is

\[
\mathrm{AP\ excess}=\mathrm{AP}-\pi.
\]

The current serialized key retains the legacy name `auprc_lift_over_prevalence`, but the
quantity is an unnormalized excess, not \((\mathrm{AP}-\pi)/(1-\pi)\). Precision at the
true number of edges uses expected credit within a score tie. AUROC is secondary under
sparsity. If truth has no active edge, AP is undefined and absolute null scores are used.

### 5.4 Long-horizon rollout component

The rollout scorecard requires
`observed[case, time, target]` and
`forecast[case, draw, time, target]`. Draws must be conditionally exchangeable **joint
trajectories** from one forecast. Independent one-step draws assembled across time violate
the estimand by destroying temporal dependence.

The fair off-diagonal energy U-statistic on a flattened finite path block is the primary
D2 proper score under finite-first-moment conditions. Center, target scale, block length,
and stride are fixed from training data or simulator units before evaluation. Its finite
ensemble value has Monte Carlo error and can be slightly negative.

Autocovariance nRMSE and full cross-spectral nISE are stationary second-order diagnostics,
not proper scores for the full path law. Escape rate/first-passage curves and extreme-event
frequency answer registered risk questions; their fair Brier terms properly score the
corresponding binary event probabilities. Extreme frequency is not the standardized
mean/variance-independent shape-tail field. Uncertainty clusters by case/worm, especially
when path blocks overlap. When several forecast origins belong to one episode, the case is
the point-estimand grain but the episode/worm remains the independent uncertainty grain.
The adversarial test uses independent oracle path draws and shows that finite-sample oracle
scores retain a sampling floor while beating a forecast with correct marginals and wrong
temporal dependence; copied observed paths are not a lawful oracle forecast.

## 6. Latent-state and parameter identifiability

Latent assignment and affine calibration are fit on validation worms and then frozen for
test worms. This supports L1 tracking only under the declared alignment. It does not prove
that the learned coordinate is molecular concentration.

The Hill map illustrates the main gauge:

\[
O_{ik}=E_{ik}\frac{M_k^{h_{ik}}}{K_{d,ik}^{h_{ik}}+M_k^{h_{ik}}}.
\]

Scaling both \(M_k\) and \(K_{d,ik}\) by the same positive constant leaves occupancy
unchanged. Expression magnitude can also trade against downstream effect magnitude in
the represented output map. More general latent-state reparameterizations can preserve
the complete input-output law.

Consequently:

- receptor support, latent alignment, clearance-shape diagnostics, and typed effect
  leakage can be reported as synthetic diagnostics under the model-matched coordinate;
- raw expression/effect errors require the declared gauge and are not generally physical
  identification;
- the benchmark makes **no receptor-\(K_d\) or Hill-coefficient recovery claim**;
- physical \(K_d\), Hill, expression magnitude, and concentration require concentration
  calibration, receptor-specific perturbations/dose anchors, or a proved identifiable
  experimental design;
- physical parameter recovery is omitted for calcium observations because the compact
  latent model has no identified calcium-emission/deconvolution layer.

## 7. Intervention estimands

The registered component branches every pair from an exactly shared realized history and
reuses the same named pulse, process, mixture, common-factor, and observation innovations
after the branch. Pair index \(b\) also reuses the same baseline/exogenous object across
all registered operations, preserving cross-operation pairing. Common random numbers
reduce Monte Carlo contrast variance; they do not supply identification in observational
data.

Three estimands must remain separate.

**Passive arm-history transfer (implemented/integrated, P1).** Each arm evolves after its
operation, so its history \(H_t^e\) differs from the factual history. An ordinary predictor
is evaluated on each realized arm history. Population-mean kernel error estimates the
response averaged over the registered history/exogenous distribution; history-conditional
pairwise error first scores each paired history and then averages. Pairwise normalized
error reports both its defined fraction and count because null pair truths have no relative
denominator. This is neither a common-history response beyond lag one nor autonomous
counterfactual rollout.

**Represented-operation arm-history response (implemented/integrated, secondary C1).** An
eligible latent model invokes its own receptor-specific knockout operation while consuming
the registered post-operation arm history. With one modulator its label is fixed; with
multiple modulators the model must declare an anchored label mapping. This tests the
represented operation on those histories, not receptor-specific biological causality and
not a recursively controlled full kernel.

**Controlled common-history lag one (implemented/integrated, primary C1).** At the first
post-branch transition, baseline and intervention predictions condition on the same
realized history. The primary C1 endpoint is therefore

\[
R_i^e(1;h)=E[Y_{t+1,i}\mid do(e),H_t=h]
-E[Y_{t+1,i}\mid do(e_0),H_t=h],
\]

scored by nRMSE, sign, and null leakage. At later lags the current adapter teacher-forces
the distinct realized arm histories, so the full curve remains the arm-history estimand
above. A recursively sampled controlled multi-lag model kernel remains pending.

The response runner records each method's effective information set. Generic
complete-state baselines observe current modulator concentration, whereas the explicit
latent SSM deliberately reconstructs it from neural history and stimulus. Their response
errors answer different difficulty levels and must not be read as an equal-input P1
leaderboard.

Operations include receptor-specific knockout, release-source silencing, finite-duration
ligand pulses, and expected-null shams. Positive operations must exceed a preregistered,
dimensionless beta-min defined as truth-kernel RMS divided by baseline innovation RMS;
expected-null operations must fall below the same registered tolerance. The response truth
reports Monte Carlo standard-error RMS and its ratio to truth RMS. Active truths must stay
below a preregistered MC-SE/truth-RMS ceiling; transient active truths must also meet a
preregistered minimum fraction of outputs whose post-peak (1/e) crossing is observable
inside the window. These are ground-truth precision/applicability preflights, not
model-performance metrics.

Peak magnitude, peak time, windowed signed cumulative response, sign, and null leakage are
separate summaries. A true peak at the final lag is right-censored and is not used for
peak-time error. A transient operation's first post-peak \(1/e\) crossing is scored only
when truth crosses within the window, with an explicit estimate-censor mismatch. Persistent
knockout/source-silencing operations do not receive a return-to-zero claim; they receive an
end-window gain over a declared fraction of the finite window.

## 8. Schrödinger-bridge contract

The implemented bridge is an experimental conditional Brownian endpoint bridge. Its
endpoint model is a heteroskedastic Gaussian regression; conditional paths are Brownian
bridges between the observed current state and a sampled endpoint. Endpoint NLL does not
identify the reference diffusion or the interior path law.

Reference diffusion is therefore a **fixed, registered sensitivity stratum**, not a
hyperparameter selected by endpoint NLL. Fair time-marginal and finite-block energy scores
compare sampled bridge paths with observed forecast paths; they are forecast-accuracy
losses whose true-law optima depend on the data distribution and are not generally zero.
The discrete-time reference diffusion has units of target activity per square root
simulation step; it is not a physical-time diffusion coefficient without an explicit
time-unit conversion.
The Brownian `reference_path_kl_mean` instead measures control effort relative to the
chosen reference. It is directionless and diagnostic: a smaller value does not establish a
better forecast. This lane has no receptor, causal-edge, or autonomous biological-path
interpretation.

## 9. Implemented versus planned dynamics grid

| Evaluation | Current status | Interpretation |
| --- | --- | --- |
| Separately fitted horizons | implemented for registered horizons | direct finite-horizon prediction, not recursive rollout |
| One-step oracle mean/variance/tail/shape operators | implemented | projected transition functionals |
| Occupied-state history and physical response fields | implemented on complete state, \(h=1\) | local-on-support derivatives |
| Experimental Brownian-bridge time-marginal and joint-path scores | implemented/integrated | observed forecast accuracy for this bridge construction |
| Brownian reference path KL | implemented/integrated diagnostic | directionless reference-relative control effort |
| Generic recursive joint trajectory sampling | integrated only for bridge; other adapters pending | needed before other methods receive rollout claims |
| Fair multistep block energy score | implemented/tested and bridge-integrated | proper finite-block D2 path-law score for valid joint rollouts |
| Autocovariance/spectrum, stability/escape, extreme-frequency error | implemented/tested and bridge-integrated | targeted path diagnostics, not full-law scores |
| Legacy zero/double ligand and all-receptor arm-history transfer | implemented | P1 transfer under divergent histories |
| Passive receptor/source/pulse arm-history kernels | implemented/integrated | population-mean and history-conditional P1 transfer |
| Explicit receptor operation on registered arm histories | implemented/integrated for eligible models | secondary represented-operation C1 |
| Receptor common-history lag-one effect | implemented/integrated for eligible, anchored models | primary controlled C1 endpoint |
| Recursive controlled multi-lag kernel | pending | required for a full controlled response-kernel claim |

## 10. Decision rule

A method passes a claim only when it improves the matching primary metric and passes that
mechanism's invariants, null leakage, numerical, split, and robustness gates. The report
must state scenario, observation regime, horizon, claim level, and independent seed/worm
unit. It must not pool incompatible strata or convert a missing capability into a loss.

The permitted conclusion has the form:

> Under scenario \(S\), observation regime \(O\), and claim level \(C\), method \(M\)
> improves metric \(Q\) on held-out units while passing invariants \(G\).

Anything stronger requires a stronger estimand, not a more favorable aggregate score.
