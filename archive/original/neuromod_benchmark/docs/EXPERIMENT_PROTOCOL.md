# Experiment protocol and decision gates

This is the preregistration-style analysis contract for the frozen confirmatory suite.
The completed 1,806-case medium sweep is **developmental**: it identified metric,
alignment, validation, and reporting defects that were corrected afterward. It can guide
resource sizing and frozen hyperparameters, but its numerical results are not pooled with
the confirmatory run.

The normative definitions are in [METRIC_CONTRACT.md](METRIC_CONTRACT.md); the theory is
in [THEORY.md](THEORY.md); executable admission gates are in
[METRIC_VALIDATION.md](METRIC_VALIDATION.md). A source/config digest identifies each
result. Any change to a frozen metric, simulator, method, split, or configuration creates
a new frozen version.

## Scientific question

For a stable stochastic neural system with explicit activity-dependent release,
persistent concentration, receptor occupancy, and typed neuromodulator entry channels,
which methods recover:

1. the held-out conditional transition law;
2. finite-time transition functionals;
3. occupied-state history and physical neuromodulator response fields;
4. latent modulator trajectories and the limited kinetic quantities justified by the
   declared coordinate;
5. transfer to altered ligand and receptor-knockout histories;
6. passive and represented-operation arm-history response plus controlled common-history
   lag-one effects, keeping a recursive controlled multi-lag kernel as a later capability;
7. calibrated changes in mean rule, dispersion, dependence, or higher-order shape?

An anatomical connectome, receptor atlas, or ordinary functional-connectivity matrix is
not simulator truth for items 1--6. Agreement with biological datasets is B1 external
consistency only and is reported outside the synthetic-recovery scorecard.

## Implemented simulator scope

The reference DGP is a bounded discrete-time release \(\rightarrow\) positive
concentration/clearance \(\rightarrow\) Hill occupancy \(\rightarrow\) neural transition
law \(\rightarrow\) calcium observation model. It is a phenomenological member of the
broader stochastic time-series class, not a calibrated molecular model and not an
implementation of arbitrary diffusion, jump, or regime-switching dynamics.

The registered mechanisms are null, additive mean, synaptic gain, intrinsic
excitability, stochastic dispersion, mean/variance-matched tail shape, correlation
routing, and `mixed`. Correlation routing is a separate pure scenario; `mixed` contains
the original five non-correlation channels.

## Mechanism-specific positive and negative controls

| Scenario | Required positive endpoint | Required invariance / negative control |
| --- | --- | --- |
| null | calibrated predictive law | every physical modulation field and represented intervention effect is zero |
| additive mean | physical mean state field; dose/knockout arm effect | no direct log-variance, dependence, or standardized-shape entry |
| synaptic gain | total gated-response field and explicit synaptic tensor lane | no direct dispersion/tail entry; baseline recurrent support is preserved |
| intrinsic excitability | self-response gating and explicit slope/threshold lane | no new cross-neuron structural edge |
| stochastic dispersion | physical log-variance state field and distribution-score benefit | one-step conditional mean is unchanged |
| matched tail | standardized two-sided shape probability, fourth moment, and law-score benefit | conditional mean and variance match the null construction exactly |
| correlation routing | off-diagonal covariance/correlation state fields | every marginal mean, variance, fixed-tail, and standardized-shape functional is invariant |
| mixed | joint predictive performance and channel-specific recovery | no single response channel may stand in for the full law |

Passing a positive endpoint while failing its invariant is not mechanism recovery. For
example, a tail detector that exploits variance leakage fails the matched-tail lane even
if its localization score is high.

## Primary metrics by claim

### P1 — conditional predictive law

For normalized methods:

- held-out per-target NLL and paired seed-level \(\Delta\)NLL;
- fair multivariate ensemble energy score when joint draws are available;
- fair marginal ensemble CRPS;
- marginal PIT CvM/ECE, own-CDF equal-tailed coverage, and within-episode serial PIT
  products;
- mean RMSE only as a point-forecast diagnostic.

`crps_gaussian_moment` is explicitly the Gaussian moment-projection CRPS. For Student-t
or mixture forecasts it is not a proper score for the submitted non-Gaussian law. PIT and
coverage are marginal, not copula tests. A per-target NLL under a factorized model does not
test cross-neuron dependence. Proper-score gains support P1 only on the tested histories;
they do not imply correct derivatives, latent coordinates, or causality.

For unnormalized score models, report held-out DSM risk under the same fixed Gaussian and
Student-t reference ladders. A clean comparison also requires the same scored variables
and conditioning domain. Conditional-outcome DSM and SBTG's joint
\([x_t,y_{t+1}]\) score therefore have domain-qualified metric IDs and are never placed
on one raw-risk leaderboard, even under the same kernel and scale. Graph-only methods
receive no predictive cell.

### D1/D2 — finite-time and rollout dynamics

Implemented on the complete-state, one-step lane:

- oracle conditional mean and log-variance error;
- linear and quadratic transition-operator NRMSE;
- fixed-threshold raw upper-tail probability error;
- two-sided standardized-shape probability error;
- occupied-state derivatives of these functionals with respect to observed history.

Tail probabilities use the method's declared exact CDF when available. A sample-only
generative method uses empirical event frequency and reports finite-draw Monte Carlo
error. Gaussian moment reconstruction is allowed only for a declared Gaussian forecast.

Separately fitted horizons are direct finite-horizon forecasts, not iterated rollouts.
The fair block-energy, autocovariance/spectral, stability/escape, and extreme-frequency
metric component is implemented and tested under a strict joint-sample shape contract.
It is integrated for the conditional Brownian bridge; other method adapters remain
pending, so those numbers cannot be inferred from the direct-horizon lane. Its
falsification uses independent oracle path draws, which retain finite-sampling error while
beating forecasts with correct marginals but wrong temporal dependence. Repeated or
overlapping forecast origins cluster at the episode/worm level for uncertainty.

Bridge fair energy is observed forecast accuracy and has a distribution-dependent
true-law optimum. Brownian reference path KL is separately registered as a directionless
control-effort diagnostic; it is never optimized as though zero KL implied a better
forecast.

### M2 — physical neuromodulator response

On common occupied complete states, report dimensionless fields for

- \(\partial_M\mu\);
- \(\partial_M\log v\);
- \(\partial_M P(Y>q)\), a raw fixed-threshold probability that mixes location, scale,
  and shape;
- \(\partial_M P(|(Y-\mu)/\sqrt v|>2)\), a raw two-sided standardized shape
  probability;
- off-diagonal \(\partial_M\Sigma\) and \(\partial_M\rho\);
- total gated response \(\partial_M\partial_X\mu\).

For nonzero truth, primary recovery is occupied-state nISE with absolute RMSE/MAE,
field cosine, and RMS-map error. For exactly zero truth, nISE and cosine are undefined;
report estimate RMS, maximum magnitude, RMS-map error, and false support. Statewise
evaluation is primary because averaging can cancel equal-and-opposite effects.

The total gated response is a physical response property, not a unique decomposition into
direct synaptic gating, intrinsic excitability, release feedback, and receptor nonlinearity.
Typing requires the explicit tensor lane plus targeted perturbations.

### Sparse support localization

Support is registered after a beta-min threshold. Report average precision, AP excess
over prevalence \(\mathrm{AP}-\pi\), tie-aware precision at the true number of edges, and
AUROC as secondary. The serialized `auprc_lift_over_prevalence` field is an unnormalized
excess despite its legacy name. For null support, report absolute null scores rather than
undefined AP.

### L1 — latent state and kinetic diagnostics

- Fit component assignment and affine calibration on validation worms only.
- Evaluate held-out Spearman, nRMSE, and \(R^2\) on test worms.
- Enforce the model-declared positive concentration orientation.
- Report clearance-time and release/tensor diagnostics only on the complete-state,
  model-matched lane; omit physical parameter recovery in calcium view and hierarchical
  lanes without a valid per-worm truth object.

The Hill map is invariant when concentration and \(K_d\) share a positive scale change,
and receptor expression magnitude can trade against downstream effect magnitude. Raw
parameter agreement is therefore a coordinate-conditional synthetic diagnostic. The
confirmatory claims include **no physical \(K_d\) or Hill-coefficient recovery**. Those
quantities require concentration calibration and/or an identifiable perturbation design.

### P1/C1 — altered environments and interventions

The registered response grid supports finite ligand pulses, receptor-specific knockout,
release-source silencing, and expected-null shams. Each pair has an exactly shared
pre-operation history and reuses named process, mixture, common-factor, pulse, and
observation innovations after branching. Pair index is also reused across operations, so
cross-operation comparisons remain paired.

This panel contains two deliberately different information sets. Ordinary
complete-state predictors receive the registered current modulator coordinate; the
latent neuromodulator SSM ignores that column and reconstructs concentration recursively
from neural history and stimulus. Records carry an effective-information-set field.
Accordingly, the ordinary methods are oracle-information baselines for the latent model,
not a like-for-like P1 leaderboard; the represented-operation lane instead asks whether
the harder latent reconstruction remains intervention-useful.

Frozen v1 uses an explicitly named **high-signal intervention-identifiability stratum**,
chosen from oracle truth diagnostics before any fitted response outcomes. It has one
modulator, 16 common-history pairs, and an 80 s response window. A zero Bernoulli release
density invokes the simulator's exactly-one-source guard; that unique source has release
gain 4 and is silenced for 20 s, leaving 60 s for recovery against clearance constants up
to 40 s. The finite ligand pulse lasts 4 s. Effect support is dense in this panel, and the
synaptic stratum uses effect strength 3.2 with reduced bias/noise so population averaging
does not erase a signed response. The fixed truth gates are RMS/innovation RMS at least
0.005, MC-SE/truth-RMS at most 0.35, and an in-window transient-decay fraction at least
0.25. Ordinary dense-release and noisier mechanisms remain in the non-response suites.
This panel establishes detectable K=1 recoverability only; it is not a claim about weak
effects, natural prevalence, or labeled multi-modulator recovery.

After an operation the arm histories diverge. The integrated scorecard therefore keeps
three claims separate:

- A passive predictor receives P1 population-mean and history-conditional arm-history
  response metrics.
- An eligible latent model may invoke its represented receptor operation on the
  registered arm history. This is secondary, narrow C1; multiple modulator labels must be
  externally anchored, while (K=1) is intrinsically labeled.
- Primary controlled C1 is scored only at lag one, when baseline and intervention
  predictions condition on the same realized branch-point history. A recursively
  controlled multi-lag kernel is not yet implemented.

Active operations must exceed the dimensionless truth-RMS/baseline-innovation-RMS
beta-min, while expected-null shams must remain below its registered tolerance. Active
truth MC-SE/truth-RMS must also fall below its preregistered ceiling. A transient active
truth must meet the preregistered minimum fraction of outputs with an in-window post-peak
(1/e) crossing. These are ground-truth precision/applicability preflights, not model
performance scores.

Population truth reports Monte Carlo SE. Pairwise nRMSE reports its defined fraction and
count. The kernel properties are nRMSE, sign, peak magnitude/time, windowed signed
cumulative response, transient (1/e)-crossing error and censor mismatch, persistent
end-window gain, and null leakage. A peak at the final lag is right-censored; persistent
operations do not receive a return-to-zero metric. Common random numbers reduce Monte
Carlo variance but do not identify an intervention from observational data.

### Change detection

- nuisance fitting, scan calibration, and final null-FPR evaluation use independent
  pools;
- maximum scan statistics, not pointwise scores, are calibrated;
- FPR, call rate, localization, detected-within, delay, and attribution remain separate;
- mean, residual dispersion, and residual tail-shape axes are not collapsed;
- finite-calibration and Monte Carlo confidence intervals are reported.

### B1 — external biology

Connectome, receptor-atlas, NeuroPAL, and perturbational-map comparisons are secondary
external analyses. They must be channel matched and replicated across animals where
possible. B1 cannot upgrade a predictive, derivative, latent, or causal synthetic result,
and B1 failure does not by itself invalidate simulator recovery.

## Metric falsification gates

The executable map is [METRIC_VALIDATION.md](METRIC_VALIDATION.md). Before freezing
confirmatory results, the suite must verify:

1. unregistered families, misspelled statistics, and incompatible suffixes fail strict
   metric-contract resolution;
2. exact oracle forecasts give numerical-zero functional and field error;
3. null mechanisms give negligible absolute estimated field magnitude;
4. matched mean/variance with changed tails moves shape metrics but not mean/variance;
5. changed dependence with invariant marginals moves off-diagonal dependence metrics only;
6. equal-and-opposite state effects defeat an average summary but not pointwise nISE;
7. a shifted stimulus fails transition-aligned oracle reconstruction;
8. mutated test targets cannot alter forecasts;
9. wrong mechanism/receptor channels are selectively penalized;
10. response truths pass beta-min, MC precision, null-sham, and transient/persistent
    applicability preflights before model scoring;
11. independent oracle path draws beat correct-marginal/wrong-dependence paths without
    requiring an impossible zero empirical sampling floor;
12. finite-difference and Monte Carlo checks agree within registered numerical/sampling
   tolerances.

A metric that fails its gate is removed or labeled exploratory before the freeze.

## Model selection and regularization

- Hyperparameters are selected using whole validation worms/episodes.
- Test worms never influence early stopping, corruption scale, sparsity, ridge, dropout,
  weight decay, mixture count, or latent-effect penalties.
- Neural models use early stopping, gradient clipping where relevant, fixed CPU threads,
  and registered regularization grids.
- Candidate failures remain in case records.
- Normalized models select on held-out NLL with numerical and extreme-variance guards;
  score-only models select on same-reference DSM risk; graph settings are fixed unless a
  graph-valid validation objective exists.
- Brownian-bridge reference diffusion is a **fixed sensitivity stratum**, not selected by
  endpoint NLL. Endpoint fit does not identify the interior reference.

### Developmental boundary decisions

The boundary sweeps in
[DEVELOPMENTAL_BOUNDARY_SUMMARY.md](../outputs/DEVELOPMENTAL_BOUNDARY_SUMMARY.md) use
seeds 0--2 only and are not confirmatory evidence. They set the following design choices:

- Corruption scale is a fixed estimand/sensitivity stratum, not a freely selected model
  hyperparameter. Gaussian score and SBTG use 0.5 with 0.25 sensitivity; SID is reported
  at fixed 0.75 and 1.0 strata. Comparisons use the fixed reference-risk ladder.
- MDN uses three components as the parsimonious primary capacity and seven as a declared
  sensitivity; the developmental mean validation-NLL advantage of seven was only 0.0067.
- The latent SSM fixes effect L1 at (10^{-4}) for observed complete-state data and
  (10^{-2}) for calcium, with weight decay (10^{-4}) because its validation surface was
  effectively flat.
- Developmental latent tracking remained poor ((R^2=0.012) in the latent view and
  (R^2=-0.018) in calcium). That is a mechanism-recovery failure to test, not a reason to
  choose a regularizer using test performance.

No value in that developmental summary is pooled into a frozen confidence interval.

## Replication and uncertainty

- The frozen resource-light methods use five independent DGP seeds; expensive neural and
  latent methods use a preregistered subset stated in every table.
- The uncertainty unit in synthetic summaries is the independent DGP seed, not a frame,
  target, or edge.
- Held-out-worm hierarchy metrics use worms as the unit and distinguish between-worm
  parameter transfer from paired within-worm observation degradation.
- Real-data inference uses animals as the resampling unit whenever the design permits.
- Exploratory selection over lags/targets requires max-statistic or equivalent correction.

Three to five synthetic seeds support engineering discrimination and failure discovery;
they are not a high-precision Monte Carlo power study. Changepoint FPR therefore uses a
larger dedicated null ensemble.

## Confirmatory suite order

1. property, oracle, null, leakage, and metric-falsification tests;
2. core conditional-law and direct-horizon suite;
3. isolated additive, synaptic, intrinsic, and dispersion channels;
4. matched-tail shape and correlation-routing suites;
5. memory-depth and held-out-worm/observation robustness;
6. latent-state/kinetic diagnostics;
7. calibrated changepoint detection and attribution;
8. passive/explicit arm-history response and controlled common-history lag-one lanes;
9. bridge-integrated joint rollout scoring, with other generative adapters and recursive
   controlled multi-lag response deferred until their capabilities exist;
10. separate reconciliation with existing real-data results.

Long jobs run sequentially at low priority, with one job and at most two numerical threads,
per-case atomic resume, a 6 GiB RSS ceiling, and a 4 GiB free-disk stop threshold. A
scientific failure is retained; it is not deleted or silently replaced.

## Decision rule

No method wins globally. A defensible conclusion has the form:

> Under mechanism \(S\), observation regime \(O\), horizon \(h\), and claim level \(C\),
> method \(M\) improves metric \(Q\) on held-out units while passing invariants \(G\) and
> numerical/resource gates \(R\).

Reports use paired seed intersections and never pool incompatible scenarios, views,
horizons, method capabilities, or source digests. Claims are downgraded when a relevant
state is hidden, a causal assumption fails, a reference kernel changes the estimand, or a
result follows exploratory selection.
