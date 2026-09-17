# Literature map and status of each model instantiation

This document distinguishes three kinds of support:

1. a biological observation that motivates a mechanism class;
2. a mathematical parameterization chosen as a minimal testable surrogate; and
3. a quantitatively calibrated biological model.

The current simulator is in categories 1–2. It is deliberately **not** represented
as a quantitatively calibrated molecular model of a particular monoamine or peptide.
That distinction is essential: synthetic ground truth is useful only when the tested
mechanism is explicit and falsifiable, not when a convenient equation is relabeled as
biology.

## Why the wired or functional connectome is not the primary truth

The key biological premise is well supported. In *C. elegans*, most cells expressing
several monoamine receptors do not receive a wired synapse from the corresponding
aminergic neuron. Bentley et al. therefore found that inferred extrasynaptic monoamine
and neuropeptide layers are largely non-overlapping with the wired connectome
([PLOS Computational Biology, 2016](https://doi.org/10.1371/journal.pcbi.1005283)).
Direct optogenetic perturbation plus whole-brain imaging further showed signal
propagation not predicted by anatomy alone and demonstrated a contribution from
dense-core-vesicle-mediated extrasynaptic signaling
([Randi et al., Nature 2023](https://doi.org/10.1038/s41586-023-06683-4)).

Consequences for this benchmark:

- anatomical overlap is a B1 external-consistency metric, not universal ground truth;
- a lagged conditional graph is not called “rewiring” without an intervention or a
  valid structural argument;
- receptor, release, and concentration variables must be represented explicitly when
  making a physical neuromodulator claim;
- functional-connectome agreement cannot validate stochastic dispersion or tail shape.

## Biological operations represented in the simulator

### Release and persistent concentration

The simulator uses activity-dependent positive release followed by first-order
clearance. This represents a generic separation between release and extracellular
concentration, not transmitter-specific chemistry.

The mechanism class is motivated by experiments showing both fast and persistent
neuromodulatory responses. In whole-brain *C. elegans* experiments, distinct serotonin
receptors mediated responses to sudden versus persistent serotonin release, and
optogenetic activation of the serotonergic NSM neuron produced widespread changes in
brain dynamics ([Dag et al., Cell 2023](https://doi.org/10.1016/j.cell.2023.04.023)).
Tyramine can likewise act through a fast ionotropic receptor and a slower extrasynaptic
metabotropic receptor in the same behavioral sequence
([Donnelly et al., Neuron 2013](https://doi.org/10.1016/j.neuron.2013.02.013)).

Implementation:

\[
R_t=\operatorname{softplus}(b_R+W_R\tanh X_t+U_t),\qquad
M_{t+1}=\rho M_t+(1-\rho)R_t.
\]

Interpretation and limitations:

- softplus enforces positive release and supplies a smooth release Jacobian;
- exponential clearance gives a physical time constant
  \(\tau=-\Delta t/\log\rho\);
- neither equation is claimed to be a fitted vesicle-release or transporter model;
- randomized ligand pulses provide persistent excitation for recovery tests;
- clearance-shape and release diagnostics are reported separately from generic
  latent-state correlation, but raw parameter agreement remains conditional on the
  represented coordinate and excitation design.

The implemented ligand-dose intervention reuses one randomized pulse train \(I_t\)
and changes only its amplitude:

\[
U_t^{(d)}=\rho_U U_{t-1}^{(d)}+dA I_t,\qquad d\in\{0,1,2\}.
\]

Initial state, neural innovations, tail-component uniforms, fluorescence innovations,
and pulse times are common across factual, zero-dose, and double-dose trajectories.
This is a common-random-number structural counterfactual: it reduces Monte Carlo
variance and makes the within-episode effect paired. Histories diverge after treatment,
so the current predictor metrics test one-step transfer on realized arm-specific
histories and paired total trajectory effects—not a same-history controlled response
or autonomous counterfactual rollout. Common random numbers do not establish
identification in observational data, and arms must not be analyzed as independent.
The estimand is this registered pulse-amplitude operation, not a universal
pharmacological dose-response curve.

### Receptor occupancy, expression, and knockout

The model uses expression-masked Hill occupancy,

\[
O_{ik}=E_{ik}\frac{M_k^{h_{ik}}}{K_{d,ik}^{h_{ik}}+M_k^{h_{ik}}}.
\]

This is a minimal saturating dose-response map. It provides positive, bounded occupancy,
a receptor-support mask, and a differentiable susceptibility with respect to physical
concentration. It is not a fitted receptor kinetic scheme.

Receptor-specific and even antagonistic effects are biologically necessary. D1- and
D2-like dopamine receptors act extrasynaptically in the same motor neurons through
opposing G-protein pathways
([Chase et al., Nature Neuroscience 2004](https://doi.org/10.1038/nn1316)).
Serotonin also acts through both metabotropic receptors and the ionotropic MOD-1
chloride channel
([Ranganathan et al., Nature 2000](https://doi.org/10.1038/35044083)).
Dag et al. used receptor mutants and whole-brain imaging to separate receptor roles,
which motivates receptor-specific knockouts rather than only removing the releasing
neuron.

Implementation status:

- expression support, \(K_d\), Hill coefficient, and occupancy are explicit truth;
- receptor knockout sets selected occupancies to zero without changing release or
  concentration;
- response pairs share the exact pre-operation history and named innovations, and pair
  indices reuse the same baseline/exogenous object across registered operations;
- the frozen-v1 response benchmark is an oracle-calibrated high-signal K=1
  identifiability panel with one unique dominant release source and a window long enough
  to observe decay; ordinary dense/noisier mechanisms are retained in separate suites;
- ordinary dose and knockout prediction on their evolved histories is P1 arm-history
  transfer, not graph overlap and not a common-history causal response;
- the intervention-capable latent model additionally exposes receptor-specific knockout;
  the full arm-history comparison is secondary represented C1, while only its exactly
  common-history lag-one contrast is primary controlled C1.

#### Identifiability boundary

Explicit simulator truth does not make every parameter identifiable from observations.
For any positive \(a_k\), replacing \(M_k\) and every corresponding \(K_{d,ik}\) by
\(a_kM_k\) and \(a_kK_{d,ik}\) leaves Hill occupancy unchanged. Receptor-expression
magnitude can also trade against downstream effect magnitude in the output map. Structural
identifiability is a prerequisite for interpreting a fitted biological parameter, not a
property conferred by low prediction error
([Villaverde, Barreiro, and Papachristodoulou 2016](https://doi.org/10.1371/journal.pcbi.1005153)).

Accordingly, the package may report coordinate-conditional latent alignment, support,
clearance-shape, expression, and typed-effect diagnostics, but it makes **no physical
\(K_d\) or Hill-coefficient recovery claim**. Such claims require concentration
calibration, dose/receptor anchors, or a separate proof of identifiability for the
experimental design.

### Additive drive, intrinsic excitability, and synaptic gating

The conditional neural mean has separately switchable entries:

\[
\mu(X,M)=b+d\,\tanh\{\kappa(O)(X-\theta(O))\}
 + [A\odot\exp G(O)]\tanh X + B O.
\]

They instantiate:

- **additive drive** \(BO\): a modulator-dependent offset;
- **intrinsic excitability** \(\kappa(O),\theta(O)\): a changed self response curve;
- **synaptic gating** \(A\odot\exp G(O)\): receptor-dependent modulation restricted
  to existing recurrent support.

The distinct channels matter because “gain” is not synonymous with offset. For
example, tonic GABA receptor activation can alter neuronal offset without altering
gain ([Pavlov et al., Journal of Neuroscience 2009](https://doi.org/10.1523/JNEUROSCI.2747-09.2009)).
Neuromodulators can also alter excitability and inter-individual variability even when
a mean output feature does not change
([Khorkova and Golowasch, eNeuro 2022](https://doi.org/10.1523/ENEURO.0061-22.2022)).

These equations are bounded phenomenological transition maps. They intentionally avoid
claiming membrane-potential, conductance, or spiking units. Their purpose is to create
separable ground-truth operations with analytic Jacobians and stability certificates.

#### Total modulator-gated response is not a direct-synapse label

The new complete-state lane evaluates the total finite-time susceptibility

\[
T_{ijk}(x,m)=\frac{\partial}{\partial m_k}
\left\{\frac{\partial E(Y_{t+1,i}\mid x,m)}{\partial x_j}\right\}.
\]

For a row-wise normalized predictor it is estimated by the four-corner central
difference in \(x_j\) and \(m_k\) on at most 32 occupied states and then made
dimensionless with the occupied probe's neural, modulator, and target standard
deviations. The pointwise field is the primary recovery object; its average is retained
only as a descriptive summary. If the fitted mean is \(C^4\) near
those states, the truncation error is \(O(\delta_x^2+\delta_m^2)\); this guarantee does
not apply at nonsmooth kinks. The oracle differentiates its analytic neural Jacobian
by a concentration secant, which is second order in the interior and first order when
positive-concentration clipping creates a one-sided difference. The fixed numerical
steps are \(\delta_x=10^{-3}\max\{\operatorname{sd}(x_j),1\}\) and
\(\delta_m=10^{-3}\max\{\operatorname{sd}(m_k),1\}\) for the fitted map, versus
\(10^{-4}\max(1,|m_k|)\) for the oracle secant. They require sensitivity checks and
are an approximation, not exact automatic differentiation.

All occupied-state fields use the same registered states for oracle and estimate. For a
nonzero truth field, nISE measures integrated squared error relative to oracle field
energy. For exactly zero truth, nISE and cosine are undefined; absolute estimated RMS,
maximum magnitude, RMS-map error, and false support are the relevant leakage controls.
This prevents equal-and-opposite state responses from disappearing in an average and
prevents an arbitrary epsilon denominator from turning null leakage into a relative score.

This total response cannot uniquely type a direct synaptic mechanism. In the simulator,

\[
c^+=\rho m+(1-\rho)R(x),\qquad o=O(c^+),\qquad \mu=F(x,o),
\]

so \(T_{ijk}\) contains direct response-gating terms and chain-rule terms through
activity-dependent release, clearance, the Hill operating point, additive drive, and
intrinsic excitability. A source neuron can change broadcast release and create an
off-diagonal total response even when its direct synaptic-gain tensor entry is zero.
Accordingly, total self/cross-neuron recovery is an M2 response metric; direct
synaptic-versus-intrinsic typing still requires the separate learned/oracle mechanism
tensors and receptor/source/dose interventions.

### Stochastic dispersion

The innovation law has a separate conditional log-variance channel,

\[
\log v_i(M)=\log v_{0,i}+\sum_k D_{ik}O_{ik}.
\]

This tests whether a method can recover changed reliability/dispersion even when the
conditional mean rule is unchanged. It is a scientifically plausible mechanism class,
but the current coefficient ranges are synthetic stress-test values rather than a fit to
a particular worm transmitter experiment.

Required claims:

- call \(\partial_M\log v\) **stochastic dispersion susceptibility**;
- do not call it response gain;
- validate it with a proper predictive law and oracle derivatives;
- use a mean-only detector as a negative control.

### Mean/variance-matched tail modulation

Both the oracle DGP and the learned explicit latent SSM can use the normalized
two-component scale mixture

\[
Y\mid H\sim(1-p)\mathcal N\!\left(\mu,\frac{v a^2}{D(p)}\right)
+p\mathcal N\!\left(\mu,\frac{v b^2}{D(p)}\right),\qquad
D(p)=(1-p)a^2+pb^2,
\]

with \(0<a<1<b\). The learned SSM predicts \(\mu(H)\), \(v(H)>0\), and
\(p(H)\in(0,1)\), evaluates the exact log mixture density and CDF, and samples the
same law. Each target emission is normalized, and the implemented joint emission is
factorized across targets conditional on the learned state.

Moment matching is algebraic, not approximate:

\[
E(Y\mid H)=\mu,\qquad
\operatorname{Var}(Y\mid H)=
\frac{v\{(1-p)a^2+pb^2\}}{D(p)}=v.
\]

For \(0<p<1\) and \(a\ne b\),

\[
E[(Y-\mu)^4\mid H]
=3v^2\frac{(1-p)a^4+pb^4}{D(p)^2}>3v^2,
\]

because the numerator minus \(D(p)^2\) is
\(p(1-p)(b^2-a^2)^2>0\). This is an exact counterexample to evaluating a learned
law by mean and variance alone. A model that detects the lane through a variance
leak fails the construction test.

This is **not** a claim that *C. elegans* calcium innovations follow this particular
mixture. It is an adversarial higher-order identifiability control, and its
factorization does not represent general cross-neuron tail dependence.

The associated response metrics differentiate raw probabilities, not logits. The fixed
upper-tail functional \(P(Y_i>q)\) responds to location, scale, and shape. The shape lane
uses the two-sided standardized probability

\[
P\!\left(\left|(Y_i-\mu_i)/\sqrt{v_i}\right|>2\right).
\]

For a Gaussian marginal this is the constant \(2\Phi(-2)\), so its derivative is exactly
zero under pure mean or variance modulation. That invariant is what lets the metric
isolate the represented matched-tail mechanism rather than generic extreme values.

### Observation model and worm hierarchy

Latent neural state is rendered through a causal low-pass calcium process and additive
fluorescence noise. Whole-brain calcium imaging is a powerful but indirect observation
of neuronal dynamics; freely behaving whole-brain imaging itself requires substantial
tracking and measurement machinery
([Nguyen et al., PNAS 2016](https://doi.org/10.1073/pnas.1507110112)).

The hierarchy varies recurrent weights, clearance time, receptor expression, receptor
affinity, and calcium decay across worms while preserving structural support, coefficient
sign, positivity, and the recurrent stability bound. Observation variants share bitwise
identical latent trajectories. This creates two distinct tests:

- biological generalization to unseen parameter draws;
- measurement robustness under a fixed living-system realization.

The implemented robustness suite makes the distinction operational. Worm IDs and
parameter/simulation seeds are disjoint across train, validation, and test. Mean-one
lognormal deviations are applied to active recurrent coefficients, clearance in
physical seconds, receptor expression, receptor \(K_d\), and calcium decay; zero
support, coefficient sign, positivity, and the recurrent row-norm bound are preserved.
The model is tuned once on reference-rendered train/validation worms and then frozen.

Each held-out test worm is rendered as reference, high-measurement-noise, and
slow-calcium data using the same exogenous object. The latent neural arrays are required
to be bitwise identical and their hashes are recorded. Thus within-worm transfer deltas
isolate the declared observation change conditional on one simulated organism, whereas
variation across held-out worms tests transfer to new parameter draws.

These are not interchangeable replication units. Frames within a worm are dependent,
and paired renderings are deliberately dependent; uncertainty belongs at the worm
level. Generalization is only to the declared synthetic hierarchy. It does not by
itself establish transport to real worms, unmodeled biological variability, different
structural support, or a different observation family.

## Statistical and dynamical baselines: exact scope

| Family | Implemented output | Routine claim ceiling |
| --- | --- | --- |
| VAR / Granger | finite-lag conditional mean/dispersion and held-out source deletion | P1 plus reduced-form M1; structural responses need identified shocks |
| SINDy / PySINDy | sparse discrete transition map in a declared feature library | projected D1/M1 only when coordinates and library are adequate |
| PCMCI-ParCorr | lagged conditional-dependence parents in multiple-dataset mode | reduced-form M1; C2 needs causal sufficiency and the PCMCI assumptions |
| VAR-LiNGAM | per-worm endogenous neural VAR, averaged lag coefficients | reduced-form M1 unless linear non-Gaussian independent-error/no-confounding assumptions hold |
| SID / Hyvärinen | quadratic score and, when precision is valid, normalized Gaussian law | P1 for the valid law and typed M1; not automatic M2 |
| DSM neural score | fixed-kernel corrupted outcome score | S1/M1 at that kernel; clean-law claims need an inverse or sampler |
| SBTG | joint consecutive-state score on \([x_t,y_{t+1}]\) and score-localization statistics | fixed-reference S1 and reduced-form M1 localization; no fabricated likelihood |
| Neuromodulator SSM | normalized emission, positive latent recurrence, typed tensors, explicit receptor operation | P1, equivalence-aware L1, model-matched M2, narrow C1 under gauges |
| Conditional Brownian bridge | normalized endpoint and reference-conditioned joint paths | P1/D1 and bridge-supplied D2; no M2/C1 |

These ceilings apply inside the implemented bounded simulator subclass. They do not claim
uniform recovery over the continuous-time umbrella model.

### Proper-score scope

Strictly proper scores identify a forecast law only within their domain and on the tested
history distribution ([Gneiting and Raftery 2007](https://doi.org/10.1198/016214506000001437)).
The log-score regret is conditional KL when both laws are normalized on the same observed
space. This does not identify a latent explanation, a derivative, or a causal graph.

The implementation therefore distinguishes exact normalized NLL, fair multivariate
ensemble energy score, fair marginal ensemble CRPS, and calibration diagnostics. The
Gaussian moment CRPS is a projection diagnostic for non-Gaussian forecasts; marginal PIT
does not test cross-target dependence; and a per-target factorized NLL does not test a
copula. DSM is handled separately because it identifies a corrupted score at a declared
kernel and scale rather than a normalized clean forecast.

### Finite-time operator probes

At the observed sampling interval, the benchmark compares methods through three
one-step transition-operator probes:

\[
P_\Delta y_i=\mu_i,\qquad
P_\Delta y_i^2=\mu_i^2+v_i,\qquad
P_\Delta 1_{\{y_i>q\}}=P(Y_i>q\mid H).
\]

The linear metric is target-scale-normalized RMSE; the quadratic metric is normalized
by the oracle second-moment RMS; and the tail lanes report raw-probability RMSE, bias,
and correlation. The registered analytic CDF is used when the forecast exposes one. A
sample-only forecast uses empirical event frequency; conditional on that forecast its
Monte Carlo estimate has variance \(p(1-p)/M\) and converges with sample count. A Gaussian
CDF reconstructed from mean and variance is valid only for a declared Gaussian forecast,
not as a shortcut for Student-t or mixture laws. The implementation exposes these metrics
only for applicable normalized predictors on complete-state, horizon-one data; a missing
probe is not scored as failure.

These probes are exact for the corresponding functionals but not complete for the
law. Linear plus quadratic probes cannot distinguish the matched-tail mixture from a
Gaussian with the same moments, and a single tail threshold cannot determine the full
tail. The implemented standardized two-sided shape event adds a deliberately
location/scale-invariant probe, but it too is only one functional. These are a common
finite-time interface, not a replacement for proper joint-law scores. Generic recursive
rollout metrics are implemented and tested as a separate component and are integrated for
the conditional Brownian bridge. Other method adapters and joint-trajectory generators
remain pending. The rollout falsification uses independent oracle trajectories, not copied
observations, and verifies a finite-sampling floor while detecting wrong temporal
dependence. Dependent forecast origins cluster at the episode/worm level.

### Finite-model predictive-memory curves

For method family \(m\), dense history \(1{:}L\), and largest tested history
\(L_{\max}\), the implemented curve is

\[
\widehat\Delta_{m,L}
=\widehat{\mathrm{NLL}}(\widehat f_{m,L})
-\widehat{\mathrm{NLL}}(\widehat f_{m,L_{\max}}).
\]

Every lag gets a separate tuned fit on the same grouped worm split; the reference gap
at \(L_{\max}\) is zero by construction. This is a finite-model useful-memory
diagnostic. It can be negative when a shorter model estimates better and includes
architecture, regularization, optimization, and finite-sample effects. All lanes are
explicitly restricted to the prediction events available at \(L_{\max}\), giving
identical target rows and held-out worms across history lengths.

The population conditional-mutual-information identity applies only to true
conditional laws evaluated on a common target distribution:

\[
R_{\log}(P_L)-R_{\log}(P_{\mathrm{full}})
=I(Y_{t+1};O_t^{(L)}\mid H_t^{(L)}).
\]

The implementation fixes worm splits and restricts every lane to the common
\(L_{\max}\) target rows. That eliminates row-composition shift, but finite-model
estimation, regularization, and optimization still separate this curve from CMI. A
same-corruption DSM-risk gap is never CMI.

### VAR and Granger source deletion

The VAR family estimates a finite-lag conditional mean and, in heteroskedastic variants,
a conditional variance. Granger's original operational notion is predictive: one series
helps when its past improves prediction of another
([Granger 1969](https://doi.org/10.2307/1912791)). Hidden modulators, temporal aggregation,
measurement filtering, and misspecified memory can all prevent a structural causal
interpretation.

Benchmark contract:

- use held-out episode source deletion, not in-sample coefficient magnitude alone;
- report normalized Gaussian/Student-t scores only for methods that define them;
- label hidden-state results reduced-form;
- never equate a VAR coefficient with receptor or synaptic truth.

### SINDy

SINDy searches for a sparse transition or differential equation in a declared feature
library ([Brunton, Proctor, and Kutz, PNAS 2016](https://doi.org/10.1073/pnas.1517384113)).
The benchmark's data are discrete and noisy, so both the internal implementation and
official PySINDy comparator fit \(X_{t+1}=F(X_t,U_t)\). They never pass the future state
as a continuous-time derivative. Equation recovery is meaningful only if the true law is
represented by the library and coordinates are appropriate.

### PCMCI

PCMCI combines conditional-independence testing with time-series parent selection
([Runge et al., Science Advances 2019](https://doi.org/10.1126/sciadv.aau4996)). The
official adapter uses ParCorr and multiple-dataset mode, so worm boundaries are not
concatenated. Its causal reading requires the relevant graphical assumptions, including
causal sufficiency for the configured test. With a hidden modulator it is scored only as
reduced-form lagged conditional-dependence localization.

### VAR-LiNGAM

VAR-LiNGAM identifies a linear structural VAR using non-Gaussianity under mutually
independent disturbances, an acyclic contemporaneous structure, and no latent confounding
([Hyvärinen et al., JMLR 2010](https://www.jmlr.org/papers/v11/hyvarinen10a.html)). The
official adapter fits each worm independently and averages lag coefficients. Its
instantaneous matrix is retained but is not scored against lagged neural truth.

### SID, Hyvärinen score matching, and DSM

Hyvärinen score matching estimates an unnormalized density through derivatives of its
log density ([Hyvärinen 2005](https://jmlr.org/papers/v6/hyvarinen05a.html)). Denoising
score matching learns the score of a corrupted distribution
([Vincent 2011](https://doi.org/10.1162/NECO_a_00142)). Consequently:

- a DSM risk is comparable only at the same corruption kernel, scale, scored
  variables, and conditioning domain;
- a single nonzero corruption level does not itself recover the clean law;
- a quadratic SID parameterization supplies a normalized Gaussian law only when its
  inferred precision remains numerically valid;
- invalid, near-boundary, extreme-variance, and held-out DSM diagnostics are reported;
- Student-t corruption is included to test reference-kernel sensitivity.

### SBTG

The isolated SBTG adapters learn a joint score of the standardized consecutive-state
vector \([x_t,y_{t+1}]\) under linear or feature-bilinear coupling. This is not the
conditional outcome score learned by the neural DSM adapter. Their risks receive
different domain-qualified metric IDs and cannot rank against one another even under
the same noise kernel and scale. SBTG joint-score cross-moments and squared-score
covariances are localization statistics, not automatically conditional mean or
log-variance derivatives. SBTG is therefore evaluated on within-domain held-out DSM
risk and oracle support localization, without manufacturing a normalized likelihood.

### Explicit learned neuromodulator state-space model

The learned SSM mirrors the benchmark causal chain with positive activity-dependent
release, persistent positive concentration, saturating receptor occupancy, and
separate additive, synaptic, intrinsic, dispersion, and tail tensors. It is fitted by
held-out normalized NLL with truncated recurrent training and validation-based early
stopping. Its Gaussian and matched-tail-mixture emissions are both normalized; the
latter has the exact moment-matching property proved above.

This is the strongest direct M2 instantiation in the package, but it remains a compact
phenomenological model. Latent coordinates require held-out alignment, and learned
clearance-shape, receptor-support/expression, and tensor diagnostics are evaluated
separately from forecast quality and interpreted in the declared synthetic coordinate.
Even on a `complete_state` response panel, this adapter deliberately ignores the supplied
current-modulator feature and reconstructs concentration recursively from neural history
and stimulus. Other complete-state predictors receive that oracle coordinate, so their
P1/response scores are easier-information baselines rather than equal-input competitors.
The expression/effect and concentration/\(K_d\) gauges above preclude a general physical
parameter claim; \(K_d\) and Hill recovery are not confirmatory endpoints.
Because its concentration is reconstructed recursively over a full sequence, the
row-subsampled total-gated finite-difference lane is not applied to it; its explicit
learned mechanism tensors provide the corresponding typing lane.

### Schrödinger bridge

A Schrödinger bridge minimizes path-space relative entropy to a declared reference law
subject to marginal constraints. Its interior dynamics depend on that reference in
general ([Léonard 2014](https://doi.org/10.3934/dcds.2014.34.1533)). The current bridge
is a conditional Brownian endpoint bridge and is labeled experimental. Endpoint fit is
insufficient. Its reference diffusion is a fixed sensitivity stratum rather than a value
selected by endpoint NLL. The current implementation reports fair time-marginal and
finite-block path energy against observed forecast paths, RMSE, ACF/spectral and
escape/extreme diagnostics, and Brownian reference path KL. Energy is forecast accuracy
with a distribution-dependent optimum. Reference KL is instead directionless control
effort relative to the chosen Brownian law; smaller KL does not by itself mean a better
forecast. Generic rollout integration is available for this bridge only. Interventions,
control-field validation, and other methods' recursive adapters remain pending.
The registered diffusion values have units of target activity per square root simulation
step; no physical-time diffusion interpretation is claimed.

## Metric gates implied by the literature

| Claim | Necessary evidence in this benchmark | Explicit non-evidence |
| --- | --- | --- |
| Conditional-law recovery | held-out NLL/energy/CRPS and calibration; finite-time probes as separate functionals | training loss, Gaussian moment CRPS for a non-Gaussian law, or a few probes alone |
| Mean dynamics | oracle conditional mean/Jacobian and multi-step response | marginal correlation |
| Stochastic dispersion | oracle log-variance susceptibility and proper distribution score | mean gain or residual variance alone |
| Tail modulation | normalized matched-mixture likelihood plus tail/fourth-moment recovery | variance change |
| Neuromodulator mechanism | occupied-state physical concentration susceptibility, typed leakage/support, or a represented intervention | generic history derivative or unanchored \(K_d\)/Hill agreement |
| Total gated response | occupied-state mixed-derivative recovery with step sensitivity | a direct synaptic label |
| Synaptic/intrinsic typing | direct tensor recovery plus targeted receptor/source/dose interventions when implemented | total mixed derivative, all-receptor transfer, or connectome AUROC alone |
| Latent recovery | held-out alignment under the declared equivalence class and input-output behavior | raw latent correlation chosen on test data |
| Predictive memory | finite-model NLL curve on fixed worm splits and, ideally, common target rows | automatic conditional mutual information |
| Arm-history transfer | population-mean and history-conditional response error on realized operation-arm histories | recursive controlled response, free-roll, or causal-identification claims |
| Represented operation | explicit receptor operation on registered arm histories, with (K=1) or anchored labels | passive transfer, C2, or a controlled multi-lag kernel |
| Controlled causal usefulness | common-history lag-one nRMSE/sign/null leakage after truth beta-min, MC-precision, and applicability preflights | later teacher-forced arm histories, observational graphs, or treating CRN arms as independent |
| Worm robustness | unseen-worm performance plus paired observation transfer deltas | frame-level replication or real-population transport |
| Change detection | independent matched-null calibration, power, localization, delay, attribution | maximum scan score alone |

## Current implementation boundary

| Evidence family | Implemented now | Planned extension |
| --- | --- | --- |
| dynamics | direct registered horizons; one-step oracle probes and occupied-state fields; bridge-integrated rollout metric component | joint-trajectory adapters for non-bridge methods |
| interventions | passive population/history-conditional arm-history response; explicit receptor arm-history operation; primary common-history lag-one comparison | recursively controlled multi-lag response kernel |
| bridge | conditional Brownian endpoint/interior paths, observed path scores, and directionless reference effort under fixed reference-diffusion strata | biologically structured references, interventions, and control-field validation |
| latent biophysics | held-out state alignment and coordinate-conditional clearance/release/expression/effect diagnostics | calibrated identifiable concentration, \(K_d\), Hill, and receptor-magnitude design |
| biology | separate connectome/receptor/perturbational-map consistency | experimental replication matched to each stochastic channel |

This boundary prevents a broad theoretical desideratum from appearing as an already-run
metric. In particular, a separately trained horizon is not a rollout, and a later-lag arm
evolved under an operation is not a common-history controlled response.

### Developmental tuning status

The seed-0--2 boundary sweep is design evidence, not a result lane. It fixes corruption
scales as estimand/sensitivity strata and compares score methods on one fixed reference
ladder. Gaussian score/SBTG use 0.5 with 0.25 sensitivity; SID uses fixed 0.75 and 1.0
strata. MDN uses three components primarily and seven as a capacity sensitivity. The
latent SSM freezes view-specific effect penalties ((10^{-4}) complete state,
(10^{-2}) calcium) and conventional (10^{-4}) weight decay. Developmental latent
tracking was poor ((R^2=0.012) and (-0.018), respectively), so forecast fit cannot be
substituted for L1/M2 recovery. None of these test values enters confirmatory uncertainty.

## What remains empirical rather than established

- The simulator's numerical ranges are calibrated for stable, observable benchmark
  effects, not fitted to transmitter concentrations or receptor kinetics.
- Matched-tail and stochastic-dispersion channels are mechanism stress tests, not claims
  about a specific experimental dataset.
- The total gated-response tensor depends on occupied states and fixed finite-difference
  steps; numerical convergence and direct mechanism typing are separate checks.
- Finite-model memory gaps combine representational memory with estimation, tuning,
  optimization, and possibly target-row composition; they are not automatically CMI.
- Calcium filtering is a controlled observation family, not a detailed GCaMP kinetic
  model.
- Held-out-worm transfer is only to the declared synthetic hierarchy, while paired
  observation variants condition on one shared latent realization.
- A model that recovers synthetic parameter labels can still fail under biological
  misspecification. Real-data evaluation must therefore retain perturbation prediction,
  across-worm generalization, and observation sensitivity as independent axes.
