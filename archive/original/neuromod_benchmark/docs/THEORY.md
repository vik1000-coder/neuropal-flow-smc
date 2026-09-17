# Theory and metric contract

The normative metric names, applicability rules, optima, and implementation status are
listed in [METRIC_CONTRACT.md](METRIC_CONTRACT.md). This document supplies the theory
behind that contract. Where a broader theoretical target is discussed, it is labeled as
such rather than presented as an implemented result.

Metric admission is operationalized in
[METRIC_VALIDATION.md](METRIC_VALIDATION.md): the strict registry attaches an estimand,
claim, direction, attainable optimum, unit, and role to every identifier, then rejects
unknown terminal statistics before reporting.

This package benchmarks stochastic neural time-series models with latent
neuromodulator dynamics. Its common target is the controlled conditional path law

\[ P^u(Y_{t+1:t+K}\in\cdot\mid H_t=h). \]

The target is shared by VAR, nonlinear autoregression, state-space models, SID,
SBTG, denoising score models, implicit samplers, and Schrödinger bridges. It does
not by itself identify a latent coordinate, a causal graph, or derivatives of the
law with respect to history.

The benchmark uses a claim ladder:

1. predict a held-out conditional law;
2. recover dynamics and typed mechanisms against simulator truth;
3. recover latent state only modulo its identifiable transformation;
4. predict unseen interventions before making causal claims;
5. pass change-detection calibration and observation-stress tests.

An anatomical or mean functional-connectome match is a secondary biological check,
not ground truth for stochastic dispersion, tail shape, or a latent modulator.

## Terminology

- A **forecast score** is a loss \(S(Q,y)\) for a predictive distribution.
- A **density score** is \(\nabla_y\log p(y\mid h)\); a **history score** is
  \(\nabla_h\log p(y\mid h)\).
- **SID** means score-identified dynamics; **SBTG** means Score-Based Transition Graph.
- **Response gain** is an input-output, excitability, or modulator-gated mean slope.
- **Stochastic dispersion** is conditional noise-scale or log-variance modulation.
- A **physical susceptibility** differentiates with respect to calibrated \(M_k\).
- A **reduced-form channel** differentiates with respect to observed history \(H\).
- A Schrödinger reference is a path law \(R\); an \(\ell_q\) bridge penalty is
  unrelated regression shrinkage.

Legacy code may call a log-variance derivative gain. Reports should call it
stochastic dispersion and reserve response gain for a response slope.

## 1. Model class and claim levels

A continuous-time umbrella is

\[
\begin{aligned}
dZ_t&=f(Z_t,M_t,U_t)dt+G(Z_t,M_t,U_t)dW_t+J(Z_{t-},M_{t-},U_t)dN_t,\\
dM_t&=\{R(Z_t,U_t)-\Lambda M_t+D\mathcal A M_t\}dt+QdB_t,\quad
Y_k=\mathcal O(Z_{[k\Delta-c,k\Delta]},M_{k\Delta})+\varepsilon_k .
\end{aligned}
\]

\(Z\) is fast neural state; \(M\) is release, concentration, or receptor
occupancy; \(G\) controls process dispersion; \(JdN\) represents bursts or jumps;
and \(\mathcal O\) includes calcium filtering, sampling, missing neurons, and
measurement noise. A finite-memory discrete kernel is equally valid.

### 1.1 Implemented simulator subclass

The current synthetic ground truth is a bounded, discrete-time subclass of that umbrella,
not a numerical solver for the full jump-diffusion. With elementwise operations where
appropriate,

\[
\begin{aligned}
u_t&=\rho_u u_{t-1}+A I_t,\\
r_t&=\operatorname{softplus}\{b_R+u_t+W_R\tanh x_t\},\\
m_{t+1}&=\rho_m\odot m_t+(1-\rho_m)\odot r_t,\\
o_{ik,t+1}&=E_{ik}\frac{m_{k,t+1}^{h_{ik}}}
{K_{d,ik}^{h_{ik}}+m_{k,t+1}^{h_{ik}}},\\
x_{t+1}&=\mu(x_t,o_{t+1})+\eta_{t+1},\\
c_{t+1}&=\rho_c c_t+(1-\rho_c)x_{t+1},\qquad
Y^{\mathrm{complete}}_{t+1}=x_{t+1},\quad
Y^{\mathrm{calcium}}_{t+1}=c_{t+1}+\varepsilon_{t+1}.
\end{aligned}
\]

The bounded mean map contains separately switchable additive drive, intrinsic slope and
threshold, and receptor-gated recurrent efficacy. The innovation law is either Gaussian,
a factorized two-scale Gaussian mixture whose conditional mean and marginal variance are
exactly matched, or a positive-definite one-factor correlation-routing law. The pure
correlation scenario is separate; `mixed` contains the original five mean, intrinsic,
synaptic, dispersion, and matched-tail channels and does not include correlation routing.

All complete-state one-step oracles condition on the registered transition state and
ligand drive. The simulator supplies exact conditional moments and analytic or
controlled-finite-difference response fields. Thus simulator-ground-truth conclusions
apply to this stable phenomenological family. They do not automatically extend to the
entire continuous-time umbrella, arbitrary jump processes, or a quantitatively calibrated
worm transmitter system.

The DGP registry separately instantiates direct mean drive, intrinsic excitability,
synaptic gating, stochastic dispersion, correlation routing, matched-tail modulation,
and release/receptor kinetics. Regime switching and general jump/Lévy laws remain
broader design targets rather than members of this reference simulator.

Every metric record must declare a claim level.

| Level | Estimand | Defensible interpretation |
| --- | --- | --- |
| P0 | marginal or rollout law of \(Y\) | generative faithfulness |
| P1 | \(P(Y_{t+1:t+K}\mid H_t,U)\) | observed predictive dynamics |
| D1 | transition, generator, drift, diffusion, jump law | observed-effective dynamics |
| D2 | coherent joint conditional rollout law | recursive path dynamics |
| M1 | history-derived typed channel and lag kernel | reduced-form mechanism |
| M2 | physical \(M\)-susceptibility, release, gating | neuromodulator mechanism |
| L1 | latent state modulo an equivalence class | latent tracking |
| C1 | interventional path law | represented causal response |
| C2 | direct causal graph or mechanism | structural claim |
| B1 | atlas, connectome, or experimental-map overlap | external biological consistency only |

An observational hidden-driver result is at most D1 or M1. It must not silently
become M2 or C2. B1 is not a synthetic recovery gate and is never evidence that an
unobserved stochastic or molecular quantity was recovered.

## 2. Theoretical guardrails

### 2.1 Proper prediction identifies a law, not its explanation

**Theorem 1 — strict propriety.** For true law \(P\) and loss-oriented strictly
proper score \(S\),

\[ E_{Y\sim P}S(P,Y)<E_{Y\sim P}S(Q,Y)\quad\text{for all }Q\ne P. \]

For the log score, regret is \(\mathrm{KL}(P\Vert Q)\). Conditional identification
holds only for \(P_H\)-almost every tested history. With a frozen forecast,
stationary ergodicity, and integrability,

\[ \frac1T\sum_{t=1}^T S\{\widehat P(\cdot\mid H_t),Y_{t+1}\}\longrightarrow E S\{\widehat P(\cdot\mid H),Y\} \]

almost surely. This says nothing about off-support histories or causality.
See [Gneiting and Raftery 2007](https://doi.org/10.1198/016214506000001437)
and [Dawid 1984](https://doi.org/10.2307/2981683).

**Counterexample — predictive convergence need not imply derivative convergence.**
On compact history coordinate \(h\), let

\[ p_n(y\mid h)=p_0(y)\{1+n^{-1/2}a(y)\sin(nh)\}, \]

where \(a\) is bounded, nonzero, and centered under \(p_0\). These densities are
normalized and positive for large \(n\). Their KL regret is \(O(n^{-1})\to0\),
while \(\partial_h p_n\) has amplitude \(O(\sqrt n)\).

Therefore NLL, energy score, or DSM loss cannot alone validate a graph or channel
read from \(\partial_H\log p\), \(\partial_H\mu\), or
\(\partial_H\log\Sigma\). Derivative metrics and regularization are separate.

The implemented score names require further care. NLL is primary only when a model
supplies a normalized density on the declared observed space; a per-target factorized
log score does not test cross-target dependence. The fair ensemble energy U-statistic
tests a joint sampled forecast but has finite-ensemble Monte Carlo error. Ensemble CRPS
and PIT are marginal. `crps_gaussian_moment` scores the Gaussian moment projection, so
for a Student-t or mixture forecast it is a moment diagnostic rather than a strictly
proper score for the submitted non-Gaussian law. Pooled PIT uniformity also does not
imply conditional or serial calibration. These qualifications are part of P1, not
optional reporting caveats.

### 2.2 Generator and finite-time transition recovery

For an observed Itô diffusion,

\[ \mathcal L\phi(x)=b(x)^\top\nabla\phi(x)+\tfrac12\operatorname{tr}\{a(x)\nabla^2\phi(x)\}. \]

**Proposition 1 — generator characterization.** Equality of two generators on a
core, a well-posed martingale problem, and the same initial law characterize the
same process law. A finite probe set is only a projected diagnostic.

At coarse sampling the identified object is usually \(P_\Delta\), not
\(\mathcal L\). Use the oracle operator metric

\[ D_{P_\Delta}=\frac1R\sum_{r=1}^R\frac{E_\nu[\{(\widehat P_\Delta\phi_r)(H)-(P_\Delta\phi_r)(H)\}^2]}{E_\nu[(P_\Delta\phi_r)(H)^2]+\epsilon}. \]

Evaluate on occupancy-weighted histories and an intervention-designed state grid.
Use linear, quadratic, localized, occupancy, and tail probes. This is the common
target for wrapped VAR/SINDy models, SID/SBTG, samplers, and bridges.

The implemented one-step probes specialize this operator identity to

\[
\phi_i(y)=y_i,\quad \phi_i^{(2)}(y)=y_i^2,\quad
\phi_{i,q}^{\mathrm{tail}}(y)=1_{\{y_i>q\}},
\]

so that

\[
P_\Delta\phi_i=\mu_i,\qquad
P_\Delta\phi_i^{(2)}=\mu_i^2+v_i,\qquad
P_\Delta\phi_{i,q}^{\mathrm{tail}}=P(Y_i>q\mid H).
\]

**Proposition 1a — implemented probe exactness and incompleteness.** Exact conditional
mean and variance make the linear and quadratic probe errors zero. A model's declared
analytic CDF gives its exact fitted threshold probability. If only conditional forecast
draws are available, the empirical tail exceedance converges almost surely to that fitted
probability as draw count grows, with conditional Monte Carlo variance \(p(1-p)/M\). At
finite \(M\), the sample fallback contains simulation noise. Conversely, zero error on
these probes does not identify the full transition law: the first two fix only two
moments, and a tail probe fixes one event.

Implementation uses target-scale-normalized RMSE for the linear probe, oracle
second-moment RMS normalization for the quadratic probe, and raw probability RMSE, bias,
and correlation for fixed and standardized tail probes. It uses the exact declared CDF
when supplied and empirical event frequency for a sample-only generative forecast. A
Gaussian CDF reconstructed from mean and variance is permitted only for a declared
Gaussian forecast; it is not a fallback for Student-t, mixture, or otherwise
non-Gaussian models. These probes run only for applicable normalized predictions in the
complete-state, horizon-one lane; unnormalized methods are not assigned a fabricated
failure value.

### 2.3 Functional derivatives from a history score

Let \(S_k^H(y,h)=\partial_{h_k}\log p(y\mid h)\).

**Proposition 2 — score-covariance identity.** If differentiation may pass through
the conditional integral,

\[ \partial_{h_k}E\{\phi(Y)\mid h\}=\operatorname{Cov}\{\phi(Y),S_k^H(Y,h)\mid h\}. \]

Consequently,

\[
\begin{aligned}
\partial_{h_k}\mu_i&=E[(Y_i-\mu_i)S_k^H\mid h],\quad
\partial_{h_k}v_i=E[((Y_i-\mu_i)^2-v_i)S_k^H\mid h],\\
\partial_{h_k}\Sigma_{ab}&=E[((Y_a-\mu_a)(Y_b-\mu_b)-\Sigma_{ab})S_k^H\mid h],\quad
\partial_{h_k}P(Y_i>q\mid h)=E[(1_{\{Y_i>q\}}-P(Y_i>q\mid h))S_k^H\mid h].
\end{aligned}
\]

The conditional neural DSM learns the outcome score
\(\nabla_y\log p_\sigma(y\mid h)\). The implemented SBTG instead learns the joint
score of the standardized consecutive-state vector \([x_t,y_{t+1}]\), using one
current coordinate per neuron. Neither is \(S^H\); conversion requires another
identity or a sampleable fitted law. Their DSM risks are not mutually comparable,
because their scored variables and dimensions differ even when the corruption kernel
and scale agree.

For \(Y\mid H=h\sim\mathcal N\{\mu(h),\Sigma\}\) with constant
positive-definite \(\Sigma\),

\[ \Sigma\,\partial_h\nabla_y\log p(y\mid h)=\partial_h\mu(h). \]

This is exact. If \(\Sigma=\Sigma(h)\), the score-Jacobian mixes mean and
covariance derivatives; explicit \(\mu(H)\) and positive-definite \(\Sigma(H)\)
heads are preferable for mechanism typing.

### 2.4 Latent coordinates are equivalence classes

For \(z_{t+1}=F(z_t,u_t)\), \(y_t=g(z_t)\), any invertible differentiable
\(\varphi\) gives

\[ \widetilde F=\varphi\circ F\circ\varphi^{-1},\qquad \widetilde g=g\circ\varphi^{-1}, \]

with identical input-output behavior. Minimal linear realizations are unique only
up to similarity; suitable nonlinear minimal realizations may be unique only up to
diffeomorphism. Without minimality and observability, ambiguity can be worse.

Fit alignment on validation systems and test it on held-out systems. Use
affine/similarity, componentwise monotone, or input-output/conjugacy metrics as
justified. The implemented Hill map has an explicit concentration-scale gauge:

\[
(M_k,K_{d,ik})\mapsto(a_kM_k,a_kK_{d,ik}),\qquad a_k>0,
\]

which leaves receptor occupancy unchanged. Expression magnitude can likewise trade
against downstream effect magnitude in the observed map. Therefore latent tracking,
support, clearance-shape, and effect-leakage diagnostics can be reported under a declared
model-matched coordinate, but they do not prove unique molecular parameters. The current
benchmark makes no receptor-\(K_d\) or Hill-coefficient recovery claim. Physical
concentration, receptor magnitude, \(K_d\), Hill coefficient, and transport length need
calibrated inputs/outputs, intervention anchors, or a proved identifiable design. This is
the required distinction between structural identifiability and numerical fit
([Villaverde et al. 2016](https://doi.org/10.1371/journal.pcbi.1005153)).

### 2.5 Predictive memory has an exact log-score meaning

Partition history into recent \(H_t^{(L)}\) and older \(O_t^{(L)}\).

**Proposition 3 — memory gap.**

\[ E[-\log p(Y_{t+1}\mid H_t^{(L)})]-E[-\log p(Y_{t+1}\mid H_t^{(L)},O_t^{(L)})]=I(Y_{t+1};O_t^{(L)}\mid H_t^{(L)})\ge0. \]

The implemented quantity is instead the method-specific finite-model gap

\[
\widehat\Delta_{m,L}
=\widehat{\mathrm{NLL}}(\widehat f_{m,L})
-\widehat{\mathrm{NLL}}(\widehat f_{m,L_{\max}}),
\]

where each lag \(L\) uses dense history \(1{:}L\), the same grouped worm split, and a
separately fitted/tuned instance of method family \(m\). The \(L_{\max}\) gap is zero
by construction. A negative finite gap is possible because shorter histories can
generalize better, and because estimation, optimization, regularization, and finite
samples differ. The implementation intersects every lane to the prediction events
available at \(L_{\max}\), so target rows and held-out worms are identical across
history lengths.

**Qualification.** \(\widehat\Delta_{m,L}\) estimates the conditional-mutual-information
identity only if both fitted models converge to the corresponding true conditional
laws, \(L_{\max}\) contains the relevant older history, NLLs are evaluated on the same
target rows/distribution, and tuning error vanishes. The current suite fixes worm
splits and explicitly intersects all lanes to the \(L_{\max}\) target rows. This
removes row-composition shift but not finite-model estimation or tuning error. The
same-kernel DSM gap is an objective comparison at one corruption scale, never CMI.

### 2.6 Schrödinger bridges are reference-relative

\[ Q_R^*=\arg\min_{Q:\,Q_0=\mu_0,Q_T=\mu_T}\mathrm{KL}(Q\Vert R). \]

Under standard feasibility conditions the solution is unique
[Léonard 2014](https://doi.org/10.3934/dcds.2014.34.1533).
It can, and generally does, depend on \(R\). Endpoint-factor reweightings
\(dR'/dR=a(X_0)b(X_T)\) leave the optimizer unchanged because they add a constant
on the constrained set.

Endpoint fit does not identify interior dynamics. Evaluate intermediate marginals,
joint paths, finite-time operators, interventions, control fields, and sensitivity
to scientifically distinct references. In the implemented Brownian baseline, reference
diffusion is consequently a fixed, registered sensitivity stratum. It is not selected by
endpoint NLL, because the endpoint regression does not identify it. The bridge now supplies
joint paths to the generic rollout component. Fair time-marginal and finite-block energy
scores are observed forecast-accuracy losses; their expected true-law optimum is
distribution-dependent rather than zero. The Brownian path KL has a different semantics:
it is directionless control effort relative to the registered reference and is meaningful
only on an accuracy-constrained comparison. Intervention and control-field evaluation,
and generic joint-path adapters for other methods, remain pending.
The implemented diffusion parameter is measured in target activity per square root
simulation step, not per square root physical second.

### 2.7 DSM identifies a corrupted law

Let \(p_\sigma(\widetilde y\mid h)=\int q_\sigma(\widetilde y\mid y)p(y\mid h)\,dy\).
Under differentiation and integrability conditions,

\[ \nabla_{\widetilde y}\log p_\sigma(\widetilde y\mid h)=E[\nabla_{\widetilde y}\log q_\sigma(\widetilde y\mid Y)\mid\widetilde Y=\widetilde y,H=h]. \]

DSM identifies the corrupted conditional score, not the clean score from one
nonzero noise level [Vincent 2011](https://doi.org/10.1162/NECO_a_00142).
Clean-law claims require a validated inverse, annealed reverse process, or direct
clean sampler. Compare derivatives at the same corruption scale unless such a
route is supplied. Jump/Lévy corruption may require a generator or Stein operator.

### 2.8 The learned mixture SSM matches mean and variance exactly

For each target, the learned matched-tail emission is

\[
Y\mid H\sim(1-p)\,\mathcal N\!\left(\mu,\frac{v a^2}{D(p)}\right)
+p\,\mathcal N\!\left(\mu,\frac{v b^2}{D(p)}\right),
\quad D(p)=(1-p)a^2+pb^2,
\]

with \(0<a<1<b\) and learned \(\mu(H),v(H),p(H)\). It is a normalized factorized
conditional law because its Gaussian component weights are nonnegative and sum to
one.

**Proposition 4 — exact moment matching.**

\[
E(Y\mid H)=\mu,\qquad
\operatorname{Var}(Y\mid H)
=\frac{v\{(1-p)a^2+pb^2\}}{D(p)}=v.
\]

For \(0<p<1\) and \(a\ne b\), its fourth centered moment is

\[
E[(Y-\mu)^4\mid H]
=3v^2\frac{(1-p)a^4+pb^4}{D(p)^2}>3v^2,
\]

because the numerator minus \(D(p)^2\) is
\(p(1-p)(b^2-a^2)^2>0\). Thus the learned mixture is an exact counterexample to
the claim that conditional mean and variance determine tail shape. It does not model
arbitrary cross-target covariance: the implemented joint emission is factorized
across targets conditional on the learned state.

### 2.9 Sequential validity requires a filtration

If \(d_t\) is \(\mathcal F_t\)-measurable and

\[ E_0\{\exp(\lambda d_t)\mid\mathcal F_{t-1}\}\le\exp(\lambda^2c_t/2) \]

for predictable \(c_t\), then

\[ M_t(\lambda)=\exp\{\sum_{s\le t}(\lambda d_s-\tfrac12\lambda^2c_s)\} \]

is a nonnegative supermartingale and
\(P_0(\sup_tM_t\ge1/\alpha)\le\alpha\). Learned increments and calibration must
be frozen or proved safe under the global filtration. Matched null simulations are
diagnostics, not a validity proof
[Howard et al. 2021](https://doi.org/10.1214/20-AOS1991).

## 3. Physical neuromodulator quantities versus reduced form

Let \(\mu_i(h,m,u)\), \(v_i(h,m,u)\), and \(\rho_{ab}(h,m,u)\) be oracle
conditional moments. Dimensionless physical susceptibilities are

\[
\begin{aligned}
C^{M,\mu}_{i\leftarrow k}&=(s_{m,k}/s_{y,i})\partial_{m_k}\mu_i,\quad
C^{M,\mathrm{disp}}_{i\leftarrow k}=s_{m,k}\partial_{m_k}\log v_i,\\
C^{M,\rho}_{ab\leftarrow k}&=s_{m,k}\partial_{m_k}\rho_{ab},\quad
C^{M,\mathrm{upper}}_{i\leftarrow k}(q)&=s_{m,k}\partial_{m_k}P(Y_i>q\mid h,m,u),\\
C^{M,\mathrm{shape}}_{i\leftarrow k}(a)&=s_{m,k}\partial_{m_k}
P\!\left(\left|(Y_i-\mu_i)/\sqrt{v_i}\right|>a\mid h,m,u\right).
\end{aligned}
\]

These are derivatives of raw probabilities, not logits. The fixed-threshold upper-tail
field mixes location, scale, and shape. The standardized field is deliberately two-sided;
for a Gaussian marginal it equals \(2\Phi(-a)\) and has zero derivative even when mean or
variance changes.

The implemented **total finite-time gated response** and a direct physical-drift
component are

\[
T^{\mathrm{total}}_{i\leftarrow j;k}(x,m)
=\frac{s_{m,k}s_{x,j}}{s_{y,i}}
\frac{\partial^2\mu_i(x,m)}{\partial m_k\,\partial x_j},
\qquad
C^{\mathrm{gate},f}_{i\leftarrow j;k}
=\frac{s_{m,k}s_{z,j}}{s_{f,i}}
\frac{\partial^2 f_i}{\partial m_k\,\partial z_j}.
\]

\(T^{\mathrm{total}}\) is a response property, not a unique parameter type. To see
why, write \(c^+=\rho m+(1-\rho)R(x)\), \(o=O(c^+)\), and
\(\mu_i=F_i(x,o)\). Its mixed derivative contains direct response gating
\(F_{i,x_jo}o_{m_k}\), but also terms such as
\(F_{i,oo}o_{x_j}o_{m_k}\) and \(F_{i,o}o_{x_jm_k}\) produced by
activity-dependent release, clearance, nonlinear receptor occupancy, additive drive,
and intrinsic feedback. Consequently even an off-diagonal source-target entry can be
nonzero without a direct synaptic-gain parameter. Self versus cross-neuron summaries
are descriptive; direct synaptic versus intrinsic versus release/receptor typing
requires the separate structural tensors and receptor/source/dose interventions.

For a fitted row-wise conditional mean \(F_i\), implementation uses

\[
\widehat T_{ijk}
=\frac{F_i(x+\delta_xe_j,m+\delta_me_k)
-F_i(x+\delta_xe_j,m-\delta_me_k)
-F_i(x-\delta_xe_j,m+\delta_me_k)
+F_i(x-\delta_xe_j,m-\delta_me_k)}
{4\delta_x\delta_m}.
\]

**Proposition 5 — finite-difference consistency.** If \(F_i\) is \(C^4\) on a
neighborhood of the probed state and the relevant fourth derivatives are bounded,
then \(\widehat T_{ijk}=\partial_{x_jm_k}F_i+O(\delta_x^2+\delta_m^2)\).
Thus the rectangle difference is consistent as both steps vanish. Fixed steps still
trade truncation error against floating-point cancellation; nonsmooth predictors do
not have this pointwise guarantee. The oracle uses a secant of its analytic
\(\partial_x\mu\): it is second order away from the positive-concentration boundary
and only first order if clipping turns it into a one-sided difference.

The gated-response metric evaluates at most 32 occupied complete states, uses fixed
scale-relative steps, and reports both the occupied-state field and its average. It
therefore estimates local-on-support response, not a global derivative field. It is
restricted to normalized row-wise horizon-one predictors; the recurrent latent SSM is
scored through its explicit learned mechanism tensors because row subsampling would alter
its state.

The upstream release kernel is

\[ C^{\mathrm{release}}_{k\leftarrow j,\ell}=\frac{s_{z,j\ell}}{s_{m,k}}\partial_{z_{j,\ell}}E(M_{t+1,k}\mid\text{physical state},U_t). \]

These M2 quantities, or targeted source/receptor/dose interventions, justify a
neuromodulator claim.

If \(M\) is hidden, the M1 reduced-form channels are

\[ C^\mu_{i\leftarrow j,\ell}=(s_{h,j\ell}/s_{y,i})\partial_{h_{j\ell}}\mu_i,\quad C^{\mathrm{disp}}_{i\leftarrow j,\ell}=s_{h,j\ell}\partial_{h_{j\ell}}\log v_i,\quad C^\rho_{ab\leftarrow j,\ell}=s_{h,j\ell}\partial_{h_{j\ell}}\rho_{ab}. \]

The fixed-threshold and standardized shape-tail channels are

\[
C^{\mathrm{tail,total}}_{i\leftarrow j,\ell}
=s_{h,j\ell}\partial_{h_{j\ell}}P(Y_i>q_i\mid H=h),
\quad
C^{\mathrm{tail,shape}}_{i\leftarrow j,\ell}
=s_{h,j\ell}\partial_{h_{j\ell}}
P\!\left(\left|(Y_i-\mu_i)/\sqrt{v_i}\right|>a\mid H=h\right).
\]

Both are raw-probability derivatives. The total tail mixes location, scale, and shape;
the standardized two-sided probability removes affine location/scale effects. For every
channel report

\[ \bar C=E_H C(H),\qquad C_{\mathrm{rms}}=\{E_H C(H)^2\}^{1/2}, \]

plus state-conditioned summaries and quantiles. RMS prevents sign cancellation.
Channel typing can mix under nonlinear target reparameterization.

## 4. Metric definitions

### 4.1 Predictive law

For density forecasts report held-out NLL and paired skill

\[ \Delta\mathrm{NLL}=\mathrm{NLL}_{\mathrm{baseline}}-\mathrm{NLL}_{\mathrm{method}}. \]

All densities must use the same observed space and dominating measure, including
change-of-variables Jacobians. For scalar targets,

\[ \mathrm{CRPS}(F,y)=\int\{F(z)-1(y\le z)\}^2\,dz. \]

For iid ensemble draws \(X_1,\ldots,X_M\), use the fair energy U-statistic

\[
\widehat{\mathrm{ES}}=\frac1M\sum_m\|X_m-y\|
-\frac{1}{2M(M-1)}\sum_{m\ne m'}\|X_m-X_{m'}\|.
\]

Use fixed train-derived scaling and fair U-statistics
[Ferro 2014](https://doi.org/10.1002/qj.2270).

The true-law expected energy or CRPS value depends on the outcome distribution and is
not generally zero. Zero is attainable for a degenerate copied observation, which is not
a lawful independent forecast draw; comparisons therefore use paired held-out outcomes
and the same registered sampling contract.

For atoms, use randomized PIT

\[ U=F(Y^-)+V\{F(Y)-F(Y^-)\},\qquad V\sim\mathrm{Uniform}(0,1). \]

Report PIT Cramér-von Mises error, coverage, within-episode serial PIT products,
and calibration by history, forecast scale, stimulus, environment, and worm.

Score joint paths at physical horizons. For one coherent, possibly
time-inhomogeneous process, test

\[ P_{s,t}(\cdot\mid h)=\int P_{u,t}(\cdot\mid x)P_{s,u}(dx\mid h),\quad s<u<t. \]

Only a declared homogeneous augmented-state model should be tested by \(P_\Delta^K\).

### 4.2 Dynamics, channels, and lag

For nonzero oracle field \(T\), use

\[ \operatorname{nISE}(\widehat T,T;\nu)=\frac{E_\nu\|\widehat T-T\|_F^2}{E_\nu\|T\|_F^2}. \]

Here \(\nu\) is the registered held-out occupied-state distribution, and truth and
estimate are evaluated on the same selected states. The implementation reports nISE,
absolute RMSE/MAE, field cosine, truth/estimate RMS, and RMS-map error. Pointwise error is
primary because an average Jacobian can cancel equal-and-opposite state effects. For an
exactly zero field, nISE and cosine are undefined; report estimated RMS, maximum absolute
magnitude, RMS-map error, and false support instead. Do not add an arbitrary denominator
and call it relative recovery.

The implemented rollout metric component retains fixed scaling for a fair finite-block
path energy score and targeted autocovariance, cross-spectral, escape/first-passage, and
extreme-frequency diagnostics. It requires joint samples with shape
`[case, draw, time, target]`; independently resampling each time point is invalid. The
bridge adapter is integrated; generic recursive samplers for the other methods remain
pending, so direct-horizon results cannot be relabeled as rollouts. The falsification uses
independent oracle path draws and retains their finite-sampling floor while detecting a
forecast with the right marginals and wrong temporal dependence. When origins or path
blocks overlap within an episode, uncertainty clusters by episode/worm. Occupancy, dwell,
switching, oscillation, and jump diagnostics remain future extensions.

For active typed channels report nISE, correlation/cosine, calibration slope, sign
accuracy above a preregistered beta-min threshold, and average precision. If \(\pi\) is
the active-edge prevalence, the implemented localization improvement is

\[ \mathrm{AP\ excess}=\mathrm{AP}-\pi. \]

The serialized key retains the legacy name `auprc_lift_over_prevalence`, but it is an
unnormalized excess, not a normalized lift. Precision at the true number of active edges
uses expected credit within a score tie.

Keep AUROC secondary under sparsity. For truth mechanism \(c\) and report \(d\),

\[
L_{cd}=
\frac{E\|\widehat C_d\,1_{\{\mathrm{truth}=c\}}\|^2}
{\sum_{d'}E\|\widehat C_{d'}\,1_{\{\mathrm{truth}=c\}}\|^2}.
\]

If the denominator is zero, record no recovered mass. Also report off-support false
energy because row-normalized leakage cannot see spurious edges.

Aggregate a state-dependent lag profile by

\[ a_\ell=\{E_\nu[C_\ell(H)^2]\}^{1/2},\qquad p_\ell=a_\ell/\sum_m a_m. \]

Report \(W_1(\widehat p,p)\) in seconds, peak/center-of-mass error, fast/slow mass,
signed profiles, and a no-recovered-mass flag.

### 4.3 Latent, interventional, and change metrics

Fit latent alignment only on validation systems. Report held-out Spearman, aligned
nRMSE, subspace angle, or conjugacy/input-output error appropriate to the
identification class. A causal filtering encoder may not use future data.

The desired common-history intervention target is, for intervention \(e\),

\[
R_i^e(k;h)=E[Y_{t+k,i}\mid do(e),H_t=h]
-E[Y_{t+k,i}\mid do(e_0),H_t=h].
\]

The signed law contrast \(P^e-P^0\) is not a probability distribution: score each arm
and compare response functionals. The integrated panel constructs an exactly shared
pre-operation history and reuses all named innovations after the branch. Pair index
\(b\) also reuses the same baseline and exogenous fingerprint across operations, so both
within-operation and cross-operation contrasts remain paired. Common random numbers are a
Monte Carlo design, not an identification theorem.

Three model-side estimands are deliberately different:

1. A passive predictor evaluated on each realized post-operation arm history receives P1
   population-mean and history-conditional transfer metrics.
2. An eligible latent model can invoke its represented receptor-specific operation on
   those arm histories. This is secondary, narrow C1. Multiple latent modulator labels
   require an anchored mapping; with one modulator the label is fixed.
3. Only lag one receives primary controlled C1, because both predictions then condition
   on the same realized branch-point history. Later predictions are teacher-forced on
   diverged arm histories. A recursively controlled multi-lag kernel is still pending.

The panel supports receptor-specific knockout, release-source silencing,
finite-duration ligand pulses, and expected-null shams. Population-mean response and
history-conditional pairwise error are both reported; pairwise nRMSE includes the defined
fraction and count. The ground truth reports Monte Carlo-SE RMS and its ratio to truth RMS.
Before any model is scored, an active operation must exceed a dimensionless
truth-RMS/innovation-RMS beta-min and remain below a registered MC-SE/truth-RMS precision
ceiling. A transient active operation must also meet a registered minimum fraction of
outputs with an observable post-peak \(1/e\) crossing. Expected-null shams must remain
below the null tolerance. These are truth precision/applicability gates, not model scores.

The response properties are nRMSE, sign, peak magnitude, peak time, finite-window signed
cumulative response, transient first-\(1/e\)-crossing error/censoring, persistent
end-window gain, and null leakage. A peak at the final lag is right-censored. Decay is
applicable only to a transient truth that crosses within the window; a persistent knockout
or source silencing instead receives end-window gain over the declared final-window
fraction. Real-data intervention claims still require consistency, randomization or
exchangeability, positivity, and a well-defined intervention.

For changepoints use independent null streams for threshold tuning and final FPR.
Report no-change FPR/alarms/ARL, calibrated power, unconditional and
call-conditional localization, detected-within-tolerance, detection and attribution
delay, post-change channel recovery, and multi-event matched precision/recall.
Post-selection estimation needs independent data, cross-fitting, or a caveat.

The independent unit is a DGP seed, system, worm, or intervention trial, not a time
point or edge. Use paired/hierarchical bootstrap at that level and block/HAC methods
for within-unit dependence.

The held-out-worm robustness lane draws disjoint train, validation, and test worms
with worm-specific recurrent coefficients, clearance times, receptor expression,
receptor affinity, and calcium decay. Support, coefficient signs, positivity, and
the recurrent stability bound are preserved. Tuning uses only reference-rendered
training/validation worms. The frozen estimator is evaluated on unseen test worms
and on their paired reference, high-noise, and slow-calcium renderings. Within a
test worm these renderings reuse exogenous noise and have a bitwise-identical latent
neural trajectory, so paired transfer deltas isolate observation-process degradation
for that simulated realization. Across-worm generalization and within-worm
measurement robustness remain different estimands; uncertainty must resample worms,
and neither establishes transport to a real worm population outside the simulated
hierarchy.

### 4.4 Implemented versus planned dynamical evidence

| Evaluation | Status | Claim |
| --- | --- | --- |
| separately fitted registered horizons | implemented | direct finite-horizon P1, not recursive rollout |
| one-step mean/variance/fixed-tail/shape operator probes | implemented | projected D1 |
| occupied-state history and physical response fields | implemented on complete state at horizon one | local-on-support D1/M2 |
| Brownian-bridge time-marginal energy/RMSE | implemented/integrated experimental lane | observed path-forecast D1 |
| Brownian reference path KL | implemented/integrated diagnostic | directionless reference-relative control effort |
| fair block energy, ACF/spectrum, escape, and extreme-frequency component | implemented/tested and bridge-integrated; other adapters pending | path-law D2 for valid joint rollouts |
| legacy zero/double ligand and all-receptor arm-history transfer | implemented | P1 |
| passive receptor/source/pulse arm-history responses | implemented/integrated | population-mean/history-conditional P1 |
| explicit latent-model receptor operation on registered arm histories | implemented/integrated for eligible labels | secondary narrow C1 |
| common-history receptor effect at lag one | implemented/integrated | primary controlled C1 |
| recursive controlled multi-lag response | pending | required for full controlled-kernel C1 |

## 5. Gate table

A general predictive-law model should pass these gates. A targeted estimator may
mark incompatible law metrics not applicable if its narrower estimand is declared.
VAR or SINDy needs an innovation and observation wrapper before NLL/path scoring.

| Gate | Primary evidence | Failure means |
| --- | --- | --- |
| 0 Numerical | finite losses, valid covariance, stable sampling | invalid model |
| 1 Law | NLL/CRPS/ES, PIT/coverage, path score | no predictive value/equivalence |
| 2 Dynamics | implemented finite-time probes/memory; rollout when a coherent sampler exists | wrong dynamics |
| 3 Mechanism | M1/M2 channel error, leakage, lag \(W_1\) | channel not recovered |
| 4 Latent | equivalence-aware state and kinetic metrics | latent claim unsupported |
| 5 Causal | secondary explicit arm-history operation and primary common-history lag-one effect | causal claim unsupported beyond the scored operation/lag |
| 6 Detection | FPR/ARL, power, delay, post-change recovery | monitoring unsupported |
| 7 Robustness | stress curves and worst-group loss | fragile result |
| 8 Biology | channel-matched enrichment and replication | no external consistency |

Do not average gates into one overall winner. Report a claim-specific profile.

## 6. Assumptions behind consistency

1. **Sufficient history:** history is adequate for the horizon, or truncation is
   treated as misspecification.
2. **Stationary or labeled regimes:** ergodic risk arguments apply within environments.
3. **Independent top-level units:** uncertainty uses systems, seeds, or animals.
4. **Score-compatible laws:** scores are finite and densities share a measure.
5. **Smooth positive law:** claimed derivatives exist and variance stays positive.
6. **Persistent excitation:** support covers the claimed state/intervention region.
7. **Identifiable frame:** state is physical, observation is identified, or metrics
   are quotient-invariant.
8. **Causal assumptions:** randomization or an explicit structural model matches.
9. **Beta-min/separation:** exact support needs signals above estimation error.
10. **Honest selection:** candidates are frozen, finite, or uniformly controlled.
11. **Finite-difference regularity:** mixed-derivative claims require a smooth
    predictor, interior support, and step-size sensitivity checks.
12. **Memory comparability:** a CMI interpretation requires the same target
    distribution/rows and consistent conditional-law fits at each history length.
13. **Population and intervention design:** held-out worms are draws from the declared
    hierarchy; paired response arms share history/innovations and differ only through the
    registered operation, with truth beta-min, Monte Carlo precision, and finite-window
    applicability checked before model scoring.

Log-score regret implies total variation through Pinsker. Energy/kernel regret
generally gives weak convergence only under its moment/kernel/domain conditions.
Moments additionally need uniform integrability; thresholds need no atom; quantiles
need uniqueness and local regularity. Derivative recovery needs a stronger
derivative-controlling norm, such as an appropriate Sobolev norm.

## 7. Method-specific interpretation

| Method family | What the implementation estimates | Highest routine benchmark claim | Ceiling / required assumption |
| --- | --- | --- | --- |
| VAR / Granger deletion | finite-lag conditional mean, optionally dispersion, and held-out source-deletion relevance | P1 and reduced-form M1 | structural impulse responses require identified shocks; hidden modulators and calcium filtering break a direct causal reading |
| SINDy / PySINDy | sparse discrete transition map in a declared library | projected D1 and reduced-form M1 | equation recovery requires correct coordinates/library; no NLL without an innovation wrapper |
| PCMCI-ParCorr | lagged conditional-dependence parents with worm datasets kept separate | reduced-form M1 | C2 requires the graphical assumptions and causal sufficiency; this is PCMCI with ParCorr, not PCMCI+ |
| VAR-LiNGAM | lagged linear coefficients from the endogenous neural state; instantaneous matrix retained separately | reduced-form M1, or structural only under its model | linearity, non-Gaussian mutually independent errors, acyclic contemporaneous structure, and no latent confounding are essential |
| SID / Hyvärinen score matching | quadratic conditional score; a normalized Gaussian law only when the fitted precision is valid | P1 for the valid normalized fit; typed M1 fields | numerical validity and the parameterization are part of the claim; no automatic M2 |
| DSM neural score | conditional outcome score corrupted by one registered reference kernel/scale | S1 and compatible M1 probes | cross-kernel, cross-scale, or cross-domain risks are different estimands; a clean law needs a validated inverse or sampler |
| SBTG | joint consecutive-state score on \([x_t,y_{t+1}]\) and score-based localization statistics | fixed-reference S1 and reduced-form M1 localization | its risk cannot rank against conditional-outcome DSM; a good DSM risk does not make a normalized density or guarantee a correct graph readout |
| Learned neuromodulator SSM | normalized Gaussian/matched-tail law, positive latent recurrence, typed tensors, and explicit receptor operation | P1, equivalence-aware L1, model-matched M2, narrow C1 | concentration/\(K_d\) and expression/effect gauges prevent general physical parameter identification; factorized tail emissions do not learn a copula |
| Conditional Brownian bridge | normalized endpoint law and reference-conditioned joint paths | endpoint P1, time-marginal D1, bridge-supplied D2 | reference diffusion is a fixed stratum; reference KL is directionless effort, and no M2/C1 follows |

Mean atlas or connectome overlap remains B1 secondary for every method; it is not
dispersion, tail, latent-concentration, or physical \(M\) truth.

## Primary literature

- [Proper scores: Gneiting and Raftery 2007](https://doi.org/10.1198/016214506000001437)
- [Calibration: Gneiting et al. 2007](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
- [Fair ensembles: Ferro 2014](https://doi.org/10.1002/qj.2270)
- [Prequential evaluation: Dawid 1984](https://doi.org/10.2307/2981683)
- [Score matching: Hyvärinen 2005](https://www.jmlr.org/papers/v6/hyvarinen05a.html)
- [Denoising score matching: Vincent 2011](https://doi.org/10.1162/NECO_a_00142)
- [SINDy: Brunton et al. 2016](https://doi.org/10.1073/pnas.1517384113)
- [Latent identification: Khemakhem et al. 2020](https://proceedings.mlr.press/v108/khemakhem20a.html)
- [Structural identifiability: Villaverde et al. 2016](https://doi.org/10.1371/journal.pcbi.1005153)
- [Granger causality: Granger 1969](https://doi.org/10.2307/1912791)
- [VAR-LiNGAM: Hyvärinen et al. 2010](https://www.jmlr.org/papers/v11/hyvarinen10a.html)
- [PCMCI: Runge et al. 2019](https://doi.org/10.1126/sciadv.aau4996)
- [DYNOTEARS: Pamfil et al. 2020](https://proceedings.mlr.press/v108/pamfil20a.html)
- [Schrödinger problem: Léonard 2014](https://doi.org/10.3934/dcds.2014.34.1533)
- [Diffusion bridges: De Bortoli et al. 2021](https://papers.nips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html)
- [Time-uniform inference: Howard et al. 2021](https://doi.org/10.1214/20-AOS1991)
- [Neuromodulation: Marder 2012](https://doi.org/10.1016/j.neuron.2012.09.010)
- [Synaptic modulation: Nadim and Bucher 2014](https://doi.org/10.1016/j.conb.2014.05.003)
- [Multilayer connectome: Bentley et al. 2016](https://doi.org/10.1371/journal.pcbi.1005283)
- [Signal-propagation atlas: Randi et al. 2023](https://doi.org/10.1038/s41586-023-06683-4)
