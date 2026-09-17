# Compatibility-Aware Neural Path Responses with Conditional Flows

## A detailed technical guide to wide-flow direct importance weighting and progressive bridge SMC

**Document status:** Technical methods guide and results interpretation  
**Project:** learned stochastic dynamics for whole-brain *C. elegans* calcium activity  
**Methods covered:** wide conditional flow + direct paired importance weighting; conditional flow + progressive bridge sequential Monte Carlo (SMC)  
**Methods deliberately excluded from the tutorial:** SBTG and Hessian/score-product estimators  
**Last updated:** 2026-08-28

---

## Executive summary

The project has two strong non-SBTG approaches for turning a learned neural dynamics model into a source-neuron-by-target-neuron response matrix.

1. **Wide-flow direct importance weighting** was the strongest complete historical pipeline on the independent Randi and Cook comparisons. It first learns a conditional distribution for the next 54-neuron activity vector. It then generates natural neural paths and reweights those same paths according to whether a selected source neuron is compatible with a low or high activity state, while the rest of the population remains close to a real episode. It is fast and simple. It can fail when only a few generated paths receive nearly all of the weight.

2. **Progressive bridge SMC** was the strongest finite-particle estimator of the declared repaired-path distribution in a frozen estimator benchmark. It does not wait until the end of a path to impose the source constraint. It introduces that constraint gradually, branches particles, monitors effective sample size, and resamples when necessary. It is slower, but it reduced both bias and Monte Carlo variance relative to direct importance weighting and terminal-clamp SMC.

These two conclusions are not contradictory:

- The **best complete biological-correspondence pipeline** was the stronger wide generator combined with direct importance weighting.
- The **best sampler for a fixed repaired-path law** was progressive bridge SMC.

The historical external-reference results were:

| Complete pipeline | Randi WT AUROC / AUPRC | Cook structural AUROC / AUPRC |
| --- | ---: | ---: |
| Wide-flow direct importance | **0.619 / 0.298** | **0.591 / 0.349** |
| Progressive bridge SMC | 0.596 / 0.251 | 0.564 / 0.325 |

The two pipelines did **not** use the same fitted generator in these historical comparisons. Therefore, these values do not isolate the sampler. In the fixed-generator sampler benchmark, progressive SMC had response mean-squared error 0.0110, versus 0.0307 for direct importance weighting.

The central scientific qualification is equally important. These matrices are **model-relative, compatibility-aware observational responses**. They are not automatically causal interventions, synapses, receptor effects, or physical transmission delays. Moreover, the strongest historical matrices varied the future response horizon while keeping the source window immediately before the cut. They therefore show broad effective response dynamics, but they do not yet establish reliable source-lag localization.

---

## 1. The scientific question

Let the observed whole-brain calcium activity at frame \(t\) be

\[
X_t = (X_{t,1},\ldots,X_{t,d}) \in \mathbb{R}^d,
\]

where \(d=54\) in the current shared-neuron analysis. Let \(S_t\) describe the externally supplied stimulus state. The first modeling goal is to learn the complete conditional distribution

\[
p(X_{t+1}\mid X_{t-L+1:t},S_{t-L+1:t}).
\]

This is more ambitious than predicting only a conditional mean. The conditional distribution can represent uncertainty, correlated fluctuations across neurons, heavy or asymmetric tails, and potentially multiple response modes.

The second goal is to use that learned stochastic law to ask a source-target question:

> If source neuron \(j\) occupies a naturally compatible high state instead of a naturally compatible low state, how does the learned future distribution of target neuron \(k\) change?

The phrase **naturally compatible** is essential. A naive method could replace one neural coordinate by an arbitrary number and roll the model forward. Such a hard edit may create a history that never occurred in the data. A neural network will still output a prediction, but that prediction can depend on an unconstrained off-support extrapolation.

The compatibility-aware method instead samples histories from the learned natural dynamics and changes their relative probability. It favors histories that meet the requested source condition while remaining compatible with a real population episode.

---

## 2. What each component does

The complete workflow has three conceptually separate components:

\[
\text{observed data}
\longrightarrow
\text{learned transition generator}
\longrightarrow
\text{repaired-path sampler}
\longrightarrow
\text{response matrix}.
\]

### 2.1 The learned transition generator

The generator approximates

\[
Q_\theta(X_{t+1}\mid H_t,\Sigma_t),
\]

where

\[
H_t=(X_{t-L+1},\ldots,X_t),
\qquad
\Sigma_t=(S_{t-L+1},\ldots,S_t).
\]

It answers a predictive question: given a recent observed history and known stimulus schedule, what next neural vectors are plausible?

### 2.2 The repair potential

The repair potential gives a high score to generated prefixes that:

- have the requested low or high source-neuron statistic; and
- keep the remaining population close to a factual held-out episode.

It does not alter the learned transition rule after the cut. It alters which naturally generated prefixes are emphasized before the cut.

### 2.3 The Monte Carlo estimator

The exact repaired law is usually impossible to integrate analytically. Monte Carlo methods approximate it.

- Direct importance weighting generates a complete bank of natural paths and reweights them at the end.
- Progressive SMC changes the particle population during prefix generation so that more particles survive in regions relevant to the final repair query.

This separation matters. A weak final result can arise from at least three different sources:

1. **Generator error:** \(Q_\theta\) does not approximate the observational dynamics well enough.
2. **Sampler error:** the Monte Carlo procedure poorly approximates the repaired law under \(Q_\theta\).
3. **Semantic gap:** the repaired observational response is not the same scientific object as a physical intervention.

No single metric diagnoses all three.

---

## 3. Notation and time indexing

| Symbol | Meaning |
| --- | --- |
| \(t\) | discrete frame index |
| \(d\) | number of neurons; \(d=54\) for the shared analysis |
| \(X_t\in\mathbb R^d\) | standardized observed neural activity vector |
| \(S_t\) | aligned stimulus encoding |
| \(L\) | declared input history length; historically 80 frames |
| \(H_t\) | neural history \(X_{t-L+1:t}\) |
| \(\Sigma_t\) | stimulus history \(S_{t-L+1:t}\) |
| \(j\) | source-neuron index |
| \(k\) | target-neuron index |
| \(B\) | repair-prefix length; historically 4 frames |
| \(c\) | temporal cut: repair ends and free future rollout begins |
| \(h\) | future response horizon after the cut |
| \(\tau\) | source-window offset before the cut |
| \(A_j\) | scalar statistic of source \(j\) over its repair window |
| \(a_j^-,a_j^+\) | fold-local low and high targets, usually 25th and 75th percentiles |
| \(\epsilon_j\) | width of the soft source clamp |
| \(\lambda\) | strength of the factual population anchor |
| \(P\) | natural path law induced by the learned generator |
| \(Q_j^-,Q_j^+\) | low- and high-repaired path laws for source \(j\) |
| \(F_h^k\) | selected future feature of target \(k\) through horizon \(h\) |
| \(M_h(k,j)\) | normalized response coefficient; target on row, source on column |

At 4 Hz:

- 1 frame = 0.25 seconds;
- 4 frames = 1 second;
- 31 frames = 7.75 seconds;
- 80 frames = 20 seconds;
- 40 future frames = 10 seconds.

---

## 4. Data and stimulus scope

### 4.1 Historical response cohort

The historical strong results used:

- 20 worms;
- 54 complete-case neurons shared across the analysis;
- nominal 4 Hz analysis time;
- 80 input frames, corresponding to 20 seconds of declared history;
- whole-worm cross-validation rather than random overlapping-window splits;
- five outer folds and three generator seeds for the response-ready ensemble.

Fold-local standardization was used. Means, scales, source quantiles, anchor projections, and thresholds were estimated from training worms only.

### 4.2 The raw data contain three chemicals

The source recordings contain three distinct stimuli:

| Code | Chemical |
| ---: | --- |
| 1 | butanone |
| 2 | pentanedione |
| 3 | NaCl |

Their presentation order is counterbalanced across worms. Therefore, presentation number is not equivalent to chemical identity.

### 4.3 What the historical best model actually conditioned on

The historical wide-flow winner used a single **binary any-stimulus channel**:

\[
S_t=
\begin{cases}
1,&\text{some stimulus is active at frame }t,\\
0,&\text{no stimulus is active.}
\end{cases}
\]

Thus, the historical external-reference performance is evidence for a generic stimulus-conditioned neural model. It is **not** evidence that the model learned butanone-, pentanedione-, and NaCl-specific transition laws.

A previous presentation-position encoding used temporal slot identity rather than chemical identity. Because the chemical order was counterbalanced, that run must not be interpreted as a chemical-specific experiment.

### 4.4 Cohort-provenance warning

The pooled 20-worm cohort contains 17 OH16230 worms and 3 OH15500 worms. The strains have slightly different native sampling and schedule details; OH15500 was acquired at 4.1 Hz and has a 20-second final stimulus epoch. Corrected chemical-specific work now either:

- uses OH16230 head recordings at their native 4.0 Hz as the primary cohort; or
- explicitly resamples OH15500 from 4.1 Hz to 4.0 Hz before windowing and preserves its true event duration as a sensitivity analysis.

This does not erase the historical generic-response results, but it limits how specifically they can be interpreted. Any new chemical claim must use the corrected schedule-aware pipeline.

---

## 5. The wide conditional flow model

### 5.1 Why learn a distribution rather than a point prediction?

A deterministic regression model estimates a function such as

\[
\mu_\theta(H_t,\Sigma_t)
=
\mathbb E[X_{t+1}\mid H_t,\Sigma_t].
\]

This is useful, but it discards other features of the conditional law. Two neural states can have the same conditional mean while differing in:

- variance or gain;
- cross-neuron covariance;
- tail probability;
- switching probability between response regimes;
- probability of crossing a behaviorally meaningful threshold.

A conditional generative model instead provides samples

\[
\widetilde X_{t+1}^{(m)}
\sim
Q_\theta(\cdot\mid H_t,\Sigma_t),
\qquad m=1,\ldots,M.
\]

These samples can approximate any future feature that can be computed from simulated paths.

### 5.2 Residual target

The winning flow predicts the one-step residual

\[
R_{t+1}=X_{t+1}-X_t
\]

rather than \(X_{t+1}\) directly. A sampled next state is reconstructed as

\[
\widetilde X_{t+1}=X_t+\widetilde R_{t+1}.
\]

This is often easier for slowly varying calcium traces because persistence is represented explicitly and the stochastic model focuses on changes around that persistent state.

### 5.3 Temporal convolutional encoder

The model concatenates the neural and stimulus values at every history frame. For the historical binary model, the input tensor per window has shape

\[
80\times(54+1)=80\times55.
\]

A causal temporal convolutional network (TCN) maps this tensor to a context vector \(C_t\in\mathbb R^{128}\). Each TCN block uses:

- a causal width-3 convolution;
- a dilation;
- GroupNorm;
- SiLU nonlinearity;
- dropout;
- a residual connection.

The historical winner used dilations

\[
1,2,4,8.
\]

For kernel width 3, the receptive field is

\[
1+2(1+2+4+8)=31\text{ frames}.
\]

This is a crucial distinction:

- the checkpoint declares and receives an 80-frame input;
- the legacy TCN's final representation can depend directly on only the most recent 31 frames.

Therefore, the model had a 20-second input tensor but an effective causal receptive field of 7.75 seconds. Claims about effects beyond 31 frames are structurally unsupported by this specific encoder. Later full-history TCNs added larger dilations, but they are not the historical wide-flow model described here.

### 5.4 Conditional flow matching

The distributional head learns a time-dependent velocity field that transports standard Gaussian noise into a neural residual sample.

Let

\[
Z_0\sim\mathcal N(0,I_d)
\]

be a base sample and let

\[
Z_1=R_{t+1}
\]

be the observed standardized residual. Draw an interpolation time

\[
U\sim\operatorname{Uniform}(0,1)
\]

and define the linear interpolation

\[
Z_U=(1-U)Z_0+UZ_1.
\]

The velocity along this straight path is

\[
\frac{dZ_U}{dU}=Z_1-Z_0.
\]

The network \(v_\theta(z,u,C_t)\) is trained by squared-error regression:

\[
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E
\left[
\left\|
v_\theta(Z_U,U,C_t)-(Z_1-Z_0)
\right\|_2^2
\right].
\]

The implementation supplies three time features to the head:

\[
U,\qquad \sin(\pi U),\qquad \cos(\pi U).
\]

For the historical wide model:

- TCN width: 128;
- encoder dropout: 0.10;
- weight decay: \(5\times10^{-4}\);
- learning rate: \(3\times10^{-4}\);
- flow-head hidden width: 128;
- flow-head residual layers: 4;
- numerical sampling steps: 24.

### 5.5 Sampling from the flow

At prediction time, draw \(Z(0)\sim\mathcal N(0,I_d)\) and solve

\[
\frac{dZ(u)}{du}=v_\theta(Z(u),u,C_t),
\qquad u\in[0,1].
\]

The implementation uses 24 fixed Heun steps. One Heun update from \(u_r\) to \(u_{r+1}\) is:

\[
\begin{aligned}
k_1 &= v_\theta(Z_r,u_r,C_t),\\
\widetilde Z_{r+1} &= Z_r+\Delta u\,k_1,\\
k_2 &= v_\theta(\widetilde Z_{r+1},u_{r+1},C_t),\\
Z_{r+1} &= Z_r+\frac{\Delta u}{2}(k_1+k_2).
\end{aligned}
\]

The terminal value \(Z(1)\) is treated as a residual sample. Adding the latest observed neural vector gives the next state.

### 5.6 From a one-step model to a path law

The one-step generator is applied recursively. Given a fixed initial history and a fixed future stimulus schedule, it induces the path law

\[
P_\theta(x_{1:T})
=
\prod_{r=1}^{T}
Q_\theta(x_r\mid x_{r-L:r-1},s_{r-L:r-1}).
\]

At every step, the oldest frame is dropped, the sampled frame is appended, and the next state is sampled from the updated history. This ancestral recursion is the natural proposal law for both direct importance weighting and SMC.

The stimulus is exogenous. Its observed schedule is supplied during rollout rather than generated by the neural model.

---

## 6. Predictive model selection

The generator was selected before loading Randi, Cook, Bentley, SBTG, or any anatomical reference. This **atlas firewall** prevents direct tuning of the predictive model to the downstream biological comparisons.

### 6.1 Energy score

For one observed target vector \(y\in\mathbb R^d\) and \(M\) predictive samples \(x^{(1)},\ldots,x^{(M)}\), the sample energy score is

\[
\widehat{\operatorname{ES}}
=
\frac{1}{M}\sum_{m=1}^{M}\|x^{(m)}-y\|_2
-
\frac{1}{2M^2}\sum_{m=1}^{M}\sum_{m'=1}^{M}
\|x^{(m)}-x^{(m')}\|_2.
\]

Lower is better. The first term rewards closeness to the observation. The second term rewards an appropriately dispersed predictive sample rather than a collapsed point mass. The energy score is proper: in expectation, the true predictive distribution minimizes it.

### 6.2 Stimulus-balanced energy

Quiet windows are much more common than transition or active-stimulus windows. A simple pooled average could therefore select a model that performs well during quiet periods but poorly around stimulation. The stimulus-balanced score first averages within the observed stimulus-history strata and then gives those strata equal influence.

### 6.3 Variogram score

The variogram score evaluates whether the predictive samples reproduce pairwise relationships among output dimensions. In generic form,

\[
\operatorname{VS}
=
\sum_{k<\ell}w_{k\ell}
\left(
|y_k-y_\ell|^p
-
\mathbb E|X_k-X_\ell|^p
\right)^2.
\]

It complements the energy score by emphasizing cross-neuron dependence. Lower is better.

### 6.4 Calibration and sharpness

Coverage90 is the fraction of observed components that fall within the nominal 90% predictive interval. Ideal marginal calibration is near 0.90. Coverage alone is insufficient because arbitrarily wide intervals can achieve high coverage; sharpness reports their width.

### 6.5 Frozen predictive result

The confirmation results for the selected wide flow were:

| Metric | Value |
| --- | ---: |
| Energy | 0.850241 |
| Stimulus-balanced energy | 0.855514 |
| Variogram score | 0.022471 |
| Coverage90 | 0.8965 |

Its natural-energy advantage over the next regularized flow was about 0.21%, smaller than the reported fold/seed standard errors. The appropriate conclusion is a near tie between the leading regularized flows, with the wide model selected by the frozen rule for reproducibility. It is not evidence that width 128 is uniquely optimal.

---

## 7. Why naive history editing is unsafe

Suppose genuine observations satisfy an approximate relationship between two coordinates, such as \(U_t\approx V_t\). A fitted predictor may agree perfectly on the observed manifold but behave arbitrarily when only \(U_t\) is changed.

For example, consider

\[
m_\beta(h)=m_0(h)+\beta(U_t-V_t).
\]

Whenever genuine histories satisfy \(U_t=V_t\), every value of \(\beta\) gives the same predictions on observed data. After the hard edit

\[
(U_t,V_t)=(u_0,u_0)\mapsto(u,u_0),
\]

the prediction changes by

\[
\beta(u-u_0).
\]

Thus, ordinary held-out accuracy cannot determine the hard-edit response if the edit leaves the support of the observed history distribution.

The repaired-path strategy avoids assigning scientific meaning to a single artificial vector. It keeps the natural path law and changes the relative weight assigned to naturally generated prefixes.

---

## 8. The compatibility-aware repaired-path estimand

### 8.1 Natural prefix and future

Choose a cut time \(c\). The repair interval contains \(B\) generated frames ending at \(c\). The future contains up to \(K\) unweighted generated frames after \(c\).

For the historical primary setting:

\[
B=4\text{ frames}=1\text{ second},
\]

and

\[
K\in\{1,2,4,8,16,24,32,40\}\text{ frames}.
\]

The factual history before repair initializes the generator. Let \(\omega\) denote a complete generated repair-prefix-plus-future path. The natural learned path law is \(P_\theta(d\omega)\).

### 8.2 Source statistic

For source neuron \(j\), define its average activity over the source window:

\[
A_j(\omega)
=
\frac{1}{W}
\sum_{r\in\mathcal W_\tau}
X_{r,j}(\omega),
\]

where \(W=4\) frames in the historical primary setting and \(\mathcal W_\tau\) is the source window at offset \(\tau\) before the cut.

For each cross-validation fold and episode phase, training worms determine:

\[
a_j^-=Q_{0.25}(A_j),
\qquad
a_j^+=Q_{0.75}(A_j),
\qquad
\operatorname{IQR}_j=a_j^+-a_j^-.
\]

The implementation floors the IQR at 0.20 standardized units.

### 8.3 Soft source clamp

For target level \(a\), the Gaussian source potential is

\[
K_{\epsilon_j}(A_j-a)
=
\exp\left[
-\frac{1}{2}
\left(\frac{A_j-a}{\epsilon_j}\right)^2
\right].
\]

The primary bandwidth is

\[
\epsilon_j
=
\max\{0.25\operatorname{IQR}_j,0.10\}.
\]

This is a soft condition, not an exact equality. A path close to the target receives high weight; a path far from the target receives low weight.

The bandwidth controls a bias-variance tradeoff:

- a small \(\epsilon_j\) approximates a sharp condition but can select almost no path mass;
- a large \(\epsilon_j\) has better overlap but weakens the distinction between low and high states.

### 8.4 Factual population anchor

Let \(x_r^\star\) be the factual population vector during the selected held-out episode. The method penalizes generated repair prefixes that depart too far from the factual population context.

The implementation learns a rank-12 whitened PCA projection \(P\in\mathbb R^{d\times12}\) from the training worms. For generated prefix \(x_r\), define a projected deviation

\[
z_r=(x_r-x_r^\star)P.
\]

During the source window, the designated source coordinate is subtracted from this projected deviation. This prevents the anchor from simultaneously pulling source \(j\) toward its factual value while the clamp pulls it toward \(a_j^-\) or \(a_j^+\).

The source-specific anchor cost is, schematically,

\[
C_j(\omega,x^\star)
=
\frac{1}{2Br}
\sum_{r=1}^{B}
\left\|
\left[M_{j,r}(x_r-x_r^\star)\right]P
\right\|_2^2,
\]

where \(r=12\) is the anchor rank and \(M_{j,r}\) excludes the source only during its designated source window.

The anchor potential is

\[
G_{\mathrm{anchor},j}(\omega)
=
\exp[-\lambda C_j(\omega,x^\star)].
\]

The historical primary setting used

\[
\lambda=0.25.
\]

### 8.5 Total repair potential

For low or high target \(a_j^q\), where \(q\in\{-,+\}\), define

\[
G_j^q(\omega)
=
K_{\epsilon_j}(A_j(\omega)-a_j^q)
\times
\exp[-\lambda C_j(\omega,x^\star)].
\]

The repaired law is

\[
Q_j^q(d\omega)
=
\frac{G_j^q(\omega)}{Z_j^q}
P_\theta(d\omega),
\]

with normalizer

\[
Z_j^q
=
\mathbb E_{P_\theta}[G_j^q(\omega)].
\]

This formula gives the method its precise meaning:

- \(P_\theta\) says which paths the learned natural dynamics generate;
- \(G_j^q\) says which natural paths are compatible with the query;
- \(Z_j^q\) renormalizes the selected paths into a probability distribution.

The factual episode is an anchor. The primary contrast is **high-repaired versus low-repaired**, not “edited versus factual.”

### 8.6 What repair changes

Repair changes the mixing distribution over prefixes. After the cut, particles are released and rolled forward using the unchanged learned transition kernel:

\[
Q_j^q(X_{c+1:c+K}\mid X_{c-B+1:c})
=
P_\theta(X_{c+1:c+K}\mid X_{c-B+1:c}).
\]

Therefore, the method does not insert an additional force or clamp into the future. It asks what the existing learned dynamics predict from differently selected, but model-generated, prefix states.

---

## 9. Future response features and matrix construction

### 9.1 Cumulative future mean

The primary historical matrix used the cumulative future mean of target neuron \(k\):

\[
F_h^k(\omega)
=
\frac{1}{h}
\sum_{r=1}^{h}X_{c+r,k}(\omega).
\]

For source \(j\), define

\[
\mu_{h,k,j}^q
=
\mathbb E_{Q_j^q}[F_h^k],
\qquad q\in\{-,+\}.
\]

The raw high-minus-low response is

\[
\Delta_{h,k,j}
=
\mu_{h,k,j}^{+}-\mu_{h,k,j}^{-}.
\]

### 9.2 Achieved source displacement

Because the source clamp is soft, the selected particles do not generally achieve the requested quartiles exactly. Define

\[
\widetilde a_j^q
=
\mathbb E_{Q_j^q}[A_j].
\]

The achieved source gap is

\[
g_j=\widetilde a_j^+-\widetilde a_j^-.
\]

The matrix coefficient is

\[
M_h(k,j)
=
\frac{\Delta_{h,k,j}}
{\max(g_j,0.10)}.
\]

The denominator makes responses more comparable across source neurons. A raw response of 0.2 following an achieved source separation of 1.0 is not treated as equivalent to a raw response of 0.2 following a separation of 0.1.

The 0.10 floor prevents numerical explosion. It is not a substitute for the validity gate.

### 9.3 Matrix orientation

The archived orientation is:

- **row \(k\): target neuron**;
- **column \(j\): source neuron**.

Thus, \(M_h(k,j)\) reads “response of target \(k\) associated with a high-versus-low repaired state of source \(j\).”

### 9.4 Sign interpretation

- \(M_h(k,j)>0\): high source-compatible paths predict a larger target feature than low source-compatible paths.
- \(M_h(k,j)<0\): high source-compatible paths predict a smaller target feature.
- \(M_h(k,j)\approx0\): the selected feature changes little, or the estimate has insufficient information.

The sign is a property of a specified model, anchor, phase, source window, outcome, and horizon. It is not automatically excitatory or inhibitory synaptic sign.

### 9.5 Other features already supported

The path-response implementation also computes:

- endpoint mean \(X_{c+h,k}\);
- cumulative mean through \(h\);
- running peak mean;
- probability that the running peak exceeds a fold-local 90th-percentile threshold;
- endpoint standard deviation.

The external Randi/Cook results discussed below primarily used the normalized cumulative-mean response.

---

## 10. Method 1: wide-flow direct paired importance weighting

### 10.1 Core identity

For any integrable future feature \(F\),

\[
\mathbb E_{Q_j^q}[F]
=
\frac{
\mathbb E_{P_\theta}[F(\omega)G_j^q(\omega)]
}{
\mathbb E_{P_\theta}[G_j^q(\omega)]
}.
\]

This follows directly from the repaired-law density:

\[
\begin{aligned}
\mathbb E_{Q_j^q}[F]
&=
\int F(\omega)Q_j^q(d\omega)\\
&=
\int F(\omega)
\frac{G_j^q(\omega)}{Z_j^q}
P_\theta(d\omega)\\
&=
\frac{
\mathbb E_{P_\theta}[FG_j^q]
}{Z_j^q}\\
&=
\frac{
\mathbb E_{P_\theta}[FG_j^q]
}{
\mathbb E_{P_\theta}[G_j^q]
}.
\end{aligned}
\]

This identity allows estimation using only samples from the learned generator and evaluations of the repair potential. A normalized flow density is not required.

### 10.2 Self-normalized importance estimator

Generate \(N\) complete natural paths

\[
\omega^{(1)},\ldots,\omega^{(N)}
\overset{\mathrm{iid}}{\sim}P_\theta.
\]

For source \(j\) and condition \(q\), calculate

\[
w_{n,j}^q=G_j^q(\omega^{(n)}),
\]

then normalize:

\[
\bar w_{n,j}^q
=
\frac{w_{n,j}^q}{\sum_{m=1}^{N}w_{m,j}^q}.
\]

The repaired expectation is approximated by

\[
\widehat\mu_{h,k,j}^q
=
\sum_{n=1}^{N}
\bar w_{n,j}^qF_h^k(\omega^{(n)}).
\]

The same natural path bank is reused for:

- every source neuron;
- the low and high conditions;
- every target neuron;
- every response horizon;
- every supported future feature.

This reuse is why the method is fast. Low and high are also statistically paired because both are weighted versions of the same generated paths.

### 10.3 Historical direct configuration

The primary historical ensemble used:

| Setting | Value |
| --- | ---: |
| Natural paths per episode/checkpoint | 128 |
| Repair frames | 4 |
| Source-window frames | 4 |
| Source lag before cut | 0 |
| Anchor rank | 12 |
| Anchor strength \(\lambda\) | 0.25 |
| Clamp bandwidth fraction | 0.25 of fold-local IQR |
| Horizon frames | 1, 2, 4, 8, 16, 24, 32, 40 |
| Whole-worm folds | 5 |
| Generator seeds | 3 |
| Neurons | 54 |

The direct algorithm generates the 4-frame repair prefix and the entire free future before weights are computed. There is no resampling.

### 10.4 Small numerical example

Suppose four generated paths have source statistics and future target features:

| Path | Source statistic \(A_j\) | Target feature \(F\) |
| ---: | ---: | ---: |
| 1 | 0.0 | 0.1 |
| 2 | 0.5 | 0.4 |
| 3 | 1.0 | 0.8 |
| 4 | 1.5 | 1.0 |

Assume the normalized high weights are

\[
\bar w^+=(0.006,0.077,0.346,0.571)
\]

and the normalized low weights are

\[
\bar w^-=(0.571,0.346,0.077,0.006).
\]

Then

\[
\widehat\mu^+
=
0.006(0.1)+0.077(0.4)+0.346(0.8)+0.571(1.0)
=0.879,
\]

while

\[
\widehat\mu^-
=
0.571(0.1)+0.346(0.4)+0.077(0.8)+0.006(1.0)
=0.263.
\]

The raw response is

\[
0.879-0.263=0.616.
\]

The achieved source levels are approximately 1.241 and 0.259, so the achieved gap is 0.982. The normalized response is approximately

\[
0.616/0.982=0.627.
\]

The important point is that no path was edited. The method changed how strongly each naturally generated path contributed to each expectation.

### 10.5 Effective sample size

The effective sample size is

\[
\operatorname{ESS}
=
\frac{(\sum_n w_n)^2}{\sum_nw_n^2}
=
\frac{1}{\sum_n\bar w_n^2}.
\]

Interpretation:

- equal weights give \(\operatorname{ESS}=N\);
- one dominant path gives \(\operatorname{ESS}\approx1\);
- intermediate values measure concentration of the weighted sample.

ESS is a Monte Carlo overlap diagnostic. It does not prove that the learned generator is biologically correct.

### 10.6 Historical validity gate

A source query was declared valid only when both low and high systems satisfied:

- ESS at least 12.8 for the 128-path production configuration;
- maximum normalized weight at most 0.20;
- achieved source gap at least 25% of the requested fold-local interquartile gap.

The method also archives source, total, and repair normalizers; achieved source levels; anchor costs; entropy-equivalent ancestor counts; and sensitivity to anchor/clamp settings.

### 10.7 Strengths

- Simple derivation.
- No transition density or score is needed.
- One path bank supports all source-target-horizon queries.
- Low and high estimates use the same underlying paths.
- Very fast relative to SMC.
- Easy to audit using weights, ESS, and achieved source displacement.

### 10.8 Weaknesses

- If the desired source state is rare, a few paths dominate.
- Generating more future steps for paths that later receive negligible weight wastes computation.
- Self-normalized importance estimates are biased at finite \(N\), although consistent as \(N\to\infty\) under standard overlap conditions.
- A finite bank can miss an important compatible mode entirely.
- Weight stability does not guarantee stability across fitted generator seeds or worms.

### 10.9 Direct pseudocode

~~~text
INPUT:
    fitted generator Q_theta
    factual boundary history and fixed stimulus schedule
    source low/high targets and IQRs from training worms
    factual anchor projection
    N natural paths

1. Generate N natural repair prefixes from Q_theta.
2. Continue each prefix freely through the largest future horizon.
3. For every source j:
       a. Compute source statistic A_j for every path.
       b. Compute source-specific factual-anchor cost C_j for every path.
       c. For q in {low, high}:
              log_weight[n] =
                  -0.5 * ((A_j[n] - target_j[q]) / epsilon_j)^2
                  -lambda * C_j[n]
              normalize log weights with log-sum-exp
              compute ESS, max weight, normalizers, and achieved source level
              compute weighted future features for every target and horizon
       d. Subtract low future features from high future features.
       e. Divide by max(achieved high-low source gap, 0.10).
       f. Mark the source invalid if any diagnostic gate fails.
4. Average episode-level matrices within the declared phase, worm, seed,
   and fold hierarchy.
OUTPUT:
    response matrices and complete support/Monte Carlo diagnostics
~~~

---
## 11. Method 2: progressive bridge sequential Monte Carlo

### 11.1 Why SMC is needed

Direct importance weighting draws all paths from the unmodified natural law. If the final query is selective, most paths contribute almost nothing.

SMC addresses this by maintaining a population of partial paths. At each repair frame, it:

1. proposes new partial paths;
2. scores their compatibility;
3. preserves or replicates promising paths;
4. discards unpromising paths;
5. continues from the adapted particle population.

The goal is not to change the declared terminal repaired law. The goal is to allocate finite computation more intelligently while estimating that law.

### 11.2 Feynman-Kac sequence

Write the full repair potential as a product of incremental potentials:

\[
G_j^q(x_{1:B})
=
\prod_{r=1}^{B}g_{r,j}^q(x_{1:r}).
\]

Define a sequence of partial-path targets

\[
\pi_{r,j}^q(dx_{1:r})
\propto
\left[
\prod_{u=1}^{r}
Q_\theta(dx_u\mid H_{u-1},\Sigma_{u-1})
\right]
\left[
\prod_{u=1}^{r}g_{u,j}^q(x_{1:u})
\right].
\]

At \(r=B\), the target equals the desired repaired prefix distribution. The future is then sampled freely from the learned transition model.

### 11.3 Bootstrap proposal

For particle \(n\) at step \(r-1\), propose

\[
x_r^{(n)}
\sim
Q_\theta(\cdot\mid H_{r-1}^{(n)},\Sigma_{r-1}).
\]

Because the proposal is the learned natural transition itself, the incremental weight contains only the repair-potential increment. The implementation never evaluates the flow log density.

### 11.4 Branching

The progressive method begins each repair step with \(N=128\) retained particles. Every parent proposes \(b=2\) children, producing

\[
N_{\mathrm{cand}}=Nb=256
\]

candidate partial paths.

Branching gives each parent more than one stochastic opportunity to enter a compatible region. After weighting and tempering, the candidate cloud is systematically pruned back to 128 retained particles.

### 11.5 Progressive source bridge

A terminal clamp evaluates the complete 4-frame source average only at the end of repair. That can cause abrupt weight collapse. Progressive SMC introduces the source information across the four frames.

After observing \(r\) of the \(W=4\) source frames, let

\[
S_r=\sum_{u=1}^{r}X_{u,j}.
\]

The unseen source frames are temporarily filled using a persistence prediction based on the current source value:

\[
\widehat A_{j,r}
=
\frac{S_r+(W-r)X_{r,j}}{W}.
\]

The provisional clamp energy is

\[
E_{j,r}^q
=
-\frac12
\left(
\frac{\widehat A_{j,r}-a_j^q}{\epsilon_j}
\right)^2.
\]

The bridge exponent is capped at

\[
\beta_r=\frac{r}{W}.
\]

Thus the source constraint is introduced at 25%, 50%, 75%, and 100% strength. At the final repair frame, \(\widehat A_{j,W}=A_j\) and \(\beta_W=1\), so the final source potential is exactly the original Gaussian clamp.

The persistence fill is an intermediate computational device. It changes how particles are selected during the bridge, not the terminal repair potential.

### 11.6 Adaptive ESS tempering

Even a planned increase in \(\beta\) can be too selective. Suppose the current exponent is \(\beta\) and the next cap is \(\beta_{\max}\). For a candidate increment \(\Delta\beta\), update log weights as

\[
\log \widetilde w_n
=
\log w_n+\Delta\beta\,E_n.
\]

The algorithm chooses the largest \(\Delta\beta\) such that normalized candidate weights retain at least the target ESS:

\[
\operatorname{ESS}(\Delta\beta)
\ge
0.65N_{\mathrm{cand}}.
\]

A binary search solves this one-dimensional problem. If no meaningful increment is possible, the candidate population is systematically resampled and tempering continues from equal weights. The production configuration allows up to eight tempering resamples per repair step.

This is why the method is called a **progressive ESS bridge**:

- progressive: the terminal clamp is introduced in stages;
- ESS-adaptive: step sizes are chosen from the current particle overlap;
- bridge: intermediate distributions connect the natural proposal to the repaired target.

### 11.7 Systematic resampling

Given normalized weights \(\bar w_1,\ldots,\bar w_N\), systematic resampling draws one random offset \(u\sim\operatorname{Uniform}(0,1/N)\) and uses points

\[
u,\;u+1/N,\;\ldots,\;u+(N-1)/N
\]

on the cumulative weight distribution.

High-weight paths receive multiple descendants. Low-weight paths may receive none. Compared with independent multinomial resampling, systematic resampling usually has lower resampling variance.

Resampling converts unequal weights into an approximately equally weighted empirical sample. It does not create new information. If every high-weight particle descends from one original root, resampling can produce many copies of the same genealogy.

### 11.8 Genealogical diversity

The algorithm tracks the original repair root of every retained particle. The number of distinct ancestors is a direct measure of genealogical collapse.

This diagnostic answers a different question from ESS:

- ESS asks how concentrated the **current weights** are.
- ancestor count asks how many independent **historical lineages** remain.

Weights can be equal immediately after resampling even if every particle is a clone of one ancestor. Therefore, ESS alone is insufficient for SMC validation.

### 11.9 Free future descendants

After the terminal repaired population is selected, every one of the 128 retained prefixes launches two free future descendants. This gives 256 future paths per low/high-source system.

Averaging multiple descendants from each repaired prefix reduces future-transition noise. It is analogous to partial Rao-Blackwellization, although the future is still sampled rather than analytically integrated.

This detail matters when interpreting the fixed-generator benchmark: the progressive bundle changed both the repair bridge and the number of future descendants. The benchmark establishes that the bundle works, but it does not attribute the full improvement to bridge tempering alone.

### 11.10 Progressive pseudocode

~~~text
INPUT:
    fitted generator Q_theta
    factual boundary history and fixed stimulus schedule
    source low/high targets and IQRs from training worms
    factual anchor projection
    N = 128 retained particles
    branch factor b = 2
    future branch factor f = 2

FOR every source j and q in {low, high}:
    initialize N identical factual boundary histories
    initialize equal weights and distinct root labels
    beta = 0

    FOR repair frame r = 1,...,4:
        branch each retained particle into b proposal children
        sample each child from Q_theta
        apply the incremental factual-anchor energy

        update the observed partial source statistic
        fill unseen source frames by persistence
        compute provisional source-clamp energy
        set the planned beta cap to r / 4

        WHILE beta is below the planned cap:
            choose the largest beta increment retaining at least 65% candidate ESS
            apply that tempered source-energy increment
            if the next increment is too selective:
                systematically resample candidates
                reset weights to equal values
            continue until the planned beta cap is reached

        record ESS, max weight, normalizer increment, beta, resampling,
        forced tempering, and root ancestry
        systematically prune 256 candidates to 128 retained particles

    at frame 4, verify that beta = 1 and the full terminal clamp is reached
    record achieved source level and terminal diagnostics
    branch every repaired prefix into f free future descendants
    roll all future descendants under the unchanged Q_theta
    average future features

subtract low from high and divide by max(achieved source gap, 0.10)
return matrices and complete stepwise diagnostics
~~~

### 11.11 Strengths

- Allocates computation toward compatible prefixes before the cut.
- Prevents a single abrupt terminal clamp from causing avoidable ESS collapse.
- Uses branching to explore more local transition outcomes.
- Uses adaptive tempering rather than a fixed bridge schedule alone.
- Preserves the same terminal source potential.
- Requires samples and repair-potential evaluations, not flow densities.
- Provides detailed stepwise ESS, weight, tempering, and ancestry diagnostics.

### 11.12 Weaknesses

- Much more computationally expensive.
- Resampling introduces genealogical dependence.
- A persistence look-ahead can be poor for oscillatory or rapidly changing sources.
- If the generator assigns negligible mass to the compatible region, SMC cannot manufacture that missing support.
- Results can still vary across fitted generator seeds, worms, or episode definitions.
- The production bundle combines bridge improvement with future-descendant averaging.

---

## 12. Direct importance versus progressive SMC

| Property | Wide-flow direct importance | Progressive bridge SMC |
| --- | --- | --- |
| Natural generator | historical wide regularized flow | historical external table used an earlier flow ensemble |
| Proposal | complete natural path | learned transition at every repair step |
| Source constraint | evaluated after complete path bank exists | introduced progressively across repair frames |
| Anchor | evaluated on complete generated prefix | applied incrementally |
| Retained repair particles | 128 paths in the primary run | 128 survivors |
| Candidate branching | none | 2 children per survivor, 256 candidates |
| Future descendants | one future already attached to each natural path | two per repaired survivor, 256 future paths |
| Tempering | none | ESS-adaptive up to full clamp |
| Resampling | none | systematic resampling and pruning |
| Main failure | weight degeneracy | genealogy collapse or bridge/model mismatch |
| Speed | fast | slow |
| Flow density required | no | no |
| Best historical role | strongest complete Randi/Cook pipeline | strongest fixed-law finite-particle estimator |

The ideal infinite-particle target can be the same repaired law. The finite-sample approximations differ because they allocate samples differently.

---

## 13. Fixed-generator estimator benchmark

The sampler benchmark froze one flow checkpoint and one fold. It compared each finite-particle estimator with the mean of two independent 4,096-particle direct-importance references.

| Estimator | Response MSE | Squared bias | Monte Carlo variance | Valid rate | Achieved/target contrast | Distinct roots | Runtime per four-worm fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct importance | 0.030689 | 0.013059 | 0.017630 | 0.440 | 0.473 | n/a | 0.5 min |
| Terminal-clamp SMC | 0.015645 | 0.008098 | 0.007547 | 0.435 | 0.472 | 55.4 | 21.2 min |
| Progressive bridge SMC | **0.011008** | **0.006580** | **0.004428** | **0.911** | **0.553** | **62.0** | 34.3 min |

Relative to terminal-clamp SMC, progressive SMC reduced:

- squared bias by 18.7%;
- Monte Carlo variance by 41.3%;
- total response MSE by 29.6%.

Relative to direct importance weighting, progressive SMC reduced response MSE by 64.1%.

The progressive matrix had signed Spearman correlation 0.854 with the finite reference, versus 0.758 for direct importance. Progressive SMC was best at every tested response horizon from 0.25 to 10 seconds.

At the repair steps, progressive SMC retained approximately 112.5/128 particle-equivalent ESS, while terminal SMC fell to roughly 46/128 when its full source clamp arrived at the final frame.

### 13.1 Benchmark qualifications

- The reference is high-particle, not exact. The two independent references had response MSE 0.0170 between them and signed Spearman correlation 0.707.
- Only one frozen generator checkpoint and fold were used.
- Three Monte Carlo repeats support a descriptive variance comparison, not asymptotic confidence intervals.
- The benchmark tests sampler approximation under a learned model. It does not validate biological edges.
- The progressive bundle includes both a progressive repair bridge and two future descendants.

---

## 14. Historical external-reference performance

### 14.1 Edge-presence metrics

The post-freeze external analysis used absolute off-diagonal response magnitudes as edge scores. The complete results for the two methods were:

| Method | Randi WT AUROC / AUPRC | Cook structural | Cook chemical | Cook gap |
| --- | ---: | ---: | ---: | ---: |
| Wide-flow direct importance | **0.619 / 0.298** | **0.591 / 0.349** | 0.579 / 0.314 | **0.653 / 0.118** |
| Progressive bridge SMC | 0.596 / 0.251 | 0.564 / 0.325 | 0.557 / 0.297 | 0.590 / 0.086 |

### 14.2 AUROC definition

For a binary edge reference and continuous score \(s_{k,j}=|M(k,j)|\), AUROC is

\[
\operatorname{AUROC}
=
\Pr(s_{\mathrm{positive}}>s_{\mathrm{negative}}),
\]

with half credit for ties. An AUROC of 0.5 is chance ranking. AUROC asks whether referenced edges tend to receive higher scores than referenced non-edges. It does not measure effect-size calibration.

### 14.3 AUPRC definition

The precision-recall curve varies the score threshold and tracks:

\[
\operatorname{precision}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},
\qquad
\operatorname{recall}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}.
\]

AUPRC summarizes this curve. Unlike AUROC, its baseline depends on edge prevalence. It is especially relevant when positive edges are rare.

### 14.4 Cook is count-valued

The bundled Cook reference stores nonnegative integer chemical-synapse and gap-junction counts. Structural count was defined as

\[
A_{\mathrm{struct}}=A_{\mathrm{chem}}+A_{\mathrm{gap}}.
\]

AUROC and AUPRC binarize these counts as \(A>0\). That tests edge presence, not whether larger model responses correspond to larger anatomical counts.

Count-sensitive metrics gave:

| Method | All-pair Spearman | Positive-edge Spearman | Positive-edge Kendall | Within-source rank Spearman | NDCG |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wide-flow direct importance | **0.143** | **0.066** | **0.046** | **0.111** | 0.719 |
| Progressive bridge SMC | 0.098 | 0.030 | 0.023 | 0.086 | 0.709 |

All-pair correlation partly repeats edge-presence information because most Cook pairs are zero. Positive-edge correlation is the cleaner strength-conditional question. Even there, anatomical counts and learned response magnitudes are not assumed to share a calibrated physical scale.

### 14.5 The two matrices are related but not interchangeable

Comparing the progressive and wide-direct shortest-horizon matrices gave:

| Relationship | Value |
| --- | ---: |
| Signed Spearman correlation | 0.576 |
| Absolute-value Spearman correlation | 0.418 |
| Sign agreement | 0.732 |
| Top-10% edge Jaccard overlap | 0.333 |

Part of this disagreement is generator variation, not just sampler variation, because the historical progressive ensemble and historical wide-direct ensemble used different fitted model families/checkpoints.

### 14.6 What external agreement does and does not show

Randi and Cook were not used to train or select the wide flow. Their post-freeze agreement is therefore useful convergent validity: strong learned response scores overlap independent reference structure more than chance.

It does not imply:

- identified causal effects;
- direct synapses;
- calibrated anatomical strength;
- transmitter-specific action;
- receptor-specific gain control;
- correct physical delay.

An observational activity dependency can align with anatomy for many reasons, including direct coupling, indirect propagation, common drive, circuit state, and calcium filtering.

---

## 15. The critical distinction between response horizon and source lag

This is the most important interpretive distinction in the project.

### 15.1 Response horizon

The response horizon \(h\) asks:

> How far into the future do we summarize the target response after the cut?

For example,

\[
F_4^k
=
\frac14(X_{c+1,k}+X_{c+2,k}+X_{c+3,k}+X_{c+4,k}).
\]

At 4 Hz, \(h=4\) is a one-second cumulative future response.

### 15.2 Source lag

The source lag \(\tau\) asks:

> Which historical source window, at what offset before the cut, is being selected?

If the source window has width \(W\), its end is shifted \(\tau\) frames before the cut.

### 15.3 Timeline

~~~text
source lag tau = 0

factual boundary       generated repair/source window        free future
------------------ | [ c-3 ][ c-2 ][ c-1 ][  c  ] | [c+1] [c+2] ...
                                                    ^
                                                    cut

source lag tau > 0

factual boundary   [source window] [bridge-to-cut frames] | [free future]
                                                          ^
                                                          cut
~~~

### 15.4 What the successful historical matrices varied

The strongest historical wide-direct and progressive results generally used:

- a 4-frame source window ending at the cut, so \(\tau=0\);
- multiple future horizons \(h\in\{1,2,4,8,16,24,32,40\}\).

Therefore, their curves across “lags” were mainly **response-horizon curves**. The source history contrast was broad and immediately pre-cut.

The artifact label “lag-1 matrix” refers to the one-frame future horizon in this response convention. It should not be read as a coefficient obtained by changing only \(X_{t-1,j}\).

### 15.5 What a genuine lag-indexed tensor requires

A genuine lag-resolved analysis must estimate

\[
M_{h,\tau}(k,j)
=
\frac{
\mathbb E_{Q_{j,\tau}^{+}}F_h^k
-
\mathbb E_{Q_{j,\tau}^{-}}F_h^k
}{
\max(g_{j,\tau},0.10)
}.
\]

Now \(h\) and \(\tau\) are separate axes:

- \(h\): how long after the cut the target is summarized;
- \(\tau\): how far before the cut the selected source window lies.

### 15.6 Why historical lag localization remains unvalidated

The explicit temporal-cut experiment varied source offset, but its selected zero-offset cell had only 35.6% compatibility-valid source clamps and did not improve prediction on untouched worms. Its external Randi/Cook values were descriptive, not confirmation of lag resolution.

Calcium dynamics create an additional identification problem. Slow fluorescence filtering can spread a fast latent interaction across many adjacent frames. Strong self-history and broad cross-neuron lag effects can therefore be properties of the observation process rather than precise neural transmission delays.

The accurate conclusion is:

> The repaired-response framework extracts a useful broad effective-response matrix, but it has not yet reliably decomposed that response into distinct historical source lags.

---

## 16. Diagnostics required for a trustworthy response

### 16.1 Generator diagnostics

Before response estimation:

- held-out whole-worm energy score;
- stimulus-balanced energy score;
- variogram score;
- calibration and sharpness;
- multistep rollout stability;
- performance by quiet, onset, active, offset, and recovery phase;
- variation across generator seeds and folds;
- explicit effective receptive field.

Good one-step prediction is necessary but not sufficient. A response query can concentrate on a low-mass part of history space where generator error is larger than its global average.

### 16.2 Compatibility diagnostics

For each source and low/high query:

- source-only normalizer;
- total source-plus-anchor normalizer;
- repair normalizer conditional on source compatibility;
- achieved source distribution;
- achieved high-minus-low gap;
- anchor-cost distribution;
- fraction passing the declared validity gate.

Compatibility estimates how much model path mass supports the query. It does not estimate biological truth.

### 16.3 Direct importance diagnostics

- low and high ESS;
- maximum normalized weight;
- weight entropy;
- Monte Carlo convergence from \(N=64\) to \(N=128\) to \(N=256\);
- sensitivity to clamp bandwidth and anchor strength;
- independent path-bank repeat agreement.

### 16.4 SMC diagnostics

- stepwise candidate ESS;
- particle-equivalent ESS after correcting for branch factor;
- maximum candidate and equivalent weight;
- number of tempering resamples;
- any forced tempering;
- distinct root ancestors;
- normalizer increments;
- independent SMC repeat agreement;
- direct-versus-SMC agreement where both estimate the same target.

### 16.5 Biological reproducibility diagnostics

- sign agreement across held-out worms;
- rank agreement across folds;
- agreement across generator seeds;
- agreement across samplers;
- matched quiet-versus-onset comparison;
- source-label shuffle;
- within-worm block-preserving time shift;
- source-lag shuffle;
- quiet pseudo-cuts;
- prospective replication on new worms.

Anatomical or receptor references should be examined only after the model and response statistic are frozen.

---

## 17. Failure modes and what they mean

| Observation | Likely interpretation | Appropriate response |
| --- | --- | --- |
| Very low importance ESS | requested repair has poor overlap with natural generator paths | broaden clamp, increase particles, use SMC, or abstain |
| High ESS but tiny achieved gap | low and high repaired laws did not separate | do not normalize into a strong coefficient; mark invalid |
| Good sampler repeat agreement but poor generator-seed agreement | model uncertainty dominates Monte Carlo uncertainty | improve generator or report ensemble uncertainty |
| Good onset matrix but equally strong quiet matrix | generic state dynamics rather than stimulus-specific recruitment | use onset-minus-matched-quiet estimand |
| Stable mean effect but unstable lag of maximum | calcium smoothing or correlated histories make lag allocation non-identifiable | use smooth distributed-lag kernels or report broad time bands |
| Stable variance/tail effect but weak mean effect | possible gain or reliability modulation | promote distributional feature, not mean coefficient |
| Strong Cook/Randi AUROC but weak held-out prediction | possible atlas-aligned artifact or overfitting | predictive validity must precede external correspondence |
| Strong held-out prediction but chance anatomy | model may capture activity dynamics not encoded by anatomy, or query may be semantically mismatched | do not tune directly to anatomy; investigate internal dynamics |
| SMC equal weights but one root ancestor | genealogical collapse hidden by resampling | increase branching/diversity or abstain |
| Effect only at lags beyond encoder receptive field | structural artifact or indexing error | treat as negative control; fix architecture/indexing |

---

## 18. What the procedure corrects—and what it does not

### 18.1 Problems it addresses

The compatibility-aware framework addresses:

- arbitrary hard replacement of a source coordinate;
- mismatch between requested and achieved source displacement;
- incompatibility of the remaining population history;
- importance-weight concentration;
- finite-particle degeneracy;
- Monte Carlo noise;
- finite SMC genealogical collapse, when diagnostics detect it.

### 18.2 Problems it does not solve automatically

It does not identify or eliminate:

- unmeasured behavioral state;
- unobserved neurons or common causes;
- feedback from unrecorded variables;
- observational-versus-interventional differences;
- calcium convolution and neuron-specific observation kinetics;
- model misspecification;
- finite-memory insufficiency;
- chemical identity when the conditioning input is only binary;
- direct versus indirect propagation;
- anatomical versus functional coupling.

For these reasons, the correct name is **model-relative repaired-response coefficient**, not causal effect or synaptic weight.

---

## 19. Distributional extensions beyond the mean

The repaired law is a distribution, so reducing it to a cumulative mean discards much of the value of the generative model.

For every source-target-lag-horizon query, one can estimate several complementary effects.

### 19.1 Mean shift

\[
\Delta\mu_{h,k,j}
=
\mathbb E_{Q_j^+}[F_h^k]
-
\mathbb E_{Q_j^-}[F_h^k].
\]

This is the historical primary response.

### 19.2 Scale or gain shift

Let \(\sigma_{h,k,j}^q\) be the standard deviation of a future feature under repaired law \(q\). A symmetric scale statistic is

\[
\Delta\log\sigma_{h,k,j}
=
\log(\sigma_{h,k,j}^++\delta)
-
\log(\sigma_{h,k,j}^-+\delta),
\]

where \(\delta>0\) prevents logarithms of zero.

A positive value means the high source state is associated with increased target variability even if its mean changes little.

### 19.3 Tail-probability shift

For threshold \(u_k\), define

\[
\Delta p_{h,k,j}^{\mathrm{tail}}
=
\Pr_{Q_j^+}(F_h^k>u_k)
-
\Pr_{Q_j^-}(F_h^k>u_k).
\]

This detects a change in the chance of unusually large responses.

### 19.4 Omnibus distributional distance

Let \(Y_1^+,\ldots,Y_m^+\) and \(Y_1^-,\ldots,Y_n^-\) be scalar or vector future features from the two repaired laws. The energy distance is

\[
\mathcal E(Q^+,Q^-)
=
2\mathbb E\|Y^+-Y^-\|
-
\mathbb E\|Y^+-\widetilde Y^+\|
-
\mathbb E\|Y^--\widetilde Y^-\|.
\]

It is zero only when the two distributions agree under standard conditions. Wasserstein distance is another possible omnibus statistic.

These statistics are particularly relevant for neuromodulation. A receptor pattern may be related to gain, variability, reliability, or tail behavior without producing a consistent mean shift.

### 19.5 Paired common-random-number refinement

For a one-step distributional contrast, factual and repaired histories can be passed through the same conditional flow using identical base-noise draws. If

\[
Y^+=T_\theta(Z;H^+),
\qquad
Y^-=T_\theta(Z;H^-)
\]

share the same \(Z\), then the paired difference \(Y^+-Y^-\) often has much lower Monte Carlo variance than two independent samples.

This is a useful next estimator, but it should not be conflated with the historical direct importance method. The historical method pairs low and high through a shared **natural path bank and different weights**. A common-random-number history-contrast method pairs low and high through matched **flow noise under two explicit histories**.

An on-manifold history perturbation still requires a justified conditional history resampler or masked-history model. A next-state conditional flow does not automatically provide the conditional distribution of one historical coordinate given all the others.

---

## 20. Improving genuine multi-lag dynamics

Particle count alone cannot solve lag non-identifiability. A better lag model should change the statistical representation of history.

### 20.1 Smooth distributed-lag kernels

Rather than estimate an unrelated matrix at every frame, write

\[
K_{k,j}(\tau)
=
\sum_{b=1}^{B_\tau}
\gamma_{k,j,b}\phi_b(\tau),
\]

where \(\phi_b\) are smooth temporal basis functions and \(B_\tau\) is much smaller than the number of frames.

This shares information across neighboring lags. It is statistically more efficient and reflects the fact that calcium-filtered responses should vary smoothly in time.

Useful regularizers include:

- ridge shrinkage of all basis coefficients;
- group lasso over the complete lag kernel for each edge;
- fused penalties between adjacent lag coefficients;
- sparse-plus-low-rank structure across source-target pairs;
- hierarchical shrinkage across worms;
- sign or smoothness stability across nearby lags.

### 20.2 Latent neural state plus calcium observation model

Observed fluorescence can be modeled as a filtered version of a faster latent state:

\[
Z_{t+1}\sim p_\theta(Z_{t+1}\mid Z_{t-L:t},S_{t-L:t}),
\]

\[
X_{t,i}\sim p_\psi(X_{t,i}\mid Z_{1:t,i},\kappa_i),
\]

where \(\kappa_i\) represents neuron-specific calcium kinetics.

Lag effects are then defined on \(Z_t\), while the observation model explains temporal smoothing in \(X_t\). This is a more defensible route to physical timing than reading delay directly from broad fluorescence kernels.

### 20.3 Hierarchical replication

With more worms, treat each worm-specific kernel as a deviation from a population kernel:

\[
K_{k,j}^{(w)}(\tau)
=
K_{k,j}^{\mathrm{pop}}(\tau)
+
U_{k,j}^{(w)}(\tau).
\]

Shrink \(U^{(w)}\) toward zero while allowing genuine biological heterogeneity. Cross-validation must continue to hold out complete worms.

### 20.4 Role of the repaired sampler after model improvement

The sampling idea remains useful:

- use direct paired sampling or direct importance for local, one-step edge discovery;
- use progressive SMC when a repair query is selective but supported;
- use temporal-cut SMC only after a local effect passes internal stability gates;
- use multi-step SMC to study propagation of a validated local dependency, not to manufacture the initial edge.

---

## 21. Recommended success criteria

A response should pass several layers before biological promotion.

### Layer A: predictive law

- lower held-out proper score than meaningful baselines;
- stable multistep rollout;
- calibrated uncertainty;
- no dependence on external atlases for selection.

### Layer B: sampler accuracy

- direct estimates converge as particle count increases;
- progressive SMC has adequate stepwise ESS and ancestry;
- independent sampler repeats agree;
- direct and SMC agree when targeting the same repaired law;
- source separation is actually achieved.

### Layer C: lag information

- exact source-lag indexing is declared;
- effects exceed lag-shuffle and time-shift nulls;
- neighboring lags show a reproducible smooth profile;
- lag maxima or centroids replicate across worms, folds, and seeds;
- no claimed effect lies outside the encoder's receptive field.

### Layer D: stimulus specificity

- onset effects exceed matched quiet pseudo-onsets;
- chemical identity is encoded correctly;
- results replicate within each chemical or in a prespecified hierarchical model;
- presentation slot is never substituted for chemical identity.

### Layer E: biological interpretation

- anatomy, perturbation, and receptor maps are examined post-freeze;
- binary edge presence and continuous strength are analyzed separately;
- mean, gain, tail, and omnibus distributional effects are reported separately;
- claims remain observational unless validated by a real intervention.

---

## 22. Implementation details that prevent silent errors

### 22.1 Fold-local quantities

For each outer fold, estimate only from training worms:

- neural mean and scale;
- source low/high quantiles and IQR;
- PCA anchor projection and whitening scales;
- future event thresholds;
- any conditional history-resampling model.

Using held-out worms in these quantities leaks information even if model weights are unchanged.

### 22.2 Stimulus alignment

- Store an explicit schedule per worm.
- Preserve native event duration.
- If resampling time, resample traces and stimulus schedule before window extraction.
- Store stimulus schema version and fingerprint in every checkpoint.
- Reject a checkpoint when its expected stimulus channels or schema fingerprint do not match the cohort.

### 22.3 Numerical weight stability

Compute in log space:

\[
\ell_n=\log G(\omega^{(n)}).
\]

Let \(m=\max_n\ell_n\). Then

\[
\bar w_n
=
\frac{\exp(\ell_n-m)}
{\sum_r\exp(\ell_r-m)}.
\]

This log-sum-exp calculation prevents numerical underflow when potentials are very small.

### 22.4 Common random numbers

When comparing low and high systems, align random-number streams whenever the particle layout permits it. Record when exact pairing is lost because resampling or chunk boundaries diverge.

### 22.5 Matrix normalization

Archive both:

- raw high-minus-low response \(\Delta_{h,k,j}\);
- normalized coefficient \(M_h(k,j)\);
- target source gap;
- achieved source gap;
- source validity mask.

Never archive only a normalized matrix. A large coefficient caused by a tiny denominator must remain detectable.

### 22.6 Required array dimensions

A robust response archive should retain explicit axes such as:

~~~text
raw_response:
    [worm, phase, event, source_lag, horizon, target, source]

achieved_gap:
    [worm, phase, event, source_lag, source]

validity:
    [worm, phase, event, source_lag, source]

step_ess:
    [worm, phase, event, source_lag, low_high, source, repair_step]
~~~

Do not encode lag or phase only in filenames. Store numeric frame offsets and seconds inside the artifact.

### 22.7 Deterministic provenance

Every run should save:

- model checkpoint checksum;
- data input checksum;
- code revision or source checksum;
- fold assignment;
- generator seed;
- Monte Carlo seed;
- stimulus schema fingerprint;
- complete response configuration;
- matrix orientation;
- completion and finite-value validation;
- artifact checksum ledger.

---

## 23. Suggested archive layout

~~~text
experiment_name/
├── README.md
├── REPORT.md
├── protocol.json
├── validation.json
├── fold_assignments.csv
├── input_artifact_checksums.csv
├── checksums.sha256
├── responses/
│   ├── fold_0_seed_1701.npz
│   ├── fold_0_seed_2903.npz
│   └── ...
├── diagnostics/
│   ├── compatibility.csv
│   ├── smc_steps.csv
│   ├── convergence.csv
│   └── null_controls.csv
├── analysis/
│   ├── matrices.npz
│   ├── internal_reliability.csv
│   ├── external_postfreeze.csv
│   └── uncertainty.csv
└── figures/
    ├── response_matrices.png
    ├── lag_horizon_heatmaps.png
    ├── ess_and_ancestry.png
    └── worm_seed_stability.png
~~~

The primary REPORT.md should lead with the internal validity result. External reference scores should appear in a clearly labeled post-freeze section.

---

## 24. Reproducible project artifacts

The main local sources for this guide are:

- [Compatibility-aware method source](SOURCE.md)
- [Preserved technical manuscript](main.tex)
- [Atlas-blind conditional-density tournament](../../results/conditional_distribution_benchmark/atlas_blind_world_model_20260827/REPORT.md)
- [Frozen wide-flow direct response ensemble](../../results/compatibility_path_response/winner_wide_direct_20260827/REPORT.md)
- [Progressive SMC fixed-generator benchmark](../../results/compatibility_path_response/frozen_estimator_benchmark_20260826/analysis/REPORT.md)
- [Full progressive ensemble and Cook analysis](../../results/compatibility_path_response/full_progressive_analysis_20260827/REPORT.md)
- [Compatibility-path experiment index](../../results/compatibility_path_response/EXPERIMENT_INDEX.md)
- [Stimulus tensor provenance audit](../../results/stimulus_provenance_audit_20260828/tensor_inspection.json)

Relevant implementation files are:

- **conditional_neural_benchmark/models.py**: causal TCN encoders and encoded conditional-model wrapper;
- **history_tangent_benchmark/src/history_tangent_benchmark/models.py**: conditional flow-matching head and Heun sampler;
- **compatibility_neural_benchmark/core.py**: repaired-response configuration, direct path generation, importance weights, anchors, features, and diagnostics;
- **compatibility_neural_benchmark/progressive_smc.py**: progressive bridge, ESS tempering, branching, resampling, and future descendants;
- **compatibility_neural_benchmark/full_progressive_analysis.py**: aligned post-freeze external and count-sensitive analyses.

---

## 25. Glossary

**Achieved contrast:** The actual difference between the mean source statistic in the high- and low-repaired particle populations.

**Anchor:** A soft penalty that keeps non-source population activity close to a real factual episode.

**Ancestral rollout:** Recursive generation in which each sampled state is appended to history before the next state is sampled.

**AUPRC:** Area under the precision-recall curve; a ranking metric sensitive to positive-class prevalence.

**AUROC:** Probability that a randomly selected positive edge receives a higher score than a randomly selected negative edge.

**Compatibility:** The amount of natural path mass receiving appreciable repair potential for a query.

**Conditional distribution:** A probability law for an output given observed context, rather than a single predicted value.

**Convergent validity:** Agreement with an independent reference that supports usefulness but does not prove identity of scientific meaning.

**Effective sample size:** A weight-concentration diagnostic, \(1/\sum_n\bar w_n^2\).

**Flow matching:** Training a velocity field that transports a base distribution to the data distribution.

**Feynman-Kac path target:** A sequential probability law formed by multiplying a natural path law by incremental nonnegative potentials.

**Heun integration:** A second-order predictor-corrector method for numerically integrating an ordinary differential equation.

**Importance weighting:** Estimating expectations under one distribution using weighted samples from another distribution.

**Model-relative response:** A response defined under the learned generator and declared repair semantics, not directly under an unknown biological law.

**Normalizer:** The expected unnormalized potential that converts weighted path mass into a probability distribution.

**Particle:** One simulated partial or complete path in an SMC algorithm.

**Proper score:** A predictive scoring rule whose expected optimum is the true data-generating distribution.

**Repaired path:** A model-generated path selected by soft source and factual-compatibility potentials.

**Response horizon:** The amount of future time summarized after the cut.

**Sequential Monte Carlo:** A particle method that alternates propagation, weighting, and resampling through a sequence of intermediate targets.

**Source lag:** The offset between the selected historical source window and the future cut.

**Systematic resampling:** A low-variance resampling scheme based on evenly spaced points on the cumulative weight distribution.

**Temporal cut:** The boundary after which repair weights are removed and all future particles follow the unchanged learned transition law.

---

## 26. Bottom line

The two methods solve different parts of the same problem.

The wide conditional flow learns a useful stochastic approximation to observed neural dynamics. Direct importance weighting then gives a fast, transparent way to estimate high-versus-low compatibility-aware responses from a shared bank of natural paths. This complete pipeline produced the strongest historical non-SBTG Randi and Cook correspondence.

Progressive bridge SMC is a more computationally careful estimator of the repaired-path law. It introduces the source condition gradually, branches particles, adapts tempering to ESS, monitors ancestry, and averages multiple future descendants. On a frozen generator, it substantially reduced finite-particle response error.

Neither result establishes exact lag-resolved causal dynamics. The successful historical matrices mostly index future response horizon, not an isolated historical source lag. The next scientifically defensible step is to combine the sampling framework with correctly encoded chemical schedules, explicit source-lag indexing, smooth distributed-lag structure, distributional response features, and eventually a latent calcium-aware state-space model.

The strongest defensible claim today is:

> A learned conditional flow plus compatibility-aware repaired-path sampling extracts reproducible broad effective-response structure from whole-brain calcium dynamics. Direct weighting is the strongest historical complete pipeline, progressive bridge SMC is the strongest tested fixed-law sampler, and precise biological lag localization remains an open validation problem.

---

## 27. Selected background references

- Del Moral, P. (2004). *Feynman-Kac Formulae: Genealogical and Interacting Particle Systems with Applications*. Springer.
- Doucet, A., and Johansen, A. M. (2009). A tutorial on particle filtering and smoothing: fifteen years later. In *The Oxford Handbook of Nonlinear Filtering*.
- Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. (2023). Flow matching for generative modeling. *International Conference on Learning Representations*.
