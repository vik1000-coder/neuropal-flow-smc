# Tier 2 preregistration: predictive closure and observation stress

Status: frozen before the definitive Tier 1 confirmatory run completed.

## Gate

Tier 2 runs only if Tier 1 Gate 4 is met by at least one distributional route: the
upper 95% DGP-seed bootstrap interval for variance- or third-cumulant-lag NRMSE is
below one, the signed lag profile is correct, and the mean-blind channel remains
null.  Passing via a correctly specified structural model opens the robustness
study but does not establish a generic SID advantage.

## Seeds and inference

- New top-level DGP seeds: 3001--3030.
- The DGP seed is the primary inferential unit.
- Any conditional-query draws are averaged within DGP seed.
- Confidence intervals use 2,000 DGP-cluster bootstrap replicates.
- No Tier 2 outcome is used to alter a method, basis, threshold, or primary grid.

## D3 hidden stochastic modulator

Let a scalar latent state be

\[
Z_t = \sum_{\ell=1}^{12} k_\ell X_{t-\ell}+\xi_t,
\qquad k_\ell \propto 0.85^{\ell-1},\quad \xi_t\sim N(0,0.45^2),
\]

with \(\pi_t=\operatorname{logit}^{-1}(\alpha+Z_t)\) and

\[
Y_{t+1}=2\{S_{t+1}-\pi_t\}+0.55\epsilon_{t+1},
\qquad S_{t+1}\mid Z_t\sim\operatorname{Bernoulli}(\pi_t).
\]

The latent-state centering makes the conditional mean exactly zero under every
observable history, while the conditional variance changes with the source
history.  For truncated contexts, the oracle integrates over missing source lags
and latent noise.

Primary context ladder:

- short: 2 observed source lags;
- oracle-sufficient: 12 observed lags;
- overcomplete: 20 lags, of which the last eight are registered nulls.

Primary sample size is 8,000.  At sufficient context only, add 2,000 and 32,000.
Primary channels are the mean null and variance lag profile.

Methods:

- zero effect;
- direct raw-moment regression;
- Gaussian mean/variance likelihood control;
- ratio critic with explicitly labelled conditional evaluation queries;
- frozen anchored score model with sample-based Hodge centering and explicitly
  labelled conditional evaluation queries.

## D5 observation process

Start from the mean-blind stochastic-gain process.  Independently draw the current
and previous latent responses conditional on their overlapping source histories,
then observe

\[
Y_t^{\mathrm{obs}}=g_0Y_t^\star+g_1Y_{t-1}^\star+\eta_t.
\]

Registered regimes are:

- clean: \((g_0,g_1,\sigma_\eta)=(1,0,0)\);
- primary filter: \((1,0.35,0.25)\);
- strong filter: \((1,0.65,0.25)\);
- observation-only null: the strong filter and identical marginal noise, but no
  history-dependent gain.

Primary channels are the mean null and observed variance lag profile.  The same
five methods and access labels as D3 are used.  A method fails the observation
stress if it calls a nonzero lag effect in the observation-only null or attributes
the filter-induced lag spread to an unfiltered biological kernel.

## Static scale and information slices

Use M3 bounded skew, M4 quartic, and M5 occupancy as representative recovered,
higher-order-failed, and information-boundary mechanisms.

- Sample-size slice: \(N\in\{2{,}000,8{,}000,32{,}000\}\) at target
  \(\Lambda=3\).
- Information slice: target \(\Lambda\in\{0.3,3,30\}\) at \(N=8{,}000\).
- Do not run their Cartesian product.
- If the positivity amplitude bound prevents a requested \(\Lambda\), report the
  achieved value and label the cell boundary-limited.

The dimension ladder is not treated as informative in the current scalar-projection
implementation because every reusable estimator is handed the known affected
coordinate.  A dense rotated effect, multiple sources/targets, and unknown subspace
are reserved for D4/Tier 3 and cannot be claimed from independent nuisance padding.

## Decisions

Recovery and advantage use the original runbook criteria.  Comparisons with stronger
information access are reported as best-case controls, not observational
superiority.  A Tier 1 score failure followed by a Tier 2 direct-method success is
evidence for targeted predictive closure, not for score-derived SID.
