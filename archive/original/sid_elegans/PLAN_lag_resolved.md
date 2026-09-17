# Plan: lag-resolved, relative connectome-correspondence analysis

**Question.** For each method, at each lag ℓ, how strongly does that lag's directed effect
matrix `E(ℓ)` correspond to each reference network — and what is the *shape* of the
correspondence-vs-lag curve, **relative to that method's own other lags**? The payoff is a
**channel × lag × reference interaction**, read relatively (absolute AUROCs are low/noisy
on receptor proxies).

**Central hypothesis.**
- Fast / short-lag **MEAN** effects → wired **structural** connectome (fast synaptic drive).
- Slow / long-lag **DISTRIBUTIONAL** effects → **neuromodulator receptor** networks
  (metabotropic monoamines, neuropeptides).
- **The asymmetry:** every classical/causal method estimates only the conditional **mean**
  — one channel. SID adds **distributional channels (gain, tail)** with no classical
  analog. If the modulatory correspondence lives at long lags in the *distributional*
  channel, that is structure the entire classical toolkit is blind to by construction.

## Design decisions (locked)
- **Headline lag effect = MARGINAL lag-ℓ** (regress `x(t+ℓ)` on `x(t)` alone), so all
  methods incl. SID-gain are directly comparable per lag. Partial/joint forms (VAR,
  PCMCI+, LiNGAM) reported alongside as the "causal" view.
- **PCMCI+ = full N=80, reduced τ** (see compute note); all other methods run the full lag
  grid.
- Lag grid: {1,2,3,5,8,10,15,20,30,40} frames = 0.25–10 s (data ~928 frames/worm).
- References — **fast/structural:** Cook chem, Cook gap. **slow/modulatory:** dopamine,
  serotonin (most metabotropic: ser-1/4/7), octopamine, tyramine (has ionotropic lgc-55),
  neuropeptide. All `[post,pre]`, aligned via `sid_elegans.ground_truth`.
- Data: per-worm **deconvolved**, **scale-preserving** standardized activity; per-worm
  de-mean then NaN→0; **never** form lag pairs across worm boundaries; **never** per-neuron
  z-score.

## Method roster + verified orientation (all `E(ℓ)[j,i]` = effect of source i → target j)

| method | channel | per-lag matrix | raw→[post,pre] (verified w/ synthetic driver) | venv |
|---|---|---|---|---|
| Pearson / cross-corr | mean | `fut.T @ past` | already `[post,pre]`, no transpose | main |
| single-lag ridge | mean | `(XtX+αI)⁻¹XtY` → `.T` | `B[i,j]=[src,tgt]` → transpose | main |
| VAR(p) | mean (partial) | statsmodels `coefs[ℓ-1]` | `[j,i]=[eq,reg]=[post,pre]` direct | main |
| VAR-LiNGAM | mean (partial, causal) | `adjacency_matrices_[ℓ]` | already `[post,pre]` | scratch |
| SINDy (time-delay, deg-1) | mean | coef of source i in target j's map | map to `[post,pre]` per snippet | scratch |
| PCMCI+ (ParCorr) | mean (partial, causal) | `val_matrix[i,j,τ]` | `E(τ)=|val|[:,:,τ].T` | scratch |
| DYNOTEARS-analog (native) | mean (sparse, causal) | lagged sparse `A_ℓ` | `A_ℓ.T` | main |
| **SBTG mu_hat** | mean | precomputed npz + multilag | already `[post,pre]` | main |
| **SBTG volatility test** | **distributional** | `Cov(s1_j², s0_i²)` per lag | already `[post,pre]` | main |
| **SID mean** | mean | `fit_distributional_connectome(horizon=ℓ)` | already `[post,pre]` | main |
| **SID gain / tail** | **distributional** | ∂logVar / ∂tail, per lag | already `[post,pre]` | main |
| **MDN mean / gain** | mean / **distributional** | non-Gaussian, `horizon=ℓ` | already `[post,pre]` | main (cpu) |

All snippets were built and orientation-verified during planning (scratchpad
`effect_matrices_xcorr_var.py`, `var_sindy_prod.py`, `facet_pcmci_dynotears.py`,
`dynotears_analog.py`, `lag_corr_metric.py`), ready to assemble.

## The relative metric (per method × channel × lag × reference)
1. `corr(ℓ) = AUROC(|E(ℓ)|_offdiag, R>0)` (+ AUPRC, rank-corr). Off-diagonal, `[post,pre]`.
2. **Controls at every lag (mandatory):** source-variance, target-variance, and cross-corr
   baselines scored *at that lag*; and **partial out source+target variance** from `E(ℓ)`
   before scoring the residual. A channel "adds value at ℓ" only if its residual beats
   chance at ℓ. (Smoke test showed raw corr is ~entirely node-activity — this is not
   optional.)
3. **Relativize within method-across-lag:** z-score `corr(ℓ)` over lags (and `corr(ℓ) −
   mean_ℓ`), to read "strong relative to this method's other lags."
4. **Shape statistics:** peak lag (argmax); monotone trend `Spearman(lag, corr)`;
   fast-vs-slow contrast (mean corr short vs long lags); lag-specificity (max/mean or
   negentropy of the normalized curve).
5. **Uncertainty:** per-worm curves → mean ± block-bootstrap CI; split-half stability of
   the *curve shape*; per-lag permutation null of the effect matrix.

## What confirms vs refutes
- **Validation (must hold):** all mean-channel methods give a *similar* structural lag
  profile (peak short, decaying); SID-mean tracks VAR/cross-corr/PCMCI+ per lag. Anchors
  everything.
- **Payoff (the test):** distributional channels (SID-gain, MDN-gain, SBTG-volatility)
  show, for metabotropic-serotonin + neuropeptide references, correspondence that is
  **relatively stronger at long lags** and **stronger than the mean channel's** modulatory
  correspondence, **and survives the variance partial-out** — a profile no classical method
  can produce. Bonus in-family test: does our principled SID-gain beat SBTG's heuristic
  volatility channel?
- **Refute:** distributional modulatory curves flat/at-chance, or indistinguishable from
  the mean channel, or die under the variance control → the distributional axis adds
  nothing lag-resolved. (We report this outcome honestly if it happens.)

## Execution phases (compute budget: hours OK)
1. **Effect matrices**: all methods × full lag grid × per worm, saved as a tensor
   `[method, channel, lag, worm, N, N]`. Fast for everything except PCMCI+.
2. **PCMCI+ (the one bottleneck)**: measured N=30/τ8 ≈ 54 s; **N=80/τ8 intractable**.
   Feasible config: **pooled across worms, ParCorr, N=80, τ_max≈10** with a wall-clock cap;
   if it exceeds budget, fall back to N≈40 (connectome/top-variance neurons) for the causal
   baselines only. PCMCI+ contributes a coarser lag axis; all other methods keep the full
   grid. (Honest caveat, not a blocker.)
3. **Curves + relativization + shape stats + per-lag nulls/variance-controls + per-worm
   CIs**, per reference.
4. **Consistency check** (mean-channel methods agree on the structural profile) and the
   **distributional-channel contrast**.
5. **Figures**: corr-vs-lag faceted by reference; mean-channel methods overlaid (agreement);
   distributional channels overlaid (new profile); lag×method heatmap; normalized/relative
   curves; the variance-partialled residual curves.
6. **Synthesis**: does the lag-resolved, variance-controlled, relative picture show
   fast-mean→wires and slow-distributional→modulators — and does SID's principled gain
   channel do this where SBTG's volatility heuristic and all classical methods cannot?

## Honest risks
- Trivial-baseline dominance at every lag → the variance partial-out is load-bearing; if
  nothing survives it, the honest answer is negative.
- Multiple comparisons across 7 references × many methods → pre-register the hypothesis
  (metabotropic-serotonin/peptide, long-lag, distributional) and treat others as
  exploratory.
- Small worm counts (6 at 80 neurons) → curve *shapes* are noisier than point estimates;
  lean on per-worm CIs and split-half shape stability, and cross-check at 20 worms/56 neu.
