# E12 result: direct versus composed horizons

**Decision:** completed, mixed confirmation.  Six of seven preregistered gates
passed.  The experiment is no longer an unrun theory proposal, but its
short-history AR(2) detection-power claim did not meet the frozen threshold.

**Canonical run:** `runs/e12_closure_confirmation_20260716/`  
**Protocol:** `E12_CLOSURE_PREREGISTRATION_20260716.md`  
**Inference:** 30 independent DGP seeds (6001--6030), 40 trajectories per
seed, three trajectory folds, 99 full trajectory-bootstrap refits.

## What was estimated

For horizon `k` and endpoint witness `phi`, the direct side was the ordinary
outer Riesz-orthogonal estimate of

`E[D m_dir,k,phi(H)]`.

The composed side was estimated separately by recursively rolling out the
cross-fitted one-step conditional Gaussian law and differentiating the whole
rollout path.  Its uncertainty refit the one-step model after resampling whole
trajectories.  The observed horizon outcome was not used to correct this side.

The registered witnesses were the endpoint mean, `sin(x)`, and `sin(2x)` at
horizons 2, 4, and 8.  The primary decision used the mean and `sin(x)`.

## Gate decision

| Gate | Frozen threshold | Result | Decision |
|---|---:|---:|---|
| Complete and finite | 1,350 rows; 30 seeds | 1,350; 30 | pass |
| Closure-null false-positive rate | <= 0.10 | 0.0148 | pass |
| AR(2), history too short, horizon-8 power | >= 0.80 | 0.70 | **fail** |
| Nonlinear one-step misspecification, horizon-8 power | >= 0.80 | 1.00 | pass |
| AR(2) mean defect grows from horizon 2 to 8 | horizon 8 > horizon 2 | 0.191 > 0.120 | pass |
| Invalid outer-score audit | within frozen tolerance | 0.0359 <= 0.0427 | pass |
| Direct-target coverage in analytic AR cells | 0.85--1.00 | 0.994 | pass |

The AR(2) short-history point estimates were accurate even though the test was
underpowered: at horizon 8 the mean-witness defect was 0.210 versus oracle
0.209, and the `sin(x)` defect was 0.146 versus oracle 0.146.  Both witnesses
rejected in 21 of 30 seeds.  This is an uncertainty/power limitation, not a
failure to recover the average defect.

The nonlinear misspecification alternative was decisive.  At horizon 8, the
mean defect was -0.649 versus oracle -0.624 and the `sin(x)` defect was -0.225
versus oracle -0.226; both had power 1.00.

## Closure-null calibration

The aggregate primary-witness false-positive rate was 0.0148.  By cell it was
0 for the closed AR(1), 0.0167 for the correctly declared AR(2), and 0.0278 for
the correctly specified nonlinear Markov system.  The bootstrap is therefore
conservative at this sample size.

## The score-construction audit

The deliberately labelled `naive_outer` diagnostic inserted the composed map
into an outer score with the observed horizon residual.  As the manuscript
predicts, this quantity stayed close to the direct estimate under the
alternatives (median absolute difference 0.0359 within the frozen 0.0427
tolerance).  It was not used to estimate the composed side.  This empirical
audit is important: using that correction would conceal rather than estimate
the closure defect.

## Interpretation and claim ceiling

The experiment supports three statements at the registered resolution:

1. Direct and composed derivatives are well calibrated in correctly declared
   linear and nonlinear Markov systems.
2. A history window shorter than the true memory produces a growing resolved
   closure defect with accurate average point estimates, but the present
   per-seed bootstrap test reaches only 0.70 power at horizon 8.
3. One-step misspecification can produce a large defect even when the declared
   history is a true predictive state.

This is evidence about compositional adequacy for the tested histories,
horizons, witnesses, and model classes.  It is not proof of exact Markov
closure.  A nonzero defect remains compatible with insufficient memory,
one-step misspecification, direct-model error, nonstationarity, weak support,
or inadequate feature resolution.

## Reproduce

```bash
distributional_sid/.venv/bin/python -m distributional_sid.closure_experiment \
  --run-dir distributional_sid/runs/e12_closure_confirmation_20260716 \
  --stage confirmation --seed-start 6001 --seed-end 6030 \
  --n-trajectories 40 --trajectory-length 220 --anchors 16 --folds 3 \
  --rollouts 32 --bootstrap-replicates 99
```

The runner is resumable and refuses to reuse a run directory with a different
frozen configuration.
