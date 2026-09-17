r"""LEVER #5 -- HELD-OUT PREDICTIVE LIKELIHOOD (predlik).

Replace connectome-AUROC with a *within-model* readout: does modeling the CONDITIONAL
VARIANCE (the SID "gain" channel) buy held-out one-step predictive likelihood, and does it
buy MORE for receptor-expressing neurons?

Design (faithful to the spec)
-----------------------------
For a config in {6w_clean_deconv, 28w_deconv}, a WORM-LEVEL held-out cross-validation:

  * K-fold over worms (K = min(NL_FOLDS, n_worms); K == n_worms => leave-one-worm-out).
  * For each held-out fold, each lag L, each target neuron j: fit a closed-form
    QuadraticScoreMatcher on the TRAIN worms with feature vector
        Psi(t) = [1, centered population sources at t]      (the history H_t)
    predicting  Y_j(t+L)  (one-step-at-lag-L target).  Both the conditional MEAN mu(t) and
    the conditional VARIANCE v(t) are linear-in-Psi natural parameters -- the variance's
    dependence on the sources IS the gain channel.
  * On each held-out worm compute (mu, var) via predict_params, then two mean Gaussian NLLs:
        FULL    NLL = mean NLL with (mu, var)     -- heteroscedastic, gain channel ON
        REDUCED NLL = mean NLL with (mu, v0)      -- homoscedastic, gain channel OFF,
                                                     mean model held FIXED, v0 = mean train var
    delta_NLL[worm, L, j] = REDUCED - FULL   (>0  =>  the gain channel helps predict j).

Two questions from the estimand:
  (1) Does the gain channel help AT ALL?  -> overall mean delta_NLL over targets, with a
      worm_bootstrap CI (resampling held-out worms).
  (2) Does it help MORE for receptor neurons? -> group_perm_test(delta_NLL, receptor-
      expressing, variance-matched controls) for neuropeptide & monoamine, per lag and
      aggregated, with a worm_bootstrap CI on the group difference.

Numerical guards (all-N linear predictors extrapolate hard out-of-worm on deconv data):
identical clips are applied to FULL and REDUCED (they share mu, so clips never bias the
FULL-vs-REDUCED contrast) -- mu clipped to a wide train-based band, var clipped to a band
around the train marginal variance; v0 is the mean of the clipped train variance.

Heavy work is under __main__.  All counts come from env vars.  Results -> predlik.json.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np

from sid_elegans.newlevers import common as cm
from sid_elegans.newlevers import harness as H
from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher

# ---------------------------------------------------------------------------- config knobs
SMOKE = os.environ.get("NEWLEVERS_SMOKE", "0") == "1"
SEED = cm.env_int("NL_SEED", 0)
RIDGE = cm.env_float("NL_RIDGE", 2.0)          # ridge on theta (mean+variance) linear solve
NFOLDS = cm.env_int("NL_FOLDS", 6)             # worm K-fold; K==n_worms => leave-one-worm-out
NBOOT = cm.env_int("NL_NBOOT", 200)            # worm bootstrap resamples (over held-out worms)
NPERM = cm.env_int("NL_NPERM", 5000)           # label permutations for the group test
THROTTLE = cm.env_float("NL_THROTTLE", 0.3)    # sleep between heavy worm-set (fold) refits
_CFG_ENV = os.environ.get("NL_CONFIG", "").strip()

# numerical guards against out-of-worm linear extrapolation (applied to FULL and REDUCED alike)
CLIP_SD = 8.0            # clip held-out mu to train mean +/- CLIP_SD * train SD
VAR_FLOOR_FRAC = 0.05    # var floor  = VAR_FLOOR_FRAC * train marginal target variance
VAR_CEIL_FRAC = 20.0     # var ceil   = VAR_CEIL_FRAC  * train marginal target variance
MIN_TRAIN = 200          # min finite train rows to attempt a target's fit
MIN_TEST = 10            # min finite held-out rows to score a (worm, target)

MODULATORS = ("neuropeptide", "monoamine")
LAGS_FULL = list(H.LAGS)                        # [1,2,3,5,8,10,15,20,30,40]
LAGS_SMOKE = [2, 5, 8]

# smoke shrinks everything so a run finishes in < ~90s
if SMOKE:
    NBOOT = min(NBOOT, 5)
    NPERM = min(NPERM, 5)
    THROTTLE = 0.0
    LAGS = LAGS_SMOKE
    CONFIGS = ["6w_clean_deconv"]
else:
    LAGS = LAGS_FULL
    CONFIGS = [_CFG_ENV] if _CFG_ENV else ["6w_clean_deconv", "28w_deconv"]


# --------------------------------------------------------------------------- small helpers
def _gauss_nll(y, mu, var):
    """Per-sample Gaussian negative log likelihood."""
    return 0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var)


def _fold_indices(W, K, rng):
    """K disjoint worm folds (interleaved after a shuffle); K==W => singletons (LOO)."""
    idx = rng.permutation(W)
    return [np.sort(idx[i::K]) for i in range(K)]


def _dbar(perworm_list):
    """Mean delta_NLL over a list of per-worm [N] vectors (NaN-aware) -> [N]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(np.asarray(perworm_list, float), axis=0)


def _group_diff(perworm_list, exp, ctrl):
    """mean(dbar[expressing]) - mean(dbar[control]) from a (resampled) list of per-worm vecs."""
    dbar = _dbar(perworm_list)
    a = dbar[exp]; b = dbar[ctrl]
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(a.mean() - b.mean())


def _overall(perworm_list):
    """Overall mean delta_NLL across all targets from a (resampled) list of per-worm vecs."""
    dbar = _dbar(perworm_list)
    dbar = dbar[np.isfinite(dbar)]
    return float(dbar.mean()) if len(dbar) else np.nan


# ------------------------------------------------------------------- the held-out CV engine
def compute_perworm_delta(X_list, names, lags, ridge, n_folds, throttle, seed):
    """Worm-level K-fold held-out delta_NLL.

    Returns ``(perworm, K, meta)`` where ``perworm[L]`` is a length-``W`` list of ``[N]``
    vectors: ``perworm[L][w][j] = REDUCED_NLL - FULL_NLL`` for target ``j`` on held-out worm
    ``w`` at lag ``L`` (NaN if unavailable).  Each worm is a held-out unit exactly once.
    """
    W = len(X_list); N = len(names)
    rng = np.random.default_rng(seed)
    K = min(int(n_folds), W)
    folds = _fold_indices(W, K, rng)
    perworm = {L: [np.full(N, np.nan) for _ in range(W)] for L in lags}
    n_fits = 0

    for test_idx in folds:
        test_set = {int(i) for i in test_idx}
        tr = [X_list[i] for i in range(W) if i not in test_set]
        if len(tr) == 0:
            continue
        for L in lags:
            # shared TRAIN design: population sources at t, target matrix at t+L
            Str = np.concatenate([x[:-L, :] for x in tr], axis=0)          # [Ttr, N] sources
            Ytr_all = np.concatenate([x[L:, :] for x in tr], axis=0)       # [Ttr, N] targets
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ctr = np.nanmean(Str, axis=0)                              # train source means
            Psi_tr = np.concatenate(
                [np.ones((Str.shape[0], 1)), np.nan_to_num(Str - ctr, nan=0.0)], axis=1)

            # held-out worm designs (share the train centering)
            tews = []
            for wo in test_idx:
                x = X_list[int(wo)]
                Pte = np.concatenate(
                    [np.ones((x.shape[0] - L, 1)), np.nan_to_num(x[:-L, :] - ctr, nan=0.0)], 1)
                tews.append((int(wo), Pte, x[L:, :]))

            for j in range(N):
                y = Ytr_all[:, j]
                fin = np.isfinite(y)
                if int(fin.sum()) < MIN_TRAIN:
                    continue
                Yj = y[fin]; Pj = Psi_tr[fin]
                yb = float(Yj.mean()); sy = float(Yj.std()) + 1e-9
                vmar = float(Yj.var()) + 1e-12
                lo, hi = yb - CLIP_SD * sy, yb + CLIP_SD * sy
                vlo, vhi = VAR_FLOOR_FRAC * vmar, VAR_CEIL_FRAC * vmar
                try:
                    m = QuadraticScoreMatcher(ridge=ridge, var_min=1e-8, eta2_min=1e-8, seed=seed)
                    m.fit(Yj, Pj)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, _, _, vtr = m.predict_params(Pj)
                    v0 = float(np.clip(vtr, vlo, vhi).mean())              # homoscedastic baseline
                except Exception:
                    continue
                n_fits += 1
                for wo, Pte, Yte_all in tews:
                    yte = Yte_all[:, j]
                    f2 = np.isfinite(yte)
                    if int(f2.sum()) < MIN_TEST:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, _, mute, vte = m.predict_params(Pte[f2])
                    mute = np.clip(mute, lo, hi)                           # same mu for both models
                    vte = np.clip(vte, vlo, vhi)
                    yv = yte[f2]
                    full = float(np.mean(_gauss_nll(yv, mute, vte)))       # gain channel ON
                    red = float(np.mean(_gauss_nll(yv, mute, v0)))        # gain channel OFF
                    perworm[L][wo][j] = red - full
        if throttle:
            time.sleep(throttle)

    meta = dict(n_fits=int(n_fits), K_folds=int(K), fold_sizes=[int(len(f)) for f in folds])
    return perworm, K, meta


# ------------------------------------------------------------------------- per-config driver
def analyse_config(cfg, lags, rng_master):
    out = {"config": cfg}
    try:
        X_list, names, fps = cm.get_data(cfg)
    except Exception as e:  # pragma: no cover - defensive
        out["error"] = f"load failed: {e!r}"
        return out
    W = len(X_list); N = len(names)
    out.update(n_worms=int(W), N=int(N), fps=float(fps), lags=list(lags),
               n_frames_total=int(sum(x.shape[0] for x in X_list)))

    # heavy step: held-out delta_NLL
    perworm, K, cvmeta = compute_perworm_delta(
        X_list, names, lags, RIDGE, NFOLDS, THROTTLE, SEED)
    out["cv"] = cvmeta

    # aggregate-over-lags per-worm vectors (each worm: mean delta over lags)
    agg_perworm = []
    for w in range(W):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agg_perworm.append(np.nanmean(np.asarray([perworm[L][w] for L in lags], float), axis=0))

    def _boot_overall(pwl, rng):
        return H.worm_bootstrap(_overall, pwl, n_boot=NBOOT, rng=rng, throttle=0.0)

    # ---- Q1: does the gain channel help at all (overall mean delta_NLL, worm-boot CI) ----
    overall = {"per_lag": {}}
    for L in lags:
        pwl = perworm[L]
        dbar = _dbar(pwl)
        overall["per_lag"][str(L)] = dict(
            mean_delta=float(np.nanmean(dbar)),
            median_delta=float(np.nanmedian(dbar)),
            frac_targets_pos=float(np.nanmean((dbar > 0).astype(float))),
            boot=_boot_overall(pwl, np.random.default_rng(SEED + L)))
    dbar_agg = _dbar(agg_perworm)
    overall["aggregate"] = dict(
        mean_delta=float(np.nanmean(dbar_agg)),
        median_delta=float(np.nanmedian(dbar_agg)),
        frac_targets_pos=float(np.nanmean((dbar_agg > 0).astype(float))),
        boot=_boot_overall(agg_perworm, np.random.default_rng(SEED + 999)))
    out["overall_gain"] = overall

    # ---- Q2: does it localize to receptor-expressing neurons? ----
    V = cm.source_variance(X_list)
    logV = np.log(V + 1e-12)                     # log-variance matching feature
    localization = {}
    for mod in MODULATORS:
        try:
            exp = cm.receptor_expressing(names, mod)
        except Exception as e:
            localization[mod] = {"error": f"receptor set failed: {e!r}"}
            continue
        # genuinely-matched, EQUAL-SIZE control set on log-variance (subsamples the larger
        # group) -- fixes the old variance_matched_controls fallback that took the whole pool.
        exp_mask, ctrl_mask, minfo = cm.matched_groups(
            exp, logV, np.random.default_rng(SEED + 7))
        rec = {"n_expressing": int(exp.sum()), "n_control": int(ctrl_mask.sum()),
               "n_pairs": int(minfo["n_pairs"]),
               "resid_imbalance_std": minfo["resid_imbalance_std"],
               "match_pool_n": int(minfo["pool_n"]), "match_exp_n": int(minfo["exp_n"]),
               "per_lag": {}}

        def _boot_diff(pwl, rng, exp=exp_mask, ctrl=ctrl_mask):
            return H.worm_bootstrap(lambda L_: _group_diff(L_, exp, ctrl),
                                    pwl, n_boot=NBOOT, rng=rng, throttle=0.0)

        for L in lags:
            pwl = perworm[L]
            dbar = _dbar(pwl)
            gp = H.group_perm_test(dbar, exp_mask, ctrl_mask, n_perm=NPERM,
                                   rng=np.random.default_rng(SEED + L))
            rec["per_lag"][str(L)] = dict(
                group_perm=gp,
                boot=_boot_diff(pwl, np.random.default_rng(SEED + 1000 + L)))
        gp = H.group_perm_test(dbar_agg, exp_mask, ctrl_mask, n_perm=NPERM,
                               rng=np.random.default_rng(SEED + 31))
        rec["aggregate"] = dict(
            group_perm=gp,
            boot=_boot_diff(agg_perworm, np.random.default_rng(SEED + 2000)))
        localization[mod] = rec
    out["localization"] = localization
    return out


# --------------------------------------------------------------------------------- __main__
def _fmt_ci(b):
    return f"[{b.get('lo', float('nan')):+.4f}, {b.get('hi', float('nan')):+.4f}]"


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t_start = time.time()
    rng_master = np.random.default_rng(SEED)

    results = {
        "lever": "predlik",
        "estimand": ("held-out one-step predictive likelihood: does the conditional-variance "
                     "(gain) channel help, and more for receptor-expressing neurons?"),
        "params": dict(smoke=SMOKE, seed=SEED, ridge=RIDGE, n_folds=NFOLDS, n_boot=NBOOT,
                       n_perm=NPERM, throttle=THROTTLE, lags=list(LAGS), configs=list(CONFIGS),
                       modulators=list(MODULATORS), clip_sd=CLIP_SD,
                       var_floor_frac=VAR_FLOOR_FRAC, var_ceil_frac=VAR_CEIL_FRAC),
        "configs": {},
    }

    for cfg in CONFIGS:
        print(f"[predlik] config={cfg} ...", flush=True)
        try:
            results["configs"][cfg] = analyse_config(cfg, LAGS, rng_master)
        except Exception as e:  # emit partial results even on failure
            import traceback
            results["configs"][cfg] = {"config": cfg, "error": repr(e),
                                       "traceback": traceback.format_exc()}
            print(f"[predlik] config={cfg} FAILED: {e!r}", flush=True)

    results["runtime_sec"] = round(time.time() - t_start, 1)
    path = cm.save_json("predlik", results)

    # ------------------------------------------------------------------ human summary
    print("\n" + "=" * 78)
    print(f"PREDLIK  (smoke={SMOKE})  ridge={RIDGE}  lags={LAGS}  nboot={NBOOT} nperm={NPERM}")
    print("=" * 78)
    verdict_q1, verdict_q2 = [], []
    for cfg, c in results["configs"].items():
        if "error" in c:
            print(f"\n[{cfg}] ERROR: {c['error']}")
            continue
        agg = c["overall_gain"]["aggregate"]
        b = agg["boot"]
        gain_sig = np.isfinite(b.get("lo", np.nan)) and b["lo"] > 0
        print(f"\n[{cfg}] worms={c['n_worms']} N={c['N']} K={c['cv']['K_folds']} "
              f"fits={c['cv']['n_fits']}")
        print(f"  Q1 gain helps? aggregate mean deltaNLL={agg['mean_delta']:+.4f} "
              f"(median={agg['median_delta']:+.4f}, {100*agg['frac_targets_pos']:.0f}% targets>0) "
              f"boot95={_fmt_ci(b)} {'**' if gain_sig else ''}")
        if gain_sig:
            verdict_q1.append(cfg)
        for mod, rec in c.get("localization", {}).items():
            if "error" in rec:
                print(f"  Q2 {mod}: {rec['error']}")
                continue
            gp = rec["aggregate"]["group_perm"]; bd = rec["aggregate"]["boot"]
            # PRIMARY verdict: permutation p_greater < 0.05. The 6-worm bootstrap CI is
            # reported as descriptive support only and does NOT drive the verdict.
            loc_sig = np.isfinite(gp.get("p_greater", np.nan)) and gp["p_greater"] < 0.05
            boot_supports = np.isfinite(bd.get("lo", np.nan)) and bd["lo"] > 0
            print(f"  Q2 {mod}: expr(n={rec['n_expressing']})-ctrl(n={rec['n_control']}) "
                  f"pairs={rec['n_pairs']} diff={gp['diff']:+.4f} p_greater={gp['p_greater']:.3f} "
                  f"[PRIMARY] boot95={_fmt_ci(bd)}{' boot+' if boot_supports else ''} "
                  f"{'**' if loc_sig else ''}")
            if loc_sig:
                verdict_q2.append(f"{cfg}:{mod}")

    print("\nVERDICT")
    print(f"  Q1 gain channel improves held-out predictive likelihood (boot lo>0) in: "
          f"{verdict_q1 or 'none'}")
    print(f"  Q2 predictive gain localizes to receptor neurons (perm p_greater<.05, PRIMARY) in: "
          f"{verdict_q2 or 'none'}")
    print(f"\nwrote {path}  ({results['runtime_sec']}s)")
