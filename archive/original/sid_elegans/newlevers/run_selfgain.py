r"""Lever #3 -- SELF-GAIN / EXCITABILITY  (the zeroed self-edge, now exposed).

The distributional estimator zeroes every matrix diagonal, but the *self* term it discards
is itself a signal: for target neuron j,

    self_gain[j] = d logVar[Y_j] / d x_j

is a neuron modulating its OWN future variance -- an intrinsic self-gain / excitability knob.
Both estimators now expose it before zeroing:
  * estimator.fit_distributional_connectome(...).diagonal['gain']   (complete-case, 6w x 80)
  * acmma.fit_acmma_connectome(...)[1]['self_diag']['gain']         (available-case, 28w x 84)

Modulator ground truths have a ZERO diagonal (no self-edges), so this is a NODE-level test on
receptor-expressing MEMBERSHIP, not an edge AUROC. We ask whether a neuron being on the receptor
(target) side of a neuromodulator layer predicts its self-gain, and -- the mechanistic version --
whether its self-gain is more STATE-DEPENDENT (changes more with brain state).

TEST 1 (static):  |self_gain| for receptor-expressing vs an EQUAL-SIZE, log-variance-matched control
                  set (cm.matched_groups; genuine 1:1 nearest-neighbour matching, reported n_pairs +
                  resid_imbalance_std), per lag (group permutation test). We also report
                  corr(|self_gain|, log-variance) so the gap is demonstrably not a variance artifact.
                  Report the fast-vs-slow-band lag profile. Metabotropic
                  neuromodulator receptors act over seconds -> the excitability signature should be
                  stronger in the SLOW band.
TEST 2 (state):   phase_masks(X,'pc1') -> hi/lo brain state; self_gain per neuron in each phase;
                  delta = |self_gain_hi - self_gain_lo|. Prediction: receptor-expressing neurons'
                  self-gain changes MORE with state than variance-matched controls (group perm test).
Inference:        group_perm_test (>= NL_NPERM) for every lag/modulator/engine; worm_bootstrap CI on
                  the primary group-difference; estimator(6w) vs acmma(28w) self-gain consistency.

All heavy counts come from env vars (see below); heavy work is under __main__ only; results ->
common.save_json('selfgain', ...) in sid_elegans/output/newlevers/.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np

# cheap module imports only at import time (no data load / no fitting here)
from sid_elegans.newlevers import common as cm
from sid_elegans.newlevers import harness as H
from sid_elegans.biolag import config as C

# --------------------------------------------------------------------------------------
# knobs (all heavy counts from the environment, with the mandated defaults)
# --------------------------------------------------------------------------------------
SMOKE = os.environ.get("NEWLEVERS_SMOKE", "0") == "1"
SEED = cm.env_int("NL_SEED", 0)

NSURR = cm.env_int("NL_NSURR", 200)      # (kept for parity; not used by this node-level lever)
NBOOT = cm.env_int("NL_NBOOT", 200)
NPERM = cm.env_int("NL_NPERM", 5000)
THROTTLE = cm.env_float("NL_THROTTLE", 0.3)

# estimator (complete-case) config + acmma (available-case) config
EST_CONFIG = os.environ.get("NL_CONFIG", "6w_clean_deconv")
ACMMA_CONFIG = os.environ.get("NL_ACMMA_CONFIG", "28w_deconv")
MIN_TRIPLE = cm.env_int("NL_MIN_TRIPLE", 2000)   # acmma hard-mask threshold (harness convention)

# lag grid + the single pre-specified lag used for the bootstrap CI (slow band, ~5 s)
LAGS = list(C.LAG_FRAMES)
PRIMARY_LAG = cm.env_int("NL_PRIMARY_LAG", 20)
# modulators: the self-gain vector is shared, only the expressing/control masks change per modulator
MODULATORS = ["neuropeptide", "monoamine", "serotonin", "tyramine"]
PRIMARY_MOD = "neuropeptide"

if SMOKE:
    # TINY: one data config, 2 lags (one fast one slow), tiny resampling, exercise BOTH engines
    LAGS = [3, 20]
    MODULATORS = ["neuropeptide"]
    NBOOT = min(NBOOT, 5)
    NPERM = min(NPERM, 5)
    NSURR = min(NSURR, 5)
    THROTTLE = 0.0
    MIN_TRIPLE = 2          # 6w-sized acmma needs a low triple floor
    ACMMA_CONFIG = EST_CONFIG   # run acmma on the SAME single config (ONE data config in smoke)

if PRIMARY_LAG not in LAGS:
    PRIMARY_LAG = LAGS[-1]


# --------------------------------------------------------------------------------------
# self-gain readouts (length-N vectors; exact 0.0 == "target not fit" -> treated as NaN)
# --------------------------------------------------------------------------------------
def self_gain_estimator(X_list, names, lag):
    from sid_elegans.estimator import fit_distributional_connectome
    res = fit_distributional_connectome(X_list, names, lag=lag)
    return np.asarray(res.diagonal["gain"], float)


def self_gain_acmma(X_list, names, lag, min_triple=MIN_TRIPLE):
    from sid_elegans.acmma import fit_acmma_connectome
    _, diag = fit_acmma_connectome(X_list, names, lag=lag, ridge=0.01, min_triple=min_triple)
    return np.asarray(diag["self_diag"]["gain"], float)


def abs_selfgain(sg):
    """|self-gain| with unfit targets (exact 0.0) mapped to NaN so they drop from group tests."""
    sg = np.asarray(sg, float)
    v = np.abs(sg)
    v[sg == 0.0] = np.nan
    return v


def variance_corr(sg_by_lag, logV):
    """corr(|self-gain|, log-variance) across neurons, per lag (Spearman). Makes the
    'the expressing-vs-control gap is NOT a marginal-variance artifact' point explicit: if
    |self-gain| tracked variance, this correlation would be large."""
    from scipy.stats import spearmanr
    logV = np.asarray(logV, float)
    out = {}
    for L, sg in sg_by_lag.items():
        v = abs_selfgain(sg)
        good = np.isfinite(v) & np.isfinite(logV)
        if good.sum() > 3:
            rho = float(spearmanr(v[good], logV[good]).statistic)
        else:
            rho = float("nan")
        out[L] = dict(spearman_absselfgain_logvar=rho, n=int(good.sum()))
    return out


# --------------------------------------------------------------------------------------
# band bookkeeping (fast vs slow, from the biolag config -- the honest 4 Hz resolution)
# --------------------------------------------------------------------------------------
def lag_band(lag_frames):
    s = lag_frames / C.FPS
    for name, (lo, hi) in C.bands().items():
        if lo <= s <= hi:
            return name
    return "gap"   # intentional 1.0-2.5 s transition gap


def band_summary(per_lag, lags):
    """Aggregate a {lag: group_perm_result} dict into fast/slow band means + a slow>fast verdict."""
    res = {}
    for band in ("fast", "slow"):
        Ls = [L for L in lags if lag_band(L) == band]
        diffs = [per_lag[L]["diff"] for L in Ls if np.isfinite(per_lag[L]["diff"])]
        pg = [per_lag[L]["p_greater"] for L in Ls if np.isfinite(per_lag[L]["p_greater"])]
        res[band] = dict(
            lags=Ls, n=len(diffs),
            mean_diff=float(np.mean(diffs)) if diffs else float("nan"),
            mean_p_greater=float(np.mean(pg)) if pg else float("nan"),
            frac_sig=float(np.mean([p < 0.05 for p in pg])) if pg else float("nan"),
        )
    fd, sd = res["fast"]["mean_diff"], res["slow"]["mean_diff"]
    ok = np.isfinite(fd) and np.isfinite(sd)
    res["slow_minus_fast_diff"] = float(sd - fd) if ok else float("nan")
    res["slow_gt_fast"] = bool(ok and sd > fd)
    return res


# --------------------------------------------------------------------------------------
# per-config expressing / variance-matched-control masks (computed once per config)
# --------------------------------------------------------------------------------------
def build_masks(names, X_list, rng):
    """Expressing vs *genuinely matched* control masks, per modulator.

    FIX (selfgain reviewer): the old ``variance_matched_controls`` silently took the whole
    non-expressing pool when the expressing set was larger than that pool (no matching, unequal
    sizes). We now use ``cm.matched_groups`` on log-variance, which subsamples the LARGER group
    to build EQUAL-SIZE, nearest-matched (expressing, control) subsets. ``info[mod]`` carries
    ``n_pairs`` and per-feature ``resid_imbalance_std`` (pool-sd units) so the 'not a variance
    artifact' claim is auditable. ``logV`` is returned for the corr(|self-gain|, log-var) check.
    """
    V = cm.source_variance(X_list)
    logV = np.log(np.asarray(V, float) + 1e-12)
    masks, info = {}, {}
    for mod in MODULATORS:
        expr = cm.receptor_expressing(names, mod)
        exp_mask, ctrl_mask, mg = cm.matched_groups(expr, logV, rng)
        masks[mod] = (exp_mask, ctrl_mask)
        info[mod] = mg
    return masks, info, logV


# --------------------------------------------------------------------------------------
# TEST 1 -- static: |self-gain| expressing vs control, per lag
# --------------------------------------------------------------------------------------
def run_static(X_list, names, self_gain_fn, masks, rng):
    out = {mod: {"per_lag": {}} for mod in masks}
    sg_by_lag = {}
    for L in LAGS:
        try:
            sg = self_gain_fn(X_list, names, L)
        except Exception as e:               # degenerate-fit guard: emit NaN, keep going
            sg = np.full(len(names), np.nan)
            out.setdefault("_errors", []).append(f"static lag {L}: {e}")
        sg_by_lag[L] = sg
        vals = abs_selfgain(sg)
        for mod, (expr, ctrl) in masks.items():
            out[mod]["per_lag"][L] = H.group_perm_test(vals, expr, ctrl, n_perm=NPERM, rng=rng)
        if THROTTLE:
            time.sleep(THROTTLE)
    for mod in masks:
        out[mod]["bands"] = band_summary(out[mod]["per_lag"], LAGS)
    return out, sg_by_lag


# --------------------------------------------------------------------------------------
# TEST 2 -- state-dependence: delta = |self_gain_hi - self_gain_lo| across pc1 brain state
# --------------------------------------------------------------------------------------
def run_state(X_list, names, self_gain_fn, masks, rng):
    hi_list, lo_list = cm.phase_masks(X_list, "pc1")
    Xhi = cm.subset_frames(X_list, hi_list)
    Xlo = cm.subset_frames(X_list, lo_list)
    out = {mod: {"per_lag": {}} for mod in masks}
    out["_phase_worms_kept"] = dict(hi=len(Xhi), lo=len(Xlo), total=len(X_list))
    usable = len(Xhi) >= 2 and len(Xlo) >= 2
    for L in LAGS:
        if not usable:
            delta = np.full(len(names), np.nan)
        else:
            try:
                sg_hi = self_gain_fn(Xhi, names, L)
                sg_lo = self_gain_fn(Xlo, names, L)
                both = (sg_hi != 0.0) & (sg_lo != 0.0)     # require the target fit in BOTH phases
                delta = np.where(both, np.abs(sg_hi - sg_lo), np.nan)
            except Exception as e:
                delta = np.full(len(names), np.nan)
                out.setdefault("_errors", []).append(f"state lag {L}: {e}")
        for mod, (expr, ctrl) in masks.items():
            out[mod]["per_lag"][L] = H.group_perm_test(delta, expr, ctrl, n_perm=NPERM, rng=rng)
        if THROTTLE:
            time.sleep(THROTTLE)
    for mod in masks:
        out[mod]["bands"] = band_summary(out[mod]["per_lag"], LAGS)
    return out


# --------------------------------------------------------------------------------------
# bootstrap statistics (resample WORMS) on the primary-modulator group difference
# --------------------------------------------------------------------------------------
def stat_static_diff(Xsub, names, lag, expr, ctrl, self_gain_fn):
    vals = abs_selfgain(self_gain_fn(Xsub, names, lag))
    a, b = vals[expr], vals[ctrl]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(a.mean() - b.mean())


def stat_state_diff(Xsub, names, lag, expr, ctrl, self_gain_fn):
    hi, lo = cm.phase_masks(Xsub, "pc1")
    Xhi, Xlo = cm.subset_frames(Xsub, hi), cm.subset_frames(Xsub, lo)
    if len(Xhi) < 2 or len(Xlo) < 2:
        return np.nan
    sh, sl = self_gain_fn(Xhi, names, lag), self_gain_fn(Xlo, names, lag)
    both = (sh != 0.0) & (sl != 0.0)
    delta = np.where(both, np.abs(sh - sl), np.nan)
    a, b = delta[expr], delta[ctrl]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(a.mean() - b.mean())


# --------------------------------------------------------------------------------------
# estimator(6w) vs acmma(28w) self-gain consistency on the shared neuron classes
# --------------------------------------------------------------------------------------
def engine_consistency(est_names, acmma_names, sg_est, sg_acmma, static_est, static_acmma):
    from scipy.stats import spearmanr
    common = [nm for nm in est_names if nm in set(acmma_names)]
    ie = [est_names.index(nm) for nm in common]
    ia = [acmma_names.index(nm) for nm in common]
    per_lag = {}
    rhos = []
    for L in LAGS:
        e = np.asarray(sg_est[L], float)[ie]
        a = np.asarray(sg_acmma[L], float)[ia]
        valid = (e != 0.0) & (a != 0.0) & np.isfinite(e) & np.isfinite(a)
        if valid.sum() > 3:
            rho = float(spearmanr(np.abs(e[valid]), np.abs(a[valid])).statistic)
        else:
            rho = float("nan")
        per_lag[L] = dict(n_common=int(valid.sum()), spearman_abs_selfgain=rho)
        if np.isfinite(rho):
            rhos.append(rho)
    # sign agreement of the static group difference (per modulator, per lag)
    sign_agree = {}
    for mod in MODULATORS:
        if mod not in static_est or mod not in static_acmma:
            continue
        agree = tot = 0
        for L in LAGS:
            de = static_est[mod]["per_lag"][L]["diff"]
            da = static_acmma[mod]["per_lag"][L]["diff"]
            if np.isfinite(de) and np.isfinite(da):
                tot += 1
                if np.sign(de) == np.sign(da):
                    agree += 1
        sign_agree[mod] = dict(agree=agree, total=tot,
                               frac=float(agree / tot) if tot else float("nan"))
    return dict(n_common_neurons=len(common),
                median_spearman=float(np.median(rhos)) if rhos else float("nan"),
                per_lag=per_lag, static_diff_sign_agreement=sign_agree)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # eta2 clamp chatter
    t0 = time.time()
    result = {
        "meta": dict(
            lever="selfgain", smoke=SMOKE, seed=SEED, lags=LAGS, primary_lag=PRIMARY_LAG,
            primary_modulator=PRIMARY_MOD, modulators=MODULATORS, bands=C.bands(),
            n_perm=NPERM, n_boot=NBOOT, throttle=THROTTLE, est_config=EST_CONFIG,
            acmma_config=ACMMA_CONFIG, min_triple=MIN_TRIPLE,
            note="NODE-level test on receptor-expressing membership (modulator GTs have zero "
                 "diagonal); self_gain = d logVar[Y_j]/d x_j (intrinsic excitability).",
        ),
        "errors": [],
    }

    # ---- data + masks ------------------------------------------------------------------
    rng_ctrl = np.random.default_rng(SEED)          # control selection (stable, modulator order)
    rng_inf = np.random.default_rng(SEED + 1)       # permutation / bootstrap inference

    X_est, est_names, _ = cm.get_data(EST_CONFIG)
    masks_est, info_est, logV_est = build_masks(est_names, X_est, rng_ctrl)
    result["data"] = {
        EST_CONFIG: dict(n_worms=len(X_est), N=len(est_names),
                         expressing={m: int(masks_est[m][0].sum()) for m in MODULATORS},
                         control={m: int(masks_est[m][1].sum()) for m in MODULATORS},
                         matching={m: dict(n_pairs=info_est[m]["n_pairs"],
                                           resid_imbalance_std=info_est[m]["resid_imbalance_std"],
                                           pool_n=info_est[m]["pool_n"], exp_n=info_est[m]["exp_n"])
                                   for m in MODULATORS})
    }

    same_config = ACMMA_CONFIG == EST_CONFIG
    if same_config:
        X_acmma, acmma_names, masks_acmma = X_est, est_names, masks_est
    else:
        X_acmma, acmma_names, _ = cm.get_data(ACMMA_CONFIG)
        masks_acmma, info_acmma, _ = build_masks(acmma_names, X_acmma, rng_ctrl)
        result["data"][ACMMA_CONFIG] = dict(
            n_worms=len(X_acmma), N=len(acmma_names),
            expressing={m: int(masks_acmma[m][0].sum()) for m in MODULATORS},
            control={m: int(masks_acmma[m][1].sum()) for m in MODULATORS},
            matching={m: dict(n_pairs=info_acmma[m]["n_pairs"],
                              resid_imbalance_std=info_acmma[m]["resid_imbalance_std"],
                              pool_n=info_acmma[m]["pool_n"], exp_n=info_acmma[m]["exp_n"])
                      for m in MODULATORS})

    def acmma_fn(Xl, nm, lag):
        return self_gain_acmma(Xl, nm, lag, min_triple=MIN_TRIPLE)

    # ---- TEST 1 (static) for both engines ---------------------------------------------
    try:
        static_est, sg_est = run_static(X_est, est_names, self_gain_estimator, masks_est, rng_inf)
    except Exception as e:
        static_est, sg_est = {"_fatal": str(e)}, {L: np.full(len(est_names), np.nan) for L in LAGS}
        result["errors"].append(f"static estimator: {e}")
    try:
        static_acmma, sg_acmma = run_static(X_acmma, acmma_names, acmma_fn, masks_acmma, rng_inf)
    except Exception as e:
        static_acmma, sg_acmma = {"_fatal": str(e)}, {L: np.full(len(acmma_names), np.nan) for L in LAGS}
        result["errors"].append(f"static acmma: {e}")
    result["test1_static"] = {"estimator_" + EST_CONFIG: static_est,
                              "acmma_" + ACMMA_CONFIG: static_acmma}

    # corr(|self-gain|, log-variance): explicit 'not a variance artifact' check (6w estimator)
    try:
        result["variance_corr"] = dict(config=EST_CONFIG,
                                       per_lag=variance_corr(sg_est, logV_est))
    except Exception as e:
        result["variance_corr"] = {"error": str(e)}
        result["errors"].append(f"variance_corr: {e}")

    # ---- TEST 2 (state-dependence) for both engines -----------------------------------
    try:
        state_est = run_state(X_est, est_names, self_gain_estimator, masks_est, rng_inf)
    except Exception as e:
        state_est = {"_fatal": str(e)}
        result["errors"].append(f"state estimator: {e}")
    try:
        state_acmma = run_state(X_acmma, acmma_names, acmma_fn, masks_acmma, rng_inf)
    except Exception as e:
        state_acmma = {"_fatal": str(e)}
        result["errors"].append(f"state acmma: {e}")
    result["test2_state"] = {"estimator_" + EST_CONFIG: state_est,
                             "acmma_" + ACMMA_CONFIG: state_acmma}

    # ---- bootstrap CI on the PRIMARY-modulator group difference (6w estimator) ---------
    expr_p, ctrl_p = masks_est[PRIMARY_MOD]
    boot = {}
    try:
        boot["test1_static"] = H.worm_bootstrap(
            lambda Xs: stat_static_diff(Xs, est_names, PRIMARY_LAG, expr_p, ctrl_p,
                                        self_gain_estimator),
            X_est, n_boot=NBOOT, rng=rng_inf, throttle=THROTTLE)
    except Exception as e:
        boot["test1_static"] = {"error": str(e)}
        result["errors"].append(f"boot static: {e}")
    try:
        boot["test2_state"] = H.worm_bootstrap(
            lambda Xs: stat_state_diff(Xs, est_names, PRIMARY_LAG, expr_p, ctrl_p,
                                       self_gain_estimator),
            X_est, n_boot=NBOOT, rng=rng_inf, throttle=THROTTLE)
    except Exception as e:
        boot["test2_state"] = {"error": str(e)}
        result["errors"].append(f"boot state: {e}")
    result["bootstrap_primary"] = dict(config=EST_CONFIG, modulator=PRIMARY_MOD,
                                       lag=PRIMARY_LAG, **boot)

    # ---- estimator vs acmma consistency ------------------------------------------------
    try:
        result["engine_consistency"] = engine_consistency(
            est_names, acmma_names, sg_est, sg_acmma,
            static_est if "_fatal" not in static_est else {},
            static_acmma if "_fatal" not in static_acmma else {})
    except Exception as e:
        result["engine_consistency"] = {"error": str(e)}
        result["errors"].append(f"consistency: {e}")

    # ---- verdict -----------------------------------------------------------------------
    verdict = {}
    try:
        ps = static_est[PRIMARY_MOD]["bands"]
        verdict["static_slow_mean_diff"] = ps["slow"]["mean_diff"]
        verdict["static_slow_mean_p"] = ps["slow"]["mean_p_greater"]
        verdict["static_slow_gt_fast"] = ps["slow_gt_fast"]
        st = state_est[PRIMARY_MOD]["bands"]
        verdict["state_slow_mean_diff"] = st["slow"]["mean_diff"]
        verdict["state_slow_mean_p"] = st["slow"]["mean_p_greater"]
        b1 = boot.get("test1_static", {})
        verdict["boot_static_ci"] = [b1.get("lo"), b1.get("hi")]
        verdict["boot_static_excludes_0"] = bool(
            np.isfinite(b1.get("lo", np.nan)) and np.isfinite(b1.get("hi", np.nan))
            and (b1["lo"] > 0 or b1["hi"] < 0))
        b2 = boot.get("test2_state", {})
        verdict["boot_state_ci"] = [b2.get("lo"), b2.get("hi")]
        verdict["boot_state_excludes_0"] = bool(
            np.isfinite(b2.get("lo", np.nan)) and np.isfinite(b2.get("hi", np.nan))
            and (b2["lo"] > 0 or b2["hi"] < 0))
        verdict["engine_median_spearman"] = result["engine_consistency"].get("median_spearman")
        mi = info_est[PRIMARY_MOD]
        verdict["match_n_pairs"] = mi["n_pairs"]
        verdict["match_resid_imbalance_std"] = mi["resid_imbalance_std"]
        vc = result.get("variance_corr", {}).get("per_lag", {})
        verdict["varcorr_primary_lag"] = (vc.get(PRIMARY_LAG, {})
                                          .get("spearman_absselfgain_logvar"))
    except Exception as e:
        result["errors"].append(f"verdict: {e}")
    result["verdict"] = verdict

    result["runtime_sec"] = round(time.time() - t0, 1)
    path = cm.save_json("selfgain", result)

    # ---- human summary -----------------------------------------------------------------
    print("\n=== SELF-GAIN / EXCITABILITY (lever #3) ===")
    print(f"smoke={SMOKE}  lags={LAGS}  primary_lag={PRIMARY_LAG}  modulators={MODULATORS}")
    print(f"est={EST_CONFIG} (worms={len(X_est)}, N={len(est_names)})  "
          f"acmma={ACMMA_CONFIG} (worms={len(X_acmma)}, N={len(acmma_names)})")
    if "_fatal" not in static_est:
        print(f"\nTEST1 static |self-gain| expressing vs control [{EST_CONFIG}, estimator]:")
        for mod in MODULATORS:
            b = static_est[mod]["bands"]
            print(f"  {mod:12s} fast diff={b['fast']['mean_diff']:+.4f} (p~{b['fast']['mean_p_greater']:.3f})"
                  f"  slow diff={b['slow']['mean_diff']:+.4f} (p~{b['slow']['mean_p_greater']:.3f})"
                  f"  slow>fast={b['slow_gt_fast']}")
    if "_fatal" not in state_est and PRIMARY_MOD in state_est:
        print(f"\nTEST2 state-dependence delta [{EST_CONFIG}, estimator]:")
        for mod in MODULATORS:
            b = state_est[mod]["bands"]
            print(f"  {mod:12s} fast diff={b['fast']['mean_diff']:+.4f} (p~{b['fast']['mean_p_greater']:.3f})"
                  f"  slow diff={b['slow']['mean_diff']:+.4f} (p~{b['slow']['mean_p_greater']:.3f})")
    b1 = boot.get("test1_static", {})
    b2 = boot.get("test2_state", {})
    print(f"\nBOOTSTRAP [{PRIMARY_MOD}, lag {PRIMARY_LAG}, {EST_CONFIG}]  (n_boot={NBOOT}):")
    print(f"  static diff obs={b1.get('obs'):+.4f}  CI=[{b1.get('lo'):+.4f},{b1.get('hi'):+.4f}]"
          if np.isfinite(b1.get("obs", np.nan)) else f"  static: {b1}")
    print(f"  state  diff obs={b2.get('obs'):+.4f}  CI=[{b2.get('lo'):+.4f},{b2.get('hi'):+.4f}]"
          if np.isfinite(b2.get("obs", np.nan)) else f"  state: {b2}")
    mi = info_est.get(PRIMARY_MOD, {})
    print(f"\nMATCHED CONTROL [{PRIMARY_MOD}, {EST_CONFIG}]: n_pairs={mi.get('n_pairs')}  "
          f"exp_n={mi.get('exp_n')} pool_n={mi.get('pool_n')}  "
          f"resid_imbalance_std(log-var)={mi.get('resid_imbalance_std')}")
    vc = result.get("variance_corr", {}).get("per_lag", {})
    if vc:
        vc_str = "  ".join(f"lag{L}: rho={vc[L]['spearman_absselfgain_logvar']:+.3f}(n={vc[L]['n']})"
                           for L in LAGS if L in vc)
        print(f"VARIANCE-ARTIFACT check corr(|self-gain|,log-var): {vc_str}")
    ec = result.get("engine_consistency", {})
    print(f"\nENGINE consistency: n_common={ec.get('n_common_neurons')}  "
          f"median spearman(|self-gain|)={ec.get('median_spearman')}")
    if isinstance(ec.get("static_diff_sign_agreement"), dict):
        for mod, sa in ec["static_diff_sign_agreement"].items():
            print(f"  sign-agree[{mod}] = {sa['agree']}/{sa['total']} ({sa['frac']})")
    print(f"\nerrors: {result['errors'] if result['errors'] else 'none'}")
    print(f"runtime: {result['runtime_sec']}s   ->  {path}")


if __name__ == "__main__":
    main()
