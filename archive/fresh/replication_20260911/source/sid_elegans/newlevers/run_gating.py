r"""LEVER: gating  —  GLOBAL BRAIN-STATE AS THE MODULATOR (idea #2).

Reframe: the C. elegans whole-brain global signal g_w(t) is usually treated as a confound to
regress out.  Here we treat it as *the modulator variable itself* and ask whether the SID
gain channel (d logVar[Y_j]/d.) carries a real state-gating signal that lives specifically on
receptor-expressing (peptidergic / aminergic) targets — i.e. does the neuromodulatory gain
signal ride the global-state axis?

Two complementary designs, both written to output/newlevers/gating.json:

  APPROACH A  (condition-as-feature).  For each worm, z-scale g_w = global_signal(X,"pc1")
    to unit sd and APPEND it as an extra source column ->  X_aug [T, N+1],
    names_aug = names+["GLOBAL"].  Fit fit_distributional_connectome on X_aug at each lag.
    The STATE-GAIN of target j is mats['gain'][j, N] = coupling of the GLOBAL source onto
    target j = d logVar[Y_j]/d g_w.  NOTE (estimand honesty): g_w = pc1 lies in the linear
    span of the neuron columns, so this coefficient is a RIDGE-REGULARIZED common-mode gain,
    NOT a clean partial (the design is rank-deficient in that direction; ridge is what makes
    it estimable at all).  We therefore (i) report a ridge-sensitivity check on the sign and
    (ii) do NOT lean on a marginal-loading control alone.
    TEST: do receptor-expressing targets (receptor_expressing, neuropeptide AND monoamine)
    have larger |state-gain| than EQUAL-SIZE genuinely-matched controls (cm.matched_groups on
    log-variance)?  A per-lag table is EXPLORATORY (garden-of-forking-paths); the confirmatory
    worm_bootstrap CI + circshift_p run at a SINGLE pre-registered slow lag (PRESPEC_SLOW_LAG).
    KEY CONTROL (headline): receptor status and global-mode loading are collinear, so we
    regress state-gain on [log-variance, loading] and test the RESIDUAL (expressing vs
    non-expressing) over ALL neurons.  A second control set matched on [log-variance, loading]
    and the per-feature residual imbalance from matched_groups are also reported.

  APPROACH B  (phase-split).  phase_masks(X,"pc1") -> hi/lo; subset_frames; fit the gain
    connectome per phase; for peptidergic/aminergic reference EDGES (build_registry
    adjacency>0) vs variance-matched control EDGES, test whether |gain| DIFFERS across
    phases more on modulator edges than on controls (group_perm_test over edges).

CONVENTIONS honoured: X_list = list of [T,N] globally-standardized float64; matrices are
[post,pre] (NEVER transposed); fps=4.0; heavy counts come from env vars; heavy work only under
__main__; all inference via harness (worm_bootstrap / circshift_p / group_perm_test); all data
/sets via common.  Emits JSON even on partial failure.
"""
from __future__ import annotations

import os
import time

import numpy as np

from sid_elegans.newlevers import common as cm
from sid_elegans.newlevers import harness as H
from sid_elegans.biolag import config as C
from sid_elegans.estimator import fit_distributional_connectome

# --------------------------------------------------------------------------------------
# env-tunable knobs (import-time cheap; no heavy work here)
# --------------------------------------------------------------------------------------
SMOKE = os.environ.get("NEWLEVERS_SMOKE", "0") == "1"
SEED = cm.env_int("NL_SEED", 0)
FPS = C.FPS

# heavy counts (tiny under smoke)
NBOOT = 5 if SMOKE else cm.env_int("NL_NBOOT", 200)
NSURR = 5 if SMOKE else cm.env_int("NL_NSURR", 200)
NPERM = 5 if SMOKE else cm.env_int("NL_NPERM", 5000)
THROTTLE = 0.0 if SMOKE else cm.env_float("NL_THROTTLE", 0.3)

# lag grid (tiny under smoke, but keep >=1 slow-band lag so the slow-band logic exercises)
LAGS = [2, 10, 20] if SMOKE else list(C.LAG_FRAMES)

# PRE-REGISTERED confirmatory slow lag (frames); ~3.75 s at fps=4.  Fixed a priori so the
# confirmatory bootstrap/circshift are NOT a garden-of-forking-paths over the argmax slow lag.
PRESPEC_SLOW_LAG = 15

# primary config + optional secondary "if time" (never under smoke -> ONE config)
PRIMARY_CONFIG = os.environ.get("NL_CONFIG", "6w_clean_deconv")
CONFIGS = [PRIMARY_CONFIG]
if not SMOKE and "28w_deconv" != PRIMARY_CONFIG:
    CONFIGS.append("28w_deconv")

# run bootstrap+circshift for BOTH modulators (else just the pre-registered neuropeptide
# confirmatory, to keep the overnight run modest).  monoamine always gets the full per-lag
# scan + a rep-lag permutation test regardless.
BOTH_MOD = os.environ.get("NL_BOTH_MOD", "0") == "1"

MODULATORS = ["neuropeptide", "monoamine"]
GLOBAL_METHOD = "pc1"     # per spec: use the pc1 global signal as the modulator g_w
FIT_KW = dict(target_mode="next", ridge=1e-2, heldout_frac=0.2, seed=SEED, sigma_frac=0.0)


# --------------------------------------------------------------------------------------
# small helpers  (pure; no global heavy state)
# --------------------------------------------------------------------------------------
def _augment(X_list, method=GLOBAL_METHOD):
    """Append the unit-sd global signal g_w as an extra source column to every worm.

    global_signal returns a centered [T] vector; we scale it to unit sd so it sits on the
    same scale as the globally-standardized neurons.  Degenerate (flat) g_w -> zero column.
    """
    out = []
    for X in X_list:
        g = cm.global_signal(X, method)
        sd = float(g.std())
        gz = g / sd if sd > 1e-9 else np.zeros_like(g)
        Xa = np.concatenate([np.asarray(X, float), gz[:, None]], axis=1)
        out.append(Xa)
    return out


def _state_gain_vector(X_list, names, lag, ridge=None):
    """|state-gain| length-N vector: |mats['gain'][j, N]| for targets j=0..N-1.

    N = len(names) (the ORIGINAL neuron count); the augmented design has N+1 columns and the
    GLOBAL source is column N.  [j, N] is off-diagonal for every real target so it is never
    zeroed by the estimator's self-edge zeroing.  ``ridge`` overrides FIT_KW['ridge'] for the
    ridge-sensitivity check (this coefficient is a ridge-regularized common-mode gain).
    """
    N = len(names)
    Xa = _augment(X_list)
    names_aug = list(names) + ["GLOBAL"]
    kw = dict(FIT_KW)
    if ridge is not None:
        kw["ridge"] = ridge
    res = fit_distributional_connectome(Xa, names_aug, lag=lag, **kw)
    G = res.matrices["gain"]                 # [N+1, N+1] [post,pre]
    return np.abs(G[:N, N])                  # coupling GLOBAL(source) -> target j


def _global_loadings(X_list, method=GLOBAL_METHOD):
    """Per-neuron global-mode LOADING = mean over worms of corr(trace_i, g_w).

    NaN columns (28w config) are handled per worm; a neuron with no finite/variable frames in
    any worm gets NaN loading (dropped from matching).
    """
    N = X_list[0].shape[1]
    acc = np.zeros(N)
    cnt = np.zeros(N)
    for X in X_list:
        Xc = np.asarray(X, float)
        g = cm.global_signal(X, method)
        if np.std(g) < 1e-9:
            continue
        for i in range(N):
            xi = Xc[:, i]
            good = np.isfinite(xi)
            if good.sum() > 10 and np.std(xi[good]) > 1e-9:
                r = np.corrcoef(xi[good], g[good])[0, 1]
                if np.isfinite(r):
                    acc[i] += r
                    cnt[i] += 1
    L = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return L


def _matched_control_neurons(expressing, feats, rng, n=None):
    """Greedy nearest-neighbour control set of NON-expressing neurons matched to the
    expressing set on a (standardized) multi-feature vector ``feats`` [N, k].

    Generalises common.variance_matched_controls to >1 matching axis (so we can match on
    log-variance AND global-mode loading simultaneously).  Neurons with any non-finite
    feature are excluded from both the target list and the control pool.
    """
    expressing = np.asarray(expressing, bool)
    F = np.asarray(feats, float)
    if F.ndim == 1:
        F = F[:, None]
    finite = np.isfinite(F).all(axis=1)
    idx_exp = np.where(expressing & finite)[0]
    idx_pool = np.where((~expressing) & finite)[0]
    N = len(expressing)
    mask = np.zeros(N, bool)
    if len(idx_exp) == 0 or len(idx_pool) == 0:
        return mask
    # standardize on the pool so distances are comparable across axes
    mu = F[idx_pool].mean(axis=0)
    sd = F[idx_pool].std(axis=0) + 1e-9
    Fn = (F - mu) / sd
    if n is None:
        n = len(idx_exp)
    n = min(n, len(idx_pool))
    used = set()
    chosen = []
    for e in rng.permutation(idx_exp):
        if len(chosen) >= n:
            break
        best, bestd = None, np.inf
        for p in idx_pool:
            if p in used:
                continue
            d = float(((Fn[p] - Fn[e]) ** 2).sum())
            if d < bestd:
                bestd, best = d, p
        if best is None:
            break
        used.add(best)
        chosen.append(best)
    mask[chosen] = True
    return mask


def _matched_control_edges(mod_mask2d, off, V, rng):
    """Variance-matched control EDGES for approach B.

    For each modulator edge (i<-j) find a NON-modulator off-diagonal edge (i'<-j') matched on
    the standardized (log source-variance, log target-variance) 2-vector, greedily without
    replacement.  Returns a [N,N] bool control-edge mask of the same count as the modulator
    edge set.
    """
    lv = np.log(np.asarray(V, float) + 1e-12)
    mod_edges = [tuple(e) for e in np.argwhere(mod_mask2d & off)]
    pool_edges = [tuple(e) for e in np.argwhere((~mod_mask2d) & off)]
    N = mod_mask2d.shape[0]
    ctrl = np.zeros((N, N), bool)
    if not mod_edges or not pool_edges:
        return ctrl
    P = np.array([[lv[i], lv[j]] for (i, j) in pool_edges])
    mu = P.mean(axis=0)
    sd = P.std(axis=0) + 1e-9
    Pn = (P - mu) / sd
    used = np.zeros(len(pool_edges), bool)
    for k in rng.permutation(len(mod_edges)):
        i, j = mod_edges[k]
        f = (np.array([lv[i], lv[j]]) - mu) / sd
        d = ((Pn - f) ** 2).sum(axis=1)
        d[used] = np.inf
        b = int(np.argmin(d))
        if not np.isfinite(d[b]):
            break
        used[b] = True
        ci, cj = pool_edges[b]
        ctrl[ci, cj] = True
    return ctrl


def _slow_lags(lags):
    lo, hi = C.bands()["slow"]           # seconds
    sl = [L for L in lags if lo <= L / FPS <= hi]
    return sl if sl else [max(lags)]     # fallback: the slowest available lag


def _modulator_adjacency(names):
    """Off-diagonal boolean edge masks for neuropeptide:all and monoamine:all, taken from
    the biolag build_registry (adjacency>0), plus their union.  Faithful to the spec's
    'build_registry adjacency>0' edge definition.
    """
    from sid_elegans.biolag.references import build_registry
    refs = build_registry(names, min_specific_edges=10_000, verbose=False)  # drop specifics
    by = {r.name: r for r in refs}
    N = len(names)
    off = ~np.eye(N, dtype=bool)
    out = {}
    for key, ref_name in [("neuropeptide", "neuropeptide:all"),
                          ("monoamine", "monoamine:all")]:
        A = by[ref_name].adjacency if ref_name in by else np.zeros((N, N))
        out[key] = (np.asarray(A, float) > 0) & off
    out["union"] = (out["neuropeptide"] | out["monoamine"]) & off
    return out, off


# --------------------------------------------------------------------------------------
# APPROACH A  — global state as an appended source; partial state-gain per target
# --------------------------------------------------------------------------------------
def run_approach_A(X_list, names, rng):
    N = len(names)
    V = cm.source_variance(X_list)
    loading = _global_loadings(X_list)
    logV = np.log(V + 1e-12)
    covars = np.column_stack([logV, loading])

    # expressing set + two EQUAL-SIZE genuinely-matched pair sets per modulator via
    # cm.matched_groups.  This FIXES the vacuous variance_matched_controls / _matched_control_
    # neurons usage: when expressing>pool (true here) the old code silently took the whole pool
    # with NO matching.  matched_groups subsamples the LARGER group so exp_mask & ctrl_mask are
    # equal-size, genuinely matched; on this data BOTH groups get subsampled.
    #   (a) variance-matched   on log-variance
    #   (b) var+loading-matched on [log-variance, global-mode loading] (loading & receptor
    #       status are collinear, so this is the stringent set)
    sets = {}
    for mod in MODULATORS:
        exp = cm.receptor_expressing(names, mod)
        exp_v, ctrl_v, info_v = cm.matched_groups(exp, logV, np.random.default_rng(SEED))
        exp_vl, ctrl_vl, info_vl = cm.matched_groups(exp, covars, np.random.default_rng(SEED))
        sets[mod] = dict(exp=exp, exp_v=exp_v, ctrl_v=ctrl_v, info_v=info_v,
                         exp_vl=exp_vl, ctrl_vl=ctrl_vl, info_vl=info_vl)

    # ---- observed per-lag state-gain (one augmented fit per lag, shared by both modulators)
    sg_by_lag = {}
    for L in LAGS:
        try:
            sg_by_lag[L] = _state_gain_vector(X_list, names, L)
        except Exception as e:
            sg_by_lag[L] = np.full(N, np.nan)
            print(f"[A] lag {L} fit failed: {e}")

    # ---- EXPLORATORY per-lag table: expr vs variance-matched subset at every lag.  This is a
    # garden-of-forking-paths scan and is NOT used for confirmatory inference (that runs at the
    # single pre-registered PRESPEC_SLOW_LAG below).
    per_lag = {}
    for L in LAGS:
        sg = sg_by_lag[L]
        row = {}
        for mod in MODULATORS:
            s = sets[mod]
            gp = H.group_perm_test(sg, s["exp_v"], s["ctrl_v"], n_perm=NPERM,
                                   rng=np.random.default_rng(SEED))
            row[mod] = gp
        per_lag[str(L)] = row

    slow = _slow_lags(LAGS)

    # EXPLORATORY argmax slow lag (slow-band lag with the largest observed diff) — reported for
    # transparency only; confirmatory tests use PRESPEC_SLOW_LAG, not this.
    def _argmax_slow_lag(mod):
        cand = [(per_lag[str(L)][mod]["diff"], L) for L in slow
                if np.isfinite(per_lag[str(L)][mod]["diff"])]
        if not cand:
            return slow[0]
        return max(cand, key=lambda t: t[0])[1]

    # ---- PRE-REGISTERED confirmatory state-gain at the single fixed slow lag (no argmax)
    try:
        sg_prespec = _state_gain_vector(X_list, names, PRESPEC_SLOW_LAG)
    except Exception as e:
        sg_prespec = np.full(N, np.nan)
        print(f"[A] prespec lag {PRESPEC_SLOW_LAG} fit failed: {e}")

    modulators_out = {}
    for mod in MODULATORS:
        s = sets[mod]

        # (1) confirmatory permutation test at the PRE-REGISTERED slow lag, variance-matched
        perm_var = H.group_perm_test(sg_prespec, s["exp_v"], s["ctrl_v"], n_perm=NPERM,
                                     rng=np.random.default_rng(SEED))
        # (1b) same at the stringent var+loading-matched subset
        perm_load = H.group_perm_test(sg_prespec, s["exp_vl"], s["ctrl_vl"], n_perm=NPERM,
                                      rng=np.random.default_rng(SEED))

        # (2) HEADLINE confound control: regress state-gain on [log-variance, loading] and test
        # the RESIDUAL over ALL neurons (expressing vs non-expressing).  Robust when receptor
        # status & loading are collinear.  Survives here => not merely a loading difference;
        # vanishes here => the headline was a global-mode-loading artefact.
        resid = cm.regress_out(sg_prespec, covars)
        perm_resid = H.group_perm_test(resid, s["exp"], ~s["exp"], n_perm=NPERM,
                                       rng=np.random.default_rng(SEED))

        # loading means (transparency: are expressing neurons just high global-mode loaders?)
        def _mean(mask):
            v = loading[np.asarray(mask, bool) & np.isfinite(loading)]
            return float(v.mean()) if len(v) else float("nan")
        loading_means = dict(expressing=_mean(s["exp"]),
                             var_ctrl=_mean(s["ctrl_v"]),
                             load_ctrl=_mean(s["ctrl_vl"]))

        # (4) ridge-sensitivity of the expr-ctrl diff at the pre-registered lag: mats['gain']
        # [j,N] is a RIDGE-REGULARIZED common-mode gain, so check the sign is stable in ridge.
        ridge_diffs = {}
        for rg in (1e-3, 1e-2, 1e-1):
            try:
                sgr = _state_gain_vector(X_list, names, PRESPEC_SLOW_LAG, ridge=rg)
                a = sgr[np.asarray(s["exp_v"], bool)]; a = a[np.isfinite(a)]
                b = sgr[np.asarray(s["ctrl_v"], bool)]; b = b[np.isfinite(b)]
                ridge_diffs[f"{rg:g}"] = (float(a.mean() - b.mean())
                                          if len(a) and len(b) else float("nan"))
            except Exception as e:
                ridge_diffs[f"{rg:g}"] = float("nan")
                print(f"[A] ridge {rg} fit failed: {e}")
        finite_rd = [d for d in ridge_diffs.values() if np.isfinite(d)]
        ridge_sign_stable = bool(finite_rd and (all(d > 0 for d in finite_rd)
                                                or all(d < 0 for d in finite_rd)))

        entry = dict(
            prespec_slow_lag=int(PRESPEC_SLOW_LAG),
            exploratory_argmax_lag=int(_argmax_slow_lag(mod)),
            confirm_perm_var=perm_var,
            confirm_perm_loadmatched=perm_load,
            loading_regression_headline=perm_resid,
            loading_means=loading_means,
            match_info=dict(
                var=dict(n_pairs=s["info_v"]["n_pairs"],
                         resid_imbalance_std=s["info_v"]["resid_imbalance_std"],
                         pool_n=s["info_v"]["pool_n"], exp_n=s["info_v"]["exp_n"]),
                var_loading=dict(n_pairs=s["info_vl"]["n_pairs"],
                                 resid_imbalance_std=s["info_vl"]["resid_imbalance_std"],
                                 pool_n=s["info_vl"]["pool_n"], exp_n=s["info_vl"]["exp_n"])),
            ridge_sensitivity=dict(diffs=ridge_diffs, sign_stable=ridge_sign_stable),
        )

        # bootstrap CI + circshift null on the group-mean-difference statistic at the
        # PRE-REGISTERED slow lag (not the argmax lag).  stat_fn takes ORIGINAL worm arrays;
        # augmentation happens inside so circshift (which rolls each neuron column) destroys
        # the global mode -> the correct gating null.
        run_bc = BOTH_MOD or mod == "neuropeptide"
        if run_bc:
            exp_m, ctrl_m = s["exp_v"], s["ctrl_v"]

            def _stat(Xs, _L=PRESPEC_SLOW_LAG, _e=exp_m, _c=ctrl_m):
                try:
                    sg = _state_gain_vector(Xs, names, _L)
                except Exception:
                    return np.nan
                a = sg[np.asarray(_e, bool)]
                b = sg[np.asarray(_c, bool)]
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if len(a) == 0 or len(b) == 0:
                    return np.nan
                return float(a.mean() - b.mean())

            entry["bootstrap"] = H.worm_bootstrap(_stat, X_list, n_boot=NBOOT,
                                                  rng=np.random.default_rng(SEED),
                                                  throttle=THROTTLE)
            entry["circshift"] = H.circshift_p(_stat, X_list, n_surr=NSURR,
                                               rng=np.random.default_rng(SEED),
                                               throttle=THROTTLE)
        else:
            entry["bootstrap"] = "skipped (set NL_BOTH_MOD=1)"
            entry["circshift"] = "skipped (set NL_BOTH_MOD=1)"

        entry["n_expressing"] = int(s["exp"].sum())
        entry["n_matched_pairs"] = int(s["info_v"]["n_pairs"])
        modulators_out[mod] = entry

    return dict(per_lag=per_lag,
                per_lag_note="EXPLORATORY (garden-of-forking-paths); confirmatory inference "
                             "uses prespec_slow_lag only",
                slow_band_lags=[int(L) for L in slow],
                prespec_slow_lag=int(PRESPEC_SLOW_LAG),
                modulators=modulators_out)


# --------------------------------------------------------------------------------------
# APPROACH B  — phase-split; per-edge |gain| phase difference on modulator vs control edges
# --------------------------------------------------------------------------------------
def run_approach_B(X_list, names, rng):
    N = len(names)
    V = cm.source_variance(X_list)
    slow = _slow_lags(LAGS)
    repL = int(slow[len(slow) // 2])     # a middle slow-band lag (deterministic)

    hi, lo = cm.phase_masks(X_list, GLOBAL_METHOD)
    Xhi = cm.subset_frames(X_list, hi)
    Xlo = cm.subset_frames(X_list, lo)
    if len(Xhi) < 2 or len(Xlo) < 2:
        return dict(error="too few worms survive the phase split", rep_slow_lag=repL,
                    n_hi=len(Xhi), n_lo=len(Xlo))

    res_hi = fit_distributional_connectome(Xhi, names, lag=repL, **FIT_KW)
    res_lo = fit_distributional_connectome(Xlo, names, lag=repL, **FIT_KW)
    Ghi = np.abs(res_hi.matrices["gain"])
    Glo = np.abs(res_lo.matrices["gain"])
    delta = np.abs(Ghi - Glo)            # [N,N] per-edge cross-phase change in |gain|

    adj, off = _modulator_adjacency(names)
    delta_flat = delta.reshape(-1)

    modulators_out = {}
    for key in ["neuropeptide", "monoamine", "union"]:
        mod_mask = adj[key]
        ctrl_mask = _matched_control_edges(mod_mask, off, V, np.random.default_rng(SEED))
        gp = H.group_perm_test(delta_flat, mod_mask.reshape(-1), ctrl_mask.reshape(-1),
                               n_perm=NPERM, rng=np.random.default_rng(SEED))
        modulators_out[key] = dict(n_mod_edges=int(mod_mask.sum()),
                                   n_ctrl_edges=int(ctrl_mask.sum()), **gp)

    return dict(rep_slow_lag=repL, n_hi=len(Xhi), n_lo=len(Xlo),
                mean_abs_gain_hi=float(Ghi[off].mean()), mean_abs_gain_lo=float(Glo[off].mean()),
                modulators=modulators_out)


# --------------------------------------------------------------------------------------
# per-config driver
# --------------------------------------------------------------------------------------
def run_config(config):
    rng = np.random.default_rng(SEED)
    X_list, names, fps = cm.get_data(config)
    N = len(names)
    out = dict(N=int(N), n_worms=int(len(X_list)), fps=float(fps))
    try:
        out["approach_A"] = run_approach_A(X_list, names, rng)
    except Exception as e:
        import traceback
        out["approach_A"] = {"error": f"{e}", "trace": traceback.format_exc()[-800:]}
        print(f"[{config}] approach A failed: {e}")
    try:
        out["approach_B"] = run_approach_B(X_list, names, rng)
    except Exception as e:
        import traceback
        out["approach_B"] = {"error": f"{e}", "trace": traceback.format_exc()[-800:]}
        print(f"[{config}] approach B failed: {e}")
    return out


def _summarize(results):
    print("\n" + "=" * 74)
    print("GATING — global brain-state AS the modulator (higher |state-gain| on")
    print("receptor-expressing targets => gating rides the global-state axis)")
    print("=" * 74)
    for cfg, r in results["configs"].items():
        if "error" in r:
            print(f"\n[{cfg}] ERROR: {r['error']}"); continue
        print(f"\n[{cfg}]  worms={r['n_worms']}  N={r['N']}")
        A = r.get("approach_A", {})
        if "modulators" in A:
            print(f"  A) slow-band lags (frames): {A['slow_band_lags']}  "
                  f"prespec_slow_lag={A.get('prespec_slow_lag')}  (per-lag table EXPLORATORY)")
            for mod, e in A["modulators"].items():
                pv = e["confirm_perm_var"]
                lp = e["confirm_perm_loadmatched"]
                rh = e["loading_regression_headline"]
                lm = e["loading_means"]
                mi = e["match_info"]
                print(f"   {mod:12s} prespec_lag={e['prespec_slow_lag']:>2} "
                      f"(explor argmax={e['exploratory_argmax_lag']:>2})  "
                      f"diff(expr-ctrl)={pv['diff']:+.4f}  p_greater={pv['p_greater']:.4f}  "
                      f"(n_pairs={e['n_matched_pairs']}, n_expr={e['n_expressing']})")
                print(f"   {'':12s} HEADLINE loading-regression residual: "
                      f"diff={rh['diff']:+.4f} p_greater={rh['p_greater']:.4f} "
                      f"(n_a={rh['n_a']}, n_b={rh['n_b']})")
                print(f"   {'':12s} var+loading-matched: diff={lp['diff']:+.4f} "
                      f"p_greater={lp['p_greater']:.4f}  "
                      f"| loading expr={lm['expressing']:+.3f} varC={lm['var_ctrl']:+.3f} "
                      f"loadC={lm['load_ctrl']:+.3f}")
                rs = e["ridge_sensitivity"]
                rd = ", ".join(f"{k}:{v:+.4f}" for k, v in rs["diffs"].items())
                print(f"   {'':12s} ridge-sensitivity diffs={{{rd}}} "
                      f"sign_stable={rs['sign_stable']}  "
                      f"| resid_imbalance(var)={mi['var']['resid_imbalance_std']}")
                bc = e.get("bootstrap")
                if isinstance(bc, dict):
                    cs = e.get("circshift", {})
                    print(f"   {'':12s} bootstrap diff CI=[{bc['lo']:+.4f},{bc['hi']:+.4f}] "
                          f"(obs {bc['obs']:+.4f})  circshift p={cs.get('p'):.4f} "
                          f"z={cs.get('z'):.2f}")
        B = r.get("approach_B", {})
        if "modulators" in B:
            print(f"  B) phase-split rep_lag={B['rep_slow_lag']}  "
                  f"|gain| hi={B['mean_abs_gain_hi']:.3f} lo={B['mean_abs_gain_lo']:.3f}")
            for key, e in B["modulators"].items():
                print(f"   {key:12s} edges mod={e['n_mod_edges']:>4} ctrl={e['n_ctrl_edges']:>4}  "
                      f"delta(mod-ctrl)={e['diff']:+.4f}  p_greater={e['p_greater']:.4f}")
        elif "error" in B:
            print(f"  B) ERROR: {B['error']}")
    print("=" * 74)


if __name__ == "__main__":
    t0 = time.time()
    print(f"[gating] SMOKE={SMOKE} configs={CONFIGS} lags={LAGS}")
    print(f"[gating] NBOOT={NBOOT} NSURR={NSURR} NPERM={NPERM} THROTTLE={THROTTLE} "
          f"BOTH_MOD={BOTH_MOD} seed={SEED}")
    results = {
        "meta": dict(lever="gating", smoke=SMOKE, seed=SEED, global_method=GLOBAL_METHOD,
                     lags=[int(L) for L in LAGS], configs=CONFIGS,
                     nboot=NBOOT, nsurr=NSURR, nperm=NPERM, throttle=THROTTLE,
                     both_mod=BOTH_MOD, fit_kw={k: v for k, v in FIT_KW.items()},
                     slow_band_s=list(C.bands()["slow"]),
                     idea="global brain-state as THE modulator (not a confound to remove)"),
        "configs": {},
    }
    for cfg in CONFIGS:
        print(f"\n[gating] === running config {cfg} ===")
        try:
            results["configs"][cfg] = run_config(cfg)
        except Exception as e:
            import traceback
            results["configs"][cfg] = {"error": f"{e}", "trace": traceback.format_exc()[-800:]}
            print(f"[gating] config {cfg} failed hard: {e}")
        # write incrementally so a crash on config 2 still leaves config 1's results
        cm.save_json("gating", results)

    results["meta"]["runtime_s"] = round(time.time() - t0, 1)
    path = cm.save_json("gating", results)
    _summarize(results)
    print(f"\n[gating] wrote {path}  ({results['meta']['runtime_s']}s)")
