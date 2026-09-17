r"""New-levers analysis #1 — FUNCTIONAL TARGET + KINETICS (the highest-ceiling lever).

Two things anatomy/receptor targets can never give us live only in the Randi/Leifer
optogenetic *functional* atlas: (A) a directed, causally-measured ground-truth connectome
to score the distributional channels against on *confirmed* edges (fake-negative-safe), and
(B) a per-edge measured causal *timescale* (kinetics) we can line up against the SID
gain-channel's own preferred lag — a channel x lag test with a real physical anchor.

PART A (static target).  For SID channels {mean, gain, tail} and baselines
{pearson_lag, ridge_var, source-variance control}, compute AUROC-vs-lag against the
FUNCTIONAL target (atlas['positive']) restricted to atlas['eval_mask'] (confirmed edges
only — critical: the unconfirmed cells are NOT scored as negatives).  The SAME channels are
also scored against the Cook structural:chemical target for contrast (both on the full
off-diagonal and re-restricted to the funatlas eval mask for an apples-to-apples number).
Key questions: (i) does any distributional channel beat the baselines vs functional truth?
(ii) is the gain connectome closer to FUNCTIONAL than to ANATOMICAL truth?  The best
distributional (channel, lag) and the best baseline get a worm-bootstrap CI and a
circular-shift surrogate p on the PRIMARY config.

PART B (kinetics — the novel test).  For every edge in atlas['eval_mask'] with a finite
atlas['timescale'], the SID gain channel's PREFERRED LAG = the lag (seconds) that maximises
|gain[j,i]| across the lag grid (one fit per lag).  Spearman-correlate that preferred-lag
vector against the measured timescale across edges; do the same for the mean channel as a
control (theory: mean tracks fast/wired, gain tracks the slow causal timescale).  A
permutation p comes from shuffling edge labels (the measured-timescale vector); the paired
gain-minus-mean advantage is tested under the same shuffles.

Conventions (never violated): X_list = per-worm [T,N] globally-standardized; every matrix is
[post,pre] = [target,source]; fps = 4.0.  All heavy counts come from env vars; all heavy work
is under __main__; all numbers are saved via common.save_json to output/newlevers/functional.json.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

# --------------------------------------------------------------------------------------
# env-tunable knobs (HARD RULE 1/2/5): nothing heavy is hardcoded, everything is seeded.
# --------------------------------------------------------------------------------------
SMOKE = os.environ.get("NEWLEVERS_SMOKE", "0") == "1"
SEED = int(os.environ.get("NL_SEED", "0"))

# defaults per the shared spec; capped hard in smoke so a run finishes in <~90 s.
_NSURR = int(os.environ.get("NL_NSURR", "200"))
_NBOOT = int(os.environ.get("NL_NBOOT", "200"))
_NPERM = int(os.environ.get("NL_NPERM", "5000"))
_THROTTLE = float(os.environ.get("NL_THROTTLE", "0.3"))

PRIMARY = os.environ.get("NL_CONFIG", "6w_clean_deconv")   # functional lever's primary config
ROBUSTNESS = "6w_clean_raw"                                 # same worms/neurons, raw (non-deconv)

if SMOKE:
    NSURR = min(_NSURR, 5)
    NBOOT = min(_NBOOT, 5)
    NPERM = min(_NPERM, 5)
    THROTTLE = 0.0
    LAGS_SMOKE = [1, 5, 20]        # 3 lags spanning fast->slow so kinetics has variation
    CONFIGS = [PRIMARY]            # ONE data config in smoke
else:
    NSURR, NBOOT, NPERM = _NSURR, _NBOOT, _NPERM
    THROTTLE = _THROTTLE
    LAGS_SMOKE = None
    CONFIGS = [PRIMARY, ROBUSTNESS]

FPS = 4.0
FIT_KW = dict(target_mode="next", ridge=1e-2, fps=FPS)   # matches harness/estimator defaults
SID_CHANNELS = ("mean", "gain", "tail")
DIST_CHANNELS = ("gain", "tail")                          # distributional channels
BASELINES = ("pearson", "ridge", "control")
METHODS = SID_CHANNELS + BASELINES


# --------------------------------------------------------------------------------------
# small helpers (import-time cheap: only numpy at module scope)
# --------------------------------------------------------------------------------------
def _control_matrix(X_list, N):
    """Source-variance control: C[j,i] = Var(source i) tiled down every target row (lag-flat,
    model-free). Its AUROC is identical at every lag by construction (the 'is it just marginal
    variance?' null)."""
    from sid_elegans.newlevers.common import source_variance
    V = source_variance(X_list)                 # [N]
    C = np.tile(V[None, :], (N, 1))             # C[j,i] = V[i]
    np.fill_diagonal(C, 0.0)
    return C


def _best_over_curve(curve_dict, keys):
    """Return (best_key, best_lag_index, best_value) = argmax over methods `keys` and lags of a
    {method: [auroc per lag]} dict, ignoring NaNs."""
    best = (None, None, -np.inf)
    for k in keys:
        arr = np.asarray(curve_dict[k], float)
        if not np.isfinite(arr).any():
            continue
        li = int(np.nanargmax(arr))
        if arr[li] > best[2]:
            best = (k, li, float(arr[li]))
    return best


def _rank(a):
    from scipy.stats import rankdata
    return rankdata(np.asarray(a, float))


def _corr(a, b):
    """Pearson correlation of two vectors (used on ranks -> Spearman). NaN-safe on degenerate."""
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _kinetics_perm(pref_gain, pref_mean, ts, n_perm, rng):
    """Paired Spearman-vs-timescale permutation test.

    Ranks are fixed; the null shuffles the measured-timescale vector (edge labels), which
    de-correlates BOTH channels simultaneously so the gain-minus-mean advantage is tested on
    the identical shuffle. Returns Spearman(gain), Spearman(mean), their difference, and
    one-sided (greater) + two-sided permutation p-values.
    """
    ts = np.asarray(ts, float)
    ok = np.isfinite(pref_gain) & np.isfinite(pref_mean) & np.isfinite(ts)
    pg, pm, y = np.asarray(pref_gain, float)[ok], np.asarray(pref_mean, float)[ok], ts[ok]
    n = len(y)
    out = dict(n_edges=int(n), n_perm=int(n_perm))
    # degenerate guards: need >=3 edges and non-constant vectors on each side
    if n < 3 or np.unique(y).size < 2 or np.unique(pg).size < 2 or np.unique(pm).size < 2:
        out.update(spearman_gain=np.nan, spearman_mean=np.nan, diff=np.nan,
                   p_gain_greater=np.nan, p_gain_two=np.nan, p_mean_greater=np.nan,
                   p_mean_two=np.nan, p_diff_greater=np.nan)
        return out
    rg, rm, ry = _rank(pg), _rank(pm), _rank(y)
    obs_g, obs_m = _corr(rg, ry), _corr(rm, ry)
    obs_d = obs_g - obs_m
    cg = cm = cd = 0
    cg_abs = cm_abs = 0
    for _ in range(n_perm):
        ryp = rng.permutation(ry)
        g, m = _corr(rg, ryp), _corr(rm, ryp)
        d = g - m
        cg += g >= obs_g; cm += m >= obs_m; cd += d >= obs_d
        cg_abs += abs(g) >= abs(obs_g); cm_abs += abs(m) >= abs(obs_m)
    out.update(
        spearman_gain=obs_g, spearman_mean=obs_m, diff=obs_d,
        p_gain_greater=(cg + 1) / (n_perm + 1), p_gain_two=(cg_abs + 1) / (n_perm + 1),
        p_mean_greater=(cm + 1) / (n_perm + 1), p_mean_two=(cm_abs + 1) / (n_perm + 1),
        p_diff_greater=(cd + 1) / (n_perm + 1),
    )
    return out


# --------------------------------------------------------------------------------------
# per-config analysis (Part A curves + Part B kinetics); inference is added for PRIMARY only
# --------------------------------------------------------------------------------------
def analyze_config(config, lags, do_inference, rng):
    """Run Parts A and B for one data config. Robust: returns a dict with an 'error' key on
    failure rather than raising, so partial results are still saved."""
    from sid_elegans.newlevers.common import get_data
    from sid_elegans.newlevers.funatlas import functional_reference
    from sid_elegans.newlevers.harness import (auroc_at_lag, channel_matrix, circshift_p,
                                               fit_all_lags, worm_bootstrap)
    from sid_elegans.biolag.references import build_registry
    from sid_elegans.baselines import pearson_lag, ridge_var
    from sid_elegans.estimator import fit_distributional_connectome

    res = {"config": config, "lags_frames": list(lags),
           "lags_seconds": [L / FPS for L in lags]}
    try:
        X, names, fps = get_data(config)
    except Exception as e:                                  # data load failed -> bail cleanly
        res["error"] = f"get_data({config}): {e!r}"
        return res
    N = len(names)
    res["n_worms"] = len(X); res["n_neurons"] = N

    # --- ground truths -------------------------------------------------------------------
    ref, atlas = functional_reference(names)
    if atlas is None:
        res["error"] = "functional_reference returned None (atlas unavailable)"
        return res
    positive = np.asarray(atlas["positive"], float)         # binary functional target [post,pre]
    eval_mask = np.asarray(atlas["eval_mask"], bool)        # confirmed edges only (scoring mask)
    timescale = np.asarray(atlas["timescale"], float)       # per-edge measured causal timescale (s)
    res["n_eval_edges"] = int(eval_mask.sum())
    res["n_positive_in_mask"] = int((positive[eval_mask] > 0).sum())

    try:
        registry = build_registry(names, verbose=False)
        struct = next(r.adjacency for r in registry if r.name == "structural:chemical")
        struct = np.asarray(struct, float)
    except Exception as e:
        struct = None
        res["structural_error"] = repr(e)

    # --- fit SID once per lag (reused for BOTH Part A scoring and Part B preferred-lag) ----
    fits = fit_all_lags(X, names, lags=lags, channels=SID_CHANNELS, **FIT_KW)
    control = _control_matrix(X, N)

    def method_matrix(L, m):
        if m in SID_CHANNELS:
            return fits[L][m]
        if m == "pearson":
            return pearson_lag(X, L)
        if m == "ridge":
            return ridge_var(X, L, ridge=1.0)
        if m == "control":
            return control
        raise ValueError(m)

    # ================= PART A: AUROC vs functional (eval mask) and structural =============
    targets = {"functional_eval": (positive, eval_mask)}
    if struct is not None:
        targets["structural_full"] = (struct, None)          # anatomy: full off-diagonal
        targets["structural_evalmask"] = (struct, eval_mask)  # anatomy on the SAME edge set
    curves = {tk: {m: [] for m in METHODS} for tk in targets}
    for L in lags:
        mats = {m: method_matrix(L, m) for m in METHODS}
        for tk, (tgt, msk) in targets.items():
            for m in METHODS:
                curves[tk][m].append(auroc_at_lag(mats[m], tgt, msk))
    res["auroc_curves"] = curves

    # per-channel best AUROC vs each target (for "is gain closer to functional than anatomy?")
    def best_val(tk, m):
        arr = np.asarray(curves[tk][m], float)
        return float(np.nanmax(arr)) if np.isfinite(arr).any() else float("nan")
    res["best_auroc"] = {tk: {m: best_val(tk, m) for m in METHODS} for tk in targets}

    # headline selections vs functional truth
    fk = "functional_eval"
    best_sid = _best_over_curve(curves[fk], SID_CHANNELS)
    best_dist = _best_over_curve(curves[fk], DIST_CHANNELS)
    best_base = _best_over_curve(curves[fk], BASELINES)
    res["best_sid"] = dict(channel=best_sid[0], lag=int(lags[best_sid[1]]) if best_sid[1] is not None else None,
                           lag_s=(lags[best_sid[1]] / FPS) if best_sid[1] is not None else None, auroc=best_sid[2])
    res["best_distributional"] = dict(channel=best_dist[0],
                                      lag=int(lags[best_dist[1]]) if best_dist[1] is not None else None,
                                      lag_s=(lags[best_dist[1]] / FPS) if best_dist[1] is not None else None,
                                      auroc=best_dist[2])
    res["best_baseline"] = dict(method=best_base[0],
                                lag=int(lags[best_base[1]]) if best_base[1] is not None else None,
                                lag_s=(lags[best_base[1]] / FPS) if best_base[1] is not None else None,
                                auroc=best_base[2])
    # (i) does the best distributional channel beat the best baseline?
    res["dist_beats_baseline_delta"] = (best_dist[2] - best_base[2]
                                        if np.isfinite(best_dist[2]) and np.isfinite(best_base[2]) else None)
    # (ii) is gain closer to functional than to anatomical truth?
    if struct is not None:
        gf = best_val("functional_eval", "gain")
        ga = best_val("structural_evalmask", "gain")         # same edge set -> comparable
        res["gain_functional_minus_anatomical"] = (gf - ga if np.isfinite(gf) and np.isfinite(ga) else None)
        mf = best_val("functional_eval", "mean")
        ma = best_val("structural_evalmask", "mean")
        res["mean_functional_minus_anatomical"] = (mf - ma if np.isfinite(mf) and np.isfinite(ma) else None)

    # ================= PART B: KINETICS (preferred lag vs measured timescale) ==============
    lag_s = np.array([L / FPS for L in lags], float)
    # edge sets: spec = eval_mask & finite timescale; robustness = positive & finite timescale.
    sets = {
        "eval_mask": np.asarray(eval_mask & np.isfinite(timescale), bool),
        "positive_only": np.asarray((positive > 0) & np.isfinite(timescale), bool),
    }
    kin = {}
    for si, (set_name, sel) in enumerate(sets.items()):
        js, is_ = np.where(sel)
        if len(js) == 0:
            kin[set_name] = {"n_edges": 0}
            continue
        ts_vec = timescale[js, is_]
        # preferred lag (s) per edge = argmax over lags of |channel[j,i]|
        gain_stack = np.abs(np.stack([fits[L]["gain"][js, is_] for L in lags], axis=1))  # [E, nlag]
        mean_stack = np.abs(np.stack([fits[L]["mean"][js, is_] for L in lags], axis=1))
        pref_gain = lag_s[np.argmax(gain_stack, axis=1)]
        pref_mean = lag_s[np.argmax(mean_stack, axis=1)]
        # deterministic per-set seed (NOT hash(): str hashing is process-randomized)
        rng_k = np.random.default_rng(SEED + 100 + si)
        kin[set_name] = _kinetics_perm(pref_gain, pref_mean, ts_vec, NPERM, rng_k)
        kin[set_name]["median_pref_lag_gain_s"] = float(np.median(pref_gain))
        kin[set_name]["median_pref_lag_mean_s"] = float(np.median(pref_mean))
        kin[set_name]["median_timescale_s"] = float(np.median(ts_vec))
    res["kinetics"] = kin

    # ================= inference (PRIMARY config only): CI + surrogate p ===================
    if do_inference:
        inf = {}
        # (a) best distributional SID channel -> refit-per-resample worm bootstrap + circshift
        if best_dist[0] is not None and np.isfinite(best_dist[2]):
            ch, L = best_dist[0], int(lags[best_dist[1]])

            def sid_stat(Xs, ch=ch, L=L):
                r = fit_distributional_connectome(Xs, names, lag=L, **FIT_KW)
                return auroc_at_lag(channel_matrix(r, ch), positive, eval_mask)

            inf["distributional"] = {
                "channel": ch, "lag": L, "lag_s": L / FPS,
                "bootstrap": worm_bootstrap(sid_stat, X, n_boot=NBOOT,
                                            rng=np.random.default_rng(SEED + 1), throttle=THROTTLE),
                "circshift": circshift_p(sid_stat, X, n_surr=NSURR,
                                         rng=np.random.default_rng(SEED + 2), throttle=THROTTLE),
            }
        # (b) best baseline -> same inference (cheap: no estimator refit, so no throttle)
        if best_base[0] is not None and np.isfinite(best_base[2]):
            bm, bL = best_base[0], int(lags[best_base[1]])

            def base_stat(Xs, bm=bm, bL=bL):
                M = method_matrix_for(Xs, bm, bL)
                return auroc_at_lag(M, positive, eval_mask)

            def method_matrix_for(Xs, m, L):
                if m == "pearson":
                    return pearson_lag(Xs, L)
                if m == "ridge":
                    return ridge_var(Xs, L, ridge=1.0)
                if m == "control":
                    return _control_matrix(Xs, N)
                raise ValueError(m)

            # The source-variance control is a lag-flat marginal-variance statistic: it is
            # invariant under per-neuron circular time-shift, so its circshift surrogate p is a
            # degenerate 1.0 (z undefined). Exclude it from the circshift comparator and annotate
            # p/z as N/A; the worm bootstrap (which resamples worms, not time) stays meaningful.
            if bm == "control":
                circ = dict(obs=float(base_stat(X)), p=None, null_mean=None, null_sd=None,
                            z=None, n_surr=0,
                            note="N/A: temporal circular-shift surrogate is meaningless for a "
                                 "lag-flat marginal-variance statistic (invariant under per-neuron "
                                 "circshift -> degenerate p=1).")
            else:
                circ = circshift_p(base_stat, X, n_surr=NSURR,
                                   rng=np.random.default_rng(SEED + 4), throttle=0.0)
            inf["baseline"] = {
                "method": bm, "lag": bL, "lag_s": bL / FPS,
                "bootstrap": worm_bootstrap(base_stat, X, n_boot=NBOOT,
                                            rng=np.random.default_rng(SEED + 3), throttle=0.0),
                "circshift": circ,
            }
        res["inference"] = inf
    return res


# --------------------------------------------------------------------------------------
# entry point (HARD RULE 3: all heavy work only under __main__)
# --------------------------------------------------------------------------------------
def main():
    from sid_elegans.newlevers.common import save_json
    # the estimator legitimately clamps invalid eta2 rows and warns; silence to keep stdout clean
    warnings.filterwarnings("ignore", message="invalid eta2")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    rng = np.random.default_rng(SEED)
    out = {
        "lever": "functional",
        "smoke": SMOKE,
        "params": dict(NSURR=NSURR, NBOOT=NBOOT, NPERM=NPERM, THROTTLE=THROTTLE,
                       seed=SEED, configs=CONFIGS, primary=PRIMARY),
        "conventions": "matrices [post,pre]; scoring on atlas eval_mask (confirmed edges only)",
        "results": {},
    }
    lags = LAGS_SMOKE if LAGS_SMOKE is not None else __import__(
        "sid_elegans.newlevers.harness", fromlist=["LAGS"]).LAGS

    for cfg in CONFIGS:
        try:
            out["results"][cfg] = analyze_config(cfg, lags, do_inference=(cfg == PRIMARY), rng=rng)
        except Exception as e:                              # never let one config kill the run
            import traceback
            out["results"][cfg] = {"config": cfg, "error": repr(e),
                                   "traceback": traceback.format_exc()}

    path = save_json("functional", out)

    # ---------------- concise human summary + verdict ----------------
    print("\n" + "=" * 78)
    print(f"FUNCTIONAL LEVER  (smoke={SMOKE})  ->  {path}")
    print("=" * 78)
    for cfg, r in out["results"].items():
        print(f"\n[{cfg}]")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  worms={r['n_worms']} neurons={r['n_neurons']} "
              f"eval_edges={r['n_eval_edges']} (pos_in_mask={r['n_positive_in_mask']})")
        bd, bb, bs = r["best_distributional"], r["best_baseline"], r["best_sid"]
        print(f"  PART A  vs FUNCTIONAL (eval mask):")
        print(f"    best SID overall : {bs['channel']:>7} @ lag {bs['lag']:>3} "
              f"({bs['lag_s']:.2f}s)  AUROC={bs['auroc']:.3f}")
        print(f"    best distribut'l : {bd['channel']:>7} @ lag {bd['lag']:>3} "
              f"({bd['lag_s']:.2f}s)  AUROC={bd['auroc']:.3f}")
        print(f"    best baseline    : {bb['method']:>7} @ lag {bb['lag']:>3} "
              f"({bb['lag_s']:.2f}s)  AUROC={bb['auroc']:.3f}")
        d = r.get("dist_beats_baseline_delta")
        if d is not None:
            print(f"    dist - baseline  : {d:+.3f}  "
                  f"({'distributional WINS' if d > 0 else 'baseline wins'})")
        gfa = r.get("gain_functional_minus_anatomical")
        if gfa is not None:
            print(f"    gain: AUROC(functional) - AUROC(anatomical, same edges) = {gfa:+.3f}  "
                  f"({'closer to FUNCTIONAL' if gfa > 0 else 'closer to anatomical'})")
        kin = r.get("kinetics", {}).get("eval_mask", {})
        if kin.get("n_edges", 0) >= 3 and np.isfinite(kin.get("spearman_gain", np.nan)):
            print(f"  PART B  KINETICS (eval mask, {kin['n_edges']} edges, n_perm={kin['n_perm']}):")
            print(f"    Spearman(pref-lag, timescale)  gain={kin['spearman_gain']:+.3f} "
                  f"(p={kin['p_gain_greater']:.4f})   mean={kin['spearman_mean']:+.3f} "
                  f"(p={kin['p_mean_greater']:.4f})")
            print(f"    gain - mean = {kin['diff']:+.3f}  (p_diff={kin['p_diff_greater']:.4f})  "
                  f"[theory: gain tracks slow causal timescale > mean]")
        if "inference" in r:
            di = r["inference"].get("distributional")
            if di:
                b, s = di["bootstrap"], di["circshift"]
                print(f"  INFERENCE (primary) {di['channel']} @ lag {di['lag']}: "
                      f"AUROC={b['obs']:.3f}  boot95%=[{b['lo']:.3f},{b['hi']:.3f}]  "
                      f"circshift p={s['p']:.4f} (z={s['z']:.2f})")
            bi = r["inference"].get("baseline")
            if bi:
                b, s = bi["bootstrap"], bi["circshift"]
                cp = ("N/A (lag-flat control)" if s.get("p") is None
                      else f"{s['p']:.4f}")
                print(f"                      {bi['method']} @ lag {bi['lag']}: "
                      f"AUROC={b['obs']:.3f}  boot95%=[{b['lo']:.3f},{b['hi']:.3f}]  "
                      f"circshift p={cp}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
