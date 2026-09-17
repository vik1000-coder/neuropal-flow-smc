r"""Lever #7 -- HIERARCHICAL / JAMES-STEIN SHRINKAGE OF THE ACMMA GAIN CONNECTOME
TOWARD A GANGLION-CLASS PRIOR (SBTG pooling, closed-form, post-fit).

Idea
----
The ACMMA gain matrix (available-case, all 28 worms, no imputation) is a noisy [post, pre]
estimate. Classical hierarchical / James-Stein logic says a per-edge estimate can be improved
in mean-squared error by shrinking it toward a coarser group prior. Here the group prior is the
**ganglion block mean**: for gain edge ``g[j, i]`` we form the mean of gain over all off-diagonal
edges whose (post-ganglion, pre-ganglion) block matches ``(ganglion[j], ganglion[i])`` and shrink

        shrunk[j, i] = (1 - w) * g[j, i] + w * block_mean[j, i].

``w`` is tuned SELF-SUPERVISED (no label peeking) by split-half stability of the recovered graph
(``sid_elegans.stability.split_half_stability``; higher Spearman = more reproducible), exactly the
"does shrinkage toward a class prior buy reproducibility?" question.

We then compare RAW vs BEST-SHRUNK gain on:
  (i)   split-half stability  (expect UP -- the whole point of shrinkage),
  (ii)  AUROC vs neuropeptide:all and monoamine:all reference layers (expect >=, not worse),
  (iii) a biolag-style band-concordance contrast dBC = BC_gain - BC_mean for the peptidergic
        network in the SLOW band -- does shrinkage change the null verdict? (worm-bootstrap CI).

Conventions (never violated): 28w_deconv, X_list = list of [T,N], matrices [post, pre],
fps = 4.0, diagonal (self-edge) is zeroed. Shrinkage is applied ONLY to the gain channel; the
mean channel stays raw as the within-fit baseline for the dBC contrast, so dBC differs between
the raw and shrunk conditions only through the (shrunk) gain channel -- isolating the effect.

Compute
-------
All heavy counts come from env vars (HARD RULE #1). ACMMA is post-fit shrunk, so the block prior
and every ``w`` are essentially free once the raw gain is fit; a per-(worm-subset, lag) memo lets
the raw and best-shrunk worm-bootstraps share every ACMMA fit. Heavy work only under __main__.
"""
from __future__ import annotations

import os
import time

import numpy as np

# Scaffolding (cheap imports; data/estimators are lazy or invoked only under __main__).
from sid_elegans.newlevers import common, harness

# ---- band definition (seconds; matches biolag config TWO_BAND fast/slow) -------------------
# fast = 0.25-1.0 s -> frames {1,2,3}; slow = 2.5-10 s -> frames {10,15,20,30,40}. We thin the
# slow band to {10,20,40} for the (heavy) bootstrap; both bands keep >=2 lags so the band mean
# is not a single-point estimate. The 1.0-2.5 s transition gap is intentionally excluded.
FAST_LAGS_FULL = [1, 2, 3]
SLOW_LAGS_FULL = [10, 20, 40]


# --------------------------------------------------------------------------------------------
# config / env
# --------------------------------------------------------------------------------------------
def get_config():
    smoke = os.environ.get("NEWLEVERS_SMOKE", "0") == "1"
    cfg = dict(
        smoke=smoke,
        # data config for this lever is ACMMA on all worms: 28w_deconv (env-overridable).
        config=os.environ.get("NL_CONFIG", "28w_deconv"),
        seed=common.env_int("NL_SEED", 0),
        # ACMMA knobs -- min_triple small so gain edges survive on the 28w available-case set.
        ridge=common.env_float("NL_RIDGE", 0.01),
        min_triple=common.env_int("NL_MIN_TRIPLE", 50),
        # shrinkage weight grid tuned by split-half stability.
        w_grid=[0.0, 0.25, 0.5, 0.75],
        top_frac=0.1,
        # forced probe weight: if stability rejects shrinkage (best_w == 0) we STILL evaluate the
        # AUROC / dBC comparison at this moderate weight so "does shrinkage change the verdict?"
        # is answered substantively (post-fit, so it reuses every cached ACMMA fit -- no extra cost).
        probe_w=common.env_float("NL_PROBE_W", 0.5),
        # standard resampling knobs (HARD RULE #1). This lever's inference is a worm-bootstrap
        # CI on dBC; NL_NSURR / NL_NPERM are not used by the shrinkage design (recorded in meta).
        n_surr=common.env_int("NL_NSURR", 200),
        n_perm=common.env_int("NL_NPERM", 5000),
        n_boot=common.env_int("NL_NBOOT", 200),
        throttle=common.env_float("NL_THROTTLE", 0.3),
        stab_lag=common.env_int("NL_STAB_LAG", 20),   # lag (frames) at which stability is tuned
        n_splits=common.env_int("NL_NSPLITS", 5),
        fast_lags=list(FAST_LAGS_FULL),
        slow_lags=list(SLOW_LAGS_FULL),
    )
    if smoke:
        # TINY settings: 1 fast + 1 slow lag, <=5 bootstrap reps, 2 splits, no throttle.
        cfg.update(
            fast_lags=[1], slow_lags=[10], stab_lag=10, n_splits=2,
            n_boot=min(cfg["n_boot"], 5), throttle=0.0,
        )
    return cfg


# --------------------------------------------------------------------------------------------
# ganglion block prior + shrinkage (post-fit, deterministic, closed-form)
# --------------------------------------------------------------------------------------------
def block_indicator(names):
    """Return ``(P, blocks)`` with ``P`` a ``[C, N]`` 0/1 ganglion indicator (P[c,i]=1 iff neuron
    i is in block c). Uses ``common.ganglion_labels`` -- the meaningful ~8-ganglion coarsening
    (neurons absent from the atlas map form singleton blocks and are shrunk within themselves)."""
    labels = common.ganglion_labels(names)
    blocks = sorted(set(labels))
    bi = {b: k for k, b in enumerate(blocks)}
    P = np.zeros((len(blocks), len(names)), float)
    for i, lab in enumerate(labels):
        P[bi[lab], i] = 1.0
    return P, blocks


def class_block_mean(M, P):
    """Off-diagonal ganglion-block mean of ``M`` [post,pre], broadcast back to ``[N,N]``.

    ``block_mean[j,i] = mean over finite off-diagonal edges (j',i') with (block[j'],block[i'])
    == (block[j],block[i])``. Diagonal self-edges are excluded from the mean and set to 0 in the
    returned target (the ACMMA diagonal is already zeroed). Vectorised via the indicator P so it
    is cheap to recompute inside the bootstrap.
    """
    N = M.shape[0]
    off = ~np.eye(N, dtype=bool)
    F = np.isfinite(M) & off                 # finite, off-diagonal cells only
    Ms = np.where(F, M, 0.0)
    bsum = P @ Ms @ P.T                       # [C,C] sum of M over each block
    bcnt = P @ F.astype(float) @ P.T          # [C,C] count of contributing cells
    with np.errstate(invalid="ignore", divide="ignore"):
        bmean = np.where(bcnt > 0, bsum / bcnt, 0.0)
    target = P.T @ bmean @ P                  # [N,N] each edge <- its block mean
    np.fill_diagonal(target, 0.0)
    return target


def shrink_gain(g, P, w):
    """James-Stein-style convex shrink of gain toward its ganglion-block prior (diag stays 0)."""
    if w <= 0.0:
        return g
    # ACMMA hard-masks unsupported/thin gain entries to a FINITE 0.0 (thin-triple masking, forced
    # non-finite->0, and whole rows for un-fit targets). Averaging those structural zeros as if
    # they were real edges dilutes the ganglion block-mean prior toward 0, so "shrink toward block
    # mean" would partly become "shrink toward 0" and trivially hurt stability -- an unfair
    # negative. A genuinely-solved continuous gain is essentially never exactly 0.0, so an
    # off-diagonal exact-zero cell IS a masked/unsupported entry: NaN it out before forming the
    # prior. class_block_mean already excludes non-finite cells, so the prior becomes a genuine
    # class mean over real edges. (Shrinkage is still applied to every entry via the raw g below;
    # only the PRIOR's mean is de-diluted.)
    off = ~np.eye(g.shape[0], dtype=bool)
    g_for_prior = np.where(off & (g == 0.0), np.nan, g)
    target = class_block_mean(g_for_prior, P)
    out = (1.0 - w) * g + w * target
    np.fill_diagonal(out, 0.0)
    return out


# --------------------------------------------------------------------------------------------
# memoised ACMMA fit (raw mean + gain per worm-subset, per lag)
# --------------------------------------------------------------------------------------------
_FIT_MEMO: dict = {}


def _subset_key(subset, lag):
    # Elements of `subset` are the persistent original X_list arrays (never copied, never GC'd
    # during the run), so their id() is a stable multiset key -- duplicates from bootstrap
    # resampling are preserved (sorted keeps multiplicity). The memo is a pure speed-up: a miss
    # only recomputes, never returns a wrong fit.
    return (tuple(sorted(id(x) for x in subset)), int(lag))


def fit_channels(subset, lag, names, ridge, min_triple):
    """Return ``(mean[N,N], gain[N,N])`` raw ACMMA channels for a worm subset at one lag (cached)."""
    from sid_elegans.acmma import fit_acmma_connectome
    key = _subset_key(subset, lag)
    hit = _FIT_MEMO.get(key)
    if hit is not None:
        return hit
    mats, _diag = fit_acmma_connectome(subset, names, lag, ridge=ridge, min_triple=min_triple)
    val = (np.asarray(mats["mean"], float).copy(), np.asarray(mats["gain"], float).copy())
    _FIT_MEMO[key] = val
    return val


# --------------------------------------------------------------------------------------------
# band concordance (within-fit, relative)
# --------------------------------------------------------------------------------------------
def _auroc(M, ref):
    """AUROC of |offdiag(M)| vs (ref>0) over all confirmed off-diagonal pairs (harness/score_matrix)."""
    try:
        return harness.auroc_at_lag(M, ref.adjacency)
    except Exception:
        return np.nan


def band_concordance_from_curves(au_by_lag, fast_lags, slow_lags):
    """BC = mean AUROC over slow-band lags - mean over fast-band lags (a within-channel, relative
    quantity; the calcium-kernel floor makes absolute latencies untrustworthy but this contrast
    is defensible). NaN-safe."""
    sl = np.array([au_by_lag[L] for L in slow_lags], float)
    fa = np.array([au_by_lag[L] for L in fast_lags], float)
    if not np.isfinite(sl).any() or not np.isfinite(fa).any():
        return np.nan
    return float(np.nanmean(sl) - np.nanmean(fa))


def dbc_stat(subset, w, ref, names, cfg, P):
    """dBC = BC_gain(shrunk with weight w) - BC_mean(raw) for a worm subset vs ``ref``.

    Positive dBC = the gain channel's above-baseline correspondence to the (slow, peptidergic)
    reference is MORE concentrated in the slow band than the mean channel's is -- the theory-
    predicted distributional, slow signature. The mean channel is never shrunk, so raw vs shrunk
    dBC differ only through the gain band-concordance.
    """
    fast, slow = cfg["fast_lags"], cfg["slow_lags"]
    au_mean, au_gain = {}, {}
    for L in fast + slow:
        m, g = fit_channels(subset, L, names, cfg["ridge"], cfg["min_triple"])
        gs = shrink_gain(g, P, w)
        au_mean[L] = _auroc(m, ref)
        au_gain[L] = _auroc(gs, ref)
    bc_mean = band_concordance_from_curves(au_mean, fast, slow)
    bc_gain = band_concordance_from_curves(au_gain, fast, slow)
    if not (np.isfinite(bc_mean) and np.isfinite(bc_gain)):
        return np.nan
    return float(bc_gain - bc_mean)


def _verdict(boot):
    """Significance verdict from a worm-bootstrap CI dict: 'pos'/'neg' if the 95% CI excludes 0,
    else 'null'."""
    lo, hi = boot.get("lo"), boot.get("hi")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "undetermined"
    if lo > 0:
        return "pos"
    if hi < 0:
        return "neg"
    return "null"


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def main():
    # ACMMA emits benign "Mean of empty slice"/divide warnings when a bootstrap resample happens
    # to drop every worm recording some neuron (its gain row is guarded to 0). Silence for a clean
    # unattended log; nothing numerically wrong.
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered")
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

    cfg = get_config()
    t_start = time.time()
    results = {"lever": "shrinkage", "meta": {
        "config": cfg["config"], "smoke": cfg["smoke"], "seed": cfg["seed"],
        "ridge": cfg["ridge"], "min_triple": cfg["min_triple"], "w_grid": cfg["w_grid"],
        "fast_lags": cfg["fast_lags"], "slow_lags": cfg["slow_lags"],
        "stab_lag": cfg["stab_lag"], "n_splits": cfg["n_splits"], "n_boot": cfg["n_boot"],
        "throttle": cfg["throttle"], "top_frac": cfg["top_frac"],
        "declared_unused_env": {"NL_NSURR": cfg["n_surr"], "NL_NPERM": cfg["n_perm"],
                                "note": "no surrogate/permutation test in the shrinkage design; "
                                        "inference is a worm-bootstrap CI on dBC."},
        "design": "shrink ACMMA gain toward ganglion-block mean; tune w by split-half stability; "
                  "compare raw vs best-shrunk on stability, AUROC (np/ma), and peptidergic dBC.",
    }}

    try:
        from sid_elegans.stability import split_half_stability
        from sid_elegans.biolag.references import build_registry

        X, names, fps = common.get_data(cfg["config"])
        N = len(names)
        P, blocks = block_indicator(names)
        results["meta"]["n_worms"] = len(X)
        results["meta"]["n_neurons"] = N
        results["meta"]["fps"] = fps
        results["meta"]["n_ganglion_blocks"] = len(blocks)
        results["meta"]["block_sizes"] = {b: int(P[k].sum()) for k, b in enumerate(blocks)}

        # references (peptidergic + monoaminergic class layers) on the 28w support
        refs = build_registry(names, min_specific_edges=10, verbose=False)
        byname = {r.name: r for r in refs}
        ref_np = byname.get("neuropeptide:all")
        ref_ma = byname.get("monoamine:all")
        results["meta"]["ref_edges"] = {
            "neuropeptide:all": int(ref_np.n_edges) if ref_np is not None else None,
            "monoamine:all": int(ref_ma.n_edges) if ref_ma is not None else None,
        }

        # ---------------------------------------------------------------------------------
        # (i) split-half stability: tune w. Raw fits are memoised, so the w-grid reuses them.
        # ---------------------------------------------------------------------------------
        def fit_matrix_for(w):
            def fm(sub):
                _, g = fit_channels(sub, cfg["stab_lag"], names, cfg["ridge"], cfg["min_triple"])
                return shrink_gain(g, P, w)
            return fm

        stab_table = []
        for w in cfg["w_grid"]:
            st = split_half_stability(fit_matrix_for(w), X, n_splits=cfg["n_splits"],
                                      seed=cfg["seed"], top_frac=cfg["top_frac"])
            stab_table.append({"w": float(w), **st})
        # best w = max split-half Spearman (reproducibility objective, no label peeking)
        best = max(stab_table, key=lambda r: r["spearman_mean"])
        best_w = float(best["w"])
        raw_row = next(r for r in stab_table if r["w"] == 0.0)
        # weight used for the raw-vs-shrunk AUROC / dBC comparisons: the stability-best if it is
        # nonzero, else a forced moderate probe so the comparison is not a trivial w=0 vs w=0.
        used_probe = best_w <= 0.0
        eval_w = float(cfg["probe_w"]) if used_probe else best_w
        results["stability"] = {
            "table": stab_table, "best_w": best_w,
            "eval_w": eval_w, "eval_w_is_stability_best": (not used_probe),
            "raw_spearman_mean": raw_row["spearman_mean"], "raw_jaccard_mean": raw_row["jaccard_mean"],
            "best_spearman_mean": best["spearman_mean"], "best_jaccard_mean": best["jaccard_mean"],
            "spearman_gain": float(best["spearman_mean"] - raw_row["spearman_mean"]),
            "jaccard_gain": float(best["jaccard_mean"] - raw_row["jaccard_mean"]),
            "shrinkage_helps_stability": bool(best_w > 0.0),
        }

        # ---------------------------------------------------------------------------------
        # (ii) AUROC vs neuropeptide:all / monoamine:all, raw vs best-shrunk (full 28w fits;
        #      also fills the memo for the full set so the dBC obs is a cache hit).
        # ---------------------------------------------------------------------------------
        auroc_out = {}
        for tag, ref in (("neuropeptide:all", ref_np), ("monoamine:all", ref_ma)):
            if ref is None:
                auroc_out[tag] = {"error": "reference missing"}
                continue
            per_lag = {"raw_gain": {}, "shrunk_gain": {}, "mean": {}}
            for L in cfg["fast_lags"] + cfg["slow_lags"]:
                m, g = fit_channels(X, L, names, cfg["ridge"], cfg["min_triple"])
                gs = shrink_gain(g, P, eval_w)
                per_lag["raw_gain"][L] = _auroc(g, ref)
                per_lag["shrunk_gain"][L] = _auroc(gs, ref)
                per_lag["mean"][L] = _auroc(m, ref)

            def _slowmean(d):
                return float(np.nanmean([d[L] for L in cfg["slow_lags"]]))

            def _maxlag(d):
                return float(np.nanmax([d[L] for L in d]))
            auroc_out[tag] = {
                "per_lag": {k: {int(L): float(v) for L, v in d.items()} for k, d in per_lag.items()},
                "raw_gain_slow_mean": _slowmean(per_lag["raw_gain"]),
                "shrunk_gain_slow_mean": _slowmean(per_lag["shrunk_gain"]),
                "delta_slow_mean": _slowmean(per_lag["shrunk_gain"]) - _slowmean(per_lag["raw_gain"]),
                "raw_gain_max": _maxlag(per_lag["raw_gain"]),
                "shrunk_gain_max": _maxlag(per_lag["shrunk_gain"]),
                "delta_max": _maxlag(per_lag["shrunk_gain"]) - _maxlag(per_lag["raw_gain"]),
                "mean_slow_mean": _slowmean(per_lag["mean"]),
                "eval_w": eval_w,
            }
        results["auroc"] = auroc_out

        # ---------------------------------------------------------------------------------
        # (iii) peptidergic dBC = BC_gain - BC_mean (slow band), worm-bootstrap CI, raw vs best.
        #       Two bootstraps with the SAME seed -> identical worm resamples -> the second
        #       (shrunk) call hits the memo the first (raw) call filled (correctness never
        #       depends on the cache -- a miss just recomputes).
        # ---------------------------------------------------------------------------------
        if ref_np is not None:
            def stat_raw(sub):
                return dbc_stat(sub, 0.0, ref_np, names, cfg, P)

            def stat_shrunk(sub):
                return dbc_stat(sub, eval_w, ref_np, names, cfg, P)

            boot_raw = harness.worm_bootstrap(
                stat_raw, X, n_boot=cfg["n_boot"],
                rng=np.random.default_rng(cfg["seed"]), throttle=cfg["throttle"])
            boot_shrunk = harness.worm_bootstrap(
                stat_shrunk, X, n_boot=cfg["n_boot"],
                rng=np.random.default_rng(cfg["seed"]), throttle=cfg["throttle"])
            v_raw, v_shr = _verdict(boot_raw), _verdict(boot_shrunk)
            results["dbc"] = {
                "reference": "neuropeptide:all", "channel_contrast": "BC_gain - BC_mean",
                "band": "slow", "best_w": best_w, "eval_w": eval_w,
                "eval_w_is_stability_best": (not used_probe),
                "raw": boot_raw, "shrunk": boot_shrunk,
                "verdict_raw": v_raw, "verdict_shrunk": v_shr,
                "verdict_changed": bool(v_raw != v_shr),
                "obs_delta": float(boot_shrunk["obs"] - boot_raw["obs"])
                if np.isfinite(boot_shrunk["obs"]) and np.isfinite(boot_raw["obs"]) else None,
            }
        else:
            results["dbc"] = {"error": "neuropeptide:all reference missing"}

        results["status"] = "ok"
    except Exception as e:  # emit partial JSON on any failure (HARD RULE #4 robustness)
        import traceback
        results["status"] = "error"
        results["error"] = repr(e)
        results["traceback"] = traceback.format_exc()

    # clear the (potentially large) fit memo before returning
    _FIT_MEMO.clear()
    results["meta"]["runtime_s"] = round(time.time() - t_start, 2)
    path = common.save_json("shrinkage", results)

    # ---- concise human summary -------------------------------------------------------------
    print("\n==================== LEVER #7: SHRINKAGE (ganglion-class prior) ====================")
    print(f"status={results['status']}  config={cfg['config']}  smoke={cfg['smoke']}  "
          f"runtime={results['meta']['runtime_s']}s  -> {path}")
    if results["status"] == "ok":
        s = results["stability"]
        print(f"\n(i) STABILITY (split-half Spearman @ lag {cfg['stab_lag']}f):  best_w={s['best_w']}"
              f"   eval_w={s['eval_w']}{'' if s['eval_w_is_stability_best'] else ' (forced probe; stability rejected shrinkage)'}")
        for row in s["table"]:
            print(f"    w={row['w']:.2f}  spearman={row['spearman_mean']:.4f}"
                  f"+/-{row['spearman_sd']:.4f}  jaccard={row['jaccard_mean']:.4f}")
        print(f"    raw->best  d(spearman)={s['spearman_gain']:+.4f}  d(jaccard)={s['jaccard_gain']:+.4f}"
              f"   {'UP (shrinkage helps)' if s['spearman_gain'] > 0 else 'NOT up (shrinkage does not help)'}")
        print(f"\n(ii) AUROC gain vs modulator layers  (raw -> shrunk@w={s['eval_w']}, slow-band mean):")
        for tag, a in results["auroc"].items():
            if "error" in a:
                print(f"    {tag}: {a['error']}"); continue
            print(f"    {tag:16s}  {a['raw_gain_slow_mean']:.4f} -> {a['shrunk_gain_slow_mean']:.4f}"
                  f"  (delta={a['delta_slow_mean']:+.4f}; max delta={a['delta_max']:+.4f})")
        d = results.get("dbc", {})
        if "error" not in d:
            print(f"\n(iii) PEPTIDERGIC dBC = BC_gain - BC_mean (slow band), worm-bootstrap CI:")
            print(f"    RAW    obs={d['raw']['obs']:+.4f}  CI=[{d['raw']['lo']:+.4f},{d['raw']['hi']:+.4f}]"
                  f"  n={d['raw']['n_boot']}  verdict={d['verdict_raw']}")
            print(f"    SHRUNK obs={d['shrunk']['obs']:+.4f}  CI=[{d['shrunk']['lo']:+.4f},"
                  f"{d['shrunk']['hi']:+.4f}]  n={d['shrunk']['n_boot']}  verdict={d['verdict_shrunk']}")
            print(f"    verdict_changed={d['verdict_changed']}  obs_delta={d['obs_delta']}")
    else:
        print("ERROR:", results.get("error"))
    print("===================================================================================\n")
    return results


if __name__ == "__main__":
    main()
