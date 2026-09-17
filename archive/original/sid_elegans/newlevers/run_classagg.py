r"""Lever #4 — CLASS / GANGLION AGGREGATION.

Question
--------
The distributional (gain/tail) channels of the SID estimator are recovered at the level of
individual (already class-level) neurons, but the modulator ground-truth connectomes are only
meaningfully resolved at a coarser anatomical granularity. Does *coarsening the estimate to the
granularity at which the ground truth actually lives* buy any statistical power — specifically for
the gain channel that theory says should carry the neuromodulator signal?

Design
------
Two aggregation levels (plus the un-pooled neuron level as the baseline):
  (a) ``coarse_class_labels`` — digit-strip. HONESTLY a near-no-op on this data: the neurons are
      already class-level, so this merges only a handful of numbered members (we report the exact
      count: N -> C_coarse). Kept in the comparison precisely to show it changes almost nothing.
  (b) ``ganglion_labels`` — the genuine ~8-14-way coarsening (the meaningful test).

For each level we build the row-normalized mean-pool operator ``G`` (``pooling_operator``) and, for
every channel {mean, gain, tail} + the two linear baselines {pearson_lag, ridge_var}, pool the
neuron-level [N,N] |coupling| to [C,C] (``pool_matrix(|M|, G)``) and score AUROC against the SAME
ground-truth adjacency pooled by the SAME ``G`` (binary: any edge present between the two classes).
Targets: ``structural:chemical`` (fast/wired control), ``monoamine:all`` and ``neuropeptide:all``
(the slow, distributional families gain/tail are supposed to prefer). AUROC is scanned over the lag
grid; we compare pooled AUROC to the neuron-level AUROC per channel/target/lag.

We then ask whether pooling improves *reproducibility*: split-half stability (Spearman + top-k
Jaccard of the off-diagonal |coupling| across disjoint worm halves) at neuron vs coarse vs ganglion
level, for each estimator channel, at one representative (slow-band) lag.

Two controls guard the headline (a reviewer confirmed the raw ganglion lift is partly a pooling
artifact AND not gain-specific):
  1. NEGATIVE CONTROL — ganglion pooling raises AUROC even for a RANDOM |M| matrix (~+0.03). For
     each target we pool ``N_RAND`` iid-random matrices the SAME way and build the ganglion-minus-
     neuron lift null; the real gain lift is reported MINUS the null mean with a permutation p, and
     only counts if it BEATS the null.
  2. CHANNEL-CONTRAST estimand — the headline is a worm-bootstrapped CONTRAST of ganglion lifts at
     the gain's best lag: (gain lift − mean lift) and (gain lift − best-linear-baseline lift). A
     GAIN-SPECIFIC benefit requires BOTH the random-null gate AND these contrast CIs to exclude 0;
     otherwise the honest verdict is that pooling raises AUROC for ALL channels ~equally.

Everything heavy is under ``__main__`` and sized by env vars (SMOKE / NL_*). All inference goes
through ``harness`` (worm_bootstrap) and ``stability`` (split-half); nothing is reinvented.
Convention: X_list = per-worm [T,N] (globally standardized), matrices [post,pre], fps=4.0.
"""
from __future__ import annotations

import os
import time
import traceback

import numpy as np

from sid_elegans.newlevers import common as cm
from sid_elegans.newlevers import harness
from sid_elegans.baselines import pearson_lag, ridge_var
from sid_elegans.biolag import config as C
from sid_elegans.biolag.references import build_registry
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.stability import _agreement


# --------------------------------------------------------------------------------------
# env / knobs  (all heavy counts come from here — never hardcode a big loop)
# --------------------------------------------------------------------------------------
SMOKE = os.environ.get("NEWLEVERS_SMOKE", "0") == "1"
SEED = cm.env_int("NL_SEED", 0)
NBOOT = cm.env_int("NL_NBOOT", 200)
NRAND = cm.env_int("NL_NRAND", 200)          # random-pooling negative-control draws (fix #1)
THROTTLE = cm.env_float("NL_THROTTLE", 0.3)
PRIMARY = os.environ.get("NL_CONFIG", "6w_clean_deconv")

# The three registry targets pooled to the ground-truth granularity.
TARGET_KEYS = ["structural:chemical", "monoamine:all", "neuropeptide:all"]
# Estimator channels + linear baselines that we pool + score.
EST_CHANNELS = ["mean", "gain", "tail"]
BASE_CHANNELS = ["pearson", "ridge"]
ALL_CHANNELS = EST_CHANNELS + BASE_CHANNELS
# Channels we compute split-half stability for (estimator channels only — baselines are not the
# object of study and refitting them adds nothing to the reproducibility question).
STAB_CHANNELS = ["mean", "gain", "tail"]

if SMOKE:
    CONFIGS = [PRIMARY]
    LAGS = [1, 8, 30]                       # tiny lag grid spanning fast..slow
    N_BOOT = min(NBOOT, 5)
    N_RAND = min(NRAND, 20)                  # smoke: tiny random-pooling null
    N_SPLITS = 2
    STAB_LAG = 8
else:
    CONFIGS = [PRIMARY] + [c for c in ["6w_clean_deconv", "28w_deconv"] if c != PRIMARY]
    LAGS = list(C.LAG_FRAMES)               # [1,2,3,5,8,10,15,20,30,40]
    N_BOOT = NBOOT
    N_RAND = NRAND
    N_SPLITS = 5
    STAB_LAG = 15                           # 3.75 s — inside the slow band (2.5-10 s)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def pool_abs(M, G):
    """Pool the |coupling| of a [N,N] matrix to [C,C] (NaN->0 first, diag zeroed by pool_matrix).

    For the neuron level ``G`` is the identity, so this returns |M| with a zeroed diagonal — i.e.
    exactly the un-pooled neuron-level scoring surface, giving a uniform interface across levels.
    """
    A = np.abs(np.nan_to_num(np.asarray(M, float), nan=0.0))
    return cm.pool_matrix(A, G)


def pooled_target(A, G):
    """Pool a binary [N,N] [post,pre] adjacency by ``G`` and re-binarize: 1 iff ANY neuron edge
    exists between the two classes (off-diagonal; self-class zeroed by pool_matrix)."""
    P = cm.pool_matrix(np.asarray(A, float), G)
    return (P > 0).astype(float)


def best_of(curve, lags):
    """(best_lag, best_auroc) ignoring NaN; (None, NaN) if the whole curve is NaN."""
    arr = np.asarray(curve, float)
    if not np.isfinite(arr).any():
        return None, float("nan")
    i = int(np.nanargmax(arr))
    return int(lags[i]), float(arr[i])


def _summ(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std()), "n": int(v.size)}


def _ci(obs, vals):
    """Percentile CI of a bootstrap sample (drops NaN); pairs it with the observed value."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return {"obs": float(obs), "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "sd": float("nan"), "n_boot": 0}
    return {"obs": float(obs), "mean": float(v.mean()),
            "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
            "sd": float(v.std()), "n_boot": int(v.size)}


def random_null_lifts(tgt_g, tgt_n, G_gang, G_id, N, n_rand, rng):
    """Negative control (fix #1): pool ``n_rand`` iid-random |M| matrices EXACTLY like a real
    channel (``pool_abs`` at ganglion vs neuron level) and record the ganglion-minus-neuron AUROC
    lift. This is the pooling artifact — pooling raises AUROC even for a random matrix — that any
    real lift must beat. Returns the null lift array (one per finite draw)."""
    lifts = []
    for _ in range(n_rand):
        R = rng.standard_normal((N, N))     # abs taken inside pool_abs; sign irrelevant
        a_g = harness.auroc_at_lag(pool_abs(R, G_gang), tgt_g, mask=None)
        a_n = harness.auroc_at_lag(pool_abs(R, G_id), tgt_n, mask=None)
        if np.isfinite(a_g) and np.isfinite(a_n):
            lifts.append(a_g - a_n)
    return np.asarray(lifts, float)


# --------------------------------------------------------------------------------------
# per-config analysis
# --------------------------------------------------------------------------------------
def analyze_config(cfg, is_primary, rng):
    X, names, fps = cm.get_data(cfg)
    N = len(names)

    # ---- aggregation levels (rebuilt per config: N and names differ 6w vs 28w) ----
    labels_coarse = cm.coarse_class_labels(names)
    G_coarse, cls_coarse = cm.pooling_operator(names, labels_coarse)
    labels_gang = cm.ganglion_labels(names)
    G_gang, cls_gang = cm.pooling_operator(names, labels_gang)
    G_id = np.eye(N)
    levels = {
        "neuron": (G_id, list(names)),
        "coarse_class": (G_coarse, cls_coarse),
        "ganglion": (G_gang, cls_gang),
    }
    # how much does each pooling actually merge (honesty about the coarse "no-op")
    n_merged_coarse = N - len(cls_coarse)
    n_merged_gang = N - len(cls_gang)

    # ---- ground-truth targets pooled to each level ----
    refs = build_registry(names, verbose=False)
    ref_by = {r.name: r for r in refs}
    targets = {k: ref_by[k].adjacency for k in TARGET_KEYS if k in ref_by}
    missing = [k for k in TARGET_KEYS if k not in ref_by]
    pooled_tgt = {lvl: {k: pooled_target(A, G) for k, A in targets.items()}
                  for lvl, (G, _) in levels.items()}
    tgt_pos = {lvl: {k: int((pooled_tgt[lvl][k] > 0).sum()) for k in targets}
               for lvl in levels}

    # ---- estimator channels (one fit per lag) + linear baselines per lag ----
    fits = harness.fit_all_lags(X, names, lags=LAGS, channels=tuple(EST_CHANNELS))
    chan_mats = {}
    for L in LAGS:
        d = dict(fits[L])
        d["pearson"] = pearson_lag(X, L)
        d["ridge"] = ridge_var(X, L, ridge=1.0)
        chan_mats[L] = d

    # ---- AUROC grid: target x level x channel over lags ----
    auroc = {}
    for tkey in targets:
        auroc[tkey] = {}
        for lvl, (G, _) in levels.items():
            tgt = pooled_tgt[lvl][tkey]
            auroc[tkey][lvl] = {}
            for ch in ALL_CHANNELS:
                curve = [harness.auroc_at_lag(pool_abs(chan_mats[L][ch], G), tgt, mask=None)
                         for L in LAGS]
                bl, ba = best_of(curve, LAGS)
                auroc[tkey][lvl][ch] = {"lags": list(LAGS), "auroc": curve,
                                        "best_lag": bl, "best_auroc": ba}

    # ---- granularity effect: pooled best AUROC minus neuron best AUROC, per channel/target ----
    gran = {}
    for tkey in targets:
        gran[tkey] = {}
        for ch in ALL_CHANNELS:
            nb = auroc[tkey]["neuron"][ch]["best_auroc"]
            cb = auroc[tkey]["coarse_class"][ch]["best_auroc"]
            gb = auroc[tkey]["ganglion"][ch]["best_auroc"]
            gran[tkey][ch] = {
                "neuron_best": nb, "coarse_best": cb, "ganglion_best": gb,
                "coarse_minus_neuron": cb - nb, "ganglion_minus_neuron": gb - nb,
            }

    # ---- FIX #1 negative control: random-pooling null for the ganglion lift, per target -------
    # The real per-channel lift is measured at the SAME (ganglion-best) lag for BOTH pool levels,
    # so it is a single matrix pooled two ways — matched to a single random matrix pooled two ways.
    # corrected_lift = real_lift - null_mean; a lift only counts if it BEATS the random null
    # (permutation-style p = (#null >= real +1)/(n+1)).
    nrng = np.random.default_rng(SEED + 999)
    random_null = {}
    for tkey in targets:
        tgt_g = pooled_tgt["ganglion"][tkey]
        tgt_n = pooled_tgt["neuron"][tkey]
        null = random_null_lifts(tgt_g, tgt_n, G_gang, G_id, N, N_RAND, nrng)
        null_mean = float(np.mean(null)) if null.size else float("nan")
        rn = {"n_rand": int(null.size), "null_mean": null_mean,
              "null_sd": float(np.std(null)) if null.size else float("nan"),
              "null_p5": float(np.percentile(null, 5)) if null.size else float("nan"),
              "null_p95": float(np.percentile(null, 95)) if null.size else float("nan")}
        for ch in ALL_CHANNELS:
            Lb = auroc[tkey]["ganglion"][ch]["best_lag"]
            if Lb is None:
                real = float("nan")
            else:
                M = chan_mats[Lb][ch]
                ag = harness.auroc_at_lag(pool_abs(M, G_gang), tgt_g, mask=None)
                an = harness.auroc_at_lag(pool_abs(M, G_id), tgt_n, mask=None)
                real = ag - an
            if null.size and np.isfinite(real):
                p = float((np.sum(null >= real) + 1) / (null.size + 1))
                corrected = real - null_mean
            else:
                p, corrected = float("nan"), float("nan")
            rn[ch] = {"real_lift": float(real), "corrected_lift": float(corrected),
                      "p": p, "lag": (int(Lb) if Lb is not None else None),
                      "beats_null": bool(np.isfinite(p) and p < 0.05 and corrected > 0)}
        random_null[tkey] = rn

    # ---- split-half stability: neuron vs coarse vs ganglion, per estimator channel ----
    # One split -> two fits reused across ALL levels & channels (cheap; no redundant refits).
    W = len(X)
    stab_acc = {lvl: {ch: {"rho": [], "jac": []} for ch in STAB_CHANNELS} for lvl in levels}
    srng = np.random.default_rng(SEED + 12345)
    n_splits_eff = N_SPLITS if W // 2 >= 1 else 0
    for _ in range(n_splits_eff):
        perm = srng.permutation(W)
        h = W // 2
        if h < 1:
            break
        idxA, idxB = perm[:h], perm[h:2 * h]
        resA = fit_distributional_connectome([X[i] for i in idxA], names, lag=STAB_LAG)
        resB = fit_distributional_connectome([X[i] for i in idxB], names, lag=STAB_LAG)
        for ch in STAB_CHANNELS:
            MA = harness.channel_matrix(resA, ch)
            MB = harness.channel_matrix(resB, ch)
            for lvl, (G, _) in levels.items():
                rho, jac = _agreement(pool_abs(MA, G), pool_abs(MB, G))
                stab_acc[lvl][ch]["rho"].append(rho)
                stab_acc[lvl][ch]["jac"].append(jac)
        if THROTTLE:
            time.sleep(THROTTLE)
    stability = {lvl: {ch: {"spearman": _summ(stab_acc[lvl][ch]["rho"]),
                            "jaccard": _summ(stab_acc[lvl][ch]["jac"]),
                            "lag": STAB_LAG}
                       for ch in STAB_CHANNELS}
                 for lvl in levels}

    # ---- FIX #2 headline = CHANNEL-CONTRAST estimand, worm-bootstrapped ---------------------
    # The ganglion lift is NOT gain-specific (mean/baselines get the same pooling boost), so the
    # headline is now a CONTRAST of lifts at the gain's best ganglion lag, all channels sharing the
    # one fit per resample: lift(ch) = ganglion-AUROC(ch) - neuron-AUROC(ch).
    #   gain_minus_mean     = lift(gain) - lift(mean)
    #   gain_minus_baseline = lift(gain) - max(lift(pearson), lift(ridge))
    # A gain-specific benefit requires these contrast CIs to EXCLUDE 0 (not just gain lift > 0).
    boot = None
    npt = "neuropeptide:all"
    if is_primary and npt in targets:
        blag = auroc[npt]["ganglion"]["gain"]["best_lag"]
        if blag is None:
            blag = STAB_LAG
        tgt_g = pooled_tgt["ganglion"][npt]
        tgt_n = pooled_tgt["neuron"][npt]

        def _lift(M):
            a_g = harness.auroc_at_lag(pool_abs(M, G_gang), tgt_g, mask=None)
            a_n = harness.auroc_at_lag(pool_abs(M, G_id), tgt_n, mask=None)
            return a_g - a_n

        def lifts_of(Xsub):
            res = fit_distributional_connectome(Xsub, names, lag=blag)
            gl = _lift(harness.channel_matrix(res, "gain"))
            ml = _lift(harness.channel_matrix(res, "mean"))
            pl = _lift(pearson_lag(Xsub, blag))
            rl = _lift(ridge_var(Xsub, blag, ridge=1.0))
            bl = np.nanmax([pl, rl]) if np.isfinite([pl, rl]).any() else float("nan")
            return {"gain": gl, "mean": ml, "pearson": pl, "ridge": rl, "best_baseline": bl,
                    "gain_minus_mean": gl - ml, "gain_minus_baseline": gl - bl}

        keys = ["gain", "mean", "gain_minus_mean", "gain_minus_baseline"]
        obs = lifts_of(X)
        acc = {k: [] for k in keys}
        brng = np.random.default_rng(SEED + 777)
        W_ = len(X)
        for _ in range(N_BOOT):
            idx = brng.integers(0, W_, W_)
            d = lifts_of([X[i] for i in idx])
            for k in keys:
                if np.isfinite(d[k]):
                    acc[k].append(d[k])
            if THROTTLE:
                time.sleep(THROTTLE)
        boot = {"channel": "gain", "target": npt, "lag": int(blag),
                "statistic": "channel_contrast_of_ganglion_lifts",
                "obs_lifts": {k: float(obs[k]) for k in
                              ["gain", "mean", "pearson", "ridge", "best_baseline"]},
                "gain_lift": _ci(obs["gain"], acc["gain"]),
                "mean_lift": _ci(obs["mean"], acc["mean"]),
                "gain_minus_mean": _ci(obs["gain_minus_mean"], acc["gain_minus_mean"]),
                "gain_minus_baseline": _ci(obs["gain_minus_baseline"], acc["gain_minus_baseline"])}

    return {
        "n_worms": int(W), "N": int(N), "fps": float(fps),
        "n_coarse_classes": len(cls_coarse), "n_ganglia": len(cls_gang),
        "n_merged_coarse": int(n_merged_coarse), "n_merged_ganglion": int(n_merged_gang),
        "coarse_note": (f"coarse_class merges {n_merged_coarse}/{N} neurons "
                        f"({N}->{len(cls_coarse)}); a near-no-op — the data is already class-level. "
                        f"ganglion is the meaningful coarsening ({N}->{len(cls_gang)})."),
        "target_positive_edges": tgt_pos,
        "missing_targets": missing,
        "auroc": auroc,
        "granularity_effect": gran,
        "random_null": random_null,
        "stability": stability,
        "bootstrap_headline": boot,
    }


# --------------------------------------------------------------------------------------
# verdict synthesis (primary config)
# --------------------------------------------------------------------------------------
def make_verdict(results):
    prim = results.get(PRIMARY)
    if not isinstance(prim, dict) or "granularity_effect" not in prim:
        return "no usable primary-config result (see errors)."
    npt = "neuropeptide:all"

    # ---- FIX #1 gate: does the gain lift BEAT the random-pooling null? (neuropeptide target) ----
    rn = prim.get("random_null", {}).get(npt, {})
    rn_gain = rn.get("gain", {})
    p_gain = rn_gain.get("p", float("nan"))
    corr_gain = rn_gain.get("corrected_lift", float("nan"))
    real_gain = rn_gain.get("real_lift", float("nan"))
    null_mean = rn.get("null_mean", float("nan"))
    beats_null = bool(np.isfinite(p_gain) and p_gain < 0.05 and np.isfinite(corr_gain)
                      and corr_gain > 0)

    # ---- FIX #2 gate: is the lift GAIN-SPECIFIC? (contrast CIs exclude 0) ----
    boot = prim.get("bootstrap_headline") or {}
    gm = boot.get("gain_minus_mean", {})
    gb = boot.get("gain_minus_baseline", {})
    gm_lo, gm_hi = gm.get("lo", float("nan")), gm.get("hi", float("nan"))
    gb_lo, gb_hi = gb.get("lo", float("nan")), gb.get("hi", float("nan"))
    beats_mean = bool(np.isfinite(gm_lo) and gm_lo > 0)
    beats_base = bool(np.isfinite(gb_lo) and gb_lo > 0)
    gain_specific = beats_mean and beats_base

    if beats_null and gain_specific:
        verdict = "HELPS (GAIN-SPECIFIC)"
    elif beats_null or gain_specific:
        verdict = "MIXED"
    else:
        verdict = "POOLING ARTIFACT — no gain-specific benefit (aggregation raises AUROC ~equally)"

    return (
        f"[{PRIMARY}] ganglion pooling of a RANDOM matrix already lifts AUROC by "
        f"null_mean {null_mean:+.3f}; real gain lift {real_gain:+.3f} => corrected "
        f"{corr_gain:+.3f} (p={p_gain:.3f}, {'BEATS' if beats_null else 'does NOT beat'} null). "
        f"Gain-specificity contrasts (worm-boot): gain-minus-mean CI [{gm_lo:+.3f},{gm_hi:+.3f}] "
        f"({'excl' if beats_mean else 'incl'} 0); gain-minus-baseline CI "
        f"[{gb_lo:+.3f},{gb_hi:+.3f}] ({'excl' if beats_base else 'incl'} 0). "
        f"coarse-class is a near-no-op (merges {prim['n_merged_coarse']} neurons). "
        f"=> {verdict}."
    )


# --------------------------------------------------------------------------------------
# entry point (all heavy work here)
# --------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    results = {}
    for cfg in CONFIGS:
        is_primary = (cfg == PRIMARY)
        try:
            results[cfg] = analyze_config(cfg, is_primary, rng)
        except Exception as e:                       # emit JSON even on partial failure
            results[cfg] = {"error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()}
            print(f"[classagg] CONFIG {cfg} FAILED: {e}")

    verdict = make_verdict(results)
    out = {
        "lever": "classagg",
        "smoke": SMOKE,
        "seed": SEED,
        "configs": CONFIGS,
        "primary_config": PRIMARY,
        "lags": list(LAGS),
        "channels": ALL_CHANNELS,
        "targets": TARGET_KEYS,
        "stability_lag": STAB_LAG,
        "n_splits": N_SPLITS,
        "n_boot": N_BOOT,
        "n_rand": N_RAND,
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
        "verdict": verdict,
        "caveat": ("pooled and neuron-level AUROCs are computed over different edge populations "
                   "(class-pairs vs neuron-pairs), so absolute values are not strictly comparable; "
                   "read the WITHIN-target neuron->ganglion direction and the bootstrap CI, not the "
                   "raw magnitudes."),
    }
    path = cm.save_json("classagg", out)

    # ---- concise human summary ----
    print("\n" + "=" * 78)
    print(f"LEVER classagg  (smoke={SMOKE})  elapsed={out['elapsed_sec']}s  -> {path}")
    for cfg in CONFIGS:
        r = results[cfg]
        if "error" in r:
            print(f"  [{cfg}] ERROR: {r['error']}")
            continue
        print(f"  [{cfg}] worms={r['n_worms']} N={r['N']} "
              f"coarse={r['n_coarse_classes']}(merged {r['n_merged_coarse']}) "
              f"ganglia={r['n_ganglia']}")
        rn_all = r.get("random_null", {})
        for tkey in r["granularity_effect"]:
            g = r["granularity_effect"][tkey]["gain"]
            print(f"      {tkey:22s} gain best-AUROC  "
                  f"neuron={g['neuron_best']:.3f}  coarse={g['coarse_best']:.3f}  "
                  f"ganglion={g['ganglion_best']:.3f}  (gangl-neuron {g['ganglion_minus_neuron']:+.3f})")
            rn = rn_all.get(tkey, {})
            rg = rn.get("gain", {})
            if rn:
                print(f"        random-null lift={rn['null_mean']:+.3f}  "
                      f"gain real={rg.get('real_lift', float('nan')):+.3f} "
                      f"corrected={rg.get('corrected_lift', float('nan')):+.3f} "
                      f"p={rg.get('p', float('nan')):.3f} "
                      f"{'BEATS' if rg.get('beats_null') else 'no-beat'} null")
        stg = r["stability"]
        print(f"      gain stability (Spearman) neuron={stg['neuron']['gain']['spearman']['mean']:.3f} "
              f"coarse={stg['coarse_class']['gain']['spearman']['mean']:.3f} "
              f"ganglion={stg['ganglion']['gain']['spearman']['mean']:.3f}")
        if r.get("bootstrap_headline"):
            b = r["bootstrap_headline"]
            ol = b["obs_lifts"]
            gm, gb = b["gain_minus_mean"], b["gain_minus_baseline"]
            print(f"      contrast headline (gain, {b['target']}, lag {b['lag']}): "
                  f"lifts gain={ol['gain']:+.3f} mean={ol['mean']:+.3f} "
                  f"baseline={ol['best_baseline']:+.3f}")
            print(f"        gain-minus-mean     obs={gm['obs']:+.3f} "
                  f"CI=[{gm['lo']:+.3f},{gm['hi']:+.3f}] n_boot={gm['n_boot']}")
            print(f"        gain-minus-baseline obs={gb['obs']:+.3f} "
                  f"CI=[{gb['lo']:+.3f},{gb['hi']:+.3f}] n_boot={gb['n_boot']}")
    print("  VERDICT: " + verdict)
    print("=" * 78)


if __name__ == "__main__":
    main()
