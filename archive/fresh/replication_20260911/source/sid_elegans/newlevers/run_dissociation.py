r"""Metabotropic-vs-ionotropic DISSOCIATION test for the self-gain lead.

Neuromodulation is metabotropic (slow GPCR). If the self-gain readout captures it, then across
neurons the SLOW-band self-gain should track a neuron's METABOTROPIC receptor load and the
FAST-band self-gain should track its IONOTROPIC load — a double dissociation that no boring
confound (variance, firing rate, calcium kinetics) reproduces, because those are mechanism-blind.

For each config we compute, per neuron, slow-band and fast-band |self-gain| (the exposed
diagonal), and CeNGEN metabotropic / ionotropic receptor loads (receptors.py). We then measure
partial Spearman correlations (controlling for log-variance, the known confound, and the OTHER
mechanism's load) with a per-neuron permutation null:
  slow ~ metabotropic   (predicted POSITIVE)      slow ~ ionotropic   (predicted ~0)
  fast ~ ionotropic     (predicted POSITIVE)       fast ~ metabotropic (predicted ~0)
plus the neuropeptide-GPCR axis (the real lead) and within-transmitter serotonin/tyramine
contrasts (each has both a metabotropic and an ionotropic receptor).

Env: NL_NPERM (default 5000, smoke 500). Heavy work under __main__.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from scipy.stats import rankdata

from sid_elegans.newlevers import common as cm
from sid_elegans.newlevers import receptors as R

SMOKE = os.environ.get("NEWLEVERS_SMOKE", "0") == "1"
SEED = cm.env_int("NL_SEED", 0)
NPERM = 500 if SMOKE else cm.env_int("NL_NPERM", 5000)
SLOW = [10, 15, 20, 30, 40]        # 2.5-10 s
FAST = [1, 2, 3]                   # 0.25-0.75 s


def self_gain_bands(config):
    """Return (slow, fast, logvar) per-neuron vectors + names for a config.
    6w -> estimator diagonal; 28w -> ACMMA self_diag."""
    X, names, fps = cm.get_data(config)
    N = len(names)
    if config.startswith("28w"):
        from sid_elegans.acmma import fit_acmma_connectome
        def sg(lags):
            acc = np.zeros(N); n = 0
            for L in lags:
                _, diag = fit_acmma_connectome(X, names, lag=L, ridge=0.01, min_triple=50)
                acc += np.abs(diag["self_diag"]["gain"]); n += 1
            return acc / n
    else:
        from sid_elegans.estimator import fit_distributional_connectome
        def sg(lags):
            acc = np.zeros(N); n = 0
            for L in lags:
                res = fit_distributional_connectome(X, names, lag=L)
                acc += np.abs(res.diagonal["gain"]); n += 1
            return acc / n
    slow, fast = sg(SLOW), sg(FAST)
    logv = np.log(cm.source_variance(X) + 1e-12)
    return slow, fast, logv, names


def partial_spearman(a, b, ctrl):
    """Partial Spearman of a,b controlling for ctrl columns. Returns (rho, n)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    C = np.asarray(ctrl, float)
    if C.ndim == 1:
        C = C[:, None]
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(C).all(axis=1)
    if good.sum() < C.shape[1] + 5:
        return np.nan, int(good.sum())
    ra, rb = rankdata(a[good]), rankdata(b[good])
    Cr = np.column_stack([rankdata(C[good, k]) for k in range(C.shape[1])])
    D = np.column_stack([np.ones(good.sum()), Cr])
    ea = ra - D @ np.linalg.lstsq(D, ra, rcond=None)[0]
    eb = rb - D @ np.linalg.lstsq(D, rb, rcond=None)[0]
    if ea.std() < 1e-9 or eb.std() < 1e-9:
        return np.nan, int(good.sum())
    return float(np.corrcoef(ea, eb)[0, 1]), int(good.sum())


def partial_spearman_perm(a, b, ctrl, rng, nperm=NPERM):
    """Partial Spearman + two-sided permutation p (shuffle b among the valid neurons)."""
    obs, n = partial_spearman(a, b, ctrl)
    if not np.isfinite(obs):
        return {"rho": np.nan, "p": np.nan, "n": n}
    a = np.asarray(a, float); b = np.asarray(b, float); C = np.asarray(ctrl, float)
    if C.ndim == 1:
        C = C[:, None]
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(C).all(axis=1)
    ai, bi, Ci = a[good], b[good], C[good]
    cnt = 0
    for _ in range(nperm):
        r, _ = partial_spearman(ai, rng.permutation(bi), Ci)
        if np.isfinite(r) and abs(r) >= abs(obs):
            cnt += 1
    return {"rho": obs, "p": (cnt + 1) / (nperm + 1), "n": n}


def analyze(config, rng):
    slow, fast, logv, names = self_gain_bands(config)
    loads = R.mechanism_loads(names)
    metab, iono, npg = loads["metab_total"], loads["iono_total"], loads["np_gpcr"]

    def pc(y, x, extra):
        ctrl = np.column_stack([logv] + extra)
        return partial_spearman_perm(y, x, ctrl, np.random.default_rng(SEED + 1), NPERM)

    res = {
        # the double dissociation (control for logvar + the OTHER mechanism)
        "slow_vs_metab": pc(slow, metab, [iono]),
        "slow_vs_iono":  pc(slow, iono, [metab]),
        "fast_vs_iono":  pc(fast, iono, [metab]),
        "fast_vs_metab": pc(fast, metab, [iono]),
        # the peptidergic axis (the real lead) — slow self-gain vs neuropeptide-GPCR load
        "slow_vs_np_gpcr": pc(slow, npg, [iono]),
        # within-transmitter (each has both receptor kinds); control logvar only
        "sero_slow_vs_metab": partial_spearman_perm(slow, loads["metab_serotonin"],
                                                    np.column_stack([logv, loads["iono_serotonin"]]),
                                                    np.random.default_rng(SEED + 2), NPERM),
        "sero_slow_vs_iono": partial_spearman_perm(slow, loads["iono_serotonin"],
                                                   np.column_stack([logv, loads["metab_serotonin"]]),
                                                   np.random.default_rng(SEED + 3), NPERM),
        "tyr_slow_vs_metab": partial_spearman_perm(slow, loads["metab_tyramine"],
                                                   np.column_stack([logv, loads["iono_tyramine"]]),
                                                   np.random.default_rng(SEED + 4), NPERM),
        "tyr_slow_vs_iono": partial_spearman_perm(slow, loads["iono_tyramine"],
                                                  np.column_stack([logv, loads["metab_tyramine"]]),
                                                  np.random.default_rng(SEED + 5), NPERM),
    }
    # dissociation scores
    res["dissociation_slow"] = res["slow_vs_metab"]["rho"] - res["slow_vs_iono"]["rho"]
    res["dissociation_fast"] = res["fast_vs_iono"]["rho"] - res["fast_vs_metab"]["rho"]
    res["n_in_cengen"] = int(loads["in_cengen"].sum())
    return res


def verdict(all_res):
    r = all_res.get("6w_clean_deconv", {})
    sm = r.get("slow_vs_metab", {}); si = r.get("slow_vs_iono", {})
    npg = r.get("slow_vs_np_gpcr", {})
    dslow = r.get("dissociation_slow", float("nan"))
    metab_pos = np.isfinite(sm.get("rho", np.nan)) and sm["rho"] > 0 and sm.get("p", 1) < 0.05
    iono_null = (not np.isfinite(si.get("rho", np.nan))) or si.get("p", 1) >= 0.05 or si["rho"] <= sm.get("rho", 0)
    if metab_pos and iono_null and dslow > 0:
        v = ("DISSOCIATION HOLDS: slow self-gain tracks METABOTROPIC receptor load "
             f"(rho={sm['rho']:+.2f}, p={sm['p']:.3f}) but not ionotropic (rho={si.get('rho', float('nan')):+.2f}), "
             f"dissociation={dslow:+.2f} — the mechanism-specific fingerprint of neuromodulation. "
             "This corroborates the self-gain lead independently of the p=0.09 group test.")
    elif np.isfinite(sm.get("rho", np.nan)) and sm.get("rho", 0) > 0 and dslow > 0:
        v = (f"PARTIAL: slow self-gain leans toward metabotropic (rho={sm['rho']:+.2f}, p={sm['p']:.3f}) "
             f"over ionotropic (dissociation={dslow:+.2f}) but not cleanly significant — suggestive, "
             "consistent with the lead but not a decisive mechanistic corroboration.")
    else:
        v = (f"NO DISSOCIATION: slow self-gain does not track metabotropic load specifically "
             f"(rho={sm.get('rho', float('nan')):+.2f}, p={sm.get('p', float('nan')):.3f}, "
             f"dissociation={dslow:+.2f}) — the self-gain signal is not mechanism-specific, weakening "
             "the neuromodulation interpretation.")
    return v


if __name__ == "__main__":
    t0 = time.time()
    print(f"[dissoc] SMOKE={SMOKE} NPERM={NPERM} SLOW={SLOW} FAST={FAST}")
    rng = np.random.default_rng(SEED)
    configs = ["6w_clean_deconv"] if SMOKE else ["6w_clean_deconv", "28w_deconv"]
    all_res = {}
    for cfg in configs:
        print(f"[dissoc] === {cfg} ===", flush=True)
        try:
            all_res[cfg] = analyze(cfg, rng)
        except Exception as e:
            import traceback
            all_res[cfg] = {"error": f"{e}", "trace": traceback.format_exc()[-600:]}
            print(f"  FAILED: {e}")
    v = verdict(all_res)
    out = {"lever": "dissociation", "smoke": SMOKE, "nperm": NPERM, "slow_lags": SLOW,
           "fast_lags": FAST, "results": all_res, "verdict": v,
           "elapsed_sec": round(time.time() - t0, 1)}
    path = cm.save_json("dissociation", out)

    print("\n" + "=" * 76)
    print("METABOTROPIC vs IONOTROPIC DISSOCIATION  (self-gain lead)")
    print("=" * 76)
    for cfg, r in all_res.items():
        if "error" in r:
            print(f"[{cfg}] ERROR: {r['error']}"); continue
        print(f"[{cfg}]  ({r['n_in_cengen']} neurons in CeNGEN)")
        for k in ["slow_vs_metab", "slow_vs_iono", "fast_vs_iono", "fast_vs_metab",
                  "slow_vs_np_gpcr", "sero_slow_vs_metab", "sero_slow_vs_iono",
                  "tyr_slow_vs_metab", "tyr_slow_vs_iono"]:
            d = r.get(k, {})
            print(f"    {k:22s} rho={d.get('rho', float('nan')):+.3f}  p={d.get('p', float('nan')):.3f}  n={d.get('n', 0)}")
        print(f"    dissociation slow (metab-iono) = {r['dissociation_slow']:+.3f}  "
              f"fast (iono-metab) = {r['dissociation_fast']:+.3f}")
    print("\n  VERDICT: " + v)
    print("=" * 76)
    print(f"wrote {path}  ({out['elapsed_sec']}s)")
