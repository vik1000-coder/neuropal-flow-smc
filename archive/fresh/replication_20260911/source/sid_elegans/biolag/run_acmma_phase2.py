r"""ACMMA band-concordance on ALL 28 worms (available-case, NO imputation) -- the decisive test of
whether using all the data (done right) makes the peptidergic gain-slow effect bootstrap-robust.

Same controlled pipeline as Phase 2 (primary + global-mode + surrogate p + worm-bootstrap CI +
source-variance control), but the estimator is ACMMA (available-case multivariate moment assembly)
at the stability-selected ridge. Compare to: imputed-28 (dBC boot CI INCLUDED 0) and clean-6w
(dBC +2.78, boot CI [+1.49,+5.27]).

Global-mode removal is NaN-aware (global-signal regression preserving the missingness mask) so the
available-case structure is respected. Gentle by default.
Run: PYTHONPATH=.:SBTG ./.venv/bin/python sid_elegans/biolag/run_acmma_phase2.py
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from sid_elegans.acmma import fit_acmma_connectome
from sid_elegans.biolag import config as C
from sid_elegans.biolag import metric as M
from sid_elegans.biolag.references import build_registry
from sid_elegans.combined_data import load_combined
from sid_elegans.lagres import metric as mt

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "sid_elegans" / "output" / "biolag"
LAGS = C.LAG_FRAMES
N_SURR = int(os.environ.get("ACMMA_NSURR", 25))
N_BOOT = int(os.environ.get("ACMMA_NBOOT", 25))
THROTTLE = float(os.environ.get("ACMMA_THROTTLE", 0.4))
# default = 0.01 (the standard ridge that reproduces the definitive result). ridge=3.0 was shown to
# be an OVER-REGULARIZATION artifact (sign flips; see ALLDATA_RESULT.md) -- do NOT use it as default.
RIDGE = float(os.environ.get("ACMMA_RIDGE", 0.01))
MIN_TRIPLE = 2000
SHRINK = 0.05
REFS = ["neuropeptide:all", "neuropeptide:pdf-1", "neuropeptide:ntc-1",
        "monoamine:serotonin", "structural:gap", "structural:chemical"]


def remove_global_mode_nan(X_list, n_pc=1):
    """Global-signal regression that PRESERVES the NaN missingness mask (for available-case)."""
    out = []
    for X in X_list:
        mask = np.isfinite(X)
        g = np.nanmean(np.where(mask, X, np.nan), axis=1)      # global signal (mean over present)
        g = np.nan_to_num(g, nan=0.0)
        R = np.array(X, dtype=float)
        gg = g @ g + 1e-9
        for k in range(X.shape[1]):
            col = X[:, k]; m = np.isfinite(col)
            if m.sum() < 5:
                continue
            beta = (g[m] @ col[m]) / (g[m] @ g[m] + 1e-9)
            R[m, k] = col[m] - beta * g[m]
        R[~mask] = np.nan
        out.append(R)
    return out


def acmma_curves(X, names, refs, mask):
    cm, cg = {r.name: [] for r in refs}, {r.name: [] for r in refs}
    for l in LAGS:
        m, _ = fit_acmma_connectome(X, names, lag=l, ridge=RIDGE, min_triple=MIN_TRIPLE, shrink=SHRINK)
        for r in refs:
            cm[r.name].append(mt.corr_score(m["mean"], r.adjacency, mask, "auroc"))
            cg[r.name].append(mt.corr_score(m["gain"], r.adjacency, mask, "auroc"))
    return {r.name: {"mean": np.array(cm[r.name]), "gain": np.array(cg[r.name])} for r in refs}


def bc_from(cur, refs, mu0, sd0):
    out = {}
    for r in refs:
        z = {ch: (cur[r.name][ch] - mu0[r.name][ch]) / sd0[r.name][ch] for ch in ("mean", "gain")}
        bc = {ch: M.band_concordance(z[ch], r.expected_band) for ch in z}
        out[r.name] = {"BC_gain": bc["gain"], "BC_mean": bc["mean"], "dBC": bc["gain"] - bc["mean"]}
    return out


def stat(x):
    x = np.array([v for v in x if np.isfinite(v)])
    return ({"mean": float(x.mean()), "lo": float(np.percentile(x, 2.5)),
             "hi": float(np.percentile(x, 97.5))} if x.size else
            {"mean": np.nan, "lo": np.nan, "hi": np.nan})


def run_block(X, names, refs, mask, tag, rng):
    obs = acmma_curves(X, names, refs, mask)
    surr = []
    for s in range(N_SURR):
        surr.append(acmma_curves(M.circshift(X, rng), names, refs, mask))
        if THROTTLE:
            time.sleep(THROTTLE)
        if (s + 1) % 5 == 0:
            print(f"  [{tag}] surrogate {s+1}/{N_SURR}", flush=True)
    mu0 = {r.name: {ch: np.nanmean([sc[r.name][ch] for sc in surr], 0) for ch in ("mean", "gain")} for r in refs}
    sd0 = {r.name: {ch: np.nanstd([sc[r.name][ch] for sc in surr], 0) + 1e-9 for ch in ("mean", "gain")} for r in refs}
    obc = bc_from(obs, refs, mu0, sd0)
    nbc = [bc_from(sc, refs, mu0, sd0) for sc in surr]
    boot = []
    for b in range(N_BOOT):
        idx = rng.integers(0, len(X), len(X)).tolist()
        boot.append(bc_from(acmma_curves([X[i] for i in idx], names, refs, mask), refs, mu0, sd0))
        if THROTTLE:
            time.sleep(THROTTLE)
        if (b + 1) % 5 == 0:
            print(f"  [{tag}] bootstrap {b+1}/{N_BOOT}", flush=True)
    res = {}
    for r in refs:
        nd = np.array([nb[r.name]["dBC"] for nb in nbc])
        ng = np.array([nb[r.name]["BC_gain"] for nb in nbc])
        res[r.name] = {**obc[r.name],
                       "p_dBC": float((np.sum(nd >= obc[r.name]["dBC"]) + 1) / (len(nd) + 1)),
                       "p_BCgain": float((np.sum(ng >= obc[r.name]["BC_gain"]) + 1) / (len(ng) + 1)),
                       "dBC_boot": stat([bb[r.name]["dBC"] for bb in boot]),
                       "BCgain_boot": stat([bb[r.name]["BC_gain"] for bb in boot])}
    prim = next(r for r in refs if r.name == "neuropeptide:all")
    res["_source_variance_BC"] = float(
        M.band_concordance(M.source_variance_curve(X, names, prim, mask), prim.expected_band))
    return res


def load_data(tag):
    if tag == "6w":
        return load_combined(coverage_frac=0.6, complete_case=True, signal="deconv", verbose=True)
    return load_combined(complete_case=False, min_worms_per_neuron=6, signal="deconv", verbose=True)


def main():
    configs = os.environ.get("ACMMA_DATA", "28w").split(",")
    print(f"[config] ridge={RIDGE} N_SURR={N_SURR} N_BOOT={N_BOOT} data={configs}", flush=True)
    results = {}
    for tag in configs:
        X, names, _ = load_data(tag)
        refs = [r for r in build_registry(names, 10) if r.name in REFS]
        mask = mt.eval_mask(len(names)); rng = np.random.default_rng(0)
        print(f"=== ACMMA {tag}: {len(X)} worms / {len(names)} neurons, ridge={RIDGE} ===", flush=True)
        primary = run_block(X, names, refs, mask, f"{tag}-prim", rng)
        results[tag] = {"n_worms": len(X), "n_neurons": len(names), "ridge": RIDGE, "primary": primary}
        json.dump(results, open(OUT / "acmma_phase2.json", "w"), indent=2, default=float)
        gmode = run_block(remove_global_mode_nan(X, 1), names, refs, mask, f"{tag}-gmode", rng)
        results[tag]["global_mode_removed"] = gmode
        json.dump(results, open(OUT / "acmma_phase2.json", "w"), indent=2, default=float)
        p = primary["neuropeptide:all"]; g = gmode["neuropeptide:all"]
        excl = "EXCLUDES 0" if (p['dBC_boot']['lo'] > 0 or p['dBC_boot']['hi'] < 0) else "includes 0"
        print(f"\n[ACMMA {tag} ridge={RIDGE}] neuropeptide dBC PRIMARY {p['dBC']:+.2f} "
              f"boot[{p['dBC_boot']['lo']:+.2f},{p['dBC_boot']['hi']:+.2f}] {excl} (p={p['p_dBC']:.3f})  "
              f"GLOBAL {g['dBC']:+.2f}  src-var={primary['_source_variance_BC']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
