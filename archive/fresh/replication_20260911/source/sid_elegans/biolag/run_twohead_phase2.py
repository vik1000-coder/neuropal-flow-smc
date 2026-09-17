r"""Run the null-contrast-tuned two-head (neural DSM + SID readout) through the biolag
band-concordance pipeline, INCLUDING the global-mode control -- to corroborate (or not) the
closed-form SID Phase-2 result with an independent neural estimator.

Two-head gives mean + gain (no tail). Modest surrogate null (neural refits are expensive);
surrogate p only (no separate bootstrap). Primary + global-mode-removed blocks.
Run: PYTHONPATH=.:SBTG ./.venv/bin/python sid_elegans/biolag/run_twohead_phase2.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from sid_elegans.biolag import config as C
from sid_elegans.biolag import metric as M
from sid_elegans.biolag.references import build_registry
from sid_elegans.combined_data import load_combined
from sid_elegans.lagres import metric as mt
from sid_elegans.twohead import fit_two_head

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "sid_elegans" / "output" / "biolag"
N_SURR = 25
EPOCHS_OBS = 150
EPOCHS_SURR = 80
REFS = ["neuropeptide:all", "neuropeptide:pdf-1", "neuropeptide:ntc-1",
        "monoamine:serotonin", "structural:gap", "structural:chemical"]


def twohead_curves(X, names, refs, mask, cfg, epochs):
    curves = {r.name: {"mean": [], "gain": []} for r in refs}
    for l in C.LAG_FRAMES:
        m = fit_two_head(X, names, horizon=l, epochs=epochs, seed=0, **cfg).matrices
        for r in refs:
            curves[r.name]["mean"].append(mt.corr_score(m["mean"], r.adjacency, mask, "auroc"))
            curves[r.name]["gain"].append(mt.corr_score(m["gain"], r.adjacency, mask, "auroc"))
    return {rn: {ch: np.array(v) for ch, v in d.items()} for rn, d in curves.items()}


def bc_block(curves, refs, mu0, sd0):
    out = {}
    for r in refs:
        z = {ch: (curves[r.name][ch] - mu0[r.name][ch]) / sd0[r.name][ch] for ch in ("mean", "gain")}
        bc = {ch: M.band_concordance(z[ch], r.expected_band) for ch in z}
        out[r.name] = {"BC_gain": bc["gain"], "BC_mean": bc["mean"],
                       "dBC": bc["gain"] - bc["mean"]}
    return out


def run_block(X, names, refs, mask, cfg, tag, rng):
    obs = twohead_curves(X, names, refs, mask, cfg, EPOCHS_OBS)
    surr = []
    for s in range(N_SURR):
        surr.append(twohead_curves(M.circshift(X, rng), names, refs, mask, cfg, EPOCHS_SURR))
        print(f"  [{tag}] surrogate {s+1}/{N_SURR}", flush=True)
    mu0 = {r.name: {ch: np.nanmean([sc[r.name][ch] for sc in surr], 0) for ch in ("mean", "gain")}
           for r in refs}
    sd0 = {r.name: {ch: np.nanstd([sc[r.name][ch] for sc in surr], 0) + 1e-9 for ch in ("mean", "gain")}
           for r in refs}
    obs_bc = bc_block(obs, refs, mu0, sd0)
    null_bc = [bc_block(sc, refs, mu0, sd0) for sc in surr]
    res = {}
    for r in refs:
        nd = np.array([nb[r.name]["dBC"] for nb in null_bc])
        ng = np.array([nb[r.name]["BC_gain"] for nb in null_bc])
        res[r.name] = {**obs_bc[r.name],
                       "p_dBC": float((np.sum(nd >= obs_bc[r.name]["dBC"]) + 1) / (len(nd) + 1)),
                       "p_BCgain": float((np.sum(ng >= obs_bc[r.name]["BC_gain"]) + 1) / (len(ng) + 1)),
                       "null_dBC_sd": float(np.std(nd))}
    prim = next(r for r in refs if r.name == "neuropeptide:all")
    res["_control_source_variance_BC"] = float(
        M.band_concordance(M.source_variance_curve(X, names, prim, mask), prim.expected_band))
    return res


def main():
    tune = json.load(open(OUT / "twohead_tuning.json"))
    cfg = tune["best"]["cfg"]
    print(f"tuned cfg (gain null-contrast {tune['best']['contrast_gain']:+.1f}): {cfg}", flush=True)
    X, names, _ = load_combined(coverage_frac=0.6, complete_case=True, signal="deconv", verbose=True)
    allrefs = build_registry(names, min_specific_edges=10)
    refs = [r for r in allrefs if r.name in REFS]
    mask = mt.eval_mask(len(names))
    rng = np.random.default_rng(0)

    print("=== TWO-HEAD PRIMARY ===", flush=True)
    primary = run_block(X, names, refs, mask, cfg, "primary", rng)
    print("=== TWO-HEAD GLOBAL-MODE REMOVED ===", flush=True)
    gmode = run_block(M.remove_global_mode(X, n_pc=1), names, refs, mask, cfg, "gmode", rng)

    out = {"cfg": cfg, "n_surr": N_SURR, "primary": primary, "global_mode_removed": gmode,
           "contrast_gain": tune["best"]["contrast_gain"]}
    json.dump(out, open(OUT / "twohead_phase2.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT/'twohead_phase2.json'}")
    for tag, blk in [("PRIMARY", primary), ("GLOBAL-MODE-REMOVED", gmode)]:
        p = blk["neuropeptide:all"]
        print(f"[{tag}] neuropeptide:all  BC_gain={p['BC_gain']:+.2f} (p={p['p_BCgain']:.3f})  "
              f"dBC={p['dBC']:+.2f} (p={p['p_dBC']:.3f})  src-var ctrl={blk['_control_source_variance_BC']:+.2f}")


if __name__ == "__main__":
    main()
