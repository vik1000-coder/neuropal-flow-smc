r"""Phase 2 driver: inferential band-concordance with the full control battery.

Pre-registered confirmatory: dBC_neuropeptide = BC_gain - BC_mean (within one SID fit).
Supported iff: dBC>0 (worm-boot CI excludes 0) AND BC_gain clears the surrogate null (p<0.05)
AND controls return null (source-variance -> 0; gain on gap/chem not slow-concentrated) AND it
survives removing the shared global brain-state mode. All other references = exploratory.

Run: PYTHONPATH=.:SBTG ./.venv/bin/python sid_elegans/biolag/run_phase2.py
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

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "sid_elegans" / "output" / "biolag"
N_SURR = 100
N_BOOT = 100
PRIMARY = "neuropeptide:all"


def bc_all(curves, refs, mu0, sd0):
    """z-score every (ref,channel) curve by (mu0,sd0) and return BC + dBC(gain-mean, tail-mean)."""
    out = {}
    for r in refs:
        z = {ch: (curves[r.name][ch] - mu0[r.name][ch]) / sd0[r.name][ch]
             for ch in ("mean", "gain", "tail")}
        bc = {ch: M.band_concordance(z[ch], r.expected_band) for ch in z}
        out[r.name] = {"BC": bc, "dBC_gain": bc["gain"] - bc["mean"],
                       "dBC_tail": bc["tail"] - bc["mean"]}
    return out


def summarize(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "p_gt0": np.nan}
    return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "p_gt0": float(np.mean(v > 0))}


def run_block(X, names, refs, mask, tag):
    """Full surrogate-null + worm-bootstrap band-concordance on one worm set."""
    rng = np.random.default_rng(0)
    obs = M.sid_curves(X, names, refs, mask)

    # --- surrogate null: mu0/sd0 per (ref,ch,lag) + null distribution of BC/dBC ---
    surr_curves = []
    for s in range(N_SURR):
        surr_curves.append(M.sid_curves(M.circshift(X, rng), names, refs, mask))
        if (s + 1) % 25 == 0:
            print(f"  [{tag}] surrogate {s+1}/{N_SURR}", flush=True)
    mu0 = {r.name: {ch: np.nanmean([sc[r.name][ch] for sc in surr_curves], 0)
                    for ch in ("mean", "gain", "tail")} for r in refs}
    sd0 = {r.name: {ch: np.nanstd([sc[r.name][ch] for sc in surr_curves], 0) + 1e-9
                    for ch in ("mean", "gain", "tail")} for r in refs}
    obs_bc = bc_all(obs, refs, mu0, sd0)
    null_bc = [bc_all(sc, refs, mu0, sd0) for sc in surr_curves]

    # --- worm bootstrap for CIs (same 6-worm count -> matched null floor) ---
    W = len(X); boot = {r.name: {"dBC_gain": [], "BC_gain": [], "BC_mean": []} for r in refs}
    for b in range(N_BOOT):
        idx = rng.integers(0, W, W).tolist()
        bc = bc_all(M.sid_curves([X[i] for i in idx], names, refs, mask), refs, mu0, sd0)
        for r in refs:
            boot[r.name]["dBC_gain"].append(bc[r.name]["dBC_gain"])
            boot[r.name]["BC_gain"].append(bc[r.name]["BC"]["gain"])
            boot[r.name]["BC_mean"].append(bc[r.name]["BC"]["mean"])
        if (b + 1) % 25 == 0:
            print(f"  [{tag}] bootstrap {b+1}/{N_BOOT}", flush=True)

    res = {}
    for r in refs:
        dobs = obs_bc[r.name]["dBC_gain"]
        gobs = obs_bc[r.name]["BC"]["gain"]
        null_d = np.array([nb[r.name]["dBC_gain"] for nb in null_bc])
        null_g = np.array([nb[r.name]["BC"]["gain"] for nb in null_bc])
        res[r.name] = {
            "family": r.family, "level": r.level, "n_edges": r.n_edges,
            "expected_band": r.expected_band,
            "BC_mean_obs": obs_bc[r.name]["BC"]["mean"], "BC_gain_obs": gobs,
            "dBC_gain_obs": dobs,
            "dBC_gain_boot": summarize(boot[r.name]["dBC_gain"]),
            "BC_gain_boot": summarize(boot[r.name]["BC_gain"]),
            "BC_mean_boot": summarize(boot[r.name]["BC_mean"]),
            "p_dBC_surrogate": float((np.sum(null_d >= dobs) + 1) / (len(null_d) + 1)),
            "p_BCgain_surrogate": float((np.sum(null_g >= gobs) + 1) / (len(null_g) + 1)),
        }
    # trivial source-variance control on the primary reference (must be ~0, lag-flat)
    prim = next(r for r in refs if r.name == PRIMARY)
    sv = M.source_variance_curve(X, names, prim, mask)
    res["_control_source_variance_BC"] = float(M.band_concordance(sv, prim.expected_band))
    return res


def main():
    X, names, _ = load_combined(coverage_frac=0.6, complete_case=True, signal="deconv",
                               verbose=True)
    refs = build_registry(names, min_specific_edges=10)
    mask = mt.eval_mask(len(names))

    print("=== PRIMARY (all 6 worms, full data) ===", flush=True)
    primary = run_block(X, names, refs, mask, "primary")

    print("=== GLOBAL-MODE CONTROL (top global PC removed) ===", flush=True)
    Xg = M.remove_global_mode(X, n_pc=1)
    gmode = run_block(Xg, names, refs, mask, "gmode")

    out = {"config": {"n_worms": len(X), "n_neurons": len(names), "n_surr": N_SURR,
                      "n_boot": N_BOOT, "bands": C.bands(), "primary": PRIMARY,
                      "caveats": C.CAVEATS},
           "primary": primary, "global_mode_removed": gmode}
    json.dump(out, open(OUT / "phase2_results.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT/'phase2_results.json'}")

    # console headline
    for tag, block in [("PRIMARY", primary), ("GLOBAL-MODE-REMOVED", gmode)]:
        p = block[PRIMARY]
        d = p["dBC_gain_boot"]; g = p["BC_gain_boot"]
        print(f"\n[{tag}] neuropeptide:all")
        print(f"   BC_mean={p['BC_mean_obs']:+.3f}  BC_gain={p['BC_gain_obs']:+.3f}")
        print(f"   dBC(gain-mean) = {d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] "
              f"surrogate p={p['p_dBC_surrogate']:.3f}")
        print(f"   BC_gain alone  = {g['mean']:+.3f} [{g['lo']:+.3f},{g['hi']:+.3f}] "
              f"surrogate p={p['p_BCgain_surrogate']:.3f}")
        print(f"   control: source-variance BC = {block['_control_source_variance_BC']:+.3f} (expect ~0)")
        print(f"   anchor: gain BC on gap={block['structural:gap']['BC_gain_obs']:+.3f} "
              f"chem={block['structural:chemical']['BC_gain_obs']:+.3f} (expect not-positive)")


if __name__ == "__main__":
    main()
