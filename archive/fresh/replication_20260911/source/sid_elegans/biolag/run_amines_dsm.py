r"""Tier-1 follow-ups on all 28 worms via ACMMA:
  PART A -- the MONOAMINE neuromodulators (dopamine, serotonin, tyramine, octopamine, pooled) tested
            fairly with the validated all-data estimator (Hyvarinen), band-concordance + bootstrap +
            global-mode + source-variance control.
  PART B -- give DENOISING score matching its best shot: ACMMA-DSM sigma-sweep (point estimates) for
            neuropeptide + pooled monoamine, to see if any noise level surfaces the effect.
Gentle by default. Writes amines_dsm.json.
"""
from __future__ import annotations
import json, os, time, warnings
from pathlib import Path
import numpy as np
from sid_elegans.acmma import fit_acmma_connectome
from sid_elegans.biolag import config as C, metric as M
from sid_elegans.biolag.references import build_registry
from sid_elegans.biolag.run_acmma_phase2 import remove_global_mode_nan
from sid_elegans.combined_data import load_combined
from sid_elegans.lagres import metric as mt

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "sid_elegans" / "output" / "biolag"
LAGS = C.LAG_FRAMES
NS = int(os.environ.get("AD_NSURR", 20)); NB = int(os.environ.get("AD_NBOOT", 20))
THR = float(os.environ.get("AD_THROTTLE", 0.4)); RIDGE = 0.01
REFS_A = ["neuropeptide:all", "monoamine:all", "monoamine:dopamine", "monoamine:serotonin",
          "monoamine:tyramine", "monoamine:octopamine", "structural:gap", "structural:chemical"]


def curves(X, names, refs, mask, sigma_frac):
    cm, cg = {r.name: [] for r in refs}, {r.name: [] for r in refs}
    for l in LAGS:
        m, _ = fit_acmma_connectome(X, names, lag=l, ridge=RIDGE, min_triple=2000, shrink=0.05,
                                    sigma_frac=sigma_frac)
        for r in refs:
            cm[r.name].append(mt.corr_score(m["mean"], r.adjacency, mask, "auroc"))
            cg[r.name].append(mt.corr_score(m["gain"], r.adjacency, mask, "auroc"))
    return {r.name: {"mean": np.array(cm[r.name]), "gain": np.array(cg[r.name])} for r in refs}


def bc(cur, refs, mu0, sd0):
    o = {}
    for r in refs:
        z = {ch: (cur[r.name][ch] - mu0[r.name][ch]) / sd0[r.name][ch] for ch in ("mean", "gain")}
        b = {ch: M.band_concordance(z[ch], r.expected_band) for ch in z}
        o[r.name] = {"BC_gain": b["gain"], "dBC": b["gain"] - b["mean"]}
    return o


def st(x):
    x = np.array([v for v in x if np.isfinite(v)])
    return {"mean": float(x.mean()), "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5))} if x.size else {}


def block(X, names, refs, mask, tag, rng, sigma_frac=0.0, boot=True):
    obs = curves(X, names, refs, mask, sigma_frac)
    surr = []
    for s in range(NS):
        surr.append(curves(M.circshift(X, rng), names, refs, mask, sigma_frac))
        if THR: time.sleep(THR)
    mu0 = {r.name: {c: np.nanmean([x[r.name][c] for x in surr], 0) for c in ("mean", "gain")} for r in refs}
    sd0 = {r.name: {c: np.nanstd([x[r.name][c] for x in surr], 0) + 1e-9 for c in ("mean", "gain")} for r in refs}
    ob = bc(obs, refs, mu0, sd0); nb = [bc(x, refs, mu0, sd0) for x in surr]
    res = {}
    bt = []
    if boot:
        for b in range(NB):
            idx = rng.integers(0, len(X), len(X)).tolist()
            bt.append(bc(curves([X[i] for i in idx], names, refs, mask, sigma_frac), refs, mu0, sd0))
            if THR: time.sleep(THR)
    for r in refs:
        nd = np.array([x[r.name]["dBC"] for x in nb])
        res[r.name] = {**ob[r.name], "p_dBC": float((np.sum(nd >= ob[r.name]["dBC"]) + 1) / (len(nd) + 1)),
                       "dBC_boot": st([x[r.name]["dBC"] for x in bt]) if boot else {}}
    return res


def main():
    print(f"[config] ridge={RIDGE} NS={NS} NB={NB}", flush=True)
    X, names, _ = load_combined(complete_case=False, min_worms_per_neuron=6, signal="deconv", verbose=True)
    reg = build_registry(names, min_specific_edges=8)
    refs = [r for r in reg if r.name in REFS_A]
    mask = mt.eval_mask(len(names)); rng = np.random.default_rng(0)
    out = {"n_worms": len(X), "n_neurons": len(names)}

    print("=== PART A: monoamines under ACMMA (Hyvarinen), 28 worms ===", flush=True)
    prim = block(X, names, refs, mask, "amines", rng, sigma_frac=0.0, boot=True)
    gm = block(remove_global_mode_nan(X, 1), names, refs, mask, "amines-gm", rng, sigma_frac=0.0, boot=False)
    out["monoamines_primary"] = prim; out["monoamines_global_removed"] = gm
    json.dump(out, open(OUT / "amines_dsm.json", "w"), indent=2, default=float)
    print("  ref                     dBC   (p)     boot CI            GLOBAL dBC", flush=True)
    for r in refs:
        p = prim[r.name]; g = gm[r.name]
        b = p.get("dBC_boot", {})
        excl = "*" if b and (b["lo"] > 0 or b["hi"] < 0) else " "
        print(f"  {r.name:22s} {p['dBC']:+6.2f} ({p['p_dBC']:.3f})  [{b.get('lo',0):+.2f},{b.get('hi',0):+.2f}]{excl}  {g['dBC']:+.2f}", flush=True)

    print("\n=== PART B: DENOISING score-matching sigma-sweep (point est), 28 worms ===", flush=True)
    refsB = [r for r in refs if r.name in ("neuropeptide:all", "monoamine:all")]
    sweep = {}
    for sf in [0.0, 0.3, 0.5, 1.0]:
        r = block(X, names, refsB, mask, f"dsm{sf}", np.random.default_rng(0), sigma_frac=sf, boot=False)
        sweep[str(sf)] = {k: {"dBC": v["dBC"], "p": v["p_dBC"]} for k, v in r.items()}
        print(f"  sigma_frac={sf}: neuropeptide dBC={r['neuropeptide:all']['dBC']:+.2f}(p={r['neuropeptide:all']['p_dBC']:.3f})  "
              f"monoamine:all dBC={r['monoamine:all']['dBC']:+.2f}(p={r['monoamine:all']['p_dBC']:.3f})", flush=True)
    out["dsm_sigma_sweep"] = sweep
    json.dump(out, open(OUT / "amines_dsm.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT/'amines_dsm.json'}")


if __name__ == "__main__":
    main()
