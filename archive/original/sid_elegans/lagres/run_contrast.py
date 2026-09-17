r"""The paired within-estimator lag-profile contrast (the payoff test).

Within the SAME SID fit, does the DISTRIBUTIONAL (gain/tail) channel's correspondence-vs-lag
curve lean slower (higher log-slope, later peak) than the MEAN channel's — specifically for
the modulatory references (serotonin, neuropeptide, dopamine) vs the structural reference
(cook_chem)? Bootstrapped over worms for CIs. This is the clean, controlled comparison:
same estimator, same data, only the channel changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sid_elegans.combined_data import load_combined
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.ground_truth import (load_all_monoamine_layers, load_cook,
                                      load_neuropeptide_layer)
from sid_elegans.lagres import metric as mt

OUT = Path(__file__).resolve().parents[1] / "output" / "lagres"
LAGS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
NBOOT = 150


def sid_curves(X_sub, names, R, mask):
    """AUROC curves over lag for SID mean/gain/tail on a worm subset."""
    cm, cg, ct = [], [], []
    for l in LAGS:
        res = fit_distributional_connectome(X_sub, names, lag=l, target_mode="next",
                                            ridge=1e-2, source_tau=None, fps=4.0, seed=0)
        m = res.matrices
        cm.append(mt.corr_score(m["mean"], R, mask, "auroc"))
        cg.append(mt.corr_score(m["gain"], R, mask, "auroc"))
        ct.append(mt.corr_score(np.abs(m["tail_hi"]) + np.abs(m["tail_lo"]), R, mask, "auroc"))
    return np.array(cm), np.array(cg), np.array(ct)


def logslope(c):
    ok = np.isfinite(c)
    if ok.sum() < 3 or np.std(c[ok]) < 1e-12:
        return np.nan
    return float(np.polyfit(np.log2(np.array(LAGS)[ok]), c[ok], 1)[0])


def main():
    import warnings; warnings.filterwarnings("ignore")
    X_list, names, fps = load_combined(coverage_frac=0.6, complete_case=True,
                                       signal="deconv", verbose=True)
    N = len(names); W = len(X_list); mask = mt.eval_mask(N)
    cook = load_cook(names); mono = load_all_monoamine_layers(names)
    refs = {"cook_chem": cook["chem"], "serotonin": mono["serotonin"],
            "dopamine": mono["dopamine"], "neuropeptide": load_neuropeptide_layer(names)}

    rng = np.random.default_rng(0)
    results = {}
    for rname, R in refs.items():
        boot = {"mean": [], "gain": [], "tail": [], "gain_minus_mean": [], "tail_minus_mean": []}
        for b in range(NBOOT):
            idx = rng.integers(0, W, W).tolist()
            Xs = [X_list[i] for i in idx]
            cm, cg, ct = sid_curves(Xs, names, R, mask)
            sm, sg, st = logslope(cm), logslope(cg), logslope(ct)
            boot["mean"].append(sm); boot["gain"].append(sg); boot["tail"].append(st)
            boot["gain_minus_mean"].append(sg - sm); boot["tail_minus_mean"].append(st - sm)
        def stat(x):
            x = np.array([v for v in x if np.isfinite(v)])
            return {"mean": float(np.mean(x)), "lo": float(np.percentile(x, 2.5)),
                    "hi": float(np.percentile(x, 97.5)),
                    "p_gt0": float(np.mean(x > 0))}
        results[rname] = {k: stat(v) for k, v in boot.items()}
        gm = results[rname]["gain_minus_mean"]
        print(f"{rname:12s}: logslope gain-mean = {gm['mean']:+.4f} "
              f"[{gm['lo']:+.4f},{gm['hi']:+.4f}] P(>0)={gm['p_gt0']:.2f} "
              f"(>0 = gain leans SLOWER than mean)")
    json.dump(results, open(OUT / "contrast_logslope.json", "w"), indent=2)
    print(f"\nwrote {OUT/'contrast_logslope.json'}")
    print("\nInterpretation: for MODULATORY refs (serotonin/dopamine/neuropeptide) we expect "
          "gain-mean logslope > 0 (distributional channel peaks at longer lags); for the "
          "STRUCTURAL ref (cook_chem) we expect ~0 or <0.")


if __name__ == "__main__":
    main()
