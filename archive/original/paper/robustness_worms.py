r"""Robustness of the lag-resolved contrast to worm count.

The headline (sid_elegans/lagres/run_contrast.py) used the strict complete-case set
(6 real worms, 80 neurons). The data has a worms x neurons tradeoff: requiring fewer
neurons admits more *real* (non-imputed) worms. This re-runs the SAME paired within-SID
bootstrap contrast on the higher-worm / fewer-neuron corners of that frontier, WITHOUT
imputation, to test whether the peptidergic channel x lag effect holds with more animals.

Writes paper/robustness_worms.json.
Run: PYTHONPATH=.:SBTG ./.venv/bin/python paper/robustness_worms.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np

from sid_elegans.combined_data import load_combined
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.ground_truth import (load_all_monoamine_layers, load_cook,
                                       load_neuropeptide_layer)
from sid_elegans.lagres import metric as mt

warnings.filterwarnings("ignore")
LAGS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
NBOOT = 150
OUT = Path(__file__).resolve().parent / "robustness_worms.json"


def logslope(c):
    ok = np.isfinite(c)
    if ok.sum() < 3 or np.std(c[ok]) < 1e-12:
        return np.nan
    return float(np.polyfit(np.log2(np.array(LAGS)[ok]), c[ok], 1)[0])


def sid_curves(Xs, names, R, mask):
    cm, cg, ct = [], [], []
    for l in LAGS:
        m = fit_distributional_connectome(Xs, names, lag=l, target_mode="next",
                                          ridge=1e-2, fps=4.0, seed=0).matrices
        cm.append(mt.corr_score(m["mean"], R, mask, "auroc"))
        cg.append(mt.corr_score(m["gain"], R, mask, "auroc"))
        ct.append(mt.corr_score(np.abs(m["tail_hi"]) + np.abs(m["tail_lo"]), R, mask, "auroc"))
    return np.array(cm), np.array(cg), np.array(ct)


def run_setting(coverage_frac):
    X, names, _ = load_combined(coverage_frac=coverage_frac, complete_case=True,
                                signal="deconv", verbose=False)
    N, W = len(names), len(X)
    mask = mt.eval_mask(N)
    cook = load_cook(names); mono = load_all_monoamine_layers(names)
    refs = {"cook_chem": cook["chem"], "serotonin": mono["serotonin"],
            "dopamine": mono["dopamine"], "neuropeptide": load_neuropeptide_layer(names)}
    rng = np.random.default_rng(0)
    out = {"n_worms": W, "n_neurons": N, "refs": {}}
    for rname, R in refs.items():
        gm, tm = [], []
        for _ in range(NBOOT):
            idx = rng.integers(0, W, W).tolist()
            cm, cg, ct = sid_curves([X[i] for i in idx], names, R, mask)
            sm = logslope(cm)
            gm.append(logslope(cg) - sm); tm.append(logslope(ct) - sm)
        def stat(x):
            x = np.array([v for v in x if np.isfinite(v)])
            return {"mean": float(np.mean(x)), "lo": float(np.percentile(x, 2.5)),
                    "hi": float(np.percentile(x, 97.5)), "p_gt0": float(np.mean(x > 0))}
        out["refs"][rname] = {"gain_minus_mean": stat(gm), "tail_minus_mean": stat(tm)}
        g = out["refs"][rname]["gain_minus_mean"]
        print(f"  [{W}w/{N}n] {rname:13s} gain-mean {g['mean']:+.4f} "
              f"[{g['lo']:+.4f},{g['hi']:+.4f}] P(>0)={g['p_gt0']:.2f}", flush=True)
    return out


def main():
    results = {}
    for cf, tag in [(0.9, "20w_56n"), (0.8, "14w_60n")]:
        print(f"--- coverage_frac={cf} ---", flush=True)
        results[tag] = run_setting(cf)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
