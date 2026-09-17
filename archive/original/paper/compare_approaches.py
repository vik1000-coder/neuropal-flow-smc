r"""Two maximal-data approaches to the lag-resolved contrast, with worm-bootstrap CIs.

Both use all 28 worms / 84 neurons (no complete-case restriction):
  Approach 1  PAIRWISE  : marginal, per-edge worm support, NO imputation (sid_elegans/pairwise.py)
  Approach 2  DONOR-MV  : multivariate/partial on random-donor-imputed data (sid_elegans/impute.py)

Bootstrap resamples worms; for the donor approach the imputation is redrawn each resample so
its randomness is folded into the CI. Statistic = Delta log-slope of AUROC-vs-lag
(distributional - mean channel). Writes paper/compare_approaches.json.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np

from sid_elegans.combined_data import load_combined
from sid_elegans.pairwise import fit_pairwise_connectome
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.impute import impute_random_donor
from sid_elegans.ground_truth import (load_all_monoamine_layers, load_cook,
                                       load_neuropeptide_layer)
from sid_elegans.lagres import metric as mt

warnings.filterwarnings("ignore")
LAGS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
NBOOT = 100
OUT = Path(__file__).resolve().parent / "compare_approaches.json"


def logslope(c):
    ok = np.isfinite(c)
    if ok.sum() < 3 or np.std(c[ok]) < 1e-12:
        return np.nan
    return float(np.polyfit(np.log2(np.array(LAGS)[ok]), c[ok], 1)[0])


def channel_logslopes(fit_at_lag, refs, mask):
    """Return {ref: (gain_minus_mean, tail_minus_mean)} for one fitted worm-set."""
    curv = {r: {"mean": [], "gain": [], "tail": []} for r in refs}
    for l in LAGS:
        m = fit_at_lag(l)
        for r, R in refs.items():
            curv[r]["mean"].append(mt.corr_score(m["mean"], R, mask, "auroc"))
            curv[r]["gain"].append(mt.corr_score(m["gain"], R, mask, "auroc"))
            curv[r]["tail"].append(mt.corr_score(np.abs(m["tail_hi"]) + np.abs(m["tail_lo"]),
                                                 R, mask, "auroc"))
    out = {}
    for r in refs:
        sm = logslope(np.array(curv[r]["mean"]))
        out[r] = (logslope(np.array(curv[r]["gain"])) - sm,
                  logslope(np.array(curv[r]["tail"])) - sm)
    return out


def stat(x):
    x = np.array([v for v in x if np.isfinite(v)])
    if x.size == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "p_gt0": np.nan}
    return {"mean": float(np.mean(x)), "lo": float(np.percentile(x, 2.5)),
            "hi": float(np.percentile(x, 97.5)), "p_gt0": float(np.mean(x > 0))}


def run_bootstrap(approach, X, names, refs, mask):
    W = len(X)
    rng = np.random.default_rng(0)
    boot = {r: {"gain": [], "tail": []} for r in refs}
    for b in range(NBOOT):
        idx = rng.integers(0, W, W).tolist()
        Xs = [X[i] for i in idx]
        if approach == "pairwise":
            fit_at_lag = lambda l, Xs=Xs: fit_pairwise_connectome(Xs, names, lag=l, ridge=1e-2)
        else:  # donor-imputed multivariate; fresh donors each resample
            Xi, _ = impute_random_donor(Xs, seed=1000 + b)
            fit_at_lag = lambda l, Xi=Xi: fit_distributional_connectome(Xi, names, lag=l).matrices
        ls = channel_logslopes(fit_at_lag, refs, mask)
        for r in refs:
            boot[r]["gain"].append(ls[r][0]); boot[r]["tail"].append(ls[r][1])
        if (b + 1) % 20 == 0:
            print(f"  [{approach}] {b+1}/{NBOOT}", flush=True)
    return {r: {"gain_minus_mean": stat(boot[r]["gain"]),
                "tail_minus_mean": stat(boot[r]["tail"])} for r in refs}


def main():
    X, names, _ = load_combined(complete_case=False, min_worms_per_neuron=6,
                                signal="deconv", verbose=True)
    N = len(names); mask = mt.eval_mask(N)
    cook = load_cook(names); mono = load_all_monoamine_layers(names)
    refs = {"cook_chem": cook["chem"], "serotonin": mono["serotonin"],
            "dopamine": mono["dopamine"], "neuropeptide": load_neuropeptide_layer(names)}
    results = {"n_worms": len(X), "n_neurons": N, "nboot": NBOOT, "approaches": {}}
    for approach in ["pairwise", "donor_multivariate"]:
        print(f"=== {approach} ===", flush=True)
        results["approaches"][approach] = run_bootstrap(approach, X, names, refs, mask)
        for r in refs:
            g = results["approaches"][approach][r]["gain_minus_mean"]
            print(f"  {r:13s} gain-mean {g['mean']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] "
                  f"P(>0)={g['p_gt0']:.2f}", flush=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
