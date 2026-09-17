r"""Two adjudicating baselines for the peptidergic partial gain>mean lag-shift.

Run on the 6-worm / 80-neuron configuration (where the effect was significant).

#4 SURROGATE NULL. Circular-shift each neuron independently per worm (preserves each
   neuron's own spectrum/autocorrelation, destroys cross-neuron coupling), refit the SAME
   partial SID, recompute the gain-mean / tail-mean log-slope contrast. If the real value
   sits inside this null -> the estimator/metric manufactures the effect (artifact). If the
   null is centered ~0 and the real value is in the tail -> the coupling is real.

#2 NON-SID PARTIAL VARIANCE-VAR. A model-free partial variance-coupling estimator with no
   SID machinery:  mean channel = ridge VAR coeff  x_i(t) -> x_j(t+l);
   gain channel   = ridge coeff  x_i(t)^2 -> r_j(t+l)^2  (r = mean-VAR residual).
   If it reproduces neuropeptide gain>mean lag-shift, the effect is not SID-specific.

Writes paper/adjudicate.json.
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
N_SURR = 120
NBOOT = 120
OUT = Path(__file__).resolve().parent / "adjudicate.json"


def logslope(c):
    ok = np.isfinite(c)
    if ok.sum() < 3 or np.std(c[ok]) < 1e-12:
        return np.nan
    return float(np.polyfit(np.log2(np.array(LAGS)[ok]), c[ok], 1)[0])


# ---------- shared: contrast from a set of per-lag matrices ----------
def contrast_from_matrices(mats_by_lag, refs, mask, gain_key, mean_key, tail_fn=None):
    curv = {r: {"mean": [], "gain": [], "tail": []} for r in refs}
    for l in LAGS:
        m = mats_by_lag[l]
        for r, R in refs.items():
            curv[r]["mean"].append(mt.corr_score(m[mean_key], R, mask, "auroc"))
            curv[r]["gain"].append(mt.corr_score(m[gain_key], R, mask, "auroc"))
            if tail_fn is not None:
                curv[r]["tail"].append(mt.corr_score(tail_fn(m), R, mask, "auroc"))
    out = {}
    for r in refs:
        sm = logslope(np.array(curv[r]["mean"]))
        gm = logslope(np.array(curv[r]["gain"])) - sm
        tm = (logslope(np.array(curv[r]["tail"])) - sm) if tail_fn is not None else np.nan
        out[r] = (gm, tm)
    return out


# ---------- #4 surrogate ----------
def circshift(X_list, rng):
    out = []
    for X in X_list:
        T, N = X.shape
        Xs = np.empty_like(X)
        for k in range(N):
            Xs[:, k] = np.roll(X[:, k], int(rng.integers(0, T)))
        out.append(Xs)
    return out


def sid_matrices(X_list, names):
    d = {}
    for l in LAGS:
        d[l] = fit_distributional_connectome(X_list, names, lag=l, target_mode="next",
                                             ridge=1e-2).matrices
    return d


def surrogate_null(X, names, refs, mask):
    tail_fn = lambda m: np.abs(m["tail_hi"]) + np.abs(m["tail_lo"])
    real = contrast_from_matrices(sid_matrices(X, names), refs, mask, "gain", "mean", tail_fn)
    rng = np.random.default_rng(0)
    null = {r: {"gain": [], "tail": []} for r in refs}
    for s in range(N_SURR):
        Xs = circshift(X, rng)
        c = contrast_from_matrices(sid_matrices(Xs, names), refs, mask, "gain", "mean", tail_fn)
        for r in refs:
            null[r]["gain"].append(c[r][0]); null[r]["tail"].append(c[r][1])
        if (s + 1) % 20 == 0:
            print(f"  [surrogate] {s+1}/{N_SURR}", flush=True)
    res = {}
    for r in refs:
        g = np.array([v for v in null[r]["gain"] if np.isfinite(v)])
        t = np.array([v for v in null[r]["tail"] if np.isfinite(v)])
        rg, rt = real[r]
        res[r] = {
            "real_gain_minus_mean": rg, "real_tail_minus_mean": rt,
            "null_gain_mean": float(g.mean()), "null_gain_lo": float(np.percentile(g, 2.5)),
            "null_gain_hi": float(np.percentile(g, 97.5)),
            "p_gain": float((np.sum(g >= rg) + 1) / (len(g) + 1)),
            "null_tail_mean": float(t.mean()),
            "p_tail": float((np.sum(t >= rt) + 1) / (len(t) + 1)),
        }
        print(f"  {r:13s} real gain-mean {rg:+.4f}  null {res[r]['null_gain_mean']:+.4f} "
              f"[{res[r]['null_gain_lo']:+.4f},{res[r]['null_gain_hi']:+.4f}]  p={res[r]['p_gain']:.3f}",
              flush=True)
    return res


# ---------- #2 partial variance-VAR ----------
def _demean_pool(X_list, lag):
    nows, futs = [], []
    for x in X_list:
        x = np.asarray(x, float)
        x = x - np.nanmean(x, 0, keepdims=True)
        x = np.nan_to_num(x, nan=0.0)
        if x.shape[0] <= lag:
            continue
        nows.append(x[:x.shape[0] - lag]); futs.append(x[lag:])
    return np.concatenate(nows, 0), np.concatenate(futs, 0)


def varvar_matrices(X_list, names, alpha=10.0):
    N = len(names); out = {}
    for l in LAGS:
        Xn, Xf = _demean_pool(X_list, l)
        G = Xn.T @ Xn + alpha * np.eye(N)
        Bmean = np.linalg.solve(G, Xn.T @ Xf)          # [i(src), j(tgt)]
        resid = Xf - Xn @ Bmean
        Z = Xn ** 2
        Gz = Z.T @ Z + alpha * np.eye(N)
        Bgain = np.linalg.solve(Gz, Z.T @ (resid ** 2))
        Em = Bmean.T.copy(); Eg = Bgain.T.copy()
        np.fill_diagonal(Em, 0.0); np.fill_diagonal(Eg, 0.0)
        out[l] = {"mean": Em, "gain": Eg}
    return out


def varvar_bootstrap(X, names, refs, mask):
    W = len(X); rng = np.random.default_rng(0)
    boot = {r: [] for r in refs}
    point = contrast_from_matrices(varvar_matrices(X, names), refs, mask, "gain", "mean")
    for b in range(NBOOT):
        idx = rng.integers(0, W, W).tolist()
        c = contrast_from_matrices(varvar_matrices([X[i] for i in idx], names), refs, mask,
                                   "gain", "mean")
        for r in refs:
            boot[r].append(c[r][0])
    res = {}
    for r in refs:
        x = np.array([v for v in boot[r] if np.isfinite(v)])
        res[r] = {"point_gain_minus_mean": point[r][0], "mean": float(x.mean()),
                  "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5)),
                  "p_gt0": float(np.mean(x > 0))}
        print(f"  {r:13s} var-mean {res[r]['mean']:+.4f} [{res[r]['lo']:+.4f},{res[r]['hi']:+.4f}] "
              f"P(>0)={res[r]['p_gt0']:.2f}", flush=True)
    return res


def main():
    X, names, _ = load_combined(coverage_frac=0.6, complete_case=True, signal="deconv", verbose=True)
    N = len(names); mask = mt.eval_mask(N)
    cook = load_cook(names); mono = load_all_monoamine_layers(names)
    refs = {"cook_chem": cook["chem"], "serotonin": mono["serotonin"],
            "dopamine": mono["dopamine"], "neuropeptide": load_neuropeptide_layer(names)}
    print("=== #2 NON-SID partial variance-VAR (gain=partial energy coupling) ===", flush=True)
    vv = varvar_bootstrap(X, names, refs, mask)
    print("\n=== #4 SURROGATE NULL (circular-shift, refit partial SID) ===", flush=True)
    sn = surrogate_null(X, names, refs, mask)
    json.dump({"n_worms": len(X), "n_neurons": N, "varvar": vv, "surrogate": sn},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
