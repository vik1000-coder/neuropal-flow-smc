r"""Main-venv per-lag directed effect matrices, all returned as [post,pre]=[target,source]
(E(l)[j,i] = effect of source i(t) on target j(t+l)). Orientations verified with synthetic
directed drivers during planning. Multi-worm: pooled without crossing worm boundaries;
per-worm de-mean then NaN->0; never per-neuron z-score.

Channels:
  mean-channel (classical analog): cross_corr, ridge_lag, var_partial, dynotears, sbtg_mu_hat, SID mean, MDN mean
  distributional (no classical analog): SID gain / tail, MDN gain
"""
from __future__ import annotations

import numpy as np


def _demean(X_list):
    out = []
    for x in X_list:
        xc = np.asarray(x, float) - np.nanmean(x, axis=0, keepdims=True)
        out.append(np.nan_to_num(xc, nan=0.0))
    return out


def _pool_pairs(X_list, lag):
    now, fut = [], []
    for x in _demean(X_list):
        if x.shape[0] <= lag:
            continue
        now.append(x[:x.shape[0] - lag]); fut.append(x[lag:])
    return np.concatenate(now, 0), np.concatenate(fut, 0)


# -------------------- mean-channel classical --------------------
def cross_corr(X_list, lag):
    """MARGINAL lagged cross-correlation E(l)[j,i]=corr(x_i(t), x_j(t+l)); [post,pre]."""
    N = X_list[0].shape[1]
    num = np.zeros((N, N)); vf = np.zeros(N); vp = np.zeros(N)
    for x in _demean(X_list):
        if x.shape[0] <= lag:
            continue
        past, fut = x[:-lag], x[lag:]
        num += fut.T @ past; vf += (fut ** 2).sum(0); vp += (past ** 2).sum(0)
    E = num / (np.sqrt(np.outer(vf, vp)) + 1e-12)
    np.fill_diagonal(E, 0.0)
    return E


def ridge_lag(X_list, lag, alpha=10.0):
    """MARGINAL single-lag multivariate ridge E(l)[j,i] (source i -> target j); [post,pre]."""
    Xn, Xf = _pool_pairs(X_list, lag)
    N = Xn.shape[1]
    G = Xn.T @ Xn + alpha * np.eye(N)
    B = np.linalg.solve(G, Xn.T @ Xf)     # B[i,j] = source i -> target j
    E = B.T.copy(); np.fill_diagonal(E, 0.0)
    return E


def var_partial(X_list, p, alpha=10.0):
    """PARTIAL pooled ridge-VAR(p). Returns {l: E(l)[post,pre]} for l=1..p (controls all lags)."""
    Zs, Ys = [], []
    for x in _demean(X_list):
        T, N = x.shape
        if T <= p:
            continue
        blocks = [x[p - l:T - l] for l in range(1, p + 1)]   # block l -> x(t-l)
        Zs.append(np.concatenate(blocks, 1)); Ys.append(x[p:])
    Z = np.concatenate(Zs, 0); Y = np.concatenate(Ys, 0); N = Y.shape[1]
    Bstack = np.linalg.solve(Z.T @ Z + alpha * np.eye(N * p), Z.T @ Y)  # [N*p regressor, N eq]
    out = {}
    for l in range(1, p + 1):
        E = Bstack[(l - 1) * N:l * N, :].T.copy()   # -> [post,pre]
        np.fill_diagonal(E, 0.0)
        out[l] = E
    return out


def dynotears(X_list, L, lambda_a=0.05, ridge=1e-3):
    """Native DYNOTEARS-analog. Returns {l: E(l)[post,pre]}."""
    from sid_elegans.lagres.dynotears import fit_dynotears_analog
    r = fit_dynotears_analog(X_list, L, lambda_a=lambda_a, contemp=False)
    return {l + 1: r["E"][l] for l in range(L)}


# -------------------- SBTG baseline (precomputed) --------------------
def sbtg_mu_hat(lag, ref_names, npz=None):
    """SBTG mean-transfer mu_hat at one lag, reindexed to ref_names; [post,pre].
    Available lags: 1,2,3,5,8,10,15,20."""
    from pathlib import Path
    if npz is None:
        npz = Path(__file__).resolve().parents[2] / "SBTG" / "merged_results" / "result_C_merged.npz"
    d = np.load(npz, allow_pickle=True)
    key = f"mu_hat_lag{lag}"
    if key not in d.files:
        return None
    mu = d[key].astype(float)
    src = [str(s).strip().upper() for s in d["neuron_names"]]
    pos = {n: k for k, n in enumerate(src)}
    idx = [pos.get(str(n).strip().upper(), -1) for n in ref_names]
    n = len(ref_names)
    out = np.full((n, n), np.nan)
    for a, ia in enumerate(idx):
        if ia < 0:
            continue
        for b, ib in enumerate(idx):
            if ib >= 0:
                out[a, b] = mu[ia, ib]
    np.fill_diagonal(out, 0.0)
    return out


# -------------------- SID distributional channels --------------------
def sid_channels(X_list, names, lag, ridge=1e-2):
    """SID mean/gain/tail at one lag (all [post,pre])."""
    from sid_elegans.estimator import fit_distributional_connectome
    res = fit_distributional_connectome(X_list, names, lag=lag, target_mode="next",
                                        ridge=ridge, source_tau=None, fps=4.0, seed=0)
    m = res.matrices
    return {"SID_mean": m["mean"], "SID_gain": m["gain"],
            "SID_tail": np.abs(m["tail_hi"]) + np.abs(m["tail_lo"])}


def mdn_channels(X_list, names, lag, epochs=300):
    """MDN (non-Gaussian) mean & gain at one lag (horizon=lag, instantaneous sources)."""
    from sid_elegans.mdn import fit_mdn
    r = fit_mdn(X_list, names, source_tau_s=None, horizon=lag, k=3, hidden=128,
                layers=2, epochs=epochs, weight_decay=1e-2, device="cpu", seed=0)
    return {"MDN_mean": r.matrices["mean"], "MDN_gain": r.matrices["gain"]}


# channel -> {"mean" | "dist"} tag for the analysis
CHANNEL_KIND = {
    "cross_corr": "mean", "ridge_lag": "mean", "var_partial": "mean",
    "dynotears": "mean", "sbtg_mu": "mean", "varlingam": "mean", "sindy": "mean",
    "pcmci": "mean", "SID_mean": "mean", "MDN_mean": "mean",
    "SID_gain": "dist", "SID_tail": "dist", "MDN_gain": "dist", "sbtg_vol": "dist",
}
