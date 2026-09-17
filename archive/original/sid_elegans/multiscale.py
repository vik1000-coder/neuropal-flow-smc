r"""Multi-timescale filter-bank distributional connectome (#2) + denoising sigma-ledger (#3).

Instead of fitting one source timescale at a time, each target is regressed on a *bank*
of exponential-filter features of every source at several timescales
``tau in taus_s``. The model then lets every source->target edge pick its own timescale,
and the theory's channel decomposition (Section 9.4) is applied to the fitted couplings:

  fast mean mass   M^fast[j,i] = sum_{tau<=tau0} |dE[Y_j]/df_{i,tau}|
  slow gain mass   G^slow[j,i] = sum_{tau> tau0} |dlogVar[Y_j]/df_{i,tau}|

(both [post,pre]). Denoising score matching (sigma>0) with the +sigma^2 ledger gives a
cross-sigma stability check on the gain readout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from sid_neuromod.features.filter_bank import exp_filter_bank
from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher


@dataclass
class MultiScaleResult:
    neuron_names: list
    taus_s: list
    tau0_s: float
    matrices: dict          # channel -> [N, N] ([post, pre])


def _bank_features(X, taus_s, fps):
    """Return [T, N, K] causal exponential-filter features for all neurons & timescales."""
    T = X.shape[0]
    t = np.arange(T, dtype=float) / fps
    return exp_filter_bank(np.nan_to_num(X, nan=0.0), t, [tau for tau in taus_s])


def _pool_target(X_list, j, taus_s, fps, horizon, target_mode):
    """Pooled (Y, F) for target j: F[t] = [source i, timescale r] filter features at t."""
    Ys, Fs = [], []
    K = len(taus_s)
    for X in X_list:
        T = X.shape[0]
        if T <= horizon:
            continue
        F = _bank_features(X, taus_s, fps)          # [T, N, K]
        Fnow = F[:T - horizon].reshape(T - horizon, -1)   # [T-h, N*K], order (i,r) -> i*K+r
        fut = X[horizon:, j]
        y = fut if target_mode == "next" else fut - X[:T - horizon, j]
        Ys.append(y)
        Fs.append(Fnow)
    Y = np.concatenate(Ys)
    F = np.concatenate(Fs, axis=0)
    v = np.isfinite(Y)
    return Y[v], np.nan_to_num(F[v], nan=0.0)


def _center_channels(res, N, K):
    """Vectorized center-history mean & gain readouts, shaped [N sources, K timescales]."""
    t1, t2 = res.theta1, res.theta2
    eta1 = t1[0]
    eta2 = min(t2[0], -res.eta2_min)
    v_eff = -1.0 / (2.0 * eta2)
    v = max(v_eff - res.sigma ** 2, res.var_min)
    d_mean = (v_eff * t1 + 2.0 * eta1 * v_eff ** 2 * t2)[1:].reshape(N, K)
    d_gain = ((2.0 * v_eff ** 2 * t2) / v)[1:].reshape(N, K)
    return d_mean, d_gain


def fit_multiscale_connectome(X_list, neuron_names, taus_s=(0.5, 1, 2, 5, 10, 20, 30),
                              tau0_s=3.0, horizon=1, target_mode="next", ridge=1e-2,
                              sigma_frac=0.0, fps=4.0, seed=0) -> MultiScaleResult:
    """Fit the multi-timescale connectome. ``sigma_frac`` (x sd(Y)) enables denoising."""
    N = len(neuron_names)
    taus_s = list(taus_s)
    K = len(taus_s)
    fast = np.array([t <= tau0_s for t in taus_s])
    mats = {c: np.zeros((N, N)) for c in
            ("mean_fast", "gain_slow", "mean_all", "gain_all", "tail_slow")}

    for j in range(N):
        Y, F = _pool_target(X_list, j, taus_s, fps, horizon, target_mode)
        if len(Y) < 3 * (N * K + 1):
            continue
        Fc = F - F.mean(axis=0, keepdims=True)
        Psi = np.concatenate([np.ones((len(Y), 1)), Fc], axis=1)
        sigma = sigma_frac * (np.std(Y) or 1.0)
        res = QuadraticScoreMatcher(sigma=sigma, ridge=ridge, seed=seed).fit(Y, Psi)
        d_mean, d_gain = _center_channels(res, N, K)     # [N, K]

        mats["mean_all"][j, :] = np.sum(np.abs(d_mean), axis=1)
        mats["gain_all"][j, :] = np.sum(np.abs(d_gain), axis=1)
        mats["mean_fast"][j, :] = np.sum(np.abs(d_mean[:, fast]), axis=1)
        mats["gain_slow"][j, :] = np.sum(np.abs(d_gain[:, ~fast]), axis=1)
        # slow-tail proxy: |gain| weighted toward slow band already; keep gain_slow as tail too
        mats["tail_slow"][j, :] = np.sum(np.abs(d_gain[:, ~fast]), axis=1)

    for c in mats:
        np.fill_diagonal(mats[c], 0.0)
    return MultiScaleResult(neuron_names=list(neuron_names), taus_s=taus_s,
                            tau0_s=tau0_s, matrices=mats)


def sigma_ledger(X_list, neuron_names, taus_s=(0.5, 1, 2, 5, 10, 20, 30),
                 tau0_s=3.0, sigma_fracs=(0.0, 0.25, 0.5, 1.0), **kw):
    """Fit the multiscale connectome across denoising sigmas; return the gain_slow matrices
    and their pairwise stability (Spearman of off-diagonal |gain|)."""
    from scipy.stats import spearmanr
    results = {}
    for sf in sigma_fracs:
        results[sf] = fit_multiscale_connectome(X_list, neuron_names, taus_s=taus_s,
                                                tau0_s=tau0_s, sigma_frac=sf, **kw)
    base = results[sigma_fracs[0]].matrices["gain_slow"]
    n = base.shape[0]
    off = ~np.eye(n, dtype=bool)
    stab = {}
    for sf in sigma_fracs[1:]:
        g = results[sf].matrices["gain_slow"]
        rho, _ = spearmanr(np.abs(base[off]), np.abs(g[off]))
        stab[sf] = float(rho)
    return results, stab
