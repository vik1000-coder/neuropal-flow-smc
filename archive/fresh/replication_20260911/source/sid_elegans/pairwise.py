r"""Pairwise (marginal) distributional connectome with per-edge worm support.

Where the multivariate estimator (``estimator.py``) fits each target on ALL sources at
once — forcing complete-case worms (every neuron present) or imputation — this fits each
directed edge ``i -> j`` on its OWN tiny design ``psi = [1, x_i(t)]``, using every worm that
recorded BOTH ``i`` and ``j`` (a missing neuron is an all-NaN column). No imputation; each
edge uses its maximal real worm support (~21 of 28 here vs 6 complete-case).

Same closed-form quadratic-T conditional score readout as the multivariate estimator
(mean / gain=∂logVar / tail), so the within-fit mean-vs-gain-vs-tail contrast is preserved;
this is the marginal analog of lagged cross-correlation. Computed in vectorized closed form
(batched 4x4 solves over all sources per target) so it is fast enough to bootstrap.

All matrices ``[post, pre] = [target, source]``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

CHANNELS = ("mean", "gain", "tail_hi", "tail_lo")


def _pool_target(X_list, present, j, lag):
    """Pooled (Xnow [R,N], Yvec [R]) for target j at lag, over worms that recorded j.

    Xnow = all sources at time t; Yvec = x_j(t+lag). NaNs kept (masked per-source below).
    """
    nows, ys = [], []
    for w, X in enumerate(X_list):
        if not present[w][j]:
            continue
        T = X.shape[0]
        if T <= lag:
            continue
        nows.append(X[:T - lag])       # sources at t  [T-lag, N]
        ys.append(X[lag:, j])          # target future [T-lag]
    if not nows:
        return None, None
    return np.concatenate(nows, 0), np.concatenate(ys, 0)


def fit_pairwise_connectome(X_list, names, lag, ridge=1e-2,
                            eta2_min=1e-6, var_min=1e-6):
    """Directed pairwise distributional connectome at one lag.

    Returns dict channel -> [N, N] matrix ([post, pre]); entry [j, i] = effect of source i
    on target j, each edge fit on the worms containing both i and j (no imputation).
    """
    N = len(names)
    present = [~np.isnan(X).all(0) for X in X_list]     # [W][N] neuron present in worm
    mats = {c: np.zeros((N, N)) for c in CHANNELS}

    for j in range(N):
        Xnow, Y = _pool_target(X_list, present, j, lag)
        if Xnow is None:
            continue
        yfin = np.isfinite(Y)
        Y0 = np.nan_to_num(Y, nan=0.0)
        X0 = np.nan_to_num(Xnow, nan=0.0)
        V = np.isfinite(Xnow) & yfin[:, None]           # [R,N] valid rows per source
        Vf = V.astype(float)

        # masked sufficient statistics per source (vectorized over N)
        n = Vf.sum(0)                                    # [N]
        Sx = (Vf * X0).sum(0)
        Sxx = (Vf * X0 ** 2).sum(0)
        SY = (Vf * Y0[:, None]).sum(0)
        SYx = (Vf * (Y0[:, None] * X0)).sum(0)
        SYY = (Vf * (Y0 ** 2)[:, None]).sum(0)
        SYxx = (Vf * (Y0[:, None] * X0 ** 2)).sum(0)
        SYYx = (Vf * ((Y0 ** 2)[:, None] * X0)).sum(0)
        SYYxx = (Vf * ((Y0 ** 2)[:, None] * X0 ** 2)).sum(0)

        ok = n >= max(3 * 2, 10)
        mx = np.where(ok, Sx / np.maximum(n, 1), 0.0)    # source mean over valid rows
        # centered moments (z = x - mx) over the valid rows
        Zz = Sxx - n * mx ** 2                            # Σ z^2
        YZ = SYx - mx * SY                                # Σ Y z
        YZZ = SYxx - 2 * mx * SYx + mx ** 2 * SY          # Σ Y z^2
        YY = SYY                                          # Σ Y^2
        YYZ = SYYx - mx * SYY                             # Σ Y^2 z
        YYZZ = SYYxx - 2 * mx * SYYx + mx ** 2 * SYY      # Σ Y^2 z^2

        # A[i] = (1/n) U'U with U=[1, z, 2Y, 2Yz]; symmetric 4x4 (Σz=0 by centering)
        A = np.zeros((N, 4, 4))
        A[:, 0, 0] = n
        A[:, 0, 2] = A[:, 2, 0] = 2 * SY
        A[:, 0, 3] = A[:, 3, 0] = 2 * YZ
        A[:, 1, 1] = Zz
        A[:, 1, 2] = A[:, 2, 1] = 2 * YZ
        A[:, 1, 3] = A[:, 3, 1] = 2 * YZZ
        A[:, 2, 2] = 4 * YY
        A[:, 2, 3] = A[:, 3, 2] = 4 * YYZ
        A[:, 3, 3] = 4 * YYZZ
        A /= np.maximum(n, 1)[:, None, None]
        A += ridge * np.eye(4)[None]
        c = np.tile(np.array([0.0, 0.0, 2.0, 0.0]), (N, 1))
        try:
            theta = -np.linalg.solve(A, c[..., None])[..., 0]   # [N,4]
        except np.linalg.LinAlgError:
            theta = np.stack([-np.linalg.lstsq(A[i], c[i], rcond=None)[0]
                              for i in range(N)])

        th1_0, th1_1 = theta[:, 0], theta[:, 1]              # eta1 = [intercept, source]
        th2_0, th2_1 = theta[:, 2], theta[:, 3]              # eta2 = [intercept, source]
        eta1 = th1_0
        eta2 = np.minimum(th2_0, -eta2_min)
        v_eff = -1.0 / (2.0 * eta2)
        v = np.maximum(v_eff, var_min)
        mu = eta1 * v_eff
        d_mean = v_eff * th1_1 + 2.0 * eta1 * v_eff ** 2 * th2_1
        d_var = 2.0 * v_eff ** 2 * th2_1
        d_gain = d_var / v
        sv = np.sqrt(v)

        # per-target tail thresholds from the target's own valid future values
        yv = Y[yfin]
        q_hi = float(np.nanquantile(yv, 0.90)) if yv.size else 0.0
        q_lo = float(np.nanquantile(yv, 0.10)) if yv.size else 0.0
        a_hi = (q_hi - mu) / sv
        a_lo = (q_lo - mu) / sv
        d_thi = norm.pdf(a_hi) * (d_mean / sv + (q_hi - mu) / (2 * v ** 1.5) * d_var)
        d_tlo = norm.pdf(a_lo) * (-d_mean / sv - (q_lo - mu) / (2 * v ** 1.5) * d_var)

        for c_name, vals in (("mean", d_mean), ("gain", d_gain),
                             ("tail_hi", d_thi), ("tail_lo", d_tlo)):
            row = np.where(ok, vals, 0.0)
            row[~np.isfinite(row)] = 0.0
            mats[c_name][j, :] = row
        for c_name in CHANNELS:
            mats[c_name][j, j] = 0.0
    return mats
