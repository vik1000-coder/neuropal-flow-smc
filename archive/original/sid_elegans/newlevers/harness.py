r"""Shared inference for the new-levers analyses — ONE validated implementation of the
statistics every script needs, so the eight levers cannot disagree on how a CI or a p-value
is computed.

Provides:
  * ``fit_all_lags`` / ``channel_matrix`` — build per-channel [N,N] estimator matrices at each
    lag (SID closed-form), with 'tail' = |tail_hi| + |tail_lo|.
  * ``worm_bootstrap`` — resample WORMS with replacement, recompute a scalar stat -> percentile CI.
  * ``circshift_p`` — circular-shift surrogate p-value for a scalar stat (destroys cross-neuron
    lag alignment AND the global mode; matches the biolag null).
  * ``group_perm_test`` — permutation test for a difference in mean between two neuron groups
    (e.g. receptor-expressing vs variance-matched controls).
  * ``auroc_curve`` — AUROC of a channel vs a target over the lag grid (restricted mask).
All conventions: X_list = list of [T,N], matrices [post,pre], fps=4.0, LAGS = config.LAG_FRAMES.
"""
from __future__ import annotations

import numpy as np

from sid_elegans.biolag import config as C
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.evaluate import score_matrix

LAGS = C.LAG_FRAMES


# --------------------------------------------------------------------------------------
# estimator matrices
# --------------------------------------------------------------------------------------
def channel_matrix(res, channel: str) -> np.ndarray:
    """[N,N] matrix for a channel from a ConnectomeResult; 'tail' = |tail_hi|+|tail_lo|."""
    if channel == "tail":
        return np.abs(res.matrices["tail_hi"]) + np.abs(res.matrices["tail_lo"])
    return res.matrices[channel]


def fit_all_lags(X_list, names, lags=LAGS, channels=("mean", "gain", "tail"), **fit_kwargs):
    """Return ``{lag: {channel: [N,N]}}`` fitting the SID estimator once per lag."""
    out = {}
    for L in lags:
        res = fit_distributional_connectome(X_list, names, lag=L, **fit_kwargs)
        out[L] = {c: channel_matrix(res, c) for c in channels}
    return out


# --------------------------------------------------------------------------------------
# scoring vs a target
# --------------------------------------------------------------------------------------
def auroc_at_lag(M, target, mask=None) -> float:
    """AUROC of |offdiag(M)| vs (target>0), optionally restricted to ``mask`` (a bool [N,N]).

    ``mask`` lets us confine scoring to atlas-confirmed edges (avoids fake negatives).
    """
    if mask is None:
        return score_matrix(M, target)["auroc"]
    Ma = np.array(M, float); Ta = np.array(target, float)
    off = ~np.eye(Ma.shape[0], dtype=bool)
    m = mask & off
    s = np.abs(Ma[m]); y = (Ta[m] > 0).astype(int)
    good = np.isfinite(s) & np.isfinite(Ta[m])
    s, y = s[good], y[good]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def auroc_curve(X_list, names, target, channel="gain", mask=None, lags=LAGS, **fit_kwargs):
    """AUROC(channel vs target) at every lag -> np.ndarray[len(lags)]."""
    fits = fit_all_lags(X_list, names, lags=lags, channels=(channel,), **fit_kwargs)
    return np.array([auroc_at_lag(fits[L][channel], target, mask) for L in lags])


# --------------------------------------------------------------------------------------
# resampling inference on a scalar statistic
# --------------------------------------------------------------------------------------
def worm_bootstrap(stat_fn, X_list, n_boot=200, rng=None, throttle=0.0):
    """Percentile CI of ``stat_fn(X_subset)`` resampling WORMS with replacement.

    Returns ``dict(obs, mean, lo, hi, sd, n_boot)``. ``stat_fn`` maps a list of worm arrays
    to a float (NaN allowed; dropped from the CI).
    """
    import time
    rng = rng or np.random.default_rng(0)
    obs = float(stat_fn(X_list))
    vals = []
    W = len(X_list)
    for _ in range(n_boot):
        idx = rng.integers(0, W, W)
        v = stat_fn([X_list[i] for i in idx])
        if np.isfinite(v):
            vals.append(float(v))
        if throttle:
            time.sleep(throttle)
    vals = np.array(vals)
    if len(vals) == 0:
        return dict(obs=obs, mean=np.nan, lo=np.nan, hi=np.nan, sd=np.nan, n_boot=0)
    return dict(obs=obs, mean=float(vals.mean()), lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)), sd=float(vals.std()), n_boot=len(vals))


def _circshift(X_list, rng):
    """Per-neuron independent circular roll (destroys cross-neuron lag alignment + global mode)."""
    out = []
    for X in X_list:
        T = X.shape[0]
        Xs = np.empty_like(X)
        for i in range(X.shape[1]):
            Xs[:, i] = np.roll(X[:, i], int(rng.integers(1, T)))
        out.append(Xs)
    return out


def circshift_p(stat_fn, X_list, n_surr=200, rng=None, throttle=0.0):
    """One-sided surrogate p that ``stat_fn`` exceeds the circular-shift null.

    Returns ``dict(obs, p, null_mean, null_sd, z, n_surr)``. p = (#surr >= obs + 1)/(n+1).
    """
    import time
    rng = rng or np.random.default_rng(0)
    obs = float(stat_fn(X_list))
    null = []
    for _ in range(n_surr):
        v = stat_fn(_circshift(X_list, rng))
        if np.isfinite(v):
            null.append(float(v))
        if throttle:
            time.sleep(throttle)
    null = np.array(null)
    if len(null) == 0:
        return dict(obs=obs, p=np.nan, null_mean=np.nan, null_sd=np.nan, z=np.nan, n_surr=0)
    p = (int((null >= obs).sum()) + 1) / (len(null) + 1)
    sd = null.std()
    z = (obs - null.mean()) / sd if sd > 1e-12 else np.nan
    return dict(obs=obs, p=float(p), null_mean=float(null.mean()), null_sd=float(sd),
                z=float(z), n_surr=len(null))


# --------------------------------------------------------------------------------------
# group difference (expressing vs control) with a permutation p
# --------------------------------------------------------------------------------------
def group_perm_test(values, mask_a, mask_b, n_perm=5000, rng=None):
    """Permutation test for mean(values[a]) - mean(values[b]).

    Labels are shuffled among the union of a and b (finite entries only). Returns
    ``dict(diff, mean_a, mean_b, p_two, p_greater, n_a, n_b)``.
    """
    rng = rng or np.random.default_rng(0)
    v = np.asarray(values, float)
    a = np.asarray(mask_a, bool) & np.isfinite(v)
    b = np.asarray(mask_b, bool) & np.isfinite(v)
    va, vb = v[a], v[b]
    if len(va) == 0 or len(vb) == 0:
        return dict(diff=np.nan, mean_a=np.nan, mean_b=np.nan, p_two=np.nan,
                    p_greater=np.nan, n_a=int(len(va)), n_b=int(len(vb)))
    obs = va.mean() - vb.mean()
    pool = np.concatenate([va, vb]); na = len(va)
    ge = tot = 0; abs_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(pool)
        d = perm[:na].mean() - perm[na:].mean()
        tot += 1
        if d >= obs:
            ge += 1
        if abs(d) >= abs(obs):
            abs_ge += 1
    return dict(diff=float(obs), mean_a=float(va.mean()), mean_b=float(vb.mean()),
                p_two=float((abs_ge + 1) / (tot + 1)), p_greater=float((ge + 1) / (tot + 1)),
                n_a=int(na), n_b=int(len(vb)))
