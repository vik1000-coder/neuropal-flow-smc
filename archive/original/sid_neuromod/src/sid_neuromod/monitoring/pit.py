r"""Probability integral transform (PIT) residuals and ECDF recalibration (11.2).

For a target ``i`` with Gaussian conditional,

.. math::

   U_{i,t} = \hat F_i(Y_{i,t}\mid H_t)
           = \Phi\!\left(\frac{Y_{i,t} - \hat\mu_i(t)}{\sqrt{\hat v_i(t)}}\right),

clamped to ``[1e-6, 1-1e-6]``. If the conditional model is correct the PITs are
uniform on ``[0,1]``; deviations are what the monitoring channels test. Recalibration
maps stream PITs through the empirical CDF of a held-out calibration split.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kstest, norm

from ..utils.rng import get_rng

PIT_CLIP = 1e-6


def gaussian_pit(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """PIT values for a Gaussian conditional, clamped away from 0/1."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.sqrt(np.asarray(var, dtype=float))
    u = norm.cdf((y - mu) / sd)
    return np.clip(u, PIT_CLIP, 1.0 - PIT_CLIP)


class ECDFRecalibrator:
    """Recalibrate PITs through a calibration-split ECDF using randomized ranks."""

    def __init__(self, seed: int = 0):
        self.calib_ = None
        self.seed = seed

    def fit(self, u_calib: np.ndarray) -> "ECDFRecalibrator":
        self.calib_ = np.sort(np.asarray(u_calib, dtype=float))
        return self

    def transform(self, u: np.ndarray, rng=None) -> np.ndarray:
        if self.calib_ is None:
            raise RuntimeError("fit the recalibrator first")
        rng = get_rng(rng if rng is not None else self.seed)
        u = np.asarray(u, dtype=float)
        c = self.calib_
        n = c.shape[0]
        # randomized rank: count strictly-less + uniform fraction of ties
        lo = np.searchsorted(c, u, side="left")
        hi = np.searchsorted(c, u, side="right")
        frac = rng.random(u.shape[0])
        rank = lo + frac * (hi - lo)
        out = (rank + rng.random(u.shape[0])) / (n + 1)
        return np.clip(out, PIT_CLIP, 1.0 - PIT_CLIP)


def pit_diagnostics(u: np.ndarray) -> dict:
    """KS statistic/p-value against uniform, plus PIT mean & variance."""
    u = np.asarray(u, dtype=float)
    ks_stat, ks_p = kstest(u, "uniform")
    return {
        "pit_ks_stat": float(ks_stat),
        "pit_ks_pvalue": float(ks_p),
        "pit_mean": float(np.mean(u)),
        "pit_variance": float(np.var(u)),
    }
