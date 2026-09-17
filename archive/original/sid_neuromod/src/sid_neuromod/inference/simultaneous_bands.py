r"""Simultaneous sup-t bands over a directed multilag graph (Section 10.4, Algorithm 2).

Given a stacked readout vector ``r`` with covariance ``C``:

  1. draw :math:`Z^{(b)} \sim \mathcal{N}(0, \mathrm{Corr}(C))`;
  2. critical value :math:`q_{1-\alpha} = \mathrm{quantile}_{1-\alpha}(\max_m |Z_m|)`;
  3. band :math:`r_m \pm q_{1-\alpha}\sqrt{C_{mm}}`.

An entry is significant if its simultaneous band excludes zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.linalg import corr_from_cov, nearest_psd
from ..utils.rng import get_rng


@dataclass
class BandResult:
    estimate: np.ndarray
    se: np.ndarray
    z_score: np.ndarray
    crit: float
    ci_low: np.ndarray
    ci_high: np.ndarray
    significant: np.ndarray

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate.tolist(),
            "se": self.se.tolist(),
            "z_score": self.z_score.tolist(),
            "crit": float(self.crit),
            "ci_low": self.ci_low.tolist(),
            "ci_high": self.ci_high.tolist(),
            "significant": self.significant.tolist(),
        }


def sup_t_critical(C: np.ndarray, alpha: float = 0.05, n_mc: int = 10000,
                   rng=None) -> float:
    """Monte-Carlo sup-t critical value from the estimated correlation matrix."""
    rng = get_rng(rng)
    corr = corr_from_cov(C)
    corr = nearest_psd(corr, eps=1e-10)
    # unit-diagonal after PSD projection
    d = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    corr = corr / np.outer(d, d)
    L = np.linalg.cholesky(corr + 1e-12 * np.eye(corr.shape[0]))
    Z = rng.standard_normal((n_mc, corr.shape[0])) @ L.T
    maxabs = np.max(np.abs(Z), axis=1)
    return float(np.quantile(maxabs, 1.0 - alpha))


def simultaneous_bands(r: np.ndarray, C: np.ndarray, alpha: float = 0.05,
                       n_mc: int = 10000, rng=None) -> BandResult:
    """Compute simultaneous sup-t bands for readout vector ``r`` with covariance ``C``."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    C = np.asarray(C, dtype=float)
    se = np.sqrt(np.clip(np.diag(C), 0.0, None))
    crit = sup_t_critical(C, alpha=alpha, n_mc=n_mc, rng=rng)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, r / se, 0.0)
    lo = r - crit * se
    hi = r + crit * se
    sig = (lo > 0) | (hi < 0)
    return BandResult(estimate=r, se=se, z_score=z, crit=crit,
                      ci_low=lo, ci_high=hi, significant=sig)
