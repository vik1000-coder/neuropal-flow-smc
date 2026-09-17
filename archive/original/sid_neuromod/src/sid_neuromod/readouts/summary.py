r"""Neuromodulator summary scores (Section 9.4).

Given directed readout kernels over timescales, compute:

  * Fast mean mass:   :math:`M^{fast}_{i\leftarrow j} = \sum_{\tau\le\tau_0}|D^{mean}(\tau)|`
  * Slow distributional mass:
    :math:`G^{slow}_{i\leftarrow j} = \sum_{\tau>\tau_0}(|D^{gain}| + |D^{tail}| + |D^{cov}|)`
  * Candidate neuromodulator index:
    :math:`\mathrm{NMI} = G^{slow} / (G^{slow} + M^{fast} + \epsilon)`.

NMI is a *ranking statistic*, not proof of a neuromodulator.
"""
from __future__ import annotations

import numpy as np


def fast_mean_mass(timescales, d_mean, tau0: float) -> float:
    timescales = np.asarray(timescales, dtype=float)
    d_mean = np.asarray(d_mean, dtype=float)
    fast = timescales <= tau0
    return float(np.sum(np.abs(d_mean[fast])))


def slow_distributional_mass(timescales, d_gain, d_tail=None, d_cov=None,
                             tau0: float = 5.0) -> float:
    timescales = np.asarray(timescales, dtype=float)
    slow = timescales > tau0
    total = np.abs(np.asarray(d_gain, dtype=float))
    if d_tail is not None:
        total = total + np.abs(np.asarray(d_tail, dtype=float))
    if d_cov is not None:
        total = total + np.abs(np.asarray(d_cov, dtype=float))
    return float(np.sum(total[slow]))


def nmi(timescales, d_mean, d_gain, d_tail=None, d_cov=None,
        tau0: float = 5.0, eps: float = 1e-8) -> float:
    """Candidate neuromodulator index in ``[0, 1]``."""
    m_fast = fast_mean_mass(timescales, d_mean, tau0)
    g_slow = slow_distributional_mass(timescales, d_gain, d_tail, d_cov, tau0)
    return g_slow / (g_slow + m_fast + eps)
