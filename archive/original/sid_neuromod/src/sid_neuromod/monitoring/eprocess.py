r"""Anytime-valid e-processes with dead-band correction (Sections 11.4-11.6).

For a bounded channel :math:`d_t \in [-1,1]` with (near-)zero mean under the null, and
a fixed grid of betting parameters :math:`\Lambda`, the sub-Gaussian e-value over a
window of length ``n`` with sum ``S`` is, under zero drift,

.. math::

   M_\lambda = \exp(\lambda S - \lambda^2 n / 2),
   \qquad M = \frac{1}{|\Lambda|}\sum_{\lambda\in\Lambda} M_\lambda.

Because each increment is bounded in ``[-1,1]`` and mean-zero, ``E[exp(lambda d)] <=
exp(lambda^2/2)`` (Hoeffding), so ``M_t = exp(lambda S_t - lambda^2 t/2)`` is a
non-negative supermartingale started at 1; the grid mixture is too. Ville's inequality
gives ``P(sup_t M_t >= 1/alpha) <= alpha`` — an anytime-valid test.

Dead-band correction (Section 11.5): with a calibration/debias offset the channel may
have residual drift up to ``eps``; we discount the worst-case drift per sign of
``lambda``:

.. math::

   M_\lambda^+ = \exp(\lambda(S - \varepsilon n) - \lambda^2 n/2)\ (\lambda>0),\quad
   M_\lambda^- = \exp(\lambda(S + \varepsilon n) - \lambda^2 n/2)\ (\lambda<0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

DEFAULT_LAMBDA_GRID = (-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0)
DEFAULT_TAU2 = 0.25  # prior variance on the betting parameter for the Robbins mixture


def robbins_log_evalue(S: np.ndarray, length: np.ndarray, c: float = 1.0,
                       eps: float = 0.0, tau2: float = DEFAULT_TAU2) -> np.ndarray:
    r"""Log of the two-sided Robbins normal-mixture e-value with dead band.

    Integrates :math:`\exp(\lambda S - \lambda^2 c L/2)` over a normal prior
    :math:`\lambda\sim\mathcal N(0,\tau^2)`, split by sign so the dead band discounts
    the worst-case drift per sign (Section 11.5). With ``b = cL + 1/\tau^2``:

    .. math::

       M^+ = \tfrac{2}{\sqrt{1+\tau^2 cL}} e^{(S-\varepsilon L)^2/2b}\,
             \Phi\!\big((S-\varepsilon L)/\sqrt b\big),\quad
       M^- = \tfrac{2}{\sqrt{1+\tau^2 cL}} e^{(S+\varepsilon L)^2/2b}\,
             \Phi\!\big(-(S+\varepsilon L)/\sqrt b\big),

    and :math:`M = \tfrac12(M^+ + M^-)`. Each half is a supermartingale for any
    per-step drift in ``[-eps, eps]``, so ``M`` is a valid e-process; unlike a fixed
    :math:`\lambda`-grid it adapts to the (unknown) drift magnitude. At ``eps=0`` this
    reduces to the classic two-sided mixture ``(1+\tau^2 cL)^{-1/2}\exp(S^2/2b)``.
    """
    L = np.asarray(length, dtype=float)
    S = np.asarray(S, dtype=float)
    b = c * L + 1.0 / tau2
    sb = np.sqrt(b)
    base = -0.5 * np.log1p(tau2 * c * L)
    Sp = S - eps * L
    Sm = S + eps * L
    log_mp = np.log(2.0) + base + Sp ** 2 / (2.0 * b) + norm.logcdf(Sp / sb)
    log_mm = np.log(2.0) + base + Sm ** 2 / (2.0 * b) + norm.logcdf(-Sm / sb)
    return logsumexp(np.stack([log_mp, log_mm]), axis=0) + np.log(0.5)


def deadband_epsilon(n_debias: int, n_channels: int, delta: float = 0.01,
                     c: float = 1.0) -> float:
    r""":math:`\varepsilon = \sqrt{2 c \log(2K/\delta)/n_{db}}` (Section 11.5).

    ``c`` is the channel's sub-Gaussian variance proxy; a smaller proxy gives a tighter
    high-probability bound on the debias-offset estimation error, hence a smaller dead
    band. ``c=1`` recovers the plain formula.
    """
    n_debias = max(int(n_debias), 1)
    return float(np.sqrt(2.0 * c * np.log(2.0 * n_channels / delta) / n_debias))


def dyadic_restarts(n: int) -> list[int]:
    """Restart times ``0, 1, 2, 4, 8, ...`` up to ``n`` (Algorithm 3 dyadic ages)."""
    restarts = [0]
    a = 1
    while a < n:
        restarts.append(a)
        a *= 2
    return restarts


@dataclass
class EProcess:
    """Running grid-mixture e-process with dead-band correction.

    Parameters
    ----------
    lambda_grid : sequence of float
        Betting parameters (mix of signs).
    eps : float
        Dead-band half-width (0 => no correction).
    """

    lambda_grid: tuple = DEFAULT_LAMBDA_GRID
    eps: float = 0.0

    def run(self, d: np.ndarray, c: float = 1.0) -> np.ndarray:
        r"""Return the running e-value trace ``M_t`` for a single channel sequence.

        ``c`` is the sub-Gaussian variance proxy of the channel increments (Hoeffding
        proxy ``(b-a)^2/4`` for a channel bounded in ``[a,b]``); the exponent uses
        ``lambda^2 c n / 2``. ``c=1`` is valid for any channel bounded in ``[-1,1]``.
        """
        d = np.asarray(d, dtype=float)
        n = d.shape[0]
        S = np.cumsum(d)
        t = np.arange(1, n + 1, dtype=float)
        lambdas = np.asarray(self.lambda_grid, dtype=float)
        # dead-band shift per sign: positive lambda discounts +eps*t, negative discounts -eps*t
        shift = np.where(lambdas > 0, -self.eps, self.eps)  # sign convention
        # log M_lambda,t = lambda*(S_t + shift_sign*t) - lambda^2 c t /2
        logM = (lambdas[None, :] * (S[:, None] + shift[None, :] * t[:, None])
                - 0.5 * c * lambdas[None, :] ** 2 * t[:, None])
        # mixture over lambda in the linear domain, stable via logsumexp
        m = logM.max(axis=1)
        mix = m + np.log(np.mean(np.exp(logM - m[:, None]), axis=1))
        return np.exp(mix)

    def run_detector(self, d: np.ndarray, c: float = 1.0,
                     tau2: float = DEFAULT_TAU2) -> np.ndarray:
        r"""Dyadic-restart Robbins-mixture e-detector (Algorithm 3).

        Averages dead-band Robbins normal-mixture e-processes restarted at dyadic times
        ``0, 1, 2, 4, ...``. Each restart contributes ``1`` before it starts. A change
        after some restart ``r`` is captured by the e-process running from ``r`` with
        full power, so the dead band no longer buries the signal in the pre-change
        accumulation, and the normal mixture adapts to the (unknown) drift magnitude.
        The average of e-processes is itself an e-process (starts at 1), so Ville's
        inequality still gives the anytime-valid guarantee. ``c`` is the channel's
        sub-Gaussian variance proxy.
        """
        d = np.asarray(d, dtype=float)
        n = d.shape[0]
        if n == 0:
            return np.zeros(0)
        Sc = np.concatenate([[0.0], np.cumsum(d)])  # Sc[k] = sum(d[:k])
        restarts = dyadic_restarts(n)
        D = np.zeros(n)
        for r in restarts:
            contrib = np.ones(n)
            idx = np.arange(r, n)
            length = (idx + 1) - r                      # window length ending at idx
            Sw = Sc[idx + 1] - Sc[r]                    # windowed sum
            contrib[idx] = np.exp(robbins_log_evalue(Sw, length, c=c,
                                                     eps=self.eps, tau2=tau2))
            D += contrib
        return D / len(restarts)

    def run_max(self, channels: dict[str, np.ndarray], mode: str = "average",
                detector: bool = True, c_map: dict | None = None):
        """Aggregate multiple channels into a single e-value trace.

        Each channel uses the dyadic-restart detector (``detector=True``, needed when a
        dead band is active) or the running-from-zero mixture. ``c_map`` gives the
        per-channel variance proxy (default 1). ``mode='average'`` averages the
        per-channel e-processes (formally valid, Section 11.6 default); ``mode='max'``
        takes a Bonferroni-corrected per-time max.
        """
        c_map = c_map or {}
        run = self.run_detector if detector else self.run
        traces = {name: run(d, c=c_map.get(name, 1.0)) for name, d in channels.items()}
        M = np.stack(list(traces.values()), axis=0)  # [K, T]
        K = M.shape[0]
        if mode == "average":
            agg = M.mean(axis=0)
        elif mode == "max":
            agg = (M / K).max(axis=0)
        else:
            raise ValueError("mode must be 'average' or 'max'")
        return agg, traces


def first_alarm(trace: np.ndarray, alpha: float = 0.01) -> int | None:
    """Index of the first time the e-value crosses ``1/alpha`` (or ``None``)."""
    trace = np.asarray(trace, dtype=float)
    thr = 1.0 / alpha
    idx = np.argmax(trace >= thr)
    if trace[idx] >= thr:
        return int(idx)
    return None
