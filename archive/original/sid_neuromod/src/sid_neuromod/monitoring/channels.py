r"""Bounded martingale-difference monitoring channels (Section 11.3).

All channels are functions of the PIT sequence ``U`` (and, for the lag channel, a
source filter). Under a correct, calibrated conditional law each channel is a
mean-zero bounded sequence, so its running sum is a martingale — the input the
e-process needs.

  * mean:        :math:`d^{mean}_t = 2(U_t - 1/2)`
  * dispersion:  :math:`d^{disp}_t = 6U_t^2 - 6U_t + 1 = \psi_2(U_t)`
  * serial:      :math:`d^{serial}_t = d^{disp}_t\, d^{disp}_{t-1}`
  * lag-kernel:  :math:`d^{lag}_t = d^{disp}_t \cdot \mathrm{clip}((g_t-\bar g)/s_g, -1, 1)`

``psi_2`` is the second shifted Legendre polynomial on ``[0,1]``; it has mean 0 and is
bounded on ``[-1/2, 1]`` for ``U in [0,1]`` (we rescale nothing — bounds are handled by
the e-process which only needs boundedness).
"""
from __future__ import annotations

import numpy as np


def psi2(u: np.ndarray) -> np.ndarray:
    r""":math:`\psi_2(u) = 6u^2 - 6u + 1` (mean-zero dispersion statistic)."""
    u = np.asarray(u, dtype=float)
    return 6.0 * u * u - 6.0 * u + 1.0


def mean_channel(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return 2.0 * (u - 0.5)


def dispersion_channel(u: np.ndarray) -> np.ndarray:
    return psi2(u)


def serial_channel(u: np.ndarray) -> np.ndarray:
    d = psi2(u)
    out = np.zeros_like(d)
    out[1:] = d[1:] * d[:-1]
    return out


def lag_channel(u: np.ndarray, g: np.ndarray, g_mean: float | None = None,
                g_std: float | None = None) -> np.ndarray:
    """Lag-dispersion channel using source filter ``g`` (Section 11.3)."""
    u = np.asarray(u, dtype=float)
    g = np.asarray(g, dtype=float)
    if g_mean is None:
        g_mean = float(np.mean(g))
    if g_std is None:
        g_std = float(np.std(g)) or 1.0
    gc = np.clip((g - g_mean) / g_std, -1.0, 1.0)
    return psi2(u) * gc


CHANNEL_FNS = {
    "mean": mean_channel,
    "dispersion": dispersion_channel,
    "serial": serial_channel,
}

# Hoeffding sub-Gaussian variance proxy (b-a)^2/4 for each channel's range.
# mean: [-1,1] -> 1;  dispersion/serial: psi2 in [-0.5,1] -> (1.5)^2/4 = 0.5625;
# lag: psi2 * clip in [-1,1] -> 1.
CHANNEL_VAR_PROXY = {
    "mean": 1.0,
    "dispersion": 0.5625,
    "serial": 0.5625,
    "lag": 1.0,
}


def var_proxy(channel_name: str) -> float:
    """Return the sub-Gaussian variance proxy for a (possibly suffixed) channel name."""
    for key, c in CHANNEL_VAR_PROXY.items():
        if channel_name.startswith(key):
            return c
    return 1.0
