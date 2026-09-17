r"""Changepoint detection and confidence sets (Sections 11.6, 13).

The alarm time is the upper confidence endpoint (Proposition 13.1). The lower endpoint
is obtained by a retrospective scan (Proposition 13.2 in spirit): among candidate change
points ``s < alarm`` we locate the point that best explains the accumulated drift of the
dominant channel (the running argmin of its cumulative sum for a positive-drift
channel), which brackets the change from below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels import var_proxy
from .eprocess import EProcess, first_alarm


@dataclass
class ChangeResult:
    alarmed: bool
    alarm_time: int | None
    lower_ci_time: int | None
    upper_ci_time: int | None
    dominant_channel: str | None
    e_trace: np.ndarray
    per_channel: dict


def detect_change(channels: dict[str, np.ndarray], alpha: float = 0.01,
                  eps: float = 0.0, lambda_grid=None, mode: str = "average",
                  eps_map: dict | None = None) -> ChangeResult:
    """Run the anytime-valid detector over PIT-derived channels.

    Parameters
    ----------
    channels : dict[str, array]
        Mean-zero (under null) bounded channel sequences.
    alpha : float
        Anytime-valid level; alarm threshold is ``1/alpha``.
    eps : float
        Default dead-band half-width; overridden per channel by ``eps_map``.
    eps_map : dict or None
        Per-channel dead-band half-widths.
    """
    lam = lambda_grid or EProcess.lambda_grid
    eps_map = eps_map or {}
    # Each channel runs its own dead-band dyadic detector with its variance proxy.
    per_channel = {}
    for name, d in channels.items():
        ep = EProcess(lambda_grid=lam, eps=eps_map.get(name, eps))
        per_channel[name] = ep.run_detector(d, c=var_proxy(name))
    M = np.stack(list(per_channel.values()), axis=0)
    K = M.shape[0]
    agg = M.mean(axis=0) if mode == "average" else (M / K).max(axis=0)
    alarm = first_alarm(agg, alpha=alpha)
    if alarm is None:
        return ChangeResult(False, None, None, None, None, agg, per_channel)

    # dominant channel = the one with the largest e-value at the alarm time
    dom = max(per_channel, key=lambda c: per_channel[c][alarm])
    # lower endpoint: argmin of cumulative sum of dominant channel up to alarm
    # (start of the sustained positive drift that triggered the alarm)
    d = channels[dom][: alarm + 1]
    S = np.cumsum(d)
    lower = int(np.argmin(S)) if S.size else 0
    return ChangeResult(True, int(alarm), lower, int(alarm), dom, agg, per_channel)
