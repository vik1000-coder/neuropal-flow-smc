"""Moving-block bootstrap for dependent series (used for CIs on scalar statistics)."""
from __future__ import annotations

import numpy as np

from ..utils.rng import get_rng


def moving_block_bootstrap(x: np.ndarray, statistic, block: int = 50,
                           n_boot: int = 200, rng=None) -> np.ndarray:
    """Return bootstrap replicates of ``statistic(resampled_x)``.

    ``x`` may be 1D ``[T]`` or 2D ``[T, ...]``; blocks are drawn along axis 0.
    """
    rng = get_rng(rng)
    x = np.asarray(x)
    T = x.shape[0]
    block = int(min(block, T))
    n_blocks = int(np.ceil(T / block))
    reps = np.empty(n_boot)
    starts_max = T - block + 1
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        reps[b] = statistic(x[idx])
    return reps


def bootstrap_ci(reps: np.ndarray, alpha: float = 0.05):
    lo = float(np.quantile(reps, alpha / 2))
    hi = float(np.quantile(reps, 1 - alpha / 2))
    return lo, hi
