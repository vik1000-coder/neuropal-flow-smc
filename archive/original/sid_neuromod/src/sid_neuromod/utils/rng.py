"""Reproducible random number generation.

NumPy's docs recommend ``default_rng`` for modern generation. We centralize seed
handling so every experiment is reproducible from a single integer.
"""
from __future__ import annotations

import numpy as np


def get_rng(seed: int | np.random.Generator | None = 0) -> np.random.Generator:
    """Return a ``numpy.random.Generator``.

    Accepts an int seed, an existing Generator (returned as-is), or ``None``
    (nondeterministic).
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def spawn(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Spawn ``n`` independent child generators from ``rng`` (for parallel work)."""
    return [np.random.default_rng(s) for s in rng.bit_generator._seed_seq.spawn(n)]
