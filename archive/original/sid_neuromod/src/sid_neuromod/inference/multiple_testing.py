"""Multiple-testing utilities: Benjamini-Hochberg FDR and two-sided p-values."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def two_sided_pvalues(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 2.0 * norm.sf(np.abs(z))


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05):
    """Return ``(qvalues, rejected)`` under Benjamini-Hochberg FDR control."""
    p = np.asarray(pvals, dtype=float)
    n = p.shape[0]
    if n == 0:
        return p.copy(), np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity (cumulative min from the right)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    qvals = np.empty_like(q)
    qvals[order] = q
    rejected = qvals <= alpha
    return qvals, rejected
