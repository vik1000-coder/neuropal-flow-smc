r"""Stability-objective hyperparameter tuning (self-supervised, no label peeking).

We select hyperparameters that maximize the **reproducibility** of the recovered graph
across disjoint splits of the worms: split the worms into two halves, fit the connectome
on each, and score the agreement of the off-diagonal ``|coupling|`` (Spearman rank
correlation + top-k edge Jaccard). This never touches the ground-truth connectome, so it
cannot overfit the evaluation — and it is exactly the "does more tuning under a stability
objective help?" question.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _agreement(A, B, top_frac=0.1):
    """Spearman + top-k Jaccard agreement of two coupling matrices (off-diagonal)."""
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    a, b = np.abs(A[off]), np.abs(B[off])
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0, 0.0
    rho = spearmanr(a, b).statistic
    k = max(1, int(top_frac * len(a)))
    ta = set(np.argsort(a)[-k:]); tb = set(np.argsort(b)[-k:])
    jac = len(ta & tb) / len(ta | tb)
    return float(rho), float(jac)


def split_half_stability(fit_matrix, X_list, n_splits=5, seed=0, top_frac=0.1):
    """Mean split-half stability of ``fit_matrix(X_subset) -> [N,N]`` over worm splits.

    Returns dict with mean/sd of Spearman and Jaccard agreement.
    """
    rng = np.random.default_rng(seed)
    W = len(X_list)
    rhos, jacs = [], []
    for _ in range(n_splits):
        perm = rng.permutation(W)
        h = W // 2
        A = fit_matrix([X_list[i] for i in perm[:h]])
        B = fit_matrix([X_list[i] for i in perm[h:2 * h]])
        rho, jac = _agreement(A, B, top_frac)
        rhos.append(rho); jacs.append(jac)
    return {"spearman_mean": float(np.mean(rhos)), "spearman_sd": float(np.std(rhos)),
            "jaccard_mean": float(np.mean(jacs)), "jaccard_sd": float(np.std(jacs))}


def tune_by_stability(fit_matrix_for, X_list, grid, n_splits=5, seed=0, top_frac=0.1):
    """Select the config in ``grid`` (list of dicts) maximizing split-half stability.

    ``fit_matrix_for(config)`` returns a function ``X_subset -> [N,N]``. Returns
    ``(best_config, table)`` where table is a list of (config, stability) dicts.
    """
    table = []
    for cfg in grid:
        fm = fit_matrix_for(cfg)
        stab = split_half_stability(fm, X_list, n_splits=n_splits, seed=seed,
                                    top_frac=top_frac)
        table.append({**cfg, **stab})
    best = max(table, key=lambda r: r["spearman_mean"])
    return best, table
