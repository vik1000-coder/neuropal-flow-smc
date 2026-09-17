r"""VAR baseline (statsmodels) for the mean-channel functional connectome.

Provides a linear conditional-mean baseline and its held-out NLL / R2, so slow
distributional effects can be contrasted against fast mean-drive effects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VARResult:
    order: int
    r2: float
    heldout_nll: float
    coefs: np.ndarray | None


def fit_var(X: np.ndarray, order: int = 2, train_frac: float = 0.7) -> VARResult:
    """Fit a VAR(order) on the train split; report held-out one-step R2 and NLL."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T, N = X.shape
    ntr = int(train_frac * T)
    try:
        from statsmodels.tsa.api import VAR

        model = VAR(X[:ntr])
        fit = model.fit(order)
        k = fit.k_ar
        # one-step predictions on the test span
        preds = []
        acts = []
        for t in range(max(ntr, k), T):
            hist = X[t - k:t]
            preds.append(fit.forecast(hist, 1)[0])
            acts.append(X[t])
        preds = np.asarray(preds)
        acts = np.asarray(acts)
        resid = acts - preds
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((acts - acts.mean(axis=0)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        sigma = np.maximum(resid.std(axis=0), 1e-8)
        nll = float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2)
                            + 0.5 * (resid / sigma) ** 2))
        coefs = fit.coefs
    except Exception:  # pragma: no cover - statsmodels edge cases
        r2, nll, coefs = 0.0, float("nan"), None
    return VARResult(order=order, r2=float(r2), heldout_nll=nll, coefs=coefs)
