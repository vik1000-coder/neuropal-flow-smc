r"""Ridge autoregression baseline (mean-channel).

Fits a ridge regression of the target on raw lagged activity (own and, optionally,
cross-neuron). Reports held-out :math:`R^2`. For the hidden thermostat this baseline's
:math:`R^2` is ~0 by construction (the conditional mean is exactly 0) — the point of
Section 12.1 / Milestone 7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


def build_lag_matrix(x: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Xlag, valid_start)`` where row ``t`` holds lags ``x[t-1..t-n_lags]``."""
    x = np.asarray(x, dtype=float).ravel()
    T = x.shape[0]
    cols = [x[n_lags - 1 - j: T - 1 - j] for j in range(n_lags)]
    Xlag = np.column_stack(cols)  # aligned to targets x[n_lags:]
    return Xlag, n_lags


@dataclass
class RidgeARResult:
    r2: float
    alpha: float
    coef: np.ndarray


def fit_ridge_ar(x: np.ndarray, n_lags: int = 12, alpha: float = 1.0,
                 train_frac: float = 0.7) -> RidgeARResult:
    """Fit ridge AR predicting next value from ``n_lags`` past values; held-out R2."""
    x = np.asarray(x, dtype=float).ravel()
    Xlag, start = build_lag_matrix(x, n_lags)
    y = x[start:]
    n = len(y)
    ntr = int(train_frac * n)
    model = Ridge(alpha=alpha)
    model.fit(Xlag[:ntr], y[:ntr])
    yhat = model.predict(Xlag[ntr:])
    yte = y[ntr:]
    ss_res = np.sum((yte - yhat) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return RidgeARResult(r2=float(r2), alpha=alpha, coef=model.coef_)
