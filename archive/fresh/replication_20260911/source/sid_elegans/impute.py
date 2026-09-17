r"""Random-donor imputation for the multivariate approach.

To run the MULTIVARIATE (partial) estimator on more than the complete-case worms without
the flat-trace confound, fill each missing ``(worm, neuron)`` cell with that neuron's REAL
activity borrowed from a random *donor* worm that recorded it. The donor trace carries a
realistic variance magnitude (so it does not bias the variance/gain channel toward zero the
way mean-imputation does), but it is decoupled from the recipient worm's own dynamics — so
the imputed edges carry no true coupling and dilute toward chance rather than biasing.

A missing neuron is an all-NaN column; sporadic within-trace NaNs (dropout frames) in
present neurons are left untouched (the estimator already handles them). Donor length is
matched to the recipient by tiling/truncation (``np.resize``).
"""
from __future__ import annotations

import numpy as np


def impute_random_donor(X_list, seed: int = 0):
    """Return a copy of ``X_list`` with every missing neuron filled by a random donor.

    For each worm and each all-NaN neuron column, choose (uniformly, without the recipient)
    a donor worm that has that neuron present, and copy its trace, resized to the recipient's
    length. Neurons present nowhere are left all-NaN (cannot be imputed).
    """
    rng = np.random.default_rng(seed)
    present = [~np.isnan(X).all(0) for X in X_list]      # [W][N]
    P = np.array(present)                                 # [W, N] bool
    N = X_list[0].shape[1]
    out = [X.copy() for X in X_list]
    n_imputed = 0
    for w in range(len(out)):
        Lw = out[w].shape[0]
        for k in range(N):
            if present[w][k]:
                continue
            donors = np.where(P[:, k])[0]
            donors = donors[donors != w]
            if donors.size == 0:
                continue                                  # neuron present nowhere else
            d = int(rng.choice(donors))
            col = X_list[d][:, k]
            col = col[np.isfinite(col)]                   # drop donor's own dropout frames
            if col.size == 0:
                continue
            out[w][:, k] = np.resize(col, Lw)             # tile/truncate to recipient length
            n_imputed += 1
    return out, n_imputed
