"""Behavior/stimulus covariate construction.

Behavior covariates are included directly in psi(t) (Section 7.4), not only as
post-hoc regressions. For the MVP we standardize them and (optionally) add a
first-difference channel.
"""
from __future__ import annotations

import numpy as np

from ..utils.arrays import as_2d


def behavior_features(behavior: np.ndarray, names, include_derivative: bool = True):
    """Return standardized behavior features and their names.

    Parameters
    ----------
    behavior : array [T, B]
    names : sequence of str, length B
    include_derivative : bool
        If True, append a first-difference channel per behavior covariate.
    """
    B = as_2d(behavior)
    names = list(names)
    feats = [B]
    out_names = list(names)
    if include_derivative:
        d = np.zeros_like(B)
        d[1:] = np.diff(B, axis=0)
        feats.append(d)
        out_names += [f"{n}_deriv" for n in names]
    return np.concatenate(feats, axis=1), out_names
