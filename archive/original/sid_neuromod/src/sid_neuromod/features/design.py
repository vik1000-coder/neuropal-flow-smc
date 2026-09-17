r"""Build the causal design matrix ``Psi`` and its :class:`FeatureRegistry`.

Assembles, for a given target neuron, the feature vector :math:`\psi(t)` from:
  * intercept,
  * exponential filter-bank features of each source signal (causal, no leakage),
  * own-history filter features,
  * behavior/stimulus covariates.

Feature scaling is fit on the train interval only; the (centered) design supports the
center-history readout used for kernels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.behavior import behavior_features
from ..features.feature_registry import FeatureRegistry
from ..features.filter_bank import exp_filter_bank
from ..features.history import source_signal
from ..features.scaling import Scaler


@dataclass
class Design:
    Psi: np.ndarray            # [T, P] scaled + centered features
    registry: FeatureRegistry
    scaler: Scaler


def build_design(
    X: np.ndarray,
    timestamps_s: np.ndarray,
    neuron_ids,
    *,
    source_signals=("raw_zscore", "delta"),
    timescales_s=(0.5, 1, 2, 5, 10, 20, 45, 90, 180, 300),
    source_neurons=None,
    include_own_history: bool = True,
    behavior: np.ndarray | None = None,
    behavior_names=None,
    train_slice: slice | None = None,
    center: bool = True,
) -> Design:
    """Construct the design matrix and registry.

    Parameters
    ----------
    source_neurons : sequence of int or None
        Which neuron columns to use as sources (default: all).
    train_slice : slice or None
        Interval used to fit feature scaling (default: whole record).
    center : bool
        Subtract the (train) feature means so readouts are "at the center history".
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(timestamps_s, dtype=float)
    T, N = X.shape
    if source_neurons is None:
        source_neurons = list(range(N))
    timescales_s = list(timescales_s)
    reg = FeatureRegistry()

    feats = [np.ones((T, 1))]
    reg.add_intercept()

    # filter-bank features for each source signal
    for j in source_neurons:
        for kind in source_signals:
            z = source_signal(X[:, j], kind)
            F = exp_filter_bank(z, t, timescales_s)[:, 0, :]  # [T, K]
            feats.append(F)
            for r, tau in enumerate(timescales_s):
                reg.add_filter(str(neuron_ids[j]), kind, tau)

    if include_own_history:
        # own-history: short lags of raw activity for each source (kept small)
        pass  # own filters already included when a neuron is its own source

    if behavior is not None:
        bnames_in = behavior_names
        if bnames_in is None:
            bnames_in = [f"b{i}" for i in range(np.asarray(behavior).shape[1])]
        else:
            bnames_in = list(bnames_in)
        B, bnames = behavior_features(behavior, bnames_in)
        feats.append(B)
        for nm in bnames:
            reg.add_named(f"behavior:{nm}", "behavior")

    Psi_raw = np.concatenate(feats, axis=1)

    tr = train_slice or slice(0, T)
    scaler = Scaler(center_only=center)
    if center:
        # center everything except the intercept column (index 0)
        means = Psi_raw[tr].mean(axis=0)
        means[0] = 0.0  # keep intercept at 1
        Psi = Psi_raw - means
        scaler.mean_ = means
        scaler.std_ = np.ones(Psi_raw.shape[1])
    else:
        Psi = Psi_raw
        scaler.mean_ = np.zeros(Psi_raw.shape[1])
        scaler.std_ = np.ones(Psi_raw.shape[1])
    return Design(Psi=Psi, registry=reg, scaler=scaler)
