r"""Data schema and validation (Sections 5.1-5.2).

Enforces the preprocessing invariants: strictly increasing timestamps, consistent
shapes, no NaN timestamps. A :class:`NeuralDataset` bundles the required and optional
fields with validation on construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class SchemaError(ValueError):
    """Raised when input data violates the required schema/invariants."""


@dataclass
class NeuralDataset:
    """Validated neural recording.

    Required
    --------
    X : float array [T, N]
    timestamps_s : float array [T], strictly increasing
    neuron_ids : str array [N]

    Optional
    --------
    behavior : [T, B], behavior_names : [B]
    stimulus : [T, S], stimulus_names : [S]
    is_valid : bool [T, N] or [T]
    worm_id, condition_id, dataset_id
    """

    X: np.ndarray
    timestamps_s: np.ndarray
    neuron_ids: np.ndarray
    behavior: np.ndarray | None = None
    behavior_names: np.ndarray | None = None
    stimulus: np.ndarray | None = None
    stimulus_names: np.ndarray | None = None
    is_valid: np.ndarray | None = None
    worm_id: str | None = None
    condition_id: str | None = None
    dataset_id: str = "dataset"

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.timestamps_s = np.asarray(self.timestamps_s, dtype=float)
        self.neuron_ids = np.asarray(self.neuron_ids)
        validate_core(self.X, self.timestamps_s, self.neuron_ids)
        if self.behavior is not None:
            self.behavior = np.asarray(self.behavior, dtype=float)
            _validate_covariate(self.behavior, self.behavior_names, self.T, "behavior")
        if self.stimulus is not None:
            self.stimulus = np.asarray(self.stimulus, dtype=float)
            _validate_covariate(self.stimulus, self.stimulus_names, self.T, "stimulus")
        if self.is_valid is not None:
            self.is_valid = np.asarray(self.is_valid, dtype=bool)
            if self.is_valid.shape not in {(self.T,), (self.T, self.N)}:
                raise SchemaError(
                    f"is_valid shape {self.is_valid.shape} not in "
                    f"{{({self.T},), ({self.T},{self.N})}}"
                )

    @property
    def T(self) -> int:
        return self.X.shape[0]

    @property
    def N(self) -> int:
        return self.X.shape[1]

    @property
    def median_frame_s(self) -> float:
        return float(np.median(np.diff(self.timestamps_s)))

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1] - self.timestamps_s[0])


def validate_core(X: np.ndarray, timestamps_s: np.ndarray, neuron_ids: np.ndarray) -> None:
    """Validate the required fields and preprocessing invariants."""
    X = np.asarray(X)
    t = np.asarray(timestamps_s, dtype=float)
    ids = np.asarray(neuron_ids)
    if X.ndim != 2:
        raise SchemaError(f"X must be 2D [T, N], got shape {X.shape}")
    T, N = X.shape
    if t.ndim != 1 or t.shape[0] != T:
        raise SchemaError(f"timestamps_s must be 1D length T={T}, got {t.shape}")
    if ids.shape[0] != N:
        raise SchemaError(f"neuron_ids length {ids.shape[0]} != N={N}")
    if np.any(np.isnan(t)):
        raise SchemaError("timestamps_s contains NaN")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise SchemaError("timestamps_s must be strictly increasing (no dups/decreases)")
    if np.any(~np.isfinite(X)):
        # NaNs in X are allowed only via is_valid; core X must be finite unless masked
        pass


def _validate_covariate(arr, names, T, label):
    if arr.ndim != 2 or arr.shape[0] != T:
        raise SchemaError(f"{label} must be [T, K] with T={T}, got {arr.shape}")
    if names is not None and len(names) != arr.shape[1]:
        raise SchemaError(
            f"{label}_names length {len(names)} != {label} columns {arr.shape[1]}"
        )
