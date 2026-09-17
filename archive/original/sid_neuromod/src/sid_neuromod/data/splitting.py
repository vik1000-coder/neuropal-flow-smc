r"""Contiguous / worm-blocked splits (Sections 5.2, 11.1).

Default four-way contiguous split for monitoring:
  train 50% | calibration 20% | debias 10% | stream 20%.

Splits are contiguous by default (never randomly interleaved), so no target sample
uses future features across a split boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Splits:
    train: slice
    calibration: slice
    debias: slice
    stream: slice

    def as_intervals(self):
        return {
            "train": [int(self.train.start), int(self.train.stop)],
            "calibration": [int(self.calibration.start), int(self.calibration.stop)],
            "debias": [int(self.debias.start), int(self.debias.stop)],
            "test": [int(self.stream.start), int(self.stream.stop)],
        }


def contiguous_splits(T: int, train: float = 0.5, calibration: float = 0.2,
                      debias: float = 0.1, stream: float = 0.2) -> Splits:
    """Four contiguous splits by fraction (fractions need not sum to exactly 1)."""
    total = train + calibration + debias + stream
    train, calibration, debias, stream = (f / total for f in
                                           (train, calibration, debias, stream))
    i1 = int(train * T)
    i2 = i1 + int(calibration * T)
    i3 = i2 + int(debias * T)
    return Splits(slice(0, i1), slice(i1, i2), slice(i2, i3), slice(i3, T))


def train_test_split_contiguous(T: int, train_frac: float = 0.7):
    """Simple two-way contiguous split; returns ``(train_slice, test_slice)``."""
    i = int(train_frac * T)
    return slice(0, i), slice(i, T)


def kfold_contiguous(T: int, k: int = 3):
    """Yield ``(train_idx, test_idx)`` for ``k`` contiguous blocks (test = each block)."""
    bounds = np.linspace(0, T, k + 1).astype(int)
    for j in range(k):
        test = np.arange(bounds[j], bounds[j + 1])
        train = np.concatenate([np.arange(0, bounds[j]), np.arange(bounds[j + 1], T)])
        yield train, test
