r"""Data loaders for ``.npz`` (and mock generation). Section 5.1.

Accepts ``.npz`` natively; ``.h5``/``.zarr``/``.parquet`` are recognized but require the
optional dependency and are left as documented extension points.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.rng import get_rng
from .schema import NeuralDataset


def load_npz(path: str | Path) -> NeuralDataset:
    """Load a neural dataset from an ``.npz`` archive."""
    d = np.load(path, allow_pickle=True)
    kw = {}
    for opt in ("behavior", "behavior_names", "stimulus", "stimulus_names", "is_valid"):
        if opt in d:
            kw[opt] = d[opt]
    for scalar in ("worm_id", "condition_id", "dataset_id"):
        if scalar in d:
            kw[scalar] = str(d[scalar])
    return NeuralDataset(
        X=d["X"], timestamps_s=d["timestamps_s"], neuron_ids=d["neuron_ids"], **kw
    )


def load_dataset(path: str | Path) -> NeuralDataset:
    path = Path(path)
    if path.suffix == ".npz":
        return load_npz(path)
    raise NotImplementedError(
        f"loader for {path.suffix} not implemented in MVP; use .npz"
    )


def save_npz(path: str | Path, ds: NeuralDataset) -> None:
    arrs = {"X": ds.X, "timestamps_s": ds.timestamps_s, "neuron_ids": ds.neuron_ids,
            "dataset_id": ds.dataset_id}
    if ds.behavior is not None:
        arrs["behavior"] = ds.behavior
        if ds.behavior_names is not None:
            arrs["behavior_names"] = ds.behavior_names
    if ds.stimulus is not None:
        arrs["stimulus"] = ds.stimulus
    np.savez(path, **arrs)


def make_mock_elegans(T: int = 3000, N: int = 6, fs: float = 4.0, seed: int = 0,
                      dataset_id: str = "mock_elegans") -> NeuralDataset:
    """Create a small mock C. elegans-like dataset with SV-driven coupling + behavior.

    A couple of latent slow "neuromodulator" factors set the gain of downstream
    neurons; one neuron drives another's mean quickly. Behavior = velocity proxy.
    """
    rng = get_rng(seed)
    t = np.arange(T) / fs
    # latent slow gain factor (AR(1), slow)
    g = np.zeros(T)
    for k in range(1, T):
        g[k] = 0.98 * g[k - 1] + 0.2 * rng.standard_normal()
    X = np.zeros((T, N))
    # neuron 0: driver with its own AR mean dynamics
    for k in range(1, T):
        X[k, 0] = 0.6 * X[k - 1, 0] + 0.5 * rng.standard_normal()
    # neuron 1: fast mean-driven by neuron 0
    X[1:, 1] = 0.7 * X[:-1, 0] + 0.5 * rng.standard_normal(T - 1)
    # neurons 2..: gain-modulated noise (variance set by slow factor g) + weak mean drive
    for j in range(2, N):
        drive = 0.2 * np.roll(X[:, 0], 1)
        X[:, j] = drive + np.exp(0.5 * g) * 0.6 * rng.standard_normal(T)
    behavior = np.column_stack([
        np.cumsum(0.02 * rng.standard_normal(T)),  # position-ish
        g + 0.1 * rng.standard_normal(T),          # a behavior correlated with the factor
    ])
    neuron_ids = np.array([f"N{j:02d}" for j in range(N)])
    return NeuralDataset(
        X=X, timestamps_s=t, neuron_ids=neuron_ids,
        behavior=behavior, behavior_names=np.array(["velocity", "state"]),
        dataset_id=dataset_id,
    )
