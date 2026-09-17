"""Load the canonical cleaned SBTG dataset (produced by the repo's 01_prepare_data)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent / "SBTG"
DATASETS = REPO / "results" / "intermediate" / "datasets"


def load_traces(dataset: str = "full_traces_imputed"):
    """Return ``(X_list, neuron_names, fps)``.

    ``X_list`` is a list of per-worm ``[T_w, N]`` standardized trace matrices (N=80),
    ``neuron_names`` the shared neuron order, ``fps`` the frame rate (4 Hz).
    """
    d = DATASETS / dataset
    X = np.load(d / "X_segments.npy", allow_pickle=True)
    X_list = [np.asarray(x, dtype=float) for x in X]
    names_file = d / "neuron_names.json"
    if names_file.exists():
        names = json.load(open(names_file))
    else:  # fall back to the node_order stored in standardization.json
        names = json.load(open(d / "standardization.json"))["node_order"]
    names = [str(n).strip().upper() for n in names]
    return X_list, names, 4.0
