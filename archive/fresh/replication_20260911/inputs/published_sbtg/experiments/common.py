#!/usr/bin/env python3
"""Shared safe I/O and evaluation helpers for robustness experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import precision_recall_curve, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "reference_snapshot/prepared_data/full_traces_imputed"
DEFAULT_SBTG = PROJECT_ROOT / "results/paper/sbtg_lag_matrices.npz"
DEFAULT_CONNECTOME = PROJECT_ROOT / "reference_data/connectome"
DEFAULT_RANDI = PROJECT_ROOT / "reference_data/functional_atlas/aligned_atlas_wild_type.npz"
DEFAULT_MONOAMINE = (
    PROJECT_ROOT
    / "reference_data/modulatory_atlas/edge_lists/edgelist_MA_classes.csv"
)
LAGS = (1, 2, 3, 5, 8, 10, 15, 20)


def load_dataset(dataset_dir: Path) -> Tuple[List[np.ndarray], List[str], pd.DataFrame]:
    """Load prepared traces and row metadata without claiming rows are worms."""
    dataset_dir = Path(dataset_dir)
    safe_file = dataset_dir / "traces.npz"
    with np.load(safe_file, allow_pickle=False) as raw:
        values = np.asarray(raw["values"], dtype=np.float64)
        missing = np.asarray(raw["missing"], dtype=bool)
        offsets = np.asarray(raw["offsets"], dtype=np.int64)
    if values.shape != missing.shape or offsets.ndim != 1:
        raise ValueError("Prepared trace archive has an invalid shape")
    values = values.copy()
    values[missing] = np.nan
    traces = [values[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    with (dataset_dir / "standardization.json").open() as handle:
        meta = json.load(handle)
    names = [str(x) for x in meta["node_order"]]
    segment_file = dataset_dir / "segments.csv"
    if segment_file.exists():
        rows = pd.read_csv(segment_file)
    else:
        rows = pd.DataFrame({"composite_row": np.arange(len(traces))})
    rows = rows.reset_index(drop=True)
    rows.insert(0, "composite_row", np.arange(len(rows))) if "composite_row" not in rows else None
    if len(rows) != len(traces):
        rows = pd.DataFrame({"composite_row": np.arange(len(traces))})
    return traces, names, rows


def reorder_square(matrix: np.ndarray, source_names: Sequence[str], target_names: Sequence[str],
                   fill: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder a square matrix to target order and return its observed-node mask."""
    source = {str(name): i for i, name in enumerate(source_names)}
    n = len(target_names)
    out = np.full((n, n), fill, dtype=np.asarray(matrix).dtype)
    present = np.array([str(name) in source for name in target_names], dtype=bool)
    target_idx = np.where(present)[0]
    source_idx = np.array([source[str(target_names[i])] for i in target_idx])
    out[np.ix_(target_idx, target_idx)] = np.asarray(matrix)[np.ix_(source_idx, source_idx)]
    observed = np.outer(present, present)
    return out, observed


def load_sbtg(path: Path = DEFAULT_SBTG) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[str]]:
    data = np.load(path, allow_pickle=False)
    names = [str(x) for x in data["neuron_names"]]
    mu, sig = {}, {}
    for lag in LAGS:
        if f"mu_hat_lag{lag}" in data:
            mu[lag] = np.asarray(data[f"mu_hat_lag{lag}"], dtype=float)
            key = f"significant_lag{lag}"
            sig[lag] = np.asarray(
                data[key] if key in data.files else np.zeros_like(mu[lag]),
                dtype=bool,
            )
    return mu, sig, names


def load_reference_networks(neuron_names: Sequence[str], connectome_dir: Path = DEFAULT_CONNECTOME,
                            randi_file: Path = DEFAULT_RANDI) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load Cook and canonically labeled Randi networks in SBTG order."""
    connectome_dir = Path(connectome_dir)
    with (connectome_dir / "nodes.json").open() as handle:
        cook_names = json.load(handle)
    cook = np.load(connectome_dir / "A_chem.npy") + np.load(connectome_dir / "A_gap.npy")
    cook, cook_observed = reorder_square(cook, cook_names, neuron_names)
    cook_labels = cook != 0
    cook_mask = cook_observed & ~np.eye(len(neuron_names), dtype=bool)

    randi_raw = np.load(randi_file, allow_pickle=False)
    randi_names = [str(x) for x in randi_raw["neuron_order"]]
    positive = np.asarray(randi_raw["q"]) < 0.05
    negative = (np.asarray(randi_raw["q_eq"]) < 0.05) & ~positive
    positive, randi_observed = reorder_square(positive, randi_names, neuron_names)
    negative, _ = reorder_square(negative, randi_names, neuron_names)
    randi_mask = randi_observed & (positive.astype(bool) | negative.astype(bool))
    randi_mask &= ~np.eye(len(neuron_names), dtype=bool)
    return {
        "Cook": (cook_labels.astype(bool), cook_mask),
        "Randi": (positive.astype(bool), randi_mask),
    }


def normalize_mono_name(name: object) -> str:
    """Normalize an already class-level monoamine name without merging it."""
    return str(name).upper().strip()


def load_monoamine_networks(neuron_names: Sequence[str], csv_path: Path = DEFAULT_MONOAMINE) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path, header=None, names=["source", "target", "transmitter", "receptor"])
    df["source"] = df["source"].map(normalize_mono_name)
    df["target"] = df["target"].map(normalize_mono_name)
    lookup = {str(name): i for i, name in enumerate(neuron_names)}
    out = {}
    for transmitter in ("dopamine", "serotonin", "tyramine", "octopamine"):
        adj = np.zeros((len(neuron_names), len(neuron_names)), dtype=bool)
        for row in df.loc[df["transmitter"] == transmitter].itertuples():
            if row.source in lookup and row.target in lookup:
                target, source = lookup[row.target], lookup[row.source]
                if target != source:
                    adj[target, source] = True
        out[transmitter] = adj
    return out


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels.astype(int), scores))


def best_f1(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0:
        return 0.0, float("nan")
    precision, recall, thresholds = precision_recall_curve(labels.astype(int), scores)
    values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.nanargmax(values))
    # precision_recall_curve adds a terminal point with no associated threshold.
    threshold = float(thresholds[best]) if best < len(thresholds) else float("inf")
    return float(values[best]), threshold


def partial_rank_correlation(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Pearson correlation of ranked residuals after linear control adjustment."""
    xr, yr, cr = rankdata(x), rankdata(y), rankdata(control)
    design = np.column_stack([np.ones(len(cr)), cr])
    x_resid = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_resid = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return float("nan")
    return float(np.corrcoef(x_resid, y_resid)[0, 1])
