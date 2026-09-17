#!/usr/bin/env python3
"""Evaluate method agreement against anatomical and functional references.

Randi labels are defined explicitly, and the number of evaluated positive and
negative pairs is saved alongside every metric.

Two Randi protocols are reported:

``q_significance``
    A measured pair is positive when q < 0.05 and negative otherwise.

``confirmed_only``
    A pair is positive when q < 0.05 and a confirmed non-connection when
    q_eq < 0.05.  Pairs satisfying both tests after bilateral aggregation, or
    neither test, are excluded.

``confirmed_positive_priority``
    The repository's canonical policy: q < 0.05 positives take priority when
    bilateral aggregation also yields q_eq < 0.05; all other q_eq < 0.05
    pairs are confirmed negatives.

The zero-lag Pearson matrix is rebuilt per recording and then averaged. Missing
samples are assigned the within-recording mean after standardization. The
lag-1 cross-correlation input is reported separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from experiments.common import load_dataset


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "reference_snapshot/prepared_data/full_traces_imputed"
DEFAULT_OUT = ROOT / "results/derived/method_agreement"
DEFAULT_SBTG = ROOT / "results/paper/sbtg_lag_matrices.npz"
DEFAULT_LAGGED = ROOT / "reference_snapshot/benchmarks/lagged_cross_correlation.npz"
DEFAULT_GRANGER = ROOT / "reference_snapshot/benchmarks/granger.npz"
DEFAULT_CONNECTOME = ROOT / "reference_data/connectome"
DEFAULT_ATLAS = ROOT / "reference_data/functional_atlas/aligned_atlas_wild_type.npz"
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET,
        help="Prepared trace directory.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sbtg-file", type=Path, default=DEFAULT_SBTG)
    parser.add_argument("--lagged-file", type=Path, default=DEFAULT_LAGGED)
    parser.add_argument("--granger-file", type=Path, default=DEFAULT_GRANGER)
    parser.add_argument("--connectome-dir", type=Path, default=DEFAULT_CONNECTOME)
    parser.add_argument("--atlas-file", type=Path, default=DEFAULT_ATLAS)
    return parser.parse_args()


def load_npz_matrix(path: Path, key: str) -> Tuple[np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=False)
    return np.asarray(data[key]), [str(x) for x in data["neuron_names"]]


def take_names(
    matrix: np.ndarray, source_names: Iterable[str], target_names: Iterable[str]
) -> np.ndarray:
    source_names = [str(x) for x in source_names]
    indices = {name: i for i, name in enumerate(source_names)}
    return matrix[np.ix_(
        [indices[str(name)] for name in target_names],
        [indices[str(name)] for name in target_names],
    )]


def common_names(*name_lists: Iterable[str]) -> List[str]:
    sets = [set(str(x) for x in names) for names in name_lists]
    return sorted(set.intersection(*sets))


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    scores = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.max(scores))


def evaluate(
    method: str,
    benchmark: str,
    scores: np.ndarray,
    positive: np.ndarray,
    evaluated: np.ndarray,
) -> Dict[str, object]:
    offdiag = ~np.eye(scores.shape[0], dtype=bool)
    mask = offdiag & evaluated & np.isfinite(scores)
    y_true = positive[mask].astype(int)
    y_score = np.abs(scores[mask])
    if np.unique(y_true).size != 2:
        raise ValueError(f"{benchmark} has fewer than two classes")
    return {
        "method": method,
        "benchmark": benchmark,
        "n_pairs": int(mask.sum()),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "best_f1": best_f1(y_true, y_score),
    }


def build_zero_lag_pearson(dataset_dir: Path) -> Tuple[np.ndarray, List[str]]:
    with (dataset_dir / "standardization.json").open() as handle:
        metadata = json.load(handle)
    names = [str(x) for x in metadata["node_order"]]
    segments, loaded_names, _ = load_dataset(dataset_dir)
    if loaded_names != names:
        raise ValueError("Prepared traces and metadata have different neuron orders")
    n = len(names)
    correlations = []
    for segment in segments:
        segment = np.asarray(segment, dtype=float)
        if segment.ndim != 2 or segment.shape[1] != n or segment.shape[0] < 10:
            continue
        # Exclude the last sample to use the same x_t rows as the lag-1
        # windows. Standardizing within each prepared recording row prevents
        # between-row offsets from becoming correlations. Remaining NaNs become
        # the within-row mean.
        x_t = segment[:-1]
        mean = np.nanmean(x_t, axis=0)
        std = np.maximum(np.nanstd(x_t, axis=0), 1e-8)
        x_t = np.nan_to_num((x_t - mean) / std, nan=0.0)
        correlations.append(np.corrcoef(x_t, rowvar=False))
    if not correlations:
        raise ValueError(f"No usable segments found in {dataset_dir}")
    pearson = np.mean(correlations, axis=0)
    np.fill_diagonal(pearson, 0.0)
    return pearson, names


def rank_agreement(
    methods: Dict[str, Tuple[np.ndarray, List[str]]], top_k: int = 100
) -> pd.DataFrame:
    names = common_names(*(item[1] for item in methods.values()))
    offdiag = ~np.eye(len(names), dtype=bool)
    vectors = {
        method: np.abs(take_names(matrix, source_names, names)[offdiag])
        for method, (matrix, source_names) in methods.items()
    }
    rows = []
    method_names = list(vectors)
    for i, left in enumerate(method_names):
        for right in method_names[i + 1 :]:
            rho = float(spearmanr(vectors[left], vectors[right]).statistic)
            left_top = set(np.argsort(-vectors[left])[:top_k])
            right_top = set(np.argsort(-vectors[right])[:top_k])
            shared = len(left_top & right_top)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n_neurons": len(names),
                    "spearman_abs_weight": rho,
                    "top_k": top_k,
                    "shared_edges": shared,
                    "jaccard": shared / len(left_top | right_top),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    methods: Dict[str, Tuple[np.ndarray, List[str]]] = {
        "SBTG": load_npz_matrix(args.sbtg_file, "mu_hat_lag1"),
        "lag-1 cross-correlation": load_npz_matrix(args.lagged_file, "mu_hat_lag1"),
        "Granger": load_npz_matrix(args.granger_file, "mu_hat_lag1"),
        "zero-lag Pearson": build_zero_lag_pearson(args.dataset_dir),
    }

    zero_pearson, zero_names = methods["zero-lag Pearson"]
    np.savez(
        args.out_dir / "zero_lag_pearson.npz",
        matrix=zero_pearson,
        neuron_names=np.asarray(zero_names),
        definition=(
            "mean across prepared recording rows of corr(x_t, x_t), after within-row "
            "standardization and mean imputation of remaining NaNs"
        ),
    )

    cook_dir = args.connectome_dir
    cook = np.load(cook_dir / "A_chem.npy") + np.load(cook_dir / "A_gap.npy")
    with (cook_dir / "nodes.json").open() as handle:
        cook_names = json.load(handle)

    atlas = np.load(args.atlas_file, allow_pickle=False)
    q = np.asarray(atlas["q"])
    q_eq = np.asarray(atlas["q_eq"])
    atlas_names = [str(x) for x in atlas["neuron_order"]]

    rows = []
    for method, (matrix, method_names) in methods.items():
        names = common_names(method_names, cook_names)
        score_aligned = take_names(matrix, method_names, names)
        cook_aligned = take_names(cook, cook_names, names)
        rows.append(
            evaluate(
                method,
                "Cook anatomy",
                score_aligned,
                positive=cook_aligned != 0,
                evaluated=np.ones_like(cook_aligned, dtype=bool),
            )
        )

        names = common_names(method_names, atlas_names)
        score_aligned = take_names(matrix, method_names, names)
        q_aligned = take_names(q, atlas_names, names)
        q_eq_aligned = take_names(q_eq, atlas_names, names)

        measured = np.isfinite(q_aligned)
        positive_q = measured & (q_aligned < ALPHA)
        rows.append(
            evaluate(
                method,
                "Randi q_significance",
                score_aligned,
                positive=positive_q,
                evaluated=measured,
            )
        )

        positive = np.isfinite(q_aligned) & (q_aligned < ALPHA)
        negative = np.isfinite(q_eq_aligned) & (q_eq_aligned < ALPHA)
        unambiguous = positive ^ negative
        rows.append(
            evaluate(
                method,
                "Randi confirmed_only",
                score_aligned,
                positive=positive,
                evaluated=unambiguous,
            )
        )

        confirmed_negative = negative & ~positive
        rows.append(
            evaluate(
                method,
                "Randi confirmed_positive_priority",
                score_aligned,
                positive=positive,
                evaluated=positive | confirmed_negative,
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "benchmark_metrics.csv", index=False)

    ranks = rank_agreement(methods)
    ranks.to_csv(args.out_dir / "edge_rank_agreement.csv", index=False)

    print(metrics.to_string(index=False))
    print()
    print(ranks.to_string(index=False))
    print(f"\nSaved outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
