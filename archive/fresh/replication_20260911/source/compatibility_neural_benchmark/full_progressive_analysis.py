from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.metrics import ndcg_score

from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
    relationship_rows,
    source_macro_metrics,
)


LABELS = {
    "importance_weighting_full": "Direct importance (15 checkpoints)",
    "winner_wide_direct": "Wide-flow direct importance (15 checkpoints)",
    "terminal_smc_full": "Terminal SMC (15 checkpoints)",
    "progressive_smc_full": "Progressive SMC (15 checkpoints)",
    "sbtg_current": "SBTG-current",
    "sbtg_published": "SBTG-published",
}

COLORS = {
    "importance_weighting_full": "#3B82F6",
    "winner_wide_direct": "#0EA5E9",
    "terminal_smc_full": "#F59E0B",
    "progressive_smc_full": "#059669",
    "sbtg_current": "#A16207",
    "sbtg_published": "#7C3AED",
}


def load_progressive_full(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(
        (run_dir / "responses").glob(
            "*__progressive_smc__B4__f*__s*.npz"
        )
    )
    if len(paths) != 15:
        raise RuntimeError(f"expected 15 progressive outputs, found {len(paths)}")
    by_worm: dict[int, list[np.ndarray]] = {}
    by_worm_valid: dict[int, list[np.ndarray]] = {}
    neurons = None
    lags = None
    seen: set[tuple[int, int]] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                raise RuntimeError(f"incomplete response: {path}")
            fold, seed = int(data["fold"]), int(data["seed"])
            if (fold, seed) in seen:
                raise RuntimeError(f"duplicate fold/seed: {(fold, seed)}")
            seen.add((fold, seed))
            current_neurons = data["neurons"].astype(str)
            current_lags = data["horizon_frames"].astype(int)
            if neurons is None:
                neurons, lags = current_neurons, current_lags
            elif not np.array_equal(neurons, current_neurons) or not np.array_equal(
                lags, current_lags
            ):
                raise RuntimeError("progressive output alignment mismatch")
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            normalized = response / np.maximum(gap, 0.10)[:, :, :, None, None]
            matrix = normalized.mean(axis=1).transpose(0, 2, 3, 1)
            validity = data["diagnostic_valid"].astype(np.float64).mean(axis=1)
            for position, worm in enumerate(data["worm_indices"].astype(int)):
                by_worm.setdefault(worm, []).append(matrix[position])
                by_worm_valid.setdefault(worm, []).append(validity[position])
    expected = {(fold, seed) for fold in range(5) for seed in (1701, 2903, 4307)}
    if seen != expected:
        raise RuntimeError(f"fold/seed coverage mismatch: {sorted(expected - seen)}")
    if sorted(by_worm) != list(range(20)):
        raise RuntimeError("progressive responses do not cover all 20 worms")
    effects = np.stack([np.mean(by_worm[worm], axis=0) for worm in range(20)])
    validity = np.stack(
        [np.mean(by_worm_valid[worm], axis=0) for worm in range(20)]
    )
    return effects.mean(axis=0), validity.mean(axis=0), np.asarray(neurons)


def load_direct_full(
    run_dir: Path, model_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(
        (run_dir / "responses").glob(f"{model_id}__B4__f*__s*.npz")
    )
    if len(paths) != 15:
        raise RuntimeError(f"expected 15 winner-direct outputs, found {len(paths)}")
    by_worm: dict[int, list[np.ndarray]] = {}
    by_worm_valid: dict[int, list[np.ndarray]] = {}
    neurons = None
    lags = None
    seen: set[tuple[int, int]] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                raise RuntimeError(f"incomplete response: {path}")
            fold, seed = int(data["fold"]), int(data["seed"])
            if (fold, seed) in seen:
                raise RuntimeError(f"duplicate winner-direct fold/seed: {(fold, seed)}")
            seen.add((fold, seed))
            current_neurons = data["neurons"].astype(str)
            current_lags = data["horizon_frames"].astype(int)
            if neurons is None:
                neurons, lags = current_neurons, current_lags
            elif not np.array_equal(neurons, current_neurons) or not np.array_equal(
                lags, current_lags
            ):
                raise RuntimeError("winner-direct output alignment mismatch")
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            normalized = response / np.maximum(gap, 0.10)[:, :, :, None, None]
            matrix = normalized.mean(axis=1).transpose(0, 2, 3, 1)
            validity = data["diagnostic_valid"].astype(np.float64).mean(axis=1)
            for position, worm in enumerate(data["worm_indices"].astype(int)):
                by_worm.setdefault(worm, []).append(matrix[position])
                by_worm_valid.setdefault(worm, []).append(validity[position])
    expected = {(fold, seed) for fold in range(5) for seed in (1701, 2903, 4307)}
    if seen != expected:
        raise RuntimeError(f"winner-direct fold/seed mismatch: {sorted(expected - seen)}")
    if sorted(by_worm) != list(range(20)):
        raise RuntimeError("winner-direct responses do not cover all 20 worms")
    effects = np.stack([np.mean(by_worm[worm], axis=0) for worm in range(20)])
    validity = np.stack(
        [np.mean(by_worm_valid[worm], axis=0) for worm in range(20)]
    )
    return (
        effects.mean(axis=0), validity.mean(axis=0),
        np.asarray(neurons), np.asarray(lags),
    )


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def _safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(kendalltau(x, y).statistic)


def count_metrics(score: np.ndarray, weight: np.ndarray, mask: np.ndarray) -> dict:
    usable = mask & np.isfinite(score) & np.isfinite(weight)
    s = np.abs(score[usable]).astype(np.float64)
    w = weight[usable].astype(np.float64)
    positive = w > 0
    result = {
        "n_edges": int(len(w)),
        "n_positive": int(positive.sum()),
        "all_pair_spearman": _safe_spearman(s, np.log1p(w)),
        "positive_edge_spearman": _safe_spearman(
            s[positive], np.log1p(w[positive])
        ),
        "positive_edge_kendall": _safe_kendall(
            s[positive], np.log1p(w[positive])
        ),
        "ndcg_all_pairs": float(ndcg_score(np.log1p(w)[None], s[None])),
    }
    order = np.argsort(s, kind="stable")[::-1]
    total = float(w.sum())
    for fraction in (0.01, 0.05, 0.10):
        k = max(1, int(round(fraction * len(w))))
        result[f"weight_recall_top_{int(100 * fraction)}pct"] = (
            float(w[order[:k]].sum() / total) if total > 0 else np.nan
        )

    source_all = []
    source_positive = []
    within_score = np.full(score.shape, np.nan, dtype=np.float64)
    within_weight = np.full(score.shape, np.nan, dtype=np.float64)
    for source in range(score.shape[1]):
        column = usable[:, source]
        cs = np.abs(score[:, source][column])
        cw = weight[:, source][column]
        value = _safe_spearman(cs, np.log1p(cw))
        if np.isfinite(value):
            source_all.append(value)
        pos = cw > 0
        value = _safe_spearman(cs[pos], np.log1p(cw[pos]))
        if np.isfinite(value):
            source_positive.append(value)
        if len(cs) >= 2:
            within_score[column, source] = rankdata(cs, method="average") / len(cs)
            within_weight[column, source] = rankdata(
                np.log1p(cw), method="average"
            ) / len(cw)
    ranked = usable & np.isfinite(within_score) & np.isfinite(within_weight)
    result.update(
        macro_source_all_pair_spearman=float(np.mean(source_all))
        if source_all
        else np.nan,
        macro_source_positive_edge_spearman=float(np.mean(source_positive))
        if source_positive
        else np.nan,
        n_evaluable_sources_all=int(len(source_all)),
        n_evaluable_sources_positive=int(len(source_positive)),
        within_source_rank_spearman=_safe_spearman(
            within_score[ranked], within_weight[ranked]
        ),
    )
    return result


def external_rows(
    methods: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for method, item in methods.items():
        for lag, matrix in zip(item["lags"], item["signed"]):
            for reference, ref in references.items():
                row = {
                    "method": method,
                    "method_label": LABELS[method],
                    "lag_frames": int(lag),
                    "lag_seconds": float(lag / 4.0),
                    "reference": reference,
                    **binary_metrics(matrix, ref["labels"], ref["mask"]),
                    **source_macro_metrics(matrix, ref["labels"], ref["mask"]),
                }
                positive = ref["mask"] & (ref["labels"] > 0)
                if "signed_value" in ref:
                    row["signed_spearman_on_positive"] = _safe_spearman(
                        matrix[positive], ref["signed_value"][positive]
                    )
                    row["sign_agreement_on_positive"] = float(
                        np.mean(
                            np.sign(matrix[positive])
                            == np.sign(ref["signed_value"][positive])
                        )
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def cook_count_rows(
    methods: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for method, item in methods.items():
        for lag, matrix in zip(item["lags"], item["signed"]):
            for reference in (
                "cook_struct_54",
                "cook_chem_54",
                "cook_gap_54",
                "cook_struct_strict44",
                "cook_chem_strict44",
                "cook_gap_strict44",
            ):
                ref = references[reference]
                rows.append(
                    {
                        "method": method,
                        "method_label": LABELS[method],
                        "lag_frames": int(lag),
                        "lag_seconds": float(lag / 4.0),
                        "reference": reference,
                        **count_metrics(matrix, ref["weight"], ref["mask"]),
                    }
                )
    return pd.DataFrame(rows)


def neuromodulator_rows(
    methods: dict[str, dict[str, np.ndarray]], networks: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Evaluate Bentley networks without relying on the contextual method registry."""
    rows = []
    d = next(iter(networks.values())).shape[0]
    off = ~np.eye(d, dtype=bool)
    for method, item in methods.items():
        for lag, matrix in zip(item["lags"], item["signed"]):
            for network, labels in networks.items():
                eligible = labels.any(axis=0)
                for scope, mask in (
                    ("all_pairs_legacy", off),
                    ("eligible_sources", off & eligible[None]),
                ):
                    rows.append(
                        {
                            "method": method,
                            "method_label": LABELS[method],
                            "lag_frames": int(lag),
                            "lag_seconds": float(lag / 4.0),
                            "network": network,
                            "scope": scope,
                            "n_eligible_sources": int(eligible.sum()),
                            **binary_metrics(matrix, labels, mask),
                            **source_macro_metrics(matrix, labels, mask),
                        }
                    )
    return pd.DataFrame(rows)


def _within_source_ranks(
    score: np.ndarray, weight: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    within_score = np.full(score.shape, np.nan, dtype=np.float64)
    within_weight = np.full(score.shape, np.nan, dtype=np.float64)
    usable = mask & np.isfinite(score) & np.isfinite(weight)
    for source in range(score.shape[1]):
        column = usable[:, source]
        n = int(column.sum())
        if n < 2:
            continue
        within_score[column, source] = (
            rankdata(np.abs(score[:, source][column]), method="average") / n
        )
        within_weight[column, source] = (
            rankdata(np.log1p(weight[:, source][column]), method="average") / n
        )
    return within_score, within_weight


def _bootstrap_bundle(
    score: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    within_score: np.ndarray,
    within_weight: np.ndarray,
) -> dict[str, float]:
    presence = binary_metrics(score, weight > 0, mask)
    usable = mask & np.isfinite(score) & np.isfinite(weight)
    s = np.abs(score[usable]).astype(np.float64)
    w = weight[usable].astype(np.float64)
    positive = w > 0
    ranked = usable & np.isfinite(within_score) & np.isfinite(within_weight)
    return {
        "presence_auroc": float(presence["auroc"]),
        "presence_auprc": float(presence["auprc"]),
        "all_pair_spearman": _safe_spearman(s, np.log1p(w)),
        "positive_edge_spearman": _safe_spearman(
            s[positive], np.log1p(w[positive])
        ),
        "within_source_rank_spearman": _safe_spearman(
            within_score[ranked], within_weight[ranked]
        ),
        "ndcg_all_pairs": float(ndcg_score(np.log1p(w)[None], s[None])),
    }


def source_bootstrap_differences(
    methods: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
    *,
    n_boot: int,
    seed: int,
    target: str = "progressive_smc_full",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if target not in methods:
        raise KeyError(f"bootstrap target not present: {target}")
    comparators = [method for method in methods if method != target]
    matrices = {
        method: item["signed"][int(np.flatnonzero(item["lags"] == 1)[0])]
        for method, item in methods.items()
    }
    metric_names = (
        "presence_auroc",
        "presence_auprc",
        "all_pair_spearman",
        "positive_edge_spearman",
        "within_source_rank_spearman",
        "ndcg_all_pairs",
    )
    rows = []
    d = matrices[target].shape[1]
    for reference in ("cook_struct_54", "cook_chem_54", "cook_gap_54"):
        ref = references[reference]
        prepared = {
            method: _within_source_ranks(
                matrix, ref["weight"], ref["mask"]
            )
            for method, matrix in matrices.items()
        }
        observed = {
            method: _bootstrap_bundle(
                matrix,
                ref["weight"],
                ref["mask"],
                prepared[method][0],
                prepared[method][1],
            )
            for method, matrix in matrices.items()
        }
        samples = {
            (method, metric): []
            for method in comparators
            for metric in metric_names
        }
        for _ in range(n_boot):
            columns = rng.integers(0, d, size=d)
            target_values = _bootstrap_bundle(
                matrices[target][:, columns],
                ref["weight"][:, columns],
                ref["mask"][:, columns],
                prepared[target][0][:, columns],
                prepared[target][1][:, columns],
            )
            for method in comparators:
                other_values = _bootstrap_bundle(
                    matrices[method][:, columns],
                    ref["weight"][:, columns],
                    ref["mask"][:, columns],
                    prepared[method][0][:, columns],
                    prepared[method][1][:, columns],
                )
                for metric in metric_names:
                    samples[(method, metric)].append(
                        target_values[metric] - other_values[metric]
                    )
        for method in comparators:
            for metric in metric_names:
                values = np.asarray(samples[(method, metric)], dtype=np.float64)
                rows.append(
                    {
                        "reference": reference,
                        "left": target,
                        "right": method,
                        "metric": metric,
                        "observed_difference": observed[target][metric]
                        - observed[method][metric],
                        "ci_low": float(np.nanquantile(values, 0.025)),
                        "ci_high": float(np.nanquantile(values, 0.975)),
                        "bootstrap_replicates": int(n_boot),
                        "bootstrap_unit": "source_column",
                    }
                )
    return pd.DataFrame(rows)


def create_figures(
    output: Path,
    methods_data: dict[str, dict[str, np.ndarray]],
    external: pd.DataFrame,
    counts: pd.DataFrame,
    neuromod: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = list(LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for ax, method in zip(axes.flat, methods):
        item = methods_data[method]
        matrix = item["signed"][int(np.flatnonzero(item["lags"] == 1)[0])]
        off = matrix[~np.eye(len(matrix), dtype=bool)]
        scale = max(float(np.quantile(np.abs(off), 0.98)), 1e-8)
        image = ax.imshow(
            matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="none"
        )
        ax.set_title(LABELS[method])
        ax.set_xlabel("source index")
        ax.set_ylabel("target index")
        fig.colorbar(image, ax=ax, shrink=0.72)
    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    fig.suptitle("Lag-1 response matrices (method-specific robust color scales)")
    fig.savefig(figures / "lag1_matrices.png", dpi=180)
    plt.close(fig)

    lag1 = external[
        (external.lag_frames == 1)
        & (external.reference.isin(["randi_wild_type", "cook_struct_54"]))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, reference in zip(axes, ["randi_wild_type", "cook_struct_54"]):
        frame = lag1[lag1.reference == reference].set_index("method").loc[methods]
        x = np.arange(len(methods))
        ax.bar(x, frame.auroc, color=[COLORS[m] for m in methods])
        ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
        ax.set_xticks(x, [LABELS[m] for m in methods], rotation=30, ha="right")
        ax.set_ylabel("AUROC")
        ax.set_title(reference.replace("_", " ").title())
    fig.savefig(figures / "lag1_presence_auroc.png", dpi=180)
    plt.close(fig)

    frame = counts[
        (counts.lag_frames == 1) & (counts.reference == "cook_struct_54")
    ].set_index("method").loc[methods]
    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(
        x - width / 2,
        frame.all_pair_spearman,
        width,
        label="all pairs",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        frame.positive_edge_spearman,
        width,
        label="positive Cook edges only",
        color="#F58518",
    )
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x, [LABELS[m] for m in methods], rotation=25, ha="right")
    ax.set_ylabel("Spearman with log(1 + Cook count)")
    ax.set_title("Cook structural count correspondence at lag 1")
    ax.legend()
    fig.savefig(figures / "cook_count_correlations.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, reference in zip(axes, ["randi_wild_type", "cook_struct_54"]):
        for method in methods:
            frame = external[
                (external.method == method) & (external.reference == reference)
            ].sort_values("lag_frames")
            ax.plot(
                frame.lag_seconds,
                frame.auroc,
                marker="o",
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
        ax.set_xlabel("lag / response horizon (seconds)")
        ax.set_ylabel("AUROC")
        ax.set_title(reference.replace("_", " ").title())
    axes[1].legend(fontsize=7.5)
    fig.savefig(figures / "external_auroc_by_lag.png", dpi=180)
    plt.close(fig)


def _table(frame: pd.DataFrame, methods: list[str]) -> list[str]:
    refs = ["randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54"]
    lines = [
        "| Method | Randi WT | Cook structural | Cook chemical | Cook gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        cells = []
        for reference in refs:
            row = frame[
                (frame.method == method) & (frame.reference == reference)
            ].iloc[0]
            cells.append(f"{row.auroc:.3f}/{row.auprc:.3f}")
        lines.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
    return lines


def write_report(
    output: Path,
    external: pd.DataFrame,
    counts: pd.DataFrame,
    bootstrap: pd.DataFrame,
    relationships: pd.DataFrame,
    validity: np.ndarray,
) -> None:
    methods = list(LABELS)
    lag1 = external[external.lag_frames == 1]
    count1 = counts[
        (counts.lag_frames == 1) & (counts.reference == "cook_struct_54")
    ].set_index("method")
    prog = lag1[lag1.method == "progressive_smc_full"].set_index("reference")
    winner = lag1[lag1.method == "winner_wide_direct"].set_index("reference")
    previous = lag1[lag1.method == "importance_weighting_full"].set_index("reference")
    published = lag1[lag1.method == "sbtg_published"].set_index("reference")
    lines = [
        "# Full progressive-SMC ensemble and continuous Cook analysis",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Bottom line",
        "",
        f"The 15-checkpoint progressive ensemble reaches lag-1 Randi WT AUROC/AUPRC **{prog.loc['randi_wild_type','auroc']:.3f}/{prog.loc['randi_wild_type','auprc']:.3f}** and Cook structural **{prog.loc['cook_struct_54','auroc']:.3f}/{prog.loc['cook_struct_54','auprc']:.3f}**. The contextual SBTG-published values are {published.loc['randi_wild_type','auroc']:.3f}/{published.loc['randi_wild_type','auprc']:.3f} and {published.loc['cook_struct_54','auroc']:.3f}/{published.loc['cook_struct_54','auprc']:.3f}.",
        "",
        f"The atlas-blind predictive winner, evaluated with the same direct repaired-response estimator, reaches Randi **{winner.loc['randi_wild_type','auroc']:.3f}/{winner.loc['randi_wild_type','auprc']:.3f}** and Cook structural **{winner.loc['cook_struct_54','auroc']:.3f}/{winner.loc['cook_struct_54','auprc']:.3f}**, versus the previous generator's direct values {previous.loc['randi_wild_type','auroc']:.3f}/{previous.loc['randi_wild_type','auprc']:.3f} and {previous.loc['cook_struct_54','auroc']:.3f}/{previous.loc['cook_struct_54','auprc']:.3f}. This is a post-freeze check of whether predictive improvement transfers to external correspondence.",
        "",
        "Cook is not binary in the bundled reference. It stores integer chemical-synapse and gap-junction counts, with `A_struct = A_chem + A_gap`. AUROC/AUPRC therefore test edge presence after thresholding counts at zero, while count correlations test a distinct question: whether stronger anatomical counts receive larger absolute response scores.",
        "",
        "## Lag-1 edge-presence results",
        "",
        *_table(lag1, methods),
        "",
        "## Cook structural count-strength results",
        "",
        "| Method | All-pair rho | Positive-edge rho | Positive-edge tau | Within-source rank rho | NDCG |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        row = count1.loc[method]
        lines.append(
            f"| {LABELS[method]} | {row.all_pair_spearman:.3f} | {row.positive_edge_spearman:.3f} | {row.positive_edge_kendall:.3f} | {row.within_source_rank_spearman:.3f} | {row.ndcg_all_pairs:.3f} |"
        )
    lines.extend(
        [
            "",
            "All-pair correlation is partly another edge-presence statistic because most Cook pairs are zero. Positive-edge correlation is the cleaner strength-conditional statistic. Neither anatomical count nor model-response magnitude is assumed to be a calibrated physical effect size.",
            "",
            "![Cook count correlations](figures/cook_count_correlations.png)",
            "",
            "## How the lag-1 matrices relate",
            "",
            "| Comparator to progressive | Signed rho | Absolute rho | Sign agreement | Top-10% Jaccard |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparator in [method for method in methods if method != "progressive_smc_full"]:
        row = relationships[
            (((relationships.left == "progressive_smc_full") & (relationships.right == comparator))
             | ((relationships.right == "progressive_smc_full") & (relationships.left == comparator)))
            & (relationships.lag_frames == 1)
        ].iloc[0]
        lines.append(
            f"| {LABELS[comparator]} | {row.signed_spearman:.3f} | "
            f"{row.absolute_spearman:.3f} | {row.sign_agreement:.3f} | "
            f"{row.top_10pct_jaccard:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Lag-1 matrices](figures/lag1_matrices.png)",
            "",
            "## Uncertainty and fairness",
            "",
            "`paired_source_bootstrap.csv` reports progressive-minus-comparator differences for edge presence, count correlation, within-source rank correlation, and NDCG. It resamples source columns of frozen matrices; it is not animal- or generator-refit uncertainty.",
            "",
            f"Mean progressive compatibility-valid rate across source neurons is **{np.mean(validity):.3f}**. The progressive, previous-direct, winner-direct, and terminal flow matrices all average five held-out-worm folds and three generator seeds. SBTG-published remains a released 80-neuron/imputed-cohort artifact subset to the same 54 neurons, so node alignment is exact but training lineage is not identical.",
            "",
            "## Claim boundary",
            "",
            "These are observational, model-relative repaired responses of learned finite-memory dynamics. External correspondence is convergent validity, not physical intervention, direct synapse recovery, or anatomical identification.",
            "",
            "## Files",
            "",
            "- `external_metrics_by_lag.csv`: Randi and Cook edge-presence metrics.",
            "- `cook_count_metrics_by_lag.csv`: hurdle/count-strength metrics.",
            "- `paired_source_bootstrap.csv`: paired uncertainty for Cook metrics.",
            "- `neuromodulator_lag_metrics.csv`: Bentley lag correspondence.",
            "- `matrix_relationships.csv`: pairwise lag-matrix rank/sign/top-edge agreement.",
            "- `aligned_full_ensemble_matrices.npz`: all six aligned method families.",
            "- `protocol.json`, `validation.json`, `input_artifact_checksums.csv`, and `checksums.sha256`: provenance and integrity.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output: Path) -> None:
    paths = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(output)}" for path in paths)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progressive-run", type=Path, required=True)
    parser.add_argument("--fair-analysis", type=Path, required=True)
    parser.add_argument("--winner-direct-run", type=Path, required=True)
    parser.add_argument("--winner-model-id", default="flow_wide128_dropout10")
    parser.add_argument("--published-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    progressive, validity, progressive_neurons = load_progressive_full(
        args.progressive_run
    )
    winner_direct, winner_validity, winner_neurons, winner_lags = load_direct_full(
        args.winner_direct_run, args.winner_model_id
    )
    fair_path = args.fair_analysis / "aligned_all_lag_matrices.npz"
    with np.load(fair_path, allow_pickle=False) as data:
        neurons = data["neurons"].astype(str).tolist()
        if progressive_neurons.astype(str).tolist() != neurons:
            raise RuntimeError("progressive and contextual neuron order mismatch")
        if winner_neurons.astype(str).tolist() != neurons:
            raise RuntimeError("winner-direct and contextual neuron order mismatch")
        methods = {
            "importance_weighting_full": {
                "lags": data["importance_weighting__lags"].astype(int),
                "signed": data["importance_weighting__signed"].astype(np.float64),
            },
            "winner_wide_direct": {
                "lags": winner_lags.astype(int),
                "signed": winner_direct.astype(np.float64),
            },
            "terminal_smc_full": {
                "lags": data["smc_ess__lags"].astype(int),
                "signed": data["smc_ess__signed"].astype(np.float64),
            },
            "progressive_smc_full": {
                "lags": data["importance_weighting__lags"].astype(int),
                "signed": progressive.astype(np.float64),
            },
            "sbtg_current": {
                "lags": data["sbtg_current__lags"].astype(int),
                "signed": data["sbtg_current__signed"].astype(np.float64),
            },
            "sbtg_published": {
                "lags": data["sbtg_published__lags"].astype(int),
                "signed": data["sbtg_published__signed"].astype(np.float64),
            },
        }
    references, networks = load_references(args.published_release, neurons)
    external = external_rows(methods, references)
    counts = cook_count_rows(methods, references)
    neuromod = neuromodulator_rows(methods, networks)
    relationships = relationship_rows(methods)
    bootstrap = source_bootstrap_differences(
        methods, references, n_boot=args.bootstrap, seed=args.seed
    )
    external.to_csv(output / "external_metrics_by_lag.csv", index=False)
    counts.to_csv(output / "cook_count_metrics_by_lag.csv", index=False)
    neuromod.to_csv(output / "neuromodulator_lag_metrics.csv", index=False)
    relationships.to_csv(output / "matrix_relationships.csv", index=False)
    bootstrap.to_csv(output / "paired_source_bootstrap.csv", index=False)
    np.savez_compressed(
        output / "aligned_full_ensemble_matrices.npz",
        neurons=np.asarray(neurons),
        progressive_validity=validity.astype(np.float32),
        winner_direct_validity=winner_validity.astype(np.float32),
        **{f"{name}__lags": item["lags"] for name, item in methods.items()},
        **{
            f"{name}__signed": item["signed"].astype(np.float32)
            for name, item in methods.items()
        },
    )
    create_figures(output, methods, external, counts, neuromod)
    write_report(output, external, counts, bootstrap, relationships, validity)

    inputs = [
        fair_path,
        args.fair_analysis / "manifest.json",
        args.progressive_run / "manifest.json",
        *sorted((args.progressive_run / "responses").glob("*.npz")),
        args.winner_direct_run / "manifest.json",
        *sorted((args.winner_direct_run / "responses").glob("*.npz")),
        args.published_release
        / "reference_data/functional_atlas/aligned_atlas_wild_type.npz",
        args.published_release / "reference_data/connectome/A_struct.npy",
        args.published_release / "reference_data/connectome/A_chem.npy",
        args.published_release / "reference_data/connectome/A_gap.npy",
    ]
    pd.DataFrame(
        [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path.resolve()),
                "size_bytes": path.resolve().stat().st_size,
            }
            for path in inputs
        ]
    ).to_csv(output / "input_artifact_checksums.csv", index=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "full 5-fold x 3-seed progressive ensemble plus continuous Cook evaluation",
        "no_retraining": True,
        "progressive_run": str(args.progressive_run.resolve()),
        "winner_direct_run": str(args.winner_direct_run.resolve()),
        "winner_model_id": args.winner_model_id,
        "fair_analysis": str(args.fair_analysis.resolve()),
        "published_release": str(args.published_release.resolve()),
        "matrix_orientation": "target row, source column",
        "cook_policy": {
            "presence": "count > 0; AUROC and AUPRC on absolute scores",
            "strength": "Spearman/Kendall/NDCG using log1p count, both all-pair and positive-edge conditional",
            "combined": "A_struct = A_chem + A_gap",
        },
        "bootstrap": {
            "unit": "source column",
            "replicates": args.bootstrap,
            "seed": args.seed,
        },
        "claim_boundary": "model-relative observational repaired response; not causal or anatomical identification",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True))
    checks = {
        "six_methods": set(methods) == set(LABELS),
        "fifteen_progressive_outputs": len(
            list((args.progressive_run / "responses").glob("*.npz"))
        ) == 15,
        "n_progressive_outputs": len(
            list((args.progressive_run / "responses").glob("*.npz"))
        ),
        "fifteen_winner_direct_outputs": len(
            list((args.winner_direct_run / "responses").glob("*.npz"))
        ) == 15,
        "n_winner_direct_outputs": len(
            list((args.winner_direct_run / "responses").glob("*.npz"))
        ),
        "all_finite": all(
            np.isfinite(item["signed"]).all() for item in methods.values()
        ),
        "all_54_by_54": all(
            item["signed"].shape[1:] == (54, 54) for item in methods.values()
        ),
        "external_rows": len(external),
        "cook_count_rows": len(counts),
        "neuromodulator_rows": len(neuromod),
        "matrix_relationship_rows": len(relationships),
        "bootstrap_rows": len(bootstrap),
        "cook_struct_equals_chem_plus_gap": bool(
            np.allclose(
                references["cook_struct_54"]["weight"],
                references["cook_chem_54"]["weight"]
                + references["cook_gap_54"]["weight"],
            )
        ),
    }
    critical = (
        "six_methods",
        "fifteen_progressive_outputs",
        "fifteen_winner_direct_outputs",
        "all_finite",
        "all_54_by_54",
        "cook_struct_equals_chem_plus_gap",
    )
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all(bool(checks[key]) for key in critical) else "failed",
        "checks": checks,
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True)
    )
    write_checksums(output)


if __name__ == "__main__":
    main()
