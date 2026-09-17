from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from compatibility_neural_benchmark.fair_atlas_analysis import (
    align_square,
    build_reference_adjacency,
)


METHOD_LABELS = {
    "reference_high_particle": "High-particle reference",
    "direct_matched": "Direct importance (matched)",
    "terminal_smc_matched": "Terminal SMC (matched)",
    "progressive_smc_matched": "Progressive SMC (matched)",
    "importance_weighting_full": "Direct importance (full ensemble)",
    "terminal_smc_full": "Terminal SMC (full ensemble)",
    "sbtg_current": "SBTG-current",
    "sbtg_published": "SBTG-published",
}

METHOD_COLORS = {
    "reference_high_particle": "#111827",
    "direct_matched": "#3B82F6",
    "terminal_smc_matched": "#F59E0B",
    "progressive_smc_matched": "#059669",
    "importance_weighting_full": "#60A5FA",
    "terminal_smc_full": "#FBBF24",
    "sbtg_current": "#A16207",
    "sbtg_published": "#7C3AED",
}

PRIMARY_REFERENCES = [
    "randi_wild_type",
    "cook_struct_54",
    "cook_chem_54",
    "cook_gap_54",
]


def binary_metrics(score: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict:
    usable = np.asarray(mask, bool) & np.isfinite(score) & np.isfinite(labels)
    y = np.asarray(labels[usable] > 0, dtype=np.int8)
    s = np.abs(np.asarray(score[usable], dtype=np.float64))
    result = {
        "n_edges": int(len(y)),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
        "precision_at_reference_density": np.nan,
    }
    if 0 < y.sum() < len(y):
        result["auroc"] = float(roc_auc_score(y, s))
        result["auprc"] = float(average_precision_score(y, s))
        n_top = int(y.sum())
        predicted = np.zeros(len(y), dtype=bool)
        predicted[np.argsort(s, kind="stable")[-n_top:]] = True
        result["precision_at_reference_density"] = float(
            np.sum(predicted & (y > 0)) / n_top
        )
    return result


def source_macro_metrics(
    score: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    aurocs: list[float] = []
    auprcs: list[float] = []
    weights: list[int] = []
    for source in range(score.shape[1]):
        usable = mask[:, source] & np.isfinite(score[:, source])
        y = labels[:, source][usable].astype(np.int8)
        if not len(y) or y.min() == y.max():
            continue
        s = np.abs(score[:, source][usable])
        aurocs.append(float(roc_auc_score(y, s)))
        auprcs.append(float(average_precision_score(y, s)))
        weights.append(int(len(y)))
    return {
        "macro_source_auroc": float(np.mean(aurocs)) if aurocs else np.nan,
        "macro_source_auprc": float(np.mean(auprcs)) if auprcs else np.nan,
        "weighted_source_auroc": float(np.average(aurocs, weights=weights))
        if aurocs
        else np.nan,
        "n_evaluable_sources": int(len(aurocs)),
    }


def load_references(
    release: Path, neurons: list[str]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    root = release / "reference_data"
    off = ~np.eye(len(neurons), dtype=bool)
    references: dict[str, dict[str, np.ndarray]] = {}
    randi_present = np.zeros(len(neurons), dtype=bool)
    for genotype, filename in (
        ("wild_type", "aligned_atlas_wild_type.npz"),
        ("unc31", "aligned_atlas_unc31.npz"),
    ):
        with np.load(root / "functional_atlas" / filename, allow_pickle=False) as data:
            names = data["neuron_order"].astype(str).tolist()
            q = align_square(data["q"], names, neurons)
            q_eq = align_square(data["q_eq"], names, neurons)
            dff = align_square(data["dff"], names, neurons)
        positive = q < 0.05
        negative = (q_eq < 0.05) & ~positive
        present = np.asarray([name in names for name in neurons])
        if genotype == "wild_type":
            randi_present = present
        references[f"randi_{genotype}"] = {
            "labels": positive.astype(np.int8),
            "mask": (positive | negative) & off,
            "signed_value": dff,
        }

    cook_names = json.loads((root / "connectome" / "nodes.json").read_text())
    strict = randi_present[:, None] & randi_present[None] & off
    for kind, filename in (
        ("struct", "A_struct.npy"),
        ("chem", "A_chem.npy"),
        ("gap", "A_gap.npy"),
    ):
        weight = align_square(
            np.load(root / "connectome" / filename), cook_names, neurons, fill=0.0
        )
        labels = (weight > 0).astype(np.int8)
        for scope, mask in (("54", off), ("strict44", strict)):
            references[f"cook_{kind}_{scope}"] = {
                "labels": labels,
                "mask": mask,
                "weight": weight,
            }

    networks: dict[str, np.ndarray] = {}
    edge_root = root / "modulatory_atlas" / "edge_lists"
    for signal in (None, "dopamine", "serotonin", "tyramine", "octopamine"):
        adjacency, _, _ = build_reference_adjacency(
            edge_root / "edgelist_MA_classes.csv", neurons, signal=signal
        )
        networks["monoamine_all" if signal is None else f"monoamine_{signal}"] = (
            adjacency
        )
    neuropeptide, _, _ = build_reference_adjacency(
        edge_root / "edgelist_NP_classes.csv", neurons
    )
    networks["neuropeptide_all"] = neuropeptide
    networks["neuromodulator_union"] = (
        (networks["monoamine_all"] + neuropeptide) > 0
    ).astype(np.int8)
    return references, networks


def evaluate_external(
    methods: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for method, item in methods.items():
        for lag, matrix in zip(item["lags"], item["signed"]):
            for reference, ref in references.items():
                row = {
                    "panel": item["panel"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "lag_frames": int(lag),
                    "lag_seconds": float(lag / 4.0),
                    "reference": reference,
                    **binary_metrics(matrix, ref["labels"], ref["mask"]),
                    **source_macro_metrics(matrix, ref["labels"], ref["mask"]),
                }
                positive = ref["mask"] & (ref["labels"] > 0)
                if "signed_value" in ref and positive.sum() >= 3:
                    row["signed_spearman_on_positive"] = float(
                        spearmanr(matrix[positive], ref["signed_value"][positive]).statistic
                    )
                    row["sign_agreement_on_positive"] = float(
                        np.mean(
                            np.sign(matrix[positive])
                            == np.sign(ref["signed_value"][positive])
                        )
                    )
                else:
                    row["signed_spearman_on_positive"] = np.nan
                    row["sign_agreement_on_positive"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate_repeats(
    repeats: dict[str, np.ndarray],
    lags: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for method, array in repeats.items():
        for repeat, matrices in enumerate(array):
            for lag, matrix in zip(lags, matrices):
                for reference, ref in references.items():
                    rows.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "repeat": int(repeat),
                            "lag_frames": int(lag),
                            "reference": reference,
                            **binary_metrics(matrix, ref["labels"], ref["mask"]),
                        }
                    )
    return pd.DataFrame(rows)


def evaluate_neuromodulators(
    methods: dict[str, dict[str, np.ndarray]], networks: dict[str, np.ndarray]
) -> pd.DataFrame:
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
                            "panel": item["panel"],
                            "method": method,
                            "method_label": METHOD_LABELS[method],
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


def relationship_rows(methods: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    off = ~np.eye(next(iter(methods.values()))["signed"].shape[-1], dtype=bool)
    for left, right in itertools.combinations(methods, 2):
        left_item, right_item = methods[left], methods[right]
        common = sorted(set(left_item["lags"]) & set(right_item["lags"]))
        for lag in common:
            x = left_item["signed"][np.flatnonzero(left_item["lags"] == lag)[0]][off]
            y = right_item["signed"][np.flatnonzero(right_item["lags"] == lag)[0]][off]
            n_top = max(1, int(round(0.10 * len(x))))
            x_top = set(np.argpartition(np.abs(x), -n_top)[-n_top:].tolist())
            y_top = set(np.argpartition(np.abs(y), -n_top)[-n_top:].tolist())
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "lag_frames": int(lag),
                    "n_edges": int(len(x)),
                    "signed_spearman": float(spearmanr(x, y).statistic),
                    "absolute_spearman": float(spearmanr(np.abs(x), np.abs(y)).statistic),
                    "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
                    "top_10pct_jaccard": float(
                        len(x_top & y_top) / len(x_top | y_top)
                    ),
                    "mse": float(np.mean((x - y) ** 2)),
                }
            )
    return pd.DataFrame(rows)


def paired_source_bootstrap(
    methods: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = "progressive_smc_matched"
    comparisons = [
        "direct_matched",
        "terminal_smc_matched",
        "reference_high_particle",
        "importance_weighting_full",
        "terminal_smc_full",
        "sbtg_current",
        "sbtg_published",
    ]
    d = methods[target]["signed"].shape[-1]
    matrices = {}
    for method, item in methods.items():
        lag_index = int(np.flatnonzero(item["lags"] == 1)[0])
        matrices[method] = item["signed"][lag_index]
    rows = []
    for reference_index, reference in enumerate(PRIMARY_REFERENCES):
        ref = references[reference]
        observed = {
            method: binary_metrics(matrix, ref["labels"], ref["mask"])
            for method, matrix in matrices.items()
        }
        boot = {
            (method, metric): []
            for method in comparisons
            for metric in ("auroc", "auprc")
        }
        local_rng = np.random.default_rng(
            rng.integers(0, np.iinfo(np.int64).max) + reference_index
        )
        for _ in range(n_boot):
            columns = local_rng.integers(0, d, size=d)
            target_metrics = binary_metrics(
                matrices[target][:, columns],
                ref["labels"][:, columns],
                ref["mask"][:, columns],
            )
            for method in comparisons:
                other = binary_metrics(
                    matrices[method][:, columns],
                    ref["labels"][:, columns],
                    ref["mask"][:, columns],
                )
                for metric in ("auroc", "auprc"):
                    boot[(method, metric)].append(
                        float(target_metrics[metric] - other[metric])
                    )
        for method in comparisons:
            for metric in ("auroc", "auprc"):
                values = np.asarray(boot[(method, metric)])
                rows.append(
                    {
                        "reference": reference,
                        "left": target,
                        "right": method,
                        "metric": metric,
                        "observed_difference": float(
                            observed[target][metric] - observed[method][metric]
                        ),
                        "ci_low": float(np.nanquantile(values, 0.025)),
                        "ci_high": float(np.nanquantile(values, 0.975)),
                        "bootstrap_replicates": int(n_boot),
                        "bootstrap_unit": "source_column",
                    }
                )
    return pd.DataFrame(rows)


def create_figures(
    output: Path,
    methods: dict[str, dict[str, np.ndarray]],
    external: pd.DataFrame,
    neuromod: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    matched = [
        "reference_high_particle",
        "direct_matched",
        "terminal_smc_matched",
        "progressive_smc_matched",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, method in zip(axes.flat, matched):
        item = methods[method]
        matrix = item["signed"][int(np.flatnonzero(item["lags"] == 1)[0])]
        off = matrix[~np.eye(len(matrix), dtype=bool)]
        scale = max(float(np.quantile(np.abs(off), 0.98)), 1e-8)
        image = ax.imshow(
            matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="none"
        )
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("source index")
        ax.set_ylabel("target index")
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle("Estimator-matched lag-1 response matrices (method-specific scales)")
    fig.savefig(figures / "matched_lag1_matrices.png", dpi=180)
    plt.close(fig)

    refs = ["randi_wild_type", "cook_struct_54"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, reference in zip(axes, refs):
        for method in matched:
            sub = external[
                (external.method == method) & (external.reference == reference)
            ].sort_values("lag_frames")
            ax.plot(
                sub.lag_seconds,
                sub.auroc,
                marker="o",
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.axhline(0.5, color="#6B7280", linewidth=1, linestyle="--")
        ax.set_title(reference.replace("_", " ").title())
        ax.set_xlabel("response horizon (seconds)")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.44, 0.68)
    axes[1].legend(fontsize=8, loc="best")
    fig.savefig(figures / "matched_external_auroc_by_lag.png", dpi=180)
    plt.close(fig)

    selected_methods = [
        "reference_high_particle",
        "progressive_smc_matched",
        "importance_weighting_full",
        "terminal_smc_full",
        "sbtg_published",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, network in zip(axes, ["monoamine_all", "neuropeptide_all"]):
        for method in selected_methods:
            sub = neuromod[
                (neuromod.method == method)
                & (neuromod.network == network)
                & (neuromod.scope == "eligible_sources")
                & (neuromod.lag_frames.isin([1, 2, 8]))
            ].sort_values("lag_frames")
            ax.plot(
                sub.lag_seconds,
                sub.auroc,
                marker="o",
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.axhline(0.5, color="#6B7280", linewidth=1, linestyle="--")
        ax.set_title(network.replace("_", " ").title())
        ax.set_xlabel("common lag (seconds)")
        ax.set_ylabel("eligible-source AUROC")
    axes[1].legend(fontsize=7.5, loc="best")
    fig.savefig(figures / "neuromodulator_common_lags.png", dpi=180)
    plt.close(fig)

    forest = bootstrap[
        (bootstrap.reference.isin(["randi_wild_type", "cook_struct_54"]))
        & (bootstrap.metric == "auroc")
    ].copy()
    forest["label"] = (
        forest.reference.str.replace("_", " ")
        + ": vs "
        + forest.right.map(METHOD_LABELS)
    )
    y = np.arange(len(forest))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(forest))), constrained_layout=True)
    ax.errorbar(
        forest.observed_difference,
        y,
        xerr=np.vstack(
            [
                forest.observed_difference - forest.ci_low,
                forest.ci_high - forest.observed_difference,
            ]
        ),
        fmt="o",
        color=METHOD_COLORS["progressive_smc_matched"],
        ecolor="#6B7280",
        capsize=3,
    )
    ax.axvline(0, color="#111827", linewidth=1, linestyle="--")
    ax.set_yticks(y, forest.label)
    ax.invert_yaxis()
    ax.set_xlabel("Progressive SMC minus comparator AUROC")
    ax.set_title("Paired source-bootstrap differences (frozen matrices)")
    fig.savefig(figures / "progressive_auroc_differences.png", dpi=180)
    plt.close(fig)


def _metric_table(frame: pd.DataFrame, methods: list[str]) -> list[str]:
    names = {
        "randi_wild_type": "Randi WT",
        "cook_struct_54": "Cook structural",
        "cook_chem_54": "Cook chemical",
        "cook_gap_54": "Cook gap",
    }
    lines = [
        "| Method | Randi WT | Cook structural | Cook chemical | Cook gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        cells = []
        for reference in names:
            row = frame[(frame.method == method) & (frame.reference == reference)].iloc[0]
            cells.append(f"{row.auroc:.3f} / {row.auprc:.3f}")
        lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
    return lines


def write_report(
    output: Path,
    neurons: list[str],
    methods: dict[str, dict[str, np.ndarray]],
    external: pd.DataFrame,
    repeat_summary: pd.DataFrame,
    relationships: pd.DataFrame,
    neuromod: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    lag1 = external[external.lag_frames == 1]
    matched = [
        "reference_high_particle",
        "direct_matched",
        "terminal_smc_matched",
        "progressive_smc_matched",
    ]
    context = [
        "importance_weighting_full",
        "terminal_smc_full",
        "sbtg_current",
        "sbtg_published",
    ]
    prog = lag1[lag1.method == "progressive_smc_matched"].set_index("reference")
    direct = lag1[lag1.method == "direct_matched"].set_index("reference")
    terminal = lag1[lag1.method == "terminal_smc_matched"].set_index("reference")
    published = lag1[lag1.method == "sbtg_published"].set_index("reference")

    rel = relationships[relationships.lag_frames == 1].set_index(["left", "right"])
    relation_values = {}
    for comparator in ("direct_matched", "terminal_smc_matched", "reference_high_particle"):
        key = (comparator, "progressive_smc_matched")
        if key not in rel.index:
            key = ("progressive_smc_matched", comparator)
        relation_values[comparator] = float(rel.loc[key, "signed_spearman"])

    common_best = []
    common = neuromod[
        (neuromod.scope == "eligible_sources")
        & (neuromod.lag_frames.isin([1, 2, 8]))
        & (neuromod.network.isin(["monoamine_all", "neuropeptide_all"]))
    ]
    for (_, _), frame in common.groupby(["method", "network"]):
        common_best.append(frame.loc[frame.auroc.idxmax()])
    common_best_frame = pd.DataFrame(common_best)

    lines = [
        "# Post-freeze external evaluation of progressive bridge SMC",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Bottom line",
        "",
        "The progressive bridge remains the best finite-particle estimator of the frozen learned repaired-response law, but external atlas correspondence is mixed and does not establish biological superiority. Its lag-1 Randi WT AUROC is "
        f"**{prog.loc['randi_wild_type', 'auroc']:.3f}**, versus "
        f"{direct.loc['randi_wild_type', 'auroc']:.3f} for matched direct importance, "
        f"{terminal.loc['randi_wild_type', 'auroc']:.3f} for matched terminal SMC, and "
        f"{published.loc['randi_wild_type', 'auroc']:.3f} for the contextual SBTG-published artifact. "
        "On Cook structural anatomy, the corresponding AUROCs are "
        f"{prog.loc['cook_struct_54', 'auroc']:.3f}, "
        f"{direct.loc['cook_struct_54', 'auroc']:.3f}, "
        f"{terminal.loc['cook_struct_54', 'auroc']:.3f}, and "
        f"{published.loc['cook_struct_54', 'auroc']:.3f}.",
        "",
        "The correct conclusion is therefore two-part: progressive SMC improves Monte Carlo recovery of the chosen learned-law estimand, while the external references do not show a uniform improvement over either the simpler estimator or SBTG-published. These are convergent-validity checks on observational effective responses, not direct synapse recovery or causal identification.",
        "",
        "## Why this analysis is post-freeze",
        "",
        "Stage A selected the estimator using only response error against two independent 4,096-particle references, validity, achieved contrast, ancestry, and runtime. Its protocol explicitly sealed Randi, Cook, Bentley, and published SBTG artifacts. Only after progressive SMC passed those gates did this Stage-B script load the external references. The sealed Stage-A directory was not modified.",
        "",
        "This resolves the earlier record that no external results were loaded during development or scoring: that statement remains true for Stage A; the comparisons below are a separately labeled one-time post-freeze evaluation.",
        "",
        "## Two comparison panels",
        "",
        "- **Estimator-matched panel:** one frozen flow checkpoint (outer fold 0, generator seed 1701), the same four held-out worms, and the same three Monte Carlo base seeds for direct, terminal SMC, and progressive SMC. The high-particle reference is the mean of two independent 4,096-particle direct-importance runs.",
        "- **Contextual panel:** the previously frozen 15-checkpoint full direct and terminal-SMC ensembles, SBTG-current, and SBTG-published. These provide biological-reference context but are not matched estimator replicates of the Stage-A checkpoint.",
        "",
        f"Every matrix is restricted to the same {len(neurons)} neurons in identical order, uses target rows and source columns, and excludes the diagonal for external metrics.",
        "",
        "## Lag-1 estimator-matched results",
        "",
        "Cells are AUROC / AUPRC using absolute continuous edge scores.",
        "",
        *_metric_table(lag1, matched),
        "",
        "The paired source bootstrap in `paired_source_bootstrap.csv` resamples source columns of already-frozen matrices. It describes atlas-source sensitivity only; it is not an animal-refit or generator-refit confidence interval.",
        "",
        "## Contextual lag-1 results",
        "",
        *_metric_table(lag1, context),
        "",
        "SBTG-published remains a contextual released artifact trained on its 80-neuron/imputed-cohort pipeline and then subset to these 54 neurons. SBTG-current and the flow models used the present 54-neuron cohort. The shared node set is fair; the training lineages are not identical.",
        "",
        "## Does progressive SMC reproduce the reference matrix?",
        "",
        f"At lag 1, progressive SMC has signed Spearman correlation **{relation_values['reference_high_particle']:.3f}** with the high-particle reference, versus **{relation_values['direct_matched']:.3f}** with matched direct importance and **{relation_values['terminal_smc_matched']:.3f}** with matched terminal SMC. The full all-lag relationship table, including absolute-score correlation, sign agreement, top-edge overlap, and raw MSE, is in `matrix_relationships_by_lag.csv`.",
        "",
        "![Estimator-matched lag-1 matrices](figures/matched_lag1_matrices.png)",
        "",
        "## External correspondence across response horizons",
        "",
        "![Matched external AUROC by lag](figures/matched_external_auroc_by_lag.png)",
        "",
        "The curves are response horizons of the cumulative future feature (0.25 to 10 seconds), not SBTG Hessian lags. Comparing their rank correspondence to an atlas is meaningful; equating their numerical entries or interpreting the best horizon as a physical transmission delay is not.",
        "",
        "## Bentley neuromodulator lag analysis",
        "",
        "The less-confounded `eligible_sources` scope restricts negatives to source columns containing at least one reference edge. The common 1/2/8-frame grid is the valid between-method comparison with SBTG-published.",
        "",
        "| Method | Network | Best common lag | AUROC / AUPRC |",
        "| --- | --- | ---: | ---: |",
    ]
    for method in [*matched, "importance_weighting_full", "terminal_smc_full", "sbtg_current", "sbtg_published"]:
        for network in ("monoamine_all", "neuropeptide_all"):
            row = common_best_frame[
                (common_best_frame.method == method)
                & (common_best_frame.network == network)
            ].iloc[0]
            lines.append(
                f"| {METHOD_LABELS[method]} | {network} | {int(row.lag_frames)} ({row.lag_seconds:.2f}s) | {row.auroc:.3f} / {row.auprc:.3f} |"
            )
    lines.extend(
        [
            "",
            "![Neuromodulator common-lag comparison](figures/neuromodulator_common_lags.png)",
            "",
            "Best-lag entries are descriptive maxima over three prespecified common lags. They do not identify receptor kinetics, validate a direct edge, or override the broader negative real neuromodulator evidence elsewhere in this repository.",
            "",
            "## Monte Carlo stability versus biological uncertainty",
            "",
            "`matched_repeat_external_metrics.csv` evaluates each of the three finite-particle repeats separately, and `matched_repeat_external_summary.csv` reports their mean, standard deviation, minimum, and maximum. Those spreads quantify Monte Carlo sensitivity for one checkpoint and four held-out worms. They do not measure generator-seed, fold, animal-population, or biological uncertainty.",
            "",
            "![Progressive paired source-bootstrap differences](figures/progressive_auroc_differences.png)",
            "",
            "## Method correspondence to the attachment",
            "",
            "- The generator conditions on the 80-frame neural history and the observed 80-frame binary stimulus history; the future stimulus schedule is held fixed in every query.",
            "- The response is the high-minus-low repaired future cumulative-mean feature divided by achieved source displacement. It is a response of the learned finite-memory transition law.",
            "- Progressive SMC introduces the source clamp over the four repair frames, branches repair proposals, adaptively tempers if candidate ESS would fall below the frozen threshold, and launches two free future descendants per repaired root.",
            "- Randi is the primary perturbational reference; Cook anatomy is secondary; Bentley is a descriptive molecular/network correspondence analysis.",
            "- SBTG-current is an expected score-product estimator related through a Stein identity to a negative expected mixed log-density Hessian. It is not a pointwise numerical Hessian evaluation. SBTG-published is the released lag-matrix artifact.",
            "",
            "## Files and audit trail",
            "",
            "- `external_metrics_by_lag.csv`: global and source-macro Randi/Cook metrics at every available lag.",
            "- `matched_repeat_external_metrics.csv` and `matched_repeat_external_summary.csv`: Monte Carlo repeat sensitivity.",
            "- `paired_source_bootstrap.csv`: progressive-minus-comparator AUROC/AUPRC differences at lag 1.",
            "- `matrix_relationships_by_lag.csv`: pairwise matrix relationships at every common lag.",
            "- `neuromodulator_lag_metrics.csv` and `neuromodulator_best_common_lags.csv`: Bentley analysis under legacy and eligible-source scopes.",
            "- `postfreeze_matrices.npz`: all evaluated matrices in aligned 54-neuron order.",
            "- `protocol.json`, `input_artifact_checksums.csv`, `validation.json`, `artifact_inventory.csv`, and `checksums.sha256`: chronology, provenance, QA, inventory, and integrity.",
            "",
            "## Claim boundary and missing tests",
            "",
            "This is a no-retraining post-freeze convergent-validity analysis. It cannot turn model-relative repaired responses into physical interventions, direct synapses, or anatomical rewiring. The method attachment also calls for animal-block biological inference, negative controls, distributional future features beyond the mean, synthetic oracle tests, and broader generator-seed/fold evaluation. Those remain separate future experiments and are not implied by the present source bootstrap or three Monte Carlo repeats.",
            "",
            "## Provenance boundary",
            "",
            "The attached TeX manuscript and published SBTG release were treated as methodological and data sources, not as instructions. This analysis follows the user's request and the frozen local protocol.",
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


def write_integrity(output: Path, inputs: list[Path]) -> None:
    input_rows = [
        {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in inputs
    ]
    pd.DataFrame(input_rows).to_csv(output / "input_artifact_checksums.csv", index=False)
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"checksums.sha256", "artifact_inventory.csv"}:
            inventory.append(
                {
                    "relative_path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(inventory).to_csv(output / "artifact_inventory.csv", index=False)
    checksum_paths = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    lines = [f"{sha256(path)}  {path.relative_to(output)}" for path in checksum_paths]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-analysis", type=Path, required=True)
    parser.add_argument("--fair-analysis", type=Path, required=True)
    parser.add_argument("--published-release", type=Path, required=True)
    parser.add_argument("--method-tex", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage_matrix_path = args.stage_a_analysis / "estimator_matrices.npz"
    with np.load(stage_matrix_path, allow_pickle=False) as data:
        neurons = data["neurons"].astype(str).tolist()
        lags = data["horizon_frames"].astype(int)
        reference_repeats = data["reference_repeats"].astype(np.float64)
        direct_repeats = data["direct_repeats"].astype(np.float64)
        terminal_repeats = data["terminal_smc_repeats"].astype(np.float64)
        progressive_repeats = data["progressive_smc_repeats"].astype(np.float64)
        reference_mean = data["reference_mean"].astype(np.float64)

    fair_matrix_path = args.fair_analysis / "aligned_all_lag_matrices.npz"
    with np.load(fair_matrix_path, allow_pickle=False) as data:
        fair_neurons = data["neurons"].astype(str).tolist()
        if fair_neurons != neurons:
            raise RuntimeError("Stage-A and prior fair-analysis neuron order disagree")
        contextual = {
            method: {
                "lags": data[f"{source}__lags"].astype(int),
                "signed": data[f"{source}__signed"].astype(np.float64),
                "panel": "contextual",
            }
            for method, source in (
                ("importance_weighting_full", "importance_weighting"),
                ("terminal_smc_full", "smc_ess"),
                ("sbtg_current", "sbtg_current"),
                ("sbtg_published", "sbtg_published"),
            )
        }

    matched = {
        "reference_high_particle": {
            "lags": lags,
            "signed": reference_mean,
            "panel": "estimator_matched",
        },
        "direct_matched": {
            "lags": lags,
            "signed": direct_repeats.mean(axis=0),
            "panel": "estimator_matched",
        },
        "terminal_smc_matched": {
            "lags": lags,
            "signed": terminal_repeats.mean(axis=0),
            "panel": "estimator_matched",
        },
        "progressive_smc_matched": {
            "lags": lags,
            "signed": progressive_repeats.mean(axis=0),
            "panel": "estimator_matched",
        },
    }
    methods = {**matched, **contextual}
    if any(item["signed"].shape[1:] != (54, 54) for item in methods.values()):
        raise RuntimeError("all matrices must be aligned 54 by 54")
    if any(not np.isfinite(item["signed"]).all() for item in methods.values()):
        raise RuntimeError("all evaluated matrices must be finite")

    references, networks = load_references(args.published_release, neurons)
    external = evaluate_external(methods, references)
    external.to_csv(output / "external_metrics_by_lag.csv", index=False)

    repeats = {
        "reference_high_particle": reference_repeats,
        "direct_matched": direct_repeats,
        "terminal_smc_matched": terminal_repeats,
        "progressive_smc_matched": progressive_repeats,
    }
    repeat_metrics = evaluate_repeats(repeats, lags, references)
    repeat_metrics.to_csv(output / "matched_repeat_external_metrics.csv", index=False)
    repeat_summary = (
        repeat_metrics.groupby(["method", "method_label", "lag_frames", "reference"])[
            ["auroc", "auprc"]
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    repeat_summary.columns = [
        "_".join([str(value) for value in column if value]).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in repeat_summary.columns
    ]
    repeat_summary.to_csv(output / "matched_repeat_external_summary.csv", index=False)

    relationships = relationship_rows(methods)
    relationships.to_csv(output / "matrix_relationships_by_lag.csv", index=False)
    neuromod = evaluate_neuromodulators(methods, networks)
    neuromod.to_csv(output / "neuromodulator_lag_metrics.csv", index=False)
    common = neuromod[neuromod.lag_frames.isin([1, 2, 8])]
    common.to_csv(output / "neuromodulator_common_lag_metrics.csv", index=False)
    best_rows = []
    for (_, _, _), frame in common.groupby(["method", "network", "scope"]):
        usable = frame[np.isfinite(frame.auroc)]
        if len(usable):
            best_rows.append(usable.loc[usable.auroc.idxmax()])
    pd.DataFrame(best_rows).to_csv(
        output / "neuromodulator_best_common_lags.csv", index=False
    )

    bootstrap = paired_source_bootstrap(
        methods, references, n_boot=args.bootstrap, seed=args.seed
    )
    bootstrap.to_csv(output / "paired_source_bootstrap.csv", index=False)

    np.savez_compressed(
        output / "postfreeze_matrices.npz",
        neurons=np.asarray(neurons),
        **{f"{method}__lags": item["lags"] for method, item in methods.items()},
        **{f"{method}__signed": item["signed"].astype(np.float32) for method, item in methods.items()},
    )

    create_figures(output, methods, external, neuromod, bootstrap)
    write_report(
        output,
        neurons,
        methods,
        external,
        repeat_summary,
        relationships,
        neuromod,
        bootstrap,
    )

    inputs = [
        stage_matrix_path,
        args.stage_a_analysis / "protocol.json",
        args.stage_a_analysis / "checksums.sha256",
        fair_matrix_path,
        args.fair_analysis / "manifest.json",
        args.method_tex,
        args.published_release / "results/paper/sbtg_lag_matrices.npz",
        args.published_release
        / "reference_data/functional_atlas/aligned_atlas_wild_type.npz",
        args.published_release
        / "reference_data/functional_atlas/aligned_atlas_unc31.npz",
        args.published_release / "reference_data/connectome/A_struct.npy",
        args.published_release / "reference_data/connectome/A_chem.npy",
        args.published_release / "reference_data/connectome/A_gap.npy",
        args.published_release
        / "reference_data/modulatory_atlas/edge_lists/edgelist_MA_classes.csv",
        args.published_release
        / "reference_data/modulatory_atlas/edge_lists/edgelist_NP_classes.csv",
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_stage": "Stage B one-time post-freeze external evaluation",
        "stage_a_external_policy": "Randi, Cook, Bentley, and published SBTG were sealed and not loaded during estimator selection",
        "stage_b_external_policy": "loaded only after progressive SMC passed the predeclared Stage-A advancement gate",
        "no_retraining": True,
        "stage_a_analysis": str(args.stage_a_analysis.resolve()),
        "fair_analysis": str(args.fair_analysis.resolve()),
        "published_release": str(args.published_release.resolve()),
        "method_attachment": str(args.method_tex.resolve()),
        "matrix_orientation": "target row, source column",
        "fps": 4.0,
        "shared_neurons": neurons,
        "matched_panel": {
            "checkpoint": "tcn_delta_flow_matching fold 0 generator seed 1701",
            "finite_particle_repeats": 3,
            "reference_repeats": 2,
            "reference_particles": 4096,
        },
        "contextual_panel": "previously frozen 15-checkpoint direct/terminal ensembles plus SBTG-current and released SBTG-published",
        "bootstrap": {
            "unit": "source column",
            "replicates": args.bootstrap,
            "seed": args.seed,
            "limitation": "frozen-matrix source sensitivity, not animal or generator refit uncertainty",
        },
        "claim_boundary": "model-relative observational repaired response; external correspondence is not causal or anatomical identification",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True))
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "checks": {
            "eight_methods_present": set(methods) == set(METHOD_LABELS),
            "stage_and_fair_neuron_order_identical": fair_neurons == neurons,
            "shared_neuron_count": len(neurons),
            "all_matrices_finite": all(
                np.isfinite(item["signed"]).all() for item in methods.values()
            ),
            "all_matrices_54_by_54": all(
                item["signed"].shape[1:] == (54, 54) for item in methods.values()
            ),
            "lag1_present": all(1 in item["lags"] for item in methods.values()),
            "external_metric_rows": int(len(external)),
            "repeat_metric_rows": int(len(repeat_metrics)),
            "relationship_rows": int(len(relationships)),
            "neuromodulator_metric_rows": int(len(neuromod)),
            "bootstrap_rows": int(len(bootstrap)),
            "randi_wild_type_pairs": int(references["randi_wild_type"]["mask"].sum()),
            "randi_wild_type_positives": int(
                (references["randi_wild_type"]["mask"]
                & (references["randi_wild_type"]["labels"] > 0)).sum()
            ),
            "cook_structural_pairs": int(references["cook_struct_54"]["mask"].sum()),
            "cook_structural_positives": int(
                (references["cook_struct_54"]["mask"]
                & (references["cook_struct_54"]["labels"] > 0)).sum()
            ),
        },
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True)
    )
    write_integrity(output, inputs)


if __name__ == "__main__":
    main()
