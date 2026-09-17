from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from compatibility_neural_benchmark.fair_atlas_analysis import align_square


METHODS = [
    "importance_weighting",
    "smc_ess",
    "sbtg_current",
    "sbtg_published",
]
LABELS = {
    "importance_weighting": "Flow importance weighting",
    "smc_ess": "Flow bootstrap SMC (ESS)",
    "sbtg_current": "SBTG-current",
    "sbtg_published": "SBTG-published",
}
COLORS = {
    "importance_weighting": "#2B6CB0",
    "smc_ess": "#C47A1A",
    "sbtg_current": "#7A6A58",
    "sbtg_published": "#805AD5",
}


def binary_metrics(score: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict:
    usable = mask & np.isfinite(score) & np.isfinite(labels)
    y = labels[usable].astype(bool)
    s = np.abs(score[usable])
    return {
        "n_edges": int(len(y)),
        "n_positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }


def load_references(release: Path, neurons: list[str]) -> dict[str, dict[str, np.ndarray]]:
    with np.load(
        release / "reference_data/functional_atlas/aligned_atlas_wild_type.npz",
        allow_pickle=False,
    ) as atlas:
        names = atlas["neuron_order"].astype(str).tolist()
        q = align_square(atlas["q"], names, neurons)
        q_eq = align_square(atlas["q_eq"], names, neurons)
    positive = q < 0.05
    negative = (q_eq < 0.05) & ~positive
    off = ~np.eye(len(neurons), dtype=bool)
    cook_names = json.loads(
        (release / "reference_data/connectome/nodes.json").read_text()
    )
    cook = align_square(
        np.load(release / "reference_data/connectome/A_struct.npy"),
        cook_names,
        neurons,
        fill=0.0,
    ) > 0
    return {
        "randi_wild_type": {"labels": positive, "mask": (positive | negative) & off},
        "cook_struct_54": {"labels": cook, "mask": off},
    }


def load_checkpoint_run(
    run_dir: Path, pattern: str, estimator: str
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], pd.DataFrame]:
    matrices: dict[tuple[int, int], np.ndarray] = {}
    phase_matrices: dict[tuple[int, int], np.ndarray] = {}
    rows = []
    for path in sorted((run_dir / "responses").glob(pattern)):
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                continue
            fold, seed = int(data["fold"]), int(data["seed"])
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            normalized = response / np.maximum(gap, 0.10)[:, :, :, None, None]
            matrices[(fold, seed)] = normalized.mean(axis=(0, 1)).transpose(1, 2, 0)
            phase_matrices[(fold, seed)] = normalized.mean(axis=0).transpose(0, 2, 3, 1)
            neurons = data["neurons"].astype(str)
            phases = data["phase_names"].astype(str)
            for worm_pos, worm in enumerate(data["worm_indices"].astype(int)):
                for phase_pos, phase in enumerate(phases):
                    for source, neuron in enumerate(neurons):
                        target_gap = float(data["diagnostic_target_gap"][worm_pos, phase_pos, source])
                        achieved_gap = float(data["diagnostic_achieved_gap"][worm_pos, phase_pos, source])
                        rows.append(
                            {
                                "estimator": estimator,
                                "fold": fold,
                                "seed": seed,
                                "worm": worm,
                                "phase": phase,
                                "source": neuron,
                                "valid_fraction": float(
                                    data["diagnostic_valid"][worm_pos, phase_pos, source]
                                ),
                                "ess_low": float(
                                    data["diagnostic_ess_low"][worm_pos, phase_pos, source]
                                ),
                                "ess_high": float(
                                    data["diagnostic_ess_high"][worm_pos, phase_pos, source]
                                ),
                                "max_weight_low": float(
                                    data["diagnostic_max_weight_low"][worm_pos, phase_pos, source]
                                ),
                                "max_weight_high": float(
                                    data["diagnostic_max_weight_high"][worm_pos, phase_pos, source]
                                ),
                                "target_gap": target_gap,
                                "achieved_gap": achieved_gap,
                                "achieved_fraction": achieved_gap / target_gap,
                            }
                        )
    if len(matrices) != 15:
        raise RuntimeError(f"expected 15 complete {estimator} checkpoints, got {len(matrices)}")
    return matrices, phase_matrices, pd.DataFrame(rows)


def source_metrics(
    neurons: list[str],
    matrices: dict[str, np.ndarray],
    references: dict[str, dict[str, np.ndarray]],
    validity: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    difference_rows = []
    rng = np.random.default_rng(20260827)
    for reference, item in references.items():
        labels, mask = item["labels"], item["mask"]
        source_values: dict[str, list[float]] = {method: [] for method in METHODS}
        source_weights = []
        within_rank = {method: np.full_like(matrices[method], np.nan) for method in METHODS}
        for source, neuron in enumerate(neurons):
            usable = mask[:, source]
            y = labels[:, source][usable].astype(bool)
            if not len(y) or y.min() == y.max():
                continue
            source_weights.append(int(len(y)))
            for method in METHODS:
                score = np.abs(matrices[method][:, source][usable])
                auroc = float(roc_auc_score(y, score))
                auprc = float(average_precision_score(y, score))
                source_values[method].append(auroc)
                within_rank[method][usable, source] = rankdata(score) / len(score)
                rows.append(
                    {
                        "reference": reference,
                        "source": neuron,
                        "method": method,
                        "method_label": LABELS[method],
                        "n_edges": int(len(y)),
                        "n_positive": int(y.sum()),
                        "auroc": auroc,
                        "auprc": auprc,
                        "smc_valid_rate": float(validity.loc[neuron]),
                    }
                )
        for method in METHODS:
            global_metric = binary_metrics(matrices[method], labels, mask)
            values = np.asarray(source_values[method])
            rank_mask = mask & np.isfinite(within_rank[method])
            summary_rows.append(
                {
                    "reference": reference,
                    "method": method,
                    "method_label": LABELS[method],
                    "global_auroc": global_metric["auroc"],
                    "global_auprc": global_metric["auprc"],
                    "macro_source_auroc": float(values.mean()),
                    "weighted_source_auroc": float(
                        np.average(values, weights=source_weights)
                    ),
                    "within_source_rank_global_auroc": float(
                        roc_auc_score(labels[rank_mask], within_rank[method][rank_mask])
                    ),
                    "n_evaluable_sources": int(len(values)),
                }
            )
        n_sources = len(source_weights)
        indices = rng.integers(0, n_sources, size=(20_000, n_sources))
        for left, right in itertools.combinations(METHODS, 2):
            delta = np.asarray(source_values[left]) - np.asarray(source_values[right])
            sampled = delta[indices].mean(axis=1)
            difference_rows.append(
                {
                    "reference": reference,
                    "left": left,
                    "right": right,
                    "macro_auroc_difference": float(delta.mean()),
                    "ci_low": float(np.quantile(sampled, 0.025)),
                    "ci_high": float(np.quantile(sampled, 0.975)),
                    "n_sources": n_sources,
                    "bootstrap_replicates": 20_000,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows), pd.DataFrame(difference_rows)


def matrix_structure_rows(methods: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for method, item in methods.items():
        for lag, matrix in zip(item["lags"], item["signed"]):
            work = matrix.astype(np.float64).copy()
            np.fill_diagonal(work, 0.0)
            off = ~np.eye(len(work), dtype=bool)
            singular = np.linalg.svd(work, compute_uv=False)
            power = singular**2
            fraction = power / power.sum()
            row_rms = np.sqrt(np.mean(work**2, axis=1))
            column_rms = np.sqrt(np.mean(work**2, axis=0))
            rows.append(
                {
                    "method": method,
                    "method_label": LABELS[method],
                    "lag_frames": int(lag),
                    "rank1_energy_fraction": float(fraction[0]),
                    "rank5_energy_fraction": float(fraction[:5].sum()),
                    "effective_rank": float(
                        np.exp(-np.sum(fraction * np.log(fraction + 1e-300)))
                    ),
                    "row_rms_cv": float(row_rms.std() / row_rms.mean()),
                    "column_rms_cv": float(column_rms.std() / column_rms.mean()),
                    "reciprocity_spearman": float(
                        spearmanr(work[off], work.T[off]).statistic
                    ),
                }
            )
    return pd.DataFrame(rows)


def stability_rows(
    checkpoint_matrices: dict[str, dict[tuple[int, int], np.ndarray]]
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260827)
    for method, matrices in checkpoint_matrices.items():
        first = next(iter(matrices.values()))[0]
        off = ~np.eye(len(first), dtype=bool)
        within_signed, within_absolute = [], []
        for fold in range(5):
            seeds = sorted(seed for current_fold, seed in matrices if current_fold == fold)
            for left, right in itertools.combinations(seeds, 2):
                a, b = matrices[(fold, left)][0][off], matrices[(fold, right)][0][off]
                within_signed.append(float(spearmanr(a, b).statistic))
                within_absolute.append(float(spearmanr(np.abs(a), np.abs(b)).statistic))
        for metric, values in (
            ("signed_spearman", within_signed),
            ("absolute_spearman", within_absolute),
        ):
            rows.append(
                {
                    "method": method,
                    "scope": "within_fold_generator_seed_pairs",
                    "ensemble_size": 1,
                    "metric": metric,
                    "median": float(np.median(values)),
                    "p10": float(np.quantile(values, 0.10)),
                    "p90": float(np.quantile(values, 0.90)),
                    "replicates": len(values),
                }
            )
        lag1 = np.stack([matrix[0] for matrix in matrices.values()])
        for size in (1, 2, 3, 5, 7):
            signed_values, absolute_values = [], []
            for _ in range(1_000):
                selection = rng.choice(len(lag1), 2 * size, replace=False)
                left = lag1[selection[:size]].mean(axis=0)[off]
                right = lag1[selection[size:]].mean(axis=0)[off]
                signed_values.append(float(spearmanr(left, right).statistic))
                absolute_values.append(
                    float(spearmanr(np.abs(left), np.abs(right)).statistic)
                )
            for metric, values in (
                ("signed_spearman", signed_values),
                ("absolute_spearman", absolute_values),
            ):
                rows.append(
                    {
                        "method": method,
                        "scope": "random_disjoint_split_half",
                        "ensemble_size": size,
                        "metric": metric,
                        "median": float(np.median(values)),
                        "p10": float(np.quantile(values, 0.10)),
                        "p90": float(np.quantile(values, 0.90)),
                        "replicates": 1_000,
                    }
                )
    return pd.DataFrame(rows)


def create_figures(
    output: Path,
    lag_metrics: pd.DataFrame,
    source_agreement: pd.DataFrame,
    structure: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, reference in zip(axes, ["randi_wild_type", "cook_struct_54"]):
        for method in METHODS:
            frame = lag_metrics[
                (lag_metrics.reference == reference) & (lag_metrics.method == method)
            ].sort_values("lag_seconds")
            ax.plot(
                frame.lag_seconds,
                frame.auroc,
                marker="o",
                label=LABELS[method],
                color=COLORS[method],
            )
        ax.axhline(0.5, color="#4A4A4A", linestyle="--", linewidth=1)
        ax.set_title(
            "Randi wild-type perturbational atlas"
            if reference.startswith("randi")
            else "Cook structural connectome"
        )
        ax.set_xlabel("Lag / response horizon (seconds)")
        ax.set_ylabel("AUROC")
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(figures / "diagnostic_external_auroc_by_lag.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.scatter(
        source_agreement.smc_valid_rate,
        source_agreement.signed_spearman,
        s=42,
        color=COLORS["importance_weighting"],
        edgecolor="#243447",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.set_title("Flow estimator agreement by source neuron")
    ax.set_xlabel("SMC compatibility-valid episode fraction")
    ax.set_ylabel("Importance vs SMC signed Spearman across targets")
    ax.grid(alpha=0.2)
    fig.savefig(figures / "diagnostic_validity_vs_estimator_agreement.png", dpi=180)
    plt.close(fig)

    lag1 = structure[structure.lag_frames == 1].set_index("method").loc[METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    x = np.arange(len(lag1))
    axes[0].bar(x, lag1.effective_rank, color=[COLORS[name] for name in lag1.index])
    axes[0].set_xticks(x, [LABELS[name] for name in lag1.index], rotation=30, ha="right")
    axes[0].set_title("Lag-1 effective matrix rank")
    axes[0].set_ylabel("Entropy effective rank")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        x,
        lag1.rank1_energy_fraction,
        color=[COLORS[name] for name in lag1.index],
    )
    axes[1].set_xticks(x, [LABELS[name] for name in lag1.index], rotation=30, ha="right")
    axes[1].set_title("Lag-1 energy in first singular component")
    axes[1].set_ylabel("Fraction of off-diagonal squared energy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(figures / "diagnostic_lag1_matrix_rank.png", dpi=180)
    plt.close(fig)

    frame = stability[
        (stability.scope == "random_disjoint_split_half")
        & (stability.metric == "signed_spearman")
    ]
    sizes = sorted(frame.ensemble_size.unique())
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for offset, method in zip((-width / 2, width / 2), ["importance_weighting", "smc_ess"]):
        current = frame[frame.method == method].set_index("ensemble_size").loc[sizes]
        ax.bar(
            np.arange(len(sizes)) + offset,
            current["median"],
            width,
            color=COLORS[method],
            label=LABELS[method],
        )
        ax.errorbar(
            np.arange(len(sizes)) + offset,
            current["median"],
            yerr=np.vstack(
                [current["median"] - current.p10, current.p90 - current["median"]]
            ),
            fmt="none",
            ecolor="#2D3748",
            capsize=2,
            linewidth=0.8,
        )
    ax.set_xticks(np.arange(len(sizes)), sizes)
    ax.set_xlabel("Checkpoints averaged per disjoint half")
    ax.set_ylabel("Median signed matrix Spearman")
    ax.set_title("Lag-1 split-half reproducibility versus ensemble size")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures / "diagnostic_split_half_reproducibility.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--smc-run", type=Path, required=True)
    parser.add_argument("--published-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(args.analysis_dir / "aligned_all_lag_matrices.npz", allow_pickle=False) as data:
        neurons = data["neurons"].astype(str).tolist()
        methods = {
            method: {
                "lags": data[f"{method}__lags"].astype(int),
                "signed": data[f"{method}__signed"].astype(np.float64),
            }
            for method in METHODS
        }
    lag1 = {
        method: item["signed"][int(np.flatnonzero(item["lags"] == 1)[0])]
        for method, item in methods.items()
    }
    references = load_references(args.published_release, neurons)

    importance_checkpoints, importance_phase, importance_diag = load_checkpoint_run(
        args.primary_run, "tcn_delta_flow_matching__B4__f*__s*.npz", "importance_weighting"
    )
    smc_checkpoints, smc_phase, smc_diag = load_checkpoint_run(
        args.smc_run, "*__smc__B4__f*__s*.npz", "smc_ess"
    )
    diagnostics = pd.concat([importance_diag, smc_diag], ignore_index=True)
    diagnostic_rows = []
    for estimator, frame in diagnostics.groupby("estimator"):
        diagnostic_rows.append(
            {
                "estimator": estimator,
                "scope": "overall",
                "detail": "all archived worm-phase-source-seed cells",
                "valid_fraction": float(frame.valid_fraction.mean()),
                "n_cells": int(len(frame)),
                "spearman_valid_achieved_fraction": float(
                    spearmanr(frame.valid_fraction, frame.achieved_fraction).statistic
                ),
                "spearman_valid_max_weight_low": float(
                    spearmanr(frame.valid_fraction, frame.max_weight_low).statistic
                ),
                "spearman_valid_max_weight_high": float(
                    spearmanr(frame.valid_fraction, frame.max_weight_high).statistic
                ),
                "spearman_valid_target_gap": float(
                    spearmanr(frame.valid_fraction, frame.target_gap).statistic
                ),
            }
        )
        for phase, phase_frame in frame.groupby("phase"):
            diagnostic_rows.append(
                {
                    "estimator": estimator,
                    "scope": "phase",
                    "detail": phase,
                    "valid_fraction": float(phase_frame.valid_fraction.mean()),
                    "n_cells": int(len(phase_frame)),
                }
            )
        proxies = {
            "average_ess_low_below_12.8": frame.ess_low < 12.8,
            "average_ess_high_below_12.8": frame.ess_high < 12.8,
            "average_max_weight_low_above_0.20": frame.max_weight_low > 0.20,
            "average_max_weight_high_above_0.20": frame.max_weight_high > 0.20,
            "average_achieved_fraction_below_0.25": frame.achieved_fraction < 0.25,
        }
        for detail, flag in proxies.items():
            diagnostic_rows.append(
                {
                    "estimator": estimator,
                    "scope": "averaged_criterion_indicator_not_episode_failure_rate",
                    "detail": detail,
                    "valid_fraction": float(flag.mean()),
                    "n_cells": int(len(frame)),
                }
            )
    diagnostic_summary = pd.DataFrame(diagnostic_rows)
    diagnostic_summary.to_csv(output / "diagnostic_compatibility_summary.csv", index=False)

    smc_valid = smc_diag.groupby("source").valid_fraction.mean().reindex(neurons)
    source_rows, source_summary, source_differences = source_metrics(
        neurons, lag1, references, smc_valid
    )
    source_rows.to_csv(output / "diagnostic_source_reference_metrics.csv", index=False)
    source_summary.to_csv(output / "diagnostic_source_macro_summary.csv", index=False)
    source_differences.to_csv(
        output / "diagnostic_macro_paired_differences.csv", index=False
    )

    off = ~np.eye(len(neurons), dtype=bool)
    agreement_rows = []
    for source, neuron in enumerate(neurons):
        usable = off[:, source]
        left = lag1["importance_weighting"][:, source][usable]
        right = lag1["smc_ess"][:, source][usable]
        agreement_rows.append(
            {
                "source": neuron,
                "smc_valid_rate": float(smc_valid.loc[neuron]),
                "importance_valid_rate": float(
                    importance_diag[importance_diag.source == neuron].valid_fraction.mean()
                ),
                "signed_spearman": float(spearmanr(left, right).statistic),
                "absolute_spearman": float(
                    spearmanr(np.abs(left), np.abs(right)).statistic
                ),
                "sign_agreement": float(np.mean(np.sign(left) == np.sign(right))),
                "mean_absolute_difference": float(np.mean(np.abs(left - right))),
            }
        )
    source_agreement = pd.DataFrame(agreement_rows)
    source_agreement.to_csv(output / "diagnostic_source_estimator_agreement.csv", index=False)

    quartiles = pd.qcut(
        smc_valid, 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
    )
    quartile_rows = []
    for reference, item in references.items():
        for quartile in quartiles.cat.categories:
            columns = (quartiles == quartile).to_numpy()
            mask = item["mask"] & columns[None]
            for method in METHODS:
                quartile_rows.append(
                    {
                        "reference": reference,
                        "validity_quartile": quartile,
                        "method": method,
                        "n_sources": int(columns.sum()),
                        "mean_smc_valid_rate": float(smc_valid[columns].mean()),
                        **binary_metrics(lag1[method], item["labels"], mask),
                    }
                )
    pd.DataFrame(quartile_rows).to_csv(
        output / "diagnostic_validity_quartile_metrics.csv", index=False
    )

    lag_rows = []
    for method, item in methods.items():
        for lag, matrix in zip(item["lags"], item["signed"]):
            for reference, reference_item in references.items():
                lag_rows.append(
                    {
                        "method": method,
                        "method_label": LABELS[method],
                        "lag_frames": int(lag),
                        "lag_seconds": float(lag / 4.0),
                        "reference": reference,
                        **binary_metrics(
                            matrix, reference_item["labels"], reference_item["mask"]
                        ),
                    }
                )
    lag_metrics = pd.DataFrame(lag_rows)
    lag_metrics.to_csv(output / "diagnostic_external_metrics_by_lag.csv", index=False)

    phase_rows = []
    for method, checkpoint_phase in (
        ("importance_weighting", importance_phase),
        ("smc_ess", smc_phase),
    ):
        phase_matrix = np.mean(list(checkpoint_phase.values()), axis=0)
        for phase_index, phase in enumerate(["baseline", "onset", "active", "offset", "recovery"]):
            for lag_index, lag in enumerate(methods[method]["lags"]):
                for reference, reference_item in references.items():
                    phase_rows.append(
                        {
                            "method": method,
                            "phase": phase,
                            "lag_frames": int(lag),
                            "lag_seconds": float(lag / 4.0),
                            "reference": reference,
                            **binary_metrics(
                                phase_matrix[phase_index, lag_index],
                                reference_item["labels"],
                                reference_item["mask"],
                            ),
                        }
                    )
    pd.DataFrame(phase_rows).to_csv(
        output / "diagnostic_flow_metrics_by_phase_and_lag.csv", index=False
    )

    structure = matrix_structure_rows(methods)
    structure.to_csv(output / "diagnostic_matrix_structure.csv", index=False)
    stability = stability_rows(
        {"importance_weighting": importance_checkpoints, "smc_ess": smc_checkpoints}
    )
    stability.to_csv(output / "diagnostic_ensemble_stability.csv", index=False)

    rank_matrices = {}
    for method, matrix in lag1.items():
        ranked = np.zeros_like(matrix)
        ranked[off] = rankdata(np.abs(matrix[off])) / int(off.sum())
        rank_matrices[method] = ranked
    ensembles = {
        "importance_plus_smc": np.mean(
            [rank_matrices["importance_weighting"], rank_matrices["smc_ess"]], axis=0
        ),
        "importance_plus_published": np.mean(
            [rank_matrices["importance_weighting"], rank_matrices["sbtg_published"]], axis=0
        ),
        "smc_plus_published": np.mean(
            [rank_matrices["smc_ess"], rank_matrices["sbtg_published"]], axis=0
        ),
        "importance_smc_published": np.mean(
            [
                rank_matrices["importance_weighting"],
                rank_matrices["smc_ess"],
                rank_matrices["sbtg_published"],
            ],
            axis=0,
        ),
    }
    ensemble_rows = []
    for name, matrix in {**rank_matrices, **ensembles}.items():
        for reference, item in references.items():
            ensemble_rows.append(
                {
                    "exploratory_score": name,
                    "reference": reference,
                    **binary_metrics(matrix, item["labels"], item["mask"]),
                }
            )
    pd.DataFrame(ensemble_rows).to_csv(
        output / "diagnostic_exploratory_rank_ensembles.csv", index=False
    )

    create_figures(output, lag_metrics, source_agreement, structure, stability)
    (output / "DIAGNOSTIC_CHART_MAP.md").write_text(
        "# Diagnostic chart map\n\n"
        "| Figure | Question | Family | Fields | Supported takeaway |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `diagnostic_external_auroc_by_lag.png` | How does external correspondence vary with lag? | Trend / ordered axis | method, lag seconds, AUROC, reference | Randi correspondence is short-lag; Cook correspondence is flatter or later for flow. |\n"
        "| `diagnostic_validity_vs_estimator_agreement.png` | Does compatibility support track agreement between the two flow estimators? | Relationship | source valid rate, signed Spearman | Agreement rises only modestly with validity; compatibility is not a substitute for edge accuracy. |\n"
        "| `diagnostic_lag1_matrix_rank.png` | Did a method collapse to a low-dimensional matrix? | Comparison | effective rank, rank-1 energy | SBTG-current is extremely low-rank relative to the other frozen matrices. |\n"
        "| `diagnostic_split_half_reproducibility.png` | Does checkpoint ensembling improve matrix reproducibility? | Comparison / progression | ensemble size, split-half Spearman | Reproducibility rises substantially as more checkpoints are averaged. |\n"
    )


if __name__ == "__main__":
    main()
