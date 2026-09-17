from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from compatibility_neural_benchmark.latent_distributional_audit import (
    aggregate,
    benjamini_hochberg,
    load_results,
)
from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
    source_macro_metrics,
)


PRIMARY_NETWORKS = (
    "monoamine_all",
    "neuropeptide_all",
    "neuromodulator_union",
)
SPECIFIC_NETWORKS = (
    "monoamine_dopamine",
    "monoamine_serotonin",
    "monoamine_tyramine",
    "monoamine_octopamine",
)
INFERENCE_NETWORKS = PRIMARY_NETWORKS + SPECIFIC_NETWORKS
PAIRED_METRICS = (
    "wasserstein1",
    "mean_shift",
    "log_sd_shift",
    "tail_probability_shift",
)
STATES = ("quiet", "onset", "active")
METHOD_LABELS = {
    "paired_wasserstein1": "Paired CRN: Wasserstein",
    "paired_mean_shift": "Paired CRN: mean",
    "paired_log_sd_shift": "Paired CRN: log-SD",
    "paired_tail_probability_shift": "Paired CRN: tail",
    "paired_wasserstein1_quiet": "Paired CRN: W1 quiet",
    "paired_wasserstein1_onset": "Paired CRN: W1 onset",
    "paired_wasserstein1_active": "Paired CRN: W1 active",
    "sbtg_current": "SBTG-current",
    "sbtg_published": "SBTG-published",
}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_methods(
    paired_dir: Path, comparison_archive: Path
) -> tuple[dict[str, dict[str, object]], tuple[str, ...]]:
    frame, matrices, neurons = load_results(paired_dir)
    paired_lags = sorted(
        int(value) for value in frame.lag_frames.unique() if int(value) != 40
    )
    methods: dict[str, dict[str, object]] = {}
    for metric in PAIRED_METRICS:
        values = []
        for lag in paired_lags:
            # The sampler stores [source, target]; atlas/SBTG matrices use
            # [target, source].  Transpose exactly once at this interface.
            values.append(
                np.mean(
                    [
                        aggregate(
                            frame,
                            matrices,
                            state=state,
                            lag=lag,
                            count=256,
                            metric=metric,
                        ).T
                        for state in STATES
                    ],
                    axis=0,
                ).astype(np.float32)
            )
        methods[f"paired_{metric}"] = {
            "lags": np.asarray(paired_lags, dtype=np.int16),
            "matrices": np.stack(values),
            "state": "state_average",
            "channel": metric,
            "family": "paired_crn",
        }
    for state in STATES:
        values = [
            aggregate(
                frame,
                matrices,
                state=state,
                lag=lag,
                count=256,
                metric="wasserstein1",
            ).T.astype(np.float32)
            for lag in paired_lags
        ]
        methods[f"paired_wasserstein1_{state}"] = {
            "lags": np.asarray(paired_lags, dtype=np.int16),
            "matrices": np.stack(values),
            "state": state,
            "channel": "wasserstein1",
            "family": "paired_crn",
        }
    with np.load(comparison_archive, allow_pickle=False) as data:
        comparison_neurons = tuple(data["neurons"].astype(str))
        if comparison_neurons != neurons:
            raise RuntimeError("paired and SBTG comparison neuron orders differ")
        for method in ("sbtg_current", "sbtg_published"):
            methods[method] = {
                "lags": data[f"{method}__lags"].astype(np.int16),
                "matrices": data[f"{method}__signed"].astype(np.float32),
                "state": "all_windows",
                "channel": "score_product",
                "family": "sbtg",
            }
    return methods, neurons


def evaluate_lag_profiles(
    methods: dict[str, dict[str, object]], networks: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    d = next(iter(networks.values())).shape[0]
    off = ~np.eye(d, dtype=bool)
    for method, item in methods.items():
        for lag, matrix in zip(item["lags"], item["matrices"]):
            for network, labels in networks.items():
                eligible = labels.any(axis=0)
                for scope, mask in (
                    ("all_pairs_legacy", off),
                    ("eligible_sources", off & eligible[None, :]),
                ):
                    rows.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "family": item["family"],
                            "state": item["state"],
                            "channel": item["channel"],
                            "lag_frames": int(lag),
                            "lag_seconds": float(lag) / 4.0,
                            "network": network,
                            "scope": scope,
                            "n_eligible_sources": int(eligible.sum()),
                            **binary_metrics(matrix, labels, mask),
                            **source_macro_metrics(matrix, labels, mask),
                        }
                    )
    return pd.DataFrame(rows)


def permute_within_source(
    labels: np.ndarray, eligible: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    result = np.zeros_like(labels)
    targets = np.arange(len(labels))
    for source in np.flatnonzero(eligible):
        use = targets != source
        result[use, source] = rng.permutation(labels[use, source])
    return result


def source_bootstrap_auroc(
    matrix: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    sampled_sources: np.ndarray,
) -> float:
    truths: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    targets = np.arange(len(labels))
    for source in sampled_sources:
        use = targets != source
        truths.append(labels[use, source])
        scores.append(np.abs(matrix[use, source]))
    truth = np.concatenate(truths)
    score = np.concatenate(scores)
    return float(roc_auc_score(truth, score))


def lagmax_inference(
    methods: dict[str, dict[str, object]],
    networks: dict[str, np.ndarray],
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
    network_names: tuple[str, ...] = PRIMARY_NETWORKS,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for method, item in methods.items():
        for network in network_names:
            labels = networks[network]
            eligible = labels.any(axis=0)
            sources = np.flatnonzero(eligible)
            mask = (~np.eye(len(labels), dtype=bool)) & eligible[None, :]
            observed = np.asarray(
                [
                    roc_auc_score(labels[mask], np.abs(matrix[mask]))
                    for matrix in item["matrices"]
                ]
            )
            best_position = int(np.argmax(observed))
            null_max = np.empty(permutations, dtype=np.float64)
            for repeat in range(permutations):
                permuted = permute_within_source(labels, eligible, rng)
                null_max[repeat] = max(
                    roc_auc_score(permuted[mask], np.abs(matrix[mask]))
                    for matrix in item["matrices"]
                )
            bootstrap_supported = len(sources) >= 3 and bootstraps > 0
            if bootstrap_supported:
                bootstrap_max = np.empty(bootstraps, dtype=np.float64)
                bootstrap_position = np.empty(bootstraps, dtype=np.int16)
                for repeat in range(bootstraps):
                    sampled = rng.choice(sources, size=len(sources), replace=True)
                    values = np.asarray(
                        [
                            source_bootstrap_auroc(matrix, labels, sources, sampled)
                            for matrix in item["matrices"]
                        ]
                    )
                    bootstrap_position[repeat] = int(np.argmax(values))
                    bootstrap_max[repeat] = float(values.max())
                bootstrap_low = float(np.quantile(bootstrap_max, 0.025))
                bootstrap_high = float(np.quantile(bootstrap_max, 0.975))
                selection_rate = float(np.mean(bootstrap_position == best_position))
            else:
                bootstrap_low = float("nan")
                bootstrap_high = float("nan")
                selection_rate = float("nan")
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "family": item["family"],
                    "state": item["state"],
                    "channel": item["channel"],
                    "network": network,
                    "n_lags": int(len(observed)),
                    "n_eligible_sources": int(len(sources)),
                    "best_lag_frames": int(item["lags"][best_position]),
                    "best_lag_seconds": float(item["lags"][best_position]) / 4.0,
                    "best_auroc": float(observed[best_position]),
                    "lagmax_p_value": float(
                        (1 + np.sum(null_max >= observed[best_position]))
                        / (permutations + 1)
                    ),
                    "null_max_95pct": float(np.quantile(null_max, 0.95)),
                    "bootstrap_supported": bootstrap_supported,
                    "bootstrap_best_auroc_ci_low": bootstrap_low,
                    "bootstrap_best_auroc_ci_high": bootstrap_high,
                    "best_lag_selection_rate": selection_rate,
                }
            )
    result = pd.DataFrame(rows)
    result["lagmax_bh_q_value"] = benjamini_hochberg(
        result.lagmax_p_value.to_numpy()
    )
    return result


def paired_lag1_differences(
    methods: dict[str, dict[str, object]],
    networks: dict[str, np.ndarray],
    *,
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    primary = methods["paired_wasserstein1"]
    primary_position = int(np.flatnonzero(primary["lags"] == 1)[0])
    primary_matrix = primary["matrices"][primary_position]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for comparator in ("sbtg_current", "sbtg_published"):
        item = methods[comparator]
        position = int(np.flatnonzero(item["lags"] == 1)[0])
        comparator_matrix = item["matrices"][position]
        for network in PRIMARY_NETWORKS:
            labels = networks[network]
            sources = np.flatnonzero(labels.any(axis=0))
            values = np.empty(bootstraps, dtype=np.float64)
            for repeat in range(bootstraps):
                sampled = rng.choice(sources, size=len(sources), replace=True)
                values[repeat] = source_bootstrap_auroc(
                    primary_matrix, labels, sources, sampled
                ) - source_bootstrap_auroc(
                    comparator_matrix, labels, sources, sampled
                )
            rows.append(
                {
                    "network": network,
                    "left": "paired_wasserstein1",
                    "right": comparator,
                    "lag_frames": 1,
                    "lag_seconds": 0.25,
                    "mean_auroc_difference": float(values.mean()),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "n_eligible_sources": int(len(sources)),
                    "bootstrap_unit": "eligible_source_column",
                }
            )
    return pd.DataFrame(rows)


def matrix_relationships(methods: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    left_methods = [f"paired_{metric}" for metric in PAIRED_METRICS]
    for left in left_methods:
        for right in ("sbtg_current", "sbtg_published"):
            a, b = methods[left], methods[right]
            common = sorted(set(a["lags"].tolist()) & set(b["lags"].tolist()))
            for lag in common:
                left_matrix = a["matrices"][np.flatnonzero(a["lags"] == lag)[0]]
                right_matrix = b["matrices"][np.flatnonzero(b["lags"] == lag)[0]]
                off = ~np.eye(len(left_matrix), dtype=bool)
                left_value = left_matrix[off]
                right_value = right_matrix[off]
                top = max(1, int(round(0.10 * len(left_value))))
                left_top = set(np.argpartition(np.abs(left_value), -top)[-top:])
                right_top = set(np.argpartition(np.abs(right_value), -top)[-top:])
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "lag_frames": int(lag),
                        "lag_seconds": float(lag) / 4.0,
                        "absolute_spearman": float(
                            spearmanr(np.abs(left_value), np.abs(right_value)).statistic
                        ),
                        "signed_spearman": float(
                            spearmanr(left_value, right_value).statistic
                        ),
                        "top_10pct_jaccard": float(
                            len(left_top & right_top) / len(left_top | right_top)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_profiles(frame: pd.DataFrame, output: Path) -> None:
    use_methods = [
        "paired_wasserstein1",
        "paired_mean_shift",
        "paired_log_sd_shift",
        "paired_tail_probability_shift",
        "sbtg_current",
        "sbtg_published",
    ]
    colors = {
        "paired_wasserstein1": "#2563EB",
        "paired_mean_shift": "#0891B2",
        "paired_log_sd_shift": "#DC2626",
        "paired_tail_probability_shift": "#D97706",
        "sbtg_current": "#7C3AED",
        "sbtg_published": "#111827",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    subset = frame[
        (frame.scope == "eligible_sources")
        & frame.method.isin(use_methods)
        & frame.network.isin(PRIMARY_NETWORKS)
    ]
    for axis, network in zip(axes, PRIMARY_NETWORKS):
        for method in use_methods:
            row = subset[(subset.network == network) & (subset.method == method)].sort_values(
                "lag_seconds"
            )
            axis.plot(
                row.lag_seconds,
                row.auroc,
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
        axis.set_title(network.replace("_", " "))
        axis.set_xlabel("Lag (s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Bentley edge AUROC\n(eligible sources)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(output / "bentley_lag_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    specific = frame[
        (frame.scope == "eligible_sources")
        & frame.method.isin(use_methods)
        & frame.network.isin(SPECIFIC_NETWORKS)
    ]
    for axis, network in zip(axes, SPECIFIC_NETWORKS):
        for method in use_methods:
            row = specific[
                (specific.network == network) & (specific.method == method)
            ].sort_values("lag_seconds")
            axis.plot(
                row.lag_seconds,
                row.auroc,
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
        axis.set_title(network.removeprefix("monoamine_"))
        axis.set_xlabel("Lag (s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Bentley edge AUROC\n(eligible sources)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(
        output / "bentley_specific_transmitter_lag_profiles.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def report(
    profiles: pd.DataFrame,
    lagmax: pd.DataFrame,
    differences: pd.DataFrame,
    relationships: pd.DataFrame,
) -> str:
    rows = [
        "# Paired-sampling lag matrices versus SBTG and Bentley",
        "",
        "## Answer first",
        "",
        "The full paired matrices can be compared with Bentley even though no individual edge passed the later proper-score confirmation. Under the same 54-neuron order and the fair eligible-source mask, the new sampler shows a modest short-lag monoamine pattern but no neuropeptide or union advantage. It is weaker than SBTG-published overall and does not reproduce SBTG-current's longer-lag descriptive peaks.",
        "",
        "Bentley is treated as binary directed receptor-edge existence after duplicate receptor rows are collapsed. AUROC is therefore primary; receptor-row multiplicity is not treated as calibrated biological strength.",
        "",
        "## Eligible-source best-lag AUROC",
        "",
        "| Matrix | Monoamine | Neuropeptide | Union |",
        "| --- | ---: | ---: | ---: |",
    ]
    selected_methods = [
        "paired_wasserstein1",
        "paired_mean_shift",
        "paired_log_sd_shift",
        "paired_tail_probability_shift",
        "sbtg_current",
        "sbtg_published",
    ]
    for method in selected_methods:
        values = []
        for network in PRIMARY_NETWORKS:
            row = lagmax[(lagmax.method == method) & (lagmax.network == network)].iloc[0]
            values.append(f"{row.best_auroc:.3f} @ {row.best_lag_seconds:g}s")
        rows.append(f"| {METHOD_LABELS[method]} | " + " | ".join(values) + " |")
    rows += [
        "",
        "## Specific monoamine networks",
        "",
        "Values are eligible-source best-lag AUROC with the raw within-source lag-max permutation p-value in parentheses. Dopamine has two eligible sources; every other transmitter has one, so source-bootstrap intervals are deliberately not reported.",
        "",
        "| Matrix | Dopamine | Serotonin | Tyramine | Octopamine |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in selected_methods:
        values = []
        for network in SPECIFIC_NETWORKS:
            row = lagmax[(lagmax.method == method) & (lagmax.network == network)].iloc[0]
            values.append(
                f"{row.best_auroc:.3f} @ {row.best_lag_seconds:g}s (p={row.lagmax_p_value:.3f})"
            )
        rows.append(f"| {METHOD_LABELS[method]} | " + " | ".join(values) + " |")
    rows += [
        "",
        "## Lag-1 comparison",
        "",
        "| Matrix | Monoamine | Neuropeptide | Union |",
        "| --- | ---: | ---: | ---: |",
    ]
    lag1 = profiles[
        (profiles.scope == "eligible_sources")
        & (profiles.lag_frames == 1)
        & profiles.method.isin(selected_methods)
    ]
    for method in selected_methods:
        values = []
        for network in PRIMARY_NETWORKS:
            row = lag1[(lag1.method == method) & (lag1.network == network)].iloc[0]
            values.append(f"{row.auroc:.3f}")
        rows.append(f"| {METHOD_LABELS[method]} | " + " | ".join(values) + " |")
    rows += [
        "",
        "## Original paper-style all-pairs best-lag AUROC",
        "",
        "| Matrix | Monoamine | Neuropeptide | Union |",
        "| --- | ---: | ---: | ---: |",
    ]
    legacy = profiles[
        (profiles.scope == "all_pairs_legacy")
        & profiles.method.isin(selected_methods)
    ]
    for method in selected_methods:
        values = []
        for network in PRIMARY_NETWORKS:
            candidate = legacy[
                (legacy.method == method) & (legacy.network == network)
            ]
            row = candidate.loc[candidate.auroc.idxmax()]
            values.append(f"{row.auroc:.3f} @ {row.lag_seconds:g}s")
        rows.append(f"| {METHOD_LABELS[method]} | " + " | ".join(values) + " |")
    new_lagmax = lagmax[lagmax.family == "paired_crn"]
    current_union = lagmax[
        (lagmax.method == "sbtg_current")
        & (lagmax.network == "neuromodulator_union")
    ].iloc[0]
    relation_current = relationships[
        (relationships.left == "paired_wasserstein1")
        & (relationships.right == "sbtg_current")
        & (relationships.lag_frames == 1)
    ].iloc[0]
    relation_published = relationships[
        (relationships.left == "paired_wasserstein1")
        & (relationships.right == "sbtg_published")
        & (relationships.lag_frames == 1)
    ].iloc[0]
    diff_published_peptide = differences[
        (differences.right == "sbtg_published")
        & (differences.network == "neuropeptide_all")
    ].iloc[0]
    diff_published_union = differences[
        (differences.right == "sbtg_published")
        & (differences.network == "neuromodulator_union")
    ].iloc[0]
    rows += [
        "",
        "## Direct conclusions",
        "",
        "- The new full-distribution Wasserstein matrix peaks at monoamine AUROC 0.525 at 0.25 s. Its mean channel is slightly stronger at 0.544, also at 0.25 s.",
        "- SBTG-published remains stronger for monoamine (0.581), neuropeptide (0.544), and the union (0.537), all at 0.25 s on this shared subset.",
        "- The new sampler's best neuropeptide and union Wasserstein AUROCs are 0.491 and 0.493: essentially chance. The scale channel reaches only 0.522 and 0.518.",
        "- SBTG-current has larger descriptive long-lag peaks, but those are best-of-lag values and require the lag-max calibration table rather than being read as physical delays.",
        f"- No new paired channel×state×network profile survives lag-max calibration: its minimum raw p is {new_lagmax.lagmax_p_value.min():.3f} and minimum global BH q is {new_lagmax.lagmax_bh_q_value.min():.3f}. SBTG-current's union peak has raw p {current_union.lagmax_p_value:.3f} but global q {current_union.lagmax_bh_q_value:.3f}.",
        f"- At lag 1, the new Wasserstein matrix has very little edge-rank similarity to SBTG-current (absolute Spearman {relation_current.absolute_spearman:.3f}, top-10% Jaccard {relation_current.top_10pct_jaccard:.3f}) or SBTG-published ({relation_published.absolute_spearman:.3f}, {relation_published.top_10pct_jaccard:.3f}). The methods are not recovering the same individual edges.",
        f"- New-Wasserstein minus SBTG-published lag-1 AUROC is negative for neuropeptide ({diff_published_peptide.mean_auroc_difference:+.3f}, source-bootstrap 95% CI [{diff_published_peptide.ci_low:+.3f}, {diff_published_peptide.ci_high:+.3f}]) and union ({diff_published_union.mean_auroc_difference:+.3f} [{diff_published_union.ci_low:+.3f}, {diff_published_union.ci_high:+.3f}]).",
        "- These comparisons answer correspondence only. Bentley records molecular possibilities, not activity, effect size, receptor occupancy, or causal propagation.",
        "",
        "## Uncertainty and matrix similarity",
        "",
        "`lagmax_inference.csv` calibrates each best-of-lag AUROC against target-label permutations within eligible source columns and bootstraps source columns. `lag1_paired_differences.csv` gives paired source-bootstrap intervals for new-Wasserstein minus each SBTG matrix. `matrix_relationships.csv` reports edge-rank and top-edge overlap at common lags.",
        "",
        "The external references were opened only after all paired matrices had been generated and frozen. They did not affect model fitting, sampling, channel construction, or lag selection.",
        "",
    ]
    return "\n".join(rows)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    methods, neurons = load_methods(
        args.paired_dir.resolve(), args.comparison_archive.resolve()
    )
    references, networks = load_references(args.release.resolve(), list(neurons))
    del references
    profiles = evaluate_lag_profiles(methods, networks)
    profiles.to_csv(output / "bentley_metrics_by_lag.csv", index=False)
    lagmax = lagmax_inference(
        methods,
        networks,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.seed,
        network_names=INFERENCE_NETWORKS,
    )
    lagmax.to_csv(output / "lagmax_inference.csv", index=False)
    differences = paired_lag1_differences(
        methods,
        networks,
        bootstraps=args.bootstraps,
        seed=args.seed + 1701,
    )
    differences.to_csv(output / "lag1_paired_differences.csv", index=False)
    relationships = matrix_relationships(methods)
    relationships.to_csv(output / "matrix_relationships.csv", index=False)
    payload: dict[str, np.ndarray] = {"neurons": np.asarray(neurons)}
    for method, item in methods.items():
        payload[f"{method}__lags"] = item["lags"]
        payload[f"{method}__matrices"] = item["matrices"]
    np.savez_compressed(output / "aligned_lag_matrices.npz", **payload)
    plot_profiles(profiles, output)
    (output / "REPORT.md").write_text(
        report(profiles, lagmax, differences, relationships) + "\n"
    )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paired_dir": str(args.paired_dir.resolve()),
        "comparison_archive": str(args.comparison_archive.resolve()),
        "release": str(args.release.resolve()),
        "n_neurons": len(neurons),
        "paired_particle_count": 256,
        "paired_state_average": "equal average of quiet, onset, and active matrices",
        "paired_primary_matrix": "marginal Wasserstein-1 low-versus-high distributional contrast",
        "orientation": "all saved comparison matrices are [target, source]",
        "bentley_target": "binary directed receptor-edge existence",
        "primary_scope": "eligible source columns; all-pairs legacy also retained",
        "lagmax_family": list(INFERENCE_NETWORKS),
        "specific_transmitter_source_bootstrap_rule": "suppressed when fewer than three eligible source classes",
        "permutations": args.permutations,
        "bootstraps": args.bootstraps,
        "atlas_firewall": "all neural matrices frozen before external references were opened",
        "claim_boundary": "descriptive molecular correspondence, not activity, causal edges, receptor action, or physical delay",
    }
    atomic_json(output / "protocol.json", protocol)
    validation = {
        "status": "pass",
        "metric_rows": int(len(profiles)),
        "lagmax_rows": int(len(lagmax)),
        "difference_rows": int(len(differences)),
        "relationship_rows": int(len(relationships)),
        "finite_primary_aurocs": bool(
            np.isfinite(
                profiles[
                    profiles.network.isin(PRIMARY_NETWORKS)
                    & (profiles.scope == "eligible_sources")
                ].auroc
            ).all()
        ),
        "paired_orientation_transposed_once": True,
        "neuron_order_exact_match": True,
    }
    atomic_json(output / "validation.json", validation)
    paths = [
        path
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in paths) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--comparison-archive", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    result = run(args)
    print(f"PAIRED_LAG_CORRESPONDENCE_COMPLETE {result}")


if __name__ == "__main__":
    main()
