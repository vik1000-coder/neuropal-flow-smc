from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.full_progressive_analysis import (
    LABELS,
    cook_count_rows,
    external_rows,
    load_direct_full,
    neuromodulator_rows,
    sha256,
    source_bootstrap_differences,
    write_checksums,
)
from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
    relationship_rows,
)


METHODS = (
    "importance_weighting_full",
    "winner_wide_direct",
    "terminal_smc_full",
    "sbtg_current",
    "sbtg_published",
)


def paired_presence_bootstrap(
    methods: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict[str, np.ndarray]],
    *, n_boot: int, seed: int,
) -> pd.DataFrame:
    target = "winner_wide_direct"
    comparators = [method for method in methods if method != target]
    matrices = {
        method: item["signed"][int(np.flatnonzero(item["lags"] == 1)[0])]
        for method, item in methods.items()
    }
    rng = np.random.default_rng(seed)
    rows = []
    d = matrices[target].shape[1]
    for reference in (
        "randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54"
    ):
        ref = references[reference]
        observed = {
            method: binary_metrics(matrix, ref["labels"], ref["mask"])
            for method, matrix in matrices.items()
        }
        samples = {
            (method, metric): []
            for method in comparators for metric in ("auroc", "auprc")
        }
        for _ in range(n_boot):
            columns = rng.integers(0, d, size=d)
            target_values = binary_metrics(
                matrices[target][:, columns], ref["labels"][:, columns],
                ref["mask"][:, columns],
            )
            for method in comparators:
                other_values = binary_metrics(
                    matrices[method][:, columns], ref["labels"][:, columns],
                    ref["mask"][:, columns],
                )
                for metric in ("auroc", "auprc"):
                    samples[(method, metric)].append(
                        target_values[metric] - other_values[metric]
                    )
        for method in comparators:
            for metric in ("auroc", "auprc"):
                values = np.asarray(samples[(method, metric)], dtype=np.float64)
                rows.append(
                    {
                        "reference": reference, "left": target, "right": method,
                        "metric": metric,
                        "observed_difference": observed[target][metric] - observed[method][metric],
                        "ci_low": float(np.nanquantile(values, 0.025)),
                        "ci_high": float(np.nanquantile(values, 0.975)),
                        "bootstrap_replicates": n_boot,
                        "bootstrap_unit": "source_column",
                    }
                )
    return pd.DataFrame(rows)


def _cell(frame: pd.DataFrame, method: str, reference: str) -> str:
    row = frame[(frame.method == method) & (frame.reference == reference)].iloc[0]
    return f"{row.auroc:.3f}/{row.auprc:.3f}"


def write_report(
    output: Path, external: pd.DataFrame, counts: pd.DataFrame,
    relationships: pd.DataFrame, neuromod: pd.DataFrame,
    presence_bootstrap: pd.DataFrame,
    validity: np.ndarray,
) -> None:
    lag1 = external[external.lag_frames == 1]
    count1 = counts[
        (counts.lag_frames == 1) & (counts.reference == "cook_struct_54")
    ].set_index("method")
    winner = lag1[lag1.method == "winner_wide_direct"].set_index("reference")
    previous = lag1[lag1.method == "importance_weighting_full"].set_index("reference")
    published = lag1[lag1.method == "sbtg_published"].set_index("reference")
    randi_old = presence_bootstrap[
        (presence_bootstrap.reference == "randi_wild_type")
        & (presence_bootstrap.right == "importance_weighting_full")
        & (presence_bootstrap.metric == "auroc")
    ].iloc[0]
    randi_published = presence_bootstrap[
        (presence_bootstrap.reference == "randi_wild_type")
        & (presence_bootstrap.right == "sbtg_published")
        & (presence_bootstrap.metric == "auroc")
    ].iloc[0]
    lines = [
        "# Post-freeze external analysis of the atlas-blind predictive winner",
        "",
        "## Bottom line",
        "",
        f"The wide-flow winner's direct repaired-response ensemble reaches lag-1 Randi WT AUROC/AUPRC **{winner.loc['randi_wild_type','auroc']:.3f}/{winner.loc['randi_wild_type','auprc']:.3f}** and Cook structural **{winner.loc['cook_struct_54','auroc']:.3f}/{winner.loc['cook_struct_54','auprc']:.3f}**. The previous generator under the same direct estimator reaches {previous.loc['randi_wild_type','auroc']:.3f}/{previous.loc['randi_wild_type','auprc']:.3f} and {previous.loc['cook_struct_54','auroc']:.3f}/{previous.loc['cook_struct_54','auprc']:.3f}; SBTG-published reaches {published.loc['randi_wild_type','auroc']:.3f}/{published.loc['randi_wild_type','auprc']:.3f} and {published.loc['cook_struct_54','auroc']:.3f}/{published.loc['cook_struct_54','auprc']:.3f}.",
        "",
        "The generator was frozen using predictive scores and multistep rollout only. These atlas values are therefore post-selection convergent validation, not architecture-selection inputs.",
        "",
        f"Under the paired source-column bootstrap, winner-minus-old-direct Randi AUROC is **{randi_old.observed_difference:+.3f}** (95% CI {randi_old.ci_low:+.3f} to {randi_old.ci_high:+.3f}); winner-minus-SBTG-published is **{randi_published.observed_difference:+.3f}** ({randi_published.ci_low:+.3f} to {randi_published.ci_high:+.3f}). These intervals describe source-neuron sensitivity of frozen matrices, not animal or refit uncertainty.",
        "",
        "## Lag-1 edge-presence results",
        "",
        "| Method | Randi WT | Cook structural | Cook chemical | Cook gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        lines.append(
            f"| {LABELS[method]} | {_cell(lag1, method, 'randi_wild_type')} | "
            f"{_cell(lag1, method, 'cook_struct_54')} | "
            f"{_cell(lag1, method, 'cook_chem_54')} | "
            f"{_cell(lag1, method, 'cook_gap_54')} |"
        )
    lines.extend(
        [
            "",
            "## Cook structural count-strength results",
            "",
            "| Method | All-pair rho | Positive-edge rho | Positive-edge tau | Within-source rho | NDCG |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        row = count1.loc[method]
        lines.append(
            f"| {LABELS[method]} | {row.all_pair_spearman:.3f} | "
            f"{row.positive_edge_spearman:.3f} | {row.positive_edge_kendall:.3f} | "
            f"{row.within_source_rank_spearman:.3f} | {row.ndcg_all_pairs:.3f} |"
        )
    lines.extend(
        [
            "",
            "All-pair correlation partly repeats edge-presence separation because most Cook pairs are zero. Positive-edge correlation is the cleaner conditional strength view. Neither score nor anatomical count is treated as a calibrated physical effect.",
            "",
            "## Matrix relationship to the previous generator",
            "",
        ]
    )
    relation = relationships[
        (relationships.left == "importance_weighting_full")
        & (relationships.right == "winner_wide_direct")
        & (relationships.lag_frames == 1)
    ].iloc[0]
    eligible = neuromod[neuromod.scope == "eligible_sources"]
    best_rows = {}
    for method in ("winner_wide_direct", "sbtg_published"):
        for network in ("monoamine_all", "neuropeptide_all", "neuromodulator_union"):
            frame = eligible[
                (eligible.method == method) & (eligible.network == network)
            ]
            best_rows[(method, network)] = frame.loc[frame.auroc.idxmax()]
    winner_mono = best_rows[("winner_wide_direct", "monoamine_all")]
    winner_pep = best_rows[("winner_wide_direct", "neuropeptide_all")]
    published_mono = best_rows[("sbtg_published", "monoamine_all")]
    published_pep = best_rows[("sbtg_published", "neuropeptide_all")]
    lines.extend(
        [
            f"At lag 1, old-direct versus winner-direct signed Spearman is **{relation.signed_spearman:.3f}**, absolute-score Spearman is **{relation.absolute_spearman:.3f}**, sign agreement is **{relation.sign_agreement:.3f}**, and top-10% Jaccard is **{relation.top_10pct_jaccard:.3f}**.",
            "",
            "## Neuromodulator lag correspondence",
            "",
            f"On eligible-source scoring, the winner's best tested monoamine-all horizon is **{int(winner_mono.lag_frames)} frames / {winner_mono.lag_seconds:g} s** (AUROC {winner_mono.auroc:.3f}, {int(winner_mono.n_eligible_sources)} eligible sources); its best neuropeptide-all horizon is **{int(winner_pep.lag_frames)} frames / {winner_pep.lag_seconds:g} s** (AUROC {winner_pep.auroc:.3f}, {int(winner_pep.n_eligible_sources)} sources). SBTG-published is higher: monoamine-all {published_mono.auroc:.3f} at lag {int(published_mono.lag_frames)}, and neuropeptide-all {published_pep.auroc:.3f} at lag {int(published_pep.lag_frames)}.",
            "",
            "These are post hoc descriptive maxima. Flow values are cumulative-response horizons, SBTG values are score-product lags, and neither estimates a physical receptor or transmission delay. The monoamine panel has only five eligible source neurons.",
            "",
            "## Condition attainment and claim boundary",
            "",
            f"Mean primary compatibility-validity for the winner-direct ensemble is **{np.mean(validity):.3f}**. Direct weighting remains a lower-attainment estimator than progressive SMC in the separate frozen estimator study, so predictive-generator and finite-particle-estimator improvements should not be conflated.",
            "",
            "These are observational, model-relative repaired responses of learned finite-memory dynamics. They are not physical interventions, direct synapses, anatomical identification, or rewiring.",
            "",
            "## Files",
            "",
            "- `external_metrics_by_lag.csv`: Randi/Cook presence metrics.",
            "- `cook_count_metrics_by_lag.csv`: continuous Cook strength metrics.",
            "- `paired_source_bootstrap.csv`: winner-minus-comparator Cook intervals.",
            "- `presence_source_bootstrap.csv`: paired Randi/Cook AUROC and AUPRC intervals.",
            "- `neuromodulator_lag_metrics.csv`: Bentley correspondence.",
            "- `matrix_relationships.csv`: aligned matrix relationships.",
            "- `aligned_winner_comparison_matrices.npz`: exact 54-neuron matrices.",
            "- `protocol.json`, `validation.json`, `input_artifact_checksums.csv`, `checksums.sha256`: provenance and integrity.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner-direct-run", type=Path, required=True)
    parser.add_argument("--fair-analysis", type=Path, required=True)
    parser.add_argument("--published-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--winner-model-id", default="flow_wide128_dropout10")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    winner, validity, winner_neurons, winner_lags = load_direct_full(
        args.winner_direct_run.resolve(), args.winner_model_id
    )
    fair_path = args.fair_analysis.resolve() / "aligned_all_lag_matrices.npz"
    with np.load(fair_path, allow_pickle=False) as data:
        neurons = data["neurons"].astype(str).tolist()
        if winner_neurons.astype(str).tolist() != neurons:
            raise RuntimeError("winner-direct and contextual neuron order mismatch")
        methods = {
            "importance_weighting_full": {
                "lags": data["importance_weighting__lags"].astype(int),
                "signed": data["importance_weighting__signed"].astype(np.float64),
            },
            "winner_wide_direct": {
                "lags": winner_lags.astype(int),
                "signed": winner.astype(np.float64),
            },
            "terminal_smc_full": {
                "lags": data["smc_ess__lags"].astype(int),
                "signed": data["smc_ess__signed"].astype(np.float64),
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
    references, networks = load_references(args.published_release.resolve(), neurons)
    external = external_rows(methods, references)
    counts = cook_count_rows(methods, references)
    neuromod = neuromodulator_rows(methods, networks)
    relationships = relationship_rows(methods)
    bootstrap = source_bootstrap_differences(
        methods, references, n_boot=args.bootstrap, seed=args.seed,
        target="winner_wide_direct",
    )
    presence_bootstrap = paired_presence_bootstrap(
        methods, references, n_boot=args.bootstrap, seed=args.seed + 17
    )
    external.to_csv(output / "external_metrics_by_lag.csv", index=False)
    counts.to_csv(output / "cook_count_metrics_by_lag.csv", index=False)
    neuromod.to_csv(output / "neuromodulator_lag_metrics.csv", index=False)
    relationships.to_csv(output / "matrix_relationships.csv", index=False)
    bootstrap.to_csv(output / "paired_source_bootstrap.csv", index=False)
    presence_bootstrap.to_csv(output / "presence_source_bootstrap.csv", index=False)
    np.savez_compressed(
        output / "aligned_winner_comparison_matrices.npz",
        neurons=np.asarray(neurons), winner_direct_validity=validity.astype(np.float32),
        **{f"{name}__lags": item["lags"] for name, item in methods.items()},
        **{f"{name}__signed": item["signed"].astype(np.float32) for name, item in methods.items()},
    )
    write_report(
        output, external, counts, relationships, neuromod,
        presence_bootstrap, validity,
    )
    inputs = [
        fair_path, args.fair_analysis.resolve() / "manifest.json",
        args.winner_direct_run.resolve() / "manifest.json",
        *sorted((args.winner_direct_run.resolve() / "responses").glob("*.npz")),
        args.published_release.resolve() / "reference_data/functional_atlas/aligned_atlas_wild_type.npz",
        args.published_release.resolve() / "reference_data/connectome/A_struct.npy",
        args.published_release.resolve() / "reference_data/connectome/A_chem.npy",
        args.published_release.resolve() / "reference_data/connectome/A_gap.npy",
    ]
    pd.DataFrame(
        [{"path": str(path.resolve()), "sha256": sha256(path.resolve()), "size_bytes": path.resolve().stat().st_size} for path in inputs]
    ).to_csv(output / "input_artifact_checksums.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "analysis": "post-freeze external analysis of atlas-blind predictive winner",
                "atlas_firewall": "winner selected before external references were loaded",
                "matrix_orientation": "target row, source column",
                "bootstrap": {"unit": "source column", "replicates": args.bootstrap, "seed": args.seed},
                "claim_boundary": "model-relative observational repaired response; not causal or anatomical identification",
            }, indent=2, sort_keys=True,
        ) + "\n"
    )
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "methods": list(methods),
        "winner_direct_outputs": len(list((args.winner_direct_run.resolve() / "responses").glob("*.npz"))),
        "all_finite": all(np.isfinite(item["signed"]).all() for item in methods.values()),
        "all_54_by_54": all(item["signed"].shape[1:] == (54, 54) for item in methods.values()),
        "external_rows": len(external), "cook_count_rows": len(counts),
        "neuromodulator_rows": len(neuromod), "matrix_relationship_rows": len(relationships),
        "bootstrap_rows": len(bootstrap),
        "presence_bootstrap_rows": len(presence_bootstrap),
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    write_checksums(output)


if __name__ == "__main__":
    main()
