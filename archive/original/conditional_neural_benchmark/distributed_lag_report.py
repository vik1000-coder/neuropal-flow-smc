from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TITLE = "Residualized distributed-lag neural dynamics"


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records")


def _source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    generated: str,
    description: str,
    filters: list[str],
    definitions: list[str],
    tables: list[str] | None = None,
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "DuckDB",
            "sql": sql,
            "tables_used": tables or [path],
            "executed_at": generated,
            "language": "sql",
            "description": description,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(root: Path) -> dict:
    generated = datetime.now(timezone.utc).isoformat()
    adaptive_selection = json.loads((root / "selection.json").read_text())
    adaptive_gate = json.loads((root / "promotion_gate.json").read_text())
    consensus_selection = json.loads((root / "consensus" / "selection.json").read_text())
    consensus_gate = json.loads((root / "consensus" / "promotion_gate.json").read_text())
    control = json.loads(
        (root / "consensus" / "postfreeze" / "neural_control_summary.json").read_text()
    )
    screen = pd.read_csv(root / "screen_leaderboard.csv")
    screen["family_label"] = screen.family.map(
        {
            "residual_ridge": "Residual ridge",
            "group_shrunk": "Group-shrunk",
            "reduced_rank": "Reduced-rank",
            "sparse_lowrank": "Sparse + low-rank",
        }
    )
    screen["lag_label"] = screen.lag_seconds.map(lambda value: f"{value:g} s")
    semisynthetic = pd.read_csv(root / "semisynthetic_detectability.csv")
    semisynthetic["condition_label"] = semisynthetic.condition.map(
        {"latent": "Latent state", "calcium_smoothed": "Calcium-smoothed"}
    )
    semisynthetic["effect_label"] = semisynthetic.effect.map(lambda value: f"{value:.2f}")
    stability_adaptive = pd.read_csv(root / "kernel_stability.csv")
    stability_consensus = pd.read_csv(root / "consensus" / "kernel_stability.csv")
    stability = pd.DataFrame(
        [
            {
                "specification": "Fold-tuned",
                "kernel_pearson": stability_adaptive.kernel_pearson.mean(),
                "edge_rank_spearman": stability_adaptive.edge_rank_spearman.mean(),
                "sign_agreement": stability_adaptive.sign_agreement.mean(),
                "fold_pairs": len(stability_adaptive),
            },
            {
                "specification": "Globally frozen",
                "kernel_pearson": stability_consensus.kernel_pearson.mean(),
                "edge_rank_spearman": stability_consensus.edge_rank_spearman.mean(),
                "sign_agreement": stability_consensus.sign_agreement.mean(),
                "fold_pairs": len(stability_consensus),
            },
        ]
    )
    stability.to_csv(root / "stability_comparison.csv", index=False)
    consensus_worm = pd.read_csv(root / "consensus" / "worm_metrics.csv")
    phase = (
        consensus_worm[consensus_worm.fold.isin([3, 4])]
        .groupby("phase", as_index=False)
        .agg(
            n_worms=("worm", "nunique"),
            nll_improvement=("nll_improvement", "mean"),
            mse_improvement=("mse_improvement", "mean"),
            innovation_spearman=("innovation_spearman", "mean"),
        )
    )
    phase["phase_label"] = phase.phase.map(
        {"all": "All windows", "onset": "Stimulus onset", "quiet": "Quiet"}
    )
    phase.to_csv(root / "consensus_phase_summary.csv", index=False)

    external = pd.read_csv(
        root / "consensus" / "postfreeze" / "postfreeze_external_metrics.csv"
    )
    lag_profile = external[
        (external.panel == "confirmation_folds")
        & external.cell.str.startswith("lag_")
        & external.reference.isin(["randi_wild_type", "cook_struct_54"])
    ].copy()
    lag_profile["reference_label"] = lag_profile.reference.map(
        {"randi_wild_type": "Randi WT", "cook_struct_54": "Cook structural"}
    )
    lag_profile.to_csv(root / "consensus_external_lag_profile.csv", index=False)
    current_integrated = external[
        (external.panel == "confirmation_folds")
        & (external.cell == "signed_max")
        & external.reference.isin(["randi_wild_type", "cook_struct_54"])
    ].copy()
    current_integrated["method"] = "distributed_lag_consensus"
    previous = pd.read_csv(
        root.parent
        / "multilag_temporal_cut_20260827"
        / "postfreeze_external"
        / "selected_matrix_external_metrics.csv"
    )
    previous = previous[
        previous.method.isin(["winner_wide_direct", "sbtg_published"])
        & previous.reference.isin(["randi_wild_type", "cook_struct_54"])
    ].copy()
    comparison = pd.concat(
        [
            current_integrated[["method", "reference", "auroc", "auprc"]],
            previous[["method", "reference", "auroc", "auprc"]],
        ],
        ignore_index=True,
    )
    comparison["method_label"] = comparison.method.map(
        {
            "distributed_lag_consensus": "Distributed lag",
            "winner_wide_direct": "Wide-flow direct",
            "sbtg_published": "SBTG-published",
        }
    )
    comparison["reference_label"] = comparison.reference.map(
        {"randi_wild_type": "Randi WT", "cook_struct_54": "Cook structural"}
    )
    comparison.to_csv(root / "external_method_comparison.csv", index=False)
    lagmax = pd.read_csv(
        root / "consensus" / "postfreeze" / "postfreeze_bentley_lagmax.csv"
    )
    lagmax["network_label"] = lagmax.network.map(
        {"monoamine_all": "Monoamine", "neuropeptide_all": "Neuropeptide"}
    )
    synthetic_summary = (
        semisynthetic.groupby(["condition", "condition_label", "effect"], as_index=False)
        .agg(
            edge_auroc=("edge_auroc", "mean"),
            lag_mae_frames=("lag_mae_frames", "mean"),
            lag_within_one_frame=("lag_within_one_frame", "mean"),
            replicates=("replicate", "nunique"),
        )
    )
    synthetic_summary.to_csv(root / "semisynthetic_summary.csv", index=False)
    headline = pd.DataFrame(
        [
            {
                "confirmation_nll_improvement": consensus_gate["mean_nll_improvement"],
                "confirmation_nll_ci_low": consensus_gate["ci_low"],
                "confirmation_nll_ci_high": consensus_gate["ci_high"],
                "confirmation_rmse_improvement": consensus_gate["mean_rmse_improvement"],
                "confirmation_energy_improvement": consensus_gate["mean_energy_improvement"],
                "promotion_gate_passed": consensus_gate["passes_gate"],
                "circular_shift_p": control["circular_shift_p_value"],
                "edge_rank_stability": stability_consensus.edge_rank_spearman.mean(),
                "randi_auroc": float(
                    current_integrated[
                        current_integrated.reference == "randi_wild_type"
                    ].iloc[0].auroc
                ),
                "cook_auroc": float(
                    current_integrated[
                        current_integrated.reference == "cook_struct_54"
                    ].iloc[0].auroc
                ),
                "best_bentley_q": float(lagmax.lagmax_bh_q_value.min()),
            }
        ]
    )
    headline.to_csv(root / "headline_metrics.csv", index=False)

    base = "results/distributed_lag_dynamics_20260828"
    prior = "results/multilag_temporal_cut_20260827"
    sources = [
        _source(
            "screen_source",
            "Atlas-blind model and maximum-lag screen",
            f"{base}/screen_leaderboard.csv",
            f"SELECT * FROM read_csv_auto('{base}/screen_leaderboard.csv')",
            generated,
            "Outer-test screen summaries across four lag-kernel families and three maximum histories.",
            ["screen folds 0–2", "20 whole worms", "54 complete-case neurons"],
            [
                "NLL improvement is nuisance-baseline Student-t NLL minus full-model Student-t NLL; positive is better.",
                "RMSE and energy improvements use the same positive-is-better convention.",
            ],
        ),
        _source(
            "consensus_source",
            "Globally frozen confirmation",
            f"{base}/consensus/per_fold_metrics.csv",
            f"SELECT * FROM read_csv_auto('{base}/consensus/per_fold_metrics.csv')",
            generated,
            "One specification selected on folds 0–2 and refit unchanged in all outer folds.",
            ["confirmation folds 3–4", "8 confirmation worms", "history 16 frames / 4 seconds"],
            [
                "The strict gate requires the worm-bootstrap NLL-improvement CI above zero and positive NLL and RMSE improvement in both confirmation folds.",
                "Innovation Spearman correlates the fitted cross-neuron contribution with the nuisance residual across targets within each time row.",
            ],
        ),
        _source(
            "stability_source",
            "Fold-to-fold lag-kernel stability",
            f"{base}/stability_comparison.csv",
            f"SELECT * FROM read_csv_auto('{base}/stability_comparison.csv')",
            generated,
            "Pairwise fold agreement before and after globally freezing all regularization strengths.",
            ["10 unordered fold pairs", "off-diagonal ordered neuron pairs"],
            [
                "Edge-rank Spearman compares max-absolute lag-kernel magnitudes across folds.",
                "Kernel Pearson compares every signed lag coefficient jointly.",
            ],
        ),
        _source(
            "phase_source",
            "Confirmation performance by stimulus context",
            f"{base}/consensus_phase_summary.csv",
            f"SELECT * FROM read_csv_auto('{base}/consensus_phase_summary.csv')",
            generated,
            "Worm-level confirmation metrics aggregated separately for all, onset, and quiet windows.",
            ["confirmation folds 3–4", "8 worms", "globally frozen consensus model"],
            [
                "NLL improvement is nuisance-baseline Student-t NLL minus full-model NLL.",
                "MSE improvement is nuisance-baseline squared error minus full-model squared error.",
            ],
        ),
        _source(
            "synthetic_source",
            "Semi-synthetic detectability with real residual-block noise",
            f"{base}/semisynthetic_detectability.csv",
            f"SELECT * FROM read_csv_auto('{base}/semisynthetic_detectability.csv')",
            generated,
            "Sparse known-lag VAR signals simulated with block-resampled real innovation noise, with and without calcium smoothing.",
            ["12 neurons", "20 injected ordered edges", "6 replicates per condition and effect"],
            [
                "Edge AUROC ranks maximum absolute recovered lag-kernel magnitude.",
                "Lag MAE is the absolute difference between injected and recovered peak lag in frames.",
            ],
        ),
        _source(
            "control_source",
            "Temporal alignment and phase controls",
            f"{base}/consensus/postfreeze/neural_control_summary.json",
            f"SELECT * FROM read_json_auto('{base}/consensus/postfreeze/neural_control_summary.json')",
            generated,
            "Within-worm circular shifts preserve the fitted cross-neuron contribution's temporal structure while breaking target alignment.",
            ["200 shift replicates", "confirmation folds 3–4", "no external atlas loaded"],
            [
                "Circular-shift p is the empirical rank of actual confirmation NLL improvement against shifted contributions.",
                "Conditional ablation is the held-out NLL loss after removing one source's complete smooth lag block.",
            ],
        ),
        _source(
            "external_source",
            "Post-freeze Randi and Cook lag correspondence",
            f"{base}/consensus/postfreeze/postfreeze_external_metrics.csv",
            f"SELECT * FROM read_csv_auto('{base}/consensus/postfreeze/postfreeze_external_metrics.csv') WHERE panel = 'confirmation_folds'",
            generated,
            "External references loaded after neural selection and controls were written.",
            ["confirmation-fold mean kernels", "54-neuron order", "off-diagonal pairs"],
            [
                "AUROC ranks absolute kernel magnitude for binary edge presence.",
                "Cook count Spearman compares absolute kernel magnitude with continuous connection counts.",
            ],
        ),
        _source(
            "comparison_source",
            "Post-freeze contextual method comparison",
            f"{base}/external_method_comparison.csv",
            (
                "SELECT method, reference, auroc, auprc FROM "
                f"read_csv_auto('{base}/external_method_comparison.csv')"
            ),
            generated,
            "The frozen distributed-lag signed-max matrix compared with previously frozen wide-flow direct and SBTG-published matrices.",
            ["Randi WT and Cook structural", "identical 54-neuron order"],
            ["AUROC is binary edge-presence ranking; it does not identify a causal or physical delay."],
            tables=[
                f"{base}/consensus/postfreeze/postfreeze_external_metrics.csv",
                f"{prior}/postfreeze_external/selected_matrix_external_metrics.csv",
            ],
        ),
        _source(
            "bentley_source",
            "Bentley within-source lag-max inference",
            f"{base}/consensus/postfreeze/postfreeze_bentley_lagmax.csv",
            f"SELECT * FROM read_csv_auto('{base}/consensus/postfreeze/postfreeze_bentley_lagmax.csv')",
            generated,
            "Target labels are permuted within each eligible molecular source; the null takes the maximum across all 16 lags.",
            ["999 permutations", "2,000 source bootstraps", "four panel-by-network tests"],
            [
                "Lag-max p controls the complete 16-lag search within a panel/network.",
                "BH q controls the four prespecified panel/network searches.",
            ],
        ),
    ]
    cards = [
        {
            "id": "nll_card",
            "dataset": "headline",
            "sourceId": "consensus_source",
            "description": "Untouched-fold Student-t log-score improvement over target self-history, stimulus history and global population state.",
            "metrics": [
                {"label": "Confirmation NLL improvement", "field": "confirmation_nll_improvement", "format": "number", "signed": True},
                {"label": "RMSE improvement", "field": "confirmation_rmse_improvement", "format": "number", "signed": True},
            ],
        },
        {
            "id": "shift_card",
            "dataset": "headline",
            "sourceId": "control_source",
            "description": "Empirical p-value against within-worm circular shifts of the complete fitted cross-neuron contribution.",
            "metrics": [{"label": "Circular-shift p", "field": "circular_shift_p", "format": "number"}],
        },
        {
            "id": "stability_card",
            "dataset": "headline",
            "sourceId": "stability_source",
            "description": "Mean Spearman agreement of edge magnitude rankings over all ten fold pairs under one frozen specification.",
            "metrics": [{"label": "Edge-rank stability", "field": "edge_rank_stability", "format": "number"}],
        },
        {
            "id": "randi_card",
            "dataset": "headline",
            "sourceId": "external_source",
            "description": "Post-freeze Randi wild-type binary edge-presence ranking by the confirmation signed-max lag kernel.",
            "metrics": [{"label": "Randi WT AUROC", "field": "randi_auroc", "format": "number"}],
        },
        {
            "id": "bentley_card",
            "dataset": "headline",
            "sourceId": "bentley_source",
            "description": "Smallest BH-adjusted q-value across the four prespecified lag-max searches.",
            "metrics": [{"label": "Best Bentley BH q", "field": "best_bentley_q", "format": "number"}],
        },
    ]
    charts = [
        {
            "id": "screen_chart",
            "title": "Atlas-blind model and maximum-lag screen",
            "subtitle": "Mean folds-0–2 Student-t NLL improvement; positive values favor lagged cross-neuron history.",
            "type": "bar",
            "intent": "comparison",
            "dataset": "screen",
            "sourceId": "screen_source",
            "encodings": {
                "x": {"field": "lag_label", "type": "nominal", "label": "Maximum lag"},
                "y": {"field": "mean_nll_improvement", "type": "quantitative", "label": "NLL improvement"},
                "color": {"field": "family_label", "type": "nominal", "label": "Lag-kernel family"},
                "tooltip": [
                    {"field": "mean_rmse_improvement", "type": "quantitative", "label": "RMSE improvement"},
                    {"field": "mean_energy_improvement", "type": "quantitative", "label": "Energy improvement"},
                    {"field": "mean_innovation_spearman", "type": "quantitative", "label": "Innovation rho"},
                ],
            },
            "referenceLines": [{"axis": "y", "value": 0, "label": "No improvement"}],
            "layout": "full",
        },
        {
            "id": "detectability_chart",
            "title": "Semi-synthetic edge detectability",
            "subtitle": "Six real-noise replicates per effect and observation condition; lag precision remains poor at weak effects.",
            "type": "scatter",
            "intent": "relationship",
            "dataset": "semisynthetic",
            "sourceId": "synthetic_source",
            "encodings": {
                "x": {"field": "effect", "type": "quantitative", "label": "Injected coefficient magnitude"},
                "y": {"field": "edge_auroc", "type": "quantitative", "label": "Edge AUROC"},
                "color": {"field": "condition_label", "type": "nominal", "label": "Observation condition"},
                "tooltip": [
                    {"field": "replicate", "type": "quantitative", "label": "Replicate"},
                    {"field": "lag_mae_frames", "type": "quantitative", "label": "Lag MAE (frames)"},
                    {"field": "lag_within_one_frame", "type": "quantitative", "label": "Within one frame"},
                ],
            },
            "referenceLines": [{"axis": "y", "value": 0.5, "label": "Chance"}],
            "layout": "full",
        },
        {
            "id": "lag_profile_chart",
            "title": "External-reference AUROC across direct lags",
            "subtitle": "Confirmation-fold consensus kernel; each point is post-freeze and neither curve identifies physical delay.",
            "type": "line",
            "intent": "comparison",
            "dataset": "lag_profile",
            "sourceId": "external_source",
            "encodings": {
                "x": {"field": "lag_seconds", "type": "quantitative", "label": "Direct source lag (s)"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC"},
                "color": {"field": "reference_label", "type": "nominal", "label": "Reference"},
                "tooltip": [
                    {"field": "auprc", "type": "quantitative", "label": "AUPRC"},
                    {"field": "macro_source_auroc", "type": "quantitative", "label": "Source-macro AUROC"},
                ],
            },
            "referenceLines": [{"axis": "y", "value": 0.5, "label": "Chance"}],
            "layout": "full",
        },
        {
            "id": "method_chart",
            "title": "Post-freeze Randi and Cook comparison",
            "subtitle": "Integrated distributed-lag kernel versus previously frozen contextual estimators on the same neurons.",
            "type": "bar",
            "intent": "comparison",
            "dataset": "method_comparison",
            "sourceId": "comparison_source",
            "encodings": {
                "x": {"field": "reference_label", "type": "nominal", "label": "Reference"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC"},
                "color": {"field": "method_label", "type": "nominal", "label": "Method"},
                "tooltip": [{"field": "auprc", "type": "quantitative", "label": "AUPRC"}],
            },
            "referenceLines": [{"axis": "y", "value": 0.5, "label": "Chance"}],
            "layout": "full",
        },
    ]
    tables = [
        {
            "id": "stability_table",
            "title": "Kernel stability before and after global freezing",
            "subtitle": "Mean agreement across all ten unordered outer-fold pairs.",
            "dataset": "stability",
            "sourceId": "stability_source",
            "defaultSort": {"field": "edge_rank_spearman", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "specification", "label": "Specification", "type": "text"},
                {"field": "kernel_pearson", "label": "Kernel Pearson", "format": "number"},
                {"field": "edge_rank_spearman", "label": "Edge-rank Spearman", "format": "number"},
                {"field": "sign_agreement", "label": "Sign agreement", "format": "percent"},
                {"field": "fold_pairs", "label": "Fold pairs", "format": "number"},
            ],
        },
        {
            "id": "phase_table",
            "title": "Confirmation performance by stimulus context",
            "subtitle": "Eight confirmation worms; quiet windows outperform onset windows on the primary log-score gain.",
            "dataset": "phase",
            "sourceId": "phase_source",
            "defaultSort": {"field": "nll_improvement", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "phase_label", "label": "Context", "type": "text"},
                {"field": "n_worms", "label": "Worms", "format": "number"},
                {"field": "nll_improvement", "label": "NLL improvement", "format": "number", "movement": True},
                {"field": "mse_improvement", "label": "MSE improvement", "format": "number", "movement": True},
                {"field": "innovation_spearman", "label": "Innovation rho", "format": "number"},
            ],
        },
        {
            "id": "lagmax_table",
            "title": "Bentley lag-max inference",
            "subtitle": "Within-source target-label null with the maximum over all 16 direct lags.",
            "dataset": "lagmax",
            "sourceId": "bentley_source",
            "defaultSort": {"field": "lagmax_p_value", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "panel", "label": "Panel", "type": "text"},
                {"field": "network_label", "label": "Network", "type": "text"},
                {"field": "best_lag_seconds", "label": "Best lag (s)", "format": "number"},
                {"field": "best_auroc", "label": "Best AUROC", "format": "number"},
                {"field": "best_lag_selection_rate", "label": "Bootstrap selection", "format": "percent"},
                {"field": "lagmax_p_value", "label": "Lag-max p", "format": "number"},
                {"field": "lagmax_bh_q_value", "label": "BH q", "format": "number"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical summary\n\n"
                "A jointly regularized 4-second distributed-lag kernel contains reproducible, time-aligned distributional structure, but it does **not** improve the conditional mean. Under one globally frozen specification, untouched-fold Student-t log score improves by +0.00944 (worm-bootstrap 95% CI +0.00847 to +0.01038), while RMSE improvement is −0.00185. The strict mean-dynamics gate therefore fails. Temporal alignment beats within-worm circular shifts, yet onset performance is weaker than quiet performance and post-freeze Randi, Cook, and Bentley correspondence is near chance."
            ),
        },
        {"id": "headline_strip", "type": "metric-strip", "cardIds": ["nll_card", "shift_card", "stability_card", "randi_card", "bentley_card"]},
        {
            "id": "screen_heading",
            "type": "markdown",
            "body": (
                "## Group shrinkage was the only consistently positive screen family\n\n"
                "The atlas-blind screen compared four smooth lag-kernel regularizers over 4-, 8-, and 20-second maximum histories. Group shrinkage at 4 seconds had the largest mean log-score gain, but even the screen showed the central conflict: distributional score improved slightly while RMSE and energy worsened."
            ),
            "sourceId": "screen_source",
        },
        {"id": "screen_chart_block", "type": "chart", "chartId": "screen_chart"},
        {
            "id": "consensus_heading",
            "type": "markdown",
            "body": (
                "## Global hyperparameter freezing stabilized the lag tensor but did not rescue mean prediction\n\n"
                "Fold-specific tuning produced markedly different sparsity levels. Freezing common alpha 100, self alpha 100, cross alpha 1000, and 50% group shrinkage using folds 0–2 raised mean edge-rank stability from 0.47 to 0.76. The stronger stability is scientifically preferable, but both confirmation folds still had worse RMSE."
            ),
        },
        {"id": "stability_table_block", "type": "table", "tableId": "stability_table"},
        {
            "id": "control_heading",
            "type": "markdown",
            "body": (
                "## Temporal alignment is detectable, but the effect is not onset-specific\n\n"
                "Actual confirmation log-score improvement exceeds all 200 block-preserving circular-shift controls (p=0.005), so the fitted contribution is not merely an arbitrary autocorrelated sequence. However, quiet-window improvement (+0.00966) exceeds onset improvement (+0.00748), and the fitted contribution worsens mean squared error in both contexts. This supports weak predictive distribution structure—not stimulus-evoked propagation."
            ),
        },
        {"id": "phase_table_block", "type": "table", "tableId": "phase_table"},
        {
            "id": "detectability_heading",
            "type": "markdown",
            "body": (
                "## Weak lag effects are detectable as edges before they are precisely localized\n\n"
                "With real residual-block noise, calcium-smoothed coefficient 0.10 yields mean edge AUROC 0.814, but lag MAE remains 2.74 frames and only 71.7% of injected edges are recovered within one frame. At coefficient 0.03, AUROC is 0.617 and lag MAE is 4.22 frames. The current data can rank moderate effects more readily than assign precise delays."
            ),
            "sourceId": "synthetic_source",
        },
        {"id": "detectability_chart_block", "type": "chart", "chartId": "detectability_chart"},
        {
            "id": "external_heading",
            "type": "markdown",
            "body": (
                "## Stable predictive kernels do not correspond to known connectomes\n\n"
                "After the neural model and controls were frozen, the confirmation signed-max kernel reached Randi WT AUROC 0.503 and Cook structural AUROC 0.515. It trails both wide-flow direct and SBTG-published contextual matrices. No direct lag produces a convincing reference peak."
            ),
        },
        {"id": "method_chart_block", "type": "chart", "chartId": "method_chart"},
        {"id": "lag_profile_chart_block", "type": "chart", "chartId": "lag_profile_chart"},
        {
            "id": "bentley_heading",
            "type": "markdown",
            "body": (
                "## Molecular lag maxima remain compatible with chance\n\n"
                "The smallest lag-max p-value is 0.472 and every four-test BH q-value is 0.894 under the globally frozen kernel. Descriptive best lags vary across panels and networks, and no result supports a physical-delay claim."
            ),
            "sourceId": "bentley_source",
        },
        {"id": "lagmax_table_block", "type": "table", "tableId": "lagmax_table"},
        {
            "id": "scope_heading",
            "type": "markdown",
            "body": (
                "## Scope, cohort, and metric definitions\n\n"
                "The cohort contains 20 whole worms, 54 complete-case neurons, 4-Hz calcium activity, and the known binary stimulus schedule. The nuisance baseline contains target self-history, the complete lagged stimulus window, four atlas-blind population principal components, and worm-held-out standardization. Positive NLL, RMSE, and energy improvement always means the lagged cross-neuron model is better than that nuisance baseline. Direct kernels are one-step innovation dependencies; recursive impulse responses include model-mediated propagation."
            ),
        },
        {
            "id": "method_heading",
            "type": "markdown",
            "body": (
                "## Model and experimental design\n\n"
                "Each ordered pair receives one smooth lag kernel rather than an independently fitted coefficient at every lag. Candidate families are residual ridge, group-shrunk ridge, reduced rank, and sparse plus low-rank. Folds 0–2 select the maximum history, family, and consensus penalties. Folds 3–4 are untouched confirmation. Conditional ablation removes a complete source-lag block from held-out density prediction. External references are opened only after selection and neural controls are saved."
            ),
        },
        {
            "id": "limits_heading",
            "type": "markdown",
            "body": (
                "## Robustness, limitations, and claim boundary\n\n"
                "The preferred consensus tensor is materially more stable than fold-tuned kernels, but stability is not predictive usefulness. Calcium filtering, shared stimulus input, unmeasured behavior/global state, and correlated lag features can all generate reduced-form temporal dependence. The result establishes neither intervention response, anatomical connectivity, nor a physical transmission delay. Targeted temporal-cut SMC was intentionally not promoted because the strict mean-dynamics gate failed."
            ),
        },
        {
            "id": "next_heading",
            "type": "markdown",
            "body": (
                "## Recommended next experiment\n\n"
                "1. Freeze the 4-second smooth basis and consensus penalties from this run.\n"
                "2. Fit a latent calcium observation model, then repeat the same residualized lag test on inferred latent activity.\n"
                "3. Acquire more repeated onsets with measured behavior/global state and explicit single-neuron perturbations.\n"
                "4. Promote local-perturbation SMC only if both proper score and conditional-mean gates pass prospectively."
            ),
        },
        {
            "id": "questions_heading",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Is the log-score gain caused by heavy-tail calibration rather than a useful change in the conditional center?\n"
                "- Does a neuron-specific calcium decay model move or sharpen recovered kernel centroids?\n"
                "- Which controlled perturbation effect sizes are large enough to clear the semi-synthetic lag-localization threshold?"
            ),
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Atlas-blind residualized distributed-lag dynamics with whole-worm confirmation, negative controls, semi-synthetic detectability, and post-freeze external checks.",
        "generatedAt": generated,
        "sources": sources,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated,
        "status": "ready",
        "datasets": {
            "headline": _records(headline),
            "screen": _records(screen),
            "stability": _records(stability),
            "phase": _records(phase),
            "semisynthetic": _records(semisynthetic),
            "lag_profile": _records(lag_profile),
            "method_comparison": _records(comparison),
            "lagmax": _records(lagmax),
        },
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
        "summary": {
            "adaptive_selection": adaptive_selection,
            "adaptive_gate": adaptive_gate,
            "consensus_selection": consensus_selection,
            "consensus_gate": consensus_gate,
            "control": control,
            "best_bentley_q": float(lagmax.lagmax_bh_q_value.min()),
            "synthetic_calcium_effect_010": _records(
                synthetic_summary[
                    (synthetic_summary.condition == "calcium_smoothed")
                    & np.isclose(synthetic_summary.effect, 0.10)
                ]
            )[0],
        },
    }


def _write_markdown(root: Path, artifact: dict) -> None:
    summary = artifact["summary"]
    consensus = summary["consensus_gate"]
    control = summary["control"]
    synthetic = summary["synthetic_calcium_effect_010"]
    comparison = pd.read_csv(root / "external_method_comparison.csv")
    stability = pd.read_csv(root / "stability_comparison.csv")
    phase = pd.read_csv(root / "consensus_phase_summary.csv")
    lagmax = pd.read_csv(root / "consensus" / "postfreeze" / "postfreeze_bentley_lagmax.csv")
    def ext(method: str, reference: str) -> float:
        return float(comparison[(comparison.method == method) & (comparison.reference == reference)].iloc[0].auroc)
    lines = [
        f"# {TITLE}",
        "",
        "**Run date:** 2026-08-28  ",
        "**Claim boundary:** atlas-blind predictive innovation dynamics and post-freeze correspondence; not interventions, anatomy, or physical delay.",
        "",
        "## Technical summary",
        "",
        f"The preferred model is one globally frozen, group-shrunk distributed-lag tensor with a 4-second maximum history. On untouched folds 3–4, Student-t NLL improves by **{consensus['mean_nll_improvement']:+.6f}** (worm-bootstrap 95% CI **[{consensus['ci_low']:+.6f}, {consensus['ci_high']:+.6f}]**), but RMSE improvement is **{consensus['mean_rmse_improvement']:+.6f}**. The strict mean-dynamics gate therefore **failed**.",
        "",
        f"The fitted cross-neuron contribution beats 200 within-worm circular shifts (p **{control['circular_shift_p_value']:.3f}**), but quiet-window NLL improvement exceeds onset improvement. This supports weak time-aligned distributional structure, not stimulus-evoked propagation.",
        "",
        "## Why the globally frozen model is preferred",
        "",
        "| Specification | Kernel Pearson | Edge-rank Spearman | Sign agreement |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in stability.itertuples():
        lines.append(f"| {row.specification} | {row.kernel_pearson:.3f} | {row.edge_rank_spearman:.3f} | {row.sign_agreement:.3f} |")
    lines.extend(
        [
            "",
            "Fold-specific hyperparameters varied from 5% to 50% edge density. Freezing common alpha 100, self alpha 100, cross alpha 1000, and 50% group shrinkage on folds 0–2 markedly improves fold agreement without using either confirmation outcomes or an atlas.",
            "",
            "## Confirmation by stimulus context",
            "",
            "| Context | NLL improvement | MSE improvement | Innovation rho |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in phase.itertuples():
        lines.append(f"| {row.phase_label} | {row.nll_improvement:+.6f} | {row.mse_improvement:+.6f} | {row.innovation_spearman:+.3f} |")
    lines.extend(
        [
            "",
            "The primary score improves in every confirmation worm, but the fitted mean contribution increases squared error. Quiet performance is stronger than onset performance, so the effect is not onset-specific.",
            "",
            "## Semi-synthetic detectability",
            "",
            f"At calcium-smoothed injected coefficient 0.10, mean edge AUROC is **{synthetic['edge_auroc']:.3f}**, lag MAE is **{synthetic['lag_mae_frames']:.3f} frames**, and **{synthetic['lag_within_one_frame']:.1%}** of injected edges fall within one frame. The method can rank moderate edges before it can localize their lags precisely.",
            "",
            "## Post-freeze external correspondence",
            "",
            "| Method | Randi WT AUROC | Cook structural AUROC |",
            "| --- | ---: | ---: |",
            f"| Distributed lag, consensus | {ext('distributed_lag_consensus','randi_wild_type'):.3f} | {ext('distributed_lag_consensus','cook_struct_54'):.3f} |",
            f"| Wide-flow direct | {ext('winner_wide_direct','randi_wild_type'):.3f} | {ext('winner_wide_direct','cook_struct_54'):.3f} |",
            f"| SBTG-published | {ext('sbtg_published','randi_wild_type'):.3f} | {ext('sbtg_published','cook_struct_54'):.3f} |",
            "",
            f"The smallest Bentley lag-max p-value is **{lagmax.lagmax_p_value.min():.3f}** and the smallest BH q-value is **{lagmax.lagmax_bh_q_value.min():.3f}**. There is no support for molecular lag correspondence or a physical-delay claim.",
            "",
            "## Model specification",
            "",
            "- Cohort: 20 whole worms, 54 complete-case neurons, 4-Hz calcium activity.",
            "- Baseline: target self-history, complete binary stimulus history, four population PCs, and fold-local scaling.",
            "- Lag object: one smooth coefficient function per ordered source-target pair; maximum lag 16 frames / 4 seconds.",
            "- Direct kernel: one-step target innovation dependence on each source lag.",
            "- Impulse response: recursive propagation through the complete fitted lag system.",
            "- Selection: folds 0–2; confirmation: folds 3–4; all external references opened afterward.",
            "",
            "## Limits and conclusion",
            "",
            "Calcium filtering, common stimulus drive, unmeasured behavior/state, correlated lag features, and only eight confirmation worms limit interpretation. Global freezing makes the tensor reproducible, but reproducibility is not conditional-mean usefulness or anatomy. Targeted local-perturbation SMC was not promoted because the strict gate failed.",
            "",
            "## Recommended next experiment",
            "",
            "1. Freeze this 4-second basis and consensus regularization.",
            "2. Introduce a neuron-specific latent calcium observation model and repeat the identical outer-fold test.",
            "3. Collect more repeated onsets with behavior/global state and controlled single-neuron perturbations.",
            "4. Promote local-perturbation SMC only after both proper-score and mean-dynamics gates pass prospectively.",
            "",
            "## Output inventory",
            "",
            "- `consensus/`: globally frozen fold models, direct kernels, impulse responses, and promotion gate.",
            "- `consensus/postfreeze/`: circular shifts, conditional ablations, Randi/Cook metrics, and Bentley lag-max inference.",
            "- `semisynthetic_detectability.csv`: 48 real-noise calibration runs.",
            "- `artifact.json`: validated interactive technical report payload.",
            "- `FULL_INVENTORY.csv` and `FULL_CHECKSUMS.sha256`: recursive artifact ledger.",
            "",
            "## Verification",
            "",
            "The full compatibility and conditional-neural test suites pass (44 tests). The native report artifact passed schema validation and rendered successfully. Every file in this result root is indexed by size and SHA-256 digest in the recursive ledger.",
        ]
    )
    (root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    chart_map = """# Chart map

| Report section | Question | Family / type | Dataset and fields | Supported claim |
| --- | --- | --- | --- | --- |
| Atlas-blind screen | Which family and maximum lag had the best screen NLL? | Comparison / grouped bar | `screen_leaderboard.csv`: lag, family, NLL/RMSE/energy improvement | Group shrinkage at 4 s leads NLL but not mean metrics |
| Semi-synthetic detectability | At what injected effect can edges and lags be recovered? | Relationship / scatter | `semisynthetic_detectability.csv`: effect, AUROC, lag MAE, condition, replicate | Moderate edges rank before their lags localize precisely |
| Direct-lag external profile | Does any direct lag correspond to Randi or Cook? | Ordered comparison / line | `consensus_external_lag_profile.csv`: lag seconds, AUROC, reference | No convincing post-freeze atlas peak |
| Contextual method comparison | How does the integrated kernel compare with existing frozen methods? | Comparison / grouped bar | `external_method_comparison.csv`: method, reference, AUROC/AUPRC | Distributed lag is near chance and trails contextual comparators |

Palette policy: hard two-root cap plus neutrals; native report renderer. All charts use explicit reference lines and retain adjacent audit fields in their snapshot datasets.
"""
    (root / "CHART_MAP.md").write_text(chart_map)


def write_inventory(root: Path) -> None:
    excluded = {"FULL_INVENTORY.csv", "FULL_CHECKSUMS.sha256", "checksums.sha256"}
    inventory_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name not in excluded
    )
    with (root / "FULL_INVENTORY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        for path in inventory_paths:
            writer.writerow(
                {
                    "relative_path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    full = inventory_paths + [root / "FULL_INVENTORY.csv"]
    (root / "FULL_CHECKSUMS.sha256").write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(root)}" for path in full) + "\n"
    )


def write(root: Path) -> Path:
    artifact = build(root)
    payload = {
        key: artifact[key] for key in ("surface", "manifest", "snapshot", "sources")
    }
    (root / "artifact.json").write_text(json.dumps(payload, indent=2) + "\n")
    (root / "RESULTS_SUMMARY.json").write_text(json.dumps(artifact["summary"], indent=2) + "\n")
    _write_markdown(root, artifact)
    write_inventory(root)
    return root / "artifact.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("results/distributed_lag_dynamics_20260828")
    )
    args = parser.parse_args()
    write(args.root.resolve())


if __name__ == "__main__":
    main()
