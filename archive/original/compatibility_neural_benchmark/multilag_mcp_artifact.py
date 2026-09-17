from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TITLE = "Multi-lag conditional dynamics and temporal-cut SMC"


def _records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")


def build(root: Path) -> dict:
    generated = datetime.now(timezone.utc).isoformat()
    ceiling_gate = json.loads((root / "ceiling" / "promotion_gate.json").read_text())
    screen_selection = json.loads(
        (root / "lagged_smc_screen_analysis" / "neural_selection.json").read_text()
    )
    confirm_selection = json.loads(
        (root / "lagged_smc_confirmation_analysis" / "neural_selection.json").read_text()
    )
    ceiling = pd.read_csv(root / "ceiling" / "leaderboard.csv")
    ceiling = ceiling[
        (ceiling.stage == "lag_screen")
        & (ceiling.evaluation_split == "screen")
        & ceiling.model.isin(["persistence", "self_ridge", "full_ridge"])
    ].copy()
    ceiling["lag_seconds"] = ceiling.lag / 4.0
    neural = pd.read_csv(
        root / "lagged_smc_screen_analysis" / "neural_prediction_summary.csv"
    )
    neural = neural[(neural.split == "screen") & (neural.phase == "onset")].copy()
    confirm = pd.read_csv(
        root / "lagged_smc_confirmation_analysis" / "neural_prediction_summary.csv"
    )
    selected_confirm = confirm[
        (confirm.split == "confirmation")
        & (confirm.phase == "onset")
        & (confirm.source_lag_frames == screen_selection["source_lag_frames"])
        & (confirm.horizon_frames == screen_selection["horizon_frames"])
    ].iloc[0]
    diagnostics = pd.read_csv(
        root / "lagged_smc_confirmation_analysis" / "smc_diagnostics.csv"
    )
    selected_diag = diagnostics[
        (diagnostics.split == "confirmation")
        & (diagnostics.phase == "onset")
        & (diagnostics.source_lag_frames == screen_selection["source_lag_frames"])
    ].iloc[0]
    external = pd.read_csv(
        root / "postfreeze_external" / "selected_matrix_external_metrics.csv"
    )
    external = external[
        external.reference.isin(["randi_wild_type", "cook_struct_54"])
    ].copy()
    external["reference_label"] = external.reference.map(
        {"randi_wild_type": "Randi WT", "cook_struct_54": "Cook structural"}
    )
    external["method_label"] = external.method.map(
        {
            "temporal_cut_smc_onset": "Temporal-cut SMC onset",
            "temporal_cut_smc_onset_minus_baseline": "Temporal-cut SMC onset−baseline",
            "winner_wide_direct": "Wide-flow direct",
            "sbtg_current": "SBTG-current",
            "sbtg_published": "SBTG-published",
        }
    )
    lagmax = pd.read_csv(root / "postfreeze_external" / "bentley_lagmax_inference.csv")
    headline = pd.DataFrame(
        [
            {
                "conditional_energy_improvement": ceiling_gate[
                    "mean_energy_improvement_vs_self"
                ],
                "conditional_gate_passed": ceiling_gate["passes_promotion_gate"],
                "selected_source_lag_seconds": screen_selection["source_lag_seconds"],
                "selected_horizon_seconds": screen_selection["horizon_seconds"],
                "confirmation_incremental_gain": selected_confirm.mean_incremental_gain,
                "confirmation_gain_ci_low": selected_confirm.incremental_gain_ci_low,
                "confirmation_gain_ci_high": selected_confirm.incremental_gain_ci_high,
                "temporal_gate_passed": confirm_selection["passes_strict_promotion_gate"],
                "confirmation_valid_fraction": selected_diag.valid_fraction,
                "confirmation_median_ess": selected_diag.median_min_ess,
            }
        ]
    )
    sources = [
        {
            "id": "ceiling_source",
            "label": "Atlas-blind conditional-density ceiling",
            "path": "results/multilag_temporal_cut_20260827/ceiling/leaderboard.csv",
            "query": {
                "engine": "DuckDB",
                "sql": (
                    "SELECT *, lag / 4.0 AS lag_seconds "
                    "FROM read_csv_auto('results/multilag_temporal_cut_20260827/ceiling/leaderboard.csv') "
                    "WHERE stage = 'lag_screen' AND evaluation_split = 'screen' "
                    "AND model IN ('persistence', 'self_ridge', 'full_ridge')"
                ),
                "tables_used": [
                    "results/multilag_temporal_cut_20260827/ceiling/leaderboard.csv"
                ],
                "executed_at": generated,
                "language": "sql",
                "description": "Whole-worm nested-CV summaries for history and model candidates.",
                "filters": ["screen folds 0–2", "complete-case 54-neuron cohort"],
                "metric_definitions": [
                    "Energy is the held-out multivariate energy score; lower is better.",
                    "Improvement versus self is self-history score minus off-diagonal-model score.",
                ],
            },
        },
        {
            "id": "neural_source",
            "label": "Neural-only temporal-cut evaluation",
            "path": "results/multilag_temporal_cut_20260827/lagged_smc_screen_analysis/neural_prediction_summary.csv",
            "query": {
                "engine": "DuckDB",
                "sql": (
                    "SELECT * FROM read_csv_auto('results/multilag_temporal_cut_20260827/"
                    "lagged_smc_screen_analysis/neural_prediction_summary.csv') "
                    "WHERE split = 'screen' AND phase = 'onset'"
                ),
                "tables_used": [
                    "results/multilag_temporal_cut_20260827/lagged_smc_screen_analysis/neural_prediction_summary.csv"
                ],
                "executed_at": generated,
                "language": "sql",
                "description": "Worm-clustered cross-fitted prediction gains across temporal cells.",
                "filters": [
                    "screen folds 0–2 for cell selection",
                    "AWC excluded as a declared direct-sensory target",
                    "invalid SMC source columns zeroed",
                ],
                "metric_definitions": [
                    "Incremental gain is full forecast Spearman minus current-state plus same-neuron-history forecast Spearman.",
                    "Worm residual subtracts the other worms' event-specific mean.",
                ],
            },
        },
        {
            "id": "external_source",
            "label": "Post-freeze Randi, Cook, Bentley, and SBTG comparison",
            "path": "results/multilag_temporal_cut_20260827/postfreeze_external/selected_matrix_external_metrics.csv",
            "query": {
                "engine": "DuckDB",
                "sql": (
                    "SELECT * FROM read_csv_auto('results/multilag_temporal_cut_20260827/"
                    "postfreeze_external/selected_matrix_external_metrics.csv') "
                    "WHERE reference IN ('randi_wild_type', 'cook_struct_54')"
                ),
                "tables_used": [
                    "results/multilag_temporal_cut_20260827/postfreeze_external/selected_matrix_external_metrics.csv"
                ],
                "executed_at": generated,
                "language": "sql",
                "description": "External-reference metrics loaded only after the neural temporal cell was frozen.",
                "filters": ["54 shared neurons", "off-diagonal pairs"],
                "metric_definitions": [
                    "AUROC ranks absolute matrix magnitude for binary reference-edge presence.",
                    "Cook count Spearman compares absolute matrix magnitude with continuous connection counts.",
                ],
            },
        },
        {
            "id": "lagmax_source",
            "label": "Bentley lag-search inference",
            "path": "results/multilag_temporal_cut_20260827/postfreeze_external/bentley_lagmax_inference.csv",
            "query": {
                "engine": "DuckDB",
                "sql": (
                    "SELECT * FROM read_csv_auto('results/multilag_temporal_cut_20260827/"
                    "postfreeze_external/bentley_lagmax_inference.csv')"
                ),
                "tables_used": [
                    "results/multilag_temporal_cut_20260827/postfreeze_external/bentley_lagmax_inference.csv"
                ],
                "executed_at": generated,
                "language": "sql",
                "description": "Within-source target-label null with maximum over all source-lag and horizon cells.",
                "filters": ["eligible molecular source columns", "four panel × network tests"],
                "metric_definitions": [
                    "Lag-max p-value compares the observed best AUROC with the maximum null AUROC over the full temporal grid.",
                    "BH q-value corrects the four prespecified panel-by-network searches.",
                ],
            },
        },
    ]
    cards = [
        {
            "id": "conditional_gate",
            "dataset": "headline",
            "sourceId": "ceiling_source",
            "description": "Confirmation energy improvement of the off-diagonal model over self history; positive favors the off-diagonal model.",
            "metrics": [
                {
                    "label": "Energy improvement vs self",
                    "field": "conditional_energy_improvement",
                    "format": "number",
                    "signed": True,
                }
            ],
        },
        {
            "id": "temporal_gain",
            "dataset": "headline",
            "sourceId": "neural_source",
            "description": "Untouched-fold worm-residual Spearman gain from adding the frozen temporal-cut matrix.",
            "metrics": [
                {
                    "label": "Confirmation Spearman gain",
                    "field": "confirmation_incremental_gain",
                    "format": "number",
                    "signed": True,
                }
            ],
        },
        {
            "id": "smc_validity",
            "dataset": "headline",
            "sourceId": "neural_source",
            "description": "Fraction of source clamps meeting ESS, maximum-weight, and achieved-gap diagnostics on confirmation worms.",
            "metrics": [
                {
                    "label": "Compatibility-valid sources",
                    "field": "confirmation_valid_fraction",
                    "format": "percent",
                },
                {
                    "label": "Median clamp ESS",
                    "field": "confirmation_median_ess",
                    "format": "number",
                },
            ],
        },
    ]
    charts = [
        {
            "id": "history_energy",
            "title": "Conditional-density energy across history windows",
            "subtitle": "Self history remains stronger than the best off-diagonal candidate on proper held-out scoring.",
            "type": "line",
            "intent": "comparison",
            "dataset": "ceiling_screen",
            "sourceId": "ceiling_source",
            "encodings": {
                "x": {"field": "lag_seconds", "type": "quantitative", "label": "History (s)"},
                "y": {"field": "energy_mean", "type": "quantitative", "label": "Energy"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "tooltip": [
                    {"field": "energy_se", "type": "quantitative", "label": "Fold SE"},
                    {"field": "rmse_mean", "type": "quantitative", "label": "RMSE"},
                    {"field": "innovation_spearman_mean", "type": "quantitative", "label": "Innovation rho"},
                ],
            },
            "yAxisTitle": "Energy score (lower is better)",
            "layout": "full",
        },
        {
            "id": "neural_grid",
            "title": "Incremental onset prediction across temporal cells",
            "subtitle": "Cell selection uses neural prediction only; external atlases remain sealed at this stage.",
            "type": "heatmap",
            "intent": "comparison",
            "dataset": "neural_grid",
            "sourceId": "neural_source",
            "encodings": {
                "x": {"field": "horizon_seconds", "type": "quantitative", "label": "Horizon (s)"},
                "y": {"field": "source_lag_seconds", "type": "quantitative", "label": "Source lag (s)"},
                "color": {"field": "mean_incremental_gain", "type": "quantitative", "label": "Spearman gain"},
                "tooltip": [
                    {"field": "incremental_gain_ci_low", "type": "quantitative", "label": "CI low"},
                    {"field": "incremental_gain_ci_high", "type": "quantitative", "label": "CI high"},
                    {"field": "mean_matrix_only_spearman", "type": "quantitative", "label": "Matrix-only rho"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "external_comparison",
            "title": "Selected-matrix external-reference AUROC",
            "subtitle": "Randi and Cook are post-freeze correspondence checks, not selection targets or causal validation.",
            "type": "bar",
            "intent": "comparison",
            "dataset": "external_selected",
            "sourceId": "external_source",
            "encodings": {
                "x": {"field": "reference_label", "type": "nominal", "label": "Reference"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC"},
                "color": {"field": "method_label", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "auprc", "type": "quantitative", "label": "AUPRC"},
                    {"field": "macro_source_auroc", "type": "quantitative", "label": "Source-macro AUROC"},
                    {"field": "count_spearman_all_pairs", "type": "quantitative", "label": "Cook count rho"},
                ],
            },
            "referenceLines": [{"axis": "y", "value": 0.5, "label": "Chance"}],
            "layout": "full",
        },
    ]
    tables = [
        {
            "id": "lagmax_table",
            "title": "Bentley temporal-grid inference",
            "subtitle": "Each p-value controls the full cell search within its panel; q-values control four panels.",
            "dataset": "lagmax",
            "sourceId": "lagmax_source",
            "defaultSort": {"field": "lagmax_p_value", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "panel", "label": "Panel", "type": "text"},
                {"field": "network", "label": "Network", "type": "text"},
                {"field": "best_source_lag_seconds", "label": "Source lag (s)", "format": "number"},
                {"field": "best_horizon_seconds", "label": "Horizon (s)", "format": "number"},
                {"field": "best_auroc", "label": "Best AUROC", "format": "number"},
                {"field": "best_cell_selection_rate", "label": "Bootstrap selection", "format": "percent"},
                {"field": "lagmax_p_value", "label": "Lag-max p", "format": "number"},
                {"field": "lagmax_bh_q_value", "label": "BH q", "format": "number"},
            ],
        }
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "summary",
            "type": "markdown",
            "body": (
                "The wider model search produced an honest negative: weak off-diagonal innovation ranking did not improve the held-out conditional distribution over self history. "
                "The temporal-cut SMC extension was therefore evaluated with a stricter question—whether it adds worm-specific onset prediction beyond current state and same-neuron history—before any atlas was opened."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": ["conditional_gate", "temporal_gain", "smc_validity"],
        },
        {
            "id": "ceiling_heading",
            "type": "markdown",
            "body": "## Conditional-density ceiling\n\nPositive innovation correlation alone did not pass the proper-score and RMSE gate on untouched folds.",
            "sourceId": "ceiling_source",
        },
        {"id": "history_chart", "type": "chart", "chartId": "history_energy"},
        {
            "id": "neural_heading",
            "type": "markdown",
            "body": "## Neural-only temporal-cell selection\n\nThe primary metric is worm-residual incremental Spearman after excluding AWC as a declared direct sensory target. Invalid SMC source columns are zeroed.",
            "sourceId": "neural_source",
        },
        {"id": "neural_chart", "type": "chart", "chartId": "neural_grid"},
        {
            "id": "external_heading",
            "type": "markdown",
            "body": "## Post-freeze external correspondence\n\nThe selected temporal cell is compared with wide-flow direct, SBTG-current, and SBTG-published on the identical 54-neuron order. Cook count correlations remain separate from binary edge-presence AUROC.",
            "sourceId": "external_source",
        },
        {"id": "external_chart", "type": "chart", "chartId": "external_comparison"},
        {
            "id": "lagmax_heading",
            "type": "markdown",
            "body": "## Neuromodulator lag-search controls\n\nSource lag and forecast horizon are distinct axes. The null permutes molecular targets within each source and uses the maximum AUROC over the complete grid; no cell is interpreted as a physical delay.",
            "sourceId": "lagmax_source",
        },
        {"id": "lagmax_results", "type": "table", "tableId": "lagmax_table"},
        {
            "id": "limits",
            "type": "markdown",
            "body": "## Limits and recommendation\n\nPassive calcium, common stimulus drive, hidden state, behavior, and finite-particle compatibility prevent anatomical or causal interpretation. The next useful test is a prospectively held-out acquisition with more repeated onsets and explicit perturbations, using this run's temporal cell and scoring rule unchanged.",
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Atlas-blind multi-lag conditional-density and temporal-cut SMC evaluation with post-freeze external checks.",
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
            "ceiling_screen": _records(ceiling),
            "neural_grid": _records(neural),
            "external_selected": _records(external),
            "lagmax": _records(lagmax),
        },
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact = build(args.root.resolve())
    text = json.dumps(artifact, indent=2) + "\n"
    if args.output:
        args.output.resolve().write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
