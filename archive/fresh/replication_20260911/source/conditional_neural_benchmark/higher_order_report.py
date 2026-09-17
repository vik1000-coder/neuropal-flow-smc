from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TITLE = "Higher-order conditional lag dynamics across 54- and 80-neuron cohorts"
COHORT_LABEL = {
    "current54": "Current 54",
    "sbtg80": "Original SBTG 80",
    "sbtg_bridge54": "SBTG bridge 54",
}
METHOD_LABEL = {
    "location": "Structured mean",
    "logvariance": "Structured log variance",
    "covariance_gain": "Structured covariance gain",
    "conditional_scale_ablation": "Conditional scale ablation",
    "importance_weighting_full": "Flow importance",
    "wide_flow_direct": "Wide-flow direct",
    "terminal_smc_full": "Terminal SMC",
    "progressive_smc_full": "Progressive ESS-SMC",
    "sbtg_current": "SBTG-current",
    "sbtg_published": "SBTG-published",
    "neural_mean": "Neural mean effect",
    "neural_logvariance": "Neural log-variance effect",
    "neural_covariance_row_energy": "Neural covariance effect",
}
REFERENCE_LABEL = {
    "randi_wild_type": "Randi WT",
    "cook_struct_54": "Cook structural",
    "cook_chem_54": "Cook chemical",
    "cook_gap_54": "Cook gap",
}


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [_safe(value) for value in frame.to_dict("records")]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bh(values: pd.Series) -> np.ndarray:
    raw = values.to_numpy(float)
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = np.minimum.accumulate((ranked * len(raw) / np.arange(1, len(raw) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _source(
    source_id: str,
    label: str,
    path: str,
    generated: str,
    description: str,
    filters: list[str],
    definitions: list[str],
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{path}')",
            "tables_used": [path],
            "executed_at": generated,
            "description": description,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def build(
    root: Path,
    structured: Path,
    neural: Path,
    postfreeze: Path,
    effects_postfreeze: Path,
    ablation_root: Path,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    folds = pd.read_csv(structured / "final_fold_metrics.csv")
    confirmation = folds[folds.fold.isin([3, 4])].copy()
    metric_columns = [
        "mean_nll_improvement", "mean_rmse_improvement",
        "scale_gaussian_nll_improvement", "scale_student_nll_improvement",
        "covariance_energy_improvement", "covariance_variogram_improvement",
    ]
    structured_summary = confirmation.groupby("cohort", as_index=False)[metric_columns].mean()
    structured_summary["cohort_label"] = structured_summary.cohort.map(COHORT_LABEL)
    gates = json.loads((structured / "promotion_gates.json").read_text())
    structured_summary["scale_ci_low"] = structured_summary.cohort.map(
        lambda value: gates[value]["ci_low"]
    )
    structured_summary["scale_ci_high"] = structured_summary.cohort.map(
        lambda value: gates[value]["ci_high"]
    )
    structured_summary["passes_scale_gate"] = structured_summary.cohort.map(
        lambda value: gates[value]["passes_scale_gate"]
    )
    structured_summary.to_csv(root / "structured_confirmation_summary.csv", index=False)

    neural_rows: list[pd.DataFrame] = []
    neural_validation: dict[str, dict] = {}
    for cohort in COHORT_LABEL:
        frame = pd.read_csv(neural / cohort / "confirmation_leaderboard.csv")
        frame["cohort"] = cohort
        frame["cohort_label"] = COHORT_LABEL[cohort]
        frame["model_label"] = frame.model_id.str.replace("tcn_", "", regex=False)
        neural_rows.append(frame)
        neural_validation[cohort] = json.loads((neural / cohort / "validation.json").read_text())
    neural_summary = pd.concat(neural_rows, ignore_index=True)
    neural_summary.to_csv(root / "neural_confirmation_summary.csv", index=False)

    control = json.loads((postfreeze / "neural_control_summary.json").read_text())
    stability = pd.read_csv(postfreeze / "kernel_stability.csv")
    stability_summary = stability.groupby(["cohort", "functional"], as_index=False).agg(
        kernel_pearson=("kernel_pearson", "mean"),
        edge_rank_spearman=("edge_rank_spearman", "mean"),
        sign_agreement=("sign_agreement", "mean"),
        fold_pairs=("kernel_pearson", "size"),
    )
    stability_summary["cohort_label"] = stability_summary.cohort.map(COHORT_LABEL)
    stability_summary.to_csv(root / "kernel_stability_summary.csv", index=False)

    weak = pd.read_csv(postfreeze / "variance_semisynthetic.csv")
    strong = pd.read_csv(postfreeze / "variance_semisynthetic_strong_effects.csv")
    synthetic = pd.concat([weak, strong], ignore_index=True)
    synthetic_summary = synthetic.groupby(
        ["noise_cohort", "condition", "effect"], as_index=False
    ).agg(
        edge_auroc=("edge_auroc", "mean"),
        edge_auroc_sd=("edge_auroc", "std"),
        lag_mae_frames=("lag_mae_frames", "mean"),
        lag_within_one_frame=("lag_within_one_frame", "mean"),
        replicates=("replicate", "size"),
    )
    synthetic_summary["series_label"] = (
        synthetic_summary.noise_cohort.map(COHORT_LABEL)
        + " / "
        + synthetic_summary.condition.str.replace("_", " ")
    )
    synthetic_summary.to_csv(root / "variance_detectability_summary.csv", index=False)

    structured_external = pd.read_csv(postfreeze / "postfreeze_randi_cook_metrics.csv")
    neural_external = pd.read_csv(effects_postfreeze / "neural_effect_randi_cook_metrics.csv")
    external = pd.concat([structured_external, neural_external], ignore_index=True)
    external["method_label"] = external.method.map(METHOD_LABEL).fillna(external.method)
    external["reference_label"] = external.reference.map(REFERENCE_LABEL).fillna(external.reference)
    external.to_csv(root / "all_postfreeze_external_metrics.csv", index=False)
    external_chart = external[
        (external.cohort == "current54")
        & external.reference.isin(["randi_wild_type", "cook_struct_54"])
        & external.method.isin([
            "covariance_gain", "neural_covariance_row_energy", "wide_flow_direct",
            "sbtg_current", "sbtg_published",
        ])
    ].copy()
    external_chart.to_csv(root / "external_chart_data.csv", index=False)

    structured_receptor = pd.read_csv(postfreeze / "postfreeze_receptor_matched_enrichment.csv")
    neural_receptor = pd.read_csv(
        effects_postfreeze / "neural_effect_receptor_matched_enrichment.csv"
    )
    receptor = pd.concat([structured_receptor, neural_receptor], ignore_index=True)
    receptor["bh_q_global"] = _bh(receptor.permutation_p)
    receptor["method_label"] = receptor.method.map(METHOD_LABEL).fillna(receptor.method)
    receptor["cohort_label"] = receptor.cohort.map(COHORT_LABEL)
    receptor.to_csv(root / "all_receptor_matched_enrichment.csv", index=False)
    receptor_table = receptor.sort_values(["bh_q_global", "permutation_p"]).head(20).copy()

    ablation = pd.read_csv(ablation_root / "ablation_summary.csv")
    ablation["cohort_label"] = ablation.cohort.map(COHORT_LABEL)
    ablation.to_csv(root / "neural_history_ablation_summary.csv", index=False)

    current_board = neural_summary[neural_summary.cohort == "current54"].sort_values("energy")
    current_winner = current_board.iloc[0]
    current_structured = structured_summary[structured_summary.cohort == "current54"].iloc[0]
    current_ablation = ablation[ablation.cohort == "current54"].iloc[0]
    current_detect = synthetic_summary[
        (synthetic_summary.noise_cohort == "current54")
        & (synthetic_summary.condition == "calcium_smoothed")
        & (synthetic_summary.edge_auroc >= 0.70)
    ]
    calcium_threshold = (
        float(current_detect.effect.min()) if not current_detect.empty else None
    )
    best_new_randi = external[
        (external.cohort == "current54")
        & (external.reference == "randi_wild_type")
        & external.method.isin([
            "location", "logvariance", "covariance_gain", "conditional_scale_ablation",
            "neural_mean", "neural_logvariance", "neural_covariance_row_energy",
        ])
    ].sort_values("auroc", ascending=False).iloc[0]
    best_new_cook = external[
        (external.cohort == "current54")
        & (external.reference == "cook_struct_54")
        & external.method.isin([
            "location", "logvariance", "covariance_gain", "conditional_scale_ablation",
            "neural_mean", "neural_logvariance", "neural_covariance_row_energy",
        ])
    ].sort_values("auroc", ascending=False).iloc[0]

    headline = pd.DataFrame([{
        "current_mean_nll_improvement": current_structured.mean_nll_improvement,
        "current_mean_rmse_improvement": current_structured.mean_rmse_improvement,
        "current_scale_nll_improvement": current_structured.scale_gaussian_nll_improvement,
        "current_neural_winner_energy": current_winner.energy,
        "current_neural_ablation_nll_increase": current_ablation.mean_nll_increase,
        "calcium_variance_auroc70_threshold": calcium_threshold,
        "best_new_randi_auroc": best_new_randi.auroc,
        "best_new_cook_auroc": best_new_cook.auroc,
    }])
    headline.to_csv(root / "headline_metrics.csv", index=False)

    base = "results/higher_order_dynamics_20260828"
    sources = [
        _source(
            "structured_source", "Structured confirmation",
            f"{base}/structured_confirmation_summary.csv", generated,
            "Frozen smooth mean, log-variance and covariance lag models on folds 3–4.",
            ["whole-worm confirmation folds 3–4", "three prespecified cohorts"],
            [
                "Every improvement is baseline score minus full score; positive favors cross-neuron lag history.",
                "The scale gate requires the worm-bootstrap Gaussian-NLL interval to lie above zero.",
            ],
        ),
        _source(
            "neural_source", "Neural density confirmation",
            f"{base}/neural_confirmation_summary.csv", generated,
            "Confirmation energy, variogram, calibration, RMSE and exact NLL where available.",
            ["atlas-blind models", "folds 3–4", "seed 1701"],
            ["Energy and variogram are proper multivariate scores where lower is better."],
        ),
        _source(
            "ablation_source", "Neural source-history ablation",
            f"{base}/neural_history_ablation_summary.csv", generated,
            "Within-worm circular source-history ablation on held-out parametric neural checkpoints.",
            ["folds 3–4", "one complete source history circularly shifted at a time"],
            ["Positive NLL increase means factual source history improves held-out density."],
        ),
        _source(
            "synthetic_source", "Variance-effect detectability",
            f"{base}/variance_detectability_summary.csv", generated,
            "Known variance-edge and lag recovery under real-noise latent and calcium-smoothed simulations.",
            ["six replicates per cell", "effect 0.05–2.0", "20 injected directed edges"],
            [
                "Edge AUROC ranks injected versus absent edges.",
                "Lag MAE is evaluated only on injected edges in frames at 4 Hz.",
            ],
        ),
        _source(
            "external_source", "Post-freeze Randi and Cook correspondence",
            f"{base}/all_postfreeze_external_metrics.csv", generated,
            "Binary and continuous/count correspondence loaded only after neural freeze.",
            ["current54, SBTG80 and bridge54", "ordered off-diagonal eligible pairs"],
            [
                "AUROC/AUPRC use binary edge presence.",
                "Randi continuous metrics use signed and absolute Spearman; Cook count metrics use log1p edge counts.",
            ],
        ),
        _source(
            "receptor_source", "Covariate-matched receptor enrichment",
            f"{base}/all_receptor_matched_enrichment.csv", generated,
            "Within-source target matching on activity variance, derivative scale and amplitude.",
            ["999 target-label permutations per cell", "global BH correction across displayed experiment family"],
            ["Matched enrichment is mean connected-minus-matched-unconnected magnitude within eligible sources."],
        ),
    ]

    cards = [
        {
            "id": "mean_card", "dataset": "headline", "sourceId": "structured_source",
            "description": "Incremental current54 conditional-mean performance over the self-history, stimulus and population-state baseline.",
            "metrics": [
                {"label": "Mean NLL improvement", "field": "current_mean_nll_improvement", "format": "number", "signed": True},
                {"label": "RMSE improvement", "field": "current_mean_rmse_improvement", "format": "number", "signed": True},
            ],
        },
        {
            "id": "scale_card", "dataset": "headline", "sourceId": "structured_source",
            "description": "Primary normalized Gaussian scale-score improvement on current54 confirmation worms.",
            "metrics": [{"label": "Scale NLL improvement", "field": "current_scale_nll_improvement", "format": "number", "signed": True}],
        },
        {
            "id": "neural_card", "dataset": "headline", "sourceId": "neural_source",
            "description": "Best atlas-blind current54 confirmation energy among the tested neural density architectures.",
            "metrics": [{"label": "Best neural energy", "field": "current_neural_winner_energy", "format": "number"}],
        },
        {
            "id": "ablation_card", "dataset": "headline", "sourceId": "ablation_source",
            "description": "Mean held-out NLL change after circularly shifting one source neuron's complete lag history.",
            "metrics": [{"label": "History-ablation NLL increase", "field": "current_neural_ablation_nll_increase", "format": "number", "signed": True}],
        },
        {
            "id": "detect_card", "dataset": "headline", "sourceId": "synthetic_source",
            "description": "Smallest injected current54 calcium-smoothed log-variance coefficient with mean edge AUROC at least 0.70.",
            "metrics": [{"label": "Variance detectability threshold", "field": "calcium_variance_auroc70_threshold", "format": "number"}],
        },
    ]

    charts = [
        {
            "id": "neural_chart", "title": "Neural density confirmation energy",
            "subtitle": "Folds 3–4; lower is better. Scores are comparable within cohort, not across differently scaled cohorts.",
            "type": "bar", "intent": "comparison", "dataset": "neural",
            "sourceId": "neural_source",
            "encodings": {
                "x": {"field": "cohort_label", "type": "nominal", "label": "Cohort"},
                "y": {"field": "energy", "type": "quantitative", "label": "Energy score"},
                "color": {"field": "model_label", "type": "nominal", "label": "Model"},
                "tooltip": [
                    {"field": "balanced_energy", "type": "quantitative", "label": "Stimulus-balanced energy"},
                    {"field": "variogram", "type": "quantitative", "label": "Variogram"},
                    {"field": "coverage90", "type": "quantitative", "label": "90% coverage"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "detectability_chart", "title": "Semi-synthetic variance-edge detectability",
            "subtitle": "Six replicates per cell; calcium smoothing shifts useful recovery toward much larger effects.",
            "type": "scatter", "intent": "relationship", "dataset": "synthetic",
            "sourceId": "synthetic_source",
            "encodings": {
                "x": {"field": "effect", "type": "quantitative", "label": "Injected log-variance coefficient"},
                "y": {"field": "edge_auroc", "type": "quantitative", "label": "Edge AUROC"},
                "color": {"field": "series_label", "type": "nominal", "label": "Cohort / observation"},
                "tooltip": [
                    {"field": "lag_mae_frames", "type": "quantitative", "label": "Lag MAE (frames)"},
                    {"field": "lag_within_one_frame", "type": "quantitative", "label": "Within one frame"},
                    {"field": "edge_auroc_sd", "type": "quantitative", "label": "AUROC SD"},
                ],
            },
            "referenceLines": [{"axis": "y", "value": 0.5, "label": "Chance"}],
            "layout": "full",
        },
        {
            "id": "external_chart", "title": "Post-freeze Randi and Cook AUROC",
            "subtitle": "Current 54-neuron panel; binary presence is descriptive and continuous/count correlations are retained in the source table.",
            "type": "bar", "intent": "comparison", "dataset": "external_chart",
            "sourceId": "external_source",
            "encodings": {
                "x": {"field": "reference_label", "type": "nominal", "label": "Reference"},
                "y": {"field": "auroc", "type": "quantitative", "label": "AUROC"},
                "color": {"field": "method_label", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "auprc", "type": "quantitative", "label": "AUPRC"},
                    {"field": "continuous_abs_spearman", "type": "quantitative", "label": "Randi absolute rho"},
                    {"field": "count_spearman_all", "type": "quantitative", "label": "Cook count rho"},
                ],
            },
            "referenceLines": [{"axis": "y", "value": 0.5, "label": "Chance"}],
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "structured_table", "title": "Structured confirmation by estimand",
            "subtitle": "Positive improvement favors lagged cross-neuron history; all scale gates fail.",
            "dataset": "structured", "sourceId": "structured_source",
            "defaultSort": {"field": "mean_nll_improvement", "direction": "desc"},
            "density": "dense", "layout": "full",
            "columns": [
                {"field": "cohort_label", "label": "Cohort", "type": "text"},
                {"field": "mean_nll_improvement", "label": "Mean NLL imp.", "format": "number", "movement": True},
                {"field": "mean_rmse_improvement", "label": "Mean RMSE imp.", "format": "number", "movement": True},
                {"field": "scale_gaussian_nll_improvement", "label": "Scale NLL imp.", "format": "number", "movement": True},
                {"field": "covariance_energy_improvement", "label": "Cov. energy imp.", "format": "number", "movement": True},
                {"field": "covariance_variogram_improvement", "label": "Cov. variogram imp.", "format": "number", "movement": True},
                {"field": "passes_scale_gate", "label": "Scale gate", "type": "boolean"},
            ],
        },
        {
            "id": "receptor_table", "title": "Smallest covariate-matched receptor permutation results",
            "subtitle": "Top 20 of the complete family; global BH q controls the full structured and neural set.",
            "dataset": "receptor", "sourceId": "receptor_source",
            "defaultSort": {"field": "bh_q_global", "direction": "asc"},
            "density": "dense", "layout": "full",
            "columns": [
                {"field": "cohort_label", "label": "Cohort", "type": "text"},
                {"field": "method_label", "label": "Method", "type": "text"},
                {"field": "network", "label": "Network", "type": "text"},
                {"field": "matched_enrichment", "label": "Matched enrichment", "format": "number"},
                {"field": "eligible_sources", "label": "Sources", "format": "number"},
                {"field": "permutation_p", "label": "Permutation p", "format": "number"},
                {"field": "bh_q_global", "label": "Global BH q", "format": "number"},
            ],
        },
    ]

    summary_text = (
        f"Across all three cohorts, structured cross-neuron histories improve Student-t mean log score only slightly "
        f"but worsen RMSE, normalized Gaussian scale NLL, or covariance variogram score. All scale gates fail. "
        f"The best current54 neural density is {current_winner.model_id} (energy {current_winner.energy:.4f}), "
        f"but the incremental source-history ablation is {current_ablation.mean_nll_increase:+.5f} NLL. "
        f"Variance edges become reliably rankable only at large injected effects; the current54 calcium-smoothed "
        f"AUROC-0.70 threshold is {calcium_threshold if calcium_threshold is not None else 'not reached'}. "
        f"The best new current54 Randi/Cook binary AUROCs are {best_new_randi.auroc:.3f}/{best_new_cook.auroc:.3f}, "
        "below the strongest prior atlas-blind or published comparators."
    )
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {"id": "technical_summary", "type": "markdown", "body": f"## Technical summary\n\n{summary_text}"},
        {"id": "headline", "type": "metric-strip", "cardIds": ["mean_card", "scale_card", "neural_card", "ablation_card", "detect_card"]},
        {"id": "structured_heading", "type": "markdown", "body": "## No structured higher-order estimand passes its full confirmation gate\n\nSmall heavy-tail log-score gains do not transfer to conditional-mean RMSE. Marginal variance fails normalized Gaussian NLL in every cohort; covariance energy and variogram disagree. The sign conflict blocks a new temporal-cut/SMC promotion.", "sourceId": "structured_source"},
        {"id": "structured_table_block", "type": "table", "tableId": "structured_table"},
        {"id": "neural_heading", "type": "markdown", "body": "## Flexible neural densities improve absolute forecasts, not automatically inter-neuron identification\n\nMDN, flow, contrastive-energy, score, and low-rank heads are selected solely by held-out predictive-law scores. Source-history circular ablation is reported separately because an absolute density winner can rely mostly on target self-history and stimulus."},
        {"id": "neural_chart_block", "type": "chart", "chartId": "neural_chart"},
        {"id": "detect_heading", "type": "markdown", "body": "## Variance/gain effects require large signals and calcium smoothing degrades lag recovery\n\nThe extended oracle test reaches strong edge ranking only when injected log-variance effects are much larger than the original 0.05–0.30 grid. Even when edge AUROC rises, lag MAE remains substantial after calcium filtering.", "sourceId": "synthetic_source"},
        {"id": "detect_chart_block", "type": "chart", "chartId": "detectability_chart"},
        {"id": "external_heading", "type": "markdown", "body": "## Continuous and count-valued references do not rescue the new matrices\n\nRandi continuous Spearman and Cook count Spearman preserve the broad binary-AUROC conclusion: new mean, variance, and covariance matrices are weak. These references describe anatomy or receptor compatibility rather than activity effects and remain secondary post-freeze context.", "sourceId": "external_source"},
        {"id": "external_chart_block", "type": "chart", "chartId": "external_chart"},
        {"id": "receptor_heading", "type": "markdown", "body": "## Matched receptor enrichment is exploratory and multiplicity-limited\n\nTargets are matched within source on activity variance, derivative scale, and amplitude before permutation. The global BH column covers the complete structured-plus-neural family; nominal cells are not treated as physical-delay or causal evidence.", "sourceId": "receptor_source"},
        {"id": "receptor_table_block", "type": "table", "tableId": "receptor_table"},
        {"id": "scope", "type": "markdown", "body": "## Scope and metric definitions\n\nThe current cohort has 20 worms and 54 neurons; the exact original SBTG cache has 20 worms and 80 neurons; the bridge uses those SBTG traces restricted to the current 54-neuron order. Histories are 8, 16, or 32 frames at 4 Hz and include the complete binary stimulus window. Folds 0–2 select; folds 3–4 confirm. Proper scores assess predictive laws, while lag tensors are local reduced-form functionals."},
        {"id": "methods", "type": "markdown", "body": "## Methodology\n\nStructured models use smooth lag bases, group shrinkage, heteroscedastic residual scale, and low-rank covariance gains. Neural candidates include heteroscedastic and low-rank Gaussians, MDN, denoising-score matching, bounded contrastive energy, Gaussian-source flow, and wide conditional flow. Neural moment tensors use central finite differences along the frozen smooth basis. External references are opened only after freeze records are written."},
        {"id": "limits", "type": "markdown", "body": "## Robustness, limitations, and claim boundary\n\nWhole-worm splits prevent window leakage, but calcium filtering, unmeasured behavior/global state, shared stimulus drive, correlated histories, and only eight confirmation worms limit identification. Receptor networks encode possible modulation, not measured gain. A predictive lag matrix is neither a direct synapse nor an intervention response, and no result supports a physical transmission delay."},
        {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\n1. Treat the current higher-order lag result as a calibrated null/ceiling, not a failed optimizer search.\n2. Fit an explicit latent calcium observation model and repeat the same frozen outer-fold tests.\n3. Add behavior/global-state measurements and repeated single-neuron perturbations.\n4. Use receptor atlases to define stratified hypotheses, not tune lag models.\n5. Reopen temporal-cut SMC only after a prospective model passes both proper-score and incremental-history gates."},
        {"id": "questions", "type": "markdown", "body": "## Further questions\n\n- Does neuron-specific deconvolution reduce the semi-synthetic variance threshold?\n- Are gain effects concentrated in a small prespecified receptor-defined source subset?\n- Can controlled perturbations separate direct modulation from shared latent state?"},
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Atlas-blind higher-order conditional lag models, whole-worm confirmation, semi-synthetic detectability, and post-freeze structural/receptor checks.",
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
            "structured": _records(structured_summary),
            "neural": _records(neural_summary),
            "ablation": _records(ablation),
            "synthetic": _records(synthetic_summary),
            "external_chart": _records(external_chart),
            "receptor": _records(receptor_table),
        },
    }
    summary = {
        "structured_confirmation": _records(structured_summary),
        "neural_validation": neural_validation,
        "neural_current54_winner": _safe(current_winner.to_dict()),
        "neural_history_ablation": _records(ablation),
        "calcium_variance_auroc70_threshold": calcium_threshold,
        "best_new_current54_randi": _safe(best_new_randi.to_dict()),
        "best_new_current54_cook": _safe(best_new_cook.to_dict()),
        "structured_scale_controls": control,
        "minimum_receptor_global_bh_q": float(receptor.bh_q_global.min()),
    }
    return {
        "surface": "report", "manifest": manifest, "snapshot": snapshot,
        "sources": sources, "summary": _safe(summary),
    }


def _write_markdown(root: Path, artifact: dict) -> None:
    summary = artifact["summary"]
    structured = pd.read_csv(root / "structured_confirmation_summary.csv")
    neural = pd.read_csv(root / "neural_confirmation_summary.csv")
    ablation = pd.read_csv(root / "neural_history_ablation_summary.csv")
    external = pd.read_csv(root / "all_postfreeze_external_metrics.csv")
    receptor = pd.read_csv(root / "all_receptor_matched_enrichment.csv")
    detect = pd.read_csv(root / "variance_detectability_summary.csv")
    lines = [
        f"# {TITLE}", "", "**Run date:** 2026-08-28  ",
        "**Claim boundary:** predictive calcium dynamics and post-freeze correspondence; not synapses, interventions, receptor activity, or physical transmission delay.", "",
        "## Technical summary", "",
        "The exhaustive atlas-blind search does **not** produce a confirmed inter-neuron higher-order lag model. Structured cross-neuron histories yield small Student-t log-score gains but worsen conditional-mean RMSE in all three cohorts. Normalized Gaussian scale NLL worsens in all three cohorts, and covariance energy/variogram scores disagree. All prespecified scale gates fail, so no new temporal-cut/ESS-SMC response run is promoted.", "",
        f"On current54, the best absolute neural density is **{summary['neural_current54_winner']['model_id']}** with confirmation energy **{summary['neural_current54_winner']['energy']:.4f}**. Absolute forecasting quality is kept separate from incremental inter-neuron usefulness; the held-out circular source-history ablation is reported below.", "",
        f"Semi-synthetic variance recovery is near chance over effects 0.05–0.30 and becomes useful only at large effects. The current54 calcium-smoothed mean AUROC first exceeds 0.70 at coefficient **{summary['calcium_variance_auroc70_threshold']}**. This identifies a signal ceiling, not merely a failed architecture search.", "",
        "## Structured confirmation", "",
        "| Cohort | Mean NLL imp. | Mean RMSE imp. | Scale Gaussian NLL imp. | Cov. energy imp. | Cov. variogram imp. | Scale gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in structured.itertuples():
        lines.append(
            f"| {row.cohort_label} | {row.mean_nll_improvement:+.6f} | {row.mean_rmse_improvement:+.6f} | "
            f"{row.scale_gaussian_nll_improvement:+.6f} | {row.covariance_energy_improvement:+.6f} | "
            f"{row.covariance_variogram_improvement:+.6f} | {'Pass' if row.passes_scale_gate else 'Fail'} |"
        )
    lines += ["", "Positive improvement favors lagged cross-neuron history. The sign conflict is consistent across cohorts: a small heavy-tail log-score gain does not translate to mean prediction, and scale/covariance proper scores do not agree.", "", "## Neural density tournament", "", "| Cohort | Model | Energy | Balanced energy | Variogram | 90% coverage |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in neural.sort_values(["cohort", "energy"]).itertuples():
        lines.append(f"| {row.cohort_label} | `{row.model_id}` | {row.energy:.4f} | {row.balanced_energy:.4f} | {row.variogram:.4f} | {row.coverage90:.3f} |")
    lines += ["", "Energy differences between the leading MDN/flow/parametric finalists are modest and fold-dependent. These scores compare conditional-law prediction within cohort; they do not validate a lag matrix by themselves.", "", "## Incremental neural source-history ablation", "", "| Cohort | Mean NLL increase | Median NLL increase | Positive source fraction |", "| --- | ---: | ---: | ---: |"]
    for row in ablation.itertuples():
        lines.append(f"| {row.cohort_label} | {row.mean_nll_increase:+.6f} | {row.median_nll_increase:+.6f} | {row.positive_source_fraction:.1%} |")
    lines += ["", "Each ablation circularly shifts one neuron's complete lag-history window within worm while keeping the target and stimulus factual. Positive NLL increase means the aligned source history helps held-out density prediction.", "", "## Semi-synthetic variance/gain detectability", "", "| Noise cohort | Observation | Effect | Edge AUROC | Lag MAE (frames) | Within one frame |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in detect[detect.effect.isin([0.3, 0.75, 1.0, 1.5, 2.0])].itertuples():
        lines.append(f"| {COHORT_LABEL[row.noise_cohort]} | {row.condition.replace('_',' ')} | {row.effect:.2f} | {row.edge_auroc:.3f} | {row.lag_mae_frames:.3f} | {row.lag_within_one_frame:.1%} |")
    lines += ["", "Calcium filtering shifts the recovery curve rightward and preserves substantial lag error even when edge ranking improves. Weak gain effects of the size originally tested are not identifiable with this estimator/data regime.", "", "## Post-freeze Randi and Cook checks", "", "| Method | Randi AUROC | Randi abs. rho | Cook AUROC | Cook count rho |", "| --- | ---: | ---: | ---: | ---: |"]
    current = external[external.cohort == "current54"]
    methods = ["covariance_gain", "neural_covariance_row_energy", "wide_flow_direct", "progressive_smc_full", "sbtg_current", "sbtg_published"]
    for method in methods:
        randi = current[(current.method == method) & (current.reference == "randi_wild_type")]
        cook = current[(current.method == method) & (current.reference == "cook_struct_54")]
        if randi.empty or cook.empty:
            continue
        r, c = randi.iloc[0], cook.iloc[0]
        lines.append(f"| {METHOD_LABEL[method]} | {r.auroc:.3f} | {r.continuous_abs_spearman:.3f} | {c.auroc:.3f} | {c.count_spearman_all:.3f} |")
    lines += ["", "The continuous/count-valued analyses do not reverse the binary ranking. Wide-flow direct and SBTG-published remain stronger contextual comparators than the new higher-order matrices. None of these references is ground-truth activity coupling.", "", "## Neuromodulator receptor enrichment", ""]
    best = receptor.sort_values(["bh_q_global", "permutation_p"]).head(12)
    lines += ["| Cohort | Method | Network | Enrichment | Sources | p | Global BH q |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in best.itertuples():
        lines.append(f"| {row.cohort_label} | {row.method_label} | {row.network} | {row.matched_enrichment:+.4f} | {row.eligible_sources} | {row.permutation_p:.3f} | {row.bh_q_global:.3f} |")
    lines += ["", "Matching adjusts targets within source for activity variance, derivative scale, and amplitude. The global BH correction spans the complete structured and neural experiment family. Receptor enrichment is a stratification clue, not a lag or causal-effect validation.", "", "## Why no new SMC run", "", "SMC can improve finite-particle estimation of a chosen repaired-response functional; it cannot repair a conditional density whose incremental mean/scale/covariance effect fails held-out gates. The previously completed importance, terminal-SMC and progressive-ESS-SMC matrices remain in the post-freeze contextual table. The exact gate record is in `SMC_DECISION.md`.", "", "## Scope and methodology", "", "- Cohorts: current 20-worm/54-neuron NeuroPAL; exact original 20-worm/80-neuron SBTG cache; and SBTG restricted to the shared 54-neuron order.", "- Sampling: 4 Hz; maximum histories 2, 4 and 8 seconds; complete binary stimulus history included.", "- Validation: whole-worm five-fold splits; folds 0–2 selection, folds 3–4 untouched confirmation.", "- Structured laws: smooth mean/log-variance kernels and low-rank history-dependent covariance.", "- Neural laws: heteroscedastic/low-rank Gaussian, MDN, DSM, contrastive energy, Gaussian-source flow and wide flow.", "- Lag matrices: structured coefficients or confirmation-fold central finite differences along the frozen lag basis.", "- Atlas firewall: all external references opened only after neural freeze artifacts were written.", "", "## Limitations and conclusion", "", "Calcium filtering, correlated histories, shared stimulus drive, unmeasured behavior/global state, and only eight confirmation worms prevent causal interpretation. The cross-cohort replication and strong-effect oracle calibration show that the weak result is not well explained by one neuron subset or one architecture. The fair conclusion is a calibrated higher-order lag ceiling: modest predictive distribution structure exists, but the present data and estimators do not support reliable variance/gain lag matrices or a physical-delay claim.", "", "## Recommended next experiment", "", "1. Add neuron-specific latent calcium dynamics/deconvolution and repeat the frozen protocol.", "2. Measure behavior/global state and collect more repeated onsets.", "3. Use receptor atlases to prespecify small source-target strata rather than tune models.", "4. Add controlled single-neuron perturbations to identify intervention effects.", "5. Promote temporal-cut SMC only after prospective proper-score and incremental-history gates pass.", "", "## Artifact map", "", "- `RUN_MANIFEST.md`: frozen cohorts, metrics, stages and claim boundary.", "- `SMC_DECISION.md`: exact no-promotion rationale.", "- `RESULTS_SUMMARY.json`: machine-readable headline results.", "- `artifact.json`: validated interactive technical report payload.", "- `CHART_MAP.md`: visual contract and provenance map.", "- `FULL_INVENTORY.csv` / `FULL_CHECKSUMS.sha256`: recursive ledger over every suite root.", ""]
    (root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")

    chart_map = """# Chart map

| Section | Analytical question | Family / type | Dataset and fields | Supported takeaway |
| --- | --- | --- | --- | --- |
| Neural density tournament | Which confirmed architecture best predicts each cohort's law? | Comparison / grouped bar | `neural_confirmation_summary.csv`: cohort, model, energy, balanced energy, variogram, coverage | Flexible heads have modest, fold-dependent differences; score within cohort |
| Variance detectability | At what injected effect can gain edges and lags be recovered? | Relationship / scatter | `variance_detectability_summary.csv`: effect, AUROC, lag MAE, cohort, condition | Useful ranking appears only at large effects; calcium smoothing worsens lag precision |
| External correspondence | How do new higher-order matrices compare with frozen contextual methods? | Comparison / grouped bar | `external_chart_data.csv`: reference, method, AUROC/AUPRC and continuous/count correlations | New matrices are weak; continuous/count metrics do not reverse the result |

Palette policy: relaxed multi-category for the two grouped comparisons and four-series detectability plot, using approved roots plus neutrals; reference lines are dark neutral. Charts use the native report renderer. Absolute energy is compared only within cohort.
"""
    (root / "CHART_MAP.md").write_text(chart_map)


def write_inventory(root: Path, artifact_roots: list[Path]) -> None:
    excluded = {"FULL_INVENTORY.csv", "FULL_CHECKSUMS.sha256"}
    rows: list[dict] = []
    for artifact_root in artifact_roots:
        for path in sorted(artifact_root.rglob("*")):
            if not path.is_file() or path.name in excluded:
                continue
            rows.append({
                "artifact_root": str(artifact_root),
                "relative_path": str(path.relative_to(artifact_root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    with (root / "FULL_INVENTORY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"{row['sha256']}  {row['artifact_root']}/{row['relative_path']}"
        for row in rows
    ]
    lines.append(f"{_sha256(root / 'FULL_INVENTORY.csv')}  {root}/FULL_INVENTORY.csv")
    (root / "FULL_CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")


def write(args) -> Path:
    artifact = build(
        args.root, args.structured, args.neural, args.postfreeze,
        args.effects_postfreeze, args.ablation,
    )
    payload = {key: artifact[key] for key in ("surface", "manifest", "snapshot", "sources")}
    (args.root / "artifact.json").write_text(json.dumps(_safe(payload), indent=2, allow_nan=False) + "\n")
    (args.root / "RESULTS_SUMMARY.json").write_text(
        json.dumps(artifact["summary"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _write_markdown(args.root, artifact)
    manifest = args.root / "RUN_MANIFEST.md"
    if manifest.exists():
        manifest.write_text(manifest.read_text().replace(
            "**Status:** running; this file is the stable index for the overnight suite.",
            "**Status:** complete; all planned gated stages and post-freeze analyses finished.",
        ))
    write_inventory(args.root, [
        args.root, args.structured, args.neural, args.postfreeze,
        args.effects, args.effects_postfreeze, args.ablation,
    ])
    return args.root / "artifact.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/higher_order_dynamics_20260828"))
    parser.add_argument("--structured", type=Path, default=Path("results/higher_order_lag_20260828"))
    parser.add_argument("--neural", type=Path, default=Path("results/higher_order_neural_20260828"))
    parser.add_argument("--postfreeze", type=Path, default=Path("results/higher_order_postfreeze_20260828"))
    parser.add_argument("--effects", type=Path, default=Path("results/higher_order_neural_effects_20260828"))
    parser.add_argument("--effects-postfreeze", type=Path, default=Path("results/higher_order_neural_effects_postfreeze_20260828"))
    parser.add_argument("--ablation", type=Path, default=Path("results/higher_order_neural_ablation_20260828"))
    args = parser.parse_args()
    for field in (
        "root", "structured", "neural", "postfreeze", "effects",
        "effects_postfreeze", "ablation",
    ):
        setattr(args, field, getattr(args, field).resolve())
    print(f"HIGHER_ORDER_REPORT_COMPLETE {write(args)}", flush=True)


if __name__ == "__main__":
    main()
