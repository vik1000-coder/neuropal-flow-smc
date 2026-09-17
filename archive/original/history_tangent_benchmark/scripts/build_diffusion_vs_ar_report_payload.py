#!/usr/bin/env python3
"""Build the bounded Data Analytics report payload for final rendering."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "diffusion_vs_ar_scientific_summary_20260713"
GENERATED_AT = "2026-07-13T02:08:00-04:00"
TITLE = "Diffusion versus autoregression for conditional history tangents"


def _records(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def build_payload() -> dict:
    mechanisms = _records(SUMMARY / "tangent_mechanism_summary.csv")
    generation = _records(SUMMARY / "g5_generation_summary.csv")

    chart_rows = []
    for row in mechanisms:
        shared = {
            "mechanism": row["mechanism"],
            "generator_id": row["generator_id"],
            "validity_threshold": 1.0,
            "generator_seeds": row["generator_seeds"],
            "oracle_clean_to_noisy_nrmse": row["oracle_clean_to_noisy_median"],
            "oracle_nll": row["oracle_nll_median"],
        }
        chart_rows.extend(
            [
                {
                    **shared,
                    "method": "Autoregressive Transformer (clean)",
                    "nrmse": row["transformer_median_nrmse"],
                    "ci_low": row["transformer_median_ci_low"],
                    "ci_high": row["transformer_median_ci_high"],
                    "valid": row["transformer_valid"],
                    "eval_ms_per_example": row["transformer_eval_ms_median"],
                    "nll": row["transformer_nll_median"],
                },
                {
                    **shared,
                    "method": "Diffusion (noisy, model-centered)",
                    "nrmse": row["diffusion_median_nrmse"],
                    "ci_low": row["diffusion_median_ci_low"],
                    "ci_high": row["diffusion_median_ci_high"],
                    "valid": row["diffusion_valid"],
                    "eval_ms_per_example": row["diffusion_eval_ms_median"],
                    "nll": None,
                },
            ]
        )

    mechanism_table = [
        {
            "mechanism": row["mechanism"],
            "transformer_nrmse": row["transformer_median_nrmse"],
            "diffusion_nrmse": row["diffusion_median_nrmse"],
            "transformer_relative_pct": row["transformer_relative_pct"],
            "both_valid": bool(row["transformer_valid"] and row["diffusion_valid"]),
            "oracle_gap": row["oracle_clean_to_noisy_median"],
            "diffusion_slower_factor": row["diffusion_slower_factor_median"],
            "transformer_nll": row["transformer_nll_median"],
            "oracle_nll": row["oracle_nll_median"],
        }
        for row in mechanisms
    ]
    g5_table = [
        {
            "model": row["model_name"],
            "estimand": row["estimand"],
            "fair_energy": row["value"],
            "relative_to_transformer_pct": row[
                "relative_to_transformer_mean_pct"
            ],
            "best_epoch": row["fit.best_epoch"],
            "stopped_epoch": row["fit.stopped_epoch"],
            "fit_wall_seconds": row["fit.wall_seconds"],
        }
        for row in generation
    ]

    tangent_source = {
        "id": "tangent_replication",
        "label": "Five-system tangent replication",
        "path": "results/diffusion_vs_ar_scientific_summary_20260713/tangent_mechanism_summary.csv",
        "query": {
            "description": "Order-averaged Transformer and model-centered diffusion tangent metrics aggregated by generator seed and mechanism.",
            "engine": "local-python",
            "language": "python",
            "tables_used": [
                "results/diffusion_vs_ar_invalidity_replication_20260713_a/metrics.csv",
                "results/diffusion_vs_ar_invalidity_replication_20260713_b/metrics.csv",
            ],
            "filters": [
                "generator seeds 101, 103, 107, 109, 113",
                "data seed 401",
                "model seed 1401",
                "Transformer orders averaged within generator seed",
                "diffusion sigma 0.05 with model-sample centering",
            ],
            "metric_definitions": [
                "Tangent NRMSE = RMS estimated-minus-oracle history tangent divided by RMS oracle tangent; values below 1 beat the zero-tangent baseline.",
                "Confidence intervals enumerate all 5^5 ordinary generator-seed bootstrap resamples.",
                "Transformer NLL is averaged across fixed coordinate orders within generator seed.",
            ],
        },
    }
    g5_source = {
        "id": "g5_generation",
        "label": "Extended G5 generation calibration",
        "path": "results/diffusion_vs_ar_scientific_summary_20260713/g5_generation_summary.csv",
        "query": {
            "description": "Fair energy score and convergence diagnostics for the 64-dimensional near-manifold G5 development system.",
            "engine": "local-python",
            "language": "python",
            "tables_used": [
                "results/diffusion_vs_ar_g5_generation_calibration_extended_20260713/metrics.csv",
                "results/diffusion_vs_ar_g5_generation_calibration_extended_20260713/cases.csv",
            ],
            "filters": [
                "generator seed 41",
                "data seed 301",
                "model seed 1301",
                "5000 training examples",
                "150 maximum epochs",
                "64 energy cases and 64 samples per case",
            ],
            "metric_definitions": [
                "Fair energy score is the finite-ensemble bias-corrected multivariate energy score; lower is better.",
                "Relative difference uses mean fair energy across the three fixed Transformer orders as baseline.",
            ],
        },
    }
    audit_source = {
        "id": "frozen_audit",
        "label": "Frozen comparison audit and decision rules",
        "path": "DIFFUSION_VS_AUTOREGRESSION_AUDIT.md",
        "query": {
            "description": "Predeclared calibration gates, estimator definitions, seed axes, and scoped decision language.",
            "engine": "local-file",
            "language": "markdown",
            "tables_used": ["DIFFUSION_VS_AUTOREGRESSION_AUDIT.md"],
            "filters": [],
            "metric_definitions": [
                "A tangent route is valid on a mechanism when median tangent NRMSE is below 1.",
                "The comparison is inconclusive due to estimator invalidity when both routes have median NRMSE at least 1 on three or more mechanisms.",
            ],
        },
    }
    sources = [tangent_source, g5_source, audit_source]
    chart_source = deepcopy(tangent_source)
    chart_source["query"].update(
        {
            "engine": "duckdb",
            "language": "sql",
            "sql": """WITH m AS (
  SELECT * FROM read_csv_auto('results/diffusion_vs_ar_scientific_summary_20260713/tangent_mechanism_summary.csv')
)
SELECT mechanism, generator_id, 'Autoregressive Transformer (clean)' AS method,
       transformer_median_nrmse AS nrmse,
       transformer_median_ci_low AS ci_low,
       transformer_median_ci_high AS ci_high,
       transformer_valid AS valid, 1.0 AS validity_threshold,
       generator_seeds, oracle_clean_to_noisy_median AS oracle_clean_to_noisy_nrmse,
       transformer_eval_ms_median AS eval_ms_per_example,
       transformer_nll_median AS nll, oracle_nll_median AS oracle_nll
FROM m
UNION ALL
SELECT mechanism, generator_id, 'Diffusion (noisy, model-centered)' AS method,
       diffusion_median_nrmse AS nrmse,
       diffusion_median_ci_low AS ci_low,
       diffusion_median_ci_high AS ci_high,
       diffusion_valid AS valid, 1.0 AS validity_threshold,
       generator_seeds, oracle_clean_to_noisy_median AS oracle_clean_to_noisy_nrmse,
       diffusion_eval_ms_median AS eval_ms_per_example,
       NULL AS nll, oracle_nll_median AS oracle_nll
FROM m""",
        }
    )
    mechanism_table_source = deepcopy(tangent_source)
    mechanism_table_source["query"].update(
        {
            "engine": "duckdb",
            "language": "sql",
            "sql": """SELECT mechanism,
       transformer_median_nrmse AS transformer_nrmse,
       diffusion_median_nrmse AS diffusion_nrmse,
       transformer_relative_pct,
       transformer_valid AND diffusion_valid AS both_valid,
       oracle_clean_to_noisy_median AS oracle_gap,
       diffusion_slower_factor_median AS diffusion_slower_factor,
       transformer_nll_median AS transformer_nll,
       oracle_nll_median AS oracle_nll
FROM read_csv_auto('results/diffusion_vs_ar_scientific_summary_20260713/tangent_mechanism_summary.csv')""",
        }
    )
    g5_table_source = deepcopy(g5_source)
    g5_table_source["query"].update(
        {
            "engine": "duckdb",
            "language": "sql",
            "sql": """SELECT model_name AS model, estimand, value AS fair_energy,
       relative_to_transformer_mean_pct AS relative_to_transformer_pct,
       \"fit.best_epoch\" AS best_epoch,
       \"fit.stopped_epoch\" AS stopped_epoch,
       \"fit.wall_seconds\" AS fit_wall_seconds
FROM read_csv_auto('results/diffusion_vs_ar_scientific_summary_20260713/g5_generation_summary.csv')""",
        }
    )

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "A falsification-oriented comparison of direct normalized-likelihood tangents and diffusion mixed-derivative reconstructions.",
        "generatedAt": GENERATED_AT,
        "sources": sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": "## Technical summary\n\nThe frozen architecture-winner decision is **inconclusive due to estimator invalidity**: both the normalized Transformer and implementable diffusion reconstruction fail the NRMSE < 1 validity threshold on covariance, skew-mixture, and multimodal-occupancy mechanisms. The replicated result is nevertheless clear: direct autoregressive tangents work well and cheaply for location effects, while neither predictive likelihood nor diffusion response-score fit validates shape-changing history derivatives. On the 64-dimensional G5 development system, diffusion sampled far faster but had worse fair energy; that endpoint remains development evidence, not multi-seed confirmation.",
            },
            {
                "id": "tangent_finding",
                "type": "markdown",
                "sourceId": "tangent_replication",
                "body": "## The replicated result is a location-versus-shape split\n\nAcross five independently generated systems, both methods are valid on G1 location and G6 rough location, and both are invalid on G2 covariance, G3 skew mixture, and G4 multimodal occupancy. Transformer orders are averaged within generator seed; diffusion uses the implementable model-sample-centered tangent at standardized noise 0.05. The clean-to-noisy oracle discrepancy stays below 0.018 on every mechanism, so the small corruption level does not explain the failures.",
            },
            {"id": "tangent_chart_block", "type": "chart", "chartId": "tangent_nrmse"},
            {
                "id": "tangent_chart_note",
                "type": "markdown",
                "body": "The chart uses one NRMSE scale; 1 is the frozen zero-estimator validity threshold. Relative ranking does not override validity: a method can be less wrong and still unusable.",
            },
            {
                "id": "predictive_derivative",
                "type": "markdown",
                "sourceId": "tangent_replication",
                "body": "## Predictive fit does not validate derivative fit\n\nTransformer NLL remains close to exact oracle NLL even where tangent error exceeds 1. Diffusion response-score NRMSE is below 0.50 on every mechanism, yet its implementable history tangent fails G2–G4. This is evidence for a measurement-interface distinction, not merely an optimization failure.",
            },
            {"id": "mechanism_table_block", "type": "table", "tableId": "mechanism_table"},
            {
                "id": "g5_finding",
                "type": "markdown",
                "sourceId": "g5_generation",
                "body": "## High-dimensional generation trades quality for throughput here\n\nAfter extending training from 60 to 150 epochs, mean Transformer fair energy is 1.1768, EDM is 1.3226 (12.4% worse), and legacy diffusion is 1.7072 (45.1% worse). EDM samples about 250 times faster because it updates all 64 coordinates in parallel. This is a one-generator development result and cannot decide the frozen multi-seed diffusion-generation claim.",
            },
            {"id": "g5_table_block", "type": "table", "tableId": "g5_table"},
            {
                "id": "definitions",
                "type": "markdown",
                "body": "## Scope and metric definitions\n\nThe estimand is the gradient with respect to conditioning history of log conditional density. Transformer values target the clean law directly. Diffusion values target the Gaussian-corrupted law at standardized noise 0.05 and require a model-sample centering correction. NRMSE below 1 means the estimator beats a zero-tangent baseline. Fair energy is a proper multivariate sample score; lower is better.",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "frozen_audit",
                "body": "## Experimental design and validation\n\nCalibration used a disjoint development seed and native validation losses only. A data-only history-score repair failed the unchanged gate and was rejected. The post-gate replication used five new generator seeds, one paired data seed, one paired model seed, 10,000 training examples, and three fixed Transformer orders for vector outputs. The generator seed is the inferential unit. All 90 expected cases completed with unique IDs and no failed metrics under one source and environment digest; all 76 repository tests pass.",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## Limitations, uncertainty, and robustness\n\nThe tangent replication measures generator-system and coordinate-order variation but uses one paired data seed and model initialization. EDM failed tangent calibration and appears only as a generation candidate. G5 has one development generator. The synthetic results support no causal, anatomical, latent-rewiring, or biological-mechanism claim. Overall status: **share with caveats**—the estimator-invalidity conclusion is well supported, while the G5 direction is not confirmatory.",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": "## Recommended next step\n\nDo not spend the next budget merely adding seeds to the same invalid tangent estimators. Add a tangent-specific normalized ratio or energy route—or a diffusion model with an explicit normalized history-ratio head—and require it to pass G2–G4 calibration before the frozen multi-seed confirmation. For current use, prefer normalized autoregression for location-like tangents; use diffusion as a fast sampler only when its proper sample score clears the application threshold.",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": "## Further questions\n\n- Can a normalized ratio or energy head recover covariance, skewness, and regime-occupancy tangents without oracle labels?\n- Can diffusion centering be normalized without relying on biased model samples?\n- Does diffusion's throughput advantage become a proper-score advantage at larger response dimension or capacity?\n- How much model-initialization variance remains after generator-seed and coordinate-order averaging?",
            },
        ],
        "charts": [
            {
                "id": "tangent_nrmse",
                "type": "bar",
                "title": "Median history-tangent NRMSE by mechanism",
                "description": "Five generator systems; lower is better. The frozen validity threshold is 1.",
                "dataset": "tangent_chart",
                "encodings": {
                    "x": {"field": "mechanism", "type": "nominal", "title": "Mechanism"},
                    "y": {"field": "nrmse", "type": "quantitative", "title": "Median tangent NRMSE"},
                    "color": {"field": "method", "type": "nominal", "title": "Method"},
                },
                "options": {
                    "orientation": "horizontal",
                    "grouping": "grouped",
                    "showValues": True,
                    "referenceLines": [
                        {"axis": "y", "value": 1.0, "label": "Validity threshold"}
                    ],
                },
                "source": chart_source,
            }
        ],
        "tables": [
            {
                "id": "mechanism_table",
                "title": "Mechanism-level tangent audit",
                "description": "Order-averaged Transformer versus model-centered diffusion across five generator systems.",
                "dataset": "mechanism_table",
                "columns": [
                    {"field": "mechanism", "label": "Mechanism", "type": "text"},
                    {"field": "transformer_nrmse", "label": "Transformer NRMSE", "type": "number"},
                    {"field": "diffusion_nrmse", "label": "Diffusion NRMSE", "type": "number"},
                    {"field": "transformer_relative_pct", "label": "Transformer relative", "type": "number", "unit": "%"},
                    {"field": "both_valid", "label": "Both valid", "type": "boolean"},
                    {"field": "oracle_gap", "label": "Clean–noisy oracle gap", "type": "number"},
                    {"field": "diffusion_slower_factor", "label": "Diffusion tangent time", "type": "number", "unit": "×"},
                    {"field": "transformer_nll", "label": "Transformer NLL", "type": "number"},
                    {"field": "oracle_nll", "label": "Oracle NLL", "type": "number"},
                ],
                "defaultSort": {"field": "mechanism", "direction": "asc"},
                "source": mechanism_table_source,
            },
            {
                "id": "g5_table",
                "title": "G5 difficult-geometry generation calibration",
                "description": "64-dimensional, eight-mode near-manifold response; extended 150-epoch development run.",
                "dataset": "g5_table",
                "columns": [
                    {"field": "model", "label": "Model", "type": "text"},
                    {"field": "fair_energy", "label": "Fair energy", "type": "number"},
                    {"field": "relative_to_transformer_pct", "label": "Relative to Transformer mean", "type": "number", "unit": "%"},
                    {"field": "best_epoch", "label": "Best epoch", "type": "number"},
                    {"field": "stopped_epoch", "label": "Stopped epoch", "type": "number"},
                ],
                "defaultSort": {"field": "fair_energy", "direction": "asc"},
                "source": g5_table_source,
            },
        ],
    }
    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": GENERATED_AT,
        "datasets": {
            "tangent_chart": chart_rows,
            "mechanism_table": mechanism_table,
            "g5_table": g5_table,
        },
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


def main() -> None:
    payload = build_payload()
    destination = SUMMARY / "report_artifact.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
