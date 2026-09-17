from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TITLE = "Biological analysis of stimulus-conditioned lag dynamics"


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")


def _source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    definitions: list[str],
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "DuckDB",
            "sql": sql,
            "tables_used": [path],
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "language": "sql",
            "description": description,
            "filters": ["20 complete-case worms", "54 frozen shared neuron classes", "4 Hz activity"],
            "metric_definitions": definitions,
        },
    }


def build(root: Path) -> dict:
    generated = datetime.now(timezone.utc).isoformat()
    aligned = pd.read_csv(root / "event_aligned_activity.csv")
    aligned = aligned[
        (aligned.scope == "event_average")
        & aligned.neuron.isin(["AWC", "AIY", "AIB", "AIZ"])
    ].copy()
    latency = pd.read_csv(root / "stimulus_response_latency.csv")
    adaptation = pd.read_csv(root / "repetition_adaptation.csv")
    adaptation = adaptation.groupby("entity", as_index=False).first()
    adaptation["entity_label"] = adaptation.entity.str.replace("class:", "Class: ", regex=False).str.replace(
        "neuron:", "Neuron: ", regex=False
    )
    prediction = pd.read_csv(root / "biological_prediction_summary.csv")
    prediction = prediction[
        (prediction.source_group == "all_sensory")
        & (prediction.target_group == "olfactory_interneurons")
        & prediction.method.isin(["wide_flow_direct", "progressive_smc"])
    ].copy()
    prediction["series"] = prediction.method.str.replace("_", " ") + " · " + prediction.episode.str.replace("_", " ")
    signatures = pd.read_csv(root / "olfactory_circuit_signatures.csv")
    signatures = signatures[
        signatures.target.isin(["AIY", "AIB"])
        & signatures.horizon_seconds.isin([0.5, 1.0, 2.0, 4.0])
    ].copy()
    candidates = pd.read_csv(root / "stable_named_edge_candidates.csv")
    candidates = candidates[candidates.candidate_pass].copy()
    response = pd.read_csv(root / "observed_response_by_neuron_horizon.csv")
    response4 = response[(response.scope == "event_average") & (response.lag_frames == 16)]
    awc_response = response4[response4.neuron == "AWC"].iloc[0]
    awc_latency = latency[latency.neuron == "AWC"].iloc[0]
    awc_adapt = adaptation[adaptation.entity == "neuron:AWC"].iloc[0]

    wide_best = prediction[prediction.episode == "stimulus_onset"].sort_values(
        "mean_spearman_gain", ascending=False
    ).groupby("method", as_index=False).first()
    headline = pd.DataFrame(
        [
            {
                "awc_4s_effect": awc_response.onset_minus_quiet,
                "awc_4s_q": awc_response.bh_q_value,
                "awc_latency_seconds": awc_latency.latency_seconds,
                "awc_latency_global_survival": bool(awc_latency.latency_survives_global_bh),
                "awc_adaptation_slope": awc_adapt.rms_adaptation_slope_per_repeat,
                "awc_adaptation_q": awc_adapt.slope_bh_q_value,
                "strict_cells": int(len(candidates)),
                "strict_unique_edges": int(len(candidates[["source", "target"]].drop_duplicates())),
                "wide_best_gain": float(
                    wide_best[wide_best.method == "wide_flow_direct"].mean_spearman_gain.iloc[0]
                ),
                "smc_best_gain": float(
                    wide_best[wide_best.method == "progressive_smc"].mean_spearman_gain.iloc[0]
                ),
            }
        ]
    )

    base = "results/biological_lag_analysis_20260828"
    sources = [
        _source(
            "aligned_source",
            "Event-aligned onset and quiet activity",
            f"{base}/event_aligned_activity.csv",
            f"SELECT * FROM read_csv_auto('{base}/event_aligned_activity.csv') WHERE scope = 'event_average' AND neuron IN ('AWC','AIY','AIB','AIZ')",
            "Worm-level event-average stimulus responses on the quarter-second grid.",
            [
                "Onset-minus-quiet is the locally baseline-corrected true-onset response minus the matched pseudo-onset response 15 seconds earlier.",
                "Intervals and tests use worms as the replication unit; global BH covers neuron-by-time tests within scope.",
            ],
        ),
        _source(
            "adaptation_source",
            "Repetition-dependent response magnitude",
            f"{base}/repetition_adaptation.csv",
            f"SELECT * FROM read_csv_auto('{base}/repetition_adaptation.csv') QUALIFY row_number() OVER (PARTITION BY entity ORDER BY event) = 1",
            "Worm-level slopes of the four-second RMS onset-minus-quiet contrast across three odor presentations.",
            [
                "RMS adaptation slope is the per-worm linear slope of onset-minus-matched-quiet response magnitude across repetitions 1–3.",
                "Negative slope means attenuation; BH q-values cover the declared neuron and functional-class entities.",
            ],
        ),
        _source(
            "prediction_source",
            "Target-restricted lag prediction",
            f"{base}/biological_prediction_summary.csv",
            f"SELECT * FROM read_csv_auto('{base}/biological_prediction_summary.csv') WHERE source_group = 'all_sensory' AND target_group = 'olfactory_interneurons'",
            "Held-out worm incremental prediction across lag horizons and matched quiet controls.",
            [
                "Spearman gain is the combined persistence-plus-matrix forecast correlation minus the persistence-only correlation.",
                "Coefficients are fit without the held-out worm; source and targets are residualized against other worms at the same repetition.",
            ],
        ),
        _source(
            "signature_source",
            "AWC-centered circuit signatures",
            f"{base}/olfactory_circuit_signatures.csv",
            f"SELECT * FROM read_csv_auto('{base}/olfactory_circuit_signatures.csv') WHERE target IN ('AIY','AIB') AND horizon_seconds IN (0.5,1,2,4)",
            "Matrix coefficients compared separately with residual activity association, population stimulus pattern, and literature sign.",
            [
                "Matrix orientation is target row by source column.",
                "Literature signs encode positive AWC-to-AIB and negative AWC-to-AIY action in the canonical odor circuit.",
            ],
        ),
        _source(
            "candidate_source",
            "Strict sampler-consistent lag candidates",
            f"{base}/stable_named_edge_candidates.csv",
            f"SELECT * FROM read_csv_auto('{base}/stable_named_edge_candidates.csv') WHERE candidate_pass",
            "Named edges passing cross-method, worm-sign, compatibility, magnitude, and observed-covariation gates.",
            [
                "Candidate passage requires direct-flow/progressive-SMC sign agreement, high magnitude rank and sign stability, compatibility validity, and observed BH q ≤ 0.10.",
                "Candidates are reduced-form activity relationships, not synapses or physical delays.",
            ],
        ),
    ]

    cards = [
        {
            "id": "awc_response_card",
            "dataset": "headline",
            "sourceId": "aligned_source",
            "description": "Four-second AWC onset-minus-quiet response and multiplicity-controlled support.",
            "metrics": [
                {"label": "AWC response at 4 s", "field": "awc_4s_effect", "format": "number", "signed": True},
                {"label": "BH q", "field": "awc_4s_q", "format": "number"},
            ],
        },
        {
            "id": "awc_timing_card",
            "dataset": "headline",
            "sourceId": "aligned_source",
            "description": "First two-frame sustained AWC response under within-neuron BH; the first point does not survive global neuron-by-time BH.",
            "metrics": [
                {"label": "AWC sustained latency (s)", "field": "awc_latency_seconds", "format": "number"},
                {"label": "Global-BH at first point", "field": "awc_latency_global_survival", "format": "boolean"},
            ],
        },
        {
            "id": "adaptation_card",
            "dataset": "headline",
            "sourceId": "adaptation_source",
            "description": "Change in the AWC onset-minus-quiet response-magnitude contrast per repeated odor presentation.",
            "metrics": [
                {"label": "AWC RMS slope / repeat", "field": "awc_adaptation_slope", "format": "number", "signed": True},
                {"label": "BH q", "field": "awc_adaptation_q", "format": "number"},
            ],
        },
        {
            "id": "candidate_card",
            "dataset": "headline",
            "sourceId": "candidate_source",
            "description": "Unique reduced-form edge surviving every strict consensus gate.",
            "metrics": [
                {"label": "Strict unique edges", "field": "strict_unique_edges", "format": "number"},
                {"label": "Horizon cells", "field": "strict_cells", "format": "number"},
            ],
        },
    ]

    charts = [
        {
            "id": "event_aligned_chart",
            "title": "Event-aligned AWC olfactory-circuit activity",
            "subtitle": "AWC suppression begins after the 0–0.75 s source window; AIY and AIZ develop later population responses.",
            "type": "line",
            "intent": "trend",
            "dataset": "aligned_circuit",
            "sourceId": "aligned_source",
            "encodings": {
                "x": {"field": "offset_seconds", "type": "quantitative", "label": "Seconds after onset"},
                "y": {"field": "onset_minus_quiet", "type": "quantitative", "label": "Onset − quiet"},
                "color": {"field": "neuron", "type": "nominal", "label": "Neuron"},
                "tooltip": [
                    {"field": "ci_low", "type": "quantitative", "label": "95% CI low"},
                    {"field": "ci_high", "type": "quantitative", "label": "95% CI high"},
                    {"field": "global_bh_q_value", "type": "quantitative", "label": "Global BH q"},
                ],
            },
            "referenceLines": [{"axis": "y", "value": 0, "label": "Matched quiet"}],
            "layout": "full",
        },
        {
            "id": "adaptation_chart",
            "title": "Matched response-magnitude adaptation across repeated odor presentations",
            "subtitle": "Negative slopes indicate attenuation beyond the matched quiet-window trend.",
            "type": "bar",
            "intent": "comparison",
            "dataset": "adaptation",
            "sourceId": "adaptation_source",
            "encodings": {
                "x": {"field": "entity_label", "type": "nominal", "label": "Neuron or class"},
                "y": {"field": "rms_adaptation_slope_per_repeat", "type": "quantitative", "label": "RMS slope per repeat"},
                "tooltip": [
                    {"field": "slope_ci_low", "type": "quantitative", "label": "95% CI low"},
                    {"field": "slope_ci_high", "type": "quantitative", "label": "95% CI high"},
                    {"field": "slope_bh_q_value", "type": "quantitative", "label": "BH q"},
                ],
            },
            "referenceLines": [{"axis": "y", "value": 0, "label": "No adaptation"}],
            "layout": "full",
        },
        {
            "id": "prediction_chart",
            "title": "Sensory-to-olfactory-interneuron prediction gain",
            "subtitle": "Neither learned-matrix series clears multiplicity; quiet controls remain near zero.",
            "type": "line",
            "intent": "comparison",
            "dataset": "prediction",
            "sourceId": "prediction_source",
            "encodings": {
                "x": {"field": "horizon_seconds", "type": "quantitative", "label": "Response horizon (s)"},
                "y": {"field": "mean_spearman_gain", "type": "quantitative", "label": "Incremental Spearman gain"},
                "color": {"field": "series", "type": "nominal", "label": "Method and episode"},
                "tooltip": [
                    {"field": "spearman_gain_ci_low", "type": "quantitative", "label": "95% CI low"},
                    {"field": "spearman_gain_ci_high", "type": "quantitative", "label": "95% CI high"},
                    {"field": "gain_bh_q_value", "type": "quantitative", "label": "BH q"},
                ],
            },
            "referenceLines": [{"axis": "y", "value": 0, "label": "No gain over persistence"}],
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "signature_table",
            "title": "AWC-centered lag-sign checks",
            "subtitle": "Residual association, population stimulus response, and literature sign are different comparisons.",
            "dataset": "signatures",
            "sourceId": "signature_source",
            "defaultSort": {"field": "horizon_seconds", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "target", "label": "Target", "type": "text"},
                {"field": "horizon_seconds", "label": "Horizon (s)", "format": "number"},
                {"field": "matrix_coefficient", "label": "Coefficient", "format": "number"},
                {"field": "empirical_worm_event_spearman", "label": "Residual rho", "format": "number"},
                {"field": "empirical_residual_sign_match", "label": "Residual-sign match", "type": "boolean"},
                {"field": "literature_sign_match", "label": "Literature-sign match", "type": "boolean"},
                {"field": "worm_sign_agreement", "label": "Worm sign agreement", "format": "percent"},
            ],
        },
        {
            "id": "candidate_table",
            "title": "Strict sampler-consistent lag candidates",
            "subtitle": "One RIB→AVB relation survives at two adjacent horizons; it remains observational and reduced-form.",
            "dataset": "candidates",
            "sourceId": "candidate_source",
            "defaultSort": {"field": "horizon_seconds", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "source", "label": "Source", "type": "text"},
                {"field": "target", "label": "Target", "type": "text"},
                {"field": "horizon_seconds", "label": "Horizon (s)", "format": "number"},
                {"field": "wide_flow_coefficient", "label": "Wide flow", "format": "number"},
                {"field": "progressive_smc_coefficient", "label": "Progressive SMC", "format": "number"},
                {"field": "observed_worm_spearman", "label": "Observed rho", "format": "number"},
                {"field": "observed_bh_q_value", "label": "Observed BH q", "format": "number"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "summary",
            "type": "markdown",
            "body": (
                "The clearest result is biological organization in measured activity, not a new atlas leaderboard. Butanone onset produces a canonical AWC-centered population response and strong repetition-dependent attenuation. "
                "The lag matrices recover parts of the expected sign structure, but their held-out downstream forecast is weak and their earliest AWC window precedes sustained AWC suppression."
            ),
        },
        {"id": "headline", "type": "metric-strip", "cardIds": ["awc_response_card", "awc_timing_card", "adaptation_card", "candidate_card"]},
        {
            "id": "observed_heading",
            "type": "markdown",
            "body": "## Observed olfactory response\n\nAWC decreases during the odor pulse, AIY increases, and AIZ decreases. The source summary used by the lag estimator covers 0–0.75 seconds, before sustained AWC suppression is detectable; this timing mismatch is central to interpreting sign instability.",
            "sourceId": "aligned_source",
        },
        {"id": "event_chart", "type": "chart", "chartId": "event_aligned_chart"},
        {
            "id": "adaptation_heading",
            "type": "markdown",
            "body": "## Repetition-dependent gain\n\nThe primary slope uses onset-minus-matched-quiet response magnitude, while onset-only and quiet-only slopes remain in the source table. Supported negative slopes indicate attenuation beyond the time-matched quiet trend. This is evidence for state- or experience-dependent activity gain, not anatomical rewiring.",
            "sourceId": "adaptation_source",
        },
        {"id": "adapt_chart", "type": "chart", "chartId": "adaptation_chart"},
        {
            "id": "matrix_heading",
            "type": "markdown",
            "body": "## What the sampled matrices recover\n\nProgressive ESS-SMC recovers the expected early AWC→AIY negative and AWC→AIB positive literature signs more often than wide-flow direct, but those signs need not match residual co-fluctuation in an early window that still has positive mean AWC activity. These are different biological questions and are reported separately.",
            "sourceId": "signature_source",
        },
        {"id": "signature_results", "type": "table", "tableId": "signature_table"},
        {
            "id": "prediction_heading",
            "type": "markdown",
            "body": "## Held-out downstream prediction\n\nThe decision metric remains incremental worm-level prediction beyond persistence. Across the declared sensory-to-olfactory-interneuron cells, neither primary matrix method survives multiplicity or yields a stable positive interval. This is the current limit on claiming useful lag-resolved propagation.",
            "sourceId": "prediction_source",
        },
        {"id": "prediction_results", "type": "chart", "chartId": "prediction_chart"},
        {
            "id": "candidate_heading",
            "type": "markdown",
            "body": "## Stable named candidate\n\nOnly one unique relation passes every strict consensus gate: RIB→AVB at 0.5 and 1 second. The adjacency of the two horizon cells is encouraging for stability, but passive calcium and common drive prevent a synaptic or physical-delay claim.",
            "sourceId": "candidate_source",
        },
        {"id": "candidate_results", "type": "table", "tableId": "candidate_table"},
        {
            "id": "definitions",
            "type": "markdown",
            "body": "## Definitions and claim boundary\n\nA lag coefficient is a target-row/source-column response of a learned conditional law to a source-compatible perturbation. Response horizon is cumulative future averaging time; it is not receptor latency. Empirical residual coupling is leave-one-worm activity association. Literature-sign match is a circuit prior. None of these alone identifies a synapse, causality, or a physical transmission delay.",
        },
        {
            "id": "methods",
            "type": "markdown",
            "body": "## Methods\n\nThe analysis uses 20 complete-case worms, 54 neuron classes, three known butanone onsets, fold-local standardization, matched quiet pseudo-onsets, worm-level intervals, and BH correction. Primary matrices are the atlas-blind wide-flow direct ensemble and full progressive ESS-SMC ensemble. The targeted temporal-cut SMC cell and globally frozen distributed-lag model are sensitivities with distinct estimands.",
        },
        {
            "id": "limits",
            "type": "markdown",
            "body": "## Limitations\n\nCalcium filtering, 4 Hz sampling, common stimulus input, locomotor feedback, unmeasured state, bilateral class collapse, and 20 worms limit temporal and causal resolution. The 1.25-second AWC latency passes within-neuron but not global neuron-by-time correction at its first point. Predictive cells were searched across a declared grid, and their maxima do not survive correction.",
        },
        {
            "id": "references",
            "type": "markdown",
            "body": "## Biological reference basis\n\n[WormAtlas](https://www.wormatlas.org/hermaphrodite/nervous/mainframe.htm) supplies the broad operational neuron categories. A [butanone/AWC behavioral study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2586605/) supports AWC as the primary declared 2-butanone sensor. The [canonical AWC circuit study](https://www.nature.com/articles/nature06292) supplies the expected positive AWC→AIB and negative AWC→AIY signs. These references are interpretation checks, not model-selection targets.",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": "## Next experiment\n\nKeep the sampling idea, but align its source functional to biology: use an AWC suppression-aligned or signed change-point source functional, then freeze AWC→AIY/AIB literature-sign tests, sensory-to-olfactory held-out gain, and the RIB→AVB 0.5–1-second candidate. Confirm them on new animals or controlled AWC/RIB perturbations with the scoring rule unchanged.",
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": "## Further questions\n\nDoes adaptation alter only response gain or also the conditional law? Does a source functional anchored after AWC suppression improve downstream prediction without reducing SMC compatibility? Does RIB→AVB replicate under locomotor-state stratification? Can higher-rate or deconvolved activity separate calcium latency from network dynamics?",
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Event-aligned biological analysis of frozen flow and ESS-SMC lag matrices.",
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
            "aligned_circuit": _records(aligned),
            "adaptation": _records(adaptation),
            "prediction": _records(prediction),
            "signatures": _records(signatures),
            "candidates": _records(candidates),
        },
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build(args.root.resolve())
    args.output.resolve().write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    main()
