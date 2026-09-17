"""Validation, tidy metric export, and answer-first Markdown reporting."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .metric_contract import resolve_metric_contract
from .serialization import atomic_json


def load_records(output_dir: str | Path) -> list[dict[str, Any]]:
    paths = sorted((Path(output_dir) / "results").glob("*.json"))
    return [json.loads(path.read_text()) for path in paths]


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    problems = []
    case_ids = [record.get("case_id") for record in records]
    duplicates = [case for case, count in Counter(case_ids).items() if count > 1]
    if duplicates:
        problems.append(f"duplicate case IDs: {duplicates[:5]}")
    for record in records:
        case = record.get("case_id", "unknown")
        if record.get("orientation") != "[target, source]":
            problems.append(f"{case}: missing canonical orientation")
        split = record.get("split", {})
        groups = [set(split.get(name, [])) for name in ("train_groups", "validation_groups", "test_groups")]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            problems.append(f"{case}: overlapping episode splits")
        if record.get("status") == "ok":
            capability = record.get("capabilities", {}).get("predictive_distribution")
            metrics = record.get("metrics", {})
            has_nll = "predictive.nll" in metrics
            if capability == "normalized" and not has_nll:
                problems.append(f"{case}: normalized model missing test NLL")
            elif capability == "normalized":
                nll = metrics.get("predictive.nll")
                if (
                    isinstance(nll, bool)
                    or not isinstance(nll, (int, float, np.integer, np.floating))
                    or not np.isfinite(nll)
                ):
                    problems.append(
                        f"{case}: normalized model has non-finite test NLL {nll!r}"
                    )
            if capability != "normalized" and has_nll:
                problems.append(f"{case}: non-normalized model was assigned a fake NLL")
            if capability == "unnormalized_score":
                ladder = [
                    value
                    for metric_id, value in metrics.items()
                    if metric_id.startswith("score.test.")
                    and ".fixed_reference." in metric_id
                    and metric_id.endswith(".ladder_mean")
                ]
                if not ladder:
                    problems.append(
                        f"{case}: score model missing fixed-reference test DSM ladder risk"
                    )
                elif len(ladder) != 1:
                    problems.append(
                        f"{case}: score model has multiple fixed-reference test DSM domains"
                    )
                elif any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)
                    for value in ladder
                ):
                    problems.append(
                        f"{case}: score model has non-finite fixed-reference test DSM ladder risk"
                    )
            for metric_id in metrics:
                try:
                    resolve_metric_contract(metric_id)
                except ValueError as error:
                    problems.append(f"{case}: {error}")
    return {
        "n_records": len(records),
        "n_ok": sum(record.get("status") == "ok" for record in records),
        "n_failed": sum(record.get("status") == "failed" for record in records),
        "valid": not problems,
        "problems": problems,
    }


def _metric_direction(metric: str) -> tuple[str, float | None]:
    contract = resolve_metric_contract(metric)
    return contract.direction, contract.optimum


def _claim(metric: str) -> tuple[str, str, str, float | None]:
    contract = resolve_metric_contract(metric)
    return (
        contract.estimand,
        contract.claim_level,
        contract.direction,
        contract.optimum,
    )


def tidy_metrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        case = record.get("case", {})
        scenario = case.get("scenario", {})
        method = case.get("method", {})
        for metric, value in record.get("metrics", {}).items():
            contract = resolve_metric_contract(metric)
            rows.append(
                {
                    "case_id": record.get("case_id"),
                    "metric_id": metric,
                    "estimand": contract.estimand,
                    "claim_level": contract.claim_level,
                    "direction": contract.direction,
                    "optimum": contract.optimum,
                    "unit": contract.unit,
                    "role": contract.role,
                    "method": method.get("name"),
                    "scenario": scenario.get("id"),
                    "dgp_family": scenario.get("family"),
                    "mechanism": scenario.get("params", {}).get("mechanism"),
                    "view": case.get("view"),
                    "horizon": case.get("horizon"),
                    "replicate_id": case.get("seed"),
                    "value": value,
                    "status": record.get("status"),
                    "oracle_level": record.get("oracle_level"),
                    "effective_information_set": record.get(
                        "effective_information_set"
                    ),
                    "uncertainty_unit": "independent_dgp_seed",
                    "split_id": record.get("case_id"),
                }
            )
    return pd.DataFrame(rows)


def aggregate(tidy: pd.DataFrame) -> pd.DataFrame:
    clean = tidy[tidy.value.notna()].copy()
    group = [
        "metric_id",
        "estimand",
        "claim_level",
        "direction",
        "optimum",
        "unit",
        "role",
        "method",
        "scenario",
        "view",
        "horizon",
    ]
    result = clean.groupby(group, dropna=False).value.agg(
        ["count", "mean", "std", "median", "min", "max"]
    ).reset_index()
    result["se"] = result["std"] / np.sqrt(result["count"])
    critical = student_t.ppf(.975, np.maximum(result["count"] - 1, 1))
    critical = np.where(result["count"] > 1, critical, np.nan)
    result["ci95_low"] = result["mean"] - critical * result["se"]
    result["ci95_high"] = result["mean"] + critical * result["se"]
    result["ci_method"] = "two-sided Student-t interval over independent DGP seeds"
    return result


def _table(
    frame: pd.DataFrame,
    metric: str,
    view: str | None = None,
    scenario: str | None = None,
    horizon: int | None = None,
    limit: int = 12,
) -> str:
    data = frame[frame.metric_id == metric]
    if view is not None:
        data = data[data.view == view]
    if scenario is not None:
        data = data[data.scenario == scenario]
    if horizon is not None:
        data = data[data.horizon == horizon]
    if data.empty:
        return "_Not available in this run._"
    direction = data.direction.iloc[0]
    optimum = data.optimum.iloc[0]
    strata = [name for name in ("scenario", "view", "horizon") if data[name].nunique() > 1]
    group = strata + ["method"]
    summary = data.groupby(group, dropna=False).agg(
        value=("value", "mean"), n=("value", "count")
    ).reset_index()
    if direction == "target" and pd.notna(optimum):
        summary["distance_to_optimum"] = np.abs(summary.value - float(optimum))
        sort_value, ascending = "distance_to_optimum", True
    else:
        sort_value, ascending = "value", direction != "up"
    if strata:
        summary = (
            summary.sort_values(strata + [sort_value], ascending=[True] * len(strata) + [ascending])
            .groupby(strata, dropna=False, group_keys=False)
            .head(limit)
        )
    else:
        summary = summary.sort_values(sort_value, ascending=ascending).head(limit)
    return summary.to_markdown(index=False, floatfmt=".4f")


def _paired_nll_skill_table(
    frame: pd.DataFrame,
    *,
    baseline: str,
    view: str,
    scenario: str,
    horizon: int,
) -> str:
    data = frame[
        (frame.metric_id == "predictive.nll")
        & (frame.view == view)
        & (frame.scenario == scenario)
        & (frame.horizon == horizon)
    ]
    if data.empty or baseline not in set(data.method):
        return "_Not available in this run._"
    pivot = data.pivot_table(
        index=["scenario", "view", "horizon", "replicate_id"],
        columns="method",
        values="value",
        aggfunc="first",
    )
    rows = []
    for method in pivot.columns:
        if method == baseline:
            continue
        paired = pivot[[baseline, method]].dropna()
        if paired.empty:
            continue
        skill = paired[baseline] - paired[method]
        rows.append(
            {
                "method": method,
                "paired_n": len(skill),
                "delta_nll_vs_ridge": float(skill.mean()),
            }
        )
    if not rows:
        return "_Not available in this run._"
    return (
        pd.DataFrame(rows)
        .sort_values("delta_nll_vs_ridge", ascending=False)
        .to_markdown(index=False, floatfmt=".4f")
    )


def write_report(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    records = load_records(output)
    validation = validate_records(records)
    tidy = tidy_metrics(records)
    summary = aggregate(tidy) if not tidy.empty else pd.DataFrame()
    tidy.to_csv(output / "metrics_tidy.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    atomic_json(output / "validation.json", validation)

    methods = sorted({record.get("case", {}).get("method", {}).get("name") for record in records})
    scenarios = sorted({record.get("case", {}).get("scenario", {}).get("id") for record in records})
    wall = sum(record.get("resources", {}).get("wall_seconds", 0.0) for record in records)
    peak = max((record.get("resources", {}).get("peak_rss_gb", 0.0) for record in records), default=0)
    lines = [
        "# Benchmark report",
        "",
        f"Validated **{validation['n_ok']}** successful cases across "
        f"**{len(methods)} methods** and **{len(scenarios)} mechanisms**. "
        f"Artifact validation: **{'PASS' if validation['valid'] else 'FAIL'}**. "
        f"Cumulative case time was {wall:.1f}s; peak process RSS was {peak:.2f} GiB.",
        "",
        "This report does not create one overall leaderboard. Normalized densities, "
        "unnormalized score models, graph-only methods, latent-state models, and bridges "
        "are compared only on capabilities they actually expose.",
        "",
        "## Predictive law",
        "",
        "Mixed-mechanism, horizon-1 complete-state held-out NLL (lower is better):",
        "",
        _table(tidy, "predictive.nll", "complete_state", "mixed", horizon=1),
        "",
        "Paired NLL skill against ridge on exactly the same seeds (higher is better):",
        "",
        _paired_nll_skill_table(
            tidy,
            baseline="ridge_var",
            view="complete_state",
            scenario="mixed",
            horizon=1,
        ),
        "",
        "Mixed-mechanism, horizon-1 calcium held-out NLL (lower is better):",
        "",
        _table(tidy, "predictive.nll", "calcium", "mixed", horizon=1),
        "",
        "Finite-time quadratic operator error on complete state (lower is better):",
        "",
        _table(
            tidy,
            "operator.quadratic_probe_nrmse",
            "complete_state",
            "mixed",
            horizon=1,
        ),
        "",
        "Tail-exceedance operator error in the matched-tail scenario (lower is better):",
        "",
        _table(
            tidy,
            "operator.tail_exceedance_rmse",
            "complete_state",
            "matched_tail",
            horizon=1,
        ),
        "",
        "Standardized two-sided shape-tail operator error (mean/variance invariant):",
        "",
        _table(
            tidy,
            "operator.shape_tail_exceedance_rmse",
            "complete_state",
            "matched_tail",
            horizon=1,
        ),
        "",
        "Coherent rollout path energy score (only methods with registered joint paths):",
        "",
        _table(tidy, "rollout.path_energy_score_fair"),
        "",
        "## Physical neuromodulator recovery",
        "",
        "Occupied-state physical mean-response field nISE (lower is better):",
        "",
        _table(
            tidy,
            "neuromodulator.physical_mean_state_field.nise",
            "complete_state",
            "additive_mean",
            horizon=1,
        ),
        "",
        "Occupied-state stochastic-dispersion field nISE (lower is better):",
        "",
        _table(
            tidy,
            "neuromodulator.physical_logvariance_state_field.nise",
            "complete_state",
            "stochastic_dispersion",
            horizon=1,
        ),
        "",
        "Occupied-state standardized shape-tail field nISE (lower is better):",
        "",
        _table(
            tidy,
            "neuromodulator.physical_shape_tail_state_field.nise",
            "complete_state",
            "matched_tail",
            horizon=1,
        ),
        "",
        "Occupied-state off-diagonal correlation-routing field nISE:",
        "",
        _table(
            tidy,
            "neuromodulator.physical_correlation_state_field_offdiagonal.nise",
            "complete_state",
            "correlation_routing",
            horizon=1,
        ),
        "",
        "Occupied-state total modulator-gated response nISE in synaptic gain:",
        "",
        _table(
            tidy,
            "neuromodulator.total_gated_response_state_field.nise",
            "complete_state",
            "synaptic_gain",
            horizon=1,
        ),
        "",
        "## Held-out interventions",
        "",
        "Double-dose arm-history conditional-mean effect RMSE (lower is better; P1 transfer, not C1):",
        "",
        _table(
            tidy,
            "arm_history_environment_transfer.ligand_double_dose.conditional_mean.effect_rmse",
            "complete_state",
            "mixed",
            horizon=1,
        ),
        "",
        "Ligand-pulse population-mean arm-history kernel nRMSE "
        "(teacher-forced P1 transfer; not a controlled rollout):",
        "",
        _table(
            tidy,
            "arm_history_environment_transfer.ligand_pulse.population_mean.response_kernel.normalized_rmse",
            "complete_state",
            horizon=1,
        ),
        "",
        "Ligand-pulse history-conditional pairwise nRMSE before population averaging "
        "(teacher-forced P1 transfer):",
        "",
        _table(
            tidy,
            "arm_history_environment_transfer.ligand_pulse.history_conditional.response_kernel.pairwise_normalized_rmse_mean",
            "complete_state",
            horizon=1,
        ),
        "",
        "Explicit receptor-knockout arm-history population kernel nRMSE "
        "(secondary represented-operation C1; still teacher-forced):",
        "",
        _table(
            tidy,
            "intervention.receptor_knockout.arm_history.population_mean.response_kernel.normalized_rmse",
            "complete_state",
            horizon=1,
        ),
        "",
        "Controlled common-history receptor-knockout lag-1 nRMSE "
        "(primary C1; K=1 or anchored labels only):",
        "",
        _table(
            tidy,
            "causal_intervention.receptor_knockout.common_history_lag1.normalized_rmse",
            "complete_state",
            horizon=1,
        ),
        "",
        "Controlled common-history receptor-knockout lag-1 null leakage "
        "(expected-null sham guardrail):",
        "",
        _table(
            tidy,
            "causal_intervention.receptor_knockout.common_history_lag1.null_leakage_rms",
            "complete_state",
            horizon=1,
        ),
        "",
        "## Observational graph stress test",
        "",
        "AP excess over prevalence with a persistent omitted common driver:",
        "",
        _table(
            tidy,
            "effect.linear_transition_coefficient_vs_mean.auprc_lift_over_prevalence",
            "complete_state",
            "strong_hidden_common_driver",
            horizon=1,
        ),
        "",
        "## Latent kinetics",
        "",
        _table(
            tidy,
            "latent.test_nrmse",
            "latent",
            "mixed",
            horizon=1,
        ),
        "",
        _table(
            tidy,
            "latent.clearance_tau_relative_mae",
            "latent",
            "mixed",
            horizon=1,
        ),
        "",
        "## Interpretation guardrails",
        "",
        "- `conditional_log_variance_derivative` is stochastic dispersion, not response gain.",
        "- SBTG joint squared-score covariance is not SID conditional dispersion.",
        "- DSM selection and reported comparisons use a fixed kernel-specific reference ladder; training-objective risks remain diagnostic.",
        "- Structural correspondence on latent/calcium views is secondary; only the "
        "  complete-state lane has the matching analytic one-step oracle.",
        "- In complete-state response suites, ordinary predictors receive the registered "
        "  current modulator feature, while the latent neuromodulator SSM deliberately "
        "  reconstructs it from neural history and stimulus. Those P1/response values "
        "  have different information sets and are not a like-for-like leaderboard.",
        "- Passive environment transfer to receptor-knockout traces is not labeled causal "
        "  unless the method exposes an explicit knockout operation.",
        "- Full response curves consume realized arm histories; only the registered "
        "  common-history lag-1 contrast is a controlled C1 endpoint in this runner.",
        "- Receptor-specific latent C1 is suppressed for exchangeable multi-modulator "
        "  coordinates unless a validation-independent label anchor is declared.",
        "- Response kernels and rollout metrics are emitted only for registered operations/path samplers; missing cells are not losses.",
        "- Tables are never pooled across horizons or mechanisms; paired NLL skill uses only exact seed intersections.",
        "",
        "See `docs/THEORY.md` and the full tidy metric export for claim-specific definitions.",
    ]
    if validation["problems"]:
        lines.extend(["", "## Validation problems", ""] + [f"- {item}" for item in validation["problems"]])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return {"validation": validation, "report": str(output / "REPORT.md")}
