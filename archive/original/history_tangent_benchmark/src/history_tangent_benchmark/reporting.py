"""Strict case-record validation and developmental smoke reporting.

The benchmark deliberately writes one JSON object per case.  This module treats
those records as the source of truth, retains explicit unavailable metrics, and
never aggregates across clean/noisy laws or across oracle/model centering.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


CASE_STATUSES = frozenset({"ok", "failed"})
METRIC_STATUSES = frozenset({"ok", "not_applicable", "failed", "skipped"})
# Native training losses and the developmental diffusion sampler are deliberately
# named as separate strata.  Neither is silently treated as a clean-law metric.
ESTIMANDS = frozenset(
    {"clean", "noisy", "native_objective", "approximate_clean_sampler"}
)
CENTERING_MODES = frozenset({"none", "oracle_samples", "model_samples"})

NATIVE_METRICS = frozenset({"native_validation_loss"})
LAW_METRICS = frozenset({"nll_original", "energy_score_v", "energy_score_fair"})
TANGENT_METRICS = frozenset(
    {
        "tangent_nrmse",
        "tangent_cosine",
        "tangent_absolute_rms",
        "centering_error",
        "finite_ratio_nrmse",
        "finite_ratio_absolute_rmse",
        "response_score_nrmse",
        "oracle_clean_to_noisy_tangent_nrmse",
    }
)
COST_METRICS = frozenset({"tangent_eval_ms_per_example", "samples_per_second"})

_REQUIRED_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "status",
        "generator",
        "data_seed",
        "model",
        "fit",
        "metrics",
        "diagnostics",
        "provenance",
    }
)
_REQUIRED_GENERATOR_FIELDS = frozenset({"id", "kind", "params", "seed"})
_REQUIRED_MODEL_FIELDS = frozenset(
    {"name", "kind", "params", "seed", "capabilities"}
)
_REQUIRED_METRIC_FIELDS = frozenset(
    {"metric_id", "status", "estimand", "centering", "unit"}
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid case record {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid case record {path}: top level must be an object")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def _load_records_and_paths(output_dir: str | Path) -> tuple[list[dict[str, Any]], list[Path]]:
    results_dir = Path(output_dir) / "results"
    if not results_dir.is_dir():
        raise ValueError(f"case-record directory does not exist: {results_dir}")
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no atomic case records found under {results_dir}")
    return [_strict_json(path) for path in paths], paths


def load_case_records(output_dir: str | Path) -> list[dict[str, Any]]:
    """Load complete JSON case files from ``<output>/results`` in stable order."""

    records, _ = _load_records_and_paths(output_dir)
    return records


def _missing(mapping: Mapping[str, Any], required: frozenset[str]) -> list[str]:
    return sorted(required - set(mapping))


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _find_nonfinite(value: Any, location: str, problems: list[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(f"{location}: non-finite numeric value")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _find_nonfinite(item, f"{location}.{key}", problems)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _find_nonfinite(item, f"{location}[{index}]", problems)


def _validate_identity(
    value: Any,
    *,
    required: frozenset[str],
    container: str,
    name_field: str,
    case_id: str,
    problems: list[str],
) -> None:
    if not isinstance(value, Mapping):
        problems.append(f"{case_id}: {container} must be an object")
        return
    missing = _missing(value, required)
    if missing:
        problems.append(f"{case_id}: {container} missing fields {missing}")
    for field in (name_field, "kind"):
        if field in value and (not isinstance(value[field], str) or not value[field]):
            problems.append(f"{case_id}: {container}.{field} must be a nonempty string")
    if "seed" in value and not _is_integer(value["seed"]):
        problems.append(f"{case_id}: {container}.seed must be an integer")
    if "params" in value and not isinstance(value["params"], Mapping):
        problems.append(f"{case_id}: {container}.params must be an object")


def _validate_metric(metric: Any, case_id: str, index: int, problems: list[str]) -> None:
    location = f"{case_id}: metrics[{index}]"
    if not isinstance(metric, Mapping):
        problems.append(f"{location} must be an object")
        return
    missing = _missing(metric, _REQUIRED_METRIC_FIELDS)
    if missing:
        problems.append(f"{location} missing fields {missing}")
    metric_id = metric.get("metric_id")
    if not isinstance(metric_id, str) or not metric_id:
        problems.append(f"{location}.metric_id must be a nonempty string")
    status = metric.get("status")
    if status not in METRIC_STATUSES:
        problems.append(f"{location}.status must be one of {sorted(METRIC_STATUSES)}")
    estimand = metric.get("estimand")
    if estimand not in ESTIMANDS:
        problems.append(f"{location}.estimand must be one of {sorted(ESTIMANDS)}")
    centering = metric.get("centering")
    if centering not in CENTERING_MODES:
        problems.append(
            f"{location}.centering must be one of {sorted(CENTERING_MODES)}"
        )
    if not isinstance(metric.get("unit"), str) or not metric.get("unit"):
        problems.append(f"{location}.unit must be a nonempty string")

    sigma = metric.get("noise_sigma_standardized")
    if estimand == "noisy":
        if not _finite_number(sigma) or float(sigma) <= 0:
            problems.append(
                f"{location}.noise_sigma_standardized must be positive for noisy metrics"
            )
    elif estimand == "clean" and sigma is not None:
        problems.append(
            f"{location}.noise_sigma_standardized must be absent/null for clean metrics"
        )

    value = metric.get("value")
    if status == "ok":
        if "value" not in metric or not _finite_number(value):
            problems.append(f"{location}.value must be finite when status is ok")
    else:
        if value is not None:
            problems.append(f"{location}.value must be null/absent when status is {status}")
        if not isinstance(metric.get("reason"), str) or not metric.get("reason"):
            problems.append(f"{location}.reason is required when status is {status}")


def validate_case_records(
    records: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[Path] | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate schema, finite values, explicit N/A, IDs, and atomic filenames."""

    problems: list[str] = []
    if not records:
        problems.append("no case records supplied")
    if paths is not None and len(paths) != len(records):
        problems.append("record/path count mismatch")
    seen_case_ids: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            problems.append(f"record[{index}] must be an object")
            continue
        _find_nonfinite(record, f"record[{index}]", problems)
        case_id = record.get("case_id")
        label = case_id if isinstance(case_id, str) and case_id else f"record[{index}]"
        missing = _missing(record, _REQUIRED_RECORD_FIELDS)
        if missing:
            problems.append(f"{label}: missing top-level fields {missing}")
        if record.get("schema_version") != "1":
            problems.append(f"{label}: unsupported schema_version {record.get('schema_version')!r}")
        if not isinstance(case_id, str) or not case_id:
            problems.append(f"{label}: case_id must be a nonempty string")
        elif case_id in seen_case_ids:
            problems.append(f"{label}: duplicate case_id")
        else:
            seen_case_ids.add(case_id)
        if paths is not None and index < len(paths) and isinstance(case_id, str):
            if paths[index].stem != case_id:
                problems.append(
                    f"{label}: filename {paths[index].name!r} does not match case_id"
                )
        status = record.get("status")
        if status not in CASE_STATUSES:
            problems.append(f"{label}: status must be one of {sorted(CASE_STATUSES)}")

        _validate_identity(
            record.get("generator"),
            required=_REQUIRED_GENERATOR_FIELDS,
            container="generator",
            name_field="id",
            case_id=label,
            problems=problems,
        )
        model = record.get("model")
        _validate_identity(
            model,
            required=_REQUIRED_MODEL_FIELDS,
            container="model",
            name_field="name",
            case_id=label,
            problems=problems,
        )
        if isinstance(model, Mapping) and "capabilities" in model:
            if not isinstance(model["capabilities"], Mapping):
                problems.append(f"{label}: model.capabilities must be an object")
        if not _is_integer(record.get("data_seed")):
            problems.append(f"{label}: data_seed must be an integer")
        for field in ("fit", "diagnostics", "provenance"):
            if not isinstance(record.get(field), Mapping):
                problems.append(f"{label}: {field} must be an object")

        metrics = record.get("metrics")
        if not isinstance(metrics, list):
            problems.append(f"{label}: metrics must be a list")
        else:
            seen_metrics: set[tuple[Any, Any, Any, Any]] = set()
            for metric_index, metric in enumerate(metrics):
                _validate_metric(metric, label, metric_index, problems)
                if isinstance(metric, Mapping):
                    identity = (
                        metric.get("metric_id"),
                        metric.get("estimand"),
                        metric.get("noise_sigma_standardized"),
                        metric.get("centering"),
                    )
                    if identity in seen_metrics:
                        problems.append(
                            f"{label}: duplicate metric stratum {identity!r}"
                        )
                    seen_metrics.add(identity)

        if status == "failed":
            error = record.get("error")
            if not isinstance(error, Mapping) or not error:
                problems.append(f"{label}: failed case requires a nonempty error object")

    if results_dir is not None:
        leftovers = sorted(Path(results_dir).glob("*.tmp"))
        if leftovers:
            problems.append(
                "incomplete temporary case files remain: "
                + ", ".join(path.name for path in leftovers[:5])
            )

    return {
        "n_records": len(records),
        "n_ok": sum(record.get("status") == "ok" for record in records),
        "n_failed": sum(record.get("status") == "failed" for record in records),
        "n_metric_ok": sum(
            metric.get("status") == "ok"
            for record in records
            if isinstance(record.get("metrics"), list)
            for metric in record["metrics"]
            if isinstance(metric, Mapping)
        ),
        "n_metric_not_applicable": sum(
            metric.get("status") == "not_applicable"
            for record in records
            if isinstance(record.get("metrics"), list)
            for metric in record["metrics"]
            if isinstance(metric, Mapping)
        ),
        "n_metric_failed": sum(
            metric.get("status") == "failed"
            for record in records
            if isinstance(record.get("metrics"), list)
            for metric in record["metrics"]
            if isinstance(metric, Mapping)
        ),
        "n_metric_skipped": sum(
            metric.get("status") == "skipped"
            for record in records
            if isinstance(record.get("metrics"), list)
            for metric in record["metrics"]
            if isinstance(metric, Mapping)
        ),
        "valid": not problems,
        "problems": problems,
    }


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def cases_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return one provenance-preserving row per case."""

    rows: list[dict[str, Any]] = []
    for record in records:
        generator = record.get("generator", {})
        model = record.get("model", {})
        metrics = record.get("metrics", [])
        error = record.get("error", {})
        row: dict[str, Any] = {
            "case_id": record.get("case_id"),
            "status": record.get("status"),
            "generator_id": generator.get("id"),
            "generator_kind": generator.get("kind"),
            "generator_seed": generator.get("seed"),
            "data_seed": record.get("data_seed"),
            "model_name": model.get("name"),
            "model_kind": model.get("kind"),
            "model_seed": model.get("seed"),
            "generator_params_json": _json_cell(generator.get("params", {})),
            "model_params_json": _json_cell(model.get("params", {})),
            "capabilities_json": _json_cell(model.get("capabilities", {})),
            "n_metrics": len(metrics) if isinstance(metrics, list) else 0,
            "n_metric_ok": sum(
                metric.get("status") == "ok"
                for metric in metrics
                if isinstance(metric, Mapping)
            ),
            "n_metric_not_applicable": sum(
                metric.get("status") == "not_applicable"
                for metric in metrics
                if isinstance(metric, Mapping)
            ),
            "n_metric_failed": sum(
                metric.get("status") == "failed"
                for metric in metrics
                if isinstance(metric, Mapping)
            ),
            "n_metric_skipped": sum(
                metric.get("status") == "skipped"
                for metric in metrics
                if isinstance(metric, Mapping)
            ),
            "error_type": error.get("type") if isinstance(error, Mapping) else None,
            "error_message": (
                error.get("message") if isinstance(error, Mapping) else None
            ),
            "diagnostics_json": _json_cell(record.get("diagnostics", {})),
            "provenance_json": _json_cell(record.get("provenance", {})),
        }
        fit = record.get("fit", {})
        if isinstance(fit, Mapping):
            for key, value in sorted(fit.items()):
                row[f"fit.{key}"] = (
                    value
                    if value is None or isinstance(value, (str, int, float, bool))
                    else _json_cell(value)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def tidy_metrics(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return one row per metric, including explicit unavailable/failed rows."""

    columns = [
        "case_id",
        "case_status",
        "generator_id",
        "generator_kind",
        "generator_seed",
        "data_seed",
        "model_name",
        "model_kind",
        "model_seed",
        "metric_id",
        "value",
        "metric_status",
        "estimand",
        "noise_sigma_standardized",
        "centering",
        "unit",
        "reason",
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        generator = record.get("generator", {})
        model = record.get("model", {})
        for metric in record.get("metrics", []):
            rows.append(
                {
                    "case_id": record.get("case_id"),
                    "case_status": record.get("status"),
                    "generator_id": generator.get("id"),
                    "generator_kind": generator.get("kind"),
                    "generator_seed": generator.get("seed"),
                    "data_seed": record.get("data_seed"),
                    "model_name": model.get("name"),
                    "model_kind": model.get("kind"),
                    "model_seed": model.get("seed"),
                    "metric_id": metric.get("metric_id"),
                    "value": metric.get("value"),
                    "metric_status": metric.get("status"),
                    "estimand": metric.get("estimand"),
                    "noise_sigma_standardized": metric.get(
                        "noise_sigma_standardized"
                    ),
                    "centering": metric.get("centering"),
                    "unit": metric.get("unit"),
                    "reason": metric.get("reason"),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _display(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, limit: int = 80) -> str:
    available = [column for column in columns if column in frame.columns]
    if frame.empty or not available:
        return "_None in this run._"
    shown = frame.loc[:, available].head(limit)
    header = "| " + " | ".join(available) + " |"
    rule = "| " + " | ".join("---" for _ in available) + " |"
    body = [
        "| " + " | ".join(_display(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    if len(frame) > limit:
        body.append(f"\n_{len(frame) - limit} additional rows are retained in the CSV export._")
    return "\n".join([header, rule, *body])


def _metric_table(metrics: pd.DataFrame, metric_ids: frozenset[str]) -> str:
    selected = metrics[metrics["metric_id"].isin(metric_ids)].copy()
    selected = selected.sort_values(
        [
            "generator_id",
            "generator_seed",
            "model_name",
            "model_seed",
            "metric_id",
            "estimand",
            "noise_sigma_standardized",
            "centering",
        ],
        na_position="first",
    )
    return _markdown_table(
        selected,
        [
            "generator_id",
            "generator_seed",
            "model_name",
            "model_seed",
            "metric_id",
            "estimand",
            "noise_sigma_standardized",
            "centering",
            "value",
            "unit",
            "metric_status",
        ],
    )


def write_developmental_smoke_report(
    output_dir: str | Path,
    records: Sequence[Mapping[str, Any]],
    metrics: pd.DataFrame | None = None,
    cases: pd.DataFrame | None = None,
) -> Path:
    """Write a concise, non-confirmatory report without cross-estimand pooling."""

    destination = Path(output_dir)
    metrics = tidy_metrics(records) if metrics is None else metrics
    cases = cases_frame(records) if cases is None else cases
    failed_cases = cases[cases["status"] == "failed"]
    unavailable = metrics[metrics["metric_status"] != "ok"].sort_values(
        ["generator_id", "model_name", "metric_id", "estimand", "centering"]
    )

    fit_cost_columns = [
        column
        for column in cases.columns
        if column.startswith("fit.")
        and any(
            token in column.lower()
            for token in ("wall", "second", "memory", "rss", "parameter", "epoch", "step")
        )
    ]
    case_cost = _markdown_table(
        cases,
        [
            "generator_id",
            "generator_seed",
            "model_name",
            "model_seed",
            *fit_cost_columns,
        ],
    )
    metric_cost = _metric_table(metrics, COST_METRICS)

    report = f"""# Developmental smoke report

> **H1-H7: not evaluated.** This run is a developmental correctness and cost gate. It is not confirmatory evidence for or against any preregistered hypothesis.

## Scope and status

- Atomic case records: {len(records)}
- Successful cases: {int((cases['status'] == 'ok').sum())}
- Failed cases: {int((cases['status'] == 'failed').sum())}
- Explicit unavailable or failed metric rows: {len(unavailable)}

Every result below remains in its original `estimand`, `noise_sigma_standardized`, and `centering` stratum. Clean and noisy laws are never pooled. Oracle-sample and model-sample centering are never pooled.

## Native validation loss

Native losses select checkpoints within a model family; unlike objectives are not a common leaderboard.

{_metric_table(metrics, NATIVE_METRICS)}

## Conditional-law scores

Energy scores are shown only where the model supplies samples. NLL is shown only where a normalized density is available.

{_metric_table(metrics, LAW_METRICS)}

## History-tangent and finite-change diagnostics

{_metric_table(metrics, TANGENT_METRICS)}

## Computational cost

Case-level fit metadata:

{case_cost}

Metric-level evaluation throughput:

{metric_cost}

## Failures and unavailable capabilities

Failed cases:

{_markdown_table(failed_cases, ['case_id', 'generator_id', 'model_name', 'error_type', 'error_message'])}

Unavailable or failed metric rows:

{_markdown_table(unavailable, ['case_id', 'generator_id', 'model_name', 'metric_id', 'estimand', 'noise_sigma_standardized', 'centering', 'metric_status', 'reason'])}

## Hypothesis decisions

**H1-H7: not evaluated.** A repaired and newly frozen confirmatory matrix is required before hypothesis decisions.
"""
    path = destination / "developmental_smoke_report.md"
    _atomic_text(path, report)
    return path


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def summarize_output(output_dir: str | Path) -> dict[str, Any]:
    """Validate records and export ``metrics.csv``, ``cases.csv``, and the report."""

    output = Path(output_dir)
    validation_path = output / "validation.json"
    try:
        records, paths = _load_records_and_paths(output)
    except ValueError as error:
        validation = {
            "n_records": 0,
            "n_ok": 0,
            "n_failed": 0,
            "n_metric_ok": 0,
            "n_metric_not_applicable": 0,
            "n_metric_failed": 0,
            "n_metric_skipped": 0,
            "valid": False,
            "problems": [str(error)],
        }
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(validation_path, validation)
        raise ValueError(f"case-record validation failed: {error}") from error

    validation = validate_case_records(
        records, paths=paths, results_dir=output / "results"
    )
    _atomic_json(validation_path, validation)
    if not validation["valid"]:
        preview = "; ".join(validation["problems"][:5])
        raise ValueError(f"case-record validation failed: {preview}")

    metrics = tidy_metrics(records)
    cases = cases_frame(records)
    metrics_path = output / "metrics.csv"
    cases_path = output / "cases.csv"
    _atomic_csv(metrics, metrics_path)
    _atomic_csv(cases, cases_path)
    report_path = write_developmental_smoke_report(output, records, metrics, cases)
    return {
        "output_dir": str(output.resolve()),
        "validation": validation,
        "metrics_csv": str(metrics_path.resolve()),
        "cases_csv": str(cases_path.resolve()),
        "report": str(report_path.resolve()),
    }


# Small compatibility aliases for callers used to the sibling benchmark package.
load_records = load_case_records
validate_records = validate_case_records
write_report = summarize_output


__all__ = [
    "cases_frame",
    "load_case_records",
    "load_records",
    "summarize_output",
    "tidy_metrics",
    "validate_case_records",
    "validate_records",
    "write_developmental_smoke_report",
    "write_report",
]
