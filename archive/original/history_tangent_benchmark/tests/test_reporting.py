from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from history_tangent_benchmark.cli import build_parser, main
from history_tangent_benchmark.reporting import (
    summarize_output,
    tidy_metrics,
    validate_case_records,
)


def _metric(
    metric_id: str,
    value: float | None,
    *,
    status: str = "ok",
    estimand: str = "clean",
    sigma: float | None = None,
    centering: str = "none",
    reason: str | None = None,
    unit: str = "dimensionless",
) -> dict:
    result = {
        "metric_id": metric_id,
        "value": value,
        "status": status,
        "estimand": estimand,
        "centering": centering,
        "unit": unit,
    }
    if sigma is not None:
        result["noise_sigma_standardized"] = sigma
    if reason is not None:
        result["reason"] = reason
    return result


def _record(case_id: str = "case-1", *, model: str = "normalized") -> dict:
    return {
        "schema_version": "1",
        "case_id": case_id,
        "status": "ok",
        "generator": {"id": "g1", "kind": "location_gaussian", "params": {}, "seed": 3},
        "data_seed": 11,
        "model": {
            "name": model,
            "kind": "gaussian",
            "params": {},
            "seed": 7,
            "capabilities": {"normalized_density": True, "sampler": True},
        },
        "fit": {
            "native_objective": "nll",
            "training_wall_seconds": 1.25,
            "parameter_count": 128,
        },
        "metrics": [
            _metric("native_validation_loss", 1.2, unit="nats_per_example"),
            _metric("energy_score_fair", 0.4, unit="response_norm"),
            _metric(
                "tangent_nrmse",
                0.3,
                centering="oracle_samples",
            ),
            _metric(
                "tangent_nrmse",
                0.5,
                estimand="noisy",
                sigma=0.1,
                centering="model_samples",
            ),
            _metric(
                "nll_original",
                None,
                status="not_applicable",
                reason="this fixture demonstrates explicit N/A retention",
                unit="nats_per_example",
            ),
        ],
        "diagnostics": {"gradient_check": "ok"},
        "provenance": {"config_sha256": "abc"},
    }


def _write_record(output: Path, record: dict, *, filename: str | None = None) -> None:
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    name = filename or f"{record['case_id']}.json"
    (results / name).write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_summarize_exports_explicit_strata_and_developmental_warning(tmp_path: Path):
    record = _record()
    _write_record(tmp_path, record)

    result = summarize_output(tmp_path)

    assert result["validation"]["valid"] is True
    assert Path(result["metrics_csv"]).is_file()
    assert Path(result["cases_csv"]).is_file()
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert report.count("H1-H7: not evaluated") >= 2
    assert "Clean and noisy laws are never pooled" in report
    assert "Oracle-sample and model-sample centering are never pooled" in report

    metrics = pd.read_csv(result["metrics_csv"])
    tangent = metrics[metrics.metric_id == "tangent_nrmse"]
    assert len(tangent) == 2
    assert set(tangent.estimand) == {"clean", "noisy"}
    assert set(tangent.centering) == {"oracle_samples", "model_samples"}
    unavailable = metrics[metrics.metric_status == "not_applicable"]
    assert len(unavailable) == 1
    assert "explicit N/A" in unavailable.reason.iloc[0]

    cases = pd.read_csv(result["cases_csv"])
    assert cases.loc[0, "generator_id"] == "g1"
    assert cases.loc[0, "fit.parameter_count"] == 128


def test_validate_rejects_missing_values_nonfinite_and_implicit_na():
    missing_value = _record("missing-value")
    del missing_value["metrics"][0]["value"]
    implicit_na = _record("implicit-na")
    implicit_na["metrics"][-1].pop("reason")
    nonfinite = _record("nonfinite")
    nonfinite["metrics"][0]["value"] = float("inf")

    validation = validate_case_records([missing_value, implicit_na, nonfinite])

    assert validation["valid"] is False
    joined = "\n".join(validation["problems"])
    assert "value must be finite" in joined
    assert "reason is required" in joined
    assert "non-finite numeric value" in joined


def test_summarize_rejects_filename_mismatch_and_writes_validation(tmp_path: Path):
    _write_record(tmp_path, _record(), filename="wrong-name.json")

    with pytest.raises(ValueError, match="filename"):
        summarize_output(tmp_path)

    validation = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert validation["valid"] is False
    assert any("filename" in problem for problem in validation["problems"])
    assert not (tmp_path / "metrics.csv").exists()


def test_tidy_metrics_keeps_oracle_and_model_centering_as_distinct_rows():
    frame = tidy_metrics([_record()])
    tangent = frame[frame.metric_id == "tangent_nrmse"]
    identity = set(
        zip(
            tangent.estimand,
            tangent.noise_sigma_standardized.fillna(-1.0),
            tangent.centering,
        )
    )
    assert identity == {
        ("clean", -1.0, "oracle_samples"),
        ("noisy", 0.1, "model_samples"),
    }


def test_validator_retains_native_sampler_and_skipped_strata():
    record = _record()
    record["metrics"].extend(
        [
            _metric(
                "native_validation_loss",
                0.2,
                estimand="native_objective",
            ),
            _metric(
                "energy_score_fair",
                0.8,
                estimand="approximate_clean_sampler",
            ),
            _metric(
                "finite_ratio_nrmse",
                None,
                status="skipped",
                estimand="noisy",
                sigma=0.1,
                centering="oracle_samples",
                reason="developmental route is not implemented",
            ),
        ]
    )

    validation = validate_case_records([record])

    assert validation["valid"] is True
    assert validation["n_metric_skipped"] == 1


def test_cli_exposes_required_subcommands_and_summarize(tmp_path: Path, capsys):
    parser = build_parser()
    assert parser.parse_args(["summarize", "--output", str(tmp_path)]).command == "summarize"
    assert parser.parse_args(["validate-oracles"]).command == "validate-oracles"
    assert parser.parse_args(["run", "--config", "config.yaml"]).command == "run"
    assert parser.parse_args(["finite-contrast", "--config", "v1.yaml"]).command == "finite-contrast"

    _write_record(tmp_path, _record())
    assert main(["summarize", "--output", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["validation"]["valid"] is True
