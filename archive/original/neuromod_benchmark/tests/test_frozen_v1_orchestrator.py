from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from frozen_v1_lib import (  # noqa: E402
    FrozenV1Error,
    SuitePlan,
    plan_config,
    verify_suite_completion,
)


def test_main_plan_counts_cases_candidates_and_nested_refits(tmp_path: Path) -> None:
    config = tmp_path / "counted.yaml"
    config.write_text(
        """
schema_version: "1"
name: counted
output_dir: outputs/frozen_v1/test_counted
history_lags: [1]
horizons: [1, 2]
views: [latent, calcium]
seeds: [3, 4]
validation_fraction: 0.2
test_fraction: 0.2
resource:
  max_threads: 2
  max_parallel_jobs: 1
  max_rss_gb: 2.0
  max_output_gb: 0.1
  nice: 19
  stop_free_disk_gb: 4.0
scenarios:
  - id: "null"
    family: mechanistic
    n_trajectories: 3
    params: {n_neurons: 3, n_modulators: 1, n_steps: 40, burn_in: 10, mechanism: "null"}
methods:
  - {name: constant_gaussian}
  - name: ridge_var
    seeds: [3]
    views: [latent]
    horizons: [1]
    tune: {ridge: [0.1, 1.0, 10.0]}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plan = plan_config(config)
    assert plan.kind == "main"
    assert plan.case_records == 9
    assert plan.tuning_candidate_evaluations == 11
    # Eight single constant fits plus three ridge candidates and one selected refit.
    assert plan.fit_invocations == 12
    assert plan.dgp_constructions == 2


def test_duplicate_aliases_are_allowed_only_on_disjoint_contexts(tmp_path: Path) -> None:
    base = """
schema_version: "1"
name: duplicate_views
output_dir: outputs/frozen_v1/test_duplicate_views
history_lags: [1]
horizons: [1]
views: [latent, calcium]
seeds: [3]
resource: {max_threads: 1, max_parallel_jobs: 1, max_rss_gb: 2.0, max_output_gb: 0.1, nice: 19, stop_free_disk_gb: 4.0}
scenarios:
  - id: "null"
    family: mechanistic
    n_trajectories: 3
    params: {n_neurons: 3, n_modulators: 1, n_steps: 40, burn_in: 10, mechanism: "null"}
methods:
  - {name: ridge_var, views: [latent], params: {ridge: 0.1}}
  - {name: ridge_var, views: [%s], params: {ridge: 10.0}}
"""
    disjoint = tmp_path / "disjoint.yaml"
    disjoint.write_text(base % "calcium", encoding="utf-8")
    plan = plan_config(disjoint)
    assert plan.case_records == 2
    assert [item["spec_id"] for item in plan.method_grids] == [
        "ridge_var__spec1",
        "ridge_var__spec2",
    ]

    overlapping = tmp_path / "overlap.yaml"
    overlapping.write_text(base % "latent", encoding="utf-8")
    with pytest.raises(FrozenV1Error, match="overlaps one execution context"):
        plan_config(overlapping)


def test_preflight_rejects_tuning_a_fixed_corruption_stratum(tmp_path: Path) -> None:
    config = tmp_path / "corruption_tune.yaml"
    config.write_text(
        """
schema_version: "1"
name: corruption_tune
output_dir: outputs/frozen_v1/test_corruption_tune
history_lags: [1]
horizons: [1]
views: [latent]
seeds: [3]
resource: {max_threads: 1, max_parallel_jobs: 1, max_rss_gb: 2.0, max_output_gb: 0.1, nice: 19, stop_free_disk_gb: 4.0}
scenarios:
  - id: "null"
    family: mechanistic
    n_trajectories: 3
    params: {n_neurons: 3, n_modulators: 1, n_steps: 40, burn_in: 10, mechanism: "null"}
methods:
  - name: sid_dsm
    tune: {sigma_fraction: [0.25, 0.5]}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FrozenV1Error, match="fixed aliases"):
        plan_config(config)


def test_changepoint_plan_counts_disjoint_calibration_and_evaluation(tmp_path: Path) -> None:
    config = tmp_path / "changepoint.json"
    config.write_text(
        json.dumps(
            {
                "name": "small_changepoint",
                "mechanistic": {
                    "n_neurons": 3,
                    "n_modulators": 1,
                    "n_steps": 80,
                    "burn_in": 20,
                    "mechanism": "null",
                },
                "benchmark": {
                    "methods": ["mean_cusum", "energy_distance"],
                    "n_calibration": 9,
                    "n_null_test": 2,
                    "n_changed": 3,
                    "alpha": 0.1,
                    "boundary_fraction": 0.5,
                    "tolerance": 5,
                    "min_segment": 10,
                    "stride": 2,
                    "scan_window": 12,
                    "post_change_strengths": {
                        "mean": {"mean_multiplier": 1.0},
                        "dispersion": {"logvariance_multiplier": 2.0},
                        "matched_tail": {"tail_logit_multiplier": 1.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    plan = plan_config(config)
    # 9 calibration + 2 null evaluation + 3 mechanisms * 3 changed events.
    assert plan.case_records == 20
    assert plan.expected_result_records == 22
    assert plan.detector_scans == 40
    assert plan.fit_invocations == 0


def test_completion_rejects_null_primary_score_even_when_record_says_ok(
    tmp_path: Path,
) -> None:
    output = tmp_path / "memory"
    results = output / "results"
    results.mkdir(parents=True)
    plan = SuitePlan(
        identifier="memory",
        kind="memory",
        name="memory",
        config_path=str(tmp_path / "memory.yaml"),
        config_sha256="config",
        suite_config_digest="suite",
        output_dir=str(output),
        output_file=None,
        resource={},
        case_records=1,
        expected_result_records=1,
        fit_invocations=1,
        tuning_candidate_evaluations=1,
        dgp_constructions=1,
        detector_scans=0,
        method_grids=(
            {
                "name": "ridge_var",
                "predictive_distribution": "normalized",
            },
        ),
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "suite_config_digest": "suite",
                "environment": {"benchmark_source_digest": "source"},
            }
        ),
        encoding="utf-8",
    )
    result_path = results / "case.json"
    result_path.write_text(
        json.dumps(
            {
                "case_id": "case",
                "status": "ok",
                "method": "ridge_var",
                "metrics": {"predictive.nll": None},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FrozenV1Error, match="primary score"):
        verify_suite_completion(plan, source_digest="source", require_report=False)

    result_path.write_text(
        json.dumps(
            {
                "case_id": "case",
                "status": "ok",
                "method": "ridge_var",
                "metrics": {"predictive.nll": 1.25},
            }
        ),
        encoding="utf-8",
    )
    verification = verify_suite_completion(
        plan, source_digest="source", require_report=False
    )
    assert verification["result_records"] == 1


def test_main_score_model_requires_finite_fixed_reference_ladder_mean(
    tmp_path: Path,
) -> None:
    output = tmp_path / "score"
    results = output / "results"
    results.mkdir(parents=True)
    plan = SuitePlan(
        identifier="score",
        kind="main",
        name="score",
        config_path=str(tmp_path / "score.yaml"),
        config_sha256="config",
        suite_config_digest="suite",
        output_dir=str(output),
        output_file=None,
        resource={},
        case_records=1,
        expected_result_records=1,
        fit_invocations=1,
        tuning_candidate_evaluations=1,
        dgp_constructions=1,
        detector_scans=0,
        method_grids=(
            {
                "name": "score_mlp_gaussian_s025",
                "predictive_distribution": "unnormalized_score",
            },
        ),
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "suite_config_digest": "suite",
                "environment": {"benchmark_source_digest": "source"},
            }
        ),
        encoding="utf-8",
    )
    path = results / "case.json"
    base = {
        "case_id": "case",
        "status": "ok",
        "case": {"method": {"name": "score_mlp_gaussian_s025"}},
    }
    path.write_text(json.dumps({**base, "metrics": {}}), encoding="utf-8")
    with pytest.raises(FrozenV1Error, match="exactly one fixed-reference ladder"):
        verify_suite_completion(plan, source_digest="source", require_report=False)
    path.write_text(
        json.dumps(
            {
                **base,
                "metrics": {
                    "score.test.fixed_reference.gaussian.ladder_mean": 0.75
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FrozenV1Error, match="exactly one fixed-reference ladder"):
        verify_suite_completion(plan, source_digest="source", require_report=False)
    path.write_text(
        json.dumps(
            {
                **base,
                "metrics": {
                    "score.test.conditional_outcome.fixed_reference.gaussian.ladder_mean": 0.75
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        verify_suite_completion(plan, source_digest="source", require_report=False)[
            "result_records"
        ]
        == 1
    )


def test_official_causal_worker_version_must_match_frozen_environment(
    tmp_path: Path,
) -> None:
    output = tmp_path / "causal"
    results = output / "results"
    results.mkdir(parents=True)
    plan = SuitePlan(
        identifier="causal",
        kind="robustness",
        name="causal",
        config_path=str(tmp_path / "causal.yaml"),
        config_sha256="config",
        suite_config_digest="suite",
        output_dir=str(output),
        output_file=None,
        resource={},
        case_records=1,
        expected_result_records=1,
        fit_invocations=1,
        tuning_candidate_evaluations=1,
        dgp_constructions=1,
        detector_scans=0,
        method_grids=(
            {
                "name": "pcmci_parcorr",
                "predictive_distribution": "none",
            },
        ),
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "suite_config_digest": "suite",
                "environment": {
                    "benchmark_source_digest": "source",
                    "sid_neuromod_source_digest": "sid",
                    "execution_environment_digest": "environment",
                },
            }
        ),
        encoding="utf-8",
    )
    result_path = results / "case.json"
    record = {
        "case_id": "case",
        "status": "ok",
        "method": "pcmci_parcorr",
        "metrics": {},
        "method_metadata": {"package": "tigramite", "version": "5.2.10.1"},
    }
    result_path.write_text(json.dumps(record), encoding="utf-8")
    kwargs = {
        "source_digest": "source",
        "require_report": False,
        "sid_neuromod_source_digest": "sid",
        "execution_environment_digest": "environment",
        "causal_environment_packages": {"tigramite": "5.2.10.1"},
    }
    assert verify_suite_completion(plan, **kwargs)["result_records"] == 1
    record["method_metadata"]["version"] = "unknown"
    result_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FrozenV1Error, match="does not match frozen environment"):
        verify_suite_completion(plan, **kwargs)
