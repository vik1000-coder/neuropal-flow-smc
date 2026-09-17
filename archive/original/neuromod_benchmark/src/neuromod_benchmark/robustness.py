"""Held-out-worm and paired observation-process robustness benchmark.

Training, validation, and test units are distinct simulated worms.  Observation
variants share the exact latent realization within each worm, so degradation under
measurement noise or calcium filtering cannot be mistaken for a biological change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .config import MethodSpec, ScenarioSpec
from .evaluation import effect_metrics, latent_recovery_metrics, oracle_functional_metrics
from .features import build_supervised, subset
from .hierarchy import (
    DEFAULT_OBSERVATION_VARIANTS,
    HierarchyConfig,
    ObservationVariant,
    generate_hierarchical_dataset,
)
from .mechanistic import MechanisticConfig
from .mechanistic_dataset import _support, _trajectory
from .metrics import pit_serial_metrics, predictive_metrics
from .resources import enforce_storage_budget, limited_threads
from .schema import Dataset, ResourceBudget
from .serialization import atomic_json, environment_manifest, stable_hash
from .tuning import tune_method


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class RobustnessSuite:
    name: str
    scenarios: tuple[ScenarioSpec, ...]
    methods: tuple[MethodSpec, ...]
    seeds: tuple[int, ...]
    coefficient_cvs: tuple[float, ...]
    views: tuple[str, ...]
    history_lags: tuple[int, ...]
    horizon: int
    hierarchy_counts: Mapping[str, int]
    observation_variants: tuple[ObservationVariant, ...]
    output_dir: str
    resource: ResourceBudget
    n_predictive_samples: int = 16
    schema_version: str = "1"

    def validate(self) -> None:
        if self.schema_version != "1":
            raise ValueError(
                f"unsupported robustness suite schema {self.schema_version!r}"
            )
        if not self.scenarios or not self.methods or not self.seeds:
            raise ValueError("robustness suite requires scenarios, methods, and seeds")
        if not self.coefficient_cvs or min(self.coefficient_cvs) < 0:
            raise ValueError("coefficient CVs must be non-negative")
        if set(self.views) - {"complete_state", "latent", "calcium"}:
            raise ValueError("unknown robustness view")
        if self.horizon != 1:
            raise ValueError("hierarchical mechanism robustness currently requires horizon=1")
        if min(self.history_lags) < 1:
            raise ValueError("history lags must be positive")
        required = {"train", "validation", "test"}
        if set(self.hierarchy_counts) != required:
            raise ValueError("hierarchy_counts must contain train/validation/test")
        if min(self.hierarchy_counts.values()) < 1:
            raise ValueError("every worm split must be non-empty")
        self.resource.validate()


def load_robustness_suite(path: str | Path) -> RobustnessSuite:
    raw = yaml.safe_load(Path(path).read_text())
    scenarios = tuple(
        ScenarioSpec(
            id=item["id"],
            family="mechanistic",
            params=item.get("params", {}),
            n_trajectories=1,
        )
        for item in raw["scenarios"]
    )
    methods = tuple(
        MethodSpec(
            name=item["name"],
            params=item.get("params", {}),
            tune=item.get("tune", {}),
            views=None if item.get("views") is None else tuple(item["views"]),
            horizons=(1,),
            scenarios=(
                None
                if item.get("scenarios") is None
                else tuple(item["scenarios"])
            ),
            seeds=(
                None
                if item.get("seeds") is None
                else tuple(int(seed) for seed in item["seeds"])
            ),
        )
        for item in raw["methods"]
    )
    variants = tuple(
        ObservationVariant(
            name=item["name"],
            measurement_noise_multiplier=float(item.get("measurement_noise_multiplier", 1.0)),
            calcium_decay_multiplier=float(item.get("calcium_decay_multiplier", 1.0)),
        )
        for item in raw.get("observation_variants", [asdict(v) for v in DEFAULT_OBSERVATION_VARIANTS])
    )
    suite = RobustnessSuite(
        name=str(raw["name"]),
        scenarios=scenarios,
        methods=methods,
        seeds=tuple(int(value) for value in raw.get("seeds", [0])),
        coefficient_cvs=tuple(float(value) for value in raw.get("coefficient_cvs", [0.0, .2])),
        views=tuple(raw.get("views", ["calcium"])),
        history_lags=tuple(int(value) for value in raw.get("history_lags", [1])),
        horizon=int(raw.get("horizon", 1)),
        hierarchy_counts={
            key: int(value)
            for key, value in raw.get(
                "hierarchy_counts", {"train": 6, "validation": 2, "test": 3}
            ).items()
        },
        observation_variants=variants,
        output_dir=str(raw.get("output_dir", f"outputs/{raw['name']}")),
        resource=ResourceBudget(**raw.get("resource", {})),
        n_predictive_samples=int(raw.get("n_predictive_samples", 16)),
        schema_version=str(raw.get("schema_version", "1")),
    )
    suite.validate()
    return suite


def _as_dataset(hierarchical, variant: str) -> Dataset:
    trajectories = [
        _trajectory(worm.observations[variant].trajectory) for worm in hierarchical.worms
    ]
    return Dataset(
        config=hierarchical.population_config,
        parameters=hierarchical.population_parameters,
        trajectories=trajectories,
        no_modulation=None,
        mechanism_support=_support(
            hierarchical.population_parameters,
            trajectories,
            hierarchical.population_config.mechanism,
        ),
        metadata={
            **dict(hierarchical.metadata),
            "observation_variant": variant,
            "worm_ids": [worm.spec.worm_id for worm in hierarchical.worms],
            "worm_splits": [worm.spec.split for worm in hierarchical.worms],
            "oracle_caveat": "physical parameters vary by worm; population parameters are not per-worm truth",
        },
    )


def _split_data(dataset: Dataset, hierarchical, view: str, lags: tuple[int, ...]):
    data = build_supervised(dataset, view=view, history_lags=lags, horizon=1)
    group_split = {
        group: hierarchical.worms[int(group)].spec.split for group in np.unique(data.groups)
    }
    return {
        split: subset(
            data,
            np.asarray([group_split[int(group)] == split for group in data.groups]),
        )
        for split in ("train", "validation", "test")
    }


def _evaluate_robustness(
    estimator,
    dataset: Dataset,
    validation,
    test,
    n_samples: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    capability = estimator.capabilities.predictive_distribution
    if capability == "normalized":
        prediction = estimator.predict(test, n_samples=n_samples)
        metrics.update({f"predictive.{key}": value for key, value in predictive_metrics(prediction, test.targets).items()})
        if prediction.cdf is not None:
            metrics.update({f"predictive.{key}": value for key, value in pit_serial_metrics(prediction.cdf, test.groups).items()})
        if test.view == "complete_state":
            metrics.update(oracle_functional_metrics(prediction, test))
    elif capability == "unnormalized_score":
        score_domain = getattr(estimator, "score_domain", None)
        if score_domain not in {"conditional_outcome", "joint_consecutive_state"}:
            raise TypeError("score estimator lacks a registered score domain")
        if not hasattr(estimator, "dsm_reference_metrics"):
            raise TypeError("score estimator lacks fixed-reference DSM metrics")
        metrics.update(
            {
                f"score.test.{score_domain}.{key}": float(value)
                for key, value in estimator.dsm_reference_metrics(
                    test, seed=71_117
                ).items()
            }
        )
    metrics.update(effect_metrics(estimator, dataset, test))
    metrics.update(
        latent_recovery_metrics(
            estimator,
            dataset,
            validation,
            test,
            include_parameter_recovery=False,
        )
    )
    return metrics


def _add_transfer_deltas(records: list[dict[str, Any]]) -> None:
    reference: dict[tuple[Any, ...], dict[str, float]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        key = tuple(record[name] for name in ("scenario", "seed", "coefficient_cv", "view", "method"))
        if record["observation_variant"] == "reference":
            reference[key] = record["metrics"]
    for record in records:
        if record.get("status") != "ok":
            continue
        key = tuple(record[name] for name in ("scenario", "seed", "coefficient_cv", "view", "method"))
        baseline = reference.get(key)
        if baseline is None:
            continue
        for metric in (
            "predictive.nll",
            "predictive.rmse_mean",
            "predictive.energy_score",
        ):
            if metric in record["metrics"] and metric in baseline:
                record["metrics"][f"transfer.delta_{metric}"] = float(
                    record["metrics"][metric] - baseline[metric]
                )


def run_robustness_suite(path_or_suite: str | Path | RobustnessSuite) -> dict[str, Any]:
    suite = (
        path_or_suite
        if isinstance(path_or_suite, RobustnessSuite)
        else load_robustness_suite(path_or_suite)
    )
    suite.validate()
    output = Path(suite.output_dir)
    if not output.is_absolute():
        output = PACKAGE_ROOT / output
    output = output.resolve()
    if PACKAGE_ROOT.resolve() not in output.parents:
        raise ValueError("robustness output must remain inside benchmark root")
    enforce_storage_budget(output, suite.resource)
    output.mkdir(parents=True, exist_ok=True)
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()
    frozen_config = asdict(suite)
    suite_digest = stable_hash(frozen_config)
    environment = environment_manifest(WORKSPACE_ROOT)
    manifest_path = output / "run_manifest.json"
    existing_results = list(results_dir.glob("*.json"))
    if manifest_path.exists() and existing_results:
        previous = json.loads(manifest_path.read_text())
        previous_digest = previous.get("suite_config_digest")
        previous_source = previous.get("environment", {}).get(
            "benchmark_source_digest"
        )
        previous_execution_environment = previous.get("environment", {}).get(
            "execution_environment_digest"
        )
        if previous_digest != suite_digest:
            raise RuntimeError(
                "robustness output contains a different frozen suite configuration; "
                "choose a new output_dir"
            )
        if previous_source != environment.get("benchmark_source_digest"):
            raise RuntimeError(
                "robustness output contains a different benchmark source digest; "
                "choose a new output_dir"
            )
        if previous_execution_environment != environment.get(
            "execution_environment_digest"
        ):
            raise RuntimeError(
                "robustness output contains a different execution environment digest; "
                "choose a new output_dir"
            )
    manifest = {
        "suite": suite.name,
        "suite_config_digest": suite_digest,
        "status": "running",
        "records": 0,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "started_unix": started,
        "config": frozen_config,
        "environment": environment,
    }
    atomic_json(manifest_path, manifest)

    def checkpoint(record: dict[str, Any]) -> None:
        records.append(record)
        atomic_json(results_dir / f"{record['case_id']}.json", record)
        manifest["records"] = len(records)
        manifest["completed"] = sum(item.get("status") == "ok" for item in records)
        manifest["failed"] = sum(item.get("status") != "ok" for item in records)
        manifest["last_case_id"] = record["case_id"]
        manifest["last_update_unix"] = time.time()
        atomic_json(manifest_path, manifest)
    for scenario in suite.scenarios:
        for seed in suite.seeds:
            population = MechanisticConfig(**{**dict(scenario.params), "seed": seed})
            for cv in suite.coefficient_cvs:
                hierarchy = HierarchyConfig(
                    n_train_worms=suite.hierarchy_counts["train"],
                    n_validation_worms=suite.hierarchy_counts["validation"],
                    n_test_worms=suite.hierarchy_counts["test"],
                    recurrent_weight_cv=cv,
                    clearance_time_cv=cv,
                    receptor_expression_cv=cv,
                    receptor_kd_cv=cv,
                    calcium_decay_cv=cv,
                    seed=91_337 + 10_007 * seed + int(round(10_000 * cv)),
                    observation_variants=suite.observation_variants,
                )
                hierarchical = generate_hierarchical_dataset(population, hierarchy)
                datasets = {
                    variant.name: _as_dataset(hierarchical, variant.name)
                    for variant in suite.observation_variants
                }
                for view in suite.views:
                    reference_parts = _split_data(
                        datasets["reference"], hierarchical, view, suite.history_lags
                    )
                    test_parts = {
                        name: _split_data(dataset, hierarchical, view, suite.history_lags)["test"]
                        for name, dataset in datasets.items()
                    }
                    for method in suite.methods:
                        if (
                            method.scenarios is not None
                            and scenario.id not in method.scenarios
                        ):
                            continue
                        if method.seeds is not None and seed not in method.seeds:
                            continue
                        if method.views is not None and view not in method.views:
                            continue
                        identities = {}
                        pending = []
                        for variant, test in test_parts.items():
                            case_payload = {
                                "suite_config_digest": suite_digest,
                                "scenario": asdict(scenario),
                                "seed": seed,
                                "coefficient_cv": cv,
                                "view": view,
                                "method": asdict(method),
                                "observation_variant": variant,
                            }
                            case_id = stable_hash(case_payload)
                            case_path = results_dir / f"{case_id}.json"
                            identities[variant] = (case_id, case_path)
                            if case_path.exists():
                                records.append(json.loads(case_path.read_text()))
                                manifest["skipped"] += 1
                            else:
                                pending.append((variant, test))
                        if not pending:
                            continue
                        fit_started = time.perf_counter()
                        try:
                            with limited_threads(suite.resource):
                                tuned = tune_method(
                                    method,
                                    reference_parts["train"],
                                    reference_parts["validation"],
                                    seed=seed,
                                )
                            fit_error = None
                        except Exception as error:
                            tuned = None
                            fit_error = (error, traceback.format_exc(limit=12))
                        fit_seconds = time.perf_counter() - fit_started
                        for variant, test in pending:
                            case_id, _ = identities[variant]
                            base = {
                                "case_id": case_id,
                                "scenario": scenario.id,
                                "scenario_spec": asdict(scenario),
                                "seed": seed,
                                "coefficient_cv": cv,
                                "view": view,
                                "method": method.name,
                                "method_spec": asdict(method),
                                "observation_variant": variant,
                                "declared_cvs": dict(hierarchy.declared_coefficient_cvs),
                                "realized_cvs": dict(hierarchical.metadata["realized_coefficient_cvs"]),
                                "latent_hash_paired": bool(
                                    hierarchical.metadata["observation_variants_latent_hash_invariant"]
                                ),
                                "wall_seconds_fit_shared": fit_seconds,
                            }
                            if fit_error is not None:
                                error, trace = fit_error
                                checkpoint(
                                    {
                                        **base,
                                        "status": "failed",
                                        "error_type": type(error).__name__,
                                        "error": str(error),
                                        "traceback": trace,
                                    }
                                )
                                continue
                            try:
                                with limited_threads(suite.resource):
                                    metrics = _evaluate_robustness(
                                        tuned.estimator,
                                        datasets[variant],
                                        reference_parts["validation"],
                                        test,
                                        suite.n_predictive_samples,
                                    )
                                checkpoint(
                                    {
                                        **base,
                                        "status": "ok",
                                        "selected_params": tuned.params,
                                        "selection_objective": tuned.objective,
                                        "selection_score": tuned.score,
                                        "method_metadata": tuned.estimator.metadata(),
                                        "metrics": metrics,
                                    }
                                )
                            except Exception as error:
                                checkpoint(
                                    {
                                        **base,
                                        "status": "failed",
                                        "error_type": type(error).__name__,
                                        "error": str(error),
                                        "traceback": traceback.format_exc(limit=12),
                                    }
                                )
                        enforce_storage_budget(output, suite.resource)
    _add_transfer_deltas(records)
    for record in records:
        atomic_json(results_dir / f"{record['case_id']}.json", record)
    atomic_json(output / "results.json", records)
    rows = []
    for record in records:
        for metric, value in record.get("metrics", {}).items():
            rows.append(
                {
                    key: record[key]
                    for key in ("scenario", "seed", "coefficient_cv", "view", "method", "observation_variant")
                }
                | {"metric": metric, "value": value}
            )
    pd.DataFrame(rows).to_csv(output / "metrics_tidy.csv", index=False)
    manifest.update(
        {
            "status": (
                "complete"
                if all(r["status"] == "ok" for r in records)
                else "complete_with_failures"
            ),
            "records": len(records),
            "completed": sum(r["status"] == "ok" for r in records),
            "failed": sum(r["status"] != "ok" for r in records),
            "finished_unix": time.time(),
        }
    )
    atomic_json(manifest_path, manifest)
    return manifest


__all__ = ["RobustnessSuite", "load_robustness_suite", "run_robustness_suite"]
