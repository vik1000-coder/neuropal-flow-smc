"""Cross-fitted predictive-memory curves for hidden neuromodulator dynamics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import traceback
from typing import Any

import pandas as pd
import yaml

from .config import MethodSpec, ScenarioSpec
from .features import build_supervised, grouped_split, subset
from .metrics import pit_serial_metrics, predictive_metrics
from .registry import simulate_scenario
from .resources import enforce_storage_budget, limited_threads
from .schema import ResourceBudget
from .serialization import atomic_json, environment_manifest, stable_hash
from .tuning import tune_method


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class MemorySuite:
    name: str
    scenarios: tuple[ScenarioSpec, ...]
    methods: tuple[MethodSpec, ...]
    seeds: tuple[int, ...]
    views: tuple[str, ...]
    max_lags: tuple[int, ...]
    output_dir: str
    resource: ResourceBudget
    validation_fraction: float = .2
    test_fraction: float = .2
    n_predictive_samples: int = 16
    schema_version: str = "1"

    def validate(self) -> None:
        if self.schema_version != "1":
            raise ValueError(f"unsupported memory suite schema {self.schema_version!r}")
        if not self.scenarios or not self.methods or not self.seeds:
            raise ValueError("memory suite requires scenarios, methods, and seeds")
        if not self.max_lags or min(self.max_lags) < 1:
            raise ValueError("max_lags must be positive")
        if tuple(sorted(set(self.max_lags))) != self.max_lags:
            raise ValueError("max_lags must be sorted and unique")
        if set(self.views) - {"complete_state", "latent", "calcium"}:
            raise ValueError("unknown memory view")
        if not 0 < self.validation_fraction < .5 or not 0 < self.test_fraction < .5:
            raise ValueError("invalid split fractions")
        self.resource.validate()


def load_memory_suite(path: str | Path) -> MemorySuite:
    raw = yaml.safe_load(Path(path).read_text())
    scenarios = tuple(
        ScenarioSpec(
            id=item["id"],
            family=item.get("family", "mechanistic"),
            params=item.get("params", {}),
            n_trajectories=int(item.get("n_trajectories", 8)),
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
    suite = MemorySuite(
        name=str(raw["name"]),
        scenarios=scenarios,
        methods=methods,
        seeds=tuple(int(value) for value in raw.get("seeds", [0])),
        views=tuple(raw.get("views", ["latent", "calcium"])),
        max_lags=tuple(int(value) for value in raw.get("max_lags", [1, 2, 4, 8, 16])),
        output_dir=str(raw.get("output_dir", f"outputs/{raw['name']}")),
        resource=ResourceBudget(**raw.get("resource", {})),
        validation_fraction=float(raw.get("validation_fraction", .2)),
        test_fraction=float(raw.get("test_fraction", .2)),
        n_predictive_samples=int(raw.get("n_predictive_samples", 16)),
        schema_version=str(raw.get("schema_version", "1")),
    )
    suite.validate()
    return suite


def _score(estimator, test, n_samples: int) -> dict[str, float]:
    capability = estimator.capabilities.predictive_distribution
    if capability == "normalized":
        prediction = estimator.predict(test, n_samples=n_samples)
        metrics = {
            f"predictive.{key}": value
            for key, value in predictive_metrics(prediction, test.targets).items()
        }
        if prediction.cdf is not None:
            metrics.update(
                {
                    f"predictive.{key}": value
                    for key, value in pit_serial_metrics(prediction.cdf, test.groups).items()
                }
            )
        return metrics
    if capability == "unnormalized_score":
        score_domain = getattr(estimator, "score_domain", None)
        if score_domain not in {"conditional_outcome", "joint_consecutive_state"}:
            raise TypeError("score estimator lacks a registered score domain")
        if not hasattr(estimator, "dsm_reference_metrics"):
            raise TypeError("score estimator lacks fixed-reference DSM metrics")
        return {
            f"score.test.{score_domain}.{key}": float(value)
            for key, value in estimator.dsm_reference_metrics(
                test, seed=71_117
            ).items()
        }
    return {}


def _memory_gaps(records: list[dict[str, Any]], max_lag: int) -> None:
    reference: dict[tuple[Any, ...], dict[str, float]] = {}
    for record in records:
        if record.get("status") != "ok" or record["max_lag"] != max_lag:
            continue
        key = tuple(record[name] for name in ("scenario", "seed", "view", "method"))
        reference[key] = record["metrics"]
    for record in records:
        if record.get("status") != "ok":
            continue
        key = tuple(record[name] for name in ("scenario", "seed", "view", "method"))
        baseline = reference.get(key, {})
        if "predictive.nll" in record["metrics"] and "predictive.nll" in baseline:
            record["metrics"]["memory.finite_model_nll_gap_to_max_lag"] = float(
                record["metrics"]["predictive.nll"] - baseline["predictive.nll"]
            )
        score_references = [
            metric
            for metric in record["metrics"]
            if metric.startswith("score.test.")
            and ".fixed_reference." in metric
            and metric.endswith(".ladder_mean")
        ]
        if len(score_references) > 1:
            raise ValueError("memory record has multiple fixed-reference DSM ladders")
        if score_references:
            score_metric = score_references[0]
            if score_metric in baseline:
                score_domain = score_metric.split(".")[2]
                record["metrics"][
                    f"memory.{score_domain}.same_kernel_dsm_gap_to_max_lag"
                ] = float(
                    record["metrics"][score_metric] - baseline[score_metric]
                )


def run_memory_suite(path_or_suite: str | Path | MemorySuite) -> dict[str, Any]:
    suite = path_or_suite if isinstance(path_or_suite, MemorySuite) else load_memory_suite(path_or_suite)
    suite.validate()
    output = Path(suite.output_dir)
    if not output.is_absolute():
        output = PACKAGE_ROOT / output
    output = output.resolve()
    if PACKAGE_ROOT.resolve() not in output.parents:
        raise ValueError("memory output must remain inside benchmark root")
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
                "memory output contains a different frozen suite configuration; "
                "choose a new output_dir"
            )
        if previous_source != environment.get("benchmark_source_digest"):
            raise RuntimeError(
                "memory output contains a different benchmark source digest; "
                "choose a new output_dir"
            )
        if previous_execution_environment != environment.get(
            "execution_environment_digest"
        ):
            raise RuntimeError(
                "memory output contains a different execution environment digest; "
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
    for scenario in suite.scenarios:
        for seed in suite.seeds:
            dataset = simulate_scenario(scenario, seed)
            for view in suite.views:
                # Every history lane must score exactly the same prediction events.
                # ``times`` is the last history index t (the target is t + 1), so
                # the largest dense history first becomes available at L_max - 1.
                common_target_start = max(suite.max_lags) - 1
                for max_lag in suite.max_lags:
                    # Dense recent history is required for the conditional-information
                    # interpretation; geometric subsampling would change the sigma-field.
                    lags = tuple(range(1, max_lag + 1))
                    data = build_supervised(dataset, view=view, history_lags=lags, horizon=1)
                    data = subset(data, data.times >= common_target_start)
                    split = grouped_split(
                        data.groups,
                        validation_fraction=suite.validation_fraction,
                        test_fraction=suite.test_fraction,
                        seed=seed + 911,
                    )
                    train_mask, validation_mask, test_mask = split.masks(data)
                    train, validation, test = (
                        subset(data, train_mask),
                        subset(data, validation_mask),
                        subset(data, test_mask),
                    )
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
                        case_payload = {
                            "suite_config_digest": suite_digest,
                            "scenario": asdict(scenario),
                            "seed": seed,
                            "view": view,
                            "method": asdict(method),
                            "max_lag": max_lag,
                            "common_target_start": common_target_start,
                        }
                        case_id = stable_hash(case_payload)
                        case_path = results_dir / f"{case_id}.json"
                        if case_path.exists():
                            records.append(json.loads(case_path.read_text()))
                            manifest["skipped"] += 1
                            continue
                        base = {
                            "case_id": case_id,
                            "scenario": scenario.id,
                            "scenario_spec": asdict(scenario),
                            "seed": seed,
                            "view": view,
                            "method": method.name,
                            "method_spec": asdict(method),
                            "max_lag": max_lag,
                            "history_lags": lags,
                            "common_target_start": common_target_start,
                            "sample_counts": {
                                "all": len(data.targets),
                                "train": len(train.targets),
                                "validation": len(validation.targets),
                                "test": len(test.targets),
                            },
                            "claim": "finite-model useful-memory curve; not automatically conditional mutual information",
                            "split": {
                                "train_groups": split.train_groups,
                                "validation_groups": split.validation_groups,
                                "test_groups": split.test_groups,
                            },
                        }
                        case_started = time.perf_counter()
                        try:
                            with limited_threads(suite.resource):
                                tuned = tune_method(method, train, validation, seed=seed)
                                metrics = _score(tuned.estimator, test, suite.n_predictive_samples)
                            record = {
                                    **base,
                                    "status": "ok",
                                    "selected_params": tuned.params,
                                    "selection_objective": tuned.objective,
                                    "selection_score": tuned.score,
                                    "method_metadata": tuned.estimator.metadata(),
                                    "metrics": metrics,
                                    "wall_seconds": time.perf_counter() - case_started,
                                }
                        except Exception as error:
                            record = {
                                    **base,
                                    "status": "failed",
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                    "traceback": traceback.format_exc(limit=12),
                                    "wall_seconds": time.perf_counter() - case_started,
                                }
                        records.append(record)
                        atomic_json(case_path, record)
                        manifest["records"] = len(records)
                        manifest["completed"] = sum(
                            item.get("status") == "ok" for item in records
                        )
                        manifest["failed"] = sum(
                            item.get("status") != "ok" for item in records
                        )
                        manifest["last_case_id"] = case_id
                        manifest["last_update_unix"] = time.time()
                        atomic_json(manifest_path, manifest)
                        enforce_storage_budget(output, suite.resource)
    _memory_gaps(records, max(suite.max_lags))
    for record in records:
        atomic_json(results_dir / f"{record['case_id']}.json", record)
    atomic_json(output / "results.json", records)
    rows = []
    for record in records:
        for metric, value in record.get("metrics", {}).items():
            rows.append(
                {
                    key: record[key]
                    for key in ("scenario", "seed", "view", "method", "max_lag")
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


__all__ = ["MemorySuite", "load_memory_suite", "run_memory_suite"]
