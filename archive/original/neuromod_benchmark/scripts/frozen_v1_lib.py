#!/usr/bin/env python3
"""Freeze-gate and sequential-execution helpers for the frozen-v1 benchmark.

This module deliberately lives under ``scripts``.  It orchestrates the public
benchmark loaders/runners without becoming part of the scientific estimator API.
Preflight constructs configurations and estimator grids, but never calls ``fit``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neuromod_benchmark.changepoint import (  # noqa: E402
    METHOD_CLAIM_AXIS,
    PostChangeStrengths,
    run_changepoint_benchmark,
)
from neuromod_benchmark.config import MethodSpec, ScenarioSpec, SuiteConfig, load_suite  # noqa: E402
from neuromod_benchmark.hierarchy import HierarchyConfig  # noqa: E402
from neuromod_benchmark.mechanistic import MechanisticConfig  # noqa: E402
from neuromod_benchmark.memory import MemorySuite, load_memory_suite  # noqa: E402
from neuromod_benchmark.metric_contract import resolve_metric_contract  # noqa: E402
from neuromod_benchmark.registry import make_method  # noqa: E402
from neuromod_benchmark.resources import THREAD_ENV, directory_size  # noqa: E402
from neuromod_benchmark.response_evaluation import prepare_response_panels  # noqa: E402
from neuromod_benchmark.robustness import RobustnessSuite, load_robustness_suite  # noqa: E402
from neuromod_benchmark.schema import DGPConfig, ResourceBudget  # noqa: E402
from neuromod_benchmark.serialization import (  # noqa: E402
    atomic_json,
    environment_manifest,
    stable_hash,
)
from neuromod_benchmark.tuning import parameter_grid  # noqa: E402
from neuromod_benchmark.registry import simulate_scenario  # noqa: E402


SCHEMA_VERSION = 1
MIN_FREE_DISK_GIB = 4.0
MAX_THREADS = 2
REQUIRED_NICE = 19
FINAL_CONFIG_DIR = PACKAGE_ROOT / "configs" / "frozen_v1"
DRAFT_CONFIG_DIR = PACKAGE_ROOT / "configs" / "frozen_v1_draft"
FROZEN_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "frozen_v1"
STATE_DIR = FROZEN_OUTPUT_ROOT / "_orchestrator"
DEFAULT_PREFLIGHT_PATH = STATE_DIR / "preflight_manifest.json"
DEFAULT_RUN_MANIFEST_PATH = STATE_DIR / "run_manifest.json"
DEFAULT_EXECUTION_SUMMARY_PATH = FROZEN_OUTPUT_ROOT / "EXECUTION_SUMMARY.json"

CONFIG_SUFFIXES = {".yaml", ".yml", ".json"}
FIXED_STRATUM_TUNING_KEYS = {
    "components",
    "dsm_sigma",
    "kernel",
    "kernel_df",
    "noise_scale",
    "noise_scales",
    "reference_diffusion",
    "sigma_fraction",
}
PREFERRED_ORDER = (
    "causal_stress",
    "bridge_path",
    "intervention_response_null",
    "intervention_response_active",
    "correlation",
    "mechanism_channels",
    "tail",
    "core",
    "latent_recovery",
    "memory_light",
    "memory_neural",
    "robustness_light",
    "robustness_neural",
    "robustness_latent",
    "changepoint_matched_null",
)
EXTERNAL_CAUSAL_PACKAGES = {
    "pcmci_parcorr": "tigramite",
    "pysindy_discrete_official": "pysindy",
    "var_lingam_official": "lingam",
}
DOMAIN_LADDER_PATTERN = re.compile(
    r"score\.test\.(?:conditional_outcome|joint_consecutive_state)\."
    r"fixed_reference\.[a-z0-9_]+\.ladder_mean"
)


class FrozenV1Error(RuntimeError):
    """Raised when a freeze or execution invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_config_dir(requested: str | Path | None = None) -> tuple[Path, str]:
    """Resolve final configs, falling back to drafts only when final is absent."""

    if requested is not None:
        selected = Path(requested).expanduser().resolve()
        mode = (
            "final"
            if selected == FINAL_CONFIG_DIR.resolve()
            else "draft"
            if selected == DRAFT_CONFIG_DIR.resolve()
            else "explicit"
        )
    elif FINAL_CONFIG_DIR.is_dir() and any(
        path.suffix.lower() in CONFIG_SUFFIXES for path in FINAL_CONFIG_DIR.iterdir()
    ):
        selected, mode = FINAL_CONFIG_DIR.resolve(), "final"
    else:
        selected, mode = DRAFT_CONFIG_DIR.resolve(), "draft"
    if not selected.is_dir():
        raise FrozenV1Error(f"configuration directory does not exist: {selected}")
    return selected, mode


def config_paths(config_dir: Path) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            (
                path.resolve()
                for path in config_dir.iterdir()
                if path.is_file() and path.suffix.lower() in CONFIG_SUFFIXES
            ),
            key=lambda path: path.name,
        )
    )
    if not paths:
        raise FrozenV1Error(f"no suite configurations found in {config_dir}")
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise FrozenV1Error("configuration stems must be unique across YAML/JSON files")
    return paths


def config_hashes(config_dir: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in config_paths(config_dir)}


def classify_config(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return "changepoint"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise FrozenV1Error(f"{path.name}: YAML must contain a mapping")
    if "max_lags" in raw:
        return "memory"
    if "coefficient_cvs" in raw or "hierarchy_counts" in raw:
        return "robustness"
    return "main"


def _resolved_output(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resource_dict(resource: ResourceBudget) -> dict[str, Any]:
    resource.validate()
    if resource.max_threads > MAX_THREADS:
        raise FrozenV1Error(
            f"max_threads={resource.max_threads} exceeds frozen ceiling {MAX_THREADS}"
        )
    if resource.max_parallel_jobs != 1:
        raise FrozenV1Error("frozen execution requires max_parallel_jobs=1")
    if resource.nice != REQUIRED_NICE:
        raise FrozenV1Error(f"frozen execution requires nice={REQUIRED_NICE}")
    if resource.stop_free_disk_gb < MIN_FREE_DISK_GIB:
        raise FrozenV1Error(
            f"stop_free_disk_gb must be at least {MIN_FREE_DISK_GIB:.1f} GiB"
        )
    return asdict(resource)


def _validate_scenario_ids(scenarios: Sequence[ScenarioSpec]) -> set[str]:
    ids = [scenario.id for scenario in scenarios]
    if not all(isinstance(identifier, str) and identifier for identifier in ids):
        raise FrozenV1Error("scenario IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise FrozenV1Error("scenario IDs must be unique within a suite")
    return set(ids)


def _construct_scenarios(
    scenarios: Sequence[ScenarioSpec], seeds: Sequence[int]
) -> int:
    """Construct and validate every DGP parameterization, without simulation."""

    constructed = 0
    for scenario in scenarios:
        for seed in seeds:
            params = {**dict(scenario.params), "seed": int(seed)}
            if scenario.family == "mechanistic":
                config = MechanisticConfig(**params)
                config.validate()
            elif scenario.family == "legacy_generalized":
                config = DGPConfig(**params)
                if config.n_trajectories != scenario.n_trajectories:
                    config = DGPConfig(
                        **{
                            **asdict(config),
                            "n_trajectories": int(scenario.n_trajectories),
                        }
                    )
                config.validate()
            else:
                raise FrozenV1Error(
                    f"scenario {scenario.id!r} has unknown family {scenario.family!r}"
                )
            constructed += 1
    return constructed


def _validate_filter_subset(
    method: MethodSpec,
    *,
    scenario_ids: set[str],
    seeds: set[int],
    views: set[str],
    horizons: set[int],
) -> None:
    checks = (
        ("scenarios", method.scenarios, scenario_ids),
        ("seeds", method.seeds, seeds),
        ("views", method.views, views),
        ("horizons", method.horizons, horizons),
    )
    for label, selected, allowed in checks:
        if selected is None:
            continue
        unknown = set(selected) - allowed
        if unknown:
            raise FrozenV1Error(
                f"method {method.name!r} filters unknown {label}: {sorted(unknown)}"
            )


def _method_plans(
    methods: Sequence[MethodSpec],
    *,
    suite_seeds: Sequence[int],
) -> dict[str, dict[str, Any]]:
    names = [method.name for method in methods]
    name_counts = {name: names.count(name) for name in set(names)}
    occurrences: dict[str, int] = {}
    plans: dict[str, dict[str, Any]] = {}
    for method_index, method in enumerate(methods):
        occurrence = occurrences.get(method.name, 0) + 1
        occurrences[method.name] = occurrence
        spec_id = (
            method.name
            if name_counts[method.name] == 1
            else f"{method.name}__spec{occurrence}"
        )
        for key, values in method.tune.items():
            if isinstance(values, (str, bytes, Mapping)):
                raise FrozenV1Error(
                    f"method {method.name!r} tuning values for {key!r} must be a sequence"
                )
            if not tuple(values):
                raise FrozenV1Error(
                    f"method {method.name!r} has an empty tuning axis {key!r}"
                )
        prohibited = FIXED_STRATUM_TUNING_KEYS & set(method.tune)
        if prohibited:
            raise FrozenV1Error(
                f"method {method.name!r} tunes frozen corruption/capacity strata "
                f"{sorted(prohibited)}; encode them as named fixed aliases"
            )
        grid = list(parameter_grid(dict(method.params), dict(method.tune)))
        if not grid:
            raise FrozenV1Error(f"method {method.name!r} has an empty parameter grid")
        eligible_seeds = (
            tuple(seed for seed in suite_seeds if seed in method.seeds)
            if method.seeds is not None
            else tuple(suite_seeds)
        )
        if not eligible_seeds:
            raise FrozenV1Error(f"method {method.name!r} has no eligible suite seed")
        capabilities = []
        classes = []
        score_domains = []
        for params in grid:
            estimator = make_method(method.name, params, int(eligible_seeds[0]))
            capabilities.append(estimator.capabilities.predictive_distribution)
            classes.append(type(estimator).__qualname__)
            score_domains.append(getattr(estimator, "score_domain", None))
        if len(set(capabilities)) != 1 or len(set(classes)) != 1:
            raise FrozenV1Error(
                f"method {method.name!r} changes estimator class/capability within its grid"
            )
        graph_only = capabilities[0] == "none"
        if graph_only and len(grid) > 1:
            raise FrozenV1Error(
                f"graph-only method {method.name!r} has no registered tuning objective"
            )
        if capabilities[0] == "unnormalized_score" and set(score_domains) not in (
            {"conditional_outcome"},
            {"joint_consecutive_state"},
        ):
            raise FrozenV1Error(
                f"score method {method.name!r} lacks one stable registered score domain"
            )
        plans[spec_id] = {
            "spec_id": spec_id,
            "method_index": method_index,
            "name": method.name,
            "estimator_class": classes[0],
            "predictive_distribution": capabilities[0],
            "score_domain": score_domains[0],
            "graph_only": graph_only,
            "grid_size": len(grid),
            "fits_per_context": len(grid) + int(len(grid) > 1),
            "grid_hash": stable_hash(grid),
            "tune_keys": sorted(method.tune),
            "context_count": 0,
        }
    return plans


def _eligible(
    method: MethodSpec,
    *,
    scenario: str,
    seed: int,
    view: str,
    horizon: int,
) -> bool:
    return not (
        method.scenarios is not None
        and scenario not in method.scenarios
        or method.seeds is not None
        and seed not in method.seeds
        or method.views is not None
        and view not in method.views
        or method.horizons is not None
        and horizon not in method.horizons
    )


@dataclass(frozen=True)
class SuitePlan:
    identifier: str
    kind: str
    name: str
    config_path: str
    config_sha256: str
    suite_config_digest: str
    output_dir: str
    output_file: str | None
    resource: Mapping[str, Any]
    case_records: int
    expected_result_records: int
    fit_invocations: int
    tuning_candidate_evaluations: int
    dgp_constructions: int
    detector_scans: int
    method_grids: tuple[Mapping[str, Any], ...]
    response_truth_preflight: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _counts_from_method_contexts(
    method_plans: Mapping[str, Mapping[str, Any]]
) -> tuple[int, int]:
    candidate_evaluations = sum(
        int(item["context_count"]) * int(item["grid_size"])
        for item in method_plans.values()
    )
    fits = sum(
        int(item["context_count"]) * int(item["fits_per_context"])
        for item in method_plans.values()
    )
    return candidate_evaluations, fits


def _response_truth_preflight(
    suite: SuiteConfig,
    *,
    progress: Callable[[str], None],
) -> tuple[dict[str, Any], ...]:
    specification = suite.response_kernel_evaluation
    if specification is None:
        return ()
    summaries: list[dict[str, Any]] = []
    for scenario in suite.scenarios:
        for seed in suite.seeds:
            progress(
                f"response truth preflight: suite={suite.name} "
                f"scenario={scenario.id} seed={seed}"
            )
            dataset = simulate_scenario(scenario, seed)
            panels = prepare_response_panels(
                dataset,
                specification,
                history_lags=suite.history_lags,
            )
            if len(panels) != len(specification.operations):
                raise FrozenV1Error(
                    f"response suite {suite.name!r} produced {len(panels)} panels for "
                    f"{len(specification.operations)} registered operations"
                )
            for panel in panels:
                summaries.append(
                    {
                        "scenario": scenario.id,
                        "seed": int(seed),
                        "operation": panel.identifier,
                        "expected_effect": panel.expected_effect,
                        "n_pairs": len(panel.pairs),
                        "horizon": int(panel.truth.values.shape[0]),
                        "truth_rms_to_innovation_rms": float(
                            panel.truth_rms_to_innovation_rms
                        ),
                        "truth_mc_se_to_truth_rms_ratio": float(
                            panel.truth_mc_se_to_truth_rms_ratio
                        ),
                        "transient_decay_estimable_fraction": float(
                            panel.transient_decay_estimable_fraction
                        ),
                    }
                )
    return tuple(summaries)


def _plan_main(
    path: Path,
    *,
    run_response_truth: bool,
    progress: Callable[[str], None],
) -> SuitePlan:
    suite = load_suite(path)
    scenario_ids = _validate_scenario_ids(suite.scenarios)
    dgp_constructions = _construct_scenarios(suite.scenarios, suite.seeds)
    method_plans = _method_plans(suite.methods, suite_seeds=suite.seeds)
    effective_views = set(suite.views)
    for scenario in suite.scenarios:
        if scenario.views is not None:
            effective_views.update(scenario.views)
    for method in suite.methods:
        _validate_filter_subset(
            method,
            scenario_ids=scenario_ids,
            seeds=set(suite.seeds),
            views=effective_views,
            horizons=set(suite.horizons),
        )
    method_keys = tuple(method_plans)
    case_records = 0
    for scenario in suite.scenarios:
        for seed in suite.seeds:
            for view in scenario.views or suite.views:
                for horizon in suite.horizons:
                    selected_aliases: set[str] = set()
                    for method_key, method in zip(method_keys, suite.methods):
                        if _eligible(
                            method,
                            scenario=scenario.id,
                            seed=seed,
                            view=view,
                            horizon=horizon,
                        ):
                            if method.name in selected_aliases:
                                raise FrozenV1Error(
                                    f"duplicate method alias {method.name!r} overlaps one "
                                    f"execution context ({scenario.id}, {seed}, {view}, {horizon})"
                                )
                            selected_aliases.add(method.name)
                            case_records += 1
                            method_plans[method_key]["context_count"] += 1
    if case_records < 1:
        raise FrozenV1Error(f"suite {suite.name!r} contains no executable cases")
    empty_methods = [
        name for name, item in method_plans.items() if int(item["context_count"]) == 0
    ]
    if empty_methods:
        raise FrozenV1Error(f"methods have no executable context: {empty_methods}")
    for item in method_plans.values():
        item["result_record_count"] = int(item["context_count"])
    candidates, fits = _counts_from_method_contexts(method_plans)
    response = (
        _response_truth_preflight(suite, progress=progress)
        if run_response_truth
        else ()
    )
    output = _resolved_output(suite.output_dir)
    return SuitePlan(
        identifier=path.stem,
        kind="main",
        name=suite.name,
        config_path=str(path),
        config_sha256=sha256_file(path),
        suite_config_digest=stable_hash(asdict(suite)),
        output_dir=str(output),
        output_file=None,
        resource=_resource_dict(suite.resource),
        case_records=case_records,
        expected_result_records=case_records,
        fit_invocations=fits,
        tuning_candidate_evaluations=candidates,
        dgp_constructions=dgp_constructions,
        detector_scans=0,
        method_grids=tuple(method_plans.values()),
        response_truth_preflight=response,
    )


def _plan_memory(path: Path) -> SuitePlan:
    suite: MemorySuite = load_memory_suite(path)
    scenario_ids = _validate_scenario_ids(suite.scenarios)
    dgp_constructions = _construct_scenarios(suite.scenarios, suite.seeds)
    method_plans = _method_plans(suite.methods, suite_seeds=suite.seeds)
    for method in suite.methods:
        _validate_filter_subset(
            method,
            scenario_ids=scenario_ids,
            seeds=set(suite.seeds),
            views=set(suite.views),
            horizons={1},
        )
    method_keys = tuple(method_plans)
    records = 0
    for scenario in suite.scenarios:
        for seed in suite.seeds:
            for view in suite.views:
                for _max_lag in suite.max_lags:
                    selected_aliases: set[str] = set()
                    for method_key, method in zip(method_keys, suite.methods):
                        if _eligible(
                            method,
                            scenario=scenario.id,
                            seed=seed,
                            view=view,
                            horizon=1,
                        ):
                            if method.name in selected_aliases:
                                raise FrozenV1Error(
                                    f"duplicate method alias {method.name!r} overlaps one "
                                    f"memory context ({scenario.id}, {seed}, {view})"
                                )
                            selected_aliases.add(method.name)
                            records += 1
                            method_plans[method_key]["context_count"] += 1
    empty = [name for name, item in method_plans.items() if not item["context_count"]]
    if empty:
        raise FrozenV1Error(f"methods have no executable memory context: {empty}")
    for item in method_plans.values():
        item["result_record_count"] = int(item["context_count"])
    candidates, fits = _counts_from_method_contexts(method_plans)
    return SuitePlan(
        identifier=path.stem,
        kind="memory",
        name=suite.name,
        config_path=str(path),
        config_sha256=sha256_file(path),
        suite_config_digest=stable_hash(asdict(suite)),
        output_dir=str(_resolved_output(suite.output_dir)),
        output_file=None,
        resource=_resource_dict(suite.resource),
        case_records=records,
        expected_result_records=records,
        fit_invocations=fits,
        tuning_candidate_evaluations=candidates,
        dgp_constructions=dgp_constructions,
        detector_scans=0,
        method_grids=tuple(method_plans.values()),
    )


def _plan_robustness(path: Path) -> SuitePlan:
    suite: RobustnessSuite = load_robustness_suite(path)
    scenario_ids = _validate_scenario_ids(suite.scenarios)
    dgp_constructions = _construct_scenarios(suite.scenarios, suite.seeds)
    method_plans = _method_plans(suite.methods, suite_seeds=suite.seeds)
    for method in suite.methods:
        _validate_filter_subset(
            method,
            scenario_ids=scenario_ids,
            seeds=set(suite.seeds),
            views=set(suite.views),
            horizons={1},
        )
    method_keys = tuple(method_plans)
    for scenario in suite.scenarios:
        for seed in suite.seeds:
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
                hierarchy.validate()
    fit_contexts = 0
    for scenario in suite.scenarios:
        for seed in suite.seeds:
            for _cv in suite.coefficient_cvs:
                for view in suite.views:
                    selected_aliases: set[str] = set()
                    for method_key, method in zip(method_keys, suite.methods):
                        if _eligible(
                            method,
                            scenario=scenario.id,
                            seed=seed,
                            view=view,
                            horizon=1,
                        ):
                            if method.name in selected_aliases:
                                raise FrozenV1Error(
                                    f"duplicate method alias {method.name!r} overlaps one "
                                    f"robustness context ({scenario.id}, {seed}, {view})"
                                )
                            selected_aliases.add(method.name)
                            fit_contexts += 1
                            method_plans[method_key]["context_count"] += 1
    empty = [name for name, item in method_plans.items() if not item["context_count"]]
    if empty:
        raise FrozenV1Error(f"methods have no executable robustness context: {empty}")
    for item in method_plans.values():
        item["result_record_count"] = int(item["context_count"]) * len(
            suite.observation_variants
        )
    candidates, fits = _counts_from_method_contexts(method_plans)
    records = fit_contexts * len(suite.observation_variants)
    return SuitePlan(
        identifier=path.stem,
        kind="robustness",
        name=suite.name,
        config_path=str(path),
        config_sha256=sha256_file(path),
        suite_config_digest=stable_hash(asdict(suite)),
        output_dir=str(_resolved_output(suite.output_dir)),
        output_file=None,
        resource=_resource_dict(suite.resource),
        case_records=records,
        expected_result_records=records,
        fit_invocations=fits,
        tuning_candidate_evaluations=candidates,
        dgp_constructions=dgp_constructions,
        detector_scans=0,
        method_grids=tuple(method_plans.values()),
    )


def _positive_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FrozenV1Error(f"changepoint {key} must be a positive integer")
    return value


def _validate_changepoint_options(config: MechanisticConfig, options: Mapping[str, Any]) -> None:
    signature = inspect.signature(run_changepoint_benchmark)
    try:
        signature.bind(config, **options)
    except TypeError as error:
        raise FrozenV1Error(f"invalid changepoint benchmark arguments: {error}") from error
    methods = tuple(options.get("methods", ()))
    if not methods or len(methods) != len(set(methods)):
        raise FrozenV1Error("changepoint methods must be non-empty and unique")
    unknown = set(methods) - set(METHOD_CLAIM_AXIS)
    if unknown:
        raise FrozenV1Error(f"unknown changepoint detectors: {sorted(unknown)}")
    for key in ("n_calibration", "n_null_test", "n_changed"):
        _positive_int(options, key)
    alpha = float(options.get("alpha", 0.05))
    boundary_fraction = float(options.get("boundary_fraction", 0.5))
    confidence = float(options.get("summary_confidence", 0.95))
    if not 0 < alpha < 1 or not 0 < boundary_fraction < 1 or not 0 < confidence < 1:
        raise FrozenV1Error("changepoint alpha, boundary_fraction, and confidence must lie in (0,1)")
    min_segment = int(options.get("min_segment", 20))
    stride = int(options.get("stride", 1))
    scan_window = int(options.get("scan_window", 32))
    tolerance = int(options.get("tolerance", 10))
    if min_segment < 3 or stride < 1 or scan_window < 3 or tolerance < 0:
        raise FrozenV1Error("invalid changepoint segment/stride/window/tolerance settings")
    boundary = int(round(config.n_steps * boundary_fraction))
    if boundary < min_segment or config.n_steps - boundary < min_segment:
        raise FrozenV1Error("changepoint boundary violates min_segment")
    if float(options.get("ridge", 0.1)) < 0:
        raise FrozenV1Error("changepoint ridge must be nonnegative")
    reference_fraction = float(options.get("reference_fraction", 0.25))
    if not 0.10 <= reference_fraction <= 0.45:
        raise FrozenV1Error("changepoint reference_fraction must lie in [0.10,0.45]")
    if int(options.get("crossfit_folds", 5)) < 2 or int(options.get("crossfit_purge", 2)) < 0:
        raise FrozenV1Error("invalid changepoint crossfit settings")
    if int(options.get("residual_lags", 1)) < 1:
        raise FrozenV1Error("changepoint residual_lags must be positive")
    decays = tuple(float(value) for value in options.get("residual_ewma_decays", ()))
    if any(not 0 < value < 1 for value in decays):
        raise FrozenV1Error("changepoint residual EWMA decays must lie in (0,1)")
    thresholds = tuple(float(value) for value in options.get("tail_thresholds", (2.0, 3.0)))
    if not thresholds or any(value <= 1 for value in thresholds):
        raise FrozenV1Error("changepoint tail thresholds must exceed one")
    if thresholds != tuple(sorted(set(thresholds))):
        raise FrozenV1Error("changepoint tail thresholds must be strictly increasing")
    if options.get("variance_estimator", "block_median") not in {"sample", "block_median"}:
        raise FrozenV1Error("invalid changepoint variance estimator")
    if int(options.get("variance_block_size", 32)) < 8:
        raise FrozenV1Error("changepoint variance block size must be at least eight")
    if options.get("variance_channel_aggregation", "max") not in {"max", "sum"}:
        raise FrozenV1Error("invalid changepoint variance channel aggregation")
    if (
        {"variance_reliability", "residual_tail_shape"} & set(methods)
        and options.get("residual_mode", "frozen_prefix") == "frozen_prefix"
        and boundary <= math.ceil(config.n_steps * reference_fraction) + 3
    ):
        raise FrozenV1Error("changepoint boundary overlaps the frozen reference prefix")


def _plan_changepoint(path: Path) -> SuitePlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"name", "mechanistic", "benchmark"}:
        raise FrozenV1Error(
            f"{path.name}: changepoint config requires exactly name/mechanistic/benchmark"
        )
    config = MechanisticConfig(**raw["mechanistic"])
    config.validate()
    benchmark = dict(raw["benchmark"])
    strengths_raw = benchmark.pop("post_change_strengths", None)
    if not isinstance(strengths_raw, Mapping):
        raise FrozenV1Error("changepoint post_change_strengths must be a mapping")
    strengths = {
        scenario: PostChangeStrengths(**settings)
        for scenario, settings in strengths_raw.items()
    }
    if set(strengths) - {"mean", "dispersion", "matched_tail"}:
        raise FrozenV1Error("changepoint post_change_strengths has an unknown scenario")
    for item in strengths.values():
        item.validate()
    benchmark["post_change_strengths"] = strengths
    if "tail_thresholds" in benchmark:
        benchmark["tail_thresholds"] = tuple(benchmark["tail_thresholds"])
    if "methods" in benchmark:
        benchmark["methods"] = tuple(benchmark["methods"])
    if "residual_ewma_decays" in benchmark:
        benchmark["residual_ewma_decays"] = tuple(benchmark["residual_ewma_decays"])
    _validate_changepoint_options(config, benchmark)
    n_calibration = int(benchmark["n_calibration"])
    n_null = int(benchmark["n_null_test"])
    n_changed = int(benchmark["n_changed"])
    methods = tuple(benchmark["methods"])
    evaluation_events = n_null + 3 * n_changed
    output_dir = (FROZEN_OUTPUT_ROOT / path.stem).resolve()
    output_file = output_dir / "results.json"
    resource = ResourceBudget(
        max_threads=1,
        max_parallel_jobs=1,
        max_rss_gb=6.0,
        max_output_gb=0.8,
        nice=REQUIRED_NICE,
        checkpoint_minutes=15,
        stop_free_disk_gb=MIN_FREE_DISK_GIB,
    )
    return SuitePlan(
        identifier=path.stem,
        kind="changepoint",
        name=str(raw["name"]),
        config_path=str(path),
        config_sha256=sha256_file(path),
        suite_config_digest=stable_hash(raw),
        output_dir=str(output_dir),
        output_file=str(output_file),
        resource=_resource_dict(resource),
        case_records=n_calibration + evaluation_events,
        expected_result_records=evaluation_events * len(methods),
        fit_invocations=0,
        tuning_candidate_evaluations=0,
        dgp_constructions=1,
        detector_scans=(n_calibration + evaluation_events) * len(methods),
        method_grids=tuple(
            {
                "name": method,
                "claim_axis": METHOD_CLAIM_AXIS[method],
                "grid_size": 1,
                "context_count": n_calibration + evaluation_events,
                "result_record_count": evaluation_events,
            }
            for method in methods
        ),
    )


def plan_config(
    path: str | Path,
    *,
    run_response_truth: bool = False,
    progress: Callable[[str], None] = lambda _message: None,
) -> SuitePlan:
    source = Path(path).resolve()
    kind = classify_config(source)
    progress(f"validating {source.name} ({kind})")
    if kind == "main":
        return _plan_main(
            source,
            run_response_truth=run_response_truth,
            progress=progress,
        )
    if kind == "memory":
        return _plan_memory(source)
    if kind == "robustness":
        return _plan_robustness(source)
    return _plan_changepoint(source)


def ordered_plans(plans: Iterable[SuitePlan]) -> tuple[SuitePlan, ...]:
    rank = {name: index for index, name in enumerate(PREFERRED_ORDER)}
    return tuple(sorted(plans, key=lambda item: (rank.get(item.identifier, len(rank)), item.identifier)))


def _validate_output_isolation(plans: Sequence[SuitePlan]) -> None:
    root = FROZEN_OUTPUT_ROOT.resolve()
    state = STATE_DIR.resolve()
    outputs = [Path(plan.output_dir).resolve() for plan in plans]
    for plan, output in zip(plans, outputs):
        if not _is_relative_to(output, root) or output == root:
            raise FrozenV1Error(
                f"suite {plan.identifier!r} output is not isolated below {root}: {output}"
            )
        if output == state or _is_relative_to(output, state):
            raise FrozenV1Error(f"suite output overlaps orchestrator state: {output}")
    if len(outputs) != len(set(outputs)):
        raise FrozenV1Error("suite output directories must be unique")
    for index, left in enumerate(outputs):
        for right in outputs[index + 1 :]:
            if _is_relative_to(left, right) or _is_relative_to(right, left):
                raise FrozenV1Error(f"nested suite outputs are forbidden: {left} and {right}")


def _visible_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [
        item
        for item in path.rglob("*")
        if item.is_file() and item.name not in {".DS_Store"} and not item.name.startswith("._")
    ]


def _validate_existing_output(
    plan: SuitePlan, environment_identity: Mapping[str, str]
) -> None:
    output = Path(plan.output_dir)
    files = _visible_files(output)
    if not files:
        return
    output_gib = directory_size(output) / 1024**3
    if output_gib > float(plan.resource["max_output_gb"]):
        raise FrozenV1Error(
            f"{plan.identifier}: existing output {output_gib:.2f} GiB exceeds its quota"
        )
    provenance_path = output / "frozen_v1_provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for key, expected in environment_identity.items():
            if provenance.get(key) != expected:
                raise FrozenV1Error(
                    f"{plan.identifier}: existing output has another {key}"
                )
        if provenance.get("config_sha256") != plan.config_sha256:
            raise FrozenV1Error(f"{plan.identifier}: existing output has another config hash")
    if plan.kind == "changepoint":
        result = Path(plan.output_file or "")
        if result.exists() and not provenance_path.exists():
            raise FrozenV1Error(
                f"{plan.identifier}: changepoint result lacks frozen provenance; use a new output"
            )
        unknown = [item for item in files if item not in {result, provenance_path}]
        if unknown:
            raise FrozenV1Error(
                f"{plan.identifier}: unrecognized files already occupy output: {unknown[:3]}"
            )
        return
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        raise FrozenV1Error(f"{plan.identifier}: nonempty output lacks run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("suite_config_digest") != plan.suite_config_digest:
        raise FrozenV1Error(f"{plan.identifier}: output suite-config digest differs")
    existing_environment = manifest.get("environment", {})
    for key, expected in environment_identity.items():
        manifest_key = "benchmark_source_digest" if key == "source_tree_digest" else key
        if existing_environment.get(manifest_key) != expected:
            raise FrozenV1Error(
                f"{plan.identifier}: output {manifest_key} differs"
            )
    allowed_resume_statuses = {"running", "partial_max_cases", "complete"}
    if manifest.get("status") not in allowed_resume_statuses:
        raise FrozenV1Error(
            f"{plan.identifier}: existing manifest status {manifest.get('status')!r} "
            "is not resumable into frozen-v1"
        )
    for result_path in sorted((output / "results").glob("*.json")):
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record.get("status") != "ok":
            raise FrozenV1Error(
                f"{plan.identifier}: existing case {record.get('case_id')} is "
                f"{record.get('status')!r}; failed/partial cases cannot enter frozen-v1"
            )


def _environment_identity(environment: Mapping[str, Any]) -> dict[str, str]:
    identity = {
        "source_tree_digest": environment.get("benchmark_source_digest"),
        "sid_neuromod_source_digest": environment.get("sid_neuromod_source_digest"),
        "execution_environment_digest": environment.get("execution_environment_digest"),
    }
    missing = [key for key, value in identity.items() if not isinstance(value, str) or not value]
    if missing:
        raise FrozenV1Error(f"could not freeze required environment digests: {missing}")
    return {key: str(value) for key, value in identity.items()}


def _validate_external_causal_environment(
    plans: Sequence[SuitePlan], environment: Mapping[str, Any]
) -> None:
    methods = {
        str(grid.get("name"))
        for plan in plans
        for grid in plan.method_grids
    }
    required = {
        package
        for method, package in EXTERNAL_CAUSAL_PACKAGES.items()
        if method in methods
    }
    if not required:
        return
    packages = environment.get("causal_environment_packages")
    if not isinstance(packages, Mapping):
        raise FrozenV1Error("external causal methods require the isolated .causal_venv")
    missing = sorted(package for package in required if package not in packages)
    if missing:
        raise FrozenV1Error(f"isolated causal environment is missing packages: {missing}")


def preflight(
    *,
    requested_config_dir: str | Path | None = None,
    run_response_truth: bool = True,
    validate_existing_outputs: bool = True,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    config_dir, mode = resolve_config_dir(requested_config_dir)
    selected_paths = config_paths(config_dir)
    if mode in {"final", "draft"}:
        observed = {path.stem for path in selected_paths}
        expected = set(PREFERRED_ORDER)
        if observed != expected:
            raise FrozenV1Error(
                "frozen-v1 suite set differs from the preregistered 15 suites; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
    started = time.time()
    environment_before = environment_manifest(WORKSPACE_ROOT)
    environment_identity = _environment_identity(environment_before)
    digest_before = environment_identity["source_tree_digest"]
    hashes_before = config_hashes(config_dir)
    free_gib = shutil.disk_usage(PACKAGE_ROOT).free / 1024**3
    if free_gib < MIN_FREE_DISK_GIB:
        raise FrozenV1Error(
            f"free disk {free_gib:.2f} GiB is below the {MIN_FREE_DISK_GIB:.2f} GiB floor"
        )
    plans = ordered_plans(
        plan_config(
            path,
            run_response_truth=run_response_truth,
            progress=progress,
        )
        for path in selected_paths
    )
    _validate_output_isolation(plans)
    _validate_external_causal_environment(plans, environment_before)
    environment_after = environment_manifest(WORKSPACE_ROOT)
    environment_identity_after = _environment_identity(environment_after)
    hashes_after = config_hashes(config_dir)
    if environment_identity_after != environment_identity or hashes_after != hashes_before:
        raise FrozenV1Error(
            "source/dependency/environment/config identity changed during preflight; "
            "rerun after all editors and environment changes stop"
        )
    if validate_existing_outputs:
        for plan in plans:
            _validate_existing_output(plan, environment_identity)
    totals = {
        "suites": len(plans),
        "case_records": sum(plan.case_records for plan in plans),
        "expected_result_records": sum(plan.expected_result_records for plan in plans),
        "fit_invocations": sum(plan.fit_invocations for plan in plans),
        "tuning_candidate_evaluations": sum(
            plan.tuning_candidate_evaluations for plan in plans
        ),
        "dgp_constructions": sum(plan.dgp_constructions for plan in plans),
        "detector_scans": sum(plan.detector_scans for plan in plans),
        "response_truth_panels": sum(
            len(plan.response_truth_preflight) for plan in plans
        ),
    }
    identity = {
        **environment_identity,
        "config_hashes": hashes_before,
        "suite_config_digests": {
            plan.identifier: plan.suite_config_digest for plan in plans
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "preflight_id": stable_hash(identity),
        "started_unix": started,
        "finished_unix": time.time(),
        "finished_utc": utc_now(),
        "config_mode": mode,
        "config_dir": str(config_dir),
        "source_tree_digest": digest_before,
        "sid_neuromod_source_digest": environment_identity[
            "sid_neuromod_source_digest"
        ],
        "execution_environment_digest": environment_identity[
            "execution_environment_digest"
        ],
        "execution_environment": {
            "python": environment_before.get("python"),
            "platform": environment_before.get("platform"),
            "packages": environment_before.get("packages"),
            "causal_environment_packages": environment_before.get(
                "causal_environment_packages"
            ),
        },
        "config_hashes": hashes_before,
        "free_disk_gib": free_gib,
        "policy": {
            "max_threads": MAX_THREADS,
            "max_parallel_jobs": 1,
            "nice": REQUIRED_NICE,
            "minimum_free_disk_gib": MIN_FREE_DISK_GIB,
            "source_immutable_between_suites": True,
            "response_truth_preflight": run_response_truth,
            "fail_on_any_case_failure": True,
        },
        "totals": totals,
        "execution_order": [plan.identifier for plan in plans],
        "suites": [plan.to_dict() for plan in plans],
    }


def write_preflight_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_json(path, dict(payload))


def assert_baseline_immutable(preflight_manifest: Mapping[str, Any]) -> None:
    config_dir = Path(str(preflight_manifest["config_dir"])).resolve()
    expected_environment = {
        key: preflight_manifest.get(key)
        for key in (
            "source_tree_digest",
            "sid_neuromod_source_digest",
            "execution_environment_digest",
        )
    }
    current_environment = _environment_identity(environment_manifest(WORKSPACE_ROOT))
    if current_environment != expected_environment:
        raise FrozenV1Error(
            "source/dependency/execution-environment digest drift: "
            f"expected {expected_environment}, observed {current_environment}"
        )
    expected_hashes = dict(preflight_manifest.get("config_hashes", {}))
    current_hashes = config_hashes(config_dir)
    if current_hashes != expected_hashes:
        raise FrozenV1Error("frozen configuration hashes have changed since preflight")


def _plan_from_dict(raw: Mapping[str, Any]) -> SuitePlan:
    return SuitePlan(
        identifier=str(raw["identifier"]),
        kind=str(raw["kind"]),
        name=str(raw["name"]),
        config_path=str(raw["config_path"]),
        config_sha256=str(raw["config_sha256"]),
        suite_config_digest=str(raw["suite_config_digest"]),
        output_dir=str(raw["output_dir"]),
        output_file=None if raw.get("output_file") is None else str(raw["output_file"]),
        resource=dict(raw["resource"]),
        case_records=int(raw["case_records"]),
        expected_result_records=int(raw["expected_result_records"]),
        fit_invocations=int(raw["fit_invocations"]),
        tuning_candidate_evaluations=int(raw["tuning_candidate_evaluations"]),
        dgp_constructions=int(raw["dgp_constructions"]),
        detector_scans=int(raw["detector_scans"]),
        method_grids=tuple(raw.get("method_grids", ())),
        response_truth_preflight=tuple(raw.get("response_truth_preflight", ())),
    )


def load_preflight(path: Path = DEFAULT_PREFLIGHT_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FrozenV1Error(f"successful preflight manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "passed":
        raise FrozenV1Error("preflight manifest is not a successful supported manifest")
    if not payload.get("policy", {}).get("response_truth_preflight"):
        raise FrozenV1Error("frozen execution requires response-truth preflight")
    assert_baseline_immutable(payload)
    return payload


def command_for_plan(plan: SuitePlan) -> list[str]:
    python = sys.executable
    config = plan.config_path
    if plan.kind == "main":
        return [python, "-m", "neuromod_benchmark.cli", config]
    if plan.kind == "memory":
        return [python, str(PACKAGE_ROOT / "scripts" / "run_memory.py"), config]
    if plan.kind == "robustness":
        return [python, str(PACKAGE_ROOT / "scripts" / "run_robustness.py"), config]
    if plan.kind == "changepoint":
        return [
            python,
            str(PACKAGE_ROOT / "scripts" / "run_changepoint.py"),
            "--config",
            config,
            "--output",
            str(plan.output_file),
        ]
    raise FrozenV1Error(f"unsupported plan kind {plan.kind!r}")


def report_command_for_plan(plan: SuitePlan) -> list[str] | None:
    if plan.kind != "main":
        return None
    return [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / "report_results.py"),
        plan.output_dir,
    ]


def _execution_environment(max_threads: int) -> dict[str, str]:
    if not 1 <= max_threads <= MAX_THREADS:
        raise FrozenV1Error(f"invalid execution thread ceiling {max_threads}")
    environment = os.environ.copy()
    for name in THREAD_ENV:
        environment[name] = str(1 if name == "OMP_NUM_THREADS" else max_threads)
    environment.update(
        {
            "PYTHONPATH": str(SRC_ROOT),
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
            "TORCH_NUM_THREADS": "1",
        }
    )
    return environment


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _process_tree_rss_gib(root_pid: int) -> float | None:
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        rows: dict[int, tuple[int, int]] = {}
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) == 3:
                rows[int(fields[0])] = (int(fields[1]), int(fields[2]))
        descendants = {root_pid}
        changed = True
        while changed:
            changed = False
            for pid, (parent, _rss) in rows.items():
                if parent in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        return sum(rows.get(pid, (0, 0))[1] for pid in descendants) / 1024**2
    except Exception:
        return None


def _underlying_progress(plan: SuitePlan) -> dict[str, Any]:
    output = Path(plan.output_dir)
    if plan.kind == "changepoint":
        result = Path(plan.output_file or "")
        return {
            "result_exists": result.is_file(),
            "result_bytes": result.stat().st_size if result.is_file() else 0,
        }
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        return {"result_files": len(list((output / "results").glob("*.json")))}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            key: manifest.get(key)
            for key in (
                "status",
                "completed",
                "failed",
                "skipped",
                "records",
                "last_case_id",
                "last_update_unix",
            )
            if key in manifest
        } | {"result_files": len(list((output / "results").glob("*.json")))}
    except Exception as error:
        return {"manifest_read_error": str(error)}


def _run_monitored(
    command: Sequence[str],
    *,
    plan: SuitePlan,
    stage: str,
    log_path: Path,
    heartbeat_seconds: int,
    preflight_manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
) -> None:
    if heartbeat_seconds < 5 or heartbeat_seconds > 60:
        raise FrozenV1Error("heartbeat_seconds must lie in [5,60]")
    assert_baseline_immutable(preflight_manifest)
    free_gib = shutil.disk_usage(PACKAGE_ROOT).free / 1024**3
    disk_floor = max(MIN_FREE_DISK_GIB, float(plan.resource["stop_free_disk_gb"]))
    if free_gib < disk_floor:
        raise FrozenV1Error(f"free disk {free_gib:.2f} GiB is below {disk_floor:.2f} GiB")
    output_gib = directory_size(Path(plan.output_dir)) / 1024**3
    max_output_gib = float(plan.resource["max_output_gb"])
    if output_gib > max_output_gib:
        raise FrozenV1Error(
            f"output quota stop: {output_gib:.2f} GiB > {max_output_gib:.2f} GiB"
        )
    max_threads = int(plan.resource["max_threads"])
    environment = _execution_environment(max_threads)
    nice = shutil.which("nice")
    if nice is None:
        raise FrozenV1Error("the POSIX nice executable is required for frozen runs")
    actual_command = [nice, "-n", str(REQUIRED_NICE), *command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            f"\n[{utc_now()}] stage={stage} command={json.dumps(actual_command)}\n"
        )
        process = subprocess.Popen(
            actual_command,
            cwd=PACKAGE_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while True:
            try:
                return_code = process.wait(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                try:
                    assert_baseline_immutable(preflight_manifest)
                    free_gib = shutil.disk_usage(PACKAGE_ROOT).free / 1024**3
                    if free_gib < disk_floor:
                        raise FrozenV1Error(
                            f"disk safety stop: {free_gib:.2f} GiB < {disk_floor:.2f} GiB"
                        )
                    output_gib = directory_size(Path(plan.output_dir)) / 1024**3
                    if output_gib > max_output_gib:
                        raise FrozenV1Error(
                            f"output quota stop: {output_gib:.2f} GiB > "
                            f"{max_output_gib:.2f} GiB"
                        )
                    rss_gib = _process_tree_rss_gib(process.pid)
                    max_rss = float(plan.resource["max_rss_gb"])
                    if rss_gib is not None and rss_gib > max_rss:
                        raise FrozenV1Error(
                            f"memory safety stop: process tree RSS {rss_gib:.2f} GiB "
                            f"> {max_rss:.2f} GiB"
                        )
                except Exception:
                    _terminate_process_group(process)
                    raise
                heartbeat = {
                    "suite": plan.identifier,
                    "utc": utc_now(),
                    "stage": stage,
                    "elapsed_seconds": time.time() - started,
                    "free_disk_gib": free_gib,
                    "output_gib": output_gib,
                    "process_tree_rss_gib": rss_gib,
                    "log_bytes": log_path.stat().st_size,
                    "progress": _underlying_progress(plan),
                }
                state["last_heartbeat"] = heartbeat
                state["suites"][plan.identifier]["last_heartbeat"] = heartbeat
                atomic_json(state_path, state)
                print(
                    json.dumps(
                        {
                            "heartbeat": plan.identifier,
                            **heartbeat,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        log.write(f"[{utc_now()}] stage={stage} exit_code={return_code}\n")
    if return_code != 0:
        raise FrozenV1Error(
            f"{plan.identifier} {stage} exited with code {return_code}; see {log_path}"
        )
    output_gib = directory_size(Path(plan.output_dir)) / 1024**3
    if output_gib > max_output_gib:
        raise FrozenV1Error(
            f"output quota stop: {output_gib:.2f} GiB > {max_output_gib:.2f} GiB"
        )
    assert_baseline_immutable(preflight_manifest)


def _result_records(plan: SuitePlan) -> list[dict[str, Any]]:
    results_dir = Path(plan.output_dir) / "results"
    records = []
    for path in sorted(results_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _record_method_name(plan: SuitePlan, record: Mapping[str, Any]) -> Any:
    return (
        record.get("case", {}).get("method", {}).get("name")
        if plan.kind == "main"
        else record.get("method")
    )


def _validate_method_record_counts(
    plan: SuitePlan, records: Sequence[Mapping[str, Any]]
) -> None:
    expected: Counter[str] = Counter()
    for grid in plan.method_grids:
        if "result_record_count" in grid:
            expected[str(grid["name"])] += int(grid["result_record_count"])
    if not expected:
        return
    observed = Counter(str(_record_method_name(plan, record)) for record in records)
    if observed != expected:
        raise FrozenV1Error(
            f"{plan.identifier}: method-level result counts differ; "
            f"expected={dict(expected)}, observed={dict(observed)}"
        )


def _validate_primary_scores(plan: SuitePlan, records: Sequence[Mapping[str, Any]]) -> None:
    capabilities_by_name: dict[str, set[str]] = {}
    score_domains_by_name: dict[str, set[str]] = {}
    for grid in plan.method_grids:
        capability = grid.get("predictive_distribution")
        if capability is not None:
            capabilities_by_name.setdefault(str(grid["name"]), set()).add(str(capability))
        score_domain = grid.get("score_domain")
        if score_domain is not None:
            score_domains_by_name.setdefault(str(grid["name"]), set()).add(
                str(score_domain)
            )
    for record in records:
        method_name = _record_method_name(plan, record)
        if not isinstance(method_name, str) or method_name not in capabilities_by_name:
            raise FrozenV1Error(
                f"{plan.identifier}: result has unknown/missing method alias {method_name!r}"
            )
        capabilities = capabilities_by_name[method_name]
        if len(capabilities) != 1:
            raise FrozenV1Error(
                f"{plan.identifier}: method {method_name!r} has ambiguous capabilities"
            )
        capability = next(iter(capabilities))
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            raise FrozenV1Error(
                f"{plan.identifier}: ok record {record.get('case_id')} lacks metrics"
            )
        for metric_id in metrics:
            try:
                resolve_metric_contract(str(metric_id))
            except ValueError as error:
                raise FrozenV1Error(
                    f"{plan.identifier}: case {record.get('case_id')} emits {error}"
                ) from error
        if capability == "normalized":
            required = "predictive.nll"
        elif capability == "unnormalized_score":
            matching = [
                str(key)
                for key in metrics
                if DOMAIN_LADDER_PATTERN.fullmatch(str(key)) is not None
            ]
            if len(matching) != 1:
                raise FrozenV1Error(
                    f"{plan.identifier}: score-model case {record.get('case_id')} must "
                    f"contain exactly one fixed-reference ladder mean, found {matching}"
                )
            required = matching[0]
            expected_domains = score_domains_by_name.get(method_name, set())
            if expected_domains and (
                len(expected_domains) != 1
                or required.split(".")[2] != next(iter(expected_domains))
            ):
                raise FrozenV1Error(
                    f"{plan.identifier}: score domain in {required!r} does not match "
                    f"method {method_name!r} domain {sorted(expected_domains)}"
                )
        else:
            required = None
        checked_keys = {
            key
            for key in metrics
            if key in {
                "predictive.nll",
                "score.test_fixed_reference_dsm_ladder_risk",
            }
            or DOMAIN_LADDER_PATTERN.fullmatch(str(key)) is not None
        }
        if required is not None:
            checked_keys.add(required)
        for key in checked_keys:
            value = metrics.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                qualifier = "required" if key == required else "present"
                raise FrozenV1Error(
                    f"{plan.identifier}: {qualifier} primary score {key!r} is not "
                    f"finite for case {record.get('case_id')}"
                )


def _validate_causal_worker_metadata(
    plan: SuitePlan,
    records: Sequence[Mapping[str, Any]],
    causal_environment_packages: Mapping[str, Any] | None,
) -> None:
    configured = {
        str(grid.get("name"))
        for grid in plan.method_grids
        if str(grid.get("name")) in EXTERNAL_CAUSAL_PACKAGES
    }
    if not configured:
        return
    if not isinstance(causal_environment_packages, Mapping):
        raise FrozenV1Error(
            f"{plan.identifier}: causal package identity is absent from preflight"
        )
    seen: dict[str, set[tuple[str, str]]] = {name: set() for name in configured}
    for record in records:
        method_name = _record_method_name(plan, record)
        if method_name not in configured:
            continue
        expected_package = EXTERNAL_CAUSAL_PACKAGES[str(method_name)]
        expected_version = causal_environment_packages.get(expected_package)
        metadata = record.get("method_metadata")
        if not isinstance(metadata, Mapping):
            raise FrozenV1Error(
                f"{plan.identifier}: {method_name} record lacks worker metadata"
            )
        observed = (str(metadata.get("package")), str(metadata.get("version")))
        expected = (expected_package, str(expected_version))
        if expected_version is None or observed != expected:
            raise FrozenV1Error(
                f"{plan.identifier}: {method_name} worker package/version {observed} "
                f"does not match frozen environment {expected}"
            )
        seen[str(method_name)].add(observed)
    for method_name, identities in seen.items():
        if len(identities) != 1:
            raise FrozenV1Error(
                f"{plan.identifier}: {method_name} worker identity is missing or "
                f"inconsistent across records: {sorted(identities)}"
            )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def _validate_changepoint_payload(
    plan: SuitePlan,
    payload: Mapping[str, Any],
    raw_config: Mapping[str, Any],
) -> dict[str, Any]:
    benchmark = raw_config["benchmark"]
    expected_methods = tuple(benchmark["methods"])
    n_calibration = int(benchmark["n_calibration"])
    detections = payload.get("detections")
    if not isinstance(detections, list) or len(detections) != plan.expected_result_records:
        raise FrozenV1Error(
            f"{plan.identifier}: expected {plan.expected_result_records} detections, "
            f"found {None if not isinstance(detections, list) else len(detections)}"
        )
    calibrations = payload.get("calibrations")
    if not isinstance(calibrations, Mapping) or set(calibrations) != set(expected_methods):
        raise FrozenV1Error(f"{plan.identifier}: calibration method set differs from config")
    for method, calibration in calibrations.items():
        if not isinstance(calibration, Mapping):
            raise FrozenV1Error(f"{plan.identifier}: invalid calibration for {method}")
        maxima = calibration.get("null_max_scores")
        if (
            not isinstance(maxima, list)
            or len(maxima) != n_calibration
            or not all(_finite_number(value) for value in maxima)
        ):
            raise FrozenV1Error(
                f"{plan.identifier}: {method} calibration maxima are incomplete/nonfinite"
            )
        for key in (
            "alpha",
            "threshold",
            "minimum_p_value",
            "attainable_null_call_rate",
        ):
            if not _finite_number(calibration.get(key)):
                raise FrozenV1Error(
                    f"{plan.identifier}: {method} calibration {key} is nonfinite"
                )
    identities: set[tuple[str, str]] = set()
    for row in detections:
        if not isinstance(row, Mapping):
            raise FrozenV1Error(f"{plan.identifier}: a detection row is not a mapping")
        identity = (str(row.get("series_id")), str(row.get("method")))
        if identity in identities:
            raise FrozenV1Error(f"{plan.identifier}: duplicate detection {identity}")
        identities.add(identity)
        if row.get("method") not in expected_methods:
            raise FrozenV1Error(f"{plan.identifier}: detection uses an unknown method")
        for key in ("max_score", "effect_size", "threshold", "p_value"):
            if not _finite_number(row.get(key)):
                raise FrozenV1Error(
                    f"{plan.identifier}: detection {identity} has nonfinite {key}"
                )
        if not 0 <= float(row["p_value"]) <= 1:
            raise FrozenV1Error(f"{plan.identifier}: detection p-value is outside [0,1]")
        if isinstance(row.get("tau_hat"), bool) or not isinstance(row.get("tau_hat"), int):
            raise FrozenV1Error(f"{plan.identifier}: detection tau_hat is not an integer")
        if not isinstance(row.get("called"), bool):
            raise FrozenV1Error(f"{plan.identifier}: detection called flag is invalid")
        if row.get("scenario") == "null":
            if not isinstance(row.get("false_positive"), bool):
                raise FrozenV1Error(f"{plan.identifier}: null detection lacks an FPR flag")
        else:
            for key in ("effect_size_at_truth", "score_at_truth"):
                if not _finite_number(row.get(key)):
                    raise FrozenV1Error(
                        f"{plan.identifier}: changed detection {identity} has nonfinite {key}"
                    )
            if not isinstance(row.get("hit"), bool) or not isinstance(
                row.get("detected_within"), bool
            ):
                raise FrozenV1Error(
                    f"{plan.identifier}: changed detection {identity} lacks hit flags"
                )
            if isinstance(row.get("localization_delay"), bool) or not isinstance(
                row.get("localization_delay"), int
            ):
                raise FrozenV1Error(
                    f"{plan.identifier}: changed detection {identity} lacks integer delay"
                )
    evaluation_events = plan.expected_result_records // len(expected_methods)
    observed_method_counts = Counter(str(row["method"]) for row in detections)
    expected_method_counts = Counter(
        {method: evaluation_events for method in expected_methods}
    )
    if observed_method_counts != expected_method_counts:
        raise FrozenV1Error(
            f"{plan.identifier}: detection counts by method differ; "
            f"expected={dict(expected_method_counts)}, "
            f"observed={dict(observed_method_counts)}"
        )
    attributions = payload.get("attributions")
    if not isinstance(attributions, list) or len(attributions) != evaluation_events:
        raise FrozenV1Error(
            f"{plan.identifier}: expected {evaluation_events} attribution rows"
        )
    if not payload.get("detection_summary") or not payload.get("attribution_summary"):
        raise FrozenV1Error(f"{plan.identifier}: changepoint summaries are missing")
    return {
        "status": "complete",
        "result_records": len(detections),
        "result_path": str(plan.output_file),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "calibration_records": n_calibration * len(expected_methods),
        "attribution_records": len(attributions),
    }


def verify_suite_completion(
    plan: SuitePlan,
    *,
    source_digest: str,
    require_report: bool,
    sid_neuromod_source_digest: str | None = None,
    execution_environment_digest: str | None = None,
    causal_environment_packages: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(plan.output_dir)
    if plan.kind == "changepoint":
        result_path = Path(plan.output_file or "")
        if not result_path.is_file():
            raise FrozenV1Error(f"{plan.identifier}: missing changepoint result")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise FrozenV1Error(f"{plan.identifier}: unsupported result schema")
        raw_config = json.loads(Path(plan.config_path).read_text(encoding="utf-8"))
        if payload.get("config") != raw_config:
            raise FrozenV1Error(f"{plan.identifier}: result embeds a different config")
        return _validate_changepoint_payload(plan, payload, raw_config)
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise FrozenV1Error(f"{plan.identifier}: missing run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("suite_config_digest") != plan.suite_config_digest:
        raise FrozenV1Error(f"{plan.identifier}: suite-config digest mismatch")
    observed_source = manifest.get("environment", {}).get("benchmark_source_digest")
    if observed_source != source_digest:
        raise FrozenV1Error(f"{plan.identifier}: benchmark-source digest mismatch")
    manifest_environment = manifest.get("environment", {})
    if (
        sid_neuromod_source_digest is not None
        and manifest_environment.get("sid_neuromod_source_digest")
        != sid_neuromod_source_digest
    ):
        raise FrozenV1Error(f"{plan.identifier}: SID source digest mismatch")
    if (
        execution_environment_digest is not None
        and manifest_environment.get("execution_environment_digest")
        != execution_environment_digest
    ):
        raise FrozenV1Error(
            f"{plan.identifier}: execution-environment digest mismatch"
        )
    if manifest.get("status") != "complete":
        raise FrozenV1Error(
            f"{plan.identifier}: underlying status is {manifest.get('status')!r}"
        )
    records = _result_records(plan)
    if len(records) != plan.expected_result_records:
        raise FrozenV1Error(
            f"{plan.identifier}: expected {plan.expected_result_records} result records, "
            f"found {len(records)}"
        )
    failures = [record for record in records if record.get("status") != "ok"]
    if failures:
        first = failures[0]
        raise FrozenV1Error(
            f"{plan.identifier}: {len(failures)} result records failed; "
            f"first={first.get('case_id')} {first.get('error_type')}: {first.get('error')}"
        )
    case_ids = [record.get("case_id") for record in records]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise FrozenV1Error(f"{plan.identifier}: a result has a missing case ID")
    if len(case_ids) != len(set(case_ids)):
        raise FrozenV1Error(f"{plan.identifier}: duplicate case IDs are present")
    _validate_method_record_counts(plan, records)
    _validate_primary_scores(plan, records)
    _validate_causal_worker_metadata(plan, records, causal_environment_packages)
    if require_report:
        validation_path = output / "validation.json"
        report_path = output / "REPORT.md"
        if not validation_path.is_file() or not report_path.is_file():
            raise FrozenV1Error(f"{plan.identifier}: report/validation artifacts are missing")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("valid") is not True:
            raise FrozenV1Error(
                f"{plan.identifier}: report validation failed: "
                f"{validation.get('problems', [])}"
            )
    return {
        "status": "complete",
        "result_records": len(records),
        "manifest_completed": manifest.get("completed"),
        "manifest_skipped": manifest.get("skipped"),
        "manifest_path": str(manifest_path),
    }


def _write_provenance(
    plan: SuitePlan,
    *,
    preflight_manifest: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> None:
    atomic_json(
        Path(plan.output_dir) / "frozen_v1_provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "suite": plan.identifier,
            "preflight_id": preflight_manifest["preflight_id"],
            "source_tree_digest": preflight_manifest["source_tree_digest"],
            "sid_neuromod_source_digest": preflight_manifest[
                "sid_neuromod_source_digest"
            ],
            "execution_environment_digest": preflight_manifest[
                "execution_environment_digest"
            ],
            "config_sha256": plan.config_sha256,
            "suite_config_digest": plan.suite_config_digest,
            "finished_utc": utc_now(),
            "verification": dict(verification),
        },
    )


class ExecutionLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "ExecutionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            raise FrozenV1Error("another frozen-v1 orchestrator holds the execution lock") from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "started_utc": utc_now()}) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _initial_execution_state(
    preflight_manifest: Mapping[str, Any], plans: Sequence[SuitePlan]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "preflight_id": preflight_manifest["preflight_id"],
        "source_tree_digest": preflight_manifest["source_tree_digest"],
        "sid_neuromod_source_digest": preflight_manifest[
            "sid_neuromod_source_digest"
        ],
        "execution_environment_digest": preflight_manifest[
            "execution_environment_digest"
        ],
        "config_hashes": preflight_manifest["config_hashes"],
        "started_utc": utc_now(),
        "started_unix": time.time(),
        "execution_order": [plan.identifier for plan in plans],
        "suites": {
            plan.identifier: {
                "status": "pending",
                "kind": plan.kind,
                "config_sha256": plan.config_sha256,
                "output_dir": plan.output_dir,
                "expected_result_records": plan.expected_result_records,
            }
            for plan in plans
        },
    }


def _load_or_initialize_state(
    state_path: Path,
    *,
    preflight_manifest: Mapping[str, Any],
    plans: Sequence[SuitePlan],
) -> dict[str, Any]:
    if not state_path.is_file():
        state = _initial_execution_state(preflight_manifest, plans)
        atomic_json(state_path, state)
        return state
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("preflight_id") != preflight_manifest.get("preflight_id"):
        raise FrozenV1Error(
            "execution state belongs to another preflight; use a new frozen version/output"
        )
    for key in (
        "source_tree_digest",
        "sid_neuromod_source_digest",
        "execution_environment_digest",
    ):
        if state.get(key) != preflight_manifest.get(key):
            raise FrozenV1Error(f"execution state has another frozen {key}")
    if state.get("execution_order") != [plan.identifier for plan in plans]:
        raise FrozenV1Error("execution order differs from the frozen preflight")
    for plan in plans:
        suite_state = state.get("suites", {}).get(plan.identifier, {})
        if suite_state.get("config_sha256") != plan.config_sha256:
            raise FrozenV1Error(f"{plan.identifier}: execution-state config hash differs")
    state["status"] = "running"
    state["resumed_utc"] = utc_now()
    atomic_json(state_path, state)
    return state


def execute_all(
    *,
    preflight_path: Path = DEFAULT_PREFLIGHT_PATH,
    state_path: Path = DEFAULT_RUN_MANIFEST_PATH,
    heartbeat_seconds: int = 30,
    allow_draft_run: bool = False,
    generate_reports: bool = True,
) -> dict[str, Any]:
    preflight_manifest = load_preflight(preflight_path)
    if preflight_manifest.get("config_mode") != "final" and not allow_draft_run:
        raise FrozenV1Error(
            "refusing to execute non-final configs; copy reviewed configs to "
            "configs/frozen_v1 and rerun preflight"
        )
    plans = tuple(_plan_from_dict(raw) for raw in preflight_manifest["suites"])
    _validate_output_isolation(plans)
    lock_path = state_path.parent / "execution.lock"
    with ExecutionLock(lock_path):
        state = _load_or_initialize_state(
            state_path,
            preflight_manifest=preflight_manifest,
            plans=plans,
        )
        for index, plan in enumerate(plans, start=1):
            assert_baseline_immutable(preflight_manifest)
            suite_state = state["suites"][plan.identifier]
            require_report = generate_reports and plan.kind == "main"
            if suite_state.get("status") == "complete":
                verification = verify_suite_completion(
                    plan,
                    source_digest=preflight_manifest["source_tree_digest"],
                    require_report=require_report,
                    sid_neuromod_source_digest=preflight_manifest[
                        "sid_neuromod_source_digest"
                    ],
                    execution_environment_digest=preflight_manifest[
                        "execution_environment_digest"
                    ],
                    causal_environment_packages=preflight_manifest[
                        "execution_environment"
                    ]["causal_environment_packages"],
                )
                suite_state["verification"] = verification
                suite_state["resume_verified_utc"] = utc_now()
                atomic_json(state_path, state)
                print(f"resume verified {plan.identifier}", flush=True)
                continue
            # Recover the narrow crash window after output completion but before the
            # orchestrator checkpoint.  A pending suite is never accepted this way.
            if suite_state.get("status") in {"running", "reporting", "verifying"}:
                try:
                    verification = verify_suite_completion(
                        plan,
                        source_digest=preflight_manifest["source_tree_digest"],
                        require_report=require_report,
                        sid_neuromod_source_digest=preflight_manifest[
                            "sid_neuromod_source_digest"
                        ],
                        execution_environment_digest=preflight_manifest[
                            "execution_environment_digest"
                        ],
                        causal_environment_packages=preflight_manifest[
                            "execution_environment"
                        ]["causal_environment_packages"],
                    )
                except Exception:
                    pass
                else:
                    _write_provenance(
                        plan,
                        preflight_manifest=preflight_manifest,
                        verification=verification,
                    )
                    suite_state.update(
                        {
                            "status": "complete",
                            "recovered_after_interruption": True,
                            "finished_utc": utc_now(),
                            "verification": verification,
                        }
                    )
                    atomic_json(state_path, state)
                    continue
            log_path = state_path.parent / "logs" / f"{index:02d}_{plan.identifier}.log"
            suite_state.update(
                {
                    "status": "running",
                    "sequence_index": index,
                    "started_utc": utc_now(),
                    "command": command_for_plan(plan),
                    "log_path": str(log_path),
                }
            )
            atomic_json(state_path, state)
            try:
                _run_monitored(
                    command_for_plan(plan),
                    plan=plan,
                    stage="benchmark",
                    log_path=log_path,
                    heartbeat_seconds=heartbeat_seconds,
                    preflight_manifest=preflight_manifest,
                    state=state,
                    state_path=state_path,
                )
                suite_state["status"] = "verifying"
                atomic_json(state_path, state)
                # Verify cases before reporting so a runner's zero exit status cannot
                # conceal caught per-case failures.
                verification = verify_suite_completion(
                    plan,
                    source_digest=preflight_manifest["source_tree_digest"],
                    require_report=False,
                    sid_neuromod_source_digest=preflight_manifest[
                        "sid_neuromod_source_digest"
                    ],
                    execution_environment_digest=preflight_manifest[
                        "execution_environment_digest"
                    ],
                    causal_environment_packages=preflight_manifest[
                        "execution_environment"
                    ]["causal_environment_packages"],
                )
                report_command = report_command_for_plan(plan) if generate_reports else None
                if report_command is not None:
                    suite_state["status"] = "reporting"
                    atomic_json(state_path, state)
                    _run_monitored(
                        report_command,
                        plan=plan,
                        stage="report",
                        log_path=log_path,
                        heartbeat_seconds=heartbeat_seconds,
                        preflight_manifest=preflight_manifest,
                        state=state,
                        state_path=state_path,
                    )
                    verification = verify_suite_completion(
                        plan,
                        source_digest=preflight_manifest["source_tree_digest"],
                        require_report=True,
                        sid_neuromod_source_digest=preflight_manifest[
                            "sid_neuromod_source_digest"
                        ],
                        execution_environment_digest=preflight_manifest[
                            "execution_environment_digest"
                        ],
                        causal_environment_packages=preflight_manifest[
                            "execution_environment"
                        ]["causal_environment_packages"],
                    )
                assert_baseline_immutable(preflight_manifest)
                _write_provenance(
                    plan,
                    preflight_manifest=preflight_manifest,
                    verification=verification,
                )
                suite_state.update(
                    {
                        "status": "complete",
                        "finished_utc": utc_now(),
                        "verification": verification,
                        "log_path": str(log_path),
                    }
                )
                atomic_json(state_path, state)
                print(f"completed {plan.identifier}", flush=True)
            except Exception as error:
                suite_state.update(
                    {
                        "status": "failed",
                        "failed_utc": utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "log_path": str(log_path),
                    }
                )
                state.update(
                    {
                        "status": "failed",
                        "failed_suite": plan.identifier,
                        "failed_utc": utc_now(),
                    }
                )
                atomic_json(state_path, state)
                raise
        assert_baseline_immutable(preflight_manifest)
        state.update(
            {
                "status": "complete",
                "finished_utc": utc_now(),
                "finished_unix": time.time(),
            }
        )
        atomic_json(state_path, state)
    summary = build_execution_summary(
        preflight_path=preflight_path,
        state_path=state_path,
        require_reports=generate_reports,
    )
    atomic_json(DEFAULT_EXECUTION_SUMMARY_PATH, summary)
    return state


def build_execution_summary(
    *,
    preflight_path: Path = DEFAULT_PREFLIGHT_PATH,
    state_path: Path = DEFAULT_RUN_MANIFEST_PATH,
    require_reports: bool = True,
) -> dict[str, Any]:
    preflight_manifest = load_preflight(preflight_path)
    if not state_path.is_file():
        raise FrozenV1Error(f"execution manifest does not exist: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise FrozenV1Error(f"execution status is not complete: {state.get('status')!r}")
    if state.get("preflight_id") != preflight_manifest.get("preflight_id"):
        raise FrozenV1Error("execution and preflight identities differ")
    suites = []
    plans = tuple(_plan_from_dict(raw) for raw in preflight_manifest["suites"])
    for plan in plans:
        suite_state = state.get("suites", {}).get(plan.identifier, {})
        if suite_state.get("status") != "complete":
            raise FrozenV1Error(f"{plan.identifier}: orchestrator status is not complete")
        verification = verify_suite_completion(
            plan,
            source_digest=preflight_manifest["source_tree_digest"],
            require_report=require_reports and plan.kind == "main",
            sid_neuromod_source_digest=preflight_manifest[
                "sid_neuromod_source_digest"
            ],
            execution_environment_digest=preflight_manifest[
                "execution_environment_digest"
            ],
            causal_environment_packages=preflight_manifest[
                "execution_environment"
            ]["causal_environment_packages"],
        )
        provenance_path = Path(plan.output_dir) / "frozen_v1_provenance.json"
        if not provenance_path.is_file():
            raise FrozenV1Error(f"{plan.identifier}: missing frozen provenance")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("preflight_id") != preflight_manifest["preflight_id"]:
            raise FrozenV1Error(f"{plan.identifier}: provenance preflight ID differs")
        for key in (
            "source_tree_digest",
            "sid_neuromod_source_digest",
            "execution_environment_digest",
        ):
            if provenance.get(key) != preflight_manifest[key]:
                raise FrozenV1Error(
                    f"{plan.identifier}: provenance {key} differs from preflight"
                )
        suites.append(
            {
                "identifier": plan.identifier,
                "kind": plan.kind,
                "status": "complete",
                "config_sha256": plan.config_sha256,
                "suite_config_digest": plan.suite_config_digest,
                "expected_result_records": plan.expected_result_records,
                "verified_result_records": verification["result_records"],
                "fit_invocations_planned": plan.fit_invocations,
                "detector_scans_planned": plan.detector_scans,
                "output_dir": plan.output_dir,
                "verification": verification,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "generated_utc": utc_now(),
        "claim_scope": "mechanical execution/provenance validation only; no scientific conclusions",
        "preflight_id": preflight_manifest["preflight_id"],
        "source_tree_digest": preflight_manifest["source_tree_digest"],
        "sid_neuromod_source_digest": preflight_manifest[
            "sid_neuromod_source_digest"
        ],
        "execution_environment_digest": preflight_manifest[
            "execution_environment_digest"
        ],
        "config_hashes": preflight_manifest["config_hashes"],
        "totals": {
            "suites": len(suites),
            "verified_result_records": sum(
                item["verified_result_records"] for item in suites
            ),
            "fit_invocations_planned": sum(
                item["fit_invocations_planned"] for item in suites
            ),
            "detector_scans_planned": sum(
                item["detector_scans_planned"] for item in suites
            ),
            "failures": 0,
        },
        "suites": suites,
    }


__all__ = [
    "DEFAULT_EXECUTION_SUMMARY_PATH",
    "DEFAULT_PREFLIGHT_PATH",
    "DEFAULT_RUN_MANIFEST_PATH",
    "FINAL_CONFIG_DIR",
    "FrozenV1Error",
    "SuitePlan",
    "assert_baseline_immutable",
    "build_execution_summary",
    "config_hashes",
    "execute_all",
    "plan_config",
    "preflight",
    "resolve_config_dir",
    "verify_suite_completion",
    "write_preflight_manifest",
]
