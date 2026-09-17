"""Resumable, resource-capped benchmark runner."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .config import SuiteConfig, load_suite
from .evaluation import evaluate_estimator
from .features import build_supervised, grouped_split, subset
from .registry import simulate_scenario
from .resources import enforce_storage_budget, limited_threads, resource_snapshot
from .response_evaluation import prepare_response_panels
from .serialization import atomic_json, environment_manifest, stable_hash
from .tuning import tune_method


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent


def _case_payload(suite, scenario, method, seed, view, horizon, suite_config_digest):
    return {
        "suite": suite.name,
        "suite_config_digest": suite_config_digest,
        "scenario": asdict(scenario),
        "method": asdict(method),
        "seed": seed,
        "view": view,
        "horizon": horizon,
        "history_lags": suite.history_lags,
    }


def run_suite(
    suite_or_path: SuiteConfig | str | Path,
    *,
    max_cases: int | None = None,
    resume: bool = True,
) -> dict:
    suite = load_suite(suite_or_path) if not isinstance(suite_or_path, SuiteConfig) else suite_or_path
    suite.validate()
    output = Path(suite.output_dir)
    if not output.is_absolute():
        output = PACKAGE_ROOT / output
    output = output.resolve()
    if PACKAGE_ROOT.resolve() not in output.parents and output != PACKAGE_ROOT.resolve():
        raise ValueError(f"output {output} escapes isolated benchmark root {PACKAGE_ROOT}")
    enforce_storage_budget(output, suite.resource)
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    frozen_config = asdict(suite)
    suite_config_digest = stable_hash(frozen_config)
    environment = environment_manifest(WORKSPACE_ROOT)
    existing_results = list(results_dir.glob("*.json")) if results_dir.exists() else []
    if manifest_path.exists() and existing_results:
        previous = json.loads(manifest_path.read_text())
        previous_config_digest = previous.get("suite_config_digest")
        if previous_config_digest is None and previous.get("config") is not None:
            previous_config_digest = stable_hash(previous["config"])
        previous_source = previous.get("environment", {}).get(
            "benchmark_source_digest"
        )
        current_source = environment.get("benchmark_source_digest")
        previous_execution_environment = previous.get("environment", {}).get(
            "execution_environment_digest"
        )
        current_execution_environment = environment.get(
            "execution_environment_digest"
        )
        if previous_config_digest != suite_config_digest:
            raise RuntimeError(
                "output directory contains a different frozen suite configuration; "
                "choose a new output_dir rather than mixing case populations"
            )
        if previous_source != current_source:
            raise RuntimeError(
                "output directory contains results from a different benchmark source "
                "digest; choose a new output_dir rather than mixing implementations"
            )
        if previous_execution_environment != current_execution_environment:
            raise RuntimeError(
                "output directory contains results from a different execution "
                "environment digest; choose a new output_dir rather than mixing "
                "Python/package/SID implementations"
            )
    manifest = {
        "suite": suite.name,
        "suite_config_digest": suite_config_digest,
        "status": "running",
        "started_unix": time.time(),
        "config": frozen_config,
        "environment": environment,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
    }
    atomic_json(manifest_path, manifest)

    attempted = 0
    for scenario in suite.scenarios:
        for seed in suite.seeds:
            dataset = simulate_scenario(scenario, seed)
            try:
                response_panels = prepare_response_panels(
                    dataset,
                    suite.response_kernel_evaluation,
                    history_lags=suite.history_lags,
                )
            except Exception as error:
                manifest.update(
                    {
                        "status": "failed_scientific_preflight",
                        "failed": manifest["failed"] + 1,
                        "preflight_failure": {
                            "scenario": scenario.id,
                            "seed": seed,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(limit=12),
                        },
                        "finished_unix": time.time(),
                    }
                )
                atomic_json(manifest_path, manifest)
                raise
            views = scenario.views or suite.views
            for view in views:
                for horizon in suite.horizons:
                    data = build_supervised(
                        dataset,
                        view=view,
                        history_lags=suite.history_lags,
                        horizon=horizon,
                    )
                    split = grouped_split(
                        data.groups,
                        validation_fraction=suite.validation_fraction,
                        test_fraction=suite.test_fraction,
                        # Freeze held-out worms across horizons so horizon
                        # degradation is not confounded with a different test set.
                        seed=seed + 97,
                    )
                    train_mask, validation_mask, test_mask = split.masks(data)
                    train, validation, test = (
                        subset(data, train_mask),
                        subset(data, validation_mask),
                        subset(data, test_mask),
                    )
                    for method in suite.methods:
                        if method.scenarios is not None and scenario.id not in method.scenarios:
                            continue
                        if method.seeds is not None and seed not in method.seeds:
                            continue
                        if method.views is not None and view not in method.views:
                            continue
                        if method.horizons is not None and horizon not in method.horizons:
                            continue
                        if max_cases is not None and attempted >= max_cases:
                            manifest["status"] = "partial_max_cases"
                            manifest["finished_unix"] = time.time()
                            atomic_json(manifest_path, manifest)
                            return manifest
                        attempted += 1
                        payload = _case_payload(
                            suite,
                            scenario,
                            method,
                            seed,
                            view,
                            horizon,
                            suite_config_digest,
                        )
                        case_id = stable_hash(payload)
                        path = results_dir / f"{case_id}.json"
                        if resume and path.exists():
                            manifest["skipped"] += 1
                            continue
                        enforce_storage_budget(output, suite.resource)
                        before = resource_snapshot()
                        started = time.perf_counter()
                        record = {
                            "case_id": case_id,
                            "status": "running",
                            "case": payload,
                            "split": {
                                "train_groups": split.train_groups,
                                "validation_groups": split.validation_groups,
                                "test_groups": split.test_groups,
                            },
                            "oracle_level": (
                                "latent_confounder_omitted"
                                if view == "complete_state"
                                and float(
                                    getattr(dataset.config, "hidden_driver_strength", 0.0)
                                )
                                > 0
                                else "complete_state_conditional"
                                if view == "complete_state"
                                else "observed_history_predictive"
                            ),
                            "orientation": "[target, source]",
                            "benchmark_source_digest": environment.get(
                                "benchmark_source_digest"
                            ),
                            "sid_neuromod_source_digest": environment.get(
                                "sid_neuromod_source_digest"
                            ),
                            "execution_environment_digest": environment.get(
                                "execution_environment_digest"
                            ),
                        }
                        try:
                            with limited_threads(suite.resource):
                                tuned = tune_method(method, train, validation, seed=seed)
                                metrics, channels = evaluate_estimator(
                                    tuned.estimator,
                                    dataset,
                                    test,
                                    validation=validation,
                                    history_lags=suite.history_lags,
                                    n_samples=suite.n_predictive_samples,
                                    response_panels=response_panels,
                                )
                            record.update(
                                {
                                    "status": "ok",
                                    "selected_params": tuned.params,
                                    "selection_objective": tuned.objective,
                                    "selection_score": tuned.score,
                                    "tuning_candidates": tuned.candidates,
                                    "method_metadata": tuned.estimator.metadata(),
                                    "response_panel_metadata": [
                                        {
                                            "id": panel.identifier,
                                            "specification": asdict(panel.specification),
                                            "expected_effect": panel.expected_effect,
                                            "n_pairs": len(panel.pairs),
                                            "horizon": panel.truth.values.shape[0],
                                            "truth_rms_to_innovation_rms": (
                                                panel.truth_rms_to_innovation_rms
                                            ),
                                            "truth_mc_se_to_truth_rms_ratio": (
                                                panel.truth_mc_se_to_truth_rms_ratio
                                            ),
                                            "transient_decay_estimable_fraction": (
                                                panel.transient_decay_estimable_fraction
                                            ),
                                            "filtering_context_rows_per_pair": int(
                                                np.sum(
                                                    (panel.baseline.groups == 0)
                                                    & (panel.baseline.times < 0)
                                                )
                                            ),
                                        }
                                        for panel in response_panels
                                    ],
                                    "capabilities": asdict(tuned.estimator.capabilities),
                                    "metrics": metrics,
                                }
                            )
                            method_metadata = record["method_metadata"]
                            if (
                                view == "complete_state"
                                and method_metadata.get(
                                    "uses_current_modulator_features"
                                )
                                is False
                            ):
                                record["oracle_level"] = (
                                    "observed_neural_history_predictive_with_latent_modulator"
                                )
                                record["effective_information_set"] = method_metadata.get(
                                    "effective_information_set"
                                )
                            elif view == "complete_state":
                                record["effective_information_set"] = (
                                    "registered_neural_history_stimulus_and_current_modulator"
                                )
                            else:
                                record["effective_information_set"] = (
                                    "registered_observed_history_and_stimulus"
                                )
                            if suite.save_channels:
                                record["channels"] = channels
                            manifest["completed"] += 1
                        except Exception as error:
                            record.update(
                                {
                                    "status": "failed",
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                    "traceback": traceback.format_exc(limit=12),
                                }
                            )
                            manifest["failed"] += 1
                        after = resource_snapshot()
                        record["resources"] = {
                            "wall_seconds": time.perf_counter() - started,
                            "peak_rss_gb": after["peak_rss_gb"],
                            "user_cpu_seconds_delta": after["user_cpu_seconds"]
                            - before["user_cpu_seconds"],
                            "system_cpu_seconds_delta": after["system_cpu_seconds"]
                            - before["system_cpu_seconds"],
                        }
                        atomic_json(path, record)
                        manifest["last_case_id"] = case_id
                        manifest["last_update_unix"] = time.time()
                        manifest["resource_snapshot"] = after
                        atomic_json(manifest_path, manifest)
                        if after["peak_rss_gb"] > suite.resource.max_rss_gb:
                            manifest["status"] = "stopped_memory_safety_gate"
                            manifest["memory_limit_gb"] = suite.resource.max_rss_gb
                            manifest["finished_unix"] = time.time()
                            atomic_json(manifest_path, manifest)
                            return manifest

    manifest["status"] = "complete" if manifest["failed"] == 0 else "complete_with_failures"
    manifest["finished_unix"] = time.time()
    atomic_json(manifest_path, manifest)
    return manifest
