from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark.core import causal_fill
from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.distribution_structure_evaluate import sensitivity_indices
from conditional_neural_benchmark.distribution_structure_scoring import score_samples
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.inference import load_checkpoint
from conditional_neural_benchmark.runner import (
    ModelConfig,
    RunState,
    _neural_trial,
    _split_windows,
)

from .core import (
    HistorySupportModel,
    construct_matched_query_pair,
    direct_importance_sampling,
    effective_sample_size,
    normalize_log_weights,
    progressive_bridge_smc,
)
from .oracles import oracle_controls, oracle_soft_effect
from .protocol import (
    FOLD_RUN,
    ROOT,
    RUN_ROOT,
    atomic_csv,
    atomic_json,
    freeze_protocol,
    sha256,
    update_status,
)


NEW_MODEL_CONFIGS = (
    ModelConfig(
        "query_autoregressive_mdn4",
        "tcn",
        "autoregressive_mdn",
        True,
        width=128,
        dropout=0.15,
        head_params={"hidden": 64, "layers": 2, "components": 4},
        learning_rate=3e-4,
        weight_decay=7.5e-4,
        training_scheme="natural",
        history_noise_std=0.01,
        history_noise_copies=1,
    ),
    ModelConfig(
        "query_conditional_edm_diffusion",
        "tcn",
        "conditional_edm_diffusion",
        True,
        width=128,
        dropout=0.15,
        head_params={
            "hidden": 128,
            "layers": 4,
            "sample_steps": 24,
            "sigma_min": 0.01,
            "sigma_max": 3.0,
            "sigma_data": 0.7,
            "p_mean": -0.8,
            "p_std": 1.2,
            "rho": 7.0,
        },
        learning_rate=3e-4,
        weight_decay=7.5e-4,
        training_scheme="natural",
        history_noise_std=0.01,
        history_noise_copies=1,
    ),
)

FAMILY_ORDER = (
    "flow",
    "autoregressive_mdn4",
    "conditional_edm_diffusion",
    "student_t_rank8",
    "gaussian_rank8",
    "gaussian_diagonal",
)


def keyed_seed(*parts: Any) -> int:
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def _checkpoint_path(run_root: Path, family: str, fold: int, seed: int) -> Path:
    if family == "flow":
        return (
            ROOT
            / "results/conditional_distribution_benchmark/chemical_encoding_primary_20260828_oh16230/checkpoints/chemical_full_cv"
            / f"stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01__L80__f{fold}__s{seed}.pt"
        )
    if family in {"student_t_rank8", "gaussian_rank8", "gaussian_diagonal"}:
        model_id = {
            "student_t_rank8": "matched_student_t_rank8",
            "gaussian_rank8": "matched_gaussian_rank8",
            "gaussian_diagonal": "matched_gaussian_diag",
        }[family]
        return (
            ROOT / "results/distribution_structure_20260831/checkpoints/matched_full_cv"
            / f"{model_id}__L80__f{fold}__s{seed}.pt"
        )
    model_id = {
        "autoregressive_mdn4": "query_autoregressive_mdn4",
        "conditional_edm_diffusion": "query_conditional_edm_diffusion",
    }[family]
    return (
        run_root / "neural_training/checkpoints/matched_primary"
        / f"{model_id}__L80__f{fold}__s{seed}.pt"
    )


def _verify_checkpoint(
    payload: dict,
    cohort,
    scaler,
    fold: int,
    seed: int,
) -> None:
    if int(payload["fold"]) != fold or int(payload["seed"]) != seed:
        raise RuntimeError("checkpoint fold/seed mismatch")
    if payload["neurons"] != list(cohort.neurons) or int(payload["lag"]) != 80:
        raise RuntimeError("checkpoint cohort/history mismatch")
    if not payload["model_config"].get("residual_target", False):
        raise RuntimeError("checkpoint is not a residual model")
    for key in ("mean", "scale"):
        np.testing.assert_array_equal(payload["scaler"][key], getattr(scaler, key))
    if "stimulus_schema_fingerprint" in payload:
        if payload["stimulus_schema_fingerprint"] != cohort.stimulus_schema_fingerprint:
            raise RuntimeError("checkpoint stimulus provenance mismatch")


def train_neural(
    run_root: Path,
    *,
    folds_to_run: list[int] | None = None,
    seeds_to_run: list[int] | None = None,
    smoke: bool = False,
) -> None:
    protocol = freeze_protocol(run_root)
    update_status(run_root, "neural_training", "running")
    torch.set_num_threads(2)
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    training_root = run_root / "neural_training"
    training_root.mkdir(parents=True, exist_ok=True)
    state = RunState(training_root, time.monotonic() + 24 * 3600)
    metrics = training_root / "trial_metrics.csv"
    if metrics.exists():
        state.records = pd.read_csv(metrics).replace({np.nan: None}).to_dict("records")
    selected_folds = [0] if smoke else list(range(5))
    selected_seeds = [1701] if smoke else [1701, 2903]
    if folds_to_run is not None:
        selected_folds = [fold for fold in selected_folds if fold in folds_to_run]
    if seeds_to_run is not None:
        selected_seeds = [seed for seed in selected_seeds if seed in seeds_to_run]
    phase = "matched_smoke" if smoke else "matched_primary"
    for fold in selected_folds:
        scaler, training, validation, testing = _split_windows(cohort, folds, fold, 80)
        for config in NEW_MODEL_CONFIGS:
            for seed in selected_seeds:
                existing = [
                    row
                    for row in state.records
                    if row.get("phase") == phase
                    and row.get("model_id") == config.model_id
                    and int(row.get("fold", -1)) == fold
                    and int(row.get("seed", -1)) == seed
                    and row.get("status") == "ok"
                ]
                if existing:
                    print(f"TRAIN_SKIP family={config.model_id} fold={fold} seed={seed}", flush=True)
                    continue
                print(f"TRAIN_START family={config.model_id} fold={fold} seed={seed}", flush=True)
                _neural_trial(
                    state=state,
                    phase=phase,
                    config=config,
                    cohort=cohort,
                    lag=80,
                    fold=fold,
                    seed=seed,
                    train=training,
                    validation=validation,
                    test=testing,
                    scaler=scaler,
                    device=protocol["resources"]["device"],
                    max_epochs=2 if smoke else 50,
                    patience=2 if smoke else 9,
                    eval_rows=160 if smoke else 1600,
                    n_samples=8 if smoke else 32,
                    batch_size=256,
                    trial_metadata={
                        "cohort_mode": "oh16230_head",
                        "stimulus_encoding": "binary_any_stimulus",
                        "protocol_fingerprint": protocol["fingerprint"],
                    },
                )
                if state.records[-1].get("status") != "ok":
                    raise RuntimeError(f"neural fit failed: {state.records[-1]}")
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    frame = pd.DataFrame(state.records)
    frame = frame[frame.phase == phase].copy()
    manifest_rows = []
    for row in frame.to_dict("records"):
        path = training_root / str(row["checkpoint"])
        manifest_rows.append(
            {
                "dataset": "neuropal_oh16230",
                "model_family": (
                    "autoregressive_mdn4"
                    if row["model_id"] == "query_autoregressive_mdn4"
                    else "conditional_edm_diffusion"
                ),
                "model_id": row["model_id"],
                "fold": int(row["fold"]),
                "model_seed": int(row["seed"]),
                "checkpoint": str(path),
                "sha256": sha256(path),
                "best_epoch": int(row["best_epoch"]),
                "stopped_epoch": int(row["stopped_epoch"]),
                "training_seconds": float(row["train_seconds"]),
                "parameter_count": int(row["parameter_count"]),
                "status": row["status"],
                "reused_checkpoint": False,
            }
        )
    new_completed = len(manifest_rows)
    reusable_families = ["flow"] if smoke else [
        "flow", "student_t_rank8", "gaussian_rank8", "gaussian_diagonal"
    ]
    for family in reusable_families:
        for fold in selected_folds:
            for seed in selected_seeds:
                path = _checkpoint_path(run_root, family, fold, seed)
                manifest_rows.append(
                    {
                        "dataset": "neuropal_oh16230",
                        "model_family": family,
                        "model_id": family,
                        "fold": fold,
                        "model_seed": seed,
                        "checkpoint": str(path),
                        "sha256": sha256(path),
                        "best_epoch": np.nan,
                        "stopped_epoch": np.nan,
                        "training_seconds": np.nan,
                        "parameter_count": np.nan,
                        "status": "reused_frozen",
                        "reused_checkpoint": True,
                    }
                )
    atomic_csv(run_root / "model_manifest.csv", pd.DataFrame(manifest_rows))
    expected = len(selected_folds) * len(selected_seeds) * len(NEW_MODEL_CONFIGS)
    completed = new_completed
    update_status(
        run_root,
        "neural_training",
        "complete" if completed >= expected else "partial",
        expected=expected,
        completed=completed,
    )


def run_analytic_controls(run_root: Path, *, smoke: bool = False) -> None:
    protocol = freeze_protocol(run_root)
    update_status(run_root, "analytic_controls", "running")
    controls = oracle_controls()
    amplitudes = [0.5, 2.0] if smoke else [0.5, 2.0, 3.5]
    particle_counts = [128, 256] if smoke else protocol["smc"]["particle_counts"]
    sampling_seeds = protocol["smc"]["sampling_seeds"][:2] if smoke else protocol["smc"]["sampling_seeds"]
    references: list[dict] = []
    results: list[dict] = []
    diagnostics: list[dict] = []
    for control in controls:
        for high_target in amplitudes:
            low_target = -0.5
            bandwidth = 0.35
            truth, truth_mcse, rarity = oracle_soft_effect(
                control,
                history=0.4,
                low_target=low_target,
                high_target=high_target,
                bandwidth=bandwidth,
                n_reference=80_000 if smoke else protocol["synthetic"]["analytic_oracle_reference_draws"],
                seed=keyed_seed("oracle", control.name, high_target),
            )
            query_class = (
                "common" if high_target <= 0.5 else "rare" if high_target <= 2 else "extreme"
            )
            references.append(
                {
                    "run_id": run_root.name,
                    "dataset": "analytic_control",
                    "system": control.name,
                    "generator_seed": 0,
                    "query_id": f"{control.name}__a{high_target:g}",
                    "query_class": query_class,
                    "source": control.source,
                    "target": control.target,
                    "low_target": low_target,
                    "high_target": high_target,
                    "bandwidth": bandwidth,
                    "oracle_effect": truth,
                    "oracle_mcse": truth_mcse,
                    "event_rarity": rarity,
                    "reference_draws": 80_000 if smoke else protocol["synthetic"]["analytic_oracle_reference_draws"],
                }
            )

            def sampler(n: int, seed: int, control=control) -> np.ndarray:
                return control.sample(n, seed, 0.4)

            def statistic(draw: np.ndarray, control=control) -> np.ndarray:
                return draw[:, control.target]

            for n_particles in particle_counts:
                for sampling_seed in sampling_seeds:
                    arm_results: dict[tuple[str, str], Any] = {}
                    for arm, requested in (("low", low_target), ("high", high_target)):
                        potential = lambda draw, requested=requested, control=control: -0.5 * np.square(
                            (draw[:, control.source] - requested) / bandwidth
                        )
                        progressive = progressive_bridge_smc(
                            sampler,
                            potential,
                            statistic,
                            n_particles=n_particles,
                            seed=keyed_seed(control.name, high_target, n_particles, sampling_seed, arm, "smc"),
                            tempering_ess_fraction=0.70,
                            resample_ess_fraction=0.70,
                            rejuvenation_steps=2,
                        )
                        direct = direct_importance_sampling(
                            sampler,
                            potential,
                            statistic,
                            n_particles=n_particles,
                            seed=keyed_seed(control.name, high_target, n_particles, sampling_seed, arm, "direct"),
                        )
                        matched = direct_importance_sampling(
                            sampler,
                            potential,
                            statistic,
                            n_particles=progressive.model_evaluations,
                            seed=keyed_seed(control.name, high_target, n_particles, sampling_seed, arm, "matched"),
                        )
                        for method, value in (
                            ("progressive_bridge", progressive),
                            ("direct_importance", direct),
                            ("direct_importance_cost_matched", matched),
                        ):
                            arm_results[(method, arm)] = value
                        for stage_row in progressive.stages.to_dict("records"):
                            diagnostics.append(
                                {
                                    "run_id": run_root.name,
                                    "dataset": "analytic_control",
                                    "system": control.name,
                                    "generator_seed": 0,
                                    "model_family": "oracle",
                                    "model_seed": 0,
                                    "sampling_seed": sampling_seed,
                                    "query_id": f"{control.name}__a{high_target:g}",
                                    "query_class": query_class,
                                    "source_neuron": control.source,
                                    "target_functional": f"coordinate_{control.target}_mean",
                                    "amplitude": high_target,
                                    "history_support_value": 1.0,
                                    "event_rarity": progressive.event_probability,
                                    "particle_count": n_particles,
                                    "smc_method": "progressive_bridge",
                                    "arm": arm,
                                    **stage_row,
                                }
                            )
                    for method in (
                        "progressive_bridge",
                        "direct_importance",
                        "direct_importance_cost_matched",
                    ):
                        low = arm_results[(method, "low")]
                        high = arm_results[(method, "high")]
                        estimate = high.estimate - low.estimate
                        results.append(
                            {
                                "run_id": run_root.name,
                                "dataset": "analytic_control",
                                "system": control.name,
                                "generator_seed": 0,
                                "worm_id": "",
                                "fold": -1,
                                "model_family": "oracle",
                                "model_seed": 0,
                                "sampling_seed": sampling_seed,
                                "query_id": f"{control.name}__a{high_target:g}",
                                "query_class": query_class,
                                "source_neuron": control.source,
                                "target_functional": f"coordinate_{control.target}_mean",
                                "amplitude": high_target,
                                "history_support_value": 1.0,
                                "event_rarity": min(low.event_probability, high.event_probability),
                                "particle_count": n_particles,
                                "smc_method": method,
                                "effect_estimate": estimate,
                                "monte_carlo_error": math.sqrt(low.mcse**2 + high.mcse**2),
                                "oracle_effect": truth,
                                "absolute_error": abs(estimate - truth),
                                "minimum_ess_fraction": min(low.minimum_ess_fraction, high.minimum_ess_fraction),
                                "final_ess_fraction": min(low.final_ess_fraction, high.final_ess_fraction),
                                "max_normalized_weight": max(low.final_max_weight, high.final_max_weight),
                                "resampling_events": low.resampling_events + high.resampling_events,
                                "unique_ancestor_fraction": min(low.unique_ancestors, high.unique_ancestors) / n_particles,
                                "rejuvenation_acceptance": (
                                    float(np.mean([
                                        value for value in (
                                            low.rejuvenation_acceptance,
                                            high.rejuvenation_acceptance,
                                        ) if np.isfinite(value)
                                    ]))
                                    if any(np.isfinite(value) for value in (
                                        low.rejuvenation_acceptance,
                                        high.rejuvenation_acceptance,
                                    )) else np.nan
                                ),
                                "model_evaluations": low.model_evaluations + high.model_evaluations,
                                "healthy_smc": bool(
                                    low.finite and high.finite
                                    and min(low.minimum_ess_fraction, high.minimum_ess_fraction) >= 0.20
                                    and min(low.unique_ancestors, high.unique_ancestors) >= 0.10 * n_particles
                                ),
                            }
                        )
            print(f"ANALYTIC_DONE system={control.name} amplitude={high_target}", flush=True)
    atomic_csv(run_root / "synthetic_oracle_effects.csv", pd.DataFrame(references))
    atomic_csv(run_root / "analytic_smc_runs.csv", pd.DataFrame(results))
    atomic_csv(run_root / "analytic_smc_stages.csv", pd.DataFrame(diagnostics))
    update_status(
        run_root,
        "analytic_controls",
        "complete",
        oracle_queries=len(references),
        estimator_rows=len(results),
    )


def _nearest_row(rows: np.ndarray, values: np.ndarray, requested: float) -> int:
    return int(rows[np.argmin(np.abs(values[rows] - requested))])


def build_support_and_queries(run_root: Path, *, smoke: bool = False) -> None:
    protocol = freeze_protocol(run_root)
    update_status(run_root, "support_queries", "running")
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    support_rows: list[dict] = []
    query_rows: list[dict] = []
    bank_directory = run_root / "query_banks"
    bank_directory.mkdir(exist_ok=True)
    selected_folds = [0] if smoke else list(range(5))
    for fold in selected_folds:
        scaler, training, validation, testing = _split_windows(cohort, folds, fold, 80)
        rng = np.random.default_rng(keyed_seed("support", fold))
        train_index = np.arange(len(training.target))
        calibration_index = np.arange(len(validation.target))
        if len(train_index) > (800 if smoke else 5000):
            train_index = np.sort(rng.choice(train_index, 800 if smoke else 5000, replace=False))
        if len(calibration_index) > (300 if smoke else 1500):
            calibration_index = np.sort(rng.choice(calibration_index, 300 if smoke else 1500, replace=False))
        support = HistorySupportModel(
            k=protocol["support"]["k"],
            pca_cap=16 if smoke else protocol["support"]["pca_cap"],
            seed=keyed_seed("support_model", fold),
        ).fit(
            training.history[train_index, :, : cohort.n_neurons],
            training.history[train_index, :, cohort.n_neurons :],
            validation.history[calibration_index, :, : cohort.n_neurons],
            validation.history[calibration_index, :, cohort.n_neurons :],
            max_train_rows=800 if smoke else 5000,
        )
        heldout_index = sensitivity_indices(
            testing,
            2 if smoke else protocol["predictive_evaluation"]["heldout_rows_per_worm_stratum"],
        )
        factual_profile = support.score(
            testing.history[heldout_index, :, : cohort.n_neurons],
            testing.history[heldout_index, :, cohort.n_neurons :],
        )
        for position, row_index in enumerate(heldout_index):
            support_rows.append(
                {
                    "run_id": run_root.name,
                    "fold": fold,
                    "worm_id": cohort.worm_ids[int(testing.worm[row_index])],
                    "row_index": int(row_index),
                    "row_kind": "heldout_factual",
                    "query_id": "",
                    **factual_profile.iloc[position].to_dict(),
                }
            )

        current_train = training.history[:, -1, : cohort.n_neurons]
        source_scale = current_train.std(axis=0)
        ranks = np.argsort(source_scale)
        selected_sources = ranks[np.linspace(4, len(ranks) - 5, 6, dtype=int)]
        query_ids: list[str] = []
        query_histories: list[np.ndarray] = []
        factual_histories: list[np.ndarray] = []
        event_low: list[float] = []
        event_high: list[float] = []
        event_bandwidth: list[float] = []
        source_store: list[int] = []
        worm_store: list[int] = []
        anchor_time_store: list[int] = []
        class_store: list[str] = []
        for worm_position, worm_index in enumerate(np.unique(testing.worm)):
            rows = np.flatnonzero(testing.worm == worm_index)
            source = int(selected_sources[worm_position % len(selected_sources)])
            train_current = current_train[:, source]
            q50, q90, q95, q99 = np.quantile(train_current, [0.50, 0.90, 0.95, 0.99])
            center = float(np.median(train_current))
            deviation_max = float(np.max(np.abs(train_current - center)))
            common_row = _nearest_row(rows, testing.history[:, -1, source], q50)
            rare_row = _nearest_row(rows, testing.history[:, -1, source], q99)
            factual_common = testing.history[common_row].astype(np.float64)
            factual_rare = testing.history[rare_row].astype(np.float64)
            stimulus = factual_common[:, cohort.n_neurons :]
            smooth_target = center + 1.10 * deviation_max
            extreme_target = center + 1.25 * deviation_max
            pair = construct_matched_query_pair(
                factual_common[:, : cohort.n_neurons],
                stimulus,
                source=source,
                target_value=smooth_target,
            )
            extreme_pair = construct_matched_query_pair(
                factual_common[:, : cohort.n_neurons],
                stimulus,
                source=source,
                target_value=extreme_target,
            )
            matching = rows[
                testing.history[rows, -1, cohort.n_neurons]
                == factual_common[-1, cohort.n_neurons]
            ]
            if len(matching) < 2:
                matching = rows
            terminal = testing.history[matching, -1, : cohort.n_neurons]
            donor_row = int(matching[np.argmax(np.linalg.norm(terminal - factual_common[-1, :cohort.n_neurons], axis=1))])
            population = testing.history[donor_row, :, : cohort.n_neurons].astype(np.float64).copy()
            population[:, source] = pair.coherent[:, source]
            population_full = np.concatenate([population, stimulus], axis=1)
            extreme_population = population.copy()
            extreme_population[:, source] = extreme_pair.incoherent[:, source]
            extreme_population_full = np.concatenate([extreme_population, stimulus], axis=1)
            candidates = [
                ("supported_history_common_event", factual_common, factual_common, q50, int(testing.time[common_row])),
                ("supported_history_rare_event", factual_common, factual_common, q99, int(testing.time[common_row])),
                ("naturally_rare_coherent_history", factual_rare, factual_rare, q95, int(testing.time[rare_row])),
                (
                    "extrapolated_dynamically_coherent_history",
                    np.concatenate([pair.coherent, stimulus], axis=1),
                    factual_common,
                    smooth_target,
                    int(testing.time[common_row]),
                ),
                (
                    "amplitude_matched_temporally_incoherent_history",
                    np.concatenate([pair.incoherent, stimulus], axis=1),
                    factual_common,
                    smooth_target,
                    int(testing.time[common_row]),
                ),
                ("population_incoherent_history", population_full, factual_common, smooth_target, int(testing.time[common_row])),
                ("extreme_incoherent_history", extreme_population_full, factual_common, extreme_target, int(testing.time[common_row])),
            ]
            absolute_next = training.target[:, source]
            low = float(np.quantile(absolute_next, 0.25))
            iqr = float(np.quantile(absolute_next, 0.75) - np.quantile(absolute_next, 0.25))
            for query_class, query_history, factual_history, requested, anchor_time in candidates:
                query_id = f"f{fold}_w{int(worm_index):02d}_s{source:02d}_{query_class}"
                profile = support.score(
                    query_history[None, :, : cohort.n_neurons],
                    query_history[None, :, cohort.n_neurons :],
                    source=source,
                ).iloc[0]
                perturbation_norm = float(
                    np.linalg.norm(
                        query_history[:, : cohort.n_neurons]
                        - factual_history[:, : cohort.n_neurons]
                    )
                )
                query_rows.append(
                    {
                        "run_id": run_root.name,
                        "dataset": "neuropal_oh16230",
                        "generator_seed": -1,
                        "worm_id": cohort.worm_ids[int(worm_index)],
                        "worm_index": int(worm_index),
                        "fold": fold,
                        "anchor_time": anchor_time,
                        "query_id": query_id,
                        "query_class": query_class,
                        "source_neuron": cohort.neurons[source],
                        "source_index": source,
                        "target_functional": "population_mean_excluding_source",
                        "amplitude": float(requested),
                        "perturbation_norm": perturbation_norm,
                        "event_low": low,
                        "event_high": float(requested),
                        "event_bandwidth": max(0.10, 0.25 * iqr),
                        "stimulus_context": str(testing.stratum[common_row]),
                        "construction": query_class,
                        **profile.to_dict(),
                    }
                )
                support_rows.append(
                    {
                        "run_id": run_root.name,
                        "fold": fold,
                        "worm_id": cohort.worm_ids[int(worm_index)],
                        "row_index": -1,
                        "row_kind": "query",
                        "query_id": query_id,
                        **profile.to_dict(),
                    }
                )
                query_ids.append(query_id)
                query_histories.append(query_history.astype(np.float32))
                factual_histories.append(factual_history.astype(np.float32))
                event_low.append(low)
                event_high.append(float(requested))
                event_bandwidth.append(max(0.10, 0.25 * iqr))
                source_store.append(source)
                worm_store.append(int(worm_index))
                anchor_time_store.append(anchor_time)
                class_store.append(query_class)
        np.savez_compressed(
            bank_directory / f"fold_{fold}.npz",
            query_id=np.asarray(query_ids),
            query_history=np.asarray(query_histories, dtype=np.float32),
            factual_history=np.asarray(factual_histories, dtype=np.float32),
            event_low=np.asarray(event_low),
            event_high=np.asarray(event_high),
            event_bandwidth=np.asarray(event_bandwidth),
            source_index=np.asarray(source_store, dtype=np.int16),
            worm_index=np.asarray(worm_store, dtype=np.int16),
            anchor_time=np.asarray(anchor_time_store, dtype=np.int32),
            query_class=np.asarray(class_store),
        )
        print(f"SUPPORT_DONE fold={fold} queries={len(query_ids)}", flush=True)
    manifest = pd.DataFrame(query_rows)
    primary_ids = (
        manifest.sort_values(["query_class", "fold", "worm_index"])
        .groupby("query_class", sort=False)
        .head(1)
        .query_id
    )
    manifest["smc_primary"] = manifest.query_id.isin(set(primary_ids))
    atomic_csv(run_root / "query_manifest.csv", manifest)
    atomic_csv(run_root / "support_calibration.csv", pd.DataFrame(support_rows))
    update_status(
        run_root,
        "support_queries",
        "complete",
        query_count=len(manifest),
        smc_primary_count=int(manifest.smc_primary.sum()),
    )


def _sample_model(
    model,
    device: torch.device,
    histories: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    residual_target: bool,
    n_neurons: int,
    chunk: int,
) -> np.ndarray:
    result = []
    for start in range(0, len(histories), chunk):
        part = histories[start : start + chunk]
        tensor = torch.as_tensor(part.reshape(len(part), -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            draw = model.sample(tensor, n_samples, seed=seed + start).cpu().numpy()
        if residual_target:
            draw = draw + part[:, None, -1, :n_neurons]
        result.append(draw)
    return np.concatenate(result)


def evaluate_predictive(run_root: Path, *, smoke: bool = False) -> None:
    protocol = freeze_protocol(run_root)
    update_status(run_root, "predictive_evaluation", "running")
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    archive_dir = run_root / "predictive_archives"
    archive_dir.mkdir(exist_ok=True)
    selected_folds = [0] if smoke else list(range(5))
    families = FAMILY_ORDER[:3] if smoke else FAMILY_ORDER
    model_seeds = [1701] if smoke else protocol["models"]["model_seeds"]
    sample_seeds = [731] if smoke else protocol["predictive_evaluation"]["sampling_seeds"]
    sample_count = 16 if smoke else protocol["predictive_evaluation"]["sample_count"]
    output_rows: list[dict] = []
    sampler_rows: list[dict] = []
    for fold in selected_folds:
        scaler, training, _, testing = _split_windows(cohort, folds, fold, 80)
        selected_index = sensitivity_indices(
            testing, 2 if smoke else protocol["predictive_evaluation"]["heldout_rows_per_worm_stratum"]
        )
        selected = testing.take(selected_index)
        innovation = training.target - training.history[:, -1, : cohort.n_neurons]
        threshold = 2 * np.maximum(
            np.sqrt(np.mean(innovation.astype(float) ** 2, axis=0)), 1e-3
        )
        for family in families:
            for model_seed in model_seeds:
                checkpoint = _checkpoint_path(run_root, family, fold, model_seed)
                if smoke and family in {"autoregressive_mdn4", "conditional_edm_diffusion"}:
                    checkpoint = Path(str(checkpoint).replace("/matched_primary/", "/matched_smoke/"))
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                model, payload, device = load_checkpoint(checkpoint, protocol["resources"]["device"])
                _verify_checkpoint(payload, cohort, scaler, fold, model_seed)
                chunk = 2 if family == "autoregressive_mdn4" else 8
                for sample_seed in sample_seeds:
                    archive = archive_dir / f"{family}__f{fold}__m{model_seed}__s{sample_seed}__N{sample_count}.npz"
                    if archive.exists():
                        with np.load(archive, allow_pickle=False) as saved:
                            scores = saved["scores"]
                            metric_names = list(saved["metric_names"].astype(str))
                            nll = saved["nll"]
                            wall = float(saved["wall_seconds"])
                    else:
                        started = time.perf_counter()
                        history = selected.history
                        tensor_target = selected.target - history[:, -1, : cohort.n_neurons]
                        draws = _sample_model(
                            model,
                            device,
                            history,
                            n_samples=sample_count,
                            seed=keyed_seed("predictive", family, fold, model_seed, sample_seed),
                            residual_target=False,
                            n_neurons=cohort.n_neurons,
                            chunk=chunk,
                        )
                        values = score_samples(
                            draws,
                            tensor_target,
                            threshold,
                            variogram_offset=history[:, -1, : cohort.n_neurons],
                        )
                        metric_names = list(values)
                        scores = np.stack([values[name] for name in metric_names], axis=1)
                        nll = np.full(len(history), np.nan)
                        if model.capabilities.normalized_density:
                            pieces = []
                            with torch.no_grad():
                                for start in range(0, len(history), chunk):
                                    part = history[start : start + chunk]
                                    h = torch.as_tensor(part.reshape(len(part), -1), dtype=torch.float32, device=device)
                                    y = torch.as_tensor(tensor_target[start : start + chunk], dtype=torch.float32, device=device)
                                    pieces.append((-model.log_prob(y, h) / cohort.n_neurons).cpu().numpy())
                            nll = np.concatenate(pieces)
                        wall = time.perf_counter() - started
                        temporary = archive.with_suffix(".tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(
                                handle,
                                scores=scores,
                                metric_names=np.asarray(metric_names),
                                nll=nll,
                                worm=selected.worm,
                                stratum=selected.stratum,
                                history_index=selected_index,
                                wall_seconds=np.asarray(wall),
                                checkpoint_sha256=np.asarray(sha256(checkpoint)),
                                protocol_fingerprint=np.asarray(protocol["fingerprint"]),
                            )
                        os.replace(temporary, archive)
                    for worm_index in np.unique(selected.worm):
                        mask = selected.worm == worm_index
                        row = {
                            "run_id": run_root.name,
                            "dataset": "neuropal_oh16230",
                            "generator_seed": -1,
                            "worm_id": cohort.worm_ids[int(worm_index)],
                            "fold": fold,
                            "model_family": family,
                            "model_seed": model_seed,
                            "sampling_seed": sample_seed,
                            "query_id": "factual_prediction",
                            "query_class": "heldout_factual",
                            "source_neuron": "",
                            "target_functional": "joint_next_state",
                            "amplitude": np.nan,
                            "history_support_value": np.nan,
                            "event_rarity": np.nan,
                            "particle_count": sample_count,
                            "smc_method": "ancestral",
                            "n_histories": int(mask.sum()),
                            "nll_per_neuron": float(np.nanmean(nll[mask])) if np.isfinite(nll[mask]).any() else np.nan,
                            "sampling_throughput_histories_per_second": len(selected_index) / wall,
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": sha256(checkpoint),
                            "training_seconds": float(payload["fit_trace"]["wall_seconds"]),
                            "parameter_count": int(model.parameter_count),
                        }
                        for metric_index, metric in enumerate(metric_names):
                            row[metric] = float(scores[mask, metric_index].mean())
                        output_rows.append(row)
                    print(
                        f"PREDICTIVE_DONE family={family} fold={fold} model_seed={model_seed} sample_seed={sample_seed}",
                        flush=True,
                    )
                if family in {"flow", "conditional_edm_diffusion"} and model_seed == 1701:
                    head = getattr(model, "head", model)
                    if not hasattr(head, "sample_steps"):
                        raise RuntimeError(f"{family} checkpoint has no configurable sampler step count")
                    original_steps = int(head.sample_steps)
                    for steps in ([12] if smoke else [12, 24, 48]):
                        head.sample_steps = int(steps)
                        started = time.perf_counter()
                        history = selected.history
                        target = selected.target - history[:, -1, :cohort.n_neurons]
                        draws = _sample_model(
                            model,
                            device,
                            history,
                            n_samples=sample_count,
                            seed=keyed_seed("sampler_sensitivity", family, fold, steps),
                            residual_target=False,
                            n_neurons=cohort.n_neurons,
                            chunk=chunk,
                        )
                        values = score_samples(
                            draws, target, threshold,
                            variogram_offset=history[:, -1, :cohort.n_neurons],
                        )
                        for worm_index in np.unique(selected.worm):
                            mask = selected.worm == worm_index
                            sampler_rows.append({
                                "run_id": run_root.name,
                                "dataset": "neuropal_oh16230",
                                "fold": fold,
                                "worm_id": cohort.worm_ids[int(worm_index)],
                                "model_family": family,
                                "model_seed": model_seed,
                                "sampling_seed": keyed_seed("sampler_sensitivity", family, fold, steps),
                                "sampler_steps": steps,
                                "particle_count": sample_count,
                                "wall_seconds": time.perf_counter() - started,
                                **{name: float(metric[mask].mean()) for name, metric in values.items()},
                            })
                    head.sample_steps = original_steps
                del model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    frame = pd.DataFrame(output_rows)
    atomic_csv(run_root / "predictive_scores.csv", frame)
    atomic_csv(run_root / "sampler_sensitivity.csv", pd.DataFrame(sampler_rows))
    update_status(run_root, "predictive_evaluation", "complete", rows=len(frame))


def run_neural_query_effects(run_root: Path, *, smoke: bool = False) -> None:
    protocol = freeze_protocol(run_root)
    update_status(run_root, "neural_query_effects", "running")
    manifest = pd.read_csv(run_root / "query_manifest.csv")
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    families = FAMILY_ORDER[:3] if smoke else FAMILY_ORDER
    model_seeds = [1701] if smoke else protocol["models"]["model_seeds"]
    sample_seeds = [731] if smoke else protocol["predictive_evaluation"]["sampling_seeds"]
    direct_n = 32 if smoke else 128
    effect_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for fold in ([0] if smoke else range(5)):
        bank_path = run_root / f"query_banks/fold_{fold}.npz"
        with np.load(bank_path, allow_pickle=False) as bank:
            query_ids = bank["query_id"].astype(str)
            query_history = bank["query_history"]
            factual_history = bank["factual_history"]
            source_index = bank["source_index"].astype(int)
            event_low = bank["event_low"]
            event_high = bank["event_high"]
            bandwidth = bank["event_bandwidth"]
            worm_index = bank["worm_index"].astype(int)
            query_class = bank["query_class"].astype(str)
        scaler, _, _, _ = _split_windows(cohort, folds, fold, 80)
        metadata = manifest.set_index("query_id").loc[query_ids]
        for family in families:
            for model_seed in model_seeds:
                checkpoint = _checkpoint_path(run_root, family, fold, model_seed)
                if smoke and family in {"autoregressive_mdn4", "conditional_edm_diffusion"}:
                    checkpoint = Path(str(checkpoint).replace("/matched_primary/", "/matched_smoke/"))
                model, payload, device = load_checkpoint(checkpoint, protocol["resources"]["device"])
                _verify_checkpoint(payload, cohort, scaler, fold, model_seed)
                chunk = 1 if family == "autoregressive_mdn4" else 8
                for sample_seed in sample_seeds:
                    seed = keyed_seed("history_effect", family, fold, model_seed, sample_seed)
                    queried = _sample_model(
                        model,
                        device,
                        query_history,
                        n_samples=direct_n,
                        seed=seed,
                        residual_target=True,
                        n_neurons=cohort.n_neurons,
                        chunk=chunk,
                    )
                    factual = _sample_model(
                        model,
                        device,
                        factual_history,
                        n_samples=direct_n,
                        seed=seed,
                        residual_target=True,
                        n_neurons=cohort.n_neurons,
                        chunk=chunk,
                    )
                    for position, query_id in enumerate(query_ids):
                        keep = np.arange(cohort.n_neurons) != source_index[position]
                        per_particle = queried[position][:, keep].mean(axis=1) - factual[position][:, keep].mean(axis=1)
                        effect_rows.append(
                            {
                                "run_id": run_root.name,
                                "dataset": "neuropal_oh16230",
                                "system": "observed_neural",
                                "generator_seed": -1,
                                "worm_id": cohort.worm_ids[worm_index[position]],
                                "fold": fold,
                                "model_family": family,
                                "model_seed": model_seed,
                                "sampling_seed": sample_seed,
                                "query_id": query_id,
                                "query_class": query_class[position],
                                "source_neuron": cohort.neurons[source_index[position]],
                                "target_functional": "population_mean_excluding_source",
                                "amplitude": float(metadata.loc[query_id, "amplitude"]),
                                "history_support_value": float(metadata.loc[query_id, "history_support_value"]),
                                "event_rarity": np.nan,
                                "particle_count": direct_n,
                                "smc_method": "paired_ancestral_history_contrast",
                                "effect_estimate": float(per_particle.mean()),
                                "monte_carlo_error": float(per_particle.std(ddof=1) / np.sqrt(direct_n)),
                                "minimum_ess_fraction": 1.0,
                                "healthy_smc": np.nan,
                            }
                        )
                        statistic = queried[position][:, keep].mean(axis=1).astype(np.float64)
                        arm_estimates, arm_variances, arm_ess, arm_mass = {}, {}, {}, {}
                        for arm, requested in (("low", event_low[position]), ("high", event_high[position])):
                            log_weight = -0.5 * np.square(
                                (queried[position, :, source_index[position]] - requested)
                                / bandwidth[position]
                            )
                            weight, log_total = normalize_log_weights(log_weight)
                            arm_estimates[arm] = float(np.sum(weight * statistic))
                            arm_ess[arm] = effective_sample_size(weight)
                            arm_variances[arm] = float(
                                np.sum(weight * np.square(statistic - arm_estimates[arm]))
                                / max(1.0, arm_ess[arm])
                            )
                            arm_mass[arm] = float(np.exp(log_total) / direct_n)
                        effect_rows.append(
                            {
                                "run_id": run_root.name,
                                "dataset": "neuropal_oh16230",
                                "system": "observed_neural",
                                "generator_seed": -1,
                                "worm_id": cohort.worm_ids[worm_index[position]],
                                "fold": fold,
                                "model_family": family,
                                "model_seed": model_seed,
                                "sampling_seed": sample_seed,
                                "query_id": query_id,
                                "query_class": query_class[position],
                                "source_neuron": cohort.neurons[source_index[position]],
                                "target_functional": "soft_event_conditioned_population_mean_excluding_source",
                                "amplitude": float(metadata.loc[query_id, "amplitude"]),
                                "history_support_value": float(metadata.loc[query_id, "history_support_value"]),
                                "event_rarity": min(arm_mass.values()),
                                "particle_count": direct_n,
                                "smc_method": "direct_soft_event",
                                "effect_estimate": arm_estimates["high"] - arm_estimates["low"],
                                "monte_carlo_error": float(np.sqrt(sum(arm_variances.values()))),
                                "minimum_ess_fraction": min(arm_ess.values()) / direct_n,
                                "healthy_smc": np.nan,
                            }
                        )
                if smoke:
                    del model
                    continue
                primary_ids = set(metadata[metadata.smc_primary.astype(bool)].index)
                for position, query_id in enumerate(query_ids):
                    if query_id not in primary_ids or model_seed != 1701:
                        continue
                    history = query_history[position : position + 1]
                    source = int(source_index[position])
                    keep = np.arange(cohort.n_neurons) != source

                    def sample_fn(n: int, draw_seed: int) -> np.ndarray:
                        return _sample_model(
                            model,
                            device,
                            history,
                            n_samples=n,
                            seed=draw_seed,
                            residual_target=True,
                            n_neurons=cohort.n_neurons,
                            chunk=1,
                        )[0]

                    statistic = lambda draw, keep=keep: draw[:, keep].mean(axis=1)
                    for n_particles in protocol["smc"]["particle_counts"]:
                        for sampling_seed in protocol["smc"]["sampling_seeds"]:
                            arms = {}
                            directs = {}
                            for arm, requested in (("low", event_low[position]), ("high", event_high[position])):
                                potential = lambda draw, requested=requested: -0.5 * np.square(
                                    (draw[:, source] - requested) / bandwidth[position]
                                )
                                bridge = progressive_bridge_smc(
                                    sample_fn,
                                    potential,
                                    statistic,
                                    n_particles=n_particles,
                                    seed=keyed_seed("neural_smc", family, query_id, n_particles, sampling_seed, arm),
                                    tempering_ess_fraction=0.70,
                                    resample_ess_fraction=0.70,
                                    rejuvenation_steps=2,
                                )
                                direct = direct_importance_sampling(
                                    sample_fn,
                                    potential,
                                    statistic,
                                    n_particles=n_particles,
                                    seed=keyed_seed("neural_direct", family, query_id, n_particles, sampling_seed, arm),
                                )
                                arms[arm], directs[arm] = bridge, direct
                                for stage in bridge.stages.to_dict("records"):
                                    diagnostic_rows.append(
                                        {
                                            "run_id": run_root.name,
                                            "dataset": "neuropal_oh16230",
                                            "system": "observed_neural",
                                            "generator_seed": -1,
                                            "worm_id": cohort.worm_ids[worm_index[position]],
                                            "fold": fold,
                                            "model_family": family,
                                            "model_seed": model_seed,
                                            "sampling_seed": sampling_seed,
                                            "query_id": query_id,
                                            "query_class": query_class[position],
                                            "source_neuron": cohort.neurons[source],
                                            "target_functional": "population_mean_excluding_source",
                                            "amplitude": float(event_high[position]),
                                            "history_support_value": float(metadata.loc[query_id, "history_support_value"]),
                                            "event_rarity": bridge.event_probability,
                                            "particle_count": n_particles,
                                            "smc_method": "progressive_bridge",
                                            "arm": arm,
                                            **stage,
                                        }
                                    )
                            for method, arm_values in (("progressive_bridge", arms), ("direct_importance", directs)):
                                low, high = arm_values["low"], arm_values["high"]
                                effect_rows.append(
                                    {
                                        "run_id": run_root.name,
                                        "dataset": "neuropal_oh16230",
                                        "system": "observed_neural",
                                        "generator_seed": -1,
                                        "worm_id": cohort.worm_ids[worm_index[position]],
                                        "fold": fold,
                                        "model_family": family,
                                        "model_seed": model_seed,
                                        "sampling_seed": sampling_seed,
                                        "query_id": query_id,
                                        "query_class": query_class[position],
                                        "source_neuron": cohort.neurons[source],
                                        "target_functional": "soft_event_conditioned_population_mean_excluding_source",
                                        "amplitude": float(event_high[position]),
                                        "history_support_value": float(metadata.loc[query_id, "history_support_value"]),
                                        "event_rarity": min(low.event_probability, high.event_probability),
                                        "particle_count": n_particles,
                                        "smc_method": method,
                                        "effect_estimate": high.estimate - low.estimate,
                                        "monte_carlo_error": math.sqrt(low.mcse**2 + high.mcse**2),
                                        "minimum_ess_fraction": min(low.minimum_ess_fraction, high.minimum_ess_fraction),
                                        "final_ess_fraction": min(low.final_ess_fraction, high.final_ess_fraction),
                                        "max_normalized_weight": max(low.final_max_weight, high.final_max_weight),
                                        "unique_ancestor_fraction": min(low.unique_ancestors, high.unique_ancestors) / n_particles,
                                        "healthy_smc": bool(
                                            low.finite and high.finite
                                            and min(low.minimum_ess_fraction, high.minimum_ess_fraction) >= 0.20
                                            and min(low.unique_ancestors, high.unique_ancestors) >= 0.10 * n_particles
                                        ),
                                    }
                                )
                    print(f"NEURAL_SMC_DONE family={family} query={query_id}", flush=True)
                del model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    atomic_csv(run_root / "query_effects.csv", pd.DataFrame(effect_rows))
    atomic_csv(run_root / "smc_diagnostics_neural.csv", pd.DataFrame(diagnostic_rows))
    update_status(run_root, "neural_query_effects", "complete", rows=len(effect_rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "freeze",
            "analytic",
            "train-neural",
            "support",
            "predictive",
            "neural-effects",
        ),
    )
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if args.stage == "freeze":
        freeze_protocol(run_root)
    elif args.stage == "analytic":
        run_analytic_controls(run_root, smoke=args.smoke)
    elif args.stage == "train-neural":
        train_neural(
            run_root,
            folds_to_run=args.folds,
            seeds_to_run=args.seeds,
            smoke=args.smoke,
        )
    elif args.stage == "support":
        build_support_and_queries(run_root, smoke=args.smoke)
    elif args.stage == "predictive":
        evaluate_predictive(run_root, smoke=args.smoke)
    elif args.stage == "neural-effects":
        run_neural_query_effects(run_root, smoke=args.smoke)


if __name__ == "__main__":
    main()
