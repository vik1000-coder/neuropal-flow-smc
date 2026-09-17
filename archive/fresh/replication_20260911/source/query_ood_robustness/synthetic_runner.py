from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from conditional_neural_benchmark import models as _conditional_models  # noqa: F401
from conditional_neural_benchmark.distribution_structure_models import ConditionalElliptical
from conditional_neural_benchmark.distribution_structure_scoring import score_samples
from history_tangent_benchmark.models import build_model, fit_model, resolve_device

from .core import direct_importance_sampling, progressive_bridge_smc
from .protocol import (
    RUN_ROOT,
    atomic_csv,
    freeze_protocol,
    sha256,
    update_status,
)
from .study import keyed_seed
from .synthetic import FitzHughNagumoNetwork, NonlinearVAR, VAR_MECHANISMS


SYNTHETIC_MODELS: dict[str, tuple[str, dict[str, Any]]] = {
    "flow": (
        "conditional_flow_matching",
        {"hidden": 48, "layers": 2, "sample_steps": 12},
    ),
    "autoregressive_mdn4": (
        "autoregressive_mdn",
        {"hidden": 48, "layers": 1, "components": 4},
    ),
    "conditional_edm_diffusion": (
        "conditional_edm_diffusion",
        {
            "hidden": 48,
            "layers": 2,
            "sample_steps": 12,
            "sigma_min": 0.01,
            "sigma_max": 2.5,
            "sigma_data": 0.7,
        },
    ),
    "gaussian_diagonal": (
        "heteroscedastic_gaussian",
        {"hidden": 48, "layers": 2},
    ),
    "student_t_rank2": (
        "structure_student_t",
        {"hidden": 48, "layers": 2, "rank": 2},
    ),
}


def _build_model(family: str, q: int, dy: int):
    head, params = SYNTHETIC_MODELS[family]
    if head == "structure_student_t":
        return ConditionalElliptical(q=q, dy=dy, student=True, **params)
    return build_model(head, q=q, dy=dy, params=params)


def _standardize(dataset) -> dict[str, np.ndarray]:
    h_mean = dataset.train_h.mean(axis=0, dtype=np.float64)
    h_scale = dataset.train_h.std(axis=0, dtype=np.float64)
    h_scale = np.where(h_scale > 1e-5, h_scale, 1.0)
    y_mean = dataset.train_y.mean(axis=0, dtype=np.float64)
    y_scale = dataset.train_y.std(axis=0, dtype=np.float64)
    y_scale = np.where(y_scale > 1e-5, y_scale, 1.0)
    return {
        "h_mean": h_mean,
        "h_scale": h_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def _checkpoint_path(run_root: Path, dataset, family: str, model_seed: int) -> Path:
    return (
        run_root
        / "synthetic_checkpoints"
        / dataset.kind
        / f"g{dataset.generator_seed:02d}__{dataset.mechanism}__{family}__m{model_seed}.pt"
    )


def _fit_or_load(
    run_root: Path,
    dataset,
    family: str,
    model_seed: int,
    *,
    device: str,
    smoke: bool,
):
    checkpoint = _checkpoint_path(run_root, dataset, family, model_seed)
    scale = _standardize(dataset)
    model = _build_model(family, dataset.train_h.shape[1], dataset.train_y.shape[1])
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["family"] != family or payload["generator_seed"] != dataset.generator_seed:
            raise RuntimeError("synthetic checkpoint identity mismatch")
        for key in scale:
            np.testing.assert_allclose(payload["scale"][key], scale[key], rtol=0, atol=0)
        model.load_state_dict(payload["state_dict"])
        resolved = resolve_device(device)
        model.to(resolved).eval()
        return model, resolved, payload, checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_h = (dataset.train_h - scale["h_mean"]) / scale["h_scale"]
    validation_h = (dataset.validation_h - scale["h_mean"]) / scale["h_scale"]
    train_y = (dataset.train_y - scale["y_mean"]) / scale["y_scale"]
    validation_y = (dataset.validation_y - scale["y_mean"]) / scale["y_scale"]
    trace = fit_model(
        model,
        train_h.astype(np.float32),
        train_y.astype(np.float32),
        validation_h.astype(np.float32),
        validation_y.astype(np.float32),
        seed=model_seed,
        device=device,
        learning_rate=5e-4,
        weight_decay=5e-4,
        batch_size=256,
        max_epochs=3 if smoke else 20,
        patience=2 if smoke else 5,
        gradient_clip=1.0,
    )
    payload = {
        "format_version": 1,
        "kind": dataset.kind,
        "mechanism": dataset.mechanism,
        "generator_seed": dataset.generator_seed,
        "family": family,
        "model_seed": model_seed,
        "q": dataset.train_h.shape[1],
        "dy": dataset.train_y.shape[1],
        "model_name": SYNTHETIC_MODELS[family][0],
        "model_params": SYNTHETIC_MODELS[family][1],
        "scale": scale,
        "fit_trace": trace.to_dict(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    return model, resolve_device(device), payload, checkpoint


def _sample_model(model, device, payload, history: np.ndarray, n: int, seed: int) -> np.ndarray:
    scale = payload["scale"]
    standardized = (np.asarray(history) - scale["h_mean"]) / scale["h_scale"]
    h = torch.as_tensor(standardized[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        draw = model.sample(h, n, seed=seed)[0].cpu().numpy()
    return draw * scale["y_scale"] + scale["y_mean"]


def _sample_model_batch(model, device, payload, histories: np.ndarray, seed: int) -> np.ndarray:
    scale = payload["scale"]
    standardized = (np.asarray(histories) - scale["h_mean"]) / scale["h_scale"]
    h = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    with torch.no_grad():
        draw = model.sample(h, 1, seed=seed)[:, 0].cpu().numpy()
    return draw * scale["y_scale"] + scale["y_mean"]


def _oracle_effect(
    dataset,
    history: np.ndarray,
    low: float,
    high: float,
    bandwidth: float,
    *,
    n: int,
    seed: int,
) -> tuple[float, float, float]:
    sums = {arm: np.zeros(3) for arm in ("low", "high")}
    remaining = n
    part = 0
    while remaining:
        count = min(100_000, remaining)
        draw = dataset.oracle_conditioned(history, count, seed + part * 104729)
        statistic = draw[:, dataset.target]
        for arm, requested in (("low", low), ("high", high)):
            weight = np.exp(-0.5 * np.square((draw[:, dataset.source] - requested) / bandwidth))
            sums[arm] += [weight.sum(), np.sum(weight * statistic), np.sum(weight * statistic**2)]
        remaining -= count
        part += 1
    estimates, variances, masses = {}, {}, {}
    for arm in ("low", "high"):
        weight, weighted, squared = sums[arm]
        estimates[arm] = weighted / weight
        variance = max(0.0, squared / weight - estimates[arm] ** 2)
        masses[arm] = weight / n
        variances[arm] = variance / max(1.0, n * masses[arm])
    return (
        float(estimates["high"] - estimates["low"]),
        float(np.sqrt(variances["high"] + variances["low"])),
        float(min(masses.values())),
    )


def _support_values(dataset, histories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(dataset.train_h)
    train = scaler.transform(dataset.train_h)
    validation = scaler.transform(dataset.validation_h)
    query = scaler.transform(np.asarray(histories))
    neighbors = NearestNeighbors(n_neighbors=10).fit(train)
    calibration = neighbors.kneighbors(validation, return_distance=True)[0].mean(axis=1)
    distance = neighbors.kneighbors(query, return_distance=True)[0].mean(axis=1)
    support = (1 + (calibration[None] >= distance[:, None]).sum(axis=1)) / (1 + len(calibration))
    return distance, support


def _queries(dataset) -> list[dict[str, Any]]:
    source = dataset.source
    train_source = dataset.train_h[:, source]
    center = float(np.median(train_source))
    maximum_deviation = float(np.max(np.abs(train_source - center)))
    anchor = dataset.anchor_h.astype(np.float64).copy()
    extrapolated = anchor.copy()
    extreme = anchor.copy()
    if dataset.kind == "fitzhugh_nagumo":
        pulse_index = len(anchor) - 1
        maximum = float(dataset.metadata["training_source_pulse_max"])
        extrapolated[pulse_index] = 1.10 * maximum
        extreme[pulse_index] = 1.50 * maximum
    else:
        extrapolated[source] = center + 1.10 * maximum_deviation
        extreme[source] = center + 1.25 * maximum_deviation
    low = float(np.quantile(dataset.train_y[:, source], 0.25))
    q75 = float(np.quantile(dataset.train_y[:, source], 0.75))
    q99 = float(np.quantile(dataset.train_y[:, source], 0.99))
    response_center = float(np.median(dataset.train_y[:, source]))
    response_max = float(np.max(np.abs(dataset.train_y[:, source] - response_center)))
    iqr = float(np.quantile(dataset.train_y[:, source], 0.75) - low)
    bandwidth = max(0.03, 0.25 * iqr)
    result = [
        {"query_class": "supported_common", "history": anchor, "high": q75},
        {"query_class": "supported_rare", "history": anchor, "high": q99},
        {"query_class": "extrapolated_coherent", "history": extrapolated, "high": q99},
        {
            "query_class": "extreme_ood",
            "history": extreme,
            "high": response_center + 1.25 * response_max,
        },
    ]
    distances, supports = _support_values(dataset, [value["history"] for value in result])
    for index, value in enumerate(result):
        value["low"] = low
        value["bandwidth"] = bandwidth
        value["distance"] = float(distances[index])
        value["support"] = float(supports[index])
        value["query_id"] = (
            f"{dataset.kind}__g{dataset.generator_seed:02d}__{dataset.mechanism}__{value['query_class']}"
        )
    return result


def _model_reference_effect(
    model,
    device,
    payload,
    dataset,
    query,
    *,
    n: int,
    seed: int,
) -> tuple[float, float]:
    def sampler(count: int, draw_seed: int) -> np.ndarray:
        return _sample_model(model, device, payload, query["history"], count, draw_seed)

    statistic = lambda draw: draw[:, dataset.target]
    arms = {}
    for arm in ("low", "high"):
        requested = query[arm]
        potential = lambda draw, requested=requested: -0.5 * np.square(
            (draw[:, dataset.source] - requested) / query["bandwidth"]
        )
        arms[arm] = direct_importance_sampling(
            sampler,
            potential,
            statistic,
            n_particles=n,
            seed=keyed_seed(seed, arm),
        )
    return (
        float(arms["high"].estimate - arms["low"].estimate),
        float(np.sqrt(arms["high"].mcse**2 + arms["low"].mcse**2)),
    )


def run_synthetic(run_root: Path, *, smoke: bool = False) -> None:
    protocol = freeze_protocol(run_root)
    update_status(run_root, "synthetic_confirmation", "running")
    torch.set_num_threads(2)
    systems = []
    seeds = [0] if smoke else protocol["synthetic"]["confirmation_generator_seeds"]
    for seed in seeds:
        systems.append(
            NonlinearVAR(20_000 + seed, VAR_MECHANISMS[seed % len(VAR_MECHANISMS)]).dataset(
                900 if smoke else 4200
            )
        )
        systems.append(FitzHughNagumoNetwork(30_000 + seed).dataset(700 if smoke else 3600))
    model_seeds = [7001] if smoke else protocol["synthetic"]["model_seeds"]
    families = list(SYNTHETIC_MODELS)[:2] if smoke else list(SYNTHETIC_MODELS)
    manifest_rows, predictive_rows = [], []
    oracle_rows, model_rows, decomposition_rows, smc_stage_rows = [], [], [], []
    pulse_rows = []
    for dataset in systems:
        manifest_rows.append(
            {
                "run_id": run_root.name,
                "dataset": dataset.kind,
                "system": dataset.mechanism,
                "generator_seed": dataset.generator_seed,
                "dimension": dataset.train_y.shape[1],
                "training_rows": len(dataset.train_h),
                "validation_rows": len(dataset.validation_h),
                "test_rows": len(dataset.test_h),
                "stability_radius": dataset.stability_radius,
                "source_index": dataset.source,
                "target_index": dataset.target,
                "metadata_json": json.dumps(dataset.metadata, sort_keys=True),
            }
        )
        queries = _queries(dataset)
        oracle_reference = 20_000 if smoke else (
            protocol["synthetic"]["var_oracle_reference_draws"]
            if dataset.kind == "nonlinear_var"
            else protocol["synthetic"]["fhn_oracle_reference_draws"]
        )
        for query in queries:
            truth, truth_mcse, rarity = _oracle_effect(
                dataset,
                query["history"],
                query["low"],
                query["high"],
                query["bandwidth"],
                n=oracle_reference,
                seed=keyed_seed("synthetic_oracle", query["query_id"]),
            )
            query["oracle_effect"] = truth
            query["oracle_mcse"] = truth_mcse
            query["oracle_rarity"] = rarity
            oracle_rows.append(
                {
                    "run_id": run_root.name,
                    "dataset": dataset.kind,
                    "system": dataset.mechanism,
                    "generator_seed": dataset.generator_seed,
                    "query_id": query["query_id"],
                    "query_class": query["query_class"],
                    "source": dataset.source,
                    "target": dataset.target,
                    "low_target": query["low"],
                    "high_target": query["high"],
                    "bandwidth": query["bandwidth"],
                    "history_support_value": query["support"],
                    "nearest_history_distance": query["distance"],
                    "oracle_effect": truth,
                    "oracle_mcse": truth_mcse,
                    "event_rarity": rarity,
                    "reference_draws": oracle_reference,
                }
            )
        for family in families:
            for model_seed in model_seeds:
                started = time.perf_counter()
                model, device, payload, checkpoint = _fit_or_load(
                    run_root,
                    dataset,
                    family,
                    model_seed,
                    device=protocol["resources"]["device"],
                    smoke=smoke,
                )
                manifest_rows.append(
                    {
                        "run_id": run_root.name,
                        "dataset": dataset.kind,
                        "system": dataset.mechanism,
                        "generator_seed": dataset.generator_seed,
                        "model_family": family,
                        "model_seed": model_seed,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "training_seconds": payload["fit_trace"]["wall_seconds"],
                        "best_epoch": payload["fit_trace"]["best_epoch"],
                        "parameter_count": sum(value.numel() for value in model.parameters()),
                    }
                )
                test_index = np.linspace(0, len(dataset.test_h) - 1, min(256, len(dataset.test_h)), dtype=int)
                test_h = dataset.test_h[test_index]
                test_y = dataset.test_y[test_index]
                draws = []
                for row, history in enumerate(test_h):
                    draws.append(
                        _sample_model(
                            model,
                            device,
                            payload,
                            history,
                            16 if smoke else 64,
                            keyed_seed("synthetic_predictive", dataset.generator_seed, family, model_seed, row),
                        )
                    )
                draws = np.asarray(draws)
                predictive = score_samples(draws, test_y)
                predictive_rows.append(
                    {
                        "run_id": run_root.name,
                        "dataset": dataset.kind,
                        "system": dataset.mechanism,
                        "generator_seed": dataset.generator_seed,
                        "model_family": family,
                        "model_seed": model_seed,
                        "sampling_seed": 0,
                        "particle_count": draws.shape[1],
                        **{key: float(value.mean()) for key, value in predictive.items()},
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                for query in queries:
                    model_truth, model_truth_mcse = _model_reference_effect(
                        model,
                        device,
                        payload,
                        dataset,
                        query,
                        n=2048 if smoke else 8192,
                        seed=keyed_seed("model_truth", dataset.generator_seed, family, model_seed, query["query_id"]),
                    )

                    def sampler(n: int, draw_seed: int, query=query) -> np.ndarray:
                        return _sample_model(model, device, payload, query["history"], n, draw_seed)

                    statistic = lambda draw, dataset=dataset: draw[:, dataset.target]
                    particle_counts = [128] if smoke else [128, 512]
                    smc_seeds = [3101] if smoke else [3101, 3109]
                    for n_particles in particle_counts:
                        for sampling_seed in smc_seeds:
                            arms = {}
                            for arm in ("low", "high"):
                                requested = query[arm]
                                potential = lambda draw, requested=requested, query=query, dataset=dataset: -0.5 * np.square(
                                    (draw[:, dataset.source] - requested) / query["bandwidth"]
                                )
                                arms[arm] = progressive_bridge_smc(
                                    sampler,
                                    potential,
                                    statistic,
                                    n_particles=n_particles,
                                    seed=keyed_seed(
                                        "synthetic_smc", dataset.generator_seed, family,
                                        model_seed, query["query_id"], n_particles,
                                        sampling_seed, arm,
                                    ),
                                    rejuvenation_steps=2,
                                )
                                for stage in arms[arm].stages.to_dict("records"):
                                    smc_stage_rows.append(
                                        {
                                            "run_id": run_root.name,
                                            "dataset": dataset.kind,
                                            "system": dataset.mechanism,
                                            "generator_seed": dataset.generator_seed,
                                            "model_family": family,
                                            "model_seed": model_seed,
                                            "sampling_seed": sampling_seed,
                                            "query_id": query["query_id"],
                                            "query_class": query["query_class"],
                                            "source_neuron": dataset.source,
                                            "target_functional": f"coordinate_{dataset.target}_mean",
                                            "amplitude": query["high"],
                                            "history_support_value": query["support"],
                                            "event_rarity": arms[arm].event_probability,
                                            "particle_count": n_particles,
                                            "smc_method": "progressive_bridge",
                                            "arm": arm,
                                            **stage,
                                        }
                                    )
                            estimate = arms["high"].estimate - arms["low"].estimate
                            mcse = math.sqrt(arms["high"].mcse**2 + arms["low"].mcse**2)
                            healthy = bool(
                                arms["high"].finite and arms["low"].finite
                                and min(arms["high"].minimum_ess_fraction, arms["low"].minimum_ess_fraction) >= 0.20
                                and min(arms["high"].unique_ancestors, arms["low"].unique_ancestors) >= 0.10 * n_particles
                            )
                            model_rows.append(
                                {
                                    "run_id": run_root.name,
                                    "dataset": dataset.kind,
                                    "system": dataset.mechanism,
                                    "generator_seed": dataset.generator_seed,
                                    "model_family": family,
                                    "model_seed": model_seed,
                                    "sampling_seed": sampling_seed,
                                    "query_id": query["query_id"],
                                    "query_class": query["query_class"],
                                    "source_neuron": dataset.source,
                                    "target_functional": f"coordinate_{dataset.target}_mean",
                                    "amplitude": query["high"],
                                    "history_support_value": query["support"],
                                    "nearest_history_distance": query["distance"],
                                    "event_rarity": min(arms["high"].event_probability, arms["low"].event_probability),
                                    "particle_count": n_particles,
                                    "smc_method": "progressive_bridge",
                                    "effect_estimate": estimate,
                                    "monte_carlo_error": mcse,
                                    "model_implied_effect": model_truth,
                                    "model_reference_mcse": model_truth_mcse,
                                    "oracle_effect": query["oracle_effect"],
                                    "minimum_ess_fraction": min(arms["high"].minimum_ess_fraction, arms["low"].minimum_ess_fraction),
                                    "unique_ancestor_fraction": min(arms["high"].unique_ancestors, arms["low"].unique_ancestors) / n_particles,
                                    "healthy_smc": healthy,
                                }
                            )
                            decomposition_rows.append(
                                {
                                    **model_rows[-1],
                                    "smc_particle_error": estimate - model_truth,
                                    "learned_model_error": model_truth - query["oracle_effect"],
                                    "total_error": estimate - query["oracle_effect"],
                                    "stable_but_wrong": bool(
                                        healthy
                                        and abs(estimate - model_truth) <= max(0.10, 2 * mcse)
                                        and abs(model_truth - query["oracle_effect"]) > 0.25
                                    ),
                                }
                            )
                if dataset.kind == "fitzhugh_nagumo":
                    for amplitude in (0.15, 0.30, 0.45):
                        for horizon in (1, 4, 8):
                            truth = dataset.pulse_truth(
                                amplitude,
                                horizon,
                                256 if smoke else 2000,
                                keyed_seed("fhn_pulse", dataset.generator_seed, amplitude, horizon),
                            )
                            n_rollout = 128 if smoke else 512
                            rng_seed = keyed_seed("fhn_model_pulse", dataset.generator_seed, family, model_seed, amplitude, horizon)
                            base_state = np.broadcast_to(dataset.anchor_h, (n_rollout, len(dataset.anchor_h))).copy()
                            pulse_state = base_state.copy()
                            base_values = base_state[:, : dataset.train_y.shape[1]].copy()
                            pulse_values = pulse_state[:, : dataset.train_y.shape[1]].copy()
                            for step in range(horizon):
                                base_state[:, : dataset.train_y.shape[1]] = base_values
                                pulse_state[:, : dataset.train_y.shape[1]] = pulse_values
                                base_state[:, -1] = 0.0
                                pulse_state[:, -1] = amplitude if step == 0 else 0.0
                                base_draw = _sample_model_batch(
                                    model, device, payload, base_state, rng_seed + step
                                )
                                pulse_draw = _sample_model_batch(
                                    model, device, payload, pulse_state, rng_seed + step
                                )
                                base_values += base_draw
                                pulse_values += pulse_draw
                            estimate_vector = (pulse_values - base_values).mean(axis=0)
                            pulse_rows.append(
                                {
                                    "run_id": run_root.name,
                                    "dataset": dataset.kind,
                                    "system": dataset.mechanism,
                                    "generator_seed": dataset.generator_seed,
                                    "model_family": family,
                                    "model_seed": model_seed,
                                    "source": dataset.source,
                                    "target": dataset.target,
                                    "pulse_amplitude": amplitude,
                                    "horizon_frames": horizon,
                                    "oracle_effect": float(truth[dataset.target]),
                                    "model_effect": float(estimate_vector[dataset.target]),
                                    "absolute_error": abs(float(estimate_vector[dataset.target] - truth[dataset.target])),
                                }
                            )
                print(
                    f"SYNTHETIC_MODEL_DONE kind={dataset.kind} seed={dataset.generator_seed} family={family} model_seed={model_seed}",
                    flush=True,
                )
                del model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    atomic_csv(run_root / "synthetic_system_manifest.csv", pd.DataFrame(manifest_rows))
    atomic_csv(run_root / "synthetic_predictive_scores.csv", pd.DataFrame(predictive_rows))
    existing_oracle = run_root / "synthetic_oracle_effects.csv"
    combined_oracle = pd.DataFrame(oracle_rows)
    if existing_oracle.exists():
        combined_oracle = pd.concat([pd.read_csv(existing_oracle), combined_oracle], ignore_index=True, sort=False)
    atomic_csv(existing_oracle, combined_oracle)
    atomic_csv(run_root / "synthetic_model_effects.csv", pd.DataFrame(model_rows))
    atomic_csv(run_root / "synthetic_error_decomposition.csv", pd.DataFrame(decomposition_rows))
    atomic_csv(run_root / "smc_diagnostics_synthetic.csv", pd.DataFrame(smc_stage_rows))
    atomic_csv(run_root / "fhn_intervention_effects.csv", pd.DataFrame(pulse_rows))
    update_status(
        run_root,
        "synthetic_confirmation",
        "complete",
        systems=len(systems),
        model_effect_rows=len(model_rows),
        pulse_rows=len(pulse_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_synthetic(args.run_root.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
