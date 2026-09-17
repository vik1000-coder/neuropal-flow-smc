from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d

from .dgps import (
    GaussianCovarianceDGP,
    GaussianMeanDGP,
    SmoothTiltDGP,
    influence_from_raw,
    make_static_dgp,
    raw_to_functional,
    trap_weights,
)
from .estimators import (
    BalancedEndpointAdapter,
    NormalizedStaticModel,
    PolynomialScoreModel,
    RatioCritic,
    RawMomentRegressor,
    finite_from_witness,
)
from .metrics import rmse, typed_metrics
from .oracles import run_all_oracle_checks


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if "extends" in config:
        base_path = (config_path.parent / config.pop("extends")).resolve()
        base, _ = load_config(base_path)

        def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
            result = dict(left)
            for key, value in right.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = value
            return result

        config = merge(base, config)
    return config, config_path


def resolve_output(config: dict[str, Any]) -> Path:
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    return output.resolve()


def enforce_resources(config: dict[str, Any], output: Path) -> None:
    limits = config["resource"]
    disk = shutil.disk_usage(output.parent if output.parent.exists() else PROJECT_ROOT)
    free_gb = disk.free / 1024**3
    if free_gb < float(limits["min_free_disk_gb"]):
        raise RuntimeError(
            f"disk safety stop: {free_gb:.3f} GiB free < {limits['min_free_disk_gb']} GiB"
        )
    if output.exists():
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) / 1024**3
        if size > float(limits["max_output_gb"]):
            raise RuntimeError(
                f"output safety stop: {size:.3f} GiB > {limits['max_output_gb']} GiB"
            )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss / 1024**3 if platform.system() == "Darwin" else rss / 1024**2
    if rss_gb > float(limits["max_rss_gb"]):
        raise RuntimeError(f"RSS safety stop: {rss_gb:.3f} GiB > {limits['max_rss_gb']} GiB")


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "source_tree_sha256": source_tree_hash(),
    }


def run_stage0(config: dict[str, Any], output: Path) -> dict[str, Any]:
    enforce_resources(config, output)
    started = time.time()
    checks, topology = run_all_oracle_checks(config["registered_scales"]["topology_epsilon"])
    frame = pd.DataFrame([check.__dict__ for check in checks])
    atomic_csv(output / "stage0" / "oracle_checks.csv", frame)
    atomic_csv(output / "stage0" / "topology_hodge.csv", pd.DataFrame(topology))
    failures = frame.loc[~frame["passed"]]
    payload = {
        "stage": 0,
        "status": "pass" if failures.empty else "fail",
        "checks": int(frame.shape[0]),
        "failed": int(failures.shape[0]),
        "started_unix": started,
        "finished_unix": time.time(),
        "source_tree_sha256": source_tree_hash(),
    }
    atomic_json(output / "stage0" / "gate.json", payload)
    report = [
        "# Gate 0: algebra and software correctness",
        "",
        f"Status: **{payload['status'].upper()}**",
        "",
        f"{payload['checks']} oracle/operator checks were executed; {payload['failed']} failed.",
        "",
    ]
    if not failures.empty:
        report.extend(["## Failures", "", failures.to_markdown(index=False), ""])
    else:
        report.extend(
            [
                "All registered normalization, positivity, moment-orthogonality, local derivative, influence-function, finite-witness, Gaussian corruption, dynamic-lag, and topology checks passed.",
                "",
            ]
        )
    (output / "stage0" / "gate_0_report.md").write_text("\n".join(report))
    return payload


def calibrate_information(config: dict[str, Any], output: Path) -> dict[str, Any]:
    target = 3.0
    n_train = int(config["data"]["n_train"])
    delta = float(config["registered_scales"]["primary_delta"])
    target_information = target / (n_train * delta**2)
    choices: dict[str, Any] = {}
    for mechanism in config["static_mechanisms"]:
        pilot = make_static_dgp(mechanism, amplitude=0.5)
        coefficient = pilot.information(0.0) / 0.5**2
        amplitude = float(np.clip(np.sqrt(target_information / max(coefficient, 1e-12)), 0.15, 0.85))
        dgp = make_static_dgp(mechanism, amplitude=amplitude)
        achieved_information = dgp.information(0.0)
        choices[mechanism] = {
            "amplitude": amplitude,
            "target_information": target_information,
            "achieved_information": achieved_information,
            "target_lambda": target,
            "achieved_lambda": n_train * delta**2 * achieved_information,
            "amplitude_bound_active": bool(amplitude in {0.15, 0.85}),
        }
    payload = {
        "status": "frozen_before_confirmatory",
        "selection_uses_outcomes": False,
        "formula": "Lambda = n_train * delta^2 * I_v at h=0",
        "mechanisms": choices,
    }
    atomic_json(output / "development" / "information_calibration.json", payload)
    return payload


def _threshold(dgp) -> float | None:
    if isinstance(dgp, SmoothTiltDGP):
        return dgp.definition.threshold
    return None


def _empirical_target(y: np.ndarray, channel: str, threshold: float | None) -> float:
    y = np.asarray(y, dtype=float)
    if channel == "covariance":
        return float(np.mean((y[:, 0] - np.mean(y[:, 0])) * (y[:, 1] - np.mean(y[:, 1]))))
    value = y[:, 0]
    if channel == "tail_probability" or channel == "mode_occupancy":
        return float(np.mean(value > float(threshold)))
    raw = np.array([np.mean(value**power) for power in range(1, 5)])
    return float(raw_to_functional(raw, channel))


def _grid_context(
    dgp, history: float, size: int = 801, sigma: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(dgp, SmoothTiltDGP):
        grid = np.linspace(-5.0, 5.0, size)
        weights = trap_weights(grid)
        density = np.interp(grid, dgp.grid, dgp.density_grid(history))
        if sigma > 0:
            density = gaussian_filter1d(
                density,
                sigma=sigma / (grid[1] - grid[0]),
                mode="constant",
                cval=0.0,
            )
        density /= np.sum(weights * density)
        return grid, weights, density
    if isinstance(dgp, GaussianMeanDGP):
        grid = np.linspace(-6.0, 6.0, size)
        weights = trap_weights(grid)
        variance = 1.0 + sigma**2
        density = np.exp(-0.5 * (grid - dgp.mean(history)) ** 2 / variance) / np.sqrt(
            2.0 * np.pi * variance
        )
        density /= np.sum(weights * density)
        return grid, weights, density
    raise TypeError("scalar grid context is unavailable")


def _influence_on_grid(
    dgp, grid: np.ndarray, weights: np.ndarray, density: np.ndarray, channel: str
) -> np.ndarray:
    if channel in {"tail_probability", "mode_occupancy"}:
        return (grid > float(_threshold(dgp))).astype(float) - float(
            np.sum(weights * density * (grid > float(_threshold(dgp))))
        )
    raw = np.asarray([np.sum(weights * density * grid**power) for power in range(1, 5)])
    return influence_from_raw(grid, raw, channel)


def _score_typed_effect(
    model: PolynomialScoreModel, dgp, histories: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray, float]:
    estimates = []
    truths = []
    residuals = []
    for history in np.asarray(histories, dtype=float):
        grid, weights, density = _grid_context(dgp, float(history), sigma=sigma)
        tangent, residual = model.reconstructed_tangent_grid(
            float(history), grid, density, weights, sigma=sigma
        )
        influence = _influence_on_grid(dgp, grid, weights, density, dgp.channel)
        estimates.append(float(np.sum(weights * density * influence * tangent)))
        true_tangent = dgp.noised_density_score_tangent(
            grid, np.full(grid.size, history), sigma
        )[1]
        truths.append(float(np.sum(weights * density * influence * true_tangent)))
        residuals.append(residual)
    return np.asarray(estimates), np.asarray(truths), float(np.mean(residuals))


def _ratio_typed_effect(
    critic: RatioCritic,
    dgp,
    histories: np.ndarray,
    rng: np.random.Generator,
    draws: int = 256,
) -> np.ndarray:
    estimates = []
    for history in np.asarray(histories, dtype=float):
        anchors = np.full(draws, history)
        y = dgp.sample(anchors, rng)
        tangent = critic.tangent(anchors, y)
        tangent -= np.mean(tangent)
        if isinstance(dgp, GaussianCovarianceDGP):
            influence = dgp.influence(y, anchors, dgp.channel)
        else:
            influence = dgp.influence(y[:, 0], anchors, dgp.channel)
        estimates.append(float(np.mean(influence * tangent)))
    return np.asarray(estimates)


def _endpoint_dataset(
    dgp,
    rng: np.random.Generator,
    delta: float,
    anchors: int,
    draws_per_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = rng.uniform(-0.9, 0.9, anchors)
    repeated = np.repeat(base, draws_per_side)
    plus = dgp.sample(repeated + delta, rng)
    minus = dgp.sample(repeated - delta, rng)
    h = np.concatenate([repeated, repeated])
    y = np.concatenate([plus, minus], axis=0)
    label = np.concatenate([np.ones(repeated.size, dtype=int), np.zeros(repeated.size, dtype=int)])
    permutation = rng.permutation(label.size)
    return h[permutation], y[permutation], label[permutation]


def _finite_endpoint_estimates(
    dgp,
    histories: np.ndarray,
    rng: np.random.Generator,
    delta: float,
    draws: int,
    adapter: BalancedEndpointAdapter | None = None,
) -> np.ndarray:
    estimates = []
    threshold = _threshold(dgp)
    for history in np.asarray(histories, dtype=float):
        plus_h = np.full(draws, history + delta)
        minus_h = np.full(draws, history - delta)
        plus = dgp.sample(plus_h, rng)
        minus = dgp.sample(minus_h, rng)
        if adapter is None:
            estimates.append(
                (_empirical_target(plus, dgp.channel, threshold) - _empirical_target(minus, dgp.channel, threshold))
                / (2.0 * delta)
            )
        else:
            mixture = np.concatenate([plus, minus], axis=0)
            anchor = np.full(mixture.shape[0], history)
            witness = adapter.witness(anchor, mixture)
            estimates.append(
                finite_from_witness(mixture, witness, delta, dgp.channel, threshold=threshold)
            )
    return np.asarray(estimates)


def _score_endpoint_effect(
    model: PolynomialScoreModel,
    dgp,
    histories: np.ndarray,
    delta: float,
    sigma: float,
) -> np.ndarray:
    estimates = []
    threshold = _threshold(dgp)
    for history in histories:
        targets = []
        for endpoint in [history - delta, history + delta]:
            grid, weights, _truth_density = _grid_context(dgp, float(endpoint), sigma=sigma)
            density = model.density_grid(float(endpoint), grid, weights, sigma=sigma)
            if dgp.channel in {"tail_probability", "mode_occupancy"}:
                targets.append(float(np.sum(weights * density * (grid > float(threshold)))))
            else:
                raw = np.asarray(
                    [np.sum(weights * density * grid**power) for power in range(1, 5)]
                )
                targets.append(float(raw_to_functional(raw, dgp.channel)))
        estimates.append((targets[1] - targets[0]) / (2.0 * delta))
    return np.asarray(estimates)


def _row_context(
    config: dict[str, Any], phase: str, seed: int, dgp, method: str, estimand: str, access: str
) -> dict[str, Any]:
    return {
        "study_id": config["study_id"],
        "phase": phase,
        "dgp_family": dgp.mechanism,
        "dgp_seed": seed,
        "data_seed": seed + 100_000,
        "model_seed": seed + 200_000,
        "adapter_seed": seed + 300_000,
        "eval_seed": seed + 400_000,
        "method_id": method,
        "estimand_type": estimand,
        "channel": dgp.channel,
        "delta": config["registered_scales"]["primary_delta"] if estimand == "finite" else 0.0,
        "sigma": 0.0,
        "information_access": access,
        "status": "ok",
        "failure_code": "",
    }


def _append_metrics(
    rows: list[dict[str, Any]], context: dict[str, Any], values: dict[str, float]
) -> None:
    for name, value in values.items():
        row = dict(context)
        row["metric_name"] = name
        row["metric_value"] = float(value)
        rows.append(row)


def _operator_rows(
    config: dict[str, Any],
    phase: str,
    seed: int,
    dgp,
    model: PolynomialScoreModel,
    method_id: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if isinstance(dgp, GaussianCovarianceDGP):
        return [], {}
    sigma = 0.0 if model.kind == "hyvarinen" else float(config["registered_scales"]["primary_sigma"])
    anchors = np.linspace(-0.75, 0.75, 16)
    h = np.repeat(anchors, 24)
    clean = dgp.sample(h, rng)[:, 0]
    if sigma > 0:
        z = clean + sigma * rng.normal(size=clean.size)
    else:
        z = clean
    true_score, true_tangent = dgp.noised_density_score_tangent(z, h, sigma)
    estimated_score = model.score(h, z, sigma)
    step = 1e-3
    true_score_plus = dgp.noised_density_score_tangent(z, h + step, sigma)[0]
    true_score_minus = dgp.noised_density_score_tangent(z, h - step, sigma)[0]
    true_field = (true_score_plus - true_score_minus) / (2.0 * step)
    estimated_field = model.mixed_field(h, z, sigma)
    estimated_tangent = np.empty_like(z)
    hodge_residuals = []
    for anchor in anchors:
        selected = h == anchor
        grid, weights, density = _grid_context(dgp, float(anchor), sigma=sigma)
        tangent_grid, residual = model.reconstructed_tangent_grid(
            float(anchor), grid, density, weights, sigma=sigma
        )
        estimated_tangent[selected] = np.interp(z[selected], grid, tangent_grid)
        hodge_residuals.append(residual)
    context = _row_context(config, phase, seed, dgp, method_id, "operator", "response_samples")
    rows: list[dict[str, Any]] = []
    score_error = rmse(estimated_score, true_score)
    field_error = rmse(estimated_field, true_field)
    tangent_error = rmse(estimated_tangent, true_tangent)
    _append_metrics(
        rows,
        context,
        {
            "response_score_rmse": score_error,
            "response_score_nrmse": score_error / max(np.sqrt(np.mean(true_score**2)), 1e-12),
            "mixed_field_rmse": field_error,
            "mixed_field_nrmse": field_error / max(np.sqrt(np.mean(true_field**2)), 1e-12),
            "history_tangent_rmse": tangent_error,
            "history_tangent_nrmse": tangent_error / max(np.sqrt(np.mean(true_tangent**2)), 1e-12),
            "field_error_amplification": field_error / max(score_error, 1e-12),
            "hodge_projection_residual": float(np.mean(hodge_residuals)),
            "centering_error": float(abs(np.mean(estimated_tangent))),
        },
    )
    for row in rows:
        row["sigma"] = sigma
    arrays = {
        "operator_h": h,
        "operator_y": z,
        "operator_true_score": true_score,
        "operator_estimated_score": estimated_score,
        "operator_true_mixed_field": true_field,
        "operator_estimated_mixed_field": estimated_field,
        "operator_true_history_tangent": true_tangent,
        "operator_estimated_history_tangent": estimated_tangent,
    }
    return rows, arrays


def run_static_case(
    config: dict[str, Any],
    phase: str,
    seed: int,
    mechanism: str,
    amplitude: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    dgp = make_static_dgp(mechanism, amplitude=amplitude)
    n_train = int(config["data"]["n_train"])
    h_train = rng.uniform(-1.2, 1.2, n_train)
    y_train = dgp.sample(h_train, rng)
    histories = rng.uniform(-0.8, 0.8, int(config["data"]["n_test_histories"]))
    delta = float(config["registered_scales"]["primary_delta"])
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"evaluation_histories": histories}

    zero_local = np.zeros(histories.size)
    truth_local = dgp.local_effect(histories, dgp.channel)
    arrays["local__truth"] = np.asarray(truth_local)
    direct = RawMomentRegressor(dgp.channel, threshold=_threshold(dgp)).fit(h_train, y_train)
    normalized = NormalizedStaticModel(dgp).fit(h_train, y_train)
    critic = RatioCritic(covariance=isinstance(dgp, GaussianCovarianceDGP)).fit(h_train, y_train, rng)
    local_estimates: dict[str, tuple[np.ndarray, str]] = {
        "B0_ZERO": (zero_local, "observational"),
        "B2_RAW_MOMENTS": (direct.local_effect(histories), "observational"),
        "S1_NLL_SCORE__A1_DENSITY_AD": (
            normalized.local_effect(histories, dgp.channel),
            "normalized_density",
        ),
        "A2_RATIO_CRITIC": (
            _ratio_typed_effect(critic, dgp, histories, rng),
            "observational",
        ),
    }

    score_models: dict[str, PolynomialScoreModel] = {}
    if not isinstance(dgp, GaussianCovarianceDGP):
        value = y_train[:, 0]
        model_specs = {
            "S2_HYVARINEN__A7_HODGE": "hyvarinen",
            "S5_DSM_MULTI__A7_HODGE": "dsm",
            "S6_GAUSS_DSM__A3_CENTER": "gaussian_dsm",
            "S7_ANCHORED_ENERGY__A3_CENTER": "anchored",
        }
        for method_id, kind in model_specs.items():
            ridge = (
                float(config.get("score_selection", {}).get("anchored_ridge", 1.0))
                if kind == "anchored"
                else 1e-2
            )
            schedule = tuple(
                config.get("score_selection", {}).get(
                    "anchored_noise_schedule", [0.03, 0.06, 0.12, 0.25]
                )
            )
            model = PolynomialScoreModel(kind, ridge=ridge).fit(
                h_train,
                value,
                rng,
                sigma_ladder=schedule if kind == "anchored" else (0.03, 0.06, 0.12, 0.25),
            )
            score_models[method_id] = model
            score_sigma = 0.0 if kind == "hyvarinen" else float(
                config["registered_scales"]["primary_sigma"]
            )
            estimate, score_truth, residual = _score_typed_effect(
                model, dgp, histories, score_sigma
            )
            if kind == "hyvarinen" or dgp.channel not in {"tail_probability", "mode_occupancy"}:
                local_estimates[method_id] = (estimate, "response_score")
            else:
                context = _row_context(
                    config, phase, seed, dgp, method_id, "local_smoothed", "response_score"
                )
                context["sigma"] = score_sigma
                _append_metrics(rows, context, typed_metrics(estimate, score_truth))
            operator_rows, operator_arrays = _operator_rows(
                config,
                phase,
                seed,
                dgp,
                model,
                method_id.split("__")[0],
                rng,
            )
            rows.extend(operator_rows)
            prefix = method_id.split("__")[0].lower()
            arrays.update({f"{prefix}__{name}": value for name, value in operator_arrays.items()})
            context = _row_context(config, phase, seed, dgp, method_id, "local", "response_score")
            _append_metrics(rows, context, {"typed_hodge_residual": residual})
    elif dgp.channel == "covariance":
        local_estimates["S6_GAUSS_DSM__A3_CENTER"] = (
            normalized.local_effect(histories, dgp.channel),
            "response_score",
        )

    for method_id, (estimate, access) in local_estimates.items():
        arrays[f"local__{method_id.lower()}"] = np.asarray(estimate)
        context = _row_context(config, phase, seed, dgp, method_id, "local", access)
        _append_metrics(rows, context, typed_metrics(estimate, truth_local))

    truth_finite = dgp.finite_effect(histories, delta, dgp.channel)
    arrays["finite__truth"] = np.asarray(truth_finite)
    endpoint_train = _endpoint_dataset(dgp, rng, delta, anchors=500, draws_per_side=8)
    endpoint_calibration = _endpoint_dataset(dgp, rng, delta, anchors=200, draws_per_side=4)
    classifier = BalancedEndpointAdapter(
        "classifier", delta, covariance=isinstance(dgp, GaussianCovarianceDGP)
    ).fit(*endpoint_train, *endpoint_calibration)
    riesz = BalancedEndpointAdapter(
        "riesz", delta, covariance=isinstance(dgp, GaussianCovarianceDGP)
    ).fit(*endpoint_train, *endpoint_calibration)
    finite_estimates: dict[str, tuple[np.ndarray, str]] = {
        "B0_ZERO": (np.zeros_like(truth_finite), "observational"),
        "DIRECT_ENDPOINT_MOMENTS": (
            _finite_endpoint_estimates(
                dgp,
                histories,
                rng,
                delta,
                int(config["data"]["endpoint_draws_per_side"]),
            ),
            "conditional_endpoint_queries",
        ),
        "B2_RAW_MOMENTS_FINITE": (direct.finite_effect(histories, delta), "observational"),
        "S1_NLL_ENDPOINTS": (
            normalized.finite_effect(histories, delta, dgp.channel),
            "normalized_density",
        ),
        "A8_ENDPOINT_CLASSIFIER": (
            _finite_endpoint_estimates(
                dgp,
                histories,
                rng,
                delta,
                int(config["data"]["endpoint_draws_per_side"]),
                classifier,
            ),
            "endpoint_labels_and_queries",
        ),
        "A10_SIGNED_RIESZ": (
            _finite_endpoint_estimates(
                dgp,
                histories,
                rng,
                delta,
                int(config["data"]["endpoint_draws_per_side"]),
                riesz,
            ),
            "endpoint_labels_and_queries",
        ),
    }
    if "S7_ANCHORED_ENERGY__A3_CENTER" in score_models:
        finite_estimates["S7_SCORE_INTEGRATED_ENDPOINTS"] = (
            _score_endpoint_effect(
                score_models["S7_ANCHORED_ENERGY__A3_CENTER"],
                dgp,
                histories,
                delta,
                float(config["registered_scales"]["primary_sigma"]),
            ),
            "response_score",
        )
        if dgp.channel in {"tail_probability", "mode_occupancy"}:
            finite_estimates.pop("S7_SCORE_INTEGRATED_ENDPOINTS")
    for method_id, (estimate, access) in finite_estimates.items():
        arrays[f"finite__{method_id.lower()}"] = np.asarray(estimate)
        context = _row_context(config, phase, seed, dgp, method_id, "finite", access)
        _append_metrics(rows, context, typed_metrics(estimate, truth_finite))

    information = dgp.information(0.0)
    for row in rows:
        row["n_train"] = n_train
        row["response_dim"] = int(config["data"]["response_dim"])
        row["history_dim"] = int(config["data"]["history_dim"])
        row["information_index"] = n_train * delta**2 * information
        row["wall_time"] = time.perf_counter() - started
        row["peak_memory"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return pd.DataFrame(rows), arrays


def _valid_case(path: Path, arrays_path: Path | None = None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if arrays_path is not None and (not arrays_path.exists() or arrays_path.stat().st_size == 0):
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    required = {"study_id", "dgp_seed", "method_id", "metric_name", "metric_value", "status"}
    return required.issubset(frame.columns) and not frame.empty


def run_static_phase(config: dict[str, Any], output: Path, phase: str) -> dict[str, Any]:
    gate_path = output / "stage0" / "gate.json"
    if not gate_path.exists() or json.loads(gate_path.read_text())["status"] != "pass":
        raise RuntimeError("Gate 0 has not passed")
    calibration_path = output / "development" / "information_calibration.json"
    if not calibration_path.exists():
        calibrate_information(config, output)
    calibration = json.loads(calibration_path.read_text())["mechanisms"]
    seeds = list(config["seeds"][phase])
    cases_dir = output / "cases" / phase
    cases_dir.mkdir(parents=True, exist_ok=True)
    expected = len(seeds) * len(config["static_mechanisms"])
    completed = 0
    failures = 0
    started = time.time()
    for seed in seeds:
        for mechanism in config["static_mechanisms"]:
            enforce_resources(config, output)
            path = cases_dir / f"{mechanism}__seed_{seed}.csv"
            arrays_path = cases_dir / f"{mechanism}__seed_{seed}.npz"
            if _valid_case(path, arrays_path):
                completed += 1
                continue
            try:
                frame, arrays = run_static_case(
                    config,
                    phase,
                    int(seed),
                    mechanism,
                    float(calibration[mechanism]["amplitude"]),
                )
                atomic_npz(arrays_path, arrays)
                frame["array_uri"] = str(arrays_path.relative_to(output))
                frame["array_sha256"] = sha256_file(arrays_path)
                atomic_csv(path, frame)
                completed += 1
            except Exception as exc:
                failures += 1
                failure = pd.DataFrame(
                    [
                        {
                            "study_id": config["study_id"],
                            "phase": phase,
                            "dgp_family": mechanism,
                            "dgp_seed": seed,
                            "method_id": "CASE_FAILURE",
                            "estimand_type": "unknown",
                            "channel": "unknown",
                            "metric_name": "failure",
                            "metric_value": np.nan,
                            "status": "failed",
                            "failure_code": type(exc).__name__,
                            "failure_message": str(exc),
                        }
                    ]
                )
                atomic_csv(path, failure)
                if failures / max(completed + failures, 1) > 0.10:
                    raise RuntimeError("more than 10% of phase cases failed") from exc
    payload = {
        "phase": phase,
        "expected_cases": expected,
        "completed_cases": completed,
        "failed_cases": failures,
        "status": "complete" if completed + failures == expected else "incomplete",
        "started_unix": started,
        "finished_unix": time.time(),
        "source_tree_sha256": source_tree_hash(),
    }
    atomic_json(output / f"stage_static_{phase}.json", payload)
    return payload


def run_dynamic_phase(config: dict[str, Any], output: Path) -> dict[str, Any]:
    from .dynamic import run_dynamic_case

    gate_path = output / "stage0" / "gate.json"
    if not gate_path.exists() or json.loads(gate_path.read_text())["status"] != "pass":
        raise RuntimeError("Gate 0 has not passed")
    seeds = list(config["seeds"]["confirmatory"])
    cases_dir = output / "cases" / "dynamic_confirmatory"
    cases_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures = 0
    started = time.time()
    for seed in seeds:
        enforce_resources(config, output)
        path = cases_dir / f"dynamic__seed_{seed}.csv"
        arrays_path = cases_dir / f"dynamic__seed_{seed}.npz"
        if _valid_case(path, arrays_path):
            completed += 1
            continue
        try:
            frame, arrays = run_dynamic_case(config, int(seed))
            atomic_npz(arrays_path, arrays)
            frame["array_uri"] = str(arrays_path.relative_to(output))
            frame["array_sha256"] = sha256_file(arrays_path)
            atomic_csv(path, frame)
            completed += 1
        except Exception as exc:
            failures += 1
            atomic_csv(
                path,
                pd.DataFrame(
                    [
                        {
                            "study_id": config["study_id"],
                            "phase": "confirmatory",
                            "dgp_family": "dynamic",
                            "dgp_seed": seed,
                            "method_id": "CASE_FAILURE",
                            "estimand_type": "dynamic",
                            "channel": "unknown",
                            "metric_name": "failure",
                            "metric_value": np.nan,
                            "status": "failed",
                            "failure_code": type(exc).__name__,
                            "failure_message": str(exc),
                        }
                    ]
                ),
            )
            if failures / max(completed + failures, 1) > 0.10:
                raise RuntimeError("more than 10% of dynamic cases failed") from exc
    payload = {
        "phase": "dynamic_confirmatory",
        "expected_cases": len(seeds),
        "completed_cases": completed,
        "failed_cases": failures,
        "status": "complete" if completed + failures == len(seeds) else "incomplete",
        "started_unix": started,
        "finished_unix": time.time(),
        "source_tree_sha256": source_tree_hash(),
    }
    atomic_json(output / "stage_dynamic_confirmatory.json", payload)
    return payload


def initialize_manifest(config: dict[str, Any], config_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frozen_config = output / "config.yaml"
    if not frozen_config.exists():
        frozen_config.write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copy2(config_path, output / "config_source.yaml")
        shutil.copytree(PROJECT_ROOT / "src", output / "source_snapshot", dirs_exist_ok=False)
    manifest = {
        "schema_version": "1",
        "study_id": config["study_id"],
        "status": "running",
        "started_unix": time.time(),
        "config_sha256": sha256_file(config_path),
        "runbook_sha256": config["runbook_sha256"],
        "source_tree_sha256": source_tree_hash(),
        "environment": environment_record(),
        "resource_limits": config["resource"],
        "stages": {},
    }
    existing = output / "manifest.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        manifest["started_unix"] = old.get("started_unix", manifest["started_unix"])
        manifest["stages"] = old.get("stages", {})
    atomic_json(existing, manifest)
    return manifest


def update_manifest(output: Path, stage: str, payload: dict[str, Any]) -> None:
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["stages"][stage] = payload
    manifest["updated_unix"] = time.time()
    manifest["source_tree_sha256"] = source_tree_hash()
    atomic_json(path, manifest)


def finalize_manifest(output: Path, status: str = "complete") -> None:
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = status
    manifest["finished_unix"] = time.time()
    atomic_json(path, manifest)
