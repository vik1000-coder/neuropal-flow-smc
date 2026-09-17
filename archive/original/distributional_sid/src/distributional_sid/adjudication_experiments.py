from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from .core import (
    CharacteristicBank,
    HistorySieve,
    atomic_csv,
    atomic_json,
    independent_folds,
    resource_guard,
    ridge_coefficients,
    riesz_coefficients,
    vector_metrics,
)
from .experiments import freeze_source_and_environment, write_frozen_config
from .remaining_experiments import (
    GeometryCell,
    MeasurementCell,
    _dense_oracle_complex,
    _dense_parameters_general,
    _estimate_filter_rho,
    _latent_gain_oracle,
    _measurement_oracle,
    _sample_measurement_paths,
    _standard_normal_qmc,
)
from .tier_a_extensions import (
    RieszCell,
    _matrix_from_symmetric,
    _observation_kernel,
    _package_rows,
    _sample_dense,
    _sample_riesz_cell,
    _symmetric_vector,
)


def _force_thread_limits() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def _metric_row(
    *,
    experiment_id: str,
    cell_id: str,
    family: str,
    seed: int,
    method: str,
    n_train: int,
    truth: np.ndarray,
    values: np.ndarray,
    method_init: int = 0,
    **extra: object,
) -> dict[str, object]:
    truth = np.atleast_1d(np.asarray(truth, dtype=float))
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    estimate = np.mean(values, axis=0)
    standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(len(values))
    metrics = vector_metrics(estimate, truth)
    for key in ("calibration_slope", "sign_accuracy"):
        if not np.isfinite(metrics[key]):
            metrics[key] = 0.0
    return {
        "experiment_id": experiment_id,
        "cell_id": cell_id,
        "dgp_family": family,
        "dgp_seed": seed,
        "method": method,
        "method_init": method_init,
        "n_train": n_train,
        **metrics,
        "target_abs_error": float(np.mean(np.abs(estimate - truth))),
        "coverage_fraction": float(
            np.mean(
                (truth >= estimate - 1.96 * standard_error)
                & (truth <= estimate + 1.96 * standard_error)
            )
        ),
        "max_standard_error": float(np.max(standard_error)),
        "fit_status": "ok",
        "warning_code": "",
        **extra,
    }


def _parse_number_list(value: str, caster) -> list:
    return [caster(item.strip()) for item in value.split(",") if item.strip()]


def _riesz_population_truth(cell: RieszCell) -> float:
    rng = np.random.default_rng(910_001 + sum(ord(char) for char in cell.name))
    batch = 100_000
    total = 0.0
    count = 0
    for _ in range(10):
        _, _, _, _, bdm = _sample_riesz_cell(cell, rng, batch)
        total += float(np.sum(bdm))
        count += len(bdm)
    return total / count


def _weight_derivative(cell: RieszCell, h: np.ndarray, b: np.ndarray) -> np.ndarray:
    if cell.family == "uniform_valid":
        return -2.0 * h[:, 0]
    if cell.family == "gap":
        return -h[:, 0] * b
    return np.zeros(len(h), dtype=float)


def _gmm_density_score(
    h_train: np.ndarray,
    h_test: np.ndarray,
    seed: int,
    component_grid: tuple[int, ...] = (1, 2, 4, 8),
    maximum_fit_rows: int = 20_000,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    if len(h_train) > maximum_fit_rows:
        fit_index = rng.choice(len(h_train), size=maximum_fit_rows, replace=False)
        fit_h = h_train[fit_index]
    else:
        fit_h = h_train
    if len(fit_h) > 5_000:
        validation_index = rng.choice(len(fit_h), size=5_000, replace=False)
        validation_h = fit_h[validation_index]
    else:
        validation_h = fit_h
    candidates: list[tuple[float, GaussianMixture]] = []
    for components in component_grid:
        model = GaussianMixture(
            n_components=components,
            covariance_type="full",
            reg_covar=1e-5,
            max_iter=200,
            n_init=1,
            random_state=seed + components,
        ).fit(fit_h)
        candidates.append((float(model.bic(validation_h)), model))
    _, selected = min(candidates, key=lambda item: item[0])
    responsibilities = selected.predict_proba(h_test)
    component_scores = np.empty(
        (len(h_test), selected.n_components, h_test.shape[1]), dtype=float
    )
    for component in range(selected.n_components):
        precision = np.linalg.inv(selected.covariances_[component])
        component_scores[:, component] = -(
            h_test - selected.means_[component]
        ) @ precision
    score = np.sum(responsibilities[:, :, None] * component_scores, axis=1)
    return score[:, 0], int(selected.n_components)


def _e6_probe_residual(
    h: np.ndarray,
    b: np.ndarray,
    alpha: np.ndarray,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    residuals: list[float] = []
    for _ in range(12):
        direction = rng.normal(size=h.shape[1])
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        omega = float(rng.uniform(0.25, 1.5))
        phase = omega * (h @ direction)
        lhs = float(np.mean(alpha * np.sin(phase)))
        rhs = float(np.mean(b * omega * direction[0] * np.cos(phase)))
        residuals.append(abs(lhs - rhs))
    return float(np.quantile(residuals, 0.95))


def _fit_e6_adjudication(
    cell: RieszCell,
    seed: int,
    n_train: int,
    width: int,
    riesz_ridge: float,
    feature_ridge: float,
    include_density_score: bool,
    population_truth: float,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(911_000 + 1009 * seed)
    h, b, oracle_alpha, m, bdm = _sample_riesz_cell(cell, rng, n_train)
    observed = m + rng.normal(scale=0.35, size=n_train)
    score_names = [
        "PLUG",
        "RIESZ_TARGETED",
        "ORTH_TARGETED",
        "OR_M_TARGETED",
        "OR_A",
        "OR_R",
        "OR_PW",
    ]
    if include_density_score:
        score_names.extend(["RIESZ_DENSITY_SCORE", "ORTH_DENSITY_SCORE"])
    scores = {name: np.zeros(n_train, dtype=float) for name in score_names}
    targeted_alpha = np.zeros(n_train, dtype=float)
    density_alpha = np.zeros(n_train, dtype=float)
    selected_components = np.zeros(n_train, dtype=float)
    for fold_id, (train, test) in enumerate(
        independent_folds(n_train, 5, 912_000 + seed)
    ):
        resource_guard()
        sieve = HistorySieve(
            cell.dimension, width, 913_000 + 100 * seed + fold_id
        ).fit(h[train])
        z_train = sieve.transform(h[train])
        z_test = sieve.transform(h[test])
        dz_train = sieve.derivative(h[train], 0)
        dz_test = sieve.derivative(h[test], 0)
        regression_coefficient = ridge_coefficients(
            z_train, observed[train, None], feature_ridge
        )
        prediction = (z_test @ regression_coefficient)[:, 0]
        derivative = (dz_test @ regression_coefficient)[:, 0]
        alpha_coefficient = riesz_coefficients(
            z_train, b[train, None] * dz_train, riesz_ridge
        )
        alpha_targeted = z_test @ alpha_coefficient
        targeted_alpha[test] = alpha_targeted
        residual = observed[test] - prediction
        scores["PLUG"][test] = b[test] * derivative
        scores["RIESZ_TARGETED"][test] = alpha_targeted * observed[test]
        scores["ORTH_TARGETED"][test] = b[test] * derivative + alpha_targeted * residual
        scores["OR_M_TARGETED"][test] = bdm[test] + alpha_targeted * (
            observed[test] - m[test]
        )
        scores["OR_A"][test] = b[test] * derivative + oracle_alpha[test] * residual
        scores["OR_R"][test] = oracle_alpha[test] * observed[test]
        scores["OR_PW"][test] = bdm[test]
        if include_density_score:
            density_score, components = _gmm_density_score(
                h[train], h[test], 914_000 + 100 * seed + fold_id
            )
            alpha_density = -_weight_derivative(cell, h[test], b[test]) - (
                b[test] * density_score
            )
            density_alpha[test] = alpha_density
            selected_components[test] = components
            scores["RIESZ_DENSITY_SCORE"][test] = alpha_density * observed[test]
            scores["ORTH_DENSITY_SCORE"][test] = (
                b[test] * derivative + alpha_density * residual
            )
    targeted_nrmse = vector_metrics(targeted_alpha, oracle_alpha)["nrmse"]
    targeted_probe = _e6_probe_residual(
        h, b, targeted_alpha, 915_000 + seed
    )
    density_nrmse = (
        vector_metrics(density_alpha, oracle_alpha)["nrmse"]
        if include_density_score
        else 0.0
    )
    density_probe = (
        _e6_probe_residual(h, b, density_alpha, 916_000 + seed)
        if include_density_score
        else 0.0
    )
    rows: list[dict[str, object]] = []
    for method, values in scores.items():
        if "DENSITY" in method:
            alpha_nrmse = density_nrmse
            probe_q95 = density_probe
            representer_norm = float(np.sqrt(np.mean(density_alpha**2)))
        elif method in {"RIESZ_TARGETED", "ORTH_TARGETED", "OR_M_TARGETED"}:
            alpha_nrmse = targeted_nrmse
            probe_q95 = targeted_probe
            representer_norm = float(np.sqrt(np.mean(targeted_alpha**2)))
        elif method in {"OR_A", "OR_R"}:
            alpha_nrmse = 0.0
            probe_q95 = _e6_probe_residual(
                h, b, oracle_alpha, 917_000 + seed
            )
            representer_norm = float(np.sqrt(np.mean(oracle_alpha**2)))
        else:
            alpha_nrmse = 0.0
            probe_q95 = 0.0
            representer_norm = 0.0
        warning = ""
        if probe_q95 > 0.05:
            warning = "probe_residual"
        if representer_norm > 5.0:
            warning = (
                f"{warning}+large_representer_norm"
                if warning
                else "large_representer_norm"
            )
        row = _metric_row(
            experiment_id="E6A",
            cell_id=f"{cell.name}_n{n_train}_w{width}_r{riesz_ridge:.0e}",
            family=cell.family,
            seed=seed,
            method=method,
            n_train=n_train,
            truth=np.asarray([population_truth]),
            values=values,
            width=width,
            riesz_ridge=riesz_ridge,
            feature_ridge=feature_ridge,
            alpha_nrmse=alpha_nrmse,
            probe_q95_abs=probe_q95,
            estimated_representer_norm=representer_norm,
            oracle_representer_norm=float(np.sqrt(np.mean(oracle_alpha**2))),
            gmm_selected_components=float(np.mean(selected_components)),
            warning_emitted=float(bool(warning)),
        )
        row["warning_code"] = warning
        rows.append(row)
    return rows


def command_e6_adjudication(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cell_map = {
        "gaussian": RieszCell("gaussian_d4", "gaussian", 4),
        "t3": RieszCell("student_t_nu3", "student_t", 4, 3.0),
        "t5": RieszCell("student_t_nu5", "student_t", 4, 5.0),
        "t10": RieszCell("student_t_nu10", "student_t", 4, 10.0),
        "gap3": RieszCell("gap_mu3", "gap", 1, 3.0),
        "gap4": RieszCell("gap_mu4", "gap", 1, 4.0),
    }
    cells = [cell_map[name] for name in _parse_number_list(args.cells, str)]
    sample_sizes = _parse_number_list(args.sample_sizes, int)
    widths = _parse_number_list(args.widths, int)
    ridges = _parse_number_list(args.riesz_ridges, float)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    truths = {cell.name: _riesz_population_truth(cell) for cell in cells}
    configuration = {
        "command": "e6-adjudication",
        "cells": [cell.__dict__ for cell in cells],
        "sample_sizes": sample_sizes,
        "widths": widths,
        "riesz_ridges": ridges,
        "feature_ridge": args.feature_ridge,
        "include_density_score": bool(args.include_density_score),
        "seeds": seeds,
        "population_truths": truths,
        "tuning_status": args.stage,
    }
    write_frozen_config(run_dir, configuration)
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for cell in cells:
        for n_train in sample_sizes:
            for width in widths:
                for ridge in ridges:
                    for seed in seeds:
                        print(
                            f"E6A cell={cell.name} n={n_train} width={width} "
                            f"ridge={ridge:g} seed={seed}",
                            flush=True,
                        )
                        rows.extend(
                            _fit_e6_adjudication(
                                cell,
                                seed,
                                n_train,
                                width,
                                ridge,
                                args.feature_ridge,
                                bool(args.include_density_score),
                                truths[cell.name],
                            )
                        )
    method_count = 9 if args.include_density_score else 7
    expected = (
        len(cells)
        * len(sample_sizes)
        * len(widths)
        * len(ridges)
        * len(seeds)
        * method_count
    )
    validation = _package_rows(run_dir, rows, expected)
    validation.update(
        {
            "full_estimator_decomposition_complete": True,
            "oracle_riesz_control_complete": True,
            "direct_density_score_complete": bool(args.include_density_score),
            "stage": args.stage,
        }
    )
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def _scalar_characteristic_bank(band: str) -> CharacteristicBank:
    if band == "low":
        magnitudes = np.linspace(0.15, 0.8, 8)
    elif band == "wide":
        magnitudes = np.linspace(0.25, 2.0, 12)
    else:
        raise ValueError(band)
    signs = np.where(np.arange(len(magnitudes)) % 2 == 0, 1.0, -1.0)
    frequencies = (signs * magnitudes)[:, None]
    return CharacteristicBank(
        median=np.zeros(1),
        scale=np.ones(1),
        frequencies=frequencies,
        seed=92610 if band == "low" else 92611,
    )


def _projected_latent_feature(
    observed: np.ndarray,
    kernel: np.ndarray,
    bank: CharacteristicBank,
) -> np.ndarray:
    denominator = max(float(kernel @ kernel), 1e-12)
    projected = (observed @ kernel / denominator)[:, None]
    return bank.transform(projected)


def _deconvolved_characteristic_feature(
    observed: np.ndarray,
    kernel: np.ndarray,
    noise_sd: float,
    bank: CharacteristicBank,
) -> tuple[np.ndarray, float, float]:
    denominator = max(float(kernel @ kernel), 1e-12)
    latent_frequencies = bank.raw_frequencies[:, 0]
    observation_frequencies = np.outer(
        kernel / denominator, latent_frequencies
    )
    attenuation = np.exp(
        -0.5 * noise_sd**2 * np.sum(observation_frequencies**2, axis=0)
    )
    inverse = 1.0 / np.maximum(attenuation, 1e-300)
    phase = observed @ observation_frequencies + bank.raw_phase_offset[None, :]
    transformed = np.exp(1j * phase) * inverse[None, :]
    feature = np.concatenate([transformed.real, transformed.imag], axis=1)
    return feature, float(np.max(inverse)), float(np.median(inverse))


def _estimate_filter_rho_mixture_likelihood(
    h: np.ndarray,
    observed: np.ndarray,
    noise_sd: float,
    seed: int,
    maximum_rows: int = 5_000,
) -> float:
    rng = np.random.default_rng(seed)
    if len(h) > maximum_rows:
        index = rng.choice(len(h), size=maximum_rows, replace=False)
        h_fit = h[index]
        y_fit = observed[index]
    else:
        h_fit = h
        y_fit = observed
    pi = expit(-0.3 + h_fit[:, 0])
    log_pi = np.log(np.maximum(pi, 1e-12))
    log_one_minus_pi = np.log(np.maximum(1.0 - pi, 1e-12))
    inverse_variance = 1.0 / max(noise_sd**2, 1e-12)

    def objective(rho: float) -> float:
        kernel = _observation_kernel(float(rho), y_fit.shape[1])
        residual_zero = y_fit + pi[:, None] * kernel[None, :]
        residual_one = y_fit - (1.0 - pi)[:, None] * kernel[None, :]
        log_zero = log_one_minus_pi - 0.5 * inverse_variance * np.sum(
            residual_zero**2, axis=1
        )
        log_one = log_pi - 0.5 * inverse_variance * np.sum(
            residual_one**2, axis=1
        )
        return float(-np.mean(np.logaddexp(log_zero, log_one)))

    result = minimize_scalar(
        objective,
        bounds=(0.01, 0.995),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 80},
    )
    return float(np.clip(result.x, 0.01, 0.995))


def _feature_result_rows(
    *,
    cell_id: str,
    seed: int,
    prefix: str,
    h: np.ndarray,
    feature: np.ndarray,
    oracle_mean: np.ndarray,
    oracle_derivative: np.ndarray,
    truth: np.ndarray,
    n_train: int,
    rho: float,
    horizon: int,
    band: str,
    rho_estimate: float,
    max_inverse_attenuation: float,
    median_inverse_attenuation: float,
    access_tier: str,
) -> list[dict[str, object]]:
    from .tier_a_extensions import _feature_score

    result = _feature_score(
        h,
        feature,
        oracle_mean,
        oracle_derivative,
        truth,
        seed + 30_000 + horizon,
        0,
    )
    rows: list[dict[str, object]] = []
    for method, metrics in result.items():
        rows.append(
            {
                "experiment_id": "E8A",
                "cell_id": cell_id,
                "dgp_family": "filtered_latent_recovery",
                "dgp_seed": seed,
                "method": f"{prefix}_{method}",
                "method_init": 0,
                "access_tier": access_tier,
                "n_train": n_train,
                "rho": rho,
                "horizon": horizon,
                "frequency_band": band,
                "rho_estimate": rho_estimate,
                "rho_abs_error": abs(rho_estimate - rho),
                "kernel_l2_norm": float(
                    np.linalg.norm(_observation_kernel(rho, horizon))
                ),
                "max_inverse_attenuation": max_inverse_attenuation,
                "median_inverse_attenuation": median_inverse_attenuation,
                **metrics,
                "warning_code": "",
                "fit_status": "ok",
            }
        )
    return rows


def _fit_e8_adjudication(
    seed: int,
    n_train: int,
    rho: float,
    horizon: int,
    band: str,
    noise_sd: float,
    include_mixture_mle: bool,
    mle_only: bool,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(
        920_000 + 1009 * seed + 10 * horizon + int(round(100 * rho))
    )
    cell = MeasurementCell(
        name=f"rho{rho:g}_h{horizon}",
        rho=rho,
        noise_sd=noise_sd,
        heteroskedastic_strength=0.0,
    )
    bank = _scalar_characteristic_bank(band)
    h = rng.normal(size=(n_train, 4))
    observed, latent, _ = _sample_measurement_paths(
        h, rng, cell, horizon=horizon
    )
    truth_h = _standard_normal_qmc(4, 16, 921_000 + seed)
    oracle_mean, oracle_derivative = _latent_gain_oracle(h, bank)
    _, truth_derivative = _latent_gain_oracle(truth_h, bank)
    truth = np.mean(truth_derivative, axis=0)
    true_kernel = _observation_kernel(rho, horizon)
    rho_estimate = _estimate_filter_rho(observed)
    estimated_kernel = _observation_kernel(rho_estimate, horizon)
    rho_mle = (
        _estimate_filter_rho_mixture_likelihood(
            h, observed, noise_sd, 922_000 + seed + horizon
        )
        if include_mixture_mle
        else rho_estimate
    )
    mle_kernel = _observation_kernel(rho_mle, horizon)
    latent_feature = bank.transform(latent[:, None])
    point_known = _projected_latent_feature(observed, true_kernel, bank)
    point_estimated = _projected_latent_feature(observed, estimated_kernel, bank)
    cf_known, known_max_inverse, known_median_inverse = (
        _deconvolved_characteristic_feature(
            observed, true_kernel, noise_sd, bank
        )
    )
    cf_estimated, estimated_max_inverse, estimated_median_inverse = (
        _deconvolved_characteristic_feature(
            observed, estimated_kernel, noise_sd, bank
        )
    )
    point_mle = _projected_latent_feature(observed, mle_kernel, bank)
    cf_mle, mle_max_inverse, mle_median_inverse = (
        _deconvolved_characteristic_feature(
            observed, mle_kernel, noise_sd, bank
        )
    )
    cell_id = f"rho{rho:g}_h{horizon}_n{n_train}_{band}"
    rows: list[dict[str, object]] = []
    rows.extend(
        _feature_result_rows(
            cell_id=cell_id,
            seed=seed,
            prefix="LATENT_ORACLE",
            h=h,
            feature=latent_feature,
            oracle_mean=oracle_mean,
            oracle_derivative=oracle_derivative,
            truth=truth,
            n_train=n_train,
            rho=rho,
            horizon=horizon,
            band=band,
            rho_estimate=rho_estimate,
            max_inverse_attenuation=1.0,
            median_inverse_attenuation=1.0,
            access_tier="M1",
        )
    )
    if include_mixture_mle:
        rows.extend(
            _feature_result_rows(
                cell_id=cell_id,
                seed=seed,
                prefix="POINT_MIXTURE_MLE",
                h=h,
                feature=point_mle,
                oracle_mean=oracle_mean,
                oracle_derivative=oracle_derivative,
                truth=truth,
                n_train=n_train,
                rho=rho,
                horizon=horizon,
                band=band,
                rho_estimate=rho_mle,
                max_inverse_attenuation=mle_max_inverse,
                median_inverse_attenuation=mle_median_inverse,
                access_tier="O0+estimated_measurement",
            )
        )
        rows.extend(
            _feature_result_rows(
                cell_id=cell_id,
                seed=seed,
                prefix="CF_DECONV_MIXTURE_MLE",
                h=h,
                feature=cf_mle,
                oracle_mean=oracle_mean,
                oracle_derivative=oracle_derivative,
                truth=truth,
                n_train=n_train,
                rho=rho,
                horizon=horizon,
                band=band,
                rho_estimate=rho_mle,
                max_inverse_attenuation=mle_max_inverse,
                median_inverse_attenuation=mle_median_inverse,
                access_tier="O0+estimated_measurement",
            )
        )
    rows.extend(
        _feature_result_rows(
            cell_id=cell_id,
            seed=seed,
            prefix="POINT_KNOWN",
            h=h,
            feature=point_known,
            oracle_mean=oracle_mean,
            oracle_derivative=oracle_derivative,
            truth=truth,
            n_train=n_train,
            rho=rho,
            horizon=horizon,
            band=band,
            rho_estimate=rho,
            max_inverse_attenuation=known_max_inverse,
            median_inverse_attenuation=known_median_inverse,
            access_tier="M1",
        )
    )
    rows.extend(
        _feature_result_rows(
            cell_id=cell_id,
            seed=seed,
            prefix="POINT_ESTIMATED",
            h=h,
            feature=point_estimated,
            oracle_mean=oracle_mean,
            oracle_derivative=oracle_derivative,
            truth=truth,
            n_train=n_train,
            rho=rho,
            horizon=horizon,
            band=band,
            rho_estimate=rho_estimate,
            max_inverse_attenuation=estimated_max_inverse,
            median_inverse_attenuation=estimated_median_inverse,
            access_tier="O0+estimated_measurement",
        )
    )
    rows.extend(
        _feature_result_rows(
            cell_id=cell_id,
            seed=seed,
            prefix="CF_DECONV_KNOWN",
            h=h,
            feature=cf_known,
            oracle_mean=oracle_mean,
            oracle_derivative=oracle_derivative,
            truth=truth,
            n_train=n_train,
            rho=rho,
            horizon=horizon,
            band=band,
            rho_estimate=rho,
            max_inverse_attenuation=known_max_inverse,
            median_inverse_attenuation=known_median_inverse,
            access_tier="M1",
        )
    )
    rows.extend(
        _feature_result_rows(
            cell_id=cell_id,
            seed=seed,
            prefix="CF_DECONV_ESTIMATED",
            h=h,
            feature=cf_estimated,
            oracle_mean=oracle_mean,
            oracle_derivative=oracle_derivative,
            truth=truth,
            n_train=n_train,
            rho=rho,
            horizon=horizon,
            band=band,
            rho_estimate=rho_estimate,
            max_inverse_attenuation=estimated_max_inverse,
            median_inverse_attenuation=estimated_median_inverse,
            access_tier="O0+estimated_measurement",
        )
    )
    rows.append(
        {
            "experiment_id": "E8A",
            "cell_id": cell_id,
            "dgp_family": "filtered_latent_recovery",
            "dgp_seed": seed,
            "method": "ZERO",
            "method_init": 0,
            "access_tier": "O0",
            "n_train": n_train,
            "rho": rho,
            "horizon": horizon,
            "frequency_band": band,
            "rho_estimate": rho_estimate,
            "rho_abs_error": abs(rho_estimate - rho),
            "kernel_l2_norm": float(np.linalg.norm(true_kernel)),
            "max_inverse_attenuation": known_max_inverse,
            "median_inverse_attenuation": known_median_inverse,
            "nrmse": 1.0,
            "calibration_slope": 0.0,
            "sign_accuracy": 0.0,
            "calibration_slope_defined": 1.0,
            "sign_accuracy_defined": 1.0,
            "coverage_fraction": 0.0,
            "max_abs_z": 0.0,
            "detected_bonferroni": 0.0,
            "warning_code": "normalization_baseline",
            "fit_status": "ok",
        }
    )
    if mle_only:
        return [row for row in rows if "MIXTURE_MLE" in str(row["method"])]
    return rows


def command_e8_adjudication(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = _parse_number_list(args.sample_sizes, int)
    rhos = _parse_number_list(args.rhos, float)
    horizons = _parse_number_list(args.horizons, int)
    bands = _parse_number_list(args.frequency_bands, str)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    write_frozen_config(
        run_dir,
        {
            "command": "e8-adjudication",
            "sample_sizes": sample_sizes,
            "rhos": rhos,
            "horizons": horizons,
            "frequency_bands": bands,
            "noise_sd": args.noise_sd,
            "include_mixture_mle": bool(
                args.include_mixture_mle or args.mle_only
            ),
            "mle_only": bool(args.mle_only),
            "seeds": seeds,
            "estimators": [
                "latent_oracle",
                "point_projection_known",
                "point_projection_estimated",
                "exact_characteristic_deconvolution_known",
                "exact_characteristic_deconvolution_estimated",
                "zero",
            ],
            "stage": args.stage,
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for n_train in sample_sizes:
        for rho in rhos:
            for horizon in horizons:
                for band in bands:
                    for seed in seeds:
                        resource_guard()
                        print(
                            f"E8A rho={rho:g} horizon={horizon} n={n_train} "
                            f"band={band} seed={seed}",
                            flush=True,
                        )
                        rows.extend(
                            _fit_e8_adjudication(
                                seed,
                                n_train,
                                rho,
                                horizon,
                                band,
                                args.noise_sd,
                                bool(args.include_mixture_mle or args.mle_only),
                                bool(args.mle_only),
                            )
                        )
    method_count = (
        6
        if args.mle_only
        else 22
        if args.include_mixture_mle
        else 16
    )
    expected = (
        len(sample_sizes)
        * len(rhos)
        * len(horizons)
        * len(bands)
        * len(seeds)
        * method_count
    )
    validation = _package_rows(run_dir, rows, expected)
    validation.update(
        {
            "known_exact_characteristic_deconvolution_complete": bool(
                not args.mle_only
            ),
            "estimated_filter_deconvolution_complete": bool(not args.mle_only),
            "mixture_likelihood_filter_estimation_complete": bool(
                args.include_mixture_mle or args.mle_only
            ),
            "horizon_scaling_complete": len(horizons) > 1,
            "sample_scaling_complete": len(sample_sizes) > 1,
            "frequency_conditioning_ablation_complete": len(bands) > 1,
            "stage": args.stage,
        }
    )
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


@dataclass(frozen=True)
class GeometryAuditCell:
    history_dim: int
    response_dim: int
    nominal_rank: int
    n_train: int
    query_count: int
    intercept_scheme: str
    frequency_band: str


def _geometry_intercepts(scheme: str, rank: int) -> np.ndarray:
    if rank != 3:
        return np.linspace(-1.1, 0.7, rank)
    if scheme == "symmetric":
        return np.asarray([-0.8, 0.0, 0.8])
    if scheme == "identified":
        return np.asarray([-1.2, -0.4, 0.9])
    raise ValueError(scheme)


def _geometry_bank(
    response_dim: int,
    query_count: int,
    band: str,
    seed: int,
) -> CharacteristicBank:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(query_count, response_dim))
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    if band == "very_low":
        amplitudes = np.resize(np.asarray([0.025, 0.05, 0.075, 0.1]), query_count)
    elif band == "low":
        amplitudes = np.resize(np.asarray([0.05, 0.1, 0.15, 0.2]), query_count)
    elif band == "current":
        amplitudes = np.resize(np.asarray([0.1, 0.15, 0.2, 0.3]), query_count)
    else:
        raise ValueError(band)
    amplitudes *= rng.uniform(0.95, 1.05, size=query_count)
    return CharacteristicBank(
        median=np.zeros(response_dim),
        scale=np.ones(response_dim),
        frequencies=directions * amplitudes[:, None],
        seed=seed,
    )


def _truth_geometry(
    h: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    intercept: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi = expit(intercept[None, :] + h @ w.T)
    coefficient = np.mean(
        pi[:, :, None]
        * (1.0 - pi[:, :, None])
        * (1.0 - 2.0 * pi[:, :, None])
        * w[None, :, :],
        axis=0,
    ).T
    covariance = np.zeros((h.shape[1], b.shape[1], b.shape[1]))
    for direction in range(h.shape[1]):
        covariance[direction] = sum(
            coefficient[direction, component]
            * np.outer(b[component], b[component])
            for component in range(len(intercept))
        )
    return covariance, coefficient, pi


def _mean_dense_feature_derivative(
    h: np.ndarray,
    bank: CharacteristicBank,
    w: np.ndarray,
    b: np.ndarray,
    intercept: np.ndarray,
    batch_size: int = 2_048,
) -> np.ndarray:
    total = np.zeros((h.shape[1], 2 * bank.q), dtype=float)
    count = 0
    for start in range(0, len(h), batch_size):
        batch = h[start : start + batch_size]
        _, derivative = _dense_oracle_complex(
            batch, bank, w, b, intercept
        )
        total[:, : bank.q] += np.sum(derivative.real, axis=0)
        total[:, bank.q :] += np.sum(derivative.imag, axis=0)
        count += len(batch)
    return total / max(count, 1)


def _effective_rank(values: np.ndarray, relative_threshold: float = 1e-4) -> int:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or float(np.max(values)) <= 1e-14:
        return 0
    return int(np.sum(values > relative_threshold * float(np.max(values))))


def _geometry_subspace_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    aggregate_estimate = np.sum(
        [matrix @ matrix.T for matrix in estimate], axis=0
    )
    aggregate_truth = np.sum([matrix @ matrix.T for matrix in truth], axis=0)
    estimate_values, estimate_vectors = np.linalg.eigh(aggregate_estimate)
    truth_values, truth_vectors = np.linalg.eigh(aggregate_truth)
    truth_values = np.maximum(truth_values, 0.0)
    response_rank = _effective_rank(truth_values)
    selected_rank = _effective_rank(np.maximum(estimate_values, 0.0), 0.05)
    if response_rank > 0:
        estimate_space = estimate_vectors[:, np.argsort(estimate_values)[-response_rank:]]
        truth_space = truth_vectors[:, np.argsort(truth_values)[-response_rank:]]
        response_angle = float(
            np.max(np.degrees(subspace_angles(estimate_space, truth_space)))
        )
    else:
        response_angle = 0.0
    flattened_estimate = estimate.reshape(estimate.shape[0], -1)
    flattened_truth = truth.reshape(truth.shape[0], -1)
    _, source_values, _ = np.linalg.svd(flattened_truth, full_matrices=False)
    source_rank = _effective_rank(source_values)
    if source_rank > 0:
        estimate_left = np.linalg.svd(
            flattened_estimate, full_matrices=False
        )[0][:, :source_rank]
        truth_left = np.linalg.svd(flattened_truth, full_matrices=False)[0][
            :, :source_rank
        ]
        source_angle = float(
            np.max(np.degrees(subspace_angles(estimate_left, truth_left)))
        )
    else:
        source_angle = 0.0
    positive_truth = truth_values[truth_values > 1e-14]
    response_condition = (
        float(np.max(positive_truth) / np.min(positive_truth))
        if positive_truth.size
        else 0.0
    )
    return {
        "response_effective_rank": float(response_rank),
        "source_effective_rank": float(source_rank),
        "selected_rank": float(selected_rank),
        "rank_error_effective": float(abs(selected_rank - response_rank)),
        "response_subspace_angle_degrees": response_angle,
        "source_subspace_angle_degrees": source_angle,
        "truth_response_condition_number": response_condition,
    }


def _low_rank_project(
    matrices: np.ndarray,
    rank: int,
) -> np.ndarray:
    if rank <= 0:
        return np.zeros_like(matrices)
    aggregate = np.sum([matrix @ matrix.T for matrix in matrices], axis=0)
    _, vectors = np.linalg.eigh(aggregate)
    space = vectors[:, -rank:]
    projector = space @ space.T
    return np.stack([projector @ matrix @ projector for matrix in matrices])


def _oracle_response_project(
    matrices: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    aggregate = np.sum([matrix @ matrix.T for matrix in truth], axis=0)
    values, vectors = np.linalg.eigh(aggregate)
    rank = _effective_rank(np.maximum(values, 0.0))
    if rank <= 0:
        return np.zeros_like(matrices)
    space = vectors[:, -rank:]
    projector = space @ space.T
    return np.stack([projector @ matrix @ projector for matrix in matrices])


def _characteristic_covariance_inversion(
    feature_tangent: np.ndarray,
    bank: CharacteristicBank,
    ridge: float,
) -> np.ndarray:
    response_dim = bank.raw_frequencies.shape[1]
    pairs = [
        (first, second)
        for first in range(response_dim)
        for second in range(first, response_dim)
    ]
    u = bank.raw_frequencies
    design = np.column_stack(
        [
            u[:, first] ** 2
            if first == second
            else 2.0 * u[:, first] * u[:, second]
            for first, second in pairs
        ]
    )
    matrices: list[np.ndarray] = []
    for direction in range(feature_tangent.shape[0]):
        shifted = (
            feature_tangent[direction, : bank.q]
            + 1j * feature_tangent[direction, bank.q :]
        )
        raw = shifted * np.exp(-1j * bank.raw_phase_offset)
        coefficient = ridge_coefficients(design, -2.0 * raw.real, ridge)
        matrices.append(
            _matrix_from_symmetric(coefficient, pairs, response_dim)
        )
    return np.stack(matrices)


def _fit_e9_adjudication(
    cell: GeometryAuditCell,
    seed: int,
    feature_ridge: float,
    riesz_ridge: float,
    inversion_ridge: float,
) -> list[dict[str, object]]:
    base_cell = GeometryCell(
        name="audit",
        history_dim=cell.history_dim,
        response_dim=cell.response_dim,
        rank=cell.nominal_rank,
        n_train=cell.n_train,
        rotation=True,
        source_sparsity="dense",
        query_count=cell.query_count,
    )
    w, b, _ = _dense_parameters_general(seed, base_cell)
    intercept = _geometry_intercepts(cell.intercept_scheme, cell.nominal_rank)
    bank = _geometry_bank(
        cell.response_dim,
        cell.query_count,
        cell.frequency_band,
        92700 + cell.response_dim + cell.query_count,
    )
    rng = np.random.default_rng(930_000 + 1009 * seed)
    h = rng.normal(size=(cell.n_train, cell.history_dim))
    y = _sample_dense(h, rng, w, b, intercept)
    feature = bank.transform(y)
    covariance_feature, covariance_pairs = _symmetric_vector(y)
    truth_h = _standard_normal_qmc(cell.history_dim, 17, 931_000 + seed)
    truth_covariance, coefficient, _ = _truth_geometry(
        truth_h, w, b, intercept
    )
    truth_feature = _mean_dense_feature_derivative(
        truth_h, bank, w, b, intercept
    )
    methods = ("PLUG", "RIESZ", "ORTH")
    feature_sum = {
        method: np.zeros_like(truth_feature) for method in methods
    }
    feature_sumsq = {
        method: np.zeros_like(truth_feature) for method in methods
    }
    truth_covariance_vector = np.asarray(
        [
            [truth_covariance[p, first, second] for first, second in covariance_pairs]
            for p in range(cell.history_dim)
        ]
    )
    covariance_sum = {
        method: np.zeros_like(truth_covariance_vector) for method in methods
    }
    covariance_sumsq = {
        method: np.zeros_like(truth_covariance_vector) for method in methods
    }
    for fold_id, (train, test) in enumerate(
        independent_folds(cell.n_train, 5, 932_000 + seed)
    ):
        resource_guard()
        sieve = HistorySieve(
            cell.history_dim, 64, 933_000 + 100 * seed + fold_id
        ).fit(h[train])
        z_train = sieve.transform(h[train])
        z_test = sieve.transform(h[test])
        feature_coefficient = ridge_coefficients(
            z_train, feature[train], feature_ridge
        )
        covariance_coefficient = ridge_coefficients(
            z_train, covariance_feature[train], feature_ridge
        )
        feature_prediction = z_test @ feature_coefficient
        covariance_prediction = z_test @ covariance_coefficient
        for direction in range(cell.history_dim):
            dz_train = sieve.derivative(h[train], direction)
            dz_test = sieve.derivative(h[test], direction)
            alpha_coefficient = riesz_coefficients(
                z_train, dz_train, riesz_ridge
            )
            alpha = z_test @ alpha_coefficient
            feature_values = {
                "PLUG": dz_test @ feature_coefficient,
                "RIESZ": alpha[:, None] * feature[test],
                "ORTH": dz_test @ feature_coefficient
                + alpha[:, None] * (feature[test] - feature_prediction),
            }
            covariance_values = {
                "PLUG": dz_test @ covariance_coefficient,
                "RIESZ": alpha[:, None] * covariance_feature[test],
                "ORTH": dz_test @ covariance_coefficient
                + alpha[:, None]
                * (covariance_feature[test] - covariance_prediction),
            }
            for method in methods:
                feature_sum[method][direction] += np.sum(
                    feature_values[method], axis=0
                )
                feature_sumsq[method][direction] += np.sum(
                    feature_values[method] ** 2, axis=0
                )
                covariance_sum[method][direction] += np.sum(
                    covariance_values[method], axis=0
                )
                covariance_sumsq[method][direction] += np.sum(
                    covariance_values[method] ** 2, axis=0
                )
    estimated_features = {
        method: feature_sum[method] / cell.n_train for method in methods
    }
    estimated_covariance_vectors = {
        method: covariance_sum[method] / cell.n_train for method in methods
    }
    direct_matrices = {
        method: np.stack(
            [
                _matrix_from_symmetric(
                    estimated_covariance_vectors[method][direction],
                    covariance_pairs,
                    cell.response_dim,
                )
                for direction in range(cell.history_dim)
            ]
        )
        for method in methods
    }
    characteristic_matrices = {
        method: _characteristic_covariance_inversion(
            estimated_features[method], bank, inversion_ridge
        )
        for method in methods
    }
    oracle_characteristic = _characteristic_covariance_inversion(
        truth_feature, bank, inversion_ridge
    )
    truth_geometry_metrics = _geometry_subspace_metrics(
        truth_covariance, truth_covariance
    )
    effective_rank = int(truth_geometry_metrics["response_effective_rank"])
    matrix_methods: list[tuple[str, np.ndarray, str]] = [
        (f"{method}_DIRECT_COV", direct_matrices[method], "direct")
        for method in methods
    ]
    matrix_methods.extend(
        [
            (
                "ORTH_DIRECT_COV_LOW_RANK_EFFECTIVE",
                _low_rank_project(direct_matrices["ORTH"], effective_rank),
                "direct",
            ),
            (
                "ORACLE_RESPONSE_SUBSPACE_PROJECTED",
                _oracle_response_project(
                    direct_matrices["ORTH"], truth_covariance
                ),
                "direct",
            ),
        ]
    )
    matrix_methods.extend(
        [
            (
                f"{method}_CHARACTERISTIC_INVERSION",
                characteristic_matrices[method],
                "characteristic",
            )
            for method in methods
        ]
    )
    matrix_methods.append(
        (
            "ORACLE_CHARACTERISTIC_INVERSION",
            oracle_characteristic,
            "characteristic_oracle",
        )
    )
    rows: list[dict[str, object]] = []
    cell_id = (
        f"{cell.intercept_scheme}_dy{cell.response_dim}_n{cell.n_train}_"
        f"q{cell.query_count}_{cell.frequency_band}"
    )
    for method, matrices, route in matrix_methods:
        geometry_metrics = _geometry_subspace_metrics(matrices, truth_covariance)
        covariance_metrics = vector_metrics(
            matrices.reshape(-1), truth_covariance.reshape(-1)
        )
        if route.startswith("characteristic"):
            base_method = method.split("_", 1)[0]
            feature_estimate = (
                truth_feature
                if route == "characteristic_oracle"
                else estimated_features[base_method]
            )
            feature_nrmse = vector_metrics(
                feature_estimate.reshape(-1), truth_feature.reshape(-1)
            )["nrmse"]
            if route == "characteristic_oracle":
                coverage = 1.0
                null_fwer = 0.0
            else:
                feature_variance = np.maximum(
                    (
                        feature_sumsq[base_method]
                        - cell.n_train * estimated_features[base_method] ** 2
                    )
                    / max(cell.n_train - 1, 1),
                    0.0,
                )
                feature_se = np.sqrt(feature_variance / cell.n_train)
                coverage = float(
                    np.mean(
                        (truth_feature >= feature_estimate - 1.96 * feature_se)
                        & (truth_feature <= feature_estimate + 1.96 * feature_se)
                    )
                )
                null_z = np.abs(
                    feature_estimate[-2:]
                    / np.maximum(feature_se[-2:], 1e-12)
                )
                critical = norm.ppf(1.0 - 0.05 / (2.0 * null_z.size))
                null_fwer = float(np.any(null_z > critical))
        else:
            feature_nrmse = covariance_metrics["nrmse"]
            base_method = (
                method.split("_", 1)[0]
                if method.startswith(tuple(methods))
                else "ORTH"
            )
            covariance_estimate = estimated_covariance_vectors[base_method]
            covariance_variance = np.maximum(
                (
                    covariance_sumsq[base_method]
                    - cell.n_train * covariance_estimate**2
                )
                / max(cell.n_train - 1, 1),
                0.0,
            )
            covariance_se = np.sqrt(covariance_variance / cell.n_train)
            coverage = float(
                np.mean(
                    (
                        truth_covariance_vector
                        >= covariance_estimate - 1.96 * covariance_se
                    )
                    & (
                        truth_covariance_vector
                        <= covariance_estimate + 1.96 * covariance_se
                    )
                )
            )
            null_z = np.abs(
                covariance_estimate[-2:]
                / np.maximum(covariance_se[-2:], 1e-12)
            )
            critical = norm.ppf(1.0 - 0.05 / (2.0 * null_z.size))
            null_fwer = float(np.any(null_z > critical))
        rows.append(
            {
                "experiment_id": "E9A",
                "cell_id": cell_id,
                "dgp_family": "dense_geometry_identifiability_audit",
                "dgp_seed": seed,
                "method": method,
                "method_init": 0,
                "access_tier": "Q1" if method.startswith("ORACLE") else "O0",
                "n_train": cell.n_train,
                "history_dim": cell.history_dim,
                "response_dim": cell.response_dim,
                "nominal_rank": cell.nominal_rank,
                "intercept_scheme": cell.intercept_scheme,
                "query_count": cell.query_count,
                "frequency_band": cell.frequency_band,
                "feature_tensor_nrmse": feature_nrmse,
                "covariance_tensor_nrmse": covariance_metrics["nrmse"],
                "covariance_calibration_slope": covariance_metrics[
                    "calibration_slope"
                ],
                "coverage_fraction": coverage,
                "null_edge_familywise_call": null_fwer,
                "coefficient_min_singular_value": float(
                    np.min(np.linalg.svd(coefficient, compute_uv=False))
                ),
                "nominal_rank_error": float(
                    abs(geometry_metrics["selected_rank"] - cell.nominal_rank)
                ),
                **geometry_metrics,
                "warning_code": (
                    "nominal_rank_not_identified"
                    if effective_rank < cell.nominal_rank
                    else ""
                ),
                "fit_status": "ok",
            }
        )
    rows.append(
        {
            "experiment_id": "E9A",
            "cell_id": cell_id,
            "dgp_family": "dense_geometry_identifiability_audit",
            "dgp_seed": seed,
            "method": "ZERO",
            "method_init": 0,
            "access_tier": "O0",
            "n_train": cell.n_train,
            "history_dim": cell.history_dim,
            "response_dim": cell.response_dim,
            "nominal_rank": cell.nominal_rank,
            "intercept_scheme": cell.intercept_scheme,
            "query_count": cell.query_count,
            "frequency_band": cell.frequency_band,
            "feature_tensor_nrmse": 1.0,
            "covariance_tensor_nrmse": 1.0,
            "covariance_calibration_slope": 0.0,
            "coverage_fraction": 0.0,
            "null_edge_familywise_call": 0.0,
            "coefficient_min_singular_value": float(
                np.min(np.linalg.svd(coefficient, compute_uv=False))
            ),
            "nominal_rank_error": float(cell.nominal_rank),
            **truth_geometry_metrics,
            "selected_rank": 0.0,
            "rank_error_effective": float(effective_rank),
            "response_subspace_angle_degrees": 90.0,
            "source_subspace_angle_degrees": 90.0,
            "warning_code": "normalization_baseline",
            "fit_status": "ok",
        }
    )
    return rows


def _fit_e9_direct_only(
    cell: GeometryAuditCell,
    seed: int,
    feature_ridge: float,
    riesz_ridge: float,
) -> list[dict[str, object]]:
    base_cell = GeometryCell(
        "direct_audit",
        cell.history_dim,
        cell.response_dim,
        cell.nominal_rank,
        cell.n_train,
        True,
        "dense",
        0,
    )
    w, b, _ = _dense_parameters_general(seed, base_cell)
    intercept = _geometry_intercepts(cell.intercept_scheme, cell.nominal_rank)
    rng = np.random.default_rng(940_000 + 1009 * seed)
    h = rng.normal(size=(cell.n_train, cell.history_dim))
    y = _sample_dense(h, rng, w, b, intercept)
    covariance_feature, pairs = _symmetric_vector(y)
    truth_h = _standard_normal_qmc(cell.history_dim, 17, 941_000 + seed)
    truth_covariance, coefficient, _ = _truth_geometry(
        truth_h, w, b, intercept
    )
    truth_vector = np.asarray(
        [
            [truth_covariance[p, first, second] for first, second in pairs]
            for p in range(cell.history_dim)
        ]
    )
    methods = ("PLUG", "RIESZ", "ORTH")
    sums = {method: np.zeros_like(truth_vector) for method in methods}
    sumsquares = {method: np.zeros_like(truth_vector) for method in methods}
    for fold_id, (train, test) in enumerate(
        independent_folds(cell.n_train, 5, 942_000 + seed)
    ):
        resource_guard()
        sieve = HistorySieve(
            cell.history_dim, 64, 943_000 + 100 * seed + fold_id
        ).fit(h[train])
        z_train = sieve.transform(h[train])
        z_test = sieve.transform(h[test])
        coefficient_fit = ridge_coefficients(
            z_train, covariance_feature[train], feature_ridge
        )
        prediction = z_test @ coefficient_fit
        for direction in range(cell.history_dim):
            dz_train = sieve.derivative(h[train], direction)
            dz_test = sieve.derivative(h[test], direction)
            alpha_coefficient = riesz_coefficients(
                z_train, dz_train, riesz_ridge
            )
            alpha = z_test @ alpha_coefficient
            values = {
                "PLUG": dz_test @ coefficient_fit,
                "RIESZ": alpha[:, None] * covariance_feature[test],
                "ORTH": dz_test @ coefficient_fit
                + alpha[:, None] * (covariance_feature[test] - prediction),
            }
            for method in methods:
                sums[method][direction] += np.sum(values[method], axis=0)
                sumsquares[method][direction] += np.sum(
                    values[method] ** 2, axis=0
                )
    estimates = {
        method: sums[method] / cell.n_train for method in methods
    }
    matrices = {
        method: np.stack(
            [
                _matrix_from_symmetric(
                    estimates[method][direction], pairs, cell.response_dim
                )
                for direction in range(cell.history_dim)
            ]
        )
        for method in methods
    }
    truth_geometry = _geometry_subspace_metrics(
        truth_covariance, truth_covariance
    )
    effective_rank = int(truth_geometry["response_effective_rank"])
    candidates: list[tuple[str, np.ndarray, str]] = [
        (f"{method}_DIRECT_COV", matrices[method], method)
        for method in methods
    ]
    candidates.extend(
        [
            (
                "ORTH_DIRECT_COV_LOW_RANK_EFFECTIVE",
                _low_rank_project(matrices["ORTH"], effective_rank),
                "ORTH",
            ),
            (
                "ORACLE_RESPONSE_SUBSPACE_PROJECTED",
                _oracle_response_project(matrices["ORTH"], truth_covariance),
                "ORTH",
            ),
        ]
    )
    cell_id = (
        f"{cell.intercept_scheme}_dy{cell.response_dim}_n{cell.n_train}_direct"
    )
    rows: list[dict[str, object]] = []
    for method, matrix, base_method in candidates:
        metric = vector_metrics(matrix.reshape(-1), truth_covariance.reshape(-1))
        geometry = _geometry_subspace_metrics(matrix, truth_covariance)
        estimate = estimates[base_method]
        variance = np.maximum(
            (
                sumsquares[base_method] - cell.n_train * estimate**2
            )
            / max(cell.n_train - 1, 1),
            0.0,
        )
        standard_error = np.sqrt(variance / cell.n_train)
        coverage = float(
            np.mean(
                (truth_vector >= estimate - 1.96 * standard_error)
                & (truth_vector <= estimate + 1.96 * standard_error)
            )
        )
        null_z = np.abs(
            estimate[-2:] / np.maximum(standard_error[-2:], 1e-12)
        )
        critical = norm.ppf(1.0 - 0.05 / (2.0 * null_z.size))
        rows.append(
            {
                "experiment_id": "E9A_DIRECT",
                "cell_id": cell_id,
                "dgp_family": "dense_geometry_direct_scaling",
                "dgp_seed": seed,
                "method": method,
                "method_init": 0,
                "access_tier": "Q1" if method.startswith("ORACLE") else "O0",
                "n_train": cell.n_train,
                "history_dim": cell.history_dim,
                "response_dim": cell.response_dim,
                "nominal_rank": cell.nominal_rank,
                "intercept_scheme": cell.intercept_scheme,
                "query_count": 0,
                "frequency_band": "not_applicable",
                "feature_tensor_nrmse": metric["nrmse"],
                "covariance_tensor_nrmse": metric["nrmse"],
                "covariance_calibration_slope": metric["calibration_slope"],
                "coverage_fraction": coverage,
                "null_edge_familywise_call": float(np.any(null_z > critical)),
                "coefficient_min_singular_value": float(
                    np.min(np.linalg.svd(coefficient, compute_uv=False))
                ),
                "nominal_rank_error": float(
                    abs(geometry["selected_rank"] - cell.nominal_rank)
                ),
                **geometry,
                "warning_code": "",
                "fit_status": "ok",
            }
        )
    zero = {
        **rows[0],
        "method": "ZERO",
        "access_tier": "O0",
        "feature_tensor_nrmse": 1.0,
        "covariance_tensor_nrmse": 1.0,
        "covariance_calibration_slope": 0.0,
        "coverage_fraction": 0.0,
        "null_edge_familywise_call": 0.0,
        "selected_rank": 0.0,
        "rank_error_effective": float(effective_rank),
        "nominal_rank_error": float(cell.nominal_rank),
        "response_subspace_angle_degrees": 90.0,
        "source_subspace_angle_degrees": 90.0,
        "warning_code": "normalization_baseline",
    }
    rows.append(zero)
    return rows


def command_e9_adjudication(args: argparse.Namespace) -> int:
    _force_thread_limits()
    if args.direct_only and args.inversion_only:
        raise ValueError("--direct-only and --inversion-only are mutually exclusive")
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = _parse_number_list(args.sample_sizes, int)
    response_dims = _parse_number_list(args.response_dims, int)
    query_counts = _parse_number_list(args.query_counts, int)
    intercept_schemes = _parse_number_list(args.intercept_schemes, str)
    frequency_bands = _parse_number_list(args.frequency_bands, str)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    cells = [
        GeometryAuditCell(
            history_dim=8,
            response_dim=response_dim,
            nominal_rank=3,
            n_train=n_train,
            query_count=query_count,
            intercept_scheme=intercept_scheme,
            frequency_band=frequency_band,
        )
        for n_train in sample_sizes
        for response_dim in response_dims
        for query_count in query_counts
        for intercept_scheme in intercept_schemes
        for frequency_band in frequency_bands
    ]
    write_frozen_config(
        run_dir,
        {
            "command": "e9-adjudication",
            "cells": [cell.__dict__ for cell in cells],
            "seeds": seeds,
            "feature_ridge": args.feature_ridge,
            "riesz_ridge": args.riesz_ridge,
            "inversion_ridge": args.inversion_ridge,
            "inversion_only": bool(args.inversion_only),
            "direct_only": bool(args.direct_only),
            "rank_definition": "relative eigenvalue threshold 1e-4 on population integrated covariance tensor",
            "stage": args.stage,
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for cell in cells:
        for seed in seeds:
            resource_guard()
            print(
                f"E9A scheme={cell.intercept_scheme} dy={cell.response_dim} "
                f"n={cell.n_train} q={cell.query_count} "
                f"band={cell.frequency_band} seed={seed}",
                flush=True,
            )
            cell_rows = (
                _fit_e9_direct_only(
                    cell,
                    seed,
                    args.feature_ridge,
                    args.riesz_ridge,
                )
                if args.direct_only
                else _fit_e9_adjudication(
                    cell,
                    seed,
                    args.feature_ridge,
                    args.riesz_ridge,
                    args.inversion_ridge,
                )
            )
            if args.inversion_only:
                cell_rows = [
                    row
                    for row in cell_rows
                    if "CHARACTERISTIC_INVERSION" in str(row["method"])
                ]
            rows.extend(cell_rows)
    method_count = 6 if args.direct_only else 4 if args.inversion_only else 10
    expected = len(cells) * len(seeds) * method_count
    validation = _package_rows(run_dir, rows, expected)
    validation.update(
        {
            "nominal_vs_effective_rank_audit_complete": True,
            "oracle_characteristic_inversion_complete": bool(
                not args.direct_only
            ),
            "plug_riesz_orth_decomposition_complete": True,
            "sample_scaling_complete": len(sample_sizes) > 1,
            "query_scaling_complete": len(query_counts) > 1,
            "stage": args.stage,
        }
    )
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def command_e9_oracle_inversion(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    response_dims = _parse_number_list(args.response_dims, int)
    query_counts = _parse_number_list(args.query_counts, int)
    schemes = _parse_number_list(args.intercept_schemes, str)
    bands = _parse_number_list(args.frequency_bands, str)
    ridges = _parse_number_list(args.inversion_ridges, float)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    write_frozen_config(
        run_dir,
        {
            "command": "e9-oracle-inversion",
            "response_dims": response_dims,
            "query_counts": query_counts,
            "intercept_schemes": schemes,
            "frequency_bands": bands,
            "inversion_ridges": ridges,
            "seeds": seeds,
            "access_tier": "Q1",
            "stage": "development",
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for response_dim in response_dims:
        for query_count in query_counts:
            for scheme in schemes:
                for band in bands:
                    for seed in seeds:
                        base = GeometryCell(
                            "oracle",
                            8,
                            response_dim,
                            3,
                            0,
                            True,
                            "dense",
                            query_count,
                        )
                        w, b, _ = _dense_parameters_general(seed, base)
                        intercept = _geometry_intercepts(scheme, 3)
                        bank = _geometry_bank(
                            response_dim,
                            query_count,
                            band,
                            92700 + response_dim + query_count,
                        )
                        truth_h = _standard_normal_qmc(8, 17, 931_000 + seed)
                        truth_covariance, coefficient, _ = _truth_geometry(
                            truth_h, w, b, intercept
                        )
                        truth_feature = _mean_dense_feature_derivative(
                            truth_h, bank, w, b, intercept
                        )
                        for ridge in ridges:
                            estimate = _characteristic_covariance_inversion(
                                truth_feature, bank, ridge
                            )
                            metrics = vector_metrics(
                                estimate.reshape(-1), truth_covariance.reshape(-1)
                            )
                            geometry = _geometry_subspace_metrics(
                                estimate, truth_covariance
                            )
                            rows.append(
                                {
                                    "experiment_id": "E9A_ORACLE",
                                    "cell_id": (
                                        f"{scheme}_dy{response_dim}_q{query_count}_"
                                        f"{band}_r{ridge:.0e}"
                                    ),
                                    "dgp_family": "oracle_characteristic_inversion",
                                    "dgp_seed": seed,
                                    "method": "ORACLE_CHARACTERISTIC_INVERSION",
                                    "method_init": 0,
                                    "access_tier": "Q1",
                                    "n_train": 0,
                                    "response_dim": response_dim,
                                    "query_count": query_count,
                                    "intercept_scheme": scheme,
                                    "frequency_band": band,
                                    "inversion_ridge": ridge,
                                    "feature_tensor_nrmse": 0.0,
                                    "covariance_tensor_nrmse": metrics["nrmse"],
                                    "covariance_calibration_slope": metrics[
                                        "calibration_slope"
                                    ],
                                    "coefficient_min_singular_value": float(
                                        np.min(
                                            np.linalg.svd(
                                                coefficient, compute_uv=False
                                            )
                                        )
                                    ),
                                    **geometry,
                                    "warning_code": "oracle_access",
                                    "fit_status": "ok",
                                }
                            )
    expected = (
        len(response_dims)
        * len(query_counts)
        * len(schemes)
        * len(bands)
        * len(seeds)
        * len(ridges)
    )
    validation = _package_rows(run_dir, rows, expected)
    validation.update(
        {
            "oracle_only_no_observational_fit": True,
            "query_frequency_ridge_grid_complete": True,
            "stage": "development",
        }
    )
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distributional-sid-adjudication")
    subparsers = parser.add_subparsers(dest="command", required=True)

    e6 = subparsers.add_parser("e6-adjudication")
    e6.add_argument("--run-dir", required=True)
    e6.add_argument("--cells", default="gaussian,t3,t5,gap3,gap4")
    e6.add_argument("--sample-sizes", default="8000")
    e6.add_argument("--widths", default="32")
    e6.add_argument("--riesz-ridges", default="1e-2")
    e6.add_argument("--feature-ridge", type=float, default=1e-3)
    e6.add_argument("--include-density-score", action="store_true")
    e6.add_argument("--seed-start", type=int, default=2001)
    e6.add_argument("--seed-end", type=int, default=2010)
    e6.add_argument("--stage", choices=("development", "posthoc"), default="posthoc")
    e6.set_defaults(func=command_e6_adjudication)

    e8 = subparsers.add_parser("e8-adjudication")
    e8.add_argument("--run-dir", required=True)
    e8.add_argument("--sample-sizes", default="8000,32000")
    e8.add_argument("--rhos", default="0.9,0.98")
    e8.add_argument("--horizons", default="4,16,64")
    e8.add_argument("--frequency-bands", default="low,wide")
    e8.add_argument("--noise-sd", type=float, default=0.05)
    e8.add_argument("--include-mixture-mle", action="store_true")
    e8.add_argument("--mle-only", action="store_true")
    e8.add_argument("--seed-start", type=int, default=2001)
    e8.add_argument("--seed-end", type=int, default=2010)
    e8.add_argument("--stage", choices=("development", "posthoc"), default="posthoc")
    e8.set_defaults(func=command_e8_adjudication)

    e9 = subparsers.add_parser("e9-adjudication")
    e9.add_argument("--run-dir", required=True)
    e9.add_argument("--sample-sizes", default="8000,32000")
    e9.add_argument("--response-dims", default="8")
    e9.add_argument("--query-counts", default="64")
    e9.add_argument("--intercept-schemes", default="symmetric,identified")
    e9.add_argument("--frequency-bands", default="current")
    e9.add_argument("--feature-ridge", type=float, default=1e-3)
    e9.add_argument("--riesz-ridge", type=float, default=1e-2)
    e9.add_argument("--inversion-ridge", type=float, default=1e-4)
    e9.add_argument("--inversion-only", action="store_true")
    e9.add_argument("--direct-only", action="store_true")
    e9.add_argument("--seed-start", type=int, default=2001)
    e9.add_argument("--seed-end", type=int, default=2010)
    e9.add_argument("--stage", choices=("development", "posthoc"), default="posthoc")
    e9.set_defaults(func=command_e9_adjudication)

    e9_oracle = subparsers.add_parser("e9-oracle-inversion")
    e9_oracle.add_argument("--run-dir", required=True)
    e9_oracle.add_argument("--response-dims", default="8")
    e9_oracle.add_argument("--query-counts", default="64,256")
    e9_oracle.add_argument("--intercept-schemes", default="identified")
    e9_oracle.add_argument("--frequency-bands", default="very_low,low,current")
    e9_oracle.add_argument("--inversion-ridges", default="1e-8,1e-6,1e-4")
    e9_oracle.add_argument("--seed-start", type=int, default=1101)
    e9_oracle.add_argument("--seed-end", type=int, default=1110)
    e9_oracle.set_defaults(func=command_e9_oracle_inversion)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
