from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.special import expit, logit, ndtri
from scipy.stats import norm, qmc

from .core import (
    AnalyticGaussianPathDGP,
    CharacteristicBank,
    HistorySieve,
    MomentBlindDiscreteDGP,
    MomentBlindLegendreDGP,
    SupportMotionDGP,
    atomic_csv,
    atomic_json,
    embargoed_block_splits,
    independent_folds,
    resource_guard,
    ridge_coefficients,
    riesz_coefficients,
    vector_metrics,
)
from .experiments import freeze_source_and_environment, write_frozen_config
from .tier_a_extensions import (
    ObservationCell,
    _dense_oracle_complex,
    _feature_score,
    _matrix_from_symmetric,
    _observation_kernel,
    _observation_oracle,
    _package_rows,
    _path_feature_oracle,
    _path_templates,
    _sample_dense,
    _sample_observation_paths,
    _sample_paths,
    _subspace_metrics,
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


def _standard_normal_qmc(dimension: int, power: int, seed: int) -> np.ndarray:
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)
    return ndtri(np.clip(unit, 1e-12, 1.0 - 1e-12))


def _ar1_histories(
    rng: np.random.Generator, n: int, dimension: int, rho: float
) -> np.ndarray:
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    innovation = rng.normal(size=(n, dimension))
    output = np.empty_like(innovation)
    output[0] = innovation[0]
    scale = math.sqrt(1.0 - rho**2)
    for index in range(1, n):
        output[index] = rho * output[index - 1] + scale * innovation[index]
    return output


def _hac_standard_error(values: np.ndarray, bandwidth: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    n = centered.shape[0]
    long_variance = np.sum(centered * centered, axis=0) / n
    for lag in range(1, min(int(bandwidth), n - 1) + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        covariance = np.sum(centered[lag:] * centered[:-lag], axis=0) / n
        long_variance += 2.0 * weight * covariance
    return np.sqrt(np.maximum(long_variance, 0.0) / n)


def _feature_score_with_splits(
    h: np.ndarray,
    feature: np.ndarray,
    truth: np.ndarray,
    dgp_seed: int,
    method_init: int,
    split_mode: str,
    footprint: int = 8,
) -> dict[str, dict[str, float]]:
    n = len(h)
    if split_mode == "random":
        splits = independent_folds(n, 5, 811_000 + dgp_seed)
    elif split_mode == "blocked":
        splits = embargoed_block_splits(n, 5, footprint)
    else:
        raise ValueError(split_mode)
    scores = {
        method: np.zeros((n, feature.shape[1]), dtype=float)
        for method in ("PLUG", "RIESZ", "ORTH")
    }
    for fold_id, (train, test) in enumerate(splits):
        sieve = HistorySieve(
            h.shape[1], 64, 812_000 + 100 * method_init + fold_id
        ).fit(h[train])
        z_train = sieve.transform(h[train])
        z_test = sieve.transform(h[test])
        dz_train = sieve.derivative(h[train], 0)
        dz_test = sieve.derivative(h[test], 0)
        feature_coefficient = ridge_coefficients(z_train, feature[train], 1e-3)
        alpha_coefficient = riesz_coefficients(z_train, dz_train, 1e-2)
        prediction = z_test @ feature_coefficient
        derivative = dz_test @ feature_coefficient
        alpha = z_test @ alpha_coefficient
        scores["PLUG"][test] = derivative
        scores["RIESZ"][test] = alpha[:, None] * feature[test]
        scores["ORTH"][test] = derivative + alpha[:, None] * (
            feature[test] - prediction
        )
    bandwidth = max(footprint, int(math.ceil(1.5 * n ** (1.0 / 3.0))))
    output: dict[str, dict[str, float]] = {}
    for method, score in scores.items():
        estimate = np.mean(score, axis=0)
        iid_se = np.std(score, axis=0, ddof=1) / math.sqrt(n)
        hac_se = _hac_standard_error(score, bandwidth)
        metrics = vector_metrics(estimate, truth)
        if not np.isfinite(metrics["calibration_slope"]):
            metrics["calibration_slope"] = 0.0
        if not np.isfinite(metrics["sign_accuracy"]):
            metrics["sign_accuracy"] = 0.0
        iid_variance = np.mean(iid_se**2)
        hac_variance = np.mean(hac_se**2)
        output[method] = {
            **metrics,
            "coverage_fraction_iid": float(
                np.mean(
                    (truth >= estimate - 1.96 * iid_se)
                    & (truth <= estimate + 1.96 * iid_se)
                )
            ),
            "coverage_fraction_hac": float(
                np.mean(
                    (truth >= estimate - 1.96 * hac_se)
                    & (truth <= estimate + 1.96 * hac_se)
                )
            ),
            "effective_sample_fraction": float(
                np.clip(iid_variance / max(hac_variance, 1e-16), 0.0, 1.0)
            ),
            "hac_bandwidth": float(bandwidth),
            "max_abs_z_iid": float(
                np.max(np.abs(estimate / np.maximum(iid_se, 1e-12)))
            ),
            "max_abs_z_hac": float(
                np.max(np.abs(estimate / np.maximum(hac_se, 1e-12)))
            ),
        }
    return output


def command_e3_dependent(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        {
            "cell_id": "d16_n8000_rho0p5",
            "history_dim": 16,
            "n_train": 8000,
            "rho": 0.5,
        },
        {
            "cell_id": "d16_n8000_rho0p9",
            "history_dim": 16,
            "n_train": 8000,
            "rho": 0.9,
        },
        {
            "cell_id": "d32_n32000_rho0p9",
            "history_dim": 32,
            "n_train": 32000,
            "rho": 0.9,
        },
    ]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    initializations = list(range(args.initializations))
    payload = {
        "command": "confirm-e3-dependent-extension",
        "stage": "confirmation_extension",
        "cells": cells,
        "seeds": seeds,
        "initializations": initializations,
        "split_modes": ["blocked", "random"],
        "uncertainty": "Newey-West/Bartlett HAC with frozen 1.5*n^(1/3) bandwidth and minimum footprint 8",
        "query_count": 64,
        "one_future_per_history": True,
    }
    write_frozen_config(run_dir, payload)
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(cells):
        for seed in seeds:
            resource_guard()
            dgp = AnalyticGaussianPathDGP(
                seed, history_dim=cell["history_dim"], response_dim=4
            )
            rng = np.random.default_rng(810_000 + 1009 * seed + 10_000 * cell_index)
            h_bank = dgp.sample_histories(rng, 1024)
            y_bank = dgp.sample_responses(h_bank, rng)
            bank = CharacteristicBank.fit(y_bank, 64, seed=92501)
            h = _ar1_histories(
                rng, cell["n_train"], cell["history_dim"], cell["rho"]
            )
            y = dgp.sample_responses(h, rng)
            feature = bank.transform(y)
            truth_h = dgp.qmc_histories(15, 813_000 + seed)
            truth = np.mean(
                dgp.conditional_feature_derivative(truth_h, 0, bank), axis=0
            )
            empirical_rho = float(np.corrcoef(h[:-1, 0], h[1:, 0])[0, 1])
            for method_init in initializations:
                for split_mode in ("blocked", "random"):
                    print(
                        f"E3-dependent cell={cell['cell_id']} seed={seed} "
                        f"init={method_init} split={split_mode}",
                        flush=True,
                    )
                    result = _feature_score_with_splits(
                        h, feature, truth, seed + 10_000 * cell_index,
                        method_init, split_mode
                    )
                    for method, metrics in result.items():
                        rows.append(
                            {
                                "experiment_id": "E3",
                                "cell_id": cell["cell_id"],
                                "dgp_family": "dependent_conditional_gaussian",
                                "dgp_seed": seed,
                                "method": method,
                                "method_init": method_init,
                                "access_tier": "O0",
                                "n_train": cell["n_train"],
                                "history_dim": cell["history_dim"],
                                "rho": cell["rho"],
                                "empirical_rho": empirical_rho,
                                "split_mode": split_mode,
                                **metrics,
                                "warning_code": (
                                    "random_split_negative_control"
                                    if split_mode == "random" else ""
                                ),
                                "fit_status": "ok",
                            }
                        )
    expected = len(cells) * len(seeds) * len(initializations) * 2 * 3
    validation = _package_rows(run_dir, rows, expected)
    validation.update(
        {
            "dependent_cells_complete": True,
            "blocked_and_random_splits_complete": True,
            "hac_uncertainty_complete": True,
            "observation_score_storage_complete": False,
        }
    )
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


@dataclass(frozen=True)
class MeasurementCell:
    name: str
    rho: float = 0.5
    latent_amplitude: float = 1.0
    crosstalk: float = 0.0
    noise_sd: float = 0.2
    heteroskedastic_strength: float = 0.0
    missing_rate: float = 0.0
    missing_history_coefficient: float = 0.0
    include_mask: bool = False
    seeds: int = 30


def _measurement_noise(
    h: np.ndarray, cell: MeasurementCell
) -> tuple[np.ndarray, np.ndarray]:
    activation = expit(h[:, 0])
    sigma = cell.noise_sd * (1.0 + cell.heteroskedastic_strength * activation)
    derivative = (
        cell.noise_sd
        * cell.heteroskedastic_strength
        * activation
        * (1.0 - activation)
    )
    return sigma, derivative


def _measurement_observation_probability(
    h: np.ndarray, cell: MeasurementCell
) -> tuple[np.ndarray, np.ndarray]:
    if cell.missing_rate <= 0.0:
        return np.ones(len(h)), np.zeros(len(h))
    intercept = logit(1.0 - cell.missing_rate)
    probability = expit(intercept + cell.missing_history_coefficient * h[:, 0])
    derivative = (
        cell.missing_history_coefficient * probability * (1.0 - probability)
    )
    return probability, derivative


def _sample_measurement_paths(
    h: np.ndarray,
    rng: np.random.Generator,
    cell: MeasurementCell,
    horizon: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi = expit(-0.3 + h[:, 0])
    state = rng.uniform(size=len(h)) < pi
    latent = cell.latent_amplitude * (state.astype(float) - pi)
    kernel = _observation_kernel(cell.rho, horizon)
    signal = (
        latent[:, None] * kernel[None, :]
        + cell.crosstalk * h[:, [0]] * kernel[None, :]
    )
    sigma, _ = _measurement_noise(h, cell)
    observed = signal + sigma[:, None] * rng.normal(size=signal.shape)
    probability, _ = _measurement_observation_probability(h, cell)
    mask = rng.uniform(size=len(h)) < probability
    if cell.missing_rate > 0.0:
        observed = observed * mask[:, None]
    output = (
        np.column_stack([observed, mask.astype(float)])
        if cell.include_mask else observed
    )
    return output, latent, mask.astype(float)


def _measurement_oracle(
    h: np.ndarray,
    bank: CharacteristicBank,
    cell: MeasurementCell,
    horizon: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    u_all = bank.raw_frequencies
    if cell.include_mask:
        u = u_all[:, :horizon]
        u_mask = u_all[:, horizon]
    else:
        u = u_all
        u_mask = np.zeros(bank.q)
    pi = expit(-0.3 + h[:, 0])
    dpi = pi * (1.0 - pi)
    kernel = _observation_kernel(cell.rho, horizon)
    uk = u @ kernel
    qv = cell.latent_amplitude * uk
    exp_iq = np.exp(1j * qv)
    centered_zero = np.exp(-1j * pi[:, None] * qv[None, :])
    mixture = (1.0 - pi[:, None]) + pi[:, None] * exp_iq[None, :]
    shift = np.exp(1j * cell.crosstalk * h[:, [0]] * uk[None, :])
    signal_cf = shift * centered_zero * mixture
    d_centered = centered_zero * (-1j * dpi[:, None] * qv[None, :])
    d_mixture = dpi[:, None] * (exp_iq[None, :] - 1.0)
    d_shift = shift * (1j * cell.crosstalk * uk[None, :])
    d_signal = (
        d_shift * centered_zero * mixture
        + shift * (d_centered * mixture + centered_zero * d_mixture)
    )
    sigma, dsigma = _measurement_noise(h, cell)
    frequency_norm = np.sum(u**2, axis=1)
    attenuation = np.exp(
        -0.5 * sigma[:, None] ** 2 * frequency_norm[None, :]
    )
    d_attenuation = attenuation * (
        -sigma[:, None] * dsigma[:, None] * frequency_norm[None, :]
    )
    observed_cf = signal_cf * attenuation
    d_observed = d_signal * attenuation + signal_cf * d_attenuation
    probability, d_probability = _measurement_observation_probability(h, cell)
    present_phase = np.exp(1j * u_mask)[None, :]
    masked_cf = (
        1.0 - probability[:, None]
        + probability[:, None] * present_phase * observed_cf
    )
    d_masked = (
        d_probability[:, None] * (present_phase * observed_cf - 1.0)
        + probability[:, None] * present_phase * d_observed
    )
    phase = np.exp(1j * bank.raw_phase_offset)[None, :]
    masked_cf *= phase
    d_masked *= phase
    return (
        np.concatenate([masked_cf.real, masked_cf.imag], axis=1),
        np.concatenate([d_masked.real, d_masked.imag], axis=1),
    )


def _estimate_filter_rho(y: np.ndarray) -> float:
    covariance = np.cov(np.asarray(y, dtype=float), rowvar=False)
    ratios: list[float] = []
    for first, second, third in ((0, 1, 2), (0, 2, 3), (1, 2, 3)):
        denominator = covariance[first, second]
        numerator = covariance[first, third]
        if abs(denominator) > 1e-6:
            ratios.append(float(numerator / denominator))
    if not ratios:
        return 0.5
    return float(np.clip(np.median(ratios), 0.01, 0.995))


def _latent_gain_oracle(
    h: np.ndarray, bank: CharacteristicBank, amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    pi = expit(-0.3 + h[:, 0])
    dpi = pi * (1.0 - pi)
    u = bank.raw_frequencies[:, 0]
    qv = amplitude * u
    exp_iq = np.exp(1j * qv)
    centered = np.exp(-1j * pi[:, None] * qv[None, :])
    mixture = 1.0 - pi[:, None] + pi[:, None] * exp_iq[None, :]
    cf = np.exp(1j * bank.raw_phase_offset)[None, :] * centered * mixture
    derivative = np.exp(1j * bank.raw_phase_offset)[None, :] * (
        centered * (-1j * dpi[:, None] * qv[None, :]) * mixture
        + centered * dpi[:, None] * (exp_iq[None, :] - 1.0)
    )
    return (
        np.concatenate([cf.real, cf.imag], axis=1),
        np.concatenate([derivative.real, derivative.imag], axis=1),
    )


def command_e8_measurement(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_cells = [
        MeasurementCell("hetero_active_0p5", heteroskedastic_strength=0.5),
        MeasurementCell("hetero_active_1p0", heteroskedastic_strength=1.0),
        MeasurementCell(
            "hetero_latent_null_observed_active", latent_amplitude=0.0,
            heteroskedastic_strength=1.0
        ),
        MeasurementCell("mcar_active_mask_omitted", missing_rate=0.3),
        MeasurementCell(
            "mcar_active_mask_included", missing_rate=0.3, include_mask=True
        ),
        MeasurementCell(
            "history_missing_active_mask_omitted", missing_rate=0.3,
            missing_history_coefficient=0.8
        ),
        MeasurementCell(
            "history_missing_active_mask_included", missing_rate=0.3,
            missing_history_coefficient=0.8, include_mask=True
        ),
        MeasurementCell(
            "history_missing_latent_null_mask_omitted", latent_amplitude=0.0,
            missing_rate=0.3, missing_history_coefficient=0.8
        ),
        MeasurementCell(
            "history_missing_latent_null_mask_included", latent_amplitude=0.0,
            missing_rate=0.3, missing_history_coefficient=0.8,
            include_mask=True
        ),
    ]
    correction_cells = [
        ("known_deconv_rho0p5", 0.5, "known", 0.5),
        ("known_deconv_rho0p9", 0.9, "known", 0.9),
        ("known_deconv_rho0p98", 0.98, "known", 0.98),
        ("estimated_deconv_rho0p5", 0.5, "estimated", None),
        ("estimated_deconv_rho0p9", 0.9, "estimated", None),
        ("estimated_deconv_rho0p98", 0.98, "estimated", None),
        ("misspecified_deconv_rho0p9_as0p5", 0.9, "misspecified", 0.5),
    ]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    initializations = list(range(args.initializations))
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e8-measurement-extension",
            "raw_cells": [asdict(cell) for cell in raw_cells],
            "correction_cells": correction_cells,
            "seeds": seeds,
            "initializations": initializations,
            "n_train": args.n_train,
            "query_count": 32,
            "claim_rule": "latent and observed predictive-law activity are stored separately",
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(raw_cells):
        for seed in seeds:
            resource_guard()
            rng = np.random.default_rng(820_000 + 1009 * seed + 10_000 * cell_index)
            h_bank = rng.normal(size=(1024, 4))
            y_bank, _, _ = _sample_measurement_paths(h_bank, rng, cell)
            bank = CharacteristicBank.fit(
                y_bank, 32, seed=92510, amplitude_scales=(0.5, 1.0, 2.0, 4.0)
            )
            h = rng.normal(size=(args.n_train, 4))
            y, _, mask = _sample_measurement_paths(h, rng, cell)
            feature = bank.transform(y)
            oracle_mean, oracle_derivative = _measurement_oracle(h, bank, cell)
            truth_h = _standard_normal_qmc(4, 15, 821_000 + seed)
            _, truth_derivative = _measurement_oracle(truth_h, bank, cell)
            truth = np.mean(truth_derivative, axis=0)
            for method_init in initializations:
                print(
                    f"E8-measurement cell={cell.name} seed={seed} init={method_init}",
                    flush=True,
                )
                result = _feature_score(
                    h, feature, oracle_mean, oracle_derivative, truth,
                    seed + 10_000 * cell_index, method_init
                )
                for method, metrics in result.items():
                    latent_active = float(cell.latent_amplitude != 0.0)
                    observed_active = float(np.linalg.norm(truth) > 1e-8)
                    warning = ""
                    if not latent_active and observed_active:
                        warning = "observed_effect_not_latent_effect"
                    if cell.missing_rate and not cell.include_mask:
                        warning = (
                            warning + "+mask_omitted" if warning else "mask_omitted"
                        )
                    rows.append(
                        {
                            "experiment_id": "E8",
                            "cell_id": cell.name,
                            "dgp_family": "measurement_model_extension",
                            "dgp_seed": seed,
                            "method": method,
                            "method_init": method_init,
                            "access_tier": "O0",
                            "n_train": args.n_train,
                            "rho": cell.rho,
                            "heteroskedastic_strength": cell.heteroskedastic_strength,
                            "missing_rate_realized": float(1.0 - np.mean(mask)),
                            "mask_included": float(cell.include_mask),
                            "latent_target_active": latent_active,
                            "observed_target_active": observed_active,
                            "correction_mode": "raw",
                            "rho_estimate": 0.0,
                            "rho_abs_error": 0.0,
                            **metrics,
                            "warning_code": warning,
                            "fit_status": "ok",
                        }
                    )
    for correction_index, (name, true_rho, mode, frozen_rho) in enumerate(correction_cells):
        for seed in seeds:
            resource_guard()
            rng = np.random.default_rng(
                830_000 + 1009 * seed + 10_000 * correction_index
            )
            base = MeasurementCell(
                name=name, rho=true_rho, noise_sd=0.05,
                heteroskedastic_strength=0.0
            )
            h_bank = rng.normal(size=(1024, 4))
            _, latent_bank, _ = _sample_measurement_paths(h_bank, rng, base)
            bank = CharacteristicBank.fit(
                latent_bank[:, None], 32, seed=92511,
                amplitude_scales=(0.5, 1.0, 2.0, 4.0)
            )
            h = rng.normal(size=(args.n_train, 4))
            raw, _, _ = _sample_measurement_paths(h, rng, base)
            estimated_rho = _estimate_filter_rho(raw)
            assumed_rho = estimated_rho if mode == "estimated" else frozen_rho
            assumed_kernel = _observation_kernel(float(assumed_rho), 4)
            corrected = (
                raw @ assumed_kernel / max(float(assumed_kernel @ assumed_kernel), 1e-12)
            )[:, None]
            feature = bank.transform(corrected)
            oracle_mean, oracle_derivative = _latent_gain_oracle(h, bank)
            truth_h = _standard_normal_qmc(4, 15, 831_000 + seed)
            _, truth_derivative = _latent_gain_oracle(truth_h, bank)
            truth = np.mean(truth_derivative, axis=0)
            for method_init in initializations:
                print(
                    f"E8-correction cell={name} seed={seed} init={method_init}",
                    flush=True,
                )
                result = _feature_score(
                    h, feature, oracle_mean, oracle_derivative, truth,
                    seed + 20_000 * correction_index, method_init
                )
                for method, metrics in result.items():
                    rows.append(
                        {
                            "experiment_id": "E8",
                            "cell_id": name,
                            "dgp_family": "measurement_correction",
                            "dgp_seed": seed,
                            "method": method,
                            "method_init": method_init,
                            "access_tier": "O0" if mode != "known" else "M1",
                            "n_train": args.n_train,
                            "rho": true_rho,
                            "heteroskedastic_strength": 0.0,
                            "missing_rate_realized": 0.0,
                            "mask_included": 0.0,
                            "latent_target_active": 1.0,
                            "observed_target_active": 1.0,
                            "correction_mode": mode,
                            "rho_estimate": float(assumed_rho),
                            "rho_abs_error": float(abs(assumed_rho - true_rho)),
                            **metrics,
                            "warning_code": (
                                "oracle_measurement_parameter"
                                if mode == "known" else
                                "measurement_model_misspecified"
                                if mode == "misspecified" else ""
                            ),
                            "fit_status": "ok",
                        }
                    )
    expected = (
        len(raw_cells) + len(correction_cells)
    ) * len(seeds) * len(initializations) * 3
    validation = _package_rows(run_dir, rows, expected)
    validation.update(
        {
            "heteroskedastic_cells_complete": True,
            "missingness_cells_complete": True,
            "mask_ablation_complete": True,
            "known_deconvolution_complete": True,
            "estimated_observation_model_complete": True,
            "misspecification_cell_complete": True,
            "full_observation_category_coverage": True,
            "matched_prior_observation_null_source": "e8_confirmation_20260714",
        }
    )
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


@dataclass(frozen=True)
class GeometryCell:
    name: str
    history_dim: int
    response_dim: int
    rank: int
    n_train: int
    rotation: bool
    source_sparsity: str
    query_count: int = 64
    seeds: int = 10


def _dense_parameters_general(seed: int, cell: GeometryCell):
    rng = np.random.default_rng(840_000 + seed)
    w = rng.normal(
        scale=0.55 / math.sqrt(max(cell.history_dim / 8.0, 1.0)),
        size=(cell.rank, cell.history_dim),
    )
    if cell.source_sparsity == "sparse":
        mask = np.zeros_like(w)
        for component in range(cell.rank):
            active = np.arange(component, cell.history_dim - 2, max(cell.rank, 1))[:3]
            mask[component, active] = 1.0
        w *= mask
    elif cell.source_sparsity != "dense":
        raise ValueError(cell.source_sparsity)
    w[:, -2:] = 0.0
    if cell.rotation:
        b = rng.normal(size=(cell.rank, cell.response_dim))
        rotation, _ = np.linalg.qr(
            rng.normal(size=(cell.response_dim, cell.response_dim))
        )
        b = b @ rotation
    else:
        b = np.zeros((cell.rank, cell.response_dim))
        for component in range(cell.rank):
            b[component, component % cell.response_dim] = 1.0
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    b *= 1.5
    intercept = np.linspace(-0.8, 0.8, cell.rank)
    return w, b, intercept


def _source_subspace_angle(
    estimate: np.ndarray, truth: np.ndarray, rank: int
) -> float:
    left_estimate = np.linalg.svd(estimate.reshape(estimate.shape[0], -1), full_matrices=False)[0]
    left_truth = np.linalg.svd(truth.reshape(truth.shape[0], -1), full_matrices=False)[0]
    usable_rank = min(rank, left_estimate.shape[1], left_truth.shape[1])
    return float(
        np.max(np.degrees(subspace_angles(
            left_estimate[:, :usable_rank], left_truth[:, :usable_rank]
        )))
    )


def _selected_low_rank(matrices: np.ndarray) -> tuple[np.ndarray, int]:
    aggregate = np.sum([matrix @ matrix.T for matrix in matrices], axis=0)
    values, vectors = np.linalg.eigh(aggregate)
    maximum = max(float(np.max(values)), 1e-12)
    selected = max(1, int(np.sum(values > 0.05 * maximum)))
    space = vectors[:, np.argsort(values)[-selected:]]
    projector = space @ space.T
    return np.stack([projector @ matrix @ projector for matrix in matrices]), selected


def _fit_geometry_seed(
    cell: GeometryCell, seed: int, method_init: int
) -> list[dict[str, object]]:
    w, b, intercept = _dense_parameters_general(seed, cell)
    rng = np.random.default_rng(841_000 + 1009 * seed)
    h_bank = rng.normal(size=(1024, cell.history_dim))
    y_bank = _sample_dense(h_bank, rng, w, b, intercept)
    bank = CharacteristicBank.fit(
        y_bank, cell.query_count, seed=92520,
        amplitude_scales=(0.1, 0.15, 0.2, 0.3)
    )
    h = rng.normal(size=(cell.n_train, cell.history_dim))
    y = _sample_dense(h, rng, w, b, intercept)
    feature = bank.transform(y)
    pair_feature, pairs = _symmetric_vector(y)
    truth_h = _standard_normal_qmc(cell.history_dim, 14, 842_000 + seed)
    _, truth_derivative_complex = _dense_oracle_complex(
        truth_h, bank, w, b, intercept
    )
    truth_feature = np.mean(
        np.concatenate(
            [truth_derivative_complex.real, truth_derivative_complex.imag], axis=2
        ),
        axis=0,
    )
    pi_truth = expit(intercept[None, :] + truth_h @ w.T)
    truth_covariance = np.zeros(
        (cell.history_dim, cell.response_dim, cell.response_dim)
    )
    for direction in range(cell.history_dim):
        coefficient = np.mean(
            pi_truth * (1.0 - pi_truth) * (1.0 - 2.0 * pi_truth)
            * w[:, direction][None, :],
            axis=0,
        )
        truth_covariance[direction] = sum(
            coefficient[k] * np.outer(b[k], b[k]) for k in range(cell.rank)
        )
    truth_cov_vector = np.asarray(
        [
            [truth_covariance[p, i, j] for i, j in pairs]
            for p in range(cell.history_dim)
        ]
    )
    feature_sum = np.zeros_like(truth_feature)
    feature_sumsq = np.zeros_like(truth_feature)
    covariance_sum = np.zeros_like(truth_cov_vector)
    covariance_sumsq = np.zeros_like(truth_cov_vector)
    for fold_id, (train, test) in enumerate(
        independent_folds(cell.n_train, 5, 843_000 + seed)
    ):
        sieve = HistorySieve(
            cell.history_dim, 64, 844_000 + 100 * method_init + fold_id
        ).fit(h[train])
        z_train = sieve.transform(h[train])
        z_test = sieve.transform(h[test])
        feature_coefficient = ridge_coefficients(z_train, feature[train], 1e-3)
        covariance_coefficient = ridge_coefficients(
            z_train, pair_feature[train], 1e-3
        )
        feature_prediction = z_test @ feature_coefficient
        covariance_prediction = z_test @ covariance_coefficient
        for direction in range(cell.history_dim):
            dz_train = sieve.derivative(h[train], direction)
            dz_test = sieve.derivative(h[test], direction)
            alpha_coefficient = riesz_coefficients(z_train, dz_train, 1e-2)
            alpha = z_test @ alpha_coefficient
            feature_score = dz_test @ feature_coefficient + alpha[:, None] * (
                feature[test] - feature_prediction
            )
            covariance_score = dz_test @ covariance_coefficient + alpha[:, None] * (
                pair_feature[test] - covariance_prediction
            )
            feature_sum[direction] += np.sum(feature_score, axis=0)
            feature_sumsq[direction] += np.sum(feature_score**2, axis=0)
            covariance_sum[direction] += np.sum(covariance_score, axis=0)
            covariance_sumsq[direction] += np.sum(covariance_score**2, axis=0)
    estimate_feature = feature_sum / cell.n_train
    estimate_cov_vector = covariance_sum / cell.n_train
    se_feature = np.sqrt(
        np.maximum(
            (feature_sumsq - cell.n_train * estimate_feature**2)
            / max(cell.n_train - 1, 1),
            0.0,
        ) / cell.n_train
    )
    se_cov_vector = np.sqrt(
        np.maximum(
            (covariance_sumsq - cell.n_train * estimate_cov_vector**2)
            / max(cell.n_train - 1, 1),
            0.0,
        ) / cell.n_train
    )
    direct_matrices = np.stack(
        [
            _matrix_from_symmetric(
                estimate_cov_vector[p], pairs, cell.response_dim
            )
            for p in range(cell.history_dim)
        ]
    )
    u = bank.raw_frequencies
    design_pairs = [
        (i, j) for i in range(cell.response_dim) for j in range(i, cell.response_dim)
    ]
    design = np.column_stack(
        [
            u[:, i] ** 2 if i == j else 2.0 * u[:, i] * u[:, j]
            for i, j in design_pairs
        ]
    )
    characteristic_matrices = []
    for direction in range(cell.history_dim):
        shifted = (
            estimate_feature[direction, : bank.q]
            + 1j * estimate_feature[direction, bank.q :]
        )
        raw = shifted * np.exp(-1j * bank.raw_phase_offset)
        coefficient = ridge_coefficients(design, -2.0 * raw.real, 1e-4)
        characteristic_matrices.append(
            _matrix_from_symmetric(coefficient, design_pairs, cell.response_dim)
        )
    characteristic_matrices = np.stack(characteristic_matrices)
    low_rank_direct, selected_rank = _selected_low_rank(direct_matrices)
    rows: list[dict[str, object]] = []
    for method, matrices in (
        ("ORTH_CHARACTERISTIC_INVERSION", characteristic_matrices),
        ("DIRECT_COV_ORTH", direct_matrices),
        ("DIRECT_COV_ORTH_LOW_RANK_SELECTED", low_rank_direct),
    ):
        angle, rank_error = _subspace_metrics(
            matrices, truth_covariance, cell.rank
        )
        covariance_nrmse = vector_metrics(
            matrices.reshape(-1), truth_covariance.reshape(-1)
        )["nrmse"]
        source_angle = _source_subspace_angle(
            matrices, truth_covariance, cell.rank
        )
        if method == "ORTH_CHARACTERISTIC_INVERSION":
            feature_metrics = vector_metrics(
                estimate_feature.reshape(-1), truth_feature.reshape(-1)
            )
            null_z = np.abs(
                estimate_feature[-2:] / np.maximum(se_feature[-2:], 1e-12)
            )
            coverage = float(
                np.mean(
                    (truth_feature >= estimate_feature - 1.96 * se_feature)
                    & (truth_feature <= estimate_feature + 1.96 * se_feature)
                )
            )
        else:
            feature_metrics = {
                "nrmse": covariance_nrmse,
                "calibration_slope": float(
                    np.dot(matrices.reshape(-1), truth_covariance.reshape(-1))
                    / max(np.dot(truth_covariance.reshape(-1), truth_covariance.reshape(-1)), 1e-12)
                ),
            }
            null_z = np.abs(
                estimate_cov_vector[-2:] / np.maximum(se_cov_vector[-2:], 1e-12)
            )
            coverage = float(
                np.mean(
                    (truth_cov_vector >= estimate_cov_vector - 1.96 * se_cov_vector)
                    & (truth_cov_vector <= estimate_cov_vector + 1.96 * se_cov_vector)
                )
            )
        critical = norm.ppf(1.0 - 0.05 / (2.0 * null_z.size))
        rows.append(
            {
                "experiment_id": "E9",
                "cell_id": cell.name,
                "dgp_family": "centered_multiple_gain_staged",
                "dgp_seed": seed,
                "method": method,
                "method_init": method_init,
                "access_tier": "O0",
                "n_train": cell.n_train,
                "history_dim": cell.history_dim,
                "response_dim": cell.response_dim,
                "rank": cell.rank,
                "rotation": float(cell.rotation),
                "source_sparsity_dense": float(cell.source_sparsity == "dense"),
                "query_count": cell.query_count,
                "feature_tensor_nrmse": feature_metrics["nrmse"],
                "feature_calibration_slope": feature_metrics["calibration_slope"],
                "feature_coverage_fraction": coverage,
                "covariance_tensor_nrmse": covariance_nrmse,
                "largest_subspace_angle_degrees": angle,
                "source_subspace_angle_degrees": source_angle,
                "rank_absolute_error": rank_error,
                "selected_rank": float(selected_rank),
                "null_edge_familywise_call": float(np.any(null_z > critical)),
                "warning_code": (
                    "posthoc_low_rank_selected"
                    if method.endswith("LOW_RANK_SELECTED") else ""
                ),
                "fit_status": "ok",
            }
        )
    return rows


def command_e9_staged(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        GeometryCell("axis_aligned_K1", 8, 8, 1, 8000, False, "dense"),
        GeometryCell("axis_aligned_K3", 8, 8, 3, 8000, False, "dense"),
        GeometryCell("rotated_K1", 8, 8, 1, 8000, True, "dense"),
        GeometryCell("rotated_K5", 8, 8, 5, 8000, True, "dense"),
        GeometryCell("rotated_sparse_K3", 8, 8, 3, 8000, True, "sparse"),
        GeometryCell("history16_rotated_K3", 16, 8, 3, 8000, True, "dense"),
        GeometryCell("response32_rotated_K3", 8, 32, 3, 8000, True, "dense", 96),
        GeometryCell("n32000_rotated_K3", 8, 8, 3, 32000, True, "dense"),
    ]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    initializations = list(range(args.initializations))
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e9-staged-extension",
            "cells": [asdict(cell) for cell in cells],
            "seeds": seeds,
            "initializations": initializations,
            "streaming_score_moments": True,
            "existing_preregistered_medium_cell": "e9_confirmation_20260714",
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for cell in cells:
        for seed in seeds:
            for method_init in initializations:
                resource_guard()
                print(
                    f"E9-staged cell={cell.name} seed={seed} init={method_init}",
                    flush=True,
                )
                rows.extend(_fit_geometry_seed(cell, seed, method_init))
    expected = len(cells) * len(seeds) * len(initializations) * 3
    validation = _package_rows(run_dir, rows, expected)
    validation.update(
        {
            "staged_geometry_grid_complete": True,
            "rotation_ablation_complete": True,
            "rank_scaling_complete": True,
            "history_scaling_complete": True,
            "response_scaling_complete": True,
            "sample_scaling_complete": True,
            "source_sparsity_ablation_complete": True,
            "full_cartesian_grid_intentionally_not_required": True,
        }
    )
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


@dataclass
class BenchmarkDataset:
    name: str
    h: np.ndarray
    y: np.ndarray
    bank: CharacteristicBank
    truth: np.ndarray
    oracle_mean: np.ndarray
    oracle_derivative: np.ndarray


def _e10_dataset(name: str, seed: int, n: int) -> BenchmarkDataset:
    rng = np.random.default_rng(850_000 + 1009 * seed)
    if name == "E1_gaussian":
        dgp = AnalyticGaussianPathDGP(seed, history_dim=8, response_dim=4)
        h = dgp.sample_histories(rng, n)
        y = dgp.sample_responses(h, rng)
        bank = CharacteristicBank.fit(y[: max(1024, n // 2)], 16, seed=92530)
        oracle_mean = dgp.conditional_feature_mean(h, bank)
        oracle_derivative = dgp.conditional_feature_derivative(h, 0, bank)
        truth_h = dgp.qmc_histories(15, 851_000 + seed)
        truth = np.mean(dgp.conditional_feature_derivative(truth_h, 0, bank), axis=0)
    elif name in {"E4_discrete", "E4_continuous"}:
        dgp = (
            MomentBlindDiscreteDGP(seed, amplitude_fraction=0.5)
            if name == "E4_discrete" else MomentBlindLegendreDGP(seed, amplitude=0.5)
        )
        h = dgp.sample_histories(rng, n)
        y = dgp.sample_responses(h, rng)
        bank = CharacteristicBank.fit(
            y[: max(1024, n // 2)], 16, seed=92531,
            amplitude_scales=(1.0, 2.0, 4.0, 8.0)
        )
        oracle_mean = dgp.conditional_feature_mean(h, bank)
        oracle_derivative = dgp.conditional_feature_derivative(h, 0, bank)
        truth_h = _standard_normal_qmc(1, 15, 852_000 + seed)
        truth = np.mean(dgp.conditional_feature_derivative(truth_h, 0, bank), axis=0)
    elif name == "E5_support_motion":
        dgp = SupportMotionDGP(seed, noise_sigma=0.1)
        h = dgp.sample_histories(rng, n)
        y = dgp.sample_responses(h, rng)
        bank = CharacteristicBank.fit(y[: max(1024, n // 2)], 16, seed=92532)
        oracle_mean = dgp.conditional_feature_mean(h, bank)
        oracle_derivative = dgp.conditional_feature_derivative(h, 0, bank)
        truth_h = _standard_normal_qmc(1, 15, 853_000 + seed)
        truth = np.mean(dgp.conditional_feature_derivative(truth_h, 0, bank), axis=0)
    elif name == "E8_filtered":
        cell = ObservationCell("e10_filtered", 0.5, 1.0, 0.0, 30)
        h = rng.normal(size=(n, 4))
        y = _sample_observation_paths(h, rng, cell)
        bank = CharacteristicBank.fit(y[: max(1024, n // 2)], 16, seed=92533)
        oracle_mean, oracle_derivative = _observation_oracle(h, bank, cell)
        truth_h = _standard_normal_qmc(4, 15, 854_000 + seed)
        _, truth_derivative = _observation_oracle(truth_h, bank, cell)
        truth = np.mean(truth_derivative, axis=0)
    elif name == "E9_dense":
        cell = GeometryCell("e10_dense", 8, 8, 3, n, True, "dense", 16)
        w, b, intercept = _dense_parameters_general(seed, cell)
        h = rng.normal(size=(n, 8))
        y = _sample_dense(h, rng, w, b, intercept)
        bank = CharacteristicBank.fit(
            y[: max(1024, n // 2)], 16, seed=92534,
            amplitude_scales=(0.1, 0.15, 0.2, 0.3)
        )
        complex_mean, complex_derivative = _dense_oracle_complex(
            h, bank, w, b, intercept
        )
        oracle_mean = np.concatenate([complex_mean.real, complex_mean.imag], axis=1)
        oracle_derivative = np.concatenate(
            [complex_derivative[:, 0].real, complex_derivative[:, 0].imag], axis=1
        )
        truth_h = _standard_normal_qmc(8, 15, 855_000 + seed)
        _, truth_derivative = _dense_oracle_complex(
            truth_h, bank, w, b, intercept
        )
        truth = np.mean(
            np.concatenate(
                [truth_derivative[:, 0].real, truth_derivative[:, 0].imag], axis=1
            ),
            axis=0,
        )
    elif name == "E11_ordering":
        forward, reverse = _path_templates()
        h = rng.normal(size=(n, 4))
        y = _sample_paths(h, rng, forward, reverse, 0.15)
        bank = CharacteristicBank.fit(
            y[: max(1024, n // 2)], 16, seed=92535,
            amplitude_scales=(1.0, 2.0, 4.0, 8.0)
        )
        oracle_mean, oracle_derivative = _path_feature_oracle(
            h, bank, forward, reverse, 0.15
        )
        truth_h = _standard_normal_qmc(4, 15, 856_000 + seed)
        _, truth_derivative = _path_feature_oracle(
            truth_h, bank, forward, reverse, 0.15
        )
        truth = np.mean(truth_derivative, axis=0)
    else:
        raise ValueError(name)
    return BenchmarkDataset(
        name, h, y, bank, truth, oracle_mean, oracle_derivative
    )


def _sample_features(samples: np.ndarray, bank: CharacteristicBank) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    flat = samples.reshape(-1, samples.shape[-1])
    return bank.transform(flat).reshape(samples.shape[0], samples.shape[1], -1).mean(axis=1)


def _energy_score(samples: np.ndarray, observed: np.ndarray) -> float:
    first = np.mean(np.linalg.norm(samples - observed[:, None, :], axis=-1))
    half = samples.shape[1] // 2
    second = 0.5 * np.mean(
        np.linalg.norm(samples[:, :half] - samples[:, half : 2 * half], axis=-1)
    )
    return float(first - second)


def _sampling_metrics(
    sampler: Callable[[np.ndarray, int, int], np.ndarray],
    h_validation: np.ndarray,
    y_validation: np.ndarray,
    evaluation_h: np.ndarray,
    bank: CharacteristicBank,
    truth: np.ndarray,
    budget: int,
    seed: int,
    delta: float = 0.02,
) -> dict[str, float]:
    validation_limit = min(32, len(h_validation))
    samples = sampler(h_validation[:validation_limit], budget, seed)
    prediction = _sample_features(samples, bank)
    observed_feature = bank.transform(y_validation[:validation_limit])
    energy = _energy_score(samples, y_validation[:validation_limit])
    del samples
    gc.collect()
    plus = evaluation_h.copy()
    minus = evaluation_h.copy()
    plus[:, 0] += delta
    minus[:, 0] -= delta
    plus_feature = _sample_features(sampler(plus, budget, seed + 1), bank)
    minus_feature = _sample_features(sampler(minus, budget, seed + 1), bank)
    tangent = np.mean((plus_feature - minus_feature) / (2.0 * delta), axis=0)
    tangent_metrics = vector_metrics(tangent, truth)
    if not np.isfinite(tangent_metrics["calibration_slope"]):
        tangent_metrics["calibration_slope"] = 0.0
    if not np.isfinite(tangent_metrics["sign_accuracy"]):
        tangent_metrics["sign_accuracy"] = 0.0
    complex_prediction = prediction[:, : bank.q] + 1j * prediction[:, bank.q :]
    return {
        **tangent_metrics,
        "feature_prediction_mse": float(
            np.mean((observed_feature - prediction) ** 2)
        ),
        "energy_score": energy,
        "max_characteristic_modulus": float(np.max(np.abs(complex_prediction))),
        "characteristic_modulus_valid": float(
            np.max(np.abs(complex_prediction)) <= 1.0 + 1e-6
        ),
    }


def _fit_nsf(
    train_h: np.ndarray,
    train_y: np.ndarray,
    validation_h: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
    transforms: int,
    max_epochs: int,
):
    import torch
    from zuko.flows import NSF

    torch.manual_seed(seed)
    flow = NSF(
        features=train_y.shape[1], context=train_h.shape[1], bins=8,
        transforms=transforms, hidden_features=(32, 32)
    )
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-4)
    h_train = torch.as_tensor(train_h, dtype=torch.float32)
    y_train = torch.as_tensor(train_y, dtype=torch.float32)
    h_validation = torch.as_tensor(validation_h, dtype=torch.float32)
    y_validation = torch.as_tensor(validation_y, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(flow.state_dict())
    patience = 4
    wait = 0
    started = time.perf_counter()
    for epoch in range(max_epochs):
        flow.train()
        permutation = rng.permutation(len(h_train))
        for start in range(0, len(permutation), 256):
            index = torch.as_tensor(permutation[start : start + 256], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = -flow(h_train[index]).log_prob(y_train[index]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite NSF likelihood")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
        flow.eval()
        with torch.no_grad():
            validation_loss = float(-flow(h_validation).log_prob(y_validation).mean())
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(flow.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    flow.load_state_dict(best_state)
    parameter_count = sum(parameter.numel() for parameter in flow.parameters())
    return flow, {
        "validation_loss": best_loss,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "wall_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count,
    }


def _nsf_sampler(flow):
    def sample(h: np.ndarray, count: int, seed: int) -> np.ndarray:
        import torch

        context = torch.as_tensor(h, dtype=torch.float32)
        flow.eval()
        pieces = []
        for start in range(0, count, 32):
            chunk = min(32, count - start)
            torch.manual_seed(seed + start)
            with torch.no_grad():
                values = flow(context).sample((chunk,)).permute(1, 0, 2)
            pieces.append(values.cpu().numpy())
            del values
        return np.concatenate(pieces, axis=1)

    return sample


def _mdn_sampler(model):
    def sample(h: np.ndarray, count: int, seed: int) -> np.ndarray:
        import torch

        model.eval()
        context = torch.as_tensor(h, dtype=torch.float32)
        pieces = []
        for start in range(0, count, 32):
            chunk = min(32, count - start)
            with torch.no_grad():
                values = model.sample(context, chunk, seed=seed + start)
            pieces.append(values.cpu().numpy())
            del values
        return np.concatenate(pieces, axis=1)

    return sample


def _fit_full_covariance_mdn(
    train_h: np.ndarray,
    train_y: np.ndarray,
    validation_h: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
    components: int,
    max_epochs: int,
):
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class FullCovarianceMDN(nn.Module):
        def __init__(self, history_dim: int, response_dim: int, mixture_count: int):
            super().__init__()
            self.response_dim = response_dim
            self.mixture_count = mixture_count
            self.triangle_size = response_dim * (response_dim + 1) // 2
            output_dim = mixture_count * (1 + response_dim + self.triangle_size)
            self.network = nn.Sequential(
                nn.Linear(history_dim, 64), nn.SiLU(),
                nn.Linear(64, 64), nn.SiLU(),
                nn.Linear(64, output_dim),
            )
            rows, columns = torch.tril_indices(response_dim, response_dim)
            self.register_buffer("triangle_rows", rows)
            self.register_buffer("triangle_columns", columns)

        def parameters_at(self, history: torch.Tensor):
            batch = len(history)
            raw = self.network(history).reshape(
                batch, self.mixture_count,
                1 + self.response_dim + self.triangle_size,
            )
            logits = raw[..., 0]
            mean = raw[..., 1 : 1 + self.response_dim]
            triangle = raw[..., 1 + self.response_dim :]
            cholesky = torch.zeros(
                batch, self.mixture_count, self.response_dim, self.response_dim,
                dtype=history.dtype, device=history.device,
            )
            cholesky[..., self.triangle_rows, self.triangle_columns] = triangle
            diagonal = torch.arange(self.response_dim, device=history.device)
            raw_diagonal = cholesky[..., diagonal, diagonal]
            cholesky[..., diagonal, diagonal] = functional.softplus(raw_diagonal) + 0.03
            return logits, mean, cholesky

        def log_prob(self, response: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            logits, mean, cholesky = self.parameters_at(history)
            difference = response[:, None, :] - mean
            standardized = torch.linalg.solve_triangular(
                cholesky, difference[..., None], upper=False
            ).squeeze(-1)
            log_determinant = torch.log(
                torch.diagonal(cholesky, dim1=-2, dim2=-1)
            ).sum(dim=-1)
            component = -0.5 * (
                standardized.square().sum(dim=-1)
                + self.response_dim * math.log(2.0 * math.pi)
            ) - log_determinant
            return torch.logsumexp(
                torch.log_softmax(logits, dim=-1) + component, dim=-1
            )

        def characteristic(
            self,
            history: torch.Tensor,
            frequency: torch.Tensor,
            phase_offset: torch.Tensor,
        ):
            logits, mean, cholesky = self.parameters_at(history)
            phase = (
                torch.einsum("bkd,qd->bkq", mean, frequency)
                + phase_offset[None, None, :]
            )
            transposed_product = torch.einsum(
                "bkij,qi->bkjq", cholesky, frequency
            )
            quadratic = transposed_product.square().sum(dim=2)
            mixture_weight = torch.softmax(logits, dim=-1)[..., None]
            value = torch.sum(
                mixture_weight
                * torch.exp(torch.complex(-0.5 * quadratic, phase)),
                dim=1,
            )
            return torch.cat([value.real, value.imag], dim=-1)

    torch.manual_seed(seed)
    model = FullCovarianceMDN(
        train_h.shape[1], train_y.shape[1], components
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_history = torch.as_tensor(train_h, dtype=torch.float32)
    train_response = torch.as_tensor(train_y, dtype=torch.float32)
    validation_history = torch.as_tensor(validation_h, dtype=torch.float32)
    validation_response = torch.as_tensor(validation_y, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        permutation = rng.permutation(len(train_history))
        for start in range(0, len(permutation), 256):
            index = torch.as_tensor(permutation[start : start + 256], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = -model.log_prob(
                train_response[index], train_history[index]
            ).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite analytic MDN likelihood")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                -model.log_prob(validation_response, validation_history).mean()
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= 4:
                break
    model.load_state_dict(best_state)
    return model, {
        "validation_loss": best_loss,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "wall_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _analytic_mdn_metrics(
    model,
    h_validation: np.ndarray,
    y_validation: np.ndarray,
    evaluation_h: np.ndarray,
    bank: CharacteristicBank,
    truth: np.ndarray,
) -> dict[str, float]:
    import torch

    frequency = torch.as_tensor(bank.raw_frequencies, dtype=torch.float32)
    phase_offset = torch.as_tensor(bank.raw_phase_offset, dtype=torch.float32)
    validation_limit = min(32, len(h_validation))
    validation_history = torch.as_tensor(
        h_validation[:validation_limit], dtype=torch.float32
    )
    validation_response = torch.as_tensor(
        y_validation[:validation_limit], dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        prediction = model.characteristic(
            validation_history, frequency, phase_offset
        )
        heldout_log_likelihood = float(
            model.log_prob(
                torch.as_tensor(y_validation, dtype=torch.float32),
                torch.as_tensor(h_validation, dtype=torch.float32),
            ).mean()
        )
    observed_feature = bank.transform(validation_response.numpy())
    history = torch.as_tensor(evaluation_h, dtype=torch.float32).requires_grad_(True)
    feature_mean = model.characteristic(history, frequency, phase_offset)
    derivative_coordinates = []
    for coordinate in range(feature_mean.shape[1]):
        gradient = torch.autograd.grad(
            feature_mean[:, coordinate].sum(), history,
            retain_graph=coordinate + 1 < feature_mean.shape[1],
        )[0]
        derivative_coordinates.append(float(gradient[:, 0].mean()))
    tangent = np.asarray(derivative_coordinates)
    tangent_metrics = vector_metrics(tangent, truth)
    if not np.isfinite(tangent_metrics["calibration_slope"]):
        tangent_metrics["calibration_slope"] = 0.0
    if not np.isfinite(tangent_metrics["sign_accuracy"]):
        tangent_metrics["sign_accuracy"] = 0.0
    complex_prediction = prediction[:, : bank.q] + 1j * prediction[:, bank.q :]
    return {
        **tangent_metrics,
        "heldout_log_likelihood": heldout_log_likelihood,
        "feature_prediction_mse": float(
            np.mean((observed_feature - prediction.detach().numpy()) ** 2)
        ),
        "max_characteristic_modulus": float(
            np.max(np.abs(complex_prediction.detach().numpy()))
        ),
        "characteristic_modulus_valid": float(
            np.max(np.abs(complex_prediction.detach().numpy())) <= 1.0 + 1e-6
        ),
    }


def command_e10_full(args: argparse.Namespace) -> int:
    _force_thread_limits()
    workspace = Path(__file__).resolve().parents[3]
    benchmark_source = workspace / "history_tangent_benchmark" / "src"
    if str(benchmark_source) not in sys.path:
        sys.path.insert(0, str(benchmark_source))
    from history_tangent_benchmark.models import build_model, fit_model
    import torch

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    families = [
        "E1_gaussian", "E4_discrete", "E4_continuous",
        "E5_support_motion", "E8_filtered", "E9_dense", "E11_ordering"
    ]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    initializations = list(range(args.initializations))
    sampling_budgets = [64, 256, 1024]
    neural_methods = [
        ("MDN_K5", "mdn", 5),
        ("MDN_K10", "mdn", 10),
        ("MDN_K20", "mdn", 20),
        ("NSF_6", "nsf", 6),
        ("NSF_10", "nsf", 10),
    ]
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e10-full-extension",
            "families": families,
            "seeds": seeds,
            "initializations": initializations,
            "n_train_total": args.n_train,
            "max_epochs": args.max_epochs,
            "sampling_budgets": sampling_budgets,
            "evaluation_histories": 32,
            "methods": ["PLUG", "RIESZ", "ORTH", "ZERO"]
            + [method[0] for method in neural_methods],
            "normalized_likelihood_methods": [
                "MDN_K5", "MDN_K10", "MDN_K20", "NSF_6", "NSF_10"
            ],
            "flow_backend": "zuko.flows.NSF 1.6.0",
            "access_tier": "O0",
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for family_index, family in enumerate(families):
        for seed in seeds:
            resource_guard()
            dataset = _e10_dataset(family, seed, args.n_train)
            split = int(0.8 * args.n_train)
            h_train, h_validation = dataset.h[:split], dataset.h[split:]
            y_train, y_validation = dataset.y[:split], dataset.y[split:]
            evaluation_h = _standard_normal_qmc(
                dataset.h.shape[1], 5, 860_000 + seed
            )
            feature = dataset.bank.transform(dataset.y)
            for method_init in initializations:
                print(
                    f"E10-full family={family} method=ORTH seed={seed} init={method_init}",
                    flush=True,
                )
                started = time.perf_counter()
                feature_estimators = _feature_score(
                    dataset.h, feature, dataset.oracle_mean,
                    dataset.oracle_derivative, dataset.truth,
                    seed + 10_000 * family_index, method_init
                )
                feature_runtime = time.perf_counter() - started
                for feature_method, feature_metrics in feature_estimators.items():
                    rows.append(
                        {
                            "experiment_id": "E10",
                            "cell_id": family,
                            "dgp_family": family,
                            "dgp_seed": seed,
                            "method": feature_method,
                            "method_init": method_init,
                            "access_tier": "O0",
                            "compute_budget": "matched_local",
                            "n_train": args.n_train,
                            "history_dim": dataset.h.shape[1],
                            "response_dim": dataset.y.shape[1],
                            "query_count": dataset.bank.q,
                            "sampling_budget": 0,
                            "parameter_count": float(
                                1 + 2 * dataset.h.shape[1] + 32
                            ),
                            "runtime_seconds": feature_runtime,
                            "validation_native_loss": float(
                                np.mean((feature - dataset.oracle_mean) ** 2)
                            ),
                            "heldout_log_likelihood": 0.0,
                            "log_likelihood_defined": 0.0,
                            "feature_prediction_mse": float(
                                np.mean((feature - dataset.oracle_mean) ** 2)
                            ),
                            "energy_score": 0.0,
                            "energy_score_defined": 0.0,
                            "max_characteristic_modulus": 0.0,
                            "characteristic_modulus_valid": 1.0,
                            **feature_metrics,
                            "exact_normalized_likelihood": 0.0,
                            "warning_code": "target_specific_feature_estimator",
                            "fit_status": "ok",
                        }
                    )
                rows.append(
                    {
                        "experiment_id": "E10",
                        "cell_id": family,
                        "dgp_family": family,
                        "dgp_seed": seed,
                        "method": "ZERO",
                        "method_init": method_init,
                        "access_tier": "O0",
                        "compute_budget": "matched_local",
                        "n_train": args.n_train,
                        "history_dim": dataset.h.shape[1],
                        "response_dim": dataset.y.shape[1],
                        "query_count": dataset.bank.q,
                        "sampling_budget": 0,
                        "parameter_count": 0.0,
                        "runtime_seconds": 0.0,
                        "validation_native_loss": float(
                            np.mean(feature**2)
                        ),
                        "heldout_log_likelihood": 0.0,
                        "log_likelihood_defined": 0.0,
                        "feature_prediction_mse": float(np.mean(feature**2)),
                        "energy_score": 0.0,
                        "energy_score_defined": 0.0,
                        "max_characteristic_modulus": 0.0,
                        "characteristic_modulus_valid": 1.0,
                        "nrmse": 1.0,
                        "calibration_slope": 0.0,
                        "sign_accuracy": 0.0,
                        "coverage_fraction": 0.0,
                        "max_abs_z": 0.0,
                        "detected_bonferroni": 0.0,
                        "calibration_slope_defined": 1.0,
                        "sign_accuracy_defined": 1.0,
                        "exact_normalized_likelihood": 0.0,
                        "warning_code": "normalization_baseline",
                        "fit_status": "ok",
                    }
                )
                for method_label, kind, complexity in neural_methods:
                    resource_guard()
                    training_seed = (
                        861_000 + 100 * seed + method_init + 10_000 * family_index
                        + complexity
                    )
                    print(
                        f"E10-full family={family} method={method_label} "
                        f"seed={seed} init={method_init}",
                        flush=True,
                    )
                    if kind == "mdn":
                        model = build_model(
                            "autoregressive_mdn", q=dataset.h.shape[1],
                            dy=dataset.y.shape[1],
                            params={"hidden": 32, "layers": 2, "components": complexity},
                        )
                        trace = fit_model(
                            model, h_train, y_train, h_validation, y_validation,
                            seed=training_seed, device="cpu", learning_rate=1e-3,
                            weight_decay=1e-4, batch_size=256,
                            max_epochs=args.max_epochs, patience=4
                        )
                        sampler = _mdn_sampler(model)
                        with torch.no_grad():
                            heldout_ll = float(
                                model.log_prob(
                                    torch.as_tensor(y_validation, dtype=torch.float32),
                                    torch.as_tensor(h_validation, dtype=torch.float32),
                                ).mean()
                            )
                        trace_payload = trace.to_dict()
                    else:
                        model, trace_payload = _fit_nsf(
                            h_train, y_train, h_validation, y_validation,
                            training_seed, complexity, args.max_epochs
                        )
                        sampler = _nsf_sampler(model)
                        with torch.no_grad():
                            heldout_ll = float(
                                model(torch.as_tensor(h_validation, dtype=torch.float32))
                                .log_prob(torch.as_tensor(y_validation, dtype=torch.float32))
                                .mean()
                            )
                    traces.append(
                        {
                            "family": family,
                            "dgp_seed": seed,
                            "method_init": method_init,
                            "method": method_label,
                            **trace_payload,
                        }
                    )
                    native_validation = trace_payload["validation_loss"]
                    if isinstance(native_validation, list):
                        best_epoch = int(trace_payload["best_epoch"])
                        native_validation = native_validation[best_epoch]
                    for budget in sampling_budgets:
                        metrics = _sampling_metrics(
                            sampler, h_validation, y_validation, evaluation_h,
                            dataset.bank, dataset.truth, budget,
                            862_000 + seed + 1000 * budget
                        )
                        rows.append(
                            {
                                "experiment_id": "E10",
                                "cell_id": family,
                                "dgp_family": family,
                                "dgp_seed": seed,
                                "method": method_label,
                                "method_init": method_init,
                                "access_tier": "O0",
                                "compute_budget": "matched_local",
                                "n_train": args.n_train,
                                "history_dim": dataset.h.shape[1],
                                "response_dim": dataset.y.shape[1],
                                "query_count": dataset.bank.q,
                                "sampling_budget": budget,
                                "parameter_count": float(trace_payload["parameter_count"]),
                                "runtime_seconds": float(trace_payload["wall_seconds"]),
                                "validation_native_loss": float(native_validation),
                                "heldout_log_likelihood": heldout_ll,
                                "log_likelihood_defined": 1.0,
                                "energy_score_defined": 1.0,
                                **metrics,
                                "coverage_fraction": 0.0,
                                "max_abs_z": 0.0,
                                "detected_bonferroni": 0.0,
                                "calibration_slope_defined": 1.0,
                                "sign_accuracy_defined": 1.0,
                                "exact_normalized_likelihood": 1.0,
                                "warning_code": "sampling_based_tangent",
                                "fit_status": "ok",
                            }
                        )
                    del sampler, model
                    gc.collect()
    atomic_csv(run_dir / "training_traces.csv", pd.DataFrame(traces))
    expected = len(families) * len(seeds) * len(initializations) * (
        4 + len(neural_methods) * len(sampling_budgets)
    )
    validation = _package_rows(run_dir, rows, expected)
    validation.update(
        {
            "all_required_reference_families_complete": True,
            "mdn_component_grid_complete": True,
            "normalized_spline_flow_included": True,
            "flow_depth_grid_complete": True,
            "internal_sampling_budget_grid_complete": True,
            "plug_riesz_orth_zero_decomposition_complete": True,
            "full_e10_reference_grid_complete": True,
            "superiority_requires_paired_ratio_analysis": True,
        }
    )
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def command_e10_analytic_mdn(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    families = [
        "E1_gaussian", "E4_discrete", "E4_continuous",
        "E5_support_motion", "E8_filtered", "E9_dense", "E11_ordering",
    ]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    initializations = list(range(args.initializations))
    component_grid = [5, 10, 20]
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e10-analytic-mdn-extension",
            "families": families,
            "seeds": seeds,
            "initializations": initializations,
            "n_train_total": args.n_train,
            "max_epochs": args.max_epochs,
            "components": component_grid,
            "covariance": "full_cholesky",
            "characteristic_readout": "analytic_autodiff",
            "evaluation_histories": 32,
            "access_tier": "O0",
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for family_index, family in enumerate(families):
        for seed in seeds:
            resource_guard()
            dataset = _e10_dataset(family, seed, args.n_train)
            split = int(0.8 * args.n_train)
            h_train, h_validation = dataset.h[:split], dataset.h[split:]
            y_train, y_validation = dataset.y[:split], dataset.y[split:]
            evaluation_h = _standard_normal_qmc(
                dataset.h.shape[1], 5, 860_000 + seed
            )
            for method_init in initializations:
                for components in component_grid:
                    resource_guard()
                    method = f"ANALYTIC_MDN_K{components}"
                    print(
                        f"E10-analytic family={family} method={method} "
                        f"seed={seed} init={method_init}",
                        flush=True,
                    )
                    training_seed = (
                        891_000 + 100 * seed + method_init
                        + 10_000 * family_index + components
                    )
                    model, trace = _fit_full_covariance_mdn(
                        h_train, y_train, h_validation, y_validation,
                        training_seed, components, args.max_epochs,
                    )
                    metrics = _analytic_mdn_metrics(
                        model, h_validation, y_validation, evaluation_h,
                        dataset.bank, dataset.truth,
                    )
                    traces.append(
                        {
                            "family": family,
                            "dgp_seed": seed,
                            "method_init": method_init,
                            "method": method,
                            **trace,
                        }
                    )
                    rows.append(
                        {
                            "experiment_id": "E10",
                            "cell_id": family,
                            "dgp_family": family,
                            "dgp_seed": seed,
                            "method": method,
                            "method_init": method_init,
                            "access_tier": "O0",
                            "compute_budget": "matched_local",
                            "n_train": args.n_train,
                            "history_dim": dataset.h.shape[1],
                            "response_dim": dataset.y.shape[1],
                            "query_count": dataset.bank.q,
                            "sampling_budget": 0,
                            "parameter_count": float(trace["parameter_count"]),
                            "runtime_seconds": float(trace["wall_seconds"]),
                            "validation_native_loss": float(trace["validation_loss"]),
                            "log_likelihood_defined": 1.0,
                            "energy_score": 0.0,
                            "energy_score_defined": 0.0,
                            **metrics,
                            "coverage_fraction": 0.0,
                            "max_abs_z": 0.0,
                            "detected_bonferroni": 0.0,
                            "calibration_slope_defined": 1.0,
                            "sign_accuracy_defined": 1.0,
                            "exact_normalized_likelihood": 1.0,
                            "warning_code": "analytic_characteristic_autodiff",
                            "fit_status": "ok",
                        }
                    )
                    del model
                    gc.collect()
    atomic_csv(run_dir / "training_traces.csv", pd.DataFrame(traces))
    expected = len(families) * len(seeds) * len(initializations) * len(component_grid)
    validation = _package_rows(run_dir, rows, expected)
    validation.update(
        {
            "all_required_reference_families_complete": True,
            "full_covariance_mdn_grid_complete": True,
            "analytic_characteristic_readout": True,
            "paired_comparison_uses_full_e10_orth_rows": True,
        }
    )
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distributional-sid-remaining")
    subparsers = parser.add_subparsers(dest="command", required=True)
    e3 = subparsers.add_parser("e3-dependent")
    e3.add_argument("--run-dir", required=True)
    e3.add_argument("--seed-start", type=int, default=1001)
    e3.add_argument("--seed-end", type=int, default=1030)
    e3.add_argument("--initializations", type=int, default=3)
    e3.set_defaults(func=command_e3_dependent)
    e8 = subparsers.add_parser("e8-measurement")
    e8.add_argument("--run-dir", required=True)
    e8.add_argument("--n-train", type=int, default=8000)
    e8.add_argument("--seed-start", type=int, default=1001)
    e8.add_argument("--seed-end", type=int, default=1030)
    e8.add_argument("--initializations", type=int, default=3)
    e8.set_defaults(func=command_e8_measurement)
    e9 = subparsers.add_parser("e9-staged")
    e9.add_argument("--run-dir", required=True)
    e9.add_argument("--seed-start", type=int, default=1001)
    e9.add_argument("--seed-end", type=int, default=1010)
    e9.add_argument("--initializations", type=int, default=3)
    e9.set_defaults(func=command_e9_staged)
    e10 = subparsers.add_parser("e10-full")
    e10.add_argument("--run-dir", required=True)
    e10.add_argument("--n-train", type=int, default=4000)
    e10.add_argument("--max-epochs", type=int, default=10)
    e10.add_argument("--seed-start", type=int, default=1001)
    e10.add_argument("--seed-end", type=int, default=1010)
    e10.add_argument("--initializations", type=int, default=3)
    e10.set_defaults(func=command_e10_full)
    e10_analytic = subparsers.add_parser("e10-analytic-mdn")
    e10_analytic.add_argument("--run-dir", required=True)
    e10_analytic.add_argument("--n-train", type=int, default=4000)
    e10_analytic.add_argument("--max-epochs", type=int, default=10)
    e10_analytic.add_argument("--seed-start", type=int, default=1001)
    e10_analytic.add_argument("--seed-end", type=int, default=1010)
    e10_analytic.add_argument("--initializations", type=int, default=3)
    e10_analytic.set_defaults(func=command_e10_analytic_mdn)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
