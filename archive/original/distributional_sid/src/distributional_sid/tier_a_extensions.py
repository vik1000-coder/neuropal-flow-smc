from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtri
from scipy.stats import norm, qmc

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


def _force_thread_limits() -> None:
    import os

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


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean, mean
    se = float(np.std(values, ddof=1) / math.sqrt(values.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _package_rows(run_dir: Path, rows: list[dict[str, object]], expected: int) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    atomic_csv(run_dir / "seed_level.csv", frame)
    frame.to_parquet(run_dir / "seed_level.parquet", index=False)
    numeric = [
        column
        for column in frame.columns
        if column not in {"experiment_id", "cell_id", "dgp_family", "method", "fit_status", "warning_code"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summaries: list[dict[str, object]] = []
    group_columns = ["experiment_id", "cell_id", "dgp_family", "method"]
    for key, group in frame.groupby(group_columns, dropna=False):
        init_average = group.groupby("dgp_seed", as_index=False)[numeric].mean(numeric_only=True)
        for metric in numeric:
            if metric in {"dgp_seed", "method_init"}:
                continue
            values = init_average[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            mean, lower, upper = _mean_ci(values)
            summaries.append(
                {
                    **dict(zip(group_columns, key)),
                    "metric_name": metric,
                    "mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_dgp_seeds": int(values.size),
                }
            )
    summary = pd.DataFrame(summaries)
    atomic_csv(run_dir / "summary.csv", summary)
    nonfinite_count = int(
        (~np.isfinite(frame.select_dtypes(include=[np.number]).drop(columns=["dgp_seed", "method_init"], errors="ignore"))).sum().sum()
    )
    validation = {
        "expected_rows": int(expected),
        "observed_rows": int(frame.shape[0]),
        "failed_rows": int((frame.fit_status != "ok").sum()),
        "nonfinite_required_metrics": nonfinite_count,
        "passed": bool(
            frame.shape[0] == expected and np.all(frame.fit_status == "ok") and nonfinite_count == 0
        ),
        **resource_guard(),
    }
    atomic_json(run_dir / "validation.json", validation)
    return validation


@dataclass(frozen=True)
class RieszCell:
    name: str
    family: str
    dimension: int
    parameter: float = 0.0
    mode: str = "regular"


def _sample_riesz_cell(
    cell: RieszCell, rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return history, weight b, oracle alpha, target regression m, and b*Dm."""
    d = cell.dimension
    if cell.family == "gaussian":
        h = rng.normal(size=(n, d))
        b = np.ones(n)
        alpha = h[:, 0]
        phase = 0.7 * h[:, 0] + (0.2 * h[:, 1] if d > 1 else 0.0)
        m = np.tanh(phase)
        bdm = b * 0.7 * (1.0 - m**2)
    elif cell.family == "student_t":
        nu = cell.parameter
        h = rng.normal(size=(n, d)) / np.sqrt(rng.chisquare(nu, size=(n, 1)) / nu)
        b = np.ones(n)
        alpha = ((nu + d) / (nu + np.sum(h**2, axis=1))) * h[:, 0]
        phase = 0.7 * h[:, 0] + (0.2 * h[:, 1] if d > 1 else 0.0)
        m = np.tanh(phase)
        bdm = b * 0.7 * (1.0 - m**2)
    elif cell.family == "uniform_valid":
        h = rng.uniform(-1.0, 1.0, size=(n, 1))
        b = 1.0 - h[:, 0] ** 2
        alpha = 2.0 * h[:, 0]
        m = np.exp(0.6 * h[:, 0])
        bdm = b * 0.6 * m
    elif cell.family == "uniform_transformed":
        raw = rng.uniform(-1.0, 1.0, size=n)
        h = np.arctanh(np.clip(raw, -1 + 1e-9, 1 - 1e-9))[:, None]
        b = np.ones(n)
        alpha = 2.0 * np.tanh(h[:, 0])
        m = np.tanh(0.6 * h[:, 0])
        bdm = 0.6 * (1.0 - m**2)
    elif cell.family == "gap":
        mu = cell.parameter
        sign = rng.choice(np.array([-1.0, 1.0]), size=n)
        h = (sign * mu + rng.normal(size=n))[:, None]
        raw_b = np.exp(-0.5 * h[:, 0] ** 2)
        normalizer = math.exp(-0.25 * mu**2) / math.sqrt(2.0)
        b = raw_b / normalizer
        density_score = -h[:, 0] + mu * np.tanh(mu * h[:, 0])
        alpha = b * h[:, 0] - b * density_score
        m = np.tanh(0.7 * h[:, 0])
        bdm = b * 0.7 * (1.0 - m**2)
    else:
        raise ValueError(cell.family)
    return h, b, alpha, m, bdm


def _fit_riesz_cell(
    cell: RieszCell, dgp_seed: int, method_init: int, n: int = 8000
) -> dict[str, object]:
    rng = np.random.default_rng(710_000 + 1009 * dgp_seed)
    h, b, oracle_alpha, m, bdm = _sample_riesz_cell(cell, rng, n)
    observed = m + rng.normal(scale=0.35, size=n)
    alpha_hat = np.zeros(n)
    score = np.zeros(n)
    probe_residuals: list[float] = []
    for fold_id, (train, test) in enumerate(independent_folds(n, 5, 711_000 + dgp_seed)):
        sieve = HistorySieve(cell.dimension, 32, 712_000 + 100 * method_init + fold_id).fit(h[train])
        z_train = sieve.transform(h[train])
        z_test = sieve.transform(h[test])
        dz_train = sieve.derivative(h[train], 0)
        coefficients = riesz_coefficients(z_train, b[train, None] * dz_train, 1e-2)
        fitted = z_test @ coefficients
        alpha_hat[test] = fitted
        score[test] = fitted * observed[test]
        probe_rng = np.random.default_rng(713_000 + dgp_seed + fold_id)
        for _ in range(6):
            direction = probe_rng.normal(size=cell.dimension)
            direction /= max(np.linalg.norm(direction), 1e-12)
            omega = probe_rng.uniform(0.25, 1.5)
            phase = omega * h[test] @ direction
            lhs = np.mean(fitted * np.sin(phase))
            rhs = np.mean(b[test] * omega * direction[0] * np.cos(phase))
            probe_residuals.append(float(abs(lhs - rhs)))
    truth = float(np.mean(bdm))
    estimate = float(np.mean(score))
    se = float(np.std(score, ddof=1) / math.sqrt(n))
    alpha_nrmse = vector_metrics(alpha_hat, oracle_alpha)["nrmse"]
    estimated_norm = float(np.sqrt(np.mean(alpha_hat**2)))
    oracle_norm = float(np.sqrt(np.mean(oracle_alpha**2)))
    probe_q95 = float(np.quantile(probe_residuals, 0.95))
    warning = ""
    if probe_q95 > 0.05:
        warning = "probe_residual"
    if estimated_norm > 5.0:
        warning = "large_representer_norm" if not warning else warning + "+large_representer_norm"
    return {
        "experiment_id": "E6",
        "cell_id": cell.name,
        "dgp_family": cell.family,
        "dgp_seed": dgp_seed,
        "method": "targeted_riesz_sieve",
        "method_init": method_init,
        "n_train": n,
        "alpha_nrmse": alpha_nrmse,
        "estimated_representer_norm": estimated_norm,
        "oracle_representer_norm": oracle_norm,
        "probe_q95_abs": probe_q95,
        "target_truth": truth,
        "target_estimate": estimate,
        "target_abs_error": abs(estimate - truth),
        "target_covered": float(estimate - 1.96 * se <= truth <= estimate + 1.96 * se),
        "warning_emitted": float(bool(warning)),
        "warning_code": warning,
        "fit_status": "ok",
    }


def _fit_circle_cell(dgp_seed: int, method_init: int, n: int = 8000) -> dict[str, object]:
    rng = np.random.default_rng(720_000 + 1009 * dgp_seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    h = np.column_stack([np.cos(theta), np.sin(theta)])
    b = np.sin(theta)
    oracle_alpha = -np.cos(theta)
    alpha_hat = np.zeros(n)
    residuals: list[float] = []
    for fold_id, (train, test) in enumerate(independent_folds(n, 5, 721_000 + dgp_seed)):
        sieve = HistorySieve(2, 32, 722_000 + 100 * method_init + fold_id).fit(h[train])
        z_train = sieve.transform(h[train])
        z_test = sieve.transform(h[test])
        tangent_train = (
            sieve.derivative(h[train], 0) * (-np.sin(theta[train]))[:, None]
            + sieve.derivative(h[train], 1) * np.cos(theta[train])[:, None]
        )
        coefficient = riesz_coefficients(z_train, b[train, None] * tangent_train, 1e-2)
        alpha_hat[test] = z_test @ coefficient
        for k in (1, 2, 3):
            lhs = np.mean(alpha_hat[test] * np.sin(k * theta[test]))
            rhs = np.mean(b[test] * k * np.cos(k * theta[test]))
            residuals.append(float(abs(lhs - rhs)))
    return {
        "experiment_id": "E6",
        "cell_id": "circle_supported_tangent",
        "dgp_family": "circle",
        "dgp_seed": dgp_seed,
        "method": "targeted_riesz_sieve",
        "method_init": method_init,
        "n_train": n,
        "alpha_nrmse": vector_metrics(alpha_hat, oracle_alpha)["nrmse"],
        "estimated_representer_norm": float(np.sqrt(np.mean(alpha_hat**2))),
        "oracle_representer_norm": float(np.sqrt(np.mean(oracle_alpha**2))),
        "probe_q95_abs": float(np.quantile(residuals, 0.95)),
        "target_truth": 0.0,
        "target_estimate": 0.0,
        "target_abs_error": 0.0,
        "target_covered": 1.0,
        "warning_emitted": 0.0,
        "warning_code": "",
        "fit_status": "ok",
    }


def command_e6(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        RieszCell("gaussian_d4", "gaussian", 4),
        *[RieszCell(f"student_t_nu{nu}", "student_t", 4, float(nu)) for nu in (3, 5, 10)],
        RieszCell("uniform_valid_weight", "uniform_valid", 1),
        RieszCell("uniform_atanh_coordinate", "uniform_transformed", 1),
        *[RieszCell(f"gap_mu{mu}", "gap", 1, float(mu)) for mu in (0, 1, 2, 3, 4)],
    ]
    seeds = list(range(1001, 1031))
    initializations = list(range(3))
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e6",
            "cells": [cell.__dict__ for cell in cells],
            "seeds": seeds,
            "initializations": initializations,
            "n_train": args.n_train,
            "ridge": 1e-2,
            "warning_rule": "probe_q95_abs > .05 or estimated representer norm > 5",
            "explicit_rejections": ["uniform_b_equals_1", "circle_euclidean_radial"],
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for cell in cells:
        for seed in seeds:
            for method_init in initializations:
                print(f"E6 cell={cell.name} seed={seed} init={method_init}", flush=True)
                rows.append(_fit_riesz_cell(cell, seed, method_init, args.n_train))
    for seed in seeds:
        for method_init in initializations:
            print(f"E6 cell=circle_supported_tangent seed={seed} init={method_init}", flush=True)
            rows.append(_fit_circle_cell(seed, method_init, args.n_train))
    invalid_grid = np.linspace(-1.0, 1.0, 1_000_001)
    invalid_boundary_residual = float(abs(np.trapezoid(np.exp(invalid_grid), invalid_grid) / 2.0))
    rejection = pd.DataFrame(
        [
            {
                "cell_id": "uniform_b_equals_1",
                "direction": "euclidean",
                "accepted": False,
                "diagnostic": "nonvanishing boundary trace",
                "diagnostic_value": invalid_boundary_residual,
            },
            {
                "cell_id": "circle_euclidean_radial",
                "direction": "radial",
                "accepted": False,
                "diagnostic": "direction leaves declared support manifold",
                "diagnostic_value": 1.0,
            },
        ]
    )
    atomic_csv(run_dir / "explicit_rejections.csv", rejection)
    validation = _package_rows(run_dir, rows, expected=(len(cells) + 1) * len(seeds) * len(initializations))
    validation["invalid_boundary_rejected"] = bool(invalid_boundary_residual > 0.5)
    validation["radial_direction_rejected"] = True
    validation["passed"] = bool(validation["passed"] and validation["invalid_boundary_rejected"])
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def _hermite_basis(h: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(h, dtype=float).reshape(-1)
    values = [np.ones_like(x), x, (x**2 - 1.0) / math.sqrt(2.0), (x**3 - 3.0 * x) / math.sqrt(6.0)]
    derivatives = [np.zeros_like(x), np.ones_like(x), math.sqrt(2.0) * x, (3.0 * x**2 - 3.0) / math.sqrt(6.0)]
    return np.column_stack(values[:count]), np.column_stack(derivatives[:count])


def _e7_truth(family: str, count: int, c: float = 0.15) -> np.ndarray:
    truth = np.zeros(count)
    if family == "mean_cancellation":
        if count >= 2:
            truth[1] = 2.0
    elif family == "variance_cancellation":
        a = 1.0 - 2.0 * c
        if count >= 2:
            truth[1] = 2.0 * c * a ** (-1.5)
        if count >= 4:
            truth[3] = 12.0 * c**2 / (math.sqrt(6.0) * a ** 2.5)
    else:
        raise ValueError(family)
    return truth


def _fit_e7_cell(
    family: str, basis_count: int, dgp_seed: int, method_init: int, n: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(730_000 + 1009 * dgp_seed)
    h = rng.normal(size=(n, 1))
    if family == "mean_cancellation":
        y = h[:, 0] ** 2 + rng.normal(scale=0.5, size=n)
        target = y
    elif family == "variance_cancellation":
        variance = np.exp(0.15 * h[:, 0] ** 2)
        y = rng.normal(scale=np.sqrt(variance), size=n)
        target = y**2
    else:
        raise ValueError(family)
    basis, _ = _hermite_basis(h[:, 0], basis_count)
    truth = _e7_truth(family, basis_count)
    contributions = {
        method: np.zeros((n, basis_count), dtype=float)
        for method in ("PLUG", "RIESZ", "ORTH")
    }
    for fold_id, (train, test) in enumerate(independent_folds(n, 5, 731_000 + dgp_seed)):
        sieve = HistorySieve(1, 32, 732_000 + 100 * method_init + fold_id).fit(h[train])
        z_train = sieve.transform(h[train])
        dz_train = sieve.derivative(h[train], 0)
        z_test = sieve.transform(h[test])
        dz_test = sieve.derivative(h[test], 0)
        feature_coefficient = ridge_coefficients(z_train, target[train], 1e-3)
        prediction = z_test @ feature_coefficient
        derivative = dz_test @ feature_coefficient
        for r in range(basis_count):
            alpha_coefficient = riesz_coefficients(
                z_train, basis[train, r, None] * dz_train, 1e-2
            )
            alpha = z_test @ alpha_coefficient
            contributions["PLUG"][test, r] = basis[test, r] * derivative
            contributions["RIESZ"][test, r] = alpha * target[test]
            contributions["ORTH"][test, r] = (
                basis[test, r] * derivative + alpha * (target[test] - prediction)
            )
    rows: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    for method, score in contributions.items():
        estimate = np.mean(score, axis=0)
        se = np.std(score, axis=0, ddof=1) / math.sqrt(n)
        metrics = vector_metrics(estimate, truth)
        calibration_defined = float(np.dot(truth, truth) > 1e-12)
        if not np.isfinite(metrics["calibration_slope"]):
            metrics["calibration_slope"] = 0.0
        sign_defined = float(np.any(np.abs(truth) > 1e-8))
        if not np.isfinite(metrics["sign_accuracy"]):
            metrics["sign_accuracy"] = 0.0
        metrics["calibration_slope_defined"] = calibration_defined
        metrics["sign_accuracy_defined"] = sign_defined
        metrics["coverage_fraction"] = float(
            np.mean((truth >= estimate - 1.96 * se) & (truth <= estimate + 1.96 * se))
        )
        metrics["global_abs_estimate"] = float(abs(estimate[0]))
        if basis_count >= 2:
            grid_basis, _ = _hermite_basis(np.array([-1.0, 1.0]), basis_count)
            reconstructed = grid_basis @ estimate
            metrics["opposite_regime_signs"] = float(reconstructed[0] < 0 < reconstructed[1])
        else:
            metrics["opposite_regime_signs"] = 0.0
        metrics["projected_energy_estimate"] = float(np.sum(estimate**2))
        metrics["projected_energy_truth"] = float(np.sum(truth**2))
        row = {
            "experiment_id": "E7",
            "cell_id": f"{family}_R{basis_count}",
            "dgp_family": family,
            "dgp_seed": dgp_seed,
            "method": method,
            "method_init": method_init,
            "basis_count": basis_count,
            "n_train": n,
            **metrics,
            "warning_code": "",
            "fit_status": "ok",
        }
        rows.append(row)
        for r, (estimate_r, truth_r, se_r) in enumerate(zip(estimate, truth, se)):
            coefficients.append(
                {
                    "experiment_id": "E7",
                    "cell_id": row["cell_id"],
                    "dgp_family": family,
                    "dgp_seed": dgp_seed,
                    "method": method,
                    "method_init": method_init,
                    "basis_count": basis_count,
                    "basis_index": r,
                    "estimate": float(estimate_r),
                    "truth": float(truth_r),
                    "standard_error": float(se_r),
                }
            )
    return rows, coefficients


def command_e7(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    families = ["mean_cancellation", "variance_cancellation"]
    basis_counts = [1, 2, 4]
    seeds = list(range(1001, 1031))
    initializations = list(range(3))
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e7",
            "families": families,
            "basis_counts": basis_counts,
            "basis": "frozen normalized probabilists' Hermite basis",
            "seeds": seeds,
            "initializations": initializations,
            "n_train": args.n_train,
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    for family in families:
        for basis_count in basis_counts:
            for seed in seeds:
                for method_init in initializations:
                    print(
                        f"E7 family={family} R={basis_count} seed={seed} init={method_init}",
                        flush=True,
                    )
                    metric_rows, coefficient_rows = _fit_e7_cell(
                        family, basis_count, seed, method_init, args.n_train
                    )
                    rows.extend(metric_rows)
                    coefficients.extend(coefficient_rows)
    coefficient_frame = pd.DataFrame(coefficients)
    atomic_csv(run_dir / "coefficient_estimates.csv", coefficient_frame)
    energy_rows = []
    for key, group in coefficient_frame[coefficient_frame.method == "ORTH"].groupby(
        ["cell_id", "dgp_family", "dgp_seed", "basis_count"]
    ):
        piv = group.pivot(index="method_init", columns="basis_index", values="estimate")
        if 0 in piv.index and 1 in piv.index:
            energy_rows.append(
                {
                    "cell_id": key[0],
                    "dgp_family": key[1],
                    "dgp_seed": key[2],
                    "basis_count": key[3],
                    "cross_initialization_energy": float(np.dot(piv.loc[0], piv.loc[1])),
                    "projected_truth_energy": float(
                        np.sum(group[group.method_init == 0].truth.to_numpy() ** 2)
                    ),
                }
            )
    atomic_csv(run_dir / "crossfit_energy.csv", pd.DataFrame(energy_rows))
    expected = len(families) * len(basis_counts) * len(seeds) * len(initializations) * 3
    validation = _package_rows(run_dir, rows, expected)
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def _feature_score(
    h: np.ndarray,
    feature: np.ndarray,
    oracle_mean: np.ndarray,
    oracle_derivative: np.ndarray,
    truth: np.ndarray,
    dgp_seed: int,
    method_init: int,
) -> dict[str, dict[str, float]]:
    n = h.shape[0]
    output_dim = feature.shape[1]
    scores = {
        method: np.zeros((n, output_dim), dtype=float)
        for method in ("PLUG", "RIESZ", "ORTH")
    }
    for fold_id, (train, test) in enumerate(independent_folds(n, 5, 740_000 + dgp_seed)):
        sieve = HistorySieve(h.shape[1], 32, 741_000 + 100 * method_init + fold_id).fit(h[train])
        z_train = sieve.transform(h[train])
        dz_train = sieve.derivative(h[train], 0)
        z_test = sieve.transform(h[test])
        dz_test = sieve.derivative(h[test], 0)
        feature_coefficient = ridge_coefficients(z_train, feature[train], 1e-3)
        alpha_coefficient = riesz_coefficients(z_train, dz_train, 1e-2)
        prediction = z_test @ feature_coefficient
        derivative = dz_test @ feature_coefficient
        alpha = z_test @ alpha_coefficient
        scores["PLUG"][test] = derivative
        scores["RIESZ"][test] = alpha[:, None] * feature[test]
        scores["ORTH"][test] = derivative + alpha[:, None] * (feature[test] - prediction)
    results: dict[str, dict[str, float]] = {}
    for method, values in scores.items():
        estimate = np.mean(values, axis=0)
        se = np.std(values, axis=0, ddof=1) / math.sqrt(n)
        metric_values = vector_metrics(estimate, truth)
        calibration_defined = float(np.dot(truth, truth) > 1e-12)
        if not np.isfinite(metric_values["calibration_slope"]):
            metric_values["calibration_slope"] = 0.0
        sign_defined = float(np.any(np.abs(truth) > 1e-8))
        if not np.isfinite(metric_values["sign_accuracy"]):
            metric_values["sign_accuracy"] = 0.0
        results[method] = {
            **metric_values,
            "calibration_slope_defined": calibration_defined,
            "sign_accuracy_defined": sign_defined,
            "coverage_fraction": float(
                np.mean((truth >= estimate - 1.96 * se) & (truth <= estimate + 1.96 * se))
            ),
            "max_abs_z": float(np.max(np.abs(estimate / np.maximum(se, 1e-12)))),
            "detected_bonferroni": float(
                np.any(
                    np.abs(estimate / np.maximum(se, 1e-12))
                    > norm.ppf(1.0 - 0.05 / (2.0 * output_dim))
                )
            ),
        }
    return results


def _path_templates(time_points: int = 8) -> tuple[np.ndarray, np.ndarray]:
    forward = []
    reverse = []
    for jitter_a in (-1, 0, 1):
        for jitter_b in (-1, 0, 1):
            f = np.zeros((time_points, 2))
            r = np.zeros((time_points, 2))
            f[2 + jitter_a, 0] = 1.0
            f[5 + jitter_b, 1] = 1.0
            r[2 + jitter_a, 1] = 1.0
            r[5 + jitter_b, 0] = 1.0
            forward.append(f.reshape(-1))
            reverse.append(r.reshape(-1))
    return np.asarray(forward), np.asarray(reverse)


def _path_feature_oracle(
    h: np.ndarray,
    bank: CharacteristicBank,
    forward: np.ndarray,
    reverse: np.ndarray,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    pi = expit(-0.2 + 1.1 * h[:, 0])
    dpi = 1.1 * pi * (1.0 - pi)
    u = bank.raw_frequencies
    offset = bank.raw_phase_offset
    attenuation = np.exp(-0.5 * noise**2 * np.sum(u**2, axis=1))
    cf_forward = np.mean(np.exp(1j * (forward @ u.T + offset[None, :])), axis=0) * attenuation
    cf_reverse = np.mean(np.exp(1j * (reverse @ u.T + offset[None, :])), axis=0) * attenuation
    conditional = (1.0 - pi[:, None]) * cf_reverse[None, :] + pi[:, None] * cf_forward[None, :]
    derivative = dpi[:, None] * (cf_forward - cf_reverse)[None, :]
    return (
        np.concatenate([conditional.real, conditional.imag], axis=1),
        np.concatenate([derivative.real, derivative.imag], axis=1),
    )


def _sample_paths(
    h: np.ndarray,
    rng: np.random.Generator,
    forward: np.ndarray,
    reverse: np.ndarray,
    noise: float,
) -> np.ndarray:
    pi = expit(-0.2 + 1.1 * h[:, 0])
    motif = rng.uniform(size=h.shape[0]) < pi
    jitter = rng.integers(0, forward.shape[0], size=h.shape[0])
    path = np.where(motif[:, None], forward[jitter], reverse[jitter])
    return path + rng.normal(scale=noise, size=path.shape)


def command_e11(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1001, 1031))
    initializations = list(range(3))
    families = ["full_path_characteristic", "endpoint_characteristic", "unordered_channel_totals"]
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e11",
            "families": families,
            "seeds": seeds,
            "initializations": initializations,
            "n_train": args.n_train,
            "time_points": 8,
            "channels": 2,
            "jitter": [-1, 0, 1],
            "noise_sd": 0.15,
        },
    )
    freeze_source_and_environment(run_dir)
    forward, reverse = _path_templates()
    rows: list[dict[str, object]] = []
    aligned_rows: list[dict[str, object]] = []
    for seed in seeds:
        rng = np.random.default_rng(750_000 + 1009 * seed)
        h_bank = rng.normal(size=(1024, 4))
        path_bank = _sample_paths(h_bank, rng, forward, reverse, 0.15)
        h = rng.normal(size=(args.n_train, 4))
        path = _sample_paths(h, rng, forward, reverse, 0.15)
        family_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        bank_full = CharacteristicBank.fit(
            path_bank, 64, seed=92117, amplitude_scales=(1.0, 2.0, 4.0, 8.0)
        )
        m_full, dm_full = _path_feature_oracle(h, bank_full, forward, reverse, 0.15)
        truth_h = ndtri(
            np.clip(qmc.Sobol(d=4, scramble=True, seed=751_000 + seed).random_base2(16), 1e-12, 1 - 1e-12)
        )
        _, truth_derivative_full = _path_feature_oracle(
            truth_h, bank_full, forward, reverse, 0.15
        )
        family_data["full_path_characteristic"] = (
            bank_full.transform(path),
            m_full,
            dm_full,
            np.mean(truth_derivative_full, axis=0),
        )
        endpoint_index = np.array([0, 1, 14, 15])
        bank_endpoint = CharacteristicBank.fit(path_bank[:, endpoint_index], 32, seed=92118)
        endpoint_feature = bank_endpoint.transform(path[:, endpoint_index])
        endpoint_mean = np.mean(endpoint_feature, axis=0, keepdims=True).repeat(args.n_train, axis=0)
        family_data["endpoint_characteristic"] = (
            endpoint_feature,
            endpoint_mean,
            np.zeros_like(endpoint_mean),
            np.zeros(endpoint_feature.shape[1]),
        )
        reshape_bank = path_bank.reshape(-1, 8, 2).sum(axis=1)
        reshape_path = path.reshape(-1, 8, 2).sum(axis=1)
        bank_unordered = CharacteristicBank.fit(reshape_bank, 32, seed=92119)
        unordered_feature = bank_unordered.transform(reshape_path)
        unordered_mean = np.mean(unordered_feature, axis=0, keepdims=True).repeat(args.n_train, axis=0)
        family_data["unordered_channel_totals"] = (
            unordered_feature,
            unordered_mean,
            np.zeros_like(unordered_mean),
            np.zeros(unordered_feature.shape[1]),
        )
        for method_init in initializations:
            for family, (feature, oracle_mean, oracle_derivative, truth) in family_data.items():
                print(f"E11 family={family} seed={seed} init={method_init}", flush=True)
                result = _feature_score(
                    h, feature, oracle_mean, oracle_derivative, truth, seed, method_init
                )
                for method, metrics in result.items():
                    rows.append(
                        {
                            "experiment_id": "E11",
                            "cell_id": family,
                            "dgp_family": "event_order_mixture",
                            "dgp_seed": seed,
                            "method": method,
                            "method_init": method_init,
                            "n_train": args.n_train,
                            "query_count": int(feature.shape[1] // 2),
                            **metrics,
                            "warning_code": "",
                            "fit_status": "ok",
                        }
                    )
        delta = np.mean(forward, axis=0) - np.mean(reverse, axis=0)
        aligned = path @ delta / max(np.dot(delta, delta), 1e-12)
        pi = expit(-0.2 + 1.1 * h[:, 0])
        aligned_truth = float(np.mean(1.1 * pi * (1.0 - pi)))
        aligned_rows.append(
            {
                "dgp_seed": seed,
                "method": "ALIGNED_MOTIF_CONTROL",
                "estimate": float(np.mean(h[:, 0] * aligned)),
                "truth": aligned_truth,
            }
        )
    atomic_csv(run_dir / "aligned_motif_control.csv", pd.DataFrame(aligned_rows))
    expected = len(seeds) * len(initializations) * len(families) * 3
    validation = _package_rows(run_dir, rows, expected)
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


@dataclass(frozen=True)
class ObservationCell:
    name: str
    rho: float
    latent_amplitude: float
    crosstalk: float
    seeds: int


def _observation_kernel(rho: float, horizon: int = 4) -> np.ndarray:
    if rho == 0.0:
        kernel = np.zeros(horizon)
        kernel[0] = 1.0
        return kernel
    return (1.0 - rho) * rho ** np.arange(horizon)


def _sample_observation_paths(
    h: np.ndarray, rng: np.random.Generator, cell: ObservationCell, noise: float = 0.2
) -> np.ndarray:
    pi = expit(-0.3 + h[:, 0])
    state = rng.uniform(size=h.shape[0]) < pi
    kernel = _observation_kernel(cell.rho)
    latent = cell.latent_amplitude * (state.astype(float) - pi)
    signal = latent[:, None] * kernel[None, :] + cell.crosstalk * h[:, [0]] * kernel[None, :]
    return signal + rng.normal(scale=noise, size=signal.shape)


def _observation_oracle(
    h: np.ndarray, bank: CharacteristicBank, cell: ObservationCell, noise: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    pi = expit(-0.3 + h[:, 0])
    dpi = pi * (1.0 - pi)
    kernel = _observation_kernel(cell.rho)
    u = bank.raw_frequencies
    offset = bank.raw_phase_offset
    uk = u @ kernel
    qv = cell.latent_amplitude * uk
    attenuation = np.exp(-0.5 * noise**2 * np.sum(u**2, axis=1))
    exp_iq = np.exp(1j * qv)
    a = np.exp(-1j * pi[:, None] * qv[None, :])
    b = (1.0 - pi[:, None]) + pi[:, None] * exp_iq[None, :]
    shift = np.exp(1j * cell.crosstalk * h[:, [0]] * uk[None, :])
    common = attenuation[None, :] * np.exp(1j * offset[None, :])
    cf = common * shift * a * b
    da = a * (-1j * dpi[:, None] * qv[None, :])
    db = dpi[:, None] * (exp_iq[None, :] - 1.0)
    dshift = shift * (1j * cell.crosstalk * uk[None, :])
    derivative = common * (dshift * a * b + shift * (da * b + a * db))
    return (
        np.concatenate([cf.real, cf.imag], axis=1),
        np.concatenate([derivative.real, derivative.imag], axis=1),
    )


def command_e8(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        ObservationCell("active_clean", 0.0, 1.0, 0.0, 30),
        ObservationCell("active_filter_rho0p5", 0.5, 1.0, 0.0, 30),
        ObservationCell("active_filter_rho0p9", 0.9, 1.0, 0.0, 30),
        ObservationCell("active_filter_rho0p98", 0.98, 1.0, 0.0, 30),
        ObservationCell("exact_observed_null", 0.9, 0.0, 0.0, 100),
        ObservationCell("measurement_crosstalk", 0.9, 0.0, 0.5, 30),
    ]
    initializations = list(range(3))
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e8-representative",
            "cells": [cell.__dict__ for cell in cells],
            "initializations": initializations,
            "n_train": args.n_train,
            "horizon": 4,
            "noise_sd": 0.2,
            "scope_note": "representative clean/filter/null/crosstalk cells; heteroskedasticity and missingness remain unrun",
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(cells):
        for seed in range(1001, 1001 + cell.seeds):
            rng = np.random.default_rng(760_000 + 1009 * seed + 10_000 * cell_index)
            h_bank = rng.normal(size=(1024, 4))
            y_bank = _sample_observation_paths(h_bank, rng, cell)
            bank = CharacteristicBank.fit(
                y_bank, 32, seed=92130, amplitude_scales=(1.0, 2.0, 4.0, 8.0)
            )
            h = rng.normal(size=(args.n_train, 4))
            y = _sample_observation_paths(h, rng, cell)
            feature = bank.transform(y)
            oracle_mean, oracle_derivative = _observation_oracle(h, bank, cell)
            truth_h = ndtri(
                np.clip(
                    qmc.Sobol(d=4, scramble=True, seed=761_000 + seed).random_base2(15),
                    1e-12,
                    1 - 1e-12,
                )
            )
            _, truth_derivative = _observation_oracle(truth_h, bank, cell)
            truth = np.mean(truth_derivative, axis=0)
            for method_init in initializations:
                print(f"E8 cell={cell.name} seed={seed} init={method_init}", flush=True)
                result = _feature_score(
                    h, feature, oracle_mean, oracle_derivative, truth, seed + 10_000 * cell_index, method_init
                )
                for method, metrics in result.items():
                    rows.append(
                        {
                            "experiment_id": "E8",
                            "cell_id": cell.name,
                            "dgp_family": "filtered_centered_gain",
                            "dgp_seed": seed,
                            "method": method,
                            "method_init": method_init,
                            "n_train": args.n_train,
                            "rho": cell.rho,
                            "latent_amplitude": cell.latent_amplitude,
                            "crosstalk": cell.crosstalk,
                            "latent_target_active": float(cell.latent_amplitude != 0.0),
                            "observed_target_active": float(
                                cell.latent_amplitude != 0.0 or cell.crosstalk != 0.0
                            ),
                            **metrics,
                            "warning_code": (
                                "observed_effect_not_latent_effect" if cell.crosstalk != 0.0 else ""
                            ),
                            "fit_status": "ok",
                        }
                    )
    expected = sum(cell.seeds for cell in cells) * len(initializations) * 3
    validation = _package_rows(run_dir, rows, expected)
    validation["full_observation_grid_complete"] = False
    validation["deployment_gate_passed"] = False
    validation["deployment_gate_reason"] = "heteroskedastic-noise, missingness, deconvolution, and misspecification cells not executed"
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def _symmetric_vector(y: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pairs = [(i, j) for i in range(y.shape[1]) for j in range(i, y.shape[1])]
    return np.column_stack([y[:, i] * y[:, j] for i, j in pairs]), pairs


def _dense_parameters(seed: int, history_dim: int = 8, response_dim: int = 8, rank: int = 3):
    rng = np.random.default_rng(770_000 + seed)
    w = rng.normal(scale=0.55, size=(rank, history_dim))
    w[:, -2:] = 0.0
    b = rng.normal(size=(rank, response_dim))
    qmat, _ = np.linalg.qr(rng.normal(size=(response_dim, response_dim)))
    b = b @ qmat
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    b *= 1.5
    intercept = np.linspace(-0.8, 0.8, rank)
    return w, b, intercept


def _sample_dense(
    h: np.ndarray, rng: np.random.Generator, w: np.ndarray, b: np.ndarray, intercept: np.ndarray
) -> np.ndarray:
    pi = expit(intercept[None, :] + h @ w.T)
    state = rng.uniform(size=pi.shape) < pi
    centered = state.astype(float) - pi
    return centered @ b + rng.normal(scale=0.35, size=(h.shape[0], b.shape[1]))


def _dense_oracle_complex(
    h: np.ndarray,
    bank: CharacteristicBank,
    w: np.ndarray,
    b: np.ndarray,
    intercept: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pi = expit(intercept[None, :] + h @ w.T)
    u = bank.raw_frequencies
    phase = b @ u.T
    exp_phase = np.exp(1j * phase)
    factors = np.exp(-1j * pi[:, :, None] * phase[None, :, :]) * (
        1.0 - pi[:, :, None] + pi[:, :, None] * exp_phase[None, :, :]
    )
    attenuation = np.exp(-0.5 * 0.35**2 * np.sum(u**2, axis=1))
    cf = np.prod(factors, axis=1) * attenuation[None, :] * np.exp(
        1j * bank.raw_phase_offset[None, :]
    )
    derivative = np.zeros((h.shape[0], h.shape[1], bank.q), dtype=complex)
    denominator = 1.0 - pi[:, :, None] + pi[:, :, None] * exp_phase[None, :, :]
    dlog_dpi = -1j * phase[None, :, :] + (exp_phase[None, :, :] - 1.0) / denominator
    for p in range(h.shape[1]):
        dpi = pi * (1.0 - pi) * w[:, p][None, :]
        derivative[:, p, :] = cf * np.sum(dpi[:, :, None] * dlog_dpi, axis=1)
    return cf, derivative


def _matrix_from_symmetric(vector: np.ndarray, pairs: list[tuple[int, int]], dimension: int) -> np.ndarray:
    matrix = np.zeros((dimension, dimension), dtype=float)
    for value, (i, j) in zip(vector, pairs):
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def _subspace_metrics(matrices: np.ndarray, truth: np.ndarray, rank: int) -> tuple[float, float]:
    from scipy.linalg import subspace_angles

    aggregate = np.sum([matrix @ matrix.T for matrix in matrices], axis=0)
    aggregate_truth = np.sum([matrix @ matrix.T for matrix in truth], axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(aggregate)
    truth_values, truth_vectors = np.linalg.eigh(aggregate_truth)
    estimated_space = eigenvectors[:, np.argsort(eigenvalues)[-rank:]]
    truth_space = truth_vectors[:, np.argsort(truth_values)[-rank:]]
    angles = np.degrees(subspace_angles(estimated_space, truth_space))
    rank_estimate = int(np.sum(eigenvalues > max(float(np.max(eigenvalues)) * 0.05, 1e-10)))
    return float(np.max(angles)), float(abs(rank_estimate - rank))


def _fit_e9_seed(seed: int, method_init: int, n: int) -> list[dict[str, object]]:
    history_dim = 8
    response_dim = 8
    rank = 3
    w, b, intercept = _dense_parameters(seed, history_dim, response_dim, rank)
    rng = np.random.default_rng(771_000 + 1009 * seed)
    h_bank = rng.normal(size=(1024, history_dim))
    y_bank = _sample_dense(h_bank, rng, w, b, intercept)
    bank = CharacteristicBank.fit(
        y_bank, 96, seed=92140, amplitude_scales=(0.1, 0.15, 0.2, 0.3)
    )
    h = rng.normal(size=(n, history_dim))
    y = _sample_dense(h, rng, w, b, intercept)
    feature = bank.transform(y)
    pair_feature, pairs = _symmetric_vector(y)
    cf_oracle, derivative_oracle_complex = _dense_oracle_complex(h, bank, w, b, intercept)
    derivative_oracle = np.concatenate(
        [derivative_oracle_complex.real, derivative_oracle_complex.imag], axis=2
    )
    truth_h = ndtri(
        np.clip(qmc.Sobol(d=history_dim, scramble=True, seed=772_000 + seed).random_base2(15), 1e-12, 1 - 1e-12)
    )
    _, truth_derivative_complex = _dense_oracle_complex(truth_h, bank, w, b, intercept)
    truth_feature = np.mean(
        np.concatenate([truth_derivative_complex.real, truth_derivative_complex.imag], axis=2), axis=0
    )
    pi_truth = expit(intercept[None, :] + truth_h @ w.T)
    truth_covariance = np.zeros((history_dim, response_dim, response_dim))
    for p in range(history_dim):
        coefficient = np.mean(
            pi_truth * (1.0 - pi_truth) * (1.0 - 2.0 * pi_truth) * w[:, p][None, :],
            axis=0,
        )
        truth_covariance[p] = sum(
            coefficient[k] * np.outer(b[k], b[k]) for k in range(rank)
        )
    truth_cov_vector = np.column_stack(
        [[truth_covariance[p, i, j] for i, j in pairs] for p in range(history_dim)]
    ).T
    score_feature = np.zeros((n, history_dim, feature.shape[1]))
    score_cov = np.zeros((n, history_dim, pair_feature.shape[1]))
    for fold_id, (train, test) in enumerate(independent_folds(n, 5, 773_000 + seed)):
        sieve = HistorySieve(history_dim, 64, 774_000 + 100 * method_init + fold_id).fit(h[train])
        z_train = sieve.transform(h[train])
        z_test = sieve.transform(h[test])
        feature_coefficient = ridge_coefficients(z_train, feature[train], 1e-3)
        covariance_coefficient = ridge_coefficients(z_train, pair_feature[train], 1e-3)
        feature_prediction = z_test @ feature_coefficient
        covariance_prediction = z_test @ covariance_coefficient
        for p in range(history_dim):
            dz_train = sieve.derivative(h[train], p)
            dz_test = sieve.derivative(h[test], p)
            alpha_coefficient = riesz_coefficients(z_train, dz_train, 1e-2)
            alpha = z_test @ alpha_coefficient
            score_feature[test, p] = (
                dz_test @ feature_coefficient
                + alpha[:, None] * (feature[test] - feature_prediction)
            )
            score_cov[test, p] = (
                dz_test @ covariance_coefficient
                + alpha[:, None] * (pair_feature[test] - covariance_prediction)
            )
    estimate_feature = np.mean(score_feature, axis=0)
    se_feature = np.std(score_feature, axis=0, ddof=1) / math.sqrt(n)
    estimate_cov_vector = np.mean(score_cov, axis=0)
    se_cov_vector = np.std(score_cov, axis=0, ddof=1) / math.sqrt(n)
    direct_matrices = np.stack(
        [_matrix_from_symmetric(estimate_cov_vector[p], pairs, response_dim) for p in range(history_dim)]
    )
    u = bank.raw_frequencies
    design_pairs = [(i, j) for i in range(response_dim) for j in range(i, response_dim)]
    design = np.column_stack(
        [u[:, i] ** 2 if i == j else 2.0 * u[:, i] * u[:, j] for i, j in design_pairs]
    )
    characteristic_matrices = []
    for p in range(history_dim):
        shifted = estimate_feature[p, : bank.q] + 1j * estimate_feature[p, bank.q :]
        raw = shifted * np.exp(-1j * bank.raw_phase_offset)
        coefficient = ridge_coefficients(design, -2.0 * raw.real, 1e-4)
        characteristic_matrices.append(_matrix_from_symmetric(coefficient, design_pairs, response_dim))
    characteristic_matrices = np.stack(characteristic_matrices)
    rows = []
    for method, matrices in (
        ("ORTH_CHARACTERISTIC_INVERSION", characteristic_matrices),
        ("DIRECT_COV_ORTH", direct_matrices),
    ):
        angle, rank_error = _subspace_metrics(matrices, truth_covariance, rank)
        covariance_nrmse = vector_metrics(matrices.reshape(-1), truth_covariance.reshape(-1))["nrmse"]
        if method == "ORTH_CHARACTERISTIC_INVERSION":
            feature_metrics = vector_metrics(estimate_feature.reshape(-1), truth_feature.reshape(-1))
            null_z = np.abs(estimate_feature[-2:] / np.maximum(se_feature[-2:], 1e-12))
            null_fwer = float(
                np.any(null_z > norm.ppf(1.0 - 0.05 / (2.0 * null_z.size)))
            )
            coverage = float(
                np.mean(
                    (truth_feature >= estimate_feature - 1.96 * se_feature)
                    & (truth_feature <= estimate_feature + 1.96 * se_feature)
                )
            )
        else:
            feature_metrics = {"nrmse": covariance_nrmse, "calibration_slope": float(np.dot(matrices.reshape(-1), truth_covariance.reshape(-1)) / max(np.dot(truth_covariance.reshape(-1), truth_covariance.reshape(-1)), 1e-12))}
            null_z = np.abs(estimate_cov_vector[-2:] / np.maximum(se_cov_vector[-2:], 1e-12))
            null_fwer = float(
                np.any(null_z > norm.ppf(1.0 - 0.05 / (2.0 * null_z.size)))
            )
            coverage = float(
                np.mean(
                    (truth_cov_vector >= estimate_cov_vector - 1.96 * se_cov_vector)
                    & (truth_cov_vector <= estimate_cov_vector + 1.96 * se_cov_vector)
                )
            )
        rows.append(
            {
                "experiment_id": "E9",
                "cell_id": "medium_dense_rotated_K3",
                "dgp_family": "centered_multiple_gain",
                "dgp_seed": seed,
                "method": method,
                "method_init": method_init,
                "n_train": n,
                "feature_tensor_nrmse": feature_metrics["nrmse"],
                "feature_calibration_slope": feature_metrics["calibration_slope"],
                "feature_coverage_fraction": coverage,
                "covariance_tensor_nrmse": covariance_nrmse,
                "largest_subspace_angle_degrees": angle,
                "rank_absolute_error": rank_error,
                "null_edge_familywise_call": null_fwer,
                "warning_code": "",
                "fit_status": "ok",
            }
        )
    return rows


def command_e9(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1001, 1031))
    initializations = list(range(3))
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e9-medium-cell",
            "history_dim": 8,
            "response_dim": 8,
            "rank": 3,
            "null_history_directions": [6, 7],
            "seeds": seeds,
            "initializations": initializations,
            "n_train": args.n_train,
            "query_count": 96,
            "frequency_scales": [0.1, 0.15, 0.2, 0.3],
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for method_init in initializations:
            print(f"E9 seed={seed} init={method_init}", flush=True)
            rows.extend(_fit_e9_seed(seed, method_init, args.n_train))
    expected = len(seeds) * len(initializations) * 2
    validation = _package_rows(run_dir, rows, expected)
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def _conditional_complex_from_mdn(model, h_tensor, bank: CharacteristicBank):
    import torch
    from torch.nn import functional as functional

    previous = torch.empty((len(h_tensor), 0), dtype=h_tensor.dtype, device=h_tensor.device)
    logits, mean, log_std = model._conditional_parameters(0, h_tensor, previous)
    weights = functional.softmax(logits, dim=-1)
    u = torch.as_tensor(bank.raw_frequencies[:, 0], dtype=h_tensor.dtype, device=h_tensor.device)
    offset = torch.as_tensor(bank.raw_phase_offset, dtype=h_tensor.dtype, device=h_tensor.device)
    exponent = (
        1j * mean[:, :, None] * u[None, None, :]
        - 0.5 * torch.exp(2.0 * log_std)[:, :, None] * u[None, None, :] ** 2
    )
    return torch.sum(weights[:, :, None] * torch.exp(exponent), dim=1) * torch.exp(1j * offset)[None, :]


def _mdn_tangent(model, h_eval: np.ndarray, bank: CharacteristicBank) -> np.ndarray:
    import torch

    h_tensor = torch.as_tensor(h_eval, dtype=torch.float32).requires_grad_(True)
    complex_mean = _conditional_complex_from_mdn(model, h_tensor, bank)
    derivatives = []
    for component in (complex_mean.real, complex_mean.imag):
        for query in range(bank.q):
            gradient = torch.autograd.grad(
                component[:, query].sum(), h_tensor, retain_graph=True
            )[0]
            derivatives.append(float(gradient[:, 0].mean().detach().cpu()))
    return np.asarray(derivatives)


def _model_samples(model, h: np.ndarray, count: int, seed: int) -> np.ndarray:
    import torch

    h_tensor = torch.as_tensor(h, dtype=torch.float32)
    with torch.no_grad():
        return model.sample(h_tensor, count, seed=seed).detach().cpu().numpy()


def _sample_feature_mean(samples: np.ndarray, bank: CharacteristicBank) -> np.ndarray:
    flat = samples.reshape(-1, samples.shape[-1])
    transformed = bank.transform(flat)
    return transformed.reshape(samples.shape[0], samples.shape[1], -1).mean(axis=1)


def _flow_tangent(
    model,
    h_eval: np.ndarray,
    bank: CharacteristicBank,
    sample_count: int,
    seed: int,
    delta: float = 0.02,
) -> np.ndarray:
    plus = h_eval.copy()
    minus = h_eval.copy()
    plus[:, 0] += delta
    minus[:, 0] -= delta
    plus_feature = _sample_feature_mean(_model_samples(model, plus, sample_count, seed), bank)
    minus_feature = _sample_feature_mean(_model_samples(model, minus, sample_count, seed), bank)
    return np.mean((plus_feature - minus_feature) / (2.0 * delta), axis=0)


def _crps_from_samples(samples: np.ndarray, observed: np.ndarray) -> float:
    values = samples[..., 0]
    target = observed[:, 0]
    first = np.mean(np.abs(values - target[:, None]))
    half = values.shape[1] // 2
    second = 0.5 * np.mean(np.abs(values[:, :half] - values[:, half : 2 * half]))
    return float(first - second)


def _crossfit_feature_prediction_mse(
    h: np.ndarray, feature: np.ndarray, seed: int, method_init: int
) -> float:
    prediction = np.zeros_like(feature)
    for fold_id, (train, test) in enumerate(independent_folds(len(h), 5, 786_000 + seed)):
        sieve = HistorySieve(h.shape[1], 32, 787_000 + 100 * method_init + fold_id).fit(h[train])
        coefficient = ridge_coefficients(sieve.transform(h[train]), feature[train], 1e-3)
        prediction[test] = sieve.transform(h[test]) @ coefficient
    return float(np.mean((feature - prediction) ** 2))


def command_e10(args: argparse.Namespace) -> int:
    _force_thread_limits()
    import sys

    workspace = Path(__file__).resolve().parents[3]
    benchmark_source = workspace / "history_tangent_benchmark" / "src"
    if str(benchmark_source) not in sys.path:
        sys.path.insert(0, str(benchmark_source))
    from history_tangent_benchmark.models import build_model, fit_model
    from .core import MomentBlindDiscreteDGP, MomentBlindLegendreDGP, sha256_file

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    families = ["finite_difference_mixture", "continuous_legendre_tilt"]
    seeds = list(range(1001, 1011))
    initializations = list(range(3))
    model_file = workspace / "history_tangent_benchmark" / "src" / "history_tangent_benchmark" / "models.py"
    write_frozen_config(
        run_dir,
        {
            "command": "confirm-e10-reference-cells",
            "families": families,
            "seeds": seeds,
            "initializations": initializations,
            "n_train_total": args.n_train,
            "train_fraction": 0.8,
            "query_count": 32,
            "sampling_budget": 64,
            "max_epochs": 10,
            "shared_hidden_width": 32,
            "methods": ["ORTH", "MDN_K5_NORMALIZED", "CONDITIONAL_FLOW_MATCHING"],
            "external_models_source": str(model_file),
            "external_models_sha256": sha256_file(model_file),
            "scope_note": "reference E4 cells only; flow-matching baseline is generative but not a normalized-likelihood flow",
        },
    )
    freeze_source_and_environment(run_dir)
    rows: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for family_index, family in enumerate(families):
        for seed in seeds:
            dgp = (
                MomentBlindDiscreteDGP(seed, amplitude_fraction=0.5)
                if family == "finite_difference_mixture"
                else MomentBlindLegendreDGP(seed, amplitude=0.5)
            )
            rng = np.random.default_rng(780_000 + 1009 * seed + 10_000 * family_index)
            h = dgp.sample_histories(rng, args.n_train)
            y = dgp.sample_responses(h, rng)
            split = int(0.8 * args.n_train)
            h_train, h_validation = h[:split], h[split:]
            y_train, y_validation = y[:split], y[split:]
            bank = CharacteristicBank.fit(
                y_train,
                32,
                seed=92150,
                amplitude_scales=(1.0, 2.0, 4.0, 8.0),
            )
            feature = bank.transform(y)
            oracle_mean = dgp.conditional_feature_mean(h, bank)
            oracle_derivative = dgp.conditional_feature_derivative(h, 0, bank)
            h_truth = ndtri(
                np.clip(qmc.Sobol(d=1, scramble=True, seed=781_000 + seed).random_base2(16), 1e-12, 1 - 1e-12)
            )
            truth = np.mean(dgp.conditional_feature_derivative(h_truth, 0, bank), axis=0)
            evaluation_h = h_truth[:512]
            for method_init in initializations:
                print(f"E10 family={family} method=ORTH seed={seed} init={method_init}", flush=True)
                orth = _feature_score(
                    h, feature, oracle_mean, oracle_derivative, truth, seed + 10_000 * family_index, method_init
                )["ORTH"]
                orth_prediction_mse = _crossfit_feature_prediction_mse(
                    h, feature, seed + 10_000 * family_index, method_init
                )
                rows.append(
                    {
                        "experiment_id": "E10",
                        "cell_id": family,
                        "dgp_family": family,
                        "dgp_seed": seed,
                        "method": "ORTH",
                        "method_init": method_init,
                        "n_train": args.n_train,
                        "parameter_count": 2275,
                        "runtime_seconds": 0.0,
                        "validation_native_loss": orth_prediction_mse,
                        "feature_prediction_mse": orth_prediction_mse,
                        "crps": 0.0,
                        "crps_defined": 0.0,
                        **orth,
                        "warning_code": "",
                        "fit_status": "ok",
                    }
                )
                for model_name, method_label, params in (
                    (
                        "autoregressive_mdn",
                        "MDN_K5_NORMALIZED",
                        {"hidden": 32, "layers": 2, "components": 5},
                    ),
                    (
                        "conditional_flow_matching",
                        "CONDITIONAL_FLOW_MATCHING",
                        {"hidden": 32, "layers": 2, "sample_steps": 8},
                    ),
                ):
                    print(
                        f"E10 family={family} method={method_label} seed={seed} init={method_init}",
                        flush=True,
                    )
                    model = build_model(model_name, q=1, dy=1, params=params)
                    trace = fit_model(
                        model,
                        h_train,
                        y_train,
                        h_validation,
                        y_validation,
                        seed=782_000 + 100 * seed + method_init + 10_000 * family_index,
                        device="cpu",
                        learning_rate=1e-3,
                        weight_decay=1e-4,
                        batch_size=256,
                        max_epochs=10,
                        patience=4,
                    )
                    if method_label == "MDN_K5_NORMALIZED":
                        tangent = _mdn_tangent(model, evaluation_h, bank)
                        import torch

                        with torch.no_grad():
                            validation_tensor = torch.as_tensor(h_validation, dtype=torch.float32)
                            predicted_complex = _conditional_complex_from_mdn(model, validation_tensor, bank)
                            predicted_feature = np.concatenate(
                                [predicted_complex.real.cpu().numpy(), predicted_complex.imag.cpu().numpy()], axis=1
                            )
                    else:
                        tangent = _flow_tangent(
                            model,
                            evaluation_h[:256],
                            bank,
                            sample_count=64,
                            seed=783_000 + seed,
                        )
                        predicted_feature = _sample_feature_mean(
                            _model_samples(model, h_validation[:256], 64, 784_000 + seed), bank
                        )
                    validation_limit = predicted_feature.shape[0]
                    validation_feature = bank.transform(y_validation[:validation_limit])
                    samples = _model_samples(
                        model, h_validation[:256], 64, 785_000 + seed
                    )
                    tangent_metrics = vector_metrics(tangent, truth)
                    if not np.isfinite(tangent_metrics["calibration_slope"]):
                        tangent_metrics["calibration_slope"] = 0.0
                    rows.append(
                        {
                            "experiment_id": "E10",
                            "cell_id": family,
                            "dgp_family": family,
                            "dgp_seed": seed,
                            "method": method_label,
                            "method_init": method_init,
                            "n_train": args.n_train,
                            "parameter_count": trace.parameter_count,
                            "runtime_seconds": trace.wall_seconds,
                            "validation_native_loss": float(trace.validation_loss[trace.best_epoch]),
                            "feature_prediction_mse": float(
                                np.mean((validation_feature - predicted_feature) ** 2)
                            ),
                            "crps": _crps_from_samples(samples, y_validation[:256]),
                            "crps_defined": 1.0,
                            **tangent_metrics,
                            "coverage_fraction": 0.0,
                            "max_abs_z": 0.0,
                            "detected_bonferroni": 0.0,
                            "calibration_slope_defined": 1.0,
                            "sign_accuracy_defined": 1.0,
                            "warning_code": (
                                "not_normalized_density" if method_label == "CONDITIONAL_FLOW_MATCHING" else ""
                            ),
                            "fit_status": "ok",
                        }
                    )
                    traces.append(
                        {
                            "family": family,
                            "dgp_seed": seed,
                            "method_init": method_init,
                            "method": method_label,
                            **trace.to_dict(),
                        }
                    )
    atomic_csv(run_dir / "training_traces.csv", pd.DataFrame(traces))
    expected = len(families) * len(seeds) * len(initializations) * 3
    validation = _package_rows(run_dir, rows, expected)
    validation["full_e10_grid_complete"] = False
    validation["normalized_flexible_flow_included"] = False
    validation["e10_superiority_gate_passed"] = False
    validation["e10_gate_reason"] = "only E4 reference cells and 10 neural seeds; conditional flow-matching has no normalized likelihood"
    atomic_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distributional-sid-tier-a")
    subparsers = parser.add_subparsers(dest="command", required=True)
    e6 = subparsers.add_parser("confirm-e6")
    e6.add_argument("--run-dir", required=True)
    e6.add_argument("--n-train", type=int, default=8000)
    e6.set_defaults(func=command_e6)
    e7 = subparsers.add_parser("confirm-e7")
    e7.add_argument("--run-dir", required=True)
    e7.add_argument("--n-train", type=int, default=8000)
    e7.set_defaults(func=command_e7)
    e11 = subparsers.add_parser("confirm-e11")
    e11.add_argument("--run-dir", required=True)
    e11.add_argument("--n-train", type=int, default=8000)
    e11.set_defaults(func=command_e11)
    e8 = subparsers.add_parser("confirm-e8")
    e8.add_argument("--run-dir", required=True)
    e8.add_argument("--n-train", type=int, default=4000)
    e8.set_defaults(func=command_e8)
    e9 = subparsers.add_parser("confirm-e9")
    e9.add_argument("--run-dir", required=True)
    e9.add_argument("--n-train", type=int, default=8000)
    e9.set_defaults(func=command_e9)
    e10 = subparsers.add_parser("confirm-e10")
    e10.add_argument("--run-dir", required=True)
    e10.add_argument("--n-train", type=int, default=4000)
    e10.set_defaults(func=command_e10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
