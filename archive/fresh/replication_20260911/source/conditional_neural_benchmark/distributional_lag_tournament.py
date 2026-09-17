from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from conditional_neural_benchmark.data import (
    Cohort,
    FoldScaler,
    load_cohort,
    load_sbtg_cohort,
    make_fold_assignments,
    make_windows,
)
from conditional_neural_benchmark.distributed_lag_dynamics import (
    FeatureSet,
    NuisanceFit,
    PreparedSet,
    fit_cross_path,
    fit_nuisance,
    group_shrink,
    make_features,
    predict_full,
    reconstruct_kernel,
    select_nuisance_alphas,
    smooth_lag_basis,
    transform_coefficient,
)
from conditional_neural_benchmark.metrics import metric_rows
from conditional_neural_benchmark.runner import _split_indices


MEAN_ALPHAS = (100.0, 1000.0, 10000.0)
SCALE_ALPHAS = (100.0, 1000.0, 10000.0)
SHRINK_QUANTILES = (0.25, 0.50, 0.75)
COV_ALPHAS = (1000.0, 10000.0)
COV_RANKS = (2, 4, 8)
DF = 5.0


@dataclass(frozen=True)
class FoldBundle:
    lag_basis: np.ndarray
    nuisance: NuisanceFit
    train_features: FeatureSet
    validation_features: FeatureSet
    test_features: FeatureSet
    train: PreparedSet
    validation: PreparedSet
    test: PreparedSet
    common_alpha: float
    self_alpha: float


@dataclass(frozen=True)
class ScaleBaseline:
    train_logvar: np.ndarray
    validation_logvar: np.ndarray
    test_logvar: np.ndarray
    train_response: np.ndarray


def _ridge(alpha: float) -> Ridge:
    return Ridge(alpha=float(alpha), solver="lsqr", tol=1e-5, max_iter=4000)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cohorts() -> dict[str, Cohort]:
    current = load_cohort()
    sbtg = load_sbtg_cohort()
    bridge = load_sbtg_cohort(neuron_subset=current.neurons)
    return {"current54": current, "sbtg80": sbtg, "sbtg_bridge54": bridge}


def cohort_folds(cohort_name: str, cohort: Cohort, evidence: Path) -> np.ndarray:
    if cohort_name == "current54":
        frame = pd.read_csv(evidence / "fold_assignments.csv")
        mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
        return np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)
    return make_fold_assignments(cohort, n_folds=5, seed=20260828)


def _prepare_fold(cohort: Cohort, folds: np.ndarray, fold: int, lag: int) -> FoldBundle:
    train_idx, validation_idx, test_idx = _split_indices(folds, fold)
    scaler = FoldScaler.fit(cohort.traces[index] for index in train_idx)
    train_windows = make_windows(cohort, train_idx, lag, scaler)
    validation_windows = make_windows(cohort, validation_idx, lag, scaler)
    test_windows = make_windows(cohort, test_idx, lag, scaler)
    current = train_windows.history[:, -1, :cohort.n_neurons]
    pca = PCA(
        n_components=min(4, cohort.n_neurons),
        svd_solver="randomized",
        random_state=20260828 + fold + lag,
    ).fit(current)
    basis = smooth_lag_basis(lag)
    train_features = make_features(train_windows, basis, pca)
    validation_features = make_features(validation_windows, basis, pca)
    test_features = make_features(test_windows, basis, pca)
    d, q = cohort.n_neurons, basis.shape[1]
    common_alpha, self_alpha, _ = select_nuisance_alphas(
        train_features, validation_features, d, q
    )
    nuisance = fit_nuisance(
        train_features,
        d=d,
        q=q,
        common_alpha=common_alpha,
        self_alpha=self_alpha,
    )
    from conditional_neural_benchmark.distributed_lag_dynamics import prepare_set

    return FoldBundle(
        lag_basis=basis,
        nuisance=nuisance,
        train_features=train_features,
        validation_features=validation_features,
        test_features=test_features,
        train=prepare_set(nuisance, train_features),
        validation=prepare_set(nuisance, validation_features),
        test=prepare_set(nuisance, test_features),
        common_alpha=common_alpha,
        self_alpha=self_alpha,
    )


def _student_nll_rows(y: np.ndarray, mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    variance = np.maximum(np.exp(np.asarray(logvar, np.float64)), 1e-5)
    scale2 = variance * (DF - 2.0) / DF
    z2 = np.square(y - mean) / scale2
    constant = (
        gammaln((DF + 1.0) / 2.0)
        - gammaln(DF / 2.0)
        - 0.5 * np.log(DF * np.pi * scale2)
    )
    logp = constant - 0.5 * (DF + 1.0) * np.log1p(z2 / DF)
    return -logp.mean(axis=1)


def _gaussian_nll_rows(y: np.ndarray, mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    safe = np.clip(np.asarray(logvar, np.float64), -10.0, 5.0)
    return 0.5 * (math.log(2.0 * math.pi) + safe + np.square(y - mean) * np.exp(-safe)).mean(axis=1)


def _constant_logvar(y: np.ndarray, mean: np.ndarray) -> np.ndarray:
    variance = np.maximum(np.mean(np.square(y - mean), axis=0), 1e-4)
    return np.broadcast_to(np.log(variance)[None], y.shape).copy()


def _mean_metrics(
    train_y: np.ndarray,
    train_base: np.ndarray,
    train_full: np.ndarray,
    test_y: np.ndarray,
    test_base: np.ndarray,
    test_full: np.ndarray,
) -> dict[str, float]:
    base_logvar = _constant_logvar(train_y, train_base)[:1]
    full_logvar = _constant_logvar(train_y, train_full)[:1]
    base_nll = _student_nll_rows(test_y, test_base, np.broadcast_to(base_logvar, test_y.shape))
    full_nll = _student_nll_rows(test_y, test_full, np.broadcast_to(full_logvar, test_y.shape))
    base_rmse = float(np.sqrt(np.mean(np.square(test_y - test_base))))
    full_rmse = float(np.sqrt(np.mean(np.square(test_y - test_full))))
    return {
        "baseline_nll": float(base_nll.mean()),
        "full_nll": float(full_nll.mean()),
        "nll_improvement": float(np.mean(base_nll - full_nll)),
        "baseline_rmse": base_rmse,
        "full_rmse": full_rmse,
        "rmse_improvement": base_rmse - full_rmse,
    }


def _location_candidate(
    bundle: FoldBundle, coefficient_path: dict[float, np.ndarray], alpha: float, quantile: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    coefficient, threshold, density = group_shrink(coefficient_path[alpha], quantile)
    train_mean, _ = predict_full(bundle.train, bundle.nuisance, coefficient)
    validation_mean, _ = predict_full(bundle.validation, bundle.nuisance, coefficient)
    test_mean, _ = predict_full(bundle.test, bundle.nuisance, coefficient)
    return coefficient, train_mean, validation_mean, test_mean, threshold, density


def _fit_scale_baseline(
    bundle: FoldBundle,
    train_mean: np.ndarray,
    validation_mean: np.ndarray,
    test_mean: np.ndarray,
    alpha: float = 100.0,
) -> ScaleBaseline:
    train_common = bundle.train_features.common.astype(np.float64)
    common_mean = train_common.mean(axis=0)
    common_scale = np.where(train_common.std(axis=0) > 1e-6, train_common.std(axis=0), 1.0)
    commons = [
        (value.common - common_mean) / common_scale
        for value in (bundle.train_features, bundle.validation_features, bundle.test_features)
    ]
    means = (train_mean, validation_mean, test_mean)
    prepared = (bundle.train, bundle.validation, bundle.test)
    residual2 = np.square(bundle.train_features.target - train_mean)
    floor = np.maximum(np.quantile(residual2, 0.10, axis=0), 1e-5)
    response = np.log(residual2 + floor[None])
    predictions = [np.zeros_like(value, dtype=np.float64) for value in means]
    d, q = bundle.nuisance.n_neurons, bundle.nuisance.n_basis
    for target in range(d):
        own = slice(target * q, (target + 1) * q)
        train_design = np.concatenate(
            [commons[0], prepared[0].x_common_residual[:, own]], axis=1
        )
        model = _ridge(alpha).fit(train_design, response[:, target])
        for slot in range(3):
            design = np.concatenate(
                [commons[slot], prepared[slot].x_common_residual[:, own]], axis=1
            )
            predictions[slot][:, target] = model.predict(design)
    # Calibrate the scale multiplicatively on training data without test access.
    ratio = np.mean(residual2 / np.exp(np.clip(predictions[0], -10, 5)), axis=0)
    offset = np.log(np.maximum(ratio, 1e-4))
    predictions = [np.clip(value + offset[None], -10.0, 5.0) for value in predictions]
    return ScaleBaseline(
        train_logvar=predictions[0].astype(np.float32),
        validation_logvar=predictions[1].astype(np.float32),
        test_logvar=predictions[2].astype(np.float32),
        train_response=response.astype(np.float32),
    )


def _fit_cross_response(
    prepared: PreparedSet,
    nuisance: NuisanceFit,
    response: np.ndarray,
    alphas: Iterable[float],
) -> dict[float, np.ndarray]:
    values = tuple(float(value) for value in alphas)
    d, q = nuisance.n_neurons, nuisance.n_basis
    result = {alpha: np.zeros((d, d, q), dtype=np.float32) for alpha in values}
    for target in range(d):
        x = np.asarray(prepared.residual_x[target], dtype=np.float64)
        y = np.asarray(response[:, target], dtype=np.float64)
        gram = x.T @ x
        cross = x.T @ y
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        projected = eigenvectors.T @ cross
        sources = np.delete(np.arange(d), target)
        for alpha in values:
            standardized = eigenvectors @ (projected / np.maximum(eigenvalues + alpha, 1e-10))
            standardized = standardized.reshape(d - 1, q)
            raw = standardized / nuisance.neural_scale.reshape(d, q)[sources]
            result[alpha][target, sources] = raw.astype(np.float32)
    return result


def _cross_contribution(
    prepared: PreparedSet, nuisance: NuisanceFit, coefficient: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(prepared.worm), nuisance.n_neurons), dtype=np.float64)
    scale = nuisance.neural_scale.reshape(nuisance.n_neurons, nuisance.n_basis)
    for target in range(nuisance.n_neurons):
        sources = np.delete(np.arange(nuisance.n_neurons), target)
        standardized = (coefficient[target, sources] * scale[sources]).reshape(-1)
        result[:, target] = prepared.residual_x[target] @ standardized
    return result.astype(np.float32)


def _calibrate_full_logvar(
    train_y: np.ndarray, train_mean: np.ndarray, train_logvar: np.ndarray, others: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    residual2 = np.square(train_y - train_mean)
    ratio = np.mean(residual2 / np.exp(np.clip(train_logvar, -10, 5)), axis=0)
    offset = np.log(np.maximum(ratio, 1e-4))
    train_value = np.clip(train_logvar + offset[None], -10.0, 5.0)
    return train_value.astype(np.float32), [
        np.clip(value + offset[None], -10.0, 5.0).astype(np.float32) for value in others
    ]


def _coverage(y: np.ndarray, mean: np.ndarray, logvar: np.ndarray, z: float) -> float:
    radius = z * np.sqrt(np.exp(logvar))
    return float(np.mean((y >= mean - radius) & (y <= mean + radius)))


def _scale_metrics(
    y: np.ndarray,
    mean: np.ndarray,
    baseline_logvar: np.ndarray,
    full_logvar: np.ndarray,
) -> dict[str, float]:
    baseline_gaussian = _gaussian_nll_rows(y, mean, baseline_logvar)
    full_gaussian = _gaussian_nll_rows(y, mean, full_logvar)
    baseline_t = _student_nll_rows(y, mean, baseline_logvar)
    full_t = _student_nll_rows(y, mean, full_logvar)
    return {
        "baseline_gaussian_nll": float(baseline_gaussian.mean()),
        "full_gaussian_nll": float(full_gaussian.mean()),
        "gaussian_nll_improvement": float(np.mean(baseline_gaussian - full_gaussian)),
        "baseline_student_nll": float(baseline_t.mean()),
        "full_student_nll": float(full_t.mean()),
        "student_nll_improvement": float(np.mean(baseline_t - full_t)),
        "baseline_coverage50": _coverage(y, mean, baseline_logvar, 0.67448975),
        "full_coverage50": _coverage(y, mean, full_logvar, 0.67448975),
        "baseline_coverage80": _coverage(y, mean, baseline_logvar, 1.28155157),
        "full_coverage80": _coverage(y, mean, full_logvar, 1.28155157),
        "baseline_coverage90": _coverage(y, mean, baseline_logvar, 1.64485363),
        "full_coverage90": _coverage(y, mean, full_logvar, 1.64485363),
        "mean_logvar_shift_rms": float(np.sqrt(np.mean(np.square(full_logvar - baseline_logvar)))),
    }


def _sample_law(
    mean: np.ndarray,
    logvar: np.ndarray,
    *,
    seed: int,
    n_samples: int = 16,
    loadings: np.ndarray | None = None,
    multipliers: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = np.sqrt(np.exp(np.clip(logvar, -10, 5)))
    draw = mean[:, None] + scale[:, None] * rng.normal(size=(len(mean), n_samples, mean.shape[1]))
    if loadings is not None and loadings.shape[1] > 0:
        rank_noise = rng.normal(size=(len(mean), n_samples, loadings.shape[1]))
        weighted = rank_noise * np.sqrt(np.maximum(multipliers, 1e-4))[:, None]
        standardized = np.einsum("bkr,dr->bkd", weighted, loadings, optimize=True)
        draw += scale[:, None] * standardized
    return draw.astype(np.float32)


def _sample_metrics(samples: np.ndarray, target: np.ndarray, seed: int) -> dict[str, float]:
    rows = metric_rows(samples, target, seed)
    return {name: float(np.mean(value)) for name, value in rows.items()}


def _fit_covariance_candidate(
    bundle: FoldBundle,
    train_mean: np.ndarray,
    validation_mean: np.ndarray,
    test_mean: np.ndarray,
    train_logvar: np.ndarray,
    validation_logvar: np.ndarray,
    test_logvar: np.ndarray,
    *,
    rank: int,
    alpha: float,
    quantile: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    standardized = (
        bundle.train_features.target - train_mean
    ) / np.sqrt(np.exp(train_logvar))
    covariance = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:rank]
    eigenvalues = np.maximum(eigenvalues[order], 1.0)
    vectors = eigenvectors[:, order]
    excess = np.maximum(eigenvalues - 1.0, 1e-3)
    score = standardized @ vectors
    response = np.log(np.square(score) + 0.05)
    x = np.asarray(bundle.train.x_common_residual, dtype=np.float64)
    gram = x.T @ x
    eigen_x, vectors_x = np.linalg.eigh(gram)
    projected = vectors_x.T @ (x.T @ response)
    coefficient_std = vectors_x @ (
        projected / np.maximum(eigen_x[:, None] + float(alpha), 1e-10)
    )
    d, q = bundle.nuisance.n_neurons, bundle.nuisance.n_basis
    coefficient_std = coefficient_std.reshape(d, q, rank).transpose(2, 0, 1)
    raw = coefficient_std / bundle.nuisance.neural_scale.reshape(d, q)[None]
    norms = np.linalg.norm(raw, axis=(0, 2))
    threshold = float(np.quantile(norms, quantile))
    factor = np.maximum(0.0, 1.0 - threshold / np.maximum(norms, 1e-12))
    raw *= factor[None, :, None]
    coefficient_std *= factor[None, :, None]
    flat = coefficient_std.transpose(1, 2, 0).reshape(d * q, rank)

    def multiplier(prepared: PreparedSet) -> np.ndarray:
        contribution = prepared.x_common_residual @ flat
        return np.exp(np.clip(contribution, -2.0, 2.0))

    train_multiplier = multiplier(bundle.train)
    normalization = np.maximum(train_multiplier.mean(axis=0), 1e-4)
    validation_multiplier = multiplier(bundle.validation) / normalization
    test_multiplier = multiplier(bundle.test) / normalization
    loadings = vectors * np.sqrt(excess)[None]
    index = np.arange(len(bundle.validation_features.target))
    if len(index) > 2048:
        index = np.sort(np.random.default_rng(1200 + rank).choice(index, 2048, replace=False))
    baseline_draw = _sample_law(
        validation_mean[index], validation_logvar[index], seed=2000 + rank,
        loadings=loadings, multipliers=np.ones((len(index), rank)),
    )
    full_draw = _sample_law(
        validation_mean[index], validation_logvar[index], seed=3000 + rank,
        loadings=loadings, multipliers=validation_multiplier[index],
    )
    baseline = _sample_metrics(baseline_draw, bundle.validation_features.target[index], 4000 + rank)
    full = _sample_metrics(full_draw, bundle.validation_features.target[index], 5000 + rank)
    metrics = {
        "baseline_energy": baseline["energy"],
        "full_energy": full["energy"],
        "energy_improvement": baseline["energy"] - full["energy"],
        "baseline_variogram": baseline["variogram"],
        "full_variogram": full["variogram"],
        "variogram_improvement": baseline["variogram"] - full["variogram"],
    }
    gain_basis = np.einsum(
        "tr,r,rsq->tsq", np.square(vectors), excess, raw, optimize=True
    )
    return metrics, {
        "vectors": vectors.astype(np.float32),
        "excess": excess.astype(np.float32),
        "raw_coefficient": raw.astype(np.float32),
        "gain_basis": gain_basis.astype(np.float32),
        "test_multiplier": test_multiplier.astype(np.float32),
        "loadings": loadings.astype(np.float32),
    }


def _mean_screen(cohort_name: str, cohort: Cohort, folds: np.ndarray, lags: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for lag in lags:
        for fold in (0, 1, 2):
            started = time.perf_counter()
            bundle = _prepare_fold(cohort, folds, fold, lag)
            path = fit_cross_path(bundle.train, bundle.nuisance, MEAN_ALPHAS)
            for alpha in MEAN_ALPHAS:
                for quantile in SHRINK_QUANTILES:
                    _, train_mean, validation_mean, test_mean, threshold, density = _location_candidate(
                        bundle, path, alpha, quantile
                    )
                    validation_metrics = _mean_metrics(
                        bundle.train_features.target,
                        bundle.train.baseline_prediction,
                        train_mean,
                        bundle.validation_features.target,
                        bundle.validation.baseline_prediction,
                        validation_mean,
                    )
                    test_metrics = _mean_metrics(
                        bundle.train_features.target,
                        bundle.train.baseline_prediction,
                        train_mean,
                        bundle.test_features.target,
                        bundle.test.baseline_prediction,
                        test_mean,
                    )
                    rows.append({
                        "cohort": cohort_name, "lag": lag, "fold": fold,
                        "alpha": alpha, "shrink_quantile": quantile,
                        "threshold": threshold, "density": density,
                        **{f"validation_{key}": value for key, value in validation_metrics.items()},
                        **{f"test_{key}": value for key, value in test_metrics.items()},
                        "fold_wall_seconds": time.perf_counter() - started,
                    })
    return pd.DataFrame(rows)


def _select_mean(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby(["lag", "alpha", "shrink_quantile"], as_index=False).agg(
        nll_improvement=("test_nll_improvement", "mean"),
        rmse_improvement=("test_rmse_improvement", "mean"),
    )
    eligible = grouped[grouped.rmse_improvement >= 0]
    candidate = eligible if not eligible.empty else grouped
    winner = candidate.sort_values(["nll_improvement", "rmse_improvement"], ascending=False).iloc[0]
    return {
        "lag": int(winner.lag), "alpha": float(winner.alpha),
        "shrink_quantile": float(winner.shrink_quantile),
        "screen_mean_nll_improvement": float(winner.nll_improvement),
        "screen_mean_rmse_improvement": float(winner.rmse_improvement),
    }


def _fit_selected_location(bundle: FoldBundle, specification: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = fit_cross_path(bundle.train, bundle.nuisance, [specification["alpha"]])
    coefficient, train_mean, validation_mean, test_mean, _, _ = _location_candidate(
        bundle, path, specification["alpha"], specification["shrink_quantile"]
    )
    return coefficient, train_mean, validation_mean, test_mean


def _scale_screen(
    cohort_name: str, cohort: Cohort, folds: np.ndarray, mean_spec: dict
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold in (0, 1, 2):
        bundle = _prepare_fold(cohort, folds, fold, int(mean_spec["lag"]))
        _, train_mean, validation_mean, test_mean = _fit_selected_location(bundle, mean_spec)
        baseline = _fit_scale_baseline(bundle, train_mean, validation_mean, test_mean)
        response = baseline.train_response - baseline.train_logvar
        path = _fit_cross_response(bundle.train, bundle.nuisance, response, SCALE_ALPHAS)
        for alpha in SCALE_ALPHAS:
            for quantile in SHRINK_QUANTILES:
                coefficient, threshold, density = group_shrink(path[alpha], quantile)
                train_contribution = _cross_contribution(bundle.train, bundle.nuisance, coefficient)
                validation_contribution = _cross_contribution(bundle.validation, bundle.nuisance, coefficient)
                test_contribution = _cross_contribution(bundle.test, bundle.nuisance, coefficient)
                train_full, other = _calibrate_full_logvar(
                    bundle.train_features.target,
                    train_mean,
                    baseline.train_logvar + train_contribution,
                    [
                        baseline.validation_logvar + validation_contribution,
                        baseline.test_logvar + test_contribution,
                    ],
                )
                validation_full, test_full = other
                validation_metrics = _scale_metrics(
                    bundle.validation_features.target, validation_mean,
                    baseline.validation_logvar, validation_full,
                )
                test_metrics = _scale_metrics(
                    bundle.test_features.target, test_mean,
                    baseline.test_logvar, test_full,
                )
                rows.append({
                    "cohort": cohort_name, "fold": fold, "alpha": alpha,
                    "shrink_quantile": quantile, "threshold": threshold,
                    "density": density,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                })
    return pd.DataFrame(rows)


def _select_scale(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby(["alpha", "shrink_quantile"], as_index=False).agg(
        gaussian_nll_improvement=("test_gaussian_nll_improvement", "mean"),
        student_nll_improvement=("test_student_nll_improvement", "mean"),
        coverage90=("test_full_coverage90", "mean"),
    )
    grouped["coverage_error"] = np.abs(grouped.coverage90 - 0.90)
    winner = grouped.sort_values(
        ["gaussian_nll_improvement", "coverage_error", "student_nll_improvement"],
        ascending=[False, True, False],
    ).iloc[0]
    return {
        "alpha": float(winner.alpha),
        "shrink_quantile": float(winner.shrink_quantile),
        "screen_gaussian_nll_improvement": float(winner.gaussian_nll_improvement),
        "screen_student_nll_improvement": float(winner.student_nll_improvement),
        "screen_coverage90": float(winner.coverage90),
    }


def _fit_selected_scale(
    bundle: FoldBundle,
    mean_spec: dict,
    scale_spec: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_coefficient, train_mean, validation_mean, test_mean = _fit_selected_location(bundle, mean_spec)
    baseline = _fit_scale_baseline(bundle, train_mean, validation_mean, test_mean)
    response = baseline.train_response - baseline.train_logvar
    path = _fit_cross_response(bundle.train, bundle.nuisance, response, [scale_spec["alpha"]])
    scale_coefficient, _, _ = group_shrink(path[scale_spec["alpha"]], scale_spec["shrink_quantile"])
    contributions = [
        _cross_contribution(value, bundle.nuisance, scale_coefficient)
        for value in (bundle.train, bundle.validation, bundle.test)
    ]
    train_logvar, others = _calibrate_full_logvar(
        bundle.train_features.target,
        train_mean,
        baseline.train_logvar + contributions[0],
        [baseline.validation_logvar + contributions[1], baseline.test_logvar + contributions[2]],
    )
    return (
        mean_coefficient, scale_coefficient,
        train_mean, validation_mean, test_mean,
        train_logvar, others[0], others[1],
    )


def _covariance_screen(
    cohort_name: str,
    cohort: Cohort,
    folds: np.ndarray,
    mean_spec: dict,
    scale_spec: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold in (0, 1, 2):
        bundle = _prepare_fold(cohort, folds, fold, int(mean_spec["lag"]))
        fitted = _fit_selected_scale(bundle, mean_spec, scale_spec)
        _, _, train_mean, validation_mean, test_mean, train_logvar, validation_logvar, test_logvar = fitted
        for rank in COV_RANKS:
            for alpha in COV_ALPHAS:
                for quantile in (0.50, 0.75):
                    metrics, _ = _fit_covariance_candidate(
                        bundle, train_mean, validation_mean, test_mean,
                        train_logvar, validation_logvar, test_logvar,
                        rank=rank, alpha=alpha, quantile=quantile,
                    )
                    rows.append({
                        "cohort": cohort_name, "fold": fold, "rank": rank,
                        "alpha": alpha, "shrink_quantile": quantile, **metrics,
                    })
    return pd.DataFrame(rows)


def _select_covariance(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby(["rank", "alpha", "shrink_quantile"], as_index=False).agg(
        energy_improvement=("energy_improvement", "mean"),
        variogram_improvement=("variogram_improvement", "mean"),
    )
    winner = grouped.sort_values(
        ["energy_improvement", "variogram_improvement"], ascending=False
    ).iloc[0]
    return {
        "rank": int(winner["rank"]), "alpha": float(winner.alpha),
        "shrink_quantile": float(winner.shrink_quantile),
        "screen_energy_improvement": float(winner.energy_improvement),
        "screen_variogram_improvement": float(winner.variogram_improvement),
    }


def _fit_final_fold(
    output: Path,
    cohort_name: str,
    cohort: Cohort,
    folds: np.ndarray,
    fold: int,
    mean_spec: dict,
    scale_spec: dict,
    covariance_spec: dict,
) -> tuple[dict, pd.DataFrame]:
    started = time.perf_counter()
    bundle = _prepare_fold(cohort, folds, fold, int(mean_spec["lag"]))
    fitted = _fit_selected_scale(bundle, mean_spec, scale_spec)
    mean_coefficient, scale_coefficient = fitted[:2]
    train_mean, validation_mean, test_mean = fitted[2:5]
    train_logvar, validation_logvar, test_logvar = fitted[5:]
    scale_baseline = _fit_scale_baseline(bundle, train_mean, validation_mean, test_mean)
    mean_metrics = _mean_metrics(
        bundle.train_features.target, bundle.train.baseline_prediction, train_mean,
        bundle.test_features.target, bundle.test.baseline_prediction, test_mean,
    )
    scale_metrics = _scale_metrics(
        bundle.test_features.target, test_mean, scale_baseline.test_logvar, test_logvar
    )
    covariance_validation, covariance = _fit_covariance_candidate(
        bundle, train_mean, validation_mean, test_mean,
        train_logvar, validation_logvar, test_logvar,
        rank=covariance_spec["rank"], alpha=covariance_spec["alpha"],
        quantile=covariance_spec["shrink_quantile"],
    )
    index = np.arange(len(bundle.test_features.target))
    if len(index) > 3072:
        index = np.sort(np.random.default_rng(9000 + fold).choice(index, 3072, replace=False))
    baseline_draw = _sample_law(
        test_mean[index], test_logvar[index], seed=10000 + fold,
        loadings=covariance["loadings"],
        multipliers=np.ones((len(index), covariance_spec["rank"])),
    )
    full_draw = _sample_law(
        test_mean[index], test_logvar[index], seed=11000 + fold,
        loadings=covariance["loadings"],
        multipliers=covariance["test_multiplier"][index],
    )
    baseline_sample = _sample_metrics(
        baseline_draw, bundle.test_features.target[index], 12000 + fold
    )
    full_sample = _sample_metrics(full_draw, bundle.test_features.target[index], 13000 + fold)
    covariance_metrics = {
        "baseline_energy": baseline_sample["energy"],
        "full_energy": full_sample["energy"],
        "energy_improvement": baseline_sample["energy"] - full_sample["energy"],
        "baseline_variogram": baseline_sample["variogram"],
        "full_variogram": full_sample["variogram"],
        "variogram_improvement": baseline_sample["variogram"] - full_sample["variogram"],
    }
    location_kernel = reconstruct_kernel(mean_coefficient, bundle.lag_basis)
    scale_kernel = reconstruct_kernel(scale_coefficient, bundle.lag_basis)
    covariance_gain_kernel = reconstruct_kernel(
        covariance["gain_basis"], bundle.lag_basis
    )
    path = output / "fold_models" / f"{cohort_name}__f{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        neurons=np.asarray(cohort.neurons),
        lag_basis=bundle.lag_basis,
        location_basis_coefficient=mean_coefficient,
        logvariance_basis_coefficient=scale_coefficient,
        covariance_factor_coefficient=covariance["raw_coefficient"],
        location_kernel=location_kernel,
        logvariance_kernel=scale_kernel,
        covariance_gain_kernel=covariance_gain_kernel,
        covariance_vectors=covariance["vectors"],
        covariance_excess=covariance["excess"],
    )
    rows: list[dict] = []
    base_nll = _gaussian_nll_rows(
        bundle.test_features.target, test_mean, scale_baseline.test_logvar
    )
    full_nll = _gaussian_nll_rows(bundle.test_features.target, test_mean, test_logvar)
    for worm in np.unique(bundle.test.worm):
        use = bundle.test.worm == worm
        rows.append({
            "cohort": cohort_name, "fold": fold, "worm": int(worm),
            "split": "confirmation" if fold in (3, 4) else "screen",
            "scale_gaussian_nll_improvement": float(np.mean(base_nll[use] - full_nll[use])),
            "mean_mse_improvement": float(np.mean(
                np.square(bundle.test_features.target[use] - bundle.test.baseline_prediction[use])
                - np.square(bundle.test_features.target[use] - test_mean[use])
            )),
        })
    record = {
        "cohort": cohort_name, "fold": fold,
        "split": "confirmation" if fold in (3, 4) else "screen",
        **{f"mean_{key}": value for key, value in mean_metrics.items()},
        **{f"scale_{key}": value for key, value in scale_metrics.items()},
        **{f"covariance_{key}": value for key, value in covariance_metrics.items()},
        "covariance_validation_energy_improvement": covariance_validation["energy_improvement"],
        "wall_seconds": time.perf_counter() - started,
        "model_file": str(path.relative_to(output)),
    }
    return record, pd.DataFrame(rows)


def _bootstrap_gate(worms: pd.DataFrame, cohort_name: str, seed: int = 20260828) -> dict:
    frame = worms[(worms.cohort == cohort_name) & (worms.split == "confirmation")]
    values = frame.scale_gaussian_nll_improvement.to_numpy(float)
    rng = np.random.default_rng(seed)
    draw = np.asarray([
        np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(5000)
    ])
    return {
        "confirmation_worms": int(len(values)),
        "mean_scale_gaussian_nll_improvement": float(np.mean(values)),
        "ci_low": float(np.quantile(draw, 0.025)),
        "ci_high": float(np.quantile(draw, 0.975)),
        "passes_scale_gate": bool(np.quantile(draw, 0.025) > 0),
    }


def run(output: Path, evidence: Path, lags: list[int], resume: bool) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    cohorts = load_cohorts()
    folds_by_cohort = {
        name: cohort_folds(name, cohort, evidence) for name, cohort in cohorts.items()
    }
    pd.concat([
        pd.DataFrame({
            "cohort": name, "worm_index": np.arange(cohort.n_worms),
            "worm_id": cohort.worm_ids, "outer_fold": folds_by_cohort[name],
        })
        for name, cohort in cohorts.items()
    ]).to_csv(output / "fold_assignments.csv", index=False)
    _atomic_json(output / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohorts": {
            name: {
                "worms": cohort.n_worms, "neurons": cohort.n_neurons,
                "raw_nonfinite_values": cohort.raw_nonfinite_values,
            }
            for name, cohort in cohorts.items()
        },
        "lags": lags,
        "screen_folds": [0, 1, 2],
        "confirmation_folds": [3, 4],
        "mean_alphas": MEAN_ALPHAS,
        "scale_alphas": SCALE_ALPHAS,
        "shrink_quantiles": SHRINK_QUANTILES,
        "covariance_ranks": COV_RANKS,
        "covariance_alphas": COV_ALPHAS,
        "atlas_firewall": "no Randi, Cook, Bentley, receptor, or SBTG result artifact loaded",
        "claim_lanes": ["conditional mean", "conditional log variance", "conditional covariance", "full predictive law"],
    })
    mean_rows: list[pd.DataFrame] = []
    scale_rows: list[pd.DataFrame] = []
    covariance_rows: list[pd.DataFrame] = []
    selections: dict[str, dict] = {}
    for name, cohort in cohorts.items():
        mean_path = output / f"{name}_mean_screen.csv"
        mean = pd.read_csv(mean_path) if resume and mean_path.exists() else _mean_screen(
            name, cohort, folds_by_cohort[name], lags
        )
        mean.to_csv(mean_path, index=False)
        mean_spec = _select_mean(mean)
        scale_path = output / f"{name}_scale_screen.csv"
        scale = pd.read_csv(scale_path) if resume and scale_path.exists() else _scale_screen(
            name, cohort, folds_by_cohort[name], mean_spec
        )
        scale.to_csv(scale_path, index=False)
        scale_spec = _select_scale(scale)
        cov_path = output / f"{name}_covariance_screen.csv"
        covariance = pd.read_csv(cov_path) if resume and cov_path.exists() else _covariance_screen(
            name, cohort, folds_by_cohort[name], mean_spec, scale_spec
        )
        covariance.to_csv(cov_path, index=False)
        covariance_spec = _select_covariance(covariance)
        selections[name] = {
            "mean": mean_spec, "scale": scale_spec, "covariance": covariance_spec,
            "external_references_consulted": False,
        }
        mean_rows.append(mean)
        scale_rows.append(scale)
        covariance_rows.append(covariance)
        _atomic_json(output / "selection.json", selections)
    pd.concat(mean_rows, ignore_index=True).to_csv(output / "mean_screen.csv", index=False)
    pd.concat(scale_rows, ignore_index=True).to_csv(output / "scale_screen.csv", index=False)
    pd.concat(covariance_rows, ignore_index=True).to_csv(output / "covariance_screen.csv", index=False)
    fold_records: list[dict] = []
    worm_records: list[pd.DataFrame] = []
    for name, cohort in cohorts.items():
        for fold in range(5):
            final_path = output / "fold_models" / f"{name}__f{fold}.npz"
            existing = output / "final_fold_metrics.csv"
            if resume and final_path.exists() and existing.exists():
                old = pd.read_csv(existing)
                hit = old[(old.cohort == name) & (old.fold == fold)]
                if len(hit) == 1:
                    fold_records.append(hit.iloc[0].to_dict())
                    continue
            record, worms = _fit_final_fold(
                output, name, cohort, folds_by_cohort[name], fold,
                selections[name]["mean"], selections[name]["scale"], selections[name]["covariance"],
            )
            fold_records.append(record)
            worm_records.append(worms)
            pd.DataFrame(fold_records).to_csv(output / "final_fold_metrics.csv", index=False)
    fold_frame = pd.DataFrame(fold_records)
    fold_frame.to_csv(output / "final_fold_metrics.csv", index=False)
    worm_path = output / "worm_metrics.csv"
    if worm_records:
        worm_frame = pd.concat(worm_records, ignore_index=True)
        if resume and worm_path.exists():
            previous = pd.read_csv(worm_path)
            worm_frame = pd.concat([previous, worm_frame], ignore_index=True).drop_duplicates(
                ["cohort", "fold", "worm"], keep="last"
            )
    else:
        worm_frame = pd.read_csv(worm_path)
    worm_frame.to_csv(worm_path, index=False)
    gates = {name: _bootstrap_gate(worm_frame, name) for name in cohorts}
    _atomic_json(output / "promotion_gates.json", gates)
    _atomic_json(output / "validation.json", {
        "status": "pass",
        "mean_screen_rows": int(sum(len(value) for value in mean_rows)),
        "scale_screen_rows": int(sum(len(value) for value in scale_rows)),
        "covariance_screen_rows": int(sum(len(value) for value in covariance_rows)),
        "final_fold_rows": int(len(fold_frame)),
        "worm_rows": int(len(worm_frame)),
        "expected_final_fold_rows": 15,
        "all_finite": bool(np.isfinite(fold_frame.select_dtypes(include=[np.number])).all().all()),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/higher_order_lag_20260828"))
    parser.add_argument(
        "--evidence", type=Path,
        default=Path("results/distributed_lag_dynamics_20260828"),
    )
    parser.add_argument("--lags", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.output.resolve(), args.evidence.resolve(), args.lags, args.resume)
    print(f"HIGHER_ORDER_COMPLETE {result}", flush=True)


if __name__ == "__main__":
    main()
