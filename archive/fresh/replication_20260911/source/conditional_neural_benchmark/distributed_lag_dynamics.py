from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from conditional_neural_benchmark.data import (
    FoldScaler,
    Windows,
    load_cohort,
    make_windows,
    split_history,
)
from conditional_neural_benchmark.runner import _split_indices


COMMON_ALPHAS = (0.1, 1.0, 10.0, 100.0)
SELF_ALPHAS = (0.1, 1.0, 10.0, 100.0)
CROSS_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
SHRINK_QUANTILES = (0.50, 0.75, 0.90, 0.95)
RANKS = (4, 8, 16)
FAMILIES = ("residual_ridge", "group_shrunk", "reduced_rank", "sparse_lowrank")
DEFAULT_LAGS = (16, 32, 80)
STUDENT_DF = 5.0


@dataclass(frozen=True)
class FeatureSet:
    neural: np.ndarray
    common: np.ndarray
    target: np.ndarray
    worm: np.ndarray
    time: np.ndarray
    stratum: np.ndarray


@dataclass
class NuisanceFit:
    neural_mean: np.ndarray
    neural_scale: np.ndarray
    common_mean: np.ndarray
    common_scale: np.ndarray
    common_model: Ridge
    self_models: list[Ridge]
    off_indices: list[np.ndarray]
    n_neurons: int
    n_basis: int


@dataclass(frozen=True)
class PreparedSet:
    x: np.ndarray
    x_common_residual: np.ndarray
    y_common_residual: np.ndarray
    common_prediction: np.ndarray
    baseline_prediction: np.ndarray
    residual_x: tuple[np.ndarray, ...]
    residual_y: tuple[np.ndarray, ...]
    worm: np.ndarray
    time: np.ndarray
    stratum: np.ndarray


def _ridge(alpha: float) -> Ridge:
    return Ridge(
        alpha=float(alpha), fit_intercept=True, solver="lsqr", tol=1e-5, max_iter=4000
    )


def smooth_lag_basis(max_lag: int, n_basis: int | None = None) -> np.ndarray:
    """Return a smooth partition-of-unity basis indexed by lag 1..max_lag."""
    if max_lag < 2:
        raise ValueError("max_lag must be at least two")
    q = int(n_basis or min(10, max(5, round(2.4 * math.log2(max_lag)))))
    q = min(q, max_lag)
    lag_coordinate = np.log1p(np.arange(1, max_lag + 1, dtype=np.float64))
    centers = np.linspace(lag_coordinate[0], lag_coordinate[-1], q)
    spacing = float(centers[1] - centers[0]) if q > 1 else 1.0
    phase = (lag_coordinate[:, None] - centers[None]) / max(spacing, 1e-8)
    basis = np.zeros_like(phase)
    inside = np.abs(phase) <= 1.0
    basis[inside] = 0.5 + 0.5 * np.cos(np.pi * phase[inside])
    empty = basis.sum(axis=1) <= 1e-12
    if np.any(empty):
        nearest = np.argmin(np.abs(phase[empty]), axis=1)
        basis[empty, nearest] = 1.0
    basis /= basis.sum(axis=1, keepdims=True)
    return basis.astype(np.float32)


def reconstruct_kernel(basis_coef: np.ndarray, basis_lag: np.ndarray) -> np.ndarray:
    """Expand [target, source, basis] coefficients to [lag, target, source]."""
    if basis_coef.ndim != 3 or basis_coef.shape[2] != basis_lag.shape[1]:
        raise ValueError("coefficient and lag-basis shapes disagree")
    return np.einsum("tsq,lq->lts", basis_coef, basis_lag, optimize=True)


def impulse_response(kernel: np.ndarray, horizons: int) -> np.ndarray:
    """Recursive VAR impulse response with psi[0] equal to the identity."""
    if kernel.ndim != 3 or kernel.shape[1] != kernel.shape[2]:
        raise ValueError("kernel must have shape [lag, target, source]")
    d = kernel.shape[1]
    result = np.zeros((horizons + 1, d, d), dtype=np.float64)
    result[0] = np.eye(d)
    for horizon in range(1, horizons + 1):
        for lag in range(1, min(len(kernel), horizon) + 1):
            result[horizon] += kernel[lag - 1] @ result[horizon - lag]
    return result.astype(np.float32)


def _causal_fill(trace: np.ndarray) -> np.ndarray:
    result = np.asarray(trace, dtype=np.float32).copy()
    for column in range(result.shape[1]):
        finite = np.isfinite(result[:, column])
        last = np.maximum.accumulate(np.where(finite, np.arange(len(result)), -1))
        usable = last >= 0
        result[usable, column] = result[last[usable], column]
    return result


def _fit_pca(windows: Windows, components: int, seed: int) -> PCA:
    neural, _ = split_history(windows.history, windows.target.shape[1])
    current = neural[:, -1]
    usable = np.isfinite(current).all(axis=1)
    return PCA(
        n_components=min(int(components), current.shape[1]),
        svd_solver="randomized",
        random_state=int(seed),
    ).fit(current[usable])


def make_features(windows: Windows, basis_lag: np.ndarray, pca: PCA) -> FeatureSet:
    history, stimulus = split_history(windows.history, windows.target.shape[1])
    history = np.asarray(history, dtype=np.float32)
    stimulus = np.asarray(stimulus, dtype=np.float32)
    chronological_basis = basis_lag[::-1]
    projected = np.einsum(
        "nld,lq->ndq", history, chronological_basis, optimize=True
    )
    neural = projected.reshape(len(history), -1)
    stimulus_basis = np.einsum(
        "nlc,lq->ncq", stimulus, chronological_basis, optimize=True
    ).reshape(len(history), -1)
    global_history = history @ pca.components_.T.astype(np.float32)
    global_basis = np.einsum(
        "nlg,lq->ngq", global_history, chronological_basis, optimize=True
    ).reshape(len(history), -1)
    active_stimulus = np.max(stimulus, axis=2)
    recent = active_stimulus[:, -min(4, active_stimulus.shape[1]) :]
    onset_recent = np.any(np.diff(recent, axis=1) > 0.5, axis=1).astype(np.float32)
    offset_recent = np.any(np.diff(recent, axis=1) < -0.5, axis=1).astype(np.float32)
    common = np.concatenate(
        [
            stimulus_basis,
            global_basis,
            active_stimulus[:, -1, None],
            onset_recent[:, None],
            offset_recent[:, None],
        ],
        axis=1,
    )
    return FeatureSet(
        neural=neural.astype(np.float32),
        common=common.astype(np.float32),
        target=np.asarray(windows.target, dtype=np.float32),
        worm=np.asarray(windows.worm),
        time=np.asarray(windows.time),
        stratum=np.asarray(windows.stratum),
    )


def _standardize(
    train: np.ndarray, others: Iterable[np.ndarray]
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    scale = train.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)
    transformed_train = ((train - mean) / scale).astype(np.float32)
    transformed_others = tuple(((value - mean) / scale).astype(np.float32) for value in others)
    return transformed_train, transformed_others, mean.astype(np.float32), scale.astype(np.float32)


def _common_residuals(
    model: Ridge, x: np.ndarray, common: np.ndarray, y: np.ndarray, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = model.predict(common)
    common_y = prediction[:, :d]
    common_x = prediction[:, d:]
    return y - common_y, x - common_x, common_y


def select_nuisance_alphas(
    train: FeatureSet, validation: FeatureSet, d: int, q: int
) -> tuple[float, float, pd.DataFrame]:
    x_train, (x_validation,), _, _ = _standardize(train.neural, (validation.neural,))
    c_train, (c_validation,), _, _ = _standardize(train.common, (validation.common,))
    rows: list[dict] = []
    for common_alpha in COMMON_ALPHAS:
        common = _ridge(common_alpha).fit(
            c_train, np.concatenate([train.target, x_train], axis=1)
        )
        yc_train, xc_train, _ = _common_residuals(
            common, x_train, c_train, train.target, d
        )
        yc_validation, xc_validation, common_y_validation = _common_residuals(
            common, x_validation, c_validation, validation.target, d
        )
        for self_alpha in SELF_ALPHAS:
            prediction = common_y_validation.copy()
            for target in range(d):
                sl = slice(target * q, (target + 1) * q)
                fitted = _ridge(self_alpha).fit(xc_train[:, sl], yc_train[:, target])
                prediction[:, target] += fitted.predict(xc_validation[:, sl])
            rows.append(
                {
                    "common_alpha": common_alpha,
                    "self_alpha": self_alpha,
                    "validation_mse": float(np.square(prediction - validation.target).mean()),
                }
            )
    frame = pd.DataFrame(rows).sort_values("validation_mse").reset_index(drop=True)
    winner = frame.iloc[0]
    return float(winner.common_alpha), float(winner.self_alpha), frame


def fit_nuisance(
    train: FeatureSet,
    *,
    d: int,
    q: int,
    common_alpha: float,
    self_alpha: float,
) -> NuisanceFit:
    x, _, x_mean, x_scale = _standardize(train.neural, ())
    common, _, common_mean, common_scale = _standardize(train.common, ())
    common_model = _ridge(common_alpha).fit(
        common, np.concatenate([train.target, x], axis=1)
    )
    yc, xc, _ = _common_residuals(common_model, x, common, train.target, d)
    self_models: list[Ridge] = []
    off_indices: list[np.ndarray] = []
    all_indices = np.arange(d * q)
    for target in range(d):
        self_index = np.arange(target * q, (target + 1) * q)
        off = np.setdiff1d(all_indices, self_index, assume_unique=True)
        response = np.concatenate([yc[:, target, None], xc[:, off]], axis=1)
        self_models.append(_ridge(self_alpha).fit(xc[:, self_index], response))
        off_indices.append(off)
    return NuisanceFit(
        neural_mean=x_mean,
        neural_scale=x_scale,
        common_mean=common_mean,
        common_scale=common_scale,
        common_model=common_model,
        self_models=self_models,
        off_indices=off_indices,
        n_neurons=d,
        n_basis=q,
    )


def prepare_set(model: NuisanceFit, features: FeatureSet) -> PreparedSet:
    x = ((features.neural - model.neural_mean) / model.neural_scale).astype(np.float32)
    common = ((features.common - model.common_mean) / model.common_scale).astype(np.float32)
    yc, xc, common_y = _common_residuals(
        model.common_model, x, common, features.target, model.n_neurons
    )
    baseline = common_y.copy()
    residual_x: list[np.ndarray] = []
    residual_y: list[np.ndarray] = []
    for target, (self_model, off) in enumerate(zip(model.self_models, model.off_indices)):
        self_index = np.arange(target * model.n_basis, (target + 1) * model.n_basis)
        prediction = self_model.predict(xc[:, self_index])
        baseline[:, target] += prediction[:, 0]
        residual_y.append((yc[:, target] - prediction[:, 0]).astype(np.float32))
        residual_x.append((xc[:, off] - prediction[:, 1:]).astype(np.float32))
    return PreparedSet(
        x=x,
        x_common_residual=xc.astype(np.float32),
        y_common_residual=yc.astype(np.float32),
        common_prediction=common_y.astype(np.float32),
        baseline_prediction=baseline.astype(np.float32),
        residual_x=tuple(residual_x),
        residual_y=tuple(residual_y),
        worm=features.worm,
        time=features.time,
        stratum=features.stratum,
    )


def fit_cross_path(
    prepared: PreparedSet, nuisance: NuisanceFit, alphas: Iterable[float]
) -> dict[float, np.ndarray]:
    alphas = tuple(float(value) for value in alphas)
    d, q = nuisance.n_neurons, nuisance.n_basis
    result = {alpha: np.zeros((d, d, q), dtype=np.float32) for alpha in alphas}
    for target in range(d):
        x = np.asarray(prepared.residual_x[target], dtype=np.float64)
        y = np.asarray(prepared.residual_y[target], dtype=np.float64)
        gram = x.T @ x
        cross = x.T @ y
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        projected = eigenvectors.T @ cross
        off_sources = np.delete(np.arange(d), target)
        for alpha in alphas:
            coefficient_std = eigenvectors @ (projected / np.maximum(eigenvalues + alpha, 1e-10))
            coefficient_std = coefficient_std.reshape(d - 1, q)
            raw = coefficient_std / nuisance.neural_scale.reshape(d, q)[off_sources]
            result[alpha][target, off_sources] = raw.astype(np.float32)
    return result


def group_shrink(coefficient: np.ndarray, quantile: float) -> tuple[np.ndarray, float, float]:
    result = np.asarray(coefficient, dtype=np.float64).copy()
    d = result.shape[0]
    off = ~np.eye(d, dtype=bool)
    norms = np.linalg.norm(result, axis=2)
    threshold = float(np.quantile(norms[off], float(quantile)))
    factor = np.maximum(0.0, 1.0 - threshold / np.maximum(norms, 1e-12))
    result *= factor[..., None]
    result[np.arange(d), np.arange(d)] = 0.0
    density = float(np.mean(np.linalg.norm(result, axis=2)[off] > 0))
    return result.astype(np.float32), threshold, density


def rank_reduce(coefficient: np.ndarray, rank: int) -> np.ndarray:
    d, _, q = coefficient.shape
    matrix = np.asarray(coefficient, dtype=np.float64).reshape(d, d * q)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    r = min(int(rank), len(singular))
    reduced = ((u[:, :r] * singular[:r]) @ vt[:r]).reshape(d, d, q)
    reduced[np.arange(d), np.arange(d)] = 0.0
    return reduced.astype(np.float32)


def transform_coefficient(
    coefficient: np.ndarray, family: str, *, shrink_quantile: float | None, rank: int | None
) -> tuple[np.ndarray, dict]:
    if family == "residual_ridge":
        return coefficient.copy(), {}
    if family == "group_shrunk":
        transformed, threshold, density = group_shrink(coefficient, float(shrink_quantile))
        return transformed, {"threshold": threshold, "group_density": density}
    if family == "reduced_rank":
        return rank_reduce(coefficient, int(rank)), {"rank": int(rank)}
    if family == "sparse_lowrank":
        shrunk, threshold, density = group_shrink(coefficient, float(shrink_quantile))
        return rank_reduce(shrunk, int(rank)), {
            "rank": int(rank), "threshold": threshold, "group_density": density
        }
    raise ValueError(f"unknown family: {family}")


def predict_full(
    prepared: PreparedSet, nuisance: NuisanceFit, coefficient: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = prepared.baseline_prediction.copy().astype(np.float64)
    contribution = np.zeros_like(prediction)
    x_scale = nuisance.neural_scale.reshape(nuisance.n_neurons, nuisance.n_basis)
    for target, off in enumerate(nuisance.off_indices):
        off_sources = np.delete(np.arange(nuisance.n_neurons), target)
        coefficient_std = (coefficient[target, off_sources] * x_scale[off_sources]).reshape(-1)
        contribution[:, target] = prepared.residual_x[target] @ coefficient_std
    prediction += contribution
    return prediction.astype(np.float32), contribution.astype(np.float32)


def total_basis_kernel(
    nuisance: NuisanceFit, coefficient: np.ndarray
) -> np.ndarray:
    result = np.asarray(coefficient, dtype=np.float64).copy()
    d, q = nuisance.n_neurons, nuisance.n_basis
    x_scale = nuisance.neural_scale.reshape(d, q)
    for target, (self_model, off) in enumerate(zip(nuisance.self_models, nuisance.off_indices)):
        off_sources = np.delete(np.arange(d), target)
        coefficient_std = (coefficient[target, off_sources] * x_scale[off_sources]).reshape(-1)
        self_std = self_model.coef_[0] - self_model.coef_[1:].T @ coefficient_std
        result[target, target] = self_std / x_scale[target]
    return result.astype(np.float32)


def _student_nll_rows(target: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> np.ndarray:
    df = STUDENT_DF
    safe = np.maximum(np.asarray(scale, dtype=np.float64), 0.03)
    z2 = np.square((target - prediction) / safe)
    constant = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * np.log(df * np.pi)
        - np.log(safe)
    )
    logp = constant - 0.5 * (df + 1.0) * np.log1p(z2 / df)
    return -logp.mean(axis=1)


def _row_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_rank = rankdata(left, axis=1)
    right_rank = rankdata(right, axis=1)
    left_rank -= left_rank.mean(axis=1, keepdims=True)
    right_rank -= right_rank.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left_rank, axis=1) * np.linalg.norm(right_rank, axis=1)
    return np.divide(
        np.sum(left_rank * right_rank, axis=1),
        denominator,
        out=np.full(len(left), np.nan),
        where=denominator > 0,
    )


def _energy_mean(
    target: np.ndarray, prediction: np.ndarray, scale: np.ndarray, seed: int, max_rows: int = 2048
) -> float:
    rng = np.random.default_rng(seed)
    index = np.arange(len(target))
    if len(index) > max_rows:
        index = np.sort(rng.choice(index, size=max_rows, replace=False))
    noise = rng.standard_t(STUDENT_DF, size=(len(index), 32, target.shape[1]))
    samples = prediction[index, None] + noise * scale[None, None]
    first = np.linalg.norm(samples - target[index, None], axis=2).mean(axis=1)
    paired = np.linalg.norm(samples[:, ::2] - samples[:, 1::2], axis=2).mean(axis=1)
    return float(np.mean(first - 0.5 * paired))


def evaluate_predictions(
    train_target: np.ndarray,
    train_baseline: np.ndarray,
    train_full: np.ndarray,
    test: FeatureSet,
    prepared_test: PreparedSet,
    full_prediction: np.ndarray,
    contribution: np.ndarray,
    *,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    baseline_scale = np.sqrt(np.mean(np.square(train_target - train_baseline), axis=0))
    full_scale = np.sqrt(np.mean(np.square(train_target - train_full), axis=0))
    baseline_nll = _student_nll_rows(test.target, prepared_test.baseline_prediction, baseline_scale)
    full_nll = _student_nll_rows(test.target, full_prediction, full_scale)
    baseline_sq = np.square(test.target - prepared_test.baseline_prediction).mean(axis=1)
    full_sq = np.square(test.target - full_prediction).mean(axis=1)
    innovation = _row_spearman(contribution, test.target - prepared_test.baseline_prediction)
    baseline_energy = _energy_mean(
        test.target, prepared_test.baseline_prediction, baseline_scale, seed + 11
    )
    full_energy = _energy_mean(test.target, full_prediction, full_scale, seed + 23)
    metrics = {
        "baseline_nll": float(baseline_nll.mean()),
        "full_nll": float(full_nll.mean()),
        "nll_improvement": float((baseline_nll - full_nll).mean()),
        "baseline_rmse": float(np.sqrt(baseline_sq.mean())),
        "full_rmse": float(np.sqrt(full_sq.mean())),
        "rmse_improvement": float(np.sqrt(baseline_sq.mean()) - np.sqrt(full_sq.mean())),
        "baseline_energy": baseline_energy,
        "full_energy": full_energy,
        "energy_improvement": baseline_energy - full_energy,
        "innovation_spearman": float(np.nanmean(innovation)),
        "offdiag_prediction_rms": float(np.sqrt(np.square(contribution).mean())),
    }
    worm_rows = []
    for worm in np.unique(test.worm):
        use = test.worm == worm
        for phase, phase_mask in (
            ("all", np.ones(use.sum(), dtype=bool)),
            ("onset", test.stratum[use] == "onset"),
            ("quiet", test.stratum[use] == "off"),
        ):
            indices = np.flatnonzero(use)
            indices = indices[phase_mask]
            if len(indices) == 0:
                continue
            worm_rows.append(
                {
                    "worm": int(worm),
                    "phase": phase,
                    "n_rows": len(indices),
                    "nll_improvement": float(np.mean(baseline_nll[indices] - full_nll[indices])),
                    "mse_improvement": float(np.mean(baseline_sq[indices] - full_sq[indices])),
                    "innovation_spearman": float(np.nanmean(innovation[indices])),
                }
            )
    return metrics, pd.DataFrame(worm_rows)


def _candidate_specs() -> list[dict]:
    rows: list[dict] = []
    for alpha in CROSS_ALPHAS:
        rows.append({"family": "residual_ridge", "cross_alpha": alpha})
        for quantile in SHRINK_QUANTILES:
            rows.append(
                {"family": "group_shrunk", "cross_alpha": alpha, "shrink_quantile": quantile}
            )
        for rank in RANKS:
            rows.append({"family": "reduced_rank", "cross_alpha": alpha, "rank": rank})
        for quantile in SHRINK_QUANTILES:
            for rank in RANKS:
                rows.append(
                    {
                        "family": "sparse_lowrank",
                        "cross_alpha": alpha,
                        "shrink_quantile": quantile,
                        "rank": rank,
                    }
                )
    return rows


def select_family_specs(
    train: FeatureSet,
    validation: FeatureSet,
    nuisance: NuisanceFit,
    prepared_train: PreparedSet,
    prepared_validation: PreparedSet,
) -> tuple[dict[str, dict], pd.DataFrame]:
    path = fit_cross_path(prepared_train, nuisance, CROSS_ALPHAS)
    rows: list[dict] = []
    for spec in _candidate_specs():
        transformed, details = transform_coefficient(
            path[float(spec["cross_alpha"])],
            str(spec["family"]),
            shrink_quantile=spec.get("shrink_quantile"),
            rank=spec.get("rank"),
        )
        train_prediction, _ = predict_full(prepared_train, nuisance, transformed)
        validation_prediction, contribution = predict_full(
            prepared_validation, nuisance, transformed
        )
        scale = np.sqrt(np.mean(np.square(train.target - train_prediction), axis=0))
        nll = float(
            _student_nll_rows(validation.target, validation_prediction, scale).mean()
        )
        rows.append(
            {
                **spec,
                **details,
                "validation_nll": nll,
                "validation_rmse": float(
                    np.sqrt(np.square(validation.target - validation_prediction).mean())
                ),
                "validation_innovation_spearman": float(
                    np.nanmean(
                        _row_spearman(
                            contribution,
                            validation.target - prepared_validation.baseline_prediction,
                        )
                    )
                ),
            }
        )
    frame = pd.DataFrame(rows)
    selected: dict[str, dict] = {}
    for family in FAMILIES:
        winner = frame[frame.family == family].sort_values(
            ["validation_nll", "validation_rmse"]
        ).iloc[0]
        selected[family] = {
            key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in winner.to_dict().items()
            if key not in {"validation_nll", "validation_rmse", "validation_innovation_spearman", "threshold", "group_density"}
        }
    return selected, frame


def _refit_and_evaluate(
    train: FeatureSet,
    test: FeatureSet,
    basis_lag: np.ndarray,
    *,
    common_alpha: float,
    self_alpha: float,
    specs: dict[str, dict],
    seed: int,
) -> tuple[list[dict], pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    d = train.target.shape[1]
    q = basis_lag.shape[1]
    nuisance = fit_nuisance(
        train, d=d, q=q, common_alpha=common_alpha, self_alpha=self_alpha
    )
    prepared_train = prepare_set(nuisance, train)
    prepared_test = prepare_set(nuisance, test)
    requested_alphas = sorted({float(spec["cross_alpha"]) for spec in specs.values()})
    path = fit_cross_path(prepared_train, nuisance, requested_alphas)
    rows: list[dict] = []
    worms: list[pd.DataFrame] = []
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for family, spec in specs.items():
        transformed, details = transform_coefficient(
            path[float(spec["cross_alpha"])],
            family,
            shrink_quantile=spec.get("shrink_quantile"),
            rank=spec.get("rank"),
        )
        train_prediction, _ = predict_full(prepared_train, nuisance, transformed)
        test_prediction, contribution = predict_full(prepared_test, nuisance, transformed)
        metrics, worm_frame = evaluate_predictions(
            train.target,
            prepared_train.baseline_prediction,
            train_prediction,
            test,
            prepared_test,
            test_prediction,
            contribution,
            seed=seed + 101 * len(rows),
        )
        direct = reconstruct_kernel(transformed, basis_lag)
        total = reconstruct_kernel(total_basis_kernel(nuisance, transformed), basis_lag)
        impulse = impulse_response(total, max(32, len(basis_lag)))
        rows.append({"family": family, **spec, **details, **metrics})
        worm_frame.insert(0, "family", family)
        worms.append(worm_frame)
        artifacts[family] = {
            "basis_coefficient": transformed,
            "direct_kernel": direct,
            "total_kernel": total,
            "impulse_response": impulse,
            "test_contribution": contribution,
            "test_prediction": test_prediction,
            "baseline_prediction": prepared_test.baseline_prediction,
        }
    return rows, pd.concat(worms, ignore_index=True), artifacts


def _fold_features(
    cohort,
    folds: np.ndarray,
    fold: int,
    lag: int,
    basis: np.ndarray,
    pca_components: int,
    seed: int,
) -> tuple[FeatureSet, FeatureSet, FeatureSet, dict]:
    train_idx, validation_idx, test_idx = _split_indices(folds, fold)
    scaler = FoldScaler.fit(cohort.traces[i] for i in train_idx)
    train_windows = make_windows(cohort, train_idx, lag, scaler)
    validation_windows = make_windows(cohort, validation_idx, lag, scaler)
    test_windows = make_windows(cohort, test_idx, lag, scaler)
    pca = _fit_pca(train_windows, pca_components, seed)
    return (
        make_features(train_windows, basis, pca),
        make_features(validation_windows, basis, pca),
        make_features(test_windows, basis, pca),
        {
            "train_worms": train_idx.tolist(),
            "validation_worms": validation_idx.tolist(),
            "test_worms": test_idx.tolist(),
            "scaler_mean": scaler.mean.tolist(),
            "scaler_scale": scaler.scale.tolist(),
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        },
    )


def _refit_features(
    cohort,
    folds: np.ndarray,
    fold: int,
    lag: int,
    basis: np.ndarray,
    pca_components: int,
    seed: int,
) -> tuple[FeatureSet, FeatureSet]:
    train_idx, validation_idx, test_idx = _split_indices(folds, fold)
    fit_idx = np.concatenate([train_idx, validation_idx])
    scaler = FoldScaler.fit(cohort.traces[i] for i in fit_idx)
    fit_windows = make_windows(cohort, fit_idx, lag, scaler)
    test_windows = make_windows(cohort, test_idx, lag, scaler)
    pca = _fit_pca(fit_windows, pca_components, seed)
    return make_features(fit_windows, basis, pca), make_features(test_windows, basis, pca)


def _save_fold_artifact(
    path: Path,
    *,
    family: str,
    lag: int,
    fold: int,
    basis: np.ndarray,
    neurons: Iterable[str],
    artifact: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        status=np.asarray("complete"),
        family=np.asarray(family),
        lag_frames=np.asarray(lag),
        fold=np.asarray(fold),
        lag_basis=basis,
        neurons=np.asarray(tuple(neurons)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **artifact,
    )


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draw, 0.025)), float(np.quantile(draw, 0.975))


def _semisynthetic_sequence(
    noise_blocks: list[np.ndarray], *, seed: int, effect: float, smooth: bool, length: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = min(12, noise_blocks[0].shape[1])
    max_lag = 16
    true = np.zeros((max_lag, d, d), dtype=np.float64)
    true[0, np.arange(d), np.arange(d)] = 0.38
    pairs = [(t, s) for t in range(d) for s in range(d) if t != s]
    selected = rng.choice(len(pairs), size=20, replace=False)
    delays = np.asarray([1, 2, 4, 8, 16])
    for position, pair_index in enumerate(selected):
        target, source = pairs[int(pair_index)]
        delay = int(delays[position % len(delays)])
        true[delay - 1, target, source] = rng.choice([-1.0, 1.0]) * effect
    noise = np.empty((length, d), dtype=np.float64)
    cursor = 0
    while cursor < length:
        block = np.asarray(noise_blocks[int(rng.integers(len(noise_blocks)))], dtype=np.float64)
        start = int(rng.integers(max(1, len(block) - 64)))
        take = min(64, length - cursor, len(block) - start)
        noise[cursor : cursor + take] = block[start : start + take, :d]
        cursor += take
    noise /= np.maximum(noise.std(axis=0, keepdims=True), 1e-6)
    noise *= 0.16
    latent = np.zeros((length, d), dtype=np.float64)
    latent[:max_lag] = noise[:max_lag]
    for index in range(max_lag, length):
        history = latent[index - max_lag : index][::-1]
        latent[index] = np.einsum("ls,lts->t", history, true) + noise[index]
    observed = latent.copy()
    if smooth:
        calcium_alpha = 0.32
        for index in range(1, length):
            observed[index] = (1.0 - calcium_alpha) * observed[index - 1] + calcium_alpha * latent[index]
    return observed.astype(np.float32), true.astype(np.float32)


def semisynthetic_calibration(
    cohort, *, basis: np.ndarray, effects: Iterable[float], replicates: int, seed: int
) -> pd.DataFrame:
    scaler = FoldScaler.fit(cohort.traces)
    blocks = []
    for trace in cohort.traces:
        filled = _causal_fill(scaler.transform(trace))
        # Causal fill intentionally leaves leading values missing when a
        # neuron has not yet been observed.  Semi-synthetic innovation blocks
        # have no factual-prefix semantics, so replace only those remaining
        # leading values with the standardized training mean (zero).
        filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
        blocks.append(np.diff(filled, axis=0))
    rows: list[dict] = []
    lag = len(basis)
    q = basis.shape[1]
    for smooth in (False, True):
        for effect in effects:
            for replicate in range(replicates):
                observed, true = _semisynthetic_sequence(
                    blocks,
                    seed=seed + 1009 * replicate + int(round(effect * 10000)) + 31 * smooth,
                    effect=float(effect),
                    smooth=smooth,
                )
                history = np.lib.stride_tricks.sliding_window_view(
                    observed, window_shape=lag, axis=0
                )[:-1].transpose(0, 2, 1)
                x = np.einsum("nld,lq->ndq", history, basis[::-1], optimize=True).reshape(len(history), -1)
                y = observed[lag:]
                first, second = 2800, 3900
                x_train, (x_val, x_test), mean, scale = _standardize(
                    x[:first], (x[first:second], x[second:])
                )
                best = None
                for alpha in CROSS_ALPHAS:
                    fitted = _ridge(alpha).fit(x_train, y[:first])
                    mse = float(np.square(fitted.predict(x_val) - y[first:second]).mean())
                    if best is None or mse < best[0]:
                        best = (mse, alpha)
                fitted = _ridge(best[1]).fit(
                    np.concatenate([x_train, x_val]), y[:second]
                )
                raw = fitted.coef_.reshape(y.shape[1], y.shape[1], q) / scale.reshape(y.shape[1], q)[None]
                raw[np.arange(y.shape[1]), np.arange(y.shape[1])] = 0.0
                estimated = reconstruct_kernel(raw, basis)
                d = y.shape[1]
                off = ~np.eye(d, dtype=bool)
                labels = (np.max(np.abs(true), axis=0) > 0) & off
                score = np.max(np.abs(estimated), axis=0)
                true_lag = np.argmax(np.abs(true), axis=0) + 1
                estimated_lag = np.argmax(np.abs(estimated), axis=0) + 1
                rows.append(
                    {
                        "condition": "calcium_smoothed" if smooth else "latent",
                        "effect": effect,
                        "replicate": replicate,
                        "selected_alpha": best[1],
                        "edge_auroc": float(roc_auc_score(labels[off], score[off])),
                        "lag_mae_frames": float(np.mean(np.abs(true_lag[labels] - estimated_lag[labels]))),
                        "lag_within_one_frame": float(np.mean(np.abs(true_lag[labels] - estimated_lag[labels]) <= 1)),
                        "test_rmse": float(np.sqrt(np.square(fitted.predict(x_test) - y[second:]).mean())),
                    }
                )
    return pd.DataFrame(rows)


def _kernel_stability(paths: list[Path], selected_family: str, selected_lag: int) -> pd.DataFrame:
    kernels = []
    folds = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["family"].item()) != selected_family or int(data["lag_frames"]) != selected_lag:
                continue
            kernels.append(data["direct_kernel"].astype(np.float64))
            folds.append(int(data["fold"]))
    rows = []
    for left in range(len(kernels)):
        for right in range(left + 1, len(kernels)):
            d = kernels[left].shape[1]
            off = ~np.eye(d, dtype=bool)
            a = kernels[left][:, off].ravel()
            b = kernels[right][:, off].ravel()
            correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
            edge_a = np.max(np.abs(kernels[left]), axis=0)[off]
            edge_b = np.max(np.abs(kernels[right]), axis=0)[off]
            rank_a = rankdata(edge_a)
            rank_b = rankdata(edge_b)
            rows.append(
                {
                    "fold_left": folds[left],
                    "fold_right": folds[right],
                    "kernel_pearson": correlation,
                    "edge_rank_spearman": float(np.corrcoef(rank_a, rank_b)[0, 1]),
                    "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(output: Path) -> None:
    paths = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "checksums.sha256"]
    (output / "checksums.sha256").write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(output)}" for path in paths) + "\n"
    )


def _write_report(output: Path, metrics: pd.DataFrame, selection: dict, confirmation: dict) -> None:
    selected = metrics[
        (metrics.family == selection["family"]) & (metrics.lag_frames == selection["lag_frames"])
    ]
    lines = [
        "# Residualized distributed-lag dynamics",
        "",
        "**Claim boundary:** atlas-blind predictive innovation dynamics; not physical intervention, direct anatomy, or transmission delay.",
        "",
        "## Answer first",
        "",
        f"The screen selected **{selection['family']}** with a {selection['lag_seconds']:.1f}-second maximum lag. "
        f"On untouched folds 3–4, mean Student-t log-score improvement over self + stimulus + global state was "
        f"**{confirmation['mean_nll_improvement']:+.6f}** with worm-bootstrap 95% CI "
        f"**[{confirmation['ci_low']:+.6f}, {confirmation['ci_high']:+.6f}]**. "
        f"The promotion gate **{'passed' if confirmation['passes_gate'] else 'did not pass'}**.",
        "",
        "## What is new",
        "",
        "- The baseline removes target self-history, the complete binary stimulus history, and cross-fitted global population components.",
        "- Each ordered pair is represented by one smooth lag kernel and regularized jointly across all lag coefficients.",
        "- `direct_kernel` is a one-step innovation dependency; `impulse_response` is its recursively propagated response. They are saved separately.",
        "- Selection uses folds 0–2 only; folds 3–4 are untouched confirmation. No connectome or SBTG artifact is loaded here.",
        "",
        "## Confirmation diagnostics",
        "",
        f"- Mean RMSE improvement: **{confirmation['mean_rmse_improvement']:+.6f}**.",
        f"- Mean energy improvement: **{confirmation['mean_energy_improvement']:+.6f}**.",
        f"- Mean innovation Spearman: **{confirmation['mean_innovation_spearman']:+.4f}**.",
        f"- Both confirmation folds improved log score: **{confirmation['both_folds_positive_nll']}**.",
        "",
        "## Interpretation",
        "",
        "A nonzero lag kernel is evidence of held-out conditional predictive structure after the declared nuisance model. Calcium filtering, unmeasured state, and common drive can still create reduced-form lag structure. A kernel peak is therefore not a physical delay.",
        "",
        "## Files",
        "",
        "- `per_fold_metrics.csv`: outer-fold proper scores and guardrails.",
        "- `worm_metrics.csv`: worm-clustered improvements for confirmation inference.",
        "- `fold_models/`: direct kernels, total kernels, and impulse responses for every fold/family/history.",
        "- `semisynthetic_detectability.csv`: lag recovery using real-residual block noise.",
        "- `kernel_stability.csv`: fold-to-fold kernel and edge-ranking stability.",
        "- `selection.json` and `promotion_gate.json`: frozen selection and confirmation decision.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cohort = load_cohort()
    fold_frame = pd.read_csv(args.fold_assignments)
    fold_by_worm = dict(zip(fold_frame.worm_id.astype(str), fold_frame.outer_fold.astype(int)))
    folds = np.asarray([fold_by_worm[worm] for worm in cohort.worm_ids], dtype=np.int64)
    fold_frame.to_csv(output / "fold_assignments.csv", index=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "one-step target innovation conditional on target self-history, full binary stimulus history, global population PCs, and lagged source history",
        "lags_frames": list(args.lags),
        "lags_seconds": [lag / cohort.fps for lag in args.lags],
        "families": list(FAMILIES),
        "screen_folds": [0, 1, 2],
        "confirmation_folds": [3, 4],
        "primary_metric": "held-out diagonal Student-t log-score improvement over nuisance baseline",
        "atlas_firewall": "no Randi, Cook, Bentley, SBTG, or neuromodulator file loaded",
        "claim_boundary": "observational predictive innovation dynamics, not causal/anatomical/physical delay",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    metrics_path = output / "per_fold_metrics.csv"
    worms_path = output / "worm_metrics.csv"
    tuning_path = output / "tuning_grid.csv"
    if args.resume and metrics_path.exists() and worms_path.exists() and tuning_path.exists():
        existing_metrics = pd.read_csv(metrics_path)
        metric_rows: list[dict] = existing_metrics.to_dict("records")
        worm_frames: list[pd.DataFrame] = [pd.read_csv(worms_path)]
        tuning_frames: list[pd.DataFrame] = [pd.read_csv(tuning_path)]
        artifacts: list[Path] = sorted((output / "fold_models").glob("*.npz"))
    else:
        metric_rows = []
        worm_frames = []
        tuning_frames = []
        artifacts = []
    completed = {
        (int(lag), int(fold))
        for (lag, fold), group in pd.DataFrame(metric_rows).groupby(
            ["lag_frames", "fold"]
        )
        if len(group) == len(FAMILIES)
        and all(
            (output / "fold_models" / f"{family}__L{int(lag)}__f{int(fold)}.npz").exists()
            for family in FAMILIES
        )
    } if metric_rows else set()
    for lag in args.lags:
        basis = smooth_lag_basis(lag, args.n_basis)
        np.save(output / f"lag_basis_L{lag}.npy", basis)
        for fold in range(5):
            if (int(lag), int(fold)) in completed:
                print(f"DISTRIBUTED_LAG_SKIP lag={lag} fold={fold}", flush=True)
                continue
            trial_start = time.perf_counter()
            print(f"DISTRIBUTED_LAG_START lag={lag} fold={fold}", flush=True)
            train, validation, _, split_metadata = _fold_features(
                cohort, folds, fold, lag, basis, args.pca_components, args.seed + 101 * fold + lag
            )
            d, q = cohort.n_neurons, basis.shape[1]
            common_alpha, self_alpha, nuisance_grid = select_nuisance_alphas(
                train, validation, d, q
            )
            nuisance = fit_nuisance(
                train,
                d=d,
                q=q,
                common_alpha=common_alpha,
                self_alpha=self_alpha,
            )
            prepared_train = prepare_set(nuisance, train)
            prepared_validation = prepare_set(nuisance, validation)
            specs, family_grid = select_family_specs(
                train, validation, nuisance, prepared_train, prepared_validation
            )
            nuisance_grid.insert(0, "grid", "nuisance")
            nuisance_grid.insert(0, "fold", fold)
            nuisance_grid.insert(0, "lag_frames", lag)
            family_grid.insert(0, "grid", "lag_family")
            family_grid.insert(0, "fold", fold)
            family_grid.insert(0, "lag_frames", lag)
            tuning_frames.extend([nuisance_grid, family_grid])
            fit, test = _refit_features(
                cohort, folds, fold, lag, basis, args.pca_components, args.seed + 7001 + 101 * fold + lag
            )
            rows, worms, fold_artifacts = _refit_and_evaluate(
                fit,
                test,
                basis,
                common_alpha=common_alpha,
                self_alpha=self_alpha,
                specs=specs,
                seed=args.seed + 7919 * fold + lag,
            )
            for row in rows:
                row.update(
                    {
                        "lag_frames": lag,
                        "lag_seconds": lag / cohort.fps,
                        "fold": fold,
                        "common_alpha": common_alpha,
                        "self_alpha": self_alpha,
                        "wall_seconds": time.perf_counter() - trial_start,
                    }
                )
                metric_rows.append(row)
            worms.insert(0, "fold", fold)
            worms.insert(0, "lag_seconds", lag / cohort.fps)
            worms.insert(0, "lag_frames", lag)
            worm_frames.append(worms)
            for family, artifact in fold_artifacts.items():
                path = output / "fold_models" / f"{family}__L{lag}__f{fold}.npz"
                _save_fold_artifact(
                    path,
                    family=family,
                    lag=lag,
                    fold=fold,
                    basis=basis,
                    neurons=cohort.neurons,
                    artifact=artifact,
                    metadata={
                        **split_metadata,
                        "common_alpha": common_alpha,
                        "self_alpha": self_alpha,
                        "family_spec": specs[family],
                    },
                )
                artifacts.append(path)
            pd.DataFrame(metric_rows).to_csv(output / "per_fold_metrics.csv", index=False)
            pd.concat(worm_frames, ignore_index=True).to_csv(output / "worm_metrics.csv", index=False)
            pd.concat(tuning_frames, ignore_index=True, sort=False).to_csv(output / "tuning_grid.csv", index=False)
            print(
                f"DISTRIBUTED_LAG_DONE lag={lag} fold={fold} seconds={time.perf_counter()-trial_start:.1f}",
                flush=True,
            )
    metrics = pd.DataFrame(metric_rows)
    worms = pd.concat(worm_frames, ignore_index=True)
    screen = (
        metrics[metrics.fold.isin([0, 1, 2])]
        .groupby(["family", "lag_frames", "lag_seconds"], as_index=False)
        .agg(
            mean_nll_improvement=("nll_improvement", "mean"),
            mean_rmse_improvement=("rmse_improvement", "mean"),
            mean_energy_improvement=("energy_improvement", "mean"),
            mean_innovation_spearman=("innovation_spearman", "mean"),
        )
        .sort_values(["mean_nll_improvement", "mean_rmse_improvement"], ascending=False)
    )
    screen.to_csv(output / "screen_leaderboard.csv", index=False)
    winner = screen.iloc[0]
    selection = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": str(winner.family),
        "lag_frames": int(winner.lag_frames),
        "lag_seconds": float(winner.lag_seconds),
        "selection_metric": "mean outer-test Student-t NLL improvement on folds 0-2",
        "external_references_consulted": False,
        "screen_metrics": {
            key: float(winner[key]) for key in winner.index if key.startswith("mean_")
        },
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    confirmation_rows = metrics[
        (metrics.family == selection["family"])
        & (metrics.lag_frames == selection["lag_frames"])
        & (metrics.fold.isin([3, 4]))
    ]
    confirmation_worms = worms[
        (worms.family == selection["family"])
        & (worms.lag_frames == selection["lag_frames"])
        & (worms.fold.isin([3, 4]))
        & (worms.phase == "all")
    ]
    ci_low, ci_high = _bootstrap_ci(
        confirmation_worms.nll_improvement.to_numpy(), args.seed + 9091
    )
    confirmation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": selection["family"],
        "lag_frames": selection["lag_frames"],
        "confirmation_folds": [3, 4],
        "confirmation_worms": int(len(confirmation_worms)),
        "mean_nll_improvement": float(confirmation_worms.nll_improvement.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_rmse_improvement": float(confirmation_rows.rmse_improvement.mean()),
        "mean_energy_improvement": float(confirmation_rows.energy_improvement.mean()),
        "mean_innovation_spearman": float(confirmation_rows.innovation_spearman.mean()),
        "both_folds_positive_nll": bool((confirmation_rows.nll_improvement > 0).all()),
        "passes_gate": bool(
            ci_low > 0
            and (confirmation_rows.nll_improvement > 0).all()
            and (confirmation_rows.rmse_improvement > 0).all()
        ),
        "promotion_rule": "worm-bootstrap NLL-improvement CI > 0 and both confirmation folds improve NLL and RMSE",
        "external_references_consulted": False,
    }
    (output / "promotion_gate.json").write_text(json.dumps(confirmation, indent=2) + "\n")
    stability = _kernel_stability(artifacts, selection["family"], selection["lag_frames"])
    stability.to_csv(output / "kernel_stability.csv", index=False)
    selected_basis = np.load(output / f"lag_basis_L{selection['lag_frames']}.npy")
    semisynthetic = semisynthetic_calibration(
        cohort,
        basis=selected_basis,
        effects=args.synthetic_effects,
        replicates=args.synthetic_replicates,
        seed=args.seed + 12001,
    )
    semisynthetic.to_csv(output / "semisynthetic_detectability.csv", index=False)
    _write_report(output, metrics, selection, confirmation)
    validation = {
        "status": "pass",
        "fold_model_files": len(artifacts),
        "expected_fold_model_files": len(args.lags) * 5 * len(FAMILIES),
        "metric_rows": len(metrics),
        "worm_metric_rows": len(worms),
        "semisynthetic_rows": len(semisynthetic),
        "all_finite_primary_metrics": bool(
            np.isfinite(metrics[["nll_improvement", "rmse_improvement", "energy_improvement"]].to_numpy()).all()
        ),
        "neural_wall_minutes": float(
            metrics.groupby(["lag_frames", "fold"]).wall_seconds.max().sum() / 60.0
        ),
        "finalization_wall_minutes": (time.perf_counter() - started) / 60.0,
    }
    if validation["fold_model_files"] != validation["expected_fold_model_files"]:
        validation["status"] = "fail"
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    _write_checksums(output)
    if validation["status"] != "pass":
        raise RuntimeError(validation)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path(
            "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827/fold_assignments.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distributed_lag_dynamics_20260828"),
    )
    parser.add_argument("--lags", nargs="+", type=int, default=list(DEFAULT_LAGS))
    parser.add_argument("--n-basis", type=int)
    parser.add_argument("--pca-components", type=int, default=4)
    parser.add_argument("--synthetic-effects", nargs="+", type=float, default=[0.03, 0.06, 0.10, 0.15])
    parser.add_argument("--synthetic-replicates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
