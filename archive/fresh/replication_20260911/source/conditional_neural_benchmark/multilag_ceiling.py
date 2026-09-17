from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from conditional_neural_benchmark.data import (
    Windows,
    choose_evaluation_indices,
    load_cohort,
    split_history,
)
from conditional_neural_benchmark.metrics import metric_rows, summarize_metrics
from conditional_neural_benchmark.runner import _split_windows


FPS = 4.0
LAGS = (4, 8, 16, 32, 80)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
RANKS = (4, 8, 16)
SHRINK_QUANTILES = (0.0, 0.25, 0.50, 0.75, 0.90)
MODEL_NAMES = (
    "persistence",
    "self_ridge",
    "full_ridge",
    "offdiag_reduced_rank",
    "group_shrunk_ridge",
    "sparse_plus_lowrank",
    "phase_modulated_ridge",
)


@dataclass(frozen=True)
class LinearLagModel:
    name: str
    lag: int
    basis: np.ndarray
    coef: np.ndarray | None
    intercept: np.ndarray | None
    chol: np.ndarray
    feature_mode: str = "base"

    @property
    def n_neurons(self) -> int:
        return int(self.chol.shape[0])

    def features(self, history: np.ndarray) -> np.ndarray:
        base, neural, stimulus = basis_features(
            history, self.basis, self.n_neurons
        )
        if self.feature_mode == "base":
            return base
        if self.feature_mode != "phase_modulated":
            raise ValueError(f"unknown feature mode {self.feature_mode}")
        _, stimulus_channels = split_history(history, self.n_neurons)
        stim_history = np.max(stimulus_channels, axis=-1)
        active = stim_history[:, -1]
        onset = np.any(np.diff(stim_history, axis=1) > 0.5, axis=1).astype(np.float32)
        return np.concatenate(
            [neural, stimulus, neural * active[:, None], neural * onset[:, None]],
            axis=1,
        ).astype(np.float32, copy=False)

    def mean(self, history: np.ndarray) -> np.ndarray:
        if self.coef is None:
            return np.asarray(
                history[:, -1, :self.n_neurons], dtype=np.float32
            )
        return self.features(history) @ self.coef.T + self.intercept

    def sample(self, history: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
        mean = self.mean(history)
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((len(mean), n_samples, self.n_neurons))
        return (mean[:, None] + noise @ self.chol.T).astype(np.float32)


def temporal_basis(lag: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Piecewise-constant causal lag basis with exact recent-frame resolution."""
    if lag < 1:
        raise ValueError("lag must be positive")
    proposed = (
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 6),
        (7, 8),
        (9, 12),
        (13, 16),
        (17, 24),
        (25, 32),
        (33, 48),
        (49, 64),
        (65, 80),
    )
    intervals: list[tuple[int, int]] = []
    covered = 0
    for lo, hi in proposed:
        if lo > lag:
            break
        hi = min(hi, lag)
        intervals.append((lo, hi))
        covered = hi
    if covered < lag:
        intervals.append((covered + 1, lag))
    basis = np.zeros((lag, len(intervals)), dtype=np.float32)
    for column, (lo, hi) in enumerate(intervals):
        chronological = np.arange(lag - hi, lag - lo + 1)
        basis[chronological, column] = 1.0 / len(chronological)
    np.testing.assert_allclose(basis.sum(axis=0), 1.0)
    return basis, intervals


def basis_features(
    history: np.ndarray, basis: np.ndarray, n_neurons: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if history.ndim != 3 or history.shape[1] != basis.shape[0]:
        raise ValueError("history and basis disagree")
    neural_history, stimulus_history = split_history(history, n_neurons)
    neural_history = np.asarray(neural_history, dtype=np.float32)
    stimulus_history = np.asarray(stimulus_history, dtype=np.float32)
    projected = np.einsum("eld,lk->edk", neural_history, basis, optimize=True)
    neural = projected.reshape(len(history), -1)
    stimulus = np.einsum(
        "elc,lk->eck", stimulus_history, basis, optimize=True
    ).reshape(len(history), -1)
    return (
        np.concatenate([neural, stimulus], axis=1).astype(np.float32, copy=False),
        neural.astype(np.float32, copy=False),
        stimulus.astype(np.float32, copy=False),
    )


def _ridge(alpha: float) -> Ridge:
    return Ridge(
        alpha=float(alpha),
        fit_intercept=True,
        solver="lsqr",
        tol=1e-4,
        max_iter=2000,
    )


def _select_alpha(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> float:
    best = (math.inf, ALPHAS[0])
    for alpha in ALPHAS:
        model = _ridge(alpha).fit(train_x, train_y)
        mse = float(np.square(model.predict(validation_x) - validation_y).mean())
        if mse < best[0]:
            best = (mse, alpha)
    return float(best[1])


def _fit_full(
    train: Windows,
    validation: Windows,
    basis: np.ndarray,
    *,
    feature_mode: str = "base",
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    helper = LinearLagModel(
        "feature_helper",
        len(basis),
        basis,
        np.zeros((train.target.shape[1], 1), dtype=np.float32),
        np.zeros(train.target.shape[1], dtype=np.float32),
        np.eye(train.target.shape[1], dtype=np.float32),
        feature_mode,
    )
    train_x = helper.features(train.history)
    validation_x = helper.features(validation.history)
    alpha = _select_alpha(train_x, train.target, validation_x, validation.target)
    fit_x = np.concatenate([train_x, validation_x])
    fit_y = np.concatenate([train.target, validation.target])
    fitted = _ridge(alpha).fit(fit_x, fit_y)
    return (
        fitted.coef_.astype(np.float32),
        fitted.intercept_.astype(np.float32),
        alpha,
        fit_x,
        fit_y,
    )


def _fit_self(
    train: Windows,
    validation: Windows,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    d = train.target.shape[1]
    train_base, train_neural, train_stimulus = basis_features(
        train.history, basis, d
    )
    validation_base, validation_neural, validation_stimulus = basis_features(
        validation.history, basis, d
    )
    del train_base, validation_base
    k = basis.shape[1]
    scores: dict[float, list[float]] = {alpha: [] for alpha in ALPHAS}
    for alpha in ALPHAS:
        for target in range(d):
            train_x = np.concatenate(
                [train_neural[:, target * k : (target + 1) * k], train_stimulus],
                axis=1,
            )
            validation_x = np.concatenate(
                [
                    validation_neural[:, target * k : (target + 1) * k],
                    validation_stimulus,
                ],
                axis=1,
            )
            fitted = _ridge(alpha).fit(train_x, train.target[:, target])
            scores[alpha].append(
                float(
                    np.square(
                        fitted.predict(validation_x) - validation.target[:, target]
                    ).mean()
                )
            )
    alpha = min(ALPHAS, key=lambda candidate: float(np.mean(scores[candidate])))
    fit_history = np.concatenate([train.history, validation.history])
    fit_target = np.concatenate([train.target, validation.target])
    _, fit_neural, fit_stimulus = basis_features(fit_history, basis, d)
    stimulus_width = fit_stimulus.shape[1]
    full_coef = np.zeros((d, d * k + stimulus_width), dtype=np.float32)
    intercept = np.zeros(d, dtype=np.float32)
    for target in range(d):
        x = np.concatenate(
            [fit_neural[:, target * k : (target + 1) * k], fit_stimulus], axis=1
        )
        fitted = _ridge(alpha).fit(x, fit_target[:, target])
        full_coef[target, target * k : (target + 1) * k] = fitted.coef_[:k]
        full_coef[target, d * k :] = fitted.coef_[k:]
        intercept[target] = fitted.intercept_
    fit_base, _, _ = basis_features(fit_history, basis, d)
    return full_coef, intercept, float(alpha), fit_base, fit_target


def _residual_chol(
    coef: np.ndarray | None,
    intercept: np.ndarray | None,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    *,
    persistence_history: np.ndarray | None = None,
) -> np.ndarray:
    if coef is None:
        if persistence_history is None:
            raise ValueError("persistence history required")
        prediction = persistence_history[:, -1, :fit_y.shape[1]]
    else:
        prediction = fit_x @ coef.T + intercept
    residual = np.asarray(fit_y - prediction, dtype=np.float64)
    covariance = LedoitWolf().fit(residual).covariance_.astype(np.float64)
    covariance.flat[:: len(covariance) + 1] += 1e-6
    return np.linalg.cholesky(covariance).astype(np.float32)


def _offdiagonal_tensor(coef: np.ndarray, d: int, k: int) -> np.ndarray:
    tensor = coef[:, : d * k].reshape(d, d, k).copy()
    tensor[np.arange(d), np.arange(d)] = 0.0
    return tensor


def _replace_offdiagonal(
    coef: np.ndarray, replacement: np.ndarray, d: int, k: int
) -> np.ndarray:
    result = coef.copy()
    tensor = result[:, : d * k].reshape(d, d, k)
    off = ~np.eye(d, dtype=bool)
    tensor[off] = replacement[off]
    return result


def _rank_reduce_offdiagonal(
    coef: np.ndarray, d: int, k: int, rank: int
) -> np.ndarray:
    off = _offdiagonal_tensor(coef, d, k).reshape(d, d * k)
    u, singular, vt = np.linalg.svd(off, full_matrices=False)
    r = min(int(rank), len(singular))
    reduced = (u[:, :r] * singular[:r]) @ vt[:r]
    return _replace_offdiagonal(coef, reduced.reshape(d, d, k), d, k)


def _group_shrink_offdiagonal(
    coef: np.ndarray, d: int, k: int, quantile: float
) -> tuple[np.ndarray, float, float]:
    tensor = _offdiagonal_tensor(coef, d, k)
    off = ~np.eye(d, dtype=bool)
    norms = np.linalg.norm(tensor, axis=2)
    threshold = float(np.quantile(norms[off], quantile))
    factors = np.maximum(0.0, 1.0 - threshold / np.maximum(norms, 1e-12))
    shrunk = tensor * factors[..., None]
    density = float(np.mean(np.linalg.norm(shrunk, axis=2)[off] > 0))
    return _replace_offdiagonal(coef, shrunk, d, k), threshold, density


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


def _crps_rows(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    first = np.abs(samples - target[:, None]).mean(axis=(1, 2))
    paired_n = samples.shape[1] - samples.shape[1] % 2
    paired = np.abs(samples[:, :paired_n:2] - samples[:, 1:paired_n:2]).mean(
        axis=(1, 2)
    )
    return first - 0.5 * paired


def evaluate_model(
    model: LinearLagModel,
    test: Windows,
    self_model: LinearLagModel,
    *,
    eval_rows: int,
    n_samples: int,
    seed: int,
) -> dict[str, float]:
    index = choose_evaluation_indices(test.stratum, eval_rows, seed)
    history = test.history[index]
    target = test.target[index]
    strata = test.stratum[index]
    samples = model.sample(history, n_samples, seed + 17)
    rows = metric_rows(samples, target, seed + 29)
    rows["crps"] = _crps_rows(samples, target)
    summary = summarize_metrics(
        rows,
        strata,
        None,
        target.shape[1],
        population_strata=test.stratum,
    )
    prediction = model.mean(history)
    self_prediction = self_model.mean(history)
    summary["target_spearman"] = float(np.nanmean(_row_spearman(prediction, target)))
    if model.name == "self_ridge":
        summary["innovation_spearman"] = 0.0
        summary["offdiag_prediction_rms"] = 0.0
    else:
        summary["innovation_spearman"] = float(
            np.nanmean(
                _row_spearman(
                    prediction - self_prediction, target - self_prediction
                )
            )
        )
        summary["offdiag_prediction_rms"] = float(
            np.sqrt(np.square(prediction - self_prediction).mean())
        )
    return summary


def _save_model(
    path: Path,
    model: LinearLagModel,
    *,
    scaler,
    neurons: Iterable[str],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": np.asarray(model.name),
        "lag": np.asarray(model.lag),
        "basis": model.basis,
        "chol": model.chol,
        "feature_mode": np.asarray(model.feature_mode),
        "scaler_mean": scaler.mean,
        "scaler_scale": scaler.scale,
        "neurons": np.asarray(tuple(neurons)),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if model.coef is not None:
        payload["coef"] = model.coef
        payload["intercept"] = model.intercept
    np.savez_compressed(path, **payload)


def _fit_primary_models(
    train: Windows,
    validation: Windows,
    basis: np.ndarray,
) -> tuple[LinearLagModel, LinearLagModel, LinearLagModel, dict]:
    fit_history = np.concatenate([train.history, validation.history])
    fit_target = np.concatenate([train.target, validation.target])
    d = train.target.shape[1]
    fit_base, _, _ = basis_features(fit_history, basis, d)
    persistence_chol = _residual_chol(
        None,
        None,
        fit_base,
        fit_target,
        persistence_history=fit_history,
    )
    persistence = LinearLagModel(
        "persistence", len(basis), basis, None, None, persistence_chol
    )
    self_coef, self_intercept, self_alpha, self_fit_x, self_fit_y = _fit_self(
        train, validation, basis
    )
    self_chol = _residual_chol(
        self_coef, self_intercept, self_fit_x, self_fit_y
    )
    self_model = LinearLagModel(
        "self_ridge", len(basis), basis, self_coef, self_intercept, self_chol
    )
    full_coef, full_intercept, full_alpha, full_fit_x, full_fit_y = _fit_full(
        train, validation, basis
    )
    full_chol = _residual_chol(
        full_coef, full_intercept, full_fit_x, full_fit_y
    )
    full_model = LinearLagModel(
        "full_ridge", len(basis), basis, full_coef, full_intercept, full_chol
    )
    return persistence, self_model, full_model, {
        "self_alpha": self_alpha,
        "full_alpha": full_alpha,
        "full_fit_x": full_fit_x,
        "full_fit_y": full_fit_y,
    }


def _validation_mse(
    coef: np.ndarray,
    intercept: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> float:
    return float(np.square(validation_x @ coef.T + intercept - validation_y).mean())


def _fit_structured_models(
    train: Windows,
    validation: Windows,
    basis: np.ndarray,
) -> tuple[list[LinearLagModel], list[dict]]:
    d = train.target.shape[1]
    k = basis.shape[1]
    train_x, _, _ = basis_features(train.history, basis, d)
    validation_x, _, _ = basis_features(validation.history, basis, d)
    alpha = _select_alpha(train_x, train.target, validation_x, validation.target)
    training_fit = _ridge(alpha).fit(train_x, train.target)
    training_coef = training_fit.coef_.astype(np.float32)
    training_intercept = training_fit.intercept_.astype(np.float32)
    fit_x = np.concatenate([train_x, validation_x])
    fit_y = np.concatenate([train.target, validation.target])
    refit = _ridge(alpha).fit(fit_x, fit_y)
    base_coef = refit.coef_.astype(np.float32)
    base_intercept = refit.intercept_.astype(np.float32)

    rank_scores = []
    for rank in RANKS:
        candidate = _rank_reduce_offdiagonal(training_coef, d, k, rank)
        rank_scores.append(
            (_validation_mse(candidate, training_intercept, validation_x, validation.target), rank)
        )
    selected_rank = min(rank_scores)[1]
    reduced_coef = _rank_reduce_offdiagonal(base_coef, d, k, selected_rank)

    shrink_scores = []
    for quantile in SHRINK_QUANTILES:
        candidate, threshold, density = _group_shrink_offdiagonal(
            training_coef, d, k, quantile
        )
        shrink_scores.append(
            (
                _validation_mse(
                    candidate, training_intercept, validation_x, validation.target
                ),
                quantile,
                threshold,
                density,
            )
        )
    _, selected_quantile, _, _ = min(shrink_scores)
    shrunk_coef, threshold, density = _group_shrink_offdiagonal(
        base_coef, d, k, selected_quantile
    )

    sparse_rank_scores = []
    for quantile in SHRINK_QUANTILES:
        candidate, _, _ = _group_shrink_offdiagonal(training_coef, d, k, quantile)
        for rank in RANKS:
            candidate_rank = _rank_reduce_offdiagonal(candidate, d, k, rank)
            sparse_rank_scores.append(
                (
                    _validation_mse(
                        candidate_rank,
                        training_intercept,
                        validation_x,
                        validation.target,
                    ),
                    quantile,
                    rank,
                )
            )
    _, sparse_quantile, sparse_rank = min(sparse_rank_scores)
    sparse_coef, sparse_threshold, sparse_density = _group_shrink_offdiagonal(
        base_coef, d, k, sparse_quantile
    )
    sparse_lowrank_coef = _rank_reduce_offdiagonal(sparse_coef, d, k, sparse_rank)

    definitions = [
        (
            "offdiag_reduced_rank",
            reduced_coef,
            {"alpha": alpha, "rank": selected_rank},
        ),
        (
            "group_shrunk_ridge",
            shrunk_coef,
            {
                "alpha": alpha,
                "shrink_quantile": selected_quantile,
                "threshold": threshold,
                "offdiag_group_density": density,
            },
        ),
        (
            "sparse_plus_lowrank",
            sparse_lowrank_coef,
            {
                "alpha": alpha,
                "shrink_quantile": sparse_quantile,
                "threshold": sparse_threshold,
                "offdiag_group_density": sparse_density,
                "rank": sparse_rank,
            },
        ),
    ]
    models: list[LinearLagModel] = []
    metadata: list[dict] = []
    for name, coef, details in definitions:
        chol = _residual_chol(coef, base_intercept, fit_x, fit_y)
        models.append(
            LinearLagModel(
                name, len(basis), basis, coef, base_intercept, chol, "base"
            )
        )
        metadata.append(details)

    phase_coef, phase_intercept, phase_alpha, phase_fit_x, phase_fit_y = _fit_full(
        train, validation, basis, feature_mode="phase_modulated"
    )
    phase_chol = _residual_chol(
        phase_coef, phase_intercept, phase_fit_x, phase_fit_y
    )
    models.append(
        LinearLagModel(
            "phase_modulated_ridge",
            len(basis),
            basis,
            phase_coef,
            phase_intercept,
            phase_chol,
            "phase_modulated",
        )
    )
    metadata.append({"alpha": phase_alpha})
    return models, metadata


def _synthetic_sequence(
    *, seed: int, smooth: bool, d: int = 12, length: int = 6500
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_lag = 16
    true = np.zeros((max_lag, d, d), dtype=np.float64)
    true[0, np.arange(d), np.arange(d)] = 0.55
    pairs = [(target, source) for target in range(d) for source in range(d) if target != source]
    selected = rng.choice(len(pairs), size=24, replace=False)
    delays = np.asarray([1, 2, 4, 8])
    for index in selected:
        target, source = pairs[int(index)]
        delay = int(rng.choice(delays))
        true[delay - 1, target, source] = rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.16)
    stimulus = np.zeros(length, dtype=np.float64)
    for start in (800, 2200, 4000, 5400):
        stimulus[start : start + 80] = 1.0
    stimulus_loading = rng.normal(scale=0.08, size=d)
    latent = np.zeros((length, d), dtype=np.float64)
    latent[:max_lag] = rng.normal(scale=0.25, size=(max_lag, d))
    for time_index in range(max_lag, length):
        history = latent[time_index - max_lag : time_index][::-1]
        latent[time_index] = (
            np.einsum("ls,lts->t", history, true, optimize=True)
            + stimulus[time_index] * stimulus_loading
            + rng.normal(scale=0.16, size=d)
        )
    observed = latent.copy()
    if smooth:
        alpha = 0.32
        for time_index in range(1, length):
            observed[time_index] = (
                (1.0 - alpha) * observed[time_index - 1] + alpha * latent[time_index]
            )
        observed += rng.normal(scale=0.05, size=observed.shape)
    return observed.astype(np.float32), stimulus.astype(np.float32), true.astype(np.float32)


def synthetic_calibration(replicates: int = 12) -> pd.DataFrame:
    rows = []
    lag = 16
    basis, _ = temporal_basis(lag)
    for smooth in (False, True):
        for replicate in range(replicates):
            observed, stimulus, true = _synthetic_sequence(
                seed=71_000 + 101 * replicate, smooth=smooth
            )
            history = np.lib.stride_tricks.sliding_window_view(
                observed, window_shape=lag, axis=0
            )[:-1].transpose(0, 2, 1)
            stim_history = np.lib.stride_tricks.sliding_window_view(stimulus, lag)[:-1]
            combined = np.concatenate([history, stim_history[..., None]], axis=2)
            target = observed[lag:]
            split1, split2 = 3600, 5000
            d = observed.shape[1]
            train_x, _, _ = basis_features(combined[:split1], basis, d)
            validation_x, _, _ = basis_features(combined[split1:split2], basis, d)
            test_x, _, _ = basis_features(combined[split2:], basis, d)
            alpha = _select_alpha(
                train_x, target[:split1], validation_x, target[split1:split2]
            )
            fitted = _ridge(alpha).fit(
                np.concatenate([train_x, validation_x]), target[:split2]
            )
            prediction = fitted.predict(test_x)
            coef = fitted.coef_[:, : observed.shape[1] * basis.shape[1]].reshape(
                observed.shape[1], observed.shape[1], basis.shape[1]
            )
            estimated = np.einsum("tsk,lk->lts", coef, basis, optimize=True)[::-1]
            d = observed.shape[1]
            off = ~np.eye(d, dtype=bool)
            true_edge = np.max(np.abs(true), axis=0) > 0
            true_edge &= off
            score = np.max(np.abs(estimated), axis=0)
            auroc = roc_auc_score(true_edge[off], score[off])
            true_delays = np.argmax(np.abs(true), axis=0) + 1
            estimated_delays = np.argmax(np.abs(estimated), axis=0) + 1
            edge_delays_true = true_delays[true_edge]
            edge_delays_estimated = estimated_delays[true_edge]
            rows.append(
                {
                    "condition": "calcium_smoothed" if smooth else "latent_observed",
                    "replicate": replicate,
                    "alpha": alpha,
                    "edge_auroc": float(auroc),
                    "lag_mae_frames": float(
                        np.mean(np.abs(edge_delays_true - edge_delays_estimated))
                    ),
                    "lag_within_one_frame": float(
                        np.mean(np.abs(edge_delays_true - edge_delays_estimated) <= 1)
                    ),
                    "test_rmse": float(np.sqrt(np.square(prediction - target[split2:]).mean())),
                }
            )
    return pd.DataFrame(rows)


def _leaderboard(frame: pd.DataFrame, folds: Iterable[int]) -> pd.DataFrame:
    work = frame[frame.fold.isin(tuple(folds))].copy()
    metrics = [
        "energy",
        "energy__stim_balanced",
        "crps",
        "rmse",
        "coverage90",
        "target_spearman",
        "innovation_spearman",
        "offdiag_prediction_rms",
    ]
    rows = []
    for (stage, model, lag), group in work.groupby(["stage", "model", "lag"]):
        row = {"stage": stage, "model": model, "lag": int(lag), "n_folds": len(group)}
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_se"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["stage", "energy_mean", "rmse_mean"]).reset_index(drop=True)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksums(output: Path) -> None:
    paths = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(output)}" for path in paths
        )
        + "\n"
    )


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    folds_frame = pd.read_csv(args.fold_assignments)
    fold_by_worm = dict(
        zip(folds_frame.worm_id.astype(str), folds_frame.outer_fold.astype(int))
    )
    folds = np.asarray([fold_by_worm[worm] for worm in cohort.worm_ids], dtype=np.int64)
    if set(folds.tolist()) != set(range(5)):
        raise RuntimeError("expected the immutable five-fold assignment")
    folds_frame.to_csv(output / "fold_assignments.csv", index=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "p(x[t+1] | neural history, binary stimulus history)",
        "candidate_history_frames": list(args.lags),
        "candidate_history_seconds": [lag / cohort.fps for lag in args.lags],
        "screen_folds": [0, 1, 2],
        "confirmation_folds": [3, 4],
        "alpha_grid": list(ALPHAS),
        "rank_grid": list(RANKS),
        "shrink_quantiles": list(SHRINK_QUANTILES),
        "primary_score": "held-out multivariate energy; lower is better",
        "offdiagonal_gate": "held-out future-innovation Spearman gain over self history",
        "atlas_firewall": "no Randi, Cook, Bentley, SBTG, or neuromodulator reference used here",
        "claim_boundary": "observational predictive dynamics, not anatomical or causal identification",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    synthetic = synthetic_calibration(args.synthetic_replicates)
    synthetic.to_csv(output / "synthetic_delay_recovery.csv", index=False)

    rows: list[dict] = []
    basis_rows: list[dict] = []
    primary_by_lag_fold: dict[tuple[int, int], tuple] = {}
    for lag in args.lags:
        basis, intervals = temporal_basis(lag)
        for index, (lo, hi) in enumerate(intervals):
            basis_rows.append(
                {
                    "lag": lag,
                    "basis_index": index,
                    "source_lag_lo_frames": lo,
                    "source_lag_hi_frames": hi,
                    "source_lag_lo_seconds": lo / cohort.fps,
                    "source_lag_hi_seconds": hi / cohort.fps,
                }
            )
        for fold in range(5):
            print(f"MULTILAG_START stage=lag_screen lag={lag} fold={fold}", flush=True)
            trial_started = time.perf_counter()
            scaler, train, validation, test = _split_windows(cohort, folds, fold, lag)
            persistence, self_model, full_model, details = _fit_primary_models(
                train, validation, basis
            )
            primary_by_lag_fold[(lag, fold)] = (
                scaler,
                train,
                validation,
                test,
                persistence,
                self_model,
                full_model,
                details,
            )
            for model in (persistence, self_model, full_model):
                metrics = evaluate_model(
                    model,
                    test,
                    self_model,
                    eval_rows=args.eval_rows,
                    n_samples=args.samples,
                    seed=args.seed + 1009 * lag + 97 * fold + len(rows),
                )
                metadata = {
                    "fold": fold,
                    "lag": lag,
                    "alpha": None
                    if model.name == "persistence"
                    else details[f"{model.name.removesuffix('_ridge')}_alpha"]
                    if model.name in {"self_ridge", "full_ridge"}
                    else None,
                }
                checkpoint = output / "models" / f"{model.name}__L{lag}__f{fold}.npz"
                _save_model(
                    checkpoint,
                    model,
                    scaler=scaler,
                    neurons=cohort.neurons,
                    metadata=metadata,
                )
                rows.append(
                    {
                        "stage": "lag_screen",
                        "model": model.name,
                        "lag": lag,
                        "lag_seconds": lag / cohort.fps,
                        "fold": fold,
                        "alpha": metadata["alpha"],
                        "checkpoint": str(checkpoint.relative_to(output)),
                        "wall_seconds": time.perf_counter() - trial_started,
                        **metrics,
                    }
                )
            pd.DataFrame(rows).to_csv(output / "per_fold_metrics.csv", index=False)
            print(
                f"MULTILAG_DONE lag={lag} fold={fold} seconds={time.perf_counter()-trial_started:.1f}",
                flush=True,
            )
    pd.DataFrame(basis_rows).to_csv(output / "temporal_basis.csv", index=False)
    frame = pd.DataFrame(rows)
    screen = _leaderboard(frame, [0, 1, 2])
    lag_candidates = screen[
        (screen.stage == "lag_screen") & (screen.model == "full_ridge")
    ]
    selected_lag = int(lag_candidates.sort_values("energy_mean").iloc[0].lag)
    (output / "lag_selection.json").write_text(
        json.dumps(
            {
                "selected_lag": selected_lag,
                "selected_seconds": selected_lag / cohort.fps,
                "screen_folds": [0, 1, 2],
                "selection_metric": "energy_mean",
                "external_references_consulted": False,
                "candidates": lag_candidates.to_dict("records"),
            },
            indent=2,
        )
        + "\n"
    )
    structured_metadata: list[dict] = []
    for fold in range(5):
        print(f"MULTILAG_START stage=structured lag={selected_lag} fold={fold}", flush=True)
        (
            scaler,
            train,
            validation,
            test,
            _persistence,
            self_model,
            _full_model,
            _details,
        ) = primary_by_lag_fold[(selected_lag, fold)]
        basis, _ = temporal_basis(selected_lag)
        models, metadata = _fit_structured_models(train, validation, basis)
        for model, model_metadata in zip(models, metadata):
            metrics = evaluate_model(
                model,
                test,
                self_model,
                eval_rows=args.eval_rows,
                n_samples=args.samples,
                seed=args.seed + 701 * fold + len(rows),
            )
            checkpoint = output / "models" / f"{model.name}__L{selected_lag}__f{fold}.npz"
            _save_model(
                checkpoint,
                model,
                scaler=scaler,
                neurons=cohort.neurons,
                metadata={"fold": fold, "lag": selected_lag, **model_metadata},
            )
            rows.append(
                {
                    "stage": "structured_confirmation",
                    "model": model.name,
                    "lag": selected_lag,
                    "lag_seconds": selected_lag / cohort.fps,
                    "fold": fold,
                    "checkpoint": str(checkpoint.relative_to(output)),
                    **model_metadata,
                    **metrics,
                }
            )
            structured_metadata.append(
                {"model": model.name, "fold": fold, **model_metadata}
            )
        pd.DataFrame(rows).to_csv(output / "per_fold_metrics.csv", index=False)
    frame = pd.DataFrame(rows)
    leaderboard = pd.concat(
        [
            _leaderboard(frame, [0, 1, 2]).assign(evaluation_split="screen"),
            _leaderboard(frame, [3, 4]).assign(evaluation_split="confirmation"),
            _leaderboard(frame, [0, 1, 2, 3, 4]).assign(evaluation_split="full_cv"),
        ],
        ignore_index=True,
    )
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    confirmation = leaderboard[
        (leaderboard.evaluation_split == "confirmation")
        & (
            (leaderboard.stage == "structured_confirmation")
            | (
                (leaderboard.stage == "lag_screen")
                & (leaderboard.model == "full_ridge")
                & (leaderboard.lag == selected_lag)
            )
        )
    ].sort_values("energy_mean")
    winner = str(confirmation.iloc[0].model)
    (output / "winner_selection.json").write_text(
        json.dumps(
            {
                "model": winner,
                "lag": selected_lag,
                "confirmation_folds": [3, 4],
                "selection_metric": "energy_mean",
                "confirmation_leaderboard": confirmation.to_dict("records"),
                "external_references_consulted": False,
            },
            indent=2,
        )
        + "\n"
    )
    pd.DataFrame(structured_metadata).to_csv(
        output / "selected_hyperparameters.csv", index=False
    )
    validation = {
        "status": "pass",
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "folds": sorted(np.unique(folds).tolist()),
        "candidate_lags_complete": sorted(frame[frame.stage == "lag_screen"].lag.unique().tolist())
        == sorted(args.lags),
        "lag_screen_rows": int((frame.stage == "lag_screen").sum()),
        "structured_rows": int((frame.stage == "structured_confirmation").sum()),
        "selected_lag": selected_lag,
        "winner": winner,
        "all_metrics_finite": bool(
            np.isfinite(frame[["energy", "rmse", "crps", "target_spearman"]]).all().all()
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    _checksums(output)
    print(f"MULTILAG_COMPLETE output={output} winner={winner} L={selected_lag}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path(
            "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827/fold_assignments.csv"
        ),
    )
    parser.add_argument("--lags", nargs="+", type=int, default=list(LAGS))
    parser.add_argument("--eval-rows", type=int, default=1600)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--synthetic-replicates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
