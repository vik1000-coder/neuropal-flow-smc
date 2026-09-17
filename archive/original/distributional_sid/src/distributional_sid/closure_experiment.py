from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from .core import atomic_csv, atomic_json, resource_guard
from .experiments import freeze_source_and_environment, write_frozen_config


@dataclass(frozen=True)
class ClosureCell:
    name: str
    dgp: str
    history_length: int
    one_step_spec: str
    null_cell: bool
    failure_mode: str


CELLS = (
    ClosureCell("ar1_closed", "ar1", 1, "linear", True, "none"),
    ClosureCell("ar2_closed", "ar2", 2, "linear", True, "none"),
    ClosureCell("ar2_history_short", "ar2", 1, "linear", False, "insufficient_history"),
    ClosureCell("nonlinear_closed", "nonlinear", 1, "sine", True, "none"),
    ClosureCell("nonlinear_misspecified", "nonlinear", 1, "linear", False, "one_step_misspecification"),
)


@dataclass(frozen=True)
class ClosureConfig:
    stage: str = "confirmation"
    n_trajectories: int = 40
    trajectory_length: int = 220
    burn_in: int = 500
    anchors_per_trajectory: int = 16
    n_folds: int = 3
    horizons: tuple[int, ...] = (2, 4, 8)
    witnesses: tuple[str, ...] = ("mean", "sin_1.0", "sin_2.0")
    rollout_draws: int = 32
    bootstrap_replicates: int = 99
    regression_ridge: float = 1e-5
    riesz_ridge: float = 1e-2


@dataclass
class ClosureDataset:
    histories: np.ndarray
    next_value: np.ndarray
    horizon_values: dict[int, np.ndarray]
    trajectory_id: np.ndarray


def _force_thread_limits() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _simulate_paths(cell: ClosureCell, config: ClosureConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(17_071 + 1_009 * int(seed))
    total = config.burn_in + config.trajectory_length
    x = np.zeros((config.n_trajectories, total + 2), dtype=float)
    x[:, :2] = rng.normal(scale=0.6, size=(config.n_trajectories, 2))
    for t in range(1, total + 1):
        if cell.dgp == "ar1":
            mean = 0.72 * x[:, t]
            sigma = 0.50
        elif cell.dgp == "ar2":
            mean = 0.55 * x[:, t] + 0.30 * x[:, t - 1]
            sigma = 0.50
        elif cell.dgp == "nonlinear":
            mean = 0.60 * x[:, t] + 0.60 * np.sin(1.5 * x[:, t])
            sigma = 0.30
        else:
            raise ValueError(f"unknown DGP {cell.dgp}")
        x[:, t + 1] = mean + sigma * rng.normal(size=config.n_trajectories)
    return x[:, config.burn_in + 1 : config.burn_in + 1 + config.trajectory_length]


def _make_dataset(cell: ClosureCell, config: ClosureConfig, seed: int) -> ClosureDataset:
    paths = _simulate_paths(cell, config, seed)
    max_horizon = max(config.horizons)
    first = cell.history_length - 1
    last = paths.shape[1] - max_horizon - 1
    if last <= first:
        raise ValueError("trajectory too short for the registered horizons")
    base = np.linspace(first, last, config.anchors_per_trajectory, dtype=int)
    histories: list[np.ndarray] = []
    next_values: list[float] = []
    trajectory_ids: list[int] = []
    horizon_values = {horizon: [] for horizon in config.horizons}
    for trajectory in range(config.n_trajectories):
        # A seed-specific cyclic offset prevents every replication using the
        # same within-trajectory phase while retaining registered anchor count.
        available = last - first + 1
        offset = (seed + 7 * trajectory) % max(1, available // config.anchors_per_trajectory)
        anchors = np.minimum(base + offset, last)
        for t in anchors:
            histories.append(np.array([paths[trajectory, t - lag] for lag in range(cell.history_length)]))
            next_values.append(float(paths[trajectory, t + 1]))
            trajectory_ids.append(trajectory)
            for horizon in config.horizons:
                horizon_values[horizon].append(float(paths[trajectory, t + horizon]))
    return ClosureDataset(
        histories=np.asarray(histories, dtype=float),
        next_value=np.asarray(next_values, dtype=float),
        horizon_values={key: np.asarray(value, dtype=float) for key, value in horizon_values.items()},
        trajectory_id=np.asarray(trajectory_ids, dtype=int),
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if axis is None:
        return np.sum(values * weights) / np.sum(weights)
    shape = [1] * values.ndim
    shape[axis] = weights.size
    return np.sum(values * weights.reshape(shape), axis=axis) / np.sum(weights)


def _weighted_ridge(design: np.ndarray, target: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    scale = max(float(np.sum(weights)), 1.0)
    gram = design.T @ (weights[:, None] * design) / scale
    penalty = float(ridge) * np.eye(gram.shape[0])
    penalty[0, 0] *= 0.01
    right = design.T @ (weights[:, None] * target) / scale if target.ndim == 2 else design.T @ (weights * target) / scale
    try:
        return np.linalg.solve(gram + penalty, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + penalty, right, rcond=1e-10)[0]


def _weighted_riesz(
    design: np.ndarray, derivative_design: np.ndarray, weights: np.ndarray, ridge: float
) -> np.ndarray:
    design = np.asarray(design, dtype=float)
    derivative_design = np.asarray(derivative_design, dtype=float)
    weights = np.asarray(weights, dtype=float)
    scale = max(float(np.sum(weights)), 1.0)
    gram = design.T @ (weights[:, None] * design) / scale
    penalty = float(ridge) * np.eye(gram.shape[0])
    penalty[0, 0] *= 0.01
    right = np.sum(weights[:, None] * derivative_design, axis=0) / scale
    try:
        return np.linalg.solve(gram + penalty, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + penalty, right, rcond=1e-10)[0]


@dataclass
class DirectBasis:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, histories: np.ndarray, weights: np.ndarray) -> "DirectBasis":
        center = _weighted_mean(histories, weights, axis=0)
        variance = _weighted_mean((histories - center) ** 2, weights, axis=0)
        return cls(np.asarray(center), np.sqrt(np.maximum(variance, 0.05**2)))

    def transform(self, histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        histories = np.asarray(histories, dtype=float)
        z = (histories - self.center) / self.scale
        columns = [np.ones(histories.shape[0])]
        derivatives = [np.zeros(histories.shape[0])]
        max_power = 3
        for coordinate in range(histories.shape[1]):
            for power in range(1, max_power + 1):
                columns.append(z[:, coordinate] ** power)
                if coordinate == 0:
                    derivatives.append(power * z[:, coordinate] ** (power - 1) / self.scale[0])
                else:
                    derivatives.append(np.zeros(histories.shape[0]))
        if histories.shape[1] > 1:
            columns.append(z[:, 0] * z[:, 1])
            derivatives.append(z[:, 1] / self.scale[0])
        tanh_value = np.tanh(1.5 * histories[:, 0])
        columns.append(tanh_value)
        derivatives.append(1.5 * (1.0 - tanh_value**2))
        sine_value = np.sin(1.5 * histories[:, 0])
        columns.append(sine_value)
        derivatives.append(1.5 * np.cos(1.5 * histories[:, 0]))
        cosine_value = np.cos(1.5 * histories[:, 0])
        columns.append(cosine_value)
        derivatives.append(-1.5 * np.sin(1.5 * histories[:, 0]))
        return np.column_stack(columns), np.column_stack(derivatives)


@dataclass
class OneStepModel:
    spec: str
    coefficient: np.ndarray
    sigma: float
    history_length: int

    def features(self, histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        histories = np.asarray(histories, dtype=float)
        columns = [np.ones(histories.shape[0])]
        gradient_columns = [np.zeros((histories.shape[0], self.history_length))]
        for coordinate in range(self.history_length):
            columns.append(histories[:, coordinate])
            gradient = np.zeros((histories.shape[0], self.history_length))
            gradient[:, coordinate] = 1.0
            gradient_columns.append(gradient)
        if self.spec == "tanh":
            value = np.tanh(1.5 * histories[:, 0])
            columns.append(value)
            gradient = np.zeros((histories.shape[0], self.history_length))
            gradient[:, 0] = 1.5 * (1.0 - value**2)
            gradient_columns.append(gradient)
        elif self.spec == "sine":
            value = np.sin(1.5 * histories[:, 0])
            columns.append(value)
            gradient = np.zeros((histories.shape[0], self.history_length))
            gradient[:, 0] = 1.5 * np.cos(1.5 * histories[:, 0])
            gradient_columns.append(gradient)
        design = np.column_stack(columns)
        gradients = np.stack(gradient_columns, axis=1)
        return design, gradients

    def mean_and_gradient(self, histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design, gradients = self.features(histories)
        return design @ self.coefficient, np.einsum("nph,p->nh", gradients, self.coefficient)


def _fit_one_step(
    histories: np.ndarray,
    next_value: np.ndarray,
    weights: np.ndarray,
    spec: str,
    ridge: float,
) -> OneStepModel:
    prototype = OneStepModel(
        spec,
        np.zeros(histories.shape[1] + 1 + (spec in {"tanh", "sine"})),
        1.0,
        histories.shape[1],
    )
    design, _ = prototype.features(histories)
    coefficient = _weighted_ridge(design, next_value, weights, ridge)
    residual = next_value - design @ coefficient
    sigma = math.sqrt(max(float(_weighted_mean(residual**2, weights)), 1e-8))
    return OneStepModel(spec, coefficient, sigma, histories.shape[1])


def _witness_matrix(values: np.ndarray, witnesses: tuple[str, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    columns = []
    for witness in witnesses:
        if witness == "mean":
            columns.append(values)
        elif witness.startswith("sin_"):
            omega = float(witness.split("_", 1)[1])
            columns.append(np.sin(omega * values))
        else:
            raise ValueError(f"unknown witness {witness}")
    return np.column_stack(columns)


def _rollout_features(
    model: OneStepModel,
    histories: np.ndarray,
    horizons: tuple[int, ...],
    witnesses: tuple[str, ...],
    normal_draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n, draws, max_horizon = normal_draws.shape
    if histories.shape[0] != n or max_horizon < max(horizons):
        raise ValueError("rollout draw shape does not match histories/horizons")
    history = np.repeat(histories[:, None, :], draws, axis=1)
    tangent = np.zeros_like(history)
    tangent[:, :, 0] = 1.0
    values = np.empty((n, len(horizons), len(witnesses)), dtype=float)
    derivatives = np.empty_like(values)
    horizon_index = {horizon: index for index, horizon in enumerate(horizons)}
    for step in range(1, max(horizons) + 1):
        flat_history = history.reshape(n * draws, model.history_length)
        mean, gradient = model.mean_and_gradient(flat_history)
        mean = mean.reshape(n, draws)
        gradient = gradient.reshape(n, draws, model.history_length)
        next_value = mean + model.sigma * normal_draws[:, :, step - 1]
        next_tangent = np.sum(gradient * tangent, axis=2)
        if step in horizon_index:
            hidx = horizon_index[step]
            for widx, witness in enumerate(witnesses):
                if witness == "mean":
                    feature = next_value
                    derivative = next_tangent
                else:
                    omega = float(witness.split("_", 1)[1])
                    feature = np.sin(omega * next_value)
                    derivative = omega * np.cos(omega * next_value) * next_tangent
                values[:, hidx, widx] = np.mean(feature, axis=1)
                derivatives[:, hidx, widx] = np.mean(derivative, axis=1)
        if model.history_length == 1:
            history[:, :, 0] = next_value
            tangent[:, :, 0] = next_tangent
        else:
            history[:, :, 1:] = history[:, :, :-1].copy()
            history[:, :, 0] = next_value
            tangent[:, :, 1:] = tangent[:, :, :-1].copy()
            tangent[:, :, 0] = next_tangent
    return values, derivatives


def _fold_assignment(n_trajectories: int, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(91_117 + int(seed))
    order = rng.permutation(n_trajectories)
    assignment = np.empty(n_trajectories, dtype=int)
    for index, trajectory in enumerate(order):
        assignment[trajectory] = index % n_folds
    return assignment


def _estimate(
    cell: ClosureCell,
    config: ClosureConfig,
    dataset: ClosureDataset,
    trajectory_counts: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    fold = _fold_assignment(config.n_trajectories, config.n_folds, seed)
    shape = (len(config.horizons), len(config.witnesses))
    totals = {name: np.zeros(shape) for name in ("direct", "composed", "naive_outer")}
    total_weight = 0.0
    row_weights = trajectory_counts[dataset.trajectory_id].astype(float)
    for fold_index in range(config.n_folds):
        train = fold[dataset.trajectory_id] != fold_index
        evaluate = ~train
        train_weights = row_weights[train]
        eval_weights = row_weights[evaluate]
        if np.sum(train_weights) <= 0 or np.sum(eval_weights) <= 0:
            raise RuntimeError("bootstrap replicate emptied a cross-fitting fold")
        train_history = dataset.histories[train]
        eval_history = dataset.histories[evaluate]
        one_step = _fit_one_step(
            train_history,
            dataset.next_value[train],
            train_weights,
            cell.one_step_spec,
            config.regression_ridge,
        )
        basis = DirectBasis.fit(train_history, train_weights)
        train_design, train_derivative = basis.transform(train_history)
        eval_design, eval_derivative = basis.transform(eval_history)
        riesz_coefficient = _weighted_riesz(
            train_design, train_derivative, train_weights, config.riesz_ridge
        )
        alpha = eval_design @ riesz_coefficient

        rng = np.random.default_rng(700_001 + 10_007 * int(seed) + 97 * fold_index)
        normal_draws = rng.normal(
            size=(eval_history.shape[0], config.rollout_draws, max(config.horizons))
        )
        composed_value, composed_derivative = _rollout_features(
            one_step, eval_history, config.horizons, config.witnesses, normal_draws
        )
        for hidx, horizon in enumerate(config.horizons):
            train_target = _witness_matrix(dataset.horizon_values[horizon][train], config.witnesses)
            eval_target = _witness_matrix(dataset.horizon_values[horizon][evaluate], config.witnesses)
            coefficient = _weighted_ridge(
                train_design, train_target, train_weights, config.regression_ridge
            )
            prediction = eval_design @ coefficient
            plug_derivative = eval_derivative @ coefficient
            direct_score = plug_derivative + alpha[:, None] * (eval_target - prediction)
            naive_score = composed_derivative[:, hidx, :] + alpha[:, None] * (
                eval_target - composed_value[:, hidx, :]
            )
            totals["direct"][hidx] += np.sum(eval_weights[:, None] * direct_score, axis=0)
            totals["composed"][hidx] += np.sum(
                eval_weights[:, None] * composed_derivative[:, hidx, :], axis=0
            )
            totals["naive_outer"][hidx] += np.sum(eval_weights[:, None] * naive_score, axis=0)
        total_weight += float(np.sum(eval_weights))
    estimates = {name: value / total_weight for name, value in totals.items()}
    estimates["defect"] = estimates["direct"] - estimates["composed"]
    return estimates


def _ar_truth(cell: ClosureCell, config: ClosureConfig) -> dict[str, np.ndarray]:
    if cell.dgp == "ar1":
        transition = np.array([[0.72]])
        covariance = np.array([[0.50**2 / (1.0 - 0.72**2)]])
    else:
        transition = np.array([[0.55, 0.30], [1.0, 0.0]])
        covariance = solve_discrete_lyapunov(transition, np.diag([0.50**2, 0.0]))
    if cell.history_length == transition.shape[0]:
        conditional_state = np.eye(transition.shape[0])
        one_step_beta = transition[0].copy()
    else:
        conditional_state = covariance[:, [0]] / covariance[0, 0]
        one_step_beta = np.array([float(transition[0] @ conditional_state[:, 0])])
    direct = np.empty((len(config.horizons), len(config.witnesses)))
    composed = np.empty_like(direct)
    if cell.history_length == 1:
        composed_transition = np.array([[one_step_beta[0]]])
    else:
        composed_transition = np.vstack([one_step_beta, np.array([1.0, 0.0])])
    marginal_variance = float(covariance[0, 0])
    for hidx, horizon in enumerate(config.horizons):
        beta_direct = float((np.linalg.matrix_power(transition, horizon)[0] @ conditional_state)[0])
        beta_composed = float(np.linalg.matrix_power(composed_transition, horizon)[0, 0])
        for widx, witness in enumerate(config.witnesses):
            if witness == "mean":
                multiplier = 1.0
            else:
                omega = float(witness.split("_", 1)[1])
                multiplier = omega * math.exp(-0.5 * omega**2 * marginal_variance)
            direct[hidx, widx] = multiplier * beta_direct
            composed[hidx, widx] = multiplier * beta_composed
    return {"direct": direct, "composed": composed, "defect": direct - composed}


@lru_cache(maxsize=8)
def _nonlinear_truth_cached(
    horizons: tuple[int, ...], witnesses: tuple[str, ...], rollout_draws: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(33_771)
    n_stationary = 120_000
    x = np.zeros(n_stationary + 1_000)
    for t in range(x.size - 1):
        mean = 0.60 * x[t] + 0.60 * math.sin(1.5 * x[t])
        x[t + 1] = mean + 0.30 * rng.normal()
    stationary = x[1_000:]
    current = stationary[:-1]
    following = stationary[1:]
    linear_design = np.column_stack([np.ones(current.size), current])
    linear_coefficient = np.linalg.lstsq(linear_design, following, rcond=None)[0]
    linear_sigma = float(np.sqrt(np.mean((following - linear_design @ linear_coefficient) ** 2)))
    true_model = OneStepModel("sine", np.array([0.0, 0.60, 0.60]), 0.30, 1)
    linear_model = OneStepModel("linear", linear_coefficient, linear_sigma, 1)
    histories = stationary[-12_000:, None]
    truth_rng = np.random.default_rng(771_991)
    normals = truth_rng.normal(size=(histories.shape[0], rollout_draws, max(horizons)))
    _, true_derivative = _rollout_features(true_model, histories, horizons, witnesses, normals)
    _, linear_derivative = _rollout_features(linear_model, histories, horizons, witnesses, normals)
    return np.mean(true_derivative, axis=0), np.mean(linear_derivative, axis=0)


def _oracle_truth(cell: ClosureCell, config: ClosureConfig) -> dict[str, np.ndarray]:
    if cell.dgp in {"ar1", "ar2"}:
        return _ar_truth(cell, config)
    direct, misspecified = _nonlinear_truth_cached(config.horizons, config.witnesses, 256)
    composed = direct if cell.one_step_spec == "sine" else misspecified
    return {"direct": direct.copy(), "composed": composed.copy(), "defect": direct - composed}


def _basic_interval(point: np.ndarray, bootstrap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower_quantile = np.quantile(bootstrap, 0.025, axis=0)
    upper_quantile = np.quantile(bootstrap, 0.975, axis=0)
    return 2.0 * point - upper_quantile, 2.0 * point - lower_quantile


def run_cell_seed(
    cell: ClosureCell,
    config: ClosureConfig,
    seed: int,
) -> pd.DataFrame:
    started = time.perf_counter()
    dataset = _make_dataset(cell, config, seed)
    counts = np.ones(config.n_trajectories, dtype=int)
    point = _estimate(cell, config, dataset, counts, seed)
    boot = {name: [] for name in point}
    rng = np.random.default_rng(8_000_003 + 101 * int(seed) + sum(map(ord, cell.name)))
    attempts = 0
    while len(boot["defect"]) < config.bootstrap_replicates:
        attempts += 1
        if attempts > 5 * config.bootstrap_replicates:
            raise RuntimeError("too many empty-fold trajectory-bootstrap draws")
        sample = rng.integers(0, config.n_trajectories, size=config.n_trajectories)
        bootstrap_counts = np.bincount(sample, minlength=config.n_trajectories)
        try:
            estimate = _estimate(cell, config, dataset, bootstrap_counts, seed)
        except RuntimeError:
            continue
        for name in boot:
            boot[name].append(estimate[name])
    bootstrap = {name: np.stack(value) for name, value in boot.items()}
    intervals = {name: _basic_interval(point[name], bootstrap[name]) for name in point}
    oracle = _oracle_truth(cell, config)
    elapsed = time.perf_counter() - started
    rows: list[dict[str, object]] = []
    for hidx, horizon in enumerate(config.horizons):
        for widx, witness in enumerate(config.witnesses):
            defect_lower, defect_upper = intervals["defect"]
            direct_lower, direct_upper = intervals["direct"]
            rows.append(
                {
                    "stage": config.stage,
                    "seed": seed,
                    "cell": cell.name,
                    "dgp": cell.dgp,
                    "history_length": cell.history_length,
                    "one_step_spec": cell.one_step_spec,
                    "null_cell": cell.null_cell,
                    "failure_mode": cell.failure_mode,
                    "horizon": horizon,
                    "witness": witness,
                    "direct_estimate": point["direct"][hidx, widx],
                    "composed_estimate": point["composed"][hidx, widx],
                    "defect_estimate": point["defect"][hidx, widx],
                    "naive_outer_estimate": point["naive_outer"][hidx, widx],
                    "naive_minus_direct": point["naive_outer"][hidx, widx] - point["direct"][hidx, widx],
                    "direct_bootstrap_se": np.std(bootstrap["direct"][:, hidx, widx], ddof=1),
                    "composed_bootstrap_se": np.std(bootstrap["composed"][:, hidx, widx], ddof=1),
                    "defect_bootstrap_se": np.std(bootstrap["defect"][:, hidx, widx], ddof=1),
                    "direct_ci_lower": direct_lower[hidx, widx],
                    "direct_ci_upper": direct_upper[hidx, widx],
                    "defect_ci_lower": defect_lower[hidx, widx],
                    "defect_ci_upper": defect_upper[hidx, widx],
                    "reject_zero_defect": bool(
                        defect_lower[hidx, widx] > 0.0 or defect_upper[hidx, widx] < 0.0
                    ),
                    "oracle_direct": oracle["direct"][hidx, widx],
                    "oracle_composed": oracle["composed"][hidx, widx],
                    "oracle_defect": oracle["defect"][hidx, widx],
                    "direct_covered": bool(
                        direct_lower[hidx, widx] <= oracle["direct"][hidx, widx] <= direct_upper[hidx, widx]
                    ),
                    "defect_covered": bool(
                        defect_lower[hidx, widx] <= oracle["defect"][hidx, widx] <= defect_upper[hidx, widx]
                    ),
                    "fit_status": "ok",
                    "elapsed_seconds_cell_seed": elapsed,
                    "bootstrap_replicates": config.bootstrap_replicates,
                    "rollout_draws": config.rollout_draws,
                }
            )
    return pd.DataFrame(rows)


def _summarize(seed_level: pd.DataFrame) -> pd.DataFrame:
    grouped = seed_level.groupby(["cell", "null_cell", "failure_mode", "horizon", "witness"], sort=False)
    rows = []
    for keys, frame in grouped:
        cell, null_cell, failure_mode, horizon, witness = keys
        rows.append(
            {
                "cell": cell,
                "null_cell": bool(null_cell),
                "failure_mode": failure_mode,
                "horizon": int(horizon),
                "witness": witness,
                "seed_count": int(frame.seed.nunique()),
                "mean_direct_estimate": float(frame.direct_estimate.mean()),
                "mean_composed_estimate": float(frame.composed_estimate.mean()),
                "mean_defect_estimate": float(frame.defect_estimate.mean()),
                "oracle_direct": float(frame.oracle_direct.iloc[0]),
                "oracle_composed": float(frame.oracle_composed.iloc[0]),
                "oracle_defect": float(frame.oracle_defect.iloc[0]),
                "defect_rmse": float(np.sqrt(np.mean((frame.defect_estimate - frame.oracle_defect) ** 2))),
                "rejection_rate": float(frame.reject_zero_defect.mean()),
                "direct_coverage": float(frame.direct_covered.mean()),
                "defect_coverage": float(frame.defect_covered.mean()),
                "median_abs_naive_minus_direct": float(np.median(np.abs(frame.naive_minus_direct))),
                "mean_elapsed_seconds_cell_seed": float(frame.elapsed_seconds_cell_seed.mean()),
            }
        )
    return pd.DataFrame(rows)


def _validate(seed_level: pd.DataFrame, config: ClosureConfig, expected_seeds: int) -> dict[str, object]:
    primary = seed_level[seed_level.witness.isin(["mean", "sin_1.0"])].copy()
    expected_rows = expected_seeds * len(CELLS) * len(config.horizons) * len(config.witnesses)
    complete = bool(
        len(seed_level) == expected_rows
        and seed_level.seed.nunique() == expected_seeds
        and np.all(seed_level.fit_status == "ok")
        and np.all(np.isfinite(seed_level.select_dtypes(include=[np.number]).to_numpy()))
    )
    null_fpr = float(primary[primary.null_cell].reject_zero_defect.mean())
    power: dict[str, float] = {}
    for cell in ("ar2_history_short", "nonlinear_misspecified"):
        selected = primary[(primary.cell == cell) & (primary.horizon == 8)]
        power[cell] = float(selected.reject_zero_defect.mean())
    memory = seed_level[(seed_level.cell == "ar2_history_short") & (seed_level.witness == "mean")]
    h2 = float(np.median(np.abs(memory[memory.horizon == 2].defect_estimate)))
    h8 = float(np.median(np.abs(memory[memory.horizon == 8].defect_estimate)))
    alternatives = primary[~primary.null_cell]
    naive_difference = float(np.median(np.abs(alternatives.naive_minus_direct)))
    naive_tolerance = float(0.10 * np.median(np.abs(alternatives.direct_estimate)) + 0.02)
    analytic_ar = primary[primary.dgp.isin(["ar1", "ar2"])]
    direct_coverage = float(analytic_ar.direct_covered.mean())
    gates = {
        "complete_and_finite": complete,
        "closure_null_fpr_at_most_0_10": null_fpr <= 0.10,
        "ar2_short_history_power_at_least_0_80": power["ar2_history_short"] >= 0.80,
        "nonlinear_misspec_power_at_least_0_80": power["nonlinear_misspecified"] >= 0.80,
        "memory_defect_grows_h2_to_h8": h8 > h2,
        "naive_outer_collapses_to_direct": naive_difference <= naive_tolerance,
        "analytic_ar_direct_coverage_0_85_to_1_00": 0.85 <= direct_coverage <= 1.00,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "diagnostics": {
            "expected_rows": expected_rows,
            "actual_rows": len(seed_level),
            "seed_count": int(seed_level.seed.nunique()),
            "closure_null_fpr": null_fpr,
            "horizon8_power": power,
            "memory_median_abs_defect_h2": h2,
            "memory_median_abs_defect_h8": h8,
            "median_abs_naive_minus_direct": naive_difference,
            "naive_tolerance": naive_tolerance,
            "analytic_ar_direct_coverage": direct_coverage,
        },
        "claim_ceiling": (
            "Compositional adequacy or resolved defect at the registered histories, "
            "horizons, witnesses, model classes, and bootstrap resolution; not proof of exact Markov closure."
        ),
    }


def run_experiment(run_dir: Path, config: ClosureConfig, seeds: list[int]) -> dict[str, object]:
    _force_thread_limits()
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": "e12-closure",
        "config": asdict(config),
        "cells": [asdict(cell) for cell in CELLS],
        "seeds": seeds,
        "preregistration": "distributional_sid/E12_CLOSURE_PREREGISTRATION_20260716.md",
        "inference_unit": "independent trajectory; DGP seed for replication summaries",
        "composed_uncertainty": "full trajectory-bootstrap refit with fixed cross-fit grouping",
    }
    write_frozen_config(run_dir, payload)
    freeze_source_and_environment(run_dir)
    for seed in seeds:
        for cell in CELLS:
            output = run_dir / "seed_cells" / f"{cell.name}__seed_{seed}.csv"
            if output.exists():
                print(f"E12 {cell.name} seed={seed} complete", flush=True)
                continue
            resource_guard()
            print(f"E12 {cell.name} seed={seed}", flush=True)
            frame = run_cell_seed(cell, config, seed)
            atomic_csv(output, frame)
    files = sorted((run_dir / "seed_cells").glob("*.csv"))
    seed_level = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    atomic_csv(run_dir / "seed_level.csv", seed_level)
    try:
        seed_level.to_parquet(run_dir / "seed_level.parquet", index=False)
    except (ImportError, ValueError):
        pass
    summary = _summarize(seed_level)
    atomic_csv(run_dir / "summary.csv", summary)
    validation = _validate(seed_level, config, len(seeds))
    atomic_json(run_dir / "validation.json", validation)
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distributional-sid-closure")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=("smoke", "development", "confirmation"), default="confirmation")
    parser.add_argument("--seed-start", type=int, default=6001)
    parser.add_argument("--seed-end", type=int, default=6030)
    parser.add_argument("--n-trajectories", type=int, default=40)
    parser.add_argument("--trajectory-length", type=int, default=220)
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=99)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ClosureConfig(
        stage=args.stage,
        n_trajectories=args.n_trajectories,
        trajectory_length=args.trajectory_length,
        anchors_per_trajectory=args.anchors,
        n_folds=args.folds,
        rollout_draws=args.rollouts,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    validation = run_experiment(
        Path(args.run_dir).resolve(), config, list(range(args.seed_start, args.seed_end + 1))
    )
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
