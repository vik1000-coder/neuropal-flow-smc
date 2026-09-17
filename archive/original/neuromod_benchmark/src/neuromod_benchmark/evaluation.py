"""Capability-aware test evaluation against matching oracle levels."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtr
from scipy.stats import spearmanr

from .features import build_supervised, subset
from .capabilities import validate_channel_contract
from .mechanistic import MechanisticState, one_step_moments
from .mechanistic_dataset import dataset_with_trajectories
from .metrics import (
    counterfactual_metrics,
    energy_score,
    graph_metrics,
    matrix_metrics,
    pit_serial_metrics,
    predictive_metrics,
)
from .rollout_metrics import RolloutMetricConfig, rollout_metrics
from .response_evaluation import PreparedResponsePanel, evaluate_response_panels
from .schema import Dataset, SupervisedData


def _aligned_oracle_matrix(dataset: Dataset, test: SupervisedData, channel: str) -> np.ndarray:
    attribute = {
        "mean": "pointwise_mean_jacobian",
        "logvariance": "pointwise_logvariance_jacobian",
        "tail": "pointwise_tail_jacobian",
        "shape_tail": "pointwise_shape_tail_jacobian",
        "covariance": "pointwise_covariance_jacobian",
        "correlation": "pointwise_correlation_jacobian",
    }[channel]
    matrices = []
    for group, time in zip(test.groups, test.times):
        trajectory = dataset.trajectories[int(group)]
        pointwise = getattr(trajectory, attribute)
        target_time = int(time + test.horizon)
        if pointwise is not None and target_time < len(pointwise):
            matrices.append(pointwise[target_time])
        else:
            average = {
                "mean": trajectory.average_mean_jacobian,
                "logvariance": trajectory.average_logvariance_jacobian,
                "tail": trajectory.average_tail_jacobian,
                "shape_tail": trajectory.average_shape_tail_jacobian,
                "covariance": trajectory.average_covariance_jacobian,
                "correlation": trajectory.average_correlation_jacobian,
            }[channel]
            if average is not None:
                matrices.append(average)
    if not matrices:
        raise ValueError(f"dataset does not expose a {channel} derivative oracle")
    return np.mean(matrices, axis=0)


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}.{name}": value for name, value in values.items()}


def prediction_target_invariance_metric(estimator, test: SupervisedData) -> dict[str, float]:
    """Metamorphic guard against accidental use of test labels in forecasts."""

    count = min(16, len(test.targets))
    mask = np.arange(len(test.targets)) < count
    probe = subset(test, mask)
    mutated = replace(
        probe,
        targets=np.flip(probe.targets, axis=0) + 13.731,
    )
    original_prediction = estimator.predict(probe, n_samples=0)
    mutated_prediction = estimator.predict(mutated, n_samples=0)
    differences = [
        np.max(np.abs(original_prediction.mean - mutated_prediction.mean), initial=0.0),
        np.max(
            np.abs(original_prediction.variance - mutated_prediction.variance),
            initial=0.0,
        ),
    ]
    for name in ("covariance", "correlation"):
        original = getattr(original_prediction, name, None)
        changed = getattr(mutated_prediction, name, None)
        if original is not None or changed is not None:
            if original is None or changed is None:
                raise RuntimeError(f"prediction {name} availability depends on test targets")
            differences.append(np.max(np.abs(original - changed), initial=0.0))
    maximum = float(max(differences, default=0.0))
    if maximum > 1e-10:
        raise RuntimeError(
            f"predictive moments depend on supplied test targets (max change {maximum:.3g})"
        )
    return {"diagnostic.prediction_target_invariance_max_abs": maximum}


def metric_parameter_binding(estimator, dataset: Dataset) -> dict[str, float]:
    """Require estimator-side channel thresholds to match the registered DGP."""

    declared = set(estimator.capabilities.effect_channels)
    metadata = estimator.metadata() if hasattr(estimator, "metadata") else {}
    if "conditional_tail_high_derivative" in declared:
        submitted = metadata.get(
            "tail_threshold_physical", metadata.get("tail_threshold")
        )
        if submitted is None:
            raise ValueError("tail-derivative estimator does not declare its threshold")
        expected = float(getattr(dataset.config, "tail_threshold", 0.35))
        if not np.isclose(float(submitted), expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"tail threshold mismatch: estimator={submitted}, oracle={expected}"
            )
    submitted_shape = metadata.get("shape_tail_z")
    if submitted_shape is not None:
        expected_shape = float(getattr(dataset.config, "shape_tail_z", 2.0))
        if not np.isclose(
            float(submitted_shape), expected_shape, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "shape-tail threshold mismatch: "
                f"estimator={submitted_shape}, oracle={expected_shape}"
            )
    return {"diagnostic.metric_parameter_binding_valid": 1.0}


def oracle_functional_metrics(prediction: Any, test: SupervisedData) -> dict[str, float]:
    if test.oracle_mean is None or test.oracle_variance is None:
        return {}
    result = {
        "oracle.mean_rmse": float(np.sqrt(np.mean((prediction.mean - test.oracle_mean) ** 2))),
        "oracle.variance_log_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        np.log(np.maximum(prediction.variance, 1e-10))
                        - np.log(np.maximum(test.oracle_variance, 1e-10))
                    )
                    ** 2
                )
            )
        ),
        "oracle.mean_bias": float(np.mean(prediction.mean - test.oracle_mean)),
        "oracle.variance_ratio_median": float(
            np.median(prediction.variance / np.maximum(test.oracle_variance, 1e-10))
        ),
    }
    predicted_covariance = getattr(prediction, "covariance", None)
    if predicted_covariance is None:
        predicted_covariance = prediction.metadata.get("conditional_covariance")
    if predicted_covariance is None and prediction.samples is not None:
        samples = np.asarray(prediction.samples, dtype=float)
        if samples.ndim == 3 and samples.shape[1] >= 2:
            centered = samples - np.mean(samples, axis=1, keepdims=True)
            predicted_covariance = np.einsum(
                "rsi,rsj->rij", centered, centered
            ) / (samples.shape[1] - 1)
    if test.oracle_covariance is not None and predicted_covariance is not None:
        truth_covariance = np.asarray(test.oracle_covariance, dtype=float)
        predicted_covariance = np.asarray(predicted_covariance, dtype=float)
        result.update(
            _prefixed(
                "oracle.covariance_continuous",
                matrix_metrics(
                    truth_covariance.reshape(len(truth_covariance), -1),
                    predicted_covariance.reshape(len(predicted_covariance), -1),
                ),
            )
        )
        off_diagonal = ~np.eye(truth_covariance.shape[-1], dtype=bool)
        result.update(
            _prefixed(
                "oracle.covariance_offdiagonal",
                matrix_metrics(
                    truth_covariance[:, off_diagonal],
                    predicted_covariance[:, off_diagonal],
                ),
            )
        )
        predicted_correlation = getattr(prediction, "correlation", None)
        if predicted_correlation is None:
            predicted_correlation = prediction.metadata.get("conditional_correlation")
        if predicted_correlation is None:
            scale = np.sqrt(
                np.maximum(np.diagonal(predicted_covariance, axis1=1, axis2=2), 1e-12)
            )
            predicted_correlation = predicted_covariance / (
                scale[:, :, None] * scale[:, None, :]
            )
        if test.oracle_correlation is not None:
            truth_correlation = np.asarray(test.oracle_correlation, dtype=float)
            predicted_correlation = np.asarray(predicted_correlation, dtype=float)
            result.update(
                _prefixed(
                    "oracle.correlation_continuous",
                    matrix_metrics(
                        truth_correlation.reshape(len(truth_correlation), -1),
                        predicted_correlation.reshape(len(predicted_correlation), -1),
                    ),
                )
            )
            off_diagonal = ~np.eye(truth_correlation.shape[-1], dtype=bool)
            result.update(
                _prefixed(
                    "oracle.correlation_offdiagonal",
                    matrix_metrics(
                        truth_correlation[:, off_diagonal],
                        predicted_correlation[:, off_diagonal],
                    ),
                )
            )
            result["oracle.correlation_offdiagonal_rmse"] = float(
                np.sqrt(
                    np.mean(
                        (
                            predicted_correlation[:, off_diagonal]
                            - truth_correlation[:, off_diagonal]
                        )
                        ** 2
                    )
                )
            )
    return result


def _cdf_at(estimator, data: SupervisedData, threshold: np.ndarray) -> np.ndarray:
    threshold_data = replace(data, targets=np.asarray(threshold, dtype=float))
    prediction = estimator.predict(threshold_data, n_samples=0)
    if prediction.cdf is None:
        raise TypeError(
            f"{getattr(estimator, 'name', type(estimator).__name__)!r} declares a "
            "normalized forecast but does not expose its predictive CDF"
        )
    return np.asarray(prediction.cdf, dtype=float)


def _state_field_summary(
    truth: np.ndarray,
    estimate: np.ndarray,
    *,
    beta_min_fraction: float = 0.05,
) -> dict[str, float]:
    """Integrated field error plus non-cancelling RMS support diagnostics."""

    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    error = estimate - truth
    truth_energy = float(np.mean(truth**2))
    estimate_energy = float(np.mean(estimate**2))
    result = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "truth_rms": float(np.sqrt(truth_energy)),
        "estimate_rms": float(np.sqrt(estimate_energy)),
        "signed_mean_mae": float(
            np.mean(np.abs(np.mean(estimate, axis=0) - np.mean(truth, axis=0)))
        ),
    }
    if truth_energy > 1e-14:
        result["nise"] = float(np.mean(error**2) / truth_energy)
        denominator = np.linalg.norm(truth.ravel()) * np.linalg.norm(estimate.ravel())
        result["field_cosine"] = float(
            np.dot(truth.ravel(), estimate.ravel()) / denominator
            if denominator > 1e-14
            else np.nan
        )
    else:
        result["nise"] = float("nan")
        result["field_cosine"] = float("nan")
        result["null_estimate_max_abs"] = float(
            np.max(np.abs(estimate), initial=0.0)
        )
    truth_rms_map = np.sqrt(np.mean(truth**2, axis=0))
    estimate_rms_map = np.sqrt(np.mean(estimate**2, axis=0))
    result["rms_map_mae"] = float(np.mean(np.abs(estimate_rms_map - truth_rms_map)))
    beta_min = max(
        1e-8,
        beta_min_fraction * float(np.max(truth_rms_map, initial=0.0)),
    )
    support_truth = (truth_rms_map >= beta_min).astype(float)
    support_score = estimate_rms_map
    if support_truth.ndim > 2:
        support_truth = support_truth.reshape(support_truth.shape[0], -1)
        support_score = support_score.reshape(support_score.shape[0], -1)
    result["support_beta_min"] = beta_min
    result.update(
        _prefixed(
            "rms_support",
            graph_metrics(
                support_truth,
                support_score,
                exclude_diagonal=False,
            ),
        )
    )
    return result


def pointwise_state_field_metrics(
    estimator,
    dataset: Dataset,
    test: SupervisedData,
    *,
    max_rows: int = 96,
) -> dict[str, float]:
    """Finite-difference fitted fields on the exact occupied test states.

    Unlike average-Jacobian comparisons, these metrics cannot be passed by a zero
    estimator when equal-and-opposite state-dependent effects cancel in the mean.
    """

    if test.view != "complete_state" or test.horizon != 1:
        return {}
    if estimator.capabilities.predictive_distribution != "normalized":
        return {}
    if estimator.capabilities.latent_state:
        # Recursive latent filters cannot be row-batched without changing their
        # state history. Their explicit parameter/state recovery is scored instead.
        return {}
    n = test.targets.shape[1]
    neural_columns = []
    for source in range(n):
        matches = np.flatnonzero(test.source_index == source)
        if not len(matches):
            return {}
        neural_columns.append(int(matches[0]))
    neural_columns = np.asarray(neural_columns, dtype=int)
    indices = np.arange(len(test.targets))
    if len(indices) > max_rows:
        indices = indices[np.linspace(0, len(indices) - 1, max_rows, dtype=int)]
    probe = subset(test, np.isin(np.arange(len(test.targets)), indices))
    x_scale = np.maximum(np.std(probe.features[:, neural_columns], axis=0), 1e-8)
    y_scale = np.maximum(np.std(probe.targets, axis=0), 1e-8)
    steps = 1e-3 * np.maximum(x_scale, 1.0)
    feature_blocks = []
    for source, column in enumerate(neural_columns):
        plus, minus = probe.features.copy(), probe.features.copy()
        plus[:, column] += steps[source]
        minus[:, column] -= steps[source]
        feature_blocks.extend((plus, minus))
    repeats = len(feature_blocks)
    batched = replace(
        probe,
        features=np.concatenate(feature_blocks),
        targets=np.tile(probe.targets, (repeats, 1)),
        groups=np.tile(probe.groups, repeats),
        trajectory_ids=np.tile(probe.trajectory_ids, repeats),
        times=np.tile(probe.times, repeats),
        oracle_mean=None,
        oracle_variance=None,
        oracle_burst_probability=None,
        oracle_tail_probability=None,
        oracle_covariance=None,
        oracle_correlation=None,
        oracle_shape_tail_probability=None,
    )
    prediction = estimator.predict(batched, n_samples=0)
    rows = len(probe.targets)

    def derivative(values: np.ndarray) -> np.ndarray:
        blocks = np.asarray(values).reshape(n, 2, rows, n)
        source_first = (blocks[:, 0] - blocks[:, 1]) / (
            2.0 * steps[:, None, None]
        )
        return np.transpose(source_first, (1, 2, 0))

    fitted = {
        "mean": derivative(prediction.mean),
        "logvariance": derivative(
            np.log(np.maximum(prediction.variance, 1e-10))
        ),
    }
    fixed_q = float(getattr(dataset.config, "tail_threshold", 0.35))
    fixed_threshold = np.full_like(prediction.mean, fixed_q)
    fitted["tail"] = derivative(1.0 - _cdf_at(estimator, batched, fixed_threshold))
    shape_z = float(getattr(dataset.config, "shape_tail_z", 2.0))
    sd = np.sqrt(np.maximum(prediction.variance, 1e-10))
    cdf_low = _cdf_at(estimator, batched, prediction.mean - shape_z * sd)
    cdf_high = _cdf_at(estimator, batched, prediction.mean + shape_z * sd)
    fitted["shape_tail"] = derivative(cdf_low + 1.0 - cdf_high)

    attribute = {
        "mean": "pointwise_mean_jacobian",
        "logvariance": "pointwise_logvariance_jacobian",
        "tail": "pointwise_tail_jacobian",
        "shape_tail": "pointwise_shape_tail_jacobian",
    }
    truth: dict[str, list[np.ndarray]] = {name: [] for name in attribute}
    for group, time in zip(probe.groups, probe.times):
        trajectory = dataset.trajectories[int(group)]
        target_time = int(time + 1)
        for name, field in attribute.items():
            values = getattr(trajectory, field)
            if values is not None:
                truth[name].append(values[target_time])
    result: dict[str, float] = {}
    for name, values in truth.items():
        if len(values) != rows:
            continue
        truth_field = np.asarray(values)
        estimate_field = fitted[name]
        if name == "mean":
            dimensionless = x_scale[None, :] / y_scale[:, None]
        else:
            dimensionless = x_scale[None, :]
        result.update(
            _prefixed(
                f"state_field.{name}",
                _state_field_summary(
                    truth_field * dimensionless[None, :, :],
                    estimate_field * dimensionless[None, :, :],
                ),
            )
        )
    return result


def transition_operator_probe_metrics(
    prediction: Any,
    test: SupervisedData,
    *,
    tail_threshold: float,
    shape_tail_z: float = 2.0,
    estimator=None,
) -> dict[str, float]:
    """Finite-time operator probes beyond a single mean/variance summary."""

    if test.oracle_mean is None or test.oracle_variance is None:
        return {}
    scale = np.maximum(np.std(test.targets, axis=0), 1e-8)
    linear_error = (prediction.mean - test.oracle_mean) / scale[None, :]
    true_quadratic = test.oracle_mean**2 + test.oracle_variance
    fitted_quadratic = prediction.mean**2 + prediction.variance
    quadratic_scale = np.maximum(np.sqrt(np.mean(true_quadratic**2, axis=0)), 1e-8)
    result = {
        "operator.linear_probe_nrmse": float(np.sqrt(np.mean(linear_error**2))),
        "operator.quadratic_probe_nrmse": float(
            np.sqrt(np.mean(((fitted_quadratic - true_quadratic) / quadratic_scale) ** 2))
        ),
    }
    fitted_tail = None
    if test.oracle_tail_probability is not None:
        if estimator is not None:
            threshold = np.full_like(prediction.mean, tail_threshold)
            fitted_tail = 1.0 - _cdf_at(estimator, test, threshold)
        elif prediction.samples is not None:
            fitted_tail = np.mean(prediction.samples > tail_threshold, axis=1)
    if fitted_tail is not None:
        truth_tail = test.oracle_tail_probability
        result.update(
            {
                "operator.tail_exceedance_rmse": float(
                    np.sqrt(np.mean((fitted_tail - truth_tail) ** 2))
                ),
                "operator.tail_exceedance_bias": float(np.mean(fitted_tail - truth_tail)),
                "operator.tail_exceedance_correlation": float(
                    np.corrcoef(fitted_tail.ravel(), truth_tail.ravel())[0, 1]
                    if np.std(fitted_tail) > 1e-10 and np.std(truth_tail) > 1e-10
                    else np.nan
                ),
            }
        )
    fitted_shape_tail = None
    if test.oracle_shape_tail_probability is not None:
        if estimator is not None:
            sd = np.sqrt(np.maximum(prediction.variance, 1e-10))
            fitted_shape_tail = (
                _cdf_at(estimator, test, prediction.mean - shape_tail_z * sd)
                + 1.0
                - _cdf_at(estimator, test, prediction.mean + shape_tail_z * sd)
            )
        elif prediction.samples is not None:
            standardized = (
                prediction.samples - prediction.mean[:, None, :]
            ) / np.sqrt(np.maximum(prediction.variance[:, None, :], 1e-10))
            fitted_shape_tail = np.mean(
                np.abs(standardized) > shape_tail_z, axis=1
            )
    if fitted_shape_tail is not None:
        truth_shape_tail = test.oracle_shape_tail_probability
        result.update(
            {
                "operator.shape_tail_exceedance_rmse": float(
                    np.sqrt(np.mean((fitted_shape_tail - truth_shape_tail) ** 2))
                ),
                "operator.shape_tail_exceedance_bias": float(
                    np.mean(fitted_shape_tail - truth_shape_tail)
                ),
                "operator.shape_tail_exceedance_correlation": float(
                    np.corrcoef(
                        fitted_shape_tail.ravel(), truth_shape_tail.ravel()
                    )[0, 1]
                    if np.std(fitted_shape_tail) > 1e-10
                    and np.std(truth_shape_tail) > 1e-10
                    else np.nan
                ),
            }
        )
    return result


def bridge_path_metrics(estimator, dataset: Dataset, test: SupervisedData, max_rows=256):
    """Score bridge interior marginals against observed discrete intermediate states."""

    if (
        not estimator.capabilities.recursive_path_samples
        or not hasattr(estimator, "sample_paths")
        or test.horizon < 2
    ):
        return {}
    indices = np.arange(len(test.targets))
    if len(indices) > max_rows:
        indices = indices[np.linspace(0, len(indices) - 1, max_rows, dtype=int)]
    probe = subset(test, np.isin(np.arange(len(test.targets)), indices))
    paths = estimator.sample_paths(
        probe,
        n_paths=16,
        n_steps=test.horizon,
        seed=81_119,
    )
    actual = np.empty((len(indices), test.horizon + 1, test.targets.shape[1]))
    for local, row in enumerate(indices):
        group, time = int(test.groups[row]), int(test.times[row])
        trajectory = dataset.trajectories[group]
        values = trajectory.latent if test.view == "complete_state" else getattr(trajectory, test.view)
        actual[local] = values[time : time + test.horizon + 1]
    step_energy = []
    step_rmse = []
    for step in range(1, test.horizon + 1):
        step_energy.append(energy_score(actual[:, step], paths[:, :, step]))
        step_rmse.append(
            float(np.sqrt(np.mean((paths[:, :, step].mean(axis=1) - actual[:, step]) ** 2)))
        )
    intermediate = slice(0, -1) if len(step_energy) > 1 else slice(None)
    result = {
        "bridge.path_energy_mean": float(np.mean(step_energy)),
        "bridge.path_rmse_mean": float(np.mean(step_rmse)),
        "bridge.interior_energy_mean": float(np.mean(step_energy[intermediate])),
        "bridge.interior_rmse_mean": float(np.mean(step_rmse[intermediate])),
        "bridge.endpoint_energy": float(step_energy[-1]),
        "bridge.reference_path_kl_mean": float(
            np.mean(estimator.path_relative_entropy(probe))
        ),
    }
    # The initial state is fixed by conditioning and is therefore removed before
    # scoring the joint future path law. Center/scale come from training only.
    training_scaler = estimator.endpoint_model_.scaler_
    lags = tuple(lag for lag in (1, 2, 4, 8) if lag < test.horizon)
    result.update(
        rollout_metrics(
            actual[:, 1:],
            paths[:, :, 1:],
            config=RolloutMetricConfig(
                autocovariance_lags=lags,
                sample_interval=float(getattr(dataset.config, "dt_seconds", 1.0)),
                target_scale=tuple(np.asarray(training_scaler.y_scale, dtype=float)),
                center=tuple(np.asarray(training_scaler.y_mean, dtype=float)),
                escape_radius=5.0,
                extreme_threshold=2.0,
                max_energy_draws=16,
            ),
        )
    )
    return result


def effect_metrics(estimator, dataset: Dataset, test: SupervisedData) -> dict[str, float]:
    channels = getattr(estimator, "channels_", {})
    result: dict[str, float] = {}
    truth = {
        "mean": _aligned_oracle_matrix(dataset, test, "mean"),
        "logvariance": _aligned_oracle_matrix(dataset, test, "logvariance"),
        "tail": _aligned_oracle_matrix(dataset, test, "tail"),
    }
    if any(
        tr.pointwise_shape_tail_jacobian is not None
        or tr.average_shape_tail_jacobian is not None
        for tr in dataset.trajectories
    ):
        truth["shape_tail"] = _aligned_oracle_matrix(
            dataset, test, "shape_tail"
        )
    if any(tr.pointwise_covariance_jacobian is not None for tr in dataset.trajectories):
        truth["covariance"] = _aligned_oracle_matrix(dataset, test, "covariance")
    if any(tr.pointwise_correlation_jacobian is not None for tr in dataset.trajectories):
        truth["correlation"] = _aligned_oracle_matrix(dataset, test, "correlation")
    matching = {
        "conditional_mean_derivative": "mean",
        "linear_transition_coefficient": "mean",
        "conditional_log_variance_derivative": "logvariance",
        "conditional_tail_high_derivative": "tail",
        "conditional_shape_tail_derivative": "shape_tail",
        "conditional_covariance_derivative": "covariance",
        "conditional_correlation_derivative": "correlation",
    }
    for estimate_name, truth_name in matching.items():
        if estimate_name not in channels or truth_name not in truth:
            continue
        level = "effect" if test.view == "complete_state" else "secondary_structural_correspondence"
        prefix = f"{level}.{estimate_name}_vs_{truth_name}"
        truth_value = np.asarray(truth[truth_name])
        estimate_value = np.asarray(channels[estimate_name])
        if truth_value.ndim > 2:
            truth_value = truth_value.reshape(-1, truth_value.shape[-1])
            estimate_value = estimate_value.reshape(-1, estimate_value.shape[-1])
        result.update(_prefixed(prefix, matrix_metrics(truth_value, estimate_value)))
        result.update(_prefixed(prefix, graph_metrics(truth_value, estimate_value)))

    # Joint-score statistics are evaluated as coarse localization probes against
    # each channel separately; no quantitative derivative equivalence is claimed.
    for joint in ("joint_score_cross_moment", "joint_squared_score_covariance"):
        if joint not in channels:
            continue
        for truth_name in ("mean", "logvariance", "tail"):
            level = (
                "reduced_form_neural_localization"
                if getattr(estimator, "name", "") == "sbtg_joint_score"
                else "localization"
                if test.view == "complete_state"
                else "secondary_localization"
            )
            result.update(
                _prefixed(
                    f"{level}.{joint}_vs_{truth_name}",
                    graph_metrics(truth[truth_name], channels[joint]),
                )
            )
    marginal = channels.get("marginal_lagged_association")
    if marginal is not None:
        result.update(
            _prefixed("secondary.marginal_association_vs_mean", graph_metrics(truth["mean"], marginal))
        )
    conditional = channels.get("conditional_independence_strength")
    if conditional is not None:
        # Conditional-independence tests recover lagged graph support only under
        # their causal-sufficiency/stationarity/test assumptions.  Even on the
        # complete-state lane this is a localization comparison, not a numerical
        # derivative equivalence.
        level = "localization" if test.view == "complete_state" else "secondary_localization"
        result.update(
            _prefixed(
                f"{level}.conditional_independence_vs_mean",
                graph_metrics(truth["mean"], conditional),
            )
        )
    return result


def _true_modulator(dataset: Dataset, data: SupervisedData) -> np.ndarray:
    rows = []
    for group, time in zip(data.groups, data.times):
        trajectory = dataset.trajectories[int(group)]
        index = min(int(time + data.horizon), len(trajectory.modulator) - 1)
        rows.append(trajectory.modulator[index])
    return np.asarray(rows)


def latent_recovery_metrics(
    estimator,
    dataset: Dataset,
    validation,
    test,
    *,
    include_parameter_recovery: bool = True,
) -> dict[str, float]:
    if not estimator.capabilities.latent_state or not hasattr(estimator, "latent_states"):
        return {}
    learned_validation = np.asarray(estimator.latent_states(validation), dtype=float)
    learned_test = np.asarray(estimator.latent_states(test), dtype=float)
    true_validation = _true_modulator(dataset, validation)
    true_test = _true_modulator(dataset, test)
    metadata = estimator.metadata()
    positive_orientation = metadata.get("latent_orientation") == "positive_concentration"

    def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        if (
            len(left) < 2
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or np.ptp(left) <= 1e-14
            or np.ptp(right) <= 1e-14
        ):
            return float("nan")
        return float(spearmanr(left, right).statistic)

    def finite_mean(values: list[float]) -> float:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.mean(finite)) if len(finite) else float("nan")

    correlation = np.zeros((learned_validation.shape[1], true_validation.shape[1]))
    for learned in range(correlation.shape[0]):
        for truth in range(correlation.shape[1]):
            value = safe_spearman(
                learned_validation[:, learned], true_validation[:, truth]
            )
            if not np.isfinite(value):
                value = 0.0
            correlation[learned, truth] = max(0.0, value) if positive_orientation else abs(value)
    learned_index, truth_index = linear_sum_assignment(-correlation)
    aligned = np.zeros((len(test.targets), len(truth_index)))
    validation_aligned = np.zeros((len(validation.targets), len(truth_index)))
    raw_slopes = []
    for column, (learned, truth) in enumerate(zip(learned_index, truth_index)):
        design = np.column_stack([learned_validation[:, learned], np.ones(len(validation.targets))])
        slope, intercept = np.linalg.lstsq(design, true_validation[:, truth], rcond=None)[0]
        raw_slopes.append(float(slope))
        if positive_orientation and slope < 0:
            slope = 0.0
            intercept = float(np.mean(true_validation[:, truth]))
        validation_aligned[:, column] = slope * learned_validation[:, learned] + intercept
        aligned[:, column] = slope * learned_test[:, learned] + intercept
    target = true_test[:, truth_index]
    residual = aligned - target
    variance = np.var(target, axis=0)
    test_spearman = []
    raw_test_spearman = []
    for column in range(len(truth_index)):
        value = safe_spearman(aligned[:, column], target[:, column])
        test_spearman.append(value)
        raw_test_spearman.append(
            safe_spearman(
                learned_test[:, learned_index[column]], target[:, column]
            )
        )
    result = {
        "latent.matched_fraction_true": float(len(truth_index) / true_test.shape[1]),
        "latent.validation_assignment_abs_spearman": float(
            np.mean(correlation[learned_index, truth_index])
        ),
        "latent.test_spearman": finite_mean(test_spearman),
        "latent.test_physical_orientation_spearman": finite_mean(
            raw_test_spearman
        ),
        "latent.test_nrmse": float(
            np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(variance)), 1e-10)
        ),
        "latent.test_r2": float(
            1.0 - np.sum(residual**2) / max(np.sum((target - target.mean(axis=0)) ** 2), 1e-10)
        ),
        "latent.alignment_negative_slope_fraction": float(
            np.mean(np.asarray(raw_slopes) < 0)
        ),
    }
    if not include_parameter_recovery:
        # In the held-out-worm hierarchy, each worm has distinct kinetic and
        # receptor parameters.  The dataset-level population parameter object is
        # not a valid per-worm oracle, while state trajectories remain exact.
        result["latent.parameter_recovery_omitted_hierarchical_truth"] = 1.0
        return result
    if test.view == "calcium":
        # The compact latent model consumes fluorescence directly and does not
        # contain an identified calcium emission/deconvolution layer.  Its recursive
        # state may still track concentration, but fitted kinetic/receptor tensors
        # are effective observation-scale parameters, not physical neural ones.
        result["latent.parameter_recovery_omitted_unidentified_observation_model"] = 1.0
        return result
    learned_rho = metadata.get("learned_clearance_rho")
    true_rho = getattr(dataset.parameters, "clearance_rho", None)
    dt = getattr(dataset.config, "dt_seconds", None)
    if learned_rho is not None and true_rho is not None and dt is not None:
        learned_rho = np.asarray(learned_rho)[learned_index]
        true_rho = np.asarray(true_rho)[truth_index]
        learned_tau = -float(dt) / np.log(np.clip(learned_rho, 1e-6, 1 - 1e-6))
        true_tau = -float(dt) / np.log(np.clip(true_rho, 1e-6, 1 - 1e-6))
        result["latent.clearance_tau_mae_seconds"] = float(np.mean(np.abs(learned_tau - true_tau)))
        result["latent.clearance_tau_relative_mae"] = float(
            np.mean(np.abs(learned_tau - true_tau) / true_tau)
        )
    learned_release = metadata.get("learned_release_weights")
    true_release = getattr(dataset.parameters, "release_weights", None)
    if learned_release is not None and true_release is not None:
        learned_release = np.asarray(learned_release)[learned_index]
        true_release = np.asarray(true_release)[truth_index]
        learned_pattern = learned_release / np.maximum(
            np.sum(learned_release, axis=1, keepdims=True), 1e-10
        )
        true_pattern = true_release / np.maximum(
            np.sum(true_release, axis=1, keepdims=True), 1e-10
        )
        result["latent.release_pattern_mae"] = float(
            np.mean(np.abs(learned_pattern - true_pattern))
        )
        cosine = np.sum(learned_pattern * true_pattern, axis=1) / np.maximum(
            np.linalg.norm(learned_pattern, axis=1)
            * np.linalg.norm(true_pattern, axis=1),
            1e-10,
        )
        result["latent.release_pattern_cosine"] = float(np.mean(cosine))
    learned_expression = metadata.get("learned_receptor_expression")
    true_expression = getattr(dataset.parameters, "receptor_expression", None)
    if learned_expression is not None and true_expression is not None:
        learned_expression = np.asarray(learned_expression)[:, learned_index]
        true_expression = np.asarray(true_expression)[:, truth_index]
        result["latent.gauge_dependent_receptor_expression_mae"] = float(
            np.mean(np.abs(learned_expression - true_expression))
        )
        result.update(
            _prefixed(
                "latent.receptor_support",
                graph_metrics(true_expression, learned_expression, exclude_diagonal=False),
            )
        )
        result["latent.parameter_recovery_omitted_expression_effect_gauge"] = 1.0
        result["latent.parameter_recovery_omitted_kd_hill_gauge"] = 1.0
    mechanism_pairs = {
        "additive_mean": ("learned_additive_effect", "additive_effect"),
        "synaptic_gain": ("learned_synaptic_effect", "synaptic_effect"),
        "intrinsic_slope": (
            "learned_intrinsic_slope_effect",
            "intrinsic_log_slope_effect",
        ),
        "intrinsic_threshold": (
            "learned_intrinsic_threshold_effect",
            "intrinsic_threshold_effect",
        ),
        "stochastic_dispersion": ("learned_logvariance_effect", "logvariance_effect"),
        "tail_shape": ("learned_tail_logit_effect", "tail_logit_effect"),
    }
    scenario_mechanism = getattr(dataset.config, "mechanism", None)
    active_mechanisms = (
        {
            "additive_mean",
            "synaptic_gain",
            "intrinsic_excitability",
            "innovation_variance",
            "matched_tail",
        }
        if scenario_mechanism == "mixed"
        else set() if scenario_mechanism == "null" else {scenario_mechanism}
    )
    channel_mechanism = {
        "additive_mean": "additive_mean",
        "synaptic_gain": "synaptic_gain",
        "intrinsic_slope": "intrinsic_excitability",
        "intrinsic_threshold": "intrinsic_excitability",
        "stochastic_dispersion": "innovation_variance",
        "tail_shape": "matched_tail",
    }
    learned_energies = {}
    inactive_channel_rms = []
    for channel, (learned_name, truth_name) in mechanism_pairs.items():
        learned_value = metadata.get(learned_name)
        truth_value = getattr(dataset.parameters, truth_name, None)
        if learned_value is None or truth_value is None:
            continue
        learned_value = np.asarray(learned_value)
        truth_value = np.asarray(truth_value)
        channel_active = channel_mechanism[channel] in active_mechanisms
        if not channel_active:
            # Potential tensors are pre-drawn for common-seed comparability, but an
            # inactive tensor is exactly zero in the realized transition law.
            truth_value = np.zeros_like(truth_value)
        if learned_value.ndim == 2:
            learned_aligned = learned_value[:, learned_index]
            truth_aligned = truth_value[:, truth_index]
        elif learned_value.ndim == 3:
            learned_aligned = learned_value[:, :, learned_index]
            truth_aligned = truth_value[:, :, truth_index]
        else:
            continue
        learned_energies[channel] = float(np.mean(learned_aligned**2))
        learned_matrix = learned_aligned.reshape(learned_aligned.shape[0], -1)
        truth_matrix = truth_aligned.reshape(truth_aligned.shape[0], -1)
        result[f"latent.mechanism.{channel}.learned_rms"] = float(
            np.sqrt(np.mean(learned_matrix**2))
        )
        if channel_active:
            result.update(
                _prefixed(
                    f"latent.mechanism.{channel}",
                    matrix_metrics(truth_matrix, learned_matrix),
                )
            )
            result.update(
                _prefixed(
                    f"latent.mechanism.{channel}",
                    graph_metrics(truth_matrix, learned_matrix, exclude_diagonal=False),
                )
            )
        else:
            rms = float(np.sqrt(np.mean(learned_matrix**2)))
            inactive_channel_rms.append(rms)
            result[f"latent.mechanism.{channel}.inactive_rms"] = rms
            result[f"latent.mechanism.{channel}.inactive_mae"] = float(
                np.mean(np.abs(learned_matrix))
            )
            result[f"latent.mechanism.{channel}.inactive_max_abs"] = float(
                np.max(np.abs(learned_matrix), initial=0.0)
            )
    if inactive_channel_rms:
        result["latent.mechanism.inactive_channel_rms_mean"] = float(
            np.mean(inactive_channel_rms)
        )
        result["latent.mechanism.inactive_channel_rms_max"] = float(
            np.max(inactive_channel_rms)
        )
    expected = {
        "additive_mean": "additive_mean",
        "synaptic_gain": "synaptic_gain",
        "intrinsic_excitability": "intrinsic_slope",
        "innovation_variance": "stochastic_dispersion",
        "matched_tail": "tail_shape",
    }.get(scenario_mechanism)
    if learned_energies and expected is not None:
        total_energy = sum(learned_energies.values())
        result["latent.mechanism.correct_channel_energy_fraction"] = float(
            learned_energies.get(expected, 0.0) / max(total_energy, 1e-12)
        )
    return result


def _row_mechanistic_parameters(dataset: Dataset, trajectory, time: int):
    """Apply the transition-aligned controlled ligand drive to release bias."""

    parameters = dataset.parameters
    if not hasattr(parameters, "release_bias"):
        return parameters
    drive = np.atleast_1d(np.asarray(trajectory.stimulus[time], dtype=float))
    if drive.shape != np.asarray(parameters.release_bias).shape:
        raise ValueError("transition stimulus does not match modulator release bias")
    if not np.any(drive):
        return parameters
    return replace(parameters, release_bias=parameters.release_bias + drive)


def physical_modulator_susceptibility_metrics(estimator, dataset, test, max_states=128):
    """Compare physical concentration derivatives on the full-state lane.

    Mean/log-variance/tail derivatives retain the established finite-difference oracle.
    Covariance and correlation use the simulator's exact analytic susceptibilities.  The
    latter are scored continuously with ``matrix_metrics`` before a secondary support
    localization summary is added.
    """

    if test.view != "complete_state" or test.horizon != 1:
        return {}
    if not hasattr(dataset.parameters, "receptor_expression"):
        return {}
    columns = np.asarray(
        [i for i, name in enumerate(test.feature_names) if name.startswith("modulator")], dtype=int
    )
    if not len(columns):
        return {}
    channels = getattr(estimator, "channels_", {})
    channel_features = {
        "mean": channels.get(
            "conditional_mean_derivative_feature",
            channels.get("linear_transition_coefficient_feature"),
        ),
        "logvariance": channels.get("conditional_log_variance_derivative_feature"),
        "tail": channels.get("conditional_tail_high_derivative_feature"),
        "shape_tail": channels.get("conditional_shape_tail_derivative_feature"),
        "covariance": channels.get("conditional_covariance_derivative_feature"),
        "correlation": channels.get("conditional_correlation_derivative_feature"),
    }
    if not any(
        value is not None
        and np.asarray(value).ndim >= 2
        and np.asarray(value).shape[-1] == len(test.feature_names)
        for value in channel_features.values()
    ):
        return {}
    indices = np.arange(len(test.targets))
    if len(indices) > max_states:
        indices = indices[np.linspace(0, len(indices) - 1, max_states, dtype=int)]
    n, k = test.targets.shape[1], len(columns)
    truth_matrix = {
        "mean": np.zeros((n, k)),
        "logvariance": np.zeros((n, k)),
        "tail": np.zeros((n, k)),
        "shape_tail": np.zeros((n, k)),
        "covariance": np.zeros((n, n, k)),
        "correlation": np.zeros((n, n, k)),
    }
    for row in indices:
        group, time = int(test.groups[row]), int(test.times[row])
        trajectory = dataset.trajectories[group]
        neural = trajectory.latent[time]
        concentration = trajectory.modulator[time]
        row_parameters = _row_mechanistic_parameters(
            dataset, trajectory, time
        )
        oracle_center = one_step_moments(
            dataset.config,
            row_parameters,
            MechanisticState(neural, concentration, np.zeros_like(neural)),
        )
        for modulator in range(len(concentration)):
            step = 1e-4 * max(1.0, abs(float(concentration[modulator])))
            plus = concentration.copy()
            minus = concentration.copy()
            plus[modulator] += step
            minus[modulator] = max(1e-8, minus[modulator] - step)
            denominator = plus[modulator] - minus[modulator]
            oracle_plus = one_step_moments(
                dataset.config,
                row_parameters,
                MechanisticState(neural, plus, np.zeros_like(neural)),
            )
            oracle_minus = one_step_moments(
                dataset.config,
                row_parameters,
                MechanisticState(neural, minus, np.zeros_like(neural)),
            )
            truth_matrix["mean"][:, modulator] += (
                oracle_plus.conditional_mean - oracle_minus.conditional_mean
            ) / denominator
            truth_matrix["logvariance"][:, modulator] += (
                    np.log(oracle_plus.conditional_variance)
                    - np.log(oracle_minus.conditional_variance)
            ) / denominator
            truth_matrix["tail"][:, modulator] += (
                oracle_plus.upper_tail_probability
                - oracle_minus.upper_tail_probability
            ) / denominator
            truth_matrix["shape_tail"][:, modulator] += (
                oracle_plus.shape_tail_probability
                - oracle_minus.shape_tail_probability
            ) / denominator
            truth_matrix["covariance"][:, :, modulator] += (
                oracle_center.covariance_concentration_susceptibility[
                    :, :, modulator
                ]
            )
            truth_matrix["correlation"][:, :, modulator] += (
                oracle_center.correlation_concentration_susceptibility[
                    :, :, modulator
                ]
            )
    for name in truth_matrix:
        truth_matrix[name] /= max(len(indices), 1)
    result = {}
    for name, estimate_full in channel_features.items():
        if estimate_full is None:
            continue
        estimate_full = np.asarray(estimate_full)
        if estimate_full.ndim < 2 or estimate_full.shape[-1] != len(test.feature_names):
            continue
        estimate = estimate_full[..., columns]
        truth_continuous = truth_matrix[name]
        if estimate.shape != truth_continuous.shape:
            continue
        estimate_flat = estimate.reshape(-1, k)
        truth_flat = truth_continuous.reshape(-1, k)
        result.update(
            _prefixed(
                f"neuromodulator.physical_{name}_susceptibility",
                matrix_metrics(truth_flat, estimate_flat),
            )
        )
        if name in {"covariance", "correlation"}:
            off_diagonal = np.broadcast_to(
                (~np.eye(n, dtype=bool))[:, :, None],
                truth_continuous.shape,
            )
            result.update(
                _prefixed(
                    f"neuromodulator.physical_{name}_susceptibility_offdiagonal",
                    matrix_metrics(
                        truth_continuous[off_diagonal],
                        estimate[off_diagonal],
                    ),
                )
            )
        result.update(
            _prefixed(
                f"neuromodulator.physical_{name}_support",
                graph_metrics(truth_flat, estimate_flat, exclude_diagonal=False),
            )
        )
    return result


def pointwise_physical_susceptibility_metrics(
    estimator,
    dataset: Dataset,
    test: SupervisedData,
    *,
    max_rows: int = 64,
) -> dict[str, float]:
    """Physical concentration-response fields on common occupied test states."""

    if test.view != "complete_state" or test.horizon != 1:
        return {}
    if estimator.capabilities.predictive_distribution != "normalized":
        return {}
    if estimator.capabilities.latent_state:
        return {}
    if not hasattr(dataset.parameters, "receptor_expression"):
        return {}
    modulator_columns = np.asarray(
        [
            index
            for index, name in enumerate(test.feature_names)
            if name.startswith("modulator")
        ],
        dtype=int,
    )
    if not len(modulator_columns):
        return {}
    indices = np.arange(len(test.targets))
    if len(indices) > max_rows:
        indices = indices[np.linspace(0, len(indices) - 1, max_rows, dtype=int)]
    probe = subset(test, np.isin(np.arange(len(test.targets)), indices))
    n, k, rows = test.targets.shape[1], len(modulator_columns), len(probe.targets)
    m_scale = np.maximum(np.std(probe.features[:, modulator_columns], axis=0), 1e-8)
    y_scale = np.maximum(np.std(probe.targets, axis=0), 1e-8)
    steps = 1e-3 * np.maximum(m_scale, 1.0)
    blocks = []
    for modulator, column in enumerate(modulator_columns):
        plus, minus = probe.features.copy(), probe.features.copy()
        plus[:, column] += steps[modulator]
        minus[:, column] -= steps[modulator]
        blocks.extend((plus, minus))
    repeats = len(blocks)
    batched = replace(
        probe,
        features=np.concatenate(blocks),
        targets=np.tile(probe.targets, (repeats, 1)),
        groups=np.tile(probe.groups, repeats),
        trajectory_ids=np.tile(probe.trajectory_ids, repeats),
        times=np.tile(probe.times, repeats),
        oracle_mean=None,
        oracle_variance=None,
        oracle_burst_probability=None,
        oracle_tail_probability=None,
        oracle_covariance=None,
        oracle_correlation=None,
        oracle_shape_tail_probability=None,
    )
    prediction = estimator.predict(batched, n_samples=0)

    def derivative(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        reshaped = values.reshape((k, 2, rows) + values.shape[1:])
        difference = reshaped[:, 0] - reshaped[:, 1]
        first = difference / (
            2.0 * steps.reshape((k,) + (1,) * (difference.ndim - 1))
        )
        return np.moveaxis(first, 0, -1)

    fitted = {
        "mean": derivative(prediction.mean),
        "logvariance": derivative(
            np.log(np.maximum(prediction.variance, 1e-10))
        ),
    }
    fixed_q = float(getattr(dataset.config, "tail_threshold", 0.35))
    fitted["tail"] = derivative(
        1.0 - _cdf_at(estimator, batched, np.full_like(prediction.mean, fixed_q))
    )
    shape_z = float(getattr(dataset.config, "shape_tail_z", 2.0))
    sd = np.sqrt(np.maximum(prediction.variance, 1e-10))
    fitted["shape_tail"] = derivative(
        _cdf_at(estimator, batched, prediction.mean - shape_z * sd)
        + 1.0
        - _cdf_at(estimator, batched, prediction.mean + shape_z * sd)
    )
    covariance = prediction.covariance
    if covariance is None:
        covariance = prediction.metadata.get("conditional_covariance")
    if covariance is None:
        covariance = np.zeros((len(prediction.mean), n, n))
        diagonal = np.arange(n)
        covariance[:, diagonal, diagonal] = prediction.variance
    covariance = np.asarray(covariance, dtype=float)
    correlation = prediction.correlation
    if correlation is None:
        correlation = prediction.metadata.get("conditional_correlation")
    if correlation is None:
        scale = np.sqrt(
            np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12)
        )
        correlation = covariance / (scale[:, :, None] * scale[:, None, :])
    fitted["covariance"] = derivative(covariance)
    fitted["correlation"] = derivative(np.asarray(correlation, dtype=float))

    truth = {
        "mean": np.zeros((rows, n, k)),
        "logvariance": np.zeros((rows, n, k)),
        "tail": np.zeros((rows, n, k)),
        "shape_tail": np.zeros((rows, n, k)),
        "covariance": np.zeros((rows, n, n, k)),
        "correlation": np.zeros((rows, n, n, k)),
    }
    for local, (group, time) in enumerate(zip(probe.groups, probe.times)):
        trajectory = dataset.trajectories[int(group)]
        neural = trajectory.latent[int(time)]
        concentration = trajectory.modulator[int(time)]
        row_parameters = _row_mechanistic_parameters(
            dataset, trajectory, int(time)
        )
        for modulator in range(k):
            step = 1e-4 * max(1.0, abs(float(concentration[modulator])))
            plus, minus = concentration.copy(), concentration.copy()
            plus[modulator] += step
            minus[modulator] = max(1e-8, minus[modulator] - step)
            denominator = plus[modulator] - minus[modulator]
            oracle_plus = one_step_moments(
                dataset.config,
                row_parameters,
                MechanisticState(neural, plus, np.zeros_like(neural)),
            )
            oracle_minus = one_step_moments(
                dataset.config,
                row_parameters,
                MechanisticState(neural, minus, np.zeros_like(neural)),
            )
            truth["mean"][local, :, modulator] = (
                oracle_plus.conditional_mean - oracle_minus.conditional_mean
            ) / denominator
            truth["logvariance"][local, :, modulator] = (
                np.log(oracle_plus.conditional_variance)
                - np.log(oracle_minus.conditional_variance)
            ) / denominator
            truth["tail"][local, :, modulator] = (
                oracle_plus.upper_tail_probability
                - oracle_minus.upper_tail_probability
            ) / denominator
            truth["shape_tail"][local, :, modulator] = (
                oracle_plus.shape_tail_probability
                - oracle_minus.shape_tail_probability
            ) / denominator
            truth["covariance"][local, :, :, modulator] = (
                oracle_plus.conditional_covariance
                - oracle_minus.conditional_covariance
            ) / denominator
            truth["correlation"][local, :, :, modulator] = (
                oracle_plus.conditional_correlation
                - oracle_minus.conditional_correlation
            ) / denominator
    result = {}
    scales = {
        "mean": m_scale[None, :] / y_scale[:, None],
        "logvariance": m_scale[None, :],
        "tail": m_scale[None, :],
        "shape_tail": m_scale[None, :],
        "covariance": (
            m_scale[None, None, :]
            / (y_scale[:, None, None] * y_scale[None, :, None])
        ),
        "correlation": m_scale[None, None, :],
    }
    for name in truth:
        true_field = truth[name] * scales[name][None, ...]
        fitted_field = fitted[name] * scales[name][None, ...]
        result.update(
            _prefixed(
                f"neuromodulator.physical_{name}_state_field",
                _state_field_summary(true_field, fitted_field),
            )
        )
        if name in {"covariance", "correlation"}:
            off_diagonal = ~np.eye(n, dtype=bool)
            result.update(
                _prefixed(
                    f"neuromodulator.physical_{name}_state_field_offdiagonal",
                    _state_field_summary(
                        true_field[:, off_diagonal, :],
                        fitted_field[:, off_diagonal, :],
                    ),
                )
            )
    return result


def physical_gated_response_metrics(estimator, dataset, test, max_states=32):
    """Recover total modulator-gated response slopes by finite differences.

    The estimand is ``d/dM_k {d E[Y_i(t+1)|H,M] / d x_j(t)}``, evaluated on
    occupied complete states.  It is broader than a direct synaptic parameter: release
    feedback and nonlinear receptor occupancy can also contribute.  Direct mechanism
    typing therefore remains a separate parameter/intervention lane.
    """

    if test.view != "complete_state" or test.horizon != 1:
        return {}
    if estimator.capabilities.predictive_distribution != "normalized":
        return {}
    # The latent SSM's prediction recursively reconstructs hidden concentrations;
    # row-subsampling would alter that recursion. Its explicit learned mechanism
    # tensors are already evaluated in latent_recovery_metrics instead.
    if getattr(estimator, "name", "") == "latent_neuromodulated_ssm":
        return {}
    if not hasattr(dataset.parameters, "receptor_expression"):
        return {}
    modulator_columns = np.asarray(
        [i for i, name in enumerate(test.feature_names) if name.startswith("modulator")],
        dtype=int,
    )
    if not len(modulator_columns):
        return {}
    neural_columns = []
    for source in range(test.targets.shape[1]):
        matches = np.flatnonzero(test.source_index == source)
        if not len(matches):
            return {}
        neural_columns.append(int(matches[0]))
    neural_columns = np.asarray(neural_columns, dtype=int)
    indices = np.arange(len(test.targets))
    if len(indices) > max_states:
        indices = indices[np.linspace(0, len(indices) - 1, max_states, dtype=int)]
    probe = subset(test, np.isin(np.arange(len(test.targets)), indices))
    n, k = test.targets.shape[1], len(modulator_columns)
    estimate_field = np.zeros((len(probe.targets), n, n, k), dtype=float)
    truth_field = np.zeros_like(estimate_field)
    feature_blocks = []
    block_keys = []
    for source, x_column in enumerate(neural_columns):
        dx = 1e-3 * max(float(np.std(probe.features[:, x_column])), 1.0)
        for modulator, m_column in enumerate(modulator_columns):
            dm = 1e-3 * max(float(np.std(probe.features[:, m_column])), 1.0)
            for x_sign, m_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                features = probe.features.copy()
                features[:, x_column] += x_sign * dx
                features[:, m_column] += m_sign * dm
                feature_blocks.append(features)
            block_keys.append((source, modulator, dx, dm))
    # All eligible predictors are row-wise conditional maps. One batched call avoids
    # hundreds of tiny neural-network forward passes per benchmark case.
    repeats = len(feature_blocks)
    batched = replace(
        probe,
        features=np.concatenate(feature_blocks),
        targets=np.tile(probe.targets, (repeats, 1)),
        groups=np.tile(probe.groups, repeats),
        trajectory_ids=np.tile(probe.trajectory_ids, repeats),
        times=np.tile(probe.times, repeats),
        oracle_mean=None,
        oracle_variance=None,
        oracle_burst_probability=None,
        oracle_tail_probability=None,
        oracle_covariance=None,
        oracle_correlation=None,
        oracle_shape_tail_probability=None,
    )
    block_prediction = estimator.predict(batched, n_samples=0).mean.reshape(
        repeats, len(probe.targets), n
    )
    for index, (source, modulator, dx, dm) in enumerate(block_keys):
        start = 4 * index
        predictions = block_prediction[start : start + 4]
        mixed = predictions[0] - predictions[1] - predictions[2] + predictions[3]
        estimate_field[:, :, source, modulator] = mixed / (4.0 * dx * dm)

    for local, row in enumerate(indices):
        group, time = int(test.groups[row]), int(test.times[row])
        trajectory = dataset.trajectories[group]
        neural = trajectory.latent[time]
        concentration = trajectory.modulator[time]
        row_parameters = _row_mechanistic_parameters(
            dataset, trajectory, time
        )
        for modulator in range(k):
            dm = 1e-4 * max(1.0, abs(float(concentration[modulator])))
            plus, minus = concentration.copy(), concentration.copy()
            plus[modulator] += dm
            minus[modulator] = max(1e-8, minus[modulator] - dm)
            denominator = plus[modulator] - minus[modulator]
            oracle_plus = one_step_moments(
                dataset.config,
                row_parameters,
                MechanisticState(neural, plus, np.zeros_like(neural)),
            )
            oracle_minus = one_step_moments(
                dataset.config,
                row_parameters,
                MechanisticState(neural, minus, np.zeros_like(neural)),
            )
            truth_field[local, :, :, modulator] = (
                oracle_plus.mean_jacobian - oracle_minus.mean_jacobian
            ) / denominator
    truth = np.mean(truth_field, axis=0)
    estimate = np.mean(estimate_field, axis=0)

    # Dimensionless comparison makes systems with different physical units
    # commensurate while retaining signed support and tensor orientation.
    x_scale = np.std(probe.features[:, neural_columns], axis=0)
    m_scale = np.std(probe.features[:, modulator_columns], axis=0)
    y_scale = np.std(probe.targets, axis=0)
    scale = (
        np.maximum(x_scale[None, :, None], 1e-8)
        * np.maximum(m_scale[None, None, :], 1e-8)
        / np.maximum(y_scale[:, None, None], 1e-8)
    )
    truth_dimensionless, estimate_dimensionless = truth * scale, estimate * scale
    truth_field_dimensionless = truth_field * scale[None, :, :, :]
    estimate_field_dimensionless = estimate_field * scale[None, :, :, :]
    result = _prefixed(
        "neuromodulator.total_gated_response",
        matrix_metrics(
            truth_dimensionless.reshape(n, -1), estimate_dimensionless.reshape(n, -1)
        ),
    )
    result.update(
        _prefixed(
            "neuromodulator.total_gated_response_support",
            graph_metrics(
                truth_dimensionless.reshape(n, -1),
                estimate_dimensionless.reshape(n, -1),
                exclude_diagonal=False,
            ),
        )
    )
    result.update(
        _prefixed(
            "neuromodulator.total_gated_response_state_field",
            _state_field_summary(
                truth_field_dimensionless,
                estimate_field_dimensionless,
            ),
        )
    )
    diagonal = np.eye(n, dtype=bool)[:, :, None]
    off_diagonal = (~np.eye(n, dtype=bool))[:, :, None]
    for name, mask in (("self", diagonal), ("cross_neuron", off_diagonal)):
        mask = np.broadcast_to(mask, truth_dimensionless.shape)
        result.update(
            _prefixed(
                f"neuromodulator.total_gated_response_{name}",
                matrix_metrics(truth_dimensionless[mask], estimate_dimensionless[mask]),
            )
        )
    return result


def evaluate_estimator(
    estimator,
    dataset: Dataset,
    test: SupervisedData,
    validation: SupervisedData | None = None,
    *,
    history_lags: tuple[int, ...],
    n_samples: int,
    response_panels: tuple[PreparedResponsePanel, ...] = (),
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    result: dict[str, float] = {}
    result.update(metric_parameter_binding(estimator, dataset))
    validate_channel_contract(
        estimator.capabilities,
        getattr(estimator, "channels_", {}),
    )
    capability = estimator.capabilities.predictive_distribution
    prediction = None
    if capability == "normalized":
        prediction = estimator.predict(test, n_samples=n_samples)
        result.update(prediction_target_invariance_metric(estimator, test))
        result.update(_prefixed("predictive", predictive_metrics(prediction, test.targets)))
        pit = prediction.cdf
        if pit is not None:
            result.update(_prefixed("predictive", pit_serial_metrics(pit, test.groups)))
        if test.view == "complete_state" and test.horizon == 1:
            result.update(oracle_functional_metrics(prediction, test))
            result.update(
                pointwise_state_field_metrics(estimator, dataset, test)
            )
            result.update(
                transition_operator_probe_metrics(
                    prediction,
                    test,
                    tail_threshold=float(getattr(dataset.config, "tail_threshold", 0.35)),
                    shape_tail_z=float(getattr(dataset.config, "shape_tail_z", 2.0)),
                    estimator=estimator,
                )
            )
    elif capability == "unnormalized_score":
        score_domain = getattr(estimator, "score_domain", None)
        if score_domain not in {"conditional_outcome", "joint_consecutive_state"}:
            raise TypeError("score estimator lacks a registered score domain")
        result[f"score.test.{score_domain}.training_objective_dsm_risk"] = float(
            estimator.dsm_risk(test, seed=71_117)
        )
        if not hasattr(estimator, "dsm_reference_metrics"):
            raise TypeError("score estimator lacks fixed-reference DSM metrics")
        result.update(
            _prefixed(
                f"score.test.{score_domain}",
                estimator.dsm_reference_metrics(test, seed=71_117),
            )
        )

    if test.horizon == 1:
        result.update(effect_metrics(estimator, dataset, test))
        result.update(physical_modulator_susceptibility_metrics(estimator, dataset, test))
        result.update(
            pointwise_physical_susceptibility_metrics(estimator, dataset, test)
        )
        result.update(physical_gated_response_metrics(estimator, dataset, test))
    if validation is not None:
        result.update(latent_recovery_metrics(estimator, dataset, validation, test))
    result.update(bridge_path_metrics(estimator, dataset, test))
    if test.view == "complete_state" and test.horizon == 1 and response_panels:
        result.update(evaluate_response_panels(estimator, response_panels))

    if prediction is not None and dataset.no_modulation is not None:
        counterfactual_dataset = dataset_with_trajectories(dataset, dataset.no_modulation)
        cf_all = build_supervised(
            counterfactual_dataset,
            view=test.view,
            history_lags=history_lags,
            horizon=test.horizon,
        )
        test_groups = np.unique(test.groups)
        cf_test = subset(cf_all, np.isin(cf_all.groups, test_groups))
        if cf_test.targets.shape == test.targets.shape:
            explicit_intervention = bool(estimator.capabilities.interventions)
            if explicit_intervention and not hasattr(
                estimator, "predict_receptor_knockout"
            ):
                raise TypeError(
                    "estimator declares interventions but has no receptor-knockout API"
                )
            cf_prediction = (
                estimator.predict_receptor_knockout(cf_test, n_samples=0)
                if explicit_intervention
                else estimator.predict(cf_test, n_samples=0)
            )
            if test.oracle_mean is not None and cf_test.oracle_mean is not None:
                result.update(
                    _prefixed(
                        (
                            "intervention.receptor_knockout.arm_history_conditional_mean"
                            if explicit_intervention
                            else "arm_history_environment_transfer.receptor_knockout.conditional_mean"
                        ),
                        counterfactual_metrics(
                            test.oracle_mean,
                            cf_test.oracle_mean,
                            prediction.mean,
                            cf_prediction.mean,
                        ),
                    )
                )
    if prediction is not None and dataset.interventions:
        for intervention_name, trajectories in dataset.interventions.items():
            intervention_dataset = dataset_with_trajectories(dataset, trajectories)
            intervention_all = build_supervised(
                intervention_dataset,
                view=test.view,
                history_lags=history_lags,
                horizon=test.horizon,
            )
            intervention_test = subset(
                intervention_all,
                np.isin(intervention_all.groups, np.unique(test.groups)),
            )
            if intervention_test.targets.shape != test.targets.shape:
                continue
            intervention_prediction = estimator.predict(intervention_test, n_samples=0)
            if (
                test.oracle_mean is not None
                and intervention_test.oracle_mean is not None
            ):
                result.update(
                    _prefixed(
                        f"arm_history_environment_transfer.{intervention_name}.conditional_mean",
                        counterfactual_metrics(
                            test.oracle_mean,
                            intervention_test.oracle_mean,
                            prediction.mean,
                            intervention_prediction.mean,
                        ),
                    )
                )
            result.update(
                _prefixed(
                    f"arm_history_environment_transfer.{intervention_name}.predictive",
                    predictive_metrics(intervention_prediction, intervention_test.targets),
                )
            )
    channels = getattr(estimator, "channels_", {})
    return result, channels
