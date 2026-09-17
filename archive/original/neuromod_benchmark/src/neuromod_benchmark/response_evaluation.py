"""Model-side evaluation on registered common-history response panels."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import ResponseKernelEvaluationSpec, ResponseOperationSpec
from .features import concatenate
from .intervention_response import (
    CommonHistoryPair,
    InterventionSpec,
    ResponseKernel,
    aggregate_response_kernels,
    response_kernel,
    response_kernel_metrics,
    simulate_common_history_pair,
)
from .mechanistic import MechanisticConfig, draw_exogenous_noise
from .schema import Dataset, SupervisedData


@dataclass(frozen=True)
class PreparedResponsePanel:
    identifier: str
    specification: InterventionSpec
    pairs: tuple[CommonHistoryPair, ...]
    baseline: SupervisedData
    intervention: SupervisedData
    truth: ResponseKernel
    post_onset_mask: np.ndarray
    expected_effect: str
    truth_rms_to_innovation_rms: float
    truth_mc_se_to_truth_rms_ratio: float
    transient_decay_estimable_fraction: float
    n_modulators: int


def _active_receptor_targets(dataset: Dataset, modulator: int) -> np.ndarray:
    parameters = dataset.parameters
    mechanism = dataset.config.mechanism
    active = np.zeros(dataset.config.n_neurons, dtype=bool)
    if mechanism in {"additive_mean", "mixed"}:
        active |= np.abs(parameters.additive_effect[:, modulator]) > 0
    if mechanism in {"synaptic_gain", "mixed"}:
        active |= np.any(
            np.abs(parameters.synaptic_effect[:, :, modulator]) > 0, axis=1
        )
    if mechanism in {"intrinsic_excitability", "mixed"}:
        active |= (
            np.abs(parameters.intrinsic_log_slope_effect[:, modulator]) > 0
        ) | (np.abs(parameters.intrinsic_threshold_effect[:, modulator]) > 0)
    # This benchmark scores a conditional-mean response. Dispersion, tail, and
    # correlation effects are scientifically meaningful but cannot satisfy its
    # active-effect precondition merely by being nonzero.
    return np.flatnonzero(
        active & (parameters.receptor_expression[:, modulator] > 0)
    )


def _selected_index(
    candidates: np.ndarray, *, what: str, selection: str
) -> tuple[int, ...]:
    if not len(candidates):
        raise ValueError(f"no {selection.removeprefix('first_')} {what} exists")
    return (int(candidates[0]),)


def _operation(operation: ResponseOperationSpec, dataset: Dataset) -> InterventionSpec:
    if operation.kind == "null":
        return InterventionSpec.null()
    if operation.kind == "ligand_pulse":
        return InterventionSpec.ligand_pulse(
            int(operation.modulator_index),
            amplitude=operation.amplitude,
            duration_steps=int(operation.duration_steps),
        )
    if operation.kind == "receptor_knockout":
        targets = operation.target_neurons
        if not targets:
            modulator = int(operation.modulator_index)
            active = _active_receptor_targets(dataset, modulator)
            if operation.selection == "first_active":
                candidates = active
            elif operation.selection == "first_inactive":
                candidates = np.flatnonzero(
                    dataset.parameters.receptor_expression[
                        :, modulator
                    ]
                    == 0
                )
            elif len(active):
                candidates = active
            elif operation.effect_role == "null":
                candidates = np.flatnonzero(
                    dataset.parameters.receptor_expression[:, modulator] > 0
                )
            else:
                candidates = np.asarray([], dtype=int)
            targets = _selected_index(
                candidates, what="receptor target", selection=str(operation.selection)
            )
        return InterventionSpec.receptor_knockout(
            int(operation.modulator_index),
            targets,
            duration_steps=operation.duration_steps,
        )
    if operation.kind == "release_source_silencing":
        sources = operation.source_neurons
        if not sources:
            modulator = int(operation.modulator_index)
            downstream_active = len(_active_receptor_targets(dataset, modulator)) > 0
            release_sources = np.flatnonzero(
                dataset.parameters.release_weights[modulator] > 0
            )
            if operation.selection == "first_active":
                if not downstream_active:
                    raise ValueError(
                        "registered modulator has no active downstream mean mechanism"
                    )
                candidates = release_sources
            elif operation.selection == "first_inactive":
                candidates = np.flatnonzero(
                    dataset.parameters.release_weights[modulator] == 0
                )
            elif downstream_active:
                candidates = release_sources
            elif operation.effect_role == "null":
                candidates = release_sources
            else:
                candidates = np.asarray([], dtype=int)
            sources = _selected_index(
                candidates, what="release source", selection=str(operation.selection)
            )
        return InterventionSpec.release_source_silencing(
            int(operation.modulator_index),
            sources,
            duration_steps=operation.duration_steps,
        )
    raise ValueError(f"unknown response operation {operation.kind!r}")


def _path_supervised(
    pair: CommonHistoryPair,
    *,
    arm: str,
    history_lags: tuple[int, ...],
    group: int,
) -> SupervisedData:
    path = getattr(pair, arm)
    lags = tuple(sorted(set(int(lag) for lag in history_lags)))
    onset, stop = pair.onset_step, pair.onset_step + pair.horizon
    start = max(lags) - 1
    if onset - max(lags) + 1 < 0:
        raise ValueError("response branch does not contain the requested history")
    rows = []
    # Retain the full valid pre-onset trajectory as filtering context. Stateless
    # predictors simply make extra predictions; recursive latent estimators use
    # these rows to reconstruct the shared branch-point state. Only times >= 0
    # are scored downstream.
    for transition in range(start, stop):
        parts = [path.neural_state[transition - lag + 1] for lag in lags]
        parts.append(path.applied_ligand_drive[transition])
        parts.append(path.concentration_state[transition])
        rows.append(np.concatenate(parts))
    n = path.neural_state.shape[1]
    k = path.concentration_state.shape[1]
    names = [f"x{source}:lag{lag}" for lag in lags for source in range(n)]
    source_index = [source for _lag in lags for source in range(n)]
    if k == 1:
        names.append("stimulus")
    else:
        names.extend(f"stimulus{index}" for index in range(k))
    source_index.extend([-1] * k)
    names.extend(f"modulator{index}:current" for index in range(k))
    source_index.extend([-1] * k)
    length = stop - start
    return SupervisedData(
        features=np.asarray(rows),
        targets=path.neural_state[start + 1 : stop + 1],
        groups=np.full(length, group, dtype=int),
        trajectory_ids=np.full(length, group, dtype=int),
        times=np.arange(start - onset, stop - onset, dtype=int),
        feature_names=tuple(names),
        source_index=np.asarray(source_index, dtype=int),
        view="complete_state",
        horizon=1,
        oracle_mean=path.conditional_mean[start:stop],
        oracle_variance=path.conditional_variance[start:stop],
    )


def _truth_rms_to_baseline_innovation_rms(
    truth: ResponseKernel, pairs: tuple[CommonHistoryPair, ...]
) -> float:
    variances = np.stack(
        [
            pair.baseline.conditional_variance[
                pair.onset_step : pair.onset_step + pair.horizon
            ]
            for pair in pairs
        ],
        axis=0,
    )
    innovation_scale = np.sqrt(np.maximum(np.mean(variances, axis=(0, 1)), 1e-12))
    return float(np.sqrt(np.mean((truth.values / innovation_scale[None, :]) ** 2)))


def prepare_response_panels(
    dataset: Dataset,
    specification: ResponseKernelEvaluationSpec | None,
    *,
    history_lags: tuple[int, ...],
) -> tuple[PreparedResponsePanel, ...]:
    if specification is None:
        return ()
    if not isinstance(dataset.config, MechanisticConfig):
        return ()
    specification.validate()
    if specification.horizon > dataset.config.n_steps:
        raise ValueError("response-kernel horizon exceeds retained simulator length")
    # Every operation reuses pair i's exact named innovation streams. This makes
    # cross-operation contrasts paired rather than adding an avoidable MC layer.
    exogenous_bank = tuple(
        draw_exogenous_noise(
            dataset.config,
            dataset.config.seed * 1_000_003 + 700_001 + pair_index,
        )
        for pair_index in range(specification.n_pairs)
    )
    panels = []
    for operation_spec in specification.operations:
        operation = _operation(operation_spec, dataset)
        operation.validate(dataset.config)
        pairs = tuple(
            simulate_common_history_pair(
                dataset.config,
                operation,
                params=dataset.parameters,
                exogenous=exogenous_bank[pair_index],
                onset_step=dataset.config.burn_in,
                horizon=specification.horizon,
            )
            for pair_index in range(specification.n_pairs)
        )
        baseline = concatenate(
            [
                _path_supervised(
                    pair, arm="baseline", history_lags=history_lags, group=index
                )
                for index, pair in enumerate(pairs)
            ]
        )
        intervention = concatenate(
            [
                _path_supervised(
                    pair, arm="intervention", history_lags=history_lags, group=index
                )
                for index, pair in enumerate(pairs)
            ]
        )
        truth = aggregate_response_kernels(pairs, outcome="conditional_mean")
        truth_ratio = _truth_rms_to_baseline_innovation_rms(truth, pairs)
        threshold = specification.minimum_active_truth_rms_to_innovation_rms
        if operation_spec.effect_role == "active" and truth_ratio < threshold:
            raise ValueError(
                f"response operation {operation_spec.id!r} has dimensionless truth "
                f"RMS/innovation RMS {truth_ratio:.3g}, below the registered beta-min "
                f"{threshold:.3g}"
            )
        if operation_spec.effect_role == "null" and truth_ratio > threshold:
            raise ValueError(
                f"expected-null response operation {operation_spec.id!r} has "
                f"dimensionless truth RMS/innovation RMS {truth_ratio:.3g}, above "
                f"the registered null tolerance {threshold:.3g}"
            )
        truth_rms = float(np.sqrt(np.mean(truth.values**2)))
        truth_se_rms = (
            float(np.sqrt(np.mean(truth.standard_error**2)))
            if truth.standard_error is not None
            else float("nan")
        )
        truth_mc_ratio = (
            truth_se_rms / truth_rms if truth_rms > 1e-12 else float("nan")
        )
        if (
            operation_spec.effect_role == "active"
            and (
                not np.isfinite(truth_mc_ratio)
                or truth_mc_ratio
                > specification.maximum_truth_mc_se_to_truth_rms_ratio
            )
        ):
            raise ValueError(
                f"response operation {operation_spec.id!r} has truth MC-SE/truth "
                f"RMS ratio {truth_mc_ratio:.3g}, above the registered precision "
                f"ceiling {specification.maximum_truth_mc_se_to_truth_rms_ratio:.3g}"
            )
        persistent = bool(
            operation.kind
            in {"receptor_knockout", "release_source_silencing"}
            and operation.duration_steps is None
        )
        transient_decay_fraction = float(
            response_kernel_metrics(
                truth, truth, persistent_operation=persistent
            )["response_kernel.decay_estimable_fraction"]
        )
        if (
            operation_spec.effect_role == "active"
            and not persistent
            and transient_decay_fraction
            < specification.minimum_transient_decay_estimable_fraction
        ):
            raise ValueError(
                f"transient response operation {operation_spec.id!r} has 1/e "
                f"crossing estimable fraction {transient_decay_fraction:.3g}, below "
                "the registered minimum "
                f"{specification.minimum_transient_decay_estimable_fraction:.3g}"
            )
        post_onset_mask = baseline.times >= 0
        if int(np.sum(post_onset_mask)) != specification.n_pairs * specification.horizon:
            raise AssertionError("response panel scoring mask has the wrong size")
        panels.append(
            PreparedResponsePanel(
                identifier=operation_spec.id,
                specification=operation,
                pairs=pairs,
                baseline=baseline,
                intervention=intervention,
                truth=truth,
                post_onset_mask=post_onset_mask,
                expected_effect=operation_spec.effect_role,
                truth_rms_to_innovation_rms=truth_ratio,
                truth_mc_se_to_truth_rms_ratio=truth_mc_ratio,
                transient_decay_estimable_fraction=transient_decay_fraction,
                n_modulators=dataset.config.n_modulators,
            )
        )
    return tuple(panels)


def _estimated_kernel(
    estimator,
    panel: PreparedResponsePanel,
    *,
    explicit_intervention: bool,
) -> tuple[ResponseKernel, np.ndarray]:
    baseline = estimator.predict(panel.baseline, n_samples=0)
    if explicit_intervention:
        intervention = estimator.predict_intervention(
            panel.intervention,
            panel.specification,
            n_samples=0,
        )
    else:
        intervention = estimator.predict(panel.intervention, n_samples=0)
    n_pairs, horizon = len(panel.pairs), panel.truth.values.shape[0]
    difference = intervention.mean - baseline.mean
    if len(difference) != len(panel.post_onset_mask):
        raise ValueError("response prediction does not align with filtering panel")
    per_pair = difference[panel.post_onset_mask].reshape(
        n_pairs, horizon, -1
    )
    standard_error = (
        np.std(per_pair, axis=0, ddof=1) / math.sqrt(n_pairs)
        if n_pairs > 1
        else None
    )
    kernel = ResponseKernel(
        lags=panel.truth.lags.copy(),
        values=np.mean(per_pair, axis=0),
        outcome="conditional_mean",
        dt_seconds=panel.truth.dt_seconds,
        n_pairs=n_pairs,
        standard_error=standard_error,
    )
    kernel.validate()
    return kernel, per_pair


def _history_conditional_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    null_tolerance: float = 1e-12,
) -> dict[str, float]:
    """Aggregate errors computed within each history before population averaging."""

    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.shape != truth.shape or estimate.ndim != 3:
        raise ValueError("per-pair kernels must have shape [pair, lag, output]")
    pair_rmse = np.sqrt(np.mean((estimate - truth) ** 2, axis=(1, 2)))
    pair_truth_rms = np.sqrt(np.mean(truth**2, axis=(1, 2)))
    pair_nrmse = np.divide(
        pair_rmse,
        pair_truth_rms,
        out=np.full_like(pair_rmse, np.nan),
        where=pair_truth_rms > null_tolerance,
    )
    pair_null_leakage = np.asarray(
        [
            (
                float(np.sqrt(np.mean(pair_estimate[null] ** 2)))
                if np.any(null := np.abs(pair_truth) <= null_tolerance)
                else float("nan")
            )
            for pair_estimate, pair_truth in zip(estimate, truth)
        ]
    )

    def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return float("nan"), float("nan")
        standard_error = (
            float(np.std(finite, ddof=1) / math.sqrt(len(finite)))
            if len(finite) > 1
            else float("nan")
        )
        return float(np.mean(finite)), standard_error

    rmse_mean, rmse_se = _mean_and_se(pair_rmse)
    nrmse_mean, nrmse_se = _mean_and_se(pair_nrmse)
    leakage_mean, leakage_se = _mean_and_se(pair_null_leakage)
    return {
        "pairwise_rmse_mean": rmse_mean,
        "pairwise_rmse_standard_error": rmse_se,
        "pairwise_normalized_rmse_mean": nrmse_mean,
        "pairwise_normalized_rmse_standard_error": nrmse_se,
        "pairwise_normalized_rmse_defined_fraction": float(
            np.mean(np.isfinite(pair_nrmse))
        ),
        "pairwise_normalized_rmse_defined_count": float(
            np.sum(np.isfinite(pair_nrmse))
        ),
        "pairwise_null_leakage_rms_mean": leakage_mean,
        "pairwise_null_leakage_rms_standard_error": leakage_se,
    }


def _truth_mc_metrics(truth: ResponseKernel) -> dict[str, float]:
    truth_rms = float(np.sqrt(np.mean(truth.values**2)))
    standard_error_rms = (
        float(np.sqrt(np.mean(truth.standard_error**2)))
        if truth.standard_error is not None
        else float("nan")
    )
    return {
        "truth_monte_carlo_standard_error_rms": standard_error_rms,
        "truth_monte_carlo_standard_error_to_truth_rms_ratio": (
            standard_error_rms / truth_rms if truth_rms > 1e-12 else float("nan")
        ),
    }


def _modulator_mapping_is_anchored(estimator, n_modulators: int) -> bool:
    if n_modulators == 1:
        return True
    metadata = estimator.metadata() if hasattr(estimator, "metadata") else {}
    return bool(
        isinstance(metadata, dict)
        and metadata.get("modulator_label_mapping_anchored", False)
    )


def _common_history_lag1_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    beta_min_fraction: float = 0.05,
) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    error = estimate - truth
    truth_rms = float(np.sqrt(np.mean(truth**2)))
    beta_min = max(1e-12, beta_min_fraction * float(np.max(np.abs(truth))))
    active = np.abs(truth) > beta_min
    null = np.abs(truth) <= 1e-12
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": (
            float(np.sqrt(np.mean(error**2)) / truth_rms)
            if truth_rms > 1e-12
            else float("nan")
        ),
        "truth_rms": truth_rms,
        "estimate_rms": float(np.sqrt(np.mean(estimate**2))),
        "sign_accuracy": (
            float(np.mean(np.sign(estimate[active]) == np.sign(truth[active])))
            if np.any(active)
            else float("nan")
        ),
        "null_leakage_rms": (
            float(np.sqrt(np.mean(estimate[null] ** 2)))
            if np.any(null)
            else float("nan")
        ),
    }


def evaluate_response_panels(
    estimator,
    panels: tuple[PreparedResponsePanel, ...],
) -> dict[str, float]:
    if estimator.capabilities.predictive_distribution != "normalized":
        return {}
    result: dict[str, float] = {}
    for panel in panels:
        explicit = bool(
            estimator.capabilities.interventions
            and panel.specification.kind == "receptor_knockout"
            and hasattr(estimator, "predict_intervention")
            and _modulator_mapping_is_anchored(estimator, panel.n_modulators)
        )
        estimate, per_pair_estimate = _estimated_kernel(
            estimator, panel, explicit_intervention=explicit
        )
        root = (
            f"intervention.{panel.identifier}.arm_history"
            if explicit
            else f"arm_history_environment_transfer.{panel.identifier}"
        )
        persistent = bool(
            panel.specification.kind
            in {"receptor_knockout", "release_source_silencing"}
            and panel.specification.duration_steps is None
        )
        population_prefix = f"{root}.population_mean.response_kernel"
        result.update(
            {
                f"{population_prefix}.{name.removeprefix('response_kernel.')}": value
                for name, value in response_kernel_metrics(
                    estimate, panel.truth, persistent_operation=persistent
                ).items()
            }
        )
        result.update(
            {
                f"{population_prefix}.{name}": value
                for name, value in _truth_mc_metrics(panel.truth).items()
            }
        )
        result[f"{population_prefix}.truth_rms_to_innovation_rms"] = (
            panel.truth_rms_to_innovation_rms
        )
        per_pair_truth = np.stack(
            [response_kernel(pair).values for pair in panel.pairs], axis=0
        )
        conditional_prefix = f"{root}.history_conditional.response_kernel"
        result.update(
            {
                f"{conditional_prefix}.{name}": value
                for name, value in _history_conditional_metrics(
                    per_pair_estimate, per_pair_truth
                ).items()
            }
        )
        if explicit:
            result.update(
                {
                    f"causal_intervention.{panel.identifier}.common_history_lag1.{name}": value
                    for name, value in _common_history_lag1_metrics(
                        per_pair_estimate[:, 0], per_pair_truth[:, 0]
                    ).items()
                }
            )
    return result
