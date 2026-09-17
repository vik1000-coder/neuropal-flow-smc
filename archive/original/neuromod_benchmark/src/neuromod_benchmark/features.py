"""Leakage-resistant supervised views of independent trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Dataset, SupervisedData


def causal_fill(x: np.ndarray) -> np.ndarray:
    """Forward-fill missing observations without peeking into the future."""

    out = np.asarray(x, dtype=float).copy()
    if out.ndim != 2:
        raise ValueError("trajectory must have shape [time, neuron]")
    for j in range(out.shape[1]):
        last = 0.0
        for t in range(out.shape[0]):
            if np.isfinite(out[t, j]):
                last = out[t, j]
            else:
                out[t, j] = last
    return out


def build_supervised(
    dataset: Dataset,
    *,
    view: str,
    history_lags: tuple[int, ...],
    horizon: int = 1,
    include_stimulus: bool = True,
) -> SupervisedData:
    """Build samples without crossing trajectory boundaries.

    Lag 1 denotes the current observation ``x[t]``; lag ``l`` denotes
    ``x[t-l+1]``.  The target is ``x[t+horizon]``.
    """

    if view not in {"complete_state", "latent", "calcium"}:
        raise ValueError(f"unsupported view {view!r}")
    lags = tuple(sorted(set(int(lag) for lag in history_lags)))
    if not lags or lags[0] < 1 or horizon < 1:
        raise ValueError("lags and horizon must be positive")

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    times: list[np.ndarray] = []
    oracle_means: list[np.ndarray] = []
    oracle_variances: list[np.ndarray] = []
    oracle_bursts: list[np.ndarray] = []
    oracle_tails: list[np.ndarray] = []
    oracle_covariances: list[np.ndarray] = []
    oracle_correlations: list[np.ndarray] = []
    oracle_shape_tails: list[np.ndarray] = []
    n = dataset.config.n_neurons
    example_stimulus = np.asarray(dataset.trajectories[0].stimulus)
    stimulus_dim = 1 if example_stimulus.ndim == 1 else example_stimulus.shape[1]
    names = [f"x{source}:lag{lag}" for lag in lags for source in range(n)]
    source_index = [source for _lag in lags for source in range(n)]
    if include_stimulus:
        if stimulus_dim == 1:
            names.append("stimulus")
        else:
            names.extend(f"stimulus{index}" for index in range(stimulus_dim))
        source_index.extend([-1] * stimulus_dim)
    if view == "complete_state":
        for modulator in range(dataset.config.n_modulators):
            names.append(f"modulator{modulator}:current")
            source_index.append(-1)

    start = max(lags) - 1
    for group, trajectory in enumerate(dataset.trajectories):
        target_view = "latent" if view == "complete_state" else view
        raw_x = np.asarray(getattr(trajectory, target_view), dtype=float)
        x = causal_fill(raw_x)
        stop = x.shape[0] - horizon
        if stop <= start:
            continue
        rows = []
        for t in range(start, stop):
            parts = [x[t - lag + 1] for lag in lags]
            if include_stimulus:
                parts.append(np.atleast_1d(trajectory.stimulus[t]))
            if view == "complete_state":
                parts.append(trajectory.modulator[t])
            rows.append(np.concatenate(parts))
        feature = np.asarray(rows)
        target_indices = np.arange(start, stop) + horizon
        target = raw_x[target_indices]
        # Histories may be causally forward-filled, but unobserved future values
        # are never converted into pseudo-outcomes. Multivariate likelihoods use
        # rows for which every target neuron was actually observed.
        observed_target = np.all(np.isfinite(target), axis=1)
        feature = feature[observed_target]
        target = target[observed_target]
        target_indices = target_indices[observed_target]
        row_times = np.arange(start, stop, dtype=int)[observed_target]
        if not len(feature):
            continue
        features.append(feature)
        targets.append(target)
        groups.append(np.full(len(feature), group, dtype=int))
        times.append(row_times)
        if horizon == 1 and target_view == "latent":
            oracle_index = target_indices
            oracle_means.append(trajectory.conditional_mean[oracle_index])
            oracle_variances.append(trajectory.conditional_variance[oracle_index])
            oracle_bursts.append(trajectory.burst_probability[oracle_index])
            if trajectory.upper_tail_probability is not None:
                oracle_tails.append(trajectory.upper_tail_probability[oracle_index])
            if trajectory.conditional_covariance is not None:
                oracle_covariances.append(
                    trajectory.conditional_covariance[oracle_index]
                )
            if trajectory.conditional_correlation is not None:
                oracle_correlations.append(
                    trajectory.conditional_correlation[oracle_index]
                )
            if trajectory.shape_tail_probability is not None:
                oracle_shape_tails.append(
                    trajectory.shape_tail_probability[oracle_index]
                )

    if not features:
        raise ValueError("no trajectory is long enough for the requested history/horizon")
    group_array = np.concatenate(groups)
    return SupervisedData(
        features=np.concatenate(features),
        targets=np.concatenate(targets),
        groups=group_array,
        trajectory_ids=group_array.copy(),
        times=np.concatenate(times),
        feature_names=tuple(names),
        source_index=np.asarray(source_index, dtype=int),
        view=view,
        horizon=horizon,
        oracle_mean=np.concatenate(oracle_means) if oracle_means else None,
        oracle_variance=np.concatenate(oracle_variances) if oracle_variances else None,
        oracle_burst_probability=np.concatenate(oracle_bursts) if oracle_bursts else None,
        oracle_tail_probability=np.concatenate(oracle_tails) if oracle_tails else None,
        oracle_covariance=(
            np.concatenate(oracle_covariances) if oracle_covariances else None
        ),
        oracle_correlation=(
            np.concatenate(oracle_correlations) if oracle_correlations else None
        ),
        oracle_shape_tail_probability=(
            np.concatenate(oracle_shape_tails) if oracle_shape_tails else None
        ),
    )


@dataclass(frozen=True)
class GroupSplit:
    train_groups: np.ndarray
    validation_groups: np.ndarray
    test_groups: np.ndarray

    def masks(self, data: SupervisedData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.isin(data.groups, self.train_groups),
            np.isin(data.groups, self.validation_groups),
            np.isin(data.groups, self.test_groups),
        )


def grouped_split(
    groups: np.ndarray,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> GroupSplit:
    """Split whole trajectories/worms, never neighboring time samples."""

    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("need at least three independent groups")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_test = max(1, int(round(test_fraction * len(unique))))
    n_validation = max(1, int(round(validation_fraction * len(unique))))
    if n_test + n_validation >= len(unique):
        n_test = n_validation = 1
    test = shuffled[:n_test]
    validation = shuffled[n_test : n_test + n_validation]
    train = shuffled[n_test + n_validation :]
    if not len(train):
        raise ValueError("split leaves no training groups")
    return GroupSplit(train, validation, test)


def subset(data: SupervisedData, mask: np.ndarray) -> SupervisedData:
    mask = np.asarray(mask, dtype=bool)
    return SupervisedData(
        features=data.features[mask],
        targets=data.targets[mask],
        groups=data.groups[mask],
        trajectory_ids=data.trajectory_ids[mask],
        times=data.times[mask],
        feature_names=data.feature_names,
        source_index=data.source_index,
        view=data.view,
        horizon=data.horizon,
        oracle_mean=None if data.oracle_mean is None else data.oracle_mean[mask],
        oracle_variance=None if data.oracle_variance is None else data.oracle_variance[mask],
        oracle_burst_probability=(
            None if data.oracle_burst_probability is None else data.oracle_burst_probability[mask]
        ),
        oracle_tail_probability=(
            None if data.oracle_tail_probability is None else data.oracle_tail_probability[mask]
        ),
        oracle_covariance=(
            None if data.oracle_covariance is None else data.oracle_covariance[mask]
        ),
        oracle_correlation=(
            None if data.oracle_correlation is None else data.oracle_correlation[mask]
        ),
        oracle_shape_tail_probability=(
            None
            if data.oracle_shape_tail_probability is None
            else data.oracle_shape_tail_probability[mask]
        ),
    )


def concatenate(parts: list[SupervisedData]) -> SupervisedData:
    if not parts:
        raise ValueError("cannot concatenate an empty collection")
    first = parts[0]
    if any(part.feature_names != first.feature_names for part in parts):
        raise ValueError("feature schemas differ")
    return SupervisedData(
        features=np.concatenate([p.features for p in parts]),
        targets=np.concatenate([p.targets for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        trajectory_ids=np.concatenate([p.trajectory_ids for p in parts]),
        times=np.concatenate([p.times for p in parts]),
        feature_names=first.feature_names,
        source_index=first.source_index,
        view=first.view,
        horizon=first.horizon,
        oracle_mean=(
            None
            if any(p.oracle_mean is None for p in parts)
            else np.concatenate([p.oracle_mean for p in parts])
        ),
        oracle_variance=(
            None
            if any(p.oracle_variance is None for p in parts)
            else np.concatenate([p.oracle_variance for p in parts])
        ),
        oracle_burst_probability=(
            None
            if any(p.oracle_burst_probability is None for p in parts)
            else np.concatenate([p.oracle_burst_probability for p in parts])
        ),
        oracle_tail_probability=(
            None
            if any(p.oracle_tail_probability is None for p in parts)
            else np.concatenate([p.oracle_tail_probability for p in parts])
        ),
        oracle_covariance=(
            None
            if any(p.oracle_covariance is None for p in parts)
            else np.concatenate([p.oracle_covariance for p in parts])
        ),
        oracle_correlation=(
            None
            if any(p.oracle_correlation is None for p in parts)
            else np.concatenate([p.oracle_correlation for p in parts])
        ),
        oracle_shape_tail_probability=(
            None
            if any(p.oracle_shape_tail_probability is None for p in parts)
            else np.concatenate([p.oracle_shape_tail_probability for p in parts])
        ),
    )
