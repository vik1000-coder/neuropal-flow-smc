"""Long-horizon diagnostics for stochastic trajectory forecasts.

These metrics operate on *joint* rollout samples rather than independently sampled
one-step predictions.  The public shape contract is

``observed_paths[case, time, target]`` and
``forecast_samples[case, draw, time, target]``.

Each case is a forecast origin. Cases may be independent episodes or dependent
origins within an episode; dependence does not change the point estimand, but any
uncertainty calculation must cluster at the independent episode/worm level (and
overlapping origins must never be counted as independent replicates). Draws within
a case must be conditionally iid (or exchangeable) joint trajectories from the
same forecast. In particular, independently resampling every time point does not
satisfy the contract because it destroys the temporal law that these metrics are
intended to test.

The path energy score is a proper score for the finite-dimensional block law.
Autocovariance, spectral, escape-rate, and extreme-frequency errors are targeted
diagnostics, not proper scores for the complete path distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


EPS = 1e-12
ROLLOUT_SAMPLE_CAPABILITY = "recursive_path_samples"


@dataclass(frozen=True)
class RolloutSampleContract:
    """Machine-readable axes; case independence is declared in run metadata."""

    capability: str = ROLLOUT_SAMPLE_CAPABILITY
    observed_axes: tuple[str, ...] = ("case", "time", "target")
    forecast_axes: tuple[str, ...] = ("case", "draw", "time", "target")
    minimum_cases: int = 1
    minimum_times: int = 2
    minimum_draws: int = 2


ROLLOUT_CONTRACT = RolloutSampleContract()


@dataclass(frozen=True)
class RolloutMetricConfig:
    """Precommitted settings for a rollout scorecard.

    ``target_scale`` and ``center`` are either scalars or one value per target.
    They should be fixed from training data, simulator units, or an external
    scientific convention.  Estimating them from evaluation outcomes would make
    comparisons optimistic and, for the energy score, can compromise propriety.

    ``escape_radius`` is a threshold on standardized RMS amplitude across targets.
    ``extreme_threshold`` is a per-target standardized threshold.
    """

    block_length: int | None = None
    block_stride: int | None = None
    autocovariance_lags: tuple[int, ...] = (1, 2, 4, 8)
    sample_interval: float = 1.0
    target_scale: float | tuple[float, ...] = 1.0
    center: float | tuple[float, ...] = 0.0
    escape_radius: float = 5.0
    extreme_threshold: float = 2.0
    extreme_sidedness: Literal["two_sided", "upper"] = "two_sided"
    spectral_window: Literal["hann", "rectangular"] = "hann"
    max_energy_draws: int | None = 64

    def __post_init__(self) -> None:
        if self.block_length is not None and self.block_length < 2:
            raise ValueError("block_length must be at least two")
        if self.block_stride is not None and self.block_stride < 1:
            raise ValueError("block_stride must be positive")
        if not self.autocovariance_lags:
            raise ValueError("autocovariance_lags cannot be empty")
        if any(not isinstance(lag, int) or lag < 1 for lag in self.autocovariance_lags):
            raise ValueError("autocovariance lags must be positive integers")
        if len(set(self.autocovariance_lags)) != len(self.autocovariance_lags):
            raise ValueError("autocovariance lags must be unique")
        if not np.isfinite(self.sample_interval) or self.sample_interval <= 0:
            raise ValueError("sample_interval must be finite and positive")
        if not np.isfinite(self.escape_radius) or self.escape_radius <= 0:
            raise ValueError("escape_radius must be finite and positive")
        if not np.isfinite(self.extreme_threshold) or self.extreme_threshold <= 0:
            raise ValueError("extreme_threshold must be finite and positive")
        if self.max_energy_draws is not None and self.max_energy_draws < 2:
            raise ValueError("max_energy_draws must be at least two")


def validate_rollout_samples(
    observed_paths: np.ndarray,
    forecast_samples: np.ndarray,
    *,
    contract: RolloutSampleContract = ROLLOUT_CONTRACT,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return floating-point arrays under the rollout contract."""

    observed = np.asarray(observed_paths, dtype=float)
    forecast = np.asarray(forecast_samples, dtype=float)
    if observed.ndim != len(contract.observed_axes):
        raise ValueError(
            "observed_paths must have shape [case, time, target]; "
            f"received {observed.shape}"
        )
    if forecast.ndim != len(contract.forecast_axes):
        raise ValueError(
            "forecast_samples must have shape [case, draw, time, target]; "
            f"received {forecast.shape}"
        )
    if (
        forecast.shape[0] != observed.shape[0]
        or forecast.shape[2] != observed.shape[1]
        or forecast.shape[3] != observed.shape[2]
    ):
        raise ValueError(
            "forecast case, time, and target dimensions must match observed_paths"
        )
    if observed.shape[0] < contract.minimum_cases:
        raise ValueError(f"at least {contract.minimum_cases} case is required")
    if observed.shape[1] < contract.minimum_times:
        raise ValueError(f"at least {contract.minimum_times} times are required")
    if observed.shape[2] < 1:
        raise ValueError("at least one target is required")
    if forecast.shape[1] < contract.minimum_draws:
        raise ValueError(
            f"at least {contract.minimum_draws} forecast draws are required for "
            "the fair energy score"
        )
    if not np.all(np.isfinite(observed)):
        raise ValueError("observed_paths contains non-finite values")
    if not np.all(np.isfinite(forecast)):
        raise ValueError("forecast_samples contains non-finite values")
    return observed, forecast


def _target_vector(
    value: float | tuple[float, ...] | np.ndarray,
    n_targets: int,
    *,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n_targets, float(array))
    if array.shape != (n_targets,):
        raise ValueError(f"{name} must be scalar or have one value per target")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _validated_scale(
    target_scale: float | tuple[float, ...] | np.ndarray, n_targets: int
) -> np.ndarray:
    return _target_vector(target_scale, n_targets, name="target_scale", positive=True)


def fair_path_energy_score(
    observed_paths: np.ndarray,
    forecast_samples: np.ndarray,
    *,
    block_length: int | None = None,
    stride: int | None = None,
    target_scale: float | tuple[float, ...] | np.ndarray = 1.0,
    max_draws: int | None = 64,
    normalize_dimension: bool = True,
) -> float:
    r"""Fair finite-ensemble energy score on flattened path blocks.

    For an observed block :math:`y` and ensemble :math:`X_1,\ldots,X_m`, the
    estimator is

    .. math::

       m^{-1}\sum_j\|X_j-y\|
       - \{m(m-1)\}^{-1}\sum_{j<k}\|X_j-X_k\|.

    This is the U-statistic form: no zero-distance self-pairs are included.  Its
    expectation is the population energy score, although a finite realization can
    be negative.  Scaling by a fixed positive constant (including the optional
    square-root dimension normalization) preserves propriety.
    """

    observed, forecast = validate_rollout_samples(observed_paths, forecast_samples)
    n_cases, n_times, n_targets = observed.shape
    scale = _validated_scale(target_scale, n_targets)
    length = n_times if block_length is None else int(block_length)
    if length < 2 or length > n_times:
        raise ValueError("block_length must be between two and the rollout length")
    step = length if stride is None else int(stride)
    if step < 1:
        raise ValueError("stride must be positive")
    if max_draws is not None:
        if max_draws < 2:
            raise ValueError("max_draws must be at least two")
        forecast = forecast[:, :max_draws]
    n_draws = forecast.shape[1]
    if n_draws < 2:
        raise ValueError("fair path energy score requires at least two draws")

    normalizer = np.sqrt(length * n_targets) if normalize_dimension else 1.0
    scores: list[np.ndarray] = []
    for start in range(0, n_times - length + 1, step):
        stop = start + length
        truth = (observed[:, start:stop, :] / scale).reshape(n_cases, -1)
        draws = (forecast[:, :, start:stop, :] / scale).reshape(
            n_cases, n_draws, -1
        )
        first = np.linalg.norm(draws - truth[:, None, :], axis=2).mean(axis=1)

        # Accumulating unordered pairs avoids materializing [case, draw, draw,
        # block_dimension], which is important for long neural rollouts.
        pair_sum = np.zeros(n_cases, dtype=float)
        for left in range(n_draws - 1):
            distances = np.linalg.norm(
                draws[:, left + 1 :, :] - draws[:, left : left + 1, :], axis=2
            )
            pair_sum += distances.sum(axis=1)
        second = pair_sum / (n_draws * (n_draws - 1))
        scores.append((first - second) / normalizer)
    if not scores:
        raise ValueError("block settings produced no complete path blocks")
    return float(np.mean(np.concatenate(scores)))


def _forecast_as_trajectories(forecast: np.ndarray) -> np.ndarray:
    # Preserve each draw as its own trajectory.  Reshaping [case, time] together
    # would create false lag pairs across case/draw boundaries.
    return np.transpose(forecast, (0, 1, 2, 3)).reshape(
        forecast.shape[0] * forecast.shape[1], forecast.shape[2], forecast.shape[3]
    )


def _autocovariance(paths: np.ndarray, lag: int) -> np.ndarray:
    centered = paths - np.mean(paths, axis=(0, 1), keepdims=True)
    if lag == 0:
        later = earlier = centered
    else:
        later = centered[:, lag:, :]
        earlier = centered[:, :-lag, :]
    n_pairs = later.shape[0] * later.shape[1]
    return np.einsum("nti,ntj->ij", later, earlier, optimize=True) / n_pairs


def _normalized_autocovariance_errors(
    observed_paths: np.ndarray,
    forecast_samples: np.ndarray,
    *,
    lags: tuple[int, ...],
    target_scale: float | tuple[float, ...] | np.ndarray,
) -> dict[int, float]:
    observed, forecast = validate_rollout_samples(observed_paths, forecast_samples)
    n_times, n_targets = observed.shape[1:]
    if not lags or any(not isinstance(lag, int) or lag < 1 for lag in lags):
        raise ValueError("lags must be a non-empty tuple of positive integers")
    if max(lags) >= n_times:
        raise ValueError("all autocovariance lags must be shorter than the rollout")
    scale = _validated_scale(target_scale, n_targets)
    observed_scaled = observed / scale
    forecast_scaled = _forecast_as_trajectories(forecast) / scale
    denominator = np.linalg.norm(_autocovariance(observed_scaled, 0), ord="fro")
    return {
        lag: float(
            np.linalg.norm(
                _autocovariance(forecast_scaled, lag)
                - _autocovariance(observed_scaled, lag),
                ord="fro",
            )
            / max(float(denominator), EPS)
        )
        for lag in lags
    }


def normalized_autocovariance_error(
    observed_paths: np.ndarray,
    forecast_samples: np.ndarray,
    *,
    lags: tuple[int, ...] = (1, 2, 4, 8),
    target_scale: float | tuple[float, ...] | np.ndarray = 1.0,
) -> float:
    r"""RMS lagged-autocovariance error normalized by observed lag-zero scale.

    Each source is centered by its own pooled channel mean, so this diagnostic
    targets temporal/cross-target dependence rather than mean bias.  Every lagged
    matrix is divided by :math:`\|C_{\rm obs}(0)\|_F`; using a common denominator
    avoids instability when the true covariance at an individual lag is near zero.
    """

    errors = _normalized_autocovariance_errors(
        observed_paths,
        forecast_samples,
        lags=lags,
        target_scale=target_scale,
    )
    return float(np.sqrt(np.mean(np.square(list(errors.values())))))


def _spectral_density(
    paths: np.ndarray,
    *,
    sample_interval: float,
    window: Literal["hann", "rectangular"],
) -> tuple[np.ndarray, np.ndarray]:
    n_trajectories, n_times, _ = paths.shape
    centered = paths - np.mean(paths, axis=(0, 1), keepdims=True)
    if window == "hann":
        taper = np.hanning(n_times)
        if np.sum(taper**2) <= EPS:
            taper = np.ones(n_times)
    elif window == "rectangular":
        taper = np.ones(n_times)
    else:
        raise ValueError("spectral window must be 'hann' or 'rectangular'")
    transformed = np.fft.rfft(centered * taper[None, :, None], axis=1)
    sampling_frequency = 1.0 / sample_interval
    density = np.einsum(
        "kfi,kfj->fij", transformed, np.conjugate(transformed), optimize=True
    )
    density /= n_trajectories * sampling_frequency * np.sum(taper**2)

    # Convert the two-sided periodogram to a one-sided spectral density.  DC and
    # the Nyquist bin (for an even-length path) have no negative-frequency partner.
    one_sided_factor = np.full(density.shape[0], 2.0)
    one_sided_factor[0] = 1.0
    if n_times % 2 == 0:
        one_sided_factor[-1] = 1.0
    density *= one_sided_factor[:, None, None]
    frequencies = np.fft.rfftfreq(n_times, d=sample_interval)
    return frequencies, density


def _frequency_integral(values: np.ndarray, frequencies: np.ndarray) -> float:
    """Trapezoid integration on the equally spaced rFFT frequency grid."""

    if len(frequencies) < 2:
        return float(values[0])
    weights = np.ones(len(frequencies), dtype=float)
    weights[[0, -1]] = 0.5
    return float((frequencies[1] - frequencies[0]) * np.sum(weights * values))


def integrated_spectral_density_error(
    observed_paths: np.ndarray,
    forecast_samples: np.ndarray,
    *,
    sample_interval: float = 1.0,
    target_scale: float | tuple[float, ...] | np.ndarray = 1.0,
    window: Literal["hann", "rectangular"] = "hann",
) -> float:
    r"""Normalized integrated squared error of the spectral-density matrix.

    The full complex cross-spectral matrix is compared, not only the univariate
    power spectra.  The reported quantity is

    .. math::

       \frac{\int \|\widehat S(f)-S(f)\|_F^2\,df}
            {\int \|S(f)\|_F^2\,df + \epsilon}.

    This is a stationary second-order diagnostic.  It is not a proper score and
    cannot distinguish path laws that share the same second-order spectrum.
    """

    observed, forecast = validate_rollout_samples(observed_paths, forecast_samples)
    if not np.isfinite(sample_interval) or sample_interval <= 0:
        raise ValueError("sample_interval must be finite and positive")
    n_targets = observed.shape[2]
    scale = _validated_scale(target_scale, n_targets)
    frequencies, observed_density = _spectral_density(
        observed / scale, sample_interval=sample_interval, window=window
    )
    forecast_frequencies, forecast_density = _spectral_density(
        _forecast_as_trajectories(forecast) / scale,
        sample_interval=sample_interval,
        window=window,
    )
    if not np.array_equal(frequencies, forecast_frequencies):
        raise RuntimeError("internal spectral grids do not match")
    error_power = np.sum(np.abs(forecast_density - observed_density) ** 2, axis=(1, 2))
    truth_power = np.sum(np.abs(observed_density) ** 2, axis=(1, 2))
    numerator = _frequency_integral(error_power, frequencies)
    denominator = _frequency_integral(truth_power, frequencies)
    return float(numerator / max(denominator, EPS))


def _binary_ensemble_brier(
    observed_event: np.ndarray, forecast_event: np.ndarray, *, draw_axis: int
) -> tuple[float, float]:
    """Return plug-in and finite-ensemble-fair Brier scores.

    The plug-in score has an upward Monte Carlo bias because the ensemble event
    frequency is noisy.  Subtracting ``p_hat * (1-p_hat) / (m-1)`` gives an
    unbiased estimator of the Brier score of the underlying forecast probability.
    As with the fair energy score, a finite fair score can be negative.
    """

    n_draws = forecast_event.shape[draw_axis]
    probability = np.mean(forecast_event, axis=draw_axis)
    observed_float = np.asarray(observed_event, dtype=float)
    plugin_cells = (probability - observed_float) ** 2
    correction = probability * (1.0 - probability) / (n_draws - 1)
    return float(np.mean(plugin_cells)), float(np.mean(plugin_cells - correction))


def stability_escape_metrics(
    observed_paths: np.ndarray,
    forecast_samples: np.ndarray,
    *,
    escape_radius: float,
    center: float | tuple[float, ...] | np.ndarray = 0.0,
    target_scale: float | tuple[float, ...] | np.ndarray = 1.0,
) -> dict[str, float]:
    """First-passage diagnostics for a predeclared stable region.

    A path has escaped once its standardized RMS amplitude across targets exceeds
    ``escape_radius``.  ``escape_rate_error`` compares terminal population rates;
    ``escape_curve_l1`` compares the entire cumulative first-passage curve; and the
    Brier score evaluates case-conditional terminal escape probabilities.
    """

    observed, forecast = validate_rollout_samples(observed_paths, forecast_samples)
    if not np.isfinite(escape_radius) or escape_radius <= 0:
        raise ValueError("escape_radius must be finite and positive")
    n_targets = observed.shape[2]
    location = _target_vector(center, n_targets, name="center")
    scale = _validated_scale(target_scale, n_targets)
    observed_radius = np.sqrt(np.mean(((observed - location) / scale) ** 2, axis=2))
    forecast_radius = np.sqrt(
        np.mean(((forecast - location) / scale) ** 2, axis=3)
    )
    observed_escaped = np.maximum.accumulate(observed_radius > escape_radius, axis=1)
    forecast_escaped = np.maximum.accumulate(forecast_radius > escape_radius, axis=2)
    observed_curve = np.mean(observed_escaped, axis=0)
    forecast_curve = np.mean(forecast_escaped, axis=(0, 1))
    observed_terminal = observed_escaped[:, -1]
    forecast_terminal = forecast_escaped[:, :, -1]
    forecast_probability = np.mean(forecast_escaped[:, :, -1], axis=1)
    plugin_brier, fair_brier = _binary_ensemble_brier(
        observed_terminal, forecast_terminal, draw_axis=1
    )
    return {
        "escape_rate_observed": float(np.mean(observed_terminal)),
        "escape_rate_forecast": float(np.mean(forecast_probability)),
        "escape_rate_error": float(
            abs(np.mean(forecast_probability) - np.mean(observed_terminal))
        ),
        "escape_curve_l1": float(np.mean(np.abs(forecast_curve - observed_curve))),
        "escape_probability_brier_plugin": plugin_brier,
        "escape_probability_brier_fair": fair_brier,
    }


def extreme_event_frequency_metrics(
    observed_paths: np.ndarray,
    forecast_samples: np.ndarray,
    *,
    threshold: float,
    center: float | tuple[float, ...] | np.ndarray = 0.0,
    target_scale: float | tuple[float, ...] | np.ndarray = 1.0,
    sidedness: Literal["two_sided", "upper"] = "two_sided",
) -> dict[str, float]:
    """Errors in predeclared per-target extreme-event frequencies.

    The aggregate rate error and target-wise RMSE are descriptive diagnostics.
    ``extreme_probability_brier`` is a proper score for the forecast probability
    of the threshold event at each case/time/target cell.
    """

    observed, forecast = validate_rollout_samples(observed_paths, forecast_samples)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    n_targets = observed.shape[2]
    location = _target_vector(center, n_targets, name="center")
    scale = _validated_scale(target_scale, n_targets)
    observed_standardized = (observed - location) / scale
    forecast_standardized = (forecast - location) / scale
    if sidedness == "two_sided":
        observed_event = np.abs(observed_standardized) > threshold
        forecast_event = np.abs(forecast_standardized) > threshold
    elif sidedness == "upper":
        observed_event = observed_standardized > threshold
        forecast_event = forecast_standardized > threshold
    else:
        raise ValueError("sidedness must be 'two_sided' or 'upper'")

    observed_target_rate = np.mean(observed_event, axis=(0, 1))
    forecast_target_rate = np.mean(forecast_event, axis=(0, 1, 2))
    observed_rate = float(np.mean(observed_target_rate))
    forecast_rate = float(np.mean(forecast_target_rate))
    plugin_brier, fair_brier = _binary_ensemble_brier(
        observed_event, forecast_event, draw_axis=1
    )
    return {
        "extreme_rate_observed": observed_rate,
        "extreme_rate_forecast": forecast_rate,
        "extreme_frequency_error": float(abs(forecast_rate - observed_rate)),
        "extreme_frequency_target_rmse": float(
            np.sqrt(np.mean((forecast_target_rate - observed_target_rate) ** 2))
        ),
        "extreme_probability_brier_plugin": plugin_brier,
        "extreme_probability_brier_fair": fair_brier,
    }


def rollout_metrics(
    observed_paths: np.ndarray,
    forecast_samples: np.ndarray,
    *,
    config: RolloutMetricConfig = RolloutMetricConfig(),
) -> dict[str, float]:
    """Compute the complete long-horizon scorecard under one frozen config."""

    observed, forecast = validate_rollout_samples(observed_paths, forecast_samples)
    if max(config.autocovariance_lags) >= observed.shape[1]:
        raise ValueError("autocovariance lags must be shorter than the rollout")
    autocovariance_errors = _normalized_autocovariance_errors(
        observed,
        forecast,
        lags=config.autocovariance_lags,
        target_scale=config.target_scale,
    )
    result = {
        "rollout.path_energy_score_fair": fair_path_energy_score(
            observed,
            forecast,
            block_length=config.block_length,
            stride=config.block_stride,
            target_scale=config.target_scale,
            max_draws=config.max_energy_draws,
        ),
        "rollout.autocovariance_nrmse": float(
            np.sqrt(np.mean(np.square(list(autocovariance_errors.values()))))
        ),
        "rollout.spectral_density_nise": integrated_spectral_density_error(
            observed,
            forecast,
            sample_interval=config.sample_interval,
            target_scale=config.target_scale,
            window=config.spectral_window,
        ),
    }
    result.update(
        {
            f"rollout.autocovariance_relative_error_lag{lag}": error
            for lag, error in autocovariance_errors.items()
        }
    )
    result.update(
        {
            f"rollout.stability.{name}": value
            for name, value in stability_escape_metrics(
                observed,
                forecast,
                escape_radius=config.escape_radius,
                center=config.center,
                target_scale=config.target_scale,
            ).items()
        }
    )
    result.update(
        {
            f"rollout.extreme.{name.removeprefix('extreme_')}": value
            for name, value in extreme_event_frequency_metrics(
                observed,
                forecast,
                threshold=config.extreme_threshold,
                center=config.center,
                target_scale=config.target_scale,
                sidedness=config.extreme_sidedness,
            ).items()
        }
    )
    return result
