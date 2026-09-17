import numpy as np
import pytest

from neuromod_benchmark.rollout_metrics import (
    ROLLOUT_CONTRACT,
    ROLLOUT_SAMPLE_CAPABILITY,
    RolloutMetricConfig,
    extreme_event_frequency_metrics,
    fair_path_energy_score,
    integrated_spectral_density_error,
    normalized_autocovariance_error,
    rollout_metrics,
    stability_escape_metrics,
    validate_rollout_samples,
)


def _replicate_as_ensemble(paths: np.ndarray, draws: int = 4) -> np.ndarray:
    return np.repeat(paths[:, None, :, :], draws, axis=1)


def _stationary_ar_paths(
    rng: np.random.Generator, n_paths: int, n_times: int, rho: float
) -> np.ndarray:
    paths = np.empty((n_paths, n_times, 1))
    paths[:, 0, 0] = rng.standard_normal(n_paths)
    innovation_scale = np.sqrt(1.0 - rho**2)
    for time in range(1, n_times):
        paths[:, time, 0] = (
            rho * paths[:, time - 1, 0]
            + innovation_scale * rng.standard_normal(n_paths)
        )
    return paths


def test_rollout_contract_rejects_non_joint_or_misaligned_samples():
    observed = np.zeros((3, 16, 2))
    forecast = np.zeros((3, 4, 16, 2))
    validated_observed, validated_forecast = validate_rollout_samples(observed, forecast)
    assert validated_observed.shape == (3, 16, 2)
    assert validated_forecast.shape == (3, 4, 16, 2)
    assert ROLLOUT_CONTRACT.capability == ROLLOUT_SAMPLE_CAPABILITY

    with pytest.raises(ValueError, match=r"\[case, draw, time, target\]"):
        validate_rollout_samples(observed, forecast[:, 0])
    with pytest.raises(ValueError, match="must match"):
        validate_rollout_samples(observed, np.zeros((3, 4, 15, 2)))
    with pytest.raises(ValueError, match="at least 2 forecast draws"):
        validate_rollout_samples(observed, forecast[:, :1])


def test_identical_joint_paths_are_exact_for_all_path_law_metrics():
    time = np.arange(64)
    paths = np.stack(
        [
            np.sin(2 * np.pi * (time + phase) / 16)
            for phase in (0, 2, 4, 6)
        ],
        axis=0,
    )[:, :, None]
    oracle = _replicate_as_ensemble(paths)
    config = RolloutMetricConfig(
        block_length=32,
        block_stride=16,
        autocovariance_lags=(1, 2, 4, 8),
        escape_radius=2.0,
        extreme_threshold=0.8,
    )
    metrics = rollout_metrics(paths, oracle, config=config)

    assert metrics["rollout.path_energy_score_fair"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["rollout.autocovariance_nrmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["rollout.spectral_density_nise"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["rollout.stability.escape_rate_error"] == 0.0
    assert metrics["rollout.extreme.frequency_error"] == 0.0


def test_same_marginals_but_wrong_temporal_order_is_detected():
    time = np.arange(64)
    paths = np.stack(
        [
            np.sin(2 * np.pi * (time + phase) / 16)
            + 0.2 * np.cos(2 * np.pi * (time + phase) / 7)
            for phase in (0, 3, 6, 9)
        ],
        axis=0,
    )[:, :, None]
    # Every forecast path contains exactly the same scalar values as its observed
    # counterpart.  Only their temporal order is changed.
    permutation = np.random.default_rng(17).permutation(paths.shape[1])
    wrong_paths = paths[:, permutation, :]
    np.testing.assert_allclose(
        np.sort(wrong_paths, axis=1), np.sort(paths, axis=1), atol=0.0
    )
    oracle = _replicate_as_ensemble(paths)
    wrong = _replicate_as_ensemble(wrong_paths)

    oracle_energy = fair_path_energy_score(paths, oracle, block_length=64)
    wrong_energy = fair_path_energy_score(paths, wrong, block_length=64)
    oracle_acf = normalized_autocovariance_error(paths, oracle, lags=(1, 2, 4, 8))
    wrong_acf = normalized_autocovariance_error(paths, wrong, lags=(1, 2, 4, 8))
    oracle_spectrum = integrated_spectral_density_error(paths, oracle)
    wrong_spectrum = integrated_spectral_density_error(paths, wrong)

    assert oracle_energy == pytest.approx(0.0, abs=1e-14)
    assert oracle_acf == pytest.approx(0.0, abs=1e-14)
    assert oracle_spectrum == pytest.approx(0.0, abs=1e-14)
    assert wrong_energy > 0.5
    assert wrong_acf > 0.5
    assert wrong_spectrum > 0.5


def test_independent_oracle_path_draws_beat_wrong_dependence_with_sampling_floor():
    """A lawful forecast uses independent draws, never leaked observed paths."""

    rng = np.random.default_rng(1907)
    cases, draws, times = 192, 24, 48
    observed = _stationary_ar_paths(rng, cases, times, rho=0.82)
    oracle = _stationary_ar_paths(rng, cases * draws, times, rho=0.82).reshape(
        cases, draws, times, 1
    )
    # Same N(0,1) marginals and innovation scale, but no temporal dependence.
    wrong = _stationary_ar_paths(rng, cases * draws, times, rho=0.0).reshape(
        cases, draws, times, 1
    )

    oracle_energy = fair_path_energy_score(
        observed, oracle, block_length=24, stride=12, max_draws=24
    )
    wrong_energy = fair_path_energy_score(
        observed, wrong, block_length=24, stride=12, max_draws=24
    )
    oracle_acf = normalized_autocovariance_error(
        observed, oracle, lags=(1, 2, 4, 8)
    )
    wrong_acf = normalized_autocovariance_error(
        observed, wrong, lags=(1, 2, 4, 8)
    )
    oracle_spectrum = integrated_spectral_density_error(observed, oracle)
    wrong_spectrum = integrated_spectral_density_error(observed, wrong)

    assert oracle_energy < wrong_energy
    assert oracle_acf < wrong_acf
    assert oracle_spectrum < wrong_spectrum
    # Finite independent samples do not have the impossible leaked-path optimum.
    assert oracle_energy > 0.0
    assert oracle_acf > 0.0
    assert oracle_spectrum > 0.0

    small_acf = normalized_autocovariance_error(
        observed[:16], oracle[:16], lags=(1, 2, 4, 8)
    )
    small_spectrum = integrated_spectral_density_error(
        observed[:16], oracle[:16]
    )
    assert oracle_acf < small_acf
    assert oracle_spectrum < small_spectrum


def test_fair_path_energy_uses_off_diagonal_u_statistic():
    observed = np.zeros((1, 2, 1))
    forecast = np.asarray([[[[-1.0], [-1.0]], [[1.0], [1.0]]]])
    # Both forecast distances are one per coordinate.  After sqrt-dimension
    # normalization the first term is one and the fair pair correction is one.
    assert fair_path_energy_score(observed, forecast) == pytest.approx(0.0)


def test_unstable_rollout_is_detected_by_escape_rate_and_first_passage_curve():
    observed = np.zeros((5, 20, 2))
    stable = _replicate_as_ensemble(observed)
    unstable = stable.copy()
    unstable[:, :, 8:, :] = np.linspace(0.0, 12.0, 12)[None, None, :, None]

    stable_metrics = stability_escape_metrics(
        observed, stable, escape_radius=3.0, target_scale=1.0
    )
    unstable_metrics = stability_escape_metrics(
        observed, unstable, escape_radius=3.0, target_scale=1.0
    )

    assert stable_metrics["escape_rate_error"] == 0.0
    assert stable_metrics["escape_curve_l1"] == 0.0
    assert unstable_metrics["escape_rate_observed"] == 0.0
    assert unstable_metrics["escape_rate_forecast"] == 1.0
    assert unstable_metrics["escape_rate_error"] == 1.0
    assert unstable_metrics["escape_curve_l1"] > 0.25
    assert unstable_metrics["escape_probability_brier_plugin"] == 1.0
    assert unstable_metrics["escape_probability_brier_fair"] == 1.0


def test_extreme_event_frequency_detects_tail_rate_difference():
    observed = np.zeros((4, 20, 2))
    observed[:, ::10, :] = 3.0  # 10% two-sided event frequency.
    matched = _replicate_as_ensemble(observed)
    heavy = np.zeros_like(matched)
    heavy[:, :, ::2, :] = -3.0  # 50% with identical center and target scale.

    matched_metrics = extreme_event_frequency_metrics(
        observed, matched, threshold=2.0
    )
    heavy_metrics = extreme_event_frequency_metrics(observed, heavy, threshold=2.0)

    assert matched_metrics["extreme_frequency_error"] == 0.0
    assert matched_metrics["extreme_probability_brier_plugin"] == 0.0
    assert matched_metrics["extreme_probability_brier_fair"] == 0.0
    assert heavy_metrics["extreme_rate_observed"] == pytest.approx(0.1)
    assert heavy_metrics["extreme_rate_forecast"] == pytest.approx(0.5)
    assert heavy_metrics["extreme_frequency_error"] == pytest.approx(0.4)
    assert heavy_metrics["extreme_frequency_target_rmse"] == pytest.approx(0.4)
    assert heavy_metrics["extreme_probability_brier_plugin"] > 0.0
    assert heavy_metrics["extreme_probability_brier_fair"] > 0.0


def test_scales_and_thresholds_must_be_predeclared_and_valid():
    observed = np.zeros((2, 12, 2))
    forecast = _replicate_as_ensemble(observed)
    with pytest.raises(ValueError, match="one value per target"):
        fair_path_energy_score(observed, forecast, target_scale=(1.0,))
    with pytest.raises(ValueError, match="strictly positive"):
        normalized_autocovariance_error(
            observed, forecast, lags=(1,), target_scale=(1.0, 0.0)
        )
    with pytest.raises(ValueError, match="finite and positive"):
        extreme_event_frequency_metrics(observed, forecast, threshold=0.0)
