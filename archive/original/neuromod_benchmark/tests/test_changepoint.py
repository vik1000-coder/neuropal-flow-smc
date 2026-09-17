from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from neuromod_benchmark.capabilities import UnsupportedCapability
from neuromod_benchmark.changepoint import (
    DEFAULT_METHODS,
    ChangeSeries,
    PostChangeStrengths,
    calibrate_matched_nulls,
    detect_calibrated,
    run_changepoint_benchmark,
    scan_changepoint,
    simulate_changepoint_series,
    simulate_matched_nulls,
)
from neuromod_benchmark.mechanistic import (
    MechanisticConfig,
    generate_mechanistic_parameters,
)
from neuromod_benchmark.methods.monitoring import (
    CalibratedScanMonitor,
    OfflineChangepointSuite,
)


def _small_config() -> MechanisticConfig:
    return MechanisticConfig(
        n_neurons=3,
        n_modulators=1,
        n_steps=96,
        burn_in=32,
        mechanism="null",
        effect_strength=0.9,
        seed=19,
    )


def _iid_series(
    values: np.ndarray,
    *,
    scenario: str,
    identifier: str,
    seed: int,
    boundary: int | None = None,
) -> ChangeSeries:
    return ChangeSeries(
        values=values,
        scenario=scenario,
        series_id=identifier,
        simulation_seed=seed,
        match_key="iid-standard-normal-system",
        boundary=boundary,
        boundary_name="neuromodulator_onset" if boundary is not None else None,
    )


def test_mechanistic_scenarios_share_prechange_and_have_pure_entry_channels() -> None:
    config = _small_config()
    params = generate_mechanistic_parameters(config)
    boundary = 48
    common = {
        scenario: simulate_changepoint_series(
            config,
            scenario=scenario,
            boundary=boundary,
            params=params,
            simulation_seed=431,
        )
        for scenario in ("null", "mean", "dispersion", "matched_tail")
    }

    reference = common["null"].values[:boundary]
    for scenario in ("mean", "dispersion", "matched_tail"):
        np.testing.assert_array_equal(common[scenario].values[:boundary], reference)
        assert common[scenario].boundary_name == "neuromodulator_onset"
        assert common[scenario].match_key == common["null"].match_key

    mean = common["mean"]
    assert np.max(np.abs(mean.entry_mean_shift[boundary:])) > 1e-5
    np.testing.assert_allclose(mean.entry_variance_shift, 0.0, atol=1e-14)

    dispersion = common["dispersion"]
    np.testing.assert_allclose(dispersion.entry_mean_shift, 0.0, atol=1e-14)
    assert np.max(np.abs(dispersion.entry_variance_shift[boundary:])) > 1e-6

    tail = common["matched_tail"]
    np.testing.assert_allclose(tail.entry_mean_shift, 0.0, atol=1e-14)
    np.testing.assert_allclose(tail.entry_variance_shift, 0.0, atol=1e-14)
    assert np.max(np.abs(tail.entry_fourth_shift[boundary:])) > 1e-6
    assert np.max(tail.mixture_probability[boundary:]) > 0.0


def test_every_detector_returns_a_finite_curve_on_the_same_candidate_grid() -> None:
    config = _small_config()
    changed = simulate_changepoint_series(
        config,
        scenario="mean",
        boundary=48,
        simulation_seed=91,
    )
    scans = [
        scan_changepoint(
            changed,
            method=method,
            min_segment=14,
            scan_window=18,
        )
        for method in DEFAULT_METHODS
    ]
    for scan in scans:
        assert scan.claim_axis
        assert scan.detector_channel
        assert scan.tau_hat in scan.candidate_indices
        assert scan.score_curve.shape == scan.candidate_indices.shape
        assert np.isfinite(scan.score_curve).all()
    for scan in scans[1:]:
        np.testing.assert_array_equal(scan.candidate_indices, scans[0].candidate_indices)


def test_post_strengths_preserve_null_match_and_scale_only_the_named_channel() -> None:
    config = _small_config()
    params = generate_mechanistic_parameters(config)
    common = {
        "config": config,
        "scenario": "dispersion",
        "boundary": 48,
        "params": params,
        "simulation_seed": 991,
    }
    weak = simulate_changepoint_series(
        **common,
        post_change_strengths=PostChangeStrengths(logvariance_multiplier=1.0),
        series_id="weak",
    )
    strong = simulate_changepoint_series(
        **common,
        post_change_strengths=PostChangeStrengths(logvariance_multiplier=3.0),
        series_id="strong",
    )
    assert weak.match_key == strong.match_key
    np.testing.assert_array_equal(weak.values[:48], strong.values[:48])
    np.testing.assert_allclose(strong.entry_mean_shift, 0.0, atol=1e-14)
    assert np.max(np.abs(strong.entry_variance_shift[48:])) > np.max(
        np.abs(weak.entry_variance_shift[48:])
    )

    tail = simulate_changepoint_series(
        config,
        scenario="matched_tail",
        boundary=48,
        params=params,
        simulation_seed=991,
        post_change_strengths=PostChangeStrengths(
            tail_logit_multiplier=2.0,
            tail_base_probability=0.12,
            tail_low_scale=0.45,
            tail_high_scale=3.5,
        ),
    )
    assert tail.match_key == weak.match_key
    np.testing.assert_allclose(tail.entry_mean_shift, 0.0, atol=1e-14)
    np.testing.assert_allclose(tail.entry_variance_shift, 0.0, atol=1e-14)
    assert np.max(np.abs(tail.entry_fourth_shift[48:])) > 1e-5


def test_default_residual_scans_are_oracle_blind_and_exclude_prefix_fit_rows() -> None:
    config = _small_config()
    changed = simulate_changepoint_series(
        config,
        scenario="dispersion",
        boundary=48,
        simulation_seed=551,
    )
    corrupted_oracle = replace(
        changed,
        conditional_mean=np.full_like(changed.values, 1e9),
        conditional_variance=np.full_like(changed.values, 1e-9),
        centered_fourth_moment=np.full_like(changed.values, 1e12),
    )
    for method in ("variance_reliability", "residual_tail_shape"):
        original = scan_changepoint(
            changed,
            method=method,
            min_segment=14,
            scan_window=18,
            reference_fraction=0.25,
        )
        corrupted = scan_changepoint(
            corrupted_oracle,
            method=method,
            min_segment=14,
            scan_window=18,
            reference_fraction=0.25,
        )
        np.testing.assert_array_equal(original.score_curve, corrupted.score_curve)
        assert original.diagnostic_metadata["residual_mode"] == "frozen_prefix"
        assert (
            original.diagnostic_metadata["scan_residual_start"]
            == original.diagnostic_metadata["reference_end"]
            - original.diagnostic_metadata["response_start"]
        )


def test_tail_specific_residual_scan_detects_variance_matched_shape_change() -> None:
    rng = np.random.default_rng(2031)
    length, channels, boundary = 300, 3, 150
    nulls = tuple(
        _iid_series(
            rng.standard_normal((length, channels)),
            scenario="null",
            identifier=f"tail-cal:{index}",
            seed=30_000 + index,
        )
        for index in range(19)
    )
    probability, low, high = 0.14, 0.40, 4.0
    normalizer = math.sqrt((1.0 - probability) * low**2 + probability * high**2)
    values = rng.standard_normal((length, channels))
    selector = rng.random((length - boundary, channels)) < probability
    component = np.where(selector, high, low) / normalizer
    values[boundary:] *= component
    changed = _iid_series(
        values,
        scenario="matched_tail",
        identifier="evaluation:tail",
        seed=40_000,
        boundary=boundary,
    )
    calibration = calibrate_matched_nulls(
        nulls,
        methods=("residual_tail_shape",),
        alpha=0.10,
        min_segment=50,
        stride=2,
        reference_fraction=0.25,
    )["residual_tail_shape"]
    result = detect_calibrated(changed, calibration, tolerance=12)
    assert result.detector_channel == "tail_shape"
    assert result.called
    assert result.hit
    assert result.detected_within
    # The construction is variance matched in expectation, not in this finite sample.
    theoretical_variance = (
        (1.0 - probability) * low**2 + probability * high**2
    ) / normalizer**2
    assert theoretical_variance == pytest.approx(1.0)


def test_independent_null_calibration_detects_strong_mean_change_and_separates_axes() -> None:
    rng = np.random.default_rng(701)
    length, channels, boundary = 180, 3, 90
    nulls = tuple(
        _iid_series(
            rng.standard_normal((length, channels)),
            scenario="null",
            identifier=f"cal:{index}",
            seed=index,
        )
        for index in range(19)
    )
    changed_values = rng.standard_normal((length, channels))
    changed_values[boundary:] += np.array([2.8, -2.4, 2.2])
    changed = _iid_series(
        changed_values,
        scenario="mean",
        identifier="evaluation:mean",
        seed=10_000,
        boundary=boundary,
    )
    calibration = calibrate_matched_nulls(
        nulls,
        methods=("mean_cusum",),
        alpha=0.10,
        min_segment=20,
        scan_window=24,
    )["mean_cusum"]
    result = detect_calibrated(changed, calibration, tolerance=5)

    assert calibration.minimum_p_value == pytest.approx(0.05)
    assert calibration.attainable_null_call_rate == pytest.approx(0.10)
    calibration_low, calibration_high = calibration.calibration_fpr_interval()
    assert calibration_low < calibration.attainable_null_call_rate < calibration_high
    assert result.called
    assert result.false_positive is None
    assert result.hit
    assert result.detected_within
    assert abs(result.localization_delay) <= 5
    assert result.detected_delay == result.localization_delay

    null_result = detect_calibrated(
        _iid_series(
            rng.standard_normal((length, channels)),
            scenario="null",
            identifier="evaluation:null",
            seed=20_000,
        ),
        calibration,
    )
    assert null_result.false_positive == null_result.called
    assert null_result.hit is None
    assert null_result.detected_within is None
    assert null_result.localization_delay is None


def test_calibration_rejects_changes_duplicates_mismatch_and_reuse() -> None:
    config = _small_config()
    params = generate_mechanistic_parameters(config)
    nulls = simulate_matched_nulls(
        config,
        n_series=5,
        params=params,
        first_seed=800,
    )
    changed = simulate_changepoint_series(
        config,
        scenario="dispersion",
        boundary=48,
        params=params,
        simulation_seed=900,
    )
    with pytest.raises(ValueError, match="only no-change"):
        calibrate_matched_nulls((*nulls, changed), methods=("mean_cusum",))
    with pytest.raises(ValueError, match="at least one detector"):
        calibrate_matched_nulls(nulls, methods=())
    with pytest.raises(ValueError, match="IDs must be unique"):
        calibrate_matched_nulls((nulls[0], nulls[0]), methods=("mean_cusum",))

    calibration = calibrate_matched_nulls(
        nulls,
        methods=("mean_cusum",),
        alpha=0.20,
        min_segment=14,
    )["mean_cusum"]
    with pytest.raises(ValueError, match="cannot be reused"):
        detect_calibrated(nulls[0], calibration)

    mismatched = ChangeSeries(
        values=changed.values,
        scenario=changed.scenario,
        series_id="mismatch",
        simulation_seed=901,
        match_key="different-system",
        boundary=changed.boundary,
        boundary_name=changed.boundary_name,
    )
    with pytest.raises(ValueError, match="does not match"):
        detect_calibrated(mismatched, calibration)


def test_monitor_adapters_declare_offline_only_and_require_calibration() -> None:
    config = _small_config()
    params = generate_mechanistic_parameters(config)
    nulls = simulate_matched_nulls(
        config,
        n_series=5,
        params=params,
        first_seed=1_100,
    )
    changed = simulate_changepoint_series(
        config,
        scenario="mean",
        boundary=48,
        params=params,
        simulation_seed=1_200,
    )

    monitor = CalibratedScanMonitor(
        "mean_cusum",
        alpha=0.20,
        min_segment=14,
        scan_window=18,
    )
    assert monitor.capabilities.offline_change
    assert not monitor.capabilities.online_change
    with pytest.raises(RuntimeError, match="must be fit"):
        monitor.detect(changed)
    result = monitor.fit(nulls).detect(changed)
    assert result.method == "mean_cusum"
    assert monitor.metadata()["calibration_source"] == "independent_matched_no_change"
    with pytest.raises(UnsupportedCapability):
        monitor.predict(None)

    suite = OfflineChangepointSuite(
        alpha=0.20,
        min_segment=14,
        scan_window=18,
    ).fit(nulls)
    results, attribution = suite.detect(changed)
    assert {result.method for result in results} == set(DEFAULT_METHODS)
    assert attribution.series_id == changed.series_id


def test_benchmark_keeps_calibration_disjoint_and_reports_metrics_separately() -> None:
    config = MechanisticConfig(
        n_neurons=3,
        n_modulators=1,
        n_steps=72,
        burn_in=20,
        mechanism="null",
        effect_strength=0.9,
        seed=29,
    )
    report = run_changepoint_benchmark(
        config,
        n_calibration=4,
        n_null_test=1,
        n_changed=1,
        alpha=0.25,
        boundary_fraction=0.5,
        tolerance=8,
        min_segment=12,
        scan_window=12,
        seed=5_000,
    )

    assert len(report.detections) == 4 * len(DEFAULT_METHODS)
    assert len(report.attributions) == 4
    calibration_ids = set(report.metadata["calibration_ids"])
    evaluation_ids = set(report.metadata["evaluation_ids"])
    assert calibration_ids.isdisjoint(evaluation_ids)
    assert report.metadata["minimum_attainable_p_value"] == pytest.approx(0.2)

    null_rows = [row for row in report.detection_summary if row["scenario"] == "null"]
    assert len(null_rows) == len(DEFAULT_METHODS)
    for row in null_rows:
        assert row["false_positive_rate"] == row["call_rate"]
        assert math.isnan(row["hit_rate"])
        assert math.isnan(row["detected_within_rate"])
        assert 0.0 <= row["false_positive_rate_ci_low"]
        assert row["false_positive_rate_ci_high"] <= 1.0
        assert row["false_positive_rate_ci_low"] <= row["false_positive_rate"]
        assert row["false_positive_rate"] <= row["false_positive_rate_ci_high"]

    changed_rows = [row for row in report.detection_summary if row["scenario"] != "null"]
    assert changed_rows
    for row in changed_rows:
        assert math.isnan(row["false_positive_rate"])
        assert 0.0 <= row["hit_rate"] <= 1.0
        assert 0.0 <= row["detected_within_rate"] <= row["hit_rate"]
