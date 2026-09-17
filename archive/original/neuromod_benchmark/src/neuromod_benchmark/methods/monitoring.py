"""Capability-aware adapters for the isolated changepoint lane.

These adapters intentionally do not implement the predictive ``Estimator`` contract:
their fit data are independent no-change trajectories, not supervised history/target
pairs, and their output is a calibrated changepoint decision rather than a forecast.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..capabilities import Capabilities, ResourceProfile, UnsupportedCapability
from ..changepoint import (
    AttributionResult,
    ChangeSeries,
    DEFAULT_METHODS,
    DetectionResult,
    NullCalibration,
    calibrate_matched_nulls,
    detect_calibrated,
    infer_event_channel,
)


class CalibratedScanMonitor:
    """One offline detector calibrated on independent matched null trajectories."""

    capabilities = Capabilities(
        predictive_mean=False,
        predictive_distribution="none",
        online_change=False,
        offline_change=True,
    )
    resource_profile = ResourceProfile(
        scheduling_class="tiny_cpu",
        threads=1,
        uses_accelerator=False,
        estimated_peak_gb=0.25,
    )

    def __init__(
        self,
        method: str,
        *,
        alpha: float = 0.05,
        min_segment: int = 20,
        stride: int = 1,
        scan_window: int = 32,
        ridge: float = 0.1,
        residual_mode: str = "frozen_prefix",
        reference_fraction: float = 0.25,
        crossfit_folds: int = 5,
        crossfit_purge: int = 2,
        tail_thresholds: tuple[float, ...] = (2.0, 3.0),
        variance_estimator: str = "block_median",
        variance_block_size: int = 32,
        variance_channel_aggregation: str = "max",
        residual_lags: int = 1,
        residual_nonlinear: bool = False,
        residual_ewma_decays: tuple[float, ...] = (),
    ) -> None:
        self.method = method
        self.name = f"calibrated_{method}"
        self.alpha = float(alpha)
        self.scan_options = {
            "min_segment": int(min_segment),
            "stride": int(stride),
            "scan_window": int(scan_window),
            "ridge": float(ridge),
            "residual_mode": residual_mode,
            "reference_fraction": float(reference_fraction),
            "crossfit_folds": int(crossfit_folds),
            "crossfit_purge": int(crossfit_purge),
            "tail_thresholds": tuple(float(value) for value in tail_thresholds),
            "variance_estimator": variance_estimator,
            "variance_block_size": int(variance_block_size),
            "variance_channel_aggregation": variance_channel_aggregation,
            "residual_lags": int(residual_lags),
            "residual_nonlinear": bool(residual_nonlinear),
            "residual_ewma_decays": tuple(
                float(value) for value in residual_ewma_decays
            ),
        }
        self.calibration_: NullCalibration | None = None

    def fit(self, null_series: Sequence[ChangeSeries]) -> "CalibratedScanMonitor":
        calibrations = calibrate_matched_nulls(
            null_series,
            methods=(self.method,),
            alpha=self.alpha,
            **self.scan_options,
        )
        self.calibration_ = calibrations[self.method]
        return self

    def detect(self, series: ChangeSeries, *, tolerance: int = 10) -> DetectionResult:
        if self.calibration_ is None:
            raise RuntimeError("monitor must be fit on matched null trajectories first")
        return detect_calibrated(series, self.calibration_, tolerance=tolerance)

    def predict(self, *_: Any, **__: Any) -> None:
        raise UnsupportedCapability(self.name, "predictive_mean")

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "scan_options": dict(self.scan_options),
            "calibrated": self.calibration_ is not None,
            "calibration_size": (
                0 if self.calibration_ is None else self.calibration_.null_max_scores.size
            ),
            "calibration_source": "independent_matched_no_change",
        }


class OfflineChangepointSuite:
    """A calibrated detector panel plus explicit event-level channel attribution."""

    name = "offline_changepoint_suite"
    capabilities = Capabilities(
        predictive_mean=False,
        predictive_distribution="none",
        online_change=False,
        offline_change=True,
    )
    resource_profile = ResourceProfile(
        scheduling_class="tiny_cpu",
        threads=1,
        uses_accelerator=False,
        estimated_peak_gb=0.5,
    )

    def __init__(
        self,
        *,
        methods: Sequence[str] = DEFAULT_METHODS,
        alpha: float = 0.05,
        min_segment: int = 20,
        stride: int = 1,
        scan_window: int = 32,
        ridge: float = 0.1,
        residual_mode: str = "frozen_prefix",
        reference_fraction: float = 0.25,
        crossfit_folds: int = 5,
        crossfit_purge: int = 2,
        tail_thresholds: tuple[float, ...] = (2.0, 3.0),
        variance_estimator: str = "block_median",
        variance_block_size: int = 32,
        variance_channel_aggregation: str = "max",
        residual_lags: int = 1,
        residual_nonlinear: bool = False,
        residual_ewma_decays: tuple[float, ...] = (),
    ) -> None:
        self.methods = tuple(methods)
        self.alpha = float(alpha)
        self.scan_options = {
            "min_segment": int(min_segment),
            "stride": int(stride),
            "scan_window": int(scan_window),
            "ridge": float(ridge),
            "residual_mode": residual_mode,
            "reference_fraction": float(reference_fraction),
            "crossfit_folds": int(crossfit_folds),
            "crossfit_purge": int(crossfit_purge),
            "tail_thresholds": tuple(float(value) for value in tail_thresholds),
            "variance_estimator": variance_estimator,
            "variance_block_size": int(variance_block_size),
            "variance_channel_aggregation": variance_channel_aggregation,
            "residual_lags": int(residual_lags),
            "residual_nonlinear": bool(residual_nonlinear),
            "residual_ewma_decays": tuple(
                float(value) for value in residual_ewma_decays
            ),
        }
        self.calibrations_: dict[str, NullCalibration] | None = None

    def fit(self, null_series: Sequence[ChangeSeries]) -> "OfflineChangepointSuite":
        self.calibrations_ = calibrate_matched_nulls(
            null_series,
            methods=self.methods,
            alpha=self.alpha,
            **self.scan_options,
        )
        return self

    def detect(
        self,
        series: ChangeSeries,
        *,
        tolerance: int = 10,
    ) -> tuple[tuple[DetectionResult, ...], AttributionResult]:
        if self.calibrations_ is None:
            raise RuntimeError("suite must be fit on matched null trajectories first")
        results = tuple(
            detect_calibrated(
                series,
                self.calibrations_[method],
                tolerance=tolerance,
            )
            for method in self.methods
        )
        return results, infer_event_channel(results)

    def predict(self, *_: Any, **__: Any) -> None:
        raise UnsupportedCapability(self.name, "predictive_mean")

    def metadata(self) -> dict[str, Any]:
        calibration_size = 0
        minimum_p = None
        if self.calibrations_:
            first = next(iter(self.calibrations_.values()))
            calibration_size = int(first.null_max_scores.size)
            minimum_p = first.minimum_p_value
        return {
            "methods": self.methods,
            "alpha": self.alpha,
            "scan_options": dict(self.scan_options),
            "calibrated": self.calibrations_ is not None,
            "calibration_size": calibration_size,
            "minimum_attainable_p_value": minimum_p,
            "calibration_source": "independent_matched_no_change",
        }


__all__ = ["CalibratedScanMonitor", "OfflineChangepointSuite"]
