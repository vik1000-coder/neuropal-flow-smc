from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def truth_rms(truth: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean(truth**2)))


def nrmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    denominator = truth_rms(truth)
    return rmse(estimate, truth) / max(denominator, 1e-12)


def slope_through_origin(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denominator = float(np.sum(truth**2))
    return float(np.sum(estimate * truth) / max(denominator, 1e-12))


def signed_correlation(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if np.std(estimate) <= 1e-12 or np.std(truth) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(estimate, truth)[0, 1])


def sign_accuracy(estimate: np.ndarray, truth: np.ndarray, tolerance: float = 1e-10) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    active = np.abs(truth) > tolerance
    if not np.any(active):
        return float(np.mean(np.abs(estimate) <= tolerance))
    return float(np.mean(np.sign(estimate[active]) == np.sign(truth[active])))


def typed_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return {
        "rmse": rmse(estimate, truth),
        "nrmse": nrmse(estimate, truth),
        "truth_rms": truth_rms(truth),
        "signed_correlation": signed_correlation(estimate, truth),
        "calibration_slope": slope_through_origin(estimate, truth),
        "sign_accuracy": sign_accuracy(estimate, truth),
        "integrated_absolute_error": float(np.mean(np.abs(estimate - truth))),
        "estimate_rms": truth_rms(estimate),
    }


def lag_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    lag_index = np.arange(1, truth.size + 1, dtype=float)
    truth_mass = np.abs(truth)
    estimate_mass = np.abs(estimate)
    truth_center = float(np.sum(lag_index * truth_mass) / max(np.sum(truth_mass), 1e-12))
    estimate_center = float(np.sum(lag_index * estimate_mass) / max(np.sum(estimate_mass), 1e-12))
    base = typed_metrics(estimate, truth)
    base.update(
        {
            "peak_lag_error": float(abs(np.argmax(estimate_mass) - np.argmax(truth_mass))),
            "center_of_mass_error": abs(estimate_center - truth_center),
            "slow_mass_fraction_error": abs(
                float(np.sum(estimate_mass[truth.size // 2 :]) / max(np.sum(estimate_mass), 1e-12))
                - float(np.sum(truth_mass[truth.size // 2 :]) / max(np.sum(truth_mass), 1e-12))
            ),
        }
    )
    return base


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    lower: float
    upper: float
    n: int


def bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    replicates: int = 2000,
    confidence: float = 0.95,
) -> BootstrapInterval:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return BootstrapInterval(float("nan"), float("nan"), float("nan"), 0)
    draws = rng.integers(0, values.size, size=(int(replicates), values.size))
    bootstrap = np.mean(values[draws], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        float(np.mean(values)),
        float(np.quantile(bootstrap, alpha)),
        float(np.quantile(bootstrap, 1.0 - alpha)),
        int(values.size),
    )


def summarize_seed_metrics(
    seed_level: pd.DataFrame,
    group_columns: list[str],
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 99173,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for keys, frame in seed_level.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        interval = bootstrap_mean_interval(
            frame["metric_value"].to_numpy(float), rng, replicates=replicates, confidence=confidence
        )
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "mean": interval.mean,
                "ci_lower": interval.lower,
                "ci_upper": interval.upper,
                "n_dgp_seeds": interval.n,
                "failure_rate": float(np.mean(frame["status"] != "ok")),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_error_ratio_rows(
    seed_level: pd.DataFrame,
    method: str,
    baseline: str,
    grouping: list[str],
) -> pd.DataFrame:
    relevant = seed_level[
        (seed_level["metric_name"] == "nrmse") & seed_level["method_id"].isin([method, baseline])
    ]
    index = grouping + ["dgp_seed"]
    pivot = relevant.pivot_table(index=index, columns="method_id", values="metric_value", aggfunc="mean")
    pivot = pivot.dropna(subset=[method, baseline]).reset_index()
    pivot["error_ratio"] = pivot[method] / np.maximum(pivot[baseline], 1e-12)
    return pivot
