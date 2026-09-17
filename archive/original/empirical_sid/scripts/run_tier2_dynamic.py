#!/usr/bin/env python3
"""Run preregistered Tier-2 hidden-state and observation-filter stresses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit, logit
from sklearn.linear_model import Ridge

from empirical_sid.dynamic import (
    DynamicAnchoredScore,
    DynamicMomentRegressor,
    DynamicRatioCritic,
    history_design,
    history_design_derivative,
    influence_dynamic,
    ratio_effect,
)
from empirical_sid.metrics import lag_metrics


SEEDS = tuple(range(3001, 3031))
OUTPUT_CAP_GB = 0.12
MIN_FREE_DISK_GB = 2.0
MAX_RSS_GB = 8.0
PRIMARY_SIGMA = 0.12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def enforce_resources(output: Path) -> None:
    disk = shutil.disk_usage(output.parent if output.parent.exists() else output.parent.parent)
    if disk.free / 1024**3 < MIN_FREE_DISK_GB:
        raise RuntimeError("Tier-2 disk safety stop")
    if output.exists():
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if size / 1024**3 > OUTPUT_CAP_GB:
            raise RuntimeError("Tier-2 output-cap safety stop")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    if rss > MAX_RSS_GB:
        raise RuntimeError("Tier-2 RSS safety stop")


class HiddenModulatorDGP:
    mechanism = "d3_hidden_modulator"

    def __init__(self, context: int, rho: float = 0.85) -> None:
        self.context = int(context)
        base = rho ** np.arange(12, dtype=float)
        self.kernel = 0.75 * base / np.linalg.norm(base)
        self.source_sd = 0.7
        self.latent_sd = 0.45
        self.noise_sd = 0.55
        self.gain = 2.0
        self.alpha = float(logit(0.30))
        nodes, weights = hermgauss(32)
        self.gh_nodes = np.sqrt(2.0) * nodes
        self.gh_weights = weights / np.sqrt(np.pi)

    def sample_histories(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(scale=self.source_sd, size=(int(n), self.context))

    def latent_mean_sd(self, h: np.ndarray) -> tuple[np.ndarray, float]:
        h = np.asarray(h, dtype=float)
        observed = min(self.context, self.kernel.size)
        mean = h[:, :observed] @ self.kernel[:observed]
        missing_variance = self.source_sd**2 * np.sum(self.kernel[observed:] ** 2)
        return mean, float(np.sqrt(self.latent_sd**2 + missing_variance))

    def sample(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean, sd = self.latent_mean_sd(h)
        latent = mean + sd * rng.normal(size=mean.size)
        probability = expit(self.alpha + latent)
        state = rng.binomial(1, probability)
        return self.gain * (state - probability) + self.noise_sd * rng.normal(size=mean.size)

    def sample_vector(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        value = self.sample(h, rng)
        return np.stack([value, rng.normal(size=value.size)], axis=1)

    def _quadrature(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, sd = self.latent_mean_sd(h)
        latent = mean[:, None] + sd * self.gh_nodes[None, :]
        probability = expit(self.alpha + latent)
        return probability, self.gh_weights

    def lag_effect(self, h: np.ndarray, channel: str) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        if channel == "mean":
            return np.zeros_like(h)
        if channel != "variance":
            raise KeyError(channel)
        probability, weights = self._quadrature(h)
        derivative_factor = np.sum(
            weights[None, :] * probability * (1.0 - probability) * (1.0 - 2.0 * probability),
            axis=1,
        )
        result = np.zeros_like(h)
        active = min(self.context, self.kernel.size)
        result[:, :active] = (
            self.gain**2
            * derivative_factor[:, None]
            * self.kernel[None, :active]
        )
        return result

    def target(self, h: np.ndarray, channel: str) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        if channel == "mean":
            return np.zeros(h.shape[0])
        if channel != "variance":
            raise KeyError(channel)
        probability, weights = self._quadrature(h)
        return self.noise_sd**2 + self.gain**2 * np.sum(
            weights[None, :] * probability * (1.0 - probability), axis=1
        )


class ObservationFilterDGP:
    mechanism = "d5_observation_filter"

    def __init__(self, filter_weight: float, observation_sd: float, null: bool = False) -> None:
        self.lags = 7
        base = 0.8 ** np.arange(6, dtype=float)
        self.kernel = 0.75 * base / np.linalg.norm(base)
        self.filter_weight = float(filter_weight)
        self.observation_sd = float(observation_sd)
        self.null = bool(null)
        self.alpha = float(logit(0.30))
        self.gain = 2.0
        self.noise_sd = 0.55

    def sample_histories(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(scale=0.7, size=(int(n), self.lags))

    def probabilities(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.asarray(h, dtype=float)
        if self.null:
            constant = np.full(h.shape[0], expit(self.alpha))
            return constant, constant
        return (
            expit(self.alpha + h[:, :6] @ self.kernel),
            expit(self.alpha + h[:, 1:7] @ self.kernel),
        )

    def _latent_draw(self, probability: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        state = rng.binomial(1, probability)
        return self.gain * (state - probability) + self.noise_sd * rng.normal(
            size=probability.size
        )

    def sample(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        current, previous = self.probabilities(h)
        value = self._latent_draw(current, rng)
        value += self.filter_weight * self._latent_draw(previous, rng)
        value += self.observation_sd * rng.normal(size=value.size)
        return value

    def sample_vector(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        value = self.sample(h, rng)
        return np.stack([value, rng.normal(size=value.size)], axis=1)

    def lag_effect(self, h: np.ndarray, channel: str) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        result = np.zeros_like(h)
        if channel == "mean" or self.null:
            return result
        if channel != "variance":
            raise KeyError(channel)
        current, previous = self.probabilities(h)
        current_factor = self.gain**2 * current * (1.0 - current) * (1.0 - 2.0 * current)
        previous_factor = self.gain**2 * previous * (1.0 - previous) * (1.0 - 2.0 * previous)
        result[:, :6] += current_factor[:, None] * self.kernel[None, :]
        result[:, 1:7] += (
            self.filter_weight**2 * previous_factor[:, None] * self.kernel[None, :]
        )
        return result

    def target(self, h: np.ndarray, channel: str) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        if channel == "mean":
            return np.zeros(h.shape[0])
        if channel != "variance":
            raise KeyError(channel)
        current, previous = self.probabilities(h)
        current_variance = self.noise_sd**2 + self.gain**2 * current * (1.0 - current)
        previous_variance = self.noise_sd**2 + self.gain**2 * previous * (1.0 - previous)
        return (
            current_variance
            + self.filter_weight**2 * previous_variance
            + self.observation_sd**2
        )


@dataclass
class GaussianLogVarianceRegressor:
    ridge: float = 1e-2

    def fit(self, h: np.ndarray, y: np.ndarray) -> "GaussianLogVarianceRegressor":
        design = history_design(h)
        self.mean_model = Ridge(alpha=self.ridge, fit_intercept=False).fit(design, y)
        residual = y - self.mean_model.predict(design)
        self.logvar_model = Ridge(alpha=self.ridge, fit_intercept=False).fit(
            design, np.log(np.maximum(residual**2, 0.04))
        )
        return self

    def effect(self, h: np.ndarray, channel: str) -> np.ndarray:
        derivative_design = history_design_derivative(h)
        if channel == "mean":
            return np.einsum("nqd,d->nq", derivative_design, self.mean_model.coef_)
        if channel == "variance":
            design = history_design(h)
            logvar = self.logvar_model.predict(design)
            derivative = np.einsum("nqd,d->nq", derivative_design, self.logvar_model.coef_)
            return np.exp(logvar)[:, None] * derivative
        raise KeyError(channel)


def sample_centered_score_effect(
    model: DynamicAnchoredScore,
    dgp,
    histories: np.ndarray,
    channel: str,
    rng: np.random.Generator,
    draws: int = 384,
) -> np.ndarray:
    grid = np.linspace(-6.0, 6.0, 301)
    result = np.zeros((histories.shape[0], histories.shape[1]))
    for index, history in enumerate(histories):
        repeated_grid = np.repeat(history[None, :], grid.size, axis=0)
        field = model.mixed(repeated_grid, grid, PRIMARY_SIGMA)
        potential = np.stack(
            [cumulative_trapezoid(field[:, lag], grid, initial=0.0) for lag in range(history.size)],
            axis=1,
        )
        repeated = np.repeat(history[None, :], draws, axis=0)
        value = dgp.sample(repeated, rng) + PRIMARY_SIGMA * rng.normal(size=draws)
        tangent = np.stack(
            [np.interp(value, grid, potential[:, lag]) for lag in range(history.size)], axis=1
        )
        tangent -= np.mean(tangent, axis=0, keepdims=True)
        vector = np.stack([value, np.zeros_like(value)], axis=1)
        influence = influence_dynamic(vector, channel)
        result[index] = np.mean(influence[:, None] * tangent, axis=0)
    return result


def metric_rows(
    seed: int,
    cell: str,
    dgp,
    channel: str,
    method: str,
    access: str,
    values: dict[str, float],
    n_train: int,
    context: int,
    wall_time: float,
) -> list[dict[str, object]]:
    rows = []
    for name, value in values.items():
        rows.append(
            {
                "study_id": "sid_tier2_robustness_20260714",
                "phase": "tier2_confirmatory",
                "cell_id": cell,
                "dgp_family": dgp.mechanism,
                "dgp_seed": seed,
                "method_id": method,
                "estimand_type": "dynamic_local",
                "channel": channel,
                "information_access": access,
                "metric_name": name,
                "metric_value": float(value),
                "n_train": n_train,
                "context_length": context,
                "status": "ok",
                "failure_code": "",
                "wall_time": wall_time,
            }
        )
    return rows


def run_case(dgp, seed: int, n_train: int, cell: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed + int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16))
    h = dgp.sample_histories(n_train, rng)
    y = dgp.sample(h, rng)
    y_vector = np.stack([y, np.zeros_like(y)], axis=1)
    evaluation = dgp.sample_histories(32, rng)
    direct = DynamicMomentRegressor(ridge=1e-2).fit(h, y_vector)
    gaussian = GaussianLogVarianceRegressor(ridge=1e-2).fit(h, y)
    critic = DynamicRatioCritic().fit(h, y_vector, rng)
    score = DynamicAnchoredScore(ridge=1.0).fit(h, y, rng)
    arrays: dict[str, np.ndarray] = {"evaluation_histories": evaluation}
    rows: list[dict[str, object]] = []
    for channel in ["mean", "variance"]:
        truth_matrix = dgp.lag_effect(evaluation, channel)
        truth = np.mean(truth_matrix, axis=0)
        direct_matrix = direct.effect(evaluation, channel)
        gaussian_matrix = gaussian.effect(evaluation, channel)
        ratio_matrix = ratio_effect(critic, dgp, evaluation, channel, rng, draws=256)
        score_matrix = sample_centered_score_effect(score, dgp, evaluation, channel, rng)
        estimates = {
            "B0_ZERO": (np.zeros_like(truth), "observational"),
            "B2_RAW_MOMENTS": (np.mean(direct_matrix, axis=0), "observational"),
            "B4_GAUSSIAN_LOGVAR": (np.mean(gaussian_matrix, axis=0), "observational"),
            "A2_RATIO_CRITIC": (
                np.mean(ratio_matrix, axis=0),
                "observational_training_plus_conditional_evaluation_queries",
            ),
            "S7_ANCHORED_ENERGY__A7_HODGE": (
                np.mean(score_matrix, axis=0),
                "observational_training_plus_conditional_evaluation_queries",
            ),
        }
        arrays[f"{channel}__truth_matrix"] = truth_matrix
        arrays[f"{channel}__truth_profile"] = truth
        arrays[f"{channel}__b2_matrix"] = direct_matrix
        arrays[f"{channel}__b4_matrix"] = gaussian_matrix
        arrays[f"{channel}__ratio_matrix"] = ratio_matrix
        arrays[f"{channel}__score_matrix"] = score_matrix
        for method, (estimate, access) in estimates.items():
            arrays[f"{channel}__{method.lower()}__profile"] = estimate
            if channel == "mean":
                metrics = {
                    "mean_null_rms": float(np.sqrt(np.mean(estimate**2))),
                    "mean_null_max_abs": float(np.max(np.abs(estimate))),
                }
            elif np.sqrt(np.mean(truth**2)) <= 1e-12:
                metrics = {
                    "null_rms": float(np.sqrt(np.mean(estimate**2))),
                    "null_max_abs": float(np.max(np.abs(estimate))),
                }
            else:
                metrics = lag_metrics(estimate, truth)
            rows.extend(
                metric_rows(
                    seed,
                    cell,
                    dgp,
                    channel,
                    method,
                    access,
                    metrics,
                    n_train,
                    evaluation.shape[1],
                    time.perf_counter() - started,
                )
            )
    return pd.DataFrame(rows), arrays


def cells() -> list[tuple[str, object, int]]:
    result: list[tuple[str, object, int]] = []
    for context in [2, 12, 20]:
        result.append((f"d3_context_{context}__n_8000", HiddenModulatorDGP(context), 8000))
    for n_train in [2000, 32000]:
        result.append((f"d3_context_12__n_{n_train}", HiddenModulatorDGP(12), n_train))
    regimes = [
        ("clean", 0.0, 0.0, False),
        ("primary_filter", 0.35, 0.25, False),
        ("strong_filter", 0.65, 0.25, False),
        ("observation_only_null", 0.65, 0.25, True),
    ]
    for name, weight, noise, null in regimes:
        result.append(
            (
                f"d5_{name}__n_8000",
                ObservationFilterDGP(weight, noise, null),
                8000,
            )
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--upstream-package", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    manifest = {
        "study_id": "sid_tier2_robustness_20260714",
        "status": "running",
        "started_unix": time.time(),
        "seeds": list(SEEDS),
        "expected_cases": len(cells()) * len(SEEDS),
        "script_sha256": sha256_file(script_path),
        "preregistration_sha256": sha256_file(args.preregistration.resolve()),
        "upstream_package_sha256": sha256_file(args.upstream_package.resolve()),
        "resource_limits": {
            "threads": 1,
            "max_rss_gb": MAX_RSS_GB,
            "max_output_gb": OUTPUT_CAP_GB,
            "min_free_disk_gb": MIN_FREE_DISK_GB,
        },
    }
    atomic_json(output / "manifest.json", manifest)
    completed = 0
    failures = 0
    for cell, dgp, n_train in cells():
        for seed in SEEDS:
            enforce_resources(output)
            csv_path = cases_dir / f"{cell}__seed_{seed}.csv"
            npz_path = cases_dir / f"{cell}__seed_{seed}.npz"
            if csv_path.exists() and npz_path.exists():
                completed += 1
                continue
            try:
                frame, arrays = run_case(dgp, seed, n_train, cell)
                atomic_npz(npz_path, arrays)
                frame["array_uri"] = str(npz_path.relative_to(output))
                frame["array_sha256"] = sha256_file(npz_path)
                atomic_csv(csv_path, frame)
                completed += 1
            except Exception as exc:
                failures += 1
                atomic_csv(
                    csv_path,
                    pd.DataFrame(
                        [
                            {
                                "study_id": "sid_tier2_robustness_20260714",
                                "phase": "tier2_confirmatory",
                                "cell_id": cell,
                                "dgp_seed": seed,
                                "status": "failed",
                                "failure_code": type(exc).__name__,
                                "failure_message": str(exc),
                            }
                        ]
                    ),
                )
                if failures / (completed + failures) > 0.10:
                    raise
    manifest.update(
        {
            "status": "complete",
            "completed_cases": completed,
            "failed_cases": failures,
            "finished_unix": time.time(),
        }
    )
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    raise SystemExit(main())
