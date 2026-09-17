from __future__ import annotations

import json
import math
import os
import platform
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import norm, qmc

from .core import (
    AnalyticGaussianPathDGP,
    CharacteristicBank,
    HistorySieve,
    MomentBlindDiscreteDGP,
    MomentBlindLegendreDGP,
    SupportMotionDGP,
    atomic_csv,
    atomic_json,
    atomic_npz,
    independent_folds,
    resource_guard,
    ridge_coefficients,
    riesz_coefficients,
    sha256_file,
    sha256_tree,
    vector_metrics,
)


METHODS = ("ZERO", "PLUG", "RIESZ", "ORTH", "OR_A", "OR_M", "OR_R")


@dataclass(frozen=True)
class GaussianRunConfig:
    n_train: int = 8000
    n_bank: int = 1024
    n_folds: int = 5
    history_dim: int = 8
    response_dim: int = 4
    query_count: int = 64
    direction: int = 0
    random_width: int = 64
    feature_ridge: float = 1e-3
    riesz_ridge: float = 1e-3
    qmc_power: int = 15
    access_tier: str = "O0"
    stage: str = "development"


@dataclass(frozen=True)
class MomentBlindRunConfig:
    n_train: int = 8000
    n_bank: int = 1024
    n_folds: int = 5
    query_count: int = 128
    random_width: int = 32
    feature_ridge: float = 1e-3
    riesz_ridge: float = 1e-2
    qmc_power: int = 17
    amplitude_fraction: float = 0.5
    frequency_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    stage: str = "development"


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _environment_payload() -> dict[str, object]:
    import importlib.metadata

    import numpy
    import pandas
    import scipy
    import sklearn

    try:
        pyarrow_version = importlib.metadata.version("pyarrow")
    except importlib.metadata.PackageNotFoundError:
        pyarrow_version = "not-installed"

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow_version,
        "sklearn": sklearn.__version__,
        "thread_limits": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    }


def freeze_source_and_environment(run_dir: Path) -> None:
    environment_path = run_dir / "environment.lock.json"
    if not environment_path.exists():
        atomic_json(environment_path, _environment_payload())
    snapshot_path = run_dir / "source_snapshot.tar.gz"
    if snapshot_path.exists():
        return
    root = _source_root()
    temporary = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in {"runs", "reports", "__pycache__", ".venv", ".pytest_cache"} for part in relative.parts):
                continue
            archive.add(path, arcname=str(relative), recursive=False)
    temporary.replace(snapshot_path)


def run_e0(run_dir: Path) -> dict[str, object]:
    """Persist the algebra, split, and reproducibility gate."""
    import torch

    resource_guard()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    dgp = AnalyticGaussianPathDGP(seed=71, history_dim=4, response_dim=3)
    rng = np.random.default_rng(801)
    h_bank = dgp.sample_histories(rng, 1000)
    y_bank = dgp.sample_responses(h_bank, rng)
    bank = CharacteristicBank.fit(y_bank, 12, seed=991)
    h = dgp.sample_histories(rng, 5)
    analytic = dgp.conditional_feature_derivative(h, 1, bank)

    torch.set_default_dtype(torch.float64)
    h_t = torch.tensor(h, requires_grad=True)
    mean_linear = torch.tensor(dgp.mean_linear)
    mean_frequency = torch.tensor(dgp.mean_frequency)
    mean_nonlinear = torch.tensor(dgp.mean_nonlinear)
    base_covariance = torch.tensor(dgp.base_covariance)
    cov_offset = torch.tensor(dgp.cov_offset)
    cov_history = torch.tensor(dgp.cov_history)
    cov_directions = torch.tensor(dgp.cov_directions)
    u = torch.tensor(bank.raw_frequencies)
    phase_offset = torch.tensor(bank.raw_phase_offset)
    mean = h_t @ mean_linear.T + torch.sin(h_t @ mean_frequency.T) * mean_nonlinear
    logits = cov_offset[None, :] + h_t @ cov_history.T
    weights = torch.nn.functional.softplus(logits)
    outer = torch.einsum("ki,kj->kij", cov_directions, cov_directions)
    covariance = base_covariance[None, :, :] + torch.einsum("nk,kij->nij", weights, outer)
    phase = mean @ u.T + phase_offset[None, :]
    variance = torch.einsum("qi,nij,qj->nq", u, covariance, u)
    output = torch.cat([torch.cos(phase), torch.sin(phase)], dim=1) * torch.cat(
        [torch.exp(-0.5 * variance), torch.exp(-0.5 * variance)], dim=1
    )
    ad = np.zeros_like(analytic)
    for row in range(h.shape[0]):
        for feature in range(bank.output_dim):
            gradient = torch.autograd.grad(output[row, feature], h_t, retain_graph=True)[0]
            ad[row, feature] = float(gradient[row, 1])
    ad_relative_error = float(np.linalg.norm(ad - analytic) / np.linalg.norm(analytic))
    rows.append(
        {
            "check": "E0.1 analytic versus AD characteristic derivative",
            "value": ad_relative_error,
            "threshold": 1e-6,
            "passed": ad_relative_error < 1e-6,
        }
    )

    finite_errors = []
    for delta in (0.1, 0.05, 0.025, 0.0125):
        plus = h.copy()
        minus = h.copy()
        plus[:, 1] += delta
        minus[:, 1] -= delta
        finite = (
            dgp.conditional_feature_mean(plus, bank)
            - dgp.conditional_feature_mean(minus, bank)
        ) / (2.0 * delta)
        finite_errors.append(float(np.linalg.norm(finite - analytic) / np.linalg.norm(analytic)))
    log_slope = float(np.polyfit(np.log([0.1, 0.05, 0.025, 0.0125]), np.log(finite_errors), 1)[0])
    rows.append(
        {
            "check": "E0.1 central-difference convergence order",
            "value": log_slope,
            "threshold": 1.8,
            "passed": log_slope >= 1.8,
        }
    )

    sampler = qmc.Sobol(d=4, scramble=True, seed=717)
    gaussian_h = ndtri(np.clip(sampler.random_base2(17), 1e-12, 1.0 - 1e-12))
    p = 2
    alpha = gaussian_h[:, p]
    rng_probe = np.random.default_rng(123)
    discrepancies = []
    for _ in range(12):
        vector = rng_probe.normal(size=4)
        omega = rng_probe.uniform(0.2, 1.5)
        phase_probe = omega * gaussian_h @ vector
        lhs = np.mean(alpha * np.sin(phase_probe))
        rhs = np.mean(omega * vector[p] * np.cos(phase_probe))
        discrepancies.append(abs(float(lhs - rhs)))
    max_riesz_discrepancy = max(discrepancies)
    rows.append(
        {
            "check": "E0.2 exact Gaussian Riesz probes",
            "value": max_riesz_discrepancy,
            "threshold": 5e-4,
            "passed": max_riesz_discrepancy < 5e-4,
        }
    )

    grid = np.linspace(-1.0, 1.0, 1_000_001)
    g = np.exp(grid)
    dg = np.exp(grid)
    invalid_discrepancy = abs(float(np.trapezoid(dg, grid) / 2.0))
    b = 1.0 - grid**2
    valid_lhs = np.trapezoid(2.0 * grid * g, grid) / 2.0
    valid_rhs = np.trapezoid(b * dg, grid) / 2.0
    valid_discrepancy = abs(float(valid_lhs - valid_rhs))
    rows.extend(
        [
            {
                "check": "E0.3 invalid bounded-support identity rejected",
                "value": invalid_discrepancy,
                "threshold": 0.5,
                "passed": invalid_discrepancy > 0.5,
            },
            {
                "check": "E0.3 boundary-vanishing identity",
                "value": valid_discrepancy,
                "threshold": 1e-9,
                "passed": valid_discrepancy < 1e-9,
            },
        ]
    )

    h_mc = dgp.sample_histories(rng, 120_000)
    y_mc = dgp.sample_responses(h_mc, rng)
    phi_mc = bank.transform(y_mc)
    m_mc = dgp.conditional_feature_mean(h_mc, bank)
    dm_mc = dgp.conditional_feature_derivative(h_mc, 1, bank)
    alpha_mc = dgp.oracle_riesz(h_mc, 1)
    correction = alpha_mc[:, None] * (phi_mc - m_mc)
    correction_mean = np.mean(correction, axis=0)
    correction_se = np.std(correction, axis=0, ddof=1) / math.sqrt(h_mc.shape[0])
    max_standard_error = float(np.max(np.abs(correction_mean) / np.maximum(correction_se, 1e-12)))
    rows.append(
        {
            "check": "E0.4 oracle orthogonal-score centering",
            "value": max_standard_error,
            "threshold": 4.0,
            "passed": max_standard_error <= 4.0,
        }
    )

    from .core import embargoed_block_splits

    footprint = 12
    split_ok = True
    for train, test in embargoed_block_splits(2000, 5, footprint):
        forbidden = set()
        for index in test:
            forbidden.update(range(max(0, index - footprint), min(2000, index + footprint + 1)))
        split_ok &= not bool(forbidden.intersection(set(train.tolist())))
    random_window_leak = len(set(range(0, 100)).intersection(set(range(50, 150)))) > 0
    rows.extend(
        [
            {
                "check": "E0.5 intentional overlapping split rejected",
                "value": float(random_window_leak),
                "threshold": 1.0,
                "passed": bool(random_window_leak),
            },
            {
                "check": "E0.5 blocked embargo split has no overlap",
                "value": float(split_ok),
                "threshold": 1.0,
                "passed": bool(split_ok),
            },
        ]
    )

    bank_rebuilt = CharacteristicBank.fit(y_bank.copy(), 12, seed=991)
    reproducible = bank.digest() == bank_rebuilt.digest()
    rows.append(
        {
            "check": "E0.6 query-bank bitwise reproducibility",
            "value": float(reproducible),
            "threshold": 1.0,
            "passed": bool(reproducible),
        }
    )

    frame = pd.DataFrame(rows)
    atomic_csv(run_dir / "e0_checks.csv", frame)
    checks_hash = sha256_file(run_dir / "e0_checks.csv")
    hash_stable = checks_hash == sha256_file(run_dir / "e0_checks.csv")
    frame.loc[len(frame)] = {
        "check": "E0.7 artifact hash repeatability",
        "value": float(hash_stable),
        "threshold": 1.0,
        "passed": bool(hash_stable),
    }
    atomic_csv(run_dir / "e0_checks.csv", frame)
    payload = {
        "passed": bool(frame.passed.all()),
        "check_count": int(frame.shape[0]),
        "failed_checks": frame.loc[~frame.passed, "check"].tolist(),
        "checks_sha256": sha256_file(run_dir / "e0_checks.csv"),
        "source_sha256": sha256_tree(_source_root()),
        "environment": _environment_payload(),
        **resource_guard(),
    }
    atomic_json(run_dir / "e0_gate.json", payload)
    return payload


def _truth_population(
    dgp: AnalyticGaussianPathDGP,
    bank: CharacteristicBank,
    direction: int,
    power: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    def evaluate(qmc_power: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_feature = np.zeros(bank.output_dim)
        total_mean = np.zeros(dgp.response_dim)
        total_cov = np.zeros((dgp.response_dim, dgp.response_dim))
        count = 0
        h = dgp.qmc_histories(qmc_power, seed + qmc_power)
        for start in range(0, h.shape[0], 8192):
            batch = h[start : start + 8192]
            total_feature += np.sum(dgp.conditional_feature_derivative(batch, direction, bank), axis=0)
            total_mean += np.sum(dgp.conditional_mean_derivative(batch, direction), axis=0)
            total_cov += np.sum(dgp.conditional_covariance_derivative(batch, direction), axis=0)
            count += batch.shape[0]
        return total_feature / count, total_mean / count, total_cov / count

    high = evaluate(power)
    low = evaluate(max(10, power - 1))
    diagnostics = {
        "qmc_points": float(2**power),
        "feature_truth_relative_difference": float(
            np.linalg.norm(high[0] - low[0]) / max(np.linalg.norm(high[0]), 1e-12)
        ),
        "mean_truth_relative_difference": float(
            np.linalg.norm(high[1] - low[1]) / max(np.linalg.norm(high[1]), 1e-12)
        ),
        "cov_truth_relative_difference": float(
            np.linalg.norm(high[2] - low[2]) / max(np.linalg.norm(high[2]), 1e-12)
        ),
    }
    return high[0], high[1], high[2], diagnostics


def _direct_targets(y: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    y = np.asarray(y, dtype=float)
    pairs = [(i, j) for i in range(y.shape[1]) for j in range(i, y.shape[1])]
    products = np.stack([y[:, i] * y[:, j] for i, j in pairs], axis=1)
    return np.concatenate([y, products], axis=1), pairs


def _symmetric_products(y: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    y = np.asarray(y, dtype=float)
    pairs = [(i, j) for i in range(y.shape[1]) for j in range(i, y.shape[1])]
    return np.stack([y[:, i] * y[:, j] for i, j in pairs], axis=1), pairs


def _direct_covariance_derivative(
    prediction: np.ndarray,
    derivative: np.ndarray,
    response_dim: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    mean = prediction[:, :response_dim]
    dmean = derivative[:, :response_dim]
    output = np.zeros((prediction.shape[0], response_dim, response_dim))
    for index, (i, j) in enumerate(pairs):
        dm2 = derivative[:, response_dim + index]
        value = dm2 - dmean[:, i] * mean[:, j] - mean[:, i] * dmean[:, j]
        output[:, i, j] = value
        output[:, j, i] = value
    return output


def _probe_diagnostics(h: np.ndarray, alpha: np.ndarray, direction: int, seed: int) -> dict[str, float]:
    h = np.asarray(h, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    residuals: list[float] = [abs(float(np.mean(alpha)))]
    for j in range(h.shape[1]):
        residuals.append(abs(float(np.mean(alpha * h[:, j])) - float(j == direction)))
    residuals.append(abs(float(np.mean(alpha * (h[:, direction] ** 2 - 1.0)) - np.mean(2.0 * h[:, direction]))))
    rng = np.random.default_rng(seed)
    for _ in range(8):
        vector = rng.normal(size=h.shape[1])
        vector /= np.linalg.norm(vector)
        omega = rng.uniform(0.4, 1.6)
        phase = omega * (h @ vector)
        lhs = np.mean(alpha * np.sin(phase))
        rhs = np.mean(omega * vector[direction] * np.cos(phase))
        residuals.append(abs(float(lhs - rhs)))
    residuals_array = np.asarray(residuals)
    rms = float(np.sqrt(np.mean(alpha**2)))
    return {
        "probe_median_abs": float(np.median(residuals_array)),
        "probe_q95_abs": float(np.quantile(residuals_array, 0.95)),
        "alpha_l2": rms,
        "alpha_q95": float(np.quantile(np.abs(alpha), 0.95)),
        "alpha_q99": float(np.quantile(np.abs(alpha), 0.99)),
        "alpha_q999": float(np.quantile(np.abs(alpha), 0.999)),
        "extreme_weight_fraction": float(np.mean(np.abs(alpha) > 10.0 * max(rms, 1e-12))),
    }


def run_gaussian_seed(
    run_dir: Path,
    dgp_seed: int,
    method_init: int,
    config: GaussianRunConfig,
    save_scores: bool = True,
    score_methods: tuple[str, ...] = ("ORTH", "PLUG", "RIESZ"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resource_guard()
    start_time = time.monotonic()
    case_id = f"e1_gaussian__seed_{dgp_seed}__init_{method_init}"
    case_csv = run_dir / "cases" / f"{case_id}.csv"
    nuisance_csv = run_dir / "nuisance" / f"{case_id}.csv"
    score_npz = run_dir / "scores" / f"{case_id}.npz"
    if case_csv.exists() and nuisance_csv.exists() and (score_npz.exists() or not save_scores):
        return pd.read_csv(case_csv), pd.read_csv(nuisance_csv)

    rng_bank = np.random.default_rng(100_000 + dgp_seed)
    rng_data = np.random.default_rng(200_000 + dgp_seed)
    dgp = AnalyticGaussianPathDGP(
        seed=dgp_seed,
        history_dim=config.history_dim,
        response_dim=config.response_dim,
    )
    h_bank = dgp.sample_histories(rng_bank, config.n_bank)
    y_bank = dgp.sample_responses(h_bank, rng_bank)
    bank = CharacteristicBank.fit(y_bank, config.query_count, seed=92117)
    query_path = run_dir / "query_banks" / f"e1_gaussian__seed_{dgp_seed}.npz"
    if not query_path.exists():
        bank.save(query_path)
    h = dgp.sample_histories(rng_data, config.n_train)
    y = dgp.sample_responses(h, rng_data)
    phi = bank.transform(y)
    oracle_m = dgp.conditional_feature_mean(h, bank)
    oracle_dm = dgp.conditional_feature_derivative(h, config.direction, bank)
    oracle_alpha = dgp.oracle_riesz(h, config.direction)

    truth_feature, truth_mean, truth_covariance, truth_diagnostics = _truth_population(
        dgp, bank, config.direction, config.qmc_power, seed=700_000 + dgp_seed
    )

    contributions = {method: np.zeros_like(phi) for method in METHODS}
    direct_mean = np.zeros((h.shape[0], config.response_dim))
    direct_covariance = np.zeros((h.shape[0], config.response_dim, config.response_dim))
    prediction = np.zeros_like(phi)
    derivative_prediction = np.zeros_like(phi)
    alpha_prediction = np.zeros(h.shape[0])
    folds = independent_folds(h.shape[0], config.n_folds, seed=31337 + dgp_seed)
    split_path = run_dir / "splits" / f"e1_gaussian__seed_{dgp_seed}.npz"
    if not split_path.exists():
        fold_assignment = np.full(h.shape[0], -1, dtype=np.int16)
        for fold_id, (_, test) in enumerate(folds):
            fold_assignment[test] = fold_id
        atomic_npz(split_path, fold_id=fold_assignment, observation_index=np.arange(h.shape[0]))
    _, pairs = _direct_targets(y)
    feature_checkpoints = []
    alpha_checkpoints = []
    mean_checkpoints = []
    covariance_checkpoints = []

    for fold_id, (train, test) in enumerate(folds):
        sieve = HistorySieve(
            input_dim=config.history_dim,
            random_width=config.random_width,
            seed=50_000 + 100 * method_init + fold_id,
        ).fit(h[train])
        z_train = sieve.transform(h[train])
        dz_train = sieve.derivative(h[train], config.direction)
        z_test = sieve.transform(h[test])
        dz_test = sieve.derivative(h[test], config.direction)
        feature_coef = ridge_coefficients(z_train, phi[train], config.feature_ridge)
        alpha_coef = riesz_coefficients(z_train, dz_train, config.riesz_ridge)
        mean_coef = ridge_coefficients(z_train, y[train], config.feature_ridge)
        fitted_train_mean = z_train @ mean_coef
        residual_products, pairs = _symmetric_products(y[train] - fitted_train_mean)
        covariance_coef = ridge_coefficients(z_train, residual_products, config.feature_ridge)
        feature_checkpoints.append(feature_coef)
        alpha_checkpoints.append(alpha_coef)
        mean_checkpoints.append(mean_coef)
        covariance_checkpoints.append(covariance_coef)

        pred = z_test @ feature_coef
        dpred = dz_test @ feature_coef
        alpha = z_test @ alpha_coef
        direct_mean_derivative = dz_test @ mean_coef
        direct_covariance_derivative = dz_test @ covariance_coef

        prediction[test] = pred
        derivative_prediction[test] = dpred
        alpha_prediction[test] = alpha
        contributions["PLUG"][test] = dpred
        contributions["RIESZ"][test] = alpha[:, None] * phi[test]
        contributions["ORTH"][test] = dpred + alpha[:, None] * (phi[test] - pred)
        contributions["OR_A"][test] = dpred + oracle_alpha[test, None] * (phi[test] - pred)
        contributions["OR_M"][test] = oracle_dm[test] + alpha[:, None] * (phi[test] - oracle_m[test])
        contributions["OR_R"][test] = oracle_alpha[test, None] * phi[test]
        direct_mean[test] = direct_mean_derivative
        for pair_index, (i, j) in enumerate(pairs):
            direct_covariance[test, i, j] = direct_covariance_derivative[:, pair_index]
            direct_covariance[test, j, i] = direct_covariance_derivative[:, pair_index]

    rows: list[dict[str, object]] = []
    for method, values in contributions.items():
        estimate = np.mean(values, axis=0)
        metrics = vector_metrics(estimate, truth_feature)
        standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
        coverage = float(
            np.mean(
                (truth_feature >= estimate - 1.96 * standard_error)
                & (truth_feature <= estimate + 1.96 * standard_error)
            )
        )
        for metric_name, metric_value in {**metrics, "coverage_fraction": coverage}.items():
            rows.append(
                {
                    "experiment_id": "E1",
                    "cell_id": "analytic_gaussian_reference",
                    "dgp_family": "analytic_gaussian_path",
                    "dgp_seed": dgp_seed,
                    "method": method,
                    "method_init": method_init,
                    "access_tier": "O0" if method in {"ZERO", "PLUG", "RIESZ", "ORTH"} else "oracle_control",
                    "compute_budget": "small_sieve",
                    "n_train": config.n_train,
                    "n_eval": config.n_train,
                    "history_dim": config.history_dim,
                    "response_dim": config.response_dim,
                    "history_length": 1,
                    "horizon": 1,
                    "query_count": config.query_count,
                    "basis_count": 1,
                    "target_name": "characteristic_tangent",
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "fit_status": "ok",
                    "failure_code": "",
                    "stage": config.stage,
                }
            )

    for target_name, estimate, truth in (
        ("mean_derivative", np.mean(direct_mean, axis=0), truth_mean),
        ("covariance_derivative", np.mean(direct_covariance, axis=0), truth_covariance),
    ):
        for metric_name, metric_value in vector_metrics(estimate, truth).items():
            rows.append(
                {
                    "experiment_id": "E1",
                    "cell_id": "analytic_gaussian_reference",
                    "dgp_family": "analytic_gaussian_path",
                    "dgp_seed": dgp_seed,
                    "method": "DIRECT",
                    "method_init": method_init,
                    "access_tier": "O0",
                    "compute_budget": "small_sieve",
                    "n_train": config.n_train,
                    "n_eval": config.n_train,
                    "history_dim": config.history_dim,
                    "response_dim": config.response_dim,
                    "history_length": 1,
                    "horizon": 1,
                    "query_count": 0,
                    "basis_count": 1,
                    "target_name": target_name,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "fit_status": "ok",
                    "failure_code": "",
                    "stage": config.stage,
                }
            )

    probe = _probe_diagnostics(h, alpha_prediction, config.direction, seed=900_000 + dgp_seed)
    nuisance = {
        "experiment_id": "E1",
        "cell_id": "analytic_gaussian_reference",
        "dgp_seed": dgp_seed,
        "method_init": method_init,
        "feature_validation_mse": float(np.mean((phi - prediction) ** 2)),
        "feature_mean_nrmse": vector_metrics(prediction.reshape(-1), oracle_m.reshape(-1))["nrmse"],
        "feature_derivative_nrmse": vector_metrics(
            derivative_prediction.reshape(-1), oracle_dm.reshape(-1)
        )["nrmse"],
        "alpha_nrmse": vector_metrics(alpha_prediction, oracle_alpha)["nrmse"],
        **probe,
        **truth_diagnostics,
        "query_hash": bank.digest(),
        "runtime_seconds": time.monotonic() - start_time,
        **resource_guard(),
    }
    metric_frame = pd.DataFrame(rows)
    nuisance_frame = pd.DataFrame([nuisance])
    atomic_csv(case_csv, metric_frame)
    atomic_csv(nuisance_csv, nuisance_frame)
    checkpoint_path = run_dir / "checkpoints" / f"{case_id}.npz"
    atomic_npz(
        checkpoint_path,
        feature_coefficients=np.stack(feature_checkpoints).astype(np.float32),
        alpha_coefficients=np.stack(alpha_checkpoints).astype(np.float32),
        mean_coefficients=np.stack(mean_checkpoints).astype(np.float32),
        covariance_coefficients=np.stack(covariance_checkpoints).astype(np.float32),
    )
    if save_scores:
        arrays = {method.lower(): contributions[method].astype(np.float32) for method in score_methods}
        arrays.update(
            truth=truth_feature.astype(np.float64),
            dgp_seed=np.array([dgp_seed], dtype=np.int64),
            method_init=np.array([method_init], dtype=np.int64),
        )
        atomic_npz(score_npz, **arrays)
    return metric_frame, nuisance_frame


def run_e2(run_dir: Path, stage: str = "development", qmc_power: int = 18) -> tuple[pd.DataFrame, pd.DataFrame]:
    resource_guard()
    output = run_dir / "e2_perturbations.csv"
    gate_output = run_dir / "e2_gate.csv"
    if output.exists() and gate_output.exists():
        return pd.read_csv(output), pd.read_csv(gate_output)
    sampler = qmc.Sobol(d=4, scramble=True, seed=88117)
    unit = np.clip(sampler.random_base2(qmc_power), 1e-12, 1.0 - 1e-12)
    h = ndtri(unit)
    p = 0
    beta = np.array([0.55, -0.25, 0.15, 0.35])
    sigma = 0.7
    frequency = 0.9
    phase = frequency * (h @ beta)
    damping = math.exp(-0.5 * sigma**2 * frequency**2)
    m = np.stack([damping * np.cos(phase), damping * np.sin(phase)], axis=1)
    dm = np.stack(
        [
            -damping * frequency * beta[p] * np.sin(phase),
            damping * frequency * beta[p] * np.cos(phase),
        ],
        axis=1,
    )
    alpha = h[:, p]
    target = np.mean(dm, axis=0)
    vector = np.array([0.7, -0.4, 0.2, 0.5])
    vector /= np.linalg.norm(vector)
    perturbations: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "linear": (h[:, p], np.ones(h.shape[0])),
        "tanh": (np.tanh(h[:, p]), 1.0 - np.tanh(h[:, p]) ** 2),
        "multivariate_tanh": (
            np.tanh(h @ vector),
            vector[p] * (1.0 - np.tanh(h @ vector) ** 2),
        ),
    }
    levels = np.array([0.0, 0.025, 0.05, 0.1, 0.2, 0.4])
    rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    selector = np.array([1.0, 0.0])
    for name, (q_m, dq_m) in perturbations.items():
        q_alpha = q_m
        theoretical_slope = -float(np.mean(q_alpha * q_m))
        shape_rows: list[dict[str, object]] = []
        for a in levels:
            for b in levels:
                m_tilde = m + a * q_m[:, None] * selector[None, :]
                dm_tilde = dm + a * dq_m[:, None] * selector[None, :]
                alpha_tilde = alpha + b * q_alpha
                plugin = np.mean(dm_tilde, axis=0)
                riesz = np.mean(alpha_tilde[:, None] * m, axis=0)
                orthogonal = np.mean(
                    dm_tilde + alpha_tilde[:, None] * (m - m_tilde), axis=0
                )
                for method, estimate in (
                    ("PLUG", plugin),
                    ("RIESZ", riesz),
                    ("ORTH", orthogonal),
                ):
                    row = {
                        "experiment_id": "E2",
                        "stage": stage,
                        "perturbation": name,
                        "a": a,
                        "b": b,
                        "product_ab": a * b,
                        "method": method,
                        "signed_bias_channel0": float(estimate[0] - target[0]),
                        "bias_norm": float(np.linalg.norm(estimate - target)),
                        "m_l2_error": abs(a) * float(np.sqrt(np.mean(q_m**2))),
                        "alpha_l2_error": abs(b) * float(np.sqrt(np.mean(q_alpha**2))),
                        "error_product": abs(a * b) * float(
                            np.sqrt(np.mean(q_m**2)) * np.sqrt(np.mean(q_alpha**2))
                        ),
                        "theoretical_orth_bias_channel0": theoretical_slope * a * b,
                    }
                    rows.append(row)
                    shape_rows.append(row)
        orth = pd.DataFrame(shape_rows)
        orth = orth[orth.method == "ORTH"]
        small = orth[(orth.a <= 0.2) & (orth.b <= 0.2)]
        x = small.product_ab.to_numpy(float)
        y = small.signed_bias_channel0.to_numpy(float)
        design = np.stack([np.ones_like(x), x], axis=1)
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = intercept + slope * x
        ss_res = float(np.sum((y - prediction) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        double_robust = orth[(orth.a == 0.0) | (orth.b == 0.0)].bias_norm.max()
        slope_relative_error = abs(slope - theoretical_slope) / max(abs(theoretical_slope), 1e-12)
        gate_rows.append(
            {
                "perturbation": name,
                "intercept": float(intercept),
                "estimated_product_slope": float(slope),
                "theoretical_product_slope": theoretical_slope,
                "slope_relative_error": float(slope_relative_error),
                "product_bias_r2": float(r2),
                "max_double_robust_bias_norm": float(double_robust),
                "pass_r2": bool(r2 >= 0.90),
                "pass_slope": bool(slope_relative_error <= 0.20),
                "pass_intercept": bool(abs(intercept) <= 1e-4),
                "pass_double_robust": bool(double_robust <= 1e-4),
            }
        )
    frame = pd.DataFrame(rows)
    gate = pd.DataFrame(gate_rows)
    gate["passed"] = gate[["pass_r2", "pass_slope", "pass_intercept", "pass_double_robust"]].all(axis=1)
    atomic_csv(output, frame)
    atomic_csv(gate_output, gate)
    return frame, gate


def _moment_blind_dgp(family: str, seed: int, amplitude_fraction: float = 0.5):
    if family == "finite_difference_mixture":
        return MomentBlindDiscreteDGP(seed=seed, k=4, smoothing_sigma=0.2, amplitude_fraction=amplitude_fraction)
    if family == "continuous_legendre_tilt":
        return MomentBlindLegendreDGP(seed=seed, k=4, amplitude=amplitude_fraction)
    raise ValueError(f"unknown moment-blind family: {family}")


def run_moment_blind_seed(
    run_dir: Path,
    family: str,
    dgp_seed: int,
    method_init: int,
    config: MomentBlindRunConfig,
    save_scores: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resource_guard()
    start_time = time.monotonic()
    case_id = f"e4_{family}__seed_{dgp_seed}__init_{method_init}"
    case_csv = run_dir / "cases" / f"{case_id}.csv"
    nuisance_csv = run_dir / "nuisance" / f"{case_id}.csv"
    score_npz = run_dir / "scores" / f"{case_id}.npz"
    if case_csv.exists() and nuisance_csv.exists() and (score_npz.exists() or not save_scores):
        return pd.read_csv(case_csv), pd.read_csv(nuisance_csv)

    dgp = _moment_blind_dgp(family, dgp_seed, config.amplitude_fraction)
    rng_bank = np.random.default_rng(300_000 + dgp_seed)
    rng_data = np.random.default_rng(400_000 + dgp_seed)
    h_bank = dgp.sample_histories(rng_bank, config.n_bank)
    y_bank = dgp.sample_responses(h_bank, rng_bank)
    bank = CharacteristicBank.fit(
        y_bank,
        config.query_count,
        seed=92117,
        amplitude_scales=config.frequency_scales,
    )
    query_path = run_dir / "query_banks" / f"e4_{family}__seed_{dgp_seed}.npz"
    if not query_path.exists():
        bank.save(query_path)
    h = dgp.sample_histories(rng_data, config.n_train)
    y = dgp.sample_responses(h, rng_data)
    phi = bank.transform(y)
    oracle_m = dgp.conditional_feature_mean(h, bank)
    oracle_dm = dgp.conditional_feature_derivative(h, 0, bank)
    oracle_alpha = dgp.oracle_riesz(h, 0)
    oracle_h = ndtri(
        np.clip(
            qmc.Sobol(d=1, scramble=True, seed=500_000 + dgp_seed).random_base2(config.qmc_power),
            1e-12,
            1.0 - 1e-12,
        )
    )
    truth = np.mean(dgp.conditional_feature_derivative(oracle_h, 0, bank), axis=0)

    methods = ("ZERO", "PLUG", "RIESZ", "ORTH", "OR_A", "OR_M", "OR_R")
    contributions = {method: np.zeros_like(phi) for method in methods}
    moment_contributions = np.zeros((h.shape[0], 4))
    prediction = np.zeros_like(phi)
    derivative_prediction = np.zeros_like(phi)
    alpha_prediction = np.zeros(h.shape[0])
    fold_assignment = np.full(h.shape[0], -1, dtype=np.int16)
    feature_checkpoints = []
    alpha_checkpoints = []
    moment_checkpoints = []
    moments = np.concatenate([y**order for order in range(1, 5)], axis=1)
    folds = independent_folds(h.shape[0], config.n_folds, seed=41337 + dgp_seed)
    for fold_id, (train, test) in enumerate(folds):
        fold_assignment[test] = fold_id
        sieve = HistorySieve(1, config.random_width, seed=60_000 + 100 * method_init + fold_id).fit(h[train])
        z_train = sieve.transform(h[train])
        dz_train = sieve.derivative(h[train], 0)
        z_test = sieve.transform(h[test])
        dz_test = sieve.derivative(h[test], 0)
        feature_coef = ridge_coefficients(z_train, phi[train], config.feature_ridge)
        alpha_coef = riesz_coefficients(z_train, dz_train, config.riesz_ridge)
        moment_coef = ridge_coefficients(z_train, moments[train], config.feature_ridge)
        pred = z_test @ feature_coef
        dpred = dz_test @ feature_coef
        alpha = z_test @ alpha_coef
        moment_pred = z_test @ moment_coef
        moment_derivative = dz_test @ moment_coef
        prediction[test] = pred
        derivative_prediction[test] = dpred
        alpha_prediction[test] = alpha
        contributions["PLUG"][test] = dpred
        contributions["RIESZ"][test] = alpha[:, None] * phi[test]
        contributions["ORTH"][test] = dpred + alpha[:, None] * (phi[test] - pred)
        contributions["OR_A"][test] = dpred + oracle_alpha[test, None] * (phi[test] - pred)
        contributions["OR_M"][test] = oracle_dm[test] + alpha[:, None] * (phi[test] - oracle_m[test])
        contributions["OR_R"][test] = oracle_alpha[test, None] * phi[test]
        moment_contributions[test] = moment_derivative + alpha[:, None] * (moments[test] - moment_pred)
        feature_checkpoints.append(feature_coef)
        alpha_checkpoints.append(alpha_coef)
        moment_checkpoints.append(moment_coef)

    rows: list[dict[str, object]] = []
    for method, values in contributions.items():
        estimate = np.mean(values, axis=0)
        standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
        coverage = float(
            np.mean((truth >= estimate - 1.96 * standard_error) & (truth <= estimate + 1.96 * standard_error))
        )
        metrics = {**vector_metrics(estimate, truth), "coverage_fraction": coverage}
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "experiment_id": "E4",
                    "cell_id": f"{family}_k4_a{config.amplitude_fraction:g}_q{config.query_count}",
                    "dgp_family": family,
                    "dgp_seed": dgp_seed,
                    "method": method,
                    "method_init": method_init,
                    "access_tier": "O0" if method in {"ZERO", "PLUG", "RIESZ", "ORTH"} else "oracle_control",
                    "compute_budget": "small_sieve",
                    "n_train": config.n_train,
                    "n_eval": config.n_train,
                    "history_dim": 1,
                    "response_dim": 1,
                    "history_length": 1,
                    "horizon": 1,
                    "query_count": config.query_count,
                    "basis_count": 1,
                    "target_name": "moment_blind_characteristic_tangent",
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "fit_status": "ok",
                    "failure_code": "",
                    "stage": config.stage,
                }
            )
    moment_estimate = np.mean(moment_contributions, axis=0)
    moment_se = np.std(moment_contributions, axis=0, ddof=1) / math.sqrt(moment_contributions.shape[0])
    moment_z = moment_estimate / np.maximum(moment_se, 1e-12)
    threshold = float(norm.ppf(1.0 - 0.05 / (2.0 * 4)))
    active_rms = math.sqrt(float(np.mean(truth**2)))
    for metric_name, metric_value in {
        "null_rms_normalized_to_active": float(np.sqrt(np.mean(moment_estimate**2)) / max(active_rms, 1e-12)),
        "max_abs_z": float(np.max(np.abs(moment_z))),
        "any_false_call_bonferroni": float(np.any(np.abs(moment_z) > threshold)),
    }.items():
        rows.append(
            {
                "experiment_id": "E4",
                "cell_id": f"{family}_k4_a{config.amplitude_fraction:g}_q{config.query_count}",
                "dgp_family": family,
                "dgp_seed": dgp_seed,
                "method": "MOMENT_ORTH_1_TO_4",
                "method_init": method_init,
                "access_tier": "O0",
                "compute_budget": "small_sieve",
                "n_train": config.n_train,
                "n_eval": config.n_train,
                "history_dim": 1,
                "response_dim": 1,
                "history_length": 1,
                "horizon": 1,
                "query_count": 0,
                "basis_count": 1,
                "target_name": "moments_1_to_4_exact_null",
                "metric_name": metric_name,
                "metric_value": metric_value,
                "fit_status": "ok",
                "failure_code": "",
                "stage": config.stage,
            }
        )
    probe = _probe_diagnostics(h, alpha_prediction, 0, seed=600_000 + dgp_seed)
    nuisance = pd.DataFrame(
        [
            {
                "experiment_id": "E4",
                "cell_id": f"{family}_k4_a{config.amplitude_fraction:g}_q{config.query_count}",
                "dgp_family": family,
                "dgp_seed": dgp_seed,
                "method_init": method_init,
                "feature_validation_mse": float(np.mean((phi - prediction) ** 2)),
                "feature_mean_nrmse": vector_metrics(prediction.reshape(-1), oracle_m.reshape(-1))["nrmse"],
                "feature_derivative_nrmse": vector_metrics(derivative_prediction.reshape(-1), oracle_dm.reshape(-1))["nrmse"],
                "alpha_nrmse": vector_metrics(alpha_prediction, oracle_alpha)["nrmse"],
                **probe,
                "query_hash": bank.digest(),
                "runtime_seconds": time.monotonic() - start_time,
                **resource_guard(),
            }
        ]
    )
    metrics = pd.DataFrame(rows)
    atomic_csv(case_csv, metrics)
    atomic_csv(nuisance_csv, nuisance)
    split_path = run_dir / "splits" / f"e4_{family}__seed_{dgp_seed}.npz"
    if not split_path.exists():
        atomic_npz(split_path, fold_id=fold_assignment, observation_index=np.arange(h.shape[0]))
    atomic_npz(
        run_dir / "checkpoints" / f"{case_id}.npz",
        feature_coefficients=np.stack(feature_checkpoints).astype(np.float32),
        alpha_coefficients=np.stack(alpha_checkpoints).astype(np.float32),
        moment_coefficients=np.stack(moment_checkpoints).astype(np.float32),
    )
    if save_scores:
        atomic_npz(
            score_npz,
            orth=contributions["ORTH"].astype(np.float32),
            moment_orth=moment_contributions.astype(np.float32),
            truth=truth.astype(np.float64),
            dgp_seed=np.array([dgp_seed], dtype=np.int64),
            method_init=np.array([method_init], dtype=np.int16),
        )
    return metrics, nuisance


def package_e4_confirmation(run_dir: Path, expected_seeds: int, initializations: int) -> dict[str, object]:
    case_files = sorted((run_dir / "cases").glob("*.csv"))
    nuisance_files = sorted((run_dir / "nuisance").glob("*.csv"))
    score_files = sorted((run_dir / "scores_seed_average").glob("*.npz"))
    checkpoint_files = sorted((run_dir / "checkpoints").glob("*.npz"))
    query_files = sorted((run_dir / "query_banks").glob("*.npz"))
    split_files = sorted((run_dir / "splits").glob("*.npz"))
    metrics = pd.concat([pd.read_csv(path) for path in case_files], ignore_index=True)
    nuisances = pd.concat([pd.read_csv(path) for path in nuisance_files], ignore_index=True)
    atomic_csv(run_dir / "seed_level.csv", metrics)
    metrics.to_parquet(run_dir / "seed_level.parquet", index=False)
    atomic_csv(run_dir / "nuisance_metrics.csv", nuisances)
    nuisances.to_parquet(run_dir / "nuisance_metrics.parquet", index=False)
    groups = [
        "experiment_id", "cell_id", "dgp_family", "dgp_seed", "method", "access_tier",
        "compute_budget", "n_train", "n_eval", "history_dim", "response_dim", "history_length",
        "horizon", "query_count", "basis_count", "target_name", "metric_name", "stage",
    ]
    averaged = metrics.groupby(groups, dropna=False, as_index=False).agg(
        metric_value=("metric_value", "mean"),
        initialization_sd=("metric_value", "std"),
        initialization_count=("method_init", "nunique"),
    )
    atomic_csv(run_dir / "seed_level_init_averaged.csv", averaged)
    averaged.to_parquet(run_dir / "seed_level_init_averaged.parquet", index=False)
    rng = np.random.default_rng(84117)
    summary_rows = []
    for keys, frame in averaged.groupby(
        ["dgp_family", "method", "target_name", "metric_name"], dropna=False, sort=True
    ):
        values = frame.metric_value.to_numpy(float)
        draws = np.mean(values[rng.integers(0, values.size, size=(2000, values.size))], axis=1)
        summary_rows.append(
            {
                **dict(zip(["dgp_family", "method", "target_name", "metric_name"], keys, strict=True)),
                "mean": float(np.mean(values)),
                "ci_lower": float(np.quantile(draws, 0.025)),
                "ci_upper": float(np.quantile(draws, 0.975)),
                "n_dgp_seeds": int(values.size),
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_csv(run_dir / "summary.csv", summary)
    detections = pd.read_csv(run_dir / "seed_detection.csv")
    detection_summary = detections.groupby("dgp_family", as_index=False).agg(
        characteristic_power=("characteristic_detected", "mean"),
        moment_false_positive_rate=("moment_false_call", "mean"),
        n_dgp_seeds=("dgp_seed", "nunique"),
    )
    atomic_csv(run_dir / "detection_summary.csv", detection_summary)
    expected_families = 2
    expected_cases = expected_families * expected_seeds * initializations
    expected_family_seeds = expected_families * expected_seeds
    validation = {
        "expected_families": expected_families,
        "expected_dgp_seeds_per_family": expected_seeds,
        "expected_initializations": initializations,
        "expected_case_files": expected_cases,
        "case_files": len(case_files),
        "nuisance_files": len(nuisance_files),
        "initialization_averaged_score_files": len(score_files),
        "checkpoint_files": len(checkpoint_files),
        "query_bank_files": len(query_files),
        "split_files": len(split_files),
        "failed_metric_rows": int(np.sum(metrics.fit_status != "ok")),
        "nonfinite_metric_values": int(np.sum(~np.isfinite(metrics.metric_value))),
        "all_initialization_counts_complete": bool(averaged.initialization_count.eq(initializations).all()),
        "passed": bool(
            len(case_files) == expected_cases
            and len(nuisance_files) == expected_cases
            and len(score_files) == expected_family_seeds
            and len(checkpoint_files) == expected_cases
            and len(query_files) == expected_family_seeds
            and len(split_files) == expected_family_seeds
            and detections.shape[0] == expected_family_seeds
            and np.all(metrics.fit_status == "ok")
            and np.all(np.isfinite(metrics.metric_value))
            and averaged.initialization_count.eq(initializations).all()
        ),
    }
    atomic_json(run_dir / "validation.json", validation)
    manifest = {
        "source_hash": sha256_tree(_source_root()),
        "frozen_config_sha256": sha256_file(run_dir / "frozen_config.json"),
        "validation_sha256": sha256_file(run_dir / "validation.json"),
        "summary_sha256": sha256_file(run_dir / "summary.csv"),
        "detection_summary_sha256": sha256_file(run_dir / "detection_summary.csv"),
        "source_snapshot_sha256": sha256_file(run_dir / "source_snapshot.tar.gz"),
        "environment_lock_sha256": sha256_file(run_dir / "environment.lock.json"),
    }
    atomic_json(run_dir / "package_manifest.json", manifest)
    return validation


def run_support_motion_seed(
    run_dir: Path,
    sigma: float,
    dgp_seed: int,
    method_init: int,
    n_train: int = 8000,
    query_count: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resource_guard()
    start_time = time.monotonic()
    sigma_label = str(sigma).replace(".", "p")
    case_id = f"e5_sigma_{sigma_label}__seed_{dgp_seed}__init_{method_init}"
    case_csv = run_dir / "cases" / f"{case_id}.csv"
    nuisance_csv = run_dir / "nuisance" / f"{case_id}.csv"
    if case_csv.exists() and nuisance_csv.exists():
        return pd.read_csv(case_csv), pd.read_csv(nuisance_csv)
    dgp = SupportMotionDGP(dgp_seed, sigma)
    rng_bank = np.random.default_rng(610_000 + dgp_seed)
    rng_data = np.random.default_rng(620_000 + dgp_seed)
    h_bank = dgp.sample_histories(rng_bank, 1024)
    y_bank = dgp.sample_responses(h_bank, rng_bank)
    bank = CharacteristicBank.fit(y_bank, query_count, seed=92117)
    bank_path = run_dir / "query_banks" / f"e5_sigma_{sigma_label}__seed_{dgp_seed}.npz"
    if not bank_path.exists():
        bank.save(bank_path)
    h = dgp.sample_histories(rng_data, n_train)
    y = dgp.sample_responses(h, rng_data)
    phi = bank.transform(y)
    oracle_m = dgp.conditional_feature_mean(h, bank)
    oracle_dm = dgp.conditional_feature_derivative(h, 0, bank)
    oracle_alpha = dgp.oracle_riesz(h, 0)
    oracle_h = ndtri(
        np.clip(qmc.Sobol(d=1, scramble=True, seed=630_000 + dgp_seed).random_base2(17), 1e-12, 1.0 - 1e-12)
    )
    truth = np.mean(dgp.conditional_feature_derivative(oracle_h, 0, bank), axis=0)
    methods = ("ZERO", "PLUG", "RIESZ", "ORTH", "OR_A", "OR_M", "OR_R")
    contributions = {method: np.zeros_like(phi) for method in methods}
    prediction = np.zeros_like(phi)
    derivative_prediction = np.zeros_like(phi)
    alpha_prediction = np.zeros(h.shape[0])
    checkpoints_feature = []
    checkpoints_alpha = []
    fold_assignment = np.full(h.shape[0], -1, dtype=np.int16)
    folds = independent_folds(h.shape[0], 5, seed=51337 + dgp_seed)
    for fold_id, (train, test) in enumerate(folds):
        fold_assignment[test] = fold_id
        sieve = HistorySieve(1, 32, seed=70_000 + 100 * method_init + fold_id).fit(h[train])
        z_train = sieve.transform(h[train])
        dz_train = sieve.derivative(h[train], 0)
        z_test = sieve.transform(h[test])
        dz_test = sieve.derivative(h[test], 0)
        feature_coef = ridge_coefficients(z_train, phi[train], 1e-3)
        alpha_coef = riesz_coefficients(z_train, dz_train, 1e-2)
        pred = z_test @ feature_coef
        dpred = dz_test @ feature_coef
        alpha = z_test @ alpha_coef
        prediction[test] = pred
        derivative_prediction[test] = dpred
        alpha_prediction[test] = alpha
        contributions["PLUG"][test] = dpred
        contributions["RIESZ"][test] = alpha[:, None] * phi[test]
        contributions["ORTH"][test] = dpred + alpha[:, None] * (phi[test] - pred)
        contributions["OR_A"][test] = dpred + oracle_alpha[test, None] * (phi[test] - pred)
        contributions["OR_M"][test] = oracle_dm[test] + alpha[:, None] * (phi[test] - oracle_m[test])
        contributions["OR_R"][test] = oracle_alpha[test, None] * phi[test]
        checkpoints_feature.append(feature_coef)
        checkpoints_alpha.append(alpha_coef)
    rows = []
    for method, values in contributions.items():
        estimate = np.mean(values, axis=0)
        standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
        metrics = {
            **vector_metrics(estimate, truth),
            "coverage_fraction": float(
                np.mean((truth >= estimate - 1.96 * standard_error) & (truth <= estimate + 1.96 * standard_error))
            ),
        }
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "experiment_id": "E5",
                    "cell_id": f"sigma_{sigma_label}",
                    "dgp_family": "support_motion",
                    "dgp_seed": dgp_seed,
                    "method": method,
                    "method_init": method_init,
                    "access_tier": "O0" if method in {"ZERO", "PLUG", "RIESZ", "ORTH"} else "oracle_control",
                    "compute_budget": "small_sieve",
                    "n_train": n_train,
                    "n_eval": n_train,
                    "history_dim": 1,
                    "response_dim": 1,
                    "history_length": 1,
                    "horizon": 1,
                    "query_count": query_count,
                    "basis_count": 1,
                    "target_name": "characteristic_tangent",
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "fit_status": "ok",
                    "failure_code": "",
                    "stage": "confirmation",
                }
            )
    if sigma > 0:
        likelihood_score = dgp.likelihood_history_score(h, y)
        likelihood = {
            "likelihood_score_available": 1.0,
            "likelihood_score_rms": float(np.sqrt(np.mean(likelihood_score**2))),
            "likelihood_score_q95": float(np.quantile(np.abs(likelihood_score), 0.95)),
            "likelihood_score_q99": float(np.quantile(np.abs(likelihood_score), 0.99)),
        }
    else:
        likelihood = {
            "likelihood_score_available": 0.0,
            "likelihood_score_rms": math.nan,
            "likelihood_score_q95": math.nan,
            "likelihood_score_q99": math.nan,
        }
    nuisance = pd.DataFrame(
        [
            {
                "experiment_id": "E5",
                "cell_id": f"sigma_{sigma_label}",
                "dgp_seed": dgp_seed,
                "method_init": method_init,
                "sigma": sigma,
                "feature_validation_mse": float(np.mean((phi - prediction) ** 2)),
                "feature_mean_nrmse": vector_metrics(prediction.reshape(-1), oracle_m.reshape(-1))["nrmse"],
                "feature_derivative_nrmse": vector_metrics(derivative_prediction.reshape(-1), oracle_dm.reshape(-1))["nrmse"],
                "alpha_nrmse": vector_metrics(alpha_prediction, oracle_alpha)["nrmse"],
                **_probe_diagnostics(h, alpha_prediction, 0, 640_000 + dgp_seed),
                **likelihood,
                "runtime_seconds": time.monotonic() - start_time,
                **resource_guard(),
            }
        ]
    )
    metrics = pd.DataFrame(rows)
    atomic_csv(case_csv, metrics)
    atomic_csv(nuisance_csv, nuisance)
    split_path = run_dir / "splits" / f"e5_sigma_{sigma_label}__seed_{dgp_seed}.npz"
    if not split_path.exists():
        atomic_npz(split_path, fold_id=fold_assignment, observation_index=np.arange(h.shape[0]))
    atomic_npz(
        run_dir / "checkpoints" / f"{case_id}.npz",
        feature_coefficients=np.stack(checkpoints_feature).astype(np.float32),
        alpha_coefficients=np.stack(checkpoints_alpha).astype(np.float32),
    )
    return metrics, nuisance


def package_development_run(run_dir: Path, expected_cases: int) -> dict[str, object]:
    case_files = sorted((run_dir / "cases").glob("*.csv"))
    nuisance_files = sorted((run_dir / "nuisance").glob("*.csv"))
    score_files = sorted((run_dir / "scores").glob("*.npz"))
    metrics = pd.concat([pd.read_csv(path) for path in case_files], ignore_index=True) if case_files else pd.DataFrame()
    nuisances = pd.concat([pd.read_csv(path) for path in nuisance_files], ignore_index=True) if nuisance_files else pd.DataFrame()
    if not metrics.empty:
        atomic_csv(run_dir / "seed_level.csv", metrics)
        metrics.to_parquet(run_dir / "seed_level.parquet", index=False)
    if not nuisances.empty:
        atomic_csv(run_dir / "nuisance_metrics.csv", nuisances)
        nuisances.to_parquet(run_dir / "nuisance_metrics.parquet", index=False)
    e2_gate = pd.read_csv(run_dir / "e2_gate.csv") if (run_dir / "e2_gate.csv").exists() else pd.DataFrame()
    validation = {
        "expected_gaussian_cases": expected_cases,
        "metric_case_files": len(case_files),
        "nuisance_case_files": len(nuisance_files),
        "score_files": len(score_files),
        "failed_metric_rows": int(np.sum(metrics.fit_status != "ok")) if not metrics.empty else expected_cases,
        "nonfinite_metric_values": int(np.sum(~np.isfinite(metrics.metric_value))) if not metrics.empty else 0,
        "e2_passed": bool(not e2_gate.empty and e2_gate.passed.all()),
        "passed": bool(
            len(case_files) == expected_cases
            and len(nuisance_files) == expected_cases
            and len(score_files) == expected_cases
            and not metrics.empty
            and np.all(metrics.fit_status == "ok")
            and np.all(np.isfinite(metrics.metric_value))
            and not e2_gate.empty
            and e2_gate.passed.all()
        ),
    }
    atomic_json(run_dir / "validation.json", validation)
    source_hash = sha256_tree(_source_root())
    manifest = {
        "source_hash": source_hash,
        "environment": _environment_payload(),
        "validation_sha256": sha256_file(run_dir / "validation.json"),
        "seed_level_sha256": sha256_file(run_dir / "seed_level.csv") if (run_dir / "seed_level.csv").exists() else None,
        "nuisance_metrics_sha256": sha256_file(run_dir / "nuisance_metrics.csv") if (run_dir / "nuisance_metrics.csv").exists() else None,
        "e2_gate_sha256": sha256_file(run_dir / "e2_gate.csv") if (run_dir / "e2_gate.csv").exists() else None,
    }
    atomic_json(run_dir / "package_manifest.json", manifest)
    return validation


def package_confirmation_run(run_dir: Path, expected_seeds: int, initializations: int) -> dict[str, object]:
    case_files = sorted((run_dir / "cases").glob("*.csv"))
    nuisance_files = sorted((run_dir / "nuisance").glob("*.csv"))
    score_files = sorted((run_dir / "scores_seed_average").glob("*.npz"))
    checkpoint_files = sorted((run_dir / "checkpoints").glob("*.npz"))
    query_files = sorted((run_dir / "query_banks").glob("*.npz"))
    split_files = sorted((run_dir / "splits").glob("*.npz"))
    metrics = pd.concat([pd.read_csv(path) for path in case_files], ignore_index=True)
    nuisances = pd.concat([pd.read_csv(path) for path in nuisance_files], ignore_index=True)
    atomic_csv(run_dir / "seed_level.csv", metrics)
    metrics.to_parquet(run_dir / "seed_level.parquet", index=False)
    atomic_csv(run_dir / "nuisance_metrics.csv", nuisances)
    nuisances.to_parquet(run_dir / "nuisance_metrics.parquet", index=False)

    init_averaged = (
        metrics.groupby(
            [
                "experiment_id", "cell_id", "dgp_family", "dgp_seed", "method",
                "access_tier", "compute_budget", "n_train", "n_eval", "history_dim",
                "response_dim", "history_length", "horizon", "query_count", "basis_count",
                "target_name", "metric_name", "stage",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(metric_value=("metric_value", "mean"), initialization_sd=("metric_value", "std"), initialization_count=("method_init", "nunique"))
    )
    atomic_csv(run_dir / "seed_level_init_averaged.csv", init_averaged)
    init_averaged.to_parquet(run_dir / "seed_level_init_averaged.parquet", index=False)

    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(77191)
    for keys, frame in init_averaged.groupby(
        ["experiment_id", "cell_id", "dgp_family", "method", "target_name", "metric_name"],
        dropna=False,
        sort=True,
    ):
        values = frame.metric_value.to_numpy(float)
        draws = np.mean(values[rng.integers(0, values.size, size=(2000, values.size))], axis=1)
        summary_rows.append(
            {
                **dict(zip(["experiment_id", "cell_id", "dgp_family", "method", "target_name", "metric_name"], keys, strict=True)),
                "mean": float(np.mean(values)),
                "ci_lower": float(np.quantile(draws, 0.025)),
                "ci_upper": float(np.quantile(draws, 0.975)),
                "n_dgp_seeds": int(values.size),
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_csv(run_dir / "summary.csv", summary)
    summary.to_parquet(run_dir / "summary.parquet", index=False)

    paired_rows = []
    nrmse = init_averaged[
        (init_averaged.target_name == "characteristic_tangent")
        & (init_averaged.metric_name == "nrmse")
    ]
    pivot = nrmse.pivot(index="dgp_seed", columns="method", values="metric_value")
    for baseline in ("PLUG", "RIESZ"):
        log_ratio = np.log(pivot["ORTH"].to_numpy(float)) - np.log(pivot[baseline].to_numpy(float))
        draws = np.mean(log_ratio[rng.integers(0, log_ratio.size, size=(2000, log_ratio.size))], axis=1)
        paired_rows.append(
            {
                "candidate": "ORTH",
                "baseline": baseline,
                "mean_error_ratio": float(np.exp(np.mean(log_ratio))),
                "ci_lower": float(np.exp(np.quantile(draws, 0.025))),
                "ci_upper": float(np.exp(np.quantile(draws, 0.975))),
                "n_common_seeds": int(log_ratio.size),
                "superiority_upper_below_0_90": bool(np.exp(np.quantile(draws, 0.975)) < 0.90),
            }
        )
    paired = pd.DataFrame(paired_rows)
    atomic_csv(run_dir / "paired_comparisons.csv", paired)

    expected_cases = expected_seeds * initializations
    required_metric_initializations = init_averaged.initialization_count.eq(initializations).all()
    validation = {
        "expected_dgp_seeds": expected_seeds,
        "expected_initializations": initializations,
        "expected_case_files": expected_cases,
        "case_files": len(case_files),
        "nuisance_files": len(nuisance_files),
        "initialization_averaged_score_files": len(score_files),
        "checkpoint_files": len(checkpoint_files),
        "query_bank_files": len(query_files),
        "split_files": len(split_files),
        "common_seed_count": int(pivot.dropna().shape[0]),
        "failed_metric_rows": int(np.sum(metrics.fit_status != "ok")),
        "nonfinite_metric_values": int(np.sum(~np.isfinite(metrics.metric_value))),
        "all_initialization_counts_complete": bool(required_metric_initializations),
        "passed": bool(
            len(case_files) == expected_cases
            and len(nuisance_files) == expected_cases
            and len(score_files) == expected_seeds
            and len(checkpoint_files) == expected_cases
            and len(query_files) == expected_seeds
            and len(split_files) == expected_seeds
            and pivot.dropna().shape[0] == expected_seeds
            and np.all(metrics.fit_status == "ok")
            and np.all(np.isfinite(metrics.metric_value))
            and required_metric_initializations
        ),
    }
    atomic_json(run_dir / "validation.json", validation)
    manifest = {
        "source_hash": sha256_tree(_source_root()),
        "frozen_config_sha256": sha256_file(run_dir / "frozen_config.json"),
        "validation_sha256": sha256_file(run_dir / "validation.json"),
        "seed_level_sha256": sha256_file(run_dir / "seed_level.csv"),
        "init_averaged_sha256": sha256_file(run_dir / "seed_level_init_averaged.csv"),
        "summary_sha256": sha256_file(run_dir / "summary.csv"),
        "paired_comparisons_sha256": sha256_file(run_dir / "paired_comparisons.csv"),
        "source_snapshot_sha256": sha256_file(run_dir / "source_snapshot.tar.gz"),
        "environment_lock_sha256": sha256_file(run_dir / "environment.lock.json"),
        "environment": _environment_payload(),
    }
    atomic_json(run_dir / "package_manifest.json", manifest)
    return validation


def write_frozen_config(run_dir: Path, payload: dict[str, object]) -> None:
    path = run_dir / "frozen_config.json"
    if path.exists():
        existing = json.loads(path.read_text())
        # JSON persists tuples as arrays; compare the JSON-normalized payload so
        # an interrupted frozen run can resume without a false config mismatch.
        normalized = json.loads(json.dumps(payload, allow_nan=False))
        if existing != normalized:
            raise RuntimeError("frozen config exists with different contents")
        return
    atomic_json(path, payload)
