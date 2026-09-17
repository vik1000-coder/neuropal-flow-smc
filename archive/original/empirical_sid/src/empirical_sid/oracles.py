from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .dgps import (
    DynamicStochasticGainDGP,
    GaussianCovarianceDGP,
    GaussianMeanDGP,
    SmoothTiltDGP,
    TILT_DEFINITIONS,
    trap_weights,
)


@dataclass(frozen=True)
class OracleCheck:
    check_id: str
    mechanism: str
    value: float
    tolerance: float
    passed: bool
    detail: str


def _check(check_id: str, mechanism: str, value: float, tolerance: float, detail: str) -> OracleCheck:
    return OracleCheck(check_id, mechanism, float(value), float(tolerance), bool(value <= tolerance), detail)


def smooth_tilt_checks(mechanism: str) -> list[OracleCheck]:
    dgp = SmoothTiltDGP(mechanism)
    checks: list[OracleCheck] = []
    normalization = abs(float(np.sum(dgp.weights * dgp.density_grid(0.63))) - 1.0)
    checks.append(_check("density_normalization", mechanism, normalization, 1e-10, "grid quadrature"))
    minimum_factor = float(np.min(1.0 + dgp.t(1.2) * dgp.psi))
    checks.append(
        OracleCheck(
            "density_positivity",
            mechanism,
            minimum_factor,
            0.0,
            bool(minimum_factor > 0.0),
            "minimum signed-tilt factor",
        )
    )
    maximum_orthogonality = max(abs(value) for value in dgp.orthogonality.values())
    checks.append(
        _check(
            "nuisance_orthogonality",
            mechanism,
            maximum_orthogonality,
            2e-10,
            f"orders 0..{dgp.definition.nuisance_degree}",
        )
    )
    histories = np.linspace(-0.8, 0.8, 17)
    step = 1e-5
    numerical = (dgp.target(histories + step) - dgp.target(histories - step)) / (2.0 * step)
    local_error = float(np.max(np.abs(numerical - dgp.local_effect(histories))))
    checks.append(_check("local_derivative", mechanism, local_error, 2e-7, "central difference"))

    identity_errors = []
    for history in histories[::2]:
        density = dgp.density_grid(float(history))
        influence = dgp.influence(dgp.grid, np.full(dgp.grid.size, history))
        tangent = dgp.history_tangent(dgp.grid, np.full(dgp.grid.size, history))
        recovered = float(np.sum(dgp.weights * density * influence * tangent))
        identity_errors.append(abs(recovered - float(dgp.local_effect(history))))
    checks.append(
        _check(
            "influence_function_identity",
            mechanism,
            max(identity_errors),
            2e-7,
            "independent grid integration",
        )
    )

    finite_errors = []
    delta = 0.12
    for history in histories[::2]:
        plus = dgp.density_grid(float(history + delta))
        minus = dgp.density_grid(float(history - delta))
        mixture = 0.5 * (plus + minus)
        witness = (plus - minus) / np.maximum(delta * (plus + minus), 1e-15)
        if dgp.channel in {"tail_probability", "mode_occupancy"}:
            feature = dgp._declared_feature(dgp.grid)
            recovered = float(np.sum(dgp.weights * mixture * feature * witness))
        else:
            raw_mixture = []
            raw_difference = []
            for order in range(1, 5):
                feature = dgp.grid**order
                raw_mixture.append(float(np.sum(dgp.weights * mixture * feature)))
                raw_difference.append(float(np.sum(dgp.weights * mixture * feature * witness)))
            raw_mixture = np.asarray(raw_mixture)
            raw_difference = np.asarray(raw_difference)
            plus_raw = raw_mixture + delta * raw_difference
            minus_raw = raw_mixture - delta * raw_difference
            from .dgps import raw_to_functional

            recovered = float(
                (raw_to_functional(plus_raw, dgp.channel) - raw_to_functional(minus_raw, dgp.channel))
                / (2.0 * delta)
            )
        finite_errors.append(abs(recovered - float(dgp.finite_effect(history, delta))))
    checks.append(
        _check("finite_witness_identity", mechanism, max(finite_errors), 3e-7, "oracle endpoints")
    )
    return checks


def gaussian_checks() -> list[OracleCheck]:
    checks: list[OracleCheck] = []
    mean = GaussianMeanDGP()
    histories = np.linspace(-0.8, 0.8, 31)
    step = 1e-5
    numerical = (mean.target(histories + step) - mean.target(histories - step)) / (2.0 * step)
    checks.append(
        _check("local_derivative", mean.mechanism, np.max(np.abs(numerical - mean.local_effect(histories))), 1e-9, "analytic mean")
    )

    covariance = GaussianCovarianceDGP()
    numerical_covariance = (
        covariance.target(histories + step) - covariance.target(histories - step)
    ) / (2.0 * step)
    checks.append(
        _check(
            "local_derivative",
            covariance.mechanism,
            np.max(np.abs(numerical_covariance - covariance.local_effect(histories))),
            1e-9,
            "analytic covariance",
        )
    )
    rng = np.random.default_rng(412)
    repeated_h = np.repeat(histories[::5], 200_000)
    y = covariance.sample(repeated_h, rng)
    tangent = covariance.history_tangent(y, repeated_h)
    influence = covariance.influence(y, repeated_h)
    recovered = []
    truth = []
    for history in histories[::5]:
        selected = repeated_h == history
        recovered.append(float(np.mean(tangent[selected] * influence[selected])))
        truth.append(float(covariance.local_effect(history)))
    relative = np.sqrt(np.mean((np.asarray(recovered) - truth) ** 2)) / np.sqrt(np.mean(np.asarray(truth) ** 2))
    checks.append(_check("influence_function_mc", covariance.mechanism, relative, 0.025, "1.4M draws"))
    return checks


def corruption_ledger_checks() -> list[OracleCheck]:
    checks: list[OracleCheck] = []
    for mechanism in ["m3_cubic", "m4_quartic"]:
        dgp = SmoothTiltDGP(mechanism)
        history = 0.31
        sigma = 0.12
        raw = dgp.raw_moments(history)
        draw = dgp.raw_moment_derivative(history)
        corrupted = raw.copy()
        corrupted[1] += sigma**2
        corrupted[2] += 3.0 * sigma**2 * raw[0]
        corrupted[3] += 6.0 * sigma**2 * raw[1] + 3.0 * sigma**4
        corrupted_draw = draw.copy()
        corrupted_draw[2] += 3.0 * sigma**2 * draw[0]
        corrupted_draw[3] += 6.0 * sigma**2 * draw[1]
        from .dgps import raw_functional_derivative

        clean = float(raw_functional_derivative(raw, draw, dgp.channel))
        noised = float(raw_functional_derivative(corrupted, corrupted_draw, dgp.channel))
        checks.append(
            _check(
                "gaussian_corruption_typed_derivative",
                mechanism,
                abs(clean - noised),
                2e-10,
                "history-independent Gaussian noise",
            )
        )
    return checks


def topology_hodge_checks(epsilon_ladder: list[float]) -> tuple[list[OracleCheck], list[dict[str, float]]]:
    grid = np.linspace(-4.0, 4.0, 1601)
    weights = trap_weights(grid)
    left = np.exp(-0.5 * ((grid + 2.0) / 0.42) ** 2)
    right = np.exp(-0.5 * ((grid - 2.0) / 0.42) ** 2)
    bridge = np.exp(-0.5 * (grid / 1.25) ** 2)
    signed = left - right
    checks: list[OracleCheck] = []
    rows: list[dict[str, float]] = []
    t = 0.25
    dt = 0.55
    for epsilon in epsilon_ladder:
        if epsilon > 0:
            base = left + right + epsilon * bridge
            base /= np.sum(weights * base)
            psi = signed / np.maximum(left + right + epsilon * bridge, 1e-14)
            psi -= np.sum(weights * base * psi)
            psi /= np.max(np.abs(psi))
            density = base * (1.0 + t * psi)
            density /= np.sum(weights * density)
            tangent = dt * psi / np.maximum(1.0 + t * psi, 1e-14)
            field = np.gradient(tangent, grid, edge_order=2)
            reconstructed = cumulative_trapezoid(field, grid, initial=0.0)
            reconstructed -= np.sum(weights * density * reconstructed)
            error = float(
                np.sqrt(np.sum(weights * density * (reconstructed - tangent) ** 2))
                / max(np.sqrt(np.sum(weights * density * tangent**2)), 1e-12)
            )
            bridge_mass = float(np.sum(weights[np.abs(grid) < 0.5] * density[np.abs(grid) < 0.5]))
        else:
            support = (np.abs(grid + 2.0) < 0.95) | (np.abs(grid - 2.0) < 0.95)
            base = np.where(support, left + right, 0.0)
            base /= np.sum(weights * base)
            component = np.where(grid < 0.0, 1.0, -1.0)
            component -= np.sum(weights * base * component)
            density = base * (1.0 + t * component)
            density /= np.sum(weights * density)
            tangent = dt * component / np.maximum(1.0 + t * component, 1e-14)
            field = np.zeros_like(grid)
            reconstructed = np.zeros_like(grid)
            for mask in [grid < -1.05, grid > 1.05]:
                if np.any(mask):
                    reconstructed[mask] = cumulative_trapezoid(field[mask], grid[mask], initial=0.0)
                    mass = np.sum(weights[mask] * density[mask])
                    if mass > 0:
                        reconstructed[mask] -= np.sum(weights[mask] * density[mask] * reconstructed[mask]) / mass
            error = float(
                np.sqrt(np.sum(weights * density * (reconstructed - tangent) ** 2))
                / max(np.sqrt(np.sum(weights * density * tangent**2)), 1e-12)
            )
            bridge_mass = 0.0
        rows.append(
            {
                "epsilon": float(epsilon),
                "bridge_mass_proxy": bridge_mass,
                "hodge_tangent_nrmse": error,
            }
        )
        if epsilon == 0:
            checks.append(
                OracleCheck(
                    "disconnected_weight_change_nonidentifiability",
                    "o4_topology",
                    error,
                    0.8,
                    bool(error >= 0.8),
                    "component constants are absent from the gradient field",
                )
            )
        elif epsilon >= 0.1:
            checks.append(
                _check(
                    "connected_hodge_reconstruction",
                    f"o4_epsilon_{epsilon:g}",
                    error,
                    0.05,
                    "one-dimensional oracle gradient integration",
                )
            )
    return checks, rows


def dynamic_formula_checks() -> list[OracleCheck]:
    dgp = DynamicStochasticGainDGP()
    rng = np.random.default_rng(773)
    histories = dgp.sample_histories(64, rng)
    step = 1e-5
    checks: list[OracleCheck] = []
    for channel in ["variance", "third_cumulant"]:
        analytic = dgp.lag_effect(histories, channel)
        numerical = np.zeros_like(analytic)
        for lag in range(dgp.lags):
            plus = histories.copy()
            minus = histories.copy()
            plus[:, lag] += step
            minus[:, lag] -= step
            from .dgps import raw_to_functional

            numerical[:, lag] = (
                raw_to_functional(dgp.raw_moments(plus), channel)
                - raw_to_functional(dgp.raw_moments(minus), channel)
            ) / (2.0 * step)
        checks.append(
            _check(
                "dynamic_lag_formula",
                f"{dgp.mechanism}_{channel}",
                np.max(np.abs(analytic - numerical)),
                2e-8,
                "central finite difference of oracle moment",
            )
        )
    mean_leakage = float(np.max(np.abs(dgp.lag_effect(histories, "mean"))))
    checks.append(_check("mean_blind_direction", dgp.mechanism, mean_leakage, 1e-14, "population identity"))
    return checks


def run_all_oracle_checks(epsilon_ladder: list[float]) -> tuple[list[OracleCheck], list[dict[str, float]]]:
    checks: list[OracleCheck] = []
    checks.extend(gaussian_checks())
    for mechanism in TILT_DEFINITIONS:
        checks.extend(smooth_tilt_checks(mechanism))
    checks.extend(corruption_ledger_checks())
    topology_checks, topology_rows = topology_hodge_checks(epsilon_ladder)
    checks.extend(topology_checks)
    checks.extend(dynamic_formula_checks())
    return checks, topology_rows
