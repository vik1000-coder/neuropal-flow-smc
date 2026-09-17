from __future__ import annotations

import numpy as np

from distributional_sid.adjudication_experiments import (
    GeometryAuditCell,
    _deconvolved_characteristic_feature,
    _effective_rank,
    _estimate_filter_rho_mixture_likelihood,
    _fit_e6_adjudication,
    _geometry_intercepts,
    _geometry_subspace_metrics,
    _riesz_population_truth,
    _scalar_characteristic_bank,
    _truth_geometry,
)
from distributional_sid.remaining_experiments import (
    GeometryCell,
    MeasurementCell,
    _dense_parameters_general,
    _sample_measurement_paths,
    _standard_normal_qmc,
)
from distributional_sid.tier_a_extensions import RieszCell, _observation_kernel


def test_exact_characteristic_deconvolution_is_exact_without_noise() -> None:
    rng = np.random.default_rng(1)
    latent = rng.normal(size=1000)
    kernel = _observation_kernel(0.9, 16)
    observed = latent[:, None] * kernel[None, :]
    bank = _scalar_characteristic_bank("wide")
    recovered, maximum_inverse, _ = _deconvolved_characteristic_feature(
        observed, kernel, 0.0, bank
    )
    np.testing.assert_allclose(recovered, bank.transform(latent[:, None]), atol=1e-12)
    assert maximum_inverse == 1.0


def test_symmetric_geometry_has_lower_effective_rank() -> None:
    base = GeometryCell("test", 8, 8, 3, 1000, True, "dense", 64)
    w, b, _ = _dense_parameters_general(2001, base)
    h = _standard_normal_qmc(8, 17, 45)
    symmetric, _, _ = _truth_geometry(
        h, w, b, _geometry_intercepts("symmetric", 3)
    )
    identified, _, _ = _truth_geometry(
        h, w, b, _geometry_intercepts("identified", 3)
    )
    assert _geometry_subspace_metrics(symmetric, symmetric)[
        "response_effective_rank"
    ] == 2.0
    assert _geometry_subspace_metrics(identified, identified)[
        "response_effective_rank"
    ] == 3.0


def test_effective_rank_uses_relative_threshold() -> None:
    assert _effective_rank(np.asarray([10.0, 1.0, 1e-6])) == 2


def test_mixture_likelihood_filter_estimator_recovers_rho() -> None:
    rng = np.random.default_rng(9)
    h = rng.normal(size=(2500, 4))
    cell = MeasurementCell("test", rho=0.98, noise_sd=0.05)
    observed, _, _ = _sample_measurement_paths(h, rng, cell, horizon=32)
    estimate = _estimate_filter_rho_mixture_likelihood(
        h, observed, 0.05, seed=10
    )
    assert abs(estimate - 0.98) < 0.01


def test_e6_estimator_decomposition_smoke() -> None:
    cell = RieszCell("gaussian_d4", "gaussian", 4)
    rows = _fit_e6_adjudication(
        cell=cell,
        seed=2501,
        n_train=500,
        width=16,
        riesz_ridge=1e-2,
        feature_ridge=1e-3,
        include_density_score=False,
        population_truth=_riesz_population_truth(cell),
    )
    assert len(rows) == 7
    assert {row["method"] for row in rows} == {
        "PLUG",
        "RIESZ_TARGETED",
        "ORTH_TARGETED",
        "OR_M_TARGETED",
        "OR_A",
        "OR_R",
        "OR_PW",
    }
    assert all(np.isfinite(row["nrmse"]) for row in rows)


def test_geometry_audit_cell_is_frozen() -> None:
    cell = GeometryAuditCell(8, 8, 3, 8000, 64, "identified", "current")
    assert cell.nominal_rank == 3
