"""Correctness gates for the exact continuous G1--G8 oracle library."""
from __future__ import annotations

import pytest
import torch

from history_tangent_benchmark.dgps import (
    G1LocationGaussian,
    G3SkewMixture,
    G4LowRankMixture,
    G7ConcentratedHistory,
    G8FunctionalShapeMixture,
    make_dgp,
)


DTYPE = torch.float64


SMALL_DGPS = (
    ("g1", {"dy": 3}),
    ("g2", {"dy": 3, "alpha_m": 0.3}),
    ("g3", {"dy": 3}),
    ("g4", {"dy": 3, "n_components": 4, "rank": 2}),
    ("g5", {"dy": 6, "latent_dim": 2, "n_components": 4, "rank": 2}),
    ("g6", {"dy": 3, "omega": 8.0}),
    ("g7", {"q": 8, "dy": 3, "intrinsic_dim": 3}),
    ("g8", {"q": 7, "dy": 12, "n_channels": 3, "motif_rank": 3}),
)


@pytest.mark.parametrize(("name", "kwargs"), SMALL_DGPS)
def test_continuous_dgp_contract_shapes_dtype_and_determinism(name, kwargs):
    dgp = make_dgp(name, seed=7, **kwargs)
    h = dgp.sample_history(4, seed=13)
    y = dgp.sample_response(h, n_per_h=5, seed=17)
    assert h.shape == (4, dgp.q)
    assert y.shape == (4, 5, dgp.dy)
    assert h.dtype == y.dtype == DTYPE
    assert h.device.type == y.device.type == "cpu"
    assert torch.equal(h, dgp.sample_history(4, seed=13))
    assert torch.equal(y, dgp.sample_response(h, n_per_h=5, seed=17))

    log_prob = dgp.log_prob(y, h)
    tangent = dgp.history_tangent(y, h)
    response_score = dgp.response_score(y, h)
    assert log_prob.shape == (4, 5)
    assert tangent.shape == (4, 5, dgp.q)
    assert response_score.shape == (4, 5, dgp.dy)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(tangent).all()
    assert torch.isfinite(response_score).all()
    assert dgp.metadata()["generator_seed"] == 7
    assert dgp.mechanism_directions()


def test_log_prob_broadcasts_single_history_and_all_pairs():
    dgp = make_dgp("g1", seed=0, q=4, dy=2)
    h = dgp.sample_history(3, seed=1)
    y = dgp.sample_response(h, 4, seed=2)
    assert dgp.log_prob(y, h).shape == (3, 4)
    assert dgp.log_prob(y[0], h[0]).shape == (4,)
    assert dgp.log_prob(y[:, 0], h).shape == (3,)
    # Explicit singleton axes request the Cartesian product under ordinary broadcasting.
    assert dgp.log_prob(y[:, 0][None, ...], h[:, None]).shape == (3, 3)


def _g1_analytic_tangent(dgp: G1LocationGaussian, y: torch.Tensor, h: torch.Tensor):
    hidden = torch.tanh(h @ dgp.w1.T)
    jacobian = torch.einsum(
        "dk,...k,kq->...dq", dgp.w2, 1.0 - hidden.square(), dgp.w1
    )
    residual = y - dgp.mean(h)
    precision_residual = torch.linalg.solve(dgp.covariance, residual.T).T
    return torch.einsum("...dq,...d->...q", jacobian, precision_residual)


def test_g1_analytic_history_tangent_and_response_score():
    dgp = G1LocationGaussian(seed=2, q=4, dy=3)
    h = dgp.sample_history(8, seed=3)
    y = dgp.sample_response(h, 1, seed=4)[:, 0]
    expected = _g1_analytic_tangent(dgp, y, h)
    actual = dgp.history_tangent(y, h)
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-11)
    expected_score = -torch.linalg.solve(dgp.covariance, (y - dgp.mean(h)).T).T
    assert torch.allclose(dgp.response_score(y, h), expected_score, rtol=1e-10, atol=1e-11)


def test_g3_analytic_tangent_and_exact_fixed_moments():
    dgp = G3SkewMixture(seed=3, q=5, dy=2)
    h = dgp.sample_history(9, seed=4)
    y = dgp.sample_response(h, 1, seed=5)[:, 0]
    weights = dgp.weights(h)
    variance = dgp.component_sd**2
    kernels = torch.exp(
        -0.5
        * (
            torch.log(torch.tensor(2.0 * torch.pi * variance, dtype=DTYPE))
            + (y[:, 0, None] - dgp.component_means).square() / variance
        )
    )
    density = (weights * kernels).sum(dim=-1)
    coefficient = (kernels * dgp.weight_direction).sum(dim=-1) / density
    du = 0.12 * (1.0 - torch.tanh(h @ dgp.direction).square())
    expected = coefficient[:, None] * du[:, None] * dgp.direction
    assert torch.allclose(dgp.history_tangent(y, h), expected, rtol=1e-10, atol=1e-11)

    probe_h = torch.stack([-2.0 * dgp.direction, torch.zeros(dgp.q), 2.0 * dgp.direction])
    probe_weights = dgp.weights(probe_h)
    means = probe_weights @ dgp.component_means
    seconds = probe_weights @ (dgp.component_means.square() + variance)
    variances = seconds - means.square()
    assert torch.max(means) - torch.min(means) < 1e-13
    assert torch.max(variances) - torch.min(variances) < 1e-13
    centered_cube = (dgp.component_means[None, :] - means[:, None]).pow(3)
    third = (
        probe_weights
        * (centered_cube + 3 * variance * (dgp.component_means - means[:, None]))
    ).sum(-1)
    assert torch.max(third) - torch.min(third) > 0.1


def _g4_analytic_tangent(
    dgp: G4LowRankMixture,
    y: torch.Tensor,
    h: torch.Tensor,
    sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    sigma = torch.zeros(dgp.dy, dtype=DTYPE) if sigma is None else sigma
    variance = dgp.component_sigma**2 + sigma.square()
    logits = dgp.logits(h)
    log_pi = torch.log_softmax(logits, dim=-1)
    component = -0.5 * (
        torch.log(2.0 * torch.pi * variance)
        + (y[:, None, :] - dgp.component_means).square() / variance
    ).sum(-1)
    responsibility = torch.softmax(log_pi + component, dim=-1)
    return (responsibility - torch.softmax(logits, dim=-1)) @ dgp.b_matrix


def test_g4_analytic_history_tangent_clean_and_noisy():
    dgp = G4LowRankMixture(seed=4, q=5, dy=4, n_components=4, rank=2)
    h = dgp.sample_history(10, seed=5)
    y = dgp.sample_response(h, 1, seed=6)[:, 0]
    assert torch.allclose(
        dgp.history_tangent(y, h), _g4_analytic_tangent(dgp, y, h), rtol=1e-10, atol=1e-11
    )
    sigma = torch.linspace(0.05, 0.2, dgp.dy, dtype=DTYPE)
    assert torch.allclose(
        dgp.noisy_history_tangent(y, h, sigma),
        _g4_analytic_tangent(dgp, y, h, sigma),
        rtol=1e-10,
        atol=1e-11,
    )


@pytest.mark.parametrize(("name", "kwargs"), SMALL_DGPS)
def test_autograd_tangent_matches_central_finite_difference(name, kwargs):
    dgp = make_dgp(name, seed=11, **kwargs)
    h = dgp.sample_history(5, seed=12)
    y = dgp.sample_response(h, 1, seed=13)[:, 0]
    direction = next(iter(dgp.mechanism_directions().values()))
    direction = direction / direction.norm()
    epsilon = 2e-5
    finite_difference = (
        dgp.log_prob(y, h + epsilon * direction)
        - dgp.log_prob(y, h - epsilon * direction)
    ) / (2.0 * epsilon)
    derivative = dgp.history_tangent(y, h) @ direction
    assert torch.allclose(derivative, finite_difference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    (
        ("g1", {"dy": 3}),
        ("g2", {"dy": 3, "alpha_m": 0.3}),
        ("g3", {"dy": 3}),
        ("g4", {"dy": 3, "n_components": 4, "rank": 2}),
    ),
)
def test_conditional_history_tangent_is_centered(name, kwargs):
    dgp = make_dgp(name, seed=20, **kwargs)
    h = dgp.sample_history(3, seed=21)
    y = dgp.sample_response(h, n_per_h=2048, seed=22)
    tangent = dgp.history_tangent(y, h)
    center = tangent.mean(dim=1)
    scale = tangent.square().mean().sqrt().clamp_min(1e-10)
    assert center.square().mean().sqrt() / scale < 0.07


def test_finite_log_ratio_converges_to_directional_tangent():
    for name, kwargs in SMALL_DGPS:
        dgp = make_dgp(name, seed=30, **kwargs)
        h = dgp.sample_history(4, seed=31)
        y = dgp.sample_response(h, 1, seed=32)[:, 0]
        direction = next(iter(dgp.mechanism_directions().values()))
        direction = direction / direction.norm()
        delta = 1e-5
        ratio_derivative = dgp.finite_log_ratio(y, h, direction, delta) / delta
        tangent_derivative = dgp.history_tangent(y, h) @ direction
        assert torch.allclose(ratio_derivative, tangent_derivative, rtol=3e-4, atol=3e-5)


def test_g7_supported_and_orthogonal_invariants():
    dgp = G7ConcentratedHistory(
        seed=40, q=10, dy=3, intrinsic_dim=3, tau=0.02, intrinsic="g4"
    )
    h = dgp.sample_history(8, seed=41)
    y = dgp.sample_response(h, 1, seed=42)[:, 0]
    tangent = dgp.history_tangent(y, h)
    supported = dgp.mechanism_directions()["supported_0"]
    orthogonal = dgp.mechanism_directions()["orthogonal_0"]
    assert torch.max(torch.abs(tangent @ orthogonal)) < 2e-12
    assert torch.mean(torch.abs(tangent @ supported)) > 1e-5
    assert torch.max(torch.abs(dgp.finite_log_ratio(y, h, orthogonal, 0.25))) < 2e-12


def test_g8_shape_frames_have_identical_first_two_moments():
    dgp = G8FunctionalShapeMixture(
        seed=44, q=7, dy=12, n_channels=3, motif_rank=3
    )
    frame_a = dgp.component_offsets[: dgp.components_per_frame]
    frame_b = dgp.component_offsets[dgp.components_per_frame :]
    torch.testing.assert_close(frame_a.mean(0), frame_b.mean(0), atol=1e-12, rtol=0)
    covariance_a = frame_a.T @ frame_a / dgp.components_per_frame
    covariance_b = frame_b.T @ frame_b / dgp.components_per_frame
    torch.testing.assert_close(covariance_a, covariance_b, atol=1e-12, rtol=1e-12)
    h = dgp.sample_history(3, seed=45)
    direction = dgp.mechanism_directions()["shape_only_control"]
    torch.testing.assert_close(
        dgp.path_mean(h + direction), dgp.path_mean(h - direction), atol=0, rtol=0
    )
    assert torch.max(
        torch.abs(dgp.shape_probability(h + direction) - dgp.shape_probability(h - direction))
    ) > 0.1


def test_g1_noisy_density_accepts_scalar_and_coordinatewise_sigma():
    dgp = G1LocationGaussian(seed=50, q=4, dy=3)
    h = dgp.sample_history(6, seed=51)
    y = dgp.sample_response(h, 1, seed=52)[:, 0]
    sigma = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)
    covariance = dgp.covariance + torch.diag(sigma.square())
    expected = torch.distributions.MultivariateNormal(
        dgp.mean(h), covariance_matrix=covariance
    ).log_prob(y)
    assert torch.allclose(dgp.noisy_log_prob(y, h, sigma), expected, rtol=1e-11, atol=1e-11)
    assert dgp.noisy_log_prob(y, h, 0.1).shape == (6,)
    noisy_tangent = dgp.noisy_history_tangent(y, h, sigma)
    noisy_score = dgp.noisy_response_score(y, h, sigma)
    assert noisy_tangent.shape == (6, dgp.q)
    assert noisy_score.shape == (6, dgp.dy)


def test_invalid_factory_and_noise_inputs_fail_closed():
    with pytest.raises(ValueError):
        make_dgp("not-a-generator")
    dgp = make_dgp("g1", dy=3)
    h = dgp.sample_history(1, 0)
    y = dgp.sample_response(h, 1, 1)[:, 0]
    with pytest.raises(ValueError):
        dgp.noisy_log_prob(y, h, torch.ones(2, dtype=DTYPE))
    with pytest.raises(ValueError):
        dgp.noisy_log_prob(y, h, -0.1)


def test_factory_accepts_plan_and_config_parameter_aliases():
    g1 = make_dgp("g1_location_gaussian", history_dim=3, d_y=2, hidden_width=5)
    assert (g1.q, g1.dy, g1.hidden) == (3, 2, 5)
    g4 = make_dgp(
        "g4_mixture", history_dim=4, d_y=3, K=4, interaction_rank=2, R_over_sigma=1.0
    )
    assert (g4.q, g4.dy, g4.n_components, g4.rank, g4.mode_separation) == (4, 3, 4, 2, 1.0)
