"""Correctness gates for the G8 Gaussian shadow and typed quartic readout."""
from __future__ import annotations

import pytest
import torch

from history_tangent_benchmark.dgps import (
    DTYPE,
    G8FunctionalShapeMixture,
    G8GaussianShadow,
    make_dgp,
)
from history_tangent_benchmark.g8_readouts import (
    g8_frame_quartic_expectations,
    g8_geometry_diagnostics,
    g8_quartic_central_effect,
    g8_quartic_expected_readout,
    g8_quartic_readout,
    g8_quartic_separation,
    validate_g8_geometry,
)


PARAMETERS = {
    "seed": 44,
    "q": 7,
    "dy": 12,
    "n_channels": 3,
    "motif_rank": 3,
}


def test_g8_geometry_is_full_rank_orthogonal_and_quartically_nondegenerate():
    dgp = G8FunctionalShapeMixture(**PARAMETERS)
    diagnostics = validate_g8_geometry(dgp)
    assert diagnostics.motif_matrix_rank == diagnostics.motif_rank == 3
    assert diagnostics.rotation_matrix_rank == 3
    assert diagnostics.motif_orthogonality_error < 1e-12
    assert diagnostics.rotation_orthogonality_error < 1e-12
    assert diagnostics.frame_mean_error < 1e-12
    assert diagnostics.frame_second_moment_error < 1e-12
    assert diagnostics.quartic_separation > 0.1

    # A signed permutation (identity here) preserves coordinate-wise fourth
    # moments and therefore cannot define the intended higher-shape problem.
    dgp.motif_rotation = torch.eye(dgp.motif_rank, dtype=DTYPE)
    with pytest.raises(ValueError, match="quartically degenerate"):
        validate_g8_geometry(dgp)


def test_g8_constructor_rejects_rank_deficient_time_channel_motif_grid():
    with pytest.raises(ValueError, match="rank-deficient motif basis"):
        G8FunctionalShapeMixture(
            seed=1, q=2, dy=2, n_channels=1, motif_rank=2
        )


def test_g8_frames_are_exact_linear_and_quadratic_nulls():
    dgp = G8FunctionalShapeMixture(**PARAMETERS)
    frame_a = dgp.component_offsets[: dgp.components_per_frame]
    frame_b = dgp.component_offsets[dgp.components_per_frame :]
    generator = torch.Generator(device="cpu").manual_seed(91)
    linear = torch.randn(dgp.dy, generator=generator, dtype=DTYPE)
    raw_quadratic = torch.randn(
        (dgp.dy, dgp.dy), generator=generator, dtype=DTYPE
    )
    quadratic = 0.5 * (raw_quadratic + raw_quadratic.T)

    linear_a = (frame_a @ linear).mean()
    linear_b = (frame_b @ linear).mean()
    quadratic_a = torch.einsum("ni,ij,nj->n", frame_a, quadratic, frame_a).mean()
    quadratic_b = torch.einsum("ni,ij,nj->n", frame_b, quadratic, frame_b).mean()
    torch.testing.assert_close(linear_a, linear_b, atol=1e-13, rtol=0)
    torch.testing.assert_close(quadratic_a, quadratic_b, atol=1e-12, rtol=1e-12)


def test_gaussian_shadow_exactly_matches_g8_conditional_mean_and_covariance():
    dgp = G8FunctionalShapeMixture(**PARAMETERS)
    shadow = G8GaussianShadow.from_g8(dgp)
    history = dgp.sample_history(5, seed=101)
    torch.testing.assert_close(shadow.path_mean(history), dgp.path_mean(history))
    torch.testing.assert_close(
        shadow.covariance_at(history), dgp.covariance_at(history), atol=0, rtol=0
    )
    torch.testing.assert_close(shadow.motifs, dgp.motifs, atol=0, rtol=0)
    torch.testing.assert_close(
        shadow.motif_rotation, dgp.motif_rotation, atol=0, rtol=0
    )
    assert shadow.metadata()["shape_control_null"] is True
    assert isinstance(make_dgp("g8_shadow", **PARAMETERS), G8GaussianShadow)


def test_gaussian_shadow_sampling_log_prob_and_shape_null_are_exact():
    dgp = G8FunctionalShapeMixture(**PARAMETERS)
    shadow = G8GaussianShadow.from_g8(dgp)
    history = shadow.sample_history(3, seed=102)
    response = shadow.sample_response(history, n_per_h=4, seed=103)
    assert torch.equal(response, shadow.sample_response(history, n_per_h=4, seed=103))

    centered = response - shadow.path_mean(history)[:, None, :]
    expected = torch.distributions.MultivariateNormal(
        torch.zeros(shadow.dy, dtype=DTYPE),
        covariance_matrix=shadow.conditional_covariance,
    ).log_prob(centered)
    torch.testing.assert_close(shadow.log_prob(response, history), expected)

    history[..., -1] = 0.0
    direction = dgp.mechanism_directions()["shape_only_control"]
    # A frame-B component exposes the real G8 contrast, while the shadow law is
    # exactly invariant to the same control perturbation.
    probe = dgp.path_mean(history) + dgp.component_offsets[dgp.components_per_frame]
    real_contrast = dgp.log_prob(probe, history + direction) - dgp.log_prob(
        probe, history - direction
    )
    shadow_contrast = shadow.log_prob(probe, history + direction) - shadow.log_prob(
        probe, history - direction
    )
    assert torch.min(torch.abs(real_contrast)) > 0.5
    torch.testing.assert_close(shadow_contrast, torch.zeros_like(shadow_contrast), atol=0, rtol=0)


def test_quartic_frame_sign_and_closed_form_separation():
    dgp = G8FunctionalShapeMixture(**PARAMETERS)
    expected_a, expected_b = g8_frame_quartic_expectations(dgp)
    separation = torch.tensor(g8_quartic_separation(dgp), dtype=DTYPE)
    torch.testing.assert_close(expected_a, -separation, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(expected_b, separation, atol=1e-12, rtol=1e-12)


def test_quartic_readout_matches_population_expectation_under_correlated_noise():
    dgp = G8FunctionalShapeMixture(**PARAMETERS)
    history = dgp.sample_history(1, seed=104)
    history[0, -1] = 1.0
    response = dgp.sample_response(history, n_per_h=10_000, seed=105)
    readout = g8_quartic_readout(dgp, response, history)[0]
    target = g8_quartic_expected_readout(dgp, history)[0]
    standard_error = readout.std(unbiased=True) / readout.numel() ** 0.5
    assert torch.abs(readout.mean() - target) < 5.0 * standard_error + 1e-12


def test_quartic_central_effect_has_exact_probability_formula_and_shadow_null():
    dgp = G8FunctionalShapeMixture(**PARAMETERS)
    shadow = G8GaussianShadow.from_g8(dgp)
    history = dgp.sample_history(6, seed=106)
    direction = dgp.mechanism_directions()["shape_only_control"]
    delta = 0.12
    probability_plus = dgp.shape_probability(history + delta * direction)
    probability_minus = dgp.shape_probability(history - delta * direction)
    expected = (
        g8_quartic_separation(dgp)
        * (probability_plus - probability_minus)
        / delta
    )
    torch.testing.assert_close(
        g8_quartic_central_effect(dgp, history, direction, delta),
        expected,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        g8_quartic_central_effect(shadow, history, direction, delta),
        torch.zeros(len(history), dtype=DTYPE),
        atol=0,
        rtol=0,
    )


def test_g8_diagnostics_are_deterministic():
    first = g8_geometry_diagnostics(G8FunctionalShapeMixture(**PARAMETERS))
    second = g8_geometry_diagnostics(G8FunctionalShapeMixture(**PARAMETERS))
    assert first == second

