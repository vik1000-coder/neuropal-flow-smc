import numpy as np
import pytest
import torch

from history_tangent_benchmark.models import (
    AutoregressiveMixtureDensity,
    AutoregressiveTransformerDensity,
    BoundedEnergyRatio,
    ConditionalAffineFlow,
    ConditionalEDMDiffusion,
    ConditionalFlowMatching,
    ConditionalVectorDiffusion,
    ConstrainedGaussianDSM,
    HeteroscedasticGaussian,
    GaussianSourceFlowMatching,
    GaussianAnchoredEDM,
    RatioCritic,
    fit_model,
    history_score_matching_penalty,
)


def _finite_difference(model, y, h, coordinate, epsilon=1e-3):
    plus = h.clone()
    minus = h.clone()
    plus[:, coordinate] += epsilon
    minus[:, coordinate] -= epsilon
    return (model.log_prob(y, plus) - model.log_prob(y, minus)) / (2 * epsilon)


@pytest.mark.parametrize(
    "model",
    [
        HeteroscedasticGaussian(q=3, dy=2, hidden=16, layers=1),
        ConstrainedGaussianDSM(q=3, dy=2, hidden=16, layers=1),
        ConditionalAffineFlow(q=3, dy=2, hidden=16, layers=1, coupling_layers=2),
        AutoregressiveMixtureDensity(q=3, dy=2, hidden=16, layers=1, components=3),
        AutoregressiveTransformerDensity(
            q=3, dy=2, d_model=12, nhead=3, transformer_layers=1,
            feedforward=24, components=3,
        ),
    ],
)
def test_normalized_model_tangent_matches_finite_difference(model):
    torch.manual_seed(2)
    h = torch.randn(7, 3)
    y = torch.randn(7, 2)
    tangent = model.history_tangent(y, h)
    finite = _finite_difference(model, y, h, coordinate=1)
    torch.testing.assert_close(tangent[:, 1], finite, rtol=5e-3, atol=3e-3)


def test_autoregressive_samples_and_log_prob_are_finite():
    model = AutoregressiveMixtureDensity(q=2, dy=3, hidden=12, layers=1, components=4)
    h = torch.zeros(5, 2)
    samples = model.sample(h, n_samples=6, seed=9)
    assert samples.shape == (5, 6, 3)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(model.log_prob(samples[:, 0], h)).all()


@pytest.mark.parametrize("permutation_seed", [0, 17])
def test_autoregressive_transformer_samples_and_log_prob_are_finite(permutation_seed):
    model = AutoregressiveTransformerDensity(
        q=2, dy=4, d_model=16, nhead=4, transformer_layers=1,
        feedforward=32, components=4, permutation_seed=permutation_seed,
    )
    h = torch.zeros(5, 2)
    samples = model.sample(h, n_samples=6, seed=9)
    assert samples.shape == (5, 6, 4)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(model.log_prob(samples[:, 0], h)).all()
    assert sorted(model.coordinate_permutation.tolist()) == [0, 1, 2, 3]


def test_history_score_matching_penalty_is_finite_and_differentiable():
    model = HeteroscedasticGaussian(q=2, dy=1, hidden=12, layers=1)
    h = torch.randn(16, 2)
    y = torch.randn(16, 1)
    penalty = history_score_matching_penalty(model, h, y)
    assert torch.isfinite(penalty)
    penalty.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_stable_generative_samplers_are_finite_and_reproducible():
    h = torch.zeros(3, 2)
    models = [
        ConditionalEDMDiffusion(q=2, dy=2, hidden=12, layers=1, sample_steps=4),
        GaussianAnchoredEDM(q=2, dy=2, hidden=12, layers=1, sample_steps=4),
        ConditionalFlowMatching(q=2, dy=2, hidden=12, layers=1, sample_steps=4),
        GaussianSourceFlowMatching(q=2, dy=2, hidden=12, layers=1, sample_steps=4),
    ]
    for model in models:
        first = model.sample(h, n_samples=4, seed=19)
        second = model.sample(h, n_samples=4, seed=19)
        assert first.shape == (3, 4, 2)
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first, second)


def test_bounded_energy_ratio_is_tail_safe_and_centered():
    torch.manual_seed(29)
    model = BoundedEnergyRatio(
        q=2, dy=2, hidden=12, layers=1, tilt_bound=2.0,
        oversample=3, centering_samples=16,
    )
    h = torch.randn(5, 2)
    y = torch.randn(5, 2)
    tilt = model.log_tilt(y, h)
    assert torch.all(tilt.abs() <= 2.0 + 1e-6)
    samples = model.sample(h, n_samples=8, seed=19)
    assert samples.shape == (5, 8, 2)
    assert torch.isfinite(samples).all()
    tangent = model.history_tangent(y, h)
    assert tangent.shape == h.shape
    assert torch.isfinite(tangent).all()
    score = model.response_score(y, h)
    assert score.shape == y.shape
    assert torch.isfinite(score).all()


def test_ratio_critic_exposes_no_fabricated_density_or_sampler():
    critic = RatioCritic(q=2, dy=2, hidden=12, layers=1, rank=6)
    assert not critic.capabilities.normalized_density
    assert not critic.capabilities.sampler
    h = torch.randn(8, 2)
    y = torch.randn(8, 2)
    tangent = critic.history_tangent(y, h)
    assert tangent.shape == h.shape
    assert torch.isfinite(critic.native_loss(h, y))


class _ExactGaussianScore(ConditionalVectorDiffusion):
    def __init__(self, matrix):
        matrix = torch.as_tensor(matrix, dtype=torch.float32)
        super().__init__(q=matrix.shape[1], dy=matrix.shape[0], hidden=8, layers=1)
        self.register_buffer("matrix", matrix)

    def score(self, y_noisy, h, sigma):
        variance = 1.0 + float(sigma) ** 2
        return -(y_noisy - h @ self.matrix.T) / variance


def test_centered_mixed_path_recovers_gaussian_history_tangent():
    matrix = torch.tensor([[0.7, -0.2], [0.1, 0.5]])
    model = _ExactGaussianScore(matrix)
    h = torch.tensor([[0.3, -0.5], [0.1, 0.2]], dtype=torch.float32)
    y = torch.tensor([[0.4, -0.1], [-0.3, 0.7]], dtype=torch.float32)
    sigma = 0.3
    mean = h @ matrix.T
    offsets = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]],
        dtype=torch.float32,
    )
    center_samples = mean[:, None, :] + offsets
    estimated = model.mixed_history_tangent(
        y, h, sigma, center_samples, quadrature_points=4
    )
    truth = (y - mean) @ matrix / (1.0 + sigma**2)
    torch.testing.assert_close(estimated, truth, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "model",
    [
        HeteroscedasticGaussian(q=2, dy=1, hidden=12, layers=1),
        ConstrainedGaussianDSM(q=2, dy=1, hidden=12, layers=1),
        ConditionalAffineFlow(q=2, dy=1, hidden=12, layers=1, coupling_layers=2),
        AutoregressiveMixtureDensity(q=2, dy=1, hidden=12, layers=1, components=3),
        AutoregressiveTransformerDensity(
            q=2, dy=1, d_model=12, nhead=3, transformer_layers=1,
            feedforward=24, components=3,
        ),
        RatioCritic(q=2, dy=1, hidden=12, layers=1, rank=5),
        ConditionalVectorDiffusion(q=2, dy=1, hidden=12, layers=1),
        ConditionalEDMDiffusion(q=2, dy=1, hidden=12, layers=1, sample_steps=4),
        GaussianAnchoredEDM(q=2, dy=1, hidden=12, layers=1, sample_steps=4),
        ConditionalFlowMatching(q=2, dy=1, hidden=12, layers=1, sample_steps=4),
        GaussianSourceFlowMatching(q=2, dy=1, hidden=12, layers=1, sample_steps=4),
        BoundedEnergyRatio(
            q=2, dy=1, hidden=12, layers=1, oversample=2, centering_samples=8
        ),
    ],
)
def test_models_complete_short_native_objective_fit(model):
    rng = np.random.default_rng(4)
    h = rng.normal(size=(96, 2)).astype("float32")
    y = (0.5 * h[:, :1] + 0.2 * rng.normal(size=(96, 1))).astype("float32")
    trace = fit_model(
        model,
        h[:64],
        y[:64],
        h[64:],
        y[64:],
        seed=3,
        device="cpu",
        max_epochs=2,
        patience=2,
        batch_size=32,
        learning_rate=1e-3,
    )
    assert len(trace.train_loss) == 2
    assert np.isfinite(trace.validation_loss).all()
    assert trace.parameter_count > 0
