import numpy as np
import pytest
import torch
from torch import nn

from history_tangent_benchmark.models import HeteroscedasticGaussian, build_model, fit_model
from history_tangent_benchmark.stochastic_interpolant import (
    ConditionalPointSourceStochasticInterpolant,
    LearnedGaussianMeanAnchor,
)


def test_default_protocol_fixes_epsilon_and_uses_128_sampler_steps():
    model = ConditionalPointSourceStochasticInterpolant(q=1, dy=1, hidden=8, layers=1)
    assert model.epsilon == pytest.approx(1.0)
    assert model.sample_steps == 128


def test_model_is_available_through_the_common_registry():
    model = build_model(
        "conditional_point_source_stochastic_interpolant",
        q=2,
        dy=3,
        params={"hidden": 8, "layers": 1, "sample_steps": 2},
    )
    assert isinstance(model, ConditionalPointSourceStochasticInterpolant)
    assert model.capabilities.sampler
    assert not model.capabilities.normalized_density


@pytest.mark.parametrize("schedule", ["linear", "squared"])
def test_beta_schedules_have_required_endpoints_and_derivatives(schedule):
    model = ConditionalPointSourceStochasticInterpolant(
        q=1,
        dy=1,
        hidden=8,
        layers=1,
        beta_schedule=schedule,
        sample_steps=2,
    )
    time = torch.tensor([[0.0], [0.25], [1.0]])
    beta = model.beta(time)
    derivative = model.beta_derivative(time)
    assert float(beta[0]) == pytest.approx(0.0)
    assert float(beta[-1]) == pytest.approx(1.0)
    if schedule == "linear":
        torch.testing.assert_close(beta, time)
        torch.testing.assert_close(derivative, torch.ones_like(time))
    else:
        torch.testing.assert_close(beta, time.square())
        torch.testing.assert_close(derivative, 2.0 * time)
        assert float(derivative[0]) == pytest.approx(0.0)


def test_interpolant_and_target_match_declared_formula_at_endpoints():
    anchor = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        anchor.weight.copy_(torch.tensor([[0.5, -0.25]]))
    model = ConditionalPointSourceStochasticInterpolant(
        q=2,
        dy=1,
        hidden=8,
        layers=1,
        beta_schedule="squared",
        anchor=anchor,
        sample_steps=2,
    )
    h = torch.tensor([[2.0, 4.0], [1.0, -2.0]])
    y = torch.tensor([[3.0], [-1.0]])
    noise = torch.tensor([[0.7], [-0.3]])
    anchor_value = anchor(h)

    at_zero, target_zero = model.interpolant_and_target(h, y, 0.0, noise)
    torch.testing.assert_close(at_zero, anchor_value)
    torch.testing.assert_close(target_zero, -anchor_value)

    at_one, target_one = model.interpolant_and_target(h, y, 1.0, noise)
    torch.testing.assert_close(at_one, y)
    torch.testing.assert_close(target_one, -anchor_value + 2.0 * y - noise)


def test_supplied_gaussian_anchor_is_frozen_and_uses_its_mean():
    torch.manual_seed(8)
    anchor = HeteroscedasticGaussian(q=2, dy=1, hidden=8, layers=1)
    model = ConditionalPointSourceStochasticInterpolant(
        q=2,
        dy=1,
        hidden=8,
        layers=1,
        anchor=anchor,
        sample_steps=2,
    )
    h = torch.randn(5, 2)
    expected = anchor.parameters_at(h)[0]
    torch.testing.assert_close(model.anchor_mean(h), expected)
    assert model.anchor_frozen
    assert all(not parameter.requires_grad for parameter in model.anchor.parameters())
    assert all(parameter.requires_grad for parameter in model.drift_network.parameters())


def test_internal_gaussian_anchor_is_learned_by_default():
    model = ConditionalPointSourceStochasticInterpolant(
        q=2, dy=1, hidden=8, layers=1, anchor_hidden=8, anchor_layers=1, sample_steps=2
    )
    assert isinstance(model.anchor, LearnedGaussianMeanAnchor)
    assert not model.anchor_frozen
    assert all(parameter.requires_grad for parameter in model.anchor.parameters())
    h = torch.randn(6, 2)
    y = torch.randn(6, 1)
    loss = model.native_loss(h, y)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.anchor.parameters())
    assert any(parameter.grad is not None for parameter in model.drift_network.parameters())


def test_validation_loss_and_sampler_are_deterministic_and_finite():
    torch.manual_seed(11)
    model = ConditionalPointSourceStochasticInterpolant(
        q=2, dy=3, hidden=12, layers=1, beta_schedule="squared", sample_steps=4
    )
    h = torch.randn(4, 2)
    y = torch.randn(4, 3)
    first_loss = model.validation_loss(h, y)
    second_loss = model.validation_loss(h, y)
    torch.testing.assert_close(first_loss, second_loss)

    first = model.sample(h, n_samples=5, seed=23)
    second = model.sample(h, n_samples=5, seed=23)
    different = model.sample(h, n_samples=5, seed=24)
    assert first.shape == (4, 5, 3)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different)


def test_field_can_overfit_a_fixed_regression_batch():
    torch.manual_seed(31)
    anchor = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        anchor.weight.zero_()
    model = ConditionalPointSourceStochasticInterpolant(
        q=1,
        dy=1,
        hidden=24,
        layers=1,
        beta_schedule="linear",
        anchor=anchor,
        sample_steps=2,
    )
    h = torch.linspace(-1.0, 1.0, 24)[:, None]
    y = (0.8 * h + 0.4).clone()
    time = torch.linspace(0.05, 0.95, 24)[:, None]
    noise = torch.linspace(-0.7, 0.7, 24)[:, None]
    optimizer = torch.optim.Adam(model.drift_network.parameters(), lr=1e-2)
    initial = float(model.field_loss(h, y, time, noise).detach())
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        loss = model.field_loss(h, y, time, noise)
        loss.backward()
        optimizer.step()
    final = float(model.field_loss(h, y, time, noise).detach())
    assert final < 0.1 * initial


def test_model_completes_project_fit_loop_smoke():
    rng = np.random.default_rng(14)
    h = rng.normal(size=(80, 2)).astype("float32")
    y = (0.6 * h[:, :1] + 0.15 * rng.normal(size=(80, 1))).astype("float32")
    model = ConditionalPointSourceStochasticInterpolant(
        q=2, dy=1, hidden=12, layers=1, anchor_hidden=12, anchor_layers=1, sample_steps=2
    )
    trace = fit_model(
        model,
        h[:56],
        y[:56],
        h[56:],
        y[56:],
        seed=5,
        device="cpu",
        max_epochs=2,
        patience=2,
        batch_size=28,
        learning_rate=1e-3,
    )
    assert len(trace.train_loss) == 2
    assert np.isfinite(trace.train_loss).all()
    assert np.isfinite(trace.validation_loss).all()
    assert trace.parameter_count > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"beta_schedule": "cosine"}, "beta_schedule"),
        ({"epsilon": 0.5}, "epsilon"),
        ({"sample_steps": 0}, "sample_steps"),
        ({"anchor_loss_weight": -1.0}, "anchor_loss_weight"),
    ],
)
def test_invalid_protocol_parameters_fail_loudly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ConditionalPointSourceStochasticInterpolant(q=1, dy=1, **kwargs)
