import math

import torch

from history_tangent_benchmark.models import BoundedEnergyRatio


class _PiecewiseConstantTilt(BoundedEnergyRatio):
    """Analytically tractable Gaussian tilt used to test exact sampling."""

    def __init__(self, tilt_bound: float = 0.7, **kwargs):
        super().__init__(
            q=1,
            dy=1,
            hidden=8,
            layers=1,
            tilt_bound=tilt_bound,
            **kwargs,
        )
        self.max_tilt_batch = 0

    def base_parameters(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = h.new_zeros((len(h), 1))
        return zeros, zeros

    def log_tilt(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.max_tilt_batch = max(self.max_tilt_batch, len(y))
        return torch.where(
            y[:, 0] >= 0.0,
            y.new_full((len(y),), self.tilt_bound),
            y.new_full((len(y),), -self.tilt_bound),
        )


def test_exact_rejection_is_reproducible_and_seed_sensitive():
    model = _PiecewiseConstantTilt(
        oversample=4, centering_samples=8, rejection_batch_size=32
    )
    h = torch.zeros(3, 1)

    first = model.sample(h, n_samples=50, seed=314)
    first_diagnostics = model.sampling_diagnostics()
    repeated = model.sample(h, n_samples=50, seed=314)
    repeated_diagnostics = model.sampling_diagnostics()
    different = model.sample(h, n_samples=50, seed=315)

    torch.testing.assert_close(first, repeated, rtol=0.0, atol=0.0)
    assert first_diagnostics == repeated_diagnostics
    assert not torch.equal(first, different)


def test_exact_rejection_matches_known_tilted_gaussian_law():
    bound = 0.7
    model = _PiecewiseConstantTilt(
        tilt_bound=bound,
        oversample=5,
        centering_samples=8,
        rejection_batch_size=31,
    )
    draws = model.sample(torch.zeros(1, 1), n_samples=20_000, seed=902).flatten()

    # Under q=N(0,1) and u(y)=B sign(y), the target sign probability and
    # first moment are available in closed form.
    expected_positive = 1.0 / (1.0 + math.exp(-2.0 * bound))
    expected_mean = math.tanh(bound) * math.sqrt(2.0 / math.pi)
    assert abs(float((draws >= 0.0).float().mean()) - expected_positive) < 0.012
    assert abs(float(draws.mean()) - expected_mean) < 0.018

    # Separate rejection streams must not introduce serial dependence between
    # requested output slots.
    lagged_correlation = torch.corrcoef(torch.stack([draws[:-1], draws[1:]]))[0, 1]
    assert abs(float(lagged_correlation)) < 0.025

    diagnostics = model.sampling_diagnostics()
    expected_acceptance = 0.5 * (1.0 + math.exp(-2.0 * bound))
    assert abs(diagnostics["acceptance_rate"] - expected_acceptance) < 0.015
    assert diagnostics["accepted_samples"] == 20_000
    assert diagnostics["effective_proposals"] >= 20_000
    assert diagnostics["generated_proposals"] >= diagnostics["effective_proposals"]
    assert diagnostics["theoretical_acceptance_lower_bound"] == math.exp(
        -2.0 * bound
    )


def test_rejection_proposal_batches_respect_resource_cap():
    model = _PiecewiseConstantTilt(
        tilt_bound=1.2,
        oversample=7,
        centering_samples=8,
        rejection_batch_size=50,
    )
    samples = model.sample(torch.zeros(4, 1), n_samples=37, seed=71)
    diagnostics = model.sampling_diagnostics()

    assert samples.shape == (4, 37, 1)
    assert torch.isfinite(samples).all()
    assert model.max_tilt_batch <= 50
    assert diagnostics["max_generated_proposals_in_one_batch"] <= 50
    assert diagnostics["proposal_batch_size_cap"] == 50
    assert diagnostics["acceptance_rate"] >= diagnostics[
        "theoretical_acceptance_lower_bound"
    ]


def test_empty_history_batch_has_well_formed_exact_sampler_output():
    model = _PiecewiseConstantTilt(
        oversample=2, centering_samples=8, rejection_batch_size=8
    )
    samples = model.sample(torch.empty(0, 1), n_samples=3, seed=1)

    assert samples.shape == (0, 3, 1)
    diagnostics = model.sampling_diagnostics()
    assert diagnostics["accepted_samples"] == 0
    assert diagnostics["effective_proposals"] == 0
    assert diagnostics["generated_proposals"] == 0
