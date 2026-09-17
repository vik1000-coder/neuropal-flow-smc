from __future__ import annotations

import numpy as np

from distributional_sid.core import (
    CharacteristicBank,
    MomentBlindDiscreteDGP,
    MomentBlindLegendreDGP,
)


def _check_dgp(dgp) -> None:
    rng = np.random.default_rng(44)
    h_bank = dgp.sample_histories(rng, 2000)
    y_bank = dgp.sample_responses(h_bank, rng)
    bank = CharacteristicBank.fit(y_bank, 32, 99)
    h = dgp.sample_histories(rng, 16)
    analytic = dgp.conditional_feature_derivative(h, 0, bank)
    delta = 1e-5
    plus = h.copy()
    minus = h.copy()
    plus[:, 0] += delta
    minus[:, 0] -= delta
    finite = (
        dgp.conditional_feature_mean(plus, bank)
        - dgp.conditional_feature_mean(minus, bank)
    ) / (2.0 * delta)
    np.testing.assert_allclose(analytic, finite, rtol=2e-7, atol=2e-9)
    for order in range(1, 5):
        assert abs(dgp.moment_derivative(order)) < 1e-12


def test_discrete_moment_blind_identities() -> None:
    _check_dgp(MomentBlindDiscreteDGP(seed=1, k=4))


def test_continuous_moment_blind_identities() -> None:
    _check_dgp(MomentBlindLegendreDGP(seed=1, k=4))
