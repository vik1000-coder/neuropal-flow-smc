from __future__ import annotations

import numpy as np

from distributional_sid.core import CharacteristicBank, SupportMotionDGP


def test_support_motion_characteristic_derivative_at_zero_noise() -> None:
    dgp = SupportMotionDGP(seed=1, noise_sigma=0.0)
    rng = np.random.default_rng(9)
    h_bank = dgp.sample_histories(rng, 500)
    bank = CharacteristicBank.fit(dgp.sample_responses(h_bank, rng), 16, 7)
    h = dgp.sample_histories(rng, 20)
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
    np.testing.assert_allclose(analytic, finite, rtol=2e-7, atol=2e-8)


def test_likelihood_score_unavailable_at_zero_noise() -> None:
    dgp = SupportMotionDGP(seed=1, noise_sigma=0.0)
    h = np.zeros((2, 1))
    y = np.zeros((2, 1))
    try:
        dgp.likelihood_history_score(h, y)
    except ValueError:
        pass
    else:
        raise AssertionError("sigma=0 likelihood score should be unavailable")
