import numpy as np

from empirical_sid.dgps import GaussianMeanDGP, SmoothTiltDGP
from empirical_sid.estimators import (
    NormalizedStaticModel,
    PolynomialScoreModel,
    RatioCritic,
    RawMomentRegressor,
)


def test_raw_moment_regression_recovers_gaussian_mean_direction():
    rng = np.random.default_rng(11)
    dgp = GaussianMeanDGP()
    h = rng.uniform(-1.2, 1.2, 20_000)
    y = dgp.sample(h, rng)
    model = RawMomentRegressor("mean", degree=5).fit(h, y)
    anchors = np.linspace(-0.7, 0.7, 41)
    relative = np.sqrt(np.mean((model.local_effect(anchors) - dgp.local_effect(anchors)) ** 2)) / np.sqrt(
        np.mean(dgp.local_effect(anchors) ** 2)
    )
    assert relative < 0.12


def test_normalized_tilt_fit_recovers_amplitude():
    rng = np.random.default_rng(12)
    dgp = SmoothTiltDGP("m3_cubic", amplitude=0.48)
    h = rng.uniform(-1.2, 1.2, 30_000)
    y = dgp.sample(h, rng)
    model = NormalizedStaticModel(dgp).fit(h, y)
    assert abs(model.amplitude - dgp.amplitude) < 0.08


def test_score_models_produce_finite_score_and_mixed_field():
    rng = np.random.default_rng(13)
    dgp = SmoothTiltDGP("m3_bounded")
    h = rng.uniform(-1.0, 1.0, 4000)
    y = dgp.sample(h, rng)[:, 0]
    for kind in ["hyvarinen", "dsm", "gaussian_dsm", "anchored"]:
        model = PolynomialScoreModel(kind, ridge=1e-2).fit(h, y, rng)
        score = model.score(h[:100], y[:100], 0.0 if kind == "hyvarinen" else 0.12)
        field = model.mixed_field(h[:100], y[:100], 0.0 if kind == "hyvarinen" else 0.12)
        assert np.all(np.isfinite(score))
        assert np.all(np.isfinite(field))


def test_ratio_critic_tangent_is_finite():
    rng = np.random.default_rng(14)
    dgp = SmoothTiltDGP("m5_occupancy")
    h = rng.uniform(-1.0, 1.0, 3000)
    y = dgp.sample(h, rng)
    critic = RatioCritic().fit(h, y, rng)
    tangent = critic.tangent(h[:200], y[:200])
    assert np.all(np.isfinite(tangent))
