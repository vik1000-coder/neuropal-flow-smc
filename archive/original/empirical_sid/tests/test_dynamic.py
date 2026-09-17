import numpy as np

from empirical_sid.dgps import DynamicStochasticGainDGP
from empirical_sid.dynamic import CorrectGainMixture, DynamicMomentRegressor, DynamicRatioCritic


def test_dynamic_estimators_have_expected_shapes():
    rng = np.random.default_rng(81)
    dgp = DynamicStochasticGainDGP(lags=4)
    h = dgp.sample_histories(2500, rng)
    y = dgp.sample_vector(h, rng)
    anchors = dgp.sample_histories(20, rng)
    direct = DynamicMomentRegressor().fit(h, y)
    mixture = CorrectGainMixture(dgp).fit(h, y)
    critic = DynamicRatioCritic().fit(h, y, rng)
    assert direct.effect(anchors, "variance").shape == (20, 4)
    assert mixture.effect(anchors, "third_cumulant").shape == (20, 4)
    assert critic.tangent(anchors, dgp.sample_vector(anchors, rng)).shape == (20, 4)
    assert np.all(np.isfinite(direct.effect(anchors, "covariance")))
