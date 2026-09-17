import numpy as np

from history_tangent_benchmark.finite_contrast import (
    AmortizedSignedClassifier,
    FixedResponseDictionary,
    SignedRieszRegressor,
    central_mixture_witness,
    oracle_logistic_excess_risk,
    signed_mixture_readout,
)


def _normal_logpdf(y, mean, variance=1.0):
    return -0.5 * (np.log(2 * np.pi * variance) + (y - mean) ** 2 / variance)


def _balanced_shift(n, delta, seed):
    rng = np.random.default_rng(seed)
    anchors = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, 1))
    plus = anchors + delta + noise
    minus = anchors - delta + noise
    history = np.vstack([anchors, anchors])
    response = np.vstack([plus, minus])
    labels = np.concatenate([np.ones(n), -np.ones(n)])
    return history, response, labels


def test_central_gaussian_witness_is_bounded_and_has_exact_mean_readout():
    rng = np.random.default_rng(7)
    delta = 0.3
    n = 200_000
    labels = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    response = labels * delta + rng.standard_normal(n)
    witness = central_mixture_witness(
        _normal_logpdf(response, delta), _normal_logpdf(response, -delta), delta
    )
    assert np.max(np.abs(witness)) <= 1 / delta + 1e-12
    assert abs(np.mean(witness)) < 0.02
    assert abs(np.mean(response * witness) - 1.0) < 0.02


def test_signed_mixture_readout_uses_the_coefficient_norm():
    labels = np.array([1.0, 1.0, -1.0, -1.0])
    channels = np.array([[3.0], [5.0], [1.0], [1.0]])
    assert np.allclose(signed_mixture_readout(channels, labels, 2.0), [3.0])


def test_signed_riesz_recovers_a_gaussian_mean_channel():
    delta = 0.25
    h, y, labels = _balanced_shift(6_000, delta, 11)
    dictionary = FixedResponseDictionary.fit(y)
    regressor = SignedRieszRegressor(dictionary, ridge=1e-3)
    fit = regressor.fit(y, labels, 1 / delta)
    h_test, y_test, labels_test = _balanced_shift(20_000, delta, 12)
    witness = regressor.predict(y_test)
    readout = float(np.mean(y_test[:, 0] * witness))
    direct = float(signed_mixture_readout(y_test, labels_test, 1 / delta)[0])
    assert fit.witness_norm_sq >= 0
    assert fit.condition_number < 1e8
    assert abs(readout - direct) < 0.08


def test_amortized_classifier_is_bounded_centered_and_has_small_oracle_regret():
    delta = 0.3
    train = _balanced_shift(2_000, delta, 21)
    calibration = _balanced_shift(800, delta, 22)
    classifier = AmortizedSignedClassifier(1, 1, hidden=24, layers=2, seed=23)
    fit = classifier.fit(
        *train,
        *calibration,
        max_epochs=35,
        batch_size=256,
        patience=7,
    )
    history, response, _ = _balanced_shift(4_000, delta, 24)
    probability = classifier.predict_probability(history, response)
    witness = classifier.predict_witness(history, response, 1 / delta)
    true_probability = 1.0 / (
        1.0 + np.exp(-2.0 * delta * (response[:, 0] - history[:, 0]))
    )
    assert np.max(np.abs(witness)) <= 1 / delta + 1e-12
    assert fit.calibration_centering < 1e-3
    assert oracle_logistic_excess_risk(true_probability, probability) < 0.08
