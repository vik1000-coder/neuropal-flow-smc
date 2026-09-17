import numpy as np

from neuromod_benchmark.mechanistic import hill_occupancy


def test_hill_map_has_concentration_affinity_scale_gauge():
    concentration = np.asarray([0.2, 0.7])
    kd = np.asarray([[0.3, 0.5], [0.8, 0.4]])
    coefficient = np.asarray([[1.0, 1.5], [2.0, 1.2]])
    expression = np.asarray([[0.4, 0.7], [0.8, 0.5]])
    baseline, baseline_derivative = hill_occupancy(
        concentration, kd, coefficient, expression
    )

    scale = 3.25
    rescaled, rescaled_derivative = hill_occupancy(
        scale * concentration, scale * kd, coefficient, expression
    )
    np.testing.assert_allclose(rescaled, baseline, rtol=1e-14, atol=1e-14)
    # The derivative changes inversely with the coordinate scale even though the
    # observable occupancy is identical.
    np.testing.assert_allclose(
        rescaled_derivative, baseline_derivative / scale, rtol=1e-14, atol=1e-14
    )


def test_expression_and_downstream_effect_have_a_multiplicative_gauge():
    concentration = np.asarray([0.4])
    kd = np.asarray([[0.6], [0.25]])
    coefficient = np.asarray([[1.4], [1.8]])
    expression = np.asarray([[0.35], [0.45]])
    downstream_effect = np.asarray([[0.8], [-0.5]])
    occupancy, _ = hill_occupancy(
        concentration, kd, coefficient, expression
    )

    scale = 1.7
    rescaled_occupancy, _ = hill_occupancy(
        concentration, kd, coefficient, scale * expression
    )
    np.testing.assert_allclose(
        (downstream_effect / scale) * rescaled_occupancy,
        downstream_effect * occupancy,
        rtol=1e-14,
        atol=1e-14,
    )
