import numpy as np

from neuromod_benchmark.mechanistic import (
    MechanisticConfig,
    MechanisticState,
    generate_mechanistic_parameters,
    one_step_moments,
    with_mechanism,
)


def _physical_derivative(config, parameters, state, field):
    baseline = np.asarray(getattr(one_step_moments(config, parameters, state), field))
    derivative = np.empty(baseline.shape + (config.n_modulators,))
    for modulator in range(config.n_modulators):
        step = 1e-5 * max(1.0, float(state.concentration[modulator]))
        plus = state.concentration.copy()
        minus = state.concentration.copy()
        plus[modulator] += step
        minus[modulator] -= step
        upper = getattr(
            one_step_moments(
                config,
                parameters,
                MechanisticState(state.neural, plus, state.calcium),
            ),
            field,
        )
        lower = getattr(
            one_step_moments(
                config,
                parameters,
                MechanisticState(state.neural, minus, state.calcium),
            ),
            field,
        )
        derivative[..., modulator] = (upper - lower) / (2.0 * step)
    return derivative


def _signature(config, parameters, state):
    correlation = _physical_derivative(
        config, parameters, state, "conditional_correlation"
    )
    off_diagonal = ~np.eye(config.n_neurons, dtype=bool)
    return {
        "mean": np.linalg.norm(
            _physical_derivative(config, parameters, state, "conditional_mean")
        ),
        "logvariance": np.linalg.norm(
            _physical_derivative(
                config, parameters, state, "conditional_variance"
            )
            / one_step_moments(config, parameters, state).conditional_variance[
                :, None
            ]
        ),
        "upper_tail": np.linalg.norm(
            _physical_derivative(
                config, parameters, state, "upper_tail_probability"
            )
        ),
        "shape_tail": np.linalg.norm(
            _physical_derivative(
                config, parameters, state, "shape_tail_probability"
            )
        ),
        "correlation": np.linalg.norm(correlation[off_diagonal]),
    }


def test_mechanism_channels_have_the_registered_orthogonality_signature():
    base = MechanisticConfig(
        n_neurons=6,
        n_modulators=2,
        n_steps=20,
        burn_in=20,
        mechanism="mixed",
        effect_strength=1.2,
        seed=811,
    )
    parameters = generate_mechanistic_parameters(base)
    state = MechanisticState(
        neural=np.linspace(-0.7, 0.8, base.n_neurons),
        concentration=np.asarray([0.3, 0.75]),
        calcium=np.zeros(base.n_neurons),
    )
    signatures = {
        mechanism: _signature(
            with_mechanism(base, mechanism), parameters, state
        )
        for mechanism in (
            "null",
            "additive_mean",
            "synaptic_gain",
            "intrinsic_excitability",
            "innovation_variance",
            "matched_tail",
            "correlation_routing",
        )
    }
    tolerance = 1e-9
    assert all(value < tolerance for value in signatures["null"].values())

    assert signatures["additive_mean"]["mean"] > tolerance
    assert signatures["synaptic_gain"]["mean"] > tolerance
    assert signatures["intrinsic_excitability"]["mean"] > tolerance
    assert signatures["innovation_variance"]["logvariance"] > tolerance
    assert signatures["matched_tail"]["shape_tail"] > tolerance
    assert signatures["correlation_routing"]["correlation"] > tolerance

    # Standardized shape isolates non-Gaussian shape, whereas the fixed upper-tail
    # risk functional is intentionally sensitive to location, scale, or shape.
    for mechanism in (
        "additive_mean",
        "synaptic_gain",
        "intrinsic_excitability",
        "innovation_variance",
        "correlation_routing",
    ):
        assert signatures[mechanism]["shape_tail"] < tolerance
    assert signatures["additive_mean"]["upper_tail"] > tolerance
    assert signatures["innovation_variance"]["upper_tail"] > tolerance
    assert signatures["matched_tail"]["upper_tail"] > tolerance

    for mechanism in (
        "additive_mean",
        "synaptic_gain",
        "intrinsic_excitability",
        "innovation_variance",
        "matched_tail",
    ):
        assert signatures[mechanism]["correlation"] < tolerance
