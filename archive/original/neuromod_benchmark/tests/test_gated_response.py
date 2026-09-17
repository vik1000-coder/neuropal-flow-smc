import numpy as np

from neuromod_benchmark.evaluation import physical_gated_response_metrics
from neuromod_benchmark.features import build_supervised
from neuromod_benchmark.mechanistic import MechanisticConfig
from neuromod_benchmark.mechanistic_dataset import simulate_mechanistic_dataset
from neuromod_benchmark.methods.classical import ConstantGaussian


def test_null_model_has_zero_total_physical_gating_and_finite_metrics():
    dataset = simulate_mechanistic_dataset(
        MechanisticConfig(
            n_neurons=3,
            n_modulators=1,
            n_steps=55,
            burn_in=80,
            mechanism="null",
            ligand_pulse_rate_hz=.1,
            seed=31,
        ),
        n_trajectories=3,
    )
    data = build_supervised(dataset, view="complete_state", history_lags=(1,), horizon=1)
    estimator = ConstantGaussian().fit(data)
    metrics = physical_gated_response_metrics(estimator, dataset, data, max_states=8)
    assert metrics
    # Ranking metrics are deliberately undefined when the oracle graph has no
    # positives; numerical recovery metrics must still be finite and exactly zero.
    assert np.isfinite(metrics["neuromodulator.total_gated_response.mae"])
    assert metrics["neuromodulator.total_gated_response.mae"] < 1e-10
    assert np.isnan(metrics["neuromodulator.total_gated_response_support.auprc"])
