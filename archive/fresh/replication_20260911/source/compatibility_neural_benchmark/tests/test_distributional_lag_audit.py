from __future__ import annotations

import numpy as np
import torch

from compatibility_neural_benchmark.core import GeneratorAdapter
from compatibility_neural_benchmark.distributional_lag_audit import (
    build_low_high_histories,
    conditional_features,
    distributional_effects,
    evaluation_targets,
    sample_many,
)
from compatibility_neural_benchmark.latent_targeted_smc import (
    normalized_weights,
    smc_initial_histories,
)
from compatibility_neural_benchmark.paired_lag_correspondence import (
    permute_within_source,
    source_bootstrap_auroc,
)
from conditional_neural_benchmark.data import StimulusSchedule


class _ToyModel:
    def sample(self, context: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            (len(context), n_samples, 3), generator=generator, dtype=context.dtype
        ).to(context.device)
        return noise + context[:, None, :3]


def _schedule() -> StimulusSchedule:
    return StimulusSchedule(
        worm_id="OH16230:test",
        strain="OH16230",
        source_recording="test.mat",
        native_fps=4.0,
        analysis_fps=4.0,
        stimulus_names=("butanone", "pentanedione", "nacl"),
        event_intervals_seconds=((60.5, 70.5), (120.5, 130.5), (180.5, 190.5)),
        chemical_code_by_event=(2, 1, 3),
        chemical_name_by_event=("pentanedione", "butanone", "nacl"),
        resampling_provenance="none_native_grid",
    )


def test_evaluation_targets_are_state_matched_and_causal() -> None:
    targets = evaluation_targets(_schedule(), 900)
    assert len(targets) == 9
    first = {(state, event): target for state, event, target in targets if event == 0}
    onset = round(60.5 * 4)
    assert first[("quiet", 0)] == onset - 60 + 4
    assert first[("onset", 0)] == onset + 4
    assert first[("active", 0)] == onset + 24


def test_conditional_features_remove_the_candidate_source_value() -> None:
    rng = np.random.default_rng(3)
    trace = rng.normal(size=(120, 4)).astype(np.float32)
    projection = rng.normal(size=(4, 2)).astype(np.float32)
    baseline = conditional_features(trace, 100, 4, projection)
    changed = trace.copy()
    changed[96, 2] += 11.0
    candidate_changed = conditional_features(changed, 100, 4, projection)
    np.testing.assert_allclose(candidate_changed[2], baseline[2], atol=1e-5)
    assert not np.allclose(candidate_changed[1], baseline[1])


def test_low_high_history_changes_only_requested_source_lag() -> None:
    history = np.zeros((10, 4), dtype=np.float32)
    low = np.asarray([-1, -2, -3, -4], dtype=np.float32)
    high = -low
    sources = np.asarray([1, 3])
    low_history, high_history = build_low_high_histories(
        history, lag=4, low=low, high=high, source_indices=sources
    )
    assert low_history.shape == (2, 10, 4)
    assert low_history[0, -4, 1] == -2
    assert high_history[1, -4, 3] == 4
    assert np.count_nonzero(low_history) == 2
    assert np.count_nonzero(high_history) == 2


def test_common_random_numbers_make_zero_contrast_exact() -> None:
    adapter = GeneratorAdapter(
        model=_ToyModel(),
        checkpoint={
            "lag": 4,
            "neurons": ["a", "b", "c"],
            "model_config": {"residual_target": False},
        },
        device=torch.device("cpu"),
    )
    neural = np.zeros((2, 4, 3), dtype=np.float32)
    stimulus = np.zeros((2, 4, 1), dtype=np.float32)
    first = sample_many(adapter, neural, stimulus, n_particles=8, seed=17)
    second = sample_many(adapter, neural, stimulus, n_particles=8, seed=17)
    np.testing.assert_array_equal(first, second)
    metrics = distributional_effects(
        first,
        second,
        first,
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
    )
    np.testing.assert_array_equal(metrics["mean_shift"], 0)
    np.testing.assert_array_equal(metrics["log_sd_shift"], 0)
    np.testing.assert_array_equal(metrics["tail_probability_shift"], 0)
    np.testing.assert_array_equal(metrics["wasserstein1"], 0)
    np.testing.assert_array_equal(metrics["paired_transport_rms"], 0)


def test_lag32_is_outside_legacy_tcn_receptive_field() -> None:
    # Four causal kernel-3 blocks with dilations 1,2,4,8 see 1+2*(1+2+4+8)=31 frames.
    assert 1 + 2 * sum((1, 2, 4, 8)) == 31
    assert 32 > 31


def test_targeted_smc_weights_are_normalized_and_shift_source_only() -> None:
    weights = normalized_weights(np.asarray([-1000.0, -999.0, -998.0]))
    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights.sum(), 1.0)
    observed = np.zeros((32, 4), dtype=np.float32)
    innovation = np.zeros((32, 4), dtype=np.float32)
    observed_particles, latent_particles, diagnostics = smc_initial_histories(
        observed_history=observed,
        innovation_history=innovation,
        source=2,
        target_low=-0.5,
        target_high=0.5,
        proposal_sd=0.75,
        clip_low=-2.0,
        clip_high=2.0,
        seed=11,
        n_particles=128,
    )
    assert observed_particles.shape == (3, 128, 32, 4)
    np.testing.assert_array_equal(observed_particles, 0)
    changed = np.argwhere(latent_particles != 0)
    assert np.all(changed[:, 2] == 0)  # lag 32 is the oldest retained coordinate
    assert np.all(changed[:, 3] == 2)
    assert diagnostics["ess_low"] > 20
    assert diagnostics["ess_high"] > 20
    assert diagnostics["achieved_gap"] > 0


def test_bentley_null_preserves_each_source_outdegree() -> None:
    labels = np.zeros((5, 5), dtype=np.int8)
    labels[[1, 3], 0] = 1
    labels[[0, 4], 2] = 1
    eligible = labels.any(axis=0)
    permuted = permute_within_source(labels, eligible, np.random.default_rng(7))
    np.testing.assert_array_equal(permuted.sum(axis=0), labels.sum(axis=0))
    np.testing.assert_array_equal(np.diag(permuted), 0)


def test_source_bootstrap_uses_target_by_source_orientation() -> None:
    labels = np.zeros((4, 4), dtype=np.int8)
    labels[1, 0] = 1
    labels[3, 2] = 1
    score = np.zeros((4, 4), dtype=np.float32)
    score[1, 0] = 3.0
    score[3, 2] = 2.0
    sources = np.asarray([0, 2])
    assert source_bootstrap_auroc(score, labels, sources, sources) == 1.0
