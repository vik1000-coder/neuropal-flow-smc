from __future__ import annotations

import numpy as np
import torch

from compatibility_neural_benchmark.core import (
    RepairedResponseConfig,
    _chunked_standardized_next,
    estimate_repaired_responses,
    normalized_log_weights,
    smc_repaired_responses,
    source_specific_anchor_costs,
    systematic_resample,
    systematic_resample_to_n,
)
from compatibility_neural_benchmark.evaluate import lagged_animal_signflip
from compatibility_neural_benchmark.progressive_smc import progressive_smc_repaired_responses
from compatibility_neural_benchmark.sbtg_baseline import SBTGStructuredVolatilityEstimator


def test_log_weights_normalize_and_return_mean_mass():
    logw = np.log(np.asarray([0.2, 0.5, 0.8]))
    weights, log_alpha = normalized_log_weights(logw)
    np.testing.assert_allclose(weights.sum(), 1.0)
    np.testing.assert_allclose(np.exp(log_alpha), np.mean(np.exp(logw)))


def test_systematic_resampling_respects_degenerate_weight():
    indices = systematic_resample(
        np.asarray([0.0, 0.0, 1.0, 0.0]), np.random.default_rng(3)
    )
    np.testing.assert_array_equal(indices, np.asarray([2, 2, 2, 2]))


def test_systematic_resampling_can_prune_a_branched_population():
    indices = systematic_resample_to_n(
        np.asarray([0.0, 0.2, 0.0, 0.3, 0.0, 0.5]),
        3,
        np.random.default_rng(5),
        offset=0.25,
    )
    assert indices.shape == (3,)
    assert set(indices).issubset({1, 3, 5})


def test_paired_half_sampling_uses_common_random_numbers():
    class NoiseAdapter:
        @staticmethod
        def sample_standardized_next(neural_history, stimulus_history, *, seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return torch.randn(
                (len(neural_history), neural_history.shape[-1]), generator=generator
            )

    history = torch.zeros((4, 5, 2, 3))
    stimulus = torch.zeros((4, 5, 2))
    draw = _chunked_standardized_next(
        NoiseAdapter(),
        history,
        stimulus,
        seed=7,
        chunk_size=6,
        paired_halves=True,
    )
    torch.testing.assert_close(draw[:2], draw[2:])


def test_source_is_excluded_only_during_source_window():
    prefix = np.zeros((2, 3, 2), dtype=np.float32)
    factual = np.zeros((3, 2), dtype=np.float32)
    prefix[:, -1, 0] = 4.0
    projection = np.eye(2, dtype=np.float32)
    cost = source_specific_anchor_costs(prefix, factual, projection, source_window_frames=1)
    assert np.allclose(cost[:, 0], 0.0)
    assert np.all(cost[:, 1] > 0.0)


def test_lagged_source_is_excluded_at_declared_history_position():
    prefix = np.zeros((2, 4, 2), dtype=np.float32)
    factual = np.zeros((4, 2), dtype=np.float32)
    prefix[:, 1, 0] = 4.0
    projection = np.eye(2, dtype=np.float32)
    cost = source_specific_anchor_costs(
        prefix,
        factual,
        projection,
        source_window_frames=1,
        source_lag_frames=2,
    )
    assert np.allclose(cost[:, 0], 0.0)
    assert np.all(cost[:, 1] > 0.0)


def test_lagged_source_window_must_fit_inside_repair_prefix():
    config = RepairedResponseConfig(
        repair_frames=4,
        source_window_frames=2,
        source_lag_frames=3,
    )
    try:
        config.validate()
    except ValueError as error:
        assert "lagged source window" in str(error)
    else:
        raise AssertionError("invalid lagged source window was accepted")


def test_equal_source_targets_give_exact_zero_response():
    rng = np.random.default_rng(4)
    prefix = rng.normal(size=(256, 4, 3)).astype(np.float32)
    future = rng.normal(size=(256, 4, 3)).astype(np.float32)
    factual = np.zeros((4, 3), dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    config = RepairedResponseConfig(
        history_frames=2,
        repair_frames=4,
        source_window_frames=2,
        horizon_frames=(1, 2, 4),
        n_particles=256,
    )
    result = estimate_repaired_responses(
        prefix,
        future,
        factual,
        np.eye(3, dtype=np.float32),
        target,
        target,
        np.ones(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        config,
    )
    for key, value in result.items():
        if key.startswith("response_"):
            np.testing.assert_allclose(value, 0.0, atol=1e-7)


def test_repaired_response_recovers_supported_positive_association():
    rng = np.random.default_rng(9)
    n = 5000
    source = rng.normal(size=n)
    prefix = np.zeros((n, 1, 2), dtype=np.float32)
    prefix[:, 0, 0] = source
    future = np.zeros((n, 1, 2), dtype=np.float32)
    future[:, 0, 1] = source + rng.normal(scale=0.2, size=n)
    config = RepairedResponseConfig(
        history_frames=1,
        repair_frames=1,
        source_window_frames=1,
        horizon_frames=(1,),
        n_particles=n,
        anchor_lambda=0.0,
        epsilon_iqr_fraction=0.25,
        min_ess=10,
    )
    result = estimate_repaired_responses(
        prefix,
        future,
        np.zeros((1, 2), dtype=np.float32),
        np.eye(2, dtype=np.float32),
        np.asarray([-0.7, 0.0], dtype=np.float32),
        np.asarray([0.7, 0.0], dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        config,
    )
    assert result["response_endpoint_mean"][0, 0, 1] > 1.0


def test_bootstrap_smc_recovers_supported_self_response():
    class LinearAdapter:
        lag = 1
        device = torch.device("cpu")

        @staticmethod
        def sample_standardized_next(neural_history, stimulus_history, *, seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(
                (len(neural_history), neural_history.shape[-1]), generator=generator
            )
            return 0.70 * neural_history[:, -1] + 0.45 * noise

    config = RepairedResponseConfig(
        history_frames=1,
        repair_frames=1,
        source_window_frames=1,
        horizon_frames=(1,),
        n_particles=512,
        anchor_lambda=0.0,
        epsilon_iqr_fraction=0.40,
        min_ess=4,
        max_normalized_weight=1.0,
        resample_ess_fraction=0.50,
        sampling_chunk_size=128,
    )
    result = smc_repaired_responses(
        LinearAdapter(),
        np.zeros((6, 2), dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        cut_time=3,
        projection=np.eye(2, dtype=np.float32),
        source_low=np.full(2, -0.6, dtype=np.float32),
        source_high=np.full(2, 0.6, dtype=np.float32),
        source_iqr=np.ones(2, dtype=np.float32),
        thresholds=np.zeros(2, dtype=np.float32),
        config=config,
        seed=19,
    )
    response = result["response_endpoint_mean"]
    assert response.shape == (2, 1, 2)
    assert response[0, 0, 0] > 0.25
    assert response[1, 0, 1] > 0.25
    assert np.all(result["diagnostic_step_resampled_low"][:, -1] == 1)


def test_bootstrap_smc_carries_early_clamp_to_temporal_cut():
    class LinearAdapter:
        lag = 1
        device = torch.device("cpu")

        @staticmethod
        def sample_standardized_next(neural_history, stimulus_history, *, seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(
                (len(neural_history), neural_history.shape[-1]), generator=generator
            )
            return 0.75 * neural_history[:, -1] + 0.35 * noise

    config = RepairedResponseConfig(
        history_frames=1,
        repair_frames=3,
        source_window_frames=1,
        source_lag_frames=1,
        horizon_frames=(1,),
        n_particles=768,
        anchor_lambda=0.0,
        epsilon_iqr_fraction=0.40,
        min_ess=4,
        max_normalized_weight=1.0,
        resample_ess_fraction=0.50,
        sampling_chunk_size=128,
    )
    result = smc_repaired_responses(
        LinearAdapter(),
        np.zeros((8, 2), dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        cut_time=5,
        projection=np.eye(2, dtype=np.float32),
        source_low=np.full(2, -0.6, dtype=np.float32),
        source_high=np.full(2, 0.6, dtype=np.float32),
        source_iqr=np.ones(2, dtype=np.float32),
        thresholds=np.zeros(2, dtype=np.float32),
        config=config,
        seed=23,
    )
    response = result["response_endpoint_mean"]
    assert response[0, 0, 0] > 0.15
    assert response[1, 0, 1] > 0.15
    assert np.isfinite(result["diagnostic_log10_clamp_increment_low"]).all()


def test_terminal_deferred_smc_does_not_resample_between_clamp_and_cut():
    class LinearAdapter:
        lag = 1
        device = torch.device("cpu")

        @staticmethod
        def sample_standardized_next(neural_history, stimulus_history, *, seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return 0.7 * neural_history[:, -1] + 0.5 * torch.randn(
                (len(neural_history), neural_history.shape[-1]), generator=generator
            )

    config = RepairedResponseConfig(
        history_frames=1,
        repair_frames=4,
        source_window_frames=1,
        source_lag_frames=2,
        horizon_frames=(1,),
        n_particles=128,
        anchor_lambda=0.0,
        epsilon_iqr_fraction=0.10,
        min_ess=1,
        max_normalized_weight=1.0,
        resample_ess_fraction=0.99,
        sampling_chunk_size=64,
    )
    result = smc_repaired_responses(
        LinearAdapter(),
        np.zeros((9, 2), dtype=np.float32),
        np.zeros(9, dtype=np.float32),
        cut_time=6,
        projection=np.eye(2, dtype=np.float32),
        source_low=np.full(2, -0.8, dtype=np.float32),
        source_high=np.full(2, 0.8, dtype=np.float32),
        source_iqr=np.ones(2, dtype=np.float32),
        thresholds=np.zeros(2, dtype=np.float32),
        config=config,
        seed=41,
        resampling_policy="terminal_deferred",
    )
    clamp_step = config.repair_frames - config.source_lag_frames - 1
    resampled = result["diagnostic_step_resampled_low"]
    assert not resampled[:, clamp_step:-1].any()
    assert resampled[:, -1].all()


def test_progressive_smc_places_bridge_at_declared_lagged_window():
    class NoiseAdapter:
        lag = 1
        device = torch.device("cpu")

        @staticmethod
        def sample_standardized_next(neural_history, stimulus_history, *, seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return torch.randn(
                (len(neural_history), neural_history.shape[-1]), generator=generator
            )

    config = RepairedResponseConfig(
        history_frames=1,
        repair_frames=4,
        source_window_frames=1,
        source_lag_frames=2,
        horizon_frames=(1,),
        n_particles=32,
        anchor_lambda=0.0,
        min_ess=1,
        max_normalized_weight=1.0,
        sampling_chunk_size=32,
    )
    result = progressive_smc_repaired_responses(
        NoiseAdapter(),
        np.zeros((9, 2), dtype=np.float32),
        np.zeros(9, dtype=np.float32),
        cut_time=6,
        projection=np.eye(2, dtype=np.float32),
        source_low=np.full(2, -0.3, dtype=np.float32),
        source_high=np.full(2, 0.3, dtype=np.float32),
        source_iqr=np.ones(2, dtype=np.float32),
        thresholds=np.zeros(2, dtype=np.float32),
        config=config,
        seed=43,
        branch_factor=2,
        future_branch_factor=1,
    )
    beta = result["diagnostic_step_beta_low"]
    np.testing.assert_allclose(beta[:, :1], 0.0)
    np.testing.assert_allclose(beta[:, 1:], 1.0)


def test_progressive_smc_recovers_supported_self_response_and_exact_terminal_bridge():
    class LinearAdapter:
        lag = 1
        device = torch.device("cpu")

        @staticmethod
        def sample_standardized_next(neural_history, stimulus_history, *, seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(
                (len(neural_history), neural_history.shape[-1]), generator=generator
            )
            return 0.70 * neural_history[:, -1] + 0.45 * noise

    config = RepairedResponseConfig(
        history_frames=1,
        repair_frames=2,
        source_window_frames=2,
        horizon_frames=(1,),
        n_particles=256,
        anchor_lambda=0.0,
        epsilon_iqr_fraction=0.40,
        min_ess=4,
        max_normalized_weight=1.0,
        sampling_chunk_size=128,
    )
    result = progressive_smc_repaired_responses(
        LinearAdapter(),
        np.zeros((7, 2), dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        cut_time=4,
        projection=np.eye(2, dtype=np.float32),
        source_low=np.full(2, -0.6, dtype=np.float32),
        source_high=np.full(2, 0.6, dtype=np.float32),
        source_iqr=np.ones(2, dtype=np.float32),
        thresholds=np.zeros(2, dtype=np.float32),
        config=config,
        seed=29,
        branch_factor=2,
        future_branch_factor=2,
    )
    response = result["response_endpoint_mean"]
    assert response.shape == (2, 1, 2)
    assert response[0, 0, 0] > 0.20
    assert response[1, 0, 1] > 0.20
    np.testing.assert_allclose(result["diagnostic_step_beta_low"][:, -1], 1.0)
    assert np.all(result["diagnostic_distinct_ancestors_low"] > 1)


def test_progressive_smc_equal_targets_give_exact_zero_response():
    class NoiseAdapter:
        lag = 1
        device = torch.device("cpu")

        @staticmethod
        def sample_standardized_next(neural_history, stimulus_history, *, seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return torch.randn(
                (len(neural_history), neural_history.shape[-1]), generator=generator
            )

    config = RepairedResponseConfig(
        history_frames=1,
        repair_frames=1,
        source_window_frames=1,
        horizon_frames=(1,),
        n_particles=64,
        anchor_lambda=0.0,
        min_ess=2,
        max_normalized_weight=1.0,
        sampling_chunk_size=32,
    )
    target = np.zeros(2, dtype=np.float32)
    result = progressive_smc_repaired_responses(
        NoiseAdapter(),
        np.zeros((5, 2), dtype=np.float32),
        np.zeros((5, 3), dtype=np.float32),
        cut_time=2,
        projection=np.eye(2, dtype=np.float32),
        source_low=target,
        source_high=target,
        source_iqr=np.ones(2, dtype=np.float32),
        thresholds=np.zeros(2, dtype=np.float32),
        config=config,
        seed=31,
    )
    for key, value in result.items():
        if key.startswith("response_"):
            np.testing.assert_allclose(value, 0.0, atol=1e-7)


def test_lagged_signflip_uses_one_offdiagonal_test_family():
    rng = np.random.default_rng(14)
    effects = rng.normal(scale=0.02, size=(12, 2, 3, 3))
    effects[:, 1, 2, 0] += 1.0
    mean, se, p_value, q_value, mask = lagged_animal_signflip(
        effects, n_perm=1023, seed=17
    )
    assert mean.shape == se.shape == p_value.shape == q_value.shape == (2, 3, 3)
    assert mask.sum() == 12
    assert not mask[:, np.arange(3), np.arange(3)].any()
    assert q_value[1, 2, 0] <= 0.10


def test_sbtg_inner_crossfit_keeps_each_worm_in_one_fold():
    estimator = SBTGStructuredVolatilityEstimator(
        inference_mode="cross_fit",
        n_folds=2,
        cross_fit_group_by_segment=True,
        random_state=11,
        verbose=False,
    )
    worm_ids = np.repeat(np.arange(7), [5, 7, 4, 6, 8, 3, 9])
    folds = estimator._create_fold_assignments(
        worm_ids, len(worm_ids), np.random.default_rng(11)
    )
    assert set(folds) == {0, 1}
    for worm in np.unique(worm_ids):
        assert len(np.unique(folds[worm_ids == worm])) == 1
