from __future__ import annotations

import numpy as np
import copy
import pytest
import pandas as pd
from scipy.stats import multivariate_normal, multivariate_t
import torch

from conditional_neural_benchmark.models import build_encoded_model
from conditional_neural_benchmark.distribution_structure_models import ConditionalElliptical
from conditional_neural_benchmark.distribution_structure_scoring import (
    marginal_invariance, score_samples, shuffle_coordinates,
)
from conditional_neural_benchmark.distribution_structure_analysis import (
    collapse_repetitions, exact_sign_flip_p, holm, paired_interval,
)


def test_shuffle_preserves_every_history_neuron_multiset_and_marginal_score():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(7, 33, 5)) + np.arange(7)[:, None, None] * 100
    original = x.copy()
    shuffled = shuffle_coordinates(x, 27)
    np.testing.assert_array_equal(np.sort(x, axis=1), np.sort(shuffled, axis=1))
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(shuffled, shuffle_coordinates(x, 27))
    y = rng.normal(size=(7, 5)) + np.arange(7)[:, None] * 100
    a, b = score_samples(x, y, np.ones(5)), score_samples(shuffled, y, np.ones(5))
    assert marginal_invariance(a, b) < 1e-10
    assert not np.array_equal(x, shuffled)


def test_common_permutation_preserves_joint_scores_exactly():
    rng = np.random.default_rng(19)
    x, y = rng.normal(size=(6, 25, 4)), rng.normal(size=(6, 4))
    permuted = x[:, rng.permutation(25)]
    for key, value in score_samples(x, y).items():
        np.testing.assert_allclose(value, score_samples(permuted, y)[key], atol=1e-12)


def test_shuffle_removes_cross_neuron_signal_and_proper_scores_detect_it():
    rng = np.random.default_rng(37)
    z = rng.normal(size=(600, 128, 1))
    x = np.concatenate([z, 0.9 * z + 0.2 * rng.normal(size=z.shape)], axis=2)
    truth = rng.normal(size=(600, 1))
    y = np.concatenate([truth, 0.9 * truth + 0.2 * rng.normal(size=truth.shape)], axis=1)
    shuffled = shuffle_coordinates(x, 87)
    assert abs(np.corrcoef(shuffled.reshape(-1, 2).T)[0, 1]) < 0.04
    a, b = score_samples(x, y), score_samples(shuffled, y)
    assert (b["energy"] - a["energy"]).mean() > 0.025
    assert (b["variogram"] - a["variogram"]).mean() > 0.20


def test_independent_gaussian_negative_control_has_no_material_shuffle_effect():
    rng = np.random.default_rng(43)
    x, y = rng.normal(size=(1000, 64, 3)), rng.normal(size=(1000, 3))
    a = score_samples(x, y)
    differences = [score_samples(shuffle_coordinates(x, seed), y)["energy"] - a["energy"]
                   for seed in (71, 89, 113)]
    assert abs(np.mean(differences)) < 0.006


def test_energy_and_crps_match_brute_force_u_statistics():
    x = np.asarray([[[1., -2.], [2., 4.], [-1., 3.]]])
    y = np.asarray([[0.5, 1.]])
    pair = [x[0, i] - x[0, j] for i in range(3) for j in range(3) if i != j]
    expected_energy = np.linalg.norm(x[0] - y[0], axis=1).mean() - 0.5 * np.linalg.norm(pair, axis=1).mean()
    expected_crps = np.abs(x[0] - y[0]).mean() - 0.5 * np.abs(pair).mean()
    result = score_samples(x, y)
    assert result["energy"][0] == pytest.approx(expected_energy)
    assert result["crps"][0] == pytest.approx(expected_crps)


def test_next_state_offset_changes_only_absolute_variogram():
    rng = np.random.default_rng(103)
    x, y = rng.normal(size=(5, 32, 4)), rng.normal(size=(5, 4))
    offset = rng.normal(size=(5, 4)) * 7
    before = score_samples(x, y, np.ones(4))
    after = score_samples(x, y, np.ones(4), variogram_offset=offset)
    direct = score_samples(x + offset[:, None], y + offset)
    np.testing.assert_allclose(after["variogram"], direct["variogram"], atol=1e-12)
    for name in before:
        if name not in {"variogram", "variogram_naive"}:
            np.testing.assert_allclose(before[name], after[name], atol=1e-12)


def test_variogram_correction_matches_explicit_cross_sample_products():
    x = np.asarray([[[0., 1.], [2., -2.], [1., 3.], [4., 0.]]])
    y = np.asarray([[1., -1.]])
    q = np.sqrt(np.abs(x[0, :, 0] - x[0, :, 1]))
    observed = np.sqrt(2.)
    pair_product = np.mean([q[i] * q[j] for i in range(4) for j in range(4) if i != j])
    expected = observed ** 2 - 2 * observed * q.mean() + pair_product
    assert score_samples(x, y)["variogram"][0] == pytest.approx(expected)


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("student", [False, True])
def test_elliptical_density_agrees_with_scipy(rank, student):
    torch.manual_seed(11)
    model = ConditionalElliptical(3, 4, hidden=8, layers=1, rank=rank, student=student).double()
    h, y = torch.randn(3, 3, dtype=torch.double), torch.randn(3, 4, dtype=torch.double)
    mu, log_sd, factor = model.parameters_at(h)
    covariance = torch.diag_embed(torch.exp(2 * log_sd)) + factor @ factor.transpose(1, 2)
    actual = model.log_prob(y, h)
    for i in range(3):
        mean, cov = mu[i].detach().numpy(), covariance[i].detach().numpy()
        if student:
            nu = float(model.degrees_of_freedom.detach())
            expected = multivariate_t.logpdf(y[i].numpy(), loc=mean, shape=cov * (nu - 2) / nu, df=nu)
        else:
            expected = multivariate_normal.logpdf(y[i].numpy(), mean=mean, cov=cov)
        assert actual[i].item() == pytest.approx(expected, abs=1e-9)
    (-actual.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("student", [False, True])
def test_sampling_preserves_declared_covariance_and_seed(student):
    torch.manual_seed(31)
    model = ConditionalElliptical(2, 3, hidden=8, layers=1, rank=2, student=student).eval()
    h = torch.zeros(1, 2)
    with torch.no_grad():
        mean, log_sd, factor = model.parameters_at(h)
        truth = (torch.diag_embed(torch.exp(2 * log_sd)) + factor @ factor.transpose(1, 2))[0].numpy()
        samples = model.sample(h, 80000, seed=57)[0].numpy()
        first = model.sample(h, 15, seed=81)
        second = model.sample(h, 15, seed=81)
    np.testing.assert_allclose(np.cov(samples.T), truth, rtol=0.05, atol=0.025)
    np.testing.assert_allclose(samples.mean(0), mean[0].numpy(), atol=0.02)
    torch.testing.assert_close(first, second)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("student", [False, True])
def test_training_and_evaluation_devices_agree_on_elliptical_density(student):
    torch.manual_seed(37)
    cpu = ConditionalElliptical(3, 5, hidden=8, layers=1, rank=2, student=student).eval()
    mps = copy.deepcopy(cpu).to("mps")
    h, y = torch.randn(9, 3), torch.randn(9, 5)
    with torch.no_grad():
        a = cpu.log_prob(y, h)
        b = mps.log_prob(y.to("mps"), h.to("mps")).cpu()
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-5)


def test_matched_heads_start_with_identical_encoder_and_covariance_network():
    models = []
    for head in ("structure_gaussian", "structure_student_t"):
        torch.manual_seed(17)
        models.append(build_encoded_model(head_name=head, encoder_name="tcn", lag=8,
                      channels=5, dy=4, width=16, dropout=0.15,
                      head_params={"hidden": 16, "layers": 2, "rank": 2}))
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key])


def test_inference_counts_animals_after_averaging_nuisance_replicates():
    rows = []
    for worm, value in (("worm_a", 2.0), ("worm_b", 6.0)):
        for seed in (1, 2):
            for draw in (3, 4):
                rows.append(dict(samples=128, worm_id=worm, context="all", model="flow",
                                 model_seed=seed, sample_seed=draw, fold=0, histories=200,
                                 energy=value + (seed - 1.5)))
    collapsed = collapse_repetitions(pd.DataFrame(rows))
    assert len(collapsed) == 2
    np.testing.assert_allclose(collapsed.energy, [2.0, 6.0])
    np.testing.assert_array_equal(collapsed.repetitions, [4, 4])
    mean, low, high = paired_interval(collapsed.energy.to_numpy())
    assert mean == 4 and low == 2 and high == 6


def test_exact_sign_enumeration_and_holm_known_cases():
    assert exact_sign_flip_p(np.ones(17)) == 1 / 65536
    assert exact_sign_flip_p(np.zeros(17)) == 1.0
    assert exact_sign_flip_p(np.arange(1, 5)) == exact_sign_flip_p(-np.arange(1, 5))
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_particle_sensitivity_rejects_unmatched_histories():
    from conditional_neural_benchmark.distribution_structure_analysis import validate_sensitivity
    protocol = {"folds": list(range(5)), "seeds": [1701, 2903],
                "worm_ids": [f"worm_{i}" for i in range(17)]}
    metadata, rows = [], []
    for n in (128, 256):
        for seed in protocol["seeds"]:
            for fold in protocol["folds"]:
                metadata.append(dict(fold=fold, model_seed=seed, samples=n,
                                     model_id="flow", sample_seed=731, shuffle_repeats=4,
                                     history_key_sha256=f"histories_{fold}"))
            for worm in protocol["worm_ids"]:
                rows.append(dict(samples=n, model_seed=seed, context="all",
                                 model="flow", worm_id=worm, histories=64))
    frame = pd.DataFrame(rows)
    validate_sensitivity(frame, metadata, protocol)
    metadata[-1]["history_key_sha256"] = "different_histories"
    with pytest.raises(RuntimeError, match="different histories"):
        validate_sensitivity(frame, metadata, protocol)
