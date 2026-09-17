"""Smoke + correctness tests for the sid_elegans adapter."""
import numpy as np
import pytest

from sid_elegans.data import load_traces
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.evaluate import score_matrix
from sid_elegans.ground_truth import load_all_monoamine_layers, load_cook
from sid_elegans.run_eval import reindex_matrix


def test_estimator_orientation_post_pre():
    """A synthetic driver 0 -> 1 (0's past sets 1's future mean) must put the coupling
    at D_mean[target=1, source=0] (the [post, pre] convention), not the transpose."""
    rng = np.random.default_rng(0)
    N = 5
    X = rng.standard_normal((4000, N))
    # neuron 1's next value is driven by neuron 0's current value
    X[1:, 1] = 0.8 * X[:-1, 0] + 0.3 * rng.standard_normal(4000 - 1)
    names = [f"N{i}" for i in range(N)]
    res = fit_distributional_connectome([X], names, lag=1, target_mode="next", ridge=1e-3)
    D = res.matrices["mean"]
    assert abs(D[1, 0]) > abs(D[0, 1]), "coupling must be at [post=1, pre=0]"
    assert abs(D[1, 0]) > 0.3
    # off-diagonal, no self-coupling
    assert D[0, 0] == 0.0 and D[1, 1] == 0.0


def test_gain_channel_detects_variance_driver():
    """A source that sets the target's future VARIANCE (not mean) shows up in the gain
    channel but not the mean channel."""
    rng = np.random.default_rng(1)
    N = 4
    X = rng.standard_normal((8000, N))
    # neuron 2's next-step variance is modulated by neuron 0's current value (gain effect)
    scale = np.exp(0.5 * X[:-1, 0])
    X[1:, 2] = scale * rng.standard_normal(8000 - 1)
    names = [f"N{i}" for i in range(N)]
    res = fit_distributional_connectome([X], names, lag=1, target_mode="next", ridge=1e-3)
    assert abs(res.matrices["gain"][2, 0]) > abs(res.matrices["mean"][2, 0])
    assert abs(res.matrices["gain"][2, 0]) > 0.1


def test_reindex_matrix_roundtrip():
    M = np.arange(9.0).reshape(3, 3)
    names = ["A", "B", "C"]
    perm = ["C", "A", "B"]
    Mp = reindex_matrix(M, names, perm)
    # reindexing back recovers the original
    back = reindex_matrix(Mp, perm, names)
    assert np.allclose(back, M)


def test_score_matrix_matches_manual_auroc():
    """score_matrix AUROC equals sklearn on the off-diagonal |scores| vs (gt>0)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(2)
    n = 10
    gt = (rng.random((n, n)) < 0.3).astype(float)
    np.fill_diagonal(gt, 0)
    scores = gt * 2 + rng.standard_normal((n, n))  # correlated with gt
    s = score_matrix(scores, gt)
    mask = ~np.eye(n, dtype=bool)
    expected = roc_auc_score((gt[mask] > 0).astype(int), np.abs(scores[mask]))
    assert abs(s["auroc"] - expected) < 1e-9


def test_ground_truth_orientation():
    """The loaded Cook connectome must be [post,pre]: AIY->RIA sits at [RIA, AIY]."""
    _, names, _ = load_traces()
    cook = load_cook(names)
    idx = {n.upper(): i for i, n in enumerate(names)}
    if "AIY" in idx and "RIA" in idx:
        assert cook["chem"][idx["RIA"], idx["AIY"]] > cook["chem"][idx["AIY"], idx["RIA"]]
    # gap junctions symmetric
    assert np.allclose(cook["gap"], cook["gap"].T)


def test_monoamine_layers_nonempty_and_binary():
    _, names, _ = load_traces()
    mono = load_all_monoamine_layers(names)
    for nt in ["dopamine", "serotonin", "tyramine", "octopamine"]:
        A = mono[nt]
        assert set(np.unique(A)).issubset({0.0, 1.0})
        assert A.shape == (len(names), len(names))
    assert (mono["monoamine_all"] > 0).sum() >= (mono["dopamine"] > 0).sum()


@pytest.mark.smoke
def test_full_fit_smoke():
    """End-to-end fit on the real data at lag 1 produces finite, sane matrices."""
    X_list, names, fps = load_traces()
    assert len(X_list) >= 15 and len(names) == 80
    res = fit_distributional_connectome(X_list, names, lag=1, ridge=1e-2)
    for c in ("mean", "gain", "tail_hi", "tail_lo"):
        M = res.matrices[c]
        assert M.shape == (80, 80)
        assert np.all(np.isfinite(M))
        assert np.allclose(np.diag(M), 0)
    # sane invalid-variance fraction on real data
    assert np.nanmean(res.invalid_fraction) < 0.05
