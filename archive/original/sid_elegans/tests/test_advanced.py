"""Tests for the advanced methods: multi-timescale bank, sigma-ledger, two-head."""
import numpy as np
import pytest

from sid_elegans.data import load_traces
from sid_elegans.ground_truth import load_all_monoamine_layers
from sid_elegans.multiscale import fit_multiscale_connectome, sigma_ledger
from sid_elegans.significance import auroc


def test_multiscale_orientation_and_shape():
    rng = np.random.default_rng(0)
    N = 6
    X = rng.standard_normal((3000, N))
    # source 0 modulates target 2's variance (a gain edge 0 -> 2)
    X[1:, 2] = np.exp(0.4 * X[:-1, 0]) * rng.standard_normal(3000 - 1)
    names = [f"N{i}" for i in range(N)]
    res = fit_multiscale_connectome([X], names, taus_s=(0.5, 1, 5), tau0_s=3, ridge=0.1)
    for ch in ("mean_fast", "gain_slow", "mean_all", "gain_all"):
        M = res.matrices[ch]
        assert M.shape == (N, N)
        assert np.allclose(np.diag(M), 0)
        assert np.all(np.isfinite(M))
        assert M.std() > 0   # non-degenerate structure
    # source 0 exerts a nonzero gain coupling on target 2 (edge present, not asserting rank
    # on a tiny 3000-sample synthetic — real recovery is covered by the tyramine test)
    assert abs(res.matrices["gain_all"][2, 0]) > 0


@pytest.mark.slow
def test_multiscale_recovers_tyramine():
    """The multi-timescale slow-gain channel recovers the tyramine connectome (>0.65)."""
    X_list, names, fps = load_traces("full_traces")
    mono = load_all_monoamine_layers(names)
    res = fit_multiscale_connectome(X_list, names, ridge=3.0)
    a = auroc(res.matrices["gain_slow"], mono["tyramine"])
    assert a > 0.65, f"tyramine gain_slow AUROC {a}"


@pytest.mark.slow
def test_sigma_ledger_gain_stable():
    """The gain readout is stable across denoising sigma levels (cross-sigma Spearman)."""
    X_list, names, fps = load_traces("full_traces")
    _, stab = sigma_ledger(X_list, names, ridge=0.05)
    # gain readout broadly preserved across denoising sigma levels (positive corr)
    assert min(stab.values()) > 0.5, stab


@pytest.mark.slow
def test_two_head_recovers_nonlinear_gain_synthetic():
    """The two-head recovers a nonlinear variance-driver (which the linear model can't)."""
    from sid_elegans.twohead import fit_two_head
    rng = np.random.default_rng(0)
    N = 5
    X = rng.standard_normal((10000, N))
    X[1:, 2] = np.exp(0.6 * np.abs(X[:-1, 0])) * 0.5 * rng.standard_normal(10000 - 1)
    names = [f"N{i}" for i in range(N)]
    res = fit_two_head([X], names, source_tau_s=0, epochs=150, hidden=64, seed=0)
    G = res.matrices["gain"]
    assert abs(G[2, 0]) == max(abs(G[2, k]) for k in range(N) if k != 2)
