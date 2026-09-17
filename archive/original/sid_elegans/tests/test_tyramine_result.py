"""Regression test pinning the headline finding: the slow-gain channel recovers the
tyramine neuromodulator connectome, above chance and above the mean channel."""
import numpy as np
import pytest

from sid_elegans.data import load_traces
from sid_elegans.estimator import fit_distributional_connectome
from sid_elegans.ground_truth import load_all_monoamine_layers
from sid_elegans.significance import auroc, perm_p


@pytest.mark.slow
def test_tyramine_slow_gain_beats_chance_and_mean():
    X_list, names, fps = load_traces("full_traces")   # non-imputed (cleaner)
    mono = load_all_monoamine_layers(names)
    res = fit_distributional_connectome(X_list, names, lag=1, ridge=1e-2,
                                        source_tau=30 * fps)  # 30 s slow filter
    gt = mono["tyramine"]
    a_g, p_g = perm_p(res.matrices["gain"], gt, n=1000)
    a_m = auroc(res.matrices["mean"], gt)
    # gain channel recovers tyramine well above chance and above the mean channel
    assert a_g > 0.65, f"tyramine gain AUROC {a_g}"
    assert p_g < 0.01, f"tyramine gain perm p {p_g}"
    assert a_g > a_m, f"gain {a_g} should beat mean {a_m}"


def test_tyramine_gain_rises_with_timescale():
    """The gain-channel tyramine AUROC increases with slower source timescales."""
    X_list, names, fps = load_traces("full_traces")
    mono = load_all_monoamine_layers(names)
    aurocs = []
    for tau in [2, 10, 30]:
        res = fit_distributional_connectome(X_list, names, lag=1, ridge=1e-2,
                                            source_tau=tau * fps)
        aurocs.append(auroc(res.matrices["gain"], mono["tyramine"]))
    # monotone-ish increase with timescale (slow neuromodulation)
    assert aurocs[-1] > aurocs[0] + 0.05, aurocs
