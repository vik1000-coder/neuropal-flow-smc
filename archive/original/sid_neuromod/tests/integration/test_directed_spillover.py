"""E6: directed spillover — predictive != structural directedness (Prop 16.4)."""
import numpy as np

from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher
from sid_neuromod.readouts.mean_gain_tail import center_readout_vector
from sid_neuromod.synthetic.directed_spillover import simulate_directed_spillover
from sid_neuromod.utils.arrays import safe_log_square


def _gain_kernel(X, target_idx, source_idx, L=6):
    xt = (X[:, target_idx] - X[:, target_idx].mean()) / X[:, target_idx].std()
    zs = safe_log_square((X[:, source_idx] - X[:, source_idx].mean()) / X[:, source_idx].std())
    zc = zs - zs.mean()
    T = len(xt) - 1
    cols = [np.ones(T)]
    for u in range(1, L + 1):
        lag = np.zeros(T); lag[u:] = zc[: T - u]; cols.append(lag)
    Psi = np.column_stack(cols); Y = xt[1: 1 + T]
    res = QuadraticScoreMatcher(ridge=1e-6).fit(Y, Psi)
    return center_readout_vector(res, Psi, list(range(1, L + 1)), "gain_log_variance")


def test_directed_spillover_predictive_nonzero_in_structural_zero_direction():
    """The structural-zero direction (1<-2) still has nonzero predictive gain kernel."""
    sp = simulate_directed_spillover(T=200000, seed=0)
    X = sp.x
    k_wire = _gain_kernel(X, target_idx=1, source_idx=0)   # 2 <- 1 (structural wire)
    k_zero = _gain_kernel(X, target_idx=0, source_idx=1)   # 1 <- 2 (structural zero)
    # both directions carry a positive, decaying gain kernel
    assert k_wire[0] > 0.02
    assert k_zero[0] > 0.01  # predictive nonzero despite no structural wire
    # kernels decay with lag
    assert k_wire[0] > k_wire[-1]
