"""Synthetic generators for covariance and partial-observation analyses.

1. generate_var_data_corr : VAR(2) with *correlated* innovations eps ~ N(0, Sigma).
   The core control uses eps ~ N(0, sigma^2 I). This generator adds a
   non-diagonal innovation-covariance regime for explicit stress testing.

   The rho=0 condition is the jointly stabilized isotropic-noise control.  It
   shares graph support and RNG graph draws with the core generator, but it
   is not an exact trajectory-level reproduction of the unstabilized generator.

2. hide_neurons : take a full multivariate series + true graph and return only an
   observed subset (columns) plus the observed-vs-observed submatrix of the truth.
   Supports the hidden-neuron / partial-observation experiment.

The sparse-matrix helper is shared with the core synthetic benchmark so graph
draws match when the seed and configuration match.
"""

from typing import Dict, List, Tuple

import numpy as np

from pipeline.SyntheticTestingUtils import _make_sparse_matrix


# ---------------------------------------------------------------------------
# 1. Correlated-innovation VAR(2)
# ---------------------------------------------------------------------------

def companion_spectral_radius(A1: np.ndarray, A2: np.ndarray) -> float:
    """Spectral radius of the VAR(2) companion matrix [[A1, A2], [I, 0]]."""
    n = A1.shape[0]
    comp = np.block([[A1, A2], [np.eye(n), np.zeros((n, n))]])
    return float(max(abs(np.linalg.eigvals(comp))))

def _structured_cov(n: int, sigma: float, rho: float, cov_type: str) -> np.ndarray:
    """Build an innovation covariance Sigma with off-diagonal structure.

    equicorr : Sigma = sigma^2 [ (1-rho) I + rho 11^T ]   (dense, PSD for rho in [0,1))
    ar1      : Sigma_ij = sigma^2 rho^|i-j|                (banded spatial correlation)
    """
    if cov_type == "equicorr":
        R = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    elif cov_type == "ar1":
        idx = np.arange(n)
        R = rho ** np.abs(idx[:, None] - idx[None, :])
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")
    return (sigma ** 2) * R


def stabilize_var(A1: np.ndarray, A2: np.ndarray, target: float = 0.9):
    """Rescale (A1, A2) so the VAR(2) is stationary, preserving support and sign.

    Scaling A1 -> s*A1 and A2 -> s^2*A2 scales every root of
    det(λ²I − λA1 − A2) by exactly s, so choosing s = target/rho puts the companion
    radius at `target`. The edge SUPPORT (the ground-truth graph) is unchanged.
    """
    rho = companion_spectral_radius(A1, A2)
    if rho <= target or rho == 0.0:
        return A1, A2, rho, rho
    s = target / rho
    return A1 * s, A2 * (s ** 2), rho, target


def generate_var_data_corr(
    n: int, T: int, m_stim: int, noise_level: str, seed: int,
    rho: float = 0.0, cov_type: str = "equicorr", stabilize: bool = True,
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
    """VAR(2) with correlated innovations.

    x_t = A1 x_{t-1} + A2 x_{t-2} + eps_t,   eps_t ~ N(0, Sigma(rho)).

    rho = 0 gives the jointly stabilized isotropic-noise control. It preserves
    the reference graph support but can differ from data generated without joint
    companion-system stabilization.

    Returns `(X_list, truth_dict)` with the core generator contract:
    X_list is a list of (T, n) arrays; truth_dict = {1: support(A1), 2: support(A2)}.
    """
    rng = np.random.default_rng(seed)
    sigma = 0.1 if noise_level == "low" else 0.5

    # Ground-truth graphs: drawn first, so they are identical across rho for a given seed.
    A1 = _make_sparse_matrix(n, sparsity=0.1, scale=0.8, rng=rng)
    A2 = _make_sparse_matrix(n, sparsity=0.1, scale=0.5, rng=rng)
    if stabilize:
        A1, A2, _, _ = stabilize_var(A1, A2, target=0.9)   # support (truth) unchanged

    Sigma = _structured_cov(n, sigma, rho, cov_type)
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))

    X_list = []
    for _ in range(m_stim):
        X = np.zeros((T, n), dtype=float)
        X[0] = rng.normal(scale=1.0, size=n)
        X[1] = rng.normal(scale=1.0, size=n)
        for t in range(2, T):
            eps = L @ rng.standard_normal(n)          # correlated innovation
            X[t] = A1 @ X[t - 1] + A2 @ X[t - 2] + eps
        X_list.append(X)

    truth_dict = {1: (np.abs(A1) > 1e-8), 2: (np.abs(A2) > 1e-8)}
    return X_list, truth_dict


def generate_var_data_stable(
    n: int, T: int, m_stim: int, noise_level: str, seed: int,
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
    """Stationary isotropic-noise VAR(2) generator.

    The lag matrices are jointly rescaled to a stationary companion radius
    while preserving the truth-support contract.
    """
    return generate_var_data_corr(
        n=n, T=T, m_stim=m_stim, noise_level=noise_level, seed=seed,
        rho=0.0, cov_type="equicorr", stabilize=True,
    )


def empirical_noise_correlation(X_list: List[np.ndarray]) -> float:
    """Mean absolute off-diagonal innovation correlation, estimated from VAR(2) residuals.

    Fits a quick least-squares VAR(2), then reports the mean |off-diagonal| of the
    residual correlation matrix — a scalar summary of how far the innovations are
    from diagonal. Used to place real/synthetic data on the rho axis.
    """
    resid = []
    for X in X_list:
        if X.shape[0] < 6:
            continue
        Y = X[2:]
        Z = np.hstack([X[1:-1], X[:-2], np.ones((X.shape[0] - 2, 1))])
        beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        resid.append(Y - Z @ beta)
    if not resid:
        return float("nan")
    R = np.corrcoef(np.vstack(resid).T)
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    return float(np.abs(R[off]).mean())


# ---------------------------------------------------------------------------
# 2. Hidden-neuron / partial observation
# ---------------------------------------------------------------------------

def hide_neurons(
    X_list: List[np.ndarray],
    truth_lag1: np.ndarray,
    frac_hidden: float,
    seed: int,
    hidden_indices: np.ndarray = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Return an observed subset of neurons (columns) and the observed-vs-observed truth.

    The full data are generated with all n neurons interacting; only a subset is
    revealed to the estimator. Edges among observed neurons are still present, but
    their activity is confounded by marginalized hidden neurons. This isolates a
    partial-observation sensitivity while holding scored edges fixed.

    Returns (X_list_obs, truth_obs, observed_idx).
    """
    n = X_list[0].shape[1]
    n_hidden = int(round(frac_hidden * n))
    if hidden_indices is None:
        rng = np.random.default_rng(seed + 10_000)
        hidden = (
            np.sort(rng.choice(n, size=n_hidden, replace=False))
            if n_hidden > 0
            else np.array([], dtype=int)
        )
    else:
        hidden = np.sort(np.asarray(hidden_indices, dtype=int))
        if hidden.size != n_hidden:
            raise ValueError(
                f"Expected {n_hidden} hidden indices for frac_hidden="
                f"{frac_hidden}, received {hidden.size}"
            )
        if np.unique(hidden).size != hidden.size or np.any((hidden < 0) | (hidden >= n)):
            raise ValueError("hidden_indices must be unique indices in [0, n)")
    observed = np.array([i for i in range(n) if i not in set(hidden.tolist())], dtype=int)

    X_list_obs = [X[:, observed] for X in X_list]
    truth_obs = truth_lag1[np.ix_(observed, observed)]
    return X_list_obs, truth_obs, observed
