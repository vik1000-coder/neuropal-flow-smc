r"""Score-identified distributional functional connectome for C. elegans.

For each target neuron ``j`` and lag ``L`` we fit sid_neuromod's closed-form
quadratic-T conditional-score model

    Y_j(t) = x_j(t+L)            (target_mode="next")   or
    Y_j(t) = x_j(t+L) - x_j(t)   (target_mode="delta")

on features ``psi(t) = [1, x_0(t), ..., x_{N-1}(t)]`` (all neurons' current activity,
including the target's own — a multivariate/partial conditioning). The fitted conditional
law ``rho(Y_j | psi)`` is Gaussian ``N(mu_j(t), v_j(t))`` with mean and (log-)variance
*linear in the sources*, so the directed distributional couplings are read out at the
centered history:

    D_mean[j, i]  = d E[Y_j] / d x_i        (VAR-comparable predictive drive)
    D_gain[j, i]  = d log Var[Y_j] / d x_i  (gain / excitability modulation)
    D_tail_hi[j,i]= d P(Y_j > q_hi) / d x_i (burst/threshold modulation)
    D_tail_lo[j,i]= d P(Y_j < q_lo) / d x_i (suppression modulation)

All matrices use the ``[post, pre] = [target, source]`` convention, matching the Cook /
monoamine ground truth. Windows are built per worm (no cross-worm leakage) and pooled;
the target uses the future value and features the present, so the readout is strictly
causal. This is the disentangled conditional head of the two-head design realised in
closed form (no optimizer confounds, per the theory's pilot-experiment protocol).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from sid_neuromod.features.filter_bank import exp_filter_bank
from sid_neuromod.models.quadratic_score import QuadraticScoreMatcher

CHANNELS = ("mean", "gain", "tail_hi", "tail_lo")


def _filter_sources(X, tau_frames, fps=4.0):
    """Causal exponential filter of every neuron column at timescale ``tau_frames``.

    Uses sid_neuromod's exact zero-order-hold filter (per worm, strictly causal). The
    filtered feature is the slow integrated drive that neuromodulation exerts.
    """
    T = X.shape[0]
    t = np.arange(T, dtype=float) / fps
    Xf = np.nan_to_num(X, nan=0.0)
    tau_s = tau_frames / fps
    F = exp_filter_bank(Xf, t, [tau_s])[:, :, 0]   # [T, N]
    return F


@dataclass
class ConnectomeResult:
    lag: int
    neuron_names: list
    matrices: dict          # channel -> [N, N] coupling matrix ([post, pre])
    invalid_fraction: np.ndarray  # per-target invalid-variance fraction
    heldout_nll: np.ndarray       # per-target held-out NLL (diagnostic)
    diagonal: dict | None = None  # channel -> [N] self-coupling d./dx_j at target j (pre-zeroing)


def _build_target_design(X_list, j, lag, target_mode, source_tau=None, fps=4.0):
    """Pooled (Y, Psi_sources) across worms for target ``j`` at ``lag``.

    Returns ``(Y, Xsrc)`` where ``Xsrc[t]`` = the source features at the feature time and
    ``Y[t]`` the target's future value/increment. If ``source_tau`` (in frames) is given,
    source features are the causal exponential filter of each neuron at that timescale
    (the slow integrated drive); otherwise instantaneous activity. NaNs in sources are set
    to 0 (the standardized mean, matching the repo); rows with a NaN target are dropped.
    """
    Ys, Xs = [], []
    for X in X_list:
        T = X.shape[0]
        if T <= lag:
            continue
        src_full = _filter_sources(X, source_tau, fps) if source_tau else X
        x_now = src_full[:T - lag]           # source features at t
        x_fut = X[lag:]                      # x(t+lag) (raw target)
        x_now_raw = X[:T - lag]
        if target_mode == "next":
            y = x_fut[:, j]
        elif target_mode == "delta":
            y = x_fut[:, j] - x_now_raw[:, j]
        else:
            raise ValueError("target_mode must be 'next' or 'delta'")
        Ys.append(y)
        Xs.append(x_now)
    Y = np.concatenate(Ys)
    Xsrc = np.concatenate(Xs, axis=0)
    valid = np.isfinite(Y)
    Y = Y[valid]
    Xsrc = np.nan_to_num(Xsrc[valid], nan=0.0)
    return Y, Xsrc


def _center_readouts(res, q_hi, q_lo):
    r"""Vectorized center-history readouts for all sources at once.

    At the centered history the source features are 0 and the intercept is 1, so
    ``eta1 = theta1[0]``, ``eta2 = theta2[0]``, ``v_eff = -1/(2 eta2)``, ``v = v_eff``.
    Returns dict channel -> length-P vector over feature columns (col 0 = intercept).
    """
    t1 = res.theta1
    t2 = res.theta2
    eta1 = t1[0]
    eta2 = min(t2[0], -res.eta2_min)
    v_eff = -1.0 / (2.0 * eta2)
    v = max(v_eff - res.sigma ** 2, res.var_min)
    mu = eta1 * v_eff

    d_mean = v_eff * t1 + 2.0 * eta1 * v_eff ** 2 * t2
    d_var = 2.0 * v_eff ** 2 * t2
    d_gain = d_var / v
    sv = np.sqrt(v)
    a_hi = (q_hi - mu) / sv
    a_lo = (q_lo - mu) / sv
    d_tail_hi = norm.pdf(a_hi) * (d_mean / sv + (q_hi - mu) / (2 * v ** 1.5) * d_var)
    d_tail_lo = norm.pdf(a_lo) * (-d_mean / sv - (q_lo - mu) / (2 * v ** 1.5) * d_var)
    return {"mean": d_mean, "gain": d_gain, "tail_hi": d_tail_hi, "tail_lo": d_tail_lo}


def fit_distributional_connectome(
    X_list,
    neuron_names,
    lag: int,
    target_mode: str = "next",
    ridge: float = 1e-2,
    heldout_frac: float = 0.2,
    source_tau: float | None = None,
    fps: float = 4.0,
    seed: int = 0,
    sigma_frac: float = 0.0,
) -> ConnectomeResult:
    """Estimate the directed distributional connectome at a single lag.

    If ``source_tau`` (in frames) is given, sources enter as their causal exponential
    filter at that timescale (slow integrated drive) rather than instantaneous activity.
    Returns a :class:`ConnectomeResult` whose ``matrices[channel][i, j]`` is the coupling
    from source ``j`` to target ``i`` (``[post, pre]``).
    """
    N = len(neuron_names)
    mats = {c: np.zeros((N, N)) for c in CHANNELS}
    diag = {c: np.zeros(N) for c in CHANNELS}   # self-coupling (self-gain/excitability), pre-zero
    invalid = np.zeros(N)
    nll = np.full(N, np.nan)

    for j in range(N):
        Y, Xsrc = _build_target_design(X_list, j, lag, target_mode,
                                       source_tau=source_tau, fps=fps)
        if len(Y) < 3 * N:
            continue
        # design: intercept + centered sources (center on the pooled data -> readout
        # "at the typical history")
        src_mean = Xsrc.mean(axis=0, keepdims=True)
        Psi = np.concatenate([np.ones((len(Y), 1)), Xsrc - src_mean], axis=1)

        # simple contiguous held-out split for an honest NLL diagnostic
        ntr = int((1 - heldout_frac) * len(Y))
        # sigma_frac=0 -> Hyvarinen score matching; sigma_frac>0 -> denoising score
        # matching at noise level sigma = sigma_frac * sd(Y) (readout subtracts sigma^2).
        sigma = float(sigma_frac) * (float(np.std(Y[:ntr])) or 1.0)
        model = QuadraticScoreMatcher(sigma=sigma, ridge=ridge, seed=seed)
        res = model.fit(Y[:ntr], Psi[:ntr])
        invalid[j] = res.invalid_variance_fraction
        try:
            nll[j] = model.nll(Y[ntr:], Psi[ntr:])
        except Exception:
            pass

        q_hi = float(np.nanquantile(Y[:ntr], 0.90))
        q_lo = float(np.nanquantile(Y[:ntr], 0.10))
        rd = _center_readouts(res, q_hi, q_lo)
        # feature col 0 = intercept; cols 1.. = sources -> [j, i] = coupling source i -> j
        for c in CHANNELS:
            mats[c][j, :] = rd[c][1:]
            diag[c][j] = rd[c][1 + j]   # capture self-coupling BEFORE it is zeroed below
        mats["mean"][j, j] = 0.0  # zero the diagonal (self) for all channels
        for c in CHANNELS:
            mats[c][j, j] = 0.0

    return ConnectomeResult(lag=lag, neuron_names=list(neuron_names), matrices=mats,
                            invalid_fraction=invalid, heldout_nll=nll, diagonal=diag)
