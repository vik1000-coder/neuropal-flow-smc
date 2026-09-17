r"""Synthetic self-gain systems — ground truth for validating the self-excitability readout.

We generate a multi-neuron system in which a KNOWN subset of neurons has slow **self-gain**
(their own recent activity gates their own future conditional variance) and the rest do not,
then push it through the *real* calcium → OASIS-deconvolution → standardization pipeline the
worm data goes through, and ask whether the exposed self-gain diagonal recovers the truth.

Generative model (per neuron j, per worm):
    s_j(t)   = alpha*s_j(t-1) + (1-alpha)*x_j(t-1)          # slow causal filter of OWN past
    mu_j(t)  = phi_j*x_j(t-1) + load_j*g(t)                 # AR mean + shared global-state drive
    logsd_j(t) = base_j + beta_j * s_j(t)                   # SELF-GAIN: own slow state -> own vol
    x_j(t)   = mu_j(t) + exp(logsd_j(t)) * eps              # eps ~ N(0,1)
where g(t) is a per-worm AR(1) global mode (the whole-brain-state confound), and beta_j = beta
for the modulated subset, 0 otherwise. The conditional MEAN carries AR + global structure (so a
mean method is not blind), but the injected self-gain lives purely in the conditional variance.

Realism: activity is made non-negative (x + baseline), convolved with an AR(1) calcium kernel
(gamma), corrupted with measurement noise, then run through the repo's own
``deconvolve_matrix`` (OASIS AR(1)) and scale-preserving standardization — identical to the real
data path. ``calcium=False`` gives the noiseless upper bound. ``confound=True`` gives the
UNMODULATED neurons boring differences (higher variance, stronger AR, slower calcium) that are
NOT self-gain, to check the readout does not mistake them for modulation.
"""
from __future__ import annotations

import numpy as np

from sid_elegans.estimator import fit_distributional_connectome


def simulate_worms(N=80, W=6, T=1000, frac_mod=0.5, beta=0.8, tau_slow=20.0,
                   phi=0.3, global_strength=0.6, gamma=0.86, obs_noise=0.3,
                   baseline=6.0, calcium=True, confound=False, seed=0,
                   label_noise=0.0, beta_het=0.0):
    """Return ``(X_list, mod_mask, beta_vec)`` — fit-ready standardized per-worm ``[T,N]``.

    ``X_list`` has passed through the calcium+deconv pipeline when ``calcium=True``.
    ``mod_mask`` [N] bool = the ANNOTATION the test uses (the "receptor-expressing" label);
    ``beta_vec`` [N] = the TRUE injected self-gain strength.

    ``label_noise``: fraction of the annotation that is WRONG (a truly-modulated neuron
    annotated as control, or vice versa) — models imperfect receptor annotation, which dilutes
    the label-based group test exactly as in the real data. ``beta_het``: lognormal spread of
    the true self-gain across modulated neurons (biological heterogeneity).
    """
    rng = np.random.default_rng(seed)
    mod_mask = rng.random(N) < frac_mod                     # the annotation the test uses
    flip = rng.random(N) < label_noise                      # imperfect annotation
    true_mod = np.where(flip, ~mod_mask, mod_mask)          # the TRUE modulation status
    het = np.exp(beta_het * rng.standard_normal(N)) if beta_het > 0 else np.ones(N)
    beta_vec = np.where(true_mod, beta * het, 0.0)
    phi_j = np.full(N, phi)
    base_j = np.zeros(N)
    gamma_j = np.full(N, gamma)
    load = rng.uniform(0.2, 0.8, N) * global_strength
    # confound: give the modulated-LABELLED neurons a BORING property (not self-gain), so a clean
    # readout should NOT separate them. Accepts True/'all' or one of 'var','ar','calcium'.
    if confound:
        cset = {"var", "ar", "calcium"} if confound in (True, "all") else {confound}
        if "var" in cset:
            base_j[mod_mask] += 0.6           # higher marginal variance
        if "ar" in cset:
            phi_j[mod_mask] = 0.65            # stronger mean autocorrelation
        if "calcium" in cset:
            gamma_j[mod_mask] = 0.93          # slower calcium decay
    alpha = np.exp(-1.0 / tau_slow)

    X_list = []
    for w in range(W):
        rw = np.random.default_rng(seed * 100003 + w)
        g = np.zeros(T)
        for t in range(1, T):
            g[t] = 0.9 * g[t - 1] + 0.5 * rw.standard_normal()
        g = (g - g.mean()) / (g.std() + 1e-9)
        x = np.zeros((T, N)); s = np.zeros((T, N))
        for t in range(1, T):
            s[t] = alpha * s[t - 1] + (1 - alpha) * x[t - 1]
            mu = phi_j * x[t - 1] + load * g[t]
            # tanh-bounded self-gain: own slow state gates own log-sd, but bounded so the
            # variance-feedback loop cannot run away (large beta stays stable & recoverable).
            logsd = base_j + beta_vec * np.tanh(s[t])
            x[t] = mu + np.exp(logsd) * rw.standard_normal(N)
        if calcium:
            a = np.maximum(x + baseline, 0.0)             # non-negative activity
            y = np.zeros((T, N))
            for t in range(1, T):
                y[t] = gamma_j * y[t - 1] + a[t]           # AR(1) calcium
            y = y + obs_noise * y.std(0, keepdims=True) * rw.standard_normal((T, N))
            from sid_elegans.deconv import deconvolve_matrix
            xhat = deconvolve_matrix(y, fps=4.0)
        else:
            xhat = x
        X_list.append(xhat.astype(float))

    allX = np.concatenate(X_list, 0)
    mu_ = allX.mean(0); sd_ = float(allX.std()) + 1e-9
    X_list = [(xx - mu_) / sd_ for xx in X_list]
    return X_list, mod_mask, beta_vec


SLOW_LAGS = [8, 10, 15, 20, 30]      # slow band (2-7.5 s at 4 Hz)
FAST_LAGS = [1, 2, 3]


def self_gain_vector(X_list, names, lags=SLOW_LAGS):
    """Mean |self-gain| over ``lags`` (the exposed estimator diagonal)."""
    acc = np.zeros(len(names)); n = 0
    for L in lags:
        res = fit_distributional_connectome(X_list, names, lag=L)
        acc = acc + np.abs(res.diagonal["gain"]); n += 1
    return acc / max(n, 1)


def recovery_metrics(X_list, mod_mask, beta_vec, lags=SLOW_LAGS):
    """Recovery + separation of the injected self-gain from ``X_list``.

    Returns dict: spearman(recovered, beta), AUROC(recovered ranks modulated), and the
    modulated-minus-unmodulated mean self-gain difference (the analog of the real
    expressing-vs-control statistic).
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    N = len(mod_mask)
    names = [f"N{i}" for i in range(N)]
    sg = self_gain_vector(X_list, names, lags)
    rec = float(spearmanr(sg, beta_vec).correlation)
    auc = (float(roc_auc_score(mod_mask.astype(int), sg))
           if 0 < mod_mask.sum() < N else float("nan"))
    diff = float(sg[mod_mask].mean() - sg[~mod_mask].mean())
    return {"spearman_recovery": rec, "separation_auroc": auc,
            "mod_minus_unmod": diff, "self_gain": sg}
