r"""Mixture-density conditional model (#3, the genuinely non-Gaussian estimator).

The two-head so far uses a Gaussian conditional; on "nonlinear complicated" data that is
too rigid. Here the conditional law of every target is a K-component Gaussian mixture with
input-dependent weights/means/variances,

    rho(y_j | h) = sum_k pi_{jk}(h) N(y_j; mu_{jk}(h), sigma^2_{jk}(h)),

fit by maximum likelihood (the natural objective for mixtures; MLE and score matching
estimate the same conditional law). This captures multimodality, skew and heavy tails that
a single Gaussian cannot. The distributional couplings are read out by autograd through the
mixture's conditional mean and (log-)variance:

    Var[y_j|h] = sum_k pi_k (sigma^2_k + mu_k^2) - (sum_k pi_k mu_k)^2,
    gain[j,i]  = < d log Var[y_j|h] / d h_i >,   mean[j,i] = < d E[y_j|h] / d h_i >.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_neuromod.features.filter_bank import exp_filter_bank


def _mlp(din, dout, hidden, layers):
    mods, d = [], din
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods += [nn.Linear(d, dout)]
    return nn.Sequential(*mods)


class MDN(nn.Module):
    """Per-target K-component Gaussian mixture conditional density over N targets."""

    def __init__(self, n, k=3, hidden=128, layers=2, logv_floor=-6.0, logv_ceil=4.0):
        super().__init__()
        self.n, self.k = n, k
        self.trunk = _mlp(n, hidden, hidden, layers)
        self.pi = nn.Linear(hidden, n * k)
        self.mu = nn.Linear(hidden, n * k)
        self.logv = nn.Linear(hidden, n * k)
        self.logv_floor, self.logv_ceil = logv_floor, logv_ceil

    def params(self, h):
        z = self.trunk(h)
        B = h.shape[0]
        pi = F.log_softmax(self.pi(z).view(B, self.n, self.k), dim=-1)
        mu = self.mu(z).view(B, self.n, self.k)
        logv = torch.clamp(self.logv(z).view(B, self.n, self.k), self.logv_floor, self.logv_ceil)
        return pi, mu, logv

    def nll(self, h, y):
        """Mean NLL of ``y`` (B,N) under the per-target mixtures."""
        logpi, mu, logv = self.params(h)
        y = y.unsqueeze(-1)                                  # (B, N, 1)
        comp = -0.5 * (np.log(2 * np.pi) + logv + (y - mu) ** 2 / torch.exp(logv))
        logp = torch.logsumexp(logpi + comp, dim=-1)        # (B, N)
        return -logp.mean()

    def moments(self, h):
        """Conditional mean and variance per target (B, N)."""
        logpi, mu, logv = self.params(h)
        pi = torch.exp(logpi)
        v = torch.exp(logv)
        mean = (pi * mu).sum(-1)
        ex2 = (pi * (v + mu ** 2)).sum(-1)
        var = torch.clamp(ex2 - mean ** 2, min=1e-4)
        return mean, var


@dataclass
class MDNResult:
    neuron_names: list
    matrices: dict
    history: list


def _slow_filter(X_list, tau_s, fps):
    out = []
    for X in X_list:
        T = X.shape[0]
        t = np.arange(T, dtype=float) / fps
        out.append(exp_filter_bank(np.nan_to_num(X, nan=0.0), t, [tau_s])[:, :, 0])
    return out


def _pool(X_list, H_list, horizon):
    Hs, Fs = [], []
    for X, H in zip(X_list, H_list):
        T = X.shape[0]
        if T <= horizon:
            continue
        Hs.append(np.nan_to_num(H[:T - horizon], nan=0.0))
        Fs.append(np.nan_to_num(X[horizon:], nan=0.0))
    return np.concatenate(Hs, 0), np.concatenate(Fs, 0)


def fit_mdn(X_list, neuron_names, source_tau_s=20.0, horizon=1, k=3, hidden=128,
            layers=2, epochs=400, lr=2e-3, weight_decay=1e-2, batch=512, fps=4.0,
            seed=0, device=None, verbose=False) -> MDNResult:
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    N = len(neuron_names)
    H_list = _slow_filter(X_list, source_tau_s, fps) if source_tau_s else \
        [np.nan_to_num(X, nan=0.0) for X in X_list]
    Hn, Xf = _pool(X_list, H_list, horizon)
    h_mu, h_sd = Hn.mean(0), Hn.std(0) + 1e-6
    Ht = torch.tensor((Hn - h_mu) / h_sd, dtype=torch.float32, device=dev)
    Ft = torch.tensor(Xf, dtype=torch.float32, device=dev)
    M = Ht.shape[0]

    model = MDN(N, k=k, hidden=hidden, layers=layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(M, device=dev); tot = 0.0; nb = 0
        for s in range(0, M, batch):
            idx = perm[s:s + batch]
            loss = model.nll(Ht[idx], Ft[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        hist.append(tot / nb)
        if verbose and (ep + 1) % max(1, epochs // 6) == 0:
            print(f"  [mdn] epoch {ep+1}/{epochs} nll={tot/nb:.4f}")

    model.eval()
    n_eval = min(4000, M)
    sel = np.random.default_rng(seed).choice(M, n_eval, replace=False)
    ev = torch.tensor((Hn[sel] - h_mu) / h_sd, dtype=torch.float32, device=dev).requires_grad_(True)
    mean, var = model.moments(ev)
    logvar = torch.log(var)
    mean_cpl = np.zeros((N, N)); gain_cpl = np.zeros((N, N))
    for j in range(N):
        gm = torch.autograd.grad(mean[:, j].sum(), ev, retain_graph=True)[0]
        gl = torch.autograd.grad(logvar[:, j].sum(), ev, retain_graph=True)[0]
        mean_cpl[j, :] = gm.mean(0).detach().cpu().numpy() / h_sd
        gain_cpl[j, :] = gl.mean(0).detach().cpu().numpy() / h_sd
    np.fill_diagonal(mean_cpl, 0.0); np.fill_diagonal(gain_cpl, 0.0)
    return MDNResult(neuron_names=list(neuron_names),
                     matrices={"mean": mean_cpl, "gain": gain_cpl}, history=hist)
