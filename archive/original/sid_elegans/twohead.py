r"""Two-head denoising-score-matching conditional density model (#1, the nonlinear head).

Realizes the theory's disentangled two-head architecture (Section 6.3; Theorem 9.1):

  * a shared feature **trunk** over the history ``h`` (all sources at t, optionally
    slow-filtered),
  * a **conditional-density head** producing ``mu_j(h)`` and ``log v_j(h)`` for every
    target j (a Gaussian conditional ``rho(x_future | h)``),
  * a **history-marginal head** producing ``s_marg(h) ~ grad log p(h)``.

Both heads are trained by **denoising score matching**:
  - conditional head: corrupt the future, ``x~_fut = x_fut + sigma_c * eps``; the model
    score ``-(x~_fut - mu(h))/v_eff(h)`` is fit to the DSM target ``-eps/sigma_c``. At the
    optimum ``v_eff = v_true + sigma_c^2`` (Prop 6.3), so we subtract ``sigma_c^2`` for
    the readout.
  - marginal head: corrupt the history and fit ``s_marg`` to its DSM target.

Giving the marginal its own head is exactly what stops the shared trunk from spending its
capacity on the (large) history marginal and starving the transition signal — the
Theorem 9.1 failure of naive joint training. The connectome couplings are then read out
by autograd through the (now nonlinear) conditional head:

  mean[j,i] = <d mu_j / d h_i>,   gain[j,i] = <d log v_j / d h_i>   (averaged over history),

in the ``[post, pre]`` convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from sid_neuromod.features.filter_bank import exp_filter_bank


def _mlp(din, dout, hidden, layers, act=nn.SiLU):
    mods, d = [], din
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), act()]
        d = hidden
    mods += [nn.Linear(d, dout)]
    return nn.Sequential(*mods)


class TwoHead(nn.Module):
    """Shared trunk + conditional (mu, log_v) head + history-marginal (score) head."""

    def __init__(self, n, hidden=128, trunk_layers=2, marg_layers=2,
                 logv_floor=-6.0, logv_ceil=4.0):
        super().__init__()
        self.n = n
        self.trunk = _mlp(n, hidden, hidden, trunk_layers)
        self.mu_head = nn.Linear(hidden, n)
        self.logv_head = nn.Linear(hidden, n)
        self.marg = _mlp(n, n, hidden, marg_layers)   # separate marginal head
        self.logv_floor = logv_floor
        self.logv_ceil = logv_ceil

    def conditional(self, h):
        z = self.trunk(h)
        mu = self.mu_head(z)
        logv = torch.clamp(self.logv_head(z), self.logv_floor, self.logv_ceil)
        return mu, logv

    def marginal_score(self, h):
        return self.marg(h)


@dataclass
class TwoHeadResult:
    neuron_names: list
    matrices: dict          # 'mean', 'gain' -> [N, N] ([post, pre])
    history: dict           # training curves
    model: object = None    # trained TwoHead (only when return_model=True) — frozen-feature trunk
    h_mu: np.ndarray = None  # history-standardization mean (trunk input)
    h_sd: np.ndarray = None  # history-standardization sd
    meta: dict = None        # {source_tau_s, horizon, sigma_c, fps}


def _slow_filter(X_list, tau_s, fps):
    out = []
    for X in X_list:
        T = X.shape[0]
        t = np.arange(T, dtype=float) / fps
        F = exp_filter_bank(np.nan_to_num(X, nan=0.0), t, [tau_s])[:, :, 0]
        out.append(F)
    return out


def _pool(X_list, H_list, horizon):
    """Return pooled (H_now, X_fut) with H the (filtered) history and X_fut the raw future."""
    Hs, Fs = [], []
    for X, H in zip(X_list, H_list):
        T = X.shape[0]
        if T <= horizon:
            continue
        Hs.append(np.nan_to_num(H[:T - horizon], nan=0.0))
        Fs.append(np.nan_to_num(X[horizon:], nan=0.0))
    return np.concatenate(Hs, 0), np.concatenate(Fs, 0)


def fit_two_head(X_list, neuron_names, source_tau_s=20.0, horizon=1, hidden=128,
                 trunk_layers=2, epochs=300, lr=2e-3, weight_decay=1e-4, batch=512,
                 sigma_c=0.3, sigma_m=0.3, lam_marg=1.0, fps=4.0, seed=0,
                 device=None, verbose=False, return_model=False) -> TwoHeadResult:
    """Train the two-head DSM model and read out the nonlinear mean/gain connectome."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dev = torch.device(device)
    N = len(neuron_names)

    H_list = _slow_filter(X_list, source_tau_s, fps) if source_tau_s else \
        [np.nan_to_num(X, nan=0.0) for X in X_list]
    Hn, Xf = _pool(X_list, H_list, horizon)
    # standardize the history features (trunk input) on the pooled data
    h_mu, h_sd = Hn.mean(0), Hn.std(0) + 1e-6
    Hn = (Hn - h_mu) / h_sd
    Ht = torch.tensor(Hn, dtype=torch.float32, device=dev)
    Ft = torch.tensor(Xf, dtype=torch.float32, device=dev)
    M = Ht.shape[0]

    model = TwoHead(N, hidden=hidden, trunk_layers=trunk_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {"cond": [], "marg": []}

    for ep in range(epochs):
        perm = torch.randperm(M, device=dev)
        ec, em, nb = 0.0, 0.0, 0
        for s in range(0, M, batch):
            idx = perm[s:s + batch]
            h = Ht[idx]; xf = Ft[idx]
            # conditional DSM: corrupt the future
            eps = torch.randn_like(xf)
            xt = xf + sigma_c * eps
            mu, logv = model.conditional(h)
            v_eff = torch.exp(logv)
            score = -(xt - mu) / v_eff
            target = -eps / sigma_c
            loss_c = ((score - target) ** 2).mean()
            # marginal DSM: corrupt the history
            eh = torch.randn_like(h)
            ht = h + sigma_m * eh
            sm = model.marginal_score(ht)
            loss_m = ((sm + eh / sigma_m) ** 2).mean()
            loss = loss_c + lam_marg * loss_m
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ec += loss_c.item(); em += loss_m.item(); nb += 1
        hist["cond"].append(ec / nb); hist["marg"].append(em / nb)
        if verbose and (ep + 1) % max(1, epochs // 6) == 0:
            print(f"  [two-head] epoch {ep+1}/{epochs} cond={ec/nb:.4f} marg={em/nb:.4f}")

    # ---- readout couplings via autograd through the conditional head ----
    model.eval()
    n_eval = min(4000, M)
    ev = torch.tensor(Hn[np.random.default_rng(seed).choice(M, n_eval, replace=False)],
                      dtype=torch.float32, device=dev).requires_grad_(True)
    mu, logv = model.conditional(ev)
    v_true = torch.clamp(torch.exp(logv) - sigma_c ** 2, min=1e-4)
    logv_true = torch.log(v_true)
    mean_cpl = np.zeros((N, N)); gain_cpl = np.zeros((N, N))
    for j in range(N):
        gmu = torch.autograd.grad(mu[:, j].sum(), ev, retain_graph=True)[0]
        glv = torch.autograd.grad(logv_true[:, j].sum(), ev, retain_graph=True)[0]
        # average the SIGNED gradient over histories (the average marginal effect =
        # the coefficient for a linear model), then undo history standardization.
        mean_cpl[j, :] = (gmu.mean(0).detach().cpu().numpy()) / h_sd
        gain_cpl[j, :] = (glv.mean(0).detach().cpu().numpy()) / h_sd
    np.fill_diagonal(mean_cpl, 0.0); np.fill_diagonal(gain_cpl, 0.0)
    res = TwoHeadResult(neuron_names=list(neuron_names),
                        matrices={"mean": mean_cpl, "gain": gain_cpl}, history=hist)
    if return_model:
        res.model = model.to("cpu").eval()
        res.h_mu = np.asarray(h_mu); res.h_sd = np.asarray(h_sd)
        res.meta = {"source_tau_s": source_tau_s, "horizon": horizon,
                    "sigma_c": sigma_c, "fps": fps}
    return res
