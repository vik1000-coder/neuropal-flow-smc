"""DYNOTEARS-analog: native lagged sparse (L1) regression, [post,pre] orientation.

Faithful-ish to Pamfil et al. 2020 "DYNOTEARS": model
    x(t) ~= W^T x(t)  +  sum_{l=1..L} A_l^T x(t-l)
with intra-slice acyclicity h(W)=tr(exp(W*W))-N=0 (NOTEARS) and L1 sparsity on
W and every A_l. Lagged inter-slice edges need NO acyclicity (time orders them).

This is a DYNOTEARS-ANALOG, not the exact package algorithm: we optimise with a
plain augmented-Lagrangian + proximal-gradient (ISTA soft-threshold for L1) loop
rather than the paper's L-BFGS. The per-lag effect matrix is A_l.

Orientation of the raw parameters (DYNOTEARS/NOTEARS "columns are targets"):
    with design rows = time and the model X ~= X_lag @ A_l, entry A_l[i, j]
    is the weight of source i (column of X_lag) onto target j (column of X),
    i.e. edge  x_i(t-l) -> x_j(t).  To get the pipeline [post, pre]:
        E(l)[j, i] = A_l[i, j]      i.e.   E(l) = A_l.T
"""
import numpy as np


def _soft(z, thr):
    return np.sign(z) * np.maximum(np.abs(z) - thr, 0.0)


def fit_dynotears_analog(X_list, L, lambda_a=0.05, lambda_w=0.05,
                         contemp=False, max_iter=400, lr=None, tol=1e-6,
                         rho=1.0, rho_max=1e8, h_tol=1e-8, seed=0):
    """Fit lagged (+ optional contemporaneous acyclic) sparse DAG.

    Returns dict with 'A' = list of [N,N] per-lag matrices in RAW [pre(source i),
    post(target j)] = A_l[i,j] convention, plus 'E' = list of [post,pre] matrices
    E(l) = A_l.T (diagonal zeroed). If contemp=False the contemporaneous slice W
    is omitted (recommended for a pure lag-resolved read-out).
    """
    rng = np.random.default_rng(seed)
    N = X_list[0].shape[1]

    # Build pooled design: target Y = x(t); predictors = [x(t-1..t-L) (+ x(t))]
    Ys, lags = [], [[] for _ in range(L)]
    contemps = []
    for X in X_list:
        X = np.nan_to_num(np.asarray(X, float), nan=0.0)
        T = X.shape[0]
        if T <= L:
            continue
        Ys.append(X[L:])                       # [T-L, N]
        for l in range(1, L + 1):
            lags[l - 1].append(X[L - l:T - l])  # x(t-l)
        contemps.append(X[L:])                  # x(t) for contemporaneous slice
    Y = np.concatenate(Ys, 0)                   # [M, N]
    Xl = [np.concatenate(c, 0) for c in lags]   # list length L of [M, N]
    M = Y.shape[0]

    # standardize columns of design + target to unit scale for conditioning
    def z(a):
        s = a.std(0) + 1e-8
        return a / s, s
    Yz, sY = z(Y)
    Xlz, sX = zip(*[z(a) for a in Xl])
    if contemp:
        Cz, sC = z(np.concatenate(contemps, 0))

    A = [np.zeros((N, N)) for _ in range(L)]    # A[l][i,j]: src i -> tgt j at lag l+1
    W = np.zeros((N, N))                         # contemporaneous
    if lr is None:
        # Lipschitz-ish step from design Gram norms
        Lip = max(np.linalg.norm(a, 2) ** 2 for a in Xlz) + (
            np.linalg.norm(Cz, 2) ** 2 if contemp else 0.0)
        lr = 1.0 / (Lip / M + 1e-8)

    alpha = 0.0  # augmented-Lagrangian multiplier for h(W)
    for outer in range(20 if contemp else 1):
        for it in range(max_iter):
            # prediction residual
            R = Yz.copy()
            for l in range(L):
                R -= Xlz[l] @ A[l]
            if contemp:
                R -= Cz @ W
            # gradients of 0.5/M ||R||^2
            for l in range(L):
                gA = -(Xlz[l].T @ R) / M
                A[l] = _soft(A[l] - lr * gA, lr * lambda_a)
            if contemp:
                gW = -(Cz.T @ R) / M
                # NOTEARS acyclicity grad: h(W)=tr(exp(W*W))-N
                E = np.zeros((N, N))
                WW = W * W
                # scipy expm via series-free eigen approx; use np for small N
                from scipy.linalg import expm
                E = expm(WW)
                h = np.trace(E) - N
                gh = E.T * W * 2.0
                gW = gW + (rho * h + alpha) * gh
                np.fill_diagonal(gW, 0.0)  # no self loops
                W = _soft(W - lr * gW, lr * lambda_w)
                np.fill_diagonal(W, 0.0)
        if contemp:
            from scipy.linalg import expm
            h = np.trace(expm(W * W)) - N
            alpha += rho * h
            if h > 0.25 * 1.0:
                rho = min(rho * 10, rho_max)
            if h <= h_tol or rho >= rho_max:
                break
        else:
            break

    # de-standardize back to raw scale: A_raw[i,j] = A[i,j]*sY[j]/sX_l[i]
    A_raw = []
    for l in range(L):
        Ar = A[l] * (sY[None, :] / sX[l][:, None])
        A_raw.append(Ar)
    E = []
    for l in range(L):
        e = A_raw[l].T.copy()   # [post=j, pre=i]
        np.fill_diagonal(e, 0.0)
        E.append(e)
    return {"A_raw": A_raw, "E": E, "W": W}


if __name__ == "__main__":
    # Verify orientation on a KNOWN driver: 0 -> 1 @ lag2, 1 -> 2 @ lag1.
    rng = np.random.default_rng(0)
    T, N = 3000, 4
    Xs = []
    for w in range(4):
        x = rng.standard_normal((T, N)) * 0.3
        for t in range(2, T):
            x[t, 1] += 0.8 * x[t - 2, 0]
            x[t, 2] += 0.8 * x[t - 1, 1]
        Xs.append(x)
    out = fit_dynotears_analog(Xs, L=3, lambda_a=0.02, contemp=False)
    for l in range(3):
        E = out["E"][l]
        big = [(E[j, i], j, i) for j in range(N) for i in range(N) if abs(E[j, i]) > 0.15]
        print("lag", l + 1, "E[post,pre] nonzero:",
              [(round(v, 2), "edge %d->%d" % (i, j)) for v, j, i in big])
    E2 = out["E"][1]
    print("\nE(lag2)[1,0] (edge 0->1, TRUE) =", round(E2[1, 0], 3), " should be LARGE")
    print("E(lag2)[0,1] (reverse)         =", round(E2[0, 1], 3), " should be ~0")
    E1 = out["E"][0]
    print("E(lag1)[2,1] (edge 1->2, TRUE) =", round(E1[2, 1], 3), " should be LARGE")
    print("E(lag1)[1,2] (reverse)         =", round(E1[1, 2], 3), " should be ~0")
