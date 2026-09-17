"""Section 18 / E2: measurement noise identification and correction (Lemma 7.4)."""
import numpy as np

from sid_neuromod.synthetic.measurement_noise import simulate_ar1_eiv, simulate_ou_noisy
from sid_neuromod.synthetic.oracles import ou_increment_stats, ou_recover_gamma_D_R


def test_ou_multiscale_noise_estimation():
    """Recover (gamma, D, R) from increment statistics across sampling intervals."""
    gamma, D, R, dt = 1.0, 1.0, 0.5, 0.02
    data = simulate_ou_noisy(T=500000, gamma=gamma, D=D, R=R, dt=dt, seed=0)
    x = data.x_obs
    # increment statistics at several downsample factors of the base dt
    factors = [1, 2, 4, 8]
    dts, var_incs, cov_adjs = [], [], []
    for k in factors:
        xk = x[::k]
        inc = np.diff(xk)
        dts.append(k * dt)
        var_incs.append(float(np.var(inc)))
        # covariance of adjacent increments
        cov_adjs.append(float(np.cov(inc[1:], inc[:-1])[0, 1]))
    est = ou_recover_gamma_D_R(dts, var_incs, cov_adjs)
    # measurement noise R within ~15%, D within ~15% (finite-sample tolerance)
    assert abs(est["R"] - R) / R < 0.15, est
    assert abs(est["D"] - D) / D < 0.20, est


def test_ou_increment_oracle_matches_empirical():
    """Closed-form increment statistics match the simulated ones."""
    gamma, D, R, dt = 1.0, 1.0, 0.5, 0.05
    data = simulate_ou_noisy(T=400000, gamma=gamma, D=D, R=R, dt=dt, seed=1)
    inc = np.diff(data.x_obs)
    var_oracle, cov_oracle = ou_increment_stats(gamma, D, R, dt)
    assert abs(np.var(inc) - var_oracle) / var_oracle < 0.05
    assert abs(np.cov(inc[1:], inc[:-1])[0, 1] - cov_oracle) / abs(cov_oracle) < 0.10


def test_ar1_attenuation():
    """Naive AR(1) coefficient under EIV is attenuated by Var(x)/(Var(x)+R)."""
    a, R = 0.8, 1.0
    x, x_obs = simulate_ar1_eiv(T=300000, a=a, R=R, seed=2)
    # naive lag-1 regression coefficient on the noisy series
    num = np.cov(x_obs[1:], x_obs[:-1])[0, 1]
    den = np.var(x_obs[:-1])
    a_naive = num / den
    var_x = np.var(x)
    oracle = a * var_x / (var_x + R)
    assert abs(a_naive - oracle) / oracle < 0.05
