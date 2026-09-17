"""E4: hidden oscillator memory kernel & hidden spectrum (ESPRIT)."""
import numpy as np

from sid_neuromod.synthetic.hidden_oscillator import (esprit_modes,
                                                      simulate_hidden_oscillator)


def _fit_ar_kernel(x, L):
    T = len(x) - 1
    Xlag = np.column_stack([x[L - 1 - j:T - j] for j in range(L)])
    y = x[L:]
    return np.linalg.lstsq(Xlag, y, rcond=None)[0]


def test_hidden_oscillator_esprit_recovers_modes():
    """ESPRIT on the fitted kernel tail recovers hidden modulus & frequency (E4)."""
    data = simulate_hidden_oscillator(T=200000, r=0.9, theta=0.5, q_hidden=0.0, seed=0)
    coef = _fit_ar_kernel(data.x, L=20)
    eig, sv = esprit_modes(coef[1:], model_order=2)  # skip lag-0 self term
    mods = np.abs(eig)
    angs = np.abs(np.angle(eig))
    # modulus error < 0.10, frequency error < 0.15 rad (Section 12.4)
    assert abs(np.max(mods) - 0.9) < 0.10, mods
    assert np.min(np.abs(angs - 0.5)) < 0.15, angs
    # Hankel singular values expose hidden order 2 (two large, rest small)
    assert sv[1] > 5 * sv[2]


def test_oscillator_deterministic_under_seed():
    d1 = simulate_hidden_oscillator(T=5000, seed=3)
    d2 = simulate_hidden_oscillator(T=5000, seed=3)
    assert np.allclose(d1.x, d2.x)
