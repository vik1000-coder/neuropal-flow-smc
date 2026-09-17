import numpy as np

from neuromod_benchmark.rethink_suite import (
    NOISE_FAMILIES,
    SuiteConfig,
    _episodes_for_cell,
    build_cells,
    make_supervised,
    simulate_episode,
)


def test_rethink_generators_are_reproducible_finite_and_out_of_family():
    cfg = SuiteConfig(n_neurons=8, n_steps=70, burn_in=20, train_episodes=3,
                      validation_episodes=1, test_episodes=1, seeds=(7,))
    for family in NOISE_FAMILIES:
        first = simulate_episode(seed=12, cfg=cfg, noise_family=family)
        second = simulate_episode(seed=12, cfg=cfg, noise_family=family)
        assert first.latent.shape == (70, 8)
        assert np.isfinite(first.latent).all()
        assert np.array_equal(first.latent, second.latent)


def test_measurement_views_share_latent_trajectory_and_missing_targets_are_excluded():
    cfg = SuiteConfig(n_neurons=8, n_steps=70, burn_in=20, train_episodes=3,
                      validation_episodes=1, test_episodes=1, seeds=(7,))
    latent = simulate_episode(seed=12, cfg=cfg, noise_family="student_t", view="latent")
    damaged = simulate_episode(seed=12, cfg=cfg, noise_family="student_t",
                               view="artifact_missing", observed_fraction=0.5)
    assert np.array_equal(latent.latent, damaged.latent)
    assert damaged.observed.shape[1] == 4
    data = make_supervised([damaged] * 3, lags=(1, 4), horizon=1)
    assert np.isfinite(data.features).all()
    assert np.isfinite(data.targets).all()


def test_suite_declares_all_experiments_and_holds_out_jump_noise_in_e3():
    cfg = SuiteConfig(n_neurons=8, n_steps=60, burn_in=10, train_episodes=3,
                      validation_episodes=1, test_episodes=1, seeds=(7,))
    cells = build_cells(cfg)
    assert {cell["experiment"] for cell in cells} == {"E1", "E2", "E3", "E4", "E5"}
    e3 = next(cell for cell in cells if cell["experiment"] == "E3")
    episodes = _episodes_for_cell(e3, cfg)
    assert episodes[-1].metadata["noise_family"] == "correlated_jumps"
    assert all(ep.metadata["noise_family"] != "correlated_jumps" for ep in episodes[:4])


def test_controlled_operations_change_the_reservoir_under_common_random_numbers():
    cfg = SuiteConfig(n_neurons=8, n_steps=70, burn_in=20)
    control = simulate_episode(seed=91, cfg=cfg, noise_family="gaussian")
    for operation in ("source_silencing", "ligand_pulse", "receptor_attenuation"):
        treated = simulate_episode(seed=91, cfg=cfg, noise_family="gaussian", intervention=operation)
        assert not np.array_equal(control.latent, treated.latent)
