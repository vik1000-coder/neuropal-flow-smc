"""Tests for hierarchical worms and paired observation-only robustness variants."""
from __future__ import annotations

import numpy as np
import pytest

from neuromod_benchmark.hierarchy import (
    HierarchyConfig,
    ObservationVariant,
    draw_worm_spec,
    generate_hierarchical_dataset,
    generate_worm_specs,
    realized_coefficient_cvs,
    simulate_observation_variants,
)
from neuromod_benchmark.mechanistic import (
    MechanisticConfig,
    generate_mechanistic_parameters,
    stability_certificate,
)


def population_config(**kwargs) -> MechanisticConfig:
    values = dict(
        n_neurons=7,
        n_modulators=2,
        n_steps=60,
        burn_in=20,
        mechanism="mixed",
        seed=31,
    )
    values.update(kwargs)
    return MechanisticConfig(**values)


def hierarchy_config(**kwargs) -> HierarchyConfig:
    values = dict(
        n_train_worms=3,
        n_validation_worms=2,
        n_test_worms=2,
        recurrent_weight_cv=0.18,
        clearance_time_cv=0.20,
        receptor_expression_cv=0.22,
        receptor_kd_cv=0.20,
        calcium_decay_cv=0.16,
        seed=2718,
    )
    values.update(kwargs)
    return HierarchyConfig(**values)


def assert_same_parameters(left, right) -> None:
    for field in left.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


def test_worm_deviations_are_reproducible_and_id_stable() -> None:
    config = population_config()
    population = generate_mechanistic_parameters(config)
    hierarchy = hierarchy_config()
    first = draw_worm_spec(
        config, population, hierarchy, split="train", split_index=1
    )
    repeated = draw_worm_spec(
        config, population, hierarchy, split="train", split_index=1
    )
    other = draw_worm_spec(
        config, population, hierarchy, split="test", split_index=1
    )

    assert first.worm_id == repeated.worm_id
    assert first.parameter_seed == repeated.parameter_seed
    assert first.simulation_seed == repeated.simulation_seed
    assert_same_parameters(first.parameters, repeated.parameters)
    assert not np.array_equal(
        first.parameters.receptor_kd, other.parameters.receptor_kd
    )


def test_worm_draws_preserve_support_sign_positivity_and_stability() -> None:
    config = population_config(n_neurons=9, n_modulators=3)
    hierarchy = hierarchy_config(
        n_train_worms=10,
        n_validation_worms=5,
        n_test_worms=5,
        recurrent_weight_cv=0.45,
        receptor_expression_cv=0.45,
    )
    population, specs = generate_worm_specs(config, hierarchy)
    base_a = population.baseline_connectivity
    base_expression = population.receptor_expression

    for spec in specs:
        worm = spec.parameters
        np.testing.assert_array_equal(worm.baseline_connectivity != 0, base_a != 0)
        np.testing.assert_array_equal(
            np.sign(worm.baseline_connectivity), np.sign(base_a)
        )
        np.testing.assert_array_equal(
            worm.receptor_expression > 0, base_expression > 0
        )
        assert np.all((worm.receptor_expression >= 0) & (worm.receptor_expression <= 1))
        assert np.all(worm.receptor_kd > 0)
        assert np.all((worm.clearance_rho > 0) & (worm.clearance_rho < 1))
        worm.validate(spec.config)
        certificate = stability_certificate(spec.config, worm)
        assert certificate.baseline_max_row_l1 <= config.recurrent_row_l1 + 1e-12
        assert np.isfinite(certificate.conditional_mean_abs_bound)


def test_zero_declared_cv_reproduces_population_coefficients_exactly() -> None:
    config = population_config()
    population = generate_mechanistic_parameters(config)
    hierarchy = hierarchy_config(
        recurrent_weight_cv=0.0,
        clearance_time_cv=0.0,
        receptor_expression_cv=0.0,
        receptor_kd_cv=0.0,
        calcium_decay_cv=0.0,
    )
    worm = draw_worm_spec(
        config, population, hierarchy, split="validation", split_index=0
    )
    assert_same_parameters(worm.parameters, population)
    assert worm.config.calcium_decay_seconds == config.calcium_decay_seconds
    np.testing.assert_array_equal(worm.multipliers.recurrent_weight, 1.0)
    np.testing.assert_array_equal(worm.multipliers.clearance_time, 1.0)
    np.testing.assert_array_equal(worm.multipliers.receptor_expression, 1.0)
    np.testing.assert_array_equal(worm.multipliers.receptor_kd, 1.0)
    assert worm.multipliers.calcium_decay == 1.0


def test_realized_multiplier_cvs_track_declared_population_cvs() -> None:
    config = population_config(
        n_neurons=8,
        n_modulators=3,
        n_steps=2,
        burn_in=0,
    )
    requested = 0.20
    hierarchy = hierarchy_config(
        n_train_worms=100,
        n_validation_worms=60,
        n_test_worms=60,
        recurrent_weight_cv=requested,
        clearance_time_cv=requested,
        receptor_expression_cv=requested,
        receptor_kd_cv=requested,
        calcium_decay_cv=requested,
        observation_variants=(ObservationVariant("reference"),),
    )
    population, specs = generate_worm_specs(config, hierarchy)
    realized = realized_coefficient_cvs(specs, population)
    assert hierarchy.declared_coefficient_cvs == {
        "recurrent_weight": requested,
        "clearance_time": requested,
        "receptor_expression": requested,
        "receptor_kd": requested,
        "calcium_decay": requested,
    }
    for name, value in realized.items():
        assert abs(value - requested) < 0.045, (name, value)


def test_observation_variants_share_exact_latent_hash_and_exogenous_noise() -> None:
    config = population_config(n_steps=100, burn_in=30)
    hierarchy = hierarchy_config()
    population = generate_mechanistic_parameters(config)
    worm = draw_worm_spec(
        config, population, hierarchy, split="test", split_index=0
    )
    variants = (
        ObservationVariant("reference"),
        ObservationVariant("noise_x4", measurement_noise_multiplier=4.0),
        ObservationVariant("fast_calcium", calcium_decay_multiplier=0.5),
        ObservationVariant(
            "slow_and_noisy",
            measurement_noise_multiplier=3.0,
            calcium_decay_multiplier=2.0,
        ),
    )
    records = simulate_observation_variants(worm, variants)
    reference = records["reference"].trajectory
    assert len({record.latent_hash for record in records.values()}) == 1
    assert len(
        {
            record.trajectory.exogenous_fingerprint
            for record in records.values()
        }
    ) == 1
    for record in records.values():
        np.testing.assert_array_equal(
            record.trajectory.latent_neural, reference.latent_neural
        )
        np.testing.assert_array_equal(
            record.trajectory.conditional_mean, reference.conditional_mean
        )
        np.testing.assert_array_equal(
            record.trajectory.conditional_variance,
            reference.conditional_variance,
        )

    noise_only = records["noise_x4"].trajectory
    np.testing.assert_array_equal(noise_only.clean_calcium, reference.clean_calcium)
    assert not np.array_equal(noise_only.fluorescence, reference.fluorescence)
    fast = records["fast_calcium"].trajectory
    assert not np.array_equal(fast.clean_calcium, reference.clean_calcium)


def test_hierarchical_dataset_has_disjoint_declared_worm_splits() -> None:
    config = population_config(n_steps=45, burn_in=15)
    hierarchy = hierarchy_config(
        n_train_worms=3,
        n_validation_worms=2,
        n_test_worms=2,
    )
    dataset = generate_hierarchical_dataset(config, hierarchy)

    assert len(dataset.worms) == 7
    assert len(dataset.worms_for_split("train")) == 3
    assert len(dataset.worms_for_split("validation")) == 2
    assert len(dataset.worms_for_split("test")) == 2
    train = set(dataset.split_worm_ids["train"])
    validation = set(dataset.split_worm_ids["validation"])
    test = set(dataset.split_worm_ids["test"])
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)

    assert dataset.metadata["declared_coefficient_cvs"] == dict(
        hierarchy.declared_coefficient_cvs
    )
    assert dataset.metadata["split_unit"] == "worm"
    assert dataset.metadata["observation_variants_latent_hash_invariant"] is True
    for worm in dataset.worms:
        assert len(worm.observations) == len(hierarchy.observation_variants)
        assert len({record.latent_hash for record in worm.observations.values()}) == 1
        for record in worm.observations.values():
            assert record.trajectory.latent_neural.shape == (45, 7)


def test_hierarchical_dataset_generation_is_reproducible() -> None:
    config = population_config(n_steps=30, burn_in=10)
    hierarchy = hierarchy_config(
        n_train_worms=2,
        n_validation_worms=1,
        n_test_worms=1,
        observation_variants=(ObservationVariant("reference"),),
    )
    first = generate_hierarchical_dataset(config, hierarchy)
    repeated = generate_hierarchical_dataset(config, hierarchy)
    assert first.split_worm_ids == repeated.split_worm_ids
    assert first.metadata == repeated.metadata
    assert_same_parameters(first.population_parameters, repeated.population_parameters)
    for left, right in zip(first.worms, repeated.worms, strict=True):
        assert left.spec.worm_id == right.spec.worm_id
        assert_same_parameters(left.spec.parameters, right.spec.parameters)
        np.testing.assert_array_equal(
            left.observations["reference"].trajectory.latent_neural,
            right.observations["reference"].trajectory.latent_neural,
        )
        assert left.latent_hash == right.latent_hash


def test_different_worms_have_distinct_parameters_and_latent_realizations() -> None:
    config = population_config(n_steps=40, burn_in=10)
    hierarchy = hierarchy_config(
        n_train_worms=2,
        n_validation_worms=1,
        n_test_worms=1,
        observation_variants=(ObservationVariant("reference"),),
    )
    dataset = generate_hierarchical_dataset(config, hierarchy)
    left, right = dataset.worms[:2]
    assert left.spec.parameter_seed != right.spec.parameter_seed
    assert left.spec.simulation_seed != right.spec.simulation_seed
    assert not np.array_equal(
        left.spec.parameters.receptor_kd, right.spec.parameters.receptor_kd
    )
    assert left.latent_hash != right.latent_hash


def test_duplicate_observation_variant_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        hierarchy_config(
            observation_variants=(
                ObservationVariant("same"),
                ObservationVariant("same", calcium_decay_multiplier=2.0),
            )
        ).validate()


def test_unknown_split_is_rejected() -> None:
    config = population_config()
    hierarchy = hierarchy_config()
    population = generate_mechanistic_parameters(config)
    with pytest.raises(ValueError, match="unknown split"):
        draw_worm_spec(
            config,
            population,
            hierarchy,
            split="development",
            split_index=0,
        )
