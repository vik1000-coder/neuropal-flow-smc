from neuromod_benchmark.reporting import (
    _paired_nll_skill_table,
    _table,
    tidy_metrics,
    validate_records,
)


def record(
    capability="normalized",
    metrics=None,
    *,
    method="m",
    horizon=1,
    case_id="one",
):
    return {
        "case_id": case_id,
        "status": "ok",
        "orientation": "[target, source]",
        "case": {
            "method": {"name": method},
            "scenario": {"id": "s", "family": "mechanistic", "params": {"mechanism": "null"}},
            "view": "complete_state",
            "horizon": horizon,
            "seed": 0,
        },
        "split": {"train_groups": [0], "validation_groups": [1], "test_groups": [2]},
        "capabilities": {"predictive_distribution": capability},
        "oracle_level": "complete_state_conditional",
        "metrics": {"predictive.nll": 1.0} if metrics is None else metrics,
    }


def test_validator_rejects_fake_likelihood_for_score_model():
    assert validate_records([record()])["valid"]
    invalid = record(capability="unnormalized_score")
    assert not validate_records([invalid])["valid"]


def test_validator_rejects_null_or_nonfinite_normalized_nll():
    for value in (None, float("nan"), float("inf")):
        validation = validate_records(
            [record(metrics={"predictive.nll": value})]
        )
        assert not validation["valid"]
        assert any("non-finite test NLL" in item for item in validation["problems"])


def test_validator_requires_finite_fixed_reference_risk_for_score_models():
    missing = validate_records(
        [record(capability="unnormalized_score", metrics={"score.test_dsm_risk": 1.0})]
    )
    assert not missing["valid"]
    invalid = validate_records(
        [
            record(
                capability="unnormalized_score",
                metrics={
                    "score.test.conditional_outcome.fixed_reference.gaussian.ladder_mean": None
                },
            )
        ]
    )
    assert not invalid["valid"]
    valid = validate_records(
        [
            record(
                capability="unnormalized_score",
                metrics={
                    "score.test.conditional_outcome.fixed_reference.gaussian.ladder_mean": 1.0
                },
            )
        ]
    )
    assert valid["valid"]


def test_score_domains_resolve_to_distinct_estimands():
    tidy = tidy_metrics(
        [
            record(
                capability="unnormalized_score",
                metrics={
                    "score.test.conditional_outcome.fixed_reference.gaussian.ladder_mean": 1.0,
                    "score.test.joint_consecutive_state.fixed_reference.gaussian.ladder_mean": 2.0,
                },
            )
        ]
    ).set_index("metric_id")
    assert (
        tidy.loc[
            "score.test.conditional_outcome.fixed_reference.gaussian.ladder_mean",
            "estimand",
        ]
        != tidy.loc[
            "score.test.joint_consecutive_state.fixed_reference.gaussian.ladder_mean",
            "estimand",
        ]
    )


def test_tidy_records_carry_claim_and_direction():
    tidy = tidy_metrics([record()])
    assert tidy.iloc[0].claim_level == "P1"
    assert tidy.iloc[0].direction == "down"
    assert tidy.iloc[0].unit == "nats_per_scalar_target"
    assert tidy.iloc[0].role == "primary"


def test_external_biology_is_separate_from_synthetic_secondary_metrics():
    synthetic = tidy_metrics(
        [record(capability="none", metrics={"secondary_localization.x.auprc": .5})]
    ).iloc[0]
    external = tidy_metrics(
        [record(capability="none", metrics={"connectome.support.auprc": .5})]
    ).iloc[0]
    assert synthetic.claim_level == "M1"
    assert external.claim_level == "B1"


def test_validator_rejects_unregistered_metric_family():
    invalid = record(capability="none", metrics={"mystery.score": 1.0})
    validation = validate_records([invalid])
    assert not validation["valid"]
    assert any("unregistered metric family" in item for item in validation["problems"])


def test_validator_rejects_unknown_statistic_inside_registered_family():
    for metric_id in ("predictive.typo", "rollout.typo"):
        validation = validate_records(
            [record(capability="none", metrics={metric_id: 1.0})]
        )
        assert not validation["valid"]
        assert any(
            "unregistered metric statistic" in item
            for item in validation["problems"]
        )


def test_operator_and_conditional_localization_have_non_diagnostic_claim_levels():
    operator = tidy_metrics(
        [record(metrics={"operator.quadratic_probe_nrmse": .2})]
    ).iloc[0]
    assert operator.claim_level == "D1"
    localization = tidy_metrics(
        [record(capability="none", metrics={"localization.conditional_independence_vs_mean.auprc": .6})]
    ).iloc[0]
    assert localization.claim_level == "M1"


def test_metric_optima_distinguish_ratio_correlation_and_zero_target():
    tidy = tidy_metrics(
        [
            record(
                metrics={
                    "oracle.variance_ratio_median": 1.0,
                    "oracle.correlation_offdiagonal.cosine": .5,
                    "predictive.pit_centered_product_lag1": -.1,
                    "latent.test_r2": .2,
                    "latent.test_spearman": .4,
                    "latent.release_pattern_cosine": .5,
                }
            )
        ]
    ).set_index("metric_id")
    assert tidy.loc["oracle.variance_ratio_median", "direction"] == "target"
    assert tidy.loc["oracle.variance_ratio_median", "optimum"] == 1.0
    assert tidy.loc["oracle.correlation_offdiagonal.cosine", "direction"] == "up"
    assert tidy.loc["predictive.pit_centered_product_lag1", "direction"] == "target"
    assert tidy.loc["predictive.pit_centered_product_lag1", "optimum"] == 0.0
    assert tidy.loc["latent.test_r2", "direction"] == "up"
    assert tidy.loc["latent.test_spearman", "direction"] == "up"
    assert tidy.loc["latent.release_pattern_cosine", "direction"] == "up"


def test_bridge_scores_and_reference_effort_have_distinct_contracts():
    tidy = tidy_metrics(
        [
            record(
                metrics={
                    "bridge.path_energy_mean": 0.5,
                    "bridge.path_rmse_mean": 0.4,
                    "bridge.reference_path_kl_mean": 2.0,
                }
            )
        ]
    ).set_index("metric_id")
    assert tidy.loc["bridge.path_energy_mean", "direction"] == "down"
    assert tidy.loc["bridge.path_energy_mean", "optimum"] != 0.0
    assert tidy.loc["bridge.path_energy_mean", "unit"] == "target_activity"
    assert (
        tidy.loc["bridge.path_energy_mean", "estimand"]
        == "time_marginal_bridge_forecast_law"
    )
    assert tidy.loc["bridge.reference_path_kl_mean", "direction"] == "none"
    assert tidy.loc["bridge.reference_path_kl_mean", "role"] == "diagnostic"
    assert tidy.loc["bridge.reference_path_kl_mean", "unit"] == "nats_per_path"


def test_signed_changepoint_delay_targets_zero():
    item = tidy_metrics(
        [record(capability="none", metrics={"changepoint.delay": -2.0})]
    ).iloc[0]
    assert item.direction == "target"
    assert item.optimum == 0.0


def test_nested_predictive_and_effect_metrics_keep_physical_units():
    tidy = tidy_metrics(
        [
            record(
                metrics={
                    "arm_history_environment_transfer.ligand_double_dose.predictive.energy_score": 0.3,
                    "arm_history_environment_transfer.ligand_double_dose.predictive.rmse_mean": 0.2,
                    "arm_history_environment_transfer.ligand_double_dose.conditional_mean.effect_rmse": 0.1,
                    "transfer.delta_predictive.rmse_mean": 0.05,
                    "transfer.delta_predictive.energy_score": 0.04,
                }
            )
        ]
    ).set_index("metric_id")
    assert set(tidy.unit) == {"target_activity"}
    assert tidy.loc["transfer.delta_predictive.rmse_mean", "optimum"] != 0.0


def test_score_risks_and_memory_gaps_have_comparative_not_zero_optima():
    tidy = tidy_metrics(
        [
            record(
                capability="unnormalized_score",
                metrics={
                    "score.test_fixed_reference_dsm_ladder_risk": 1.2,
                    "memory.finite_model_nll_gap_to_max_lag": -0.1,
                    "memory.same_kernel_dsm_gap_to_max_lag": 0.2,
                },
            )
        ]
    ).set_index("metric_id")
    for metric_id in tidy.index:
        assert tidy.loc[metric_id, "direction"] == "down"
        assert tidy.loc[metric_id, "optimum"] != 0.0
    assert (
        tidy.loc["memory.finite_model_nll_gap_to_max_lag", "unit"]
        == "nats_per_scalar_target"
    )


def test_tables_never_pool_methods_over_different_horizon_case_mixes():
    records = [
        record(method="easy_only", horizon=1, case_id="a", metrics={"predictive.nll": 0.0}),
        record(method="all_horizons", horizon=1, case_id="b", metrics={"predictive.nll": 1.0}),
        record(method="all_horizons", horizon=16, case_id="c", metrics={"predictive.nll": 10.0}),
    ]
    table = _table(
        tidy_metrics(records),
        "predictive.nll",
        view="complete_state",
        scenario="s",
    )
    assert "horizon" in table
    assert table.count("all_horizons") == 2
    assert table.count("easy_only") == 1


def test_paired_nll_skill_table_excludes_baseline_self_comparison():
    records = [
        record(method="ridge_var", case_id="a", metrics={"predictive.nll": 1.0}),
        record(method="better", case_id="b", metrics={"predictive.nll": 0.75}),
    ]
    table = _paired_nll_skill_table(
        tidy_metrics(records),
        baseline="ridge_var",
        view="complete_state",
        scenario="s",
        horizon=1,
    )
    assert "better" in table
    assert "ridge_var" not in table
    assert "0.2500" in table
