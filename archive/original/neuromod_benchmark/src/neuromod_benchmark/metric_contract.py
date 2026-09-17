"""Machine-checked semantic contracts for benchmark metric identifiers.

Metric names are part of the scientific API.  A new top-level family must be
registered here before reporting will accept it.  This prevents an implementation
detail (for example, a synthetic latent-vs-oracle comparison) from silently being
promoted to an external-biological claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class MetricFamily:
    prefix: str
    estimand: str
    claim_level: str


@dataclass(frozen=True)
class MetricContract:
    metric_id: str
    estimand: str
    claim_level: str
    direction: str
    optimum: float | None
    unit: str
    role: str


# Longest/special prefixes precede generic ones.
METRIC_FAMILIES: tuple[MetricFamily, ...] = (
    MetricFamily(
        "arm_history_environment_transfer.",
        "post_intervention_history_predictive_transfer",
        "P1",
    ),
    MetricFamily(
        "secondary_knockout_environment_transfer.",
        "legacy_post_intervention_history_predictive_transfer",
        "P1",
    ),
    MetricFamily(
        "secondary_structural_correspondence.",
        "reduced_form_structural_correspondence_secondary",
        "M1",
    ),
    MetricFamily(
        "secondary_localization.",
        "reduced_form_support_localization_secondary",
        "M1",
    ),
    MetricFamily(
        "reduced_form_neural_localization.",
        "reduced_form_neural_joint_localization",
        "M1",
    ),
    MetricFamily("predictive.", "held_out_conditional_law", "P1"),
    MetricFamily("oracle.", "complete_state_transition_functional", "D1"),
    MetricFamily("operator.", "finite_time_transition_operator", "D1"),
    MetricFamily("rollout.", "recursive_path_law", "D2"),
    MetricFamily("path.", "recursive_path_law", "D2"),
    MetricFamily("bridge.", "conditional_bridge_path_forecast", "D1"),
    MetricFamily("state_field.", "occupied_state_transition_field", "D1"),
    MetricFamily("effect.", "typed_complete_state_channel", "M1"),
    MetricFamily(
        "neuromodulator.", "physical_neuromodulator_response_field", "M2"
    ),
    MetricFamily("latent.", "equivalence_aware_latent_state", "L1"),
    MetricFamily(
        "intervention.", "represented_intervention_on_registered_history", "C1"
    ),
    MetricFamily("causal_intervention.", "controlled_common_history_effect", "C1"),
    MetricFamily("localization.", "complete_state_support_localization", "M1"),
    MetricFamily("score.", "fixed_reference_corrupted_score", "S1"),
    MetricFamily("memory.", "history_length_sensitivity", "D1"),
    MetricFamily("transfer.", "held_out_environment_transfer", "P1"),
    MetricFamily("changepoint.", "calibrated_change_detection", "C0"),
    MetricFamily("diagnostic.", "implementation_guardrail", "D0"),
    # External biology is deliberately isolated from simulator truth.
    MetricFamily("biology.", "external_biological_consistency", "B1"),
    MetricFamily("atlas.", "external_atlas_consistency", "B1"),
    MetricFamily("connectome.", "external_connectome_consistency", "B1"),
    # Legacy synthetic secondary metrics remain mechanistic/reduced-form, not B1.
    MetricFamily("secondary.", "reduced_form_correspondence_secondary", "M1"),
)


# Metric identifiers are a public scientific API, not free-form logging labels.
# The middle components identify a registered channel/operation; the terminal
# component identifies the statistic.  Keeping the terminals explicit catches
# spelling mistakes such as ``predictive.rmes_mean`` before they enter a frozen
# result table while still allowing registered channel and operation IDs.
PREDICTIVE_TERMINALS = frozenset(
    {
        "nll",
        "crps_gaussian_moment",
        "crps_ensemble_fair",
        "energy_score",
        "rmse_mean",
        "pit_ece",
        "pit_cvm",
        "coverage_error_90",
        "invalid_variance_fraction",
        "near_natural_variance_boundary_fraction",
        "extreme_variance_fraction",
        "standardized_variance_p99",
        "numerical_valid",
        "min_component_usage",
        "component_entropy",
    }
)
GRAPH_TERMINALS = frozenset(
    {
        "prevalence",
        "beta_min",
        "auprc",
        "auprc_lift_over_prevalence",
        "auroc",
        "precision_at_true_k",
        "null_max_score",
        "null_mean_score",
    }
)
MATRIX_TERMINALS = frozenset(
    {"relative_frobenius", "mae", "cosine", "sign_accuracy_active"}
)
STATE_FIELD_TERMINALS = frozenset(
    {
        "rmse",
        "mae",
        "truth_rms",
        "estimate_rms",
        "signed_mean_mae",
        "nise",
        "field_cosine",
        "null_estimate_max_abs",
        "rms_map_mae",
        "support_beta_min",
    }
) | GRAPH_TERMINALS
RESPONSE_TERMINALS = frozenset(
    {
        "rmse",
        "normalized_rmse",
        "truth_rms",
        "estimate_rms",
        "peak_magnitude_mae",
        "peak_magnitude_normalized_rmse",
        "time_to_peak_mae_steps",
        "time_to_peak_mae_seconds",
        "time_to_peak_estimable_fraction",
        "truth_peak_at_final_lag_fraction",
        "estimate_peak_at_final_lag_fraction",
        "peak_boundary_censored_fraction",
        "windowed_cumulative_response_rmse",
        "windowed_cumulative_response_normalized_rmse",
        # Legacy names remain registered for developmental result readability.
        "integrated_response_rmse",
        "integrated_response_normalized_rmse",
        "first_crossing_1e_mae_steps",
        "first_crossing_1e_mae_seconds",
        "first_crossing_1e_estimable_fraction",
        "first_crossing_1e_censor_mismatch_fraction",
        "decay_time_mae_steps",
        "decay_time_mae_seconds",
        "decay_estimable_fraction",
        "decay_censor_mismatch_fraction",
        "end_window_gain_rmse",
        "end_window_gain_normalized_rmse",
        "end_window_gain_applicable",
        "end_window_fraction",
        "sign_accuracy",
        "active_entry_fraction",
        "null_leakage_rms",
        "null_entry_fraction",
        "truth_mc_standard_error_rms",
        "truth_mc_standard_error_to_signal",
        "truth_monte_carlo_standard_error_rms",
        "truth_monte_carlo_standard_error_to_truth_rms_ratio",
        "truth_rms_to_innovation_rms",
        "history_conditional_rmse",
        "history_conditional_normalized_rmse",
        "history_conditional_truth_rms",
        "history_conditional_estimate_rms",
        "pairwise_rmse_mean",
        "pairwise_rmse_standard_error",
        "pairwise_normalized_rmse_mean",
        "pairwise_normalized_rmse_standard_error",
        "pairwise_normalized_rmse_defined_fraction",
        "pairwise_normalized_rmse_defined_count",
        "pairwise_null_leakage_rms_mean",
        "pairwise_null_leakage_rms_standard_error",
    }
)
LATENT_TERMINALS = frozenset(
    {
        "matched_fraction_true",
        "validation_assignment_abs_spearman",
        "test_spearman",
        "test_physical_orientation_spearman",
        "test_nrmse",
        "test_r2",
        "alignment_negative_slope_fraction",
        "clearance_tau_mae_seconds",
        "clearance_tau_relative_mae",
        "release_pattern_mae",
        "release_pattern_cosine",
        "gauge_dependent_receptor_expression_mae",
        # Legacy developmental identifiers.
        "release_weight_mae",
        "release_weight_cosine",
        "receptor_expression_mae",
        "parameter_recovery_omitted_hierarchical_truth",
        "parameter_recovery_omitted_unidentified_observation_model",
        "parameter_recovery_omitted_expression_effect_gauge",
        "parameter_recovery_omitted_kd_hill_gauge",
        "learned_rms",
        "inactive_rms",
        "inactive_mae",
        "inactive_max_abs",
        "inactive_channel_rms_mean",
        "inactive_channel_rms_max",
        "correct_channel_energy_fraction",
    }
) | GRAPH_TERMINALS | MATRIX_TERMINALS


def _terminal(metric_id: str) -> str:
    return metric_id.rsplit(".", 1)[-1]


def _predictive_terminal(terminal: str) -> bool:
    return terminal in PREDICTIVE_TERMINALS or bool(
        re.fullmatch(r"pit_(?:centered|squared)_product_lag[1-9][0-9]*", terminal)
    )


def _registered_statistic(family: MetricFamily, metric_id: str) -> bool:
    """Return whether ``metric_id`` is a registered statistic in its family."""

    suffix = metric_id.removeprefix(family.prefix)
    terminal = _terminal(metric_id)
    if not suffix or ".." in suffix:
        return False
    if family.prefix == "predictive.":
        return "." not in suffix and _predictive_terminal(terminal)
    if family.prefix in {
        "arm_history_environment_transfer.",
        "secondary_knockout_environment_transfer.",
    }:
        return (
            (".predictive." in suffix and _predictive_terminal(terminal))
            or terminal in {"effect_rmse", "effect_bias", "effect_correlation"}
            or ("response_kernel." in suffix and terminal in RESPONSE_TERMINALS)
        )
    if family.prefix == "intervention.":
        return (
            (".predictive." in suffix and _predictive_terminal(terminal))
            or terminal in {"effect_rmse", "effect_bias", "effect_correlation"}
            or ("response_kernel." in suffix and terminal in RESPONSE_TERMINALS)
        )
    if family.prefix == "causal_intervention.":
        return "common_history_lag1." in suffix and terminal in RESPONSE_TERMINALS
    if family.prefix in {
        "secondary_structural_correspondence.",
        "secondary_localization.",
        "reduced_form_neural_localization.",
        "effect.",
        "localization.",
        "secondary.",
        "biology.",
        "atlas.",
        "connectome.",
    }:
        return terminal in GRAPH_TERMINALS | MATRIX_TERMINALS
    if family.prefix == "oracle.":
        return terminal in (
            GRAPH_TERMINALS
            | MATRIX_TERMINALS
            | {
                "mean_rmse",
                "variance_log_rmse",
                "mean_bias",
                "variance_ratio_median",
                "correlation_offdiagonal_rmse",
            }
        )
    if family.prefix == "operator.":
        return terminal in {
            "linear_probe_nrmse",
            "quadratic_probe_nrmse",
            "tail_exceedance_rmse",
            "tail_exceedance_bias",
            "tail_exceedance_correlation",
            "shape_tail_exceedance_rmse",
            "shape_tail_exceedance_bias",
            "shape_tail_exceedance_correlation",
        }
    if family.prefix in {"rollout.", "path."}:
        return bool(
            re.fullmatch(
                r"(?:path_energy_score_fair|autocovariance_nrmse|spectral_density_nise|"
                r"autocovariance_relative_error_lag[1-9][0-9]*|"
                r"stability\.(?:escape_rate_observed|escape_rate_forecast|escape_rate_error|"
                r"escape_curve_l1|escape_probability_brier_plugin|escape_probability_brier_fair)|"
                r"extreme\.(?:rate_observed|rate_forecast|frequency_error|"
                r"frequency_target_rmse|probability_brier_plugin|probability_brier_fair))",
                suffix,
            )
        )
    if family.prefix == "bridge.":
        return suffix in {
            "path_energy_mean",
            "path_rmse_mean",
            "interior_energy_mean",
            "interior_rmse_mean",
            "endpoint_energy",
            "reference_path_kl_mean",
        }
    if family.prefix == "state_field.":
        return terminal in STATE_FIELD_TERMINALS
    if family.prefix == "neuromodulator.":
        return terminal in STATE_FIELD_TERMINALS | MATRIX_TERMINALS | GRAPH_TERMINALS
    if family.prefix == "latent.":
        return terminal in LATENT_TERMINALS
    if family.prefix == "score.":
        return (
            suffix in {
                "test_training_objective_dsm_risk",
                "test_dsm_risk",
                "test_fixed_reference_dsm_ladder_risk",
            }
            or bool(
                re.fullmatch(
                    r"test\.fixed_reference\.[a-z0-9_]+\.scale_[0-9]+(?:\.[0-9]+)?",
                    suffix,
                )
            )
            or bool(
                re.fullmatch(
                    r"test\.(?:conditional_outcome|joint_consecutive_state)\."
                    r"training_objective_dsm_risk",
                    suffix,
                )
            )
            or bool(
                re.fullmatch(
                    r"test\.(?:conditional_outcome|joint_consecutive_state)\."
                    r"fixed_reference\.[a-z0-9_]+\."
                    r"(?:scale_[0-9]+(?:\.[0-9]+)?|ladder_mean)",
                    suffix,
                )
            )
            # Legacy pre-domain developmental output.
            or bool(
                re.fullmatch(
                    r"test\.fixed_reference\.[a-z0-9_]+\.ladder_mean",
                    suffix,
                )
            )
        )
    if family.prefix == "memory.":
        return suffix in {
            "finite_model_nll_gap_to_max_lag",
            "same_kernel_dsm_gap_to_max_lag",
        } or bool(
            re.fullmatch(
                r"(?:conditional_outcome|joint_consecutive_state)\."
                r"same_kernel_dsm_gap_to_max_lag",
                suffix,
            )
        )
    if family.prefix == "transfer.":
        return suffix in {
            "delta_predictive.nll",
            "delta_predictive.rmse_mean",
            "delta_predictive.energy_score",
        }
    if family.prefix == "changepoint.":
        return suffix in {"false_positive", "n_false_alarms", "detected", "delay"}
    if family.prefix == "diagnostic.":
        return suffix in {
            "prediction_target_invariance_max_abs",
            "metric_parameter_binding_valid",
        }
    return False


def _direction(metric_id: str) -> tuple[str, float | None]:
    if metric_id == "bridge.reference_path_kl_mean":
        # Control effort relative to a chosen Brownian reference is not forecast
        # accuracy.  It is meaningful only on an accuracy-constrained frontier.
        return "none", None
    if metric_id == "changepoint.delay":
        # Early and late detections are both errors; signed delay targets zero.
        return "target", 0.0
    if metric_id.startswith("score."):
        # A Hyvarinen/DSM risk is minimized by the registered (possibly
        # corrupted) law, but its absolute minimum is the kernel- and
        # distribution-dependent entropy term rather than numerical zero.
        return "down", None
    if metric_id.startswith("memory.") or metric_id.startswith(
        "transfer.delta_predictive."
    ):
        # These are signed comparative gaps.  Zero is the reference value, not
        # an attainable optimum: a shorter-memory model or altered observation
        # regime can legitimately score better than its reference.
        return "down", None
    if metric_id.endswith("variance_ratio_median"):
        return "target", 1.0
    if metric_id.endswith(("truth_rms", "estimate_rms")):
        return "none", None
    if any(
        token in metric_id
        for token in (
            "pit_centered_product",
            "pit_squared_product",
            "effect_bias",
            "mean_bias",
            "exceedance_bias",
        )
    ):
        return "target", 0.0
    if "auprc_lift_over_prevalence" in metric_id:
        # This is AP - prevalence, whose attainable maximum depends on prevalence.
        return "up", None
    if any(
        metric_id.endswith(suffix)
        for suffix in (
            ".cosine",
            ".field_cosine",
            ".correlation",
            ".auprc",
            ".auroc",
            ".precision_at_true_k",
            ".sign_accuracy_active",
            ".r2",
            ".spearman",
            ".matched_fraction_true",
            ".correct_channel_energy_fraction",
            ".numerical_valid",
            ".metric_parameter_binding_valid",
            ".detected",
            ".power",
            "_cosine",
            "_correlation",
            "_r2",
            "_spearman",
            ".sign_accuracy",
        )
    ):
        return "up", 1.0
    if any(
        token in metric_id
        for token in (
            ".prevalence",
            ".beta_min",
            ".truth_rms",
            ".estimate_rms",
            ".learned_rms",
            "component_entropy",
            "component_usage",
            "parameter_recovery_omitted",
            "active_entry_fraction",
            "null_entry_fraction",
            "decay_estimable_fraction",
            "first_crossing_1e_estimable_fraction",
            "peak_boundary_censored_fraction",
            "truth_mc_standard_error",
            "truth_monte_carlo_standard_error",
            "truth_rms_to_innovation_rms",
            "time_to_peak_estimable_fraction",
            "truth_peak_at_final_lag_fraction",
            "estimate_peak_at_final_lag_fraction",
            "end_window_gain_applicable",
            "end_window_fraction",
            "pairwise_normalized_rmse_defined_fraction",
            "pairwise_normalized_rmse_defined_count",
        )
    ):
        return "none", None
    if metric_id.endswith("_standard_error"):
        return "none", None
    if metric_id.endswith(
        ("_rate_observed", "_rate_forecast", ".rate_observed", ".rate_forecast")
    ):
        return "none", None
    if ".nll" in metric_id:
        return "down", None
    if (
        "energy_score" in metric_id
        or "energy_mean" in metric_id
        or metric_id.endswith(".endpoint_energy")
        or "crps" in metric_id
        or "brier" in metric_id
    ):
        return "down", None
    return "down", 0.0


def _unit(metric_id: str) -> str:
    if ".nll" in metric_id:
        return "nats_per_scalar_target"
    if metric_id == "memory.finite_model_nll_gap_to_max_lag":
        return "nats_per_scalar_target"
    if metric_id == "bridge.reference_path_kl_mean":
        return "nats_per_path"
    if metric_id.endswith("_seconds") or "tau_mae_seconds" in metric_id:
        return "seconds"
    if metric_id.endswith("_steps") or metric_id.endswith("delay"):
        return "steps"
    if "runtime_seconds" in metric_id:
        return "seconds"
    if any(token in metric_id for token in ("probability", "coverage", "pit_")):
        return "probability"
    if "response_kernel." in metric_id or ".common_history_lag1." in metric_id:
        if metric_id.endswith("_defined_count"):
            return "count"
        if "windowed_cumulative_response_rmse" in metric_id or metric_id.endswith(
            "integrated_response_rmse"
        ):
            return "target_activity_seconds"
        if any(
            token in metric_id
            for token in (
                "normalized",
                "fraction",
                "ratio",
                "applicable",
                "truth_rms_to_innovation_rms",
                "sign_accuracy",
                "_steps",
            )
        ):
            return "dimensionless"
        return "target_activity"
    if metric_id.startswith("bridge."):
        return "target_activity"
    if metric_id.endswith(
        (
            ".predictive.rmse_mean",
            ".predictive.energy_score",
            ".predictive.crps_gaussian_moment",
            ".predictive.crps_ensemble_fair",
            ".conditional_mean.effect_rmse",
            ".conditional_mean.effect_bias",
            "delta_predictive.rmse_mean",
            "delta_predictive.energy_score",
        )
    ) or metric_id in {
        "predictive.rmse_mean",
        "predictive.energy_score",
        "predictive.crps_gaussian_moment",
        "predictive.crps_ensemble_fair",
        "oracle.mean_rmse",
        "oracle.mean_bias",
    }:
        return "target_activity"
    return "dimensionless"


def _role(metric_id: str) -> str:
    if metric_id == "bridge.reference_path_kl_mean":
        return "diagnostic"
    if metric_id in {
        "predictive.nll",
        "predictive.energy_score",
        "rollout.path_energy_score_fair",
        "latent.test_nrmse",
    }:
        return "primary"
    if (
        metric_id.endswith(".nise")
        and (
            metric_id.startswith("state_field.")
            or (
                metric_id.startswith("neuromodulator.")
                and ".state_field" in metric_id
            )
        )
    ) or (
        metric_id.startswith("causal_intervention.")
        and metric_id.endswith(".normalized_rmse")
    ):
        return "primary"
    if any(
        token in metric_id
        for token in (
            "pit_",
            "coverage_error",
            "null_",
            "numerical_valid",
            "prediction_target_invariance",
            "metric_parameter_binding",
            "truth_mc_standard_error",
            "truth_monte_carlo_standard_error",
            "truth_rms_to_innovation_rms",
            "peak_boundary_censored_fraction",
            "peak_at_final_lag_fraction",
            "time_to_peak_estimable_fraction",
            "end_window_gain_applicable",
            "pairwise_normalized_rmse_standard_error",
            "pairwise_normalized_rmse_defined_fraction",
            "pairwise_normalized_rmse_defined_count",
            "pairwise_rmse_standard_error",
            "pairwise_null_leakage_rms_standard_error",
        )
    ):
        return "guardrail"
    return "secondary"


def _estimand(family: MetricFamily, metric_id: str) -> str:
    if metric_id == "bridge.reference_path_kl_mean":
        return "brownian_reference_control_effort_diagnostic"
    if metric_id.startswith("bridge."):
        if "energy" in metric_id:
            return "time_marginal_bridge_forecast_law"
        return "time_marginal_bridge_conditional_mean"
    if ".history_conditional." in metric_id:
        return "history_conditional_intervention_response"
    if ".population_mean.response_kernel." in metric_id:
        return "population_mean_intervention_response_kernel"
    if ".common_history_lag1." in metric_id:
        return "controlled_common_history_lag1_effect"
    if metric_id.startswith("score.test.conditional_outcome."):
        return "conditional_outcome_fixed_reference_corrupted_score"
    if metric_id.startswith("score.test.joint_consecutive_state."):
        return "joint_consecutive_state_fixed_reference_corrupted_score"
    if metric_id.startswith("memory.conditional_outcome."):
        return "conditional_outcome_history_length_score_sensitivity"
    if metric_id.startswith("memory.joint_consecutive_state."):
        return "joint_consecutive_state_history_length_score_sensitivity"
    return family.estimand


def resolve_metric_contract(metric_id: str) -> MetricContract:
    """Resolve a metric or reject an unregistered semantic family."""

    family = next(
        (candidate for candidate in METRIC_FAMILIES if metric_id.startswith(candidate.prefix)),
        None,
    )
    if family is None:
        raise ValueError(f"unregistered metric family for {metric_id!r}")
    if not _registered_statistic(family, metric_id):
        raise ValueError(f"unregistered metric statistic for {metric_id!r}")
    direction, optimum = _direction(metric_id)
    return MetricContract(
        metric_id=metric_id,
        estimand=_estimand(family, metric_id),
        claim_level=family.claim_level,
        direction=direction,
        optimum=optimum,
        unit=_unit(metric_id),
        role=_role(metric_id),
    )
