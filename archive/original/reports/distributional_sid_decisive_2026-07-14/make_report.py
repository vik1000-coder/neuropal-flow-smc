from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "distributional_sid" / "runs"
FIG = HERE / "figures"
TAB = HERE / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

BLUE = "#356AA0"
ORANGE = "#D9822B"
PURPLE = "#7561A8"
GRAY = "#697386"
LIGHT = "#D8E2EC"
GREEN = "#3B7D67"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)


def read_summary(name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / name / "summary.csv")


def point(frame: pd.DataFrame, metric: str, method: str, **filters: str) -> pd.Series:
    selected = frame[(frame.metric_name == metric) & (frame.method == method)]
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {metric=} {method=} {filters=}; found {len(selected)}")
    return selected.iloc[0]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def e1_figure() -> None:
    frame = read_summary("e1_confirmation_20260714")
    methods = ["PLUG", "RIESZ", "ORTH"]
    colors = [GRAY, PURPLE, BLUE]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for axis, metric, target, title in (
        (axes[0], "nrmse", "characteristic_tangent", "Characteristic tangent error"),
        (axes[1], "coverage_fraction", "characteristic_tangent", "Coordinatewise 95% coverage"),
    ):
        rows = [point(frame, metric, method, target_name=target) for method in methods]
        means = [row["mean"] for row in rows]
        lower = [row["mean"] - row["ci_lower"] for row in rows]
        upper = [row["ci_upper"] - row["mean"] for row in rows]
        axis.bar(methods, means, color=colors, width=0.65)
        axis.errorbar(range(3), means, yerr=[lower, upper], fmt="none", color="#222222", capsize=3)
        axis.set_title(title)
        if metric == "coverage_fraction":
            axis.axhline(0.95, color=ORANGE, linestyle="--", linewidth=1.2)
            axis.set_ylim(0, 1.05)
        else:
            axis.axhline(1.0, color=ORANGE, linestyle="--", linewidth=1.2)
            axis.set_ylabel("NRMSE (zero estimator = 1)")
    targets = ["mean_derivative", "covariance_derivative"]
    rows = [point(frame, "nrmse", "DIRECT", target_name=target) for target in targets]
    labels = ["Mean", "Full covariance"]
    means = [row["mean"] for row in rows]
    axes[2].bar(labels, means, color=[BLUE, ORANGE], width=0.62)
    axes[2].axhline(1.0, color=GRAY, linestyle="--", linewidth=1.2)
    axes[2].set_title("Typed direct baselines")
    axes[2].set_ylabel("NRMSE")
    for index, value in enumerate(means):
        axes[2].text(index, value + 0.08, f"{value:.2f}", ha="center")
    fig.suptitle("E1: ORTH calibrates inference, but does not improve point accuracy", y=1.03, fontsize=11)
    fig.tight_layout()
    save(fig, "e1_recovery")


def e2_e3_figure() -> None:
    perturb = pd.read_csv(RUNS / "development_milestone_20260714_v2" / "e2_perturbations.csv")
    e3 = read_summary("e3_confirmation_20260714")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    perturb = perturb[(perturb.method == "ORTH") & (perturb.error_product > 0) & (perturb.bias_norm > 0)]
    for name, group in perturb.groupby("perturbation"):
        group = group.sort_values("error_product")
        axes[0].plot(group["error_product"], group["bias_norm"], "o-", label=name)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Nuisance product $\|\delta_m\|\|\delta_\alpha\|$")
    axes[0].set_ylabel("ORTH bias norm")
    axes[0].set_title("E2: product-order remainder")
    axes[0].legend(frameon=False, fontsize=8)
    order = ["d4_n8000", "d16_n8000", "d32_n32000"]
    labels = ["d=4\nn=8k", "d=16\nn=8k", "d=32\nn=32k"]
    rows = [point(e3, "nrmse", "ORTH", cell_id=cell, target_name="characteristic_tangent") for cell in order]
    means = [row["mean"] for row in rows]
    lower = [row["mean"] - row["ci_lower"] for row in rows]
    upper = [row["ci_upper"] - row["mean"] for row in rows]
    axes[1].errorbar(range(3), means, yerr=[lower, upper], fmt="o-", color=BLUE, capsize=3)
    axes[1].set_xticks(range(3), labels)
    axes[1].axhline(1, color=ORANGE, linestyle="--")
    axes[1].set_ylabel("ORTH NRMSE")
    axes[1].set_title("E3: one future per history")
    axes[1].set_ylim(0, 0.18)
    fig.tight_layout()
    save(fig, "e2_e3_orthogonality_scaling")


def e4_figure() -> None:
    selection = pd.read_csv(RUNS / "e4_frequency_audit_20260714" / "bank_selection.csv")
    detection = pd.read_csv(RUNS / "e4_confirmation_20260714" / "detection_summary.csv")
    summary = read_summary("e4_confirmation_20260714")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    labels = ["Low\n0.25–2", "Wide\n0.5–4", "High\n1–8"]
    lookup = selection.set_index("bank")
    values = [lookup.loc[name, "mean_orth_nrmse"] for name in ["default_0p25_2", "wide_0p5_4", "high_1_8"]]
    axes[0].bar(labels, values, color=[GRAY, PURPLE, BLUE])
    axes[0].axhline(1, color=ORANGE, linestyle="--")
    axes[0].set_ylabel("Mean ORTH NRMSE")
    axes[0].set_title("Development frequency audit")
    family_labels = ["Discrete\nfinite difference", "Continuous\nLegendre"]
    family_order = ["finite_difference_mixture", "continuous_legendre_tilt"]
    powers = [float(detection.loc[detection.dgp_family == family, "characteristic_power"].iloc[0]) for family in family_order]
    false = [float(detection.loc[detection.dgp_family == family, "moment_false_positive_rate"].iloc[0]) for family in family_order]
    x = np.arange(2)
    axes[1].bar(x - 0.18, powers, 0.36, color=BLUE, label="Characteristic power")
    axes[1].bar(x + 0.18, false, 0.36, color=GRAY, label="Moment FWER")
    axes[1].axhline(0.05, color=ORANGE, linestyle="--")
    axes[1].set_xticks(x, family_labels)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Confirmatory detection")
    axes[1].legend(frameon=False, fontsize=7)
    rows = [point(summary, "nrmse", "ORTH", dgp_family=family) for family in family_order]
    means = [row["mean"] for row in rows]
    axes[2].bar(family_labels, means, color=[BLUE, PURPLE])
    axes[2].axhline(1, color=ORANGE, linestyle="--")
    axes[2].set_ylabel("ORTH NRMSE")
    axes[2].set_title("Resolved law-tangent error")
    fig.tight_layout()
    save(fig, "e4_frequency_moment_blind")


def e5_figure() -> None:
    summary = read_summary("e5_confirmation_20260714")
    diagnostic = pd.read_csv(RUNS / "e5_confirmation_20260714" / "likelihood_score_diagnostic.csv")
    sigma = np.array([1.0, 0.3, 0.1, 0.03, 0.01, 0.0])
    cell = ["sigma_1p0", "sigma_0p3", "sigma_0p1", "sigma_0p03", "sigma_0p01", "sigma_0p0"]
    nrmse = [point(summary, "nrmse", "ORTH", cell_id=value)["mean"] for value in cell]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    axes[0].plot(np.arange(len(sigma)), nrmse, "o-", color=BLUE)
    axes[0].set_xticks(np.arange(len(sigma)), [str(value) for value in sigma])
    axes[0].set_xlabel(r"Noise $\sigma$")
    axes[0].set_ylabel("ORTH NRMSE")
    axes[0].set_title("Weak-feature estimator")
    positive = diagnostic[diagnostic.sigma > 0].sort_values("sigma", ascending=False)
    axes[1].loglog(positive.sigma, positive.likelihood_score_rms, "o-", color=ORANGE)
    axes[1].set_xlabel(r"Noise $\sigma$")
    axes[1].set_ylabel("Likelihood-score RMS")
    axes[1].set_title("Likelihood route diverges")
    axes[1].invert_xaxis()
    fig.tight_layout()
    save(fig, "e5_support_motion")


def e6_figure() -> None:
    frame = read_summary("e6_confirmation_20260714")
    order = [
        "gaussian_d4",
        "student_t_nu10",
        "student_t_nu5",
        "student_t_nu3",
        "uniform_valid_weight",
        "uniform_atanh_coordinate",
        "gap_mu0",
        "gap_mu1",
        "gap_mu2",
        "gap_mu3",
        "gap_mu4",
        "circle_supported_tangent",
    ]
    labels = ["Gaussian", "t10", "t5", "t3", "Uniform\nweighted", "Uniform\natanh", "gap0", "gap1", "gap2", "gap3", "gap4", "Circle\ntangent"]
    alpha = [point(frame, "alpha_nrmse", "targeted_riesz_sieve", cell_id=cell)["mean"] for cell in order]
    norm_values = [point(frame, "oracle_representer_norm", "targeted_riesz_sieve", cell_id=cell)["mean"] for cell in order]
    coverage = [point(frame, "target_covered", "targeted_riesz_sieve", cell_id=cell)["mean"] for cell in order]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    x = np.arange(len(order))
    axes[0].bar(x - 0.19, alpha, 0.38, color=BLUE, label=r"$\alpha$ NRMSE")
    axes[0].bar(x + 0.19, norm_values, 0.38, color=ORANGE, label="Oracle representer norm")
    axes[0].axhline(1, color=GRAY, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_title("E6: approximation fails as tails/overlap become difficult")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].bar(x, coverage, color=[BLUE if value >= 0.9 else ORANGE for value in coverage])
    axes[1].axhline(0.95, color=GRAY, linestyle="--")
    axes[1].set_ylabel("Target coverage")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    fig.tight_layout()
    save(fig, "e6_riesz_stability")


def e7_e11_figure() -> None:
    e7 = read_summary("e7_confirmation_20260714")
    e11 = read_summary("e11_confirmation_20260714")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    cells = ["mean_cancellation_R1", "mean_cancellation_R2", "variance_cancellation_R1", "variance_cancellation_R2"]
    labels = ["Mean R1", "Mean R2", "Variance R1", "Variance R2"]
    estimated = [point(e7, "projected_energy_estimate", "ORTH", cell_id=cell)["mean"] for cell in cells]
    truth = [point(e7, "projected_energy_truth", "ORTH", cell_id=cell)["mean"] for cell in cells]
    x = np.arange(4)
    axes[0].bar(x - 0.18, truth, 0.36, color=GRAY, label="Truth")
    axes[0].bar(x + 0.18, estimated, 0.36, color=BLUE, label="ORTH")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Projected local energy")
    axes[0].set_title("E7: global cancellation is not no effect")
    axes[0].legend(frameon=False)
    families = ["full_path_characteristic", "endpoint_characteristic", "unordered_channel_totals"]
    labels = ["Full path", "Endpoint", "Unordered totals"]
    power = [point(e11, "detected_bonferroni", "ORTH", cell_id=family)["mean"] for family in families]
    axes[1].bar(labels, power, color=[BLUE, GRAY, GRAY])
    axes[1].axhline(0.05, color=ORANGE, linestyle="--")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Bonferroni detection rate")
    axes[1].set_title("E11: ordering requires path features")
    fig.tight_layout()
    save(fig, "e7_e11_heterogeneity_path")


def e8_e9_e10_figure() -> None:
    e8 = read_summary("e8_confirmation_20260714")
    e9 = read_summary("e9_confirmation_20260714")
    e10 = read_summary("e10_confirmation_20260714")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    cells = ["active_clean", "active_filter_rho0p5", "active_filter_rho0p9", "active_filter_rho0p98", "exact_observed_null", "measurement_crosstalk"]
    labels = ["clean", "ρ=.5", "ρ=.9", "ρ=.98", "null", "cross-talk"]
    values = [point(e8, "detected_bonferroni", "ORTH", cell_id=cell)["mean"] for cell in cells]
    axes[0].bar(labels, values, color=[BLUE, BLUE, PURPLE, PURPLE, GRAY, ORANGE])
    axes[0].axhline(0.05, color=GRAY, linestyle="--")
    axes[0].set_ylim(0, 1.08)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("E8: filter attenuation and cross-talk")
    axes[0].set_ylabel("Detection rate")
    methods = ["DIRECT_COV_ORTH", "ORTH_CHARACTERISTIC_INVERSION"]
    labels = ["Direct covariance", "Characteristic inversion"]
    angles = [point(e9, "largest_subspace_angle_degrees", method, cell_id="medium_dense_rotated_K3")["mean"] for method in methods]
    axes[1].bar(labels, angles, color=[BLUE, PURPLE])
    axes[1].axhline(20, color=ORANGE, linestyle="--")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Largest angle (degrees)")
    axes[1].set_title("E9: dense response subspace")
    methods = ["ORTH", "MDN_K5_NORMALIZED", "CONDITIONAL_FLOW_MATCHING"]
    labels = ["ORTH", "MDN", "Flow matching"]
    x = np.arange(3)
    for index, family in enumerate(["finite_difference_mixture", "continuous_legendre_tilt"]):
        values = [point(e10, "nrmse", method, cell_id=family)["mean"] for method in methods]
        axes[2].bar(x + (index - 0.5) * 0.32, values, 0.32, color=[BLUE, ORANGE][index], label=["Discrete", "Continuous"][index])
    axes[2].set_xticks(x, labels, rotation=25, ha="right")
    axes[2].axhline(1, color=GRAY, linestyle="--")
    axes[2].set_ylabel("Tangent NRMSE")
    axes[2].set_title("E10: partial equal-access test")
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "e8_e9_e10_limits")


def completion_gate_figure() -> None:
    e3 = pd.read_csv(RUNS / "e3_dependent_confirmation_20260714" / "seed_level.csv")
    e8 = pd.read_csv(RUNS / "e8_measurement_confirmation_20260714" / "seed_level.csv")
    e9 = pd.read_csv(RUNS / "e9_staged_confirmation_20260714" / "seed_level.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))

    e3_cells = ["d16_n8000_rho0p5", "d16_n8000_rho0p9", "d32_n32000_rho0p9"]
    e3_labels = ["d16, ρ=.5", "d16, ρ=.9", "d32, ρ=.9"]
    blocked = e3[(e3.method == "ORTH") & (e3.split_mode == "blocked")]
    iid = [blocked.loc[blocked.cell_id == cell, "coverage_fraction_iid"].mean() for cell in e3_cells]
    hac = [blocked.loc[blocked.cell_id == cell, "coverage_fraction_hac"].mean() for cell in e3_cells]
    x = np.arange(len(e3_cells))
    axes[0].bar(x - 0.18, iid, 0.36, color=GRAY, label="IID SE")
    axes[0].bar(x + 0.18, hac, 0.36, color=BLUE, label="HAC SE")
    axes[0].axhline(0.95, color=ORANGE, linestyle="--", linewidth=1.1)
    axes[0].set_xticks(x, e3_labels, rotation=25, ha="right")
    axes[0].set_ylim(0.75, 1.01)
    axes[0].set_ylabel("Coordinatewise coverage")
    axes[0].set_title("E3: dependence changes uncertainty")
    axes[0].legend(frameon=False, fontsize=8)

    e8_cells = [
        "hetero_active_1p0",
        "history_missing_active_mask_included",
        "history_missing_active_mask_omitted",
        "known_deconv_rho0p5",
        "known_deconv_rho0p9",
        "known_deconv_rho0p98",
        "misspecified_deconv_rho0p9_as0p5",
    ]
    e8_labels = ["hetero", "hist miss\n+ mask", "hist miss\n− mask", "deconv\nρ=.5", "deconv\nρ=.9", "deconv\nρ=.98", "wrong\nfilter"]
    e8_values = [e8.loc[(e8.cell_id == cell) & (e8.method == "ORTH"), "nrmse"].mean() for cell in e8_cells]
    axes[1].bar(np.arange(len(e8_cells)), e8_values, color=[GREEN, BLUE, PURPLE, GREEN, ORANGE, ORANGE, "#A64B4B"])
    axes[1].axhline(1.0, color=GRAY, linestyle="--", linewidth=1.1)
    axes[1].set_xticks(np.arange(len(e8_cells)), e8_labels, rotation=25, ha="right")
    axes[1].set_ylabel("ORTH NRMSE")
    axes[1].set_title("E8: moderate correction, severe failure")

    e9_cells = ["axis_aligned_K1", "axis_aligned_K3", "n32000_rotated_K3", "response32_rotated_K3", "rotated_K5"]
    e9_labels = ["axis K1", "axis K3", "n=32k", "response 32", "rotated K5"]
    method_specs = [
        ("DIRECT_COV_ORTH", "Direct", BLUE, "o-"),
        ("DIRECT_COV_ORTH_LOW_RANK_SELECTED", "Low-rank", GREEN, "s-"),
        ("ORTH_CHARACTERISTIC_INVERSION", "Characteristic", PURPLE, "^-"),
    ]
    for method, label, color, style in method_specs:
        values = [
            e9.loc[(e9.cell_id == cell) & (e9.method == method), "largest_subspace_angle_degrees"].mean()
            for cell in e9_cells
        ]
        axes[2].plot(np.arange(len(e9_cells)), values, style, color=color, label=label)
    axes[2].axhline(20.0, color=ORANGE, linestyle="--", linewidth=1.1)
    axes[2].set_xticks(np.arange(len(e9_cells)), e9_labels, rotation=25, ha="right")
    axes[2].set_ylabel("Largest response angle (degrees)")
    axes[2].set_title("E9: typed geometry remains stronger")
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "completion_e3_e8_e9")


def _e10_primary() -> pd.DataFrame:
    raw = pd.read_csv(RUNS / "e10_full_confirmation_20260714" / "seed_level.csv")
    analytic = pd.read_csv(
        RUNS / "e10_analytic_mdn_confirmation_20260714" / "seed_level.csv"
    )
    primary = pd.concat(
        [
            raw[
                (raw.method == "ORTH")
                | (raw.method.str.startswith("NSF_") & (raw.sampling_budget == 1024))
            ],
            analytic,
        ],
        ignore_index=True,
    )
    return (
        primary.groupby(["dgp_family", "dgp_seed", "method"], as_index=False)
        .agg(
            nrmse=("nrmse", "mean"),
            feature_prediction_mse=("feature_prediction_mse", "mean"),
            heldout_log_likelihood=("heldout_log_likelihood", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        )
    )


def e10_paired_ratios(primary: pd.DataFrame) -> pd.DataFrame:
    pivot = primary.pivot(index=["dgp_family", "dgp_seed"], columns="method", values="nrmse")
    methods = [column for column in pivot.columns if column != "ORTH"]
    rng = np.random.default_rng(20260714)
    rows: list[dict[str, object]] = []
    for family in list(primary.dgp_family.unique()) + ["ALL_7_FAMILIES"]:
        family_pivot = pivot if family == "ALL_7_FAMILIES" else pivot.loc[[family]]
        for method in methods:
            log_ratios = np.log(family_pivot[method] / family_pivot["ORTH"])
            if family == "ALL_7_FAMILIES":
                log_ratios = log_ratios.groupby(level="dgp_seed").mean()
            values = log_ratios.to_numpy()
            draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
            estimate, lower, upper = np.exp(
                [values.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)]
            )
            rows.append(
                {
                    "dgp_family": family,
                    "candidate": method,
                    "baseline": "ORTH",
                    "geometric_mean_error_ratio": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "candidate_superior": bool(upper < 0.90),
                    "orth_superior": bool(1.0 / lower < 0.90),
                    "n_dgp_seeds": len(values),
                    "bootstrap_replicates": 2000,
                }
            )
    return pd.DataFrame(rows)


def e10_full_figure() -> None:
    raw = pd.read_csv(RUNS / "e10_full_confirmation_20260714" / "seed_level.csv")
    primary = _e10_primary()
    families = [
        "E1_gaussian", "E4_discrete", "E4_continuous", "E5_support_motion",
        "E8_filtered", "E9_dense", "E11_ordering",
    ]
    family_labels = ["E1", "E4-D", "E4-C", "E5", "E8", "E9", "E11"]
    methods = [
        "ANALYTIC_MDN_K5", "ANALYTIC_MDN_K10", "ANALYTIC_MDN_K20",
        "NSF_6", "NSF_10",
    ]
    method_labels = ["MDN-A5", "MDN-A10", "MDN-A20", "NSF-6", "NSF-10"]
    pivot = primary.pivot(index=["dgp_family", "dgp_seed"], columns="method", values="nrmse")
    ratio = np.array(
        [
            [np.exp(np.log(pivot.loc[family, method] / pivot.loc[family, "ORTH"]).mean()) for family in families]
            for method in methods
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw={"width_ratios": [1.35, 1.0]})
    image = axes[0].imshow(ratio, cmap="RdYlBu_r", vmin=0.5, vmax=2.0, aspect="auto")
    axes[0].set_xticks(np.arange(len(families)), family_labels)
    axes[0].set_yticks(np.arange(len(methods)), method_labels)
    axes[0].set_title("Reusable-law / ORTH tangent error")
    for row in range(ratio.shape[0]):
        for column in range(ratio.shape[1]):
            axes[0].text(column, row, f"{ratio[row, column]:.2f}", ha="center", va="center", fontsize=7.5)
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Geometric mean NRMSE ratio")

    neural = raw[raw.method != "ORTH"].copy()
    seed_budget = neural.groupby(["method", "dgp_family", "dgp_seed", "sampling_budget"], as_index=False).nrmse.mean()
    reference = seed_budget[seed_budget.sampling_budget == 1024][["method", "dgp_family", "dgp_seed", "nrmse"]].rename(columns={"nrmse": "nrmse_1024"})
    seed_budget = seed_budget.merge(reference, on=["method", "dgp_family", "dgp_seed"])
    seed_budget["sampling_ratio"] = seed_budget.nrmse / seed_budget.nrmse_1024
    for method, label, color, style in [
        ("MDN_K5", "AR-MDN-5", ORANGE, "o-"),
        ("MDN_K10", "AR-MDN-10", "#B76E2E", "s-"),
        ("MDN_K20", "AR-MDN-20", "#8C5A2B", "^-"),
        ("NSF_6", "NSF-6", PURPLE, "o--"),
        ("NSF_10", "NSF-10", BLUE, "s--"),
    ]:
        group = seed_budget[seed_budget.method == method].groupby("sampling_budget").sampling_ratio.apply(
            lambda values: float(np.exp(np.log(values).mean()))
        )
        axes[1].plot(group.index, group.values, style, color=color, label=label)
    axes[1].axhline(1.0, color=GRAY, linestyle=":", linewidth=1.1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([64, 256, 1024], ["64", "256", "1024"])
    axes[1].set_xlabel("Samples per evaluation history")
    axes[1].set_ylabel("NRMSE relative to B=1024")
    axes[1].set_title("Internal sampling error remains material")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "e10_full_tournament")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_tables() -> None:
    e3_dependent = pd.read_csv(RUNS / "e3_dependent_confirmation_20260714" / "seed_level.csv")
    e3_dependent.groupby(["cell_id", "method", "split_mode"], as_index=False).agg(
        nrmse=("nrmse", "mean"),
        calibration_slope=("calibration_slope", "mean"),
        iid_coverage=("coverage_fraction_iid", "mean"),
        hac_coverage=("coverage_fraction_hac", "mean"),
        effective_sample_fraction=("effective_sample_fraction", "mean"),
        n_dgp_seeds=("dgp_seed", "nunique"),
    ).to_csv(TAB / "e3_dependence_summary.csv", index=False)
    e8_measurement = pd.read_csv(RUNS / "e8_measurement_confirmation_20260714" / "seed_level.csv")
    e8_measurement.groupby(["cell_id", "method"], as_index=False).agg(
        nrmse=("nrmse", "mean"),
        calibration_slope=("calibration_slope", "mean"),
        coverage=("coverage_fraction", "mean"),
        detection_rate=("detected_bonferroni", "mean"),
        rho_abs_error=("rho_abs_error", "mean"),
        n_dgp_seeds=("dgp_seed", "nunique"),
    ).to_csv(TAB / "e8_measurement_summary.csv", index=False)
    e9_geometry = pd.read_csv(RUNS / "e9_staged_confirmation_20260714" / "seed_level.csv")
    e9_geometry.groupby(["cell_id", "method"], as_index=False).agg(
        covariance_nrmse=("covariance_tensor_nrmse", "mean"),
        feature_nrmse=("feature_tensor_nrmse", "mean"),
        response_angle=("largest_subspace_angle_degrees", "mean"),
        source_angle=("source_subspace_angle_degrees", "mean"),
        rank_error=("rank_absolute_error", "mean"),
        null_fwer=("null_edge_familywise_call", "mean"),
        n_dgp_seeds=("dgp_seed", "nunique"),
    ).to_csv(TAB / "e9_geometry_summary.csv", index=False)
    e10_primary = _e10_primary()
    e10_ratios = e10_paired_ratios(e10_primary)
    e10_ratios.to_csv(TAB / "e10_paired_ratios.csv", index=False)
    e10_means = e10_primary.groupby(["dgp_family", "method"], as_index=False).nrmse.mean()
    e10_means.to_csv(TAB / "e10_primary_nrmse.csv", index=False)
    e10_pivot = e10_means.pivot(index="dgp_family", columns="method", values="nrmse")
    family_labels = {
        "E1_gaussian": "E1 Gaussian",
        "E4_discrete": "E4 discrete",
        "E4_continuous": "E4 continuous",
        "E5_support_motion": "E5 support motion",
        "E8_filtered": "E8 filtered",
        "E9_dense": "E9 dense",
        "E11_ordering": "E11 ordering",
    }
    method_order = [
        "ORTH", "ANALYTIC_MDN_K5", "ANALYTIC_MDN_K10",
        "ANALYTIC_MDN_K20", "NSF_6", "NSF_10",
    ]
    table_lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Family & ORTH & MDN-A5 & MDN-A10 & MDN-A20 & NSF-6 & NSF-10 \\",
        r"\midrule",
    ]
    for family in family_labels:
        values = [float(e10_pivot.loc[family, method]) for method in method_order]
        best = int(np.argmin(values))
        formatted = [f"\\textbf{{{value:.3f}}}" if index == best else f"{value:.3f}" for index, value in enumerate(values)]
        table_lines.append(f"{family_labels[family]} & " + " & ".join(formatted) + r" \\")
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TAB / "e10_primary_table.tex").write_text("\n".join(table_lines) + "\n")

    gate_rows = [
        ("E0", "Algebra, leakage, reproducibility", "PASS", "10/10 checks pass; 14 package tests pass", "Core implementation gate"),
        ("E1", "Equal-access Gaussian recovery", "PASS / NO ADVANTAGE", "ORTH NRMSE 0.104, coverage 0.940; error ratio vs PLUG 1.002", "Inference benefit, not point-error benefit"),
        ("E2", "Controlled orthogonality", "PASS", "Product-bias R² > 0.99999997 in three perturbations", "Theory realized in population experiment"),
        ("E3", "One future/high-dimensional history", "PASS WITH HAC", "Blocked rho=.9 NRMSE 0.145/0.093; HAC coverage 0.949/0.951", "Blocked splitting and dependence-aware SE required"),
        ("E4", "Moment-blind law change", "PASS AFTER AMENDMENT", "Power 1.00/0.867; moment FWER 0.067/0.067", "High-frequency bank required; default bank failed"),
        ("E5", "Support motion/low-noise limit", "PASS", "ORTH NRMSE 0.016 at σ=0; likelihood score undefined", "Strong distinctive result"),
        ("E6", "Riesz stability and honest failure", "FAIL", "Coverage 0 for t3 and gaps 3–4; warning calibration imperfect", "No general observational-identification claim"),
        ("E7", "Heterogeneous cancellation", "PASS AT R=2", "Opposite signs in 100%; energy 3.982 vs truth 4.000", "Declare history-basis resolution"),
        ("E8", "Observation process", "FAIL", "Moderate noise/missingness works; rho=.98 and misspecified deconvolution fail", "Hard latent-deployment gate remains closed"),
        ("E9", "Dense unknown geometry", "FAIL", "Feature NRMSE 1.56–5.95; source angles remain 58–81 degrees", "Use typed head, but do not claim source recovery"),
        ("E10", "Flexible reusable-law baselines", "PASS POOLED / LOCAL", "Pooled neural/ORTH ratios 1.57–1.77; exact MDN readout changes family rankings", "Aggregate advantage, not universal dominance or a scale frontier"),
        ("E11", "Path ordering", "PASS", "Full-path power 1.00, NRMSE 0.209; endpoint/unordered ORTH FWER 0.056", "Second distinctive capability"),
    ]
    gates = pd.DataFrame(gate_rows, columns=["experiment", "question", "decision", "evidence", "consequence"])
    gates.to_csv(TAB / "gate_matrix.csv", index=False)
    claim_rows = [
        ("Orthogonal score is implemented correctly", "SUPPORTED", "E0,E2", "Exact algebra and population perturbations"),
        ("Declared bounded feature tangents are estimable in regular synthetic cells", "SUPPORTED", "E1,E3,E5", "IID and dependent Gaussian plus support-motion families"),
        ("ORTH improves point accuracy over PLUG/RIESZ", "NOT SUPPORTED", "E1,E4", "Paired error ratios include one"),
        ("ORTH supplies usable uncertainty where PLUG does not", "SUPPORTED IN REGULAR CELLS", "E1,E3,E4,E8,E11", "Requires HAC under dependence and fails under ill-posed deconvolution"),
        ("Finite characteristic banks add information beyond four moments", "SUPPORTED AT DECLARED BANK", "E4", "Two exact moment-blind families"),
        ("Weak-feature route survives deterministic support motion", "SUPPORTED", "E5", "Stable through sigma zero"),
        ("Current Riesz sieve is robust to heavy tails and weak overlap", "REFUTED", "E6", "Tail/gap coverage failures"),
        ("Global zero means no local effect", "REFUTED", "E7", "R=2 recovers sign reversal and energy"),
        ("Current method is ready for filtered neural observations", "REFUTED", "E8", "Severe filtering and observation-model misspecification fail"),
        ("Characteristic route is best for unknown dense geometry", "NOT SUPPORTED", "E9", "Direct covariance head is stronger"),
        ("ORTH dominates flexible normalized-law models", "SUPPORTED POOLED AT LOCAL BUDGET", "E10", "Every pooled ratio passes; E1/E5/E9 have no universal ORTH winner"),
        ("Path-aware features detect order changes invisible to endpoints", "SUPPORTED", "E11", "Generic path bank plus two null summaries"),
    ]
    pd.DataFrame(claim_rows, columns=["claim", "decision", "evidence", "scope"]).to_csv(TAB / "claim_registry.csv", index=False)
    ledgers = []
    selected_runs = [
        "e1_confirmation_20260714", "e3_confirmation_20260714", "e4_confirmation_20260714",
        "e5_confirmation_20260714", "e6_confirmation_20260714", "e7_confirmation_20260714",
        "e8_confirmation_20260714", "e9_confirmation_20260714", "e10_confirmation_20260714",
        "e11_confirmation_20260714",
        "e3_dependent_confirmation_20260714", "e8_measurement_confirmation_20260714",
        "e9_staged_confirmation_20260714", "e10_full_confirmation_20260714",
        "e10_analytic_mdn_confirmation_20260714",
    ]
    for name in selected_runs:
        directory = RUNS / name
        summary = directory / "summary.csv"
        validation = json.loads((directory / "validation.json").read_text())
        ledgers.append(
            {
                "run": name,
                "summary_rows": len(pd.read_csv(summary)),
                "validation_passed": validation.get("passed", False),
                "summary_sha256": sha256(summary),
                "size_bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
            }
        )
    pd.DataFrame(ledgers).to_csv(TAB / "artifact_ledger.csv", index=False)

    e1 = read_summary("e1_confirmation_20260714")
    e4 = pd.read_csv(RUNS / "e4_confirmation_20260714" / "detection_summary.csv")
    e5 = read_summary("e5_confirmation_20260714")
    e9 = read_summary("e9_confirmation_20260714")
    macros = {
        "EOneOrthNrmse": point(e1, "nrmse", "ORTH", target_name="characteristic_tangent")["mean"],
        "EOneOrthCoverage": point(e1, "coverage_fraction", "ORTH", target_name="characteristic_tangent")["mean"],
        "EOneDirectMeanNrmse": point(e1, "nrmse", "DIRECT", target_name="mean_derivative")["mean"],
        "EOneDirectCovNrmse": point(e1, "nrmse", "DIRECT", target_name="covariance_derivative")["mean"],
        "EFourDiscretePower": e4.loc[e4.dgp_family == "finite_difference_mixture", "characteristic_power"].iloc[0],
        "EFourContinuousPower": e4.loc[e4.dgp_family == "continuous_legendre_tilt", "characteristic_power"].iloc[0],
        "EFiveZeroNrmse": point(e5, "nrmse", "ORTH", cell_id="sigma_0p0")["mean"],
        "ENineDirectAngle": point(e9, "largest_subspace_angle_degrees", "DIRECT_COV_ORTH", cell_id="medium_dense_rotated_K3")["mean"],
        "ENineCharAngle": point(e9, "largest_subspace_angle_degrees", "ORTH_CHARACTERISTIC_INVERSION", cell_id="medium_dense_rotated_K3")["mean"],
    }
    (TAB / "generated_metrics.tex").write_text(
        "\n".join(f"\\newcommand{{\\{name}}}{{{value:.3f}}}" for name, value in macros.items()) + "\n"
    )
    snapshot = {
        "decision": "no-go for general deployment; go for restricted predictive-feature research claims",
        "gate_matrix": gates.to_dict("records"),
        "claim_registry": pd.DataFrame(claim_rows, columns=["claim", "decision", "evidence", "scope"]).to_dict("records"),
        "selected_run_count": len(selected_runs),
    }
    (HERE / "results_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")


def main() -> None:
    e1_figure()
    e2_e3_figure()
    e4_figure()
    e5_figure()
    e6_figure()
    e7_e11_figure()
    e8_e9_e10_figure()
    completion_gate_figure()
    e10_full_figure()
    make_tables()


if __name__ == "__main__":
    main()
