#!/usr/bin/env python3
"""Create report-quality figures from validated Tier-1/Tier-2 summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


FAMILIES = [
    "m1_mean",
    "m2_covariance",
    "m3_cubic",
    "m3_bounded",
    "m3_local",
    "m3_asymmetric",
    "m4_quartic",
    "m4_tail",
    "m5_occupancy",
]
FAMILY_LABELS = [
    "M1\nmean",
    "M2\ncovariance",
    "M3\ncubic",
    "M3\nbounded",
    "M3\nlocal",
    "M3\nasymmetric",
    "M4\nquartic",
    "M4\ntail",
    "M5\noccupancy",
]
METHODS = [
    "S1_NLL_SCORE__A1_DENSITY_AD",
    "B2_RAW_MOMENTS",
    "A2_RATIO_CRITIC",
    "S2_HYVARINEN__A7_HODGE",
    "S5_DSM_MULTI__A7_HODGE",
    "S6_GAUSS_DSM__A3_CENTER",
    "S7_ANCHORED_ENERGY__A3_CENTER",
    "B0_ZERO",
]
FINITE_METHODS = [
    "S1_NLL_ENDPOINTS",
    "B2_RAW_MOMENTS_FINITE",
    "DIRECT_ENDPOINT_MOMENTS",
    "A8_ENDPOINT_CLASSIFIER",
    "A10_SIGNED_RIESZ",
    "S7_SCORE_INTEGRATED_ENDPOINTS",
    "B0_ZERO",
]
LABEL = {
    "S1_NLL_SCORE__A1_DENSITY_AD": "Matched normalized law",
    "S1_NLL_ENDPOINTS": "Matched normalized endpoints",
    "B2_RAW_MOMENTS": "Direct moments",
    "B2_RAW_MOMENTS_FINITE": "Direct moments (finite)",
    "DIRECT_ENDPOINT_MOMENTS": "Endpoint Monte Carlo",
    "A2_RATIO_CRITIC": "Ratio critic",
    "S2_HYVARINEN__A7_HODGE": "Clean Hyvarinen + Hodge",
    "S5_DSM_MULTI__A7_HODGE": "Multi-noise DSM + Hodge",
    "S6_GAUSS_DSM__A3_CENTER": "Gaussian score control",
    "S7_ANCHORED_ENERGY__A3_CENTER": "Anchored score",
    "S7_SCORE_INTEGRATED_ENDPOINTS": "Integrated anchored score",
    "A8_ENDPOINT_CLASSIFIER": "Endpoint classifier",
    "A10_SIGNED_RIESZ": "Signed Riesz",
    "B0_ZERO": "Zero",
    "B1_MEAN_POLY": "Mean regression",
    "B4_GAUSSIAN_NLL": "Gaussian likelihood",
    "B4_GAUSSIAN_LOGVAR": "Gaussian log-variance",
    "B7_MDN_CORRECT": "Matched gain mixture",
    "S7_ANCHORED_ENERGY__A7_HODGE": "Anchored score + Hodge",
}
COLORS = {
    "truth": "#1A1A1A",
    "B2_RAW_MOMENTS": "#2B73B7",
    "B4_GAUSSIAN_LOGVAR": "#3B8EA5",
    "B7_MDN_CORRECT": "#238B68",
    "A2_RATIO_CRITIC": "#7651A8",
    "S2_HYVARINEN__A7_HODGE": "#D18A27",
    "S7_ANCHORED_ENERGY__A7_HODGE": "#D85C41",
    "S7_ANCHORED_ENERGY__A3_CENTER": "#D85C41",
    "S1_NLL_SCORE__A1_DENSITY_AD": "#238B68",
}


def save(fig: mpl.figure.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heatmap(
    summary: pd.DataFrame,
    estimand: str,
    methods: list[str],
    title: str,
    output: Path,
    name: str,
    vmax: float,
) -> None:
    data = summary[(summary.estimand_type == estimand) & (summary.metric_name == "nrmse")]
    value = data.pivot_table(index="method_id", columns="dgp_family", values="mean")
    upper = data.pivot_table(index="method_id", columns="dgp_family", values="ci_upper")
    matrix = value.reindex(index=methods, columns=FAMILIES).to_numpy(float)
    upper_matrix = upper.reindex(index=methods, columns=FAMILIES).to_numpy(float)
    masked = np.ma.masked_invalid(matrix)
    fig, axis = plt.subplots(figsize=(12.2, 5.2))
    cmap = mpl.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("#EEEEEE")
    image = axis.imshow(masked, aspect="auto", norm=LogNorm(vmin=0.05, vmax=vmax), cmap=cmap)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            number = matrix[row, column]
            if not np.isfinite(number):
                axis.text(column, row, "N/A", ha="center", va="center", color="#888888", fontsize=7)
                continue
            star = "*" if upper_matrix[row, column] < 1.0 else ""
            color = "white" if number < 0.13 or number > 3.0 else "#222222"
            axis.text(column, row, f"{number:.2f}{star}", ha="center", va="center", color=color, fontsize=8, weight="bold" if star else "normal")
    axis.set_xticks(range(len(FAMILIES)), FAMILY_LABELS, fontsize=8)
    axis.set_yticks(range(len(methods)), [LABEL.get(x, x) for x in methods], fontsize=8)
    axis.set_title(title, loc="left", fontsize=13, weight="bold")
    axis.set_xlabel("* upper 95% DGP-bootstrap NRMSE bound < 1")
    colorbar = fig.colorbar(image, ax=axis, fraction=.027, pad=.02)
    colorbar.set_label("Mean NRMSE (log color scale)")
    axis.spines[:].set_visible(False)
    save(fig, output, name)


def derivative_gap(summary: pd.DataFrame, output: Path) -> None:
    methods = ["S2_HYVARINEN", "S5_DSM_MULTI", "S6_GAUSS_DSM", "S7_ANCHORED_ENERGY"]
    families = [x for x in FAMILIES if x != "m2_covariance"]
    labels = [FAMILY_LABELS[FAMILIES.index(x)] for x in families]
    metric_titles = [
        ("response_score_nrmse", "Response score"),
        ("history_tangent_nrmse", "Reconstructed history tangent"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 6.2), sharex=True)
    norm = LogNorm(vmin=0.03, vmax=15.0)
    cmap = mpl.colormaps["RdYlBu_r"]
    for axis, (metric, title) in zip(axes, metric_titles, strict=True):
        panel = summary[(summary.estimand_type == "operator") & (summary.metric_name == metric)]
        values = panel.pivot_table(index="method_id", columns="dgp_family", values="mean").reindex(index=methods, columns=families)
        uppers = panel.pivot_table(index="method_id", columns="dgp_family", values="ci_upper").reindex(index=methods, columns=families)
        matrix = values.to_numpy(float)
        image = axis.imshow(matrix, aspect="auto", norm=norm, cmap=cmap)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                number = matrix[row, column]
                star = "*" if uppers.iloc[row, column] < 1 else ""
                color = "white" if number < .07 or number > 4 else "#222222"
                axis.text(column, row, f"{number:.2f}{star}", ha="center", va="center", fontsize=8, color=color)
        axis.set_yticks(range(len(methods)), [x.replace("_", " ").title() for x in methods], fontsize=8)
        axis.set_title(title, loc="left", fontsize=10, weight="bold")
        axis.spines[:].set_visible(False)
    axes[-1].set_xticks(range(len(families)), labels, fontsize=8)
    axes[-1].set_xlabel("* upper 95% bound < 1; identical fitted score model in both panels")
    fig.suptitle("The response-score-to-history-tangent derivative gap", x=.1, ha="left", fontsize=14, weight="bold")
    colorbar = fig.colorbar(image, ax=axes, fraction=.02, pad=.02)
    colorbar.set_label("Mean NRMSE (log scale)")
    save(fig, output, "derivative_gap_heatmap")


def dynamic_forest(summary: pd.DataFrame, output: Path) -> None:
    panels = [
        ("d1_gaussian_lag", "mean", "D1 mean"),
        ("d2_observed_stochastic_gain", "variance", "D2 variance"),
        ("d2_observed_stochastic_gain", "covariance", "D2 covariance"),
        ("d2_observed_stochastic_gain", "third_cumulant", "D2 third cumulant"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.4), sharex=True)
    for axis, (family, channel, title) in zip(axes, panels, strict=True):
        data = summary[
            (summary.dgp_family == family)
            & (summary.channel == channel)
            & (summary.estimand_type == "dynamic_local")
            & (summary.metric_name == "nrmse")
        ].sort_values("mean")
        for y, (_, row) in enumerate(data.iterrows()):
            method = row.method_id
            color = COLORS.get(method, "#777777")
            axis.errorbar(row["mean"], y, xerr=[[row["mean"] - row.ci_lower], [row.ci_upper - row["mean"]]], fmt="o", color=color, capsize=2, ms=5)
        axis.set_yticks(range(len(data)), [LABEL.get(x, x) for x in data.method_id], fontsize=7)
        axis.axvline(1, color="#999999", ls="--", lw=1)
        axis.set_title(title, fontsize=10, weight="bold")
        axis.grid(axis="x", color="#E6E6E6")
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_xlim(0, 1.48)
    fig.supxlabel("Mean lag-profile NRMSE (95% DGP-bootstrap CI); dashed line = zero estimator")
    fig.suptitle("Dynamic local recovery", x=.06, ha="left", fontsize=14, weight="bold")
    save(fig, output, "dynamic_nrmse")


def bootstrap_curve(values: np.ndarray, seed: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.shape[0], (2000, values.shape[0]))].mean(axis=1)
    return values.mean(axis=0), np.quantile(draws, .025, axis=0), np.quantile(draws, .975, axis=0)


def dynamic_profiles(run: Path, output: Path) -> None:
    files = sorted((run / "cases" / "dynamic_confirmatory").glob("*.npz"))
    methods = [
        ("truth", "truth"),
        ("B2_RAW_MOMENTS", "b2_raw_moments"),
        ("B7_MDN_CORRECT", "b7_mdn_correct"),
        ("A2_RATIO_CRITIC", "a2_ratio_critic"),
        ("S7_ANCHORED_ENERGY__A7_HODGE", "s7_anchored_energy__a7_hodge"),
    ]
    channels = ["variance", "covariance", "third_cumulant"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for panel_index, (axis, channel) in enumerate(zip(axes, channels, strict=True)):
        for method, key in methods:
            array_key = f"d2__{channel}__{key}__profile" if key != "truth" else f"d2__{channel}__truth_matrix"
            values = []
            for file in files:
                with np.load(file) as archive:
                    if array_key not in archive:
                        continue
                    value = archive[array_key]
                    values.append(value.mean(axis=0) if value.ndim == 2 else value)
            if not values:
                continue
            mean, lower, upper = bootstrap_curve(np.stack(values), seed=40 + panel_index)
            x = np.arange(1, mean.size + 1)
            color = COLORS.get(method, COLORS["truth"])
            label = "Oracle" if method == "truth" else LABEL.get(method, method)
            axis.plot(x, mean, marker="o", ms=3, lw=2 if method == "truth" else 1.4, color=color, label=label)
            axis.fill_between(x, lower, upper, color=color, alpha=.13, lw=0)
        axis.axhline(0, color="#AAAAAA", lw=.8)
        axis.set_title(channel.replace("_", " ").title(), fontsize=10, weight="bold")
        axis.set_xlabel("Source lag")
        axis.grid(color="#ECECEC")
    axes[0].set_ylabel("Typed local effect")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 0.0), ncol=5, frameon=False, fontsize=8)
    fig.suptitle("D2 source-to-target lag profiles", x=.06, ha="left", fontsize=14, weight="bold")
    save(fig, output, "dynamic_lag_profiles")


def topology(run: Path, output: Path) -> None:
    data = pd.read_csv(run / "stage0" / "topology_hodge.csv")
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(data.shape[0])
    axis.plot(x, data.hodge_tangent_nrmse, "o-", color="#2B73B7", lw=1.7)
    axis.set_yscale("log")
    axis.set_xticks(x, [f"{value:g}" for value in data.epsilon])
    axis.set(xlabel="Bridge parameter epsilon (0 = disconnected)", ylabel="Oracle Hodge tangent NRMSE")
    axis.set_title("Hodge reconstruction requires connected support", loc="left", fontsize=13, weight="bold")
    axis.grid(color="#E8E8E8", which="both")
    save(fig, output, "topology_hodge")


def g8_figure(summary_path: Path, output: Path) -> None:
    data = pd.read_csv(summary_path)
    keep = ["oracle_bayes_witness", "mlp_motif", "typed_quartic", "polynomial4_motif", "mlp_full_path_history", "quadratic_motif", "linear_motif"]
    data = data.set_index("estimator").reindex(keep).reset_index()
    labels = [x.replace("_", " ").title() for x in data.estimator]
    y = np.arange(data.shape[0])
    fig, axis = plt.subplots(figsize=(8.8, 4.4))
    axis.hlines(y, data.typed_effect_relative_error_median, data.typed_effect_relative_error_max, color="#B0B0B0", lw=3, label="median to max")
    axis.plot(data.typed_effect_relative_error_median, y, "o", color="#2B73B7", label="median")
    axis.axvline(1.0, color="#999999", ls="--")
    axis.set_yticks(y, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set(xlabel="Typed-effect relative error", xscale="log")
    axis.set_title("G8 finite contrast rewards representation alignment", loc="left", fontsize=13, weight="bold")
    axis.grid(axis="x", color="#E8E8E8", which="both")
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    save(fig, output, "g8_typed_classifier")


def tier2_dynamic(summary: pd.DataFrame, output: Path) -> None:
    data = summary[(summary.study_component == "dynamic_robustness") & (summary.channel == "variance") & (summary.metric_name == "nrmse")]
    methods = ["B4_GAUSSIAN_LOGVAR", "B2_RAW_MOMENTS", "A2_RATIO_CRITIC", "S7_ANCHORED_ENERGY__A7_HODGE"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    context = data[data.cell_id.str.contains(r"d3_context_(?:2|12|20)__n_8000", regex=True)].copy()
    context["x"] = context.cell_id.str.extract(r"context_(\d+)", expand=False).astype(int)
    sample = data[data.cell_id.str.contains(r"d3_context_12__n_", regex=True)].copy()
    sample["x"] = sample.n_train.astype(int)
    d5 = data[data.cell_id.str.startswith("d5_")].copy()
    d5 = d5[~d5.cell_id.str.contains("null")]
    d5["x"] = d5.cell_id.map({"d5_clean__n_8000": 0, "d5_primary_filter__n_8000": 1, "d5_strong_filter__n_8000": 2})
    for axis, panel, title, xlabel in [
        (axes[0], context, "Hidden state: context", "Observed context length"),
        (axes[1], sample, "Hidden state: sample size", "Training examples"),
        (axes[2], d5, "Observation filtering", "Filter regime"),
    ]:
        for method in methods:
            lines = panel[panel.method_id == method].sort_values("x")
            if lines.empty:
                continue
            axis.errorbar(lines.x, lines["mean"], yerr=[lines["mean"] - lines.ci_lower, lines.ci_upper - lines["mean"]], marker="o", lw=1.5, capsize=2, color=COLORS[method], label=LABEL[method])
        axis.axhline(1, color="#999999", ls="--", lw=1)
        axis.set_title(title, fontsize=10, weight="bold")
        axis.set_xlabel(xlabel)
        axis.grid(color="#ECECEC")
    axes[1].set_xscale("log")
    axes[0].set_xticks([2, 12, 20], ["2", "12", "20"])
    axes[1].set_xticks([2_000, 8_000, 32_000], ["2k", "8k", "32k"])
    axes[1].get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    axes[2].set_xticks([0, 1, 2], ["clean", "primary", "strong"], fontsize=8)
    axes[0].set_ylabel("Variance lag-profile NRMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 0.0), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Tier 2 dynamic robustness", x=.06, ha="left", fontsize=14, weight="bold")
    save(fig, output, "tier2_dynamic")


def tier2_static(summary: pd.DataFrame, output: Path) -> None:
    registered = summary[
        (summary.study_component == "static_slices")
        & (summary.estimand_type == "local")
        & (summary.metric_name == "nrmse")
    ]
    fixed = summary[
        (summary.study_component == "fixed_amplitude_diagnostic")
        & (summary.estimand_type == "local")
        & (summary.metric_name == "nrmse")
    ]
    mechanisms = ["m3_bounded", "m4_quartic", "m5_occupancy"]
    methods = ["S1_NLL_SCORE__A1_DENSITY_AD", "B2_RAW_MOMENTS", "A2_RATIO_CRITIC", "S2_HYVARINEN__A7_HODGE", "S7_ANCHORED_ENERGY__A3_CENTER"]
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 9.4), sharey="row")
    for column, mechanism in enumerate(mechanisms):
        sample = registered[
            (registered.dgp_family == mechanism)
            & registered.slice_name.astype(str).str.startswith("sample")
        ]
        fixed_sample = fixed[fixed.dgp_family == mechanism]
        info = registered[
            (registered.dgp_family == mechanism)
            & registered.slice_name.astype(str).str.startswith("information")
        ]
        base = registered[
            (registered.dgp_family == mechanism)
            & (registered.slice_name == "sample_n_8000")
        ]
        info = pd.concat([info, base], ignore_index=True)
        for row, (axis, panel, xname, xlabel) in enumerate([
            (axes[0, column], sample, "n_train", r"Training examples (target $\Lambda=3$)"),
            (axes[1, column], fixed_sample, "n_train", "Training examples (fixed amplitude)"),
            (axes[2, column], info, "achieved_lambda", r"Achieved information index ($N=8$k)"),
        ]):
            for method in methods:
                lines = panel[panel.method_id == method].drop_duplicates([xname]).sort_values(xname)
                if lines.empty:
                    continue
                axis.plot(lines[xname], lines["mean"], marker="o", lw=1.3, color=COLORS.get(method, "#777777"), label=LABEL[method])
            axis.axhline(1, color="#999999", ls="--", lw=1)
            axis.set_xscale("log")
            axis.set_yscale("log")
            if row in {0, 1}:
                axis.set_xticks([2_000, 8_000, 32_000], ["2k", "8k", "32k"])
                axis.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
            else:
                ticks = np.sort(panel[xname].dropna().unique().astype(float))
                axis.set_xticks(ticks, [f"{value:.3g}" for value in ticks])
                axis.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
            axis.set_xlabel(xlabel)
            axis.grid(color="#ECECEC", which="both")
            if row == 0:
                axis.set_title(FAMILY_LABELS[FAMILIES.index(mechanism)].replace("\n", " "), fontsize=10, weight="bold")
    axes[0, 0].set_ylabel("Local NRMSE")
    axes[1, 0].set_ylabel("Local NRMSE")
    axes[2, 0].set_ylabel("Local NRMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 0.0), ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Static efficiency, learning, and information slices", x=.06, ha="left", fontsize=14, weight="bold")
    save(fig, output, "tier2_static_slices")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1-summary", required=True, type=Path)
    parser.add_argument("--tier1-run", required=True, type=Path)
    parser.add_argument("--tier2-summary", required=True, type=Path)
    parser.add_argument("--g8-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelcolor": "#222222", "text.color": "#222222"})
    tier1 = pd.read_csv(args.tier1_summary)
    tier2 = pd.read_csv(args.tier2_summary)
    output = args.output.resolve()
    heatmap(tier1, "local", METHODS, "Local SID: mean NRMSE by mechanism and method", output, "static_local_heatmap", 10)
    heatmap(tier1, "finite", FINITE_METHODS, "Finite contrast: mean NRMSE by mechanism and method", output, "static_finite_heatmap", 35)
    derivative_gap(tier1, output)
    dynamic_forest(tier1, output)
    dynamic_profiles(args.tier1_run.resolve(), output)
    topology(args.tier1_run.resolve(), output)
    g8_figure(args.g8_summary.resolve(), output)
    tier2_dynamic(tier2, output)
    tier2_static(tier2, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
