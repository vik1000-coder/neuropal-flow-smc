"""Build the small, frozen figure set for the simple SID teaching guide.

Chart contracts
---------------
1. four_systems_schematic
   Decision: understand which part of P(Y|H) changes in G1--G4.
   Comparison: two nearby histories within each synthetic system.
   Metric/dimension/grain: schematic densities or component weights; one panel per DGP.
   Time/filter: timeless; G1--G4 only; explicitly not an empirical result.

2. simple_recovery_results
   Decision: determine whether the best current route recovers each simple SID target.
   Comparison: best clean infinitesimal route versus best clean finite typed route.
   Metric: NRMSE (lower is better; 1 is the zero-effect baseline).
   Dimension/grain: generator; eight frozen generator/data/model seed cells per bar.
   Time/filter: July 13 frozen core and finite runs; delta=0.12 for finite results.

3. diffusion_score_vs_sid
   Decision: test whether accurate response-score estimation implies accurate history SID.
   Comparison: legacy diffusion noisy response score versus its model-centered history tangent.
   Metric: NRMSE; dimension/grain: generator, eight frozen cells per bar.
   Time/filter: July 13 frozen core; standardized response noise sigma=0.12.

4. law_fit_improvement
   Decision: decide when a flexible full-law model materially improves on Gaussian NLL.
   Comparison: winning normalized likelihood model versus heteroscedastic Gaussian.
   Metric: median NLL reduction in nats/response vector; dimension: generator.
   Time/filter: July 13 frozen core; G1--G4.

5. g8_effect_ladder
   Decision: determine which design choice preserves the hard fourth-order G8 effect.
   Comparison: full-law generators, repairs, and direct contrast classifiers.
   Metric: median effect slope (1 is correct, 0 misses the effect).
   Dimension/grain: method family; best named member in each family, eight cells except
   oracle-midpoint guidance (four cells).
   Time/filter: July 13 frozen G8 audits.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CORE = ROOT / "history_tangent_benchmark/results/stable_sid_core_20260713/metrics.csv"
FINITE = ROOT / "history_tangent_benchmark/results/stable_sid_finite_20260713/finite_contrast_metrics.csv"
G8_CLASSIFIER = ROOT / "analysis/g8_typed_classifier_benchmark_2026-07-13/case_metrics.csv"
G8_GUIDED = ROOT / "analysis/g8_classifier_guided_generator_2026-07-13/model_summary.csv"
G8_EXPERT = ROOT / "analysis/g8_endpoint_expert_decomposition_2026-07-13/model_summary.csv"


NAVY = "#1f3b5b"
BLUE = "#3f7cac"
TEAL = "#2a9d8f"
ORANGE = "#e07a3f"
RED = "#c94c4c"
GOLD = "#d8a72e"
GRAY = "#6b7280"
LIGHT = "#e8edf2"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9aa4af",
        "axes.linewidth": 0.8,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "text.color": "#17212b",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    plt.close(fig)


def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def four_systems_schematic() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.7))
    ax = axes[0, 0]
    x = np.linspace(-3, 3, 500)
    ax.plot(x, normal_pdf(x, -0.75, 0.72), color=BLUE, lw=2.2, label=r"$h_-$")
    ax.plot(x, normal_pdf(x, 0.75, 0.72), color=ORANGE, lw=2.2, label=r"$h_+$")
    ax.set_title("G1: location changes")
    ax.set_xlabel("response coordinate")
    ax.set_yticks([])
    ax.legend(frameon=False, ncol=2, loc="upper right")

    ax = axes[0, 1]
    for width, height, angle, color, label in [
        (3.7, 1.25, 28, BLUE, r"$h_-$"),
        (1.25, 3.7, 28, ORANGE, r"$h_+$"),
    ]:
        ax.add_patch(Ellipse((0, 0), width, height, angle=angle, fill=False, lw=2.4,
                             edgecolor=color, label=label))
    ax.scatter([0], [0], s=24, color=NAVY, zorder=3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_title("G2: covariance changes")
    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.02, 0.05, "same center; rotated spread", transform=ax.transAxes, color=GRAY)

    ax = axes[1, 0]
    x = np.linspace(-4.2, 3.2, 800)
    means = np.array([-3.0, -1.0, 1.0, 2.0])
    base = np.full(4, 0.25)
    direction = np.array([-0.2, 2 / 3, -1.0, 8 / 15])
    for sign, color, label in [(-1, BLUE, r"$h_-$"), (1, ORANGE, r"$h_+$")]:
        weights = base + sign * 0.08 * direction
        density = sum(w * normal_pdf(x, m, 0.25) for w, m in zip(weights, means))
        ax.plot(x, density, color=color, lw=2.2, label=label)
    ax.set_title("G3: skew changes, mean/variance fixed")
    ax.set_xlabel("response coordinate")
    ax.set_yticks([])
    ax.legend(frameon=False, ncol=2, loc="upper right")

    ax = axes[1, 1]
    centers = np.array([[-1.55, -0.75], [-0.75, 1.25], [0.75, -1.25], [1.55, 0.75]])
    wminus = np.array([0.40, 0.31, 0.18, 0.11])
    wplus = np.array([0.10, 0.18, 0.31, 0.41])
    ax.scatter(centers[:, 0] - 0.07, centers[:, 1], s=1300 * wminus, color=BLUE,
               alpha=0.55, edgecolor="white", linewidth=1.2, label=r"$h_-$")
    ax.scatter(centers[:, 0] + 0.07, centers[:, 1], s=1300 * wplus, color=ORANGE,
               alpha=0.55, edgecolor="white", linewidth=1.2, label=r"$h_+$")
    ax.set_title("G4: separated-mode occupancy changes")
    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.0, 2.0)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.02, 0.05, "circle area represents mixture weight", transform=ax.transAxes, color=GRAY)

    for panel, label in zip(axes.flat, ["A", "B", "C", "D"]):
        panel.text(-0.13, 1.05, label, transform=panel.transAxes, fontsize=12,
                   fontweight="bold", color=NAVY)
    fig.suptitle("What changes between two nearby histories? (schematic)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.01)
    fig.tight_layout(h_pad=2.0, w_pad=1.8)
    save(fig, "four_systems_schematic")


def median_iqr(values: pd.Series) -> tuple[float, float, float]:
    med = float(values.median())
    return med, med - float(values.quantile(0.25)), float(values.quantile(0.75)) - med


def simple_recovery_results() -> None:
    core = pd.read_csv(CORE)
    finite = pd.read_csv(FINITE, low_memory=False)
    ids = ["g1_gaussian_sanity", "g2_covariance", "g3_matched_skew", "g4_separated_modes"]
    core_rows = core[
        (core.model_name == "ratio_critic")
        & (core.metric_id == "tangent_nrmse")
        & (core.estimand == "clean")
        & core.generator_id.isin(ids)
    ]
    finite_specs = [
        ("g1_location", "diffusion_gaussian_anchored", "direct_coupled_samples"),
        ("g2_covariance", "flow_matching", "direct_coupled_samples"),
        ("g3_skew", "diffusion_legacy", "signed_riesz_dictionary"),
        ("g4_low_rank_mixture", "autoregressive_mdn", "central_model_log_ratio"),
    ]
    labels = ["G1\nlocation", "G2\ncovariance", "G3\nskew", "G4\nmodes"]
    core_values, finite_values = [], []
    for core_id, (fg, fm, route) in zip(ids, finite_specs):
        core_values.append(core_rows[core_rows.generator_id == core_id].value.to_numpy())
        finite_values.append(
            finite[
                (finite.generator_id == fg)
                & (finite.model_name == fm)
                & (finite.method == route)
                & (finite.metric_id == "channel_readout_nrmse")
                & (finite.delta == 0.12)
                & (finite.law == "clean")
            ].value.to_numpy()
        )

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(4)
    width = 0.34
    for offset, arrays, color, name in [
        (-width / 2, core_values, BLUE, "best clean tangent (ratio critic)"),
        (width / 2, finite_values, ORANGE, r"best finite typed route ($\delta=0.12$)"),
    ]:
        stats = [median_iqr(pd.Series(v)) for v in arrays]
        med = np.array([s[0] for s in stats])
        err = np.array([[s[1] for s in stats], [s[2] for s in stats]])
        ax.bar(x + offset, med, width, color=color, alpha=0.92, label=name, zorder=2)
        ax.errorbar(x + offset, med, yerr=err, fmt="none", ecolor="#1f2937",
                    capsize=3, lw=1, zorder=3)
        for i, vals in enumerate(arrays):
            jitter = np.linspace(-0.065, 0.065, len(vals))
            ax.scatter(np.full(len(vals), x[i] + offset) + jitter, vals, s=13,
                       color="#17212b", alpha=0.52, linewidth=0, zorder=4)
        for xpos, value in zip(x + offset, med):
            ax.text(xpos, value + 0.055, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(1.0, ls="--", color=RED, lw=1.5, label="zero-effect baseline")
    ax.fill_between([-0.6, 3.6], 1.0, 2.25, color=RED, alpha=0.045, zorder=0)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 2.18)
    ax.set_xticks(x, labels)
    ax.set_ylabel("NRMSE (lower is better)")
    ax.set_title("Three simple systems are recovered; matched skew is not",
                 loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="y", color=LIGHT, lw=0.8, zorder=0)
    ax.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.14), fontsize=8.5)
    ax.text(3.52, 1.03, "no-effect predictor", ha="right", va="bottom", color=RED, fontsize=8.5)
    fig.tight_layout()
    save(fig, "simple_recovery_results")


def diffusion_score_vs_sid() -> None:
    core = pd.read_csv(CORE)
    ids = ["g1_gaussian_sanity", "g2_covariance", "g3_matched_skew", "g4_separated_modes"]
    labels = ["G1\nlocation", "G2\ncovariance", "G3\nskew", "G4\nmodes"]
    rows = core[(core.model_name == "diffusion_legacy") & core.generator_id.isin(ids)]
    collections = []
    for metric, centering in [("response_score_nrmse", "none"), ("tangent_nrmse", "model_samples")]:
        collections.append([
            rows[(rows.generator_id == gid) & (rows.metric_id == metric)
                 & (rows.estimand == "noisy") & (rows.centering == centering)].value.to_numpy()
            for gid in ids
        ])

    fig, ax = plt.subplots(figsize=(8.2, 4.35))
    x = np.arange(4)
    width = 0.34
    for offset, arrays, color, name in [
        (-width / 2, collections[0], TEAL, "noisy response score"),
        (width / 2, collections[1], GOLD, "model-centered history tangent"),
    ]:
        stats = [median_iqr(pd.Series(v)) for v in arrays]
        med = np.array([s[0] for s in stats])
        err = np.array([[s[1] for s in stats], [s[2] for s in stats]])
        ax.bar(x + offset, med, width, color=color, alpha=0.95, label=name, zorder=2)
        ax.errorbar(x + offset, med, yerr=err, fmt="none", ecolor="#1f2937",
                    capsize=3, lw=1, zorder=3)
        for xpos, value in zip(x + offset, med):
            ax.text(xpos, value + 0.07, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(1.0, ls="--", color=RED, lw=1.5, label="zero-effect baseline")
    ax.set_xticks(x, labels)
    ax.set_ylabel("NRMSE (lower is better)")
    ax.set_ylim(0, 2.75)
    ax.grid(axis="y", color=LIGHT, lw=0.8, zorder=0)
    ax.set_title("A good diffusion response score does not guarantee good SID",
                 loc="left", fontweight="bold", color=NAVY)
    ax.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.14), fontsize=8.5)
    ax.annotate("score is useful,\nSID still fails", xy=(2.17, 2.39), xytext=(2.72, 2.48),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1), color=RED,
                ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "diffusion_score_vs_sid")


def law_fit_improvement() -> None:
    # Frozen paired-median NLL gains from stable_results.tex; positive here means improvement.
    values = np.array([0.118, 0.227, 0.665, 2.556])
    labels = ["G1 location", "G2 covariance", "G3 skew", "G4 modes"]
    colors = [BLUE, BLUE, TEAL, ORANGE]
    fig, ax = plt.subplots(figsize=(7.5, 3.45))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, alpha=0.92)
    for yi, value in zip(y, values):
        ax.text(value + 0.05, yi, f"{value:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.9)
    ax.set_xlabel("median NLL reduction vs Gaussian (nats/response vector)")
    ax.grid(axis="x", color=LIGHT, lw=0.8, zorder=0)
    ax.set_title("Flexible likelihood matters most for separated modes",
                 loc="left", fontweight="bold", color=NAVY)
    fig.tight_layout()
    save(fig, "law_fit_improvement")


def g8_effect_ladder() -> None:
    direct = pd.read_csv(G8_CLASSIFIER)
    direct["effect_slope"] = direct.typed_effect / direct.typed_effect_truth
    direct_slope = direct.groupby("estimator").effect_slope.median()
    guided = pd.read_csv(G8_GUIDED)
    expert = pd.read_csv(G8_EXPERT)
    values = [
        float(guided.raw_slope_median.max()),
        float(guided.loc[guided.model_name != "oracle_midpoint_law", "guided_slope_median"].max()),
        float(expert.quartic_slope.max()),
        float(direct_slope["mlp_full_path_history"]),
        float(direct_slope["mlp_motif"]),
    ]
    labels = [
        "best raw\nfull-law generator",
        "best guided frozen\ngenerator",
        "best endpoint\nexpert",
        "direct raw-path\nclassifier",
        "direct motif\nclassifier",
    ]
    colors = [GRAY, GOLD, ORANGE, TEAL, BLUE]
    fig, ax = plt.subplots(figsize=(8.2, 4.15))
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, width=0.68, alpha=0.95, zorder=2)
    ax.axhline(1.0, color=NAVY, ls="--", lw=1.4, label="correct effect")
    for xi, value in zip(x, values):
        ax.text(xi, value + 0.035, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("median effect slope (1 = correct)")
    ax.grid(axis="y", color=LIGHT, lw=0.8, zorder=0)
    ax.set_title("G8: contrast-aligned classification preserves the fourth-order effect",
                 loc="left", fontweight="bold", color=NAVY)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.01, -0.23,
            "Bars show the best named member of each family; repairs help generators but remain attenuated.",
            transform=ax.transAxes, color=GRAY, fontsize=8.5)
    fig.tight_layout()
    save(fig, "g8_effect_ladder")


def main() -> None:
    four_systems_schematic()
    simple_recovery_results()
    diffusion_score_vs_sid()
    law_fit_improvement()
    g8_effect_ladder()


if __name__ == "__main__":
    main()
