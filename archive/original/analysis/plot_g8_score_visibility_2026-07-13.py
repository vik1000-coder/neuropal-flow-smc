"""Render the report figure for the G8 oracle score-visibility audit."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("analysis/g8_score_visibility_audit_2026-07-13")
SOURCE = ROOT / "noise_summary.csv"


def main() -> None:
    frame = pd.read_csv(SOURCE).sort_values("sigma").reset_index(drop=True)
    labels = [f"{value:g}" for value in frame.sigma]
    x = np.arange(len(frame))
    retained_js = 100 * frame.jensen_shannon_nats_median / frame.jensen_shannon_nats_median.iloc[0]
    retained_tangent = 100 * frame.history_tangent_rms_median / frame.history_tangent_rms_median.iloc[0]
    visible_score = 100 * frame.relative_response_score_gap_median

    blue = "#2F6B9A"
    orange = "#C56A2D"
    ink = "#25282D"
    grid = "#D9DDE3"
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 0.95], "hspace": 0.12},
    )
    fig.patch.set_facecolor("white")

    top, bottom = axes
    top.plot(
        x,
        retained_js,
        color=blue,
        marker="o",
        linewidth=2.2,
        markersize=5.5,
        label="Jensen–Shannon information retained",
    )
    top.plot(
        x,
        retained_tangent,
        color=orange,
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.2,
        linestyle="--",
        markersize=5.5,
        label="History-tangent RMS retained",
    )
    top.axhline(0, color=ink, linewidth=0.8)
    top.set_ylabel("Percent of clean-scale value")
    top.set_ylim(-4, 108)
    top.legend(loc="lower left", frameon=False, ncol=1, fontsize=9)

    bottom.plot(
        x,
        visible_score,
        color=blue,
        marker="o",
        linewidth=2.2,
        markersize=5.5,
    )
    peak = int(np.argmax(visible_score))
    bottom.scatter([peak], [visible_score.iloc[peak]], s=85, facecolor="white", edgecolor=orange, linewidth=2, zorder=5)
    bottom.annotate(
        f"Peak visibility: {visible_score.iloc[peak]:.1f}%",
        xy=(peak, visible_score.iloc[peak]),
        xytext=(peak + 0.45, visible_score.iloc[peak] + 1.7),
        color=ink,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": ink, "linewidth": 0.8},
    )
    bottom.set_ylabel("Outcome-score gap\n(% of score RMS)")
    bottom.set_xlabel("Added response-noise standard deviation, σ (evaluated scales)")
    bottom.set_ylim(0, max(16.5, float(visible_score.max()) + 3.5))
    bottom.set_xticks(x, labels)

    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(axis="y", color=grid, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(ink)
        axis.spines["bottom"].set_color(ink)
        axis.tick_params(colors=ink, labelsize=9)
        axis.yaxis.label.set_color(ink)
        axis.xaxis.label.set_color(ink)

    fig.suptitle(
        "G8 mode-weight signal across denoising scales",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="semibold",
        color=ink,
    )
    fig.text(
        0.08,
        0.945,
        "Four independent systems; medians from 5,000 exact draws per control side and system",
        ha="left",
        va="top",
        fontsize=9,
        color="#5D626B",
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.87, bottom=0.12)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"g8_score_visibility.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
