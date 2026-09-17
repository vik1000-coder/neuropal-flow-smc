#!/usr/bin/env python3
"""Plot discrete peak lags produced by ``analyze_peaks.py``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION = PROJECT_ROOT / "results" / "derived" / "evaluation"
DEFAULT_FIGURES = PROJECT_ROOT / "results" / "derived" / "figures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_EVALUATION / "discrete_peak_lags.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_FIGURES / "discrete_peak_lags.png"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Peak summary not found: {args.input}. Run analyze_peaks.py first."
        )
    frame = pd.read_csv(args.input).sort_values("peak_time_s")
    if frame.empty:
        raise ValueError(f"Peak summary is empty: {args.input}")

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(frame["network"], frame["peak_time_s"], color="#4c78a8")
    for bar, value in zip(bars, frame["peak_time_s"]):
        ax.text(
            value + max(frame["peak_time_s"].max() * 0.015, 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:g} s",
            va="center",
        )
    ax.set_xlabel("Lag at maximum sampled F1 (s)")
    ax.set_ylabel("")
    ax.set_title("Discrete peak agreement by reference network")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
