#!/usr/bin/env python3
"""Render phase-specific cell-type summaries from release CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "reference_snapshot" / "phase_analysis" / "comparison"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "derived" / "figures" / "phase_analysis"

PHASES = (
    ("baseline", "Baseline", "#7f8c8d"),
    ("on", "On", "#e74c3c"),
    ("steady", "Steady", "#3498db"),
    ("off", "Off", "#9b59b6"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tables: list[tuple[str, str, pd.DataFrame]] = []
    for code, label, color in PHASES:
        path = args.input_dir / f"celltype_by_lag_{code}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Phase summary not found: {path}")
        tables.append((label, color, pd.read_csv(path, index_col=0)))

    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(1, len(tables), figsize=(15, 5), sharey=True)
    finite_values = np.concatenate(
        [np.log10(np.maximum(table.to_numpy(dtype=float), 1e-12)).ravel() for _, _, table in tables]
    )
    vmin, vmax = np.nanpercentile(finite_values, [2, 98])
    for index, (label, color, table) in enumerate(tables):
        sns.heatmap(
            np.log10(np.maximum(table.astype(float), 1e-12)),
            ax=axes[index],
            cmap="RdYlBu_r",
            vmin=vmin,
            vmax=vmax,
            cbar=index == len(tables) - 1,
            cbar_kws={"label": "log10 mean |weight|"},
        )
        axes[index].set_title(label, color=color, fontweight="bold")
        axes[index].set_xlabel("Lag")
        axes[index].set_ylabel("Cell-type pair" if index == 0 else "")
        axes[index].set_xticklabels(
            [str(column).removeprefix("lag") for column in table.columns], rotation=0
        )
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "celltype_phase_heatmaps.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
