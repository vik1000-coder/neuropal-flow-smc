from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"energy": "#315A7D", "balanced": "#D17A22", "winner": "#2A9D6F"}


def create_figures(run_dir: Path) -> list[Path]:
    run_dir = run_dir.resolve()
    board = pd.read_csv(run_dir / "leaderboard.csv")
    trials = pd.read_csv(run_dir / "trial_metrics.csv")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    lag = board[board.phase == "lag_screen"].sort_values("lag")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    seconds = lag.lag.to_numpy() / 4.0
    ax.errorbar(
        seconds, lag.energy__mean, yerr=lag.energy__se, marker="o", lw=2,
        color=COLORS["energy"], label="Natural-frequency energy",
    )
    ax.errorbar(
        seconds, lag.energy__stim_balanced__mean,
        yerr=lag.energy__stim_balanced__se, marker="s", lw=2,
        color=COLORS["balanced"], label="Stimulus-balanced energy",
    )
    selected = lag.iloc[lag.energy__stim_balanced__mean.argmin()]
    ax.axvline(selected.lag / 4.0, color="#777777", ls="--", lw=1)
    ax.set(xlabel="History length (seconds)", ylabel="Energy score (lower is better)")
    ax.set_title("Whole-worm cross-validated lag screen")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figures / "lag_screen.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    final = board[board.phase == "final_confirmation"].sort_values("rank")
    labels = final.model_id.str.replace("tcn_delta_", "", regex=False).str.replace(
        "_full_gaussian", "", regex=False
    )
    y = np.arange(len(final))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(
        final.energy__mean, y - 0.10, xerr=final.energy__se, fmt="o",
        color=COLORS["energy"], capsize=3, label="Natural-frequency",
    )
    ax.errorbar(
        final.energy__stim_balanced__mean, y + 0.10,
        xerr=final.energy__stim_balanced__se, fmt="s",
        color=COLORS["balanced"], capsize=3, label="Stimulus-balanced",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Energy score (lower is better)")
    ax.set_title("Five-fold, three-seed final confirmation")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = figures / "final_leaderboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    keep = ["tcn_delta_flow_matching", "tcn_delta_mdn4", "ridge_full_gaussian"]
    frame = trials[
        (trials.phase == "final_confirmation") & trials.model_id.isin(keep)
    ]
    summary = frame.groupby(["model_id", "fold"]).energy.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model_id in keep:
        ax.plot(
            summary.index, summary[model_id], marker="o", lw=2,
            label=model_id.replace("tcn_delta_", "").replace("_full_gaussian", ""),
        )
    ax.set(xlabel="Held-out worm fold", ylabel="Energy score (lower is better)")
    ax.set_xticks(summary.index)
    ax.set_title("Fold heterogeneity in final confirmation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figures / "per_fold_energy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    for path in create_figures(args.run_dir):
        print(path)


if __name__ == "__main__":
    main()

