"""Paired summaries for the frozen July 13 stable-SID benchmark.

This script never tunes a model or filters a failed cell.  It reads the frozen
atomic outputs, summarizes common metrics, and computes exploratory paired
bootstrap intervals over the crossed seed cells.  With only two independent
generator seeds, intervals are developmental and are not treated as
confirmatory system-level uncertainty.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED_KEYS = ["generator_id", "generator_seed", "data_seed", "model_seed"]
CORE_GROUP_KEYS = [
    "generator_id",
    "model_name",
    "metric_id",
    "estimand",
    "noise_sigma_standardized",
    "centering",
]


def _ok(frame: pd.DataFrame) -> pd.DataFrame:
    status = "metric_status" if "metric_status" in frame else "status"
    return frame.loc[frame[status].eq("ok")].copy()


def _bootstrap_mean(values: np.ndarray, seed: int = 713, draws: int = 20_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sampled.mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_core(core: Path, output: Path) -> None:
    cases = pd.read_csv(core / "cases.csv")
    metrics = _ok(pd.read_csv(core / "metrics.csv"))
    cases.to_csv(output / "core_cases.csv", index=False)
    summary = (
        metrics.groupby(CORE_GROUP_KEYS, as_index=False, dropna=False)
        .agg(median=("value", "median"), q25=("value", lambda x: x.quantile(0.25)),
             q75=("value", lambda x: x.quantile(0.75)), n=("value", "size"))
    )
    summary.to_csv(output / "core_metric_summary.csv", index=False)

    energy = metrics.loc[metrics.metric_id.eq("energy_score_fair")].copy()
    baseline = energy.loc[energy.model_name.eq("gaussian_nll"), SEED_KEYS + ["value"]].rename(
        columns={"value": "baseline_value"}
    )
    paired = energy.merge(baseline, on=SEED_KEYS, how="inner", validate="many_to_one")
    paired["relative_change"] = (paired.value - paired.baseline_value) / paired.baseline_value.abs().clip(lower=1e-12)
    rows = []
    for (generator, model), cell in paired.groupby(["generator_id", "model_name"]):
        mean, low, high = _bootstrap_mean(cell.relative_change.to_numpy())
        rows.append({
            "generator_id": generator,
            "model_name": model,
            "n_paired": len(cell),
            "mean_relative_energy_change_vs_gaussian": mean,
            "bootstrap_low": low,
            "bootstrap_high": high,
            "win_fraction": float((cell.relative_change < 0).mean()),
            "equivalent_within_5pct_fraction": float((cell.relative_change.abs() <= 0.05).mean()),
        })
    relative = pd.DataFrame(rows)
    relative["primary_eligible"] = ~relative.model_name.eq("bounded_energy")
    relative.to_csv(output / "paired_energy_vs_gaussian.csv", index=False)

    nll = metrics.loc[metrics.metric_id.eq("nll_original")].copy()
    nll_baseline = nll.loc[
        nll.model_name.eq("gaussian_nll"), SEED_KEYS + ["value"]
    ].rename(columns={"value": "baseline_value"})
    nll_paired = nll.merge(
        nll_baseline, on=SEED_KEYS, how="inner", validate="many_to_one"
    )
    nll_paired["difference_nats_vs_gaussian"] = (
        nll_paired.value - nll_paired.baseline_value
    )
    nll_rows = []
    for (generator, model), cell in nll_paired.groupby(
        ["generator_id", "model_name"]
    ):
        mean, low, high = _bootstrap_mean(
            cell.difference_nats_vs_gaussian.to_numpy()
        )
        nll_rows.append(
            {
                "generator_id": generator,
                "model_name": model,
                "n_paired": len(cell),
                "mean_nll_difference_nats_vs_gaussian": mean,
                "bootstrap_low": low,
                "bootstrap_high": high,
                "win_fraction": float(
                    (cell.difference_nats_vs_gaussian < 0).mean()
                ),
            }
        )
    pd.DataFrame(nll_rows).to_csv(
        output / "paired_nll_vs_gaussian.csv", index=False
    )

    gate_specs = (
        (
            "clean_tangent",
            metrics.loc[
                metrics.metric_id.eq("tangent_nrmse")
                & metrics.estimand.eq("clean")
                & metrics.centering.eq("none")
            ],
        ),
        (
            "clean_finite_log_ratio",
            metrics.loc[
                metrics.metric_id.eq("finite_ratio_nrmse")
                & metrics.estimand.eq("clean")
            ],
        ),
        (
            "noisy_tangent_model_center_sigma012",
            metrics.loc[
                metrics.metric_id.eq("tangent_nrmse")
                & metrics.estimand.eq("noisy")
                & metrics.noise_sigma_standardized.eq(0.12)
                & metrics.centering.eq("model_samples")
            ],
        ),
    )
    gate_rows = []
    for gate, frame in gate_specs:
        for (generator, model), cell in frame.groupby(
            ["generator_id", "model_name"]
        ):
            gate_rows.append(
                {
                    "gate": gate,
                    "generator_id": generator,
                    "model_name": model,
                    "n": len(cell),
                    "median_nrmse": float(cell.value.median()),
                    "fraction_beating_zero": float((cell.value < 1).mean()),
                    "fraction_nrmse_below_08": float((cell.value < 0.8).mean()),
                }
            )
    pd.DataFrame(gate_rows).to_csv(output / "core_sid_gates.csv", index=False)

    primary_panels = (
        (
            "fair_energy",
            summary.loc[
                summary.metric_id.eq("energy_score_fair")
                & ~summary.model_name.eq("bounded_energy")
            ],
        ),
        ("exact_density_nll", summary.loc[summary.metric_id.eq("nll_original")]),
        (
            "clean_history_tangent",
            summary.loc[
                summary.metric_id.eq("tangent_nrmse")
                & summary.estimand.eq("clean")
                & summary.centering.eq("none")
            ],
        ),
        (
            "noisy_response_score_sigma012",
            summary.loc[
                summary.metric_id.eq("response_score_nrmse")
                & summary.estimand.eq("noisy")
                & summary.noise_sigma_standardized.eq(0.12)
            ],
        ),
        (
            "noisy_history_tangent_model_center_sigma012",
            summary.loc[
                summary.metric_id.eq("tangent_nrmse")
                & summary.estimand.eq("noisy")
                & summary.noise_sigma_standardized.eq(0.12)
                & summary.centering.eq("model_samples")
            ],
        ),
    )
    best_rows = []
    for panel_name, panel in primary_panels:
        ranked = panel.sort_values(["generator_id", "median", "model_name"]).copy()
        ranked["rank"] = ranked.groupby("generator_id").cumcount() + 1
        ranked.insert(0, "panel", panel_name)
        best_rows.append(ranked.loc[ranked["rank"] <= 5])
    pd.concat(best_rows, ignore_index=True).to_csv(
        output / "core_primary_top5.csv", index=False
    )

    failures = (
        cases.groupby(["generator_id", "model_name", "status"], as_index=False)
        .size().rename(columns={"size": "cases"})
    )
    failures.to_csv(output / "core_failure_counts.csv", index=False)

    plot_specs = (
        (
            "energy_score_fair",
            "core_energy_score.png",
            "Fair energy score (lower is better)",
            lambda frame: frame,
        ),
        (
            "tangent_nrmse",
            "core_clean_tangent_nrmse.png",
            "Clean history-tangent NRMSE (zero baseline = 1)",
            lambda frame: frame.loc[
                frame.estimand.eq("clean") & frame.centering.eq("none")
            ],
        ),
        (
            "tangent_nrmse",
            "core_noisy_tangent_model_center_nrmse.png",
            "Noisy tangent NRMSE, model-sample centered (zero baseline = 1)",
            lambda frame: frame.loc[
                frame.estimand.eq("noisy")
                & frame.noise_sigma_standardized.eq(0.12)
                & frame.centering.eq("model_samples")
            ],
        ),
        (
            "response_score_nrmse",
            "core_noisy_response_score_nrmse.png",
            "Noisy response-score NRMSE at sigma = 0.12",
            lambda frame: frame.loc[
                frame.estimand.eq("noisy")
                & frame.noise_sigma_standardized.eq(0.12)
            ],
        ),
    )
    for metric_id, filename, label, selector in plot_specs:
        chart = selector(summary.loc[summary.metric_id.eq(metric_id)].copy())
        if metric_id == "energy_score_fair":
            # The current bounded-energy sampler is finite-pool SIR, not iid.
            # Its numerical rows remain auditable but are not primary proper scores.
            chart = chart.loc[~chart.model_name.eq("bounded_energy")]
        if chart.empty:
            continue
        pivot = chart.pivot(index="model_name", columns="generator_id", values="median")
        order = pivot.median(axis=1).sort_values().index
        pivot = pivot.loc[order]
        height = max(5, 0.28 * len(pivot))
        fig, ax = plt.subplots(figsize=(11, height))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
        ax.set_yticks(range(len(pivot)), labels=pivot.index)
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
        ax.set_title(label)
        for row in range(len(pivot)):
            for column in range(len(pivot.columns)):
                value = pivot.iloc[row, column]
                if np.isfinite(value):
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if value > np.nanmedian(pivot.to_numpy()) else "black")
        fig.colorbar(image, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def summarize_finite(finite: Path, output: Path) -> None:
    metrics = _ok(pd.read_csv(finite / "finite_contrast_metrics.csv"))
    summary = (
        metrics.groupby(
            ["law", "delta", "generator_id", "model_name", "method", "metric_id"],
            as_index=False,
            dropna=False,
        )
        .agg(median=("value", "median"), q25=("value", lambda x: x.quantile(0.25)),
             q75=("value", lambda x: x.quantile(0.75)), n=("value", "size"))
    )
    summary.to_csv(output / "finite_metric_summary.csv", index=False)
    focus = summary.loc[
        summary.metric_id.isin(["witness_nrmse", "channel_readout_nrmse"])
        & summary.delta.eq(0.12)
    ].copy()
    focus.to_csv(output / "finite_primary_delta012.csv", index=False)

    channel = focus.loc[
        focus.metric_id.eq("channel_readout_nrmse") & focus.law.eq("clean")
    ].copy()
    if not channel.empty:
        channel["label"] = channel.model_name + " / " + channel.method
        pivot = channel.pivot(index="label", columns="generator_id", values="median")
        order = pivot.median(axis=1).sort_values().index[:30]
        pivot = pivot.loc[order]
        fig, ax = plt.subplots(figsize=(11, max(6, 0.3 * len(pivot))))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma_r", vmin=0)
        ax.set_yticks(range(len(pivot)), labels=pivot.index)
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
        ax.set_title("Finite typed-channel NRMSE at delta=0.12 (best 30 routes)")
        for row in range(len(pivot)):
            for column in range(len(pivot.columns)):
                value = pivot.iloc[row, column]
                if np.isfinite(value):
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if value > np.nanmedian(pivot.to_numpy()) else "black")
        fig.colorbar(image, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(output / "finite_channel_nrmse.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--finite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summarize_core(args.core, args.output)
    if args.finite is not None:
        summarize_finite(args.finite, args.output)


if __name__ == "__main__":
    main()
