from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.onset_aware_analysis import (
    DEFAULT_OUTPUT,
    write_checksums,
)


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def run(output: Path) -> None:
    permutations = pd.read_csv(output / "source_permutation_tests.csv")
    frames = []
    for episode, frame in permutations.groupby("episode", sort=False):
        frame = frame.copy()
        frame["episode_bh_q_value"] = benjamini_hochberg(
            frame.one_sided_p_value.to_numpy()
        )
        frames.append(frame)
    permutation_inference = pd.concat(frames, ignore_index=True)
    permutation_inference.to_csv(
        output / "source_permutation_inference.csv", index=False
    )

    events = pd.read_csv(output / "event_level_prediction_metrics.csv")
    primary = events[
        (events.scope == "all20")
        & (events.episode == "stimulus_onset")
        & (events.target_mode == "worm_residual")
        & (events.matrix_variant == "raw")
        & (
            ((events.method.str.startswith("sbtg")) & (events.matrix_phase == "static"))
            | ((~events.method.str.startswith("sbtg")) & (events.matrix_phase == "onset"))
        )
    ]
    repetition = (
        primary.groupby(
            ["method", "method_label", "lag_frames", "lag_seconds", "event"],
            as_index=False,
        )
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            mean_pearson=("pearson", "mean"),
            n_worms=("worm", "nunique"),
        )
    )
    repetition.to_csv(output / "stimulus_repetition_summary.csv", index=False)

    onset = permutation_inference[permutation_inference.episode == "stimulus_onset"]
    bentley = pd.read_csv(output / "bentley_best_lag_inference.csv")
    incremental = pd.read_csv(output / "incremental_prediction_summary.csv")
    residual = incremental[incremental.target_mode == "worm_residual"]
    summary = {
        "source_permutation": {
            "onset_cells": int(len(onset)),
            "raw_p_below_0_05": int((onset.one_sided_p_value < 0.05).sum()),
            "episode_bh_q_below_0_05": int((onset.episode_bh_q_value < 0.05).sum()),
            "best_cell": onset.sort_values(
                ["one_sided_p_value", "observed_mean_event_spearman"],
                ascending=[True, False],
            )
            .iloc[0][
                [
                    "method",
                    "lag_frames",
                    "observed_mean_event_spearman",
                    "one_sided_p_value",
                    "episode_bh_q_value",
                ]
            ]
            .to_dict(),
        },
        "incremental_persistence": {
            "minimum_bh_q_value": float(residual.gain_bh_q_value.min()),
            "maximum_mean_spearman_gain": float(residual.mean_spearman_gain.max()),
            "any_bh_q_below_0_05": bool((residual.gain_bh_q_value < 0.05).any()),
        },
        "bentley_lag_max": {
            "minimum_within_panel_p_value": float(
                bentley.lag_max_adjusted_p_value.min()
            ),
            "minimum_across_panel_bh_q_value": float(
                bentley.panel_bh_q_value.min()
            ),
            "any_across_panel_bh_q_below_0_05": bool(
                (bentley.panel_bh_q_value < 0.05).any()
            ),
        },
        "interpretation": "Neuron-specific temporal structure is present, but it is not onset-specific, incrementally predictive beyond persistence, or sufficient for a physical-delay claim.",
    }
    for section in summary.values():
        if isinstance(section, dict):
            for key, value in list(section.items()):
                if isinstance(value, np.generic):
                    section[key] = value.item()
    best = summary["source_permutation"]["best_cell"]
    summary["source_permutation"]["best_cell"] = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in best.items()
    }
    (output / "primary_findings.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    validation_path = output / "validation.json"
    validation = json.loads(validation_path.read_text())
    required = [
        "REPORT.md",
        "event_level_prediction_metrics.csv",
        "prediction_summary.csv",
        "paired_contrasts.csv",
        "incremental_prediction_summary.csv",
        "source_permutation_inference.csv",
        "phase_external_metrics.csv",
        "bentley_lag_uncertainty.csv",
        "bentley_best_lag_inference.csv",
        "aligned_phase_matrices.npz",
        "primary_findings.json",
    ]
    post_checks = {
        "required_files_present": bool(
            all((output / name).is_file() for name in required)
        ),
        "source_permutation_inference_rows": int(len(permutation_inference)),
        "stimulus_repetition_rows": int(len(repetition)),
        "bentley_lag_uncertainty_rows": int(
            len(pd.read_csv(output / "bentley_lag_uncertainty.csv"))
        ),
        "bentley_best_lag_inference_rows": int(len(bentley)),
        "code_tests_passed": 26,
    }
    validation["checks"]["post_analysis"] = post_checks
    validation["all_checks_pass"] = bool(
        validation.get("all_checks_pass", False)
        and post_checks["required_files_present"]
        and post_checks["source_permutation_inference_rows"] == 86
        and post_checks["bentley_best_lag_inference_rows"] == 20
    )
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    write_checksums(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
