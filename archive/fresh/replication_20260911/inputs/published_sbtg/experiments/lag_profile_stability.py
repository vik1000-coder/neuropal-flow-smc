#!/usr/bin/env python3
"""Profile-level stability analysis for lag-profile retrainings.

The default, primary analysis deliberately uses only the common Approach-C
production grid ``[2, 3, 5, 8, 10, 15, 20]``. A separate, explicitly secondary
mode evaluates all eight newly retrained Approach-C lags as a
sensitivity diagnostic.  That mode cannot assess stability of the published
full curve because the released lag-1 estimate came from an
incompatible estimator.

The script is read-only with respect to fitted models.  It validates and loads
the three full-data seed controls and the prespecified B=4 recording-row bootstrap
campaign (replicates 0--3, model seeds 42 and 43), skips fits whose lag grid is
not yet complete, and writes descriptive CSV, figure, JSON, and Markdown
artifacts.  It never treats the two model seeds as independent bootstrap
resamples and never constructs a confidence interval from B=4.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from experiments.common import (
    DEFAULT_CONNECTOME,
    DEFAULT_MONOAMINE,
    DEFAULT_RANDI,
    DEFAULT_SBTG,
    load_dataset,
)
from experiments.snapshot import PREPARED_DATA, portable_path
from experiments.lag_profile_fit import DEFAULT_DATA_SEED, fit_label
from experiments.lag_profile_summary import (
    LAG2PLUS,
    MONOAMINES,
    add_fit_records,
    default_full_control_roots,
    load_external_full_control,
    load_fit,
    parse_full_control_roots,
)
from experiments.lag_profile_metrics import (
    atomic_json,
    build_networks,
    f1_curve_rows,
    load_paper_matrices,
    write_csv,
)


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = (
    ANALYSIS_DIR.parent / "results/derived/lag_profile_bootstrap"
)
DEFAULT_OUT_DIR = DEFAULT_INPUT_ROOT / "profile_stability"
DEFAULT_ALL8_OUT_DIR = DEFAULT_INPUT_ROOT / "profile_stability_all8_diagnostic"
PRIMARY_LAGS = tuple(int(lag) for lag in LAG2PLUS)
ALL_EIGHT_LAGS = (1, 2, 3, 5, 8, 10, 15, 20)
PRIMARY_SCOPE = "primary_lag2plus"
ALL8_DIAGNOSTIC_SCOPE = "secondary_all8_diagnostic"
ANALYSIS_SCOPE = PRIMARY_SCOPE
ANALYSIS_LAGS = PRIMARY_LAGS
BOOTSTRAP_REPLICATES = (0, 1, 2, 3)
BOOTSTRAP_MODEL_SEEDS = (42, 43)
FULL_CONTROL_SEEDS = (42, 43, 44)
SECONDS_PER_FRAME = 0.25
TIE_ATOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze profile-level stability of completed production lag-profile "
            "full-data controls and B=4 bootstrap retrainings."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--analysis-scope",
        choices=(PRIMARY_SCOPE, ALL8_DIAGNOSTIC_SCOPE),
        default=PRIMARY_SCOPE,
        help=(
            "Primary lag-2+ production-lineage analysis, or the secondary "
            "all-eight-lag retraining sensitivity diagnostic."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to a scope-specific directory under the bootstrap root.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=PREPARED_DATA)
    parser.add_argument("--paper-result", type=Path, default=DEFAULT_SBTG)
    parser.add_argument("--connectome-dir", type=Path, default=DEFAULT_CONNECTOME)
    parser.add_argument("--randi-file", type=Path, default=DEFAULT_RANDI)
    parser.add_argument("--monoamine-file", type=Path, default=DEFAULT_MONOAMINE)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument(
        "--full-control-root",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help=(
            "Override a full-control directory containing lag_XX/result.npz. "
            "May be repeated; defaults use full-data controls under --input-root."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail after writing the summary if fewer than all 11 fits are complete.",
    )
    return parser.parse_args()


def configure_analysis_scope(scope: str) -> None:
    """Select the analysis grid for one process; primary remains the default."""
    global ANALYSIS_SCOPE, ANALYSIS_LAGS
    if scope == PRIMARY_SCOPE:
        ANALYSIS_SCOPE = PRIMARY_SCOPE
        ANALYSIS_LAGS = PRIMARY_LAGS
    elif scope == ALL8_DIAGNOSTIC_SCOPE:
        ANALYSIS_SCOPE = ALL8_DIAGNOSTIC_SCOPE
        ANALYSIS_LAGS = ALL_EIGHT_LAGS
    else:
        raise ValueError(f"Unknown lag-profile analysis scope: {scope!r}")


def is_all8_diagnostic() -> bool:
    return ANALYSIS_SCOPE == ALL8_DIAGNOSTIC_SCOPE


def scoped_output_name(primary_name: str) -> str:
    """Keep primary filenames stable and make every secondary file unmistakable."""
    if not is_all8_diagnostic():
        return primary_name
    path = Path(primary_name)
    return f"{path.stem}_all8_diagnostic{path.suffix}"


def ordered_complete_profile(
    group: pd.DataFrame,
    context: str,
    candidate_lags: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Return one finite F1 value at every candidate lag in ascending order."""
    if candidate_lags is None:
        candidate_lags = ANALYSIS_LAGS
    required = {"lag", "f1"}
    missing_columns = required.difference(group.columns)
    if missing_columns:
        raise RuntimeError(f"{context} lacks columns: {sorted(missing_columns)}")
    if group["lag"].duplicated().any():
        duplicated = sorted(
            group.loc[group["lag"].duplicated(keep=False), "lag"]
            .astype(int)
            .unique()
            .tolist()
        )
        raise RuntimeError(f"{context} has duplicate lags: {duplicated}")
    expected = tuple(sorted(int(value) for value in candidate_lags))
    observed = tuple(sorted(group["lag"].astype(int).tolist()))
    if observed != expected:
        raise RuntimeError(
            f"{context} has lags {observed}, expected the complete grid {expected}"
        )
    ordered = group.sort_values("lag").copy()
    values = ordered["f1"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{context} contains non-finite F1 values")
    if np.any((values < 0) | (values > 1)):
        raise RuntimeError(f"{context} contains F1 values outside [0, 1]")
    if "time_s" in ordered:
        expected_time = ordered["lag"].to_numpy(dtype=float) * SECONDS_PER_FRAME
        observed_time = ordered["time_s"].to_numpy(dtype=float)
        if not np.allclose(observed_time, expected_time, rtol=0, atol=TIE_ATOL):
            raise RuntimeError(f"{context} has time_s inconsistent with lag * 0.25")
    return ordered


def weighted_profile_moments(
    lags: Sequence[float], f1_values: Sequence[float]
) -> dict[str, float | bool]:
    """Compute normalized-F1 center of mass and population weighted SD.

    A zero-sum nonnegative profile has no defined normalized distribution.  It
    is represented explicitly with ``moments_defined=False`` and NaN moments,
    rather than silently dividing by zero.
    """
    lag_array = np.asarray(lags, dtype=float)
    values = np.asarray(f1_values, dtype=float)
    if lag_array.ndim != 1 or values.ndim != 1 or len(lag_array) != len(values):
        raise ValueError("lags and F1 values must be equal-length one-dimensional arrays")
    if len(values) == 0:
        raise ValueError("a lag profile must contain at least one point")
    if not np.all(np.isfinite(lag_array)) or not np.all(np.isfinite(values)):
        raise ValueError("lags and F1 values must be finite")
    if np.any(values < 0):
        raise ValueError("F1 profile weights must be nonnegative")
    total = float(np.sum(values))
    if total <= 0:
        return {
            "f1_sum": total,
            "moments_defined": False,
            "center_of_mass_lag": float("nan"),
            "weighted_sd_lag": float("nan"),
        }
    weights = values / total
    center = float(np.sum(weights * lag_array))
    variance = float(np.sum(weights * (lag_array - center) ** 2))
    # Protect against a tiny negative caused only by floating point rounding.
    variance = max(variance, 0.0)
    return {
        "f1_sum": total,
        "moments_defined": True,
        "center_of_mass_lag": center,
        "weighted_sd_lag": float(np.sqrt(variance)),
    }


def uniform_profile_reference(
    lags: Sequence[float] | None = None,
) -> dict[str, float | bool]:
    """Moments of equal weight at every analyzed grid point.

    This is a diagnostic baseline for the positive F1 floor: normalization
    cancels multiplicative scale, but it does not subtract a common positive
    floor, which pulls moments toward this uniform-grid reference.
    """
    if lags is None:
        lags = ANALYSIS_LAGS
    moments = weighted_profile_moments(lags, np.ones(len(lags), dtype=float))
    center = float(moments["center_of_mass_lag"])
    width = float(moments["weighted_sd_lag"])
    return {
        **moments,
        "center_of_mass_s": center * SECONDS_PER_FRAME,
        "weighted_sd_s": width * SECONDS_PER_FRAME,
    }


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation with an explicit NaN for constant profiles."""
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("correlation inputs must be equal-length vectors")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("correlation inputs must be finite")
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(left * right) / denominator)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation using average ranks for tied F1 values."""
    left_ranks = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    right_ranks = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    return pearson_correlation(left_ranks.to_numpy(), right_ranks.to_numpy())


def run_display_label(sample_kind: str, replicate: int, model_seed: int) -> str:
    if sample_kind == "full":
        return f"Full data · seed {model_seed}"
    return f"Bootstrap {replicate} · seed {model_seed}"


def run_short_label(sample_kind: str, replicate: int, model_seed: int) -> str:
    if sample_kind == "full":
        return f"full_s{model_seed}"
    return f"b{replicate}_s{model_seed}"


def add_run_metadata(curves: pd.DataFrame) -> pd.DataFrame:
    enriched = curves.copy()
    enriched["run_label"] = [
        run_short_label(str(kind), int(replicate), int(seed))
        for kind, replicate, seed in zip(
            enriched["sample_kind"],
            enriched["replicate"],
            enriched["model_seed"],
        )
    ]
    enriched["display_label"] = [
        run_display_label(str(kind), int(replicate), int(seed))
        for kind, replicate, seed in zip(
            enriched["sample_kind"],
            enriched["replicate"],
            enriched["model_seed"],
        )
    ]
    enriched["run_order"] = [
        int(seed) - 42
        if str(kind) == "full"
        else 10 + int(replicate) * 2 + BOOTSTRAP_MODEL_SEEDS.index(int(seed))
        for kind, replicate, seed in zip(
            enriched["sample_kind"],
            enriched["replicate"],
            enriched["model_seed"],
        )
    ]
    totals = enriched.groupby(["fit_label", "network"])["f1"].transform("sum")
    enriched["profile_weight"] = np.where(
        totals > 0, enriched["f1"] / totals, np.nan
    )
    return enriched.sort_values(
        ["run_order", "network", "lag"]
    ).reset_index(drop=True)


def collect_completed_curves(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, Path], pd.DataFrame]:
    """Load only fully completed individual fits; never launch training."""
    input_root = args.input_root.resolve()
    dataset_dir = args.dataset_dir.resolve()
    full_roots = parse_full_control_roots(args.full_control_root, input_root)
    traces, dataset_names, _ = load_dataset(dataset_dir)
    networks = build_networks(
        dataset_names,
        args.connectome_dir,
        args.randi_file,
        args.monoamine_file,
    )
    records: list[dict] = []
    completion_rows: list[dict] = []

    for seed in FULL_CONTROL_SEEDS:
        matrices, missing_lags = load_external_full_control(
            full_roots[seed],
            seed,
            dataset_dir,
            len(traces),
            dataset_names,
            args.data_seed,
        )
        label = fit_label("full", -1, seed)
        is_complete = matrices is not None
        completion_rows.append(
            {
                "fit_label": label,
                "run_label": run_short_label("full", -1, seed),
                "sample_kind": "full",
                "replicate": -1,
                "model_seed": seed,
                "complete": is_complete,
                "missing_lags": ",".join(str(value) for value in missing_lags),
            }
        )
        if is_complete:
            add_fit_records(records, matrices, networks, "full", -1, seed)

    for replicate in BOOTSTRAP_REPLICATES:
        for seed in BOOTSTRAP_MODEL_SEEDS:
            matrices, missing_lags = load_fit(
                input_root,
                "bootstrap",
                replicate,
                seed,
                args.data_seed,
                dataset_dir,
                len(traces),
                dataset_names,
            )
            label = fit_label("bootstrap", replicate, seed)
            is_complete = matrices is not None
            completion_rows.append(
                {
                    "fit_label": label,
                    "run_label": run_short_label("bootstrap", replicate, seed),
                    "sample_kind": "bootstrap",
                    "replicate": replicate,
                    "model_seed": seed,
                    "complete": is_complete,
                    "missing_lags": ",".join(str(value) for value in missing_lags),
                }
            )
            if is_complete:
                add_fit_records(
                    records, matrices, networks, "bootstrap", replicate, seed
                )

    columns = (
        "fit_label",
        "omit_index",
        "network",
        "lag",
        "time_s",
        "f1",
        "best_threshold",
        "sample_kind",
        "replicate",
        "model_seed",
    )
    curves = pd.DataFrame(records, columns=columns)
    curves = curves.loc[
        curves["network"].isin(MONOAMINES) & curves["lag"].isin(ANALYSIS_LAGS)
    ].copy()
    if curves.empty:
        raise RuntimeError("No complete lag-profile full-control or bootstrap fit is available")
    curves = add_run_metadata(curves)

    for (fit, network), group in curves.groupby(["fit_label", "network"]):
        ordered_complete_profile(group, f"{fit}/{network}")
    counts = curves.groupby("fit_label")["network"].nunique()
    malformed = counts.loc[counts != len(MONOAMINES)]
    if not malformed.empty:
        raise RuntimeError(
            "Completed fits lack monoamine networks: " + malformed.to_json()
        )
    completion = pd.DataFrame(completion_rows).sort_values(
        ["sample_kind", "replicate", "model_seed"]
    )

    paper_matrices = load_paper_matrices(
        args.paper_result.resolve(), dataset_names
    )
    paper_curves = pd.DataFrame(
        f1_curve_rows(
            paper_matrices,
            networks,
            "paper_reference",
            -2,
        )
    )
    paper_curves = paper_curves.loc[
        paper_curves["network"].isin(MONOAMINES)
        & paper_curves["lag"].isin(ANALYSIS_LAGS)
    ].copy()
    paper_curves["sample_kind"] = "published_paper"
    paper_curves["replicate"] = -2
    paper_curves["model_seed"] = -1
    paper_curves["run_label"] = "published_paper"
    paper_curves["display_label"] = "Released paper reference"
    paper_curves["run_order"] = -1
    totals = paper_curves.groupby(["fit_label", "network"])["f1"].transform(
        "sum"
    )
    paper_curves["profile_weight"] = np.where(
        totals > 0, paper_curves["f1"] / totals, np.nan
    )
    for network, group in paper_curves.groupby("network"):
        ordered_complete_profile(group, f"paper_reference/{network}")
    if set(paper_curves["network"]) != set(MONOAMINES):
        raise RuntimeError("Released paper curves lack a monoamine network")
    return (
        curves,
        completion.reset_index(drop=True),
        full_roots,
        paper_curves.sort_values(["network", "lag"]).reset_index(drop=True),
    )


def run_metric_rows(curves: pd.DataFrame) -> pd.DataFrame:
    """Compute peak, margin, center-of-mass, and effective width per run."""
    rows: list[dict] = []
    for (fit, network), group in curves.groupby(
        ["fit_label", "network"], sort=False
    ):
        ordered = ordered_complete_profile(group, f"{fit}/{network}")
        lags = ordered["lag"].to_numpy(dtype=int)
        values = ordered["f1"].to_numpy(dtype=float)
        moments = weighted_profile_moments(lags, values)
        ranking = np.lexsort((lags, -values))
        top_index = int(ranking[0])
        second_index = int(ranking[1])
        top_value = float(values[top_index])
        tied_count = int(
            np.count_nonzero(np.isclose(values, top_value, rtol=0, atol=TIE_ATOL))
        )
        first = ordered.iloc[0]
        center = float(moments["center_of_mass_lag"])
        width = float(moments["weighted_sd_lag"])
        rows.append(
            {
                "fit_label": fit,
                "run_label": first["run_label"],
                "display_label": first["display_label"],
                "run_order": int(first["run_order"]),
                "sample_kind": first["sample_kind"],
                "replicate": int(first["replicate"]),
                "model_seed": int(first["model_seed"]),
                "network": network,
                "n_profile_lags": len(lags),
                "f1_sum_normalization_denominator": float(moments["f1_sum"]),
                "profile_moments_defined": bool(moments["moments_defined"]),
                "f1_min": float(np.min(values)),
                "f1_max": float(np.max(values)),
                "f1_dynamic_range": float(np.ptp(values)),
                "center_of_mass_lag": center,
                "center_of_mass_s": center * SECONDS_PER_FRAME,
                "weighted_sd_lag": width,
                "weighted_sd_s": width * SECONDS_PER_FRAME,
                "peak_lag": int(lags[top_index]),
                "peak_lag_s": float(lags[top_index] * SECONDS_PER_FRAME),
                "peak_f1": top_value,
                "second_best_lag": int(lags[second_index]),
                "second_best_lag_s": float(
                    lags[second_index] * SECONDS_PER_FRAME
                ),
                "second_best_f1": float(values[second_index]),
                "peak_margin_f1": float(top_value - values[second_index]),
                "n_lags_tied_for_peak": tied_count,
                "peak_is_tied": tied_count > 1,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["run_order", "network"]
    ).reset_index(drop=True)


def pair_scope(left: pd.Series, right: pd.Series) -> str:
    left_kind = str(left["sample_kind"])
    right_kind = str(right["sample_kind"])
    if left_kind == "full" and right_kind == "full":
        return "full_vs_full"
    if left_kind != right_kind:
        return "full_vs_bootstrap"
    if int(left["replicate"]) == int(right["replicate"]):
        return "bootstrap_within_resample_seeds"
    return "bootstrap_across_resamples"


def pairwise_profile_rows(
    curves: pd.DataFrame, metrics: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Compute every unordered run pair and symmetric correlation matrices."""
    metric_lookup = metrics.set_index(["network", "fit_label"])
    if metric_lookup.index.duplicated().any():
        raise RuntimeError("Run metrics are not unique by network and fit")
    rows: list[dict] = []
    matrices: dict[tuple[str, str], pd.DataFrame] = {}

    for network in MONOAMINES:
        network_curves = curves.loc[curves["network"] == network]
        run_metadata = (
            network_curves.loc[
                :,
                [
                    "fit_label",
                    "run_label",
                    "display_label",
                    "run_order",
                    "sample_kind",
                    "replicate",
                    "model_seed",
                ],
            ]
            .drop_duplicates()
            .sort_values("run_order")
            .reset_index(drop=True)
        )
        run_ids = run_metadata["fit_label"].tolist()
        profiles: dict[str, np.ndarray] = {}
        by_lag: dict[str, dict[int, float]] = {}
        for fit in run_ids:
            ordered = ordered_complete_profile(
                network_curves.loc[network_curves["fit_label"] == fit],
                f"{fit}/{network}",
            )
            profiles[fit] = ordered["f1"].to_numpy(dtype=float)
            by_lag[fit] = dict(
                zip(ordered["lag"].astype(int), ordered["f1"].astype(float))
            )
        pearson_matrix = pd.DataFrame(
            np.nan, index=run_ids, columns=run_ids, dtype=float
        )
        spearman_matrix = pearson_matrix.copy()
        metadata_lookup = run_metadata.set_index("fit_label")

        for left_fit in run_ids:
            for right_fit in run_ids:
                pearson_matrix.loc[left_fit, right_fit] = pearson_correlation(
                    profiles[left_fit], profiles[right_fit]
                )
                spearman_matrix.loc[left_fit, right_fit] = spearman_correlation(
                    profiles[left_fit], profiles[right_fit]
                )
        matrices[(network, "pearson")] = pearson_matrix
        matrices[(network, "spearman")] = spearman_matrix

        for left_fit, right_fit in itertools.combinations(run_ids, 2):
            left = metadata_lookup.loc[left_fit]
            right = metadata_lookup.loc[right_fit]
            left_metric = metric_lookup.loc[(network, left_fit)]
            right_metric = metric_lookup.loc[(network, right_fit)]
            left_peak = int(left_metric["peak_lag"])
            right_peak = int(right_metric["peak_lag"])
            left_peak_index = ANALYSIS_LAGS.index(left_peak)
            right_peak_index = ANALYSIS_LAGS.index(right_peak)
            steps = abs(left_peak_index - right_peak_index)
            left_selected_gap = float(
                by_lag[left_fit][left_peak] - by_lag[left_fit][right_peak]
            )
            right_selected_gap = float(
                by_lag[right_fit][right_peak] - by_lag[right_fit][left_peak]
            )
            profile_difference = profiles[left_fit] - profiles[right_fit]
            rows.append(
                {
                    "network": network,
                    "pair_scope": pair_scope(left, right),
                    "run_a": left_fit,
                    "run_a_label": left["run_label"],
                    "run_a_sample_kind": left["sample_kind"],
                    "run_a_replicate": int(left["replicate"]),
                    "run_a_model_seed": int(left["model_seed"]),
                    "run_b": right_fit,
                    "run_b_label": right["run_label"],
                    "run_b_sample_kind": right["sample_kind"],
                    "run_b_replicate": int(right["replicate"]),
                    "run_b_model_seed": int(right["model_seed"]),
                    "n_lags": len(ANALYSIS_LAGS),
                    "pearson_r_raw_f1": pearson_matrix.loc[left_fit, right_fit],
                    "spearman_rho_raw_f1": spearman_matrix.loc[
                        left_fit, right_fit
                    ],
                    "f1_profile_rmse": float(
                        np.sqrt(np.mean(profile_difference**2))
                    ),
                    "f1_profile_max_absolute_difference": float(
                        np.max(np.abs(profile_difference))
                    ),
                    "run_a_f1_dynamic_range": float(
                        np.ptp(profiles[left_fit])
                    ),
                    "run_b_f1_dynamic_range": float(
                        np.ptp(profiles[right_fit])
                    ),
                    "absolute_f1_dynamic_range_difference": abs(
                        float(
                            np.ptp(profiles[left_fit])
                            - np.ptp(profiles[right_fit])
                        )
                    ),
                    "run_a_profile_constant": bool(
                        np.ptp(profiles[left_fit]) <= TIE_ATOL
                    ),
                    "run_b_profile_constant": bool(
                        np.ptp(profiles[right_fit]) <= TIE_ATOL
                    ),
                    "run_a_peak_lag": left_peak,
                    "run_b_peak_lag": right_peak,
                    "same_peak_lag": left_peak == right_peak,
                    "peak_grid_steps_apart": steps,
                    "absolute_peak_lag_difference_frames": abs(
                        left_peak - right_peak
                    ),
                    "absolute_peak_lag_difference_s": (
                        abs(left_peak - right_peak) * SECONDS_PER_FRAME
                    ),
                    "different_peaks_are_adjacent_on_grid": bool(
                        left_peak != right_peak and steps == 1
                    ),
                    "run_a_f1_loss_at_run_b_selected_lag": left_selected_gap,
                    "run_b_f1_loss_at_run_a_selected_lag": right_selected_gap,
                    "max_cross_selected_lag_f1_loss": max(
                        left_selected_gap, right_selected_gap
                    ),
                    "mean_cross_selected_lag_f1_loss": (
                        left_selected_gap + right_selected_gap
                    )
                    / 2.0,
                    "run_a_peak_margin_f1": float(left_metric["peak_margin_f1"]),
                    "run_b_peak_margin_f1": float(right_metric["peak_margin_f1"]),
                    "absolute_center_of_mass_difference_lag": abs(
                        float(left_metric["center_of_mass_lag"])
                        - float(right_metric["center_of_mass_lag"])
                    ),
                    "absolute_center_of_mass_difference_s": abs(
                        float(left_metric["center_of_mass_s"])
                        - float(right_metric["center_of_mass_s"])
                    ),
                    "absolute_weighted_sd_difference_lag": abs(
                        float(left_metric["weighted_sd_lag"])
                        - float(right_metric["weighted_sd_lag"])
                    ),
                    "absolute_weighted_sd_difference_s": abs(
                        float(left_metric["weighted_sd_s"])
                        - float(right_metric["weighted_sd_s"])
                    ),
                }
            )
    return (
        pd.DataFrame(rows).sort_values(
            ["network", "run_a_sample_kind", "run_a", "run_b"]
        ).reset_index(drop=True),
        matrices,
    )


def run_vs_paper_profile_rows(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    paper_curves: pd.DataFrame,
    paper_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each retraining with the paper reference on the valid lag-2+ grid."""
    columns = (
        "comparison_scope",
        "method_comparable",
        "network",
        "fit_label",
        "run_label",
        "sample_kind",
        "replicate",
        "model_seed",
        "n_lags",
        "pearson_r_raw_f1_vs_paper",
        "spearman_rho_raw_f1_vs_paper",
        "f1_profile_rmse_vs_paper",
        "f1_profile_max_absolute_difference_vs_paper",
        "run_f1_dynamic_range",
        "paper_f1_dynamic_range",
        "run_peak_lag",
        "paper_peak_lag",
        "same_peak_lag_as_paper",
        "run_center_of_mass_s",
        "paper_center_of_mass_s",
        "center_of_mass_minus_paper_s",
        "absolute_center_of_mass_difference_s",
        "run_weighted_sd_s",
        "paper_weighted_sd_s",
        "weighted_sd_minus_paper_s",
        "absolute_weighted_sd_difference_s",
    )
    if is_all8_diagnostic():
        # Deliberately do not combine the incompatible published lag-1 point
        # with the consistent Approach-C retraining profiles.
        return pd.DataFrame(columns=columns)
    metric_lookup = metrics.set_index(["network", "fit_label"])
    reference_metric_lookup = paper_metrics.set_index("network")
    rows: list[dict] = []
    for network in MONOAMINES:
        reference_profile = ordered_complete_profile(
            paper_curves.loc[paper_curves["network"] == network],
            f"paper_reference/{network}",
        )["f1"].to_numpy(dtype=float)
        for fit, group in curves.loc[curves["network"] == network].groupby(
            "fit_label", sort=False
        ):
            ordered = ordered_complete_profile(group, f"{fit}/{network}")
            values = ordered["f1"].to_numpy(dtype=float)
            profile_difference = values - reference_profile
            run_metric = metric_lookup.loc[(network, fit)]
            reference_metric = reference_metric_lookup.loc[network]
            rows.append(
                {
                    "comparison_scope": (
                        "valid_common_approach_c_lag2plus_vs_paper_reference"
                    ),
                    "method_comparable": True,
                    "network": network,
                    "fit_label": fit,
                    "run_label": run_metric["run_label"],
                    "sample_kind": run_metric["sample_kind"],
                    "replicate": int(run_metric["replicate"]),
                    "model_seed": int(run_metric["model_seed"]),
                    "n_lags": len(ANALYSIS_LAGS),
                    "pearson_r_raw_f1_vs_paper": pearson_correlation(
                        values, reference_profile
                    ),
                    "spearman_rho_raw_f1_vs_paper": spearman_correlation(
                        values, reference_profile
                    ),
                    "f1_profile_rmse_vs_paper": float(
                        np.sqrt(np.mean(profile_difference**2))
                    ),
                    "f1_profile_max_absolute_difference_vs_paper": float(
                        np.max(np.abs(profile_difference))
                    ),
                    "run_f1_dynamic_range": float(np.ptp(values)),
                    "paper_f1_dynamic_range": float(
                        np.ptp(reference_profile)
                    ),
                    "run_peak_lag": int(run_metric["peak_lag"]),
                    "paper_peak_lag": int(reference_metric["peak_lag"]),
                    "same_peak_lag_as_paper": bool(
                        int(run_metric["peak_lag"])
                        == int(reference_metric["peak_lag"])
                    ),
                    "run_center_of_mass_s": float(
                        run_metric["center_of_mass_s"]
                    ),
                    "paper_center_of_mass_s": float(
                        reference_metric["center_of_mass_s"]
                    ),
                    "center_of_mass_minus_paper_s": float(
                        run_metric["center_of_mass_s"]
                        - reference_metric["center_of_mass_s"]
                    ),
                    "absolute_center_of_mass_difference_s": abs(
                        float(
                            run_metric["center_of_mass_s"]
                            - reference_metric["center_of_mass_s"]
                        )
                    ),
                    "run_weighted_sd_s": float(run_metric["weighted_sd_s"]),
                    "paper_weighted_sd_s": float(
                        reference_metric["weighted_sd_s"]
                    ),
                    "weighted_sd_minus_paper_s": float(
                        run_metric["weighted_sd_s"]
                        - reference_metric["weighted_sd_s"]
                    ),
                    "absolute_weighted_sd_difference_s": abs(
                        float(
                            run_metric["weighted_sd_s"]
                            - reference_metric["weighted_sd_s"]
                        )
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["network", "sample_kind", "replicate", "model_seed"]
    ).reset_index(drop=True)


def run_vs_paper_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "comparison_scope",
        "method_comparable",
        "network",
        "n_runs",
        "pearson_median",
        "pearson_min",
        "pearson_max",
        "spearman_median",
        "spearman_min",
        "spearman_max",
        "f1_profile_rmse_median",
        "f1_profile_rmse_max",
        "f1_profile_max_absolute_difference_median",
        "f1_profile_max_absolute_difference_max",
        "center_of_mass_abs_difference_median_s",
        "center_of_mass_abs_difference_max_s",
        "weighted_sd_abs_difference_median_s",
        "weighted_sd_abs_difference_max_s",
        "same_peak_lag_count",
    )
    if comparisons.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for network, group in comparisons.groupby("network"):
        rows.append(
            {
                "comparison_scope": group["comparison_scope"].iloc[0],
                "method_comparable": True,
                "network": network,
                "n_runs": len(group),
                "pearson_median": float(
                    group["pearson_r_raw_f1_vs_paper"].median()
                ),
                "pearson_min": float(
                    group["pearson_r_raw_f1_vs_paper"].min()
                ),
                "pearson_max": float(
                    group["pearson_r_raw_f1_vs_paper"].max()
                ),
                "spearman_median": float(
                    group["spearman_rho_raw_f1_vs_paper"].median()
                ),
                "spearman_min": float(
                    group["spearman_rho_raw_f1_vs_paper"].min()
                ),
                "spearman_max": float(
                    group["spearman_rho_raw_f1_vs_paper"].max()
                ),
                "f1_profile_rmse_median": float(
                    group["f1_profile_rmse_vs_paper"].median()
                ),
                "f1_profile_rmse_max": float(
                    group["f1_profile_rmse_vs_paper"].max()
                ),
                "f1_profile_max_absolute_difference_median": float(
                    group[
                        "f1_profile_max_absolute_difference_vs_paper"
                    ].median()
                ),
                "f1_profile_max_absolute_difference_max": float(
                    group[
                        "f1_profile_max_absolute_difference_vs_paper"
                    ].max()
                ),
                "center_of_mass_abs_difference_median_s": float(
                    group["absolute_center_of_mass_difference_s"].median()
                ),
                "center_of_mass_abs_difference_max_s": float(
                    group["absolute_center_of_mass_difference_s"].max()
                ),
                "weighted_sd_abs_difference_median_s": float(
                    group["absolute_weighted_sd_difference_s"].median()
                ),
                "weighted_sd_abs_difference_max_s": float(
                    group["absolute_weighted_sd_difference_s"].max()
                ),
                "same_peak_lag_count": int(group["same_peak_lag_as_paper"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("network").reset_index(
        drop=True
    )


def sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=1)) if len(array) > 1 else float("nan")


def metric_variability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = (
        ("all_completed_retrainings", metrics),
        ("full_data_seed_controls", metrics.loc[metrics["sample_kind"] == "full"]),
        ("bootstrap_individual_seeds", metrics.loc[metrics["sample_kind"] == "bootstrap"]),
    )
    for network in MONOAMINES:
        for scope, frame in scopes:
            group = frame.loc[frame["network"] == network]
            if group.empty:
                continue
            centers = group["center_of_mass_lag"].to_numpy(dtype=float)
            widths = group["weighted_sd_lag"].to_numpy(dtype=float)
            counts = group["peak_lag"].value_counts().sort_index()
            max_count = int(counts.max())
            modes = counts.loc[counts == max_count].index.astype(int).tolist()
            rows.append(
                {
                    "network": network,
                    "summary_scope": scope,
                    "n_runs": len(group),
                    "n_zero_sum_profiles": int(
                        (~group["profile_moments_defined"]).sum()
                    ),
                    "center_of_mass_mean_lag": float(np.mean(centers)),
                    "center_of_mass_sd_lag": sample_sd(centers),
                    "center_of_mass_min_lag": float(np.min(centers)),
                    "center_of_mass_max_lag": float(np.max(centers)),
                    "center_of_mass_range_lag": float(np.ptp(centers)),
                    "center_of_mass_mean_s": float(
                        np.mean(centers) * SECONDS_PER_FRAME
                    ),
                    "center_of_mass_sd_s": sample_sd(
                        centers * SECONDS_PER_FRAME
                    ),
                    "center_of_mass_min_s": float(
                        np.min(centers) * SECONDS_PER_FRAME
                    ),
                    "center_of_mass_max_s": float(
                        np.max(centers) * SECONDS_PER_FRAME
                    ),
                    "center_of_mass_range_s": float(
                        np.ptp(centers) * SECONDS_PER_FRAME
                    ),
                    "weighted_sd_mean_lag": float(np.mean(widths)),
                    "weighted_sd_sd_lag": sample_sd(widths),
                    "weighted_sd_min_lag": float(np.min(widths)),
                    "weighted_sd_max_lag": float(np.max(widths)),
                    "weighted_sd_range_lag": float(np.ptp(widths)),
                    "weighted_sd_mean_s": float(
                        np.mean(widths) * SECONDS_PER_FRAME
                    ),
                    "weighted_sd_sd_s": sample_sd(
                        widths * SECONDS_PER_FRAME
                    ),
                    "weighted_sd_min_s": float(
                        np.min(widths) * SECONDS_PER_FRAME
                    ),
                    "weighted_sd_max_s": float(
                        np.max(widths) * SECONDS_PER_FRAME
                    ),
                    "weighted_sd_range_s": float(
                        np.ptp(widths) * SECONDS_PER_FRAME
                    ),
                    "observed_peak_lags": ",".join(
                        str(value) for value in counts.index.astype(int)
                    ),
                    "modal_peak_lags": ",".join(str(value) for value in modes),
                    "modal_peak_count": max_count,
                    "peak_margin_f1_median": float(
                        group["peak_margin_f1"].median()
                    ),
                    "peak_margin_f1_max": float(group["peak_margin_f1"].max()),
                }
            )
    return pd.DataFrame(rows)


def correlation_similarity_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    requested_scopes = (
        "all_pairs",
        "full_vs_full",
        "full_vs_bootstrap",
        "bootstrap_within_resample_seeds",
        "bootstrap_across_resamples",
        "all_bootstrap_pairs",
    )
    for network in MONOAMINES:
        network_pairs = pairwise.loc[pairwise["network"] == network]
        for scope in requested_scopes:
            if scope == "all_pairs":
                group = network_pairs
            elif scope == "all_bootstrap_pairs":
                group = network_pairs.loc[
                    network_pairs["run_a_sample_kind"].eq("bootstrap")
                    & network_pairs["run_b_sample_kind"].eq("bootstrap")
                ]
            else:
                group = network_pairs.loc[network_pairs["pair_scope"] == scope]
            if group.empty:
                continue
            different = group.loc[~group["same_peak_lag"]]
            pearson = group["pearson_r_raw_f1"].dropna()
            spearman = group["spearman_rho_raw_f1"].dropna()
            rows.append(
                {
                    "network": network,
                    "pair_summary_scope": scope,
                    "n_pairs": len(group),
                    "n_finite_pearson": len(pearson),
                    "pearson_median": float(pearson.median())
                    if len(pearson)
                    else float("nan"),
                    "pearson_min": float(pearson.min())
                    if len(pearson)
                    else float("nan"),
                    "pearson_max": float(pearson.max())
                    if len(pearson)
                    else float("nan"),
                    "n_finite_spearman": len(spearman),
                    "spearman_median": float(spearman.median())
                    if len(spearman)
                    else float("nan"),
                    "spearman_min": float(spearman.min())
                    if len(spearman)
                    else float("nan"),
                    "spearman_max": float(spearman.max())
                    if len(spearman)
                    else float("nan"),
                    "f1_profile_rmse_median": float(
                        group["f1_profile_rmse"].median()
                    ),
                    "f1_profile_rmse_max": float(group["f1_profile_rmse"].max()),
                    "f1_profile_max_absolute_difference_median": float(
                        group["f1_profile_max_absolute_difference"].median()
                    ),
                    "f1_profile_max_absolute_difference_max": float(
                        group["f1_profile_max_absolute_difference"].max()
                    ),
                    "f1_dynamic_range_abs_difference_median": float(
                        group["absolute_f1_dynamic_range_difference"].median()
                    ),
                    "f1_dynamic_range_abs_difference_max": float(
                        group["absolute_f1_dynamic_range_difference"].max()
                    ),
                    "center_of_mass_abs_difference_median_s": float(
                        group["absolute_center_of_mass_difference_s"].median()
                    ),
                    "center_of_mass_abs_difference_max_s": float(
                        group["absolute_center_of_mass_difference_s"].max()
                    ),
                    "weighted_sd_abs_difference_median_s": float(
                        group["absolute_weighted_sd_difference_s"].median()
                    ),
                    "weighted_sd_abs_difference_max_s": float(
                        group["absolute_weighted_sd_difference_s"].max()
                    ),
                    "n_pairs_with_different_peak": len(different),
                    "n_different_peak_pairs_adjacent_on_grid": int(
                        different["different_peaks_are_adjacent_on_grid"].sum()
                    ),
                    "fraction_different_peak_pairs_adjacent_on_grid": (
                        float(
                            different[
                                "different_peaks_are_adjacent_on_grid"
                            ].mean()
                        )
                        if len(different)
                        else float("nan")
                    ),
                    "different_peak_absolute_shift_median_s": (
                        float(different["absolute_peak_lag_difference_s"].median())
                        if len(different)
                        else float("nan")
                    ),
                    "different_peak_absolute_shift_max_s": (
                        float(different["absolute_peak_lag_difference_s"].max())
                        if len(different)
                        else float("nan")
                    ),
                    "different_peak_cross_selected_f1_loss_median": (
                        float(different["max_cross_selected_lag_f1_loss"].median())
                        if len(different)
                        else float("nan")
                    ),
                    "different_peak_cross_selected_f1_loss_max": (
                        float(different["max_cross_selected_lag_f1_loss"].max())
                        if len(different)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def timescale_order_rows(
    metrics: pd.DataFrame, tie_atol: float = TIE_ATOL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank monoamines shortest-to-longest using tolerance-aware dense ties."""
    rows: list[dict] = []
    signatures: list[dict] = []
    for fit, group in metrics.groupby("fit_label", sort=False):
        if set(group["network"]) != set(MONOAMINES):
            raise RuntimeError(f"{fit} lacks one or more monoamine COM values")
        if not group["profile_moments_defined"].all():
            raise RuntimeError(f"{fit} has an undefined zero-sum monoamine profile")
        ordered = group.sort_values(
            ["center_of_mass_lag", "network"]
        ).reset_index(drop=True)
        tie_groups: list[list[pd.Series]] = []
        for _, row in ordered.iterrows():
            if not tie_groups:
                tie_groups.append([row])
                continue
            reference = float(tie_groups[-1][0]["center_of_mass_lag"])
            if math.isclose(
                float(row["center_of_mass_lag"]),
                reference,
                rel_tol=0,
                abs_tol=tie_atol,
            ):
                tie_groups[-1].append(row)
            else:
                tie_groups.append([row])
        signature_parts: list[str] = []
        has_tie = False
        for rank, tied_rows in enumerate(tie_groups, start=1):
            names = sorted(str(row["network"]) for row in tied_rows)
            signature_parts.append(" = ".join(names))
            has_tie = has_tie or len(tied_rows) > 1
            for row in tied_rows:
                rows.append(
                    {
                        "fit_label": fit,
                        "run_label": row["run_label"],
                        "display_label": row["display_label"],
                        "run_order": int(row["run_order"]),
                        "sample_kind": row["sample_kind"],
                        "replicate": int(row["replicate"]),
                        "model_seed": int(row["model_seed"]),
                        "network": row["network"],
                        "center_of_mass_lag": float(row["center_of_mass_lag"]),
                        "center_of_mass_s": float(row["center_of_mass_s"]),
                        "timescale_dense_rank_short_to_long": rank,
                        "tie_group_size": len(tied_rows),
                        "is_tied": len(tied_rows) > 1,
                    }
                )
        signature = " < ".join(signature_parts)
        first = ordered.iloc[0]
        signatures.append(
            {
                "fit_label": fit,
                "run_label": first["run_label"],
                "display_label": first["display_label"],
                "run_order": int(first["run_order"]),
                "sample_kind": first["sample_kind"],
                "replicate": int(first["replicate"]),
                "model_seed": int(first["model_seed"]),
                "timescale_order_short_to_long": signature,
                "has_tie": has_tie,
                "n_distinct_timescale_ranks": len(tie_groups),
                "tie_tolerance_lag_frames": tie_atol,
            }
        )
    rank_frame = pd.DataFrame(rows).sort_values(
        ["run_order", "timescale_dense_rank_short_to_long", "network"]
    )
    signature_frame = pd.DataFrame(signatures).sort_values("run_order")
    return rank_frame.reset_index(drop=True), signature_frame.reset_index(drop=True)


def order_frequency_rows(signatures: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = (
        ("all_completed_retrainings", signatures),
        (
            "full_data_seed_controls",
            signatures.loc[signatures["sample_kind"] == "full"],
        ),
        (
            "bootstrap_individual_seeds",
            signatures.loc[signatures["sample_kind"] == "bootstrap"],
        ),
    )
    for scope, group in scopes:
        if group.empty:
            continue
        counts = group["timescale_order_short_to_long"].value_counts()
        max_count = int(counts.max())
        for signature, count in counts.items():
            rows.append(
                {
                    "summary_scope": scope,
                    "timescale_order_short_to_long": signature,
                    "count": int(count),
                    "n_runs": len(group),
                    "frequency": float(count / len(group)),
                    "is_modal_order": int(count) == max_count,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["summary_scope", "count", "timescale_order_short_to_long"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def annotate_published_order_recovery(
    signatures: pd.DataFrame,
    paper_signatures: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a valid lag-2+ order anchor, or an explicit all8 invalid marker."""
    if len(paper_signatures) != 1:
        raise RuntimeError("Expected exactly one paper-reference order signature")
    published_order = str(
        paper_signatures.iloc[0]["timescale_order_short_to_long"]
    )
    annotated = signatures.copy()
    comparable = not is_all8_diagnostic()
    annotated["published_paper_order_on_analysis_grid"] = published_order
    annotated["published_order_method_comparable"] = comparable
    annotated["published_order_comparison_status"] = (
        "valid_common_approach_c_lag2plus"
        if comparable
        else (
            "invalid_not_compared_published_lag1_estimator_incompatible_with_"
            "retraining_approach_c_lag1"
        )
    )
    if comparable:
        annotated["recovers_published_paper_order"] = (
            annotated["timescale_order_short_to_long"] == published_order
        )
    else:
        annotated["recovers_published_paper_order"] = pd.Series(
            pd.array([pd.NA] * len(annotated), dtype="boolean"),
            index=annotated.index,
        )
    return annotated


def bootstrap_seed_ensemble_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Average the two algorithmic repeats within each independent resample."""
    bootstrap = curves.loc[curves["sample_kind"] == "bootstrap"].copy()
    rows = []
    for replicate in BOOTSTRAP_REPLICATES:
        group = bootstrap.loc[bootstrap["replicate"] == replicate]
        seeds = sorted(group["model_seed"].unique().astype(int).tolist())
        if seeds != list(BOOTSTRAP_MODEL_SEEDS):
            continue
        averaged = (
            group.groupby(["network", "lag", "time_s"], as_index=False)
            .agg(f1=("f1", "mean"), best_threshold=("best_threshold", "mean"))
        )
        averaged["fit_label"] = f"bootstrap_{replicate:03d}_seed_ensemble"
        averaged["omit_index"] = replicate
        averaged["sample_kind"] = "bootstrap_seed_ensemble"
        averaged["replicate"] = replicate
        averaged["model_seed"] = -1
        averaged["run_label"] = f"b{replicate}_seed_mean"
        averaged["display_label"] = f"Bootstrap {replicate} · two-seed mean"
        averaged["run_order"] = replicate
        averaged["n_model_seeds_averaged"] = len(seeds)
        rows.append(averaged)
    if not rows:
        return pd.DataFrame(columns=curves.columns)
    ensemble = pd.concat(rows, ignore_index=True)
    totals = ensemble.groupby(["fit_label", "network"])["f1"].transform("sum")
    ensemble["profile_weight"] = np.where(
        totals > 0, ensemble["f1"] / totals, np.nan
    )
    for (fit, network), group in ensemble.groupby(["fit_label", "network"]):
        ordered_complete_profile(group, f"{fit}/{network}")
    return ensemble.sort_values(["run_order", "network", "lag"]).reset_index(
        drop=True
    )


def plot_profile_overlays(
    curves: pd.DataFrame,
    out_dir: Path,
    log_y: bool = False,
) -> list[Path]:
    """Export a polished 2x2 individual-curve overlay in PNG/PDF/SVG."""
    if log_y and bool((curves["f1"] <= 0).any()):
        bad = curves.loc[curves["f1"] <= 0, ["fit_label", "network", "lag", "f1"]]
        raise RuntimeError(
            "True log-y overlays require every F1 value to be strictly positive: "
            + bad.to_json(orient="records")
        )
    cache_dir = Path(tempfile.gettempdir()) / "sbtg_lag_profile_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LogLocator

    full_colors = {42: "#1E4F72", 43: "#4E7FA3", 44: "#88AFC8"}
    bootstrap_colors = {
        0: "#9A4E0A",
        1: "#BD6718",
        2: "#D88732",
        3: "#E5A866",
    }
    seed_styles = {
        42: ("--", "^"),
        43: (":", "v"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), sharex=True)
    axes = axes.ravel()
    run_meta = (
        curves.loc[
            :,
            [
                "fit_label",
                "display_label",
                "run_order",
                "sample_kind",
                "replicate",
                "model_seed",
            ],
        ]
        .drop_duplicates()
        .sort_values("run_order")
    )
    legend_handles: list[Line2D] = []

    for axis, network in zip(axes, MONOAMINES):
        network_curves = curves.loc[curves["network"] == network]
        for run in run_meta.itertuples():
            profile = network_curves.loc[
                network_curves["fit_label"] == run.fit_label
            ].sort_values("lag")
            if profile.empty:
                continue
            if run.sample_kind == "full":
                color = full_colors[int(run.model_seed)]
                line_style = "-"
                marker = "o"
                line_width = 2.25
                alpha = 0.95
                marker_face = "white"
            else:
                color = bootstrap_colors[int(run.replicate)]
                line_style, marker = seed_styles[int(run.model_seed)]
                line_width = 1.45
                alpha = 0.78
                marker_face = color
            axis.plot(
                profile["lag"],
                profile["f1"],
                color=color,
                linestyle=line_style,
                linewidth=line_width,
                alpha=alpha,
                marker=marker,
                markersize=4.4,
                markerfacecolor=marker_face,
                markeredgecolor=color,
                markeredgewidth=0.8,
                zorder=3 if run.sample_kind == "full" else 2,
            )
        axis.set_title(network.capitalize(), loc="left", fontsize=13, weight="bold")
        axis.set_xticks(ANALYSIS_LAGS)
        if log_y:
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:.3f}")
            )
        else:
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.75, alpha=0.8)
        axis.grid(axis="x", color="#EDF0F2", linewidth=0.55, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#59636D")
        axis.spines["bottom"].set_color("#59636D")
        axis.tick_params(colors="#3C454D", labelsize=9)
        axis.margins(x=0.02, y=0.10)

    for axis in (axes[0], axes[2]):
        axis.set_ylabel(
            "Best F1 (true log scale)" if log_y else "Best F1",
            fontsize=10,
            color="#252B31",
        )
    for axis in (axes[2], axes[3]):
        axis.set_xlabel("Lag (frames; 0.25 s per frame)", fontsize=10)

    for run in run_meta.itertuples():
        if run.sample_kind == "full":
            color = full_colors[int(run.model_seed)]
            line_style, marker, marker_face = "-", "o", "white"
            line_width = 2.25
        else:
            color = bootstrap_colors[int(run.replicate)]
            line_style, marker = seed_styles[int(run.model_seed)]
            marker_face = color
            line_width = 1.45
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=line_style,
                linewidth=line_width,
                marker=marker,
                markersize=5,
                markerfacecolor=marker_face,
                markeredgecolor=color,
                label=run.display_label,
            )
        )

    figure_title = (
        "SECONDARY lag-profile all-eight-lag retraining sensitivity F1 profiles"
        if is_all8_diagnostic()
        else "lag-profile monoamine F1 profiles across completed individual retrainings"
    ) + (" · true log-y view" if log_y else "")
    figure_subtitle = (
        "All retrainings use Approach-C at lag 1; the released lag-1 point is "
        "estimator-incompatible, so this is not published-full-curve stability."
        if is_all8_diagnostic()
        else (
            "Primary common Approach-C grid: lags 2, 3, 5, 8, 10, 15, 20. "
            "Lag 1 excluded; focused F1 scale within each panel."
        )
    )
    if log_y:
        figure_subtitle += (
            " True log scale, with no offset or clipping, emphasizes relative "
            "differences at low absolute F1."
        )
    fig.suptitle(
        figure_title,
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=17,
        weight="bold",
        color="#20262C",
    )
    fig.text(
        0.06,
        0.94,
        figure_subtitle,
        ha="left",
        fontsize=10.5,
        color="#59636D",
    )
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.805, 0.51),
        frameon=False,
        fontsize=9,
        title="Completed retraining",
        title_fontsize=10,
        labelspacing=0.8,
    )
    fig.text(
        0.805,
        0.18,
        (
            "Encoding\n"
            "Blue solid/open circle: full data\n"
            "Orange dashed/dotted: bootstrap\n"
            "Bootstrap color: resample\n"
            "Bootstrap marker/style: seed"
        ),
        ha="left",
        va="top",
        fontsize=8.5,
        color="#59636D",
        linespacing=1.45,
    )
    fig.subplots_adjust(left=0.065, right=0.78, top=0.89, bottom=0.085, wspace=0.22, hspace=0.25)

    outputs = []
    plot_stem = (
        "lag_profile_stability_overlays_log_y"
        if log_y
        else "lag_profile_stability_overlays"
    )
    for suffix in ("png", "pdf", "svg"):
        destination = out_dir / scoped_output_name(
            f"{plot_stem}.{suffix}"
        )
        temporary = destination.with_name(destination.name + ".tmp")
        fig.savefig(
            temporary,
            format=suffix,
            dpi=220 if suffix == "png" else None,
            facecolor="white",
            bbox_inches="tight",
        )
        os.replace(temporary, destination)
        outputs.append(destination)
    plt.close(fig)
    return outputs


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def format_range(median: float, minimum: float, maximum: float) -> str:
    return f"{median:.2f} [{minimum:.2f}, {maximum:.2f}]"


def interpretation_markdown(
    completion: pd.DataFrame,
    variability: pd.DataFrame,
    similarity: pd.DataFrame,
    metrics: pd.DataFrame,
    signatures: pd.DataFrame,
    order_frequency: pd.DataFrame,
    b4_variability: pd.DataFrame,
    b4_similarity: pd.DataFrame,
    b4_signatures: pd.DataFrame,
    paper_metrics: pd.DataFrame,
    paper_signatures: pd.DataFrame,
    reference_comparison_summary: pd.DataFrame,
) -> str:
    complete = completion.loc[completion["complete"]]
    n_full = int((complete["sample_kind"] == "full").sum())
    n_bootstrap = int((complete["sample_kind"] == "bootstrap").sum())
    b4_complete = len(b4_signatures)
    grid_text = ", ".join(str(lag) for lag in ANALYSIS_LAGS)
    grid_min_s = min(ANALYSIS_LAGS) * SECONDS_PER_FRAME
    grid_max_s = max(ANALYSIS_LAGS) * SECONDS_PER_FRAME
    uniform = uniform_profile_reference()
    scope_title = (
        "SECONDARY lag-profile all-eight-lag retraining sensitivity diagnostic"
        if is_all8_diagnostic()
        else "lag-profile primary lag-2+ profile-level stability analysis"
    )
    if is_all8_diagnostic():
        scope_note = (
            "All 11 retrainings use the same Approach-C estimator at lag 1, so "
            "this mode is consistent across retrainings. The paper "
            "released lag-1 point used an incompatible estimator. This "
            "secondary analysis therefore cannot assess stability of the published "
            "full eight-lag curve; no run-versus-published all8 correlations or "
            "published-order recovery counts are computed."
        )
    else:
        scope_note = (
            "This is the primary valid production-lineage comparison. Lag 1 is "
            "excluded because the released lag-1 estimator is incompatible; "
            "lags 2+ share the Approach-C lineage."
        )

    b4_sim = b4_similarity.loc[
        b4_similarity["pair_summary_scope"] == "all_pairs"
    ].set_index("network")
    b4_var = b4_variability.loc[
        b4_variability["summary_scope"] == "all_completed_retrainings"
    ].set_index("network")
    individual_sim = similarity.loc[
        similarity["pair_summary_scope"] == "all_pairs"
    ].set_index("network")
    within_seed_sim = similarity.loc[
        similarity["pair_summary_scope"]
        == "bootstrap_within_resample_seeds"
    ].set_index("network")
    individual_var = variability.loc[
        variability["summary_scope"] == "all_completed_retrainings"
    ].set_index("network")

    lines = [
        f"# {scope_title}",
        "",
        "## Scope and estimand",
        "",
        (
            f"Complete inputs: **{n_full}/3 full-data seed controls**, "
            f"**{n_bootstrap}/8 individual bootstrap fits**, and **{b4_complete}/4 "
            "two-seed-averaged bootstrap profiles**. The four seed-averaged "
            "profiles are the B=4 recording-row resampling summary. Individual "
            "seeds and full-data controls describe algorithmic variability and are "
            "not additional independent resamples."
        ),
        "",
        scope_note,
        "",
        (
            f"Grid: {grid_text} frames ({grid_min_s:g}--{grid_max_s:g} s; "
            "0.25 s/frame). Pearson and Spearman use raw F1. F1 is normalized "
            "only to compute the equal-grid-point centroid and weighted dispersion. "
            "Normalization cancels multiplicative scale but does not remove a "
            "positive F1 floor."
        ),
        "",
        "## B=4 seed-averaged bootstrap profiles",
        "",
        (
            "Correlations are median [minimum, maximum] across the six pairs of "
            "independent resample profiles. RMSE is median [maximum]. Centroid and "
            "dispersion are mean with observed range, in seconds."
        ),
        "",
        "| Network | Pearson r | Spearman rho | F1 RMSE | Centroid, s | Dispersion, s | Peak lags | Different-peak shift, s |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for network in MONOAMINES:
        sim = b4_sim.loc[network]
        var = b4_var.loc[network]
        peak_shift = (
            f"{float(sim['different_peak_absolute_shift_median_s']):.2f} "
            f"[{float(sim['different_peak_absolute_shift_max_s']):.2f} max]"
            if int(sim["n_pairs_with_different_peak"])
            else "none"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    network.capitalize(),
                    format_range(
                        float(sim["pearson_median"]),
                        float(sim["pearson_min"]),
                        float(sim["pearson_max"]),
                    ),
                    format_range(
                        float(sim["spearman_median"]),
                        float(sim["spearman_min"]),
                        float(sim["spearman_max"]),
                    ),
                    (
                        f"{float(sim['f1_profile_rmse_median']):.4f} "
                        f"[{float(sim['f1_profile_rmse_max']):.4f}]"
                    ),
                    (
                        f"{float(var['center_of_mass_mean_s']):.3f} "
                        f"[{float(var['center_of_mass_min_s']):.3f}, "
                        f"{float(var['center_of_mass_max_s']):.3f}]"
                    ),
                    (
                        f"{float(var['weighted_sd_mean_s']):.3f} "
                        f"[{float(var['weighted_sd_min_s']):.3f}, "
                        f"{float(var['weighted_sd_max_s']):.3f}]"
                    ),
                    str(var["observed_peak_lags"]),
                    peak_shift,
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Individual-fit algorithmic sensitivity",
            "",
            "| Network | All-fit Pearson median | All-fit Spearman median | Within-resample seed Pearson/Spearman | F1 RMSE median | Centroid range, s | Dispersion range, s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for network in MONOAMINES:
        sim = individual_sim.loc[network]
        within = within_seed_sim.loc[network]
        var = individual_var.loc[network]
        lines.append(
            "| "
            + " | ".join(
                (
                    network.capitalize(),
                    f"{float(sim['pearson_median']):.2f}",
                    f"{float(sim['spearman_median']):.2f}",
                    (
                        f"{float(within['pearson_median']):.2f} / "
                        f"{float(within['spearman_median']):.2f}"
                    ),
                    f"{float(sim['f1_profile_rmse_median']):.4f}",
                    f"{float(var['center_of_mass_range_s']):.3f}",
                    f"{float(var['weighted_sd_range_s']):.3f}",
                )
            )
            + " |"
        )

    published_order = str(
        paper_signatures.iloc[0]["timescale_order_short_to_long"]
    )
    if is_all8_diagnostic():
        reference_section = (
            "The released all-eight-lag centroid, dispersion, and order are exported "
            "for provenance only and marked non-comparable. It is not used for "
            "correlation, distance, or exact-order recovery because its lag-1 point "
            "does not share the retraining estimator."
        )
    else:
        individual_recovery = int(
            signatures["recovers_published_paper_order"].fillna(False).sum()
        )
        b4_recovery = int(
            b4_signatures["recovers_published_paper_order"].fillna(False).sum()
        )
        reference_bits = []
        reference_lookup = reference_comparison_summary.set_index("network")
        for network in MONOAMINES:
            row = reference_lookup.loc[network]
            reference_bits.append(
                f"{network} r={float(row['pearson_median']):.2f}, "
                f"rho={float(row['spearman_median']):.2f}"
            )
        reference_section = (
            f"On the valid lag-2+ grid, the released centroid order is "
            f"**{published_order}**. Exact strict numerical order recovery is "
            f"{individual_recovery}/11 individual fits and {b4_recovery}/4 B=4 "
            f"seed-averaged resamples. Median run-versus-published profile agreement: "
            + "; ".join(reference_bits)
            + "."
        )

    all_b4_different = int(b4_sim["n_pairs_with_different_peak"].sum())
    all_b4_adjacent = int(
        b4_sim["n_different_peak_pairs_adjacent_on_grid"].sum()
    )
    b4_orders = int(b4_signatures["timescale_order_short_to_long"].nunique())
    min_centroid = float(b4_var["center_of_mass_min_s"].min())
    max_centroid = float(b4_var["center_of_mass_max_s"].max())
    lines.extend(
        [
            "",
            "## Published reference and exact centroid ordering",
            "",
            reference_section,
            "",
            "## Direct interpretation",
            "",
            (
                "- **Profile shape:** The B=4 Pearson/Spearman, RMSE, and maximum-"
                "absolute-difference summaries above are the resampling-level evidence; "
                "the individual-fit table separately shows optimization variability. "
                "No correlation cutoff is used."
            ),
            "",
            (
                f"- **Discrete peaks:** {all_b4_adjacent}/{all_b4_different} B=4 "
                "profile pairs with different maxima are adjacent in grid order. "
                "Because the lag grid is irregular, the table also reports physical "
                "peak shifts in seconds rather than treating adjacency as equal distance."
                if all_b4_different
                else "- **Discrete peaks:** No B=4 profile pair changes the maximum."
            ),
            "",
            (
                "- **Tested-grid F1 centroid/dispersion:** B=4 centroids span "
                f"{min_centroid:.3f}--{max_centroid:.3f} s and have {b4_orders} "
                "strict numerical monoamine orderings across four resamples. These "
                f"are descriptors of {len(ANALYSIS_LAGS)} tested points, not estimates of a "
                "continuous biological integration-time distribution."
            ),
            "",
            "## Interpretation limits",
            "",
            (
                f"The equal-weight uniform-profile reference on this irregular grid "
                f"has centroid {float(uniform['center_of_mass_s']):.3f} s and "
                f"dispersion {float(uniform['weighted_sd_s']):.3f} s. Observed "
                "centroids and widths near those values can arise from a positive, "
                "fairly flat F1 floor; normalization does not subtract that floor. "
                "Each lag's best F1 also uses an independently optimized threshold, "
                "and equal grid-point weighting is not time-interval/AUC weighting."
            ),
            "",
            (
                "All results are descriptive. B=4 is insufficient for a percentile "
                "confidence interval. F1 sum is retained only as a normalization/QC "
                "denominator, not as a stability metric. Both linear and true log-y "
                "figures are exported; the log view uses no offset or clipping and "
                "only emphasizes relative differences at low absolute F1."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def build_run_summary(
    args: argparse.Namespace,
    completion: pd.DataFrame,
    metrics: pd.DataFrame,
    variability: pd.DataFrame,
    similarity: pd.DataFrame,
    order_frequency: pd.DataFrame,
    b4_variability: pd.DataFrame,
    b4_similarity: pd.DataFrame,
    b4_signatures: pd.DataFrame,
    paper_metrics: pd.DataFrame,
    paper_signatures: pd.DataFrame,
    reference_comparison_summary: pd.DataFrame,
    full_roots: dict[int, Path],
    output_files: Sequence[Path],
) -> dict:
    complete = completion.loc[completion["complete"]]
    completed_bootstrap = complete.loc[complete["sample_kind"] == "bootstrap"]
    paired_replicates = [
        replicate
        for replicate in BOOTSTRAP_REPLICATES
        if set(
            completed_bootstrap.loc[
                completed_bootstrap["replicate"] == replicate, "model_seed"
            ].astype(int)
        )
        == set(BOOTSTRAP_MODEL_SEEDS)
    ]
    all_similarity = similarity.loc[
        similarity["pair_summary_scope"] == "all_pairs"
    ].set_index("network")
    within_resample_similarity = similarity.loc[
        similarity["pair_summary_scope"]
        == "bootstrap_within_resample_seeds"
    ].set_index("network")
    all_variability = variability.loc[
        variability["summary_scope"] == "all_completed_retrainings"
    ].set_index("network")
    b4_all_similarity = b4_similarity.loc[
        b4_similarity["pair_summary_scope"] == "all_pairs"
    ].set_index("network")
    b4_all_variability = b4_variability.loc[
        b4_variability["summary_scope"] == "all_completed_retrainings"
    ].set_index("network")
    reference_lookup = (
        reference_comparison_summary.set_index("network")
        if not reference_comparison_summary.empty
        else None
    )
    network_summary = {}
    for network in MONOAMINES:
        sim = all_similarity.loc[network]
        within_sim = within_resample_similarity.loc[network]
        var = all_variability.loc[network]
        b4_sim = b4_all_similarity.loc[network]
        b4_var = b4_all_variability.loc[network]
        network_summary[network] = {
            "b4_seed_averaged_resampling_profiles": {
                "n_profiles": int(b4_var["n_runs"]),
                "pearson_median": finite_or_none(b4_sim["pearson_median"]),
                "spearman_median": finite_or_none(b4_sim["spearman_median"]),
                "f1_profile_rmse_median": finite_or_none(
                    b4_sim["f1_profile_rmse_median"]
                ),
                "f1_profile_max_absolute_difference_median": finite_or_none(
                    b4_sim["f1_profile_max_absolute_difference_median"]
                ),
                "tested_grid_f1_centroid_mean_s": finite_or_none(
                    b4_var["center_of_mass_mean_s"]
                ),
                "tested_grid_f1_centroid_range_s": finite_or_none(
                    b4_var["center_of_mass_range_s"]
                ),
                "tested_grid_f1_dispersion_mean_s": finite_or_none(
                    b4_var["weighted_sd_mean_s"]
                ),
                "tested_grid_f1_dispersion_range_s": finite_or_none(
                    b4_var["weighted_sd_range_s"]
                ),
                "observed_peak_lags": str(b4_var["observed_peak_lags"]),
                "different_peak_pair_count": int(
                    b4_sim["n_pairs_with_different_peak"]
                ),
                "adjacent_different_peak_pair_count": int(
                    b4_sim["n_different_peak_pairs_adjacent_on_grid"]
                ),
                "different_peak_absolute_shift_median_s": finite_or_none(
                    b4_sim["different_peak_absolute_shift_median_s"]
                ),
                "different_peak_absolute_shift_max_s": finite_or_none(
                    b4_sim["different_peak_absolute_shift_max_s"]
                ),
            },
            "algorithmic_sensitivity_11_individual_fits": {
                "n_runs": int(var["n_runs"]),
                "pearson_median_all_pairs": finite_or_none(sim["pearson_median"]),
                "spearman_median_all_pairs": finite_or_none(
                    sim["spearman_median"]
                ),
                "f1_profile_rmse_median": finite_or_none(
                    sim["f1_profile_rmse_median"]
                ),
                "pearson_median_within_resample_seed_pairs": finite_or_none(
                    within_sim["pearson_median"]
                ),
                "spearman_median_within_resample_seed_pairs": finite_or_none(
                    within_sim["spearman_median"]
                ),
                "tested_grid_f1_centroid_range_s": finite_or_none(
                    var["center_of_mass_range_s"]
                ),
                "tested_grid_f1_dispersion_range_s": finite_or_none(
                    var["weighted_sd_range_s"]
                ),
                "observed_peak_lags": str(var["observed_peak_lags"]),
                "peak_margin_f1_median": finite_or_none(
                    var["peak_margin_f1_median"]
                ),
            },
            "valid_run_vs_paper_lag2plus_summary": (
                {
                    key: finite_or_none(reference_lookup.loc[network, key])
                    for key in (
                        "pearson_median",
                        "spearman_median",
                        "f1_profile_rmse_median",
                        "center_of_mass_abs_difference_median_s",
                        "weighted_sd_abs_difference_median_s",
                    )
                }
                if reference_lookup is not None
                else None
            ),
        }
    all_orders = order_frequency.loc[
        (order_frequency["summary_scope"] == "all_completed_retrainings")
        & order_frequency["is_modal_order"]
    ]
    complete_count = int(completion["complete"].sum())
    comparable_reference = not is_all8_diagnostic()
    published_order = str(
        paper_signatures.iloc[0]["timescale_order_short_to_long"]
    )
    published_order_recovery = (
        int(b4_signatures["recovers_published_paper_order"].fillna(False).sum())
        if comparable_reference
        else None
    )
    uniform = uniform_profile_reference()
    grid_payload = {
        "lags_frames": list(ANALYSIS_LAGS),
        "lags_seconds": [value * SECONDS_PER_FRAME for value in ANALYSIS_LAGS],
        "seconds_per_frame": SECONDS_PER_FRAME,
        "irregular_grid": True,
        "equal_grid_point_weighting_not_time_interval_weighting": True,
    }
    return {
        "status": "complete" if complete_count == len(completion) else "in_progress",
        "analysis": (
            "production lag-profile secondary all-eight-lag retraining sensitivity diagnostic"
            if is_all8_diagnostic()
            else "production lag-profile primary lag-2+ profile-level stability"
        ),
        "analysis_scope": ANALYSIS_SCOPE,
        "analysis_grid": grid_payload,
        "scope_guard": {
            "secondary_diagnostic": is_all8_diagnostic(),
            "all_11_retrainings_use_approach_c_lag1": is_all8_diagnostic(),
            "published_paper_lag1_estimator_compatible": False,
            "published_full_eight_lag_curve_stability_assessable": False,
            "run_vs_published_profile_comparison_valid": comparable_reference,
            "published_order_recovery_valid": comparable_reference,
            "lag1_excluded_from_primary": not is_all8_diagnostic(),
            "reason": (
                "The released lag-1 point used an incompatible 180-epoch "
                "regime-gated estimator; all new retrainings use Approach C at lag 1."
            ),
        },
        "design": {
            "bootstrap_replicates": list(BOOTSTRAP_REPLICATES),
            "bootstrap_model_seeds": list(BOOTSTRAP_MODEL_SEEDS),
            "full_control_seeds": list(FULL_CONTROL_SEEDS),
            "requested_fit_count": len(completion),
            "complete_fit_count": complete_count,
            "complete_bootstrap_individual_fit_count": len(completed_bootstrap),
            "complete_paired_bootstrap_replicates": paired_replicates,
            "effective_independent_bootstrap_B": len(paired_replicates),
            "primary_resampling_profiles_are_two_seed_averages": True,
            "n_primary_seed_averaged_bootstrap_profiles": len(b4_signatures),
            "individual_bootstrap_fits_are_algorithmic_repeats": True,
            "full_control_fits_are_algorithmic_controls": True,
            "two_seeds_are_not_independent_resamples": True,
        },
        "completion": completion.to_dict(orient="records"),
        "formulas": {
            "tested_grid_f1_centroid": "sum(lag * F1) / sum(F1)",
            "tested_grid_f1_dispersion": (
                "sqrt(sum(w * (lag - centroid)^2)); w=F1/sum(F1)"
            ),
            "zero_sum_policy": "moments undefined; emit NaN in CSV and null in summary",
            "correlation_input": (
                f"raw unnormalized F1 at {len(ANALYSIS_LAGS)} tested lags"
            ),
            "spearman_ties": "average ranks",
            "constant_profile_correlation": "undefined (NaN)",
            "peak_tie_rule": "earliest lag; ties counted at absolute tolerance 1e-12",
            "monoamine_order": (
                "ascending tested-grid F1 centroid; dense ranks; equality at absolute "
                "tolerance 1e-12 frame"
            ),
            "f1_threshold_policy": (
                "best F1 threshold optimized independently for each run/network/lag"
            ),
            "area_under_curve_used_as_stability_metric": False,
        },
        "uniform_positive_floor_reference": {
            "definition": "equal F1 weight at every tested grid point",
            "tested_grid_f1_centroid_lag": finite_or_none(
                uniform["center_of_mass_lag"]
            ),
            "tested_grid_f1_centroid_s": finite_or_none(
                uniform["center_of_mass_s"]
            ),
            "tested_grid_f1_dispersion_lag": finite_or_none(
                uniform["weighted_sd_lag"]
            ),
            "tested_grid_f1_dispersion_s": finite_or_none(
                uniform["weighted_sd_s"]
            ),
            "normalization_cancels_scale_but_not_positive_f1_floor": True,
        },
        "log_y_figure": {
            "all_analyzed_f1_strictly_positive": bool((metrics["f1_min"] > 0).all()),
            "true_log_scale": True,
            "offset_added": False,
            "values_clipped": False,
        },
        "network_summary": network_summary,
        "modal_timescale_orders": [
            {
                "order": row.timescale_order_short_to_long,
                "count": int(row.count),
                "n_runs": int(row.n_runs),
                "frequency": float(row.frequency),
            }
            for row in all_orders.itertuples()
        ],
        "n_distinct_timescale_orders_across_runs": int(
            len(
                order_frequency.loc[
                    order_frequency["summary_scope"]
                    == "all_completed_retrainings"
                ]
            )
        ),
        "published_paper_reference": {
            "method_comparable_on_analysis_grid": comparable_reference,
            "reference_status": (
                "valid_common_approach_c_lag2plus"
                if comparable_reference
                else "provenance_only_not_compared_due_to_lag1_incompatibility"
            ),
            "tested_grid_f1_centroid_order": published_order,
            "b4_exact_order_recovery_count": published_order_recovery,
            "b4_exact_order_recovery_denominator": (
                len(b4_signatures) if comparable_reference else None
            ),
            "standalone_network_metrics": {
                row.network: {
                    "tested_grid_f1_centroid_s": float(row.center_of_mass_s),
                    "tested_grid_f1_dispersion_s": float(row.weighted_sd_s),
                }
                for row in paper_metrics.itertuples()
            },
        },
        "interpretation_policy": (
            "descriptive sensitivity analysis only; no bootstrap confidence interval; "
            "centroid/dispersion are tested-grid F1 descriptors, not continuous-time "
            "biological estimands"
        ),
        "input_root": portable_path(args.input_root),
        "full_control_sources": {
            str(seed): portable_path(full_roots[seed])
            for seed in FULL_CONTROL_SEEDS
        },
        "outputs": [portable_path(path) for path in output_files],
    }


def main() -> None:
    args = parse_args()
    configure_analysis_scope(args.analysis_scope)
    args.input_root = args.input_root.resolve()
    if args.out_dir is None:
        args.out_dir = (
            DEFAULT_ALL8_OUT_DIR if is_all8_diagnostic() else DEFAULT_OUT_DIR
        )
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curves, completion, full_roots, paper_curves = collect_completed_curves(args)
    metrics = run_metric_rows(curves)
    pairwise, matrices = pairwise_profile_rows(curves, metrics)
    variability = metric_variability_summary(metrics)
    similarity = correlation_similarity_summary(pairwise)
    order_ranks, order_signatures = timescale_order_rows(metrics)
    paper_metrics = run_metric_rows(paper_curves)
    paper_order_ranks, paper_order_signatures = timescale_order_rows(
        paper_metrics
    )
    comparable_reference = not is_all8_diagnostic()
    reference_status = (
        "valid_common_approach_c_lag2plus_reference"
        if comparable_reference
        else "provenance_only_not_comparable_due_to_published_lag1_estimator"
    )
    for frame in (paper_metrics, paper_order_ranks, paper_order_signatures):
        frame["method_comparable_to_retrainings"] = comparable_reference
        frame["reference_status"] = reference_status
    order_signatures = annotate_published_order_recovery(
        order_signatures, paper_order_signatures
    )
    order_frequency = order_frequency_rows(order_signatures)

    reference_comparisons = run_vs_paper_profile_rows(
        curves, metrics, paper_curves, paper_metrics
    )
    reference_comparison_summary = run_vs_paper_summary(reference_comparisons)

    b4_curves = bootstrap_seed_ensemble_curves(curves)
    b4_metrics = run_metric_rows(b4_curves)
    b4_pairwise, b4_matrices = pairwise_profile_rows(b4_curves, b4_metrics)
    b4_variability = metric_variability_summary(b4_metrics)
    b4_similarity = correlation_similarity_summary(b4_pairwise)
    b4_order_ranks, b4_order_signatures = timescale_order_rows(b4_metrics)
    b4_order_signatures = annotate_published_order_recovery(
        b4_order_signatures, paper_order_signatures
    )
    b4_order_frequency = order_frequency_rows(b4_order_signatures)
    b4_reference_comparisons = run_vs_paper_profile_rows(
        b4_curves, b4_metrics, paper_curves, paper_metrics
    )
    b4_reference_comparison_summary = run_vs_paper_summary(
        b4_reference_comparisons
    )

    output_files: list[Path] = []

    def output_csv(frame: pd.DataFrame, name: str) -> None:
        path = args.out_dir / scoped_output_name(name)
        write_csv(frame, path)
        output_files.append(path)

    output_csv(completion, "lag_profile_stability_fit_completion.csv")
    output_csv(curves, "lag_profile_stability_individual_curves.csv")
    output_csv(metrics, "lag_profile_stability_run_metrics.csv")
    output_csv(pairwise, "lag_profile_stability_pairwise_correlations_long.csv")
    output_csv(variability, "lag_profile_stability_metric_variability_summary.csv")
    output_csv(similarity, "lag_profile_stability_similarity_summary.csv")
    output_csv(order_ranks, "lag_profile_monoamine_timescale_ranks_by_run.csv")
    output_csv(order_signatures, "lag_profile_monoamine_timescale_order_by_run.csv")
    output_csv(order_frequency, "lag_profile_monoamine_timescale_order_frequency.csv")
    output_csv(paper_curves, "lag_profile_stability_paper_reference_curves.csv")
    output_csv(paper_metrics, "lag_profile_stability_paper_reference_metrics.csv")
    output_csv(
        paper_order_ranks,
        "lag_profile_stability_paper_reference_timescale_ranks.csv",
    )
    output_csv(
        paper_order_signatures,
        "lag_profile_stability_paper_reference_timescale_order.csv",
    )
    output_csv(
        reference_comparisons,
        "lag_profile_stability_run_vs_paper_comparisons.csv",
    )
    output_csv(
        reference_comparison_summary,
        "lag_profile_stability_run_vs_paper_summary.csv",
    )

    output_csv(b4_curves, "lag_profile_stability_b4_seed_ensemble_curves.csv")
    output_csv(b4_metrics, "lag_profile_stability_b4_seed_ensemble_metrics.csv")
    output_csv(
        b4_pairwise,
        "lag_profile_stability_b4_seed_ensemble_pairwise_correlations_long.csv",
    )
    output_csv(
        b4_variability,
        "lag_profile_stability_b4_seed_ensemble_variability_summary.csv",
    )
    output_csv(
        b4_similarity,
        "lag_profile_stability_b4_seed_ensemble_similarity_summary.csv",
    )
    output_csv(
        b4_order_ranks,
        "lag_profile_stability_b4_seed_ensemble_timescale_ranks.csv",
    )
    output_csv(
        b4_order_signatures,
        "lag_profile_stability_b4_seed_ensemble_timescale_order.csv",
    )
    output_csv(
        b4_order_frequency,
        "lag_profile_stability_b4_seed_ensemble_order_frequency.csv",
    )
    output_csv(
        b4_reference_comparisons,
        "lag_profile_stability_b4_seed_ensemble_vs_paper_comparisons.csv",
    )
    output_csv(
        b4_reference_comparison_summary,
        "lag_profile_stability_b4_seed_ensemble_vs_paper_summary.csv",
    )

    for (network, method), matrix in matrices.items():
        matrix_output = matrix.rename_axis("run_a").reset_index()
        output_csv(
            matrix_output,
            f"lag_profile_stability_{method}_matrix_{network}.csv",
        )

    for (network, method), matrix in b4_matrices.items():
        matrix_output = matrix.rename_axis("run_a").reset_index()
        output_csv(
            matrix_output,
            f"lag_profile_stability_b4_seed_ensemble_{method}_matrix_{network}.csv",
        )

    plot_paths = plot_profile_overlays(curves, args.out_dir)
    output_files.extend(plot_paths)
    log_plot_paths = plot_profile_overlays(curves, args.out_dir, log_y=True)
    output_files.extend(log_plot_paths)
    interpretation_path = args.out_dir / scoped_output_name(
        "lag_profile_stability_interpretation.md"
    )
    atomic_text(
        interpretation_path,
        interpretation_markdown(
            completion,
            variability,
            similarity,
            metrics,
            order_signatures,
            order_frequency,
            b4_variability,
            b4_similarity,
            b4_order_signatures,
            paper_metrics,
            paper_order_signatures,
            reference_comparison_summary,
        ),
    )
    output_files.append(interpretation_path)

    metadata_path = args.out_dir / scoped_output_name(
        "lag_profile_stability_summary.json"
    )
    metadata = build_run_summary(
        args,
        completion,
        metrics,
        variability,
        similarity,
        order_frequency,
        b4_variability,
        b4_similarity,
        b4_order_signatures,
        paper_metrics,
        paper_order_signatures,
        reference_comparison_summary,
        full_roots,
        [*output_files, metadata_path],
    )
    atomic_json(metadata_path, metadata)
    output_files.append(metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))

    if args.require_complete and metadata["status"] != "complete":
        raise RuntimeError(
            f"Only {metadata['design']['complete_fit_count']}/"
            f"{metadata['design']['requested_fit_count']} requested fits are complete"
        )


if __name__ == "__main__":
    main()
