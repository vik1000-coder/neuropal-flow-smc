#!/usr/bin/env python3
"""Summarize a deterministic lag-profile leave-out campaign.

Jackknife calculations are centered on the newly trained deterministic full-data
fit. The released paper matrices are evaluated only as a separate
reproducibility comparison; they are never combined with the new delete-one fits
in a jackknife calculation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from experiments.common import (
    DEFAULT_CONNECTOME,
    DEFAULT_MONOAMINE,
    DEFAULT_RANDI,
    DEFAULT_SBTG,
    LAGS,
    best_f1,
    load_dataset,
    load_monoamine_networks,
    load_reference_networks,
    load_sbtg,
    reorder_square,
)
from experiments.snapshot import PREPARED_DATA, portable_path


MONOAMINES = ("dopamine", "serotonin", "tyramine", "octopamine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=PREPARED_DATA)
    parser.add_argument("--paper-result", type=Path, default=DEFAULT_SBTG)
    parser.add_argument("--connectome-dir", type=Path, default=DEFAULT_CONNECTOME)
    parser.add_argument("--randi-file", type=Path, default=DEFAULT_RANDI)
    parser.add_argument("--monoamine-file", type=Path, default=DEFAULT_MONOAMINE)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_one_matrix(
    path: Path,
    expected_lag: int,
    expected_omit_index: int,
    dataset_names: Sequence[str],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as result:
        required = {"mu_hat", "neuron_names", "lag", "omit_index"}
        missing = required.difference(result.files)
        if missing:
            raise RuntimeError(f"{path} lacks required fields: {sorted(missing)}")
        if int(result["lag"]) != expected_lag:
            raise RuntimeError(
                f"{path} reports lag {int(result['lag'])}, expected {expected_lag}"
            )
        if int(result["omit_index"]) != expected_omit_index:
            raise RuntimeError(
                f"{path} reports omission {int(result['omit_index'])}, "
                f"expected {expected_omit_index}"
            )
        names = [str(value) for value in result["neuron_names"]]
        if names != list(dataset_names):
            raise RuntimeError(
                f"{path} does not use the production-snapshot neuron ordering"
            )
        matrix = np.asarray(result["mu_hat"], dtype=float)
    expected_shape = (len(dataset_names), len(dataset_names))
    if matrix.shape != expected_shape:
        raise RuntimeError(f"{path} has shape {matrix.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"{path} contains non-finite coupling estimates")
    return matrix


def load_fit(
    root: Path,
    label: str,
    omit_index: int,
    dataset_names: Sequence[str],
) -> tuple[Optional[dict[int, np.ndarray]], list[int]]:
    matrices = {}
    missing_lags = []
    for lag in LAGS:
        path = root / label / f"lag_{lag:02d}" / "result.npz"
        if not path.exists():
            missing_lags.append(lag)
            continue
        matrices[lag] = load_one_matrix(
            path, lag, omit_index, dataset_names,
        )
    if missing_lags:
        return None, missing_lags
    return matrices, []


def build_networks(
    names: Sequence[str],
    connectome_dir: Path,
    randi_file: Path,
    monoamine_file: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    networks = dict(load_reference_networks(names, connectome_dir, randi_file))
    off_diagonal = ~np.eye(len(names), dtype=bool)
    for name, labels in load_monoamine_networks(names, monoamine_file).items():
        networks[name] = (labels, off_diagonal)
    return networks


def f1_curve_rows(
    matrices: dict[int, np.ndarray],
    networks: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_label: str,
    omit_index: int,
) -> list[dict]:
    rows = []
    for network, (labels, evaluation_mask) in networks.items():
        mask = evaluation_mask & ~np.eye(len(labels), dtype=bool)
        for lag in LAGS:
            f1, threshold = best_f1(
                labels[mask], np.abs(matrices[lag])[mask],
            )
            rows.append(
                {
                    "fit_label": fit_label,
                    "omit_index": omit_index,
                    "network": network,
                    "lag": lag,
                    "time_s": lag / 4.0,
                    "f1": f1,
                    "best_threshold": threshold,
                }
            )
    return rows


def discrete_peak_rows(curves: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "fit_label",
        "omit_index",
        "network",
        "grid_peak_lag",
        "grid_peak_lag_s",
        "grid_peak_f1",
    )
    if curves.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (fit_label, omit_index, network), group in curves.groupby(
        ["fit_label", "omit_index", "network"], sort=False,
    ):
        group = group.sort_values("lag")
        lags = group["lag"].to_numpy(dtype=int)
        values = group["f1"].to_numpy(dtype=float)
        best = int(np.nanargmax(values))
        rows.append(
            {
                "fit_label": fit_label,
                "omit_index": int(omit_index),
                "network": network,
                "grid_peak_lag": int(lags[best]),
                "grid_peak_lag_s": float(lags[best] / 4.0),
                "grid_peak_f1": float(values[best]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def jackknife_description(
    values: np.ndarray,
    full_value: float,
    expected_n: int,
) -> dict:
    """Return complete-jackknife statistics and safe provisional summaries.

    The jackknife estimate and SE are deliberately withheld until all expected
    deletion replicates are present.  Partial campaigns still report their
    observed mean, standard deviation, and range for monitoring.
    """
    values = np.asarray(values, dtype=float)
    n_complete = len(values)
    complete = n_complete == expected_n
    mean = float(np.mean(values)) if n_complete else float("nan")
    sample_sd = float(np.std(values, ddof=1)) if n_complete > 1 else float("nan")
    minimum = float(np.min(values)) if n_complete else float("nan")
    maximum = float(np.max(values)) if n_complete else float("nan")
    estimate = float("nan")
    standard_error = float("nan")
    if complete:
        standard_error = float(
            np.sqrt(
                (expected_n - 1) / expected_n
                * np.sum((values - mean) ** 2)
            )
        )
        pseudo_values = expected_n * full_value - (expected_n - 1) * values
        estimate = float(np.mean(pseudo_values))
    return {
        "full_refit_value": float(full_value),
        "delete_one_mean": mean,
        "delete_one_sd": sample_sd,
        "delete_one_min": minimum,
        "delete_one_max": maximum,
        "jackknife_estimate": estimate,
        "jackknife_se": standard_error,
        "n_complete": n_complete,
        "n_expected": expected_n,
        "jackknife_complete": complete,
    }


def pointwise_summary(
    deletion_curves: pd.DataFrame,
    full_curves: pd.DataFrame,
    expected_n: int,
) -> pd.DataFrame:
    columns = (
        "network",
        "lag",
        "time_s",
        "full_refit_value",
        "delete_one_mean",
        "delete_one_sd",
        "delete_one_min",
        "delete_one_max",
        "jackknife_estimate",
        "jackknife_se",
        "n_complete",
        "n_expected",
        "jackknife_complete",
    )
    if deletion_curves.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (network, lag), group in deletion_curves.groupby(["network", "lag"]):
        full_value = float(
            full_curves.loc[
                (full_curves["network"] == network)
                & (full_curves["lag"] == lag),
                "f1",
            ].iloc[0]
        )
        rows.append(
            {
                "network": network,
                "lag": int(lag),
                "time_s": int(lag) / 4.0,
                **jackknife_description(
                    group["f1"].to_numpy(), full_value, expected_n,
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def peak_f1_summary(
    deletion_peaks: pd.DataFrame,
    full_peaks: pd.DataFrame,
    expected_n: int,
) -> pd.DataFrame:
    columns = (
        "network",
        "full_refit_value",
        "delete_one_mean",
        "delete_one_sd",
        "delete_one_min",
        "delete_one_max",
        "jackknife_estimate",
        "jackknife_se",
        "n_complete",
        "n_expected",
        "jackknife_complete",
    )
    if deletion_peaks.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for network, group in deletion_peaks.groupby("network"):
        full_value = float(
            full_peaks.loc[
                full_peaks["network"] == network, "grid_peak_f1"
            ].iloc[0]
        )
        rows.append(
            {
                "network": network,
                **jackknife_description(
                    group["grid_peak_f1"].to_numpy(), full_value, expected_n,
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def peak_selection_summary(
    deletion_peaks: pd.DataFrame,
    full_peaks: pd.DataFrame,
    expected_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frequency_columns = (
        "network",
        "grid_peak_lag",
        "grid_peak_lag_s",
        "count",
        "frequency_among_completed",
        "n_complete",
        "n_expected",
        "campaign_complete",
    )
    range_columns = (
        "network",
        "full_refit_grid_peak_lag",
        "full_refit_grid_peak_lag_s",
        "full_peak_selected_count",
        "full_peak_selected_frequency",
        "modal_grid_peak_lags_s",
        "modal_frequency",
        "delete_one_peak_lag_min_s",
        "delete_one_peak_lag_max_s",
        "n_distinct_selected_lags",
        "n_complete",
        "n_expected",
        "campaign_complete",
    )
    if deletion_peaks.empty:
        return (
            pd.DataFrame(columns=frequency_columns),
            pd.DataFrame(columns=range_columns),
        )

    frequency = (
        deletion_peaks.groupby(["network", "grid_peak_lag", "grid_peak_lag_s"])
        .size()
        .rename("count")
        .reset_index()
    )
    n_by_network = deletion_peaks.groupby("network").size()
    frequency["n_complete"] = frequency["network"].map(n_by_network).astype(int)
    frequency["frequency_among_completed"] = (
        frequency["count"] / frequency["n_complete"]
    )
    frequency["n_expected"] = expected_n
    frequency["campaign_complete"] = frequency["n_complete"] == expected_n
    frequency = frequency.loc[:, list(frequency_columns)]

    ranges = []
    for network, group in deletion_peaks.groupby("network"):
        full = full_peaks.loc[full_peaks["network"] == network].iloc[0]
        counts = group["grid_peak_lag_s"].value_counts().sort_index()
        modal_count = int(counts.max())
        modes = counts.loc[counts == modal_count].index.to_numpy(dtype=float)
        n_complete = len(group)
        full_count = int(
            np.sum(group["grid_peak_lag"].to_numpy() == int(full["grid_peak_lag"]))
        )
        ranges.append(
            {
                "network": network,
                "full_refit_grid_peak_lag": int(full["grid_peak_lag"]),
                "full_refit_grid_peak_lag_s": float(full["grid_peak_lag_s"]),
                "full_peak_selected_count": full_count,
                "full_peak_selected_frequency": full_count / n_complete,
                "modal_grid_peak_lags_s": ",".join(f"{value:g}" for value in modes),
                "modal_frequency": modal_count / n_complete,
                "delete_one_peak_lag_min_s": float(group["grid_peak_lag_s"].min()),
                "delete_one_peak_lag_max_s": float(group["grid_peak_lag_s"].max()),
                "n_distinct_selected_lags": int(group["grid_peak_lag"].nunique()),
                "n_complete": n_complete,
                "n_expected": expected_n,
                "campaign_complete": n_complete == expected_n,
            }
        )
    return frequency, pd.DataFrame(ranges, columns=range_columns)


def load_paper_matrices(
    path: Path, dataset_names: Sequence[str],
) -> dict[int, np.ndarray]:
    matrices, _, names = load_sbtg(path)
    missing = [lag for lag in LAGS if lag not in matrices]
    if missing:
        raise RuntimeError(f"Released paper result lacks lags: {missing}")
    if set(names) != set(dataset_names):
        raise RuntimeError("Released paper result and production data use different neurons")
    if names != list(dataset_names):
        matrices = {
            lag: reorder_square(matrix, names, dataset_names)[0]
            for lag, matrix in matrices.items()
        }
    return matrices


def paper_comparisons(
    full_curves: pd.DataFrame,
    full_peaks: pd.DataFrame,
    paper_curves: pd.DataFrame,
    paper_peaks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_columns = ["network", "lag", "time_s", "f1"]
    curves = full_curves.loc[:, curve_columns].merge(
        paper_curves.loc[:, curve_columns],
        on=["network", "lag", "time_s"],
        suffixes=("_full_refit", "_paper"),
        validate="one_to_one",
    )
    curves["f1_difference_full_refit_minus_paper"] = (
        curves["f1_full_refit"] - curves["f1_paper"]
    )

    peak_columns = [
        "network", "grid_peak_lag", "grid_peak_lag_s", "grid_peak_f1",
    ]
    peaks = full_peaks.loc[:, peak_columns].merge(
        paper_peaks.loc[:, peak_columns],
        on="network",
        suffixes=("_full_refit", "_paper"),
        validate="one_to_one",
    )
    peaks["grid_peak_matches_paper"] = (
        peaks["grid_peak_lag_full_refit"] == peaks["grid_peak_lag_paper"]
    )
    peaks["peak_f1_difference_full_refit_minus_paper"] = (
        peaks["grid_peak_f1_full_refit"] - peaks["grid_peak_f1_paper"]
    )
    return curves, peaks


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    traces, dataset_names, row_metadata = load_dataset(args.dataset_dir)
    expected_n = len(traces)
    full_matrices, missing_full_lags = load_fit(
        args.input_root, "full", -1, dataset_names,
    )
    if missing_full_lags:
        metadata = {
            "status": "awaiting_full_refit",
            "input_root": portable_path(args.input_root),
            "dataset_dir": portable_path(args.dataset_dir),
            "missing_full_lags": missing_full_lags,
            "required_lags": list(LAGS),
            "message": (
                "Peak and jackknife summaries are withheld until the deterministic "
                "full refit is complete on the full lag grid."
            ),
        }
        atomic_json(args.out_dir / "lag_profile_summary.json", metadata)
        print(json.dumps(metadata, indent=2))
        if args.require_complete:
            raise RuntimeError(f"Full refit lacks lags: {missing_full_lags}")
        return

    networks = build_networks(
        dataset_names, args.connectome_dir, args.randi_file, args.monoamine_file,
    )
    full_curves = pd.DataFrame(
        f1_curve_rows(full_matrices, networks, "full_refit", -1)
    )
    full_peaks = discrete_peak_rows(full_curves)

    deletion_records = []
    complete_omissions = []
    incomplete_omissions = {}
    for omission in range(expected_n):
        label = f"omit_{omission:03d}"
        matrices, missing_lags = load_fit(
            args.input_root, label, omission, dataset_names,
        )
        if missing_lags:
            incomplete_omissions[str(omission)] = missing_lags
            continue
        complete_omissions.append(omission)
        deletion_records.extend(
            f1_curve_rows(matrices, networks, label, omission)
        )

    deletion_curves = pd.DataFrame(
        deletion_records,
        columns=(
            "fit_label", "omit_index", "network", "lag", "time_s", "f1",
            "best_threshold",
        ),
    )
    deletion_peaks = discrete_peak_rows(deletion_curves)
    pointwise = pointwise_summary(deletion_curves, full_curves, expected_n)
    peak_f1 = peak_f1_summary(deletion_peaks, full_peaks, expected_n)
    peak_frequency, peak_ranges = peak_selection_summary(
        deletion_peaks, full_peaks, expected_n,
    )

    paper_matrices = load_paper_matrices(args.paper_result, dataset_names)
    paper_curves = pd.DataFrame(
        f1_curve_rows(paper_matrices, networks, "paper_reference", -2)
    )
    paper_peaks = discrete_peak_rows(paper_curves)
    comparison_curves, comparison_peaks = paper_comparisons(
        full_curves, full_peaks, paper_curves, paper_peaks,
    )

    write_csv(full_curves, args.out_dir / "lag_profile_full_refit_f1_curves.csv")
    write_csv(full_peaks, args.out_dir / "lag_profile_full_refit_discrete_peaks.csv")
    write_csv(deletion_curves, args.out_dir / "lag_profile_delete_one_f1_curves.csv")
    write_csv(deletion_peaks, args.out_dir / "lag_profile_delete_one_discrete_peaks.csv")
    write_csv(pointwise, args.out_dir / "lag_profile_pointwise_f1_jackknife.csv")
    write_csv(peak_f1, args.out_dir / "lag_profile_peak_f1_jackknife.csv")
    write_csv(peak_frequency, args.out_dir / "lag_profile_grid_peak_selection_frequency.csv")
    write_csv(peak_ranges, args.out_dir / "lag_profile_grid_peak_selection_summary.csv")
    write_csv(paper_curves, args.out_dir / "lag_profile_paper_reference_f1_curves.csv")
    write_csv(paper_peaks, args.out_dir / "lag_profile_paper_reference_discrete_peaks.csv")
    write_csv(
        comparison_curves,
        args.out_dir / "lag_profile_full_refit_vs_paper_f1_curves.csv",
    )
    write_csv(
        comparison_peaks,
        args.out_dir / "lag_profile_full_refit_vs_paper_discrete_peaks.csv",
    )

    monoamine_peak_f1 = peak_f1.loc[
        peak_f1["network"].isin(MONOAMINES)
    ].rename(columns={"jackknife_complete": "campaign_complete"})
    monoamine_ranges = peak_ranges.loc[peak_ranges["network"].isin(MONOAMINES)]
    monoamine_summary = monoamine_ranges.merge(
        monoamine_peak_f1,
        on=["network", "n_complete", "n_expected", "campaign_complete"],
        how="outer",
        validate="one_to_one",
    )
    write_csv(
        monoamine_summary,
        args.out_dir / "lag_profile_independent_monoamine_summary.csv",
    )

    campaign_complete = len(complete_omissions) == expected_n
    metadata = {
        "status": "complete" if campaign_complete else "in_progress",
        "method": (
            "deterministic production-pipeline fixed-hyperparameter "
            "leave-one-prepared-recording-out sensitivity"
        ),
        "dataset_dir": portable_path(args.dataset_dir),
        "production_neuron_order_verified": True,
        "n_rows": expected_n,
        "row_metadata": row_metadata.to_dict(orient="records"),
        "lags": list(LAGS),
        "complete_omissions": complete_omissions,
        "incomplete_omissions": incomplete_omissions,
        "jackknife_center": (
            "new deterministic full-data refit under the same implementation and "
            "model-seed policy as the delete-one fits"
        ),
        "jackknife_available": campaign_complete,
        "paper_source": portable_path(args.paper_result),
        "paper_role": (
            "separate reproducibility comparison only; released values "
            "are not used to center any jackknife calculation"
        ),
        "primary_peak_lag_outputs": (
            "discrete grid-peak selection frequencies, modes, and ranges"
        ),
        "peak_f1_output": (
            "delete-one peak-F1 summaries and descriptive jackknife SE; the SE is "
            "withheld until all deletion replicates are complete"
        ),
        "not_reported_as_primary": (
            "parabolic sub-frame peaks and symmetric confidence intervals for the "
            "non-smooth discrete peak-lag argmax"
        ),
        "scope": (
            "prepared-recording composition conditional on fixed production-selected "
            "hyperparameters and fixed donor imputation"
        ),
        "tie_breaking": "the earliest sampled lag is selected when F1 values tie exactly",
    }
    atomic_json(args.out_dir / "lag_profile_summary.json", metadata)
    print(json.dumps(metadata, indent=2))

    if args.require_complete and not campaign_complete:
        raise RuntimeError(
            f"Only {len(complete_omissions)}/{expected_n} omissions are complete"
        )


if __name__ == "__main__":
    main()
