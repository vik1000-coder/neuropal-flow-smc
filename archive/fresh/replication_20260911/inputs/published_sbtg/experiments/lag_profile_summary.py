#!/usr/bin/env python3
"""Summarize the production-pipeline lag-profile recording-row bootstrap campaign.

The campaign design had five independent recording-row resamples, two matched
model seeds per resample, and three full-data seed controls. The campaign may be
summarized at the prespecified stopping point after resamples 0--3. The two
fits of a given resample are algorithmic repeats, so summaries first average
each F1 curve over its two seeds and use B=4 or B=5, never eight or ten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from experiments.common import (
    DEFAULT_CONNECTOME,
    DEFAULT_MONOAMINE,
    DEFAULT_RANDI,
    DEFAULT_SBTG,
    LAGS,
    load_dataset,
)
from experiments.snapshot import PREPARED_DATA
from experiments.snapshot import (
    EXPECTED_MULTILAG_SHA256,
    derive_seed,
    portable_path,
    sha256,
)
from experiments.lag_profile_fit import (
    DEFAULT_DATA_SEED,
    bootstrap_indices,
    fit_label,
)
from experiments.lag_profile_metrics import (
    atomic_json,
    build_networks,
    discrete_peak_rows,
    f1_curve_rows,
    load_paper_matrices,
    write_csv,
)


MONOAMINES = ("dopamine", "serotonin", "tyramine", "octopamine")
LAG2PLUS = tuple(lag for lag in LAGS if lag != 1)
ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_FULL_CONTROL_ROOT = (
    ANALYSIS_DIR.parent / "results/derived/lag_profile_bootstrap"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=PREPARED_DATA)
    parser.add_argument("--paper-result", type=Path, default=DEFAULT_SBTG)
    parser.add_argument("--connectome-dir", type=Path, default=DEFAULT_CONNECTOME)
    parser.add_argument("--randi-file", type=Path, default=DEFAULT_RANDI)
    parser.add_argument("--monoamine-file", type=Path, default=DEFAULT_MONOAMINE)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument(
        "--full-control-seeds", type=int, nargs="+", default=[42, 43, 44],
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, nargs="+", default=[0, 1, 2, 3],
    )
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument(
        "--fit-device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Require every fitted artifact to report this device.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
        help="Require every fitted artifact to report this effective batch size.",
    )
    parser.add_argument(
        "--full-control-root",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help=(
            "External full-control directory containing lag_XX/result.npz and "
            "manifest.json. May be repeated; defaults use the full controls "
            "under results/derived/lag_profile_bootstrap."
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def default_full_control_roots(
    input_root: Path = DEFAULT_FULL_CONTROL_ROOT,
) -> dict[int, Path]:
    input_root = Path(input_root)
    return {
        seed: input_root / fit_label("full", -1, seed)
        for seed in (42, 43, 44)
    }


def parse_full_control_roots(
    values: Sequence[str],
    input_root: Path = DEFAULT_FULL_CONTROL_ROOT,
) -> dict[int, Path]:
    roots = default_full_control_roots(input_root)
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --full-control-root {value!r}; expected SEED=PATH"
            )
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if not path_text:
            raise ValueError(f"Full-control root for seed {seed} is empty")
        roots[seed] = Path(path_text).expanduser().resolve()
    return roots


def load_one_matrix(
    path: Path,
    expected_lag: int,
    sample_kind: str,
    replicate: int,
    model_seed: int,
    data_seed: int,
    expected_indices: np.ndarray,
    dataset_names: Sequence[str],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as result:
        required = {
            "mu_hat", "neuron_names", "lag", "sample_kind", "replicate",
            "model_seed", "data_seed", "source_row_indices",
            "source_row_multiplicities", "fold_seeds",
        }
        missing = required.difference(result.files)
        if missing:
            raise RuntimeError(f"{path} lacks fields: {sorted(missing)}")
        observed_kind = str(result["sample_kind"].item())
        checks = {
            "lag": (int(result["lag"]), expected_lag),
            "sample_kind": (observed_kind, sample_kind),
            "replicate": (int(result["replicate"]), replicate),
            "model_seed": (int(result["model_seed"]), model_seed),
            "data_seed": (int(result["data_seed"]), data_seed),
        }
        failures = {
            name: values for name, values in checks.items() if values[0] != values[1]
        }
        if failures:
            raise RuntimeError(f"{path} metadata mismatch: {failures}")
        names = [str(value) for value in result["neuron_names"]]
        if names != list(dataset_names):
            raise RuntimeError(f"{path} does not use the production neuron order")
        indices = np.asarray(result["source_row_indices"], dtype=np.int64)
        if not np.array_equal(indices, expected_indices):
            raise RuntimeError(f"{path} does not contain the prescribed row draw")
        multiplicities = np.asarray(
            result["source_row_multiplicities"], dtype=np.int64,
        )
        expected_multiplicities = np.bincount(
            expected_indices, minlength=len(multiplicities),
        )
        if not np.array_equal(multiplicities, expected_multiplicities):
            raise RuntimeError(f"{path} row multiplicities are inconsistent")
        expected_fold_seeds = np.asarray(
            [derive_seed(model_seed, 20260123, expected_lag, fold) for fold in range(5)],
            dtype=np.uint32,
        )
        if not np.array_equal(
            np.asarray(result["fold_seeds"], dtype=np.uint32),
            expected_fold_seeds,
        ):
            raise RuntimeError(f"{path} has unexpected fold seeds")
        matrix = np.asarray(result["mu_hat"], dtype=float)
    expected_shape = (len(dataset_names), len(dataset_names))
    if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"{path} contains a malformed coupling matrix")
    return matrix


def load_complete_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc
    if manifest.get("status") != "complete":
        return None
    return manifest


def validate_common_manifest(
    manifest: dict,
    path: Path,
    lag: int,
    model_seed: int,
    dataset_sha256: str,
    n_rows: int,
    n_neurons: int,
    expected_device: str | None = None,
    expected_effective_batch_size: int | None = None,
) -> None:
    expected = {
        "lag": lag,
        "model_seed": model_seed,
        "dataset_sha256": dataset_sha256,
        "model_source_sha256": EXPECTED_MULTILAG_SHA256,
        "n_neurons": n_neurons,
        "n_folds": 5,
    }
    if expected_device is not None:
        expected["device"] = expected_device
    if expected_effective_batch_size is not None:
        expected["effective_batch_size"] = expected_effective_batch_size
    failures = {
        key: {"observed": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    observed_rows = manifest.get("n_rows_full", manifest.get("n_source_rows"))
    if observed_rows != n_rows:
        failures["n_rows"] = {"observed": observed_rows, "expected": n_rows}
    if manifest.get("device") not in {"cpu", "cuda", "mps"}:
        failures["device"] = {
            "observed": manifest.get("device"),
            "expected": "one of cpu, cuda, or mps",
        }
    batch_size = manifest.get("effective_batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        failures["effective_batch_size"] = {
            "observed": batch_size,
            "expected": "a positive integer",
        }
    expected_fold_seeds = {
        str(fold): derive_seed(model_seed, 20260123, lag, fold)
        for fold in range(5)
    }
    observed_fold_seeds = {
        str(key): int(value)
        for key, value in (manifest.get("fold_seeds") or {}).items()
    }
    if observed_fold_seeds != expected_fold_seeds:
        failures["fold_seeds"] = {
            "observed": observed_fold_seeds,
            "expected": expected_fold_seeds,
        }
    if failures:
        raise RuntimeError(f"Full-control validation failed for {path}: {failures}")


def load_external_full_control(
    root: Path,
    model_seed: int,
    dataset_dir: Path,
    n_rows: int,
    dataset_names: Sequence[str],
    data_seed: int = DEFAULT_DATA_SEED,
    expected_device: str | None = None,
    expected_effective_batch_size: int | None = None,
) -> tuple[dict[int, np.ndarray] | None, list[int]]:
    dataset_sha256 = sha256(dataset_dir / "traces.npz")
    expected_indices = np.arange(n_rows, dtype=np.int64)
    matrices = {}
    missing_lags = []
    for lag in LAGS:
        lag_dir = root / f"lag_{lag:02d}"
        result_path = lag_dir / "result.npz"
        manifest_path = lag_dir / "manifest.json"
        manifest = load_complete_manifest(manifest_path)
        if not result_path.exists() or manifest is None:
            missing_lags.append(lag)
            continue
        validate_common_manifest(
            manifest, manifest_path, lag, model_seed, dataset_sha256,
            n_rows, len(dataset_names),
            expected_device, expected_effective_batch_size,
        )
        label = fit_label("full", -1, model_seed)
        expected_manifest = {
            "fit_label": label,
            "sample_kind": "full",
            "replicate": -1,
            "data_seed": data_seed,
            "n_sampled_rows_with_multiplicity": n_rows,
            "source_row_indices": expected_indices.tolist(),
            "source_row_multiplicities": np.ones(n_rows, dtype=int).tolist(),
        }
        failures = {
            key: {"observed": manifest.get(key), "expected": value}
            for key, value in expected_manifest.items()
            if manifest.get(key) != value
        }
        if failures:
            raise RuntimeError(
                f"Full-control validation failed for {manifest_path}: {failures}"
            )
        matrices[lag] = load_one_matrix(
            result_path,
            lag,
            "full",
            -1,
            model_seed,
            data_seed,
            expected_indices,
            dataset_names,
        )
    if missing_lags:
        return None, missing_lags
    return matrices, []


def load_fit(
    root: Path,
    sample_kind: str,
    replicate: int,
    model_seed: int,
    data_seed: int,
    dataset_dir: Path,
    n_rows: int,
    dataset_names: Sequence[str],
    expected_device: str | None = None,
    expected_effective_batch_size: int | None = None,
) -> tuple[dict[int, np.ndarray] | None, list[int]]:
    label = fit_label(sample_kind, replicate, model_seed)
    dataset_sha256 = sha256(dataset_dir / "traces.npz")
    expected_indices = (
        np.arange(n_rows, dtype=np.int64)
        if sample_kind == "full"
        else bootstrap_indices(n_rows, replicate, data_seed)
    )
    matrices = {}
    missing_lags = []
    for lag in LAGS:
        lag_dir = root / label / f"lag_{lag:02d}"
        path = lag_dir / "result.npz"
        manifest_path = lag_dir / "manifest.json"
        manifest = load_complete_manifest(manifest_path)
        if not path.exists() or manifest is None:
            missing_lags.append(lag)
            continue
        validate_common_manifest(
            manifest, manifest_path, lag, model_seed, dataset_sha256,
            n_rows, len(dataset_names),
            expected_device, expected_effective_batch_size,
        )
        expected_manifest = {
            "fit_label": label,
            "sample_kind": sample_kind,
            "replicate": replicate,
            "data_seed": data_seed,
            "n_sampled_rows_with_multiplicity": n_rows,
            "source_row_indices": expected_indices.tolist(),
            "source_row_multiplicities": np.bincount(
                expected_indices, minlength=n_rows,
            ).tolist(),
        }
        failures = {
            key: {"observed": manifest.get(key), "expected": value}
            for key, value in expected_manifest.items()
            if manifest.get(key) != value
        }
        if failures:
            raise RuntimeError(
                f"Bootstrap validation failed for {manifest_path}: {failures}"
            )
        matrices[lag] = load_one_matrix(
            path, lag, sample_kind, replicate, model_seed, data_seed,
            expected_indices, dataset_names,
        )
    if missing_lags:
        return None, missing_lags
    return matrices, []


def add_fit_records(
    records: list[dict],
    matrices: dict[int, np.ndarray],
    networks: dict,
    sample_kind: str,
    replicate: int,
    model_seed: int,
) -> None:
    label = fit_label(sample_kind, replicate, model_seed)
    for row in f1_curve_rows(matrices, networks, label, replicate):
        row.update(
            {
                "sample_kind": sample_kind,
                "replicate": replicate,
                "model_seed": model_seed,
            }
        )
        records.append(row)


def ensemble_curves(
    curves: pd.DataFrame,
    sample_kind: str,
    seeds: Sequence[int],
    expected_replicates: Sequence[int],
) -> tuple[pd.DataFrame, list[int]]:
    empty = pd.DataFrame(
        columns=(
            "network", "lag", "time_s", "f1", "fit_label", "omit_index",
            "sample_kind", "replicate", "n_model_seeds",
        )
    )
    subset = curves.loc[
        (curves["sample_kind"] == sample_kind)
        & curves["model_seed"].isin(seeds)
    ].copy()
    if sample_kind == "full":
        available_seeds = sorted(subset["model_seed"].unique().astype(int).tolist())
        if available_seeds != sorted(set(seeds)):
            return empty, []
        grouped = (
            subset.groupby(["network", "lag", "time_s"], as_index=False)["f1"]
            .mean()
        )
        grouped["fit_label"] = "full_seed_ensemble"
        grouped["omit_index"] = -1
        grouped["sample_kind"] = "full_seed_ensemble"
        grouped["replicate"] = -1
        grouped["n_model_seeds"] = len(seeds)
        return grouped, [-1]

    complete_replicates = []
    frames = []
    for replicate in expected_replicates:
        group = subset.loc[subset["replicate"] == replicate]
        available_seeds = sorted(group["model_seed"].unique().astype(int).tolist())
        if available_seeds != sorted(set(seeds)):
            continue
        averaged = (
            group.groupby(["network", "lag", "time_s"], as_index=False)["f1"]
            .mean()
        )
        averaged["fit_label"] = f"bootstrap_{replicate:03d}_seed_ensemble"
        averaged["omit_index"] = replicate
        averaged["sample_kind"] = "bootstrap_seed_ensemble"
        averaged["replicate"] = replicate
        averaged["n_model_seeds"] = len(seeds)
        frames.append(averaged)
        complete_replicates.append(replicate)
    if not frames:
        return empty, []
    return pd.concat(frames, ignore_index=True), complete_replicates


def seed_matched_contrasts(
    curves: pd.DataFrame, model_seeds: Sequence[int], replicates: Sequence[int],
) -> pd.DataFrame:
    full = curves.loc[
        (curves["sample_kind"] == "full")
        & curves["model_seed"].isin(model_seeds),
        ["model_seed", "network", "lag", "time_s", "f1"],
    ].rename(columns={"f1": "full_f1"})
    bootstrap = curves.loc[
        (curves["sample_kind"] == "bootstrap")
        & curves["model_seed"].isin(model_seeds)
        & curves["replicate"].isin(replicates),
        ["replicate", "model_seed", "network", "lag", "time_s", "f1"],
    ].rename(columns={"f1": "bootstrap_f1"})
    matched = bootstrap.merge(
        full,
        on=["model_seed", "network", "lag", "time_s"],
        how="inner",
        validate="many_to_one",
    )
    matched["bootstrap_minus_full_f1"] = (
        matched["bootstrap_f1"] - matched["full_f1"]
    )
    return matched


def bootstrap_summary(
    bootstrap_peaks: pd.DataFrame,
    full_peaks: pd.DataFrame,
    expected_b: int,
) -> pd.DataFrame:
    columns = (
        "network", "full_seed_ensemble_peak_lag_s", "full_seed_ensemble_peak_f1",
        "bootstrap_peak_f1_mean", "bootstrap_peak_f1_min",
        "bootstrap_peak_f1_max", "modal_bootstrap_peak_lags_s",
        "modal_bootstrap_peak_frequency", "bootstrap_peak_lag_min_s",
        "bootstrap_peak_lag_max_s", "n_distinct_bootstrap_peak_lags",
        "n_bootstrap_resamples_complete", "n_bootstrap_resamples_expected",
        "campaign_complete",
    )
    if bootstrap_peaks.empty or full_peaks.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for network, group in bootstrap_peaks.groupby("network"):
        full = full_peaks.loc[full_peaks["network"] == network].iloc[0]
        counts = group["grid_peak_lag_s"].value_counts().sort_index()
        modal_count = int(counts.max())
        modes = counts.loc[counts == modal_count].index.to_numpy(dtype=float)
        n_complete = len(group)
        rows.append(
            {
                "network": network,
                "full_seed_ensemble_peak_lag_s": float(full["grid_peak_lag_s"]),
                "full_seed_ensemble_peak_f1": float(full["grid_peak_f1"]),
                "bootstrap_peak_f1_mean": float(group["grid_peak_f1"].mean()),
                "bootstrap_peak_f1_min": float(group["grid_peak_f1"].min()),
                "bootstrap_peak_f1_max": float(group["grid_peak_f1"].max()),
                "modal_bootstrap_peak_lags_s": ",".join(
                    f"{value:g}" for value in modes
                ),
                "modal_bootstrap_peak_frequency": modal_count / n_complete,
                "bootstrap_peak_lag_min_s": float(group["grid_peak_lag_s"].min()),
                "bootstrap_peak_lag_max_s": float(group["grid_peak_lag_s"].max()),
                "n_distinct_bootstrap_peak_lags": int(
                    group["grid_peak_lag"].nunique()
                ),
                "n_bootstrap_resamples_complete": n_complete,
                "n_bootstrap_resamples_expected": expected_b,
                "campaign_complete": n_complete == expected_b,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _ordered_complete_profile(
    group: pd.DataFrame,
    context: str,
    candidate_lags: Sequence[int] = LAGS,
) -> pd.DataFrame:
    """Return one finite F1 value at every prespecified lag, in lag order."""
    if group["lag"].duplicated().any():
        duplicated = sorted(
            group.loc[group["lag"].duplicated(keep=False), "lag"]
            .astype(int).unique().tolist()
        )
        raise RuntimeError(f"{context} has duplicate lags: {duplicated}")
    observed = tuple(sorted(group["lag"].astype(int).tolist()))
    expected = tuple(sorted(int(lag) for lag in candidate_lags))
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
    if "time_s" in ordered.columns:
        observed_time = ordered["time_s"].to_numpy(dtype=float)
        expected_time = ordered["lag"].to_numpy(dtype=float) / 4.0
        if not np.allclose(observed_time, expected_time, rtol=0, atol=1e-12):
            raise RuntimeError(f"{context} has time_s values inconsistent with lag/4")
    return ordered


def _unique_metadata(group: pd.DataFrame, column: str, default):
    if column not in group.columns:
        return default
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise RuntimeError(
            f"Fit {group['fit_label'].iloc[0]!r} has inconsistent {column}: "
            f"{values.tolist()}"
        )
    return values.iloc[0]


def peak_margin_rows(
    curves: pd.DataFrame,
    curve_scope: str,
    candidate_lags: Sequence[int] = LAGS,
) -> pd.DataFrame:
    """Measure how far the selected grid maximum is above the runner-up lag."""
    columns = (
        "curve_scope", "fit_label", "omit_index", "sample_kind", "replicate",
        "model_seed", "network", "top_grid_lag", "top_grid_lag_s", "top_f1",
        "second_grid_lag", "second_grid_lag_s", "second_f1",
        "top_minus_second_f1", "n_lags_tied_for_top_f1", "top_f1_is_tied",
    )
    if curves.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (fit_label, omit_index, network), group in curves.groupby(
        ["fit_label", "omit_index", "network"], sort=False,
    ):
        ordered = _ordered_complete_profile(
            group, f"{curve_scope}/{fit_label}/{network}", candidate_lags,
        )
        lags = ordered["lag"].to_numpy(dtype=int)
        values = ordered["f1"].to_numpy(dtype=float)
        # Sort descending by F1 and then ascending by lag. This exactly matches
        # discrete_peak_rows' earliest-lag tie rule and gives a zero margin when
        # two lags tie for the maximum.
        ranking = np.lexsort((lags, -values))
        top_index, second_index = int(ranking[0]), int(ranking[1])
        top_value = float(values[top_index])
        tied_count = int(np.count_nonzero(np.isclose(values, top_value, rtol=0, atol=1e-12)))
        rows.append(
            {
                "curve_scope": curve_scope,
                "fit_label": fit_label,
                "omit_index": int(omit_index),
                "sample_kind": _unique_metadata(group, "sample_kind", curve_scope),
                "replicate": int(_unique_metadata(group, "replicate", omit_index)),
                "model_seed": _unique_metadata(group, "model_seed", np.nan),
                "network": network,
                "top_grid_lag": int(lags[top_index]),
                "top_grid_lag_s": float(lags[top_index] / 4.0),
                "top_f1": top_value,
                "second_grid_lag": int(lags[second_index]),
                "second_grid_lag_s": float(lags[second_index] / 4.0),
                "second_f1": float(values[second_index]),
                "top_minus_second_f1": float(
                    top_value - values[second_index]
                ),
                "n_lags_tied_for_top_f1": tied_count,
                "top_f1_is_tied": tied_count > 1,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def published_peak_recovery_rows(
    individual_peaks: pd.DataFrame,
    bootstrap_ensemble_peaks: pd.DataFrame,
    paper_peaks: pd.DataFrame,
) -> pd.DataFrame:
    """Compare every bootstrap grid peak with the released reference peak."""
    columns = (
        "recovery_level", "fit_label", "replicate", "model_seed", "network",
        "observed_grid_peak_lag", "observed_grid_peak_lag_s",
        "observed_grid_peak_f1", "paper_grid_peak_lag",
        "paper_grid_peak_lag_s", "paper_grid_peak_f1",
        "recovers_paper_grid_peak", "lag_difference_from_paper",
        "absolute_lag_difference_from_paper", "time_difference_from_paper_s",
        "absolute_time_difference_from_paper_s",
    )
    published = paper_peaks.loc[
        :, ["network", "grid_peak_lag", "grid_peak_lag_s", "grid_peak_f1"]
    ].copy()
    if published["network"].duplicated().any():
        raise RuntimeError("Released reference peaks contain duplicate networks")
    published = published.rename(
        columns={
            "grid_peak_lag": "paper_grid_peak_lag",
            "grid_peak_lag_s": "paper_grid_peak_lag_s",
            "grid_peak_f1": "paper_grid_peak_f1",
        }
    )

    individual = individual_peaks.loc[
        individual_peaks["sample_kind"] == "bootstrap"
    ].copy()
    individual["recovery_level"] = "individual_model_seed"
    ensemble = bootstrap_ensemble_peaks.copy()
    ensemble["recovery_level"] = "two_seed_curve_ensemble"
    ensemble["replicate"] = ensemble["omit_index"].astype(int)
    ensemble["model_seed"] = np.nan
    recovery_parts = [frame for frame in (individual, ensemble) if not frame.empty]
    if not recovery_parts:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(recovery_parts, ignore_index=True, sort=False)
    combined = combined.rename(
        columns={
            "grid_peak_lag": "observed_grid_peak_lag",
            "grid_peak_lag_s": "observed_grid_peak_lag_s",
            "grid_peak_f1": "observed_grid_peak_f1",
        }
    )
    combined = combined.merge(
        published, on="network", how="left", validate="many_to_one",
    )
    if combined["paper_grid_peak_lag"].isna().any():
        missing = sorted(
            combined.loc[combined["paper_grid_peak_lag"].isna(), "network"]
            .unique().tolist()
        )
        raise RuntimeError(f"No released reference peak for networks: {missing}")
    combined["recovers_paper_grid_peak"] = (
        combined["observed_grid_peak_lag"].astype(int)
        == combined["paper_grid_peak_lag"].astype(int)
    )
    combined["lag_difference_from_paper"] = (
        combined["observed_grid_peak_lag"] - combined["paper_grid_peak_lag"]
    ).astype(int)
    combined["absolute_lag_difference_from_paper"] = combined[
        "lag_difference_from_paper"
    ].abs().astype(int)
    combined["time_difference_from_paper_s"] = (
        combined["observed_grid_peak_lag_s"]
        - combined["paper_grid_peak_lag_s"]
    )
    combined["absolute_time_difference_from_paper_s"] = combined[
        "time_difference_from_paper_s"
    ].abs()
    return combined.loc[:, columns].sort_values(
        ["network", "replicate", "recovery_level", "model_seed"],
        na_position="last",
    ).reset_index(drop=True)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = np.asarray(x, dtype=float) - np.mean(x)
    y_centered = np.asarray(y, dtype=float) - np.mean(y)
    denominator = float(
        np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    )
    if denominator == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def paired_seed_agreement_rows(
    curves: pd.DataFrame,
    individual_peaks: pd.DataFrame,
    bootstrap_ensemble_peaks: pd.DataFrame,
    paper_peaks: pd.DataFrame,
    model_seeds: Sequence[int],
    replicates: Sequence[int],
    candidate_lags: Sequence[int] = LAGS,
) -> pd.DataFrame:
    """Compare the two algorithmic repeats within each independent resample."""
    columns = (
        "replicate", "network", "seed_a", "seed_b", "seed_a_grid_peak_lag",
        "seed_a_grid_peak_lag_s", "seed_b_grid_peak_lag",
        "seed_b_grid_peak_lag_s", "seeds_select_same_grid_peak",
        "absolute_seed_peak_lag_difference", "absolute_seed_peak_time_difference_s",
        "paper_grid_peak_lag", "paper_grid_peak_lag_s",
        "seed_a_recovers_paper_grid_peak", "seed_b_recovers_paper_grid_peak",
        "at_least_one_seed_recovers_paper_grid_peak",
        "both_seeds_recover_paper_grid_peak", "ensemble_grid_peak_lag",
        "ensemble_grid_peak_lag_s", "ensemble_recovers_paper_grid_peak",
        "f1_profile_pearson_r", "f1_profile_spearman_rho",
        "f1_profile_max_absolute_difference", "f1_profile_rmse",
        "n_profile_lags", "seed_a_profile_is_constant",
        "seed_b_profile_is_constant",
    )
    seeds = sorted(set(int(seed) for seed in model_seeds))
    if len(seeds) != 2:
        raise ValueError(
            "Paired-seed agreement requires exactly two bootstrap model seeds"
        )
    bootstrap_curves = curves.loc[
        (curves["sample_kind"] == "bootstrap")
        & curves["model_seed"].isin(seeds)
    ].copy()
    peak_lookup = individual_peaks.loc[
        (individual_peaks["sample_kind"] == "bootstrap")
        & individual_peaks["model_seed"].isin(seeds)
    ].set_index(["replicate", "model_seed", "network"])
    if peak_lookup.index.duplicated().any():
        raise RuntimeError("Individual bootstrap peaks are not uniquely keyed")
    ensemble_lookup = bootstrap_ensemble_peaks.set_index(
        ["omit_index", "network"]
    )
    if ensemble_lookup.index.duplicated().any():
        raise RuntimeError("Bootstrap ensemble peaks are not uniquely keyed")
    published_lookup = paper_peaks.set_index("network")
    if published_lookup.index.duplicated().any():
        raise RuntimeError("Released reference peaks are not uniquely keyed")

    rows = []
    for replicate in sorted(set(int(value) for value in replicates)):
        replicate_curves = bootstrap_curves.loc[
            bootstrap_curves["replicate"] == replicate
        ]
        available = sorted(
            replicate_curves["model_seed"].unique().astype(int).tolist()
        )
        if available != seeds:
            continue
        for network, network_curves in replicate_curves.groupby("network"):
            profiles = {}
            for seed in seeds:
                profiles[seed] = _ordered_complete_profile(
                    network_curves.loc[network_curves["model_seed"] == seed],
                    f"bootstrap {replicate}/seed {seed}/{network}",
                    candidate_lags,
                )["f1"].to_numpy(dtype=float)
            peak_a = peak_lookup.loc[(replicate, seeds[0], network)]
            peak_b = peak_lookup.loc[(replicate, seeds[1], network)]
            ensemble_peak = ensemble_lookup.loc[(replicate, network)]
            published_peak = published_lookup.loc[network]
            lag_a = int(peak_a["grid_peak_lag"])
            lag_b = int(peak_b["grid_peak_lag"])
            ensemble_lag = int(ensemble_peak["grid_peak_lag"])
            published_lag = int(published_peak["grid_peak_lag"])
            differences = profiles[seeds[0]] - profiles[seeds[1]]
            ranks_a = pd.Series(profiles[seeds[0]]).rank(method="average").to_numpy()
            ranks_b = pd.Series(profiles[seeds[1]]).rank(method="average").to_numpy()
            seed_a_recovers = lag_a == published_lag
            seed_b_recovers = lag_b == published_lag
            rows.append(
                {
                    "replicate": replicate,
                    "network": network,
                    "seed_a": seeds[0],
                    "seed_b": seeds[1],
                    "seed_a_grid_peak_lag": lag_a,
                    "seed_a_grid_peak_lag_s": lag_a / 4.0,
                    "seed_b_grid_peak_lag": lag_b,
                    "seed_b_grid_peak_lag_s": lag_b / 4.0,
                    "seeds_select_same_grid_peak": lag_a == lag_b,
                    "absolute_seed_peak_lag_difference": abs(lag_a - lag_b),
                    "absolute_seed_peak_time_difference_s": abs(lag_a - lag_b) / 4.0,
                    "paper_grid_peak_lag": published_lag,
                    "paper_grid_peak_lag_s": published_lag / 4.0,
                    "seed_a_recovers_paper_grid_peak": seed_a_recovers,
                    "seed_b_recovers_paper_grid_peak": seed_b_recovers,
                    "at_least_one_seed_recovers_paper_grid_peak": (
                        seed_a_recovers or seed_b_recovers
                    ),
                    "both_seeds_recover_paper_grid_peak": (
                        seed_a_recovers and seed_b_recovers
                    ),
                    "ensemble_grid_peak_lag": ensemble_lag,
                    "ensemble_grid_peak_lag_s": ensemble_lag / 4.0,
                    "ensemble_recovers_paper_grid_peak": (
                        ensemble_lag == published_lag
                    ),
                    "f1_profile_pearson_r": _correlation(
                        profiles[seeds[0]], profiles[seeds[1]],
                    ),
                    "f1_profile_spearman_rho": _correlation(ranks_a, ranks_b),
                    "f1_profile_max_absolute_difference": float(
                        np.max(np.abs(differences))
                    ),
                    "f1_profile_rmse": float(np.sqrt(np.mean(differences ** 2))),
                    "n_profile_lags": len(candidate_lags),
                    "seed_a_profile_is_constant": bool(
                        np.ptp(profiles[seeds[0]]) == 0
                    ),
                    "seed_b_profile_is_constant": bool(
                        np.ptp(profiles[seeds[1]]) == 0
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["network", "replicate"]
    ).reset_index(drop=True)


def published_peak_recovery_count_rows(
    paired_agreement: pd.DataFrame,
    paper_peaks: pd.DataFrame,
    model_seeds: Sequence[int],
    expected_b: int,
) -> pd.DataFrame:
    """Return integer recovery counts; never reinterpret two seeds as B=10."""
    columns = (
        "network", "paper_grid_peak_lag", "paper_grid_peak_lag_s",
        "n_complete_paired_resamples", "n_expected_resamples",
        "seed_ensemble_recovery_count",
        "at_least_one_seed_recovery_count",
        "both_seeds_recovery_count", "same_seed_peak_count",
        "seed_a", "seed_a_recovery_count", "seed_b",
        "seed_b_recovery_count", "campaign_complete",
    )
    seeds = sorted(set(int(seed) for seed in model_seeds))
    if len(seeds) != 2:
        raise ValueError("Recovery counts require exactly two bootstrap model seeds")
    rows = []
    for published in paper_peaks.itertuples():
        group = paired_agreement.loc[
            paired_agreement["network"] == published.network
        ]
        n_complete = len(group)
        if n_complete > expected_b:
            raise RuntimeError(
                f"{published.network} has {n_complete} paired resamples, expected at most {expected_b}"
            )
        rows.append(
            {
                "network": published.network,
                "paper_grid_peak_lag": int(published.grid_peak_lag),
                "paper_grid_peak_lag_s": float(published.grid_peak_lag_s),
                "n_complete_paired_resamples": n_complete,
                "n_expected_resamples": expected_b,
                "seed_ensemble_recovery_count": int(
                    group["ensemble_recovers_paper_grid_peak"].sum()
                ),
                "at_least_one_seed_recovery_count": int(
                    group["at_least_one_seed_recovers_paper_grid_peak"].sum()
                ),
                "both_seeds_recovery_count": int(
                    group["both_seeds_recover_paper_grid_peak"].sum()
                ),
                "same_seed_peak_count": int(
                    group["seeds_select_same_grid_peak"].sum()
                ),
                "seed_a": seeds[0],
                "seed_a_recovery_count": int(
                    group["seed_a_recovers_paper_grid_peak"].sum()
                ),
                "seed_b": seeds[1],
                "seed_b_recovery_count": int(
                    group["seed_b_recovers_paper_grid_peak"].sum()
                ),
                "campaign_complete": n_complete == expected_b,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def label_lag2plus_comparison(
    frame: pd.DataFrame,
    primary_paper_peaks: pd.DataFrame,
    lag2plus_paper_peaks: pd.DataFrame,
) -> pd.DataFrame:
    """Label the directly comparable common Approach-C production grid."""
    renamed = frame.rename(
        columns={
            column: column
            .replace("paper_grid_peak", "paper_lag2plus_grid_peak")
            .replace("from_paper", "from_paper_lag2plus")
            for column in frame.columns
        }
    ).copy()
    primary = primary_paper_peaks.loc[
        :, ["network", "grid_peak_lag", "grid_peak_lag_s", "grid_peak_f1"]
    ].rename(
        columns={
            "grid_peak_lag": "primary_paper_grid_peak_lag",
            "grid_peak_lag_s": "primary_paper_grid_peak_lag_s",
            "grid_peak_f1": "primary_paper_grid_peak_f1",
        }
    )
    restricted = lag2plus_paper_peaks.loc[
        :, ["network", "grid_peak_lag", "grid_peak_lag_s", "grid_peak_f1"]
    ].rename(
        columns={
            "grid_peak_lag": "paper_lag2plus_grid_peak_lag",
            "grid_peak_lag_s": "paper_lag2plus_grid_peak_lag_s",
            "grid_peak_f1": "paper_lag2plus_grid_peak_f1",
        }
    )
    for name, reference in (("primary", primary), ("lag2plus", restricted)):
        if reference["network"].duplicated().any():
            raise RuntimeError(f"{name} released reference has duplicate networks")
    if "network" not in renamed.columns:
        raise RuntimeError("Lag2+ diagnostic output lacks a network column")
    renamed = renamed.merge(
        primary, on="network", how="left", validate="many_to_one",
    )
    restricted_columns = [
        "paper_lag2plus_grid_peak_lag",
        "paper_lag2plus_grid_peak_lag_s",
        "paper_lag2plus_grid_peak_f1",
    ]
    missing_restricted_columns = [
        column for column in restricted_columns if column not in renamed.columns
    ]
    if missing_restricted_columns:
        renamed = renamed.merge(
            restricted.loc[:, ["network", *missing_restricted_columns]],
            on="network", how="left", validate="many_to_one",
        )
    missing_reference = renamed[
        "primary_paper_grid_peak_lag"
    ].isna() | renamed["paper_lag2plus_grid_peak_lag"].isna()
    if missing_reference.any():
        missing = sorted(renamed.loc[missing_reference, "network"].unique().tolist())
        raise RuntimeError(f"Lag-2+ diagnostics lack released references for: {missing}")
    renamed["lag2plus_reference_differs_from_primary_paper_peak"] = (
        renamed["paper_lag2plus_grid_peak_lag"].astype(int)
        != renamed["primary_paper_grid_peak_lag"].astype(int)
    )
    renamed.insert(0, "candidate_lags", ",".join(str(lag) for lag in LAG2PLUS))
    renamed.insert(
        0,
        "comparison_scope",
        "primary_valid_production_comparison_common_approach_c_lag2plus",
    )
    return renamed


def label_all_eight_lag_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark prespecified eight-lag results as non-reproduction diagnostics."""
    labeled = frame.copy()
    labeled.insert(0, "candidate_lags", ",".join(str(lag) for lag in LAGS))
    labeled.insert(
        0,
        "analysis_scope",
        "secondary_all_eight_lag_approach_c_sensitivity",
    )
    labeled.insert(2, "directly_paper_comparable", False)
    return labeled


def peak_tuple(peaks: pd.DataFrame) -> dict[str, int]:
    subset = peaks.loc[peaks["network"].isin(MONOAMINES)]
    return {
        str(row.network): int(row.grid_peak_lag)
        for row in subset.itertuples()
    }


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_seeds = sorted(set(args.model_seeds))
    full_control_seeds = sorted(set(args.full_control_seeds))
    replicates = sorted(set(args.bootstrap_replicates))
    prespecified_replicates = [0, 1, 2, 3, 4]
    released_replicates = [0, 1, 2, 3]
    if replicates not in (prespecified_replicates, released_replicates):
        raise ValueError(
            "Bootstrap replicates must be either the original prespecified set "
            "0,1,2,3,4 or the prespecified truncated set 0,1,2,3"
        )
    campaign_stopped_at_b4 = replicates == released_replicates
    effective_b = len(replicates)
    if len(model_seeds) != 2:
        raise ValueError(
            "This prespecified campaign has exactly two model seeds per resample"
        )
    full_control_roots = parse_full_control_roots(
        args.full_control_root, args.input_root
    )
    if not set(model_seeds).issubset(full_control_seeds):
        raise ValueError("Every bootstrap model seed needs a matched full-data control")
    roots_missing = set(full_control_seeds).difference(full_control_roots)
    if roots_missing:
        raise ValueError(f"No external full-control root for seeds: {sorted(roots_missing)}")

    traces, dataset_names, _ = load_dataset(args.dataset_dir)
    n_rows = len(traces)
    networks = build_networks(
        dataset_names, args.connectome_dir, args.randi_file, args.monoamine_file,
    )
    records = []
    missing_fits = {}

    for seed in full_control_seeds:
        matrices, missing_lags = load_external_full_control(
            full_control_roots[seed], seed, args.dataset_dir, n_rows,
            dataset_names, args.data_seed, args.fit_device,
            args.effective_batch_size,
        )
        label = fit_label("full", -1, seed)
        if missing_lags:
            missing_fits[label] = missing_lags
        else:
            add_fit_records(records, matrices, networks, "full", -1, seed)

    for replicate in replicates:
        for seed in model_seeds:
            matrices, missing_lags = load_fit(
                args.input_root, "bootstrap", replicate, seed, args.data_seed,
                args.dataset_dir, n_rows, dataset_names, args.fit_device,
                args.effective_batch_size,
            )
            label = fit_label("bootstrap", replicate, seed)
            if missing_lags:
                missing_fits[label] = missing_lags
            else:
                add_fit_records(
                    records, matrices, networks, "bootstrap", replicate, seed,
                )

    curve_columns = (
        "fit_label", "omit_index", "network", "lag", "time_s", "f1",
        "best_threshold", "sample_kind", "replicate", "model_seed",
    )
    curves = pd.DataFrame(records, columns=curve_columns)
    individual_peaks = discrete_peak_rows(curves)
    if not individual_peaks.empty:
        metadata = curves.loc[
            :, ["fit_label", "sample_kind", "replicate", "model_seed"]
        ].drop_duplicates()
        individual_peaks = individual_peaks.merge(
            metadata, on="fit_label", how="left", validate="many_to_one",
        )
    else:
        individual_peaks["sample_kind"] = pd.Series(dtype=str)
        individual_peaks["replicate"] = pd.Series(dtype=int)
        individual_peaks["model_seed"] = pd.Series(dtype=int)

    matched_full_curves, matched_full_marker = ensemble_curves(
        curves, "full", model_seeds, [-1],
    )
    all_full_curves, all_full_marker = ensemble_curves(
        curves, "full", full_control_seeds, [-1],
    )
    bootstrap_ensemble_curves, complete_replicates = ensemble_curves(
        curves, "bootstrap", model_seeds, replicates,
    )
    matched_full_peaks = discrete_peak_rows(matched_full_curves)
    all_full_peaks = discrete_peak_rows(all_full_curves)
    bootstrap_ensemble_peaks = discrete_peak_rows(bootstrap_ensemble_curves)
    matched_contrasts = seed_matched_contrasts(curves, model_seeds, replicates)
    matched_contrast_means = (
        matched_contrasts.groupby(
            ["replicate", "network", "lag", "time_s"], as_index=False,
        )["bootstrap_minus_full_f1"].mean()
        if not matched_contrasts.empty
        else pd.DataFrame(
            columns=(
                "replicate", "network", "lag", "time_s",
                "bootstrap_minus_full_f1",
            )
        )
    )
    campaign_summary = bootstrap_summary(
        bootstrap_ensemble_peaks, matched_full_peaks, len(replicates),
    )

    paper_matrices = load_paper_matrices(args.paper_result, dataset_names)
    paper_curves = pd.DataFrame(
        f1_curve_rows(paper_matrices, networks, "paper_reference", -2)
    )
    paper_peaks = discrete_peak_rows(paper_curves)
    paper_tuple = peak_tuple(paper_peaks)

    margin_parts = [
            peak_margin_rows(curves, "individual_model_fit"),
            peak_margin_rows(
                matched_full_curves, "matched_full_two_seed_curve_ensemble",
            ),
            peak_margin_rows(
                all_full_curves, "all_full_three_seed_curve_ensemble",
            ),
            peak_margin_rows(
                bootstrap_ensemble_curves,
                "bootstrap_two_seed_curve_ensemble",
            ),
            peak_margin_rows(paper_curves, "paper_reference_curve"),
    ]
    nonempty_margin_parts = [frame for frame in margin_parts if not frame.empty]
    peak_margins = (
        pd.concat(nonempty_margin_parts, ignore_index=True)
        if nonempty_margin_parts
        else margin_parts[0]
    )
    published_peak_recovery = published_peak_recovery_rows(
        individual_peaks, bootstrap_ensemble_peaks, paper_peaks,
    )
    paired_seed_agreement = paired_seed_agreement_rows(
        curves, individual_peaks, bootstrap_ensemble_peaks, paper_peaks,
        model_seeds, replicates,
    )
    published_peak_recovery_counts = published_peak_recovery_count_rows(
        paired_seed_agreement, paper_peaks, model_seeds, expected_b=effective_b,
    )
    peak_margins = label_all_eight_lag_sensitivity(peak_margins)
    published_peak_recovery = label_all_eight_lag_sensitivity(
        published_peak_recovery
    )
    paired_seed_agreement = label_all_eight_lag_sensitivity(
        paired_seed_agreement
    )
    published_peak_recovery_counts = label_all_eight_lag_sensitivity(
        published_peak_recovery_counts
    )
    campaign_summary = label_all_eight_lag_sensitivity(campaign_summary)

    # Primary valid production comparison. The paper production artifact
    # uses a separate 180-epoch regime-gated lag-1 estimate that cannot be exactly
    # retrained, whereas every newly trained production fit uses Approach C at lag
    # 1. The seven lag-2+ candidates share the same estimator lineage. The all-
    # eight-lag outputs above are retained as the prespecified production Approach-C
    # sensitivity diagnostic, but they are not a direct reproduction of the released curve.
    lag2plus_curves = curves.loc[curves["lag"].isin(LAG2PLUS)].copy()
    lag2plus_individual_peaks = discrete_peak_rows(lag2plus_curves)
    if not lag2plus_individual_peaks.empty:
        lag2plus_metadata = lag2plus_curves.loc[
            :, ["fit_label", "sample_kind", "replicate", "model_seed"]
        ].drop_duplicates()
        lag2plus_individual_peaks = lag2plus_individual_peaks.merge(
            lag2plus_metadata,
            on="fit_label",
            how="left",
            validate="many_to_one",
        )
    else:
        lag2plus_individual_peaks["sample_kind"] = pd.Series(dtype=str)
        lag2plus_individual_peaks["replicate"] = pd.Series(dtype=int)
        lag2plus_individual_peaks["model_seed"] = pd.Series(dtype=int)
    lag2plus_matched_full_curves = matched_full_curves.loc[
        matched_full_curves["lag"].isin(LAG2PLUS)
    ].copy()
    lag2plus_all_full_curves = all_full_curves.loc[
        all_full_curves["lag"].isin(LAG2PLUS)
    ].copy()
    lag2plus_bootstrap_ensemble_curves = bootstrap_ensemble_curves.loc[
        bootstrap_ensemble_curves["lag"].isin(LAG2PLUS)
    ].copy()
    lag2plus_matched_full_peaks = discrete_peak_rows(
        lag2plus_matched_full_curves
    )
    lag2plus_bootstrap_ensemble_peaks = discrete_peak_rows(
        lag2plus_bootstrap_ensemble_curves
    )
    lag2plus_paper_curves = paper_curves.loc[
        paper_curves["lag"].isin(LAG2PLUS)
    ].copy()
    lag2plus_paper_peaks = discrete_peak_rows(lag2plus_paper_curves)
    lag2plus_margin_parts = [
        peak_margin_rows(
            lag2plus_curves,
            "primary_common_approach_c_lag2plus_individual_model_fit",
            LAG2PLUS,
        ),
        peak_margin_rows(
            lag2plus_matched_full_curves,
            "primary_common_approach_c_lag2plus_matched_full_two_seed_curve_ensemble",
            LAG2PLUS,
        ),
        peak_margin_rows(
            lag2plus_all_full_curves,
            "primary_common_approach_c_lag2plus_all_full_three_seed_curve_ensemble",
            LAG2PLUS,
        ),
        peak_margin_rows(
            lag2plus_bootstrap_ensemble_curves,
            "primary_common_approach_c_lag2plus_bootstrap_two_seed_curve_ensemble",
            LAG2PLUS,
        ),
        peak_margin_rows(
            lag2plus_paper_curves,
            "primary_common_approach_c_lag2plus_paper_reference_curve",
            LAG2PLUS,
        ),
    ]
    lag2plus_nonempty_margins = [
        frame for frame in lag2plus_margin_parts if not frame.empty
    ]
    lag2plus_peak_margins = (
        pd.concat(lag2plus_nonempty_margins, ignore_index=True)
        if lag2plus_nonempty_margins
        else lag2plus_margin_parts[0]
    )
    lag2plus_published_peak_recovery = published_peak_recovery_rows(
        lag2plus_individual_peaks,
        lag2plus_bootstrap_ensemble_peaks,
        lag2plus_paper_peaks,
    )
    lag2plus_paired_seed_agreement = paired_seed_agreement_rows(
        lag2plus_curves,
        lag2plus_individual_peaks,
        lag2plus_bootstrap_ensemble_peaks,
        lag2plus_paper_peaks,
        model_seeds,
        replicates,
        candidate_lags=LAG2PLUS,
    )
    lag2plus_published_peak_recovery_counts = published_peak_recovery_count_rows(
        lag2plus_paired_seed_agreement,
        lag2plus_paper_peaks,
        model_seeds,
        expected_b=effective_b,
    )
    lag2plus_campaign_summary = bootstrap_summary(
        lag2plus_bootstrap_ensemble_peaks,
        lag2plus_matched_full_peaks,
        expected_b=effective_b,
    )
    lag2plus_peak_margins = label_lag2plus_comparison(
        lag2plus_peak_margins, paper_peaks, lag2plus_paper_peaks,
    )
    lag2plus_published_peak_recovery = label_lag2plus_comparison(
        lag2plus_published_peak_recovery,
        paper_peaks,
        lag2plus_paper_peaks,
    )
    lag2plus_paired_seed_agreement = label_lag2plus_comparison(
        lag2plus_paired_seed_agreement,
        paper_peaks,
        lag2plus_paper_peaks,
    )
    lag2plus_published_peak_recovery_counts = label_lag2plus_comparison(
        lag2plus_published_peak_recovery_counts,
        paper_peaks,
        lag2plus_paper_peaks,
    )
    lag2plus_campaign_summary = label_lag2plus_comparison(
        lag2plus_campaign_summary,
        paper_peaks,
        lag2plus_paper_peaks,
    )
    lag2plus_paper_peak_output = label_lag2plus_comparison(
        lag2plus_paper_peaks,
        paper_peaks,
        lag2plus_paper_peaks,
    )

    full_seed_peaks = individual_peaks.loc[
        individual_peaks.get("sample_kind", pd.Series(dtype=str)) == "full"
    ].copy()
    reproduction_rows = []
    for seed in full_control_seeds:
        group = full_seed_peaks.loc[full_seed_peaks["model_seed"] == seed]
        if group.empty:
            continue
        observed_tuple = peak_tuple(group)
        reproduction_rows.append(
            {
                "model_seed": seed,
                "monoamine_peak_tuple": json.dumps(observed_tuple, sort_keys=True),
                "paper_peak_tuple": json.dumps(paper_tuple, sort_keys=True),
                "all_monoamine_peaks_match_paper": observed_tuple == paper_tuple,
            }
        )
    reproduction = pd.DataFrame(
        reproduction_rows,
        columns=(
            "model_seed", "monoamine_peak_tuple", "paper_peak_tuple",
            "all_monoamine_peaks_match_paper",
        ),
    )
    reproduction.insert(
        0,
        "analysis_scope",
        "secondary_all_eight_lag_approach_c_sensitivity",
    )
    reproduction.insert(1, "directly_paper_comparable", False)

    lag2plus_paper_tuple = peak_tuple(lag2plus_paper_peaks)
    published_monoamine_peaks_unchanged_on_lag2plus = (
        lag2plus_paper_tuple == paper_tuple
    )
    lag2plus_full_seed_peaks = lag2plus_individual_peaks.loc[
        lag2plus_individual_peaks.get(
            "sample_kind", pd.Series(dtype=str)
        ) == "full"
    ].copy()
    lag2plus_reproduction_rows = []
    for seed in full_control_seeds:
        group = lag2plus_full_seed_peaks.loc[
            lag2plus_full_seed_peaks["model_seed"] == seed
        ]
        if group.empty:
            continue
        observed_tuple = peak_tuple(group)
        lag2plus_reproduction_rows.append(
            {
                "comparison_scope": (
                    "primary_valid_production_comparison_common_approach_c_lag2plus"
                ),
                "directly_paper_comparable": True,
                "candidate_lags": ",".join(str(lag) for lag in LAG2PLUS),
                "model_seed": seed,
                "monoamine_peak_tuple": json.dumps(
                    observed_tuple, sort_keys=True,
                ),
                "paper_lag2plus_peak_tuple": json.dumps(
                    lag2plus_paper_tuple, sort_keys=True,
                ),
                "primary_paper_peak_tuple": json.dumps(
                    paper_tuple, sort_keys=True,
                ),
                "published_monoamine_peaks_unchanged_on_lag2plus": (
                    published_monoamine_peaks_unchanged_on_lag2plus
                ),
                "all_monoamine_peaks_match_paper_lag2plus": (
                    observed_tuple == lag2plus_paper_tuple
                ),
            }
        )
    lag2plus_reproduction = pd.DataFrame(
        lag2plus_reproduction_rows,
        columns=(
            "comparison_scope", "directly_paper_comparable",
            "candidate_lags", "model_seed", "monoamine_peak_tuple",
            "paper_lag2plus_peak_tuple", "primary_paper_peak_tuple",
            "published_monoamine_peaks_unchanged_on_lag2plus",
            "all_monoamine_peaks_match_paper_lag2plus",
        ),
    )

    write_csv(curves, args.out_dir / "lag_profile_bootstrap_individual_f1_curves.csv")
    write_csv(
        individual_peaks,
        args.out_dir / "lag_profile_bootstrap_individual_discrete_peaks.csv",
    )
    write_csv(
        matched_full_curves,
        args.out_dir / "lag_profile_matched_full_seed_ensemble_f1_curves.csv",
    )
    write_csv(
        matched_full_peaks,
        args.out_dir / "lag_profile_matched_full_seed_ensemble_discrete_peaks.csv",
    )
    write_csv(
        all_full_curves,
        args.out_dir / "lag_profile_all_full_seed_ensemble_f1_curves.csv",
    )
    write_csv(
        all_full_peaks,
        args.out_dir / "lag_profile_all_full_seed_ensemble_discrete_peaks.csv",
    )
    write_csv(
        bootstrap_ensemble_curves,
        args.out_dir / "lag_profile_bootstrap_seed_ensemble_f1_curves.csv",
    )
    write_csv(
        bootstrap_ensemble_peaks,
        args.out_dir / "lag_profile_bootstrap_seed_ensemble_discrete_peaks.csv",
    )
    write_csv(
        matched_contrasts,
        args.out_dir / "lag_profile_bootstrap_seed_matched_f1_contrasts.csv",
    )
    write_csv(
        matched_contrast_means,
        args.out_dir / "lag_profile_bootstrap_seed_matched_f1_contrast_means.csv",
    )
    write_csv(campaign_summary, args.out_dir / "lag_profile_bootstrap_campaign_summary.csv")
    write_csv(reproduction, args.out_dir / "lag_profile_full_seed_paper_gate.csv")
    write_csv(
        lag2plus_reproduction,
        args.out_dir / "lag_profile_lag2plus_full_seed_paper_gate.csv",
    )
    write_csv(paper_curves, args.out_dir / "lag_profile_paper_reference_f1_curves.csv")
    write_csv(paper_peaks, args.out_dir / "lag_profile_paper_reference_discrete_peaks.csv")
    write_csv(
        peak_margins,
        args.out_dir / "lag_profile_peak_margin_diagnostics.csv",
    )
    write_csv(
        published_peak_recovery,
        args.out_dir / "lag_profile_bootstrap_paper_peak_recovery_by_fit.csv",
    )
    write_csv(
        paired_seed_agreement,
        args.out_dir / "lag_profile_bootstrap_paired_seed_agreement.csv",
    )
    write_csv(
        published_peak_recovery_counts,
        args.out_dir / "lag_profile_bootstrap_paper_peak_recovery_counts.csv",
    )
    write_csv(
        lag2plus_peak_margins,
        args.out_dir / "lag_profile_lag2plus_peak_margin_diagnostics.csv",
    )
    write_csv(
        lag2plus_published_peak_recovery,
        args.out_dir / "lag_profile_lag2plus_bootstrap_paper_peak_recovery_by_fit.csv",
    )
    write_csv(
        lag2plus_paired_seed_agreement,
        args.out_dir / "lag_profile_lag2plus_bootstrap_paired_seed_agreement.csv",
    )
    write_csv(
        lag2plus_published_peak_recovery_counts,
        args.out_dir / "lag_profile_lag2plus_bootstrap_paper_peak_recovery_counts.csv",
    )
    write_csv(
        lag2plus_paper_peak_output,
        args.out_dir / "lag_profile_lag2plus_paper_reference_discrete_peaks.csv",
    )
    write_csv(
        lag2plus_campaign_summary,
        args.out_dir / "lag_profile_lag2plus_bootstrap_campaign_summary.csv",
    )

    requested_fit_count = len(full_control_seeds) + len(replicates) * len(model_seeds)
    complete_fit_count = requested_fit_count - len(missing_fits)
    campaign_complete = complete_fit_count == requested_fit_count
    all_eight_lag_sensitivity_gate = (
        len(reproduction) == len(full_control_seeds)
        and bool(reproduction["all_monoamine_peaks_match_paper"].all())
    )
    common_grid_seed_gate = (
        len(lag2plus_reproduction) == len(full_control_seeds)
        and bool(
            lag2plus_reproduction[
                "all_monoamine_peaks_match_paper_lag2plus"
            ].all()
        )
    )
    report = {
        "status": "complete" if campaign_complete else "in_progress",
        "design": {
            "recording_row_bootstrap_resamples": replicates,
            "original_prespecified_bootstrap_resamples": prespecified_replicates,
            "effective_bootstrap_resample_count": effective_b,
            "campaign_stopped_at_b4": (
                campaign_stopped_at_b4
            ),
            "bootstrap_model_seeds": model_seeds,
            "full_control_seeds": full_control_seeds,
            "data_seed": args.data_seed,
            "lags": list(LAGS),
        },
        "complete_fit_count": complete_fit_count,
        "requested_fit_count": requested_fit_count,
        "complete_seed_ensemble_bootstrap_replicates": complete_replicates,
        "missing_fits": missing_fits,
        "multiplicity_policy": (
            "the length-20 bootstrap index vector is retained exactly; duplicate "
            "recordings remain duplicate training segments"
        ),
        "full_control_policy": (
            "bootstrap seeds have seed-matched full controls; the third full seed "
            "is an additional algorithmic-variability control"
        ),
        "external_full_control_sources": {
            str(seed): portable_path(full_control_roots[seed])
            for seed in full_control_seeds
        },
        "bootstrap_replicate_count": (
            f"B={effective_b} after averaging each resample's F1 curve over its "
            f"two model seeds; the {2 * effective_b} trained bootstrap fits are "
            f"not treated as {2 * effective_b} independent resamples"
        ),
        "stopping_rule": (
            "campaign stopped after completing resamples 0,1,2,3 (B=4)"
            if campaign_stopped_at_b4
            else "original prespecified B=5 campaign"
        ),
        "secondary_all_eight_lag_sensitivity_outputs": {
            "status": (
                "retained prespecified production Approach-C sensitivity; not a "
                "direct comparison with the released curve because lag 1 is estimator-"
                "incompatible"
            ),
            "lag_profile_peak_margin_diagnostics.csv": (
                "top-minus-second-best F1 margin, runner-up lag, and 1e-12-"
                "tolerance tie count for every individual, ensemble, and "
                "paper curve"
            ),
            "lag_profile_bootstrap_paper_peak_recovery_by_fit.csv": (
                "exact released discrete-grid peak recovery for every "
                "individual seed fit and every two-seed curve ensemble"
            ),
            "lag_profile_bootstrap_paired_seed_agreement.csv": (
                "within-resample peak agreement and Pearson, Spearman, maximum-"
                "absolute-difference, and RMSE agreement of the eight-lag F1 curves"
            ),
            "lag_profile_bootstrap_paper_peak_recovery_counts.csv": (
                "integer 0..B recovery counts for the seed ensemble, either seed, "
                "both seeds, and each seed separately; n_expected_resamples gives "
                "the denominator"
            ),
        },
        "peak_recovery_definition": (
            "the valid production-comparison gate uses exact equality of the "
            "selected discrete lag on the common Approach-C grid "
            "2,3,5,8,10,15,20; interpolated sub-frame peaks are not used"
        ),
        "uncertainty_output_policy": (
            f"B={effective_b} is reported through raw integer recovery counts, "
            "ranges, margins, and agreement diagnostics; no confidence interval "
            "is calculated"
        ),
        "primary_common_approach_c_production_comparison": {
            "status": (
                "PRIMARY valid production comparison and reproduction gate"
            ),
            "candidate_lags": list(LAG2PLUS),
            "excluded_lag1_estimator": "180-epoch regime-gated estimator",
            "exact_excluded_lag1_retrain_possible": False,
            "reason": (
                "the released lag-1 matrix came from an incompatible "
                "180-epoch regime-gated estimator whose exact retraining state is "
                "unavailable, while newly trained production fits use Approach C "
                "at lag 1; lags 2+ share the Approach-C production lineage"
            ),
            "reference_definition": (
                "the maximum of the released F1 curve after restricting "
                "both reference and newly fitted curves to lags 2,3,5,8,10,15,20"
            ),
            "outputs": [
                "lag_profile_lag2plus_peak_margin_diagnostics.csv",
                "lag_profile_lag2plus_bootstrap_paper_peak_recovery_by_fit.csv",
                "lag_profile_lag2plus_bootstrap_paired_seed_agreement.csv",
                "lag_profile_lag2plus_bootstrap_paper_peak_recovery_counts.csv",
                "lag_profile_lag2plus_paper_reference_discrete_peaks.csv",
                "lag_profile_lag2plus_full_seed_paper_gate.csv",
                "lag_profile_lag2plus_bootstrap_campaign_summary.csv",
            ],
            "interpretation_limit": (
                f"this is the valid descriptive production comparison, but B="
                f"{effective_b} does not support a calibrated confidence interval"
            ),
            "published_monoamine_peaks_unchanged_after_excluding_lag1": (
                published_monoamine_peaks_unchanged_on_lag2plus
            ),
            "primary_paper_monoamine_peak_tuple": paper_tuple,
            "lag2plus_paper_monoamine_peak_tuple": lag2plus_paper_tuple,
            "full_seed_reproduction_gate_passed": common_grid_seed_gate,
            "full_seed_reproduction_gate_status": (
                "PASS" if common_grid_seed_gate else "FAIL"
            ),
        },
        "paper_peak_reproduction_gate_grid": list(LAG2PLUS),
        "paper_peak_reproduction_gate_passed": common_grid_seed_gate,
        "paper_peak_reproduction_gate_status": (
            "PASS" if common_grid_seed_gate else "FAIL"
        ),
        "published_monoamine_peaks_unchanged_after_excluding_lag1": (
            published_monoamine_peaks_unchanged_on_lag2plus
        ),
        "all_eight_lag_sensitivity_gate_passed": all_eight_lag_sensitivity_gate,
        "all_eight_lag_sensitivity_directly_paper_comparable": False,
        "methodological_gate": (
            "The common-Approach-C lag2+ full-data seeds do not reproduce the "
            "released monoamine peak tuple, so the bootstrap cannot be "
            f"presented as a confidence interval for that curve. Independently, B="
            f"{effective_b} "
            "supports only raw ranges, recovery counts, and agreement diagnostics, "
            "not a calibrated confidence interval."
        ),
        "can_report_as_paper_confidence_interval": False,
        "can_report_as_campaign_sensitivity": campaign_complete,
        "fixed_scope": (
            "fixed production hyperparameters and fixed prepared-data imputation; "
            "resampling unit is a prepared recording row"
        ),
        "paper_source": portable_path(args.paper_result),
    }
    atomic_json(args.out_dir / "lag_profile_bootstrap_summary.json", report)
    print(json.dumps(report, indent=2))
    if args.require_complete and not campaign_complete:
        raise RuntimeError(
            f"Only {complete_fit_count}/{requested_fit_count} requested fits are complete"
        )


if __name__ == "__main__":
    main()
