#!/usr/bin/env python3
"""Summarize the production-compatible phase-duration analysis.

The primary statistic exactly follows the code that generated Figure 3C:
within each source/target cell-type pair and lag, average ``abs(mu_hat)`` over
FDR-significant edges.  A threshold-free analogue averages over every
off-diagonal candidate edge in the same cell-type pair.

The script also uses the reference production phase results as immutable anchors.
Before summarizing new fits, it verifies that the production cell-type computation
recreates the reference ``celltype_by_lag_*.csv`` tables to numerical precision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.snapshot import PHASE_ROOT, repository_relative, sha256


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = HERE.parent / "results/derived/phase_duration"
LAGS = (1, 2, 3, 5)
PHASES = ("baseline", "steady", "on", "off")
PHASE_LABELS = {
    "baseline": "Baseline",
    "steady": "Steady",
    "on": "On",
    "off": "Off",
}
TYPE_ORDER = ("sensory", "interneuron", "motor")
TYPE_SHORT = {"sensory": "S", "interneuron": "I", "motor": "M"}
EXPECTED_WINDOW_COUNTS = {
    "matched": {
        phase: {"1": 900, "2": 840, "3": 780, "5": 660}
        for phase in PHASES
    },
    "full": {
        "baseline": {"1": 14326, "2": 14246, "3": 14166, "5": 14006},
        "steady": {"1": 1380, "2": 1320, "3": 1260, "5": 1140},
        "on": {"1": 900, "2": 840, "3": 780, "5": 660},
        "off": {"1": 900, "2": 840, "3": 780, "5": 660},
    },
}

# This is the exact classifier behavior present when Figure 3C was generated.
# It removes a trailing L/R before lookup even when the complete class name
# itself ends in L/R. A later classifier correction does not recreate the
# released figure, so the figure-compatible mapping is retained here.
FIGURE3C_SENSORY = {
    "ASI", "ASJ", "AWA", "ASG", "AWB", "ASE", "ADF", "AFD", "AWC",
    "ASK", "ASH", "ADL", "BAG", "URX", "ALN", "PLN", "SDQ", "AQR",
    "PQR", "ALM", "AVM", "PVM", "PLM", "FLP", "DVA", "PVD", "ADE",
    "PDE", "PHA", "PHB", "PHC", "CEP", "OLQ", "OLL", "IL1", "IL2",
    "URY", "URB", "URA",
}
FIGURE3C_INTERNEURONS = {
    "AIA", "AIB", "AIY", "AIZ", "AIM", "AIN", "RIA", "RIB", "RIG",
    "RIH", "RIS", "RIF", "AVA", "AVB", "AVD", "AVE", "AVG", "AVH",
    "AVJ", "AVK", "AVF", "AVL", "PVP", "PVQ", "PVT", "PVW", "PVN",
    "DVB", "DVC", "RIM", "RIR", "RIC", "RIP", "RID", "ADA", "ALA",
    "BDU", "HSN", "LUA", "PVC", "PVR", "RMG",
}
FIGURE3C_MOTOR = {
    "RIV", "RMD", "RME", "RMF", "RMH", "SAA", "SAB", "SIA", "SIB",
    "SMB", "SMD",
}
FIGURE3C_TYPE_MAP = {
    **{name: "sensory" for name in FIGURE3C_SENSORY},
    **{name: "interneuron" for name in FIGURE3C_INTERNEURONS},
    **{name: "motor" for name in FIGURE3C_MOTOR},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=PHASE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--expected-matched-replicates", type=int, default=10)
    parser.add_argument("--anchor-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Summarize completed results but do not fail on missing expected jobs.",
    )
    return parser.parse_args()


def production_neuron_type(name: object) -> str:
    value = str(name).strip().upper()
    if len(value) > 1 and value[-1] in {"L", "R"}:
        value = value[:-1]
    return FIGURE3C_TYPE_MAP.get(value, "unknown")


def expected_jobs(
    input_root: Path, matched_replicates: int
) -> list[tuple[str, str, int, Path]]:
    jobs = [
        ("full", phase, 0, input_root / "full" / phase / "replicate_000")
        for phase in PHASES
    ]
    jobs.extend(
        (
            "matched",
            phase,
            replicate,
            input_root / "matched" / phase / f"replicate_{replicate:03d}",
        )
        for phase in ("baseline", "steady")
        for replicate in range(matched_replicates)
    )
    jobs.extend(
        (
            "matched",
            phase,
            0,
            input_root / "matched" / phase / "replicate_000",
        )
        for phase in ("on", "off")
    )
    return jobs


def validate_result_dir(
    path: Path, mode: str, phase_code: str, replicate: int
) -> list[str]:
    problems = []
    result_path = path / "result.npz"
    manifest_path = path / "manifest.json"
    if not result_path.exists():
        problems.append(f"missing result: {result_path}")
    if not manifest_path.exists():
        problems.append(f"missing manifest: {manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "complete":
                problems.append(
                    f"manifest is not complete ({manifest.get('status')}): {manifest_path}"
                )
            expected_metadata = {
                "mode": mode,
                "phase_code": phase_code,
                "replicate": replicate,
                "effective_batch_size": 256,
            }
            for field, expected in expected_metadata.items():
                if manifest.get(field) != expected:
                    problems.append(
                        f"manifest {field}={manifest.get(field)!r}, expected "
                        f"{expected!r}: {manifest_path}"
                    )
            observed_counts = {
                str(key): int(value)
                for key, value in manifest.get("window_counts", {}).items()
            }
            expected_counts = EXPECTED_WINDOW_COUNTS[mode][phase_code]
            if observed_counts != expected_counts:
                problems.append(
                    f"manifest window_counts={observed_counts}, expected "
                    f"{expected_counts}: {manifest_path}"
                )
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"invalid manifest {manifest_path}: {exc}")
    return problems


def load_result(path: Path) -> tuple[dict, list[str], tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as data:
        names = [str(value) for value in data["neuron_names"]]
        available_lags = tuple(int(value) for value in data["lags"])
        missing = [lag for lag in LAGS if lag not in available_lags]
        if missing:
            raise RuntimeError(f"{path} is missing required lags {missing}")
        n = len(names)
        arrays = {}
        for lag in LAGS:
            mu = np.asarray(data[f"mu_hat_lag{lag}"], dtype=float)
            sig = np.asarray(data[f"significant_lag{lag}"], dtype=bool)
            if mu.shape != (n, n) or sig.shape != (n, n):
                raise RuntimeError(
                    f"{path}, lag {lag}: expected {(n, n)}, got {mu.shape}/{sig.shape}"
                )
            arrays[lag] = (mu, sig)
    return arrays, names, available_lags


def summarize_result(
    path: Path,
    dataset_mode: str,
    phase_code: str,
    replicate: int,
    source_kind: str,
) -> tuple[list[dict], list[dict]]:
    arrays, names, _ = load_result(path)
    types = np.asarray([production_neuron_type(name) for name in names])
    n = len(names)
    off_diagonal = ~np.eye(n, dtype=bool)
    profile_rows: list[dict] = []
    edge_rows: list[dict] = []

    for lag in LAGS:
        mu, significant = arrays[lag]
        edge_rows.append(
            {
                "source_kind": source_kind,
                "dataset_mode": dataset_mode,
                "phase_code": phase_code,
                "phase": PHASE_LABELS[phase_code],
                "replicate": replicate,
                "lag": lag,
                "time_s": lag / 4.0,
                "significant_edge_count": int((significant & off_diagonal).sum()),
                "significant_edge_fraction": float(
                    significant[off_diagonal].mean()
                ),
            }
        )
        for source_type in TYPE_ORDER:
            for target_type in TYPE_ORDER:
                candidate = np.outer(types == target_type, types == source_type)
                significant_values = np.abs(mu)[candidate & significant]
                all_values = np.abs(mu)[candidate & off_diagonal]
                profile_rows.append(
                    {
                        "source_kind": source_kind,
                        "dataset_mode": dataset_mode,
                        "phase_code": phase_code,
                        "phase": PHASE_LABELS[phase_code],
                        "replicate": replicate,
                        "source_type": source_type,
                        "target_type": target_type,
                        "pair": (
                            f"{TYPE_SHORT[source_type]}"
                            f"→{TYPE_SHORT[target_type]}"
                        ),
                        "lag": lag,
                        "time_s": lag / 4.0,
                        "n_candidate_edges": int((candidate & off_diagonal).sum()),
                        "n_significant_edges": int((candidate & significant).sum()),
                        # Figure 3C used zero when no edge survived FDR.
                        "mean_abs_mu_significant": (
                            float(significant_values.mean())
                            if significant_values.size
                            else 0.0
                        ),
                        "mean_abs_mu_all_edges": float(all_values.mean()),
                    }
                )
    return profile_rows, edge_rows


def compute_pair_ratios(profile: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_kind", "dataset_mode", "phase_code", "phase", "replicate",
        "source_type", "target_type", "pair",
    ]
    rows = []
    for key, group in profile.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key))
        by_lag = group.set_index("lag")
        for metric, output in (
            ("mean_abs_mu_significant", "significant_long_short_ratio"),
            ("mean_abs_mu_all_edges", "all_edges_long_short_ratio"),
        ):
            short = float(by_lag.loc[[1, 2], metric].mean())
            long = float(by_lag.loc[[3, 5], metric].mean())
            row[f"{output}_short_mean"] = short
            row[f"{output}_long_mean"] = long
            row[output] = float(long / short) if short > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_phase_indices(pair_ratios: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_kind", "dataset_mode", "phase_code", "phase", "replicate"]
    rows = []
    for key, group in pair_ratios.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key))
        for metric in (
            "significant_long_short_ratio",
            "all_edges_long_short_ratio",
        ):
            values = group[metric].dropna().to_numpy(float)
            row[f"{metric}_pair_mean"] = (
                float(values.mean()) if values.size else np.nan
            )
            row[f"{metric}_pair_median"] = (
                float(np.median(values)) if values.size else np.nan
            )
            row[f"{metric}_n_pairs"] = int(values.size)
            row[f"{metric}_fraction_above_one"] = (
                float(np.mean(values > 1.0)) if values.size else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_phase_indices(indices: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_kind", "dataset_mode", "phase_code", "phase"]
    value_columns = [
        column
        for column in indices.columns
        if column.endswith("_pair_mean") or column.endswith("_pair_median")
    ]
    rows = []
    for key, group in indices.groupby(keys, sort=False, dropna=False):
        for quantity in value_columns:
            values = group[quantity].dropna().to_numpy(float)
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "quantity": quantity,
                    "n": int(values.size),
                    "mean": float(values.mean()) if values.size else np.nan,
                    "sd": (
                        float(values.std(ddof=1)) if values.size > 1 else np.nan
                    ),
                    "q025": (
                        float(np.quantile(values, 0.025))
                        if values.size
                        else np.nan
                    ),
                    "q975": (
                        float(np.quantile(values, 0.975))
                        if values.size
                        else np.nan
                    ),
                    "minimum": float(values.min()) if values.size else np.nan,
                    "maximum": float(values.max()) if values.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def contrast_record(event_value: float, reference_values: np.ndarray) -> dict:
    reference_values = reference_values[np.isfinite(reference_values)]
    if not np.isfinite(event_value) or not reference_values.size:
        return {}
    reference_mean = float(reference_values.mean())
    return {
        "event_value": float(event_value),
        "reference_mean": reference_mean,
        "reference_median": float(np.median(reference_values)),
        "event_minus_reference_mean": float(event_value - reference_mean),
        "reference_min": float(reference_values.min()),
        "reference_max": float(reference_values.max()),
        "event_within_reference_range": bool(
            reference_values.min() <= event_value <= reference_values.max()
        ),
        "event_percentile_among_reference_replicates": float(
            np.mean(reference_values <= event_value)
        ),
        "n_reference_replicates": int(reference_values.size),
    }


def compute_matched_phase_contrasts(indices: pd.DataFrame) -> pd.DataFrame:
    matched = indices.loc[
        (indices["source_kind"] == "new")
        & (indices["dataset_mode"] == "matched")
    ]
    quantities = [
        column
        for column in matched.columns
        if column.endswith("_pair_mean") or column.endswith("_pair_median")
    ]
    rows = []
    for event_phase in ("On", "Off"):
        event = matched.loc[matched.phase == event_phase]
        if len(event) != 1:
            continue
        for reference_phase in ("Baseline", "Steady"):
            reference = matched.loc[matched.phase == reference_phase]
            for quantity in quantities:
                values = reference[quantity].to_numpy(float)
                result = contrast_record(float(event.iloc[0][quantity]), values)
                if result:
                    rows.append(
                        {
                            "event_phase": event_phase,
                            "reference_phase": reference_phase,
                            "quantity": quantity,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def compute_matched_pair_contrasts(pair_ratios: pd.DataFrame) -> pd.DataFrame:
    matched = pair_ratios.loc[
        (pair_ratios["source_kind"] == "new")
        & (pair_ratios["dataset_mode"] == "matched")
    ]
    quantities = (
        "significant_long_short_ratio",
        "all_edges_long_short_ratio",
    )
    rows = []
    for event_phase in ("On", "Off"):
        for reference_phase in ("Baseline", "Steady"):
            for source_type in TYPE_ORDER:
                for target_type in TYPE_ORDER:
                    selection = (
                        (matched.source_type == source_type)
                        & (matched.target_type == target_type)
                    )
                    event = matched.loc[selection & (matched.phase == event_phase)]
                    reference = matched.loc[
                        selection & (matched.phase == reference_phase)
                    ]
                    if len(event) != 1:
                        continue
                    for quantity in quantities:
                        result = contrast_record(
                            float(event.iloc[0][quantity]),
                            reference[quantity].to_numpy(float),
                        )
                        if result:
                            rows.append(
                                {
                                    "event_phase": event_phase,
                                    "reference_phase": reference_phase,
                                    "source_type": source_type,
                                    "target_type": target_type,
                                    "pair": (
                                        f"{TYPE_SHORT[source_type]}"
                                        f"→{TYPE_SHORT[target_type]}"
                                    ),
                                    "quantity": quantity,
                                    **result,
                                }
                            )
    return pd.DataFrame(rows)


def compute_matched_edge_contrasts(edges: pd.DataFrame) -> pd.DataFrame:
    matched = edges.loc[
        (edges["source_kind"] == "new")
        & (edges["dataset_mode"] == "matched")
    ]
    rows = []
    for event_phase in ("On", "Off"):
        for reference_phase in ("Baseline", "Steady"):
            for lag in LAGS:
                event = matched.loc[
                    (matched.phase == event_phase) & (matched.lag == lag)
                ]
                reference = matched.loc[
                    (matched.phase == reference_phase) & (matched.lag == lag)
                ]
                if len(event) != 1:
                    continue
                result = contrast_record(
                    float(event.iloc[0].significant_edge_count),
                    reference.significant_edge_count.to_numpy(float),
                )
                if result:
                    rows.append(
                        {
                            "event_phase": event_phase,
                            "reference_phase": reference_phase,
                            "lag": lag,
                            "time_s": lag / 4.0,
                            "quantity": "significant_edge_count",
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def safe_spearman(left: Iterable[float], right: Iterable[float]) -> float:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    if finite.sum() < 3:
        return np.nan
    if np.std(left_array[finite]) == 0 or np.std(right_array[finite]) == 0:
        return np.nan
    return float(spearmanr(left_array[finite], right_array[finite]).statistic)


def compare_full_to_reference(
    profile: pd.DataFrame,
    pair_ratios: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_profile = profile.loc[profile.source_kind == "reference"].copy()
    full_profile = profile.loc[
        (profile.source_kind == "new") & (profile.dataset_mode == "full")
    ].copy()
    profile_keys = ["phase_code", "phase", "source_type", "target_type", "pair", "lag", "time_s"]
    profile_comparison = reference_profile.merge(
        full_profile,
        on=profile_keys,
        suffixes=("_reference", "_full"),
        validate="one_to_one",
    )
    for metric in ("mean_abs_mu_significant", "mean_abs_mu_all_edges"):
        profile_comparison[f"{metric}_absolute_error"] = np.abs(
            profile_comparison[f"{metric}_full"]
            - profile_comparison[f"{metric}_reference"]
        )

    reference_ratios = pair_ratios.loc[pair_ratios.source_kind == "reference"].copy()
    full_ratios = pair_ratios.loc[
        (pair_ratios.source_kind == "new")
        & (pair_ratios.dataset_mode == "full")
    ].copy()
    ratio_keys = ["phase_code", "phase", "source_type", "target_type", "pair"]
    ratio_comparison = reference_ratios.merge(
        full_ratios,
        on=ratio_keys,
        suffixes=("_reference", "_full"),
        validate="one_to_one",
    )
    for metric in (
        "significant_long_short_ratio",
        "all_edges_long_short_ratio",
    ):
        ratio_comparison[f"{metric}_difference"] = (
            ratio_comparison[f"{metric}_full"]
            - ratio_comparison[f"{metric}_reference"]
        )

    reference_edges = edges.loc[edges.source_kind == "reference"].copy()
    full_edges = edges.loc[
        (edges.source_kind == "new") & (edges.dataset_mode == "full")
    ].copy()
    edge_keys = ["phase_code", "phase", "lag", "time_s"]
    edge_comparison = reference_edges.merge(
        full_edges,
        on=edge_keys,
        suffixes=("_reference", "_full"),
        validate="one_to_one",
    )
    edge_comparison["edge_count_difference"] = (
        edge_comparison.significant_edge_count_full
        - edge_comparison.significant_edge_count_reference
    )
    edge_comparison["edge_count_ratio"] = (
        edge_comparison.significant_edge_count_full
        / edge_comparison.significant_edge_count_reference.replace(0, np.nan)
    )
    return profile_comparison, ratio_comparison, edge_comparison


def reproduction_summary(
    profile_comparison: pd.DataFrame,
    ratio_comparison: pd.DataFrame,
    edge_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for phase_code in PHASES:
        phase_profile = profile_comparison.loc[
            profile_comparison.phase_code == phase_code
        ]
        phase_ratios = ratio_comparison.loc[
            ratio_comparison.phase_code == phase_code
        ]
        phase_edges = edge_comparison.loc[
            edge_comparison.phase_code == phase_code
        ]
        row = {
            "phase_code": phase_code,
            "phase": PHASE_LABELS[phase_code],
        }
        for metric in ("mean_abs_mu_significant", "mean_abs_mu_all_edges"):
            reference = phase_profile[f"{metric}_reference"].to_numpy(float)
            full = phase_profile[f"{metric}_full"].to_numpy(float)
            row[f"{metric}_spearman"] = safe_spearman(reference, full)
            row[f"{metric}_mae"] = float(np.mean(np.abs(full - reference)))
        for metric in (
            "significant_long_short_ratio",
            "all_edges_long_short_ratio",
        ):
            reference = phase_ratios[f"{metric}_reference"].to_numpy(float)
            full = phase_ratios[f"{metric}_full"].to_numpy(float)
            row[f"{metric}_reference_pair_mean"] = float(np.nanmean(reference))
            row[f"{metric}_full_pair_mean"] = float(np.nanmean(full))
            row[f"{metric}_direction_matches"] = bool(
                (np.nanmean(reference) > 1.0) == (np.nanmean(full) > 1.0)
            )
        row["mean_edge_count_ratio_full_over_reference"] = float(
            phase_edges.edge_count_ratio.mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def verify_reference_figure_tables(
    reference_profile: pd.DataFrame,
    reference_root: Path,
    tolerance: float,
) -> tuple[pd.DataFrame, float]:
    rows = []
    maximum_error = 0.0
    for phase_code in PHASES:
        csv_path = reference_root / "comparison" / f"celltype_by_lag_{phase_code}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing reference Figure 3C table: {csv_path}")
        expected = pd.read_csv(csv_path, index_col=0)
        expected = expected.loc[:, [f"lag{lag}" for lag in LAGS]]
        observed_long = reference_profile.loc[
            reference_profile.phase_code == phase_code,
            ["pair", "lag", "mean_abs_mu_significant"],
        ]
        observed = observed_long.pivot(
            index="pair", columns="lag", values="mean_abs_mu_significant"
        )
        observed = observed.loc[expected.index, list(LAGS)]
        observed.columns = [f"lag{lag}" for lag in observed.columns]
        difference = observed.to_numpy(float) - expected.to_numpy(float)
        phase_maximum = float(np.max(np.abs(difference)))
        maximum_error = max(maximum_error, phase_maximum)
        rows.append(
            {
                "phase_code": phase_code,
                "phase": PHASE_LABELS[phase_code],
                "reference_table": repository_relative(csv_path),
                "reference_table_sha256": sha256(csv_path),
                "maximum_absolute_error": phase_maximum,
                "passes_tolerance": bool(phase_maximum <= tolerance),
            }
        )
    validation = pd.DataFrame(rows)
    if maximum_error > tolerance:
        raise RuntimeError(
            "Production Figure 3C calculation does not reproduce the reference tables: "
            f"maximum absolute error {maximum_error:.3g} exceeds {tolerance:.3g}"
        )
    return validation, maximum_error


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    reference_root = args.reference_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (input_root / "summary").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = expected_jobs(
        input_root, args.expected_matched_replicates
    )
    problems = [
        problem
        for mode, phase_code, replicate, directory in jobs
        for problem in validate_result_dir(
            directory, mode, phase_code, replicate
        )
    ]

    profile_rows: list[dict] = []
    edge_rows: list[dict] = []

    # Immutable production anchors.
    reference_result_hashes = {}
    for phase_code in PHASES:
        result_path = reference_root / phase_code / "result.npz"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing reference phase result: {result_path}")
        reference_result_hashes[phase_code] = sha256(result_path)
        phase_profiles, phase_edges = summarize_result(
            result_path,
            dataset_mode="reference_full",
            phase_code=phase_code,
            replicate=0,
            source_kind="reference",
        )
        profile_rows.extend(phase_profiles)
        edge_rows.extend(phase_edges)

    reference_profile = pd.DataFrame(profile_rows)
    anchor_validation, anchor_maximum_error = verify_reference_figure_tables(
        reference_profile, reference_root, args.anchor_tolerance
    )

    # Completed new fits.  Expected-directory validation above determines
    # whether missing fits are fatal; this loop permits explicit partial summaries.
    for mode, phase_code, replicate, replicate_dir in jobs:
        result_path = replicate_dir / "result.npz"
        manifest_path = replicate_dir / "manifest.json"
        if not result_path.exists() or not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "complete":
            continue
        phase_profiles, phase_edges = summarize_result(
            result_path,
            dataset_mode=mode,
            phase_code=phase_code,
            replicate=replicate,
            source_kind="new",
        )
        profile_rows.extend(phase_profiles)
        edge_rows.extend(phase_edges)

    profile = pd.DataFrame(profile_rows)
    edges = pd.DataFrame(edge_rows)
    pair_ratios = compute_pair_ratios(profile)
    phase_indices = compute_phase_indices(pair_ratios)
    phase_summary = summarize_phase_indices(phase_indices)

    profile.to_csv(out_dir / "phase_celltype_lag_profiles.csv", index=False)
    edges.to_csv(out_dir / "phase_edge_counts.csv", index=False)
    pair_ratios.to_csv(out_dir / "phase_pair_long_short_ratios.csv", index=False)
    phase_indices.to_csv(out_dir / "phase_ratio_indices.csv", index=False)
    phase_summary.to_csv(out_dir / "phase_ratio_summary.csv", index=False)
    anchor_validation.to_csv(
        out_dir / "phase_reference_figure3c_validation.csv", index=False
    )

    matched_phase_contrasts = compute_matched_phase_contrasts(phase_indices)
    matched_pair_contrasts = compute_matched_pair_contrasts(pair_ratios)
    matched_edge_contrasts = compute_matched_edge_contrasts(edges)
    matched_phase_contrasts.to_csv(
        out_dir / "phase_matched_phase_contrasts.csv", index=False
    )
    matched_pair_contrasts.to_csv(
        out_dir / "phase_matched_pair_contrasts.csv", index=False
    )
    matched_edge_contrasts.to_csv(
        out_dir / "phase_matched_edge_count_contrasts.csv", index=False
    )

    full_phase_count = len(
        phase_indices.loc[
            (phase_indices.source_kind == "new")
            & (phase_indices.dataset_mode == "full")
        ]
    )
    if full_phase_count == len(PHASES):
        profile_comparison, ratio_comparison, edge_comparison = (
            compare_full_to_reference(profile, pair_ratios, edges)
        )
        reproduction = reproduction_summary(
            profile_comparison, ratio_comparison, edge_comparison
        )
        profile_comparison.to_csv(
            out_dir / "phase_full_vs_reference_celltype_profiles.csv", index=False
        )
        ratio_comparison.to_csv(
            out_dir / "phase_full_vs_reference_pair_ratios.csv", index=False
        )
        edge_comparison.to_csv(
            out_dir / "phase_full_vs_reference_edge_counts.csv", index=False
        )
        reproduction.to_csv(
            out_dir / "phase_full_vs_reference_reproduction_summary.csv", index=False
        )
    else:
        reproduction = pd.DataFrame()

    metadata = {
        "status": "complete" if not problems else "incomplete",
        "analysis": "production-compatible phase-duration matching",
        "reference_root": repository_relative(reference_root),
        "expected_matched_baseline_steady_replicates": (
            args.expected_matched_replicates
        ),
        "required_lags": list(LAGS),
        "primary_metric": (
            "mean absolute mu among FDR-significant edges within each of the "
            "nine source-to-target cell-type pairs (exact Figure 3C definition)"
        ),
        "threshold_free_metric": (
            "mean absolute mu among all off-diagonal candidate edges within "
            "each cell-type pair"
        ),
        "long_short_ratio": "mean(lags 3,5) / mean(lags 1,2)",
        "production_type_counts_in_reference": {
            type_name: int(
                sum(
                    production_neuron_type(name) == type_name
                    for name in load_result(
                        reference_root / "baseline" / "result.npz"
                    )[1]
                )
            )
            for type_name in (*TYPE_ORDER, "unknown")
        },
        "reference_result_sha256": reference_result_hashes,
        "reference_figure3c_maximum_absolute_error": anchor_maximum_error,
        "reference_figure3c_reproduced": bool(
            anchor_maximum_error <= args.anchor_tolerance
        ),
        "completed_new_full_phase_results": int(full_phase_count),
        "completed_new_matched_phase_results": int(
            len(
                phase_indices.loc[
                    (phase_indices.source_kind == "new")
                    & (phase_indices.dataset_mode == "matched")
                ]
            )
        ),
        "full_vs_reference_reproduction_available": bool(not reproduction.empty),
        "missing_or_incomplete_results": problems,
        "interpretation": (
            "Matched contrasts test whether duration/window-count imbalance is "
            "sufficient to explain the phase lag profiles. Crop and retraining "
            "variation are sensitivity analyses, not biological confidence intervals."
        ),
    }
    (out_dir / "phase_duration_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))

    if problems and not args.allow_incomplete:
        raise RuntimeError(
            f"Missing or incomplete {len(problems)} expected artifacts; "
            f"see {out_dir / 'phase_duration_summary.json'}"
        )


if __name__ == "__main__":
    main()
