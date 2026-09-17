#!/usr/bin/env python3
"""Derive exploratory duration-matched lag-5/lag-1 phase ratios.

The calculation consumes cell-type profiles and does not retrain a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.snapshot import portable_path


METHOD_COLUMNS = {
    "all": "mean_abs_mu_all",
    "bh": "mean_abs_mu_bh",
    "by": "mean_abs_mu_by",
}
EXPECTED_PHASES = {"Baseline", "Steady", "On", "Off"}
EXPECTED_SEEDS = {42, 43, 44}
EXPECTED_LAGS = {1, 2, 3, 5}
EXPECTED_PAIRS_PER_PROFILE = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            root
            / "results"
            / "robustness"
            / "phase_duration"
            / "exploratory_endpoint"
            / "input_profiles.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "results"
            / "derived"
            / "phase_duration_endpoint"
        ),
    )
    return parser.parse_args()


def validate_input(data: pd.DataFrame) -> None:
    required = {
        "variant",
        "phase",
        "model_seed",
        "replicate",
        "pair",
        "lag",
        *METHOD_COLUMNS.values(),
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")

    if set(data["variant"].unique()) != {"distributed_production"}:
        raise ValueError("endpoint table must use only distributed_production fits")
    if set(data["phase"].unique()) != EXPECTED_PHASES:
        raise ValueError("unexpected phase set")
    if set(data["model_seed"].unique()) != EXPECTED_SEEDS:
        raise ValueError("unexpected model-seed set")
    if set(data["lag"].unique()) != EXPECTED_LAGS:
        raise ValueError("unexpected lag set")
    if data[list(METHOD_COLUMNS.values())].isna().any().any():
        raise ValueError("endpoint inputs contain missing magnitudes")

    group_cols = ["phase", "model_seed", "replicate", "lag"]
    pair_counts = data.groupby(group_cols)["pair"].nunique()
    if not (pair_counts == EXPECTED_PAIRS_PER_PROFILE).all():
        raise ValueError("every phase/seed/replicate/lag must contain nine cell-type pairs")


def calculate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Equal weight is assigned to each of the nine cell-type-pair profiles.
    group_cols = ["phase", "model_seed", "replicate", "lag"]
    profiles = data.groupby(group_cols, as_index=False)[list(METHOD_COLUMNS.values())].mean()

    endpoint_rows: list[dict[str, object]] = []
    base_cols = ["phase", "model_seed", "replicate"]
    for key, group in profiles.groupby(base_cols):
        indexed = group.set_index("lag")
        row: dict[str, object] = dict(zip(base_cols, key))
        for short_name, column in METHOD_COLUMNS.items():
            denominator = float(indexed.loc[1, column])
            if denominator <= 0:
                raise ValueError(f"non-positive lag-1 denominator for {key}, {short_name}")
            row[f"{short_name}_lag5_lag1"] = float(indexed.loc[5, column]) / denominator
        endpoint_rows.append(row)
    by_replicate = pd.DataFrame(endpoint_rows)

    # Baseline and Steady have three duration-matched draws per model seed.
    # On and Off are fixed samples.  Average replicate ratios within a seed so
    # that the reported SD reflects model initialization, not crop replication.
    ratio_columns = [f"{name}_lag5_lag1" for name in METHOD_COLUMNS]
    by_seed = (
        by_replicate.groupby(["phase", "model_seed"], as_index=False)[ratio_columns]
        .mean()
        .merge(
            by_replicate.groupby(["phase", "model_seed"], as_index=False)
            .size()
            .rename(columns={"size": "n_sampling_replicates"}),
            on=["phase", "model_seed"],
            validate="one_to_one",
        )
        .sort_values(["model_seed", "phase"])
        .reset_index(drop=True)
    )

    summary_rows: list[dict[str, object]] = []
    for phase, group in by_seed.groupby("phase"):
        row: dict[str, object] = {
            "phase": phase,
            "n_model_seeds": int(group["model_seed"].nunique()),
        }
        for method in METHOD_COLUMNS:
            values = group[f"{method}_lag5_lag1"]
            row[f"{method}_mean"] = float(values.mean())
            row[f"{method}_sd"] = float(values.std(ddof=1))
        summary_rows.append(row)
    phase_summary = pd.DataFrame(summary_rows).sort_values("phase").reset_index(drop=True)

    contrast_rows: list[dict[str, object]] = []
    indexed = by_seed.set_index(["phase", "model_seed"])
    for method in METHOD_COLUMNS:
        column = f"{method}_lag5_lag1"
        for event in ("On", "Off"):
            for reference in ("Baseline", "Steady"):
                delta = indexed.loc[event, column] - indexed.loc[reference, column]
                contrast_rows.append(
                    {
                        "method": method,
                        "event_phase": event,
                        "reference_phase": reference,
                        "mean_delta": float(delta.mean()),
                        "sd_delta": float(delta.std(ddof=1)),
                        "positive_seed_count": int((delta > 0).sum()),
                        "n_model_seeds": int(delta.size),
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    return by_seed, phase_summary, contrasts


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    data = pd.read_csv(input_path)
    validate_input(data)
    by_seed, phase_summary, contrasts = calculate(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    by_seed_path = output_dir / "lag5_lag1_by_seed.csv"
    summary_path = output_dir / "lag5_lag1_phase_summary.csv"
    contrast_path = output_dir / "lag5_lag1_contrasts.csv"
    metadata_path = output_dir / "lag5_lag1_summary.json"
    by_seed.to_csv(by_seed_path, index=False)
    phase_summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrast_path, index=False)

    bh = phase_summary.set_index("phase")
    expected_bh = {
        "Baseline": (1.160, 0.012),
        "Steady": (1.191, 0.029),
        "On": (1.251, 0.036),
        "Off": (1.240, 0.030),
    }
    for phase, (expected_mean, expected_sd) in expected_bh.items():
        actual = (round(float(bh.loc[phase, "bh_mean"]), 3), round(float(bh.loc[phase, "bh_sd"]), 3))
        if actual != (expected_mean, expected_sd):
            raise AssertionError(f"report-table mismatch for {phase}: {actual}")

    metadata = {
        "input": portable_path(input_path),
        "input_sha256": sha256(input_path),
        "variant": "distributed_production",
        "model_seeds": sorted(EXPECTED_SEEDS),
        "lags": sorted(EXPECTED_LAGS),
        "metric": (
            "For each phase/model-seed/sampling-replicate, take the equal-weight mean "
            "absolute coupling across the nine cell-type pairs at each lag, then divide "
            "lag 5 by lag 1. Average sampling-replicate ratios within a model seed and "
            "report the mean and sample SD across model seeds."
        ),
        "thresholds": {
            "all": "all candidate edges",
            "bh": "production Benjamini-Hochberg-selected edges",
            "by": "Benjamini-Yekutieli-selected edges",
        },
        "report_table_validation": "passed",
        "caveats": [
            "The lag-5/lag-1 endpoint ratio was selected after inspecting alternative summaries.",
            "Only BH-selected edges separate On and Off from both references in every model seed.",
            "All-edge and BY summaries do not establish a network-wide phase shift.",
            "The duration-matched Baseline and Steady samples are distributed draws across each phase.",
            "The production Figure 3C inference mask is BH, despite a broader paper-level description of BY.",
        ],
        "outputs": [
            by_seed_path.name,
            summary_path.name,
            contrast_path.name,
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote endpoint summary to {output_dir}")


if __name__ == "__main__":
    main()
