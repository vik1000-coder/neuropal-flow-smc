#!/usr/bin/env python3
"""Validate and bootstrap the preregistered Tier-2 robustness studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPLICATES = 2_000
BOOTSTRAP_SEED = 113_991


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_study(study: Path, source: str) -> pd.DataFrame:
    frames = []
    for path in sorted((study / "cases").glob("*.csv")):
        frame = pd.read_csv(path)
        frame["study_component"] = source
        frame["metric_uri"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def audit_access(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["reported_information_access"] = result.information_access
    static = result.study_component.isin(["static_slices", "fixed_amplitude_diagnostic"])
    score = result.method_id.astype(str).str.startswith(("S2_", "S5_", "S6_", "S7_"))
    result.loc[static & score, "information_access"] = (
        "observational_training_plus_oracle_density_quadrature"
    )
    ratio = result.method_id == "A2_RATIO_CRITIC"
    result.loc[static & ratio, "information_access"] = (
        "observational_training_plus_conditional_evaluation_queries"
    )
    normalized = static & result.method_id.astype(str).str.startswith("S1_NLL")
    result.loc[normalized, "information_access"] = "known_dgp_tilt_basis_plus_fitted_amplitude"
    return result


def bootstrap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "study_component",
        "cell_id",
        "dgp_family",
        "estimand_type",
        "channel",
        "method_id",
        "information_access",
        "metric_name",
        "n_train",
        "context_length",
        "slice_name",
        "target_lambda",
        "achieved_lambda",
    ]
    for column in groups:
        if column not in frame:
            frame[column] = np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for keys, group in frame.groupby(groups, dropna=False, sort=True):
        values = group.metric_value.dropna().to_numpy(float)
        if values.size:
            draws = values[rng.integers(0, values.size, (REPLICATES, values.size))].mean(axis=1)
            mean = float(values.mean())
            lower = float(np.quantile(draws, 0.025))
            upper = float(np.quantile(draws, 0.975))
        else:
            mean = lower = upper = np.nan
        row = dict(zip(groups, keys, strict=True))
        row.update(
            mean=mean,
            ci_lower=lower,
            ci_upper=upper,
            n_dgp_seeds=int(values.size),
            failure_rate=float(np.mean(group.status != "ok")),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def validate(studies: list[tuple[Path, int]], frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    inventory = []
    mismatches = []
    counts = {}
    for study, expected in studies:
        csvs = sorted((study / "cases").glob("*.csv"))
        arrays = sorted((study / "cases").glob("*.npz"))
        counts[study.name] = {"metric_files": len(csvs), "array_files": len(arrays), "expected": expected}
        for kind, paths in [("metric", csvs), ("array", arrays)]:
            for path in paths:
                digest = sha256(path)
                inventory.append(
                    {
                        "study": study.name,
                        "kind": kind,
                        "uri": str(path),
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                    }
                )
                if kind == "array":
                    registered = set(frame.loc[frame.array_uri == str(path.relative_to(study)), "array_sha256"].dropna())
                    if registered != {digest}:
                        mismatches.append(str(path))
        for path in [study / "manifest.json", study / "calibration.csv"]:
            if path.exists():
                inventory.append(
                    {
                        "study": study.name,
                        "kind": "frozen_input",
                        "uri": str(path),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
    missing = frame[frame.metric_value.isna()]
    expected_missing = missing.metric_name.eq("signed_correlation")
    seed_counts = (
        frame[frame.metric_name.isin(["nrmse", "mean_null_rms", "null_rms"])]
        .groupby(["study_component", "cell_id", "channel", "method_id"])
        .dgp_seed.nunique()
    )
    payload = {
        "created_unix": time.time(),
        "study_counts": counts,
        "metric_rows": int(frame.shape[0]),
        "failed_rows": int((frame.status != "ok").sum()),
        "nonfinite_values": int((~np.isfinite(frame.metric_value.dropna())).sum()),
        "expected_undefined_diagnostics": int(expected_missing.sum()),
        "unexpected_missing_values": int((~expected_missing).sum()),
        "raw_hash_mismatches": mismatches,
        "min_primary_seed_count": int(seed_counts.min()) if not seed_counts.empty else 0,
        "max_primary_seed_count": int(seed_counts.max()) if not seed_counts.empty else 0,
    }
    complete = all(
        value["metric_files"] == value["array_files"] == value["expected"]
        for value in counts.values()
    )
    payload["passed"] = bool(
        complete
        and payload["failed_rows"] == 0
        and payload["nonfinite_values"] == 0
        and payload["unexpected_missing_values"] == 0
        and not mismatches
        and payload["min_primary_seed_count"] == 30
        and payload["max_primary_seed_count"] == 30
    )
    return payload, pd.DataFrame(inventory)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic-study", required=True, type=Path)
    parser.add_argument("--static-study", required=True, type=Path)
    parser.add_argument("--fixed-amplitude-study", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dynamic = args.dynamic_study.resolve()
    static = args.static_study.resolve()
    fixed = args.fixed_amplitude_study.resolve() if args.fixed_amplitude_study else None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        read_study(dynamic, "dynamic_robustness"),
        read_study(static, "static_slices"),
    ]
    studies = [(dynamic, 270), (static, 450)]
    if fixed is not None:
        sources.append(read_study(fixed, "fixed_amplitude_diagnostic"))
        studies.append((fixed, 270))
    frame = audit_access(pd.concat(sources, ignore_index=True, sort=False))
    validation, inventory = validate(studies, frame)
    atomic_json(output / "validation.json", validation)
    atomic_csv(output / "artifact_inventory.csv", inventory)
    if not validation["passed"]:
        raise SystemExit("Tier-2 validation failed")
    atomic_csv(output / "seed_level.csv", frame)
    frame.to_parquet(output / "seed_level.parquet", index=False)
    summary = bootstrap_summary(frame)
    atomic_csv(output / "summary.csv", summary)
    summary.to_parquet(output / "summary.parquet", index=False)
    manifest = {
        "created_unix": time.time(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": REPLICATES,
        "dynamic_manifest_sha256": sha256(dynamic / "manifest.json"),
        "static_manifest_sha256": sha256(static / "manifest.json"),
        "validation_sha256": sha256(output / "validation.json"),
        "seed_level_sha256": sha256(output / "seed_level.csv"),
        "summary_sha256": sha256(output / "summary.csv"),
    }
    if fixed is not None:
        manifest["fixed_amplitude_manifest_sha256"] = sha256(fixed / "manifest.json")
    atomic_json(output / "package_manifest.json", manifest)
    print(json.dumps(validation, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
