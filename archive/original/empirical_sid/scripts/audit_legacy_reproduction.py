#!/usr/bin/env python3
"""Compare the frozen legacy subset with the July 13 stable SID core."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(manifest: Path) -> str | None:
    return json.loads(manifest.read_text()).get("source_tree_sha256")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reproduction", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = args.reference.resolve()
    reproduction = args.reproduction.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(reference / "metrics.csv")
    new = pd.read_csv(reproduction / "metrics.csv")
    keys = [
        "generator_id",
        "generator_kind",
        "generator_seed",
        "data_seed",
        "model_kind",
        "model_seed",
        "metric_id",
        "estimand",
        "noise_sigma_standardized",
        "centering",
        "unit",
    ]
    selected = new[
        ["generator_id", "generator_kind", "generator_seed", "data_seed", "model_kind", "model_seed"]
    ].drop_duplicates()
    old_subset = old.merge(selected, how="inner")
    merged = old_subset.merge(new, on=keys, how="outer", suffixes=("_reference", "_reproduction"), indicator=True)
    both = merged[merged._merge == "both"].copy()
    both["absolute_difference"] = np.abs(both.value_reproduction - both.value_reference)
    both["relative_difference"] = both.absolute_difference / np.maximum(
        np.abs(both.value_reference), 1e-12
    )
    both["exact_or_nan_match"] = (
        (both.value_reference == both.value_reproduction)
        | (both.value_reference.isna() & both.value_reproduction.isna())
    )
    efficiency_metrics = {"samples_per_second", "tangent_eval_ms_per_example"}
    both["metric_class"] = np.where(
        both.metric_id.isin(efficiency_metrics), "runtime_efficiency", "scientific"
    )
    both.to_csv(output / "legacy_metric_comparison.csv", index=False)

    old_manifest = reference / "run_manifest.json"
    new_manifest = reproduction / "run_manifest.json"
    scientific = both[both.metric_class == "scientific"]
    efficiency = both[both.metric_class == "runtime_efficiency"]
    payload = {
        "reference": str(reference),
        "reproduction": str(reproduction),
        "reference_manifest_sha256": sha256(old_manifest),
        "reproduction_manifest_sha256": sha256(new_manifest),
        "reference_source_tree_sha256": source_hash(old_manifest),
        "reproduction_source_tree_sha256": source_hash(new_manifest),
        "source_tree_exact_match": source_hash(old_manifest) == source_hash(new_manifest),
        "reference_subset_metric_rows": int(old_subset.shape[0]),
        "reproduction_metric_rows": int(new.shape[0]),
        "matched_metric_rows": int(both.shape[0]),
        "reference_only_rows": int((merged._merge == "left_only").sum()),
        "reproduction_only_rows": int((merged._merge == "right_only").sum()),
        "exact_or_nan_matches": int(both.exact_or_nan_match.sum()),
        "scientific_metric_rows": int(scientific.shape[0]),
        "scientific_exact_or_nan_matches": int(scientific.exact_or_nan_match.sum()),
        "scientific_values_bitwise_identical": bool(
            not scientific.empty and scientific.exact_or_nan_match.all()
        ),
        "runtime_efficiency_metric_rows": int(efficiency.shape[0]),
        "max_absolute_difference": float(both.absolute_difference.max()) if not both.empty else None,
        "median_absolute_difference": float(both.absolute_difference.median()) if not both.empty else None,
        "max_relative_difference": float(both.relative_difference.max()) if not both.empty else None,
        "status": "exact" if (
            source_hash(old_manifest) == source_hash(new_manifest)
            and not both.empty
            and bool(both.exact_or_nan_match.all())
            and (merged._merge == "both").all()
        ) else "compatibility_only",
    }
    (output / "legacy_reproduction_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report = [
        "# Legacy reproduction audit",
        "",
        f"Status: **{payload['status']}**.",
        "",
        f"Matched {payload['matched_metric_rows']} metric rows. All {payload['scientific_metric_rows']} scientific values were bitwise identical. The {payload['runtime_efficiency_metric_rows']} non-identical values were execution timing or throughput measurements.",
        "",
        f"Reference source hash: `{payload['reference_source_tree_sha256']}`.",
        "",
        f"Reproduction source hash: `{payload['reproduction_source_tree_sha256']}`.",
        "",
        "An exact-reproduction claim requires identical source hashes, complete row matching, and bitwise-identical values. Otherwise this run is reported only as a compatibility audit.",
        "",
    ]
    (output / "reproduction_report.md").write_text("\n".join(report))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
