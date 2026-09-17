from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from conditional_neural_benchmark.rollout_evaluator import write_report


METRICS = [
    "energy",
    "energy__stim_balanced",
    "variogram",
    "rmse",
    "coverage90",
    "sharpness90",
    "finite_fraction",
    "explosive_fraction_abs_z_gt_10",
    "mean_abs_z",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def finalize(
    shards: list[Path],
    output: Path,
    expected_model_ids: list[str],
    expected_horizons: list[int],
    expected_checkpoints_per_model: int,
) -> dict:
    frames = []
    for shard in shards:
        source = shard.resolve() / "rollout_metrics_by_checkpoint.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source)
        frame["worker_shard"] = str(shard.resolve())
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    keys = ["checkpoint", "rollout_horizon_frames"]
    duplicates = combined[combined.duplicated(keys, keep=False)]
    if not duplicates.empty:
        raise RuntimeError(
            "duplicate rollout rows: "
            + str(duplicates[keys].drop_duplicates().to_dict("records"))
        )
    combined = combined.sort_values(
        ["model_id", "fold", "seed", "rollout_horizon_frames"]
    ).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "rollout_metrics_by_checkpoint.csv", index=False)
    aggregate = (
        combined.groupby(
            ["model_id", "rollout_horizon_frames", "rollout_horizon_seconds"]
        )[METRICS]
        .agg(["mean", "std", "count"])
    )
    aggregate.columns = ["__".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(output / "rollout_metrics_aggregate.csv", index=False)
    write_report(output, aggregate)

    expected_models = set(expected_model_ids)
    observed_models = set(combined.model_id.astype(str))
    expected_horizon_set = set(expected_horizons)
    checkpoint_counts = combined.groupby("model_id").checkpoint.nunique().to_dict()
    horizon_coverage = {
        model: set(group.rollout_horizon_frames.astype(int))
        for model, group in combined.groupby("model_id")
    }
    finite = bool(np.isfinite(combined[METRICS].to_numpy(dtype=float)).all())
    complete = bool(
        observed_models == expected_models
        and all(
            int(checkpoint_counts.get(model, 0)) == expected_checkpoints_per_model
            for model in expected_models
        )
        and all(
            horizon_coverage.get(model, set()) == expected_horizon_set
            for model in expected_models
        )
        and finite
    )
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "partial",
        "expected_model_ids": sorted(expected_models),
        "observed_model_ids": sorted(observed_models),
        "expected_horizons_frames": sorted(expected_horizon_set),
        "expected_checkpoints_per_model": expected_checkpoints_per_model,
        "observed_checkpoints_per_model": {
            str(key): int(value) for key, value in checkpoint_counts.items()
        },
        "rows": int(len(combined)),
        "all_summary_metrics_finite": finite,
        "duplicate_rows": int(len(duplicates)),
    }
    _write_json(output / "validation.json", validation)
    _write_json(
        output / "protocol.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "role": "consolidated frozen multi-step rollout evaluation",
            "worker_shards": [str(path.resolve()) for path in shards],
            "external_atlas_access": "none",
            "claim_boundary": "predictive observed-activity rollout; not causal or anatomical",
        },
    )
    files = [
        path for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(output)}" for path in files
        )
        + "\n"
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-model-ids", nargs="+", required=True)
    parser.add_argument(
        "--expected-horizons", nargs="+", type=int,
        default=[1, 2, 4, 8, 16, 40],
    )
    parser.add_argument("--expected-checkpoints-per-model", type=int, default=4)
    args = parser.parse_args()
    validation = finalize(
        shards=args.shards,
        output=args.output_dir.resolve(),
        expected_model_ids=args.expected_model_ids,
        expected_horizons=args.expected_horizons,
        expected_checkpoints_per_model=args.expected_checkpoints_per_model,
    )
    if validation["status"] != "complete":
        raise RuntimeError(f"incomplete rollout consolidation: {validation}")


if __name__ == "__main__":
    main()
