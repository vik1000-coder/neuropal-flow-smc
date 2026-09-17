from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def finalize(
    shards: list[Path], output: Path, model_id: str,
    folds: list[int], seeds: list[int],
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    response_dir = output / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    records = []
    source_runs = set()
    for shard in shards:
        shard = shard.resolve()
        manifest_path = shard / "manifest.json"
        status_path = shard / "checkpoint_status.csv"
        if not manifest_path.exists() or not status_path.exists():
            raise FileNotFoundError(f"missing shard manifest/status under {shard}")
        manifest = json.loads(manifest_path.read_text())
        source_runs.add(str(Path(manifest["source_run"]).resolve()))
        with status_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    raise RuntimeError(f"non-ok response row in {shard}: {row}")
                if row.get("model_id") != model_id:
                    raise RuntimeError(f"unexpected model in {shard}: {row}")
                source = shard / row["output"]
                if not source.exists():
                    raise FileNotFoundError(source)
                destination = response_dir / source.name
                if destination.exists() or destination.is_symlink():
                    if destination.resolve() != source.resolve():
                        raise RuntimeError(f"response collision: {destination}")
                else:
                    destination.symlink_to(source.resolve())
                record = dict(row)
                record["output"] = str(destination.relative_to(output))
                record["worker_shard"] = str(shard)
                records.append(record)
    if len(source_runs) != 1:
        raise RuntimeError(f"response shards disagree on source run: {source_runs}")
    keys = [(int(row["fold"]), int(row["seed"])) for row in records]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate fold/seed response rows")
    expected = {(fold, seed) for fold in folds for seed in seeds}
    observed = set(keys)
    complete = observed == expected
    fields = sorted(set().union(*(row.keys() for row in records)))
    with (output / "checkpoint_status.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (int(row["fold"]), int(row["seed"]))))
    _write_json(
        output / "manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "role": "consolidated direct repaired-response ensemble",
            "source_run": next(iter(source_runs)),
            "worker_shards": [str(path.resolve()) for path in shards],
            "model_id": model_id,
            "folds": folds,
            "seeds": seeds,
            "estimator": "direct paired importance weighting",
            "claim_boundary": "model-relative observational repaired response; not causal or anatomical",
        },
    )
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "partial",
        "expected_fold_seed_keys": sorted([list(key) for key in expected]),
        "observed_fold_seed_keys": sorted([list(key) for key in observed]),
        "response_files": len(records),
        "duplicates": len(keys) - len(set(keys)),
    }
    _write_json(output / "validation.json", validation)
    (output / "REPORT.md").write_text(
        "# Frozen wide-flow direct repaired-response ensemble\n\n"
        f"Status: **{validation['status']}**; {len(records)}/{len(expected)} fold/seed checkpoints.\n\n"
        "This archive contains direct paired-importance repaired responses for the atlas-blind predictive winner. External-reference analysis is a separate post-freeze step. These are observational, model-relative responses, not physical interventions or anatomical edges.\n"
    )
    files = [
        path for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(
            f"{_sha256(path.resolve())}  {path.relative_to(output)}" for path in files
        )
        + "\n"
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903, 4307])
    args = parser.parse_args()
    validation = finalize(
        args.shards, args.output_dir.resolve(), args.model_id,
        args.folds, args.seeds,
    )
    if validation["status"] != "complete":
        raise RuntimeError(f"incomplete response consolidation: {validation}")


if __name__ == "__main__":
    main()
