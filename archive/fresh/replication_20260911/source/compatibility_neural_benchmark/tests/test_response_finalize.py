from pathlib import Path

import csv
import json

from compatibility_neural_benchmark.response_finalize import finalize


def test_response_finalize_exact_coverage(tmp_path: Path) -> None:
    shards = []
    for fold in (0, 1):
        shard = tmp_path / f"f{fold}"
        response = shard / "responses" / f"winner__B4__f{fold}__s7.npz"
        response.parent.mkdir(parents=True)
        response.write_bytes(b"response")
        (shard / "manifest.json").write_text(
            json.dumps({"source_run": str(tmp_path / "source")})
        )
        with (shard / "checkpoint_status.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["status", "model_id", "fold", "seed", "output"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "status": "ok", "model_id": "winner", "fold": fold,
                    "seed": 7, "output": str(response.relative_to(shard)),
                }
            )
        shards.append(shard)
    validation = finalize(shards, tmp_path / "final", "winner", [0, 1], [7])
    assert validation["status"] == "complete"
    assert validation["response_files"] == 2
