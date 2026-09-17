from pathlib import Path

import pandas as pd

from conditional_neural_benchmark.rollout_finalize import METRICS, finalize


def test_rollout_finalize_checks_complete_coverage(tmp_path: Path) -> None:
    shards = []
    for model_index, model in enumerate(("a", "b")):
        shard = tmp_path / f"shard_{model}"
        shard.mkdir()
        rows = []
        for checkpoint in range(2):
            for horizon in (1, 2):
                row = {
                    "checkpoint": str(shard / f"{model}_{checkpoint}.pt"),
                    "model_id": model,
                    "fold": checkpoint,
                    "seed": 1701,
                    "rollout_horizon_frames": horizon,
                    "rollout_horizon_seconds": horizon / 4,
                }
                row.update({metric: 0.1 + model_index for metric in METRICS})
                rows.append(row)
        pd.DataFrame(rows).to_csv(
            shard / "rollout_metrics_by_checkpoint.csv", index=False
        )
        shards.append(shard)
    validation = finalize(
        shards, tmp_path / "final", ["a", "b"], [1, 2], 2
    )
    assert validation["status"] == "complete"
    assert validation["rows"] == 8
    assert (tmp_path / "final" / "REPORT.md").exists()
