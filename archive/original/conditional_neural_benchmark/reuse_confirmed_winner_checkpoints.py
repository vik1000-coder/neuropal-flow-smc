from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament-run", type=Path, required=True)
    parser.add_argument("--output-shard", type=Path, required=True)
    args = parser.parse_args()
    tournament = args.tournament_run.resolve()
    output = args.output_shard.resolve()
    output.mkdir(parents=True, exist_ok=True)
    winner = json.loads((tournament / "winner_selection.json").read_text())["model_id"]
    frame = pd.read_csv(tournament / "trial_metrics.csv")
    frame = frame[
        (frame.phase == "focused_confirmation")
        & (frame.model_id == winner)
        & (frame.status == "ok")
    ].copy()
    if len(frame) != 4:
        raise RuntimeError(f"expected four confirmed winner trials, found {len(frame)}")
    reused = []
    for row in frame.to_dict("records"):
        source = tournament / str(row["checkpoint"])
        destination = output / "checkpoints" / "winner_full_cv" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(source.resolve())
        row["phase"] = "winner_full_cv"
        row["checkpoint"] = str(destination.relative_to(output))
        row["reuse_source_phase"] = "focused_confirmation"
        row["reuse_exact_no_retraining"] = True
        reused.append(row)
    pd.DataFrame(reused).to_csv(output / "trial_metrics.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "role": "exact checkpoint reuse for confirmed winner full-CV ensemble",
                "model_id": winner,
                "source_tournament": str(tournament),
                "source_phase": "focused_confirmation",
                "destination_phase": "winner_full_cv",
                "folds": [3, 4],
                "seeds": [1701, 2903],
                "no_retraining": True,
                "reason": "confirmation and full-CV protocols use identical fit/evaluation hyperparameters",
                "external_references_consulted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "validation.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_trials": 4,
                "completed_trials": len(reused),
                "all_checkpoints_resolve": all(
                    (output / row["checkpoint"]).resolve().exists() for row in reused
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
