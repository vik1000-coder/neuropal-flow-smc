from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from conditional_neural_benchmark.focused_world_model_runner import (
    _write_report,
    candidate_configs,
    choose_finalists,
)
from conditional_neural_benchmark.runner import _leaderboard


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_checkpoint(source: Path, output: Path, phase: str) -> str:
    destination = output / "checkpoints" / phase / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"checkpoint collision: {destination}")
    else:
        destination.symlink_to(source.resolve())
    return str(destination.relative_to(output))


def _read_shards(shards: list[Path], output: Path) -> pd.DataFrame:
    records = []
    for shard in shards:
        metrics = shard / "trial_metrics.csv"
        if not metrics.exists():
            continue
        frame = pd.read_csv(metrics)
        for row in frame.to_dict("records"):
            checkpoint = row.get("checkpoint")
            if isinstance(checkpoint, str) and checkpoint:
                row["checkpoint"] = _link_checkpoint(
                    shard / checkpoint, output, str(row["phase"])
                )
            row["worker_shard"] = str(shard)
            records.append(row)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("no worker records found")
    keys = ["phase", "model_id", "fold", "seed"]
    duplicates = frame[frame.duplicated(keys, keep=False)]
    if not duplicates.empty:
        raise RuntimeError(
            "duplicate worker trials: "
            + str(duplicates[keys].drop_duplicates().to_dict("records"))
        )
    return frame.sort_values(keys).reset_index(drop=True)


def _boards(records: list[dict]) -> pd.DataFrame:
    boards = []
    for phase in sorted({str(row.get("phase")) for row in records}):
        board = _leaderboard(records, phase)
        if not board.empty:
            board.insert(0, "phase", phase)
            boards.append(board)
    return pd.concat(boards, ignore_index=True) if boards else pd.DataFrame()


def _write_checksums(output: Path) -> None:
    paths = [
        path for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(
            f"{_sha256(path.resolve())}  {path.relative_to(output)}" for path in paths
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-run", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--finalists", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shards = [path.resolve() for path in args.shards]
    frame = _read_shards(shards, output)
    frame.to_csv(output / "trial_metrics.csv", index=False)
    records = frame.replace({np.nan: None}).to_dict("records")
    boards = _boards(records)
    boards.to_csv(output / "leaderboard.csv", index=False)
    screen = _leaderboard(records, "focused_screen")
    configs = candidate_configs()
    complete_screen = (
        not screen.empty
        and set(screen.model_id) == {config.model_id for config in configs}
        and (screen.n_trials == 3).all()
    )
    selection_path = output / "selection.json"
    finalists = []
    if complete_screen:
        proposed = choose_finalists(screen, configs, args.finalists)
        if selection_path.exists():
            selected_ids = json.loads(selection_path.read_text())["selected_model_ids"]
            if selected_ids != [config.model_id for config in proposed]:
                raise RuntimeError("frozen finalist selection disagrees with recomputation")
        else:
            selected_ids = [config.model_id for config in proposed]
            _write_json(
                selection_path,
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "selected_model_ids": selected_ids,
                    "screen_leaderboard": screen.to_dict("records"),
                    "external_references_consulted": False,
                },
            )
        by_id = {config.model_id: config for config in configs}
        finalists = [by_id[model_id] for model_id in selected_ids]

    evidence = args.evidence_run.resolve()
    fold_source = evidence / "fold_assignments.csv"
    fold_destination = output / "fold_assignments.csv"
    if not fold_destination.exists():
        shutil.copy2(fold_source, fold_destination)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "parallelized atlas-blind focused conditional-density tournament",
        "parallelization_only": True,
        "worker_shards": [str(path) for path in shards],
        "fold_assignments_sha256": _sha256(fold_source),
        "lag_frames": 80,
        "screen_folds": [0, 1, 2],
        "confirmation_folds": [3, 4],
        "screen_seed": 1701,
        "confirmation_seeds": [1701, 2903],
        "screen_eval_policy": "all held-out windows (cap 4000)",
        "selection_rule": "within one SE of best natural energy, then stimulus-balanced energy, then variogram; preserve energy/exact family coverage",
        "atlas_firewall": "no Randi, Cook, Bentley, SBTG, or external atlas",
        "candidate_configs": [config.to_dict() for config in configs],
    }
    _write_json(output / "manifest.json", manifest)

    confirmation = _leaderboard(records, "focused_confirmation")
    expected_confirmation = len(finalists) * 4 if finalists else 0
    complete_confirmation = bool(
        finalists
        and not confirmation.empty
        and set(confirmation.model_id) == {config.model_id for config in finalists}
        and (confirmation.n_trials == 4).all()
    )
    confirmed_winner = None
    if complete_confirmation:
        confirmed_winner = str(confirmation.iloc[0].model_id)
        winner_path = output / "winner_selection.json"
        if winner_path.exists():
            frozen = json.loads(winner_path.read_text())["model_id"]
            if frozen != confirmed_winner:
                raise RuntimeError("frozen confirmed winner disagrees with recomputation")
        else:
            _write_json(
                winner_path,
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model_id": confirmed_winner,
                    "confirmation_leaderboard": confirmation.to_dict("records"),
                    "external_references_consulted": False,
                    "purpose": "freeze predictive winner before rollout, response refit, or atlas access",
                },
            )
    full = _leaderboard(records, "winner_full_cv")
    complete_full = bool(
        not full.empty and len(full) == 1 and int(full.iloc[0].n_trials) == 15
    )
    complete = complete_screen and complete_confirmation
    if finalists:
        _write_report(output, finalists, complete)
    _write_json(
        output / "validation.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if complete else "partial",
            "complete_screen": bool(complete_screen),
            "screen_trials": int((frame.phase == "focused_screen").sum()),
            "selected_finalists": [config.model_id for config in finalists],
            "complete_confirmation": bool(complete_confirmation),
            "confirmed_winner": confirmed_winner,
            "expected_confirmation_trials": expected_confirmation,
            "confirmation_trials": int(
                (frame.phase == "focused_confirmation").sum()
            ),
            "complete_winner_full_cv": bool(complete_full),
            "winner_full_cv_trials": int((frame.phase == "winner_full_cv").sum()),
            "failed_trials": int((frame.status == "failed").sum()),
        },
    )
    _write_checksums(output)


if __name__ == "__main__":
    main()
