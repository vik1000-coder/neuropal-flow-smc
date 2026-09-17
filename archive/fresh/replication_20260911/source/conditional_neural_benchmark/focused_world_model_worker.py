from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.focused_world_model_runner import (
    _load_folds,
    candidate_configs,
)
from conditional_neural_benchmark.runner import (
    RunState,
    _neural_trial,
    _split_windows,
    _update_leaderboards,
)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _done(state: RunState, phase: str, model: str, fold: int, seed: int) -> bool:
    return any(
        row.get("status") == "ok"
        and row.get("phase") == phase
        and row.get("model_id") == model
        and int(row.get("fold", -1)) == fold
        and int(row.get("seed", -1)) == seed
        for row in state.records
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-run", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--model-ids", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--hours", type=float, default=9.5)
    parser.add_argument("--lag", type=int, default=80)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--eval-rows", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(run_dir, time.monotonic() + args.hours * 3600)
    metrics = run_dir / "trial_metrics.csv"
    if args.resume and metrics.exists():
        state.records = (
            pd.read_csv(metrics).replace({np.nan: None}).to_dict("records")
        )
    configs = {config.model_id: config for config in candidate_configs()}
    unknown = sorted(set(args.model_ids) - set(configs))
    if unknown:
        raise ValueError(f"unknown model identifiers: {unknown}")
    cohort = load_cohort(args.coverage)
    folds = _load_folds(cohort, args.evidence_run.resolve())
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        _write_json(
            manifest,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "role": "independent atlas-blind tournament worker",
                "phase": args.phase,
                "model_ids": args.model_ids,
                "folds": args.folds,
                "seeds": args.seeds,
                "lag": args.lag,
                "device": args.device,
                "threads": args.threads,
                "epochs": args.epochs,
                "patience": args.patience,
                "eval_rows": args.eval_rows,
                "samples": args.samples,
                "batch_size": args.batch_size,
                "evidence_run": str(args.evidence_run.resolve()),
                "atlas_firewall": "no Randi, Cook, Bentley, SBTG, or external atlas",
                "configs": [configs[model].to_dict() for model in args.model_ids],
            },
        )
    for model_id in args.model_ids:
        config = configs[model_id]
        for fold in args.folds:
            scaler, train, validation, test = _split_windows(
                cohort, folds, fold, args.lag
            )
            for seed in args.seeds:
                if _done(state, args.phase, model_id, fold, seed):
                    print(
                        f"TRIAL_SKIP phase={args.phase} model={model_id} "
                        f"fold={fold} seed={seed}",
                        flush=True,
                    )
                    continue
                if not state.can_start(10):
                    break
                _neural_trial(
                    state=state,
                    phase=args.phase,
                    config=config,
                    cohort=cohort,
                    lag=args.lag,
                    fold=fold,
                    seed=seed,
                    train=train,
                    validation=validation,
                    test=test,
                    scaler=scaler,
                    device=args.device,
                    max_epochs=args.epochs,
                    patience=args.patience,
                    eval_rows=args.eval_rows,
                    n_samples=args.samples,
                    batch_size=args.batch_size,
                )
                _update_leaderboards(state)
    expected = len(args.model_ids) * len(args.folds) * len(args.seeds)
    complete = sum(row.get("status") == "ok" for row in state.records) == expected
    _write_json(
        run_dir / "validation.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if complete else "partial",
            "expected_trials": expected,
            "completed_trials": sum(
                row.get("status") == "ok" for row in state.records
            ),
            "failed_trials": sum(
                row.get("status") == "failed" for row in state.records
            ),
        },
    )


if __name__ == "__main__":
    main()
