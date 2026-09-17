from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.runner import (
    ModelConfig,
    RunState,
    _neural_trial,
    _split_windows,
)


@dataclass(frozen=True)
class EncodingSpec:
    name: str
    encoding: str
    shuffled: bool = False
    sensitivity_only: bool = False


ENCODINGS = (
    EncodingSpec("binary_any_stimulus", "binary_any_stimulus"),
    EncodingSpec("position_onehot", "position_onehot"),
    EncodingSpec("chemical_scalar", "chemical_scalar", sensitivity_only=True),
    EncodingSpec("chemical_onehot", "chemical_onehot"),
    EncodingSpec("chemical_plus_position_onehot", "chemical_plus_position_onehot"),
    EncodingSpec(
        "chemical_onehot_subject_shuffle",
        "chemical_onehot_subject_shuffle",
        shuffled=True,
    ),
)


def config_for(spec: EncodingSpec) -> ModelConfig:
    return ModelConfig(
        model_id=f"stim_{spec.name}_tcn_flow128_dropout15_jitter_p01",
        encoder="tcn",
        head="conditional_flow_matching",
        residual_target=True,
        width=128,
        dropout=0.15,
        weight_decay=7.5e-4,
        head_params={"hidden": 128, "layers": 4, "sample_steps": 24},
        history_noise_std=0.01,
        history_noise_copies=1,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_resume_manifest(
    manifest_path: Path, stimulus_fingerprint: str, fold_fingerprint: str
) -> dict:
    actual = json.loads(manifest_path.read_text())
    if actual.get("stimulus_schema_fingerprint") != stimulus_fingerprint:
        raise RuntimeError("resume rejected: stimulus-schema fingerprint differs")
    if actual.get("fold_assignments_sha256") != fold_fingerprint:
        raise RuntimeError("resume rejected: fold-assignment fingerprint differs")
    return actual


def frozen_shuffles(cohort, seed: int = 98231) -> dict[int, tuple[int, int, int]]:
    result: dict[int, tuple[int, int, int]] = {}
    for worm, schedule in enumerate(cohort.stimulus_schedules):
        rng = np.random.default_rng(seed + 1009 * worm)
        result[worm] = tuple(
            int(value) for value in rng.permutation(schedule.chemical_code_by_event)
        )
    return result


def is_done(state: RunState, phase: str, model_id: str, fold: int, seed: int) -> bool:
    return any(
        row.get("phase") == phase
        and row.get("model_id") == model_id
        and int(row.get("fold", -1)) == fold
        and int(row.get("seed", -1)) == seed
        and row.get("status") == "ok"
        for row in state.records
    )


def write_report(run_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "trial_metrics.csv")
    use = frame[(frame.status == "ok") & frame.phase.str.startswith("chemical_")]
    metrics = [
        "energy",
        "energy__stim_balanced",
        "energy__worm_chemical_balanced",
        "energy__chemical_butanone",
        "energy__chemical_pentanedione",
        "energy__chemical_nacl",
        "variogram",
    ]
    metrics = [metric for metric in metrics if metric in use]
    board = use.groupby(["model_id", "stimulus_encoding"])[metrics].agg(
        ["mean", "std", "count"]
    )
    board.columns = ["__".join(value) for value in board.columns]
    board = board.reset_index().sort_values(
        ["energy__worm_chemical_balanced__mean", "energy__mean"]
    )
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    board.to_csv(run_dir / "chemical_leaderboard.csv", index=False)
    lines = [
        "# Corrected chemical-identity stimulus encoding experiment",
        "",
        "Models are ranked atlas-blind by held-out proper scores. Active-event energy is first averaged within each worm×chemical cell and then equally across cells, so the OH15500 20-second final epoch cannot dominate.",
        "",
        "| Rank | Model | Worm×chemical energy | Overall energy | Butanone | Pentanedione | NaCl |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in board.itertuples():
        lines.append(
            f"| {int(row.rank)} | `{row.model_id}` | "
            f"{row.energy__worm_chemical_balanced__mean:.6f} | "
            f"{row.energy__mean:.6f} | {row.energy__chemical_butanone__mean:.6f} | "
            f"{row.energy__chemical_pentanedione__mean:.6f} | "
            f"{row.energy__chemical_nacl__mean:.6f} |"
        )
    lines.extend([
        "",
        "Chemical codes are categorical. The scalar arm is a sensitivity analysis only. The shuffled control preserves event times and one occurrence of each chemical per animal while permuting the true chemical-to-epoch assignment.",
        "",
        "No Cook, Randi, SBTG, receptor, or other atlas information entered model fitting or selection.",
        "",
    ])
    (run_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-run", type=Path, required=True)
    parser.add_argument(
        "--cohort-mode", choices=["oh16230_head", "pooled_resampled"], required=True
    )
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--lag", type=int, default=80)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--eval-rows", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(args.coverage, cohort_mode=args.cohort_mode)
    folds = _load_folds(cohort, args.evidence_run.resolve())
    shuffles = frozen_shuffles(cohort)
    phase = "chemical_smoke" if args.profile == "smoke" else "chemical_full_cv"
    folds_to_run = (0,) if args.profile == "smoke" else tuple(range(5))
    seeds = (1701,) if args.profile == "smoke" else (1701, 2903)
    epochs = args.epochs or (2 if args.profile == "smoke" else 50)
    patience = args.patience or (2 if args.profile == "smoke" else 9)
    eval_rows = args.eval_rows or (160 if args.profile == "smoke" else 4000)
    samples = args.samples or (4 if args.profile == "smoke" else 32)
    state = RunState(run_dir, time.monotonic() + args.hours * 3600)
    metrics_path = run_dir / "trial_metrics.csv"
    if args.resume and metrics_path.exists():
        state.records = pd.read_csv(metrics_path).replace({np.nan: None}).to_dict("records")

    manifest_path = run_dir / "manifest.json"
    expected = {
        "protocol": "corrected worm-specific chemical stimulus encoding",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_mode": args.cohort_mode,
        "profile": args.profile,
        "lag_frames": args.lag,
        "folds": list(folds_to_run),
        "seeds": list(seeds),
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_schema": cohort.stimulus_schema_dict(),
        "fold_assignments": str((args.evidence_run / "fold_assignments.csv").resolve()),
        "fold_assignments_sha256": sha256(args.evidence_run / "fold_assignments.csv"),
        "encodings": [spec.__dict__ for spec in ENCODINGS],
        "subject_shuffle_orders": {
            cohort.worm_ids[index]: list(order) for index, order in shuffles.items()
        },
        "selection_firewall": "no atlas/connectome/receptor data",
    }
    if manifest_path.exists():
        validate_resume_manifest(
            manifest_path,
            cohort.stimulus_schema_fingerprint,
            expected["fold_assignments_sha256"],
        )
    else:
        atomic_json(manifest_path, expected)

    for fold in folds_to_run:
        for spec in ENCODINGS:
            for seed in seeds:
                model = config_for(spec)
                if is_done(state, phase, model.model_id, fold, seed):
                    continue
                scaler, train, validation, test = _split_windows(
                    cohort,
                    folds,
                    fold,
                    args.lag,
                    stimulus_encoding=spec.encoding,
                    chemical_permutations=shuffles if spec.shuffled else None,
                )
                _neural_trial(
                    state=state,
                    phase=phase,
                    config=model,
                    cohort=cohort,
                    lag=args.lag,
                    fold=fold,
                    seed=seed,
                    train=train,
                    validation=validation,
                    test=test,
                    scaler=scaler,
                    device=args.device,
                    max_epochs=epochs,
                    patience=patience,
                    eval_rows=eval_rows,
                    n_samples=samples,
                    batch_size=args.batch_size,
                    trial_metadata={
                        "stimulus_encoding": spec.encoding,
                        "chemical_labels_subject_shuffled": spec.shuffled,
                        "subject_shuffle_orders": (
                            {
                                cohort.worm_ids[index]: list(order)
                                for index, order in shuffles.items()
                            }
                            if spec.shuffled else None
                        ),
                        "sensitivity_only": spec.sensitivity_only,
                        "cohort_mode": args.cohort_mode,
                    },
                )
    write_report(run_dir)
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
