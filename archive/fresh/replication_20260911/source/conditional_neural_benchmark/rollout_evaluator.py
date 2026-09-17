from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark.core import (
    GeneratorAdapter,
    causal_fill,
    stimulus_for_trace,
)
from conditional_neural_benchmark.data import (
    choose_evaluation_indices,
    load_cohort,
)
from conditional_neural_benchmark.metrics import metric_rows, summarize_metrics
from conditional_neural_benchmark.runner import _split_windows


CHECKPOINT_PATTERN = re.compile(r"(?P<model>.+)__L(?P<lag>\d+)__f(?P<fold>\d+)__s(?P<seed>\d+)\.pt$")


def _load_folds(cohort, evidence_run: Path) -> np.ndarray:
    frame = pd.read_csv(evidence_run / "fold_assignments.csv")
    mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
    return np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)


def _standardized_trace(trace: np.ndarray, checkpoint: dict) -> np.ndarray:
    mean = np.asarray(checkpoint["scaler"]["mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler"]["scale"], dtype=np.float32)
    return ((np.asarray(trace, dtype=np.float32) - mean) / scale).astype(np.float32)


def _contexts(
    cohort, folds: np.ndarray, fold: int, lag: int, checkpoint: dict,
    max_horizon: int, eval_rows: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata = checkpoint.get("trial_metadata", {})
    encoding = str(metadata.get("stimulus_encoding", "binary_any_stimulus"))
    if bool(metadata.get("presentation_labels_subject_shuffled", False)):
        raise ValueError(
            "rollout evaluation of shuffled labels requires the frozen subject permutations"
        )
    chemical_permutations = None
    if bool(metadata.get("chemical_labels_subject_shuffled", False)):
        frozen = metadata.get("subject_shuffle_orders")
        if not isinstance(frozen, dict):
            raise ValueError("shuffled chemical checkpoint lacks frozen subject permutations")
        chemical_permutations = {}
        for worm, worm_id in enumerate(cohort.worm_ids):
            if worm_id not in frozen:
                raise ValueError(f"shuffled chemical checkpoint lacks order for {worm_id}")
            chemical_permutations[worm] = tuple(int(value) for value in frozen[worm_id])
    _, _, _, test = _split_windows(
        cohort, folds, fold, lag, stimulus_encoding=encoding,
        chemical_permutations=chemical_permutations,
    )
    valid = test.time + max_horizon - 1 < np.asarray(
        [len(cohort.traces[int(worm)]) for worm in test.worm]
    )
    for position in np.flatnonzero(valid):
        worm = int(test.worm[position])
        start = int(test.time[position])
        future = cohort.traces[worm][start : start + max_horizon]
        if not np.isfinite(future).all():
            valid[position] = False
    candidates = np.flatnonzero(valid)
    picked_local = choose_evaluation_indices(test.stratum[candidates], eval_rows, seed)
    picked = candidates[picked_local]
    return (
        test.history[picked, :, :cohort.n_neurons].astype(np.float32),
        test.history[picked, :, cohort.n_neurons:].astype(np.float32),
        test.worm[picked].astype(int),
        test.time[picked].astype(int),
        test.stratum[picked].astype(str),
    )


def evaluate_checkpoint(
    path: Path, cohort, folds: np.ndarray, horizons: tuple[int, ...],
    eval_rows: int, n_samples: int, device: str,
) -> list[dict]:
    match = CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"unrecognized checkpoint name: {path.name}")
    model_id = match.group("model")
    lag, fold, model_seed = (int(match.group(key)) for key in ("lag", "fold", "seed"))
    adapter = GeneratorAdapter.load(str(path), device=device)
    if adapter.neurons != tuple(cohort.neurons):
        raise RuntimeError("checkpoint and cohort neuron order mismatch")
    base_seed = 700_001 + 101 * fold + model_seed
    neural, stimulus, worms, times, strata = _contexts(
        cohort, folds, fold, lag, adapter.checkpoint, max(horizons),
        eval_rows, base_seed,
    )
    n_context = len(neural)
    neural_paths = np.repeat(neural, n_samples, axis=0)
    stimulus_paths = np.repeat(stimulus, n_samples, axis=0)
    repeated_times = np.repeat(times, n_samples)
    repeated_worms = np.repeat(worms, n_samples)
    unique_worms = sorted(set(worms.tolist()))
    stimulus_by_worm = {
        worm: stimulus_for_trace(
            cohort.traces[worm], cohort, worm, adapter.checkpoint
        )
        for worm in unique_worms
    }
    standardized_by_worm = {
        worm: _standardized_trace(cohort.traces[worm], adapter.checkpoint)
        for worm in unique_worms
    }
    draws_by_horizon: dict[int, np.ndarray] = {}
    for step in range(1, max(horizons) + 1):
        neural_tensor = torch.as_tensor(
            neural_paths, dtype=torch.float32, device=adapter.device
        )
        stimulus_tensor = torch.as_tensor(
            stimulus_paths, dtype=torch.float32, device=adapter.device
        )
        draw = adapter.sample_standardized_next(
            neural_tensor, stimulus_tensor, seed=base_seed + 10_007 * step
        ).detach().cpu().numpy()
        if step in horizons:
            draws_by_horizon[step] = draw.reshape(
                n_context, n_samples, cohort.n_neurons
            ).copy()
        next_stimulus = np.asarray(
            [
                stimulus_by_worm[int(worm)][time + step - 1]
                for worm, time in zip(repeated_worms, repeated_times)
            ],
            dtype=np.float32,
        )
        neural_paths = np.concatenate([neural_paths[:, 1:], draw[:, None]], axis=1)
        stimulus_paths = np.concatenate(
            [stimulus_paths[:, 1:], next_stimulus[:, None, :]], axis=1
        )

    records = []
    for horizon in horizons:
        target = np.stack(
            [
                standardized_by_worm[int(worm)][int(time) + horizon - 1]
                for worm, time in zip(worms, times)
            ]
        )
        draws = draws_by_horizon[horizon]
        row_metrics = metric_rows(draws, target, base_seed + horizon)
        summary = summarize_metrics(
            row_metrics, strata, None, cohort.n_neurons, population_strata=strata
        )
        records.append(
            {
                "checkpoint": str(path),
                "model_id": model_id,
                "fold": fold,
                "seed": model_seed,
                "lag": lag,
                "rollout_horizon_frames": horizon,
                "rollout_horizon_seconds": horizon / cohort.fps,
                "n_samples": n_samples,
                "finite_fraction": float(np.isfinite(draws).mean()),
                "explosive_fraction_abs_z_gt_10": float(np.mean(np.abs(draws) > 10)),
                "mean_abs_z": float(np.mean(np.abs(draws))),
                **summary,
            }
        )
    del adapter
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_report(output: Path, aggregate: pd.DataFrame) -> None:
    final_horizon = aggregate.rollout_horizon_frames.max()
    final = aggregate[aggregate.rollout_horizon_frames == final_horizon].sort_values(
        ["energy__stim_balanced__mean", "variogram__mean"]
    )
    winner = final.iloc[0]
    lines = [
        "# Frozen multi-step conditional rollout evaluation",
        "",
        "## Result",
        "",
        f"At the longest tested horizon ({winner.rollout_horizon_seconds:g} s), "
        f"**{winner.model_id}** has the best stimulus-balanced energy score "
        f"(**{winner.energy__stim_balanced__mean:.6f}**) among the confirmed candidates.",
        "",
        "| Model | Horizon (s) | Energy | Balanced energy | Variogram | RMSE | |z|>10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate.sort_values(["rollout_horizon_frames", "model_id"]).itertuples():
        lines.append(
            f"| {row.model_id} | {row.rollout_horizon_seconds:g} | "
            f"{row.energy__mean:.4f} | {row.energy__stim_balanced__mean:.4f} | "
            f"{row.variogram__mean:.4f} | {row.rmse__mean:.4f} | "
            f"{row.explosive_fraction_abs_z_gt_10__mean:.4f} |"
        )
    lines.extend(
        [
            "",
            "All paths are free-running ancestral samples after the observed L=80 context, with the observed future causal stimulus schedule supplied exogenously. No anatomical or functional atlas enters this evaluation.",
            "",
            "These are predictive rollouts of observed activity, not causal interventions or anatomical effects.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament-run", type=Path, required=True)
    parser.add_argument("--evidence-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", default="focused_confirmation")
    parser.add_argument(
        "--model-ids", nargs="+",
        help="optional subset of checkpoint model identifiers",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16, 40])
    parser.add_argument("--eval-rows", type=int, default=160)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    tournament = args.tournament_run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tournament_manifest = json.loads((tournament / "manifest.json").read_text())
    cohort = load_cohort(
        cohort_mode=str(tournament_manifest.get("cohort_mode", "pooled_resampled"))
    )
    if tournament_manifest.get("stimulus_schema_fingerprint") != cohort.stimulus_schema_fingerprint:
        raise RuntimeError("tournament and cohort stimulus-schema fingerprints differ")
    folds = _load_folds(cohort, args.evidence_run.resolve())
    checkpoints = sorted((tournament / "checkpoints" / args.phase).glob("*.pt"))
    if args.model_ids:
        allowed = set(args.model_ids)
        checkpoints = [
            path for path in checkpoints
            if (CHECKPOINT_PATTERN.match(path.name) is not None)
            and CHECKPOINT_PATTERN.match(path.name).group("model") in allowed
        ]
    if not checkpoints:
        raise RuntimeError(f"no checkpoints found for phase {args.phase}")
    records_path = output / "rollout_metrics_by_checkpoint.csv"
    records = (
        pd.read_csv(records_path).replace({np.nan: None}).to_dict("records")
        if args.resume and records_path.exists()
        else []
    )
    completed = {
        (str(Path(row["checkpoint"]).resolve()), int(row["rollout_horizon_frames"]))
        for row in records
    }
    horizons = tuple(sorted(set(args.horizons)))
    for checkpoint in checkpoints:
        resolved = str(checkpoint.resolve())
        if all((resolved, horizon) in completed for horizon in horizons):
            print(f"ROLLOUT_SKIP {checkpoint.name}", flush=True)
            continue
        print(f"ROLLOUT_START {checkpoint.name}", flush=True)
        fresh = evaluate_checkpoint(
            checkpoint.resolve(), cohort, folds, horizons, args.eval_rows,
            args.samples, args.device,
        )
        records = [
            row for row in records
            if str(Path(row["checkpoint"]).resolve()) != resolved
        ] + fresh
        pd.DataFrame(records).to_csv(records_path, index=False)
        print(f"ROLLOUT_DONE {checkpoint.name}", flush=True)
    frame = pd.DataFrame(records)
    metrics = [
        "energy", "energy__stim_balanced", "variogram", "rmse", "coverage90",
        "sharpness90", "finite_fraction", "explosive_fraction_abs_z_gt_10",
        "mean_abs_z",
    ]
    aggregate = (
        frame.groupby(["model_id", "rollout_horizon_frames", "rollout_horizon_seconds"])[metrics]
        .agg(["mean", "std", "count"])
    )
    aggregate.columns = ["__".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(output / "rollout_metrics_aggregate.csv", index=False)
    write_report(output, aggregate)
    _write_json(
        output / "protocol.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tournament_run": str(tournament),
            "phase": args.phase,
            "horizons_frames": horizons,
            "eval_rows_per_checkpoint": args.eval_rows,
            "samples_per_context": args.samples,
            "torch_threads": args.threads,
            "conditioning": "observed L-frame initialization plus observed exogenous future stimulus schedule",
            "selection_inputs": "none; frozen follow-up",
            "external_atlas_access": "none",
            "checkpoint_sha256": {
                str(path.resolve()): _sha256(path) for path in checkpoints
            },
            "claim_boundary": "predictive observed-activity rollout; not causal or anatomical",
        },
    )


if __name__ == "__main__":
    main()
