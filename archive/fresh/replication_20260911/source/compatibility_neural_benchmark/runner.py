from __future__ import annotations

import argparse
import csv
import json
import time
import zlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from compatibility_neural_benchmark.core import (
    PHASES,
    GeneratorAdapter,
    RepairedResponseConfig,
    causal_fill,
    checkpoint_training_context,
    episode_cuts,
    estimate_repaired_responses,
    generate_path_bank,
    standardize_for_checkpoint,
    stimulus_for_trace,
)
from conditional_neural_benchmark.data import load_cohort


MODEL_FILE_IDS = {
    "flow": "tcn_delta_flow_matching",
    "mdn4": "tcn_delta_mdn4",
    "gaussian25": "tcn_delta_gaussian_dropout25",
}
SEMANTICS = (
    ("primary", 0.25, 0.25),
    ("unanchored", 0.00, 0.25),
    ("broad_clamp", 0.25, 0.50),
    ("loose_clamp", 0.25, 1.00),
    ("strong_anchor", 4.00, 0.25),
)
RESPONSE_KEYS = (
    "response_endpoint_mean",
    "response_cumulative_mean",
    "response_peak_mean",
    "response_event_probability",
    "response_endpoint_sd",
)


def _load_folds(path: Path, n_worms: int) -> np.ndarray:
    result = np.full(n_worms, -1, dtype=np.int64)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["worm_index"])] = int(row["outer_fold"])
    if np.any(result < 0):
        raise RuntimeError("fold assignment file is incomplete")
    return result


def _checkpoint_path(
    source_run: Path, checkpoint_phase: str, model_id: str, lag: int,
    fold: int, seed: int,
) -> Path:
    return (
        source_run
        / "checkpoints"
        / checkpoint_phase
        / f"{model_id}__L{lag}__f{fold}__s{seed}.pt"
    )


def _output_path(run_dir: Path, model_id: str, fold: int, seed: int, repair: int) -> Path:
    return run_dir / "responses" / f"{model_id}__B{repair}__f{fold}__s{seed}.npz"


def _seed(base: int, model_id: str, fold: int, seed: int, worm: int, phase: int, event: int) -> int:
    model_code = zlib.crc32(model_id.encode("utf-8"))
    return int(
        (base + model_code + 1_000_003 * fold + 1009 * seed + 9176 * worm + 131 * phase + event)
        % (2**31 - 1)
    )


def run_checkpoint(
    *,
    cohort,
    folds: np.ndarray,
    source_run: Path,
    run_dir: Path,
    model_id: str,
    checkpoint_phase: str,
    lag: int,
    fold: int,
    seed: int,
    config: RepairedResponseConfig,
    device: str,
    base_seed: int,
    resume: bool,
) -> dict:
    output = _output_path(run_dir, model_id, fold, seed, config.repair_frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    if resume and output.exists():
        with np.load(output, allow_pickle=False) as existing:
            if str(existing["status"].item()) == "complete":
                return {
                    "status": "skipped",
                    "output": str(output.relative_to(run_dir)),
                    "model_id": model_id,
                    "fold": fold,
                    "seed": seed,
                    "repair_frames": config.repair_frames,
                }
    checkpoint_path = _checkpoint_path(
        source_run, checkpoint_phase, model_id, lag, fold, seed
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    started = time.perf_counter()
    adapter = GeneratorAdapter.load(str(checkpoint_path), device=device)
    if tuple(cohort.neurons) != adapter.neurons:
        raise RuntimeError("checkpoint neuron order does not match the loaded cohort")
    projection, quantiles, thresholds = checkpoint_training_context(
        cohort, folds, fold, adapter.checkpoint, config
    )
    heldout = np.flatnonzero(folds == fold)
    n_worm = len(heldout)
    d = cohort.n_neurons
    h = len(config.horizon_frames)
    response_store = {
        key: np.zeros((n_worm, len(PHASES), d, h, d), dtype=np.float32)
        for key in RESPONSE_KEYS
    }
    diagnostic_store: dict[str, np.ndarray] = {}
    event_counts = np.zeros((n_worm, len(PHASES)), dtype=np.int16)
    rollout_energy = np.zeros((n_worm, len(PHASES), h), dtype=np.float32)
    rollout_rmse = np.zeros((n_worm, len(PHASES), h), dtype=np.float32)
    rollout_coverage90 = np.zeros((n_worm, len(PHASES), h), dtype=np.float32)
    semantic_score_sum = np.zeros((len(SEMANTICS), d, d), dtype=np.float64)
    semantic_valid_sum = np.zeros((len(SEMANTICS), d), dtype=np.float64)
    semantic_count = 0

    for worm_position, worm_idx in enumerate(heldout):
        trace = cohort.traces[int(worm_idx)]
        standardized = causal_fill(standardize_for_checkpoint(trace, adapter.checkpoint))
        stimulus = stimulus_for_trace(trace, cohort, int(worm_idx), adapter.checkpoint)
        for cut in episode_cuts(
            len(trace), cohort.stimulus_schedules[int(worm_idx)],
            config.source_window_frames,
        ):
            phase_index = PHASES.index(cut.phase)
            path_seed = _seed(
                base_seed, model_id, fold, seed, int(worm_idx), phase_index, cut.event
            )
            prefix, future, factual_prefix = generate_path_bank(
                adapter,
                standardized,
                stimulus,
                cut_time=cut.time,
                config=config,
                seed=path_seed,
            )
            horizon_index = np.asarray(config.horizon_frames, dtype=np.int64) - 1
            sampled_endpoint = future[:, horizon_index]
            factual_endpoint = standardized[
                cut.time + np.asarray(config.horizon_frames, dtype=np.int64)
            ]
            first = np.linalg.norm(
                sampled_endpoint - factual_endpoint[None], axis=2
            ).mean(axis=0)
            paired_n = len(sampled_endpoint) - len(sampled_endpoint) % 2
            paired = np.linalg.norm(
                sampled_endpoint[:paired_n:2] - sampled_endpoint[1:paired_n:2], axis=2
            ).mean(axis=0)
            rollout_energy[worm_position, phase_index] += first - 0.5 * paired
            rollout_rmse[worm_position, phase_index] += np.sqrt(
                np.square(sampled_endpoint.mean(axis=0) - factual_endpoint).mean(axis=1)
            )
            lo, hi = np.quantile(sampled_endpoint, [0.05, 0.95], axis=0)
            rollout_coverage90[worm_position, phase_index] += (
                (factual_endpoint >= lo) & (factual_endpoint <= hi)
            ).mean(axis=1)
            results_by_semantic: list[dict[str, np.ndarray]] = []
            for _, anchor_lambda, eps_fraction in SEMANTICS:
                results_by_semantic.append(
                    estimate_repaired_responses(
                        prefix,
                        future,
                        factual_prefix,
                        projection,
                        quantiles[cut.phase]["low"],
                        quantiles[cut.phase]["high"],
                        quantiles[cut.phase]["iqr"],
                        thresholds,
                        config,
                        anchor_lambda=anchor_lambda,
                        epsilon_iqr_fraction=eps_fraction,
                    )
                )
            primary = results_by_semantic[0]
            for key in RESPONSE_KEYS:
                response_store[key][worm_position, phase_index] += primary[key]
            for key, value in primary.items():
                if not key.startswith("diagnostic_"):
                    continue
                if key not in diagnostic_store:
                    diagnostic_store[key] = np.zeros(
                        (n_worm, len(PHASES), d), dtype=np.float32
                    )
                diagnostic_store[key][worm_position, phase_index] += np.asarray(
                    value, dtype=np.float32
                )
            event_counts[worm_position, phase_index] += 1
            for semantic_index, semantic_result in enumerate(results_by_semantic):
                response = semantic_result["response_cumulative_mean"]
                semantic_score_sum[semantic_index] += np.max(np.abs(response), axis=1)
                semantic_valid_sum[semantic_index] += semantic_result["diagnostic_valid"]
            semantic_count += 1

    divisor = np.maximum(event_counts, 1).astype(np.float32)
    for key in response_store:
        response_store[key] /= divisor[:, :, None, None, None]
    for key in diagnostic_store:
        diagnostic_store[key] /= divisor[:, :, None]
    rollout_energy /= divisor[:, :, None]
    rollout_rmse /= divisor[:, :, None]
    rollout_coverage90 /= divisor[:, :, None]
    semantic_score = semantic_score_sum / max(1, semantic_count)
    semantic_valid_rate = semantic_valid_sum / max(1, semantic_count)
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        output,
        status=np.asarray("complete"),
        model_id=np.asarray(model_id),
        fold=np.asarray(fold),
        seed=np.asarray(seed),
        monte_carlo_base_seed=np.asarray(base_seed),
        repair_frames=np.asarray(config.repair_frames),
        history_frames=np.asarray(config.history_frames),
        source_window_frames=np.asarray(config.source_window_frames),
        horizon_frames=np.asarray(config.horizon_frames, dtype=np.int16),
        phase_names=np.asarray(PHASES),
        semantic_names=np.asarray([row[0] for row in SEMANTICS]),
        semantic_anchor_lambda=np.asarray([row[1] for row in SEMANTICS], dtype=np.float32),
        semantic_epsilon_fraction=np.asarray([row[2] for row in SEMANTICS], dtype=np.float32),
        worm_indices=heldout.astype(np.int16),
        worm_ids=np.asarray([cohort.worm_ids[int(i)] for i in heldout]),
        neurons=np.asarray(cohort.neurons),
        event_counts=event_counts,
        semantic_score=semantic_score.astype(np.float32),
        semantic_valid_rate=semantic_valid_rate.astype(np.float32),
        wall_seconds=np.asarray(elapsed),
        rollout_energy=rollout_energy,
        rollout_rmse=rollout_rmse,
        rollout_coverage90=rollout_coverage90,
        **response_store,
        **diagnostic_store,
    )
    del adapter
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {
        "status": "ok",
        "output": str(output.relative_to(run_dir)),
        "model_id": model_id,
        "fold": fold,
        "seed": seed,
        "repair_frames": config.repair_frames,
        "heldout_worms": int(n_worm),
        "wall_seconds": elapsed,
    }


def _write_state(run_dir: Path, manifest: dict, records: list[dict]) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["records"] = records
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with (run_dir / "checkpoint_status.csv").open("w", newline="") as handle:
        keys = sorted(set().union(*(record.keys() for record in records))) if records else []
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_FILE_IDS), default=["flow", "mdn4"])
    parser.add_argument(
        "--model-ids", nargs="+",
        help="literal checkpoint model identifiers; overrides --models when supplied",
    )
    parser.add_argument("--checkpoint-phase", default="final_confirmation")
    parser.add_argument("--lag", type=int, default=80)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903, 4307])
    parser.add_argument("--repair-frames", type=int, default=4)
    parser.add_argument("--n-particles", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base-seed", type=int, default=20260826)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = RepairedResponseConfig(
        repair_frames=args.repair_frames,
        source_window_frames=min(4, args.repair_frames),
        n_particles=args.n_particles,
        min_ess=min(20.0, max(4.0, 0.10 * args.n_particles)),
    )
    config.validate()
    cohort = load_cohort()
    folds = _load_folds(source_run / "fold_assignments.csv", cohort.n_worms)
    model_ids = (
        list(args.model_ids)
        if args.model_ids
        else [MODEL_FILE_IDS[name] for name in args.models]
    )
    manifest = {
        "run_id": run_dir.name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run),
        "estimand": "compatibility-aware observational repaired path response",
        "claim_boundary": "model-relative observed history-to-future response; not a physical intervention or anatomical edge",
        "models": model_ids,
        "checkpoint_phase": args.checkpoint_phase,
        "checkpoint_lag": args.lag,
        "folds": args.folds,
        "seeds": args.seeds,
        "monte_carlo_base_seed": args.base_seed,
        "config": asdict(config),
        "semantics": [
            {"name": name, "anchor_lambda": lam, "epsilon_iqr_fraction": eps}
            for name, lam, eps in SEMANTICS
        ],
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "neurons": list(cohort.neurons),
        "fps": cohort.fps,
        "status": "running",
    }
    records: list[dict] = []
    _write_state(run_dir, manifest, records)
    for model_id in model_ids:
        for fold in args.folds:
            for seed in args.seeds:
                print(
                    f"RESPONSE_START model={model_id} fold={fold} seed={seed} B={config.repair_frames}",
                    flush=True,
                )
                try:
                    record = run_checkpoint(
                        cohort=cohort,
                        folds=folds,
                        source_run=source_run,
                        run_dir=run_dir,
                        model_id=model_id,
                        checkpoint_phase=args.checkpoint_phase,
                        lag=args.lag,
                        fold=fold,
                        seed=seed,
                        config=config,
                        device=args.device,
                        base_seed=args.base_seed,
                        resume=args.resume,
                    )
                except Exception as error:
                    record = {
                        "status": "failed",
                        "model_id": model_id,
                        "fold": fold,
                        "seed": seed,
                        "repair_frames": config.repair_frames,
                        "error": repr(error),
                    }
                    print(f"RESPONSE_FAILED {record}", flush=True)
                records.append(record)
                _write_state(run_dir, manifest, records)
                print(f"RESPONSE_DONE {record}", flush=True)
    manifest["status"] = "complete" if all(r["status"] in {"ok", "skipped"} for r in records) else "partial"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_state(run_dir, manifest, records)


if __name__ == "__main__":
    main()
