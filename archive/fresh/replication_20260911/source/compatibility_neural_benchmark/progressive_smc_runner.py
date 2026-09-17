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
    standardize_for_checkpoint,
    stimulus_for_trace,
)
from compatibility_neural_benchmark.progressive_smc import (
    progressive_smc_repaired_responses,
)
from conditional_neural_benchmark.data import load_cohort


DEFAULT_MODEL_ID = "tcn_delta_flow_matching"
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


def _seed(
    base: int, model_id: str, fold: int, model_seed: int, worm: int,
    phase: int, event: int,
) -> int:
    model_code = zlib.crc32(model_id.encode("utf-8"))
    return int(
        (
            base
            + model_code
            + 1_000_003 * fold
            + 1009 * model_seed
            + 9176 * worm
            + 131 * phase
            + event
        )
        % (2**31 - 1)
    )


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


def _output_path(
    run_dir: Path, model_id: str, fold: int, seed: int, repair: int
) -> Path:
    return (
        run_dir
        / "responses"
        / f"{model_id}__progressive_smc__B{repair}__f{fold}__s{seed}.npz"
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
    branch_factor: int,
    future_branch_factor: int,
    tempering_ess_fraction: float,
    max_tempering_resamples: int,
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
                    "fold": fold,
                    "seed": seed,
                }

    started = time.perf_counter()
    adapter = GeneratorAdapter.load(
        str(
            _checkpoint_path(
                source_run, checkpoint_phase, model_id, lag, fold, seed
            )
        ),
        device=device,
    )
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

    for worm_position, worm_idx in enumerate(heldout):
        trace = cohort.traces[int(worm_idx)]
        standardized = causal_fill(standardize_for_checkpoint(trace, adapter.checkpoint))
        stimulus = stimulus_for_trace(trace, cohort, int(worm_idx), adapter.checkpoint)
        for cut in episode_cuts(
            len(trace), cohort.stimulus_schedules[int(worm_idx)],
            config.source_window_frames,
        ):
            phase_index = PHASES.index(cut.phase)
            result = progressive_smc_repaired_responses(
                adapter,
                standardized,
                stimulus,
                cut_time=cut.time,
                projection=projection,
                source_low=quantiles[cut.phase]["low"],
                source_high=quantiles[cut.phase]["high"],
                source_iqr=quantiles[cut.phase]["iqr"],
                thresholds=thresholds,
                config=config,
                seed=_seed(
                    base_seed,
                    model_id,
                    fold,
                    seed,
                    int(worm_idx),
                    phase_index,
                    cut.event,
                ),
                branch_factor=branch_factor,
                future_branch_factor=future_branch_factor,
                tempering_ess_fraction=tempering_ess_fraction,
                max_tempering_resamples=max_tempering_resamples,
            )
            for key in RESPONSE_KEYS:
                response_store[key][worm_position, phase_index] += result[key]
            for key, value in result.items():
                if not key.startswith("diagnostic_"):
                    continue
                array = np.asarray(value, dtype=np.float32)
                if key not in diagnostic_store:
                    diagnostic_store[key] = np.zeros(
                        (n_worm, len(PHASES), *array.shape), dtype=np.float32
                    )
                diagnostic_store[key][worm_position, phase_index] += array
            event_counts[worm_position, phase_index] += 1
            print(
                f"PROGRESSIVE_SMC_EPISODE fold={fold} seed={seed} "
                f"worm={int(worm_idx)} phase={cut.phase} event={cut.event}",
                flush=True,
            )

    divisor = np.maximum(event_counts, 1).astype(np.float32)
    for key in response_store:
        response_store[key] /= divisor[:, :, None, None, None]
    for key, values in diagnostic_store.items():
        extra = (None,) * (values.ndim - 2)
        diagnostic_store[key] /= divisor[(slice(None), slice(None), *extra)]

    elapsed = time.perf_counter() - started
    np.savez_compressed(
        output,
        status=np.asarray("complete"),
        estimator=np.asarray("progressive_branching_bridge_smc"),
        contrast_coupling=np.asarray("paired_common_random_numbers"),
        model_id=np.asarray(model_id),
        fold=np.asarray(fold),
        seed=np.asarray(seed),
        monte_carlo_base_seed=np.asarray(base_seed),
        repair_frames=np.asarray(config.repair_frames),
        history_frames=np.asarray(config.history_frames),
        source_window_frames=np.asarray(config.source_window_frames),
        horizon_frames=np.asarray(config.horizon_frames, dtype=np.int16),
        phase_names=np.asarray(PHASES),
        worm_indices=heldout.astype(np.int16),
        worm_ids=np.asarray([cohort.worm_ids[int(i)] for i in heldout]),
        neurons=np.asarray(cohort.neurons),
        event_counts=event_counts,
        wall_seconds=np.asarray(elapsed),
        branch_factor=np.asarray(branch_factor),
        future_branch_factor=np.asarray(future_branch_factor),
        tempering_ess_fraction=np.asarray(tempering_ess_fraction),
        max_tempering_resamples=np.asarray(max_tempering_resamples),
        **response_store,
        **diagnostic_store,
    )
    del adapter
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {
        "status": "ok",
        "output": str(output.relative_to(run_dir)),
        "fold": fold,
        "seed": seed,
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
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--checkpoint-phase", default="final_confirmation")
    parser.add_argument("--lag", type=int, default=80)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701])
    parser.add_argument("--repair-frames", type=int, default=4)
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 32, 40]
    )
    parser.add_argument("--n-particles", type=int, default=128)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--future-branch-factor", type=int, default=2)
    parser.add_argument("--tempering-ess-fraction", type=float, default=0.65)
    parser.add_argument("--max-tempering-resamples", type=int, default=8)
    parser.add_argument("--sampling-chunk-size", type=int, default=1024)
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
        horizon_frames=tuple(sorted(set(args.horizons))),
        n_particles=args.n_particles,
        min_ess=min(20.0, max(4.0, 0.10 * args.n_particles)),
        sampling_chunk_size=args.sampling_chunk_size,
    )
    config.validate()
    cohort = load_cohort()
    folds = _load_folds(source_run / "fold_assignments.csv", cohort.n_worms)
    manifest = {
        "run_id": run_dir.name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run),
        "estimator": (
            "sampler-only progressive SMC with branched repair proposals, a persistence "
            "look-ahead source bridge, adaptive ESS tempering, systematic pruning, full "
            "root ancestry, and multiple free future descendants"
        ),
        "terminal_estimand_invariance": (
            "the final bridge potential equals the primary terminal Gaussian source clamp "
            "and factual anchor exactly"
        ),
        "contrast_coupling": (
            "paired common random numbers and paired systematic offsets for low/high targets"
        ),
        "estimand": "compatibility-aware observational repaired path response",
        "claim_boundary": (
            "model-relative observed history-to-future response; not a physical intervention "
            "or anatomical edge"
        ),
        "external_atlas_access": "prohibited during this frozen-generator estimator benchmark",
        "model": args.model_id,
        "checkpoint_phase": args.checkpoint_phase,
        "checkpoint_lag": args.lag,
        "folds": args.folds,
        "seeds": args.seeds,
        "monte_carlo_base_seed": args.base_seed,
        "config": asdict(config),
        "branch_factor": args.branch_factor,
        "future_branch_factor": args.future_branch_factor,
        "tempering_ess_fraction": args.tempering_ess_fraction,
        "max_tempering_resamples": args.max_tempering_resamples,
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "neurons": list(cohort.neurons),
        "fps": cohort.fps,
        "status": "running",
    }
    records: list[dict] = []
    _write_state(run_dir, manifest, records)
    for fold in args.folds:
        for seed in args.seeds:
            print(f"PROGRESSIVE_SMC_START fold={fold} seed={seed}", flush=True)
            try:
                record = run_checkpoint(
                    cohort=cohort,
                    folds=folds,
                    source_run=source_run,
                    run_dir=run_dir,
                    model_id=args.model_id,
                    checkpoint_phase=args.checkpoint_phase,
                    lag=args.lag,
                    fold=fold,
                    seed=seed,
                    config=config,
                    device=args.device,
                    base_seed=args.base_seed,
                    branch_factor=args.branch_factor,
                    future_branch_factor=args.future_branch_factor,
                    tempering_ess_fraction=args.tempering_ess_fraction,
                    max_tempering_resamples=args.max_tempering_resamples,
                    resume=args.resume,
                )
            except Exception as error:
                record = {
                    "status": "failed",
                    "fold": fold,
                    "seed": seed,
                    "error": repr(error),
                }
                print(f"PROGRESSIVE_SMC_FAILED {record}", flush=True)
            records.append(record)
            _write_state(run_dir, manifest, records)
            print(f"PROGRESSIVE_SMC_DONE {record}", flush=True)
    manifest["status"] = (
        "complete"
        if all(record["status"] in {"ok", "skipped"} for record in records)
        else "partial"
    )
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_state(run_dir, manifest, records)


if __name__ == "__main__":
    main()
