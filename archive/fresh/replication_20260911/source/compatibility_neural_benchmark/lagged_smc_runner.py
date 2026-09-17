from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark.core import (
    PHASES,
    GeneratorAdapter,
    RepairedResponseConfig,
    causal_fill,
    episode_cuts,
    fit_anchor_projection,
    smc_repaired_responses,
    standardize_for_checkpoint,
    stimulus_for_trace,
)
from conditional_neural_benchmark.data import load_cohort


PRIMARY_PHASES = ("baseline", "onset")
RESPONSE_KEYS = (
    "response_endpoint_mean",
    "response_cumulative_mean",
    "response_peak_mean",
    "response_event_probability",
    "response_endpoint_sd",
)


def _checkpoint_path(
    source_run: Path, model_id: str, lag: int, fold: int, seed: int
) -> Path:
    return (
        source_run
        / "checkpoints"
        / "winner_full_cv"
        / f"{model_id}__L{lag}__f{fold}__s{seed}.pt"
    )


def _output_path(
    output: Path, model_id: str, source_lag: int, fold: int, seed: int, particles: int
) -> Path:
    return (
        output
        / "responses"
        / f"{model_id}__lagged_smc__ell{source_lag}__N{particles}__f{fold}__s{seed}.npz"
    )


def _load_folds(path: Path, cohort) -> np.ndarray:
    frame = pd.read_csv(path)
    mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
    folds = np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)
    if set(folds.tolist()) != set(range(5)):
        raise RuntimeError("expected immutable five-fold assignment")
    return folds


def _training_context(
    cohort,
    folds: np.ndarray,
    fold: int,
    checkpoint: dict,
    config: RepairedResponseConfig,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], np.ndarray]:
    training = np.flatnonzero(folds != fold)
    traces = [
        causal_fill(standardize_for_checkpoint(cohort.traces[int(index)], checkpoint))
        for index in training
    ]
    projection = fit_anchor_projection(traces, config.anchor_rank)
    buckets: dict[str, list[np.ndarray]] = {phase: [] for phase in PRIMARY_PHASES}
    for trace, worm in zip(traces, training):
        for cut in episode_cuts(
            len(trace), cohort.stimulus_schedules[int(worm)],
            config.source_window_frames,
        ):
            if cut.phase not in buckets:
                continue
            source_hi = cut.time - config.source_lag_frames + 1
            source_lo = source_hi - config.source_window_frames
            if source_lo >= 0:
                buckets[cut.phase].append(trace[source_lo:source_hi].mean(axis=0))
    quantiles: dict[str, dict[str, np.ndarray]] = {}
    for phase, values in buckets.items():
        if not values:
            raise RuntimeError(f"no lag-aligned source statistics for {phase}")
        array = np.asarray(values, dtype=np.float32)
        q25, q75 = np.quantile(array, [0.25, 0.75], axis=0)
        quantiles[phase] = {
            "low": q25.astype(np.float32),
            "high": q75.astype(np.float32),
            "iqr": np.maximum(q75 - q25, 0.20).astype(np.float32),
        }
    pooled = np.concatenate(traces, axis=0)
    thresholds = np.quantile(pooled, 0.90, axis=0).astype(np.float32)
    return projection, quantiles, thresholds


def _seed(
    base: int,
    model_id: str,
    fold: int,
    seed: int,
    worm: int,
    phase: int,
    event: int,
    source_lag: int,
) -> int:
    code = int(hashlib.sha256(model_id.encode()).hexdigest()[:8], 16)
    return int(
        (
            base
            + code
            + 1_000_003 * fold
            + 1009 * seed
            + 9176 * worm
            + 131 * phase
            + 17 * event
            + 53 * source_lag
        )
        % (2**31 - 1)
    )


def run_one(
    *,
    cohort,
    folds: np.ndarray,
    source_run: Path,
    output: Path,
    model_id: str,
    history_lag: int,
    fold: int,
    seed: int,
    source_lag: int,
    particles: int,
    horizons: tuple[int, ...],
    device: str,
    base_seed: int,
    min_ess: float,
) -> dict:
    path = _output_path(output, model_id, source_lag, fold, seed, particles)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if str(existing["status"].item()) == "complete":
                return {"status": "skipped", "output": str(path), "wall_seconds": 0.0}
    checkpoint_path = _checkpoint_path(source_run, model_id, history_lag, fold, seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    started = time.perf_counter()
    adapter = GeneratorAdapter.load(str(checkpoint_path), device=device)
    if tuple(cohort.neurons) != adapter.neurons:
        raise RuntimeError("checkpoint neuron order mismatch")
    config = RepairedResponseConfig(
        history_frames=history_lag,
        repair_frames=source_lag + 4,
        source_window_frames=4,
        source_lag_frames=source_lag,
        horizon_frames=horizons,
        n_particles=particles,
        min_ess=min_ess,
        resample_ess_fraction=0.50,
        sampling_chunk_size=1024,
    )
    projection, quantiles, thresholds = _training_context(
        cohort, folds, fold, adapter.checkpoint, config
    )
    heldout = np.flatnonzero(folds == fold)
    d = cohort.n_neurons
    h = len(horizons)
    phase_names = np.asarray(PRIMARY_PHASES)
    response = {
        key: np.full((len(heldout), 2, 3, d, h, d), np.nan, dtype=np.float32)
        for key in RESPONSE_KEYS
    }
    diagnostic_names: list[str] | None = None
    diagnostic: dict[str, np.ndarray] = {}
    cut_times = np.full((len(heldout), 2, 3), -1, dtype=np.int32)
    for worm_position, worm in enumerate(heldout):
        standardized = causal_fill(
            standardize_for_checkpoint(cohort.traces[int(worm)], adapter.checkpoint)
        )
        stimulus = stimulus_for_trace(
            cohort.traces[int(worm)], cohort, int(worm), adapter.checkpoint
        )
        for cut in episode_cuts(
            len(cohort.traces[int(worm)]), cohort.stimulus_schedules[int(worm)],
            config.source_window_frames,
        ):
            if cut.phase not in PRIMARY_PHASES:
                continue
            phase_index = PRIMARY_PHASES.index(cut.phase)
            draw_seed = _seed(
                base_seed,
                model_id,
                fold,
                seed,
                int(worm),
                phase_index,
                cut.event,
                source_lag,
            )
            result = smc_repaired_responses(
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
                seed=draw_seed,
            )
            cut_times[worm_position, phase_index, cut.event] = cut.time
            for key in RESPONSE_KEYS:
                response[key][worm_position, phase_index, cut.event] = result[key]
            current_names = sorted(
                key.removeprefix("diagnostic_")
                for key in result
                if key.startswith("diagnostic_")
                and np.asarray(result[key]).ndim == 1
            )
            if diagnostic_names is None:
                diagnostic_names = current_names
                diagnostic = {
                    name: np.full(
                        (len(heldout), 2, 3, d), np.nan, dtype=np.float32
                    )
                    for name in current_names
                }
            if current_names != diagnostic_names:
                raise RuntimeError("diagnostic schema changed within the run")
            for name in diagnostic_names:
                diagnostic[name][worm_position, phase_index, cut.event] = result[
                    f"diagnostic_{name}"
                ]
            if cut.phase == "onset" and cut.event == 2:
                print(
                    "LAGGED_SMC_WORM_DONE "
                    f"ell={source_lag} fold={fold} worm={int(worm)}",
                    flush=True,
                )
    if np.any(cut_times < 0):
        raise RuntimeError("one or more prespecified episode cuts were not evaluated")
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        path,
        status=np.asarray("complete"),
        created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        model_id=np.asarray(model_id),
        checkpoint=np.asarray(str(checkpoint_path)),
        fold=np.asarray(fold),
        seed=np.asarray(seed),
        history_frames=np.asarray(history_lag),
        source_lag_frames=np.asarray(source_lag),
        source_lag_seconds=np.asarray(source_lag / cohort.fps),
        repair_frames=np.asarray(config.repair_frames),
        source_window_frames=np.asarray(config.source_window_frames),
        n_particles=np.asarray(particles),
        horizon_frames=np.asarray(horizons, dtype=np.int16),
        phase_names=phase_names,
        worm_indices=heldout.astype(np.int16),
        worm_ids=np.asarray([cohort.worm_ids[int(index)] for index in heldout]),
        neurons=np.asarray(cohort.neurons),
        cut_times=cut_times,
        wall_seconds=np.asarray(elapsed),
        **response,
        **{f"diagnostic_{name}": value for name, value in diagnostic.items()},
    )
    del adapter
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"status": "ok", "output": str(path), "wall_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path(
            "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827"
        ),
    )
    parser.add_argument("--model-id", default="flow_wide128_dropout10")
    parser.add_argument("--history-lag", type=int, default=80)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701])
    parser.add_argument("--source-lags", nargs="+", type=int, default=[0, 4, 8, 16])
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-seed", type=int, default=20260827)
    parser.add_argument("--min-ess", type=float, default=12.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    folds = _load_folds(args.source_run / "fold_assignments.csv", cohort)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "literal bootstrap SMC with ESS-triggered resampling and temporal cut",
        "model_id": args.model_id,
        "source_run": str(args.source_run.resolve()),
        "history_frames": args.history_lag,
        "source_lag_frames": args.source_lags,
        "source_lag_seconds": [lag / cohort.fps for lag in args.source_lags],
        "repair_rule": "source_window + source_lag; source clamp weight carried to cut",
        "source_window_frames": 4,
        "horizon_frames": args.horizons,
        "primary_phases": list(PRIMARY_PHASES),
        "particles": args.particles,
        "resample_ess_fraction": 0.50,
        "min_ess": args.min_ess,
        "folds": args.folds,
        "seeds": args.seeds,
        "atlas_firewall": "external references not used in response estimation",
        "claim_boundary": "model-relative observed-law response, not physical intervention",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for source_lag in args.source_lags:
        for fold in args.folds:
            for seed in args.seeds:
                print(
                    f"LAGGED_SMC_START ell={source_lag} fold={fold} seed={seed}",
                    flush=True,
                )
                try:
                    result = run_one(
                        cohort=cohort,
                        folds=folds,
                        source_run=args.source_run.resolve(),
                        output=output,
                        model_id=args.model_id,
                        history_lag=args.history_lag,
                        fold=fold,
                        seed=seed,
                        source_lag=source_lag,
                        particles=args.particles,
                        horizons=tuple(sorted(set(args.horizons))),
                        device=args.device,
                        base_seed=args.base_seed,
                        min_ess=args.min_ess,
                    )
                except Exception as error:
                    result = {"status": "failed", "error": repr(error), "wall_seconds": 0.0}
                record = {
                    "source_lag_frames": source_lag,
                    "source_lag_seconds": source_lag / cohort.fps,
                    "fold": fold,
                    "seed": seed,
                    **result,
                }
                records.append(record)
                pd.DataFrame(records).to_csv(output / "run_status.csv", index=False)
                print(
                    f"LAGGED_SMC_DONE ell={source_lag} fold={fold} seed={seed} "
                    f"status={result['status']} seconds={result.get('wall_seconds', 0):.1f}",
                    flush=True,
                )
    failures = [row for row in records if row["status"] == "failed"]
    validation = {
        "status": "pass" if not failures else "failed",
        "expected_runs": len(args.source_lags) * len(args.folds) * len(args.seeds),
        "completed_or_skipped": sum(
            row["status"] in {"ok", "skipped"} for row in records
        ),
        "failed": len(failures),
        "failures": failures,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
