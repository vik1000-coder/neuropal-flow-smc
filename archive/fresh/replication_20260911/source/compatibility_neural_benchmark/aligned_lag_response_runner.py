from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark.core import (
    GeneratorAdapter,
    RepairedResponseConfig,
    causal_fill,
    estimate_repaired_responses,
    fit_anchor_projection,
    generate_path_bank,
    standardize_for_checkpoint,
    stimulus_for_trace,
)
from compatibility_neural_benchmark.progressive_smc import (
    progressive_smc_repaired_responses,
)
from conditional_neural_benchmark.data import load_cohort


PHASES = ("quiet", "onset_aligned")
RESPONSE_KEYS = (
    "response_endpoint_mean",
    "response_cumulative_mean",
    "response_peak_mean",
    "response_event_probability",
    "response_endpoint_sd",
)


def _load_folds(path: Path, cohort) -> np.ndarray:
    frame = pd.read_csv(path)
    mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
    missing = [worm for worm in cohort.worm_ids if worm not in mapping]
    if missing:
        raise RuntimeError(f"fold evidence is missing worms: {missing}")
    return np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)


def aligned_cuts(
    n_frames: int, schedule, delay_frames: int, boundary_shift_frames: int = 0
) -> list[tuple[str, int, int, int, str]]:
    result: list[tuple[str, int, int, int, str]] = []
    fps = schedule.analysis_fps
    quiet_shift = int(round(15.0 * fps))
    for event, (start_seconds, _) in enumerate(schedule.event_intervals_seconds):
        onset = int(round(start_seconds * fps)) + int(boundary_shift_frames)
        for phase, cut in (
            ("quiet", onset - quiet_shift + delay_frames),
            ("onset_aligned", onset + delay_frames),
        ):
            if 0 <= cut < n_frames:
                result.append((
                    phase,
                    event,
                    cut,
                    schedule.chemical_code_by_event[event],
                    schedule.chemical_name_by_event[event],
                ))
    return result


def training_context(
    cohort, folds: np.ndarray, fold: int, checkpoint: dict,
    config: RepairedResponseConfig, delays: tuple[int, ...], boundary_shift_frames: int,
) -> tuple[np.ndarray, dict[tuple[str, int], dict[str, np.ndarray]], np.ndarray]:
    training = np.flatnonzero(folds != fold)
    traces = [
        causal_fill(standardize_for_checkpoint(cohort.traces[int(index)], checkpoint))
        for index in training
    ]
    projection = fit_anchor_projection(traces, config.anchor_rank)
    buckets: dict[tuple[str, int], list[np.ndarray]] = {
        (phase, delay): [] for phase in PHASES for delay in delays
    }
    for trace, worm in zip(traces, training):
        schedule = cohort.stimulus_schedules[int(worm)]
        for delay in delays:
            for phase, _, cut, _, _ in aligned_cuts(
                len(trace), schedule, delay, boundary_shift_frames
            ):
                lo = cut - config.source_window_frames + 1
                if lo >= 0:
                    buckets[(phase, delay)].append(trace[lo : cut + 1].mean(axis=0))
    quantiles: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for key, values in buckets.items():
        if not values:
            raise RuntimeError(f"no training source summaries for {key}")
        array = np.asarray(values, dtype=np.float32)
        q25, q75 = np.quantile(array, [0.25, 0.75], axis=0)
        quantiles[key] = {
            "low": q25.astype(np.float32),
            "high": q75.astype(np.float32),
            "iqr": np.maximum(q75 - q25, 0.20).astype(np.float32),
        }
    pooled = np.concatenate(traces, axis=0)
    thresholds = np.quantile(pooled, 0.90, axis=0).astype(np.float32)
    return projection, quantiles, thresholds


def _seed(
    base: int, method: str, model_id: str, fold: int, model_seed: int,
    worm: int, phase: int, event: int, delay: int,
) -> int:
    code = int(hashlib.sha256(f"{method}:{model_id}".encode()).hexdigest()[:8], 16)
    return int(
        (base + code + 1_000_003 * fold + 1009 * model_seed + 9176 * worm
         + 131 * phase + 17 * event + 53 * delay) % (2**31 - 1)
    )


def _checkpoint_path(
    source_run: Path, phase: str, model_id: str, lag: int, fold: int, seed: int
) -> Path:
    return (
        source_run / "checkpoints" / phase
        / f"{model_id}__L{lag}__f{fold}__s{seed}.pt"
    )


def run_checkpoint(
    *, cohort, folds: np.ndarray, source_run: Path, output_dir: Path,
    method: str, checkpoint_phase: str, model_id: str, history_lag: int,
    fold: int, model_seed: int, delays: tuple[int, ...],
    horizons: tuple[int, ...], particles: int, device: str, base_seed: int,
    branch_factor: int, future_branch_factor: int,
    tempering_ess_fraction: float, boundary_shift_frames: int, resume: bool,
) -> dict:
    path = (
        output_dir / "responses"
        / f"{model_id}__{method}__N{particles}__f{fold}__s{model_seed}.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume and path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if str(existing["status"].item()) == "complete":
                if "stimulus_schema_fingerprint" not in existing.files:
                    raise RuntimeError("resume rejected: archive lacks stimulus schema")
                if str(existing["stimulus_schema_fingerprint"].item()) != cohort.stimulus_schema_fingerprint:
                    raise RuntimeError("resume rejected: archive stimulus schema differs")
                return {"status": "skipped", "output": str(path), "wall_seconds": 0.0}
    checkpoint_path = _checkpoint_path(
        source_run, checkpoint_phase, model_id, history_lag, fold, model_seed
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    started = time.perf_counter()
    adapter = GeneratorAdapter.load(checkpoint_path, device=device)
    if tuple(cohort.neurons) != adapter.neurons:
        raise RuntimeError("checkpoint neuron order mismatch")
    config = RepairedResponseConfig(
        history_frames=history_lag,
        repair_frames=4,
        source_window_frames=4,
        horizon_frames=horizons,
        n_particles=particles,
        min_ess=min(20.0, max(4.0, 0.10 * particles)),
        sampling_chunk_size=1024,
    )
    projection, quantiles, thresholds = training_context(
        cohort, folds, fold, adapter.checkpoint, config, delays,
        boundary_shift_frames,
    )
    heldout = np.flatnonzero(folds == fold)
    d, h = cohort.n_neurons, len(horizons)
    shape = (len(heldout), len(PHASES), 3, len(delays), d, h, d)
    response = {key: np.full(shape, np.nan, dtype=np.float32) for key in RESPONSE_KEYS}
    diagnostics: dict[str, np.ndarray] = {}
    diagnostic_shapes: dict[str, tuple[int, ...]] = {}
    cut_times = np.full((len(heldout), len(PHASES), 3, len(delays)), -1, dtype=np.int32)

    for worm_position, worm in enumerate(heldout):
        standardized = causal_fill(
            standardize_for_checkpoint(cohort.traces[int(worm)], adapter.checkpoint)
        )
        stimulus = stimulus_for_trace(
            cohort.traces[int(worm)], cohort, int(worm), adapter.checkpoint
        )
        schedule = cohort.stimulus_schedules[int(worm)]
        for delay_position, delay in enumerate(delays):
            for phase, event, cut, _, _ in aligned_cuts(
                len(standardized), schedule, delay, boundary_shift_frames
            ):
                phase_position = PHASES.index(phase)
                draw_seed = _seed(
                    base_seed, method, model_id, fold, model_seed, int(worm),
                    phase_position, event, delay,
                )
                values = quantiles[(phase, delay)]
                if method == "direct":
                    prefix, future, factual_prefix = generate_path_bank(
                        adapter, standardized, stimulus, cut_time=cut,
                        config=config, seed=draw_seed,
                    )
                    result = estimate_repaired_responses(
                        prefix, future, factual_prefix, projection,
                        values["low"], values["high"], values["iqr"],
                        thresholds, config, anchor_lambda=0.25,
                        epsilon_iqr_fraction=0.25,
                    )
                elif method == "progressive":
                    result = progressive_smc_repaired_responses(
                        adapter, standardized, stimulus, cut_time=cut,
                        projection=projection, source_low=values["low"],
                        source_high=values["high"], source_iqr=values["iqr"],
                        thresholds=thresholds, config=config, seed=draw_seed,
                        branch_factor=branch_factor,
                        future_branch_factor=future_branch_factor,
                        tempering_ess_fraction=tempering_ess_fraction,
                    )
                else:
                    raise ValueError(f"unknown method {method}")
                cut_times[worm_position, phase_position, event, delay_position] = cut
                for key in RESPONSE_KEYS:
                    response[key][
                        worm_position, phase_position, event, delay_position
                    ] = result[key]
                for key, value in result.items():
                    if not key.startswith("diagnostic_"):
                        continue
                    array = np.asarray(value, dtype=np.float32)
                    if key not in diagnostics:
                        diagnostic_shapes[key] = array.shape
                        diagnostics[key] = np.full(
                            (len(heldout), len(PHASES), 3, len(delays), *array.shape),
                            np.nan, dtype=np.float32,
                        )
                    if array.shape != diagnostic_shapes[key]:
                        raise RuntimeError(f"diagnostic shape changed for {key}")
                    diagnostics[key][
                        worm_position, phase_position, event, delay_position
                    ] = array
            print(
                f"ALIGNED_WORM method={method} fold={fold} seed={model_seed} "
                f"worm={int(worm)} delay={delay}", flush=True,
            )
    if np.any(cut_times < 0):
        raise RuntimeError("one or more aligned cuts were not evaluated")
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        path,
        status=np.asarray("complete"), created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        method=np.asarray(method), model_id=np.asarray(model_id),
        checkpoint=np.asarray(str(checkpoint_path)), fold=np.asarray(fold),
        model_seed=np.asarray(model_seed), history_frames=np.asarray(history_lag),
        source_window_frames=np.asarray(config.source_window_frames),
        delay_frames=np.asarray(delays, dtype=np.int16),
        delay_seconds=np.asarray(delays, dtype=np.float32) / cohort.fps,
        source_window_start_seconds=(np.asarray(delays, dtype=np.float32) - 3) / cohort.fps,
        horizon_frames=np.asarray(horizons, dtype=np.int16),
        horizon_seconds=np.asarray(horizons, dtype=np.float32) / cohort.fps,
        phase_names=np.asarray(PHASES), worm_indices=heldout.astype(np.int16),
        worm_ids=np.asarray([cohort.worm_ids[int(index)] for index in heldout]),
        chemical_code_by_worm_event=np.asarray([
            cohort.stimulus_schedules[int(index)].chemical_code_by_event
            for index in heldout
        ], dtype=np.int8),
        chemical_name_by_worm_event=np.asarray([
            cohort.stimulus_schedules[int(index)].chemical_name_by_event
            for index in heldout
        ]),
        stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
        stimulus_schema_fingerprint=np.asarray(cohort.stimulus_schema_fingerprint),
        neurons=np.asarray(cohort.neurons), cut_times=cut_times,
        n_particles=np.asarray(particles), wall_seconds=np.asarray(elapsed),
        boundary_shift_frames=np.asarray(boundary_shift_frames),
        **response, **diagnostics,
    )
    del adapter
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"status": "ok", "output": str(path), "wall_seconds": elapsed}


def _write_state(output: Path, manifest: dict, records: list[dict]) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["records"] = records
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if records:
        keys = sorted(set().union(*(record.keys() for record in records)))
        with (output / "checkpoint_status.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader(); writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--fold-evidence", type=Path, required=True)
    parser.add_argument("--method", choices=("direct", "progressive"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-phase", default="winner_full_cv")
    parser.add_argument("--history-lag", type=int, default=80)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903])
    parser.add_argument("--delays", nargs="+", type=int, default=[7])
    parser.add_argument("--horizons", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--future-branch-factor", type=int, default=2)
    parser.add_argument("--tempering-ess-fraction", type=float, default=0.65)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-seed", type=int, default=20260828)
    parser.add_argument("--boundary-shift-frames", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument(
        "--cohort-mode", choices=("oh16230_head", "pooled_resampled"),
        default="pooled_resampled",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(cohort_mode=args.cohort_mode)
    folds = _load_folds(args.fold_evidence.resolve(), cohort)
    delays = tuple(sorted(set(args.delays))); horizons = tuple(sorted(set(args.horizons)))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method, "source_run": str(args.source_run.resolve()),
        "fold_evidence": str(args.fold_evidence.resolve()), "model_id": args.model_id,
        "checkpoint_phase": args.checkpoint_phase, "history_frames": args.history_lag,
        "delays_frames": delays, "delays_seconds": [x / cohort.fps for x in delays],
        "source_windows_seconds": [[(x - 3) / cohort.fps, x / cohort.fps] for x in delays],
        "horizons_frames": horizons, "horizons_seconds": [x / cohort.fps for x in horizons],
        "phases": list(PHASES), "quiet_control": "same delay at 15 s pre-onset pseudo-cut",
        "particles": args.particles, "folds": args.folds, "seeds": args.seeds,
        "cohort_mode": cohort.cohort_mode,
        "boundary_shift_frames": args.boundary_shift_frames,
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_schema": cohort.stimulus_schema_dict(),
        "config": asdict(RepairedResponseConfig(
            repair_frames=4, source_window_frames=4, horizon_frames=horizons,
            n_particles=args.particles,
            min_ess=min(20.0, max(4.0, 0.10 * args.particles)),
        )),
        "claim_boundary": "model-relative observed-law finite contrast; not a physical intervention, synapse, receptor action, or anatomical edge",
        "external_atlas_access": "none during estimation",
        "status": "running",
    }
    existing_manifest = output / "manifest.json"
    records: list[dict] = []
    if args.resume and existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text())
        if previous.get("stimulus_schema_fingerprint") != cohort.stimulus_schema_fingerprint:
            raise RuntimeError("resume rejected: stimulus-schema fingerprint differs")
        records = list(previous.get("records", []))
    _write_state(output, manifest, records)
    for fold in args.folds:
        for seed in args.seeds:
            print(f"ALIGNED_START method={args.method} fold={fold} seed={seed}", flush=True)
            try:
                result = run_checkpoint(
                    cohort=cohort, folds=folds, source_run=args.source_run.resolve(),
                    output_dir=output, method=args.method,
                    checkpoint_phase=args.checkpoint_phase, model_id=args.model_id,
                    history_lag=args.history_lag, fold=fold, model_seed=seed,
                    delays=delays, horizons=horizons, particles=args.particles,
                    device=args.device, base_seed=args.base_seed,
                    branch_factor=args.branch_factor,
                    future_branch_factor=args.future_branch_factor,
                    tempering_ess_fraction=args.tempering_ess_fraction,
                    boundary_shift_frames=args.boundary_shift_frames,
                    resume=args.resume,
                )
            except Exception as error:
                result = {"status": "failed", "error": repr(error), "wall_seconds": 0.0}
            record = {"method": args.method, "fold": fold, "seed": seed, **result}
            records.append(record); _write_state(output, manifest, records)
            print(f"ALIGNED_DONE {record}", flush=True)
    manifest["status"] = "complete" if all(r["status"] in {"ok", "skipped"} for r in records) else "partial"
    _write_state(output, manifest, records)


if __name__ == "__main__":
    main()
