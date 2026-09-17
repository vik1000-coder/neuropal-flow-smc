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
    GeneratorAdapter,
    RepairedResponseConfig,
    causal_fill,
    episode_cuts,
    estimate_repaired_responses,
    fit_anchor_projection,
    generate_path_bank,
    smc_repaired_responses,
    standardize_for_checkpoint,
    stimulus_for_trace,
)
from compatibility_neural_benchmark.progressive_smc import (
    progressive_smc_repaired_responses,
)
from conditional_neural_benchmark.data import load_cohort


METHODS = (
    "direct_importance",
    "terminal_smc",
    "progressive_bridge_smc",
    "temporal_cut_smc",
)
PHASES = ("baseline", "onset", "active")
RESPONSE_KEYS = (
    "response_endpoint_mean",
    "response_cumulative_mean",
    "response_peak_mean",
    "response_event_probability",
    "response_endpoint_sd",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(
    source_run: Path,
    checkpoint_phase: str,
    model_id: str,
    history_lag: int,
    fold: int,
    seed: int,
) -> Path:
    return (
        source_run
        / "checkpoints"
        / checkpoint_phase
        / f"{model_id}__L{history_lag}__f{fold}__s{seed}.pt"
    )


def output_path(
    output: Path,
    method: str,
    model_id: str,
    source_lag: int,
    fold: int,
    seed: int,
    particles: int,
) -> Path:
    return (
        output
        / "responses"
        / method
        / (
            f"{model_id}__{method}__ell{source_lag}__N{particles}"
            f"__f{fold}__s{seed}.npz"
        )
    )


def validate_resume_archive(
    archive,
    *,
    cohort,
    method: str,
    model_id: str,
    fold: int,
    seed: int,
    source_lag: int,
    particles: int,
    horizons: tuple[int, ...],
    source_window_frames: int,
    history_lag: int,
    checkpoint: Path,
    expected_worm_indices: np.ndarray,
) -> None:
    """Fail closed before reusing a completed response archive."""
    required = {
        "status",
        "method",
        "model_id",
        "checkpoint",
        "checkpoint_sha256",
        "fold",
        "seed",
        "history_frames",
        "source_lag_frames",
        "source_window_frames",
        "n_particles",
        "horizon_frames",
        "worm_indices",
        "worm_ids",
        "neurons",
        "stimulus_schema_version",
        "stimulus_schema_fingerprint",
        "chemical_code_by_worm_event",
        "chemical_name_by_worm_event",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        raise RuntimeError(
            "resume rejected: completed archive lacks required provenance fields: "
            + ", ".join(missing)
        )
    scalar_expected = {
        "status": "complete",
        "method": method,
        "model_id": model_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "fold": int(fold),
        "seed": int(seed),
        "history_frames": int(history_lag),
        "source_lag_frames": int(source_lag),
        "source_window_frames": int(source_window_frames),
        "n_particles": int(particles),
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
    }
    for key, expected in scalar_expected.items():
        actual = archive[key].item()
        if actual != expected:
            raise RuntimeError(
                f"resume rejected: archive {key} differs "
                f"(found {actual!r}, expected {expected!r})"
            )
    expected_worm_indices = np.asarray(expected_worm_indices, dtype=np.int64)
    expected_worm_ids = np.asarray(
        [cohort.worm_ids[int(index)] for index in expected_worm_indices]
    )
    expected_codes = np.asarray(
        [
            cohort.stimulus_schedules[int(index)].chemical_code_by_event
            for index in expected_worm_indices
        ],
        dtype=np.int8,
    )
    expected_names = np.asarray(
        [
            cohort.stimulus_schedules[int(index)].chemical_name_by_event
            for index in expected_worm_indices
        ]
    )
    array_expected = {
        "horizon_frames": np.asarray(horizons, dtype=np.int16),
        "worm_indices": expected_worm_indices.astype(np.int16),
        "worm_ids": expected_worm_ids,
        "neurons": np.asarray(cohort.neurons),
        "chemical_code_by_worm_event": expected_codes,
        "chemical_name_by_worm_event": expected_names,
    }
    for key, expected in array_expected.items():
        if not np.array_equal(archive[key], expected):
            raise RuntimeError(f"resume rejected: archive {key} differs")


def load_folds(path: Path, cohort) -> np.ndarray:
    frame = pd.read_csv(path)
    if frame.worm_id.astype(str).duplicated().any():
        raise RuntimeError("fold file contains duplicated worm IDs")
    mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
    missing = sorted(set(cohort.worm_ids) - set(mapping))
    if missing:
        raise RuntimeError(f"fold file is missing cohort worms: {missing}")
    folds = np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)
    if set(folds.tolist()) != set(range(5)):
        raise RuntimeError("expected immutable five-fold assignment")
    return folds


def training_context(
    cohort,
    folds: np.ndarray,
    fold: int,
    checkpoint: dict,
    config: RepairedResponseConfig,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Fit only nuisance summaries, using training worms and the exact lag."""
    training = np.flatnonzero(folds != fold)
    traces = [
        causal_fill(standardize_for_checkpoint(cohort.traces[int(i)], checkpoint))
        for i in training
    ]
    projection = fit_anchor_projection(traces, config.anchor_rank)
    buckets: dict[str, list[np.ndarray]] = {phase: [] for phase in PHASES}
    for trace, worm in zip(traces, training):
        cuts = episode_cuts(
            len(trace),
            cohort.stimulus_schedules[int(worm)],
            config.source_window_frames,
        )
        for cut in cuts:
            if cut.phase not in buckets:
                continue
            # cut.time is the final observed frame. A lag ell means that the
            # source-window final frame is exactly ell frames before that cut.
            source_hi = cut.time - config.source_lag_frames + 1
            source_lo = source_hi - config.source_window_frames
            if source_lo >= 0:
                buckets[cut.phase].append(trace[source_lo:source_hi].mean(axis=0))
    quantiles: dict[str, dict[str, np.ndarray]] = {}
    for phase, values in buckets.items():
        if not values:
            raise RuntimeError(f"no lag-aligned training source statistics for {phase}")
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


def draw_seed(
    base: int,
    model_id: str,
    fold: int,
    seed: int,
    worm: int,
    phase: int,
    event: int,
    source_lag: int,
) -> int:
    # Deliberately omit the method: all four workflows receive the same keyed
    # seed for a matched episode, although their proposal shapes can differ.
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


def estimate(
    method: str,
    adapter: GeneratorAdapter,
    standardized: np.ndarray,
    stimulus: np.ndarray,
    *,
    cut_time: int,
    projection: np.ndarray,
    quantiles: dict[str, np.ndarray],
    thresholds: np.ndarray,
    config: RepairedResponseConfig,
    seed: int,
    progressive_branch_factor: int,
    progressive_future_branch_factor: int,
) -> dict[str, np.ndarray]:
    common = dict(
        cut_time=cut_time,
        projection=projection,
        source_low=quantiles["low"],
        source_high=quantiles["high"],
        source_iqr=quantiles["iqr"],
        thresholds=thresholds,
        config=config,
        seed=seed,
    )
    if method == "direct_importance":
        prefix, future, factual_prefix = generate_path_bank(
            adapter,
            standardized,
            stimulus,
            cut_time=cut_time,
            config=config,
            seed=seed,
        )
        return estimate_repaired_responses(
            prefix,
            future,
            factual_prefix,
            projection,
            quantiles["low"],
            quantiles["high"],
            quantiles["iqr"],
            thresholds,
            config,
        )
    if method == "terminal_smc":
        return smc_repaired_responses(
            adapter,
            standardized,
            stimulus,
            **common,
            resampling_policy="terminal_deferred",
        )
    if method == "temporal_cut_smc":
        return smc_repaired_responses(
            adapter,
            standardized,
            stimulus,
            **common,
            resampling_policy="temporal_cut",
        )
    if method == "progressive_bridge_smc":
        return progressive_smc_repaired_responses(
            adapter,
            standardized,
            stimulus,
            **common,
            branch_factor=progressive_branch_factor,
            future_branch_factor=progressive_future_branch_factor,
        )
    raise ValueError(f"unknown method {method}")


def run_one(
    *,
    cohort,
    folds: np.ndarray,
    source_run: Path,
    checkpoint_phase: str,
    output: Path,
    method: str,
    model_id: str,
    history_lag: int,
    fold: int,
    seed: int,
    source_lag: int,
    particles: int,
    horizons: tuple[int, ...],
    source_window_frames: int,
    device: str,
    base_seed: int,
    min_ess: float,
    progressive_branch_factor: int,
    progressive_future_branch_factor: int,
) -> dict[str, object]:
    path = output_path(
        output, method, model_id, source_lag, fold, seed, particles
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_path(
        source_run, checkpoint_phase, model_id, history_lag, fold, seed
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    heldout = np.flatnonzero(folds == fold)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if str(existing["status"].item()) == "complete":
                validate_resume_archive(
                    existing,
                    cohort=cohort,
                    method=method,
                    model_id=model_id,
                    fold=fold,
                    seed=seed,
                    source_lag=source_lag,
                    particles=particles,
                    horizons=horizons,
                    source_window_frames=source_window_frames,
                    history_lag=history_lag,
                    checkpoint=ckpt_path,
                    expected_worm_indices=heldout,
                )
                return {"status": "skipped", "output": str(path), "wall_seconds": 0.0}
    started = time.perf_counter()
    adapter = GeneratorAdapter.load(str(ckpt_path), device=device)
    if tuple(cohort.neurons) != adapter.neurons:
        raise RuntimeError("checkpoint neuron order mismatch")
    config = RepairedResponseConfig(
        history_frames=history_lag,
        repair_frames=source_lag + source_window_frames,
        source_window_frames=source_window_frames,
        source_lag_frames=source_lag,
        horizon_frames=horizons,
        n_particles=particles,
        min_ess=min_ess,
        resample_ess_fraction=0.50,
        sampling_chunk_size=1024,
    )
    projection, quantiles, thresholds = training_context(
        cohort, folds, fold, adapter.checkpoint, config
    )
    d, h = cohort.n_neurons, len(horizons)
    response = {
        key: np.full(
            (len(heldout), len(PHASES), 3, d, h, d), np.nan, dtype=np.float32
        )
        for key in RESPONSE_KEYS
    }
    diagnostic_names: list[str] | None = None
    diagnostic: dict[str, np.ndarray] = {}
    cut_times = np.full((len(heldout), len(PHASES), 3), -1, dtype=np.int32)
    source_window_bounds = np.full(
        (len(heldout), len(PHASES), 3, 2), -1, dtype=np.int32
    )
    for worm_position, worm in enumerate(heldout):
        standardized = causal_fill(
            standardize_for_checkpoint(cohort.traces[int(worm)], adapter.checkpoint)
        )
        stimulus = stimulus_for_trace(
            cohort.traces[int(worm)], cohort, int(worm), adapter.checkpoint
        )
        cuts = episode_cuts(
            len(cohort.traces[int(worm)]),
            cohort.stimulus_schedules[int(worm)],
            config.source_window_frames,
        )
        for cut in cuts:
            if cut.phase not in PHASES:
                continue
            phase_index = PHASES.index(cut.phase)
            keyed_seed = draw_seed(
                base_seed,
                model_id,
                fold,
                seed,
                int(worm),
                phase_index,
                cut.event,
                source_lag,
            )
            result = estimate(
                method,
                adapter,
                standardized,
                stimulus,
                cut_time=cut.time,
                projection=projection,
                quantiles=quantiles[cut.phase],
                thresholds=thresholds,
                config=config,
                seed=keyed_seed,
                progressive_branch_factor=progressive_branch_factor,
                progressive_future_branch_factor=progressive_future_branch_factor,
            )
            cut_times[worm_position, phase_index, cut.event] = cut.time
            source_hi = cut.time - source_lag + 1
            source_lo = source_hi - source_window_frames
            source_window_bounds[worm_position, phase_index, cut.event] = (
                source_lo,
                source_hi,
            )
            for key in RESPONSE_KEYS:
                response[key][worm_position, phase_index, cut.event] = result[key]
            current_names = sorted(
                key.removeprefix("diagnostic_")
                for key, value in result.items()
                if key.startswith("diagnostic_") and np.asarray(value).ndim == 1
            )
            if diagnostic_names is None:
                diagnostic_names = current_names
                diagnostic = {
                    name: np.full(
                        (len(heldout), len(PHASES), 3, d),
                        np.nan,
                        dtype=np.float32,
                    )
                    for name in current_names
                }
            if current_names != diagnostic_names:
                raise RuntimeError("diagnostic schema changed within a run")
            for name in diagnostic_names:
                diagnostic[name][worm_position, phase_index, cut.event] = result[
                    f"diagnostic_{name}"
                ]
        print(
            f"FOUR_SAMPLER_WORM_DONE method={method} ell={source_lag} "
            f"fold={fold} worm={int(worm)}",
            flush=True,
        )
    if np.any(cut_times < 0):
        raise RuntimeError("one or more prespecified episode cuts were not evaluated")
    if np.any(source_window_bounds[..., 1] - source_window_bounds[..., 0] != source_window_frames):
        raise RuntimeError("source-window bounds disagree with declared width")
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        path,
        status=np.asarray("complete"),
        created_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        method=np.asarray(method),
        model_id=np.asarray(model_id),
        checkpoint=np.asarray(str(ckpt_path.resolve())),
        checkpoint_sha256=np.asarray(sha256(ckpt_path)),
        fold=np.asarray(fold),
        seed=np.asarray(seed),
        history_frames=np.asarray(history_lag),
        source_lag_frames=np.asarray(source_lag),
        source_lag_seconds=np.asarray(source_lag / cohort.fps),
        lag_definition=np.asarray("source-window end to prediction cut"),
        source_to_readout_seconds=(
            source_lag + np.asarray(horizons, dtype=np.float32)
        ) / cohort.fps,
        repair_frames=np.asarray(config.repair_frames),
        source_window_frames=np.asarray(config.source_window_frames),
        n_particles=np.asarray(particles),
        horizon_frames=np.asarray(horizons, dtype=np.int16),
        horizon_seconds=np.asarray(horizons, dtype=np.float32) / cohort.fps,
        phase_names=np.asarray(PHASES),
        worm_indices=heldout.astype(np.int16),
        worm_ids=np.asarray([cohort.worm_ids[int(index)] for index in heldout]),
        chemical_code_by_worm_event=np.asarray(
            [
                cohort.stimulus_schedules[int(index)].chemical_code_by_event
                for index in heldout
            ],
            dtype=np.int8,
        ),
        chemical_name_by_worm_event=np.asarray(
            [
                cohort.stimulus_schedules[int(index)].chemical_name_by_event
                for index in heldout
            ]
        ),
        stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
        stimulus_schema_fingerprint=np.asarray(cohort.stimulus_schema_fingerprint),
        neurons=np.asarray(cohort.neurons),
        cut_times=cut_times,
        source_window_bounds=source_window_bounds,
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
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--fold-file", type=Path)
    parser.add_argument("--checkpoint-phase", default="chemical_full_cv")
    parser.add_argument(
        "--model-id",
        default="stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01",
    )
    parser.add_argument("--cohort-mode", default="oh16230_head")
    parser.add_argument("--history-lag", type=int, default=80)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--source-lags", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--source-window-frames", type=int, default=4)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-seed", type=int, default=20260828)
    parser.add_argument("--min-ess", type=float, default=6.0)
    parser.add_argument("--progressive-branch-factor", type=int, default=2)
    parser.add_argument("--progressive-future-branch-factor", type=int, default=2)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_run = args.source_run.resolve()
    cohort = load_cohort(cohort_mode=args.cohort_mode)
    source_manifest = json.loads((source_run / "manifest.json").read_text())
    fold_file = (
        args.fold_file.resolve()
        if args.fold_file is not None
        else Path(source_manifest["fold_assignments"]).resolve()
    )
    folds = load_folds(fold_file, cohort)
    schema = cohort.stimulus_schema_dict()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "corrected four-workflow flow-repaired explicit source-lag analysis v1",
        "methods": args.methods,
        "method_taxonomy": {
            "direct_importance": "natural path bank plus self-normalized importance weights",
            "terminal_smc": "bootstrap proposal with lagged clamp weight deferred to mandatory cut resampling",
            "progressive_bridge_smc": "branched proposal with source constraint tempered across its window",
            "temporal_cut_smc": "lagged clamp with ESS-triggered resampling at/after its window and free rollout after the cut",
        },
        "source_run": str(source_run),
        "checkpoint_phase": args.checkpoint_phase,
        "model_id": args.model_id,
        "cohort_mode": args.cohort_mode,
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "fps": cohort.fps,
        "history_frames": args.history_lag,
        "source_lag_frames": args.source_lags,
        "source_lag_seconds": [lag / cohort.fps for lag in args.source_lags],
        "lag_definition": "source-window end to prediction cut",
        "source_to_readout_seconds": {
            str(lag): [
                (lag + horizon) / cohort.fps for horizon in args.horizons
            ]
            for lag in args.source_lags
        },
        "source_window_frames": args.source_window_frames,
        "horizon_frames": args.horizons,
        "horizon_seconds": [horizon / cohort.fps for horizon in args.horizons],
        "phases": list(PHASES),
        "particles": args.particles,
        "minimum_effective_sample_size": args.min_ess,
        "maximum_normalized_weight": RepairedResponseConfig.max_normalized_weight,
        "minimum_achieved_source_fraction": RepairedResponseConfig.min_achieved_fraction,
        "source_clamp_iqr_fraction": RepairedResponseConfig.epsilon_iqr_fraction,
        "anchor_lambda": RepairedResponseConfig.anchor_lambda,
        "anchor_rank": RepairedResponseConfig.anchor_rank,
        "resampling_ess_fraction": 0.50,
        "progressive_branch_factor": args.progressive_branch_factor,
        "progressive_future_branch_factor": args.progressive_future_branch_factor,
        "folds": args.folds,
        "seeds": args.seeds,
        "stimulus_schema": schema,
        "fold_assignments": str(fold_file),
        "fold_assignments_sha256": sha256(fold_file),
        "matrix_internal_orientation": "source,horizon,target",
        "atlas_firewall": "no atlas/connectome/receptor data used in estimation or tuning",
        "claim_boundary": "model-relative observed-law response, not a causal intervention or physical delay",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    records: list[dict[str, object]] = []
    for method in args.methods:
        for source_lag in args.source_lags:
            for fold in args.folds:
                for seed in args.seeds:
                    print(
                        f"FOUR_SAMPLER_START method={method} ell={source_lag} "
                        f"fold={fold} seed={seed}",
                        flush=True,
                    )
                    try:
                        result = run_one(
                            cohort=cohort,
                            folds=folds,
                            source_run=source_run,
                            checkpoint_phase=args.checkpoint_phase,
                            output=output,
                            method=method,
                            model_id=args.model_id,
                            history_lag=args.history_lag,
                            fold=fold,
                            seed=seed,
                            source_lag=source_lag,
                            particles=args.particles,
                            horizons=tuple(sorted(set(args.horizons))),
                            source_window_frames=args.source_window_frames,
                            device=args.device,
                            base_seed=args.base_seed,
                            min_ess=args.min_ess,
                            progressive_branch_factor=args.progressive_branch_factor,
                            progressive_future_branch_factor=args.progressive_future_branch_factor,
                        )
                    except Exception as error:
                        result = {
                            "status": "failed",
                            "error": repr(error),
                            "wall_seconds": 0.0,
                        }
                    record = {
                        "method": method,
                        "source_lag_frames": source_lag,
                        "source_lag_seconds": source_lag / cohort.fps,
                        "fold": fold,
                        "seed": seed,
                        **result,
                    }
                    records.append(record)
                    pd.DataFrame(records).to_csv(output / "run_status.csv", index=False)
                    print(
                        f"FOUR_SAMPLER_DONE method={method} ell={source_lag} "
                        f"fold={fold} seed={seed} status={result['status']} "
                        f"seconds={result.get('wall_seconds', 0):.1f}",
                        flush=True,
                    )
    failures = [row for row in records if row["status"] == "failed"]
    validation = {
        "status": "pass" if not failures else "failed",
        "expected_runs": len(args.methods) * len(args.source_lags) * len(args.folds) * len(args.seeds),
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
