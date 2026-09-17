#!/usr/bin/env python3
"""Alternative, reproducible phase-duration sensitivity fits.

This script separates two questions that the original four-second crop control
combined:

1. Does matching the *number* of phase windows change the lag profile when the
   sampled windows retain coverage of the complete phase?
2. Does the answer depend on phase/lag-specific normalization and overlapping
   train/test windows?

The script therefore supports both the production estimator behavior and a
more conservative analysis with a phase-common scale, recording-row-held-out folds,
and recording-row-aware inference.  It never overwrites the reference phase-duration results.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

from experiments.common import load_dataset
from experiments.snapshot import (
    PHASE_ROOT,
    PREPARED_DATA,
    PRODUCTION_MODEL_SOURCE,
    derive_seed,
    load_production_multilag,
    load_stimulus_periods,
    repository_relative,
    seed_all,
    sha256,
    validate_snapshot,
)


PHASES = ("baseline", "steady", "on", "off")
SOURCE_PHASE_CODES = {
    "baseline": "NOTHING",
    "steady": "SHOWING",
    "on": "ON",
    "off": "OFF",
}
PHASE_LABELS = {
    "baseline": "Baseline",
    "steady": "Steady",
    "on": "On",
    "off": "Off",
}
VARIANTS = {
    # Deterministic maximum-buffer sensitivity: use the middle four seconds of
    # each Steady run and the middle of the three longest Baseline runs.
    "center_crop_production": {
        "selection": "center_crop",
        "scaling": "production",
        "folding": "production",
        "inference": "hac",
    },
    # Change only how Baseline/Steady windows are selected.
    "distributed_production": {
        "selection": "distributed_per_lag",
        "scaling": "production",
        "folding": "production",
        "inference": "hac",
    },
    # Keep the original crop geometry, but remove overlapping fold leakage and
    # use one phase-balanced scale for every block and lag.
    "crop_common_grouped": {
        "selection": "crop",
        "scaling": "phase_common",
        "folding": "recording_row",
        "inference": "recording_row",
    },
    # Preferred count-matched analysis: sample across the complete phase.
    "distributed_common_grouped": {
        "selection": "distributed_per_lag",
        "scaling": "phase_common",
        "folding": "recording_row",
        "inference": "recording_row",
    },
    # Additional lag-comparability sensitivity: every lag uses the same 660
    # anchor times, selected from the lag-5 eligible set.
    "shared_anchor_common_grouped": {
        "selection": "distributed_shared_anchor",
        "scaling": "phase_common",
        "folding": "recording_row",
        "inference": "recording_row",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PREPARED_DATA)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--sample-seed-base", type=int, default=42)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--frames-per-event", type=int, default=16)
    parser.add_argument("--events-per-recording", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--effective-batch-size", type=int, default=256)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def balanced_recording_row_folds(
    n_recording_rows: int, n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Assign complete prepared recording rows to balanced, deterministic folds."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_recording_rows)
    assignments = np.empty(n_recording_rows, dtype=int)
    assignments[permutation] = np.arange(n_recording_rows) % n_folds
    return assignments


def phase_segments(trace: np.ndarray, phase: str, stimulus_module) -> list[tuple[int, int]]:
    ranges = stimulus_module.get_4period_segments(len(trace))[SOURCE_PHASE_CODES[phase]]
    return [(int(start), int(end)) for start, end in ranges]


def stratified_indices(n_candidates: int, n_select: int, rng: np.random.Generator) -> np.ndarray:
    """Select one index per temporal quantile bin for broad phase coverage."""
    if n_select > n_candidates:
        raise ValueError(f"Cannot select {n_select} from {n_candidates} candidates")
    if n_select == n_candidates:
        return np.arange(n_candidates, dtype=int)
    boundaries = np.linspace(0, n_candidates, n_select + 1, dtype=int)
    selected = [int(rng.integers(lo, hi)) for lo, hi in zip(boundaries[:-1], boundaries[1:])]
    if len(set(selected)) != n_select:
        raise RuntimeError("Temporal-bin sampler produced duplicate indices")
    return np.asarray(selected, dtype=int)


def select_crop_ranges(
    trace: np.ndarray,
    phase: str,
    frames: int,
    events: int,
    rng: np.random.Generator,
    stimulus_module,
) -> list[tuple[int, int, int]]:
    """Reproduce the existing four-second crop selection."""
    ranges = phase_segments(trace, phase, stimulus_module)
    eligible = [(start, end, index) for index, (start, end) in enumerate(ranges) if end - start >= frames]
    if phase in {"on", "off", "steady"}:
        chosen = eligible[:events]
    else:
        if len(eligible) < events:
            raise RuntimeError(f"Only {len(eligible)} eligible {phase} runs")
        chosen = [eligible[index] for index in rng.choice(len(eligible), size=events, replace=False)]
    if len(chosen) != events:
        raise RuntimeError(f"Phase {phase} has {len(chosen)} runs; expected {events}")

    output = []
    for start, end, segment_index in chosen:
        if end - start == frames or phase in {"on", "off"}:
            crop_start = start
        else:
            crop_start = int(rng.integers(start, end - frames + 1))
        output.append((crop_start, crop_start + frames, segment_index))
    return output


def select_center_ranges(
    trace: np.ndarray,
    phase: str,
    frames: int,
    events: int,
    stimulus_module,
) -> list[tuple[int, int, int]]:
    """Choose deterministic central crops with maximal transition buffer."""
    ranges = phase_segments(trace, phase, stimulus_module)
    eligible = [
        (start, end, index)
        for index, (start, end) in enumerate(ranges)
        if end - start >= frames
    ]
    if phase == "baseline":
        # Prefer the intervals providing the largest symmetric buffer around a
        # four-second crop; resolve equal lengths by temporal order.
        chosen = sorted(
            eligible,
            key=lambda value: (-(value[1] - value[0]), value[2]),
        )[:events]
    else:
        chosen = eligible[:events]
    if len(chosen) != events:
        raise RuntimeError(f"Phase {phase} has {len(chosen)} runs; expected {events}")
    output = []
    for start, end, segment_index in chosen:
        crop_start = start + (end - start - frames) // 2
        output.append((crop_start, crop_start + frames, segment_index))
    return output


def all_candidates(
    trace: np.ndarray,
    phase: str,
    lag: int,
    stimulus_module,
) -> list[tuple[int, int]]:
    """Return (segment_index, absolute_start) for every phase-valid window."""
    candidates: list[tuple[int, int]] = []
    for segment_index, (start, end) in enumerate(phase_segments(trace, phase, stimulus_module)):
        candidates.extend((segment_index, t) for t in range(start, end - lag))
    return candidates


def build_windows(
    traces: list[np.ndarray],
    phase: str,
    lag: int,
    selection: str,
    replicate: int,
    sample_seed_base: int,
    frames: int,
    events: int,
    stimulus_module,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Construct count-matched windows and retain their complete provenance."""
    rows: list[dict] = []
    windows: list[np.ndarray] = []
    phase_index = PHASES.index(phase)

    for recording_row, trace in enumerate(traces):
        sampling_lag = 5 if selection == "distributed_shared_anchor" else lag
        rng = np.random.default_rng(
            derive_seed(
                sample_seed_base,
                7007,
                replicate,
                phase_index,
                sampling_lag,
                recording_row,
            )
        )
        if selection in {"crop", "center_crop"}:
            selected: list[tuple[int, int]] = []
            if selection == "crop":
                crop_rng = np.random.default_rng(
                    sample_seed_base + 10_000 * replicate + phase_index
                )
                # Advance the shared crop RNG through preceding recording rows
                # exactly as the reference implementation does.
                for preceding in range(recording_row + 1):
                    chosen = select_crop_ranges(
                        traces[preceding],
                        phase,
                        frames,
                        events,
                        crop_rng,
                        stimulus_module,
                    )
            else:
                chosen = select_center_ranges(
                    trace, phase, frames, events, stimulus_module
                )
            for crop_start, crop_end, segment_index in chosen:
                selected.extend((segment_index, t) for t in range(crop_start, crop_end - lag))
        else:
            anchor_lag = max(5, lag) if selection == "distributed_shared_anchor" else lag
            candidates = all_candidates(trace, phase, anchor_lag, stimulus_module)
            target = events * (frames - anchor_lag)
            selected_indices = stratified_indices(len(candidates), target, rng)
            selected = [candidates[index] for index in selected_indices]

        for segment_index, start in selected:
            block = np.asarray(trace[start : start + lag + 1], dtype=np.float64)
            if len(block) != lag + 1:
                raise RuntimeError("Selected window crosses a phase-segment boundary")
            windows.append(block.reshape(-1))
            rows.append(
                {
                    "recording_row": recording_row,
                    "phase_code": phase,
                    "replicate": replicate,
                    "lag": lag,
                    "segment_index": segment_index,
                    "absolute_start": start,
                    "absolute_end": start + lag + 1,
                }
            )

    metadata = pd.DataFrame(rows)
    ordering = np.lexsort(
        (
            metadata["absolute_start"].to_numpy(),
            metadata["segment_index"].to_numpy(),
            metadata["recording_row"].to_numpy(),
        )
    )
    metadata = metadata.iloc[ordering].reset_index(drop=True)
    array = np.asarray(windows, dtype=np.float64)[ordering]
    return array, metadata


def phase_balanced_scaler(
    traces: list[np.ndarray],
    training_rows: np.ndarray,
    stimulus_module,
) -> tuple[np.ndarray, np.ndarray]:
    """Training-only per-neuron scale giving each phase equal weight."""
    phase_means = []
    phase_second_moments = []
    for phase in PHASES:
        chunks = []
        for recording_row in training_rows:
            trace = traces[int(recording_row)]
            chunks.extend(trace[start:end] for start, end in phase_segments(trace, phase, stimulus_module))
        values = np.concatenate(chunks, axis=0)
        phase_means.append(np.nanmean(values, axis=0))
        phase_second_moments.append(np.nanmean(values * values, axis=0))
    mean = np.nanmean(np.stack(phase_means), axis=0)
    second_moment = np.nanmean(np.stack(phase_second_moments), axis=0)
    variance = np.maximum(second_moment - mean * mean, 1e-8)
    std = np.sqrt(variance)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std < 1e-4] = 1.0
    return mean, std


def apply_common_scale(windows: np.ndarray, mean: np.ndarray, std: np.ndarray, lag: int) -> np.ndarray:
    tiled_mean = np.tile(mean, lag + 1)
    tiled_std = np.tile(std, lag + 1)
    standardized = (windows - tiled_mean) / tiled_std
    return np.nan_to_num(np.clip(standardized, -10, 10), nan=0.0)


def train_cross_fitted_scores(
    module,
    traces: list[np.ndarray],
    stimulus_module,
    windows: np.ndarray,
    metadata: pd.DataFrame,
    lag: int,
    scaling: str,
    folding: str,
    model_seed: int,
    epochs: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict, dict]:
    n_neurons = traces[0].shape[1]
    n_folds = 5
    row_assignments = balanced_recording_row_folds(len(traces), n_folds=n_folds)
    recording_rows = metadata["recording_row"].to_numpy(int)
    if folding == "recording_row":
        fold_ids = row_assignments[recording_rows]
    else:
        segment_ids = (
            metadata["recording_row"].to_numpy(int) * 100
            + metadata["segment_index"].to_numpy(int)
        )
        fold_ids = module.create_fold_assignments(segment_ids, n_folds, 42)

    scores_heldout = np.zeros_like(windows)
    fold_seeds = {}
    scaler_records = {}
    for fold in range(n_folds):
        fold_seed = derive_seed(model_seed, 20260123, lag, fold)
        fold_seeds[str(fold)] = int(fold_seed)
        seed_all(fold_seed)
        train_mask = fold_ids != fold
        heldout_mask = fold_ids == fold
        if not train_mask.any() or not heldout_mask.any():
            raise RuntimeError(f"Empty train/test partition in fold {fold}")

        if scaling == "production":
            standardized, _, _ = module.standardize_windows(windows, train_mask)
        else:
            heldout_rows = np.where(row_assignments == fold)[0]
            training_rows = np.where(row_assignments != fold)[0]
            mean, std = phase_balanced_scaler(traces, training_rows, stimulus_module)
            standardized = apply_common_scale(windows, mean, std, lag)
            scaler_records[str(fold)] = {
                "training_recording_rows": training_rows.tolist(),
                "heldout_recording_rows": heldout_rows.tolist(),
                "mean_min": float(mean.min()),
                "mean_max": float(mean.max()),
                "std_min": float(std.min()),
                "std_max": float(std.max()),
            }

        if lag == 1:
            model = module.TwoBlockStructuredScoreNet(
                n_neurons=n_neurons, hidden_dim=64, num_layers=2
            )
        else:
            model = module.MultiBlockStructuredScoreNet(
                n_neurons=n_neurons, p_max=lag, hidden_dim=64, num_layers=2
            )
        module.train_score_model(
            model,
            standardized[train_mask],
            0.1,
            1e-3,
            epochs,
            batch_size,
            0.0,
            device,
            verbose=False,
        )
        scores_heldout[heldout_mask] = module.compute_scores(
            model, standardized[heldout_mask], device
        )

    fold_record = {
        "folding": folding,
        "recording_row_fold_assignments": row_assignments.tolist(),
        "fold_window_counts": {
            str(fold): int(np.sum(fold_ids == fold)) for fold in range(n_folds)
        },
    }
    return scores_heldout, {"fold_seeds": fold_seeds, **fold_record}, scaler_records


def recording_row_test(
    s_future: np.ndarray,
    s_past: np.ndarray,
    recording_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test score products using composite recording rows as independent units."""
    unique_rows = np.unique(recording_rows)
    row_means = []
    for recording_row in unique_rows:
        mask = recording_rows == recording_row
        row_means.append(
            np.einsum("nj,ni->nji", s_future[mask], s_past[mask]).mean(axis=0)
        )
    values = np.stack(row_means, axis=0)
    mu = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(unique_rows))
    statistic = np.divide(mu, se, out=np.zeros_like(mu), where=se > 1e-12)
    p_values = 2 * student_t.sf(np.abs(statistic), df=len(unique_rows) - 1)
    diagonal = np.eye(mu.shape[0], dtype=bool)
    mu[diagonal] = 0.0
    p_values[diagonal] = 1.0
    return mu, p_values, values


def main() -> None:
    args = parse_args()
    validate_snapshot()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    torch.set_num_threads(max(1, args.torch_threads))

    traces, names, row_metadata = load_dataset(args.dataset_dir)
    module = load_production_multilag()
    stimulus = load_stimulus_periods()
    settings = VARIANTS[args.variant]
    output_dir = (
        args.out_root
        / args.variant
        / args.phase
        / f"seed_{args.model_seed:04d}"
        / f"replicate_{args.replicate:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "analysis": "phase-duration alternative window/scaling sensitivity",
        "variant": args.variant,
        "variant_settings": settings,
        "phase_code": args.phase,
        "phase_label": PHASE_LABELS[args.phase],
        "replicate": args.replicate,
        "model_seed": args.model_seed,
        "sample_seed_base": args.sample_seed_base,
        "n_recording_rows": len(traces),
        "row_metadata": row_metadata.to_dict(orient="records"),
        "dataset": repository_relative(args.dataset_dir),
        "dataset_sha256": sha256(args.dataset_dir / "traces.npz"),
        "model_source": repository_relative(PRODUCTION_MODEL_SOURCE),
        "model_source_sha256": sha256(PRODUCTION_MODEL_SOURCE),
        "phase_anchor": repository_relative(PHASE_ROOT / args.phase / "result.npz"),
        "phase_anchor_sha256": sha256(PHASE_ROOT / args.phase / "result.npz"),
        "settings": {
            "lags": args.lags,
            "frames_per_event": args.frames_per_event,
            "events_per_recording": args.events_per_recording,
            "noise_std": 0.1,
            "hidden_dim": 64,
            "num_layers": 2,
            "lr": 1e-3,
            "epochs": args.epochs,
            "n_folds": 5,
            "hac_max_lag": 5,
            "fdr_alpha": 0.1,
            "saved_fdr_methods": ["bh", "by"],
            "effective_batch_size": args.effective_batch_size,
            "device": args.device,
            "torch_threads": args.torch_threads,
        },
    }
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        result_arrays = {
            "neuron_names": np.asarray(names),
            "lags": np.asarray(args.lags),
        }
        all_metadata = []
        lag_records = {}
        for lag in args.lags:
            print(
                f"[{args.variant} {args.phase} rep={args.replicate} seed={args.model_seed}] lag {lag}",
                flush=True,
            )
            windows, metadata = build_windows(
                traces,
                args.phase,
                lag,
                settings["selection"],
                args.replicate,
                args.sample_seed_base,
                args.frames_per_event,
                args.events_per_recording,
                stimulus,
            )
            expected_per_recording_row = (
                args.events_per_recording * (args.frames_per_event - 5)
                if settings["selection"] == "distributed_shared_anchor"
                else args.events_per_recording * (args.frames_per_event - lag)
            )
            observed_per_recording_row = (
                metadata.groupby("recording_row").size().to_numpy()
            )
            if not np.all(observed_per_recording_row == expected_per_recording_row):
                raise RuntimeError(
                    f"Lag {lag}: per-recording-row counts "
                    f"{observed_per_recording_row.tolist()}, expected "
                    f"{expected_per_recording_row}"
                )
            scores, fold_record, scaler_record = train_cross_fitted_scores(
                module,
                traces,
                stimulus,
                windows,
                metadata,
                lag,
                settings["scaling"],
                settings["folding"],
                args.model_seed,
                args.epochs,
                args.effective_batch_size,
                args.device,
            )
            n_neurons = len(names)
            blocks = scores.reshape(len(scores), lag + 1, n_neurons)
            s_past = blocks[:, 0, :]
            s_future = blocks[:, lag, :]
            if settings["inference"] == "recording_row":
                mu, p_values, row_means = recording_row_test(
                    s_future,
                    s_past,
                    metadata["recording_row"].to_numpy(int),
                )
                result_arrays[f"recording_row_mu_lag{lag}"] = row_means
            else:
                mu, p_values = module.hac_test_mu_hat(s_future, s_past, 5)

            sig_bh = module.apply_fdr(p_values, 0.1, "bh")
            sig_by = module.apply_fdr(p_values, 0.1, "by")
            result_arrays[f"mu_hat_lag{lag}"] = mu
            result_arrays[f"p_value_lag{lag}"] = p_values
            result_arrays[f"significant_lag{lag}"] = sig_bh.astype(bool)
            result_arrays[f"significant_bh_lag{lag}"] = sig_bh.astype(bool)
            result_arrays[f"significant_by_lag{lag}"] = sig_by.astype(bool)
            lag_records[str(lag)] = {
                "n_windows": len(windows),
                "per_recording_row_windows": expected_per_recording_row,
                "n_bh_edges": int(sig_bh.sum()),
                "n_by_edges": int(sig_by.sum()),
                **fold_record,
                "scalers": scaler_record,
            }
            all_metadata.append(metadata)

        np.savez_compressed(output_dir / "result.npz", **result_arrays)
        pd.concat(all_metadata, ignore_index=True).to_csv(
            output_dir / "selected_windows.csv", index=False
        )
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds": time.monotonic() - started,
                "lag_records": lag_records,
            }
        )
        atomic_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "elapsed_seconds": time.monotonic() - started,
                "error": repr(exc),
            }
        )
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
