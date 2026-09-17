#!/usr/bin/env python3
"""Production-code phase fit for the phase-duration matching analysis."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.common import load_dataset
from experiments.snapshot import (
    PHASE_ROOT,
    PREPARED_DATA,
    PRODUCTION_MODEL_SOURCE,
    deterministic_estimator_class,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PREPARED_DATA)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", choices=["matched", "full"], default="matched")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--frames-per-event", type=int, default=16)
    parser.add_argument("--events-per-recording", type=int, default=3)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--sample-seed-base", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--effective-batch-size", type=int, default=256)
    return parser.parse_args()


def select_segments(trace, phase, mode, frames, events, rng, stimulus_module):
    ranges = stimulus_module.get_4period_segments(len(trace))[SOURCE_PHASE_CODES[phase]]
    eligible = [(start, end) for start, end in ranges if end - start >= frames]
    if mode == "full":
        return [trace[start:end] for start, end in ranges], [
            (start, end, start, end) for start, end in ranges
        ]

    if phase in {"on", "off", "steady"}:
        chosen = eligible[:events]
    else:
        if len(eligible) < events:
            raise RuntimeError(f"Only {len(eligible)} eligible {phase} runs")
        chosen = [
            eligible[index]
            for index in rng.choice(len(eligible), size=events, replace=False)
        ]
    if len(chosen) != events:
        raise RuntimeError(f"Phase {phase} has {len(chosen)} runs; expected {events}")

    selected, records = [], []
    for start, end in chosen:
        if end - start == frames or phase in {"on", "off"}:
            crop_start = start
        else:
            crop_start = int(rng.integers(start, end - frames + 1))
        crop_end = crop_start + frames
        selected.append(trace[crop_start:crop_end])
        records.append((crop_start, crop_end, start, end))
    return selected, records


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_snapshot()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    torch.set_num_threads(max(1, args.torch_threads))

    traces, names, _ = load_dataset(args.dataset_dir)
    multilag = load_production_multilag()
    stimulus = load_stimulus_periods()
    Estimator = deterministic_estimator_class(multilag)

    sample_seed = (
        args.sample_seed_base
        + 10_000 * args.replicate
        + PHASES.index(args.phase)
    )
    rng = np.random.default_rng(sample_seed)
    segments, records = [], []
    for row_index, trace in enumerate(traces):
        selected, selected_records = select_segments(
            trace,
            args.phase,
            args.mode,
            args.frames_per_event,
            args.events_per_recording,
            rng,
            stimulus,
        )
        segments.extend(selected)
        for event, (crop_start, crop_end, source_start, source_end) in enumerate(
            selected_records
        ):
            records.append(
                {
                    "composite_row": row_index,
                    "event": event,
                    "crop_start": crop_start,
                    "crop_end": crop_end,
                    "source_run_start": source_start,
                    "source_run_end": source_end,
                }
            )

    output_dir = (
        args.out_root
        / args.mode
        / args.phase
        / f"replicate_{args.replicate:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    window_counts = {
        str(lag): int(sum(max(0, len(segment) - lag) for segment in segments))
        for lag in args.lags
    }
    manifest = {
        "status": "running",
        "analysis": "phase-duration matching sensitivity",
        "mode": args.mode,
        "phase_code": args.phase,
        "phase_label": PHASE_LABELS[args.phase],
        "replicate": args.replicate,
        "sample_seed": sample_seed,
        "model_seed": args.model_seed,
        "n_composite_rows": len(traces),
        "n_segments": len(segments),
        "segment_frames_total": int(sum(len(segment) for segment in segments)),
        "window_counts": window_counts,
        "effective_batch_size": args.effective_batch_size,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "dataset": repository_relative(args.dataset_dir),
        "dataset_sha256": sha256(args.dataset_dir / "traces.npz"),
        "model_source": repository_relative(PRODUCTION_MODEL_SOURCE),
        "model_source_sha256": sha256(PRODUCTION_MODEL_SOURCE),
        "phase_anchor": repository_relative(PHASE_ROOT / args.phase / "result.npz"),
        "reference_phase_anchor_sha256": sha256(
            PHASE_ROOT / args.phase / "result.npz"
        ),
        "settings": {
            "lags": args.lags,
            "noise_std": 0.1,
            "hidden_dim": 64,
            "num_layers": 2,
            "lr": 1e-3,
            "epochs": 100,
            "n_folds": 5,
            "hac_max_lag": 5,
            "fdr_alpha": 0.1,
            "fdr_method": "bh",
        },
        "departure_from_production": (
            "fold-local PyTorch seeds replace the unavailable reference RNG state"
        ),
    }
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        seed_all(args.model_seed)
        estimator = Estimator(
            lags=args.lags,
            tune_hp=False,
            noise_std=0.1,
            hidden_dim=64,
            num_layers=2,
            lr=1e-3,
            epochs=100,
            batch_size=args.effective_batch_size,
            n_folds=5,
            hac_max_lag=5,
            fdr_alpha=0.1,
            fdr_method="bh",
            device=args.device,
            verbose=True,
            random_state=42,
            model_seed=args.model_seed,
        )
        result = estimator.fit(segments)
        np.savez_compressed(
            output_dir / "result.npz",
            neuron_names=np.asarray(names),
            lags=np.asarray(args.lags),
            **{f"mu_hat_lag{lag}": result.mu_hat[lag] for lag in args.lags},
            **{f"p_value_lag{lag}": result.p_values[lag] for lag in args.lags},
            **{
                f"significant_lag{lag}": result.significant[lag].astype(bool)
                for lag in args.lags
            },
        )
        pd.DataFrame(records).to_csv(output_dir / "sampled_blocks.csv", index=False)
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds": time.monotonic() - started,
                "fold_seeds": {
                    f"lag{lag}_fold{fold}": estimator.fold_seeds[(lag, fold)]
                    for lag in args.lags
                    for fold in range(5)
                },
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
