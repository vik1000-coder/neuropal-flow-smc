#!/usr/bin/env python3
"""Run one full-data control or recording-row bootstrap lag-profile fit.

The runner uses the reference production architecture, prepared data, neuron
order, and fixed hyperparameters. Bootstrap draws preserve row
multiplicity: a sampled row is appended once for every occurrence in the index
vector, and omitted rows are absent.  Explicit fold-local model seeds make each
bootstrap fit comparable to its full-data control with the same model seed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import LAGS, load_dataset
from experiments.snapshot import (
    PREPARED_DATA,
    PRODUCTION_HYPERPARAMETERS,
    PRODUCTION_MODEL_SOURCE,
    deterministic_estimator_class,
    load_fixed_hyperparameters,
    load_production_multilag,
    repository_relative,
    seed_all,
    sha256,
    validate_snapshot,
)


DEFAULT_DATA_SEED = 20260803


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PREPARED_DATA)
    parser.add_argument(
        "--hyperparameters", type=Path, default=PRODUCTION_HYPERPARAMETERS
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--sample-kind", choices=("full", "bootstrap"), required=True)
    parser.add_argument(
        "--replicate",
        type=int,
        default=-1,
        help="Bootstrap replicate index; must be -1 for a full-data control.",
    )
    parser.add_argument("--lag", type=int, choices=LAGS, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=256,
        help="production base batch 128 doubled to 256 on CUDA; emulate it on CPU.",
    )
    return parser.parse_args()


def load_fixed_config(path: Path, lag: int) -> dict:
    values = load_fixed_hyperparameters(path)
    if lag not in values:
        raise KeyError(f"No fixed hyperparameters are available for lag {lag}")
    return values[lag]


def bootstrap_indices(n_rows: int, replicate: int, data_seed: int) -> np.ndarray:
    if replicate < 0:
        raise ValueError("Bootstrap replicate must be nonnegative")
    sequence = np.random.SeedSequence([int(data_seed), 20260803, int(replicate)])
    rng = np.random.default_rng(sequence)
    return rng.integers(0, n_rows, size=n_rows, dtype=np.int64)


def fit_label(sample_kind: str, replicate: int, model_seed: int) -> str:
    if sample_kind == "full":
        return f"full_seed_{model_seed:04d}"
    return f"bootstrap_{replicate:03d}_seed_{model_seed:04d}"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_npz(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_snapshot()
    if args.sample_kind == "full" and args.replicate != -1:
        raise ValueError("Full-data controls require --replicate -1")
    if args.sample_kind == "bootstrap" and args.replicate < 0:
        raise ValueError("Bootstrap fits require a nonnegative --replicate")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if args.device != "cpu" and args.effective_batch_size % 2:
        raise ValueError("Non-CPU effective batch size must be divisible by two")
    torch.set_num_threads(max(1, args.torch_threads))
    # production's trainer doubles the estimator batch argument on every non-CPU
    # device. Keep the effective batch fixed at the requested value on all backends.
    estimator_batch_size = (
        args.effective_batch_size
        if args.device == "cpu"
        else args.effective_batch_size // 2
    )

    traces, neuron_names, rows = load_dataset(args.dataset_dir)
    n_rows = len(traces)
    if args.sample_kind == "full":
        source_indices = np.arange(n_rows, dtype=np.int64)
    else:
        source_indices = bootstrap_indices(n_rows, args.replicate, args.data_seed)
    # Do not deduplicate. Repeated indices must remain repeated training segments.
    sampled_traces = [traces[int(index)] for index in source_indices]
    multiplicities = np.bincount(source_indices, minlength=n_rows).astype(np.int64)

    label = fit_label(args.sample_kind, args.replicate, args.model_seed)
    job_dir = args.out_root / label / f"lag_{args.lag:02d}"
    result_path = job_dir / "result.npz"
    manifest_path = job_dir / "manifest.json"
    config = load_fixed_config(args.hyperparameters, args.lag)
    module = load_production_multilag()
    Estimator = deterministic_estimator_class(module)

    manifest = {
        "status": "running",
        "analysis": "fixed-hyperparameter recording-row bootstrap",
        "fit_label": label,
        "sample_kind": args.sample_kind,
        "replicate": args.replicate,
        "lag": args.lag,
        "model_seed": args.model_seed,
        "data_seed": args.data_seed,
        "source_row_indices": source_indices.tolist(),
        "source_row_multiplicities": multiplicities.tolist(),
        "n_source_rows": n_rows,
        "n_sampled_rows_with_multiplicity": len(sampled_traces),
        "n_distinct_sampled_rows": int(np.count_nonzero(multiplicities)),
        "sampled_row_metadata": [
            rows.iloc[int(index)].to_dict() for index in source_indices
        ],
        "row_multiplicity_policy": (
            "every occurrence is retained as a separate segment; sampled rows are "
            "not deduplicated"
        ),
        "n_neurons": len(neuron_names),
        "n_folds": args.n_folds,
        "device": args.device,
        "effective_batch_size": args.effective_batch_size,
        "estimator_batch_size_argument": estimator_batch_size,
        "torch_threads": args.torch_threads,
        "dataset": repository_relative(args.dataset_dir),
        "dataset_sha256": sha256(args.dataset_dir / "traces.npz"),
        "hyperparameters": repository_relative(args.hyperparameters),
        "hyperparameters_sha256": sha256(args.hyperparameters),
        "model_source": repository_relative(PRODUCTION_MODEL_SOURCE),
        "model_source_sha256": sha256(PRODUCTION_MODEL_SOURCE),
        "hp_config": config,
        "scope": (
            "recording-row resampling conditional on fixed preprocessing and "
            "production-selected hyperparameters"
        ),
        "departure_from_production": (
            "fold-local PyTorch seeds replace the unavailable reference "
            "PyTorch RNG state"
        ),
    }
    atomic_json(manifest_path, manifest)

    started = time.monotonic()
    try:
        seed_all(args.model_seed)
        estimator = Estimator(
            lags=[args.lag],
            tune_hp=False,
            noise_std=float(config["noise_std"]),
            hidden_dim=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            lr=float(config["lr"]),
            epochs=int(config["epochs"]),
            batch_size=estimator_batch_size,
            n_folds=args.n_folds,
            hac_max_lag=5,
            fdr_alpha=0.1,
            fdr_method="bh",
            device=args.device,
            verbose=True,
            random_state=42,
            model_seed=args.model_seed,
        )
        result = estimator.fit(sampled_traces)
        fold_seeds = np.asarray(
            [estimator.fold_seeds[(args.lag, fold)] for fold in range(args.n_folds)],
            dtype=np.uint32,
        )
        atomic_npz(
            result_path,
            mu_hat=result.mu_hat[args.lag],
            p_value=result.p_values[args.lag],
            significant=result.significant[args.lag].astype(bool),
            neuron_names=np.asarray(neuron_names),
            lag=args.lag,
            sample_kind=args.sample_kind,
            replicate=args.replicate,
            model_seed=args.model_seed,
            data_seed=args.data_seed,
            source_row_indices=source_indices,
            source_row_multiplicities=multiplicities,
            hp_config=json.dumps(config, sort_keys=True),
            fold_seeds=fold_seeds,
        )
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds": time.monotonic() - started,
                "fold_seeds": {
                    str(fold): int(fold_seeds[fold])
                    for fold in range(args.n_folds)
                },
                "output_file": result_path.relative_to(args.out_root).as_posix(),
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
