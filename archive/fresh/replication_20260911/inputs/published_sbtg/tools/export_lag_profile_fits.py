#!/usr/bin/env python3
"""Export completed lag-profile fits to the portable release schema.

The source campaign consists of four recording-row resamples with two model
seeds each and three full-data controls. Source paths are command-line inputs;
no machine-specific path is written to an output artifact. Every exported NPZ
contains only numeric, Boolean, or Unicode arrays and loads with
``allow_pickle=False``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


LAGS = (1, 2, 3, 5, 8, 10, 15, 20)
BOOTSTRAP_REPLICATES = (0, 1, 2, 3)
BOOTSTRAP_SEEDS = (42, 43)
FULL_SEEDS = (42, 43, 44)
DATA_SEED = 20260803
MODEL_SOURCE_SHA256 = "f987cfd13748adf32106353fa6a13431c9bfb8482d223a4af268d583c15df5a7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(base_seed: int, *components: int) -> int:
    sequence = np.random.SeedSequence(
        [int(base_seed), *(int(component) for component in components)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def fit_label(sample_kind: str, replicate: int, model_seed: int) -> str:
    if sample_kind == "full":
        return f"full_seed_{model_seed:04d}"
    return f"bootstrap_{replicate:03d}_seed_{model_seed:04d}"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def parse_full_control(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("full control must use SEED=PATH syntax")
    seed_text, path_text = value.split("=", 1)
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("full-control seed must be an integer") from exc
    if not path_text:
        raise argparse.ArgumentTypeError("full-control path is empty")
    return seed, Path(path_text)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read source manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise RuntimeError(f"Source manifest is not complete: {path}")
    return value


def scalar_json(value: np.ndarray) -> dict:
    text = str(np.asarray(value).item())
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise TypeError("Hyperparameter payload is not a JSON object")
    return decoded


def source_fit_root(
    bootstrap_root: Path,
    full_controls: dict[int, Path],
    sample_kind: str,
    replicate: int,
    model_seed: int,
) -> Path:
    if sample_kind == "full":
        return full_controls[model_seed]
    return bootstrap_root / fit_label(sample_kind, replicate, model_seed)


def export_one_fit(
    source_root: Path,
    output_root: Path,
    dataset_sha256: str,
    n_rows: int,
    sample_kind: str,
    replicate: int,
    model_seed: int,
    lag: int,
) -> None:
    source_dir = source_root / f"lag_{lag:02d}"
    source_result = source_dir / "result.npz"
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    if source_manifest.get("historical_model_sha256") != MODEL_SOURCE_SHA256:
        raise RuntimeError(f"Unexpected model source digest in {source_manifest_path}")

    with np.load(source_result, allow_pickle=False) as archive:
        observed_lag = int(archive["lag"])
        if observed_lag != lag:
            raise RuntimeError(f"Lag mismatch in {source_result}: {observed_lag} != {lag}")
        mu_hat = np.asarray(archive["mu_hat"], dtype=np.float64)
        p_value = np.asarray(archive["pval"], dtype=np.float64)
        significant = np.asarray(archive["sig"], dtype=bool)
        neuron_names = np.asarray(archive["neuron_names"], dtype=str)
        fold_seeds = np.asarray(archive["fold_seeds"], dtype=np.uint32)
        hp_config = scalar_json(archive["hp_config"])
        if sample_kind == "bootstrap":
            source_indices = np.asarray(archive["source_row_indices"], dtype=np.int64)
            multiplicities = np.asarray(
                archive["source_row_multiplicities"], dtype=np.int64
            )
        else:
            source_indices = np.arange(n_rows, dtype=np.int64)
            multiplicities = np.ones(n_rows, dtype=np.int64)

    expected_shape = (len(neuron_names), len(neuron_names))
    if any(array.shape != expected_shape for array in (mu_hat, p_value, significant)):
        raise RuntimeError(f"Matrix shape mismatch in {source_result}")
    if not np.all(np.isfinite(mu_hat)) or not np.all(np.isfinite(p_value)):
        raise RuntimeError(f"Non-finite matrix values in {source_result}")
    if len(source_indices) != n_rows or len(multiplicities) != n_rows:
        raise RuntimeError(f"Recording-row metadata mismatch in {source_result}")
    expected_multiplicities = np.bincount(source_indices, minlength=n_rows)
    if not np.array_equal(multiplicities, expected_multiplicities):
        raise RuntimeError(f"Recording-row multiplicities are inconsistent in {source_result}")
    expected_fold_seeds = np.asarray(
        [derive_seed(model_seed, 20260123, lag, fold) for fold in range(5)],
        dtype=np.uint32,
    )
    if not np.array_equal(fold_seeds, expected_fold_seeds):
        raise RuntimeError(f"Fold-seed mismatch in {source_result}")

    label = fit_label(sample_kind, replicate, model_seed)
    output_dir = output_root / label / f"lag_{lag:02d}"
    output_result = output_dir / "result.npz"
    atomic_npz(
        output_result,
        mu_hat=mu_hat,
        p_value=p_value,
        significant=significant,
        neuron_names=neuron_names,
        lag=np.asarray(lag, dtype=np.int64),
        sample_kind=np.asarray(sample_kind),
        replicate=np.asarray(replicate, dtype=np.int64),
        model_seed=np.asarray(model_seed, dtype=np.int64),
        data_seed=np.asarray(DATA_SEED, dtype=np.int64),
        source_row_indices=source_indices,
        source_row_multiplicities=multiplicities,
        hp_config=np.asarray(json.dumps(hp_config, sort_keys=True)),
        fold_seeds=fold_seeds,
    )

    device = str(source_manifest.get("device", ""))
    batch_size = source_manifest.get("effective_batch_size")
    if device not in {"cpu", "cuda", "mps"}:
        raise RuntimeError(f"Invalid source device in {source_manifest_path}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise RuntimeError(f"Invalid source batch size in {source_manifest_path}")
    manifest = {
        "status": "complete",
        "format_version": 1,
        "fit_label": label,
        "sample_kind": sample_kind,
        "replicate": replicate,
        "model_seed": model_seed,
        "data_seed": DATA_SEED,
        "lag": lag,
        "n_folds": 5,
        "n_neurons": len(neuron_names),
        "n_rows_full": n_rows,
        "n_source_rows": n_rows,
        "n_sampled_rows_with_multiplicity": len(source_indices),
        "source_row_indices": source_indices.tolist(),
        "source_row_multiplicities": multiplicities.tolist(),
        "fold_seeds": {
            str(fold): int(value) for fold, value in enumerate(fold_seeds)
        },
        "dataset_sha256": dataset_sha256,
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "fixed_hyperparameters_source_sha256": source_manifest.get("fixed_hp_sha256"),
        "source_archive_sha256": sha256(source_result),
        "source_manifest_sha256": sha256(source_manifest_path),
        "device": device,
        "effective_batch_size": batch_size,
        "hp_config": hp_config,
        "schema": "pickle-free lag-profile fit; all paths are repository relative by construction",
    }
    atomic_json(output_dir / "manifest.json", manifest)


def export_campaign(
    bootstrap_root: Path,
    full_controls: dict[int, Path],
    dataset_archive: Path,
    output_root: Path,
) -> None:
    missing_seeds = sorted(set(FULL_SEEDS) - set(full_controls))
    if missing_seeds:
        raise ValueError(f"Missing full controls for seeds: {missing_seeds}")
    with np.load(dataset_archive, allow_pickle=False) as dataset:
        offsets = np.asarray(dataset["offsets"], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("Dataset archive has invalid row offsets")
    n_rows = len(offsets) - 1
    dataset_digest = sha256(dataset_archive)

    fits: list[tuple[str, int, int]] = [
        ("full", -1, seed) for seed in FULL_SEEDS
    ] + [
        ("bootstrap", replicate, seed)
        for replicate in BOOTSTRAP_REPLICATES
        for seed in BOOTSTRAP_SEEDS
    ]
    for sample_kind, replicate, model_seed in fits:
        root = source_fit_root(
            bootstrap_root, full_controls, sample_kind, replicate, model_seed
        )
        for lag in LAGS:
            export_one_fit(
                root,
                output_root,
                dataset_digest,
                n_rows,
                sample_kind,
                replicate,
                model_seed,
                lag,
            )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-root", type=Path, required=True)
    parser.add_argument(
        "--full-control",
        type=parse_full_control,
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="source full-control root; required for seeds 42, 43, and 44",
    )
    parser.add_argument("--dataset-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.full_control = dict(args.full_control)
    return args


def main() -> None:
    args = parse_args()
    export_campaign(
        args.bootstrap_root,
        args.full_control,
        args.dataset_archive,
        args.output_root,
    )
    print(f"Exported 88 portable fit artifacts to {args.output_root}")


if __name__ == "__main__":
    main()
