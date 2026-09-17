from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from compatibility_neural_benchmark.core import causal_fill
from conditional_neural_benchmark.data import FoldScaler, load_cohort


ROOT = Path(__file__).resolve().parents[1]
SBTG_ROOT = ROOT / "SBTG"
if str(SBTG_ROOT) not in sys.path:
    sys.path.insert(0, str(SBTG_ROOT))

from pipeline.models.sbtg import SBTGStructuredVolatilityEstimator  # noqa: E402


def _load_folds(path: Path, cohort) -> np.ndarray:
    result = np.full(cohort.n_worms, -1, dtype=np.int64)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "worm_id" in row and row["worm_id"] in cohort.worm_ids:
                result[cohort.worm_ids.index(row["worm_id"])] = int(row["outer_fold"])
            elif "worm_index" in row and int(row["worm_index"]) < cohort.n_worms:
                result[int(row["worm_index"])] = int(row["outer_fold"])
    if np.any(result < 0):
        raise RuntimeError("fold assignments are incomplete")
    return result


def _pearson_lag(traces: list[np.ndarray], lag: int) -> np.ndarray:
    past = np.concatenate([trace[:-lag] for trace in traces], axis=0).astype(np.float64)
    future = np.concatenate([trace[lag:] for trace in traces], axis=0).astype(np.float64)
    past -= past.mean(axis=0, keepdims=True)
    future -= future.mean(axis=0, keepdims=True)
    denom = np.sqrt(
        np.square(future).sum(axis=0)[:, None] * np.square(past).sum(axis=0)[None]
    )
    result = future.T @ past / np.maximum(denom, 1e-12)
    np.fill_diagonal(result, 0.0)
    return result.astype(np.float32)


def run_fold_lag(
    *,
    cohort,
    folds: np.ndarray,
    fold: int,
    lag: int,
    seed: int,
    epochs: int,
    inner_folds: int,
    device: str,
) -> dict[str, np.ndarray | float | int | str]:
    training_indices = np.flatnonzero(folds != fold)
    scaler = FoldScaler.fit([cohort.traces[int(i)] for i in training_indices])
    traces = [
        causal_fill(scaler.transform(cohort.traces[int(i)])) for i in training_indices
    ]
    estimator = SBTGStructuredVolatilityEstimator(
        window_length=2,
        time_lag=int(lag),
        dsm_hidden_dim=128,
        dsm_num_layers=3,
        dsm_noise_std=0.1,
        dsm_epochs=int(epochs),
        dsm_batch_size=128,
        dsm_lr=1e-3,
        train_frac=0.7,
        hac_max_lag=5,
        fdr_alpha=0.2,
        fdr_method="by",
        volatility_test=True,
        structured_hidden_dim=64,
        structured_num_layers=2,
        structured_l1_lambda=0.001,
        structured_init_scale=0.1,
        compute_undirected=False,
        model_type="feature_bilinear",
        feature_dim=16,
        feature_hidden_dim=64,
        feature_num_layers=2,
        inference_mode="cross_fit",
        n_folds=int(inner_folds),
        cross_fit_group_by_segment=True,
        random_state=int(seed + 1009 * fold + 17 * lag),
        device=None if device == "auto" else device,
        verbose=False,
    )
    started = time.perf_counter()
    result = estimator.fit(traces)
    return {
        "status": "complete",
        "fold": fold,
        "lag": lag,
        "seed": seed,
        "wall_seconds": time.perf_counter() - started,
        "mu_hat": result.mu_hat.astype(np.float32),
        "p_mean": result.p_mean.astype(np.float32),
        "significant_mean": (result.sign_adj != 0).astype(np.int8),
        "volatility_stat": result.volatility_stat.astype(np.float32),
        "significant_volatility": result.volatility_adj.astype(np.int8),
        "pearson": _pearson_lag(traces, lag),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 32, 40])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cohort-mode", choices=("oh16230_head", "pooled_resampled"),
        default="oh16230_head",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    run_dir = args.run_dir.resolve()
    outputs = run_dir / "fold_lag"
    outputs.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(cohort_mode=args.cohort_mode)
    folds = _load_folds(source_run / "fold_assignments.csv", cohort)
    records: list[dict] = []
    manifest = {
        "run_id": run_dir.name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run),
        "method": "SBTG-FeatureBilinear",
        "model_type": "feature_bilinear",
        "outer_split": "same frozen whole-worm folds as repaired responses",
        "inner_inference": f"{args.inner_folds}-fold whole-worm held-out score cross-fitting within outer-training worms",
        "lags": args.lags,
        "folds": args.folds,
        "seed": args.seed,
        "epochs": args.epochs,
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "neurons": list(cohort.neurons),
        "cohort_mode": cohort.cohort_mode,
        "data_lineage": "raw head recordings only; no tail pseudo-pairing; no donor-trace copying",
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_schema": cohort.stimulus_schema_dict(),
        "claim_boundary": "reduced-form SBTG score-product association, not an anatomical or causal edge",
        "status": "running",
    }
    for fold in args.folds:
        for lag in args.lags:
            path = outputs / f"sbtg_feature_bilinear__f{fold}__lag{lag}__s{args.seed}.npz"
            if args.resume and path.exists():
                with np.load(path, allow_pickle=False) as existing:
                    if str(existing["status"].item()) == "complete":
                        if "stimulus_schema_fingerprint" not in existing.files:
                            raise RuntimeError("resume rejected: SBTG archive lacks stimulus schema")
                        if str(existing["stimulus_schema_fingerprint"].item()) != cohort.stimulus_schema_fingerprint:
                            raise RuntimeError("resume rejected: SBTG archive stimulus schema differs")
                        records.append(
                            {"status": "skipped", "fold": fold, "lag": lag, "output": str(path.relative_to(run_dir))}
                        )
                        continue
            print(f"SBTG_START fold={fold} lag={lag}", flush=True)
            try:
                result = run_fold_lag(
                    cohort=cohort,
                    folds=folds,
                    fold=fold,
                    lag=lag,
                    seed=args.seed,
                    epochs=args.epochs,
                    inner_folds=args.inner_folds,
                    device=args.device,
                )
                np.savez_compressed(
                    path,
                    neurons=np.asarray(cohort.neurons),
                    worm_ids=np.asarray(cohort.worm_ids),
                    cohort_mode=np.asarray(cohort.cohort_mode),
                    stimulus_schema_version=np.asarray(cohort.stimulus_schema_version),
                    stimulus_schema_fingerprint=np.asarray(
                        cohort.stimulus_schema_fingerprint
                    ),
                    data_lineage=np.asarray(
                        "raw head recordings only; no tail pseudo-pairing; no donor-trace copying"
                    ),
                    **result,
                )
                record = {
                    "status": "ok",
                    "fold": fold,
                    "lag": lag,
                    "wall_seconds": result["wall_seconds"],
                    "output": str(path.relative_to(run_dir)),
                }
            except Exception as error:
                record = {"status": "failed", "fold": fold, "lag": lag, "error": repr(error)}
            records.append(record)
            manifest["records"] = records
            manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
            print(f"SBTG_DONE {record}", flush=True)
    manifest["status"] = "complete" if all(r["status"] in {"ok", "skipped"} for r in records) else "partial"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["records"] = records
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    keys = sorted(set().union(*(record.keys() for record in records)))
    with (run_dir / "status.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
