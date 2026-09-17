from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.distribution_structure_evaluate import sensitivity_indices
from conditional_neural_benchmark.distribution_structure_scoring import score_samples
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.inference import load_checkpoint
from conditional_neural_benchmark.runner import ModelConfig, RunState, _neural_trial, _split_windows

from .protocol import FOLD_RUN, RUN_ROOT, atomic_csv, freeze_protocol, sha256, update_status
from .study import _sample_model, _verify_checkpoint, keyed_seed


SENSITIVITY_CONFIGS = (
    ModelConfig(
        "query_autoregressive_mdn4_order101", "tcn", "autoregressive_mdn", True,
        width=128, dropout=0.15,
        head_params={"hidden": 64, "layers": 2, "components": 4, "permutation_seed": 101},
        learning_rate=3e-4, weight_decay=7.5e-4,
        training_scheme="natural", history_noise_std=0.01, history_noise_copies=1,
    ),
    ModelConfig(
        "query_autoregressive_mdn4_order202", "tcn", "autoregressive_mdn", True,
        width=128, dropout=0.15,
        head_params={"hidden": 64, "layers": 2, "components": 4, "permutation_seed": 202},
        learning_rate=3e-4, weight_decay=7.5e-4,
        training_scheme="natural", history_noise_std=0.01, history_noise_copies=1,
    ),
    ModelConfig(
        "query_autoregressive_mdn8_order0", "tcn", "autoregressive_mdn", True,
        width=128, dropout=0.15,
        head_params={"hidden": 64, "layers": 2, "components": 8, "permutation_seed": 0},
        learning_rate=3e-4, weight_decay=7.5e-4,
        training_scheme="natural", history_noise_std=0.01, history_noise_copies=1,
    ),
)


def run_model_sensitivities(run_root: Path, *, smoke: bool = False) -> None:
    protocol = freeze_protocol(run_root)
    update_status(run_root, "model_sensitivities", "running")
    torch.set_num_threads(2)
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    training_root = run_root / "model_sensitivity_training"
    training_root.mkdir(parents=True, exist_ok=True)
    state = RunState(training_root, time.monotonic() + 24 * 3600)
    metrics_path = training_root / "trial_metrics.csv"
    if metrics_path.exists():
        state.records = pd.read_csv(metrics_path).replace({np.nan: None}).to_dict("records")
    selected_folds = [0] if smoke else list(range(5))
    configs = SENSITIVITY_CONFIGS[:1] if smoke else SENSITIVITY_CONFIGS
    phase = "mdn_sensitivity_smoke" if smoke else "mdn_sensitivity"
    for fold in selected_folds:
        scaler, training, validation, testing = _split_windows(cohort, folds, fold, 80)
        for config in configs:
            existing = [
                row for row in state.records
                if row.get("phase") == phase and row.get("model_id") == config.model_id
                and int(row.get("fold", -1)) == fold and row.get("status") == "ok"
            ]
            if not existing:
                _neural_trial(
                    state=state, phase=phase, config=config, cohort=cohort, lag=80,
                    fold=fold, seed=1701, train=training, validation=validation,
                    test=testing, scaler=scaler, device=protocol["resources"]["device"],
                    max_epochs=2 if smoke else 50, patience=2 if smoke else 9,
                    eval_rows=160 if smoke else 1600, n_samples=8 if smoke else 32,
                    batch_size=256,
                    trial_metadata={
                        "cohort_mode": "oh16230_head",
                        "stimulus_encoding": "binary_any_stimulus",
                        "sensitivity_only": True,
                        "protocol_fingerprint": protocol["fingerprint"],
                    },
                )
                if state.records[-1].get("status") != "ok":
                    raise RuntimeError(f"model sensitivity fit failed: {state.records[-1]}")
    records = pd.DataFrame(state.records)
    records = records[(records.phase == phase) & (records.status == "ok")]
    score_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for record in records.itertuples():
        checkpoint = training_root / record.checkpoint
        manifest_rows.append({
            "run_id": run_root.name,
            "model_id": record.model_id,
            "fold": int(record.fold),
            "model_seed": int(record.seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "training_seconds": float(record.train_seconds),
            "best_epoch": int(record.best_epoch),
            "parameter_count": int(record.parameter_count),
        })
        fold = int(record.fold)
        scaler, training, _, testing = _split_windows(cohort, folds, fold, 80)
        selected_index = sensitivity_indices(testing, 2 if smoke else 8)
        selected = testing.take(selected_index)
        innovation = training.target - training.history[:, -1, :cohort.n_neurons]
        threshold = 2 * np.maximum(np.sqrt(np.mean(innovation.astype(float) ** 2, axis=0)), 1e-3)
        model, payload, device = load_checkpoint(checkpoint, protocol["resources"]["device"])
        _verify_checkpoint(payload, cohort, scaler, fold, 1701)
        target = selected.target - selected.history[:, -1, :cohort.n_neurons]
        started = time.perf_counter()
        draws = _sample_model(
            model, device, selected.history,
            n_samples=16 if smoke else 64,
            seed=keyed_seed("mdn_sensitivity", record.model_id, fold),
            residual_target=False, n_neurons=cohort.n_neurons, chunk=2,
        )
        metrics = score_samples(
            draws, target, threshold,
            variogram_offset=selected.history[:, -1, :cohort.n_neurons],
        )
        for worm_index in np.unique(selected.worm):
            mask = selected.worm == worm_index
            score_rows.append({
                "run_id": run_root.name,
                "dataset": "neuropal_oh16230",
                "fold": fold,
                "worm_id": cohort.worm_ids[int(worm_index)],
                "model_id": record.model_id,
                "model_family": "autoregressive_mdn_sensitivity",
                "model_seed": 1701,
                "sampling_seed": keyed_seed("mdn_sensitivity", record.model_id, fold),
                "components": 8 if "mdn8" in record.model_id else 4,
                "permutation_seed": 101 if "order101" in record.model_id else 202 if "order202" in record.model_id else 0,
                "particle_count": draws.shape[1],
                "wall_seconds": time.perf_counter() - started,
                **{name: float(value[mask].mean()) for name, value in metrics.items()},
            })
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    primary = pd.read_csv(run_root / "predictive_scores.csv")
    primary = primary[
        (primary.model_family == "autoregressive_mdn4")
        & (primary.model_seed == 1701)
        & (primary.sampling_seed == 731)
    ].copy()
    for row in primary.to_dict("records"):
        score_rows.append({
            **row,
            "model_id": "query_autoregressive_mdn4_order0",
            "components": 4,
            "permutation_seed": 0,
        })
    atomic_csv(run_root / "model_sensitivity_manifest.csv", pd.DataFrame(manifest_rows))
    atomic_csv(run_root / "model_sensitivity_scores.csv", pd.DataFrame(score_rows))
    update_status(
        run_root, "model_sensitivities", "complete",
        trained_models=len(manifest_rows), score_rows=len(score_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_model_sensitivities(args.run_root.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
