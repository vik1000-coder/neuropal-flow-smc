from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from conditional_neural_benchmark.baselines import GaussianBaseline
from conditional_neural_benchmark.data import multiscale_features
from conditional_neural_benchmark.distribution_structure_scoring import score_samples
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.runner import _split_windows

from .core import HistorySupportModel
from .protocol import FOLD_RUN, RUN_ROOT, atomic_csv, freeze_protocol, update_status
from .study import keyed_seed


def _temporal_exclusion(windows, candidate: np.ndarray, radius: int) -> np.ndarray:
    """Exclude candidate windows and all temporally overlapping neighbors."""
    keep = np.ones(len(windows.target), dtype=bool)
    for worm in np.unique(windows.worm[candidate]):
        held_time = np.sort(windows.time[candidate][windows.worm[candidate] == worm])
        worm_rows = np.flatnonzero(windows.worm == worm)
        if not len(held_time):
            continue
        position = np.searchsorted(held_time, windows.time[worm_rows])
        left = np.maximum(position - 1, 0)
        right = np.minimum(position, len(held_time) - 1)
        distance = np.minimum(
            np.abs(windows.time[worm_rows] - held_time[left]),
            np.abs(windows.time[worm_rows] - held_time[right]),
        )
        keep[worm_rows[distance <= radius]] = False
    return np.flatnonzero(keep)


def _subsample(index: np.ndarray, n: int, seed: int) -> np.ndarray:
    values = np.asarray(index, dtype=np.int64)
    if len(values) <= n:
        return np.sort(values)
    return np.sort(np.random.default_rng(seed).choice(values, n, replace=False))


def _stress_masks(training, n_neurons: int, fold: int, smoke: bool) -> list[dict]:
    current = training.history[:, -1, :n_neurons]
    innovation = training.target - current
    reliability = innovation.std(axis=0)
    ordered = np.argsort(reliability)
    sources = ordered[np.asarray([len(ordered) // 5, len(ordered) // 2, 4 * len(ordered) // 5])]
    standardized = (current[:, sources] - np.median(current[:, sources], axis=0)) / np.maximum(
        np.std(current[:, sources], axis=0), 1e-6
    )
    tail = np.flatnonzero(np.max(standardized, axis=1) >= np.quantile(np.max(standardized, axis=1), 0.95))

    feature = multiscale_features(training.history, n_neurons)
    rng = np.random.default_rng(keyed_seed("pseudo_cluster", fold))
    fit_index = _subsample(np.arange(len(feature)), 800 if smoke else 5000, keyed_seed("pseudo_pca", fold))
    scaler = StandardScaler().fit(feature[fit_index])
    fit_scaled = scaler.transform(feature[fit_index])
    pca = PCA(n_components=min(8 if smoke else 16, fit_scaled.shape[1], len(fit_scaled) - 1), random_state=fold)
    pca.fit(fit_scaled)
    embedded = pca.transform(scaler.transform(feature))
    cluster = KMeans(n_clusters=4 if smoke else 8, n_init=10, random_state=fold).fit(embedded[fit_index])
    labels = cluster.predict(embedded)
    compactness = []
    for label in range(cluster.n_clusters):
        member = np.flatnonzero(labels == label)
        distance = np.linalg.norm(embedded[member] - cluster.cluster_centers_[label], axis=1)
        compactness.append((float(distance.mean()), -len(member), label))
    chosen_cluster = min(compactness)[2]
    region = np.flatnonzero(labels == chosen_cluster)
    onset = np.flatnonzero(np.asarray(training.stratum).astype(str) == "onset")
    return [
        {
            "stress_split": "tail_region_holdout",
            "candidate": tail,
            "source_indices": sources,
            "definition": "maximum standardized current activity across three train-only reliability-spanning neurons above q95",
        },
        {
            "stress_split": "compact_history_region_holdout",
            "candidate": region,
            "source_indices": np.asarray([], dtype=int),
            "definition": f"most compact of {cluster.n_clusters} train-only multiscale PCA clusters",
        },
        {
            "stress_split": "stimulus_onset_context_holdout",
            "candidate": onset,
            "source_indices": np.asarray([], dtype=int),
            "definition": "all onset-stratum windows",
        },
    ]


def run_pseudo_ood(run_root: Path, *, smoke: bool = False) -> None:
    protocol = freeze_protocol(run_root)
    update_status(run_root, "pseudo_ood_stress", "running")
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    rows: list[dict] = []
    split_manifest: list[dict] = []
    selected_folds = [0] if smoke else list(range(5))
    max_eval = 32 if smoke else 192
    max_fit = 800 if smoke else 5000
    max_calibration = 300 if smoke else 1200
    for fold in selected_folds:
        _, training, validation, _ = _split_windows(cohort, folds, fold, 80)
        for specification in _stress_masks(training, cohort.n_neurons, fold, smoke):
            candidate = np.asarray(specification["candidate"], dtype=np.int64)
            if len(candidate) < 8:
                split_manifest.append({
                    "run_id": run_root.name, "fold": fold, **specification,
                    "candidate": None, "status": "skipped_insufficient_rows",
                    "heldout_rows": len(candidate),
                })
                continue
            evaluation_index = _subsample(
                candidate, max_eval, keyed_seed("pseudo_eval", fold, specification["stress_split"])
            )
            fit_index = _temporal_exclusion(training, candidate, radius=80)
            fit_index = _subsample(
                fit_index, max_fit, keyed_seed("pseudo_fit", fold, specification["stress_split"])
            )
            if len(fit_index) < 256:
                split_manifest.append({
                    "run_id": run_root.name, "fold": fold,
                    "stress_split": specification["stress_split"],
                    "definition": specification["definition"],
                    "status": "skipped_insufficient_nonoverlap_training",
                    "heldout_rows": len(candidate), "fit_rows": len(fit_index),
                })
                continue
            control_index = _subsample(
                fit_index, len(evaluation_index), keyed_seed("pseudo_control", fold, specification["stress_split"])
            )
            started = time.perf_counter()
            fit_features = multiscale_features(training.history[fit_index], cohort.n_neurons)
            model = GaussianBaseline.ridge(fit_features, training.target[fit_index], alpha=10.0)
            innovation = training.target[fit_index] - training.history[fit_index, -1, :cohort.n_neurons]
            threshold = 2 * np.maximum(np.sqrt(np.mean(innovation.astype(float) ** 2, axis=0)), 1e-3)
            calibration_index = _subsample(
                np.arange(len(validation.target)), max_calibration,
                keyed_seed("pseudo_calibration", fold, specification["stress_split"]),
            )
            support = HistorySupportModel(
                k=protocol["support"]["k"],
                pca_cap=8 if smoke else 32,
                seed=keyed_seed("pseudo_support", fold, specification["stress_split"]),
            ).fit(
                training.history[fit_index, :, :cohort.n_neurons],
                training.history[fit_index, :, cohort.n_neurons:],
                validation.history[calibration_index, :, :cohort.n_neurons],
                validation.history[calibration_index, :, cohort.n_neurons:],
                max_train_rows=max_fit,
            )
            for role, index in (("heldout_stress", evaluation_index), ("supported_control", control_index)):
                feature = multiscale_features(training.history[index], cohort.n_neurons)
                draw_absolute = model.sample(
                    protocol["predictive_evaluation"]["sample_count"] if not smoke else 16,
                    keyed_seed("pseudo_draw", fold, specification["stress_split"], role),
                    features=feature,
                )
                current = training.history[index, -1, :cohort.n_neurons]
                draw = draw_absolute - current[:, None]
                target = training.target[index] - current
                metrics = score_samples(draw, target, threshold, variogram_offset=current)
                source = int(specification["source_indices"][0]) if len(specification["source_indices"]) else None
                profile = support.score(
                    training.history[index, :, :cohort.n_neurons],
                    training.history[index, :, cohort.n_neurons:],
                    source=source,
                )
                for position, row_index in enumerate(index):
                    row = {
                        "run_id": run_root.name,
                        "dataset": "neuropal_oh16230",
                        "system": "observable_pseudo_ood",
                        "generator_seed": -1,
                        "fold": fold,
                        "row_index": int(row_index),
                        "worm_id": cohort.worm_ids[int(training.worm[row_index])],
                        "model_family": "ridge_full_gaussian_stress_probe",
                        "model_seed": 0,
                        "sampling_seed": keyed_seed("pseudo_draw", fold, specification["stress_split"], role),
                        "query_id": f"pseudo_{fold}_{specification['stress_split']}_{int(row_index)}",
                        "query_class": specification["stress_split"],
                        "stress_split": specification["stress_split"],
                        "stress_role": role,
                        "source_neuron": ";".join(cohort.neurons[int(value)] for value in specification["source_indices"]),
                        "target_functional": "observed_next_frame_joint_prediction",
                        "amplitude": float(np.max(np.abs(training.history[row_index, -1, :cohort.n_neurons]))),
                        "particle_count": draw.shape[1],
                        "smc_method": "ancestral",
                        "n_fit_rows_after_overlap_exclusion": len(fit_index),
                        "definition": specification["definition"],
                        "wall_seconds": time.perf_counter() - started,
                        **profile.iloc[position].to_dict(),
                    }
                    for name, values in metrics.items():
                        row[name] = float(values[position])
                    rows.append(row)
            split_manifest.append({
                "run_id": run_root.name,
                "fold": fold,
                "stress_split": specification["stress_split"],
                "definition": specification["definition"],
                "source_indices_json": json.dumps(specification["source_indices"].tolist()),
                "status": "complete",
                "candidate_rows": len(candidate),
                "heldout_rows": len(evaluation_index),
                "fit_rows_after_overlap_exclusion": len(fit_index),
                "overlap_radius_frames": 80,
            })
            print(
                f"PSEUDO_OOD_DONE fold={fold} split={specification['stress_split']} "
                f"heldout={len(evaluation_index)} fit={len(fit_index)}",
                flush=True,
            )
    atomic_csv(run_root / "pseudo_ood_stress_scores.csv", pd.DataFrame(rows))
    atomic_csv(run_root / "pseudo_ood_stress_manifest.csv", pd.DataFrame(split_manifest))
    update_status(run_root, "pseudo_ood_stress", "complete", rows=len(rows), splits=len(split_manifest))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_pseudo_ood(args.run_root.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
