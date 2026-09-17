from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.runner import _split_indices


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results/query_ood_robustness_20260901"
FOLD_RUN = ROOT / "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827"
DATA_PATH = ROOT / "SBTG/data/Head_Activity_OH16230.mat"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def update_status(run_root: Path, stage: str, status: str, **details: Any) -> None:
    path = run_root / "status.json"
    current = json.loads(path.read_text()) if path.exists() else {"stages": {}}
    current["updated_utc"] = datetime.now(timezone.utc).isoformat()
    current["stages"][stage] = {
        "status": status,
        "updated_utc": current["updated_utc"],
        **details,
    }
    atomic_json(path, current)


def _source_hashes() -> dict[str, str]:
    paths = list((ROOT / "query_ood_robustness").rglob("*.py"))
    paths += [
        ROOT / "compatibility_neural_benchmark/progressive_smc.py",
        ROOT / "compatibility_neural_benchmark/core.py",
        ROOT / "conditional_neural_benchmark/data.py",
        ROOT / "conditional_neural_benchmark/models.py",
        ROOT / "conditional_neural_benchmark/runner.py",
        ROOT / "conditional_neural_benchmark/distribution_structure_scoring.py",
        ROOT / "history_tangent_benchmark/src/history_tangent_benchmark/models.py",
    ]
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(set(paths))
        if path.exists()
    }


def build_protocol() -> dict[str, Any]:
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    reusable = {
        "flow": {
            "source": "results/conditional_distribution_benchmark/chemical_encoding_primary_20260828_oh16230",
            "phase": "chemical_full_cv",
            "model_id": "stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01",
        },
        "gaussian_rank8": {
            "source": "results/distribution_structure_20260831",
            "phase": "matched_full_cv",
            "model_id": "matched_gaussian_rank8",
        },
        "student_t_rank8": {
            "source": "results/distribution_structure_20260831",
            "phase": "matched_full_cv",
            "model_id": "matched_student_t_rank8",
        },
        "gaussian_diagonal": {
            "source": "results/distribution_structure_20260831",
            "phase": "matched_full_cv",
            "model_id": "matched_gaussian_diag",
        },
    }
    checkpoint_hashes: list[dict[str, Any]] = []
    for family, item in reusable.items():
        for fold in range(5):
            for seed in (1701, 2903):
                path = (
                    ROOT / item["source"] / "checkpoints" / item["phase"]
                    / f"{item['model_id']}__L80__f{fold}__s{seed}.pt"
                )
                checkpoint_hashes.append(
                    {
                        "family": family,
                        "fold": fold,
                        "seed": seed,
                        "path": str(path),
                        "sha256": sha256(path),
                    }
                )
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ROOT.name,
        "mission": "progressive-bridge SMC model and OOD robustness",
        "claim_boundary": {
            "neural": "model-implied query-conditioned predictive effect on observed calcium; not causal, anatomical, synaptic, or intervention evidence",
            "synthetic": "interventional language only for explicitly paired simulator interventions",
        },
        "dataset": {
            "cohort_mode": "oh16230_head",
            "coverage": 0.90,
            "n_worms": cohort.n_worms,
            "n_neurons": cohort.n_neurons,
            "worm_ids": list(cohort.worm_ids),
            "neurons": list(cohort.neurons),
            "fps": cohort.fps,
            "history_frames": 80,
            "legacy_tcn_receptive_field_frames": 31,
            "horizon_frames": 1,
            "residual_target": True,
            "stimulus_encoding": "binary_any_stimulus",
            "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
            "data_path": str(DATA_PATH),
            "data_sha256": sha256(DATA_PATH),
            "fold_file": str(FOLD_RUN / "fold_assignments.csv"),
            "fold_sha256": sha256(FOLD_RUN / "fold_assignments.csv"),
            "scaling": "per-neuron mean/SD fit on training worms only",
        },
        "folds": {
            "selection": [0, 1, 2],
            "confirmation": [3, 4],
            "full_after_freeze": [0, 1, 2, 3, 4],
            "validation_rule": "(test_fold + 1) modulo 5",
        },
        "models": {
            "primary": [
                "flow", "autoregressive_mdn4", "conditional_edm_diffusion",
                "student_t_rank8", "gaussian_rank8", "gaussian_diagonal",
            ],
            "model_seeds": [1701, 2903],
            "new_configs": {
                "autoregressive_mdn4": {
                    "encoder": "tcn", "width": 128, "dropout": 0.15,
                    "head": "autoregressive_mdn",
                    "head_params": {"hidden": 64, "layers": 2, "components": 4},
                    "learning_rate": 0.0003, "weight_decay": 0.00075,
                    "history_noise_std": 0.01, "history_noise_copies": 1,
                },
                "conditional_edm_diffusion": {
                    "encoder": "tcn", "width": 128, "dropout": 0.15,
                    "head": "conditional_edm_diffusion",
                    "head_params": {
                        "hidden": 128, "layers": 4, "sample_steps": 24,
                        "sigma_min": 0.01, "sigma_max": 3.0,
                        "sigma_data": 0.7, "p_mean": -0.8,
                        "p_std": 1.2, "rho": 7.0,
                    },
                    "learning_rate": 0.0003, "weight_decay": 0.00075,
                    "history_noise_std": 0.01, "history_noise_copies": 1,
                },
            },
            "sensitivities": {
                "mdn4_permutation_seeds": [0, 101, 202],
                "mdn8_permutation_seed": 0,
                "mdn_sensitivity_model_seeds": [1701],
                "mdn_sensitivity_folds": [0, 1, 2, 3, 4],
                "flow_sampler_steps": [12, 24, 48],
                "diffusion_sampler_steps": [12, 24, 48],
                "sampler_sensitivity_model_seed": 1701,
                "sampler_sensitivity_sampling_seed": 731,
            },
            "max_epochs": 50,
            "early_stopping_patience": 9,
            "batch_size": 256,
            "gradient_clip": 1.0,
            "reusable": reusable,
            "reusable_checkpoint_hashes": checkpoint_hashes,
        },
        "predictive_evaluation": {
            "heldout_rows_per_worm_stratum": 8,
            "sample_count": 64,
            "sampling_seeds": [731, 1871],
            "metrics": [
                "energy", "stimulus_balanced_energy", "marginal_crps",
                "variogram", "innovation_variogram", "sample_mean_rmse",
                "coverage90", "interval_width90", "tail_brier",
                "predicted_event_frequency", "observed_event_frequency",
                "nll_where_exact", "training_time", "sampling_throughput",
            ],
            "qualification": {
                "qualified": "energy <= 1.10 times best, finite/stable, RMSE <= 1.10 times best, absolute coverage error <= 0.08, positive finite interval width, tail Brier <= best + 0.02",
                "calibration_caveat": "energy/RMSE gates pass but coverage, interval-width, or tail-calibration gate fails",
                "not_qualified": "fails energy/RMSE or numerical stability",
            },
        },
        "query_taxonomy": [
            "supported_history_common_event",
            "supported_history_rare_event",
            "naturally_rare_coherent_history",
            "extrapolated_dynamically_coherent_history",
            "amplitude_matched_temporally_incoherent_history",
            "population_incoherent_history",
            "extreme_incoherent_history",
        ],
        "query_amplitude_grid": [
            "within_support", "q90", "q95", "q99", "training_max",
            "1.10x_max_deviation", "1.25x_max_deviation",
        ],
        "query_construction": {
            "smooth_template_frames": 8,
            "coherent": "cosine-ramp source trajectory ending at requested standardized amplitude",
            "local_manifold": "training-only local PCA direction; supplementary",
            "temporal_incoherent": "terminal spike plus matched-norm alternating preterminal edit",
            "population_incoherent": "stimulus-matched distant population donor with source trajectory preserved",
            "pair_invariants": [
                "identical terminal source value", "identical stimulus",
                "matched perturbation norm", "factual immutability", "fixed-seed determinism",
            ],
        },
        "support": {
            "primary": "mean kNN distance in scaled handcrafted plus train-only PCA representation",
            "k": 10,
            "pca_cap": 64,
            "calibration": "validation-fold factual histories only",
            "metrics": [
                "history_support_value", "amplitude_percentile",
                "max_derivative_percentile", "curvature_percentile",
                "nearest_history_distance", "distance_source_removed",
                "population_state_residual", "stimulus_incompatibility",
                "corrupted_classifier_probability", "fraction_outside_training_quantiles",
            ],
        },
        "smc": {
            "neural_primary_estimand": "existing low-versus-high compatibility-aware repaired-path response",
            "neural_primary_sensitivity": {
                "repair_frames": 4,
                "source_window_frames": 4,
                "anchor_rank": 12,
                "anchor_lambda": 0.25,
                "horizon_frames": [1],
                "fixed_factual_cuts": 7,
                "cut_selection": "support-quantile-spaced heldout factual cuts, before examining effects",
                "branch_factor": 2,
                "future_branch_factor": 2,
                "model_seed": 1701,
            },
            "analytic_synthetic_estimand": "soft-event-conditioned response under p(y|h) exp(Phi_q(y))",
            "particle_counts": [128, 256, 512, 1024],
            "sampling_seeds": [3101, 3109, 3121, 3137],
            "bridge_schedule": "adaptive largest beta increment retaining 70% ESS",
            "resampling_threshold": 0.70,
            "rejuvenation": "two sampler-only independence-MH proposals after adaptive resampling",
            "comparators": [
                "direct_ancestral", "direct_importance", "terminal_smc",
                "rejection_when_feasible", "exact_oracle", "oracle_monte_carlo",
            ],
            "success": "finite; final beta 1; particle-doubling change <= max(0.10 normalized units, 2 pooled MCSE); minimum ESS fraction >=0.20; unique ancestors >=10%; no trend over two largest N",
            "partial_support": "finite and stable but one ESS/genealogy gate fails",
            "failure": "nonfinite, endpoint failure, severe collapse, or material trend at largest N",
            "healthy_is_not_model_valid": True,
        },
        "synthetic": {
            "analytic_controls": 8,
            "var_mechanisms": 8,
            "confirmation_generator_seeds": list(range(10)),
            "model_seeds": [7001, 7009],
            "fhn_nodes": 8,
            "fhn_dt": 0.02,
            "fhn_sampling_seconds": 0.24,
            "analytic_oracle_reference_draws": 500000,
            "var_oracle_reference_draws": 150000,
            "fhn_oracle_reference_draws": 20000,
            "no_configuration_changes_after_confirmation_starts": True,
        },
        "primary_hypotheses": {
            "H1": "progressive SMC lowers rare-query oracle RMSE versus matched-cost direct importance sampling",
            "H2": "progressive SMC converges to model-specific oracle effect with particle count",
            "H3": "supported NeuroPAL model disagreement is no larger than declared equivalence margin based on within-family seed variation",
            "H4": "between-model disagreement rises as empirical support falls",
            "H5": "matched-amplitude coherent histories have better support than incoherent histories",
            "H6": "support distance and model disagreement predict synthetic oracle error better than ESS alone",
            "H7": "at least one OOD synthetic query is SMC-healthy but has large learned-model error",
        },
        "statistics": {
            "neural_unit": "worm; nuisance repetitions averaged within worm first",
            "synthetic_unit": "generator-system seed; nuisance repetitions averaged within system first",
            "intervals": "paired percentile bootstrap with fixed seed",
            "paired_tests": "exact paired sign-flip where feasible",
            "multiplicity": "Holm across H1-H7",
            "equivalence_margin": 0.10,
            "configuration_selection_firewall": "folds 0-2 only; folds 3-4 untouched until configs frozen",
        },
        "pseudo_ood": {
            "tail_region": "source-neuron histories above train-only q95 with overlapping windows excluded",
            "history_region": "compact train-only PCA cluster holdout",
            "stimulus_context": "onset-transition holdout when row counts permit",
            "whole_animal_atypicality": "heldout-worm median support distance relative to training worms",
            "stress_probe": "fixed-alpha multiscale ridge full-Gaussian refit; diagnostic only, with 80-frame temporal exclusion and observed next-frame scoring",
        },
        "validation_gates": [
            "query invariants", "support training-only fit", "heldout support calibration",
            "analytic Gaussian convergence", "particle sensitivity",
            "all requested result schemas nonempty", "existing and new tests pass",
            "artifact replay", "source/checkpoint/output hashes",
        ],
        "resources": {
            "cpu_threads": 2,
            "concurrent_neural_fits": 1,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "memory_policy": "bounded sample chunks; release accelerator cache between fits",
        },
        "source_sha256": _source_hashes(),
    }
    return protocol


def freeze_protocol(run_root: Path = RUN_ROOT) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol()
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    path = run_root / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError("incompatible resume rejected: frozen protocol or source changed")
        return previous
    frozen = {
        **protocol,
        "fingerprint": fingerprint,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(path, frozen)
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    rows = []
    for fold in range(5):
        train, validation, test = _split_indices(folds, fold)
        for split, indices in (("train", train), ("validation", validation), ("test", test)):
            for index in indices:
                rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "worm_index": int(index),
                        "worm_id": cohort.worm_ids[int(index)],
                    }
                )
    atomic_csv(run_root / "fold_assignments.csv", pd.DataFrame(rows))
    update_status(run_root, "protocol", "complete", fingerprint=fingerprint)
    return frozen
