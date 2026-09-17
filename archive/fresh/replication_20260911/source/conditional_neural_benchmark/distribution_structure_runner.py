"""Frozen, corrected-cohort comparison of predictive dependence and shape."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.chemical_encoding_runner import ENCODINGS, config_for, sha256
from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.runner import RunState, _neural_trial, _split_indices, _split_windows

ROOT = Path(__file__).resolve().parents[1]
FLOW_RUN = ROOT / "results/conditional_distribution_benchmark/chemical_encoding_primary_20260828_oh16230"
FOLD_RUN = ROOT / "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827"
DEFAULT_RUN = ROOT / "results/distribution_structure_20260831"
FLOW_ID = config_for(ENCODINGS[0]).model_id


def candidate_configs():
    base = config_for(ENCODINGS[0])
    return [replace(base, model_id=name, head=head,
                    head_params={"hidden": 128, "layers": 4, "rank": rank})
            for name, head, rank in (
                ("matched_gaussian_diag", "structure_gaussian", 0),
                ("matched_gaussian_rank8", "structure_gaussian", 8),
                ("matched_student_t_rank8", "structure_student_t", 8))]


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def flow_checkpoint(fold, seed):
    return FLOW_RUN / "checkpoints/chemical_full_cv" / f"{FLOW_ID}__L80__f{fold}__s{seed}.pt"


def freeze_protocol(run_dir: Path, *, smoke: bool = False):
    run_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    run_folds, seeds = ([0], [1701]) if smoke else (list(range(5)), [1701, 2903])
    protocol = {
        "schema": 1, "experiment": "conditional marginal uncertainty versus dependence and shape",
        "cohort_mode": "oh16230_head", "n_worms": cohort.n_worms,
        "neurons": list(cohort.neurons), "worm_ids": list(cohort.worm_ids),
        "fps": cohort.fps, "lag": 80, "stimulus_encoding": "binary_any_stimulus",
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "fold_file": str(FOLD_RUN / "fold_assignments.csv"),
        "fold_sha256": sha256(FOLD_RUN / "fold_assignments.csv"),
        "folds": run_folds, "seeds": seeds, "max_epochs": 2 if smoke else 50,
        "patience": 2 if smoke else 9, "batch_size": 256, "device": "mps",
        "training_samples_for_legacy_metrics": 8 if smoke else 32,
        "training_eval_rows": 160 if smoke else 4000,
        "evaluation": {"samples": 128, "shuffle_repeats": 8,
                       "sample_seeds": [731, 1871], "all_heldout_histories": True,
                       "primary_horizon_frames": 1,
                       "variogram_target": "absolute next activity; innovation-only variogram reported separately",
                       "particle_sensitivity": {"samples": [128, 256], "per_worm_stratum": 16,
                                                "shuffle_repeats": 4, "sample_seed": 731},
                       "primary_metrics": ["energy", "variogram", "crps"],
                       "secondary_metrics": ["rmse", "coverage90", "sharpness90", "tail_brier"],
                       "worm_weighting": "equal; average model and sampling repetitions within worm first",
                       "contexts": ["all", "off", "onset", "offset", "mixed"],
                       "null": "independent sample-index permutations per history and neuron; never permute histories",
                       "tail_event": "abs(next-current) exceeds twice training-fold RMS innovation for each neuron",
                       "energy_estimator": "all distinct sample pairs (fair U-statistic)",
                       "variogram_estimator": "p=0.5, every unordered neuron pair; finite-ensemble bias corrected",
                       "marginal_crps_estimator": "fair U-statistic from sorted samples",
                       "inference": "paired worm bootstrap; intervals conditional on overlapping fitted CV models, not independent confirmation",
                       "primary_test_family": ["flow_shuffle_energy", "flow_shuffle_variogram",
                                               "changing_variance_crps", "student_vs_gaussian_crps",
                                               "flow_vs_student_energy"],
                       "multiplicity": "two-sided exact worm sign-flip p-values; Holm adjustment across five primary tests",
                       "fixed_variance_control": "same fitted diagonal-Gaussian mean; per-neuron training-residual RMS replaces conditional SD"},
        "models": [config.to_dict() for config in candidate_configs()],
        "frozen_flow_config": config_for(ENCODINGS[0]).to_dict(),
        "frozen_flow_checkpoints": [dict(path=str(flow_checkpoint(f, s)), sha256=sha256(flow_checkpoint(f, s)))
                                    for f in run_folds for s in seeds],
        "training_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
            ROOT / "conditional_neural_benchmark/distribution_structure_models.py",
            ROOT / "conditional_neural_benchmark/distribution_structure_runner.py",
            ROOT / "conditional_neural_benchmark/models.py",
            ROOT / "conditional_neural_benchmark/runner.py",
            ROOT / "conditional_neural_benchmark/data.py",
            ROOT / "history_tangent_benchmark/src/history_tangent_benchmark/models.py"]},
        "interpretation": "matched architecture and optimization budget, independently fitted encoders/means; not exact parameter-count or mean-matched shape attribution",
        "selection_firewall": "no anatomical, functional or molecular atlas; fixed configurations, no heldout tuning",
        "smoke": smoke,
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = run_dir / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["fingerprint"] != fingerprint:
            raise RuntimeError("resume rejected: protocol, checkpoint or training source changed")
    else:
        atomic_json(path, dict(protocol, fingerprint=fingerprint, frozen_utc=datetime.now(timezone.utc).isoformat()))
        membership = []
        for fold in run_folds:
            for split, indices in zip(("train", "validation", "test"), _split_indices(folds, fold)):
                membership += [dict(fold=fold, split=split, worm_id=cohort.worm_ids[i]) for i in indices]
        pd.DataFrame(membership).to_csv(run_dir / "fold_membership.csv", index=False)
    return cohort, folds, protocol


def train(run_dir: Path, smoke: bool):
    cohort, folds, protocol = freeze_protocol(run_dir, smoke=smoke)
    torch.set_num_threads(4)
    state = RunState(run_dir, time.monotonic() + 24 * 3600)
    if (run_dir / "trial_metrics.csv").exists():
        state.records = pd.read_csv(run_dir / "trial_metrics.csv").replace({np.nan: None}).to_dict("records")
    phase = "matched_smoke" if smoke else "matched_full_cv"
    for fold in protocol["folds"]:
        scaler, training, validation, testing = _split_windows(cohort, folds, fold, 80)
        # Check reused flow's transformation and identity against current split construction.
        for seed in protocol["seeds"]:
            stored = torch.load(flow_checkpoint(fold, seed), map_location="cpu", weights_only=False)
            if stored["neurons"] != list(cohort.neurons) or stored["stimulus_schema_fingerprint"] != cohort.stimulus_schema_fingerprint:
                raise RuntimeError("flow/cohort identity mismatch")
            for key in ("mean", "scale"):
                np.testing.assert_allclose(stored["scaler"][key], getattr(scaler, key), rtol=0, atol=0)
        for config in candidate_configs():
            for seed in protocol["seeds"]:
                matches = [r for r in state.records if r.get("phase") == phase and r.get("model_id") == config.model_id
                           and int(r.get("fold", -1)) == fold and int(r.get("seed", -1)) == seed]
                if matches:
                    if len(matches) != 1 or matches[0]["status"] != "ok":
                        raise RuntimeError("ambiguous or failed existing trial; inspect before resuming")
                    continue
                _neural_trial(state=state, phase=phase, config=config, cohort=cohort, lag=80,
                              fold=fold, seed=seed, train=training, validation=validation, test=testing,
                              scaler=scaler, device="mps", max_epochs=protocol["max_epochs"],
                              patience=protocol["patience"], eval_rows=protocol["training_eval_rows"],
                              n_samples=protocol["training_samples_for_legacy_metrics"], batch_size=256,
                              trial_metadata={"cohort_mode": "oh16230_head", "stimulus_encoding": "binary_any_stimulus"})
                if state.records[-1]["status"] != "ok":
                    raise RuntimeError("training failed; see trial_metrics.csv")
    expected = len(protocol["folds"]) * len(protocol["seeds"]) * len(candidate_configs())
    if len(state.records) != expected or any(r["status"] != "ok" for r in state.records):
        raise RuntimeError("incomplete training grid")
    atomic_json(run_dir / "training_validation.json", {"status": "pass", "expected": expected,
                "completed": len(state.records), "checkpoints": [dict(path=r["checkpoint"], sha256=sha256(run_dir / r["checkpoint"])) for r in state.records]})
    print("MATCHED_TRAINING_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=("freeze", "train"), default="train")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze_protocol(args.run_dir.resolve(), smoke=args.smoke)
    else:
        train(args.run_dir.resolve(), args.smoke)


if __name__ == "__main__":
    main()
