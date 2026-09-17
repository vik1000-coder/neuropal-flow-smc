"""Build the full historical-80 atlas with the converged single-seed flow.

This module deliberately reuses the previously audited dense-atlas aggregation
and external-analysis implementation while freezing a new raw-sampling protocol:
the held-out-score-selected ``flow_lr6e4`` generator, seed 1701, N=64 particles,
repair branch factor 4, and two future descendants.  The historical SBTG80
cohort remains donor-imputed and pseudo-paired; outputs are model-relative
response sensitivities, not causal or anatomical effects.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark import sbtg80_full_progressive_atlas as base
from compatibility_neural_benchmark.prediction_atlas_runner import sha256
from conditional_neural_benchmark.data import load_sbtg_cohort


MODEL_ID = "flow_lr6e4"
GENERATOR_SEED = 1701
PARTICLES = 64
BRANCH_FACTOR = 4
FUTURE_BRANCH_FACTOR = 2
MIN_ESS = 12.0
BASE_SEED = 20_260_904
DEFAULT_OUTPUT = Path("results/sbtg80_optimized_full_atlas_20260902")
DEFAULT_SOURCE_RUN = Path("results/sbtg80_flow_optimization_20260901/training")
DEFAULT_REFERENCE_RELEASE = Path("/Users/vik/Downloads/SBTG-public-release copy")
DEFAULT_PUBLISHED_ARCHIVE = (
    DEFAULT_REFERENCE_RELEASE / "results/paper/sbtg_lag_matrices.npz"
)

EXPECTED_MODEL_CONFIG: dict[str, Any] = {
    "model_id": MODEL_ID,
    "encoder": "tcn",
    "head": "conditional_flow_matching",
    "residual_target": True,
    "width": 128,
    "dropout": 0.10,
    "head_params": {"hidden": 128, "layers": 4, "sample_steps": 20},
    "learning_rate": 6e-4,
    "weight_decay": 1e-4,
    "training_scheme": "natural",
    "history_noise_std": 0.0,
    "history_noise_copies": 0,
}

_ORIGINAL_RUN_ONE = base.run_one
_ORIGINAL_RAW_MANIFEST = base._raw_manifest
_ORIGINAL_VALIDATE_ARCHIVE = base._validate_archive
_CONFIGURED = False


def _checkpoint_config(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    found = checkpoint.get("model_config")
    if found != EXPECTED_MODEL_CONFIG:
        raise RuntimeError(f"unexpected tuned checkpoint configuration: {path}")
    if int(checkpoint.get("lag", -1)) != base.HISTORY_FRAMES:
        raise RuntimeError(f"unexpected checkpoint history length: {path}")
    return found


def _source_provenance(source_run: Path) -> dict[str, object]:
    manifest_path = source_run / "manifest.json"
    winner_path = source_run / "winner_selection.json"
    validation_path = source_run / "validation.json"
    for path in (manifest_path, winner_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    winner = json.loads(winner_path.read_text())
    validation = json.loads(validation_path.read_text())
    if (
        manifest.get("cohort_mode") != "historical_sbtg_full_traces_imputed"
        or int(manifest.get("worms", -1)) != 20
        or int(manifest.get("neurons", -1)) != 80
        or int(manifest.get("history_frames", -1)) != base.HISTORY_FRAMES
        or int(manifest.get("train_seed", -1)) != GENERATOR_SEED
        or manifest.get("external_references_consulted") is not False
    ):
        raise RuntimeError("optimization source manifest does not match the frozen protocol")
    if winner.get("model_id") != MODEL_ID:
        raise RuntimeError("optimization winner is not flow_lr6e4")
    if (
        validation.get("status") != "passed"
        or validation.get("winner") != MODEL_ID
        or int(validation.get("failed_trials", -1)) != 0
        or validation.get("external_references_consulted") is not False
    ):
        raise RuntimeError("optimization source validation did not pass")
    checkpoints: dict[str, Path] = {}
    for fold in range(5):
        path = base._checkpoint_path(source_run, fold)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _checkpoint_config(path)
        if int(checkpoint.get("fold", -1)) != fold:
            raise RuntimeError(f"checkpoint fold mismatch: {path}")
        if int(checkpoint.get("seed", -1)) != GENERATOR_SEED:
            raise RuntimeError(f"checkpoint seed mismatch: {path}")
        checkpoints[str(fold)] = path
    return {
        "path": str(source_run.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "winner_selection_sha256": sha256(winner_path),
        "validation_sha256": sha256(validation_path),
        "checkpoint_sha256": {
            fold: sha256(path) for fold, path in checkpoints.items()
        },
        "model_config": EXPECTED_MODEL_CONFIG,
        "selection_metric": "held-out stimulus-balanced energy, then energy",
        "selection_external_references_consulted": False,
    }


def _run_one_best(**kwargs: Any) -> dict[str, object]:
    kwargs["progressive_branch_factor"] = BRANCH_FACTOR
    kwargs["progressive_future_branch_factor"] = FUTURE_BRANCH_FACTOR
    return _ORIGINAL_RUN_ONE(**kwargs)


def _raw_manifest(output: Path, source_run: Path, cohort) -> dict[str, object]:
    manifest = _ORIGINAL_RAW_MANIFEST(output, source_run, cohort)
    manifest.update(
        {
            "protocol": "optimized full historical SBTG80 progressive-bridge atlas v2",
            "model_id": MODEL_ID,
            "generator_seeds": [GENERATOR_SEED],
            "generator_seed_limitation": (
                "single seed 1701 selected from held-out predictive scores; worm intervals "
                "are conditional on this fitted generator. A separate two-seed sensitivity "
                "is reported but is not used for this dashboard atlas"
            ),
            "particles": PARTICLES,
            "minimum_effective_sample_size": MIN_ESS,
            "progressive_branch_factor": BRANCH_FACTOR,
            "progressive_future_branch_factor": FUTURE_BRANCH_FACTOR,
            "base_seed": BASE_SEED,
            "single_seed_rationale": (
                "seed 1701 had the better held-out predictive energy; selection did not use "
                "Randi, Cook, Bentley, or published-SBTG correspondence"
            ),
        }
    )
    return manifest


def _validate_archive(*args: Any, **kwargs: Any) -> dict[str, object]:
    row = _ORIGINAL_VALIDATE_ARCHIVE(*args, **kwargs)
    path = Path(args[0] if args else kwargs["path"])
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "progressive_branch_factor",
            "progressive_future_branch_factor",
        )
        if any(name not in archive.files for name in required):
            raise RuntimeError(f"sampler-branch metadata missing from {path}")
        if int(archive["progressive_branch_factor"]) != BRANCH_FACTOR:
            raise RuntimeError(f"repair branch factor mismatch in {path}")
        if (
            int(archive["progressive_future_branch_factor"])
            != FUTURE_BRANCH_FACTOR
        ):
            raise RuntimeError(f"future branch factor mismatch in {path}")
    row["progressive_branch_factor"] = BRANCH_FACTOR
    row["progressive_future_branch_factor"] = FUTURE_BRANCH_FACTOR
    return row


def configure_base() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    base.MODEL_ID = MODEL_ID
    base.GENERATOR_SEED = GENERATOR_SEED
    base.PARTICLES = PARTICLES
    base.MIN_ESS = MIN_ESS
    base.BASE_SEED = BASE_SEED
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.DEFAULT_SOURCE_RUN = DEFAULT_SOURCE_RUN
    base._source_provenance = _source_provenance
    base._raw_manifest = _raw_manifest
    base._validate_archive = _validate_archive
    base.run_one = _run_one_best
    _CONFIGURED = True


def _seed_choice(output: Path) -> dict[str, object]:
    campaign = Path("results/sbtg80_flow_optimization_20260901")
    trial_paths = {
        "1701": campaign / "training/trial_metrics.csv",
        "2903": campaign / "training_seed2903/trial_metrics.csv",
    }
    model_ids = {"1701": "flow_lr6e4", "2903": "flow_lr6e4_s2903"}
    predictive: dict[str, object] = {}
    for seed, path in trial_paths.items():
        frame = pd.read_csv(path)
        chosen = frame[(frame.status == "ok") & (frame.model_id == model_ids[seed])]
        if len(chosen) != 5:
            raise RuntimeError(f"incomplete five-fold predictive record for seed {seed}")
        predictive[seed] = {
            "mean_energy": float(chosen.energy.mean()),
            "mean_balanced_energy": float(chosen["energy__stim_balanced"].mean()),
            "mean_variogram": float(chosen.variogram.mean()),
        }
    current = pd.read_csv(
        output / "external_reference_checks/primary_lag1_comparison.csv"
    )
    current = current[
        (current.method == base.METHOD) & (current.scope == "all_estimated")
    ]
    ensemble = pd.read_csv(campaign / "final/ensemble_primary_metrics.csv")
    ensemble = ensemble[
        (ensemble.evaluation_axis == "historical80")
        & (ensemble.scope == "all_estimated")
    ]
    labels = ["randi_wild_type", "cook_struct_80", "cook_chem_80", "cook_gap_80"]
    single_scores = {
        name: float(current.loc[current.reference == name, "auroc"].iloc[0])
        for name in labels
    }
    ensemble_scores = {
        name: float(ensemble.loc[ensemble.reference == name, "auroc"].iloc[0])
        for name in labels
    }
    decision = {
        "selected_generator_seed": GENERATOR_SEED,
        "selection_basis": (
            "lower held-out predictive energy and balanced energy; external references "
            "were not used to select the dashboard generator"
        ),
        "predictive_metrics_by_seed": predictive,
        "single_seed_full_grid_lag1_auroc": single_scores,
        "two_seed_high_resolution_lag1_auroc": ensemble_scores,
        "two_seed_note": (
            "two-seed averaging improved all four final lag-1 AUROCs and reduced dependence "
            "on one fitted generator; it remains a documented sensitivity rather than the "
            "dashboard's primary atlas"
        ),
    }
    base._atomic_json(output / "seed_choice.json", decision)
    return decision


def _validate_lag1_regression(output: Path) -> dict[str, object]:
    """Require exact agreement with the already frozen best-resolution lag-1 run."""
    frozen_root = Path(
        "results/sbtg80_flow_optimization_20260901/sampling/"
        "flow_lr6e4__N64__b4__f2"
    ).resolve()
    frozen_manifest = json.loads((frozen_root / "manifest.json").read_text())
    expected_manifest = {
        "model_id": MODEL_ID,
        "particles": PARTICLES,
        "repair_branch_factor": BRANCH_FACTOR,
        "future_branch_factor": FUTURE_BRANCH_FACTOR,
        "integration_steps": 20,
        "horizon_frames": [1],
        "source_lag_frames": 1,
        "external_references_consulted": False,
        "status": "sampling_complete",
    }
    for key, expected in expected_manifest.items():
        if frozen_manifest.get(key) != expected:
            raise RuntimeError(f"frozen lag-1 regression manifest mismatch: {key}")
    rows: list[dict[str, object]] = []
    for fold in range(5):
        current_path = base._archive_path(output, 1, fold)
        frozen_path = (
            frozen_root
            / "responses"
            / base.METHOD
            / current_path.name
        )
        if not current_path.is_file() or not frozen_path.is_file():
            raise FileNotFoundError(f"lag-1 regression archive missing for fold {fold}")
        with np.load(current_path, allow_pickle=False) as current, np.load(
            frozen_path, allow_pickle=False
        ) as frozen:
            if current["horizon_frames"].astype(int).tolist()[0] != 1:
                raise RuntimeError(f"current lag-1 fold {fold} does not start at horizon 1")
            if frozen["horizon_frames"].astype(int).tolist() != [1]:
                raise RuntimeError(f"frozen lag-1 fold {fold} has unexpected horizons")
            for key in base.RESPONSE_KEYS:
                if not np.array_equal(current[key][..., 0, :], frozen[key][..., 0, :]):
                    raise RuntimeError(
                        f"lag-1/horizon-1 regression mismatch for fold {fold}: {key}"
                    )
            for key in (name for name in frozen.files if name.startswith("diagnostic_")):
                if key not in current.files or not np.array_equal(current[key], frozen[key]):
                    raise RuntimeError(
                        f"lag-1 diagnostic regression mismatch for fold {fold}: {key}"
                    )
        rows.append(
            {
                "fold": fold,
                "current_sha256": sha256(current_path),
                "frozen_sha256": sha256(frozen_path),
                "seven_response_channels_exact_at_horizon_1": True,
                "all_sampler_diagnostics_exact": True,
            }
        )
    result = {
        "status": "pass",
        "comparison": "bit-for-bit against frozen flow_lr6e4__N64__b4__f2 lag-1 run",
        "folds": rows,
        "frozen_manifest_sha256": sha256(frozen_root / "manifest.json"),
    }
    base._atomic_json(output / "lag1_regression_validation.json", result)
    return result


def _finish_documentation(output: Path) -> None:
    decision = _seed_choice(output)
    report_path = output / "REPORT.md"
    report = report_path.read_text()
    old = (
        "Worm bootstrap intervals represent variation across the 20 held-out historical "
        "traces. They are conditional on generator seed 1701 because no second seed has "
        "complete checkpoints for all five folds. The chemical panels are event-stratified "
        "views of a binary-any-stimulus generator."
    )
    new = (
        "Worm bootstrap intervals represent variation across the 20 held-out historical "
        "traces and are conditional on generator seed 1701. Seed 1701 was chosen from "
        "held-out predictive scores, without external-reference selection. A separate "
        "two-seed high-resolution sensitivity improved all four lag-1 correspondence "
        "scores and is retained in `seed_choice.json`; it is not included in these worm "
        "intervals. The chemical panels are event-stratified views of a binary-any-stimulus "
        "generator."
    )
    if old not in report:
        raise RuntimeError("expected uncertainty paragraph missing from generated report")
    report = report.replace(old, new)
    insertion = (
        "\n## Single-seed dashboard decision\n\n"
        f"Seed 1701 is used because its five-fold held-out energy was "
        f"{decision['predictive_metrics_by_seed']['1701']['mean_energy']:.6f}, better than "
        f"seed 2903 at {decision['predictive_metrics_by_seed']['2903']['mean_energy']:.6f}. "
        "Both single seeds beat the published SBTG lag-1 AUROC point estimates on all four "
        "prespecified references. The two-seed ensemble remains the more stable sensitivity "
        "and is shown in the dashboard reference view.\n"
    )
    report = report.replace("\n## Head and tail recording-origin summary\n", insertion + "\n## Head and tail recording-origin summary\n")
    report_path.write_text(report)

    code_dir = output / "code"
    code_dir.mkdir(exist_ok=True)
    for source in (
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        Path("compatibility_neural_benchmark/prediction_atlas_runner.py").resolve(),
    ):
        shutil.copy2(source, code_dir / source.name)
    reproduce = f"""#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=${{1:-../..}}
cd "$PROJECT_ROOT"
./.venv/bin/python -m compatibility_neural_benchmark.sbtg80_optimized_full_atlas \\
  --output {output} --overwrite-analysis
"""
    path = output / "reproduce.sh"
    path.write_text(reproduce)
    path.chmod(0o755)
    (output / "RUNBOOK.md").write_text(
        "# Reproduce the optimized historical-80 atlas\n\n"
        "Run `./reproduce.sh /absolute/path/to/new_sbtg_neuro`. The raw stage is "
        "resume-safe and rejects checkpoint, particle, lag, horizon, or branch-setting "
        "mismatches. Seed 1701 is the held-out-score-selected generator. See "
        "`seed_choice.json` for the separate two-seed sensitivity.\n"
    )
    base.audit_final_output(output)
    base._write_checksums(output)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--reference-release", type=Path, default=DEFAULT_REFERENCE_RELEASE
    )
    parser.add_argument(
        "--published-archive", type=Path, default=DEFAULT_PUBLISHED_ARCHIVE
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--bootstrap-replicates", type=int, default=256)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--paired-source-bootstrap", type=int, default=4096)
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite-analysis", action="store_true")
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    configure_base()
    output = args.output.resolve()
    source_run = args.source_run.resolve()
    reference_release = args.reference_release.resolve()
    published_archive = args.published_archive.resolve()
    torch.set_num_threads(2)
    load_sbtg_cohort()  # fail before a long run if the historical cache is unavailable
    _source_provenance(source_run)
    if args.validate_only:
        validation = base.validate_raw(output, source_run)
        if validation["status"] == "pass":
            _validate_lag1_regression(output)
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    if args.analyze_only:
        validation = base.validate_raw(output, source_run)
        if validation["status"] != "pass":
            raise RuntimeError("analysis refused because the raw archive grid is incomplete")
    else:
        validation = base.run_raw(
            output,
            source_run,
            device=args.device,
            max_archives=args.max_archives,
        )
        if args.max_archives is not None and validation["status"] != "pass":
            print(json.dumps(validation, indent=2, sort_keys=True))
            return
        if validation["status"] != "pass":
            raise RuntimeError("raw sampling did not produce a complete validated archive grid")
    _validate_lag1_regression(output)
    if args.raw_only:
        return
    base.build_atlas(
        output,
        source_run,
        bootstrap_replicates=args.bootstrap_replicates,
        overwrite=args.overwrite_analysis,
    )
    base.run_external_metrics(
        output,
        reference_release=reference_release,
        published_archive=published_archive,
        permutations=args.permutations,
        bootstrap_repeats=args.paired_source_bootstrap,
        overwrite=args.overwrite_analysis,
    )
    base.write_report(output)
    _finish_documentation(output)
    print(json.dumps(json.loads((output / "manifest.json").read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
