from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.runner import (
    ModelConfig,
    RunState,
    _leaderboard,
    _neural_trial,
    _split_windows,
    _update_leaderboards,
)


@dataclass(frozen=True)
class EncodingSpec:
    name: str
    encoding: str
    subject_shuffled: bool = False


ENCODINGS = (
    EncodingSpec("binary", "binary"),
    EncodingSpec("presentation_scalar", "presentation_scalar"),
    EncodingSpec("presentation_onehot", "presentation_onehot"),
    EncodingSpec("presentation_onehot_subject_shuffle", "presentation_onehot", True),
)


def _config(spec: EncodingSpec) -> ModelConfig:
    return ModelConfig(
        model_id=f"stim_{spec.name}_tcn_flow128_dropout15_jitter_p01",
        encoder="tcn",
        head="conditional_flow_matching",
        residual_target=True,
        width=128,
        dropout=0.15,
        weight_decay=7.5e-4,
        head_params={"hidden": 128, "layers": 4, "sample_steps": 24},
        history_noise_std=0.01,
        history_noise_copies=1,
    )


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _permutations(n_worms: int, seed: int = 98231) -> dict[int, tuple[int, int, int]]:
    result: dict[int, tuple[int, int, int]] = {}
    for worm in range(n_worms):
        rng = np.random.default_rng(seed + 1009 * worm)
        result[worm] = tuple(int(value) for value in rng.permutation(3))
    return result


def _is_done(state: RunState, phase: str, model_id: str, fold: int, seed: int) -> bool:
    return any(
        row.get("phase") == phase
        and row.get("model_id") == model_id
        and int(row.get("fold", -1)) == fold
        and int(row.get("seed", -1)) == seed
        and row.get("status") == "ok"
        for row in state.records
    )


def _run(
    *, state: RunState, phase: str, spec: EncodingSpec, cohort, folds: np.ndarray,
    permutations: dict[int, tuple[int, int, int]], lag: int, fold: int, seed: int,
    device: str, max_epochs: int, patience: int, eval_rows: int, n_samples: int,
    batch_size: int,
) -> None:
    config = _config(spec)
    if _is_done(state, phase, config.model_id, fold, seed):
        print(
            f"TRIAL_SKIP phase={phase} model={config.model_id} fold={fold} seed={seed}",
            flush=True,
        )
        return
    scaler, train, validation, test = _split_windows(
        cohort,
        folds,
        fold,
        lag,
        stimulus_encoding=spec.encoding,
        presentation_permutations=permutations if spec.subject_shuffled else None,
    )
    _neural_trial(
        state=state,
        phase=phase,
        config=config,
        cohort=cohort,
        lag=lag,
        fold=fold,
        seed=seed,
        train=train,
        validation=validation,
        test=test,
        scaler=scaler,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        eval_rows=eval_rows,
        n_samples=n_samples,
        batch_size=batch_size,
        trial_metadata={
            "stimulus_encoding": spec.encoding,
            "presentation_label_role": "same-odor presentation number",
            "presentation_labels_subject_shuffled": spec.subject_shuffled,
        },
    )
    _update_leaderboards(state)


def _write_report(run_dir: Path, winner_nonbinary: str | None) -> None:
    records = pd.read_csv(run_dir / "trial_metrics.csv")
    confirmation = _leaderboard(records.to_dict("records"), "presentation_confirmation")
    if confirmation.empty:
        confirmation = _leaderboard(records.to_dict("records"), "presentation_screen")
    lines = [
        "# Repetition-aware stimulus encoding experiment",
        "",
        "## Scientific interpretation",
        "",
        "The three local stimulus epochs are three presentations of the same "
        "2-butanone stimulus. Labels 1/2/3 therefore encode presentation number, "
        "not three chemical identities. All baseline samples remain zero, exactly "
        "matching the proposed active-only encoding.",
        "",
        "The scalar representation is the requested 0/1/2/3 sensitivity analysis. "
        "The one-hot representation is the primary non-ordinal alternative. A "
        "subject-wise shuffled one-hot control tests whether a consistent presentation "
        "identity carries information beyond merely adding channels.",
        "",
        "## Predictive comparison",
        "",
        "| Rank | Encoding model | Energy | SE | Stimulus-balanced energy | Variogram | Trials |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in confirmation.itertuples():
        lines.append(
            f"| {int(row.rank)} | `{row.model_id}` | {row.energy__mean:.6f} | "
            f"{row.energy__se:.6f} | {row.energy__stim_balanced__mean:.6f} | "
            f"{row.variogram__mean:.6f} | {int(row.n_trials)} |"
        )
    lines.extend([
        "",
        "Energy and variogram are proper held-out predictive-distribution scores; "
        "lower is better. They do not directly score a lag matrix. Lag-matrix "
        "reliability is a separate downstream gate using the best non-binary encoding.",
        "",
        "## Frozen downstream candidate",
        "",
        f"`{winner_nonbinary}`" if winner_nonbinary else "Not selected yet.",
        "",
        "## Leakage and claim boundary",
        "",
        "Only the currently active pulse is labeled. Inter-pulse and pre-pulse "
        "baseline remain all-zero, so the encoding does not inject a persistent trial "
        "counter or future stimulus information. A gain can indicate repeat-specific "
        "adaptation or unmodeled experimental time, but cannot be interpreted as "
        "evidence for distinct stimulus chemistry or causal connectivity.",
        "",
    ])
    (run_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    raise RuntimeError(
        "quarantined: this runner encodes epoch position and was chemically "
        "mislabeled; use conditional_neural_benchmark.chemical_encoding_runner"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-run", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--lag", type=int, default=80)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--screen-epochs", type=int, default=32)
    parser.add_argument("--screen-patience", type=int, default=6)
    parser.add_argument("--final-epochs", type=int, default=50)
    parser.add_argument("--final-patience", type=int, default=9)
    parser.add_argument("--eval-rows", type=int, default=4000)
    parser.add_argument("--screen-samples", type=int, default=12)
    parser.add_argument("--final-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fit-best-nonbinary-full-cv", action="store_true")
    parser.add_argument("--fit-both-nonbinary-full-cv", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(run_dir, time.monotonic() + args.hours * 3600)
    metrics_path = run_dir / "trial_metrics.csv"
    if args.resume and metrics_path.exists():
        state.records = pd.read_csv(metrics_path).replace({np.nan: None}).to_dict("records")

    cohort = load_cohort(args.coverage)
    evidence_run = args.evidence_run.resolve()
    folds = _load_folds(cohort, evidence_run)
    permutations = _permutations(cohort.n_worms)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        _write_json(manifest_path, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "same-odor presentation-identity encoding experiment",
            "scientific_correction": "three repeats of 2-butanone, not three distinct stimuli",
            "lag_frames": args.lag,
            "fps": cohort.fps,
            "screen_folds": [0, 1, 2],
            "confirmation_folds": [3, 4],
            "screen_seed": 1701,
            "confirmation_seeds": [1701, 2903],
            "full_cv_seeds": [1701, 2903],
            "encodings": [spec.__dict__ for spec in ENCODINGS],
            "subject_shuffle_permutations": {
                cohort.worm_ids[index]: list(value)
                for index, value in permutations.items()
            },
            "candidate_configs": [_config(spec).to_dict() for spec in ENCODINGS],
            "fold_assignments": str(evidence_run / "fold_assignments.csv"),
            "fold_assignments_sha256": _sha256(evidence_run / "fold_assignments.csv"),
            "n_worms": cohort.n_worms,
            "n_neurons": cohort.n_neurons,
            "neurons": list(cohort.neurons),
            "device": args.device,
            "torch": torch.__version__,
            "python": platform.python_version(),
        })

    for fold in (0, 1, 2):
        for spec in ENCODINGS:
            if state.can_start(8):
                _run(
                    state=state, phase="presentation_screen", spec=spec,
                    cohort=cohort, folds=folds, permutations=permutations,
                    lag=args.lag, fold=fold, seed=1701, device=args.device,
                    max_epochs=args.screen_epochs, patience=args.screen_patience,
                    eval_rows=args.eval_rows, n_samples=args.screen_samples,
                    batch_size=args.batch_size,
                )

    for fold in (3, 4):
        for spec in ENCODINGS:
            for seed in (1701, 2903):
                if state.can_start(8):
                    _run(
                        state=state, phase="presentation_confirmation", spec=spec,
                        cohort=cohort, folds=folds, permutations=permutations,
                        lag=args.lag, fold=fold, seed=seed, device=args.device,
                        max_epochs=args.final_epochs, patience=args.final_patience,
                        eval_rows=args.eval_rows, n_samples=args.final_samples,
                        batch_size=args.batch_size,
                    )

    confirmation = _leaderboard(state.records, "presentation_confirmation")
    nonbinary_ids = {
        _config(spec).model_id for spec in ENCODINGS
        if spec.name in {"presentation_scalar", "presentation_onehot"}
    }
    nonbinary = confirmation[confirmation.model_id.isin(nonbinary_ids)]
    winner_nonbinary = str(nonbinary.iloc[0].model_id) if len(nonbinary) else None
    winner_spec = next(
        (spec for spec in ENCODINGS if _config(spec).model_id == winner_nonbinary), None
    )
    full_cv_specs: list[EncodingSpec] = []
    if args.fit_best_nonbinary_full_cv and winner_spec is not None:
        full_cv_specs.append(winner_spec)
    if args.fit_both_nonbinary_full_cv:
        full_cv_specs = [
            spec for spec in ENCODINGS
            if spec.name in {"presentation_scalar", "presentation_onehot"}
        ]
    if full_cv_specs and winner_spec is not None:
        _write_json(run_dir / "winner_nonbinary.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": winner_nonbinary,
            "stimulus_encoding": winner_spec.encoding,
            "selection_phase": "presentation_confirmation",
            "selection_rule": "frozen leaderboard order among scalar and one-hot, excluding shuffle control",
            "confirmation_leaderboard": confirmation.to_dict("records"),
        })
        for spec in full_cv_specs:
            for fold in range(5):
                for seed in (1701, 2903):
                    if state.can_start(8):
                        _run(
                            state=state, phase="presentation_full_cv", spec=spec,
                            cohort=cohort, folds=folds, permutations=permutations,
                            lag=args.lag, fold=fold, seed=seed, device=args.device,
                            max_epochs=args.final_epochs, patience=args.final_patience,
                            eval_rows=args.eval_rows, n_samples=args.final_samples,
                            batch_size=args.batch_size,
                        )

    _update_leaderboards(state)
    _write_report(run_dir, winner_nonbinary)
    expected = 12 + 16 + 10 * len(full_cv_specs)
    completed = sum(row.get("status") == "ok" for row in state.records)
    _write_json(run_dir / "validation.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if completed == expected else "partial",
        "expected_trials": expected,
        "completed_trials": completed,
        "failed_trials": sum(row.get("status") == "failed" for row in state.records),
        "winner_nonbinary": winner_nonbinary,
    })


if __name__ == "__main__":
    main()
