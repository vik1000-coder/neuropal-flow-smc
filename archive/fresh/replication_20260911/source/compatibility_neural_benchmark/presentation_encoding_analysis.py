from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.aligned_lag_analysis import (
    AlignedRun,
    animal_split_stability,
    delay_position,
    load_run,
    off_diagonal,
    phase_matrix,
    safe_corr,
)
from compatibility_neural_benchmark.evaluate import lagged_animal_signflip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predictive_analysis(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(run_dir / "trial_metrics.csv")
    frame = frame[
        (frame.phase == "presentation_confirmation") & (frame.status == "ok")
    ].copy()
    metrics = [
        "energy", "energy__stim_balanced", "variogram",
        "variogram__stim_balanced", "rmse", "coverage90", "sharpness90",
    ]
    board = frame.groupby(
        ["model_id", "stimulus_encoding", "presentation_labels_subject_shuffled"],
        dropna=False,
    )[metrics].agg(["mean", "std", "count"])
    board.columns = ["__".join(column) for column in board.columns]
    board = board.reset_index().sort_values(
        ["energy__mean", "energy__stim_balanced__mean"]
    ).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))

    ids = {
        "binary": "stim_binary_tcn_flow128_dropout15_jitter_p01",
        "scalar": "stim_presentation_scalar_tcn_flow128_dropout15_jitter_p01",
        "onehot": "stim_presentation_onehot_tcn_flow128_dropout15_jitter_p01",
        "shuffle": "stim_presentation_onehot_subject_shuffle_tcn_flow128_dropout15_jitter_p01",
    }
    comparisons = [
        ("scalar_minus_binary", ids["scalar"], ids["binary"]),
        ("onehot_minus_binary", ids["onehot"], ids["binary"]),
        ("onehot_minus_shuffle", ids["onehot"], ids["shuffle"]),
        ("scalar_minus_onehot", ids["scalar"], ids["onehot"]),
    ]
    paired: list[dict[str, object]] = []
    index = ["fold", "seed"]
    for label, left_id, right_id in comparisons:
        left = frame[frame.model_id == left_id].set_index(index)
        right = frame[frame.model_id == right_id].set_index(index)
        shared = left.index.intersection(right.index)
        for metric in metrics:
            delta = left.loc[shared, metric].to_numpy() - right.loc[shared, metric].to_numpy()
            paired.append({
                "comparison": label,
                "metric": metric,
                "n_pairs": len(delta),
                "mean_delta_left_minus_right": float(delta.mean()),
                "se_delta": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else np.nan,
                "wins_left": int((delta < 0).sum()) if metric != "coverage90" else np.nan,
            })
    return board, pd.DataFrame(paired)


def presentation_predictive_analysis(
    evaluation_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(evaluation_dir / "metrics_by_checkpoint_presentation.csv")
    aggregate = frame.groupby(
        ["model_id", "presentation"], dropna=False
    )[["energy", "variogram", "rmse"]].agg(["mean", "std", "count"])
    aggregate.columns = ["__".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    binary_id = "stim_binary_tcn_flow128_dropout15_jitter_p01"
    rows: list[dict[str, object]] = []
    binary = frame[frame.model_id == binary_id].set_index(
        ["fold", "seed", "presentation"]
    )
    for model_id in sorted(set(frame.model_id) - {binary_id}):
        model = frame[frame.model_id == model_id].set_index(
            ["fold", "seed", "presentation"]
        )
        shared = model.index.intersection(binary.index)
        for presentation in (1, 2, 3):
            use = [index for index in shared if index[2] == presentation]
            delta = model.loc[use, "energy"].to_numpy() - binary.loc[use, "energy"].to_numpy()
            rows.append({
                "model_id": model_id,
                "comparison": "model_minus_binary",
                "presentation": presentation,
                "n_pairs": len(delta),
                "mean_energy_delta": float(delta.mean()),
                "se_energy_delta": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else np.nan,
                "wins_model": int((delta < 0).sum()),
            })
    return aggregate, pd.DataFrame(rows)


def relationship_rows(runs: dict[str, AlignedRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def compare_arrays(
        comparison: str,
        left_label: str,
        right_label: str,
        left: np.ndarray,
        right: np.ndarray,
        left_run: AlignedRun,
        right_run: AlignedRun,
    ) -> None:
        left_delay = delay_position(left_run, 7)
        right_delay = delay_position(right_run, 7)
        for estimand in ("onset", "quiet", "onset_minus_quiet"):
            for horizon, seconds in enumerate(left_run.horizon_seconds):
                lm = phase_matrix(left, estimand, left_delay, horizon).mean(axis=0)
                rm = phase_matrix(right, estimand, right_delay, horizon).mean(axis=0)
                lv, rv = off_diagonal(lm), off_diagonal(rm)
                rows.append({
                    "comparison": comparison,
                    "left": left_label,
                    "right": right_label,
                    "estimand": estimand,
                    "horizon_seconds": float(seconds),
                    "n_edges": int(np.isfinite(lv * rv).sum()),
                    "pearson": safe_corr(lv, rv, "pearson"),
                    "spearman": safe_corr(lv, rv, "spearman"),
                })

    seed_runs = ["binary_direct", "encoded_direct", "encoded_progressive"]
    if "scalar_direct" in runs:
        seed_runs.append("scalar_direct")
    for name in seed_runs:
        run = runs[name]
        seeds = sorted(run.coefficient_by_seed)
        if len(seeds) >= 2:
            compare_arrays(
                "generator_seed",
                f"{name}_seed_{seeds[0]}",
                f"{name}_seed_{seeds[1]}",
                run.coefficient_by_seed[seeds[0]],
                run.coefficient_by_seed[seeds[1]],
                run,
                run,
            )

    compare_arrays(
        "sampler",
        "encoded_direct",
        "encoded_progressive",
        runs["encoded_direct"].coefficient,
        runs["encoded_progressive"].coefficient,
        runs["encoded_direct"],
        runs["encoded_progressive"],
    )
    compare_arrays(
        "sampler",
        "binary_direct",
        "binary_progressive",
        runs["binary_direct"].coefficient,
        runs["binary_progressive"].coefficient,
        runs["binary_direct"],
        runs["binary_progressive"],
    )
    compare_arrays(
        "encoding",
        "encoded_direct",
        "binary_direct",
        runs["encoded_direct"].coefficient,
        runs["binary_direct"].coefficient,
        runs["encoded_direct"],
        runs["binary_direct"],
    )
    if "scalar_direct" in runs:
        compare_arrays(
            "encoding",
            "scalar_direct",
            "binary_direct",
            runs["scalar_direct"].coefficient,
            runs["binary_direct"].coefficient,
            runs["scalar_direct"],
            runs["binary_direct"],
        )
    return pd.DataFrame(rows)


def _event_arrays(run_dir: Path, pattern: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    paths = sorted((run_dir / "responses").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no response files matching {run_dir / 'responses' / pattern}")
    by_seed: dict[int, list[np.ndarray | None]] = {}
    horizons = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            seed = int(data["model_seed"])
            response = data["response_cumulative_mean"].astype(np.float64)
            gap = data["diagnostic_achieved_gap"].astype(np.float64)
            normalized = response / np.maximum(gap, 0.10)[..., None, None]
            # worm, phase, event, delay, horizon, target, source
            local = normalized.transpose(0, 1, 2, 3, 5, 6, 4)
            by_seed.setdefault(seed, [None] * 20)
            for position, worm in enumerate(data["worm_indices"].astype(int)):
                by_seed[seed][worm] = local[position]
            current_horizons = data["horizon_seconds"].astype(float)
            if horizons is None:
                horizons = current_horizons
            elif not np.array_equal(horizons, current_horizons):
                raise RuntimeError("event response horizons disagree")
    result: dict[int, np.ndarray] = {}
    for seed, values in by_seed.items():
        if any(value is None for value in values):
            raise RuntimeError(f"event arrays for seed {seed} do not cover all worms")
        result[seed] = np.stack(values)
    return np.asarray(horizons), result


def event_relationships(run_dir: Path, pattern: str, run_name: str) -> pd.DataFrame:
    horizons, by_seed = _event_arrays(run_dir, pattern)
    array = np.stack(list(by_seed.values())).mean(axis=0)
    # mean across worms; only delay position zero is requested for this experiment.
    rows: list[dict[str, object]] = []
    for estimand in ("onset", "onset_minus_quiet"):
        phase = array[:, 1] if estimand == "onset" else array[:, 1] - array[:, 0]
        for horizon, seconds in enumerate(horizons):
            matrices = [phase[:, event, 0, horizon].mean(axis=0) for event in range(3)]
            for first, second in ((0, 1), (1, 2), (0, 2)):
                left, right = off_diagonal(matrices[first]), off_diagonal(matrices[second])
                rows.append({
                    "run": run_name,
                    "estimand": estimand,
                    "horizon_seconds": float(seconds),
                    "presentation_left": first + 1,
                    "presentation_right": second + 1,
                    "pearson": safe_corr(left, right, "pearson"),
                    "spearman": safe_corr(left, right, "spearman"),
                })
    return pd.DataFrame(rows)


def diagnostics(runs: dict[str, AlignedRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, run in runs.items():
        frame = run.diagnostics
        for phase in ("all", "quiet", "onset_aligned"):
            use = frame if phase == "all" else frame[frame.phase == phase]
            rows.append({
                "run": name,
                "method": run.method,
                "particles": run.particles,
                "phase": phase,
                "cells": len(use),
                "validity_rate": float(use.valid.mean()),
                "achieved_fraction_median": float(use.achieved_fraction.median()),
                "minimum_ess_median": float(use.minimum_ess.median()),
            })
    return pd.DataFrame(rows)


def edge_tests(runs: dict[str, AlignedRun], n_perm: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    supported: dict[str, np.ndarray] = {}
    signs: dict[str, np.ndarray] = {}
    for position, (name, run) in enumerate(runs.items()):
        delay = delay_position(run, 7)
        effects = np.stack([
            phase_matrix(run.coefficient, "onset_minus_quiet", delay, horizon)
            for horizon in range(len(run.horizon_seconds))
        ], axis=1)
        mean, _, _, q, mask = lagged_animal_signflip(
            effects, n_perm=n_perm, seed=4513 + 101 * position
        )
        supported[name] = (q <= 0.10) & mask
        signs[name] = np.sign(mean)
        for horizon, seconds in enumerate(run.horizon_seconds):
            rows.append({
                "run": name,
                "horizon_seconds": float(seconds),
                "tested_edges": int(mask[horizon].sum()),
                "fdr10_edges": int(supported[name][horizon].sum()),
            })
    consensus_rows = []
    for prefix in ("binary", "encoded"):
        direct, progressive = f"{prefix}_direct", f"{prefix}_progressive"
        consensus = (
            supported[direct] & supported[progressive]
            & (signs[direct] == signs[progressive])
        )
        for horizon, seconds in enumerate(runs[direct].horizon_seconds):
            consensus_rows.append({
                "encoding": prefix,
                "horizon_seconds": float(seconds),
                "cross_sampler_fdr10_sign_consensus_edges": int(consensus[horizon].sum()),
            })
    return pd.DataFrame(rows), pd.DataFrame(consensus_rows)


def mean_relation(
    frame: pd.DataFrame, comparison: str, estimand: str
) -> float:
    use = frame[(frame.comparison == comparison) & (frame.estimand == estimand)]
    return float(use.pearson.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictive-run", type=Path, required=True)
    parser.add_argument("--encoded-model-id", required=True)
    parser.add_argument("--encoded-direct", type=Path, required=True)
    parser.add_argument("--encoded-progressive", type=Path, required=True)
    parser.add_argument("--binary-direct", type=Path, required=True)
    parser.add_argument("--binary-progressive", type=Path, required=True)
    parser.add_argument("--scalar-direct", type=Path)
    parser.add_argument("--rollout-run", type=Path)
    parser.add_argument("--presentation-evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=4095)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    encoded_pattern_direct = f"{args.encoded_model_id}__direct__N256__f*__s*.npz"
    encoded_pattern_progressive = f"{args.encoded_model_id}__progressive__N64__f*__s*.npz"
    runs = {
        "binary_direct": load_run(
            "binary_direct", args.binary_direct,
            "tcn_flow128_dropout15_jitter_p01__direct__N256__f*__s*.npz",
        ),
        "binary_progressive": load_run(
            "binary_progressive", args.binary_progressive,
            "tcn_flow128_dropout15_jitter_p01__progressive__N64__f*__s*.npz",
        ),
        "encoded_direct": load_run(
            "encoded_direct", args.encoded_direct, encoded_pattern_direct,
        ),
        "encoded_progressive": load_run(
            "encoded_progressive", args.encoded_progressive,
            encoded_pattern_progressive,
        ),
    }
    if args.scalar_direct is not None:
        runs["scalar_direct"] = load_run(
            "scalar_direct",
            args.scalar_direct,
            "stim_presentation_scalar_tcn_flow128_dropout15_jitter_p01__direct__N256__f*__s*.npz",
        )
    reference = runs["binary_direct"]
    for run in runs.values():
        if not (
            np.array_equal(run.neurons, reference.neurons)
            and np.array_equal(run.horizons, reference.horizons)
            and np.any(run.delays == 7)
        ):
            raise RuntimeError("binary and encoded neuron/horizon grids are not aligned")

    board, paired = predictive_analysis(args.predictive_run)
    relationships = relationship_rows(runs)
    animal = animal_split_stability(runs, repeats=500)
    diagnostic = diagnostics(runs)
    event = pd.concat([
        event_relationships(args.encoded_direct, encoded_pattern_direct, "encoded_direct"),
        event_relationships(
            args.encoded_progressive, encoded_pattern_progressive, "encoded_progressive"
        ),
        *(
            [event_relationships(
                args.scalar_direct,
                "stim_presentation_scalar_tcn_flow128_dropout15_jitter_p01__direct__N256__f*__s*.npz",
                "scalar_direct",
            )]
            if args.scalar_direct is not None else []
        ),
    ], ignore_index=True)
    tests, consensus = edge_tests(runs, args.permutations)

    rollout = None
    if args.rollout_run is not None:
        rollout = pd.read_csv(args.rollout_run / "rollout_metrics_aggregate.csv")
    presentation_aggregate = presentation_paired = None
    if args.presentation_evaluation is not None:
        presentation_aggregate, presentation_paired = presentation_predictive_analysis(
            args.presentation_evaluation
        )

    artifacts = {
        "predictive_leaderboard.csv": board,
        "predictive_paired_deltas.csv": paired,
        "matrix_relationships.csv": relationships,
        "animal_split_stability.csv": animal,
        "sampler_diagnostics.csv": diagnostic,
        "presentation_matrix_relationships.csv": event,
        "edge_signflip_summary.csv": tests,
        "edge_consensus_summary.csv": consensus,
    }
    if rollout is not None:
        artifacts["rollout_comparison.csv"] = rollout
    if presentation_aggregate is not None and presentation_paired is not None:
        artifacts["presentation_predictive_aggregate.csv"] = presentation_aggregate
        artifacts["presentation_predictive_paired_deltas.csv"] = presentation_paired
    for name, frame in artifacts.items():
        frame.to_csv(output / name, index=False)

    selected_encoding = (
        "scalar" if "presentation_scalar" in args.encoded_model_id else "onehot"
    )
    selected_delta = paired[
        (paired.comparison == f"{selected_encoding}_minus_binary")
        & (paired.metric == "energy")
    ].iloc[0]
    shuffle_delta = paired[
        (paired.comparison == "onehot_minus_shuffle") & (paired.metric == "energy")
    ].iloc[0]
    encoded_seed = relationships[
        (relationships.comparison == "generator_seed")
        & relationships.left.str.startswith("encoded_direct")
        & (relationships.estimand == "onset_minus_quiet")
    ].pearson.mean()
    binary_seed = relationships[
        (relationships.comparison == "generator_seed")
        & relationships.left.str.startswith("binary_direct")
        & (relationships.estimand == "onset_minus_quiet")
    ].pearson.mean()
    scalar_seed = None
    if "scalar_direct" in runs:
        scalar_seed = relationships[
            (relationships.comparison == "generator_seed")
            & relationships.left.str.startswith("scalar_direct")
            & (relationships.estimand == "onset_minus_quiet")
        ].pearson.mean()
    encoded_sampler = relationships[
        (relationships.comparison == "sampler")
        & (relationships.left == "encoded_direct")
        & (relationships.estimand == "onset_minus_quiet")
    ].pearson.mean()
    binary_sampler = relationships[
        (relationships.comparison == "sampler")
        & (relationships.left == "binary_direct")
        & (relationships.estimand == "onset_minus_quiet")
    ].pearson.mean()
    encoded_consensus = int(
        consensus[consensus.encoding == "encoded"]
        .cross_sampler_fdr10_sign_consensus_edges.sum()
    )
    binary_consensus = int(
        consensus[consensus.encoding == "binary"]
        .cross_sampler_fdr10_sign_consensus_edges.sum()
    )
    rollout_sentence = "No multi-step rollout comparison was supplied."
    if rollout is not None:
        last_horizon = int(rollout.rollout_horizon_frames.max())
        final_rollout = rollout[rollout.rollout_horizon_frames == last_horizon]
        encoded_rollout = final_rollout[
            final_rollout.model_id == args.encoded_model_id
        ].iloc[0]
        binary_rollout = final_rollout[
            final_rollout.model_id == "stim_binary_tcn_flow128_dropout15_jitter_p01"
        ].iloc[0]
        rollout_delta = (
            encoded_rollout.energy__stim_balanced__mean
            - binary_rollout.energy__stim_balanced__mean
        )
        rollout_sentence = (
            f"At the longest free-running horizon ({encoded_rollout.rollout_horizon_seconds:g} s), "
            f"the paired-model stimulus-balanced energy delta was **{rollout_delta:+.6f}** "
            "for encoded minus binary (negative is better)."
        )
    presentation_lines = [
        "No per-presentation predictive evaluation was supplied."
    ]
    if presentation_paired is not None:
        selected_rows = presentation_paired[
            presentation_paired.model_id == args.encoded_model_id
        ].sort_values("presentation")
        presentation_lines = [
            "| Presentation | Encoded minus binary energy | Encoded wins |",
            "| ---: | ---: | ---: |",
            *[
                f"| {int(row.presentation)} | {row.mean_energy_delta:+.6f} | "
                f"{int(row.wins_model)}/{int(row.n_pairs)} |"
                for row in selected_rows.itertuples()
            ],
        ]

    lines = [
        "# Presentation-aware stimulus encoding and lag-matrix reliability",
        "",
        "## Bottom line",
        "",
        f"The best non-binary model used `{args.encoded_model_id}`. Relative to the "
        f"retrained binary control, its paired confirmation energy delta was "
        f"**{selected_delta.mean_delta_left_minus_right:+.6f}** (negative is better; "
        f"{int(selected_delta.wins_left)}/{int(selected_delta.n_pairs)} fold-seed pairs). "
        + (
            f"Relative to the subject-wise shuffled one-hot control, the delta was "
            f"**{shuffle_delta.mean_delta_left_minus_right:+.6f}**. The shuffled comparison "
            "is the key test of consistent presentation identity rather than merely adding channels."
            if selected_encoding == "onehot"
            else "The scalar arm has no channel-matched shuffled control; the one-hot/shuffled-one-hot comparison is reported separately."
        ),
        "",
        f"For onset-minus-matched-quiet coefficient matrices, mean generator-seed "
        f"Pearson reliability across horizons was **{encoded_seed:.3f}** with the new "
        f"encoding versus **{binary_seed:.3f}** for binary. Mean direct-versus-ESS-SMC "
        f"reliability was **{encoded_sampler:.3f}** versus **{binary_sampler:.3f}**. "
        f"Cross-sampler FDR-10% sign-consensus edge cells totaled **{encoded_consensus}** "
        f"for the new encoding and **{binary_consensus}** for binary across all horizons.",
        *(
            [
                "",
                f"The requested scalar 1/2/3 sensitivity had mean direct-sampler "
                f"generator-seed reliability **{scalar_seed:.3f}** for the same "
                "onset-minus-quiet matrices. It was not advanced to ESS-SMC because "
                "it lost to binary prediction and did not improve direct reliability.",
            ]
            if scalar_seed is not None else []
        ),
        "",
        "These are lag-matrix reliability diagnostics, not evidence of anatomical "
        "connections or causal effects. A predictive improvement does not rescue an "
        "unstable onset-minus-quiet matrix.",
        "",
        rollout_sentence,
        "",
        "## What was compared",
        "",
        "- Same 20 worms, 54 neurons, whole-worm folds, L=80 history, regularized "
        "TCN conditional flow, residual target, and training/evaluation settings.",
        "- Binary active pulse; requested scalar presentation number; non-ordinal "
        "one-hot presentation identity; and subject-wise shuffled one-hot control.",
        "- The local epochs are repetitions of the same 2-butanone stimulus. Labels "
        "1/2/3 are presentation number, not chemical identity.",
        "- Lag matrices use the frozen suppression-aligned source window, matched quiet "
        "cuts, direct N=256, and progressive ESS-SMC N=64.",
        "",
        "## Predictive leaderboard",
        "",
        "| Rank | Model | Energy | Balanced energy | Variogram |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in board.itertuples():
        lines.append(
            f"| {int(row.rank)} | `{row.model_id}` | {row.energy__mean:.6f} | "
            f"{row.energy__stim_balanced__mean:.6f} | {row.variogram__mean:.6f} |"
        )
    lines.extend([
        "",
        "## Active-pulse prediction by presentation",
        "",
        *presentation_lines,
        "",
        "## Interpretation boundary",
        "",
        "The presentation label is visible only while its pulse is active; all baseline "
        "frames are zero. It therefore does not directly reveal a future pulse or "
        "carry a persistent trial counter. Because the three pulses occur at fixed "
        "times, any repeat-specific gain can nevertheless absorb adaptation, bleaching, "
        "or other unmeasured experiment-time effects. The subject-wise permutation "
        "control and matched quiet subtraction limit, but cannot eliminate, that ambiguity.",
        "",
        "## Files",
        "",
        *[f"- `{name}`" for name in artifacts],
        "- `protocol.json`",
        "- `checksums.sha256`",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoded_model_id": args.encoded_model_id,
        "predictive_run": str(args.predictive_run.resolve()),
        "rollout_run": str(args.rollout_run.resolve()) if args.rollout_run else None,
        "presentation_evaluation": (
            str(args.presentation_evaluation.resolve())
            if args.presentation_evaluation else None
        ),
        "response_runs": {
            name: str(path.resolve()) for name, path in (
                ("encoded_direct", args.encoded_direct),
                ("encoded_progressive", args.encoded_progressive),
                ("binary_direct", args.binary_direct),
                ("binary_progressive", args.binary_progressive),
                *(
                    (("scalar_direct", args.scalar_direct),)
                    if args.scalar_direct is not None else ()
                ),
            )
        },
        "signflip_permutations": args.permutations,
        "claim_boundary": "predictive and model-relative sampled dynamics; no anatomy or causality",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )


if __name__ == "__main__":
    main()
