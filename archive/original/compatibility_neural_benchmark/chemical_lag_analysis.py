from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t, ttest_1samp

from compatibility_neural_benchmark.aligned_lag_analysis import (
    bh_adjust,
    reindex_events_by_chemical,
)


CHEMICALS = ("butanone", "pentanedione", "nacl")


@dataclass
class ChemicalLagRun:
    label: str
    method: str
    model_id: str
    neurons: np.ndarray
    delays: np.ndarray
    delay_seconds: np.ndarray
    horizons: np.ndarray
    horizon_seconds: np.ndarray
    seeds: np.ndarray
    worm_ids: np.ndarray
    # seed, worm, chemical, delay, horizon, target, source
    coefficient: np.ndarray
    # seed, worm, chemical, delay, source
    validity: np.ndarray
    schema_version: str
    schema_fingerprint: str
    boundary_shift_frames: int
    files: list[Path]


def _safe_corr(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3 or np.std(left[keep]) == 0 or np.std(right[keep]) == 0:
        return np.nan
    fn = pearsonr if kind == "pearson" else spearmanr
    return float(fn(left[keep], right[keep]).statistic)


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def load_run(label: str, run_dir: Path) -> ChemicalLagRun:
    paths = sorted((run_dir / "responses").glob("*.npz"))
    if not paths:
        raise RuntimeError(f"no response archives in {run_dir}")
    by_seed: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    metadata = None
    files: list[Path] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["status"].item()) != "complete":
                continue
            required = {
                "chemical_code_by_worm_event",
                "chemical_name_by_worm_event",
                "stimulus_schema_version",
                "stimulus_schema_fingerprint",
                "boundary_shift_frames",
            }
            missing = required.difference(data.files)
            if missing:
                raise RuntimeError(f"archive lacks corrected chemical metadata: {path}: {sorted(missing)}")
            local_metadata = (
                str(data["method"].item()),
                str(data["model_id"].item()),
                tuple(data["neurons"].astype(str)),
                tuple(data["delay_frames"].astype(int)),
                tuple(data["delay_seconds"].astype(float)),
                tuple(data["horizon_frames"].astype(int)),
                tuple(data["horizon_seconds"].astype(float)),
                str(data["stimulus_schema_version"].item()),
                str(data["stimulus_schema_fingerprint"].item()),
                int(data["boundary_shift_frames"].item()),
            )
            if metadata is None:
                metadata = local_metadata
            elif metadata != local_metadata:
                raise RuntimeError(f"archive metadata mismatch: {path}")
            codes = data["chemical_code_by_worm_event"].astype(int)
            names = data["chemical_name_by_worm_event"].astype(str)
            for worm_codes, worm_names in zip(codes, names):
                if tuple(worm_names[np.argsort(worm_codes)]) != CHEMICALS:
                    raise RuntimeError(f"chemical code/name mismatch in {path}")
            response = reindex_events_by_chemical(
                data["response_cumulative_mean"].astype(np.float64), codes
            )
            gap = reindex_events_by_chemical(
                data["diagnostic_achieved_gap"].astype(np.float64), codes
            )
            valid = reindex_events_by_chemical(
                data["diagnostic_valid"].astype(np.float64), codes
            )
            # response: worm, phase, chemical, delay, source, horizon, target.
            normalized = response / np.maximum(gap, 0.10)[..., None, None]
            contrast = (normalized[:, 1] - normalized[:, 0]).transpose(0, 1, 2, 4, 5, 3)
            paired_validity = np.minimum(valid[:, 1], valid[:, 0])
            seed = int(data["model_seed"].item())
            seed_map = by_seed.setdefault(seed, {})
            for position, worm_id in enumerate(data["worm_ids"].astype(str)):
                if worm_id in seed_map:
                    raise RuntimeError(f"duplicate worm/seed {worm_id}/{seed}")
                seed_map[worm_id] = (contrast[position], paired_validity[position])
            files.append(path)
    if metadata is None:
        raise RuntimeError(f"no complete response archives in {run_dir}")
    seeds = np.asarray(sorted(by_seed), dtype=int)
    worm_sets = [set(by_seed[int(seed)]) for seed in seeds]
    if any(worm_set != worm_sets[0] for worm_set in worm_sets[1:]):
        raise RuntimeError("generator seeds do not cover the same worms")
    worm_ids = np.asarray(sorted(worm_sets[0]))
    coefficient = np.stack(
        [np.stack([by_seed[int(seed)][worm][0] for worm in worm_ids]) for seed in seeds]
    )
    validity = np.stack(
        [np.stack([by_seed[int(seed)][worm][1] for worm in worm_ids]) for seed in seeds]
    )
    (
        method,
        model_id,
        neurons,
        delays,
        delay_seconds,
        horizons,
        horizon_seconds,
        schema_version,
        schema_fingerprint,
        boundary_shift,
    ) = metadata
    return ChemicalLagRun(
        label=label,
        method=method,
        model_id=model_id,
        neurons=np.asarray(neurons),
        delays=np.asarray(delays),
        delay_seconds=np.asarray(delay_seconds),
        horizons=np.asarray(horizons),
        horizon_seconds=np.asarray(horizon_seconds),
        seeds=seeds,
        worm_ids=worm_ids,
        coefficient=coefficient,
        validity=validity,
        schema_version=schema_version,
        schema_fingerprint=schema_fingerprint,
        boundary_shift_frames=boundary_shift,
        files=files,
    )


def _check_alignment(runs: dict[str, ChemicalLagRun]) -> None:
    first = next(iter(runs.values()))
    for run in runs.values():
        for name in (
            "neurons", "delays", "delay_seconds", "horizons", "horizon_seconds",
            "seeds", "worm_ids",
        ):
            if not np.array_equal(getattr(first, name), getattr(run, name)):
                raise RuntimeError(f"run alignment mismatch for {name}")
        if run.schema_fingerprint != first.schema_fingerprint:
            raise RuntimeError("run stimulus-schema fingerprints differ")


def stability_rows(runs: dict[str, ChemicalLagRun], split_repeats: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(20_260_828)
    rows: list[dict[str, object]] = []
    for label, run in runs.items():
        for chemical, chemical_name in enumerate(CHEMICALS):
            for delay, delay_seconds in enumerate(run.delay_seconds):
                for horizon, horizon_seconds in enumerate(run.horizon_seconds):
                    seed_matrices = run.coefficient[:, :, chemical, delay, horizon].mean(axis=1)
                    seed_pairs = []
                    for left in range(len(run.seeds)):
                        for right in range(left + 1, len(run.seeds)):
                            seed_pairs.append(
                                _safe_corr(
                                    _off_diagonal(seed_matrices[left]),
                                    _off_diagonal(seed_matrices[right]),
                                    "spearman",
                                )
                            )
                    worm_values = run.coefficient.mean(axis=0)[:, chemical, delay, horizon]
                    split_values = []
                    half = len(run.worm_ids) // 2
                    for _ in range(split_repeats):
                        order = rng.permutation(len(run.worm_ids))
                        left = worm_values[order[:half]].mean(axis=0)
                        right = worm_values[order[half : 2 * half]].mean(axis=0)
                        split_values.append(
                            _safe_corr(_off_diagonal(left), _off_diagonal(right), "spearman")
                        )
                    split_values = np.asarray(split_values, dtype=float)
                    source_validity = run.validity[:, :, chemical, delay].mean(axis=(0, 1))
                    rows.append(
                        {
                            "run": label,
                            "method": run.method,
                            "chemical": chemical_name,
                            "delay_frames": int(run.delays[delay]),
                            "delay_seconds": float(delay_seconds),
                            "horizon_frames": int(run.horizons[horizon]),
                            "horizon_seconds": float(horizon_seconds),
                            "generator_seed_spearman_mean": (
                                float(np.nanmean(seed_pairs)) if len(seed_pairs) else np.nan
                            ),
                            "split_half_spearman_median": float(np.nanmedian(split_values)),
                            "split_half_spearman_p05": float(np.nanquantile(split_values, 0.05)),
                            "split_half_spearman_p95": float(np.nanquantile(split_values, 0.95)),
                            "source_validity_mean": float(np.mean(source_validity)),
                            "source_validity_fraction_ge_50pct": float(np.mean(source_validity >= 0.5)),
                            "n_worms": len(run.worm_ids),
                            "n_generator_seeds": len(run.seeds),
                        }
                    )
    labels = list(runs)
    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            left, right = runs[labels[left_index]], runs[labels[right_index]]
            for chemical, chemical_name in enumerate(CHEMICALS):
                for delay, delay_seconds in enumerate(left.delay_seconds):
                    for horizon, horizon_seconds in enumerate(left.horizon_seconds):
                        lm = left.coefficient[:, :, chemical, delay, horizon].mean(axis=(0, 1))
                        rm = right.coefficient[:, :, chemical, delay, horizon].mean(axis=(0, 1))
                        rows.append(
                            {
                                "run": f"{left.label}_vs_{right.label}",
                                "method": "cross_sampler",
                                "chemical": chemical_name,
                                "delay_frames": int(left.delays[delay]),
                                "delay_seconds": float(delay_seconds),
                                "horizon_frames": int(left.horizons[horizon]),
                                "horizon_seconds": float(horizon_seconds),
                                "generator_seed_spearman_mean": np.nan,
                                "split_half_spearman_median": _safe_corr(
                                    _off_diagonal(lm), _off_diagonal(rm), "spearman"
                                ),
                                "split_half_spearman_p05": np.nan,
                                "split_half_spearman_p95": np.nan,
                                "source_validity_mean": float(
                                    np.mean(
                                        np.minimum(
                                            left.validity[:, :, chemical, delay].mean(axis=(0, 1)),
                                            right.validity[:, :, chemical, delay].mean(axis=(0, 1)),
                                        )
                                    )
                                ),
                                "source_validity_fraction_ge_50pct": np.nan,
                                "n_worms": len(left.worm_ids),
                                "n_generator_seeds": len(left.seeds),
                            }
                        )
    return pd.DataFrame(rows)


def edge_rows(runs: dict[str, ChemicalLagRun]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for label, run in runs.items():
        # Average model seeds within animal before animal-level inference.
        values = run.coefficient.mean(axis=0)
        tests = ttest_1samp(values, 0.0, axis=0, nan_policy="omit")
        means = np.nanmean(values, axis=0)
        standard_error = np.nanstd(values, axis=0, ddof=1) / np.sqrt(len(run.worm_ids))
        critical = float(t.ppf(0.975, len(run.worm_ids) - 1))
        sign_agreement = np.mean(np.sign(values) == np.sign(means)[None], axis=0)
        validity = run.validity.mean(axis=(0, 1))
        for chemical, chemical_name in enumerate(CHEMICALS):
            for delay, delay_seconds in enumerate(run.delay_seconds):
                for horizon, horizon_seconds in enumerate(run.horizon_seconds):
                    p_values = tests.pvalue[chemical, delay, horizon].copy()
                    np.fill_diagonal(p_values, np.nan)
                    q_values = bh_adjust(p_values.ravel()).reshape(p_values.shape)
                    for target, target_name in enumerate(run.neurons):
                        for source, source_name in enumerate(run.neurons):
                            if target == source:
                                continue
                            mean = float(means[chemical, delay, horizon, target, source])
                            se = float(standard_error[chemical, delay, horizon, target, source])
                            records.append(
                                {
                                    "run": label,
                                    "method": run.method,
                                    "chemical": chemical_name,
                                    "delay_frames": int(run.delays[delay]),
                                    "delay_seconds": float(delay_seconds),
                                    "horizon_frames": int(run.horizons[horizon]),
                                    "horizon_seconds": float(horizon_seconds),
                                    "source": str(source_name),
                                    "target": str(target_name),
                                    "mean_coefficient": mean,
                                    "ci95_low": mean - critical * se,
                                    "ci95_high": mean + critical * se,
                                    "worm_sign_agreement": float(
                                        sign_agreement[chemical, delay, horizon, target, source]
                                    ),
                                    "p_value": float(p_values[target, source]),
                                    "bh_q_value": float(q_values[target, source]),
                                    "source_validity": float(validity[chemical, delay, source]),
                                    "n_worms": len(run.worm_ids),
                                }
                            )
    return pd.DataFrame(records)


def consensus_edges(edges: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(edges.run.unique())
    if len(methods) < 2:
        return pd.DataFrame()
    keys = ["chemical", "delay_frames", "horizon_frames", "source", "target"]
    left = edges[edges.run == methods[0]].copy()
    right = edges[edges.run == methods[1]].copy()
    merged = left.merge(right, on=keys, suffixes=(f"__{methods[0]}", f"__{methods[1]}"))
    merged["same_sign"] = (
        np.sign(merged[f"mean_coefficient__{methods[0]}"])
        == np.sign(merged[f"mean_coefficient__{methods[1]}"])
    )
    merged["strict_consensus"] = (
        merged.same_sign
        & (merged[f"bh_q_value__{methods[0]}"] < 0.05)
        & (merged[f"bh_q_value__{methods[1]}"] < 0.05)
        & (merged[f"worm_sign_agreement__{methods[0]}"] >= 0.70)
        & (merged[f"worm_sign_agreement__{methods[1]}"] >= 0.70)
        & (merged[f"source_validity__{methods[0]}"] >= 0.50)
        & (merged[f"source_validity__{methods[1]}"] >= 0.50)
    )
    return merged.sort_values(
        ["strict_consensus", f"bh_q_value__{methods[0]}", f"bh_q_value__{methods[1]}"],
        ascending=[False, True, True],
    )


def write_report(
    output: Path,
    runs: dict[str, ChemicalLagRun],
    stability: pd.DataFrame,
    consensus: pd.DataFrame,
) -> None:
    strict = consensus[consensus.strict_consensus] if len(consensus) else consensus
    lines = [
        "# Corrected chemical-specific lag-matrix analysis",
        "",
        "## Result",
        "",
        f"The strict direct/ESS-SMC consensus contains **{len(strict)} edge×chemical×delay×horizon cells** after within-matrix BH correction, worm sign agreement, source compatibility, and sampler sign checks.",
        "",
        "Each matrix uses the event containing the requested chemical for each animal. The estimand is the model-relative source-compatible response at true onset minus the matched quiet response for the same animal/event. Generator seeds are averaged within animal before animal-level tests.",
        "",
        "| Run | Chemical | Horizon (s) | Seed ρ | Split-half median ρ | Validity |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    own = stability[stability.method != "cross_sampler"]
    for row in own.itertuples():
        lines.append(
            f"| {row.run} | {row.chemical} | {row.horizon_seconds:g} | "
            f"{row.generator_seed_spearman_mean:.3f} | {row.split_half_spearman_median:.3f} | "
            f"{row.source_validity_mean:.3f} |"
        )
    cross = stability[stability.method == "cross_sampler"]
    lines.extend(["", "## Direct versus ESS-SMC", ""])
    for row in cross.itertuples():
        lines.append(
            f"- {row.chemical}, {row.horizon_seconds:g} s: off-diagonal Spearman "
            f"{row.split_half_spearman_median:.3f}."
        )
    lines.extend(
        [
            "",
            "These coefficients are finite contrasts of a learned observational conditional law. They do not identify synapses, receptor-mediated effects, causal interventions, or physical transmission delays. Cook, Randi, SBTG, and receptor maps were not used here; any such comparison must be separately labeled post-freeze correspondence.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIRECTORY"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-repeats", type=int, default=500)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = {label: load_run(label, Path(directory).resolve()) for label, directory in args.run}
    _check_alignment(runs)
    stability = stability_rows(runs, args.split_repeats)
    edges = edge_rows(runs)
    consensus = consensus_edges(edges)
    stability.to_csv(output / "lag_matrix_stability.csv", index=False)
    edges.to_csv(output / "chemical_edge_statistics.csv", index=False)
    consensus.to_csv(output / "direct_smc_consensus.csv", index=False)
    first = next(iter(runs.values()))
    np.savez_compressed(
        output / "mean_chemical_lag_matrices.npz",
        chemical_names=np.asarray(CHEMICALS),
        neurons=first.neurons,
        delay_frames=first.delays,
        delay_seconds=first.delay_seconds,
        horizon_frames=first.horizons,
        horizon_seconds=first.horizon_seconds,
        stimulus_schema_version=np.asarray(first.schema_version),
        stimulus_schema_fingerprint=np.asarray(first.schema_fingerprint),
        **{
            f"{label}_onset_minus_quiet": run.coefficient.mean(axis=(0, 1))
            for label, run in runs.items()
        },
    )
    write_report(output, runs, stability, consensus)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                "runs": {label: [str(path) for path in run.files] for label, run in runs.items()},
                "stimulus_schema_version": first.schema_version,
                "stimulus_schema_fingerprint": first.schema_fingerprint,
                "chemical_event_selection": "per-worm event reindexed by raw 1-based chemical code",
                "estimand": "true onset minus matched quiet within worm/event",
                "external_atlas_access": "none",
                "claim_boundary": "model-relative observational finite contrast; not causal, anatomical, receptor, or physical delay",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
