#!/usr/bin/env python3
"""Validate and summarize the alternative phase-duration window/scaling campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.phase_window_fit import PHASES, PHASE_LABELS, VARIANTS
from experiments.phase_duration_summary import production_neuron_type


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "results/derived/phase_window_sensitivity"
LAGS = (1, 2, 3, 5)
TYPES = ("sensory", "interneuron", "motor")
TYPE_SHORT = {"sensory": "S", "interneuron": "I", "motor": "M"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--matched-replicates", type=int, default=3)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def result_dir(root: Path, variant: str, phase: str, seed: int, replicate: int) -> Path:
    return root / variant / phase / f"seed_{seed:04d}" / f"replicate_{replicate:03d}"


def validate_job(path: Path, variant: str, phase: str, seed: int, replicate: int) -> list[str]:
    problems = []
    manifest_path = path / "manifest.json"
    result_path = path / "result.npz"
    selected_path = path / "selected_windows.csv"
    for required in (manifest_path, result_path, selected_path):
        if not required.exists():
            problems.append(f"missing {required}")
    if problems:
        return problems
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid manifest {manifest_path}: {exc}"]
    expected = {
        "status": "complete",
        "variant": variant,
        "phase_code": phase,
        "model_seed": seed,
        "replicate": replicate,
        "n_recording_rows": 20,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            problems.append(f"{manifest_path}: {key}={manifest.get(key)!r}, expected {value!r}")
    selection = VARIANTS[variant]["selection"]
    expected_counts = {lag: 660 for lag in LAGS} if selection == "distributed_shared_anchor" else {
        1: 900,
        2: 840,
        3: 780,
        5: 660,
    }
    for lag, count in expected_counts.items():
        observed = manifest.get("lag_records", {}).get(str(lag), {}).get("n_windows")
        if observed != count:
            problems.append(f"{manifest_path}: lag {lag} has {observed} windows, expected {count}")
    return problems


def summarize_job(path: Path, variant: str, phase_code: str, seed: int, replicate: int):
    with np.load(path / "result.npz", allow_pickle=False) as data:
        names = [str(value) for value in data["neuron_names"]]
        types = np.asarray([production_neuron_type(name) for name in names])
        off_diagonal = ~np.eye(len(names), dtype=bool)
        profile_rows = []
        edge_rows = []
        for lag in LAGS:
            mu = np.asarray(data[f"mu_hat_lag{lag}"], dtype=float)
            masks = {
                "bh": np.asarray(data[f"significant_bh_lag{lag}"], dtype=bool),
                "by": np.asarray(data[f"significant_by_lag{lag}"], dtype=bool),
            }
            edge_rows.append(
                {
                    "variant": variant,
                    "phase_code": phase_code,
                    "phase": PHASE_LABELS[phase_code],
                    "model_seed": seed,
                    "replicate": replicate,
                    "lag": lag,
                    "bh_edges": int((masks["bh"] & off_diagonal).sum()),
                    "by_edges": int((masks["by"] & off_diagonal).sum()),
                }
            )
            for source_type in TYPES:
                for target_type in TYPES:
                    candidate = np.outer(types == target_type, types == source_type) & off_diagonal
                    base = {
                        "variant": variant,
                        "phase_code": phase_code,
                        "phase": PHASE_LABELS[phase_code],
                        "model_seed": seed,
                        "replicate": replicate,
                        "pair": f"{TYPE_SHORT[source_type]}→{TYPE_SHORT[target_type]}",
                        "source_type": source_type,
                        "target_type": target_type,
                        "lag": lag,
                        "n_candidates": int(candidate.sum()),
                        "mean_abs_mu_all": float(np.abs(mu)[candidate].mean()),
                    }
                    for method, mask in masks.items():
                        values = np.abs(mu)[candidate & mask]
                        base[f"n_{method}"] = int(values.size)
                        base[f"mean_abs_mu_{method}"] = float(values.mean()) if values.size else 0.0
                    profile_rows.append(base)
    return profile_rows, edge_rows


def compute_indices(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["variant", "phase_code", "phase", "model_seed", "replicate", "pair"]
    pair_rows = []
    for key, group in profile.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        by_lag = group.set_index("lag")
        for metric in ("mean_abs_mu_all", "mean_abs_mu_bh", "mean_abs_mu_by"):
            short = float(by_lag.loc[[1, 2], metric].mean())
            long = float(by_lag.loc[[3, 5], metric].mean())
            label = metric.removeprefix("mean_abs_mu_")
            row[f"{label}_short_mean"] = short
            row[f"{label}_long_mean"] = long
            row[f"{label}_long_short_ratio"] = long / short if short > 0 else np.nan
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)

    phase_keys = ["variant", "phase_code", "phase", "model_seed", "replicate"]
    phase_rows = []
    for key, group in pairs.groupby(phase_keys, sort=False):
        row = dict(zip(phase_keys, key))
        for method in ("all", "bh", "by"):
            values = group[f"{method}_long_short_ratio"].dropna().to_numpy(float)
            row[f"{method}_ratio_pair_mean"] = float(values.mean()) if values.size else np.nan
            row[f"{method}_ratio_pair_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{method}_ratio_n_pairs"] = int(values.size)
        phase_rows.append(row)
    return pairs, pd.DataFrame(phase_rows)


def summarize_replicates(indices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in indices.groupby(["variant", "phase", "model_seed"], sort=False):
        for method in ("all", "bh", "by"):
            values = group[f"{method}_ratio_pair_mean"].dropna().to_numpy(float)
            rows.append(
                {
                    "variant": key[0],
                    "phase": key[1],
                    "model_seed": key[2],
                    "method": method,
                    "n": len(values),
                    "mean": float(values.mean()) if values.size else np.nan,
                    "sd": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "minimum": float(values.min()) if values.size else np.nan,
                    "maximum": float(values.max()) if values.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_contrasts(indices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in indices.variant.unique():
        for seed in sorted(indices.model_seed.unique()):
            subset = indices[(indices.variant == variant) & (indices.model_seed == seed)]
            for event in ("On", "Off"):
                event_rows = subset[subset.phase == event]
                if len(event_rows) != 1:
                    continue
                for reference in ("Baseline", "Steady"):
                    reference_rows = subset[subset.phase == reference]
                    for method in ("all", "bh", "by"):
                        column = f"{method}_ratio_pair_mean"
                        event_value = float(event_rows.iloc[0][column])
                        values = reference_rows[column].dropna().to_numpy(float)
                        if not values.size or not np.isfinite(event_value):
                            continue
                        rows.append(
                            {
                                "variant": variant,
                                "model_seed": seed,
                                "event_phase": event,
                                "reference_phase": reference,
                                "method": method,
                                "event_value": event_value,
                                "reference_mean": float(values.mean()),
                                "difference": float(event_value - values.mean()),
                                "within_reference_range": bool(values.min() <= event_value <= values.max()),
                                "reference_min": float(values.min()),
                                "reference_max": float(values.max()),
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.input_root.resolve()
    out_dir = (args.out_dir or root / "summary").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = []
    for variant in args.variants:
        for phase in PHASES:
            replicates = range(args.matched_replicates) if phase in {"baseline", "steady"} else range(1)
            for seed in args.model_seeds:
                for replicate in replicates:
                    expected.append((variant, phase, seed, replicate))

    problems = []
    valid_jobs = []
    for variant, phase, seed, replicate in expected:
        path = result_dir(root, variant, phase, seed, replicate)
        job_problems = validate_job(path, variant, phase, seed, replicate)
        if job_problems:
            problems.extend(job_problems)
        else:
            valid_jobs.append((path, variant, phase, seed, replicate))
    if problems and not args.allow_incomplete:
        raise RuntimeError("Alternative phase-duration campaign is incomplete:\n" + "\n".join(problems))

    profiles = []
    edges = []
    for path, variant, phase, seed, replicate in valid_jobs:
        job_profiles, job_edges = summarize_job(path, variant, phase, seed, replicate)
        profiles.extend(job_profiles)
        edges.extend(job_edges)
    profile = pd.DataFrame(profiles)
    edge_counts = pd.DataFrame(edges)
    pairs, indices = compute_indices(profile)
    summary = summarize_replicates(indices)
    contrasts = compute_contrasts(indices)

    profile.to_csv(out_dir / "phase_alternative_profiles.csv", index=False)
    edge_counts.to_csv(out_dir / "phase_alternative_edge_counts.csv", index=False)
    pairs.to_csv(out_dir / "phase_alternative_pair_ratios.csv", index=False)
    indices.to_csv(out_dir / "phase_alternative_phase_indices.csv", index=False)
    summary.to_csv(out_dir / "phase_alternative_phase_summary.csv", index=False)
    contrasts.to_csv(out_dir / "phase_alternative_contrasts.csv", index=False)
    metadata = {
        "status": "complete" if not problems else "incomplete",
        "n_expected_jobs": len(expected),
        "n_valid_jobs": len(valid_jobs),
        "problems": problems,
        "variants": args.variants,
        "matched_replicates": args.matched_replicates,
        "model_seeds": args.model_seeds,
        "primary_classifier": "released Figure 3C classifier for exact comparability",
        "ratio_definition": "mean over 9 cell-type-pair ratios; each pair is mean(lags 3,5)/mean(lags 1,2)",
    }
    (out_dir / "phase_alternative_summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    display = summary[summary.method.isin(["bh", "all"])].copy()
    print(display.to_string(index=False))
    print(f"\nPHASE-WINDOW SUMMARY: {metadata['status'].upper()}")


if __name__ == "__main__":
    main()
