#!/usr/bin/env python3
"""Validate, summarize, and package the frozen SID empirical run.

This script is intentionally outside ``src`` so that report packaging cannot alter
the source-tree hash of the confirmatory estimators.  It treats the DGP seed as the
unit of inference and never mixes local, smoothed-local, finite, or dynamic targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 99173
BOOTSTRAP_REPLICATES = 2_000
CONFIDENCE = 0.95

METHOD_LABELS = {
    "B0_ZERO": "Zero",
    "B2_RAW_MOMENTS": "Direct moment regression",
    "B2_RAW_MOMENTS_FINITE": "Direct moment regression (finite)",
    "DIRECT_ENDPOINT_MOMENTS": "Endpoint Monte Carlo",
    "S1_NLL_SCORE__A1_DENSITY_AD": "Normalized density + AD",
    "S1_NLL_ENDPOINTS": "Normalized density endpoints",
    "A2_RATIO_CRITIC": "Ratio critic",
    "S2_HYVARINEN__A7_HODGE": "Clean Hyvarinen + Hodge",
    "S5_DSM_MULTI__A7_HODGE": "Multi-noise DSM + Hodge",
    "S6_GAUSS_DSM__A3_CENTER": "Gaussian DSM control",
    "S7_ANCHORED_ENERGY__A3_CENTER": "Anchored residual energy",
    "S7_SCORE_INTEGRATED_ENDPOINTS": "Integrated anchored score",
    "A8_ENDPOINT_CLASSIFIER": "Endpoint classifier",
    "A10_SIGNED_RIESZ": "Signed Riesz dictionary",
    "B1_MEAN_POLY": "Polynomial mean regression",
    "B4_GAUSSIAN_NLL": "Gaussian likelihood",
    "B7_MDN_CORRECT": "Matched gain mixture",
    "B7_MDN_CORRECT_ENDPOINTS": "Matched gain mixture endpoints",
    "S7_ANCHORED_ENERGY__A7_HODGE": "Anchored score + Hodge",
    "ORIGINAL_SBTG_K_NORM": "Original SBTG field norm",
}

FAMILY_LABELS = {
    "m1_mean": "M1 mean",
    "m2_covariance": "M2 covariance",
    "m3_cubic": "M3 cubic skew",
    "m3_bounded": "M3 bounded skew",
    "m3_local": "M3 local skew",
    "m3_asymmetric": "M3 asymmetric skew",
    "m4_quartic": "M4 quartic",
    "m4_tail": "M4 tail",
    "m5_occupancy": "M5 occupancy",
    "d1_gaussian_lag": "D1 Gaussian mean lag",
    "d2_observed_stochastic_gain": "D2 observed stochastic gain",
}

PALETTE = {
    "direct": "#2673B8",
    "normalized": "#2B8C6B",
    "ratio": "#7B52AB",
    "score": "#D9822B",
    "finite": "#C4475D",
    "null": "#777777",
    "other": "#4C566A",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_cases(case_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(case_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["metric_uri"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def audit_information_access(seed_level: pd.DataFrame) -> pd.DataFrame:
    """Preserve recorded access and add the access actually used by the code path."""
    frame = seed_level.copy()
    frame["reported_information_access"] = frame["information_access"]
    audited = frame["information_access"].astype(str).copy()
    static = frame.dgp_family.astype(str).str.startswith("m")
    dynamic = frame.dgp_family.astype(str).str.startswith("d")
    score_readout = frame.method_id.astype(str).str.startswith(("S2_", "S5_", "S6_", "S7_"))
    score_readout |= frame.method_id.astype(str).eq("ORIGINAL_SBTG_K_NORM")
    audited.loc[static & score_readout & frame.estimand_type.isin(["local", "local_smoothed", "operator"])] = (
        "observational_training_plus_oracle_density_quadrature"
    )
    audited.loc[dynamic & score_readout] = (
        "observational_training_plus_oracle_density_quadrature"
    )
    ratio = frame.method_id.astype(str).eq("A2_RATIO_CRITIC")
    audited.loc[ratio] = "observational_training_plus_conditional_evaluation_queries"
    matched_static = static & frame.method_id.astype(str).str.startswith("S1_NLL")
    tilt = frame.dgp_family.astype(str).str.startswith(("m3_", "m4_", "m5_"))
    audited.loc[matched_static & tilt] = "known_dgp_tilt_basis_plus_fitted_amplitude"
    matched_dynamic = dynamic & frame.method_id.astype(str).str.startswith("B7_MDN_CORRECT")
    audited.loc[matched_dynamic] = "known_gain_noise_family_plus_fitted_mixing_law"
    covariance_control = (
        (frame.dgp_family == "m2_covariance")
        & (frame.method_id == "S6_GAUSS_DSM__A3_CENTER")
    )
    audited.loc[covariance_control] = "fitted_full_covariance_gaussian_likelihood"
    frame["information_access"] = audited
    return frame


def bootstrap_interval(values: Iterable[float], rng: np.random.Generator) -> tuple[float, float, float, int]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan, 0
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    draws = np.mean(values[indices], axis=1)
    alpha = (1.0 - CONFIDENCE) / 2.0
    return (
        float(np.mean(values)),
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
        int(values.size),
    )


def summarize(seed_level: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "phase",
        "dgp_family",
        "estimand_type",
        "channel",
        "information_access",
        "method_id",
        "metric_name",
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for keys, frame in seed_level.groupby(groups, dropna=False, sort=True):
        mean, lower, upper, n = bootstrap_interval(frame.metric_value, rng)
        row = dict(zip(groups, keys, strict=True))
        row.update(
            mean=mean,
            ci_lower=lower,
            ci_upper=upper,
            n_dgp_seeds=n,
            failure_rate=float(np.mean(frame.status != "ok")),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_ratio(
    seed_level: pd.DataFrame,
    method: str,
    baseline: str,
    estimand_type: str,
    family: str,
    channel: str,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    relevant = seed_level[
        (seed_level.metric_name == "nrmse")
        & (seed_level.estimand_type == estimand_type)
        & (seed_level.dgp_family == family)
        & (seed_level.channel == channel)
        & seed_level.method_id.isin([method, baseline])
    ]
    pivot = relevant.pivot_table(index="dgp_seed", columns="method_id", values="metric_value")
    if method not in pivot or baseline not in pivot:
        return None
    pivot = pivot.dropna(subset=[method, baseline])
    if pivot.empty:
        return None
    ratio = pivot[method].to_numpy(float) / np.maximum(pivot[baseline].to_numpy(float), 1e-12)
    mean, lower, upper, n = bootstrap_interval(ratio, rng)
    return {
        "dgp_family": family,
        "channel": channel,
        "estimand_type": estimand_type,
        "method_id": method,
        "baseline_id": baseline,
        "mean_paired_nrmse_ratio": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_common_seeds": n,
        "advantage_upper_below_0_90": bool(upper < 0.90),
    }


def paired_table(seed_level: pd.DataFrame) -> pd.DataFrame:
    comparisons = {
        "local": [
            ("S1_NLL_SCORE__A1_DENSITY_AD", "B2_RAW_MOMENTS"),
            ("A2_RATIO_CRITIC", "B2_RAW_MOMENTS"),
            ("S2_HYVARINEN__A7_HODGE", "B2_RAW_MOMENTS"),
            ("S5_DSM_MULTI__A7_HODGE", "B2_RAW_MOMENTS"),
            ("S6_GAUSS_DSM__A3_CENTER", "B2_RAW_MOMENTS"),
            ("S7_ANCHORED_ENERGY__A3_CENTER", "B2_RAW_MOMENTS"),
        ],
        "finite": [
            ("S1_NLL_ENDPOINTS", "B2_RAW_MOMENTS_FINITE"),
            ("A8_ENDPOINT_CLASSIFIER", "B2_RAW_MOMENTS_FINITE"),
            ("A10_SIGNED_RIESZ", "B2_RAW_MOMENTS_FINITE"),
            ("S7_SCORE_INTEGRATED_ENDPOINTS", "B2_RAW_MOMENTS_FINITE"),
        ],
        "dynamic_local": [
            ("B7_MDN_CORRECT", "B2_RAW_MOMENTS"),
            ("A2_RATIO_CRITIC", "B2_RAW_MOMENTS"),
            ("S7_ANCHORED_ENERGY__A7_HODGE", "B2_RAW_MOMENTS"),
        ],
        "dynamic_finite": [
            ("B7_MDN_CORRECT_ENDPOINTS", "B2_RAW_MOMENTS_FINITE"),
            ("A8_ENDPOINT_CLASSIFIER", "B2_RAW_MOMENTS_FINITE"),
        ],
    }
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    primary = seed_level[seed_level.metric_name == "nrmse"]
    for (family, channel, estimand), _ in primary.groupby(
        ["dgp_family", "channel", "estimand_type"], dropna=False
    ):
        for method, baseline in comparisons.get(str(estimand), []):
            row = paired_ratio(
                seed_level, method, baseline, str(estimand), str(family), str(channel), rng
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def validate_run(run_dir: Path, seed_level: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    static_dir = run_dir / "cases" / "confirmatory"
    dynamic_dir = run_dir / "cases" / "dynamic_confirmatory"
    static_csv = sorted(static_dir.glob("*.csv"))
    static_npz = sorted(static_dir.glob("*.npz"))
    dynamic_csv = sorted(dynamic_dir.glob("*.csv"))
    dynamic_npz = sorted(dynamic_dir.glob("*.npz"))
    inventory_rows: list[dict[str, object]] = []
    hash_mismatches: list[str] = []
    for kind, paths in [
        ("static_metric", static_csv),
        ("static_array", static_npz),
        ("dynamic_metric", dynamic_csv),
        ("dynamic_array", dynamic_npz),
    ]:
        for path in paths:
            digest = sha256_file(path)
            inventory_rows.append(
                {
                    "kind": kind,
                    "relative_uri": str(path.relative_to(run_dir)),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
            )
            if kind.endswith("array"):
                rows = seed_level[seed_level.array_uri == str(path.relative_to(run_dir))]
                registered = set(rows.array_sha256.dropna().astype(str))
                if registered != {digest}:
                    hash_mismatches.append(str(path.relative_to(run_dir)))
    audit_files = [
        run_dir / "config.yaml",
        run_dir / "config_source.yaml",
        run_dir / "stage0" / "oracle_checks.csv",
        run_dir / "stage0" / "topology_hodge.csv",
        run_dir / "development" / "information_calibration.json",
    ]
    audit_files.extend(sorted((run_dir / "source_snapshot").rglob("*.py")))
    audit_files.extend(sorted((run_dir / "audit_inputs").glob("*")))
    for path in audit_files:
        if not path.exists():
            continue
        inventory_rows.append(
            {
                "kind": "frozen_input",
                "relative_uri": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    primary = seed_level[seed_level.metric_name == "nrmse"]
    seed_counts = (
        primary.groupby(["dgp_family", "estimand_type", "channel", "method_id"])
        .dgp_seed.nunique()
        .reset_index(name="n_seeds")
    )
    nonfinite = int((~np.isfinite(seed_level.metric_value.dropna().to_numpy(float))).sum())
    missing_values = seed_level[seed_level.metric_value.isna()]
    expected_missing = missing_values.metric_name.isin(["signed_correlation"])
    unexpected_missing = int((~expected_missing).sum())
    failed_rows = int((seed_level.status != "ok").sum())
    static_families = sorted(
        seed_level.loc[seed_level.phase == "confirmatory", "dgp_family"].dropna().unique()
    )
    payload = {
        "created_unix": time.time(),
        "static_metric_files": len(static_csv),
        "static_array_files": len(static_npz),
        "dynamic_metric_files": len(dynamic_csv),
        "dynamic_array_files": len(dynamic_npz),
        "static_family_count": len([x for x in static_families if str(x).startswith("m")]),
        "metric_rows": int(seed_level.shape[0]),
        "failed_rows": failed_rows,
        "nonfinite_metric_values": nonfinite,
        "missing_metric_values": int(missing_values.shape[0]),
        "expected_undefined_diagnostics": int(expected_missing.sum()),
        "unexpected_missing_metric_values": unexpected_missing,
        "raw_array_hash_mismatches": hash_mismatches,
        "min_primary_common_seed_count": int(seed_counts.n_seeds.min()) if not seed_counts.empty else 0,
        "max_primary_common_seed_count": int(seed_counts.n_seeds.max()) if not seed_counts.empty else 0,
        "static_complete": len(static_csv) == 270 and len(static_npz) == 270,
        "dynamic_complete": len(dynamic_csv) == 30 and len(dynamic_npz) == 30,
    }
    payload["passed"] = bool(
        payload["static_complete"]
        and payload["dynamic_complete"]
        and failed_rows == 0
        and nonfinite == 0
        and unexpected_missing == 0
        and not hash_mismatches
        and payload["min_primary_common_seed_count"] == 30
    )
    return payload, pd.DataFrame(inventory_rows)


def method_class(method: str) -> str:
    if method == "B0_ZERO":
        return "null"
    if method.startswith("B2") or method.startswith("DIRECT") or "MOMENT" in method:
        return "direct"
    if method.startswith("S1") or "MDN" in method or "NLL" in method:
        return "normalized"
    if "RATIO" in method:
        return "ratio"
    if method.startswith(("S2", "S5", "S6", "S7")):
        return "score"
    if method.startswith(("A8", "A10")):
        return "finite"
    return "other"


def save_figure(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def leaderboard_figure(summary: pd.DataFrame, estimand: str, path: Path) -> None:
    data = summary[(summary.metric_name == "nrmse") & (summary.estimand_type == estimand)].copy()
    panel_keys = list(
        data[["dgp_family", "channel"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    panel_keys.sort(key=lambda x: (list(FAMILY_LABELS).index(x[0]) if x[0] in FAMILY_LABELS else 999, str(x[1])))
    if not panel_keys:
        return
    methods = sorted(data.method_id.unique(), key=lambda x: METHOD_LABELS.get(x, x))
    height = max(7.0, 1.25 * len(panel_keys))
    fig, axes = plt.subplots(len(panel_keys), 1, figsize=(11.5, height), sharex=True, squeeze=False)
    for axis, (family, channel) in zip(axes[:, 0], panel_keys, strict=True):
        panel = data[(data.dgp_family == family) & (data.channel == channel)].set_index("method_id").reindex(methods).reset_index()
        y = np.arange(len(methods))
        for index, row in panel.iterrows():
            if not np.isfinite(row.get("mean", np.nan)):
                continue
            color = PALETTE[method_class(str(row.method_id))]
            axis.errorbar(
                row["mean"],
                index,
                xerr=[[row["mean"] - row["ci_lower"]], [row["ci_upper"] - row["mean"]]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2,
                ms=4,
            )
        axis.axvline(1.0, color="#9A9A9A", ls="--", lw=1)
        axis.set_yticks(y, [METHOD_LABELS.get(x, x) for x in methods], fontsize=7)
        title = FAMILY_LABELS.get(family, family)
        if data[data.dgp_family == family].channel.nunique() > 1:
            title += f" — {str(channel).replace('_', ' ')}"
        axis.set_title(title, loc="left", fontsize=9, weight="bold")
        axis.grid(axis="x", color="#E7E7E7", lw=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[-1, 0].set_xlabel("DGP-seed mean NRMSE (95% bootstrap CI); dashed line = zero estimator")
    fig.suptitle(f"{estimand.replace('_', ' ').title()} targets: like-for-like recovery", y=1.002, fontsize=14)
    save_figure(fig, path)


def derivative_gap_figure(summary: pd.DataFrame, path: Path) -> None:
    score = summary[
        (summary.estimand_type == "operator") & (summary.metric_name == "response_score_nrmse")
    ][["dgp_family", "method_id", "mean"]].rename(columns={"mean": "score_nrmse"})
    tangent = summary[
        (summary.estimand_type == "operator") & (summary.metric_name == "history_tangent_nrmse")
    ][["dgp_family", "method_id", "mean"]].rename(columns={"mean": "tangent_nrmse"})
    data = score.merge(tangent, on=["dgp_family", "method_id"])
    if data.empty:
        return
    fig, axis = plt.subplots(figsize=(7.4, 6.0))
    for method, panel in data.groupby("method_id"):
        color = PALETTE[method_class(str(method))]
        axis.scatter(panel.score_nrmse, panel.tangent_nrmse, label=METHOD_LABELS.get(method, method), color=color, s=45, alpha=.85)
        for _, row in panel.iterrows():
            axis.annotate(FAMILY_LABELS.get(row.dgp_family, row.dgp_family), (row.score_nrmse, row.tangent_nrmse), xytext=(3, 3), textcoords="offset points", fontsize=6)
    limits = [max(1e-3, min(data.score_nrmse.min(), data.tangent_nrmse.min()) * .7), max(data.score_nrmse.max(), data.tangent_nrmse.max()) * 1.4]
    axis.plot(limits, limits, color="#888888", ls="--", lw=1, label="equal error")
    axis.axhline(1.0, color="#BBBBBB", lw=1)
    axis.set(xscale="log", yscale="log", xlim=limits, ylim=limits, xlabel="Response-score NRMSE", ylabel="History-tangent NRMSE")
    axis.set_title("Derivative gap: an accurate response score need not yield an accurate history tangent", loc="left", fontsize=12)
    axis.grid(color="#ECECEC", which="both", lw=.6)
    axis.legend(fontsize=7, frameon=False, loc="best")
    save_figure(fig, path)


def information_figure(summary: pd.DataFrame, seed_level: pd.DataFrame, path: Path) -> None:
    info = seed_level.groupby("dgp_family", as_index=False).information_index.mean()
    data = summary[
        (summary.estimand_type == "local")
        & (summary.metric_name == "nrmse")
        & summary.method_id.isin(["B2_RAW_MOMENTS", "A2_RATIO_CRITIC", "S7_ANCHORED_ENERGY__A3_CENTER"])
    ].merge(info, on="dgp_family")
    if data.empty:
        return
    fig, axis = plt.subplots(figsize=(7.6, 5.5))
    for method, panel in data.groupby("method_id"):
        axis.plot(panel.information_index, panel["mean"], "o", label=METHOD_LABELS.get(method, method), color=PALETTE[method_class(method)], alpha=.85)
        for _, row in panel.iterrows():
            axis.annotate(FAMILY_LABELS.get(row.dgp_family, row.dgp_family), (row.information_index, row["mean"]), xytext=(3, 2), textcoords="offset points", fontsize=6)
    axis.axhline(1.0, color="#999999", ls="--", lw=1)
    axis.set(xlabel=r"Registered information index $\Lambda=n\delta^2I_v$", ylabel="Mean local NRMSE", yscale="log")
    axis.set_title("Information matching reveals boundary-limited mechanisms", loc="left", fontsize=12)
    axis.grid(color="#ECECEC", which="both", lw=.6)
    axis.legend(frameon=False, fontsize=8)
    save_figure(fig, path)


def make_figures(summary: pd.DataFrame, seed_level: pd.DataFrame, figures: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#252525",
            "text.color": "#252525",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        }
    )
    leaderboard_figure(summary, "local", figures / "local_leaderboard")
    leaderboard_figure(summary, "finite", figures / "finite_leaderboard")
    leaderboard_figure(summary, "dynamic_local", figures / "dynamic_local_leaderboard")
    leaderboard_figure(summary, "dynamic_finite", figures / "dynamic_finite_leaderboard")
    derivative_gap_figure(summary, figures / "derivative_gap")
    information_figure(summary, seed_level, figures / "information_vs_recovery")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--runbook", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    results = run_dir / "results"
    figures = results / "figures"
    tables = results / "tables"
    reports = run_dir / "reports"
    for directory in [figures, tables, reports]:
        directory.mkdir(parents=True, exist_ok=True)

    audit_inputs = run_dir / "audit_inputs"
    audit_inputs.mkdir(exist_ok=True)
    project_dir = Path(__file__).resolve().parents[1]
    for source in sorted(project_dir.glob("PRE_REGISTRATION*.md")) + sorted(
        project_dir.glob("PREREGISTRATION*.md")
    ):
        shutil.copy2(source, audit_inputs / source.name)
    if args.runbook is not None:
        shutil.copy2(args.runbook.resolve(), audit_inputs / args.runbook.name)

    static = read_cases(run_dir / "cases" / "confirmatory", "*.csv")
    dynamic = read_cases(run_dir / "cases" / "dynamic_confirmatory", "*.csv")
    if static.empty or dynamic.empty:
        raise SystemExit("static and dynamic confirmatory results must both exist before packaging")
    seed_level = audit_information_access(
        pd.concat([static, dynamic], ignore_index=True, sort=False)
    )
    validation, inventory = validate_run(run_dir, seed_level)
    atomic_json(results / "validation.json", validation)
    atomic_csv(results / "source_inventory.csv", inventory)
    if not validation["passed"]:
        raise SystemExit(f"validation failed; inspect {results / 'validation.json'}")

    atomic_csv(results / "seed_level.csv", seed_level)
    seed_level.to_parquet(results / "seed_level.parquet", index=False)
    access_audit = seed_level[
        ["dgp_family", "estimand_type", "method_id", "reported_information_access", "information_access"]
    ].drop_duplicates().sort_values(["dgp_family", "estimand_type", "method_id"])
    atomic_csv(tables / "information_access_audit.csv", access_audit)
    summary = summarize(seed_level)
    atomic_csv(results / "summary.csv", summary)
    summary.to_parquet(results / "summary.parquet", index=False)
    paired = paired_table(seed_level)
    atomic_csv(tables / "paired_advantage.csv", paired)
    make_figures(summary, seed_level, figures)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest["finished_unix"] = time.time()
    manifest["packaging_validation_sha256"] = sha256_file(results / "validation.json")
    atomic_json(manifest_path, manifest)
    package_manifest = {
        "created_unix": time.time(),
        "run_dir": str(run_dir),
        "run_manifest_sha256": sha256_file(manifest_path),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence": CONFIDENCE,
        "validation_sha256": sha256_file(results / "validation.json"),
        "seed_level_sha256": sha256_file(results / "seed_level.csv"),
        "summary_sha256": sha256_file(results / "summary.csv"),
        "paired_advantage_sha256": sha256_file(tables / "paired_advantage.csv"),
    }
    atomic_json(results / "package_manifest.json", package_manifest)
    print(json.dumps({"status": "complete", **validation}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
