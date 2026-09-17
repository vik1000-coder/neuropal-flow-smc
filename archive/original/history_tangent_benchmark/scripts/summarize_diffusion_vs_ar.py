#!/usr/bin/env python3
"""Reproducible final aggregation for the diffusion-versus-AR experiment.

The generator seed is the inferential unit.  Transformer coordinate orders are
averaged within generator seed before mechanism summaries or bootstrap
intervals are computed.  Clean Transformer tangents and sigma=0.05 diffusion
tangents remain explicitly labeled as different estimands; the oracle
clean-to-noisy discrepancy is reported beside them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from history_tangent_benchmark.dgps import make_dgp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "diffusion_vs_ar_scientific_summary_20260713"
REPLICATION_DIRS = [
    ROOT / "results" / "diffusion_vs_ar_invalidity_replication_20260713_a",
    ROOT / "results" / "diffusion_vs_ar_invalidity_replication_20260713_b",
]
G5_DIR = (
    ROOT
    / "results"
    / "diffusion_vs_ar_g5_generation_calibration_extended_20260713"
)

MECHANISM_LABELS = {
    "g1": "G1 location",
    "g2": "G2 covariance",
    "g3": "G3 skew mixture",
    "g4": "G4 multimodal occupancy",
    "g6": "G6 rough location",
}
GENERATOR_PARAMS: dict[str, dict[str, Any]] = {
    "g1": {"q": 8, "dy": 8},
    "g2": {"q": 8, "dy": 8, "alpha_m": 0.0},
    "g3": {"q": 8, "dy": 1},
    "g4": {
        "q": 8,
        "dy": 8,
        "n_components": 8,
        "rank": 2,
        "mode_separation": 4.0,
        "component_sigma": 1.0,
        "interaction_scale": 0.75,
    },
    "g6": {"q": 8, "dy": 8, "omega": 8.0, "roughness_amplitude": 0.5},
}


def _load_csv(name: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(directory / name) for directory in REPLICATION_DIRS],
        ignore_index=True,
    )


def _bootstrap_interval(
    values: np.ndarray, statistic=np.median, *, seed: int, draws: int = 50_000
) -> tuple[float, float]:
    if len(values) <= 7:
        # Enumerate all n**n ordinary bootstrap resamples at this small n.  This
        # removes Monte Carlo jitter from the reported confidence limits.
        indices = np.asarray(list(itertools.product(range(len(values)), repeat=len(values))))
        estimates = np.asarray([statistic(values[row]) for row in indices])
        return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = np.asarray([statistic(values[row]) for row in indices])
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def _python_source_digest() -> str:
    root = ROOT / "src" / "history_tangent_benchmark"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(relative)
        digest.update(content)
    return digest.hexdigest()


def _oracle_nll_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for generator_id, params in GENERATOR_PARAMS.items():
        for generator_seed in (101, 103, 107, 109, 113):
            dgp = make_dgp(generator_id, seed=generator_seed, **params)
            history = dgp.sample_history(512, 401 + 30_001)
            response = dgp.sample_response(history, 1, 401 + 30_002)[:, 0, :]
            rows.append(
                {
                    "generator_id": generator_id,
                    "generator_seed": generator_seed,
                    "oracle_nll": float(-dgp.log_prob(response, history).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = _load_csv("cases.csv")
    metrics = _load_csv("metrics.csv")

    manifests = [json.loads((directory / "run_manifest.json").read_text()) for directory in REPLICATION_DIRS]
    source_digests = sorted({manifest["source_tree_sha256"] for manifest in manifests})
    environment_digests = sorted(
        {manifest["environment"]["environment_sha256"] for manifest in manifests}
    )
    expected_cases = sum(int(manifest["expected_cases"]) for manifest in manifests)

    data_checks = {
        "expected_cases": expected_cases,
        "observed_cases": int(len(cases)),
        "unique_case_ids": int(cases["case_id"].nunique()),
        "duplicate_case_ids": int(cases["case_id"].duplicated().sum()),
        "failed_cases": int((cases["status"] != "ok").sum()),
        "metric_failed_rows": int((metrics["metric_status"] == "failed").sum()),
        "generator_seeds": sorted(cases["generator_seed"].unique().tolist()),
        "data_seeds": sorted(cases["data_seed"].unique().tolist()),
        "model_seeds": sorted(cases["model_seed"].unique().tolist()),
        "source_tree_digests": source_digests,
        "environment_digests": environment_digests,
        "python_source_digest_at_summary": _python_source_digest(),
    }
    required_checks = [
        len(cases) == expected_cases == 90,
        cases["case_id"].nunique() == len(cases),
        (cases["status"] == "ok").all(),
        not (metrics["metric_status"] == "failed").any(),
        len(source_digests) == 1,
        len(environment_digests) == 1,
        data_checks["generator_seeds"] == [101, 103, 107, 109, 113],
        data_checks["data_seeds"] == [401],
        data_checks["model_seeds"] == [1401],
    ]
    data_checks["ready_for_scoped_analysis"] = bool(all(required_checks))
    if not data_checks["ready_for_scoped_analysis"]:
        raise RuntimeError(f"replication data-quality gate failed: {data_checks}")

    transformer = metrics[
        metrics["model_name"].str.startswith("ar_t_")
        & (metrics["metric_id"] == "tangent_nrmse")
        & (metrics["metric_status"] == "ok")
        & (metrics["centering"] == "none")
        & (metrics["estimand"] == "clean")
    ].copy()
    diffusion = metrics[
        (metrics["model_name"] == "diffusion_legacy")
        & (metrics["metric_id"] == "tangent_nrmse")
        & (metrics["metric_status"] == "ok")
        & (metrics["centering"] == "model_samples")
        & (metrics["estimand"] == "noisy")
    ].copy()

    transformer_seed = (
        transformer.groupby(["generator_id", "generator_seed"], as_index=False)
        .agg(
            transformer_nrmse=("value", "mean"),
            transformer_order_sd=("value", "std"),
            transformer_orders=("value", "size"),
            transformer_eval_ms=("value", lambda _: np.nan),
        )
        .drop(columns="transformer_eval_ms")
    )
    diffusion_seed = diffusion[
        ["generator_id", "generator_seed", "value"]
    ].rename(columns={"value": "diffusion_nrmse"})
    seed_summary = transformer_seed.merge(
        diffusion_seed, on=["generator_id", "generator_seed"], validate="one_to_one"
    )

    gap = metrics[
        (metrics["model_name"] == "diffusion_legacy")
        & (metrics["metric_id"] == "oracle_clean_to_noisy_tangent_nrmse")
        & (metrics["metric_status"] == "ok")
        & (metrics["centering"] == "none")
    ][["generator_id", "generator_seed", "value"]].rename(
        columns={"value": "oracle_clean_to_noisy_nrmse"}
    )
    seed_summary = seed_summary.merge(
        gap, on=["generator_id", "generator_seed"], validate="one_to_one"
    )

    time_rows = metrics[
        (metrics["metric_id"] == "tangent_eval_ms_per_example")
        & (metrics["metric_status"] == "ok")
    ]
    transformer_time = (
        time_rows[time_rows["model_name"].str.startswith("ar_t_")]
        .groupby(["generator_id", "generator_seed"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "transformer_eval_ms"})
    )
    diffusion_time = time_rows[time_rows["model_name"] == "diffusion_legacy"][
        ["generator_id", "generator_seed", "value"]
    ].rename(columns={"value": "diffusion_eval_ms"})
    seed_summary = seed_summary.merge(
        transformer_time, on=["generator_id", "generator_seed"], validate="one_to_one"
    ).merge(diffusion_time, on=["generator_id", "generator_seed"], validate="one_to_one")

    ar_nll = (
        metrics[
            metrics["model_name"].str.startswith("ar_t_")
            & (metrics["metric_id"] == "nll_original")
            & (metrics["metric_status"] == "ok")
        ]
        .groupby(["generator_id", "generator_seed"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "transformer_nll"})
    )
    seed_summary = seed_summary.merge(
        ar_nll, on=["generator_id", "generator_seed"], validate="one_to_one"
    ).merge(_oracle_nll_rows(), on=["generator_id", "generator_seed"], validate="one_to_one")
    seed_summary["transformer_nll_excess"] = (
        seed_summary["transformer_nll"] - seed_summary["oracle_nll"]
    )
    seed_summary["diffusion_slower_factor"] = (
        seed_summary["diffusion_eval_ms"] / seed_summary["transformer_eval_ms"]
    )
    seed_summary["transformer_relative_to_diffusion_pct"] = 100 * (
        seed_summary["transformer_nrmse"] / seed_summary["diffusion_nrmse"] - 1
    )
    data_checks["primary_generator_mechanism_cells"] = int(len(seed_summary))
    data_checks["expected_primary_generator_mechanism_cells"] = 25
    data_checks["bootstrap_resamples_enumerated_per_mechanism"] = 5**5
    data_checks["oracle_nll_rows_recomputed"] = 25
    seed_summary.to_csv(OUTPUT / "tangent_seed_summary.csv", index=False)

    mechanism_rows: list[dict[str, Any]] = []
    for index, (generator_id, frame) in enumerate(
        seed_summary.groupby("generator_id", sort=True)
    ):
        ar = frame["transformer_nrmse"].to_numpy()
        diff = frame["diffusion_nrmse"].to_numpy()
        ar_ci = _bootstrap_interval(ar, seed=20_000 + index)
        diff_ci = _bootstrap_interval(diff, seed=30_000 + index)
        paired = ar / diff - 1.0
        paired_ci = _bootstrap_interval(paired, statistic=np.mean, seed=40_000 + index)
        mechanism_rows.append(
            {
                "generator_id": generator_id,
                "mechanism": MECHANISM_LABELS[generator_id],
                "generator_seeds": len(frame),
                "transformer_median_nrmse": float(np.median(ar)),
                "transformer_median_ci_low": ar_ci[0],
                "transformer_median_ci_high": ar_ci[1],
                "diffusion_median_nrmse": float(np.median(diff)),
                "diffusion_median_ci_low": diff_ci[0],
                "diffusion_median_ci_high": diff_ci[1],
                "transformer_relative_pct": float(100 * (np.median(ar) / np.median(diff) - 1)),
                "paired_mean_relative_ci_low_pct": 100 * paired_ci[0],
                "paired_mean_relative_ci_high_pct": 100 * paired_ci[1],
                "transformer_valid": bool(np.median(ar) < 1.0),
                "diffusion_valid": bool(np.median(diff) < 1.0),
                "oracle_clean_to_noisy_median": float(
                    frame["oracle_clean_to_noisy_nrmse"].median()
                ),
                "transformer_eval_ms_median": float(frame["transformer_eval_ms"].median()),
                "diffusion_eval_ms_median": float(frame["diffusion_eval_ms"].median()),
                "diffusion_slower_factor_median": float(
                    frame["diffusion_slower_factor"].median()
                ),
                "transformer_nll_median": float(frame["transformer_nll"].median()),
                "oracle_nll_median": float(frame["oracle_nll"].median()),
                "transformer_nll_excess_median": float(
                    frame["transformer_nll_excess"].median()
                ),
                "transformer_order_sd_median": float(
                    frame["transformer_order_sd"].median(skipna=True)
                )
                if frame["transformer_order_sd"].notna().any()
                else None,
            }
        )
    mechanism_summary = pd.DataFrame(mechanism_rows)
    mechanism_summary.to_csv(OUTPUT / "tangent_mechanism_summary.csv", index=False)

    g5_cases = pd.read_csv(G5_DIR / "cases.csv")
    g5_metrics = pd.read_csv(G5_DIR / "metrics.csv")
    g5_manifest = json.loads((G5_DIR / "run_manifest.json").read_text())
    g5_checks = {
        "expected_cases": int(g5_manifest["expected_cases"]),
        "observed_cases": int(len(g5_cases)),
        "unique_case_ids": int(g5_cases["case_id"].nunique()),
        "failed_cases": int((g5_cases["status"] != "ok").sum()),
        "metric_failed_rows": int((g5_metrics["metric_status"] == "failed").sum()),
        "source_tree_digest": g5_manifest["source_tree_sha256"],
        "environment_digest": g5_manifest["environment"]["environment_sha256"],
    }
    g5_checks["ready_for_development_interpretation"] = bool(
        g5_checks["expected_cases"] == g5_checks["observed_cases"] == 5
        and g5_checks["unique_case_ids"] == 5
        and g5_checks["failed_cases"] == 0
        and g5_checks["metric_failed_rows"] == 0
    )
    data_checks["g5_extended_calibration"] = g5_checks
    data_checks["overall_assessment"] = "share_with_caveats"
    if not g5_checks["ready_for_development_interpretation"]:
        raise RuntimeError("extended G5 calibration is incomplete or failed")
    energy = g5_metrics[
        (g5_metrics["metric_id"] == "energy_score_fair")
        & (g5_metrics["metric_status"] == "ok")
    ][["model_name", "estimand", "value"]].copy()
    ar_energy = float(energy[energy["model_name"].str.startswith("ar_t_")]["value"].mean())
    energy["relative_to_transformer_mean_pct"] = 100 * (energy["value"] / ar_energy - 1)
    energy["transformer_mean_energy"] = ar_energy
    energy = energy.merge(
        g5_cases[["model_name", "fit.best_epoch", "fit.stopped_epoch", "fit.wall_seconds"]],
        on="model_name",
        validate="one_to_one",
    )
    energy.to_csv(OUTPUT / "g5_generation_summary.csv", index=False)

    both_valid = int(
        (mechanism_summary["transformer_valid"] & mechanism_summary["diffusion_valid"]).sum()
    )
    both_invalid = int(
        (~mechanism_summary["transformer_valid"] & ~mechanism_summary["diffusion_valid"]).sum()
    )
    decision = (
        "inconclusive_due_to_estimator_invalidity"
        if both_invalid >= 3
        else "winner_rule_requires_separate_evaluation"
    )
    summary = {
        "data_quality": data_checks,
        "tangent_decision": decision,
        "mechanisms_both_valid": both_valid,
        "mechanisms_both_invalid": both_invalid,
        "all_clean_to_noisy_discrepancies_below_0_10": bool(
            (mechanism_summary["oracle_clean_to_noisy_median"] <= 0.10).all()
        ),
        "g5_transformer_mean_fair_energy": ar_energy,
        "g5_edm_fair_energy": float(
            energy.loc[energy["model_name"] == "diffusion_edm", "value"].iloc[0]
        ),
        "g5_edm_relative_pct": float(
            energy.loc[
                energy["model_name"] == "diffusion_edm",
                "relative_to_transformer_mean_pct",
            ].iloc[0]
        ),
        "g5_is_development_only": True,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "validation.json").write_text(json.dumps(data_checks, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
