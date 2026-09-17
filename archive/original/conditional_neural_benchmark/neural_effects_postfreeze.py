from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.postfreeze_external_analysis import load_references
from conditional_neural_benchmark.distributional_lag_tournament import load_cohorts
from conditional_neural_benchmark.higher_order_postfreeze import (
    _external_metrics,
    _signed_max,
    _target_covariates,
    matched_enrichment,
)


def run(
    effects_root: Path,
    output: Path,
    release: Path,
    *,
    permutations: int,
    seed: int,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    enrichment_rows: list[dict] = []
    for cohort_name, cohort in load_cohorts().items():
        effect_file = effects_root / f"{cohort_name}__confirmation_mean.npz"
        with np.load(effect_file, allow_pickle=False) as data:
            methods = {
                f"neural_{key}": _signed_max(data[key])
                for key in ("mean", "logvariance", "covariance_row_energy")
            }
        references, networks = load_references(release, list(cohort.neurons))
        covariates = _target_covariates(cohort)
        for method, matrix in methods.items():
            for row in _external_metrics(matrix, references):
                metric_rows.append({"cohort": cohort_name, "method": method, **row})
            magnitude = np.abs(matrix)
            for network in (
                "neuromodulator_union", "monoamine_all", "neuropeptide_all",
                "monoamine_dopamine", "monoamine_serotonin",
            ):
                if network not in networks:
                    continue
                result = matched_enrichment(
                    magnitude,
                    networks[network],
                    covariates,
                    seed=seed + len(enrichment_rows),
                    permutations=permutations,
                )
                enrichment_rows.append({
                    "cohort": cohort_name,
                    "method": method,
                    "network": network,
                    **result,
                })
    metrics = pd.DataFrame(metric_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    metrics.to_csv(output / "neural_effect_randi_cook_metrics.csv", index=False)
    enrichment.to_csv(output / "neural_effect_receptor_matched_enrichment.csv", index=False)
    validation = {
        "status": "pass",
        "effect_protocol": str((effects_root / "protocol.json").resolve()),
        "external_loaded_after_effect_freeze": True,
        "metric_rows": int(len(metrics)),
        "matched_enrichment_rows": int(len(enrichment)),
        "all_numeric_finite_or_missing": bool(
            np.isfinite(metrics.select_dtypes(include=[np.number]).fillna(0)).all().all()
            and np.isfinite(enrichment.select_dtypes(include=[np.number]).fillna(0)).all().all()
        ),
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--effects", type=Path,
        default=Path("results/higher_order_neural_effects_20260828"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/higher_order_neural_effects_postfreeze_20260828"),
    )
    parser.add_argument(
        "--release", type=Path,
        default=Path("/Users/vik/Downloads/SBTG-public-release copy"),
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--wait-hours", type=float, default=8.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_hours * 3600
    protocol = args.effects / "protocol.json"
    while not protocol.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("neural effects did not complete before the post-freeze deadline")
        time.sleep(20)
    result = run(
        args.effects.resolve(), args.output.resolve(), args.release.resolve(),
        permutations=args.permutations, seed=args.seed,
    )
    print(f"NEURAL_EFFECTS_POSTFREEZE_COMPLETE {result}", flush=True)


if __name__ == "__main__":
    main()
