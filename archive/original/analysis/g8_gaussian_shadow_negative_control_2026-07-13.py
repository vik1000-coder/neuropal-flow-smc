"""Finite-classifier negative control on G8's exact moment-matched Gaussian shadow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from history_tangent_benchmark.dgps import G8FunctionalShapeMixture, G8GaussianShadow
from history_tangent_benchmark.serialization import atomic_json


OUTPUT = Path("analysis/g8_gaussian_shadow_negative_control_2026-07-13")
CODE = runpy.run_path("analysis/g8_typed_classifier_benchmark_2026-07-13.py")
paired_dataset = CODE["_paired_dataset"]
fit_mlp = CODE["_fit_mlp"]
mlp_probability = CODE["_mlp_probability"]
truth_effect = CODE["_truth"]
GENERATOR_SEEDS = (61, 67, 71, 73)
DATA_SEEDS = (601, 607)


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for generator_seed in GENERATOR_SEEDS:
        source = G8FunctionalShapeMixture(
            seed=generator_seed,
            q=17,
            dy=32,
            n_channels=4,
            motif_rank=4,
            motif_scale=0.45,
            noise_sd=0.08,
            correlated_noise_scale=0.06,
            shape_sensitivity=1.2,
        )
        shadow = G8GaussianShadow.from_g8(source)
        scale = abs(truth_effect(source))
        for data_seed in DATA_SEEDS:
            train = paired_dataset(shadow, 4_000, _seed(generator_seed, data_seed, "train"))
            validation = paired_dataset(shadow, 1_000, _seed(generator_seed, data_seed, "validation"))
            test = paired_dataset(shadow, 10_000, _seed(generator_seed, data_seed, "test"))
            scaler, model, fit = fit_mlp(
                train["full"],
                train["labels"],
                validation["full"],
                validation["labels"],
                _seed(generator_seed, data_seed, "fit"),
            )
            probability = mlp_probability(scaler, model, test["full"])
            witness_effect = float(np.mean(test["phi"] * (2.0 * probability - 1.0)))
            plus = test["phi"][test["labels"] == 1]
            minus = test["phi"][test["labels"] == 0]
            empirical_effect = float((plus.mean() - minus.mean()) / 2.0)
            rows.append(
                {
                    "status": "ok",
                    "generator_seed": generator_seed,
                    "data_seed": data_seed,
                    "auc": float(roc_auc_score(test["labels"], probability)),
                    "classifier_effect": witness_effect,
                    "classifier_effect_relative_to_g8": witness_effect / scale,
                    "empirical_effect": empirical_effect,
                    "empirical_effect_relative_to_g8": empirical_effect / scale,
                    "probability_sd": float(np.std(probability)),
                    **fit,
                }
            )
            print(f"g={generator_seed} d={data_seed} auc={rows[-1]['auc']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    summary = {
        "schema_version": "1",
        "status": "complete",
        "cases": len(frame),
        "auc_median": float(frame.auc.median()),
        "auc_range": [float(frame.auc.min()), float(frame.auc.max())],
        "classifier_absolute_effect_relative_to_g8_median": float(
            frame.classifier_effect_relative_to_g8.abs().median()
        ),
        "classifier_absolute_effect_relative_to_g8_max": float(
            frame.classifier_effect_relative_to_g8.abs().max()
        ),
        "empirical_absolute_effect_relative_to_g8_median": float(
            frame.empirical_effect_relative_to_g8.abs().median()
        ),
        "train_per_side": 4_000,
        "validation_per_side": 1_000,
        "test_per_side": 10_000,
    }
    atomic_json(OUTPUT / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
