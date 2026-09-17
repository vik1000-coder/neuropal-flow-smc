"""Targeted post-benchmark SID stability sweep.

The sweep is diagnostic rather than confirmatory: it asks whether ordinary
regularization or denoising scale repairs the quadratic SID natural-variance
boundary failures seen in the frozen E1--E5 run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from neuromod_benchmark.features import subset
from neuromod_benchmark.methods.classical import RidgeGaussian
from neuromod_benchmark.methods.sid import SIDQuadratic
from neuromod_benchmark.rethink_suite import (
    SuiteConfig,
    _episodes_for_cell,
    _score_prediction,
    make_supervised,
)


RIDGES = (0.01, 0.05, 0.2, 1.0, 5.0, 20.0)
SIGMAS = (0.0, 0.2, 0.75, 1.0)


def cells(cfg: SuiteConfig) -> list[dict]:
    result = []
    for seed in cfg.seeds:
        result += [
            dict(experiment="E1", generator="reservoir", seed=seed,
                 noise_family="correlated_jumps", view="latent", horizon=1,
                 intervention="none", stress="one_step_jumps"),
            dict(experiment="E1", generator="reservoir", seed=seed,
                 noise_family="student_t", view="latent", horizon=16,
                 intervention="none", stress="long_horizon_heavy_tail"),
            dict(experiment="E2", generator="reservoir", seed=seed,
                 noise_family="student_t", view="artifact_missing", horizon=4,
                 observed_fraction=0.67, intervention="none", stress="partial_observation"),
            dict(experiment="E3", generator="reservoir", seed=seed,
                 noise_family="pooled_train_ood_test", view="artifact_missing", horizon=16,
                 observed_fraction=0.67, intervention="receptor_attenuation",
                 test_input_scale=1.7, test_state_mix=1.35, stress="joint_ood"),
            dict(experiment="E4", generator="hybrid_receptor", seed=seed,
                 noise_family="correlated_jumps", view="saturated", horizon=4,
                 intervention="receptor_attenuation", stress="receptor_kinetics"),
            dict(experiment="E5", generator="semi_markov", seed=seed,
                 noise_family="skew_mixture", view="saturated", horizon=16,
                 intervention="none", test_state_mix=1.5, stress="semi_markov"),
        ]
    return result


def split(data, cfg):
    return (
        subset(data, data.groups < cfg.train_episodes),
        subset(data, (data.groups >= cfg.train_episodes) &
                     (data.groups < cfg.train_episodes + cfg.validation_episodes)),
        subset(data, data.groups >= cfg.train_episodes + cfg.validation_episodes),
    )


def run(output: Path, neurons: int = 24) -> None:
    cfg = SuiteConfig(n_neurons=neurons, neural_cells=False)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for number, cell in enumerate(cells(cfg), start=1):
        data = make_supervised(_episodes_for_cell(cell, cfg), lags=cfg.history_lags,
                               horizon=cell["horizon"])
        train, validation, test = split(data, cfg)
        ridge = RidgeGaussian(ridge=1.0).fit(train, validation)
        ridge_score = _score_prediction(ridge.predict(test), test.targets)
        records.append(dict(cell, method="ridge_var", ridge=1.0, sigma=np.nan,
                            **ridge_score))
        for regularization in RIDGES:
            for sigma in SIGMAS:
                started = time.perf_counter()
                model = SIDQuadratic(
                    ridge=regularization, sigma_fraction=sigma,
                    n_corruptions=2 if sigma > 0 else 1, seed=cell["seed"],
                ).fit(train, validation)
                prediction = model.predict(test)
                score = _score_prediction(prediction, test.targets)
                records.append(dict(
                    cell, method="sid_quadratic", ridge=regularization, sigma=sigma,
                    runtime_seconds=time.perf_counter() - started, **score,
                ))
        pd.DataFrame(records).to_csv(output / "metrics_checkpoint.csv", index=False)
        print(f"[{number:02d}/{len(cells(cfg)):02d}] {cell['stress']} seed={cell['seed']}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "metrics_tidy.csv", index=False)
    sid = frame[frame.method == "sid_quadratic"]
    summary = sid.groupby(["ridge", "sigma"], as_index=False).agg(
        cells=("nll", "size"), valid_rate=("numerical_valid", "mean"),
        median_nll=("nll", "median"), mean_nll=("nll", "mean"),
        p95_nll=("nll", lambda x: x.quantile(.95)),
        median_nrmse=("nrmse", "median"), mean_nrmse=("nrmse", "mean"),
        invalid_variance_fraction=("invalid_variance_fraction", "mean"),
    ).sort_values(["valid_rate", "mean_nll"], ascending=[False, True])
    summary.to_csv(output / "stability_summary.csv", index=False)
    baseline = frame[frame.method == "ridge_var"].nll.mean()
    best = summary.iloc[0]
    report = [
        "# SID quadratic stability sweep", "", "## Result", "",
        f"Across {len(sid)} SID fits on {len(cells(cfg))} representative E1--E5 cells, the best validity-first setting was ridge={best.ridge:g}, sigma={best.sigma:g}: validity {best.valid_rate:.3f}, median NLL {best.median_nll:.4f}, mean NLL {best.mean_nll:.4f}. The matched ridge-VAR mean NLL was {baseline:.4f}.",
        "", "This is a post-benchmark diagnostic. It may identify a repair direction but cannot retroactively change the frozen E1--E5 ranking.",
        "", "## All settings", "",
        "| Ridge | Sigma | Valid rate | Median NLL | Mean NLL | P95 NLL | Mean NRMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        report.append(f"| {row.ridge:g} | {row.sigma:g} | {row.valid_rate:.3f} | {row.median_nll:.4f} | {row.mean_nll:.4f} | {row.p95_nll:.4f} | {row.mean_nrmse:.4f} |")
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    validation = {
        "rows": len(frame), "sid_rows": len(sid),
        "expected_sid_rows": len(cells(cfg)) * len(RIDGES) * len(SIGMAS),
        "all_finite": bool(np.isfinite(sid[["nll", "nrmse"]]).all().all()),
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neurons", type=int, default=24)
    args = parser.parse_args()
    run(args.output, args.neurons)
