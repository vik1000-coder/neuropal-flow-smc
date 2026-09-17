"""Developmental convergence sweep and preregistered readiness gate.

This module is deliberately separate from the confirmatory runner.  It selects
learning rates only by native validation loss, then evaluates tangent readiness
after selection.  Oracle tangent metrics never choose a checkpoint or learning
rate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .config import BenchmarkConfig, GeneratorSpec, ModelSpec, load_config
from .dgps import make_dgp
from .metrics import TrainScaler
from .models import (
    ConditionalModel,
    ConditionalVectorDiffusion,
    RatioCritic,
    build_model,
    resolve_device,
)
from .runner import PACKAGE_ROOT, _model_applies, run_benchmark
from .serialization import atomic_json, ensure_within


LEARNING_RATES = (1e-4, 3e-4, 1e-3)
SWEEP_MODEL_SEEDS = (1009, 1013)

# Frozen before the sweep is inspected. These gates ask whether the small
# implementations are learning their intended objects well enough to justify
# spending compute on replication; they are not H1--H7 decision thresholds.
READINESS_CRITERIA = MappingProxyType(
    {
        "all_overfit_gates_pass": {"operator": "equal", "threshold": True},
        "failed_selected_cases": {"operator": "max", "threshold": 0},
        "max_parameter_ratio": {"operator": "max", "threshold": 1.10},
        "selected_best_at_last_epoch_fraction": {"operator": "max", "threshold": 0.25},
        "g1_m1_tangent_nrmse": {"operator": "max", "threshold": 0.35},
        "g1_m2_tangent_nrmse": {"operator": "max", "threshold": 0.60},
        "g1_m4_tangent_nrmse": {"operator": "max", "threshold": 0.70},
        "g1_m5_oracle_centered_tangent_nrmse": {
            "operator": "max",
            "threshold": 0.50,
        },
        "g1_m5_response_score_nrmse": {"operator": "max", "threshold": 0.60},
        "g1_clean_centering_error_max": {"operator": "max", "threshold": 0.25},
        "g1_m5_oracle_centering_error": {"operator": "max", "threshold": 0.15},
        "g2_best_clean_tangent_nrmse": {"operator": "max", "threshold": 0.85},
        "g3_best_clean_tangent_nrmse": {"operator": "max", "threshold": 0.85},
        "g4_m4_tangent_nrmse": {"operator": "max", "threshold": 0.75},
    }
)


def _fixed_objective(
    model: ConditionalModel,
    history: torch.Tensor,
    response: torch.Tensor,
    *,
    diffusion_sigma: torch.Tensor | None = None,
    diffusion_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(model, ConditionalVectorDiffusion):
        if diffusion_sigma is None or diffusion_noise is None:
            raise ValueError("fixed diffusion objective needs frozen sigma and noise")
        noisy = response + diffusion_sigma * diffusion_noise
        predicted = model.score(noisy, history, diffusion_sigma)
        target = -diffusion_noise / diffusion_sigma
        return torch.mean(diffusion_sigma.square() * (predicted - target).square())
    if isinstance(model, RatioCritic):
        negative = torch.roll(response, shifts=1, dims=0)
        positive_logits = model.log_ratio(history, response)
        negative_logits = model.log_ratio(history, negative)
        return 0.5 * (
            F.softplus(-positive_logits).mean() + F.softplus(negative_logits).mean()
        )
    return model.native_loss(history, response)


def run_overfit_gates(
    config: BenchmarkConfig,
    *,
    steps: int = 500,
    learning_rate: float = 1e-3,
    n_examples: int = 256,
) -> dict[str, Any]:
    """Fit every configured model to one frozen batch with no resampling."""
    if steps < 1 or n_examples < 2:
        raise ValueError("overfit steps/examples are too small")
    results = []
    for model_index, model_spec in enumerate(config.models):
        generator_spec = next(
            generator
            for generator in config.generators
            if _model_applies(generator, model_spec)
        )
        parameters = dict(generator_spec.params)
        dgp = make_dgp(generator_spec.kind, seed=71 + model_index, **parameters)
        history = dgp.sample_history(n_examples, 171 + model_index)
        response = dgp.sample_response(history, 1, 271 + model_index)[:, 0, :]
        scaler = TrainScaler.fit(history.numpy(), response.numpy())
        history_s = np.asarray(scaler.transform_history(history.numpy()), dtype=np.float32)
        response_s = np.asarray(scaler.transform_response(response.numpy()), dtype=np.float32)
        model_seed = 31_001 + model_index
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        model = build_model(
            model_spec.kind,
            q=dgp.q,
            dy=dgp.dy,
            params=dict(model_spec.params),
        )
        device = resolve_device(str(config.training.get("device", "auto")))
        model.to(device)
        h = torch.as_tensor(history_s, dtype=torch.float32, device=device)
        y = torch.as_tensor(response_s, dtype=torch.float32, device=device)
        sigma = None
        noise = None
        if isinstance(model, ConditionalVectorDiffusion):
            generator = torch.Generator(device="cpu").manual_seed(model_seed + 1)
            uniform = torch.rand((n_examples, 1), generator=generator, dtype=torch.float32).to(
                device
            )
            noise = torch.randn(response_s.shape, generator=generator, dtype=torch.float32).to(
                device
            )
            sigma = torch.exp(
                math.log(model.sigma_min)
                + uniform * math.log(model.sigma_max / model.sigma_min)
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), learning_rate, weight_decay=0.0
        )
        trace = []
        model.train()
        with torch.no_grad():
            initial = float(
                _fixed_objective(
                    model,
                    h,
                    y,
                    diffusion_sigma=sigma,
                    diffusion_noise=noise,
                ).cpu()
            )
        best = initial
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = _fixed_objective(
                model,
                h,
                y,
                diffusion_sigma=sigma,
                diffusion_noise=noise,
            )
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            best = min(best, value)
            if step % 25 == 0 or step == steps - 1:
                trace.append({"step": step + 1, "loss": value})
        relative_improvement = (initial - best) / max(abs(initial), 1e-8)
        passed = bool(math.isfinite(best) and relative_improvement >= 0.50)
        results.append(
            {
                "model_name": model_spec.name,
                "model_kind": model_spec.kind,
                "generator_id": generator_spec.id,
                "n_examples": n_examples,
                "steps": steps,
                "initial_loss": initial,
                "best_loss": best,
                "relative_improvement": relative_improvement,
                "passed": passed,
                "trace": trace,
            }
        )
    return {
        "status": "passed" if all(item["passed"] for item in results) else "failed",
        "required_relative_improvement": 0.50,
        "results": results,
    }


def _lr_label(value: float) -> str:
    return f"lr_{value:.0e}".replace("-", "m").replace("+", "p")


def _read_subrun(output: Path, learning_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(output / "cases.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    cases["learning_rate"] = float(learning_rate)
    metrics["learning_rate"] = float(learning_rate)
    return cases, metrics


def select_learning_rates(metrics: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    native = metrics[
        (metrics["metric_id"] == "native_validation_loss")
        & (metrics["metric_status"] == "ok")
    ].copy()
    if native.empty:
        raise ValueError("no valid native validation losses in sweep")
    group = ["model_name", "generator_id", "model_seed"]
    native["validation_rank"] = native.groupby(group)["value"].rank(
        method="average", ascending=True
    )
    scores = (
        native.groupby(["model_name", "learning_rate"], as_index=False)
        .agg(
            mean_validation_rank=("validation_rank", "mean"),
            mean_native_loss=("value", "mean"),
            comparisons=("value", "size"),
        )
        .sort_values(
            ["model_name", "mean_validation_rank", "mean_native_loss", "learning_rate"]
        )
    )
    selected: dict[str, float] = {}
    for model_name, rows in scores.groupby("model_name", sort=True):
        selected[model_name] = float(rows.iloc[0]["learning_rate"])
    return selected, scores


def _metric_median(
    metrics: pd.DataFrame,
    *,
    generator: str,
    model: str | None,
    metric_id: str,
    estimand: str,
    centering: str,
) -> float:
    mask = (
        (metrics["generator_id"] == generator)
        & (metrics["metric_id"] == metric_id)
        & (metrics["estimand"] == estimand)
        & (metrics["centering"] == centering)
        & (metrics["metric_status"] == "ok")
    )
    if model is not None:
        mask &= metrics["model_name"] == model
    values = metrics.loc[mask, "value"]
    if values.empty:
        return float("inf")
    return float(values.median())


def evaluate_readiness(
    cases: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_learning_rates: Mapping[str, float],
    overfit: Mapping[str, Any],
    *,
    max_epochs: int,
) -> dict[str, Any]:
    selected_cases = cases[
        cases.apply(
            lambda row: math.isclose(
                float(row["learning_rate"]),
                float(selected_learning_rates[row["model_name"]]),
                rel_tol=0,
                abs_tol=1e-12,
            ),
            axis=1,
        )
    ].copy()
    selected_metrics = metrics[
        metrics.apply(
            lambda row: math.isclose(
                float(row["learning_rate"]),
                float(selected_learning_rates[row["model_name"]]),
                rel_tol=0,
                abs_tol=1e-12,
            ),
            axis=1,
        )
    ].copy()

    parameter_ratios = selected_cases.groupby("generator_id")["fit.parameter_count"].agg(
        lambda values: float(values.max() / values.min())
    )
    clean_center = selected_metrics[
        (selected_metrics["generator_id"] == "g1")
        & (selected_metrics["model_name"].isin(["m1_gaussian", "m2_ar_vector", "m4_ratio"]))
        & (selected_metrics["metric_id"] == "centering_error")
        & (selected_metrics["estimand"] == "clean")
        & (selected_metrics["metric_status"] == "ok")
    ]["value"]
    g2_values = selected_metrics[
        (selected_metrics["generator_id"] == "g2")
        & (selected_metrics["model_name"].isin(["m1_gaussian", "m2_ar_vector", "m4_ratio"]))
        & (selected_metrics["metric_id"] == "tangent_nrmse")
        & (selected_metrics["estimand"] == "clean")
        & (selected_metrics["metric_status"] == "ok")
    ].groupby("model_name")["value"].median()
    g3_values = selected_metrics[
        (selected_metrics["generator_id"] == "g3")
        & (selected_metrics["model_name"].isin(["m1_gaussian", "m2_ar_scalar", "m4_ratio"]))
        & (selected_metrics["metric_id"] == "tangent_nrmse")
        & (selected_metrics["estimand"] == "clean")
        & (selected_metrics["metric_status"] == "ok")
    ].groupby("model_name")["value"].median()
    values: dict[str, Any] = {
        "all_overfit_gates_pass": overfit.get("status") == "passed",
        "failed_selected_cases": int((selected_cases["status"] != "ok").sum()),
        "max_parameter_ratio": float(parameter_ratios.max()),
        "selected_best_at_last_epoch_fraction": float(
            (selected_cases["fit.best_epoch"] >= max_epochs - 1).mean()
        ),
        "g1_m1_tangent_nrmse": _metric_median(
            selected_metrics,
            generator="g1",
            model="m1_gaussian",
            metric_id="tangent_nrmse",
            estimand="clean",
            centering="none",
        ),
        "g1_m2_tangent_nrmse": _metric_median(
            selected_metrics,
            generator="g1",
            model="m2_ar_vector",
            metric_id="tangent_nrmse",
            estimand="clean",
            centering="none",
        ),
        "g1_m4_tangent_nrmse": _metric_median(
            selected_metrics,
            generator="g1",
            model="m4_ratio",
            metric_id="tangent_nrmse",
            estimand="clean",
            centering="none",
        ),
        "g1_m5_oracle_centered_tangent_nrmse": _metric_median(
            selected_metrics,
            generator="g1",
            model="m5_diffusion",
            metric_id="tangent_nrmse",
            estimand="noisy",
            centering="oracle_samples",
        ),
        "g1_m5_response_score_nrmse": _metric_median(
            selected_metrics,
            generator="g1",
            model="m5_diffusion",
            metric_id="response_score_nrmse",
            estimand="noisy",
            centering="none",
        ),
        "g1_clean_centering_error_max": float(clean_center.max())
        if not clean_center.empty
        else float("inf"),
        "g1_m5_oracle_centering_error": _metric_median(
            selected_metrics,
            generator="g1",
            model="m5_diffusion",
            metric_id="centering_error",
            estimand="noisy",
            centering="oracle_samples",
        ),
        "g2_best_clean_tangent_nrmse": float(g2_values.min())
        if not g2_values.empty
        else float("inf"),
        "g3_best_clean_tangent_nrmse": float(g3_values.min())
        if not g3_values.empty
        else float("inf"),
        "g4_m4_tangent_nrmse": _metric_median(
            selected_metrics,
            generator="g4",
            model="m4_ratio",
            metric_id="tangent_nrmse",
            estimand="clean",
            centering="none",
        ),
    }
    checks = []
    for name, criterion in READINESS_CRITERIA.items():
        value = values[name]
        threshold = criterion["threshold"]
        if criterion["operator"] == "equal":
            passed = value == threshold
        elif criterion["operator"] == "max":
            passed = bool(math.isfinite(float(value)) and float(value) <= float(threshold))
        else:
            raise ValueError(f"unknown readiness operator {criterion['operator']!r}")
        checks.append(
            {
                "name": name,
                "value": value,
                "operator": criterion["operator"],
                "threshold": threshold,
                "passed": passed,
            }
        )
    return {
        "ready_for_confirmatory": all(check["passed"] for check in checks),
        "criteria_frozen_before_results": True,
        "selected_learning_rates": dict(selected_learning_rates),
        "checks": checks,
        "selected_cases": int(len(selected_cases)),
        "selected_metric_rows": int(len(selected_metrics)),
    }


def run_convergence_sweep(
    config_or_path: BenchmarkConfig | str | Path,
    *,
    output_dir: str = "results/convergence_sweep_20260712_v1",
    learning_rates: Sequence[float] = LEARNING_RATES,
    model_seeds: Sequence[int] = SWEEP_MODEL_SEEDS,
    max_epochs: int = 100,
    patience: int = 20,
) -> dict[str, Any]:
    config = (
        config_or_path
        if isinstance(config_or_path, BenchmarkConfig)
        else load_config(config_or_path)
    )
    output = ensure_within(PACKAGE_ROOT, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "frozen_readiness_criteria.json",
        {"criteria": {key: dict(value) for key, value in READINESS_CRITERIA.items()}},
    )
    overfit = run_overfit_gates(config)
    atomic_json(output / "overfit_gates.json", overfit)

    case_frames = []
    metric_frames = []
    manifests = []
    for learning_rate in learning_rates:
        training = dict(config.training)
        training.update(
            {
                "learning_rate": float(learning_rate),
                "max_epochs": int(max_epochs),
                "patience": int(patience),
            }
        )
        relative_output = f"{output_dir}/{_lr_label(float(learning_rate))}"
        suite = replace(
            config,
            output_dir=relative_output,
            model_seeds=tuple(int(seed) for seed in model_seeds),
            training=MappingProxyType(training),
            metadata=MappingProxyType(
                {
                    **dict(config.metadata),
                    "developmental_convergence_sweep": True,
                    "learning_rate": float(learning_rate),
                    "readiness_criteria": "../frozen_readiness_criteria.json",
                }
            ),
        )
        suite.validate()
        manifest = run_benchmark(suite)
        manifests.append(
            {
                "learning_rate": float(learning_rate),
                "status": manifest["status"],
                "completed": manifest["completed"],
                "failed": manifest["failed"],
                "config_sha256": manifest["config_sha256"],
                "source_tree_sha256": manifest["source_tree_sha256"],
            }
        )
        cases, metrics = _read_subrun(PACKAGE_ROOT / relative_output, float(learning_rate))
        case_frames.append(cases)
        metric_frames.append(metrics)

    cases = pd.concat(case_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    cases.to_csv(output / "sweep_cases.csv", index=False)
    metrics.to_csv(output / "sweep_metrics.csv", index=False)
    selected, scores = select_learning_rates(metrics)
    scores.to_csv(output / "learning_rate_selection.csv", index=False)
    readiness = evaluate_readiness(
        cases,
        metrics,
        selected,
        overfit,
        max_epochs=max_epochs,
    )
    atomic_json(output / "readiness.json", readiness)
    summary = {
        "status": "complete",
        "output_dir": str(output),
        "overfit_status": overfit["status"],
        "subruns": manifests,
        "cases": int(len(cases)),
        "metric_rows": int(len(metrics)),
        "selected_learning_rates": selected,
        "ready_for_confirmatory": readiness["ready_for_confirmatory"],
        "failed_checks": [
            check["name"] for check in readiness["checks"] if not check["passed"]
        ],
    }
    atomic_json(output / "sweep_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="history-tangent-convergence-sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", default="results/convergence_sweep_20260712_v1"
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_convergence_sweep(
        args.config,
        output_dir=args.output,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LEARNING_RATES",
    "READINESS_CRITERIA",
    "SWEEP_MODEL_SEEDS",
    "evaluate_readiness",
    "run_convergence_sweep",
    "run_overfit_gates",
    "select_learning_rates",
]
