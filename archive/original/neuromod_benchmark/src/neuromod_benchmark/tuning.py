"""Small nested grids selected only on capability-matched validation objectives."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .config import MethodSpec
from .features import subset
from .metrics import predictive_metrics
from .registry import make_method
from .schema import SupervisedData


@dataclass
class TuningResult:
    estimator: Any
    params: dict[str, Any]
    objective: str
    score: float
    candidates: list[dict[str, Any]]


def parameter_grid(base: dict[str, Any], tune: dict[str, Any]):
    if not tune:
        yield dict(base)
        return
    names = sorted(tune)
    values = [tuple(tune[name]) for name in names]
    for combination in product(*values):
        params = dict(base)
        params.update(dict(zip(names, combination)))
        yield params


def _selection_score(estimator, validation: SupervisedData) -> tuple[str, float, dict]:
    capability = estimator.capabilities.predictive_distribution
    if capability == "normalized":
        prediction = estimator.predict(validation, n_samples=0)
        metrics = predictive_metrics(prediction, validation.targets)
        invalid = float(metrics.get("invalid_variance_fraction", 0.0))
        extreme = float(metrics.get("extreme_variance_fraction", 0.0))
        # Log score remains primary. Invalid natural variances above 5% incur a
        # predeclared deployment guardrail penalty rather than being hidden by clamps.
        score = (
            metrics["nll"]
            + 10.0 * max(0.0, invalid - 0.05)
            + 100.0 * max(0.0, extreme - 0.01)
        )
        return "validation_nll_with_invalid_variance_guardrail", score, metrics
    if capability == "unnormalized_score":
        if not hasattr(estimator, "selection_dsm_risk"):
            raise TypeError("score model has no fixed-reference DSM selection risk")
        score = float(estimator.selection_dsm_risk(validation, seed=18_881))
        return (
            "validation_fixed_reference_dsm_ladder_risk",
            score,
            {"validation_fixed_reference_dsm_ladder_risk": score},
        )
    return "no_tuning_graph_only", 0.0, {}


def tune_method(
    spec: MethodSpec,
    train: SupervisedData,
    validation: SupervisedData,
    *,
    seed: int,
) -> TuningResult:
    grid = list(parameter_grid(dict(spec.params), dict(spec.tune)))
    if not grid:
        raise ValueError("empty hyperparameter grid")
    probe = make_method(spec.name, grid[0], seed)
    if probe.capabilities.predictive_distribution == "none" and len(grid) > 1:
        raise ValueError(
            f"{spec.name} is graph-only and has no declared validation objective; "
            "fix one preregistered setting instead of tuning on an arbitrary tie"
        )

    # With multiple candidates, early stopping/calibration uses an inner group
    # split carved from training worms. The untouched outer validation worms select
    # hyperparameters. After selection, refit once on all training worms with the
    # outer validation set used only for early stopping, never for another choice.
    if len(grid) > 1:
        groups = np.unique(train.groups)
        if len(groups) < 2:
            raise ValueError("nested tuning requires at least two training groups")
        shuffled = np.random.default_rng(seed + 4_099).permutation(groups)
        n_inner_validation = max(1, int(round(0.2 * len(groups))))
        if n_inner_validation >= len(groups):
            n_inner_validation = 1
        inner_validation_groups = shuffled[:n_inner_validation]
        inner_train_mask = ~np.isin(train.groups, inner_validation_groups)
        inner_validation_mask = ~inner_train_mask
        fit_train = subset(train, inner_train_mask)
        fit_validation = subset(train, inner_validation_mask)
    else:
        fit_train, fit_validation = train, validation

    candidates = []
    best = None
    best_score = float("inf")
    best_params = None
    best_objective = None
    for index, params in enumerate(grid):
        record: dict[str, Any] = {"index": index, "params": params}
        try:
            # Common initialization/random stream across candidates prevents a
            # hyperparameter from winning merely because it received a lucky seed.
            estimator = make_method(spec.name, params, seed)
            estimator.fit(fit_train, fit_validation)
            objective, score, diagnostics = _selection_score(estimator, validation)
            record.update({"status": "ok", "objective": objective, "score": score})
            record["diagnostics"] = diagnostics
            if np.isfinite(score) and score < best_score:
                best = estimator
                best_score = score
                best_params = params
                best_objective = objective
        except Exception as error:  # failures are manifest data, not silent drops
            record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        candidates.append(record)
    if best is None:
        errors = "; ".join(
            f"{c.get('error_type')}: {c.get('error')}" for c in candidates if c["status"] == "failed"
        )
        raise RuntimeError(f"all tuning candidates failed for {spec.name}: {errors}")
    if len(grid) > 1:
        best = make_method(spec.name, best_params, seed)
        best.fit(train, validation)
    return TuningResult(best, best_params, best_objective, best_score, candidates)
