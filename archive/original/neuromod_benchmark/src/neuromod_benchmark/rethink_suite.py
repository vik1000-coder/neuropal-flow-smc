"""Out-of-family synthetic stress suite for SID-driven neuromodulation methods.

This module deliberately lives beside, rather than inside, the frozen benchmark
orchestrators.  It tests predictive-law estimation under nonlinear dynamics,
non-Gaussian innovations, partial observation, interventions, receptor kinetics,
and semi-Markov state dynamics.  It does not expose latent modulators to fitted
methods and it never treats coefficient recovery as mechanistic identification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest

from .features import subset
from .registry import make_method
from .schema import SupervisedData


NOISE_FAMILIES = ("gaussian", "student_t", "skew_mixture", "correlated_jumps")
FAST_METHODS = (
    "constant_gaussian",
    "ridge_var",
    "heteroskedastic_ridge",
    "student_t_ridge",
    "gaussian_nll",
    "sid_hyvarinen",
    "sid_dsm_s075",
)
NEURAL_METHODS = (
    "gaussian_mlp_nll",
    "gaussian_mlp_dsm_s025",
    "mdn_k3",
    "sbtg_feature_student_s025",
)


@dataclass(frozen=True)
class SuiteConfig:
    n_neurons: int = 24
    n_hidden_modulators: int = 4
    n_steps: int = 420
    burn_in: int = 180
    train_episodes: int = 12
    validation_episodes: int = 4
    test_episodes: int = 6
    history_lags: tuple[int, ...] = (1, 4, 16)
    horizons: tuple[int, ...] = (1, 4, 16)
    seeds: tuple[int, ...] = (4101, 4102, 4103)
    neural_epochs: int = 28
    neural_cells: bool = True
    output_dir: str = "outputs/rethink_suite_20260712"


@dataclass
class Episode:
    latent: np.ndarray
    observed: np.ndarray
    stimulus: np.ndarray
    regime: np.ndarray
    modulator: np.ndarray
    metadata: dict


def _stable_matrix(rng: np.random.Generator, n: int, radius: float = 0.56) -> np.ndarray:
    raw = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.18)
    eig = float(np.max(np.abs(np.linalg.eigvals(raw))))
    return raw * (radius / max(eig, 1e-8))


def _standardized_noise(
    rng: np.random.Generator, family: str, n: int, factor: np.ndarray
) -> np.ndarray:
    if family == "gaussian":
        return rng.normal(size=n)
    if family == "student_t":
        return rng.standard_t(4.0, size=n) / np.sqrt(2.0)
    if family == "skew_mixture":
        # Centered and unit-variance two-component mixture with nonzero skew.
        choose = rng.random(n) < 0.18
        value = rng.normal(-0.35, 0.55, n)
        value[choose] = rng.normal(1.5944444444, 0.85, int(choose.sum()))
        return value / 1.016
    if family == "correlated_jumps":
        base = rng.normal(size=n)
        if rng.random() < 0.065:
            base += factor * rng.normal(0.0, 4.0)
        return base / np.sqrt(1.0 + 0.065 * 16.0 * factor**2)
    raise ValueError(f"unknown noise family {family!r}")


def _observe(
    latent: np.ndarray,
    rng: np.random.Generator,
    view: str,
    observed_fraction: float = 1.0,
) -> np.ndarray:
    if view == "latent":
        out = latent.copy()
    else:
        n = latent.shape[1]
        decay = rng.uniform(0.78, 0.96, n)
        calcium = np.zeros_like(latent)
        for t in range(1, len(latent)):
            events = np.log1p(np.exp(np.clip(2.0 * (latent[t] - 0.05), -15, 15))) / 2.0
            calcium[t] = decay * calcium[t - 1] + (1.0 - decay) * events
        if view == "calcium":
            out = calcium
        elif view in {"saturated", "artifact_missing"}:
            gain = rng.uniform(0.8, 1.25, n)
            out = gain * calcium / (0.5 + calcium)
            bleach = rng.uniform(-0.16, -0.04, n)[None, :] * np.linspace(0, 1, len(out))[:, None]
            artifact = np.sin(np.linspace(0, 7 * np.pi, len(out)) + rng.uniform(0, 2 * np.pi))
            out = out + bleach + 0.16 * artifact[:, None] * rng.normal(1.0, 0.12, n)
            out += rng.normal(0, 0.055 + 0.035 * np.sqrt(np.maximum(out, 0)), out.shape)
            if view == "artifact_missing":
                # Partial observation is represented by an actually narrower
                # measurement vector.  Keeping permanently unobserved neurons as
                # all-NaN columns would make a joint predictive target undefined.
                n_observed = max(2, int(round(observed_fraction * n)))
                # Keep a common registered subset across episodes. Imperfect
                # registration is a separate stressor and should not be silently
                # entangled with ordinary neuron dropout.
                keep = np.arange(n_observed)
                out = out[:, keep]
                frame_mask = rng.random(len(out)) < 0.055
                out[frame_mask] = np.nan
        else:
            raise ValueError(view)
    return out


def simulate_episode(
    *,
    seed: int,
    cfg: SuiteConfig,
    noise_family: str,
    view: str = "latent",
    generator: str = "reservoir",
    intervention: str = "none",
    input_scale: float = 1.0,
    state_mix: float = 1.0,
    observed_fraction: float = 1.0,
) -> Episode:
    rng = np.random.default_rng(seed)
    n, k = cfg.n_neurons, cfg.n_hidden_modulators
    total = cfg.n_steps + cfg.burn_in
    a = _stable_matrix(rng, n)
    b = rng.normal(0, 0.12 / np.sqrt(n), (n, n))
    mod_load = rng.normal(0, 0.14, (n, k))
    var_load = rng.normal(0, 0.12, (n, k))
    stim_load = rng.normal(0, 0.18, n)
    factor = rng.normal(size=n)
    factor /= max(np.linalg.norm(factor), 1e-8) / np.sqrt(n)
    release_map = rng.normal(0, 0.3, (k, n)) * (rng.random((k, n)) < 0.18)
    receptor_expr = rng.uniform(0.15, 1.0, (n, k)) * (rng.random((n, k)) < 0.42)
    kd = rng.uniform(0.35, 1.2, (n, k))

    x = np.zeros((total, n))
    m = np.zeros((total, k))
    receptor = np.zeros((total, n, k))
    adapt = np.zeros((total, n, k))
    regime = np.zeros(total, dtype=int)
    stimulus = np.zeros(total)
    pulse = rng.random(total) < 0.035
    stimulus[pulse] = input_scale * rng.uniform(0.6, 1.4, pulse.sum())
    dwell = 0
    target_dwell = int(rng.integers(18, 55))
    for t in range(1, total):
        if generator == "semi_markov":
            dwell += 1
            base_hazard = 0.012 + 0.08 / (1 + np.exp(-(dwell - target_dwell) / 5))
            hazard = np.clip(base_hazard * state_mix * np.exp(0.22 * m[t - 1, 0]), 0, 0.45)
            if rng.random() < hazard:
                regime[t] = 1 - regime[t - 1]
                dwell = 0
                target_dwell = int(rng.integers(18, 75))
            else:
                regime[t] = regime[t - 1]
        else:
            regime[t] = regime[t - 1]

        release = np.tanh(release_map @ x[t - 1])
        if intervention == "source_silencing":
            release[0] = 0.0
        m[t] = 0.965 * m[t - 1] + 0.045 * release + rng.normal(0, 0.018, k)
        if intervention == "ligand_pulse":
            m[t, 0] += 0.35 * stimulus[t]
        concentration = np.log1p(np.exp(m[t]))
        occ = receptor_expr * concentration[None, :] / (kd + concentration[None, :])
        adapt[t] = 0.94 * adapt[t - 1] + 0.06 * occ
        receptor[t] = occ * (1.0 - 0.35 * adapt[t])
        if intervention == "receptor_attenuation":
            receptor[t, :, 0] *= 0.25
        if intervention == "ligand_pulse":
            receptor[t, :, 0] += 0.35 * stimulus[t]

        state_bias = 0.22 * regime[t] * np.sign(stim_load) if generator == "semi_markov" else 0.0
        if generator == "hybrid_receptor":
            hidden = np.sum(mod_load * receptor[t], axis=1)
            log_scale = np.clip(np.sum(var_load * receptor[t], axis=1), -0.7, 0.7)
        else:
            effective_modulator = m[t].copy()
            if intervention == "receptor_attenuation":
                effective_modulator[0] *= 0.25
            hidden = mod_load @ effective_modulator
            log_scale = np.clip(var_load @ effective_modulator, -0.7, 0.7)
        mean = 0.56 * x[t - 1] + a @ np.tanh(1.25 * x[t - 1] + hidden)
        mean += 0.22 * np.sin(b @ x[t - 1]) + stim_load * stimulus[t] + state_bias
        eps = _standardized_noise(rng, noise_family, n, factor)
        x[t] = np.tanh(mean + 0.13 * np.exp(0.5 * log_scale) * eps)

    sl = slice(cfg.burn_in, None)
    latent = x[sl]
    observed = _observe(latent, np.random.default_rng(seed + 9_991), view, observed_fraction)
    return Episode(
        latent=latent,
        observed=observed,
        stimulus=stimulus[sl],
        regime=regime[sl],
        modulator=m[sl],
        metadata={
            "seed": seed,
            "noise_family": noise_family,
            "view": view,
            "generator": generator,
            "intervention": intervention,
            "input_scale": input_scale,
            "state_mix": state_mix,
        },
    )


def _causal_fill(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for j in range(out.shape[1]):
        last = 0.0
        for t in range(len(out)):
            if np.isfinite(out[t, j]):
                last = out[t, j]
            else:
                out[t, j] = last
    return out


def make_supervised(
    episodes: list[Episode], *, lags: tuple[int, ...], horizon: int, use_observed: bool = True
) -> SupervisedData:
    features, targets, groups, times = [], [], [], []
    example = episodes[0].observed if use_observed else episodes[0].latent
    n = example.shape[1]
    lags = tuple(sorted(set(lags)))
    names = tuple([f"x{s}:lag{lag}" for lag in lags for s in range(n)] + ["stimulus"])
    source = np.array([s for lag in lags for s in range(n)] + [-1], dtype=int)
    start = max(lags) - 1
    for group, episode in enumerate(episodes):
        raw = episode.observed if use_observed else episode.latent
        filled = _causal_fill(raw)
        stop = len(raw) - horizon
        rows = [
            np.concatenate([*[filled[t - lag + 1] for lag in lags], [episode.stimulus[t]]])
            for t in range(start, stop)
        ]
        y = raw[np.arange(start, stop) + horizon]
        valid = np.all(np.isfinite(y), axis=1)
        if valid.any():
            features.append(np.asarray(rows)[valid])
            targets.append(y[valid])
            groups.append(np.full(valid.sum(), group, dtype=int))
            times.append(np.arange(start, stop)[valid])
    g = np.concatenate(groups)
    return SupervisedData(
        features=np.concatenate(features), targets=np.concatenate(targets), groups=g,
        trajectory_ids=g.copy(), times=np.concatenate(times), feature_names=names,
        source_index=source, view="observed" if use_observed else "latent", horizon=horizon,
    )


def _split(data: SupervisedData, n_train: int, n_validation: int):
    return (
        subset(data, data.groups < n_train),
        subset(data, (data.groups >= n_train) & (data.groups < n_train + n_validation)),
        subset(data, data.groups >= n_train + n_validation),
    )


def _method_params(name: str, cfg: SuiteConfig) -> dict:
    if name in {"sid_hyvarinen", "sid_dsm_s075"}:
        return {"ridge": 0.05, "n_corruptions": 2}
    if name == "gaussian_nll":
        return {"ridge": 0.05, "maxiter": 120}
    if name == "student_t_ridge":
        return {"mean_ridge": 1.0, "variance_ridge": 10.0, "df": 4.0}
    if name in {"gaussian_mlp_nll", "gaussian_mlp_dsm_s025", "mdn_k3"}:
        return {
            "hidden": 48, "layers": 2, "max_epochs": cfg.neural_epochs,
            "patience": 6, "batch_size": 384, "device": "mps", "weight_decay": 1e-3,
        }
    if name == "sbtg_feature_student_s025":
        return {
            "hidden": 40, "max_epochs": cfg.neural_epochs, "patience": 6,
            "batch_size": 384, "device": "mps", "weight_decay": 1e-3,
        }
    return {}


def _score_prediction(prediction, targets: np.ndarray) -> dict:
    error = prediction.mean - targets
    rmse = float(np.sqrt(np.mean(error**2)))
    scale = float(np.std(targets))
    nrmse = rmse / max(scale, 1e-8)
    result = {"rmse": rmse, "nrmse": nrmse}
    if prediction.log_prob is not None and np.isfinite(prediction.log_prob).any():
        result["nll"] = float(-np.nanmean(prediction.log_prob))
    else:
        result["nll"] = np.nan
    if prediction.cdf is not None:
        pit = np.asarray(prediction.cdf).ravel()
        pit = pit[np.isfinite(pit)]
        result["pit_ks"] = float(kstest(pit, "uniform").statistic) if len(pit) else np.nan
        result["coverage90_error"] = float(abs(np.mean((pit >= 0.05) & (pit <= 0.95)) - 0.90))
    else:
        result["pit_ks"] = np.nan
        result["coverage90_error"] = np.nan
    standardized = np.abs(error) / np.sqrt(np.maximum(prediction.variance, 1e-8))
    result["extreme_residual_rate"] = float(np.mean(standardized > 2.5))
    finite = bool(np.isfinite(prediction.mean).all() and np.isfinite(prediction.variance).all())
    result["numerical_valid"] = bool(prediction.metadata.get("numerical_valid", finite) and finite)
    result["invalid_variance_fraction"] = float(
        prediction.metadata.get("invalid_variance_fraction", 0.0)
    )
    result["extreme_variance_fraction"] = float(
        prediction.metadata.get("extreme_variance_fraction", 0.0)
    )
    return result


def evaluate_cell(
    *, cell: dict, episodes: list[Episode], cfg: SuiteConfig, methods: Iterable[str],
    control_episodes: list[Episode] | None = None,
) -> list[dict]:
    data = make_supervised(episodes, lags=cfg.history_lags, horizon=int(cell["horizon"]))
    train, validation, test = _split(data, cfg.train_episodes, cfg.validation_episodes)
    control_test = None
    if control_episodes is not None:
        control_data = make_supervised(
            control_episodes, lags=cfg.history_lags, horizon=int(cell["horizon"])
        )
        _, _, control_test = _split(control_data, cfg.train_episodes, cfg.validation_episodes)
    records = []
    for method_name in methods:
        started = time.perf_counter()
        record = dict(cell, method=method_name, n_train=len(train.targets), n_test=len(test.targets))
        try:
            model = make_method(method_name, _method_params(method_name, cfg), int(cell["seed"]))
            model.fit(train, validation)
            if method_name.startswith("sbtg_"):
                # SBTG is a score model, not a normalized density. Preserve its
                # honest held-out DSM risk without fabricating NLL or moments.
                record.update(
                    rmse=np.nan, nrmse=np.nan, nll=np.nan, pit_ks=np.nan,
                    coverage90_error=np.nan, extreme_residual_rate=np.nan,
                    numerical_valid=True, score_risk=model.dsm_risk(test, seed=int(cell["seed"]) + 17),
                )
            else:
                prediction = model.predict(test, n_samples=0)
                record.update(_score_prediction(prediction, test.targets), score_risk=np.nan)
                if control_test is not None:
                    control_prediction = model.predict(control_test, n_samples=0)
                    rows = min(len(test.targets), len(control_test.targets))
                    true_delta = test.targets[:rows] - control_test.targets[:rows]
                    predicted_delta = prediction.mean[:rows] - control_prediction.mean[:rows]
                    response_rmse = float(np.sqrt(np.mean((predicted_delta - true_delta) ** 2)))
                    record.update(
                        response_rmse=response_rmse,
                        response_nrmse=response_rmse / max(float(np.std(test.targets[:rows])), 1e-8),
                        true_response_rms=float(np.sqrt(np.mean(true_delta**2))),
                        predicted_response_rms=float(np.sqrt(np.mean(predicted_delta**2))),
                    )
            record.update(status="ok", runtime_seconds=time.perf_counter() - started)
        except Exception as exc:  # persist failures rather than silently dropping methods
            record.update(
                status="failed", error=f"{type(exc).__name__}: {exc}",
                runtime_seconds=time.perf_counter() - started, rmse=np.nan, nrmse=np.nan,
                nll=np.nan, pit_ks=np.nan, coverage90_error=np.nan,
                extreme_residual_rate=np.nan, numerical_valid=False, score_risk=np.nan,
                invalid_variance_fraction=np.nan, extreme_variance_fraction=np.nan,
            )
        records.append(record)
    return records


def _episodes_for_cell(cell: dict, cfg: SuiteConfig) -> list[Episode]:
    total = cfg.train_episodes + cfg.validation_episodes + cfg.test_episodes
    episodes = []
    for episode in range(total):
        family = cell.get("noise_family", "gaussian")
        if family == "pooled_train_ood_test":
            family = NOISE_FAMILIES[episode % 3] if episode < cfg.train_episodes + cfg.validation_episodes else "correlated_jumps"
        episodes.append(simulate_episode(
            seed=int(cell["seed"]) + 1009 * episode,
            cfg=cfg,
            noise_family=family,
            view=cell.get("view", "latent"),
            generator=cell.get("generator", "reservoir"),
            intervention=cell.get("intervention", "none") if episode >= cfg.train_episodes + cfg.validation_episodes else "none",
            input_scale=float(cell.get("test_input_scale", 1.0)) if episode >= cfg.train_episodes + cfg.validation_episodes else 1.0,
            state_mix=float(cell.get("test_state_mix", 1.0)) if episode >= cfg.train_episodes + cfg.validation_episodes else 1.0,
            observed_fraction=float(cell.get("observed_fraction", 1.0)),
        ))
    return episodes


def _paired_control_episodes(cell: dict, cfg: SuiteConfig) -> list[Episode] | None:
    if cell.get("intervention", "none") == "none":
        return None
    control = dict(cell, intervention="none")
    return _episodes_for_cell(control, cfg)


def build_cells(cfg: SuiteConfig) -> list[dict]:
    cells = []
    for seed in cfg.seeds:
        for family in NOISE_FAMILIES:
            for horizon in cfg.horizons:
                cells.append(dict(experiment="E1", generator="reservoir", seed=seed,
                                  noise_family=family, view="latent", horizon=horizon,
                                  intervention="none", stress="law_misspecification"))
        for view, fraction in (("latent", 1.0), ("calcium", 1.0), ("saturated", 1.0), ("artifact_missing", 0.67)):
            for horizon in (1, 4, 16):
                cells.append(dict(experiment="E2", generator="reservoir", seed=seed,
                                  noise_family="student_t", view=view, horizon=horizon,
                                  observed_fraction=fraction, intervention="none",
                                  stress="measurement_partial_observation"))
        for intervention in ("none", "source_silencing", "ligand_pulse", "receptor_attenuation"):
            cells.append(dict(experiment="E3", generator="reservoir", seed=seed,
                              noise_family="pooled_train_ood_test", view="artifact_missing", horizon=16,
                              observed_fraction=0.67, intervention=intervention,
                              test_input_scale=1.7, test_state_mix=1.35,
                              stress="joint_ood_intervention"))
        for family in ("student_t", "correlated_jumps"):
            cells.append(dict(experiment="E4", generator="hybrid_receptor", seed=seed,
                              noise_family=family, view="saturated", horizon=4,
                              intervention="receptor_attenuation", stress="receptor_kinetics"))
        for mix in (0.65, 1.5):
            cells.append(dict(experiment="E5", generator="semi_markov", seed=seed,
                              noise_family="skew_mixture", view="saturated", horizon=16,
                              intervention="none", test_state_mix=mix,
                              stress="semi_markov_path_law"))
    return cells


def _neural_cell(cell: dict) -> bool:
    return (
        (cell["experiment"] == "E1" and cell["noise_family"] in {"student_t", "correlated_jumps"} and cell["horizon"] in {1, 16})
        or (cell["experiment"] == "E2" and cell["view"] in {"latent", "artifact_missing"} and cell["horizon"] == 4)
        or cell["experiment"] in {"E3", "E4", "E5"}
    )


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    ok = records[records.status == "ok"].copy()
    keys = ["experiment", "stress", "method"]
    return ok.groupby(keys, as_index=False).agg(
        cells=("nll", "size"), nll_mean=("nll", "mean"), nll_sd=("nll", "std"),
        nrmse_mean=("nrmse", "mean"), pit_ks_mean=("pit_ks", "mean"),
        coverage90_error_mean=("coverage90_error", "mean"),
        extreme_residual_rate_mean=("extreme_residual_rate", "mean"),
        numerical_valid_rate=("numerical_valid", "mean"),
        runtime_seconds=("runtime_seconds", "sum"),
    )


def _paired_skill(records: pd.DataFrame) -> pd.DataFrame:
    ok = records[(records.status == "ok") & np.isfinite(records.nll)].copy()
    index = [c for c in ["experiment", "seed", "noise_family", "view", "horizon", "intervention", "test_state_mix"] if c in ok]
    ok[index] = ok[index].fillna("__not_applicable__")
    wide = ok.pivot_table(index=index, columns="method", values="nll", aggfunc="first")
    if "ridge_var" not in wide:
        return pd.DataFrame()
    rows = []
    for method in wide.columns:
        if method == "ridge_var":
            continue
        paired = wide[["ridge_var", method]].dropna()
        if not len(paired):
            continue
        delta = paired["ridge_var"] - paired[method]
        rows.append({"method": method, "paired_cells": len(delta), "nll_skill_vs_ridge": float(delta.mean()),
                     "win_rate_vs_ridge": float(np.mean(delta > 0)), "delta_sd": float(delta.std(ddof=1)) if len(delta)>1 else np.nan})
    return pd.DataFrame(rows).sort_values("nll_skill_vs_ridge", ascending=False)


def _write_report(out: Path, cfg: SuiteConfig, records: pd.DataFrame, summary: pd.DataFrame, skill: pd.DataFrame):
    ok = records[records.status == "ok"]
    failures = records[records.status != "ok"]
    best = summary.sort_values(["experiment", "nll_mean"]).groupby("experiment").first().reset_index()
    lines = [
        "# Out-of-family neuromodulation synthetic benchmark",
        "",
        "## Technical summary",
        "",
        f"The suite executed **{len(records)} method-cell fits** across E1–E5; **{len(ok)} succeeded** and **{len(failures)} failed**. "
        "Fast estimand-matched methods ran on every cell; neural density, denoising, and SBTG methods ran on predeclared representative stress cells.",
        "",
        "This is a predictive-law and robustness benchmark. It does not claim latent receptor or anatomical-edge identification from observed fluorescence.",
        "",
        "## Best average normalized likelihood by experiment",
        "",
        "| Experiment | Method | Mean NLL | Mean normalized RMSE | Mean PIT KS | Cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        lines.append(f"| {row.experiment} | {row.method} | {row.nll_mean:.4f} | {row.nrmse_mean:.4f} | {row.pit_ks_mean:.4f} | {int(row.cells)} |")
    lines += ["", "## Paired NLL skill relative to ridge VAR", ""]
    if len(skill):
        lines += ["| Method | Paired cells | NLL skill (positive is better) | Win rate |",
                  "|---|---:|---:|---:|"]
        for _, row in skill.iterrows():
            lines.append(f"| {row.method} | {int(row.paired_cells)} | {row.nll_skill_vs_ridge:.4f} | {row.win_rate_vs_ridge:.3f} |")
    else:
        lines.append("No paired likelihood comparisons were available.")
    sbtg = records[(records.status == "ok") & records.method.str.startswith("sbtg_")]
    if len(sbtg):
        lines += ["", "## SBTG held-out denoising-score risk", "",
                  "SBTG does not emit a normalized predictive density, so it is not placed in the NLL ranking. Its native held-out DSM risk is shown only within a fixed score domain.", "",
                  "| Experiment | Mean held-out DSM risk | Cells |", "|---|---:|---:|"]
        for experiment, part in sbtg.groupby("experiment"):
            lines.append(f"| {experiment} | {part.score_risk.mean():.4f} | {len(part)} |")
    lines += [
        "", "## Experiment definitions", "",
        "- **E1:** nonlinear reservoir, hidden modulators, four innovation families, horizons 1/4/16, clean latent observations.",
        "- **E2:** identical model family under latent, calcium, saturated fluorescence, and shared-artifact/structured-missing views.",
        "- **E3:** joint OOD test: held-out correlated-jump noise, stronger inputs, changed state mixture, partial observation, and controlled operations.",
        "- **E4:** stochastic activity-dependent release, saturating receptor occupancy, desensitization, and receptor attenuation.",
        "- **E5:** semi-Markov switching with modulator-dependent dwell hazard and held-out state mixtures.",
        "", "## Evidence boundaries", "",
        "- Episode-level train/validation/test separation prevents overlapping-window leakage.",
        "- E3 training excludes the test operation and holds out the correlated-jump innovation family.",
        "- Active E3/E4 operations are additionally scored by paired common-random-number response-contrast error.",
        "- NLL comparisons are valid only for methods emitting normalized predictive laws; score-only SBTG failures or missing likelihoods remain visible.",
        "- The present run is a multi-seed pilot-sized adjudication, not a final power study. A publishable claim should expand seeds after inspecting failure and truth-overlap diagnostics.",
    ]
    if len(failures):
        lines += ["", "## Persisted method failures", "", "| Method | Count | First error |", "|---|---:|---|"]
        for method, part in failures.groupby("method"):
            lines.append(f"| {method} | {len(part)} | {str(part.error.iloc[0]).replace('|','/')} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def _plot(out: Path, summary: pd.DataFrame):
    methods = list(dict.fromkeys(summary.method))
    experiments = [f"E{i}" for i in range(1, 6)]
    matrix = np.full((len(methods), len(experiments)), np.nan)
    for i, method in enumerate(methods):
        for j, experiment in enumerate(experiments):
            part = summary[(summary.method == method) & (summary.experiment == experiment)]
            if len(part): matrix[i, j] = part.nll_mean.iloc[0]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(methods))))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(experiments)), experiments)
    ax.set_yticks(range(len(methods)), methods)
    ax.set_title("Mean predictive NLL by stress experiment (lower is better)")
    for i in range(len(methods)):
        for j in range(len(experiments)):
            if np.isfinite(matrix[i, j]): ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="NLL")
    fig.tight_layout()
    fig.savefig(out / "nll_heatmap.png", dpi=180)
    plt.close(fig)


def run(cfg: SuiteConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = build_cells(cfg)
    manifest = {
        "config": asdict(cfg), "cells": cells, "fast_methods": FAST_METHODS,
        "neural_methods": NEURAL_METHODS, "noise_families": NOISE_FAMILIES,
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "started_unix": time.time(),
    }
    payload = json.dumps(manifest, sort_keys=True, default=list).encode()
    manifest["design_sha256"] = hashlib.sha256(payload).hexdigest()
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=list) + "\n")
    records: list[dict] = []
    checkpoint = out / "metrics_checkpoint.csv"
    for number, cell in enumerate(cells, start=1):
        episodes = _episodes_for_cell(cell, cfg)
        methods = list(FAST_METHODS)
        if cfg.neural_cells and _neural_cell(cell): methods += list(NEURAL_METHODS)
        records.extend(evaluate_cell(
            cell=cell, episodes=episodes, cfg=cfg, methods=methods,
            control_episodes=_paired_control_episodes(cell, cfg),
        ))
        pd.DataFrame(records).to_csv(checkpoint, index=False)
        print(f"[{number:03d}/{len(cells):03d}] {cell['experiment']} seed={cell['seed']} {cell['stress']} complete", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "metrics_tidy.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    skill = _paired_skill(frame)
    skill.to_csv(out / "paired_nll_skill.csv", index=False)
    _plot(out, summary)
    _write_report(out, cfg, frame, summary, skill)
    validation = {
        "rows": len(frame), "successful_rows": int((frame.status == "ok").sum()),
        "failed_rows": int((frame.status != "ok").sum()),
        "all_experiments_present": sorted(frame.experiment.unique().tolist()) == ["E1", "E2", "E3", "E4", "E5"],
        "all_fast_cells_covered": bool((frame.groupby(["experiment", "seed", "stress", "horizon"])["method"].apply(lambda s: set(FAST_METHODS).issubset(set(s)))).all()),
        "finite_normalized_law_metrics": bool(np.isfinite(frame.loc[(frame.status == "ok") & ~frame.method.str.startswith("sbtg_"), ["nrmse", "nll", "pit_ks"]]).all().all()),
        "finite_sbtg_score_risk": bool(np.isfinite(frame.loc[(frame.status == "ok") & frame.method.str.startswith("sbtg_"), "score_risk"]).all()),
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=SuiteConfig.output_dir)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SuiteConfig.seeds))
    parser.add_argument("--neurons", type=int, default=SuiteConfig.n_neurons)
    parser.add_argument("--steps", type=int, default=SuiteConfig.n_steps)
    parser.add_argument("--neural-epochs", type=int, default=SuiteConfig.neural_epochs)
    parser.add_argument("--skip-neural", action="store_true")
    args = parser.parse_args(argv)
    cfg = SuiteConfig(output_dir=args.output, seeds=tuple(args.seeds), n_neurons=args.neurons,
                      n_steps=args.steps, neural_epochs=args.neural_epochs,
                      neural_cells=not args.skip_neural)
    print(run(cfg))


if __name__ == "__main__":
    main()
