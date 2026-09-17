from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.core import (
    causal_fill,
    fit_anchor_projection,
    systematic_resample,
)
from compatibility_neural_benchmark.distributional_lag_analysis import (
    worm_bootstrap_interval,
)
from compatibility_neural_benchmark.distributional_lag_audit import (
    conditional_features,
    deterministic_seed,
    distributional_effects,
    evaluation_targets,
    fit_conditional_resampler,
)
from compatibility_neural_benchmark.latent_distributional_audit import (
    fold_parameters,
    predict_parameters,
    standardized_traces,
)
from compatibility_neural_benchmark.latent_paired_rollout import projected_batch
from conditional_neural_benchmark.data import _stimulus_features, load_cohort
from conditional_neural_benchmark.latent_calcium_lag import innovation_trace, load_folds


DEFAULT_N_PARTICLES = 64
SOURCE_LAG = 32
HORIZON = 16
REPEATS = (3101, 4703, 7907)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def normalized_weights(log_weight: np.ndarray) -> np.ndarray:
    shifted = np.asarray(log_weight, dtype=np.float64) - np.max(log_weight)
    weight = np.exp(shifted)
    return weight / weight.sum()


def smc_initial_histories(
    *,
    observed_history: np.ndarray,
    innovation_history: np.ndarray,
    source: int,
    target_low: float,
    target_high: float,
    proposal_sd: float,
    clip_low: float,
    clip_high: float,
    seed: int,
    n_particles: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    midpoint = 0.5 * (target_low + target_high)
    proposal = np.clip(
        rng.normal(midpoint, max(proposal_sd, 1e-3), n_particles),
        clip_low,
        clip_high,
    )
    bandwidth = max(0.25 * (target_high - target_low), 0.02)
    low_weight = normalized_weights(-0.5 * np.square((proposal - target_low) / bandwidth))
    high_weight = normalized_weights(-0.5 * np.square((proposal - target_high) / bandwidth))
    common_offset = float(rng.random())
    low_index = systematic_resample(low_weight, rng, offset=common_offset)
    high_index = systematic_resample(high_weight, rng, offset=common_offset)
    low_values, high_values = proposal[low_index], proposal[high_index]
    observed = np.broadcast_to(
        observed_history, (3, n_particles, *observed_history.shape)
    ).copy()
    innovation = np.broadcast_to(
        innovation_history, (3, n_particles, *innovation_history.shape)
    ).copy()
    innovation[0, :, -SOURCE_LAG, source] = low_values
    innovation[1, :, -SOURCE_LAG, source] = high_values
    diagnostics = {
        "target_low": float(target_low),
        "target_high": float(target_high),
        "target_gap": float(target_high - target_low),
        "achieved_low": float(low_values.mean()),
        "achieved_high": float(high_values.mean()),
        "achieved_gap": float(high_values.mean() - low_values.mean()),
        "ess_low": float(1.0 / np.square(low_weight).sum()),
        "ess_high": float(1.0 / np.square(high_weight).sum()),
        "max_weight_low": float(low_weight.max()),
        "max_weight_high": float(high_weight.max()),
        "distinct_ancestors_low": float(len(np.unique(low_index))),
        "distinct_ancestors_high": float(len(np.unique(high_index))),
    }
    diagnostics["valid"] = float(
        diagnostics["ess_low"] >= 20
        and diagnostics["ess_high"] >= 20
        and diagnostics["max_weight_low"] <= 0.20
        and diagnostics["max_weight_high"] <= 0.20
        and diagnostics["achieved_gap"] >= 0.25 * diagnostics["target_gap"]
    )
    return observed, innovation, diagnostics


def rollout_particles(
    *,
    observed: np.ndarray,
    innovation: np.ndarray,
    stimulus_history: np.ndarray,
    future_stimulus: np.ndarray,
    parameters: dict[str, np.ndarray],
    basis: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_particles = observed.shape[1]
    stimulus = np.broadcast_to(
        stimulus_history, (3, n_particles, len(stimulus_history))
    ).copy()
    rng = np.random.default_rng(seed)
    d = observed.shape[-1]
    next_observed = np.empty((3, n_particles, d), dtype=np.float32)
    for step in range(1, HORIZON + 1):
        flat_observed = observed.reshape(-1, len(basis), d)
        flat_innovation = innovation.reshape(-1, len(basis), d)
        flat_stimulus = stimulus.reshape(-1, len(basis))
        neural, common = projected_batch(
            flat_observed,
            flat_innovation,
            flat_stimulus,
            basis,
            parameters["pca_components"],
        )
        mean, logvar = predict_parameters(neural, common, parameters)
        base = (
            parameters["intercept"][None]
            + parameters["alpha"][None] * flat_observed[:, -1]
        )
        common_noise = rng.normal(size=(n_particles, d)).astype(np.float32)
        noise = np.broadcast_to(common_noise, (3, n_particles, d)).reshape(-1, d)
        next_innovation = (mean + np.exp(0.5 * logvar) * noise).reshape(
            3, n_particles, d
        )
        next_observed = (base.reshape(3, n_particles, d) + next_innovation).astype(
            np.float32
        )
        observed = np.concatenate(
            [observed[:, :, 1:], next_observed[:, :, None]], axis=2
        )
        innovation = np.concatenate(
            [innovation[:, :, 1:], next_innovation[:, :, None]], axis=2
        )
        next_stimulus = np.full(
            (3, n_particles, 1), future_stimulus[step - 1], dtype=np.float32
        )
        stimulus = np.concatenate([stimulus[:, :, 1:], next_stimulus], axis=2)
    return next_observed[0], next_observed[1], next_observed[2]


def run(
    cohort,
    folds: np.ndarray,
    model_dir: Path,
    basis: np.ndarray,
    source: int,
    target: int,
    n_particles: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        parameters = fold_parameters(model_dir / f"fold_{fold}.npz")
        traces = standardized_traces(cohort, parameters)
        innovations = [
            innovation_trace(trace, parameters["alpha"], parameters["intercept"])
            for trace in traces
        ]
        training = np.flatnonzero(folds != fold)
        heldout = np.flatnonzero(folds == fold)
        projection = fit_anchor_projection(
            [innovations[int(worm)] for worm in training], 8
        )
        resampler = fit_conditional_resampler(
            [innovations[int(worm)] for worm in training],
            [cohort.stimulus_schedules[int(worm)] for worm in training],
            state="onset",
            lag=SOURCE_LAG,
            projection=projection,
            ridge_alpha=2.0,
            residual_quantiles=(0.25, 0.75),
            clip_quantiles=(0.01, 0.99),
        )
        pooled = np.concatenate([traces[int(worm)] for worm in training], axis=0)
        tail_threshold = np.quantile(np.abs(pooled), 0.90, axis=0).astype(np.float32)
        for worm in heldout:
            trace = traces[int(worm)]
            innovation = innovations[int(worm)]
            schedule = cohort.stimulus_schedules[int(worm)]
            stimulus = _stimulus_features(
                len(trace), schedule, "binary_any_stimulus"
            )[:, 0]
            onset_targets = [
                (event, time)
                for state, event, time in evaluation_targets(schedule, len(trace))
                if state == "onset"
            ]
            for repeat in REPEATS:
                event_metrics: list[dict[str, np.ndarray]] = []
                event_diagnostics: list[dict[str, float]] = []
                for event, target_time in onset_targets:
                    features = conditional_features(
                        innovation, target_time, SOURCE_LAG, projection
                    )
                    low, high = resampler.predict_low_high(features)
                    proposal_sd = float(
                        max((high[source] - low[source]) / 1.349, 0.02)
                    )
                    observed, latent, diagnostics = smc_initial_histories(
                        observed_history=trace[target_time - len(basis) : target_time],
                        innovation_history=innovation[target_time - len(basis) : target_time],
                        source=source,
                        target_low=float(low[source]),
                        target_high=float(high[source]),
                        proposal_sd=proposal_sd,
                        clip_low=float(resampler.clip_low[source]),
                        clip_high=float(resampler.clip_high[source]),
                        seed=deterministic_seed(
                            "targeted_smc_init", fold, int(worm), event, repeat
                        ),
                        n_particles=n_particles,
                    )
                    low_samples, high_samples, factual_samples = rollout_particles(
                        observed=observed,
                        innovation=latent,
                        stimulus_history=stimulus[target_time - len(basis) : target_time],
                        future_stimulus=stimulus[target_time : target_time + HORIZON],
                        parameters=parameters,
                        basis=basis,
                        seed=deterministic_seed(
                            "targeted_smc_path", fold, int(worm), event, repeat
                        ),
                    )
                    event_metrics.append(
                        distributional_effects(
                            low_samples[None],
                            high_samples[None],
                            factual_samples[None],
                            trace[target_time + HORIZON - 1],
                            tail_threshold,
                        )
                    )
                    event_diagnostics.append(diagnostics)
                metric = {
                    key: float(np.mean([value[key][0, target] for value in event_metrics]))
                    for key in (
                        "mean_shift",
                        "log_sd_shift",
                        "tail_probability_shift",
                        "wasserstein1",
                        "proper_energy_penalty",
                    )
                }
                diagnostic = {
                    key: float(np.mean([value[key] for value in event_diagnostics]))
                    for key in event_diagnostics[0]
                }
                rows.append(
                    {
                        "fold": fold,
                        "worm_id": cohort.worm_ids[int(worm)],
                        "repeat": repeat,
                        "source": cohort.neurons[source],
                        "target": cohort.neurons[target],
                        "source_lag_frames": SOURCE_LAG,
                        "source_lag_seconds": SOURCE_LAG / cohort.fps,
                        "horizon_frames": HORIZON,
                        "horizon_seconds": HORIZON / cohort.fps,
                        "n_particles": n_particles,
                        **metric,
                        **diagnostic,
                    }
                )
            print(f"TARGETED_SMC fold={fold} worm={schedule.worm_id}", flush=True)
    return pd.DataFrame(rows)


def analyze(frame: pd.DataFrame, direct_row: pd.Series) -> tuple[dict, str]:
    particle_counts = frame.n_particles.unique()
    if len(particle_counts) != 1:
        raise RuntimeError("a targeted SMC report must contain one particle count")
    n_particles = int(particle_counts[0])
    per_worm = frame.groupby("worm_id", as_index=False).agg(
        log_sd_shift=("log_sd_shift", "mean"),
        proper_energy_penalty=("proper_energy_penalty", "mean"),
        wasserstein1=("wasserstein1", "mean"),
        valid=("valid", "mean"),
    )
    signed_low, signed_high = worm_bootstrap_interval(
        per_worm.log_sd_shift.to_numpy(), seed=20260828
    )
    proper_low, proper_high = worm_bootstrap_interval(
        per_worm.proper_energy_penalty.to_numpy(), seed=20260829
    )
    repeat_summary = frame.groupby("repeat", as_index=False).agg(
        log_sd_shift=("log_sd_shift", "mean"),
        proper_energy_penalty=("proper_energy_penalty", "mean"),
        valid_fraction=("valid", "mean"),
        median_ess_low=("ess_low", "median"),
        median_ess_high=("ess_high", "median"),
    )
    if str(direct_row.dominant_signed_metric) != "log_sd_shift":
        raise RuntimeError(
            "targeted SMC v1 expects a propagated log-SD effect, got "
            f"{direct_row.dominant_signed_metric!r}"
        )
    direct_effect = float(direct_row.signed_effect)
    smc_effect = float(per_worm.log_sd_shift.mean())
    sign_agreement = bool(np.sign(smc_effect) == np.sign(direct_effect))
    magnitude_ratio = float(abs(smc_effect) / max(abs(direct_effect), 1e-12))
    repeat_signs = np.sign(repeat_summary.log_sd_shift.to_numpy())
    repeat_consistency = float(np.mean(repeat_signs == np.sign(smc_effect)))
    valid_fraction = float(frame.valid.mean())
    passed = bool(
        sign_agreement
        and 1 / 3 <= magnitude_ratio <= 3
        and signed_high < 0
        and proper_low > 0
        and repeat_consistency == 1.0
        and valid_fraction >= 0.80
    )
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "direct_log_sd_effect": direct_effect,
        "smc_log_sd_effect": smc_effect,
        "smc_signed_95_ci": [signed_low, signed_high],
        "smc_proper_energy_penalty": float(per_worm.proper_energy_penalty.mean()),
        "smc_proper_95_ci": [proper_low, proper_high],
        "direct_smc_sign_agreement": sign_agreement,
        "smc_to_direct_magnitude_ratio": magnitude_ratio,
        "repeat_sign_consistency": repeat_consistency,
        "valid_fraction": valid_fraction,
        "all_gates_pass": passed,
        "promoted_effect": "AIY→AIZ onset-conditioned 8-s source lag / 4-s forecast scale reduction" if passed else None,
        "claim_boundary": "model-relative targeted ESS-SMC sensitivity, not causal anatomy or physical delay",
    }
    lines = [
        "# Targeted latent ESS-SMC sensitivity",
        "",
        "## Decision",
        "",
        (
            "**The targeted AIY→AIZ distributional effect passes the direct/SMC agreement gate.**"
            if passed
            else "**The targeted AIY→AIZ effect does not pass the direct/SMC agreement gate.**"
        ),
        "",
        f"The direct paired rollout estimated log-SD shift {direct_effect:+.6f}. Targeted N={n_particles} ESS-SMC estimates {smc_effect:+.6f} with worm-bootstrap 95% CI [{signed_low:+.6f}, {signed_high:+.6f}].",
        "",
        f"The SMC proper-energy penalty is {per_worm.proper_energy_penalty.mean():+.6f} [{proper_low:+.6f}, {proper_high:+.6f}]. Validity is {valid_fraction:.1%}; all three Monte Carlo repeats have the same sign: {'yes' if repeat_consistency == 1 else 'no'}.",
        "",
        "This SMC applies one ESS-diagnosed soft clamp to the conditionally resampled latent source coordinate and then performs free common-noise rollout. It is an estimator sensitivity for the learned observational law, not a physical intervention or transmission-delay estimate.",
        "",
    ]
    return decision, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--fold-evidence", type=Path, required=True)
    parser.add_argument("--n-particles", type=int, default=DEFAULT_N_PARTICLES)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rollout = args.rollout_dir.resolve()
    rollout_gate = json.loads((rollout / "gate_decision.json").read_text())
    if not rollout_gate.get("temporal_cut_ess_smc_authorized"):
        raise RuntimeError("paired rollout did not authorize ESS-SMC")
    edge_horizon = pd.read_csv(rollout / "edge_horizon_gate.csv")
    selected = edge_horizon[edge_horizon.propagation_pass]
    if len(selected) != 1:
        raise RuntimeError("targeted v1 expects exactly one propagated edge")
    row = selected.iloc[0]
    cohort = load_cohort(cohort_mode="oh16230_head")
    source = cohort.neurons.index(str(row.source))
    target = cohort.neurons.index(str(row.target))
    model_dir = args.model_dir.resolve()
    basis = np.load(model_dir / "lag_basis.npy")
    folds = load_folds(args.fold_evidence.resolve(), cohort)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "protocol": "targeted conditional-source soft-clamp ESS-SMC v1",
        "model_dir": str(model_dir),
        "rollout_dir": str(rollout),
        "source": str(row.source),
        "target": str(row.target),
        "source_lag_frames": SOURCE_LAG,
        "horizon_frames": HORIZON,
        "n_particles": args.n_particles,
        "repeats": list(REPEATS),
        "minimum_ess": 20,
        "maximum_weight": 0.20,
        "minimum_achieved_gap_fraction": 0.25,
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "selection_firewall": "no atlas/connectome/receptor/SBTG target used",
        "claim_boundary": "model-relative targeted ESS-SMC sensitivity, not causal anatomy or physical delay",
    }
    atomic_json(output / "manifest.json", manifest)
    frame = run(
        cohort, folds, model_dir, basis, source, target, args.n_particles
    )
    frame.to_csv(output / "worm_repeat_metrics.csv", index=False)
    frame.groupby("repeat", as_index=False).mean(numeric_only=True).to_csv(
        output / "repeat_summary.csv", index=False
    )
    decision, report = analyze(frame, row)
    atomic_json(output / "gate_decision.json", decision)
    (output / "REPORT.md").write_text(report)
    manifest["status"] = "complete"
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["gate_decision"] = decision
    atomic_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
