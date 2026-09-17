"""Sample-only nonlinear measurement/OOD pilot for the exact G8 path process.

The latent conditional law is exact G8, but calcium filtering, saturation,
bleaching, shared artifacts, and heteroscedastic photon noise make the observed
law sample-only.  All backends receive the same training data and are evaluated
with fair energy score and paired finite typed path contrasts.  This is a
developmental stress test, not a confirmatory biological simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from history_tangent_benchmark.dgps import make_dgp
from history_tangent_benchmark.metrics import TrainScaler, energy_score_fair
from history_tangent_benchmark.models import build_model, fit_model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "path_measurement_stress_2026-07-13"
SYSTEM_SEEDS = (73, 79, 83)
N_TRAIN = 8_000
N_VALIDATION = 1_600
N_TEST = 128
N_SAMPLES = 64
DELTA = 0.2


@dataclass(frozen=True)
class MeasurementView:
    name: str
    gain: float
    bleach: float
    artifact: float
    photon: float


ID_VIEW = MeasurementView("measured_id", 1.4, 0.18, 0.10, 0.035)
SHIFT_VIEW = MeasurementView("measured_shift", 1.9, 0.30, 0.22, 0.060)


MODELS = {
    "gaussian_nll": ("heteroscedastic_gaussian", {"hidden": 64, "layers": 2}),
    "autoregressive_mdn": (
        "autoregressive_mdn", {"hidden": 56, "layers": 2, "components": 8}
    ),
    "affine_flow": (
        "conditional_affine_flow",
        {"hidden": 64, "layers": 2, "coupling_layers": 6, "max_log_scale": 1.5},
    ),
    "bounded_energy": (
        "bounded_energy_ratio",
        {"hidden": 64, "layers": 2, "tilt_bound": 1.0, "base_nll_weight": 0.25,
         "oversample": 6, "centering_samples": 24},
    ),
    "diffusion_edm": (
        "conditional_edm_diffusion",
        {"hidden": 64, "layers": 2, "sigma_min": 0.01, "sigma_max": 4.0,
         "sigma_data": 0.5, "sample_steps": 18, "rho": 7.0},
    ),
    "diffusion_anchored": (
        "gaussian_anchored_edm",
        {"hidden": 64, "layers": 2, "sigma_min": 0.01, "sigma_max": 4.0,
         "sample_steps": 18, "rho": 7.0, "base_nll_weight": 0.5},
    ),
    "flow_matching": (
        "conditional_flow_matching", {"hidden": 64, "layers": 2, "sample_steps": 18}
    ),
    "flow_matching_gaussian_source": (
        "gaussian_source_flow_matching",
        {"hidden": 64, "layers": 2, "sample_steps": 18, "base_nll_weight": 0.25},
    ),
}


def observe(latent: np.ndarray, *, channels: int, view: MeasurementView, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    path = np.asarray(latent, dtype=float).reshape(len(latent), -1, channels)
    steps = path.shape[1]
    decays = np.linspace(0.55, 0.82, channels)
    calcium = np.empty_like(path)
    calcium[:, 0] = path[:, 0]
    for index in range(1, steps):
        calcium[:, index] = decays * calcium[:, index - 1] + (1.0 - decays) * path[:, index]
    measured = np.tanh(view.gain * calcium)
    measured *= (1.0 - view.bleach * np.linspace(0.0, 1.0, steps))[None, :, None]
    loading = np.linspace(-1.0, 1.0, channels)
    waveform = np.sin(np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False))
    amplitude = rng.normal(size=(len(path), 1, 1))
    measured += view.artifact * amplitude * waveform[None, :, None] * loading[None, None, :]
    measured += view.photon * (0.25 + np.abs(measured)) * rng.normal(size=measured.shape)
    return measured.reshape(len(latent), -1).astype("float32")


def features(value: np.ndarray, channels: int) -> np.ndarray:
    path = np.asarray(value).reshape(*np.asarray(value).shape[:-1], -1, channels)
    integral = path.mean(axis=(-2, -1))
    high_frequency = np.square(np.diff(path, axis=-2)).mean(axis=(-2, -1))
    coactivation = path.mean(axis=-1).max(axis=-1)
    exceedance = (np.abs(path).max(axis=(-2, -1)) > 0.8).astype(float)
    return np.stack([integral, high_frequency, coactivation, exceedance], axis=-1)


def model_samples(model, scaler: TrainScaler, history: np.ndarray, seed: int) -> np.ndarray:
    device = next(model.parameters()).device
    h = torch.as_tensor(scaler.transform_history(history), dtype=torch.float32, device=device)
    with torch.no_grad():
        samples = model.sample(h, N_SAMPLES, seed=seed).detach().cpu().numpy()
    return np.asarray(scaler.inverse_response(samples))


def run() -> pd.DataFrame:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for replicate, system_seed in enumerate(SYSTEM_SEEDS):
        dgp = make_dgp(
            "g8", seed=system_seed, q=17, dy=32, n_channels=4, motif_rank=4,
            motif_scale=0.45, noise_sd=0.08, correlated_noise_scale=0.06,
            shape_sensitivity=1.2,
        )
        total = N_TRAIN + N_VALIDATION
        history = dgp.sample_history(total, seed=10_000 + system_seed).numpy()
        latent = dgp.sample_response(
            torch.as_tensor(history), 1, seed=20_000 + system_seed
        )[:, 0].numpy()
        observed = observe(latent, channels=4, view=ID_VIEW, seed=30_000 + system_seed)
        train_h, val_h = history[:N_TRAIN], history[N_TRAIN:]
        train_y, val_y = observed[:N_TRAIN], observed[N_TRAIN:]
        scaler = TrainScaler.fit(train_h, train_y)

        test_h = dgp.sample_history(N_TEST, seed=40_000 + system_seed).numpy()
        direction = np.zeros(dgp.q)
        direction[-1] = 1.0

        for model_index, (model_name, (kind, params)) in enumerate(MODELS.items()):
            seed = 50_000 + 100 * replicate + model_index
            torch.manual_seed(seed)
            model = build_model(kind, q=dgp.q, dy=dgp.dy, params=params)
            trace = fit_model(
                model,
                scaler.transform_history(train_h).astype("float32"),
                scaler.transform_response(train_y).astype("float32"),
                scaler.transform_history(val_h).astype("float32"),
                scaler.transform_response(val_y).astype("float32"),
                seed=seed, device="cpu", learning_rate=1e-3, weight_decay=1e-4,
                batch_size=256, max_epochs=60, patience=10, gradient_clip=1.0,
            )
            checkpoint = OUTPUT / f"seed{system_seed}_{model_name}.pt"
            torch.save(model.state_dict(), checkpoint)

            for view_index, view in enumerate((ID_VIEW, SHIFT_VIEW)):
                target_latent = dgp.sample_response(
                    torch.as_tensor(test_h), 1, seed=60_000 + system_seed
                )[:, 0].numpy()
                target = observe(
                    target_latent, channels=4, view=view,
                    seed=70_000 + system_seed + 1_000 * view_index,
                )
                samples = model_samples(model, scaler, test_h, seed=80_000 + system_seed)
                law_score = energy_score_fair(target, samples)

                plus_h = test_h + DELTA * direction
                minus_h = test_h - DELTA * direction
                plus_model = model_samples(model, scaler, plus_h, seed=90_000 + system_seed)
                minus_model = model_samples(model, scaler, minus_h, seed=90_000 + system_seed)
                estimate = (
                    features(plus_model, 4).mean(axis=1)
                    - features(minus_model, 4).mean(axis=1)
                ) / (2.0 * DELTA)

                plus_latent = dgp.sample_response(
                    torch.as_tensor(plus_h), 256, seed=100_000 + system_seed
                ).numpy().reshape(-1, dgp.dy)
                minus_latent = dgp.sample_response(
                    torch.as_tensor(minus_h), 256, seed=100_000 + system_seed
                ).numpy().reshape(-1, dgp.dy)
                plus_oracle = observe(
                    plus_latent, channels=4, view=view,
                    seed=110_000 + system_seed + 1_000 * view_index,
                ).reshape(N_TEST, 256, dgp.dy)
                minus_oracle = observe(
                    minus_latent, channels=4, view=view,
                    seed=110_000 + system_seed + 1_000 * view_index,
                ).reshape(N_TEST, 256, dgp.dy)
                truth = (
                    features(plus_oracle, 4).mean(axis=1)
                    - features(minus_oracle, 4).mean(axis=1)
                ) / (2.0 * DELTA)
                contrast_nrmse = float(
                    np.sqrt(np.mean((estimate - truth) ** 2) / max(np.mean(truth**2), 1e-12))
                )
                rows.append({
                    "system_seed": system_seed,
                    "model_name": model_name,
                    "model_kind": kind,
                    "view": view.name,
                    "energy_score_fair": law_score,
                    "typed_contrast_nrmse": contrast_nrmse,
                    "best_epoch": trace.best_epoch,
                    "stopped_epoch": trace.stopped_epoch,
                    "fit_seconds": trace.wall_seconds,
                    "parameter_count": trace.parameter_count,
                })
            pd.DataFrame(rows).to_csv(OUTPUT / "metrics.partial.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "metrics.csv", index=False)
    manifest = {
        "status": "complete",
        "system_seeds": SYSTEM_SEEDS,
        "n_train": N_TRAIN,
        "n_validation": N_VALIDATION,
        "n_test": N_TEST,
        "n_samples": N_SAMPLES,
        "delta": DELTA,
        "models": MODELS,
        "views": [ID_VIEW.__dict__, SHIFT_VIEW.__dict__],
        "finished_unix": time.time(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=list) + "\n")
    return frame


if __name__ == "__main__":
    print(run().groupby(["view", "model_name"])[["energy_score_fair", "typed_contrast_nrmse"]].median())
