from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from conditional_neural_benchmark.models import build_encoded_model
from history_tangent_benchmark.models import resolve_device


def load_checkpoint(path: str | Path, device: str = "auto"):
    """Load a benchmark/deployment checkpoint and reconstruct its model."""
    target_device = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    input_channels = int(checkpoint.get("input_channels", len(checkpoint["neurons"]) + 1))
    model = build_encoded_model(
        head_name=config["head"],
        encoder_name=config["encoder"],
        lag=int(checkpoint["lag"]),
        channels=input_channels,
        dy=len(checkpoint["neurons"]),
        width=int(config["width"]),
        dropout=float(config["dropout"]),
        head_params=dict(config["head_params"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device).eval()
    return model, checkpoint, target_device


def sample_next(
    checkpoint_path: str | Path,
    neural_history: np.ndarray,
    stimulus_history: np.ndarray,
    *,
    n_samples: int = 128,
    seed: int = 0,
    device: str = "auto",
) -> np.ndarray:
    """Sample next-frame activity in the input coordinate system.

    ``neural_history`` must be ``[batch, lag, neurons]`` in the coordinate
    system produced by ``sid_elegans.combined_data.load_combined``. The
    stimulus array must be ``[batch, lag]`` for a one-channel checkpoint or
    ``[batch, lag, stimulus_channels]`` and use the exact same time axis.
    """
    model, checkpoint, target_device = load_checkpoint(checkpoint_path, device)
    history = np.asarray(neural_history, dtype=np.float32)
    stimulus = np.asarray(stimulus_history, dtype=np.float32)
    lag = int(checkpoint["lag"])
    n_neurons = len(checkpoint["neurons"])
    if history.ndim == 2:
        history = history[None]
    if stimulus.ndim == 1:
        stimulus = stimulus[None]
    if history.shape[1:] != (lag, n_neurons):
        raise ValueError(f"neural_history must end in {(lag, n_neurons)}")
    stimulus_channels = int(checkpoint.get("stimulus_channels", 1))
    if stimulus.ndim == 2:
        stimulus = stimulus[..., None]
    if stimulus.shape != (*history.shape[:2], stimulus_channels):
        raise ValueError(
            "stimulus_history must match batch, lag, and checkpoint stimulus channels"
        )
    mean = np.asarray(checkpoint["scaler"]["mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler"]["scale"], dtype=np.float32)
    standardized = (history - mean) / scale
    context = np.concatenate([standardized, stimulus], axis=2).reshape(
        len(history), -1
    )
    tensor = torch.as_tensor(context, dtype=torch.float32, device=target_device)
    with torch.no_grad():
        draw = model.sample(tensor, n_samples, seed=seed).detach().cpu().numpy()
    if bool(checkpoint["model_config"].get("residual_target", False)):
        draw = draw + standardized[:, -1, None, :]
    return draw * scale[None, None, :] + mean[None, None, :]
