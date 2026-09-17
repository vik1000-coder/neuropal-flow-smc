"""Permutation-invariant proper sample scores and conditional copula ablation."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def shuffle_coordinates(samples: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute sample indices for each fixed (history, neuron).

    No history, target, animal, value, or univariate empirical distribution is
    changed. At finite N a shuffled bank approximates the product of marginals;
    it does not guarantee exactly zero sample covariance.
    """
    samples = np.asarray(samples)
    if samples.ndim != 3 or samples.shape[1] < 2:
        raise ValueError("expected [history, sample>=2, neuron]")
    rng = np.random.default_rng(seed)
    index = np.broadcast_to(np.arange(samples.shape[1])[None, :, None], samples.shape)
    index = rng.permuted(index, axis=1)
    return np.take_along_axis(samples, index, axis=1)


def score_samples(samples: np.ndarray, target: np.ndarray,
                  tail_threshold: np.ndarray | None = None,
                  variogram_offset: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Every metric returns one value per history; calculations use float64.

    Unlike the legacy approximate energy implementation, all unordered draw
    pairs are used. Thus a common row permutation is exactly score-invariant.
    Variogram and Brier remove the usual iid finite-ensemble variance bias.
    """
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2 or y.shape != (x.shape[0], x.shape[2]):
        raise ValueError("sample and target axes do not match")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite samples or observations")
    offset = np.zeros_like(y) if variogram_offset is None else np.asarray(variogram_offset, dtype=float)
    if offset.shape != y.shape or not np.isfinite(offset).all():
        raise ValueError("variogram offset must match target shape")
    n = x.shape[1]
    ordered = np.sort(x, axis=1)
    coefficient = (2 * np.arange(n) - n + 1)[None, :, None]
    first = np.abs(x - y[:, None]).mean(axis=1)
    half_pair_distance = (ordered * coefficient).sum(axis=1) / (n * (n - 1))
    mean = x.mean(axis=1)
    quantiles = np.quantile(x, [0.05, 0.95], axis=1)
    energy = np.linalg.norm(x - y[:, None], axis=-1).mean(axis=1)
    energy -= 0.5 * np.asarray([pdist(bank).mean() for bank in x])
    ii, jj = np.triu_indices(x.shape[2], 1)
    if len(ii):
        # Bound temporary memory even for large particle sensitivity panels.
        vario = np.zeros(len(x))
        naive = np.zeros(len(x))
        innovation_vario = np.zeros(len(x))
        for start in range(0, len(ii), 128):
            a, b = ii[start:start + 128], jj[start:start + 128]
            raw_difference = x[:, :, a] - x[:, :, b]
            innovation = np.sqrt(np.abs(raw_difference))
            observed_innovation = np.sqrt(np.abs(y[:, a] - y[:, b]))
            innovation_vario += ((innovation.mean(axis=1) - observed_innovation) ** 2
                                 - innovation.var(axis=1, ddof=1) / n).sum(axis=1)
            shift = offset[:, a] - offset[:, b]
            difference = np.sqrt(np.abs(raw_difference + shift[:, None]))
            observed = np.sqrt(np.abs(y[:, a] - y[:, b] + shift))
            squared = (difference.mean(axis=1) - observed) ** 2
            naive += squared.sum(axis=1)
            vario += (squared - difference.var(axis=1, ddof=1) / n).sum(axis=1)
        vario, naive = vario / len(ii), naive / len(ii)
        innovation_vario /= len(ii)
    else:
        vario, naive = np.zeros(len(x)), np.zeros(len(x))
        innovation_vario = np.zeros(len(x))
    result = {
        "energy": energy,
        "variogram": vario,
        "variogram_naive": naive,
        "innovation_variogram": innovation_vario,
        "crps": (first - half_pair_distance).mean(axis=1),
        "rmse": np.sqrt(((mean - y) ** 2).mean(axis=1)),
        "coverage90": ((y >= quantiles[0]) & (y <= quantiles[1])).mean(axis=1),
        "sharpness90": (quantiles[1] - quantiles[0]).mean(axis=1),
    }
    if tail_threshold is not None:
        threshold = np.asarray(tail_threshold, dtype=float)
        if threshold.shape != (x.shape[2],) or np.any(threshold <= 0):
            raise ValueError("tail thresholds must be positive per-neuron values")
        probability = (np.abs(x) > threshold).mean(axis=1)
        event = np.abs(y) > threshold
        result["tail_brier"] = (((probability - event) ** 2)
                                - probability * (1 - probability) / (n - 1)).mean(axis=1)
        result["tail_probability"] = probability.mean(axis=1)
        result["tail_frequency"] = event.mean(axis=1)
    return result


def marginal_invariance(original: dict, shuffled: dict, tolerance: float = 1e-10) -> float:
    keys = ["crps", "rmse", "coverage90", "sharpness90", "tail_brier", "tail_probability"]
    error = max(float(np.max(np.abs(original[key] - shuffled[key])))
                for key in keys if key in original)
    if error > tolerance:
        raise AssertionError(f"conditional marginal invariance failed: {error}")
    return error
