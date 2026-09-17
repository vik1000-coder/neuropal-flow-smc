from __future__ import annotations

import numpy as np


def energy_score_rows(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Unbiased-pair Monte Carlo energy score for each conditioning window."""
    first = np.linalg.norm(samples - target[:, None, :], axis=2).mean(axis=1)
    k = samples.shape[1] - samples.shape[1] % 2
    paired = np.linalg.norm(samples[:, :k:2] - samples[:, 1:k:2], axis=2).mean(axis=1)
    return first - 0.5 * paired


def variogram_score_rows(
    samples: np.ndarray,
    target: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    power: float = 0.5,
) -> np.ndarray:
    observed = np.abs(target[:, pair_i] - target[:, pair_j]) ** power
    simulated = (
        np.abs(samples[:, :, pair_i] - samples[:, :, pair_j]) ** power
    ).mean(axis=1)
    return np.square(observed - simulated).mean(axis=1)


def metric_rows(samples: np.ndarray, target: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    n_dim = target.shape[1]
    rng = np.random.default_rng(seed)
    all_i, all_j = np.triu_indices(n_dim, k=1)
    if len(all_i) > 512:
        idx = rng.choice(len(all_i), size=512, replace=False)
        all_i, all_j = all_i[idx], all_j[idx]
    mean = samples.mean(axis=1)
    lo = np.quantile(samples, 0.05, axis=1)
    hi = np.quantile(samples, 0.95, axis=1)
    return {
        "energy": energy_score_rows(samples, target),
        "variogram": variogram_score_rows(samples, target, all_i, all_j),
        "rmse": np.sqrt(np.square(mean - target).mean(axis=1)),
        "mae": np.abs(mean - target).mean(axis=1),
        "coverage90": ((target >= lo) & (target <= hi)).mean(axis=1),
        "sharpness90": (hi - lo).mean(axis=1),
    }


def summarize_metrics(
    rows: dict[str, np.ndarray],
    strata: np.ndarray,
    log_prob: np.ndarray | None,
    n_dim: int,
    population_strata: np.ndarray | None = None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    labels = sorted(set(strata.tolist()))
    population = strata if population_strata is None else population_strata
    population_weights = {
        label: float(np.mean(population == label)) for label in labels
    }
    weight_total = sum(population_weights.values())
    for name, values in rows.items():
        per_label_values: dict[str, float] = {}
        per_label = []
        for label in labels:
            mask = strata == label
            value = float(np.mean(values[mask]))
            result[f"{name}__{label}"] = value
            per_label_values[label] = value
            per_label.append(value)
        result[name] = float(
            sum(per_label_values[label] * population_weights[label] for label in labels)
            / weight_total
        )
        result[f"{name}__stim_balanced"] = float(np.mean(per_label))
    if log_prob is not None:
        nll_by_label = {
            label: float(-np.mean(log_prob[strata == label]) / n_dim) for label in labels
        }
        for label, value in nll_by_label.items():
            result[f"nll_per_neuron__{label}"] = value
        result["nll_per_neuron"] = float(
            sum(nll_by_label[label] * population_weights[label] for label in labels)
            / weight_total
        )
        result["nll_per_neuron__stim_balanced"] = float(
            np.mean(list(nll_by_label.values()))
        )
    for label in labels:
        result[f"n__{label}"] = int(np.sum(strata == label))
        result[f"population_n__{label}"] = int(np.sum(population == label))
    result["n_eval"] = int(len(strata))
    return result
