from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


Array = np.ndarray


def effective_sample_size(weights: Array) -> float:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise ValueError("weights must be a nonempty nonnegative vector")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have positive finite mass")
    normalized = values / total
    return float(1.0 / np.square(normalized).sum())


def normalize_log_weights(log_weights: Array) -> tuple[Array, float]:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).any():
        raise ValueError("at least one log weight must be finite")
    maximum = float(np.max(values))
    raw = np.exp(values - maximum)
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("log-weight normalization failed")
    return raw / total, maximum + np.log(total)


def systematic_resample(weights: Array, rng: np.random.Generator) -> Array:
    values = np.asarray(weights, dtype=np.float64)
    values = values / values.sum()
    n = len(values)
    positions = (rng.random() + np.arange(n)) / n
    indices = np.searchsorted(np.cumsum(values), positions, side="right")
    return np.minimum(indices, n - 1).astype(np.int64)


def _entropy(weights: Array) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = values / values.sum()
    return float(-np.sum(values * np.log(np.maximum(values, 1e-300))))


def _largest_beta(
    log_weights: Array,
    potential: Array,
    beta: float,
    target_ess: float,
) -> float:
    if beta >= 1.0:
        return 1.0

    def ess_at(candidate: float) -> float:
        weights, _ = normalize_log_weights(
            log_weights + (candidate - beta) * potential
        )
        return effective_sample_size(weights)

    if ess_at(1.0) >= target_ess:
        return 1.0
    lo, hi = beta, 1.0
    for _ in range(60):
        middle = 0.5 * (lo + hi)
        if ess_at(middle) >= target_ess:
            lo = middle
        else:
            hi = middle
    return max(beta + 1e-8, lo)


@dataclass(frozen=True)
class BridgeResult:
    estimate: float
    mcse: float
    event_probability: float
    model_evaluations: int
    stages: pd.DataFrame
    minimum_ess_fraction: float
    final_ess_fraction: float
    final_max_weight: float
    final_entropy: float
    resampling_events: int
    unique_ancestors: int
    genealogical_collapse: float
    rejuvenation_acceptance: float
    constraint_mean: float
    finite: bool


def progressive_bridge_smc(
    sample_fn: Callable[[int, int], Array],
    potential_fn: Callable[[Array], Array],
    statistic_fn: Callable[[Array], Array],
    *,
    n_particles: int,
    seed: int,
    tempering_ess_fraction: float = 0.70,
    resample_ess_fraction: float = 0.70,
    rejuvenation_steps: int = 2,
    max_stages: int = 64,
) -> BridgeResult:
    """Sampler-only static bridge with independence-MH rejuvenation.

    The target is ``p(y|h) exp(Phi_q(y))``. Independence proposals come from
    ``p(y|h)``, so the unknown model density cancels from the MH ratio. This is
    the analytic/synthetic control estimator, distinct from the repository's
    repaired-path progressive SMC NeuroPAL primary estimator.
    """
    if n_particles < 2:
        raise ValueError("at least two particles are required")
    if not 0 < tempering_ess_fraction <= 1:
        raise ValueError("invalid tempering ESS fraction")
    if not 0 < resample_ess_fraction <= 1:
        raise ValueError("invalid resampling ESS fraction")
    if rejuvenation_steps < 0 or max_stages < 1:
        raise ValueError("invalid rejuvenation or stage count")

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    particles = np.asarray(sample_fn(n_particles, seed), dtype=np.float64)
    if particles.ndim != 2 or len(particles) != n_particles:
        raise ValueError("sample_fn must return [n_particles, dimension]")
    potential = np.asarray(potential_fn(particles), dtype=np.float64)
    if potential.shape != (n_particles,) or not np.isfinite(potential).all():
        raise ValueError("potential must return one finite value per particle")
    ancestors = np.arange(n_particles, dtype=np.int64)
    log_weights = np.full(n_particles, -np.log(n_particles), dtype=np.float64)
    beta = 0.0
    log_normalizer = 0.0
    model_evaluations = n_particles
    resampling_events = 0
    accepted = 0
    proposed = 0
    records: list[dict[str, float | int]] = []

    for stage in range(max_stages):
        if beta >= 1.0 - 1e-10:
            break
        chosen = _largest_beta(
            log_weights,
            potential,
            beta,
            tempering_ess_fraction * n_particles,
        )
        chosen = min(1.0, chosen)
        updated = log_weights + (chosen - beta) * potential
        weights, increment = normalize_log_weights(updated)
        log_normalizer += increment
        ess = effective_sample_size(weights)
        do_resample = chosen < 1.0 - 1e-10 or ess < resample_ess_fraction * n_particles
        stage_acceptance = np.nan
        if do_resample:
            indices = systematic_resample(weights, rng)
            particles = particles[indices]
            potential = potential[indices]
            ancestors = ancestors[indices]
            weights = np.full(n_particles, 1.0 / n_particles)
            log_weights = np.full(n_particles, -np.log(n_particles))
            resampling_events += 1
            stage_accepted = 0
            stage_proposed = 0
            for rejuvenation in range(rejuvenation_steps):
                proposal_seed = int(rng.integers(0, 2**31 - 1))
                proposals = np.asarray(
                    sample_fn(n_particles, proposal_seed), dtype=np.float64
                )
                proposal_potential = np.asarray(
                    potential_fn(proposals), dtype=np.float64
                )
                log_acceptance = chosen * (proposal_potential - potential)
                accept = np.log(rng.random(n_particles)) < np.minimum(0.0, log_acceptance)
                particles[accept] = proposals[accept]
                potential[accept] = proposal_potential[accept]
                stage_accepted += int(accept.sum())
                stage_proposed += n_particles
                model_evaluations += n_particles
            accepted += stage_accepted
            proposed += stage_proposed
            if stage_proposed:
                stage_acceptance = stage_accepted / stage_proposed
        else:
            log_weights = np.log(np.maximum(weights, 1e-300))
        records.append(
            {
                "stage": stage,
                "beta_start": beta,
                "beta_end": chosen,
                "adaptive_increment": chosen - beta,
                "ess": ess,
                "ess_fraction": ess / n_particles,
                "incremental_weight_variance": float(
                    np.var(np.exp((chosen - beta) * potential))
                ),
                "max_normalized_weight": float(weights.max()),
                "weight_entropy": _entropy(weights),
                "resampled": int(do_resample),
                "unique_ancestors": int(len(np.unique(ancestors))),
                "rejuvenation_acceptance": stage_acceptance,
                "model_evaluations": model_evaluations,
                "wall_seconds": time.perf_counter() - started,
            }
        )
        beta = chosen
    else:
        raise RuntimeError("adaptive bridge exceeded maximum stages")

    weights, _ = normalize_log_weights(log_weights)
    statistic = np.asarray(statistic_fn(particles), dtype=np.float64)
    if statistic.shape != (n_particles,) or not np.isfinite(statistic).all():
        raise ValueError("statistic must return one finite value per particle")
    estimate = float(np.sum(weights * statistic))
    ess = effective_sample_size(weights)
    variance = float(np.sum(weights * np.square(statistic - estimate)))
    event_probability = float(np.exp(log_normalizer))
    frame = pd.DataFrame(records)
    return BridgeResult(
        estimate=estimate,
        mcse=float(np.sqrt(max(0.0, variance) / max(1.0, ess))),
        event_probability=event_probability,
        model_evaluations=model_evaluations,
        stages=frame,
        minimum_ess_fraction=float(frame.ess_fraction.min()),
        final_ess_fraction=ess / n_particles,
        final_max_weight=float(weights.max()),
        final_entropy=_entropy(weights),
        resampling_events=resampling_events,
        unique_ancestors=int(len(np.unique(ancestors))),
        genealogical_collapse=1.0 - len(np.unique(ancestors)) / n_particles,
        rejuvenation_acceptance=(accepted / proposed if proposed else np.nan),
        constraint_mean=float(np.sum(weights * potential)),
        finite=bool(np.isfinite(estimate) and np.isfinite(variance)),
    )


def direct_importance_sampling(
    sample_fn: Callable[[int, int], Array],
    potential_fn: Callable[[Array], Array],
    statistic_fn: Callable[[Array], Array],
    *,
    n_particles: int,
    seed: int,
) -> BridgeResult:
    started = time.perf_counter()
    particles = np.asarray(sample_fn(n_particles, seed), dtype=np.float64)
    potential = np.asarray(potential_fn(particles), dtype=np.float64)
    weights, log_total = normalize_log_weights(potential)
    statistic = np.asarray(statistic_fn(particles), dtype=np.float64)
    estimate = float(np.sum(weights * statistic))
    ess = effective_sample_size(weights)
    variance = float(np.sum(weights * np.square(statistic - estimate)))
    frame = pd.DataFrame(
        [
            {
                "stage": 0,
                "beta_start": 0.0,
                "beta_end": 1.0,
                "adaptive_increment": 1.0,
                "ess": ess,
                "ess_fraction": ess / n_particles,
                "incremental_weight_variance": float(np.var(np.exp(potential))),
                "max_normalized_weight": float(weights.max()),
                "weight_entropy": _entropy(weights),
                "resampled": 0,
                "unique_ancestors": n_particles,
                "rejuvenation_acceptance": np.nan,
                "model_evaluations": n_particles,
                "wall_seconds": time.perf_counter() - started,
            }
        ]
    )
    return BridgeResult(
        estimate=estimate,
        mcse=float(np.sqrt(max(0.0, variance) / max(1.0, ess))),
        event_probability=float(np.exp(log_total) / n_particles),
        model_evaluations=n_particles,
        stages=frame,
        minimum_ess_fraction=ess / n_particles,
        final_ess_fraction=ess / n_particles,
        final_max_weight=float(weights.max()),
        final_entropy=_entropy(weights),
        resampling_events=0,
        unique_ancestors=n_particles,
        genealogical_collapse=0.0,
        rejuvenation_acceptance=np.nan,
        constraint_mean=float(np.sum(weights * potential)),
        finite=bool(np.isfinite(estimate) and np.isfinite(variance)),
    )


@dataclass(frozen=True)
class QueryPair:
    factual: Array
    coherent: Array
    incoherent: Array
    stimulus: Array
    source: int
    target_value: float
    coherent_norm: float
    incoherent_norm: float


def construct_matched_query_pair(
    history: Array,
    stimulus_history: Array,
    *,
    source: int,
    target_value: float,
    smooth_frames: int = 8,
) -> QueryPair:
    factual = np.asarray(history, dtype=np.float64)
    stimulus = np.asarray(stimulus_history).copy()
    if factual.ndim != 2 or not 0 <= source < factual.shape[1]:
        raise ValueError("history/source shape mismatch")
    if smooth_frames < 3 or smooth_frames > len(factual):
        raise ValueError("smooth_frames must lie in [3, history length]")
    original = factual.copy()
    delta = float(target_value - factual[-1, source])
    phase = np.linspace(0.0, np.pi, smooth_frames)
    weights = 0.5 * (1.0 - np.cos(phase))
    coherent = factual.copy()
    coherent[-smooth_frames:, source] += delta * weights

    incoherent = factual.copy()
    incoherent[-1, source] += delta
    smooth_energy = float(np.square(delta * weights).sum())
    remaining = max(0.0, smooth_energy - delta * delta)
    amplitude = np.sqrt(remaining / 2.0)
    incoherent[-3, source] += amplitude
    incoherent[-2, source] -= amplitude

    if not np.array_equal(factual, original):
        raise RuntimeError("query construction mutated the factual array")
    if coherent[-1, source] != incoherent[-1, source]:
        raise RuntimeError("terminal amplitudes do not match")
    coherent_norm = float(np.linalg.norm(coherent - factual))
    incoherent_norm = float(np.linalg.norm(incoherent - factual))
    if not np.isclose(coherent_norm, incoherent_norm, rtol=1e-10, atol=1e-10):
        raise RuntimeError("matched query norms differ")
    return QueryPair(
        factual=factual.copy(),
        coherent=coherent,
        incoherent=incoherent,
        stimulus=stimulus,
        source=int(source),
        target_value=float(target_value),
        coherent_norm=coherent_norm,
        incoherent_norm=incoherent_norm,
    )


def _stimulus_features(stimulus: Array) -> Array:
    values = np.asarray(stimulus, dtype=np.float64)
    if values.ndim == 2:
        values = values[..., None]
    binary = values.max(axis=2)
    changes = np.diff(binary, axis=1)
    onset = (changes > 0.5).sum(axis=1)
    offset = (changes < -0.5).sum(axis=1)
    last_on = np.argmax(np.flip(changes > 0.5, axis=1), axis=1)
    last_off = np.argmax(np.flip(changes < -0.5, axis=1), axis=1)
    has_on = np.any(changes > 0.5, axis=1)
    has_off = np.any(changes < -0.5, axis=1)
    last_on = np.where(has_on, last_on, changes.shape[1] + 1)
    last_off = np.where(has_off, last_off, changes.shape[1] + 1)
    return np.column_stack(
        [
            binary[:, -1],
            binary.mean(axis=1),
            binary[:, -min(4, binary.shape[1]) :].mean(axis=1),
            binary[:, -min(16, binary.shape[1]) :].mean(axis=1),
            onset,
            offset,
            last_on,
            last_off,
        ]
    )


def handcrafted_history_features(histories: Array, stimulus: Array) -> Array:
    values = np.asarray(histories, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 3:
        raise ValueError("histories must have shape [rows, time>=3, neurons]")
    features = [values[:, -1]]
    for gap in (1, 2, 4, 8, 16):
        use = min(gap, values.shape[1] - 1)
        features.append((values[:, -1] - values[:, -1 - use]) / use)
    for width in (4, 16, values.shape[1]):
        use = min(width, values.shape[1])
        section = values[:, -use:]
        features.extend([section.mean(axis=1), section.std(axis=1)])
    features.append(values[:, -1] - 2 * values[:, -2] + values[:, -3])
    population = np.column_stack(
        [
            values[:, -1].mean(axis=1),
            values[:, -1].std(axis=1),
            values[:, -4:].mean(axis=(1, 2)),
            values[:, -4:].std(axis=(1, 2)),
        ]
    )
    features.extend([population, _stimulus_features(stimulus)])
    result = np.concatenate(features, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("history representation contains nonfinite values")
    return result


def _corrupt_histories(histories: Array, rng: np.random.Generator) -> Array:
    result = np.asarray(histories, dtype=np.float64).copy()
    rows, length, neurons = result.shape
    scales = np.std(result.reshape(-1, neurons), axis=0) + 1e-6
    for row in range(rows):
        source = int(rng.integers(0, neurons))
        mode = row % 3
        if mode == 0:
            result[row, -1, source] += rng.choice((-1.0, 1.0)) * 3.0 * scales[source]
        elif mode == 1:
            result[row, :, source] = result[row, rng.permutation(length), source]
        else:
            donor = int(rng.integers(0, rows))
            keep = result[row, :, source].copy()
            result[row] = result[donor]
            result[row, :, source] = keep
    return result


class HistorySupportModel:
    """Training-only, model-independent support profile for neural histories."""

    def __init__(self, *, k: int = 10, pca_cap: int = 64, seed: int = 20260901):
        if k < 1 or pca_cap < 1:
            raise ValueError("k and PCA cap must be positive")
        self.k = int(k)
        self.pca_cap = int(pca_cap)
        self.seed = int(seed)

    def fit(
        self,
        train_histories: Array,
        train_stimulus: Array,
        calibration_histories: Array,
        calibration_stimulus: Array,
        *,
        max_train_rows: int = 6000,
    ) -> "HistorySupportModel":
        train = np.asarray(train_histories, dtype=np.float64)
        calibration = np.asarray(calibration_histories, dtype=np.float64)
        if train.ndim != 3 or calibration.shape[1:] != train.shape[1:]:
            raise ValueError("train/calibration history shapes disagree")
        rng = np.random.default_rng(self.seed)
        if len(train) > max_train_rows:
            index = np.sort(rng.choice(len(train), max_train_rows, replace=False))
            train = train[index]
            train_stimulus = np.asarray(train_stimulus)[index]
        self.train_histories_ = train.copy()
        self.train_stimulus_ = np.asarray(train_stimulus).copy()
        flat = train.reshape(len(train), -1)
        self.flat_scaler_ = StandardScaler().fit(flat)
        components = min(self.pca_cap, len(train) - 1, flat.shape[1])
        self.pca_ = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=self.seed,
        ).fit(self.flat_scaler_.transform(flat))
        handcrafted = handcrafted_history_features(train, self.train_stimulus_)
        self.handcrafted_scaler_ = StandardScaler().fit(handcrafted)
        representation = self._representation(train, self.train_stimulus_)
        self.representation_scaler_ = StandardScaler().fit(representation)
        scaled = self.representation_scaler_.transform(representation)
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(scaled))).fit(scaled)

        calibration_repr = self.representation_scaler_.transform(
            self._representation(calibration, calibration_stimulus)
        )
        self.calibration_distance_ = self.nn_.kneighbors(
            calibration_repr, return_distance=True
        )[0].mean(axis=1)

        corrupted = _corrupt_histories(train, rng)
        real_features = self._representation(train, self.train_stimulus_)
        corrupted_features = self._representation(corrupted, self.train_stimulus_)
        classifier_x = np.vstack([real_features, corrupted_features])
        classifier_y = np.r_[np.zeros(len(train)), np.ones(len(train))]
        self.corruption_scaler_ = StandardScaler().fit(classifier_x)
        self.corruption_classifier_ = LogisticRegression(
            max_iter=500, random_state=self.seed
        ).fit(self.corruption_scaler_.transform(classifier_x), classifier_y)

        terminal = train[:, -1]
        derivative = np.abs(np.diff(train, axis=1)).max(axis=(1, 2))
        curvature = np.abs(np.diff(train, n=2, axis=1)).max(axis=(1, 2))
        self.amplitude_reference_ = np.abs(terminal).reshape(-1)
        self.derivative_reference_ = derivative
        self.curvature_reference_ = curvature
        self.coordinate_low_ = np.quantile(train, 0.005, axis=(0, 1))
        self.coordinate_high_ = np.quantile(train, 0.995, axis=(0, 1))
        population = np.column_stack([terminal.mean(1), terminal.std(1)])
        self.population_center_ = population.mean(0)
        self.population_scale_ = population.std(0) + 1e-6
        return self

    def _representation(self, histories: Array, stimulus: Array) -> Array:
        values = np.asarray(histories, dtype=np.float64)
        handcrafted = self.handcrafted_scaler_.transform(
            handcrafted_history_features(values, stimulus)
        )
        flat = self.flat_scaler_.transform(values.reshape(len(values), -1))
        return np.concatenate([handcrafted, self.pca_.transform(flat)], axis=1)

    @staticmethod
    def _upper_rank(reference: Array, values: Array) -> Array:
        ref = np.sort(np.asarray(reference, dtype=np.float64))
        return np.searchsorted(ref, values, side="right") / max(1, len(ref))

    def score(
        self,
        histories: Array,
        stimulus: Array,
        *,
        source: int | None = None,
    ) -> pd.DataFrame:
        values = np.asarray(histories, dtype=np.float64)
        representation = self._representation(values, stimulus)
        scaled = self.representation_scaler_.transform(representation)
        distance = self.nn_.kneighbors(scaled, return_distance=True)[0].mean(axis=1)
        support_p = (
            1
            + (self.calibration_distance_[None] >= distance[:, None]).sum(axis=1)
        ) / (1 + len(self.calibration_distance_))
        amplitude = np.abs(values[:, -1]).max(axis=1)
        derivative = np.abs(np.diff(values, axis=1)).max(axis=(1, 2))
        curvature = np.abs(np.diff(values, n=2, axis=1)).max(axis=(1, 2))
        outside = np.mean(
            (values < self.coordinate_low_[None, None])
            | (values > self.coordinate_high_[None, None]),
            axis=(1, 2),
        )
        corruption = self.corruption_classifier_.predict_proba(
            self.corruption_scaler_.transform(representation)
        )[:, 1]
        terminal = values[:, -1]
        population = np.column_stack([terminal.mean(1), terminal.std(1)])
        population_residual = np.linalg.norm(
            (population - self.population_center_) / self.population_scale_, axis=1
        )
        stim = _stimulus_features(stimulus)
        train_stim = _stimulus_features(self.train_stimulus_)
        compatibility = np.min(
            np.mean(np.abs(stim[:, None] - train_stim[None]), axis=2), axis=1
        )
        frame = pd.DataFrame(
            {
                "nearest_history_distance": distance,
                "history_support_value": support_p,
                "amplitude_percentile": self._upper_rank(
                    self.amplitude_reference_, amplitude
                ),
                "max_derivative_percentile": self._upper_rank(
                    self.derivative_reference_, derivative
                ),
                "curvature_percentile": self._upper_rank(
                    self.curvature_reference_, curvature
                ),
                "population_state_residual": population_residual,
                "stimulus_incompatibility": compatibility,
                "corrupted_classifier_probability": corruption,
                "fraction_outside_training_quantiles": outside,
            }
        )
        if source is not None:
            if not 0 <= source < values.shape[2]:
                raise ValueError("source index out of range")
            removed = values.copy()
            removed[:, :, source] = np.median(
                self.train_histories_[:, :, source], axis=0
            )[None]
            removed_repr = self.representation_scaler_.transform(
                self._representation(removed, stimulus)
            )
            frame["distance_source_removed"] = self.nn_.kneighbors(
                removed_repr, return_distance=True
            )[0].mean(axis=1)
        else:
            frame["distance_source_removed"] = np.nan
        return frame
