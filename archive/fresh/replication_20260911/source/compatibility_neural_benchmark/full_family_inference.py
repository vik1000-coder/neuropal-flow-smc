"""Complete-family worm-level inference for NeuroPAL lag-effect matrices.

This module replaces post-selection candidate p-values with one frozen family:
every directed off-diagonal edge and every selected lag-by-horizon cell for one
channel and context.  Worms are the only replication unit.  A single
Rademacher sign is drawn for each worm and reused across the entire tensor in
each randomization, preserving the observed dependence among edges, lags, and
horizons.

The primary edge statistic is the maximum absolute studentized mean across the
selected lag-by-horizon grid.  Its sign-flip p-value therefore accounts for
selecting an edge's strongest lag and horizon.  Benjamini-Hochberg is then
applied once across all directed off-diagonal edges.  Global max-T adjusted
p-values provide an intentionally conservative but honest cell-localization
analysis.  Saved zero-null reference bands are explicitly not presented as
confidence intervals.  A parallel centered-lag analysis tests the
edge-level null that the mean effect is flat across selected lags at every
horizon.

The sign-flip null requires independent, sign-symmetric worm-level effects
under the null.  It quantifies consistency of the frozen model-relative effect;
it does not establish causality, a physical delay, or replication in new data.
Unsigned within-state distances (notably Wasserstein-1) require a sham-distance
null and are rejected by default here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ORIENTATION = "target_row_source_column"
UNSIGNED_CHANNEL = "endpoint_wasserstein1"
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 20_260_830
DEFAULT_MONTE_CARLO_REPLICATES = 65_535
DEFAULT_MAX_EXACT_PATTERNS = 65_536


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted values, retaining NaNs in place."""
    values = np.asarray(p_values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid], kind="stable")]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result


def generate_sign_patterns(
    n_worms: int,
    *,
    mode: str = "auto",
    replicates: int = DEFAULT_MONTE_CARLO_REPLICATES,
    seed: int = DEFAULT_SEED,
    max_exact_patterns: int = DEFAULT_MAX_EXACT_PATTERNS,
) -> tuple[np.ndarray, str]:
    """Return shared worm-level sign patterns and the resolved inference mode.

    Two-sided tests are invariant to negating every sign.  Exact mode therefore
    fixes the first worm's sign to +1 and enumerates the 2**(n_worms-1) unique
    sign orbits.  Monte Carlo mode samples full sign vectors and uses the usual
    +1 p-value correction downstream.
    """
    if n_worms < 2:
        raise ValueError("at least two worms are required")
    if mode not in {"auto", "exact", "monte-carlo"}:
        raise ValueError("mode must be 'auto', 'exact', or 'monte-carlo'")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if max_exact_patterns < 1:
        raise ValueError("max_exact_patterns must be positive")

    exact_count = 2 ** (n_worms - 1)
    resolved = mode
    if mode == "auto":
        resolved = "exact" if exact_count <= max_exact_patterns else "monte-carlo"
    if resolved == "exact":
        if exact_count > max_exact_patterns:
            raise ValueError(
                f"exact enumeration needs {exact_count:,} patterns, exceeding "
                f"max_exact_patterns={max_exact_patterns:,}"
            )
        if n_worms > 63:
            raise ValueError("exact bit enumeration supports at most 63 worms")
        codes = np.arange(exact_count, dtype=np.uint64)
        signs = np.ones((exact_count, n_worms), dtype=np.int8)
        bit_positions = np.arange(n_worms - 1, dtype=np.uint64)
        bits = ((codes[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
        signs[:, 1:] = bits * 2 - 1
        return signs, resolved

    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(replicates, n_worms), dtype=np.int8)
    signs = signs * 2 - 1
    return signs, resolved


def _summary_terms(
    values: np.ndarray,
    *,
    min_worms: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    count = finite.sum(axis=0).astype(np.int32)
    filled = np.where(finite, x, 0.0)
    total = filled.sum(axis=0)
    sumsq = np.square(filled).sum(axis=0)
    mean = np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    centered_ss = np.maximum(sumsq - np.square(total) / np.maximum(count, 1), 0.0)
    variance = np.divide(
        centered_ss,
        count - 1,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 1,
    )
    se = np.sqrt(variance / np.maximum(count, 1))
    rms = np.sqrt(
        np.divide(
            sumsq,
            count,
            out=np.zeros(sumsq.shape, dtype=np.float64),
            where=count > 0,
        )
    )
    se_floor = np.maximum(rms * 1e-12, np.finfo(np.float64).tiny)
    denominator = np.maximum(se, se_floor)
    statistic = np.divide(
        mean,
        denominator,
        out=np.full(mean.shape, np.nan, dtype=np.float64),
        where=np.isfinite(denominator),
    )
    invalid = count < min_worms
    mean[invalid] = np.nan
    se[invalid] = np.nan
    statistic[invalid] = np.nan
    return filled, count, sumsq, mean, se, statistic


def studentized_statistic(
    values: np.ndarray,
    *,
    axis: int = 0,
    min_worms: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, standard error, one-sample t statistic, and valid count."""
    x = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    _, count, _, mean, se, statistic = _summary_terms(x, min_worms=min_worms)
    return mean, se, statistic, count


def signed_student_t(
    values: np.ndarray,
    signs: np.ndarray,
    *,
    min_worms: int = 3,
) -> np.ndarray:
    """Studentized null statistics using one shared sign per worm.

    ``values`` has shape ``[worm, ...]`` and ``signs`` has shape
    ``[randomization, worm]``.  The returned leading axis indexes the supplied
    randomizations.  This public helper makes the shared-sign implementation
    directly auditable in tests.
    """
    x = np.asarray(values, dtype=np.float64)
    s = np.asarray(signs, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("values must have shape [worm, ...]")
    if s.ndim != 2 or s.shape[1] != x.shape[0]:
        raise ValueError("signs must have shape [randomization, worm]")
    trailing = x.shape[1:]
    flat = x.reshape(x.shape[0], -1)
    filled, count, sumsq, _, _, _ = _summary_terms(flat, min_worms=min_worms)
    signed_sum = s @ filled
    null_t = _student_t_from_sum(signed_sum, count, sumsq, min_worms=min_worms)
    return null_t.reshape((len(s),) + trailing)


def _student_t_from_sum(
    signed_sum: np.ndarray,
    count: np.ndarray,
    sumsq: np.ndarray,
    *,
    min_worms: int,
) -> np.ndarray:
    total = np.asarray(signed_sum, dtype=np.float64)
    n = np.asarray(count, dtype=np.float64)
    mean = np.divide(
        total,
        n,
        out=np.zeros(total.shape, dtype=np.float64),
        where=n > 0,
    )
    centered_ss = np.maximum(sumsq[None, :] - n[None, :] * np.square(mean), 0.0)
    variance = np.divide(
        centered_ss,
        n[None, :] - 1.0,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=n[None, :] > 1,
    )
    se = np.sqrt(variance / np.maximum(n[None, :], 1.0))
    rms = np.sqrt(
        np.divide(
            sumsq,
            n,
            out=np.zeros(sumsq.shape, dtype=np.float64),
            where=n > 0,
        )
    )
    se_floor = np.maximum(rms * 1e-12, np.finfo(np.float64).tiny)
    out = mean / np.maximum(se, se_floor[None, :])
    out[:, np.asarray(count) < min_worms] = np.nan
    return out


def _randomization_p(exceed: np.ndarray, n_patterns: int, mode: str) -> np.ndarray:
    count = np.asarray(exceed, dtype=np.float64)
    if mode == "exact":
        # The exact orbit contains the observed all-positive assignment, so an
        # exact p-value can never be zero and needs no Monte Carlo correction.
        # Recomputing the observed statistic through a batched BLAS product can
        # differ by a few ulps from the separately computed statistic, however,
        # so explicitly enforce the mathematically required observed orbit.
        return np.maximum(count, 1.0) / float(n_patterns)
    return (count + 1.0) / float(n_patterns + 1)


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    try:
        return float(np.quantile(finite, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(finite, probability, interpolation="higher"))


@dataclass(frozen=True)
class CompleteFamilyResult:
    """Dense inference results in edge-major ``[edge, lag, horizon]`` order."""

    mean: np.ndarray
    se: np.ndarray
    t_statistic: np.ndarray
    n_valid: np.ndarray
    pointwise_p_value: np.ndarray
    global_max_t_p_value: np.ndarray
    zero_null_reference_band_low: np.ndarray
    zero_null_reference_band_high: np.ndarray
    edge_max_abs_t: np.ndarray
    edge_p_value: np.ndarray
    edge_bh_q_value: np.ndarray
    edge_global_fwer_p_value: np.ndarray
    lag_contrast_mean: np.ndarray
    lag_contrast_se: np.ndarray
    lag_contrast_t: np.ndarray
    lag_contrast_pointwise_p_value: np.ndarray
    lag_contrast_global_max_t_p_value: np.ndarray
    lag_contrast_zero_null_reference_band_low: np.ndarray
    lag_contrast_zero_null_reference_band_high: np.ndarray
    flat_lag_max_abs_t: np.ndarray
    flat_lag_p_value: np.ndarray
    flat_lag_bh_q_value: np.ndarray
    flat_lag_global_fwer_p_value: np.ndarray
    signs: np.ndarray
    inference_mode: str
    global_max_abs_t: np.ndarray
    global_max_abs_lag_contrast_t: np.ndarray
    simultaneous_critical_value: float
    lag_contrast_simultaneous_critical_value: float


def _expand_edge_result(
    result: CompleteFamilyResult,
    edge_mask: np.ndarray,
) -> CompleteFamilyResult:
    """Expand an eligible-edge computation to an explicit complete edge family."""
    mask = np.asarray(edge_mask, dtype=bool)
    if int(mask.sum()) != result.mean.shape[0]:
        raise ValueError("edge mask does not match compact inference result")
    total_edges = len(mask)
    cell_shape = (total_edges,) + result.mean.shape[1:]

    def cells(values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        out = np.full(cell_shape, fill, dtype=values.dtype)
        out[mask] = values
        return out

    def edges(values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        out = np.full(total_edges, fill, dtype=values.dtype)
        out[mask] = values
        return out

    edge_p = edges(result.edge_p_value)
    edge_q = benjamini_hochberg(np.where(np.isfinite(edge_p), edge_p, 1.0))
    flat_p = edges(result.flat_lag_p_value)
    flat_q = benjamini_hochberg(np.where(np.isfinite(flat_p), flat_p, 1.0))
    return CompleteFamilyResult(
        mean=cells(result.mean),
        se=cells(result.se),
        t_statistic=cells(result.t_statistic),
        n_valid=cells(result.n_valid, fill=0),
        pointwise_p_value=cells(result.pointwise_p_value),
        global_max_t_p_value=cells(result.global_max_t_p_value),
        zero_null_reference_band_low=cells(
            result.zero_null_reference_band_low
        ),
        zero_null_reference_band_high=cells(
            result.zero_null_reference_band_high
        ),
        edge_max_abs_t=edges(result.edge_max_abs_t),
        edge_p_value=edge_p,
        edge_bh_q_value=edge_q,
        edge_global_fwer_p_value=edges(result.edge_global_fwer_p_value),
        lag_contrast_mean=cells(result.lag_contrast_mean),
        lag_contrast_se=cells(result.lag_contrast_se),
        lag_contrast_t=cells(result.lag_contrast_t),
        lag_contrast_pointwise_p_value=cells(
            result.lag_contrast_pointwise_p_value
        ),
        lag_contrast_global_max_t_p_value=cells(
            result.lag_contrast_global_max_t_p_value
        ),
        lag_contrast_zero_null_reference_band_low=cells(
            result.lag_contrast_zero_null_reference_band_low
        ),
        lag_contrast_zero_null_reference_band_high=cells(
            result.lag_contrast_zero_null_reference_band_high
        ),
        flat_lag_max_abs_t=edges(result.flat_lag_max_abs_t),
        flat_lag_p_value=flat_p,
        flat_lag_bh_q_value=flat_q,
        flat_lag_global_fwer_p_value=edges(
            result.flat_lag_global_fwer_p_value
        ),
        signs=result.signs,
        inference_mode=result.inference_mode,
        global_max_abs_t=result.global_max_abs_t,
        global_max_abs_lag_contrast_t=(
            result.global_max_abs_lag_contrast_t
        ),
        simultaneous_critical_value=result.simultaneous_critical_value,
        lag_contrast_simultaneous_critical_value=(
            result.lag_contrast_simultaneous_critical_value
        ),
    )


def infer_complete_family(
    values: np.ndarray,
    *,
    eligibility: np.ndarray | None = None,
    mode: str = "auto",
    replicates: int = DEFAULT_MONTE_CARLO_REPLICATES,
    seed: int = DEFAULT_SEED,
    max_exact_patterns: int = DEFAULT_MAX_EXACT_PATTERNS,
    batch_size: int = 64,
    min_worms: int = 3,
    alpha: float = DEFAULT_ALPHA,
) -> CompleteFamilyResult:
    """Infer a complete edge-by-lag-by-horizon family.

    Parameters
    ----------
    values:
        Array with axes ``[worm, edge, lag, horizon]``.  Diagonal edges must
        already be absent.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError("values must have shape [worm, edge, lag, horizon]")
    n_worms, n_edges, n_lags, n_horizons = x.shape
    if n_edges < 1 or n_lags < 1 or n_horizons < 1:
        raise ValueError("edge, lag, and horizon axes must be nonempty")
    if min_worms < 2 or min_worms > n_worms:
        raise ValueError("min_worms must be between 2 and n_worms")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if eligibility is not None:
        gate = np.asarray(eligibility, dtype=bool)
        try:
            gate = np.broadcast_to(gate, (n_edges, n_lags, n_horizons))
        except ValueError as error:
            raise ValueError(
                "eligibility must broadcast to [edge,lag,horizon]"
            ) from error
        # The support gate is frozen before looking at effect size.  Masking at
        # the worm tensor makes every downstream statistic explicitly absent
        # for unsupported cells rather than silently treating it as evidence.
        x = np.where(gate[None, ...], x, np.nan)

    signs, resolved_mode = generate_sign_patterns(
        n_worms,
        mode=mode,
        replicates=replicates,
        seed=seed,
        max_exact_patterns=max_exact_patterns,
    )
    n_patterns = len(signs)
    n_cells = n_edges * n_lags * n_horizons
    flat = x.reshape(n_worms, n_cells)
    filled, count, sumsq, mean_flat, se_flat, observed_t_flat = _summary_terms(
        flat, min_worms=min_worms
    )
    observed_t = observed_t_flat.reshape(n_edges, n_lags, n_horizons)
    observed_abs = np.abs(observed_t)
    observed_edge = np.max(
        np.where(np.isfinite(observed_abs), observed_abs, -np.inf), axis=(1, 2)
    )
    observed_edge[~np.isfinite(observed_edge)] = np.nan

    # Flat-lag null: for every worm, edge, and horizon, subtract the average
    # across the selected lag grid.  A nonzero population mean of any centered
    # cell implies that the lag profile is not flat at that horizon.
    lag_contrast = np.full_like(x, np.nan)
    if n_lags >= 2:
        finite_count = np.isfinite(x).sum(axis=2, keepdims=True)
        lag_average = np.divide(
            np.nansum(x, axis=2, keepdims=True),
            finite_count,
            out=np.full((n_worms, n_edges, 1, n_horizons), np.nan),
            where=finite_count > 0,
        )
        lag_contrast = x - lag_average
    contrast_flat = lag_contrast.reshape(n_worms, n_cells)
    (
        contrast_filled,
        contrast_count,
        contrast_sumsq,
        contrast_mean_flat,
        contrast_se_flat,
        contrast_t_flat,
    ) = _summary_terms(contrast_flat, min_worms=min_worms)
    contrast_t = contrast_t_flat.reshape(n_edges, n_lags, n_horizons)
    contrast_abs = np.abs(contrast_t)
    observed_flat_lag = np.max(
        np.where(np.isfinite(contrast_abs), contrast_abs, -np.inf), axis=(1, 2)
    )
    observed_flat_lag[~np.isfinite(observed_flat_lag)] = np.nan

    point_exceed = np.zeros(n_cells, dtype=np.int64)
    global_point_exceed = np.zeros(n_cells, dtype=np.int64)
    edge_exceed = np.zeros(n_edges, dtype=np.int64)
    edge_global_exceed = np.zeros(n_edges, dtype=np.int64)
    contrast_point_exceed = np.zeros(n_cells, dtype=np.int64)
    contrast_global_exceed = np.zeros(n_cells, dtype=np.int64)
    flat_lag_exceed = np.zeros(n_edges, dtype=np.int64)
    flat_lag_global_exceed = np.zeros(n_edges, dtype=np.int64)
    global_max = np.empty(n_patterns, dtype=np.float64)
    contrast_global_max = np.full(n_patterns, np.nan, dtype=np.float64)
    complete_or_absent = np.all(np.isfinite(x), axis=0) | np.all(
        ~np.isfinite(x), axis=0
    )
    derive_contrast_from_primary_sum = bool(
        n_lags >= 2 and np.all(complete_or_absent)
    )

    for start in range(0, n_patterns, batch_size):
        stop = min(start + batch_size, n_patterns)
        sign_batch = signs[start:stop].astype(np.float64, copy=False)
        signed_sum = sign_batch @ filled
        null_t_flat = _student_t_from_sum(
            signed_sum,
            count,
            sumsq,
            min_worms=min_worms,
        )
        null_abs = np.abs(null_t_flat)
        null_edge = np.max(
            np.where(
                np.isfinite(null_abs.reshape(-1, n_edges, n_lags, n_horizons)),
                null_abs.reshape(-1, n_edges, n_lags, n_horizons),
                -np.inf,
            ),
            axis=(2, 3),
        )
        batch_global = np.max(null_edge, axis=1)
        global_max[start:stop] = batch_global
        point_exceed += np.sum(null_abs >= np.abs(observed_t_flat)[None, :], axis=0)
        global_point_exceed += np.sum(
            batch_global[:, None] >= np.abs(observed_t_flat)[None, :], axis=0
        )
        edge_exceed += np.sum(null_edge >= observed_edge[None, :], axis=0)
        edge_global_exceed += np.sum(
            batch_global[:, None] >= observed_edge[None, :], axis=0
        )

        if n_lags >= 2:
            if derive_contrast_from_primary_sum:
                signed_sum_grid = signed_sum.reshape(
                    -1, n_edges, n_lags, n_horizons
                )
                contrast_signed_sum = (
                    signed_sum_grid
                    - signed_sum_grid.mean(axis=2, keepdims=True)
                ).reshape(stop - start, n_cells)
            else:
                contrast_signed_sum = sign_batch @ contrast_filled
            null_contrast_t_flat = _student_t_from_sum(
                contrast_signed_sum,
                contrast_count,
                contrast_sumsq,
                min_worms=min_worms,
            )
            null_contrast_abs = np.abs(null_contrast_t_flat)
            null_contrast_edge = np.max(
                np.where(
                    np.isfinite(
                        null_contrast_abs.reshape(
                            -1, n_edges, n_lags, n_horizons
                        )
                    ),
                    null_contrast_abs.reshape(-1, n_edges, n_lags, n_horizons),
                    -np.inf,
                ),
                axis=(2, 3),
            )
            batch_contrast_global = np.max(null_contrast_edge, axis=1)
            contrast_global_max[start:stop] = batch_contrast_global
            contrast_point_exceed += np.sum(
                null_contrast_abs >= np.abs(contrast_t_flat)[None, :], axis=0
            )
            contrast_global_exceed += np.sum(
                batch_contrast_global[:, None]
                >= np.abs(contrast_t_flat)[None, :],
                axis=0,
            )
            flat_lag_exceed += np.sum(
                null_contrast_edge >= observed_flat_lag[None, :], axis=0
            )
            flat_lag_global_exceed += np.sum(
                batch_contrast_global[:, None] >= observed_flat_lag[None, :],
                axis=0,
            )

    point_p = _randomization_p(point_exceed, n_patterns, resolved_mode)
    global_point_p = _randomization_p(
        global_point_exceed, n_patterns, resolved_mode
    )
    edge_p = _randomization_p(edge_exceed, n_patterns, resolved_mode)
    edge_fwer_p = _randomization_p(edge_global_exceed, n_patterns, resolved_mode)
    invalid_cells = ~np.isfinite(observed_t_flat)
    point_p[invalid_cells] = np.nan
    global_point_p[invalid_cells] = np.nan
    edge_p[~np.isfinite(observed_edge)] = np.nan
    edge_fwer_p[~np.isfinite(observed_edge)] = np.nan
    # Unsupported edges remain explicitly untested (raw p=NaN), but enter the
    # declared complete 54*53 edge multiplicity family as p=1.  This preserves
    # the full-family denominator without manufacturing a test result.
    edge_q = benjamini_hochberg(np.where(np.isfinite(edge_p), edge_p, 1.0))

    critical = _higher_quantile(global_max, 1.0 - alpha)
    ci_low = mean_flat - critical * se_flat
    ci_high = mean_flat + critical * se_flat

    if n_lags >= 2:
        contrast_point_p = _randomization_p(
            contrast_point_exceed, n_patterns, resolved_mode
        )
        contrast_global_p = _randomization_p(
            contrast_global_exceed, n_patterns, resolved_mode
        )
        flat_p = _randomization_p(flat_lag_exceed, n_patterns, resolved_mode)
        flat_fwer_p = _randomization_p(
            flat_lag_global_exceed, n_patterns, resolved_mode
        )
        contrast_invalid = ~np.isfinite(contrast_t_flat)
        contrast_point_p[contrast_invalid] = np.nan
        contrast_global_p[contrast_invalid] = np.nan
        flat_p[~np.isfinite(observed_flat_lag)] = np.nan
        flat_fwer_p[~np.isfinite(observed_flat_lag)] = np.nan
        flat_q = benjamini_hochberg(np.where(np.isfinite(flat_p), flat_p, 1.0))
        contrast_critical = _higher_quantile(contrast_global_max, 1.0 - alpha)
        contrast_ci_low = contrast_mean_flat - contrast_critical * contrast_se_flat
        contrast_ci_high = contrast_mean_flat + contrast_critical * contrast_se_flat
    else:
        contrast_point_p = np.full(n_cells, np.nan)
        contrast_global_p = np.full(n_cells, np.nan)
        flat_p = np.full(n_edges, np.nan)
        flat_fwer_p = np.full(n_edges, np.nan)
        flat_q = np.full(n_edges, np.nan)
        contrast_critical = float("nan")
        contrast_ci_low = np.full(n_cells, np.nan)
        contrast_ci_high = np.full(n_cells, np.nan)

    cell_shape = (n_edges, n_lags, n_horizons)
    return CompleteFamilyResult(
        mean=mean_flat.reshape(cell_shape),
        se=se_flat.reshape(cell_shape),
        t_statistic=observed_t,
        n_valid=count.reshape(cell_shape),
        pointwise_p_value=point_p.reshape(cell_shape),
        global_max_t_p_value=global_point_p.reshape(cell_shape),
        zero_null_reference_band_low=ci_low.reshape(cell_shape),
        zero_null_reference_band_high=ci_high.reshape(cell_shape),
        edge_max_abs_t=observed_edge,
        edge_p_value=edge_p,
        edge_bh_q_value=edge_q,
        edge_global_fwer_p_value=edge_fwer_p,
        lag_contrast_mean=contrast_mean_flat.reshape(cell_shape),
        lag_contrast_se=contrast_se_flat.reshape(cell_shape),
        lag_contrast_t=contrast_t,
        lag_contrast_pointwise_p_value=contrast_point_p.reshape(cell_shape),
        lag_contrast_global_max_t_p_value=contrast_global_p.reshape(cell_shape),
        lag_contrast_zero_null_reference_band_low=contrast_ci_low.reshape(
            cell_shape
        ),
        lag_contrast_zero_null_reference_band_high=contrast_ci_high.reshape(
            cell_shape
        ),
        flat_lag_max_abs_t=observed_flat_lag,
        flat_lag_p_value=flat_p,
        flat_lag_bh_q_value=flat_q,
        flat_lag_global_fwer_p_value=flat_fwer_p,
        signs=signs,
        inference_mode=resolved_mode,
        global_max_abs_t=global_max,
        global_max_abs_lag_contrast_t=contrast_global_max,
        simultaneous_critical_value=critical,
        lag_contrast_simultaneous_critical_value=contrast_critical,
    )


def _decode_scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _select_positions(
    available: np.ndarray,
    requested: Sequence[int] | None,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(available, dtype=np.int64)
    if requested is None:
        return np.arange(len(values), dtype=np.int64), values
    chosen = np.asarray(requested, dtype=np.int64)
    if not len(chosen):
        raise ValueError(f"at least one {label} must be selected")
    if len(np.unique(chosen)) != len(chosen):
        raise ValueError(f"selected {label} values must be unique")
    lookup = {int(value): index for index, value in enumerate(values)}
    missing = [int(value) for value in chosen if int(value) not in lookup]
    if missing:
        raise ValueError(
            f"requested {label} values {missing} are absent; available={values.tolist()}"
        )
    return np.asarray([lookup[int(value)] for value in chosen]), chosen


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _json_dump(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_file_provenance(path: Path | None) -> dict[str, object] | None:
    """Pin one input file and, when covered, its immediate checksum ledger.

    A parent ``checksums.sha256`` is recorded only when it contains exactly one
    entry for the input basename and that entry agrees with the file bytes.  A
    present-but-stale covering entry is an integrity error, not optional
    metadata.
    """
    if path is None:
        return None
    resolved = Path(path).resolve()
    file_sha256 = _sha256(resolved)
    provenance: dict[str, object] = {
        "path": str(resolved),
        "sha256": file_sha256,
        "parent_checksum_ledger": None,
    }
    ledger_path = resolved.parent / "checksums.sha256"
    if not ledger_path.is_file():
        return provenance

    matching: list[tuple[str, str]] = []
    for line_number, raw in enumerate(ledger_path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            expected, name = line.split(None, 1)
        except ValueError as error:
            raise ValueError(
                f"malformed checksum line {line_number} in {ledger_path}"
            ) from error
        name = name.strip()
        if Path(name) == Path(resolved.name):
            matching.append((expected, name))
    if len(matching) > 1:
        raise ValueError(
            f"checksum ledger has duplicate entries for {resolved.name}: {ledger_path}"
        )
    if matching:
        expected, name = matching[0]
        if expected != file_sha256:
            raise ValueError(
                f"checksum mismatch for {resolved}: expected {expected}, got {file_sha256}"
            )
        provenance["parent_checksum_ledger"] = {
            "path": str(ledger_path.resolve()),
            "sha256": _sha256(ledger_path),
            "entry_name": name,
            "entry_sha256": expected,
        }
    return provenance


def _is_signed_estimand(channel: str, context: str) -> bool:
    return channel != UNSIGNED_CHANNEL or context.endswith("minus_baseline")


@dataclass(frozen=True)
class SupportEligibility:
    """Effect-independent source gate evaluated across every selected lag."""

    strong_source_eligible: np.ndarray
    sensitivity_source_eligible: np.ndarray
    minimum_valid_fraction: np.ndarray
    minimum_genealogy_valid_fraction: np.ndarray
    support_contexts: tuple[str, ...]
    path: Path | None
    method: str
    strong_threshold: float
    genealogy_threshold: float
    sensitivity_threshold: float


def load_support_eligibility(
    support_path: Path | None,
    *,
    method: str,
    support_contexts: Sequence[str],
    n_neurons: int,
    selected_lags: Sequence[int],
    strong_threshold: float = 0.8,
    genealogy_threshold: float = 0.8,
    sensitivity_threshold: float = 0.5,
) -> SupportEligibility:
    """Load the frozen progressive support gate for a source-by-lag grid.

    Strong eligibility requires, independently of learned effect magnitude,
    canonical ``support_qualified``, ``valid_fraction >= strong_threshold``,
    canonical ``genealogy_strong_gate_pass``, and numeric
    ``genealogy_valid_fraction_0_10 >= genealogy_threshold`` at *every* tested
    lag and in *every* conditioning context used by the estimand.  The separate
    0.5 sensitivity flag is reported but never substituted into inference.
    """
    if not 0 <= sensitivity_threshold <= strong_threshold <= 1:
        raise ValueError(
            "support thresholds must satisfy 0 <= sensitivity <= strong <= 1"
        )
    if not 0 <= genealogy_threshold <= 1:
        raise ValueError("genealogy_threshold must lie between zero and one")
    contexts = tuple(str(value) for value in support_contexts)
    lags = tuple(int(value) for value in selected_lags)
    if not contexts or not lags:
        raise ValueError("support contexts and selected lags must be nonempty")

    if support_path is None:
        return SupportEligibility(
            strong_source_eligible=np.ones(n_neurons, dtype=bool),
            sensitivity_source_eligible=np.ones(n_neurons, dtype=bool),
            minimum_valid_fraction=np.full(n_neurons, np.nan),
            minimum_genealogy_valid_fraction=np.full(n_neurons, np.nan),
            support_contexts=contexts,
            path=None,
            method=method,
            strong_threshold=strong_threshold,
            genealogy_threshold=genealogy_threshold,
            sensitivity_threshold=sensitivity_threshold,
        )

    path = Path(support_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("support_cells must be parquet or CSV")
    required = {
        "method",
        "context",
        "source_index",
        "source_lag_frames",
        "valid_fraction",
        "support_qualified",
        "genealogy_gate_applicable",
        "genealogy_valid_fraction_0_10",
        "genealogy_strong_gate_pass",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"support table is missing columns {missing}")
    subset = frame.loc[
        (frame["method"].astype(str) == method)
        & frame["context"].astype(str).isin(contexts)
        & frame["source_lag_frames"].astype(int).isin(lags)
    ].copy()
    duplicate = subset.duplicated(
        ["context", "source_index", "source_lag_frames"], keep=False
    )
    if duplicate.any():
        examples = subset.loc[
            duplicate, ["context", "source_index", "source_lag_frames"]
        ].head(10)
        raise ValueError(
            "support table has duplicate join keys: " + examples.to_dict(orient="records").__repr__()
        )

    strong = np.zeros((len(contexts), n_neurons, len(lags)), dtype=bool)
    sensitivity = np.zeros_like(strong)
    valid = np.full(strong.shape, np.nan, dtype=np.float64)
    genealogy = np.full(strong.shape, np.nan, dtype=np.float64)
    context_position = {value: index for index, value in enumerate(contexts)}
    lag_position = {value: index for index, value in enumerate(lags)}
    for row in subset.itertuples(index=False):
        source = int(row.source_index)
        if not 0 <= source < n_neurons:
            raise ValueError(f"support source_index {source} is out of range")
        c = context_position[str(row.context)]
        ell = lag_position[int(row.source_lag_frames)]
        valid_value = float(row.valid_fraction)
        genealogy_applicable = bool(row.genealogy_gate_applicable)
        genealogy_value = (
            float(row.genealogy_valid_fraction_0_10)
            if genealogy_applicable
            else 1.0
        )
        valid[c, source, ell] = valid_value
        genealogy[c, source, ell] = genealogy_value
        canonical_support = bool(row.support_qualified)
        canonical_genealogy = bool(row.genealogy_strong_gate_pass)
        strong[c, source, ell] = bool(
            canonical_support
            and valid_value >= strong_threshold
            and canonical_genealogy
            and genealogy_value >= genealogy_threshold
        )
        sensitivity[c, source, ell] = bool(
            canonical_support
            and valid_value >= sensitivity_threshold
            and genealogy_value >= sensitivity_threshold
        )

    # All-lag joint eligibility is essential for an honest lag-flatness test:
    # every compared lag is observed through the same frozen source set.
    strong_source = strong.all(axis=(0, 2))
    sensitivity_source = sensitivity.all(axis=(0, 2))
    missing_source = ~np.isfinite(valid).all(axis=(0, 2))
    min_valid = np.min(
        np.where(np.isfinite(valid), valid, np.inf), axis=(0, 2)
    )
    min_genealogy = np.min(
        np.where(np.isfinite(genealogy), genealogy, np.inf), axis=(0, 2)
    )
    min_valid[missing_source] = np.nan
    min_genealogy[missing_source] = np.nan
    strong_source[missing_source] = False
    sensitivity_source[missing_source] = False
    return SupportEligibility(
        strong_source_eligible=strong_source,
        sensitivity_source_eligible=sensitivity_source,
        minimum_valid_fraction=min_valid,
        minimum_genealogy_valid_fraction=min_genealogy,
        support_contexts=contexts,
        path=path,
        method=method,
        strong_threshold=strong_threshold,
        genealogy_threshold=genealogy_threshold,
        sensitivity_threshold=sensitivity_threshold,
    )


def _dense_edge_array(
    edge_values: np.ndarray,
    target_index: np.ndarray,
    source_index: np.ndarray,
    n_neurons: int,
    *,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    dense = np.full((n_neurons, n_neurons) + trailing_shape, np.nan, dtype=np.float64)
    dense[target_index, source_index] = edge_values
    return dense


def run_full_family_inference(
    input_npz: Path,
    output_dir: Path,
    *,
    channel: str,
    context: str,
    contrast: str | None = None,
    lag_frames: Sequence[int] | None = None,
    horizon_frames: Sequence[int] | None = None,
    support_cells: Path | None = None,
    support_threshold: float = 0.8,
    genealogy_threshold: float = 0.8,
    sensitivity_threshold: float = 0.5,
    support_gate: str = "strong",
    mode: str = "auto",
    replicates: int = DEFAULT_MONTE_CARLO_REPLICATES,
    seed: int = DEFAULT_SEED,
    max_exact_patterns: int = DEFAULT_MAX_EXACT_PATTERNS,
    batch_size: int = 64,
    min_worms: int = 3,
    alpha: float = DEFAULT_ALPHA,
    allow_unsigned: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Load one channel/context family, run inference, and write artifacts."""
    input_path = Path(input_npz).resolve()
    output = Path(output_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if contrast not in {None, "active-minus-baseline"}:
        raise ValueError("contrast must be omitted or 'active-minus-baseline'")
    if support_gate not in {"strong", "sensitivity"}:
        raise ValueError("support_gate must be 'strong' or 'sensitivity'")
    output_context = "active_minus_baseline" if contrast else context
    if not _is_signed_estimand(channel, output_context) and not allow_unsigned:
        raise ValueError(
            f"{channel}/{output_context} is an unsigned within-state distance.  A zero-null "
            "sign-flip is invalid; use a sham-distance analysis or pass "
            "allow_unsigned=True only for a documented sensitivity run."
        )

    artifact_names = {
        "edge_inference.csv",
        "cell_inference.csv",
        "inference_arrays.npz",
        "protocol.json",
        "summary.json",
        "validation.json",
        "manifest.json",
        "checksums.sha256",
    }
    if output.exists() and any((output / name).exists() for name in artifact_names):
        if not overwrite:
            raise FileExistsError(f"inference outputs already exist in {output}")
        for name in artifact_names:
            path = output / name
            if path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=False) as archive:
        if contrast == "active-minus-baseline":
            input_keys = (
                f"normalized__{channel}__active",
                f"normalized__{channel}__baseline",
            )
            support_contexts = ("active", "baseline")
        else:
            input_keys = (f"normalized__{channel}__{context}",)
            support_contexts = (context,)
        missing_input_keys = [key for key in input_keys if key not in archive.files]
        if missing_input_keys:
            available = sorted(
                name.removeprefix("normalized__")
                for name in archive.files
                if name.startswith("normalized__")
            )
            raise KeyError(
                f"missing {missing_input_keys!r}; available channel/context keys={available}"
            )
        required = {
            "neurons",
            "worm_ids",
            "source_lag_frames",
            "source_lag_seconds",
            "horizon_frames",
            "horizon_seconds",
            "orientation",
            "primary_method",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"input archive is missing metadata keys {missing}")
        orientation = _decode_scalar(archive["orientation"])
        if orientation != ORIENTATION:
            raise ValueError(
                f"expected orientation {ORIENTATION!r}, received {orientation!r}"
            )
        neurons = np.asarray(archive["neurons"]).astype(str)
        worm_ids = np.asarray(archive["worm_ids"]).astype(str)
        available_lags = np.asarray(archive["source_lag_frames"], dtype=np.int64)
        available_lag_seconds = np.asarray(
            archive["source_lag_seconds"], dtype=np.float64
        )
        available_horizons = np.asarray(archive["horizon_frames"], dtype=np.int64)
        available_horizon_seconds = np.asarray(
            archive["horizon_seconds"], dtype=np.float64
        )
        primary_method = _decode_scalar(archive["primary_method"])
        if contrast == "active-minus-baseline":
            matrix = np.asarray(archive[input_keys[0]], dtype=np.float64) - np.asarray(
                archive[input_keys[1]], dtype=np.float64
            )
        else:
            matrix = np.asarray(archive[input_keys[0]], dtype=np.float64)

    if matrix.ndim != 5:
        raise ValueError(
            f"effect tensor must have [lag,worm,horizon,target,source] axes, got {matrix.shape}"
        )
    n_lag_all, n_worms, n_horizon_all, n_target, n_source = matrix.shape
    if n_target != n_source or n_target != len(neurons):
        raise ValueError("matrix neuron axes do not match neuron metadata")
    if n_worms != len(worm_ids):
        raise ValueError("matrix worm axis does not match worm metadata")
    if n_lag_all != len(available_lags) or n_horizon_all != len(available_horizons):
        raise ValueError("matrix lag/horizon axes do not match frame metadata")

    lag_pos, selected_lags = _select_positions(
        available_lags, lag_frames, label="lag frames"
    )
    if contrast == "active-minus-baseline" and horizon_frames is None:
        # The 8-second (h32) active forecast crosses the stimulus offset and is
        # not a clean active-minus-baseline estimand.
        horizon_frames = [int(value) for value in available_horizons if value <= 16]
    horizon_pos, selected_horizons = _select_positions(
        available_horizons, horizon_frames, label="horizon frames"
    )
    lag_seconds = available_lag_seconds[lag_pos]
    horizon_seconds = available_horizon_seconds[horizon_pos]

    target_index, source_index = np.nonzero(~np.eye(len(neurons), dtype=bool))
    resolved_support_path = Path(support_cells).resolve() if support_cells else None
    if resolved_support_path is None:
        for name in ("support_cells.parquet", "support_cells.csv"):
            candidate = input_path.parent / name
            if candidate.exists():
                resolved_support_path = candidate.resolve()
                break
    support = load_support_eligibility(
        resolved_support_path,
        method=primary_method,
        support_contexts=support_contexts,
        n_neurons=len(neurons),
        selected_lags=selected_lags,
        strong_threshold=support_threshold,
        genealogy_threshold=genealogy_threshold,
        sensitivity_threshold=sensitivity_threshold,
    )
    support_provenance = _pinned_file_provenance(support.path)
    input_provenance = _pinned_file_provenance(input_path)
    assert input_provenance is not None
    selected_source_eligible = (
        support.strong_source_eligible
        if support_gate == "strong"
        else support.sensitivity_source_eligible
    )
    edge_support_eligible = selected_source_eligible[source_index]
    edge_sensitivity_eligible = support.sensitivity_source_eligible[source_index]
    # Advanced indexing yields [lag,worm,horizon,edge].  Reorder to the public
    # engine contract [worm,edge,lag,horizon].
    selected = matrix[np.ix_(lag_pos, np.arange(n_worms), horizon_pos)]
    edge_values = selected[:, :, :, target_index, source_index].transpose(1, 3, 0, 2)
    if not edge_support_eligible.any():
        raise RuntimeError(
            "no directed edge has a source passing the frozen all-lag support gate"
        )
    compact_result = infer_complete_family(
        edge_values[:, edge_support_eligible],
        mode=mode,
        replicates=replicates,
        seed=seed,
        max_exact_patterns=max_exact_patterns,
        batch_size=batch_size,
        min_worms=min_worms,
        alpha=alpha,
    )
    result = _expand_edge_result(compact_result, edge_support_eligible)

    peak_flat = np.argmax(
        np.where(np.isfinite(result.t_statistic), np.abs(result.t_statistic), -np.inf).reshape(
            len(target_index), -1
        ),
        axis=1,
    )
    peak_lag_pos, peak_horizon_pos = np.unravel_index(
        peak_flat, (len(selected_lags), len(selected_horizons))
    )
    flat_peak_flat = np.argmax(
        np.where(
            np.isfinite(result.lag_contrast_t),
            np.abs(result.lag_contrast_t),
            -np.inf,
        ).reshape(len(target_index), -1),
        axis=1,
    )
    flat_peak_lag_pos, flat_peak_horizon_pos = np.unravel_index(
        flat_peak_flat, (len(selected_lags), len(selected_horizons))
    )

    edge_frame = pd.DataFrame(
        {
            "target_neuron": neurons[target_index],
            "source_neuron": neurons[source_index],
            "target_index": target_index,
            "source_index": source_index,
            "channel": channel,
            "context": output_context,
            "primary_method": primary_method,
            "n_worms": n_worms,
            "n_lag_horizon_cells": len(selected_lags) * len(selected_horizons),
            "support_eligible": edge_support_eligible,
            "support_gate": support_gate,
            "support_sensitivity_eligible": edge_sensitivity_eligible,
            "support_min_valid_fraction_all_contexts_lags": (
                support.minimum_valid_fraction[source_index]
            ),
            "support_min_genealogy_valid_fraction_0_10_all_contexts_lags": (
                support.minimum_genealogy_valid_fraction[source_index]
            ),
            "edge_max_abs_t": result.edge_max_abs_t,
            "edge_omnibus_p_value": result.edge_p_value,
            "edge_bh_q_value": result.edge_bh_q_value,
            "edge_global_fwer_p_value": result.edge_global_fwer_p_value,
            "edge_bh_reject": edge_support_eligible
            & (result.edge_bh_q_value <= alpha),
            "edge_global_fwer_reject": edge_support_eligible
            & (result.edge_global_fwer_p_value <= alpha),
            "peak_lag_frames": np.where(
                edge_support_eligible, selected_lags[peak_lag_pos], np.nan
            ),
            "peak_lag_seconds": np.where(
                edge_support_eligible, lag_seconds[peak_lag_pos], np.nan
            ),
            "peak_horizon_frames": np.where(
                edge_support_eligible, selected_horizons[peak_horizon_pos], np.nan
            ),
            "peak_horizon_seconds": np.where(
                edge_support_eligible, horizon_seconds[peak_horizon_pos], np.nan
            ),
            "peak_mean_normalized": result.mean[
                np.arange(len(target_index)), peak_lag_pos, peak_horizon_pos
            ],
            "peak_t_statistic": result.t_statistic[
                np.arange(len(target_index)), peak_lag_pos, peak_horizon_pos
            ],
            "flat_lag_max_abs_t": result.flat_lag_max_abs_t,
            "flat_lag_omnibus_p_value": result.flat_lag_p_value,
            "flat_lag_bh_q_value": result.flat_lag_bh_q_value,
            "flat_lag_global_fwer_p_value": result.flat_lag_global_fwer_p_value,
            "flat_lag_bh_reject": edge_support_eligible
            & (result.flat_lag_bh_q_value <= alpha),
            "flat_lag_global_fwer_reject": (
                edge_support_eligible
                & (result.flat_lag_global_fwer_p_value <= alpha)
            ),
            "flat_lag_peak_lag_frames": np.where(
                edge_support_eligible, selected_lags[flat_peak_lag_pos], np.nan
            ),
            "flat_lag_peak_horizon_frames": np.where(
                edge_support_eligible,
                selected_horizons[flat_peak_horizon_pos],
                np.nan,
            ),
            "interpretation_limit": (
                "frozen_model_relative_association_not_causal_or_physical_delay"
            ),
        }
    )
    edge_frame = edge_frame.sort_values(
        ["edge_bh_q_value", "edge_omnibus_p_value", "edge_max_abs_t"],
        ascending=[True, True, False],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    edge_frame.insert(0, "edge_rank", np.arange(1, len(edge_frame) + 1))

    edge_grid, lag_grid, horizon_grid = np.meshgrid(
        np.arange(len(target_index)),
        np.arange(len(selected_lags)),
        np.arange(len(selected_horizons)),
        indexing="ij",
    )
    ef = edge_grid.ravel()
    lf = lag_grid.ravel()
    hf = horizon_grid.ravel()
    cell_frame = pd.DataFrame(
        {
            "target_neuron": neurons[target_index[ef]],
            "source_neuron": neurons[source_index[ef]],
            "target_index": target_index[ef],
            "source_index": source_index[ef],
            "channel": channel,
            "context": output_context,
            "support_eligible": edge_support_eligible[ef],
            "support_gate": support_gate,
            "support_sensitivity_eligible": edge_sensitivity_eligible[ef],
            "support_min_valid_fraction_all_contexts_lags": (
                support.minimum_valid_fraction[source_index[ef]]
            ),
            "support_min_genealogy_valid_fraction_0_10_all_contexts_lags": (
                support.minimum_genealogy_valid_fraction[source_index[ef]]
            ),
            "source_lag_frames": selected_lags[lf],
            "source_lag_seconds": lag_seconds[lf],
            "horizon_frames": selected_horizons[hf],
            "horizon_seconds": horizon_seconds[hf],
            "source_to_readout_seconds": lag_seconds[lf] + horizon_seconds[hf],
            "n_valid_worms": result.n_valid.ravel(),
            "mean_normalized": result.mean.ravel(),
            "standard_error": result.se.ravel(),
            "student_t": result.t_statistic.ravel(),
            "pointwise_sign_flip_p_value": result.pointwise_p_value.ravel(),
            "global_max_t_p_value": result.global_max_t_p_value.ravel(),
            "zero_null_reference_band_low": (
                result.zero_null_reference_band_low.ravel()
            ),
            "zero_null_reference_band_high": (
                result.zero_null_reference_band_high.ravel()
            ),
            "global_max_t_reject": edge_support_eligible[ef]
            & (result.global_max_t_p_value.ravel() <= alpha),
            "lag_contrast_mean": result.lag_contrast_mean.ravel(),
            "lag_contrast_standard_error": result.lag_contrast_se.ravel(),
            "lag_contrast_student_t": result.lag_contrast_t.ravel(),
            "lag_contrast_pointwise_p_value": (
                result.lag_contrast_pointwise_p_value.ravel()
            ),
            "lag_contrast_global_max_t_p_value": (
                result.lag_contrast_global_max_t_p_value.ravel()
            ),
            "lag_contrast_zero_null_reference_band_low": (
                result.lag_contrast_zero_null_reference_band_low.ravel()
            ),
            "lag_contrast_zero_null_reference_band_high": (
                result.lag_contrast_zero_null_reference_band_high.ravel()
            ),
            "lag_contrast_global_max_t_reject": (
                edge_support_eligible[ef]
                & (result.lag_contrast_global_max_t_p_value.ravel() <= alpha)
            ),
            "interpretation_limit": (
                "cell_localization_is_simultaneous_across_complete_selected_family"
            ),
        }
    )

    edge_path = output / "edge_inference.csv"
    cell_path = output / "cell_inference.csv"
    _atomic_csv(edge_frame, edge_path)
    _atomic_csv(cell_frame, cell_path)

    dense_shape = (len(selected_lags), len(selected_horizons))
    array_path = output / "inference_arrays.npz"
    temporary_npz = array_path.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            neurons=neurons,
            worm_ids=worm_ids,
            source_lag_frames=selected_lags,
            source_lag_seconds=lag_seconds,
            horizon_frames=selected_horizons,
            horizon_seconds=horizon_seconds,
            orientation=np.asarray(ORIENTATION),
            channel=np.asarray(channel),
            context=np.asarray(output_context),
            primary_method=np.asarray(primary_method),
            support_eligible_source=selected_source_eligible,
            support_gate=np.asarray(support_gate),
            support_sensitivity_eligible_source=(
                support.sensitivity_source_eligible
            ),
            support_min_valid_fraction_all_contexts_lags=(
                support.minimum_valid_fraction
            ),
            support_min_genealogy_valid_fraction_0_10_all_contexts_lags=(
                support.minimum_genealogy_valid_fraction
            ),
            mean_normalized=_dense_edge_array(
                result.mean,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            standard_error=_dense_edge_array(
                result.se,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            student_t=_dense_edge_array(
                result.t_statistic,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            pointwise_sign_flip_p_value=_dense_edge_array(
                result.pointwise_p_value,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            global_max_t_p_value=_dense_edge_array(
                result.global_max_t_p_value,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            zero_null_reference_band_low=_dense_edge_array(
                result.zero_null_reference_band_low,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            zero_null_reference_band_high=_dense_edge_array(
                result.zero_null_reference_band_high,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            edge_omnibus_p_value=_dense_edge_array(
                result.edge_p_value, target_index, source_index, len(neurons)
            ),
            edge_bh_q_value=_dense_edge_array(
                result.edge_bh_q_value, target_index, source_index, len(neurons)
            ),
            edge_global_fwer_p_value=_dense_edge_array(
                result.edge_global_fwer_p_value,
                target_index,
                source_index,
                len(neurons),
            ),
            lag_contrast_mean=_dense_edge_array(
                result.lag_contrast_mean,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            lag_contrast_student_t=_dense_edge_array(
                result.lag_contrast_t,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            lag_contrast_global_max_t_p_value=_dense_edge_array(
                result.lag_contrast_global_max_t_p_value,
                target_index,
                source_index,
                len(neurons),
                trailing_shape=dense_shape,
            ),
            flat_lag_omnibus_p_value=_dense_edge_array(
                result.flat_lag_p_value,
                target_index,
                source_index,
                len(neurons),
            ),
            flat_lag_bh_q_value=_dense_edge_array(
                result.flat_lag_bh_q_value,
                target_index,
                source_index,
                len(neurons),
            ),
            flat_lag_global_fwer_p_value=_dense_edge_array(
                result.flat_lag_global_fwer_p_value,
                target_index,
                source_index,
                len(neurons),
            ),
            sign_patterns=result.signs,
            global_max_abs_t=result.global_max_abs_t,
            global_max_abs_lag_contrast_t=(
                result.global_max_abs_lag_contrast_t
            ),
            simultaneous_critical_value=np.asarray(
                result.simultaneous_critical_value
            ),
            lag_contrast_simultaneous_critical_value=np.asarray(
                result.lag_contrast_simultaneous_critical_value
            ),
        )
    temporary_npz.replace(array_path)

    smallest_possible_p = (
        1.0 / len(result.signs)
        if result.inference_mode == "exact"
        else 1.0 / (len(result.signs) + 1)
    )
    protocol = {
        "schema_version": "full-family-worm-signflip-v1",
        "family": {
            "channel": channel,
            "context": output_context,
            "primary_method": primary_method,
            "directed_edges": len(target_index),
            "support_eligible_directed_edges": int(edge_support_eligible.sum()),
            "support_sensitivity_eligible_directed_edges": int(
                edge_sensitivity_eligible.sum()
            ),
            "off_diagonal_rule": "target_index != source_index",
            "lag_frames": selected_lags.tolist(),
            "horizon_frames": selected_horizons.tolist(),
            "cells": int(len(cell_frame)),
            "support_eligible_cells": int(
                edge_support_eligible.sum()
                * len(selected_lags)
                * len(selected_horizons)
            ),
            "matrix_orientation": ORIENTATION,
        },
        "replication": {
            "independent_unit": "worm",
            "n_worms": n_worms,
            "worm_ids": worm_ids.tolist(),
            "particles_events_folds_and_model_seeds_are_not_replicates": True,
        },
        "randomization": {
            "null": "worm-level effect vectors are centered at zero and sign-symmetric",
            "shared_sign_rule": (
                "one Rademacher sign per worm per randomization, reused across "
                "every edge, lag, horizon, and lag contrast"
            ),
            "requested_mode": mode,
            "resolved_mode": result.inference_mode,
            "patterns": len(result.signs),
            "seed": seed,
            "two_sided_exact_orbits": result.inference_mode == "exact",
            "p_value_correction": (
                "exact count / exact unique two-sided sign orbits; observed orbit included"
                if result.inference_mode == "exact"
                else "(exceedances + 1) / (Monte Carlo replicates + 1)"
            ),
            "smallest_possible_p_value": smallest_possible_p,
            "monte_carlo_resolution_note": (
                "With sparse discoveries, a Monte Carlo floor larger than alpha / "
                "number_of_edges cannot resolve a best-rank BH discovery. Multiple "
                "ties at the floor can nevertheless pass BH at a later rank; exact "
                "enumeration avoids describing the limitation as an absolute impossibility."
            ),
            "sign_patterns_sha256": hashlib.sha256(result.signs.tobytes()).hexdigest(),
        },
        "edge_inference": {
            "statistic": (
                "max absolute worm-studentized mean over all selected lag x horizon cells"
            ),
            "raw_p_value": (
                "within-edge max statistic compared with the same edge maximum in "
                "each shared-sign null tensor"
            ),
            "multiplicity": (
                f"one Benjamini-Hochberg family across all {len(target_index)} "
                "directed off-diagonal edges; support-ineligible edges enter the "
                "adjustment as p=1 but retain raw p=NaN"
            ),
            "global_fwer": (
                "each edge maximum compared with the complete-family null maximum"
            ),
        },
        "cell_localization": {
            "procedure": (
                "complete-family max-T across every selected directed edge, lag, and horizon"
            ),
            "zero_null_reference_band": (
                f"descriptive mean +/- {100 * (1 - alpha):g}% zero-null max-T "
                "critical value times worm SE; this is not a confidence interval "
                "because the zero-null randomization was not inverted"
            ),
            "critical_value": result.simultaneous_critical_value,
            "alpha": alpha,
        },
        "flat_lag_inference": {
            "enabled": len(selected_lags) >= 2,
            "null": (
                "for each horizon, population mean effects are equal across all selected lags"
            ),
            "contrast": (
                "within each worm/edge/horizon, effect at a lag minus that worm's "
                "average effect across selected lags"
            ),
            "edge_statistic": "maximum absolute studentized lag contrast",
            "multiplicity": (
                f"one Benjamini-Hochberg family across all {len(target_index)} edges"
            ),
            "localization": (
                "complete-family max-T p-values for centered lag contrasts; the "
                "saved zero-null bands are descriptive, not confidence intervals"
            ),
            "interpretation": (
                "a rejection supports non-flat model coordinates, not a physical biological delay"
            ),
        },
        "claim_boundary": {
            "supported": (
                "complete-family-calibrated consistency of a frozen model-relative effect "
                "across the available worms"
            ),
            "not_supported": [
                "causality",
                "a physical propagation delay",
                "independent biological replication",
                "valid zero-null inference for unsigned within-state distances",
            ],
            "unsigned_override_used": not _is_signed_estimand(
                channel, output_context
            ),
        },
        "support_gate": {
            "applied": support.path is not None,
            "applied_gate": support_gate,
            "path": str(support.path) if support.path is not None else None,
            "sha256": (
                support_provenance["sha256"]
                if support_provenance is not None
                else None
            ),
            "parent_checksum_ledger": (
                support_provenance["parent_checksum_ledger"]
                if support_provenance is not None
                else None
            ),
            "method": primary_method,
            "join_keys": ["context", "source_index", "source_lag_frames"],
            "contexts_jointly_required": list(support.support_contexts),
            "all_selected_lags_required": True,
            "strong_rule": (
                "support_qualified AND valid_fraction >= strong_threshold AND "
                "genealogy_strong_gate_pass AND genealogy_valid_fraction_0_10 "
                ">= genealogy_threshold in every context and lag"
            ),
            "strong_threshold": support.strong_threshold,
            "genealogy_threshold": support.genealogy_threshold,
            "strong_eligible_sources": int(
                support.strong_source_eligible.sum()
            ),
            "sensitivity_rule": (
                "reported only, never used for primary inference: support_qualified "
                "AND valid_fraction and genealogy_valid_fraction_0_10 >= "
                "sensitivity_threshold in every context and lag"
            ),
            "sensitivity_threshold": support.sensitivity_threshold,
            "sensitivity_eligible_sources": int(
                support.sensitivity_source_eligible.sum()
            ),
            "inference_eligible_sources": int(selected_source_eligible.sum()),
        },
        "derived_contrast": {
            "name": contrast,
            "worm_level_operation": (
                "normalized active minus normalized baseline before inference"
                if contrast == "active-minus-baseline"
                else None
            ),
            "input_keys": list(input_keys),
            "active_horizon_guard": (
                "horizon frame 32 excluded by default because it crosses offset"
                if contrast == "active-minus-baseline"
                else None
            ),
        },
        "input": {
            "path": str(input_path),
            "sha256": input_provenance["sha256"],
            "parent_checksum_ledger": input_provenance[
                "parent_checksum_ledger"
            ],
            "keys": list(input_keys),
        },
    }
    _json_dump(output / "protocol.json", protocol)

    edge_discoveries = edge_frame.loc[edge_frame["edge_bh_reject"]]
    edge_fwer_discoveries = edge_frame.loc[edge_frame["edge_global_fwer_reject"]]
    flat_discoveries = edge_frame.loc[edge_frame["flat_lag_bh_reject"]]
    cell_discoveries = cell_frame.loc[cell_frame["global_max_t_reject"]]
    lag_cell_discoveries = cell_frame.loc[
        cell_frame["lag_contrast_global_max_t_reject"]
    ]
    top_edges = edge_frame.head(20).replace({np.nan: None}).to_dict(orient="records")
    summary = {
        "schema_version": "full-family-worm-signflip-summary-v1",
        "status": "complete",
        "channel": channel,
        "context": output_context,
        "primary_method": primary_method,
        "n_worms": n_worms,
        "n_directed_edges": len(edge_frame),
        "n_support_eligible_directed_edges": int(edge_support_eligible.sum()),
        "n_support_sensitivity_eligible_directed_edges": int(
            edge_sensitivity_eligible.sum()
        ),
        "n_cells": len(cell_frame),
        "n_support_eligible_cells": int(cell_frame["support_eligible"].sum()),
        "inference_mode": result.inference_mode,
        "sign_patterns": len(result.signs),
        "alpha": alpha,
        "edge_bh_discoveries": int(len(edge_discoveries)),
        "edge_global_fwer_discoveries": int(len(edge_fwer_discoveries)),
        "global_max_t_cell_discoveries": int(len(cell_discoveries)),
        "flat_lag_bh_discoveries": int(len(flat_discoveries)),
        "flat_lag_global_max_t_cell_discoveries": int(len(lag_cell_discoveries)),
        "simultaneous_critical_value": result.simultaneous_critical_value,
        "lag_contrast_simultaneous_critical_value": (
            result.lag_contrast_simultaneous_critical_value
        ),
        "top_edges": top_edges,
        "interpretation": (
            "exploratory frozen-model evidence with complete-family multiplicity control; "
            "not causal or independently confirmed"
        ),
    }
    _json_dump(output / "summary.json", summary)

    validation = {
        "schema_version": "full-family-worm-signflip-validation-v1",
        "status": "passed",
        "input": input_provenance,
        "support_table": support_provenance,
        "checks": {
            "input_sha256_pinned": bool(input_provenance.get("sha256")),
            "support_table_required": support.path is not None,
            "support_table_sha256_pinned": (
                support.path is None
                or bool(
                    support_provenance is not None
                    and support_provenance.get("sha256")
                )
            ),
            "support_parent_ledger_pinned_when_covering": True,
        },
    }
    _json_dump(output / "validation.json", validation)

    created_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "full-family-worm-signflip-manifest-v1",
        "status": "complete",
        "created_utc": created_utc,
        "input_npz": str(input_path),
        "output_dir": str(output),
        "inputs": {
            "worm_matrices": input_provenance,
            "support_table": support_provenance,
        },
        "artifacts": {
            "edge_inference": edge_path.name,
            "cell_inference": cell_path.name,
            "inference_arrays": array_path.name,
            "protocol": "protocol.json",
            "summary": "summary.json",
            "validation": "validation.json",
            "checksums": "checksums.sha256",
        },
    }
    _json_dump(output / "manifest.json", manifest)
    checksum_paths = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths)
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--context", default="baseline")
    parser.add_argument(
        "--contrast", choices=("active-minus-baseline",), default=None
    )
    parser.add_argument("--lags", nargs="+", type=int)
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--support-cells", type=Path)
    parser.add_argument("--support-threshold", type=float, default=0.8)
    parser.add_argument("--genealogy-threshold", type=float, default=0.8)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.5)
    parser.add_argument(
        "--support-gate", choices=("strong", "sensitivity"), default="strong"
    )
    parser.add_argument(
        "--mode", choices=("auto", "exact", "monte-carlo"), default="auto"
    )
    parser.add_argument(
        "--replicates", type=int, default=DEFAULT_MONTE_CARLO_REPLICATES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-exact-patterns", type=int, default=DEFAULT_MAX_EXACT_PATTERNS
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-worms", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--allow-unsigned", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_full_family_inference(
        args.input_npz,
        args.output_dir,
        channel=args.channel,
        context=args.context,
        contrast=args.contrast,
        lag_frames=args.lags,
        horizon_frames=args.horizons,
        support_cells=args.support_cells,
        support_threshold=args.support_threshold,
        genealogy_threshold=args.genealogy_threshold,
        sensitivity_threshold=args.sensitivity_threshold,
        support_gate=args.support_gate,
        mode=args.mode,
        replicates=args.replicates,
        seed=args.seed,
        max_exact_patterns=args.max_exact_patterns,
        batch_size=args.batch_size,
        min_worms=args.min_worms,
        alpha=args.alpha,
        allow_unsigned=args.allow_unsigned,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
