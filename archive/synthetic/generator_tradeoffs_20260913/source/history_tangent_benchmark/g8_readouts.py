"""Audited higher-shape readouts for the G8 functional-path benchmark.

G8 changes only the mixing weight between two signed, rotated motif frames.
Those frames have identical first and second moments, so linear and quadratic
statistics are population nulls.  Their fourth moments differ.  This module
implements the corresponding noise-corrected fourth-Hermite statistic and its
closed-form population contrast.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from history_tangent_benchmark.dgps import (
    DTYPE,
    G8FunctionalShapeMixture,
    G8GaussianShadow,
    _broadcast_y_h,
)


@dataclass(frozen=True)
class G8GeometryDiagnostics:
    """Numerical invariants that make the G8 quartic target identifiable."""

    motif_rank: int
    motif_matrix_rank: int
    rotation_matrix_rank: int
    motif_orthogonality_error: float
    rotation_orthogonality_error: float
    frame_mean_error: float
    frame_second_moment_error: float
    rotation_fourth_sum: float
    quartic_separation: float


def _require_g8(dgp: Any) -> G8FunctionalShapeMixture:
    if not isinstance(dgp, G8FunctionalShapeMixture):
        raise TypeError("dgp must be a G8FunctionalShapeMixture or G8GaussianShadow")
    return dgp


def _sigma_vector(sigma: float | torch.Tensor, dy: int) -> torch.Tensor:
    value = torch.as_tensor(sigma, dtype=DTYPE, device="cpu")
    if value.ndim == 0:
        value = value.expand(dy)
    elif value.ndim != 1 or value.numel() != dy:
        raise ValueError(f"sigma must be scalar or length dy={dy}")
    if not bool(torch.all(torch.isfinite(value))) or bool(torch.any(value < 0)):
        raise ValueError("sigma must be finite and non-negative")
    return value


def g8_geometry_diagnostics(dgp: G8FunctionalShapeMixture) -> G8GeometryDiagnostics:
    """Return rank, orthogonality, moment-match, and quartic-separation checks."""

    dgp = _require_g8(dgp)
    rank = dgp.motif_rank
    identity = torch.eye(rank, dtype=DTYPE)
    frame_a = dgp.component_offsets[: dgp.components_per_frame]
    frame_b = dgp.component_offsets[dgp.components_per_frame :]
    second_a = frame_a.T @ frame_a / dgp.components_per_frame
    second_b = frame_b.T @ frame_b / dgp.components_per_frame
    fourth_sum = float(dgp.motif_rotation.pow(4).sum())
    separation = float(rank**2 - rank * fourth_sum)
    return G8GeometryDiagnostics(
        motif_rank=rank,
        motif_matrix_rank=int(torch.linalg.matrix_rank(dgp.motifs).item()),
        rotation_matrix_rank=int(torch.linalg.matrix_rank(dgp.motif_rotation).item()),
        motif_orthogonality_error=float(
            torch.max(torch.abs(dgp.motifs.T @ dgp.motifs - identity))
        ),
        rotation_orthogonality_error=float(
            torch.max(
                torch.abs(dgp.motif_rotation.T @ dgp.motif_rotation - identity)
            )
        ),
        frame_mean_error=float(torch.max(torch.abs(frame_a.mean(0) - frame_b.mean(0)))),
        frame_second_moment_error=float(torch.max(torch.abs(second_a - second_b))),
        rotation_fourth_sum=fourth_sum,
        quartic_separation=separation,
    )


def validate_g8_geometry(
    dgp: G8FunctionalShapeMixture,
    *,
    tolerance: float = 1e-10,
    minimum_quartic_separation: float = 1e-8,
) -> G8GeometryDiagnostics:
    """Fail closed when a G8 construction is rank-deficient or readout-blind."""

    if tolerance <= 0 or minimum_quartic_separation < 0:
        raise ValueError("tolerances must be positive")
    diagnostics = g8_geometry_diagnostics(dgp)
    if diagnostics.motif_matrix_rank != diagnostics.motif_rank:
        raise ValueError("G8 motif matrix is rank deficient")
    if diagnostics.rotation_matrix_rank != diagnostics.motif_rank:
        raise ValueError("G8 motif rotation is rank deficient")
    if diagnostics.motif_orthogonality_error > tolerance:
        raise ValueError("G8 motif matrix is not orthonormal")
    if diagnostics.rotation_orthogonality_error > tolerance:
        raise ValueError("G8 motif rotation is not orthogonal")
    if diagnostics.frame_mean_error > tolerance:
        raise ValueError("G8 signed frames do not have the same mean")
    if diagnostics.frame_second_moment_error > tolerance:
        raise ValueError("G8 signed frames do not have the same second moment")
    if diagnostics.quartic_separation <= minimum_quartic_separation:
        raise ValueError("G8 motif rotation is quartically degenerate")
    return diagnostics


def hermite_fourth(value: torch.Tensor, variance: torch.Tensor | float) -> torch.Tensor:
    """Fourth Gaussian-noise-corrected polynomial.

    If ``X = mu + N(0, variance)``, then
    ``E[hermite_fourth(X, variance)] = mu**4``.
    """

    value = torch.as_tensor(value, dtype=DTYPE, device="cpu")
    variance = torch.as_tensor(variance, dtype=DTYPE, device="cpu")
    if not bool(torch.all(torch.isfinite(variance))) or bool(torch.any(variance < 0)):
        raise ValueError("variance must be finite and non-negative")
    return value.pow(4) - 6.0 * variance * value.square() + 3.0 * variance.square()


def g8_quartic_separation(dgp: G8FunctionalShapeMixture) -> float:
    """Return ``D = k^2 - k * sum_ij R_ij^4``, positive when frames differ."""

    return validate_g8_geometry(dgp).quartic_separation


def g8_frame_quartic_expectations(
    dgp: G8FunctionalShapeMixture,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact ``E[Q_B-Q_A | frame A]`` and its frame-B counterpart."""

    dgp = _require_g8(dgp)
    validate_g8_geometry(dgp)
    codes_a = dgp.latent_codes[: dgp.components_per_frame]
    codes_b = dgp.latent_codes[dgp.components_per_frame :]

    def frame_value(codes: torch.Tensor) -> torch.Tensor:
        q_a = codes.pow(4).sum(dim=-1)
        q_b = (codes @ dgp.motif_rotation).pow(4).sum(dim=-1)
        return (q_b - q_a).mean()

    return frame_value(codes_a), frame_value(codes_b)


def g8_projected_noise_covariance(
    dgp: G8FunctionalShapeMixture,
    sigma: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Noise covariance in motif coordinates after dividing by motif scale."""

    dgp = _require_g8(dgp)
    sigma_vector = _sigma_vector(sigma, dgp.dy)
    response_noise = dgp.base_covariance + torch.diag(sigma_vector.square())
    return (
        dgp.motifs.T @ response_noise @ dgp.motifs / dgp.motif_scale**2
    )


def g8_quartic_frame_scores(
    dgp: G8FunctionalShapeMixture,
    y: torch.Tensor,
    h: torch.Tensor,
    *,
    sigma: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute noise-corrected fourth-order scores in frames A and B.

    Returns ``(Q_A, Q_B)`` with all leading ``y``/``h`` broadcast dimensions
    preserved.  ``sigma`` is optional independent response-noise standard
    deviation, matching :meth:`ConditionalDGP.noisy_log_prob`.
    """

    dgp = _require_g8(dgp)
    validate_g8_geometry(dgp)
    y_aligned, h_aligned = _broadcast_y_h(y, h, dgp.dy, dgp.q)
    residual = y_aligned - dgp.path_mean(h_aligned)
    coordinates_a = (residual @ dgp.motifs) / dgp.motif_scale
    noise_a = g8_projected_noise_covariance(dgp, sigma)
    variance_a = torch.diagonal(noise_a)
    q_a = hermite_fourth(coordinates_a, variance_a).sum(dim=-1)

    coordinates_b = coordinates_a @ dgp.motif_rotation
    noise_b = dgp.motif_rotation.T @ noise_a @ dgp.motif_rotation
    variance_b = torch.diagonal(noise_b)
    q_b = hermite_fourth(coordinates_b, variance_b).sum(dim=-1)
    return q_a, q_b


def g8_quartic_readout(
    dgp: G8FunctionalShapeMixture,
    y: torch.Tensor,
    h: torch.Tensor,
    *,
    sigma: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Typed G8 readout ``phi(y,h) = Q_B(y,h) - Q_A(y,h)``."""

    q_a, q_b = g8_quartic_frame_scores(dgp, y, h, sigma=sigma)
    return q_b - q_a


def g8_quartic_expected_readout(
    dgp: G8FunctionalShapeMixture, h: torch.Tensor
) -> torch.Tensor:
    """Closed-form conditional expectation of :func:`g8_quartic_readout`.

    For G8 this is ``D * (2 p_B(h) - 1)``.  For the matched Gaussian shadow it
    is exactly zero because the shape control is a population null.
    """

    dgp = _require_g8(dgp)
    history = torch.as_tensor(h, dtype=DTYPE, device="cpu")
    if history.ndim < 1 or history.shape[-1] != dgp.q:
        raise ValueError(f"h must end in q={dgp.q}")
    if isinstance(dgp, G8GaussianShadow):
        return torch.zeros(history.shape[:-1], dtype=DTYPE)
    separation = g8_quartic_separation(dgp)
    return separation * (2.0 * dgp.shape_probability(history) - 1.0)


def g8_quartic_central_effect(
    dgp: G8FunctionalShapeMixture,
    h: torch.Tensor,
    direction: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Exact central finite effect of the typed quartic readout.

    With the declared unit control direction this equals
    ``D * (p_B(h+delta*v) - p_B(h-delta*v)) / delta``.
    """

    if not math.isfinite(float(delta)) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    dgp = _require_g8(dgp)
    history = torch.as_tensor(h, dtype=DTYPE, device="cpu")
    direction = torch.as_tensor(direction, dtype=DTYPE, device="cpu")
    if history.ndim < 1 or history.shape[-1] != dgp.q:
        raise ValueError(f"h must end in q={dgp.q}")
    if direction.ndim < 1 or direction.shape[-1] != dgp.q:
        raise ValueError(f"direction must end in q={dgp.q}")
    plus = g8_quartic_expected_readout(dgp, history + delta * direction)
    minus = g8_quartic_expected_readout(dgp, history - delta * direction)
    return (plus - minus) / (2.0 * delta)


__all__ = [
    "G8GeometryDiagnostics",
    "g8_frame_quartic_expectations",
    "g8_geometry_diagnostics",
    "g8_projected_noise_covariance",
    "g8_quartic_central_effect",
    "g8_quartic_expected_readout",
    "g8_quartic_frame_scores",
    "g8_quartic_readout",
    "g8_quartic_separation",
    "hermite_fourth",
    "validate_g8_geometry",
]
