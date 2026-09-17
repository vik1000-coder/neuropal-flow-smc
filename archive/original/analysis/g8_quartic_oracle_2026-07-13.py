"""Analytic and Monte Carlo validation of the G8 fourth-order typed readout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from history_tangent_benchmark.dgps import G8FunctionalShapeMixture
from history_tangent_benchmark.serialization import atomic_json


OUTPUT = Path("analysis/g8_quartic_oracle_2026-07-13")


def h4(value: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    return value.pow(4) - 6.0 * variance * value.square() + 3.0 * variance.square()


def quartic_feature(
    dgp: G8FunctionalShapeMixture, y: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    mean = dgp.path_mean(h)
    motif_coordinate = (y - mean) @ dgp.motifs / dgp.motif_scale
    motif_noise = (
        dgp.motifs.T @ dgp.base_covariance @ dgp.motifs
    ) / dgp.motif_scale**2
    rotated_coordinate = motif_coordinate @ dgp.motif_rotation
    rotated_noise = dgp.motif_rotation.T @ motif_noise @ dgp.motif_rotation
    original = h4(motif_coordinate, torch.diagonal(motif_noise)).sum(dim=-1)
    rotated = h4(rotated_coordinate, torch.diagonal(rotated_noise)).sum(dim=-1)
    return rotated - original


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    analytic_rows: list[dict[str, float | int]] = []
    mc_rows: list[dict[str, float | int]] = []
    for seed in (61, 67):
        dgp = G8FunctionalShapeMixture(
            seed=seed,
            q=17,
            dy=32,
            n_channels=4,
            motif_rank=4,
            motif_scale=0.45,
            noise_sd=0.08,
            correlated_noise_scale=0.06,
            shape_sensitivity=1.2,
        )
        k = dgp.motif_rank
        rotation_fourth_sum = float(dgp.motif_rotation.pow(4).sum())
        contrast_constant = float(k**2 - k * rotation_fourth_sum)
        for control in (-1.0, 0.0, 1.0):
            for delta in (0.05, 0.12, 0.25):
                p_plus = float(torch.sigmoid(torch.tensor(1.2 * (control + delta))))
                p_minus = float(torch.sigmoid(torch.tensor(1.2 * (control - delta))))
                analytic_rows.append(
                    {
                        "generator_seed": seed,
                        "control": control,
                        "delta": delta,
                        "rotation_fourth_sum": rotation_fourth_sum,
                        "quartic_contrast_constant": contrast_constant,
                        "p_plus": p_plus,
                        "p_minus": p_minus,
                        "analytic_central_quartic_contrast": (
                            contrast_constant * (p_plus - p_minus) / delta
                        ),
                    }
                )

        delta = 0.12
        h0 = dgp.sample_history(1, 90_000 + seed)
        h0[0, -1] = 0.0
        direction = dgp.mechanism_directions()["shape_only_control"]
        h_plus = h0 + delta * direction
        h_minus = h0 - delta * direction
        draws = 25_000
        y_plus = dgp.sample_response(h_plus, draws, 91_000 + seed)[0]
        y_minus = dgp.sample_response(h_minus, draws, 92_000 + seed)[0]
        phi_plus = quartic_feature(dgp, y_plus, h_plus.expand(draws, -1))
        phi_minus = quartic_feature(dgp, y_minus, h_minus.expand(draws, -1))
        estimate = float((phi_plus.mean() - phi_minus.mean()) / (2.0 * delta))
        standard_error = float(
            torch.sqrt(phi_plus.var(unbiased=True) / draws + phi_minus.var(unbiased=True) / draws)
            / (2.0 * delta)
        )
        p_plus = float(dgp.shape_probability(h_plus))
        p_minus = float(dgp.shape_probability(h_minus))
        truth = contrast_constant * (p_plus - p_minus) / delta

        # First and second motif moments are exact negative controls.
        u_plus = (y_plus - dgp.path_mean(h_plus)) @ dgp.motifs / dgp.motif_scale
        u_minus = (y_minus - dgp.path_mean(h_minus)) @ dgp.motifs / dgp.motif_scale
        linear_contrast = float(
            (u_plus[:, 0].mean() - u_minus[:, 0].mean()) / (2.0 * delta)
        )
        quadratic_contrast = float(
            (u_plus.square().sum(-1).mean() - u_minus.square().sum(-1).mean())
            / (2.0 * delta)
        )
        mc_rows.append(
            {
                "generator_seed": seed,
                "draws_per_side": draws,
                "delta": delta,
                "analytic_central_quartic_contrast": truth,
                "mc_central_quartic_contrast": estimate,
                "mc_standard_error": standard_error,
                "z_error": (estimate - truth) / standard_error,
                "linear_negative_control": linear_contrast,
                "quadratic_negative_control": quadratic_contrast,
            }
        )

    analytic = pd.DataFrame(analytic_rows)
    monte_carlo = pd.DataFrame(mc_rows)
    analytic.to_csv(OUTPUT / "analytic_effects.csv", index=False)
    monte_carlo.to_csv(OUTPUT / "monte_carlo_validation.csv", index=False)
    summary = {
        "schema_version": "1",
        "analytic_rows": len(analytic),
        "monte_carlo_rows": len(monte_carlo),
        "maximum_absolute_z_error": float(monte_carlo.z_error.abs().max()),
        "maximum_absolute_linear_negative_control": float(
            monte_carlo.linear_negative_control.abs().max()
        ),
        "maximum_absolute_quadratic_negative_control": float(
            monte_carlo.quadratic_negative_control.abs().max()
        ),
    }
    atomic_json(OUTPUT / "summary.json", summary)
    print(analytic.to_string(index=False))
    print(monte_carlo.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
