"""Oracle audit of mode-weight visibility to DSM response scores on G8.

For a mixture whose well-separated components do not change but whose frame
weights do, likelihood ratios and history tangents remain informative while
the outcome-score field can be nearly invariant.  This script quantifies that
gap as the observation-noise scale changes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from history_tangent_benchmark.dgps import G8FunctionalShapeMixture
from history_tangent_benchmark.serialization import atomic_json


OUTPUT = Path("analysis/g8_score_visibility_audit_2026-07-13")
GENERATOR_SEEDS = (73, 79, 83, 89)
SIGMAS = (0.0, 0.02, 0.05, 0.08, 0.12, 0.20, 0.35, 0.60, 1.0)
CONTROL_DELTA = 1.0
N_PER_SIDE = 5_000
CHUNK = 500


def _dgp(seed: int) -> G8FunctionalShapeMixture:
    return G8FunctionalShapeMixture(
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


def _evaluate(seed: int, sigma: float) -> dict[str, float | int]:
    dgp = _dgp(seed)
    h0 = dgp.sample_history(N_PER_SIDE, 100_000 + seed)
    h0[:, -1] = 0.0
    hp = h0.clone()
    hm = h0.clone()
    hp[:, -1] = CONTROL_DELTA
    hm[:, -1] = -CONTROL_DELTA
    yp = dgp.sample_response(hp, 1, 200_000 + seed)[:, 0, :]
    ym = dgp.sample_response(hm, 1, 300_000 + seed)[:, 0, :]
    y = torch.cat([yp, ym], dim=0)
    base = torch.cat([h0, h0], dim=0)
    labels = torch.cat(
        [torch.ones(N_PER_SIDE, dtype=torch.float64), torch.zeros(N_PER_SIDE, dtype=torch.float64)]
    )
    if sigma:
        generator = torch.Generator(device="cpu").manual_seed(400_000 + seed + int(10_000 * sigma))
        y = y + sigma * torch.randn(y.shape, generator=generator, dtype=y.dtype)

    log_plus_parts: list[torch.Tensor] = []
    log_minus_parts: list[torch.Tensor] = []
    score_gap_squared_parts: list[torch.Tensor] = []
    score_scale_squared_parts: list[torch.Tensor] = []
    tangent_squared_parts: list[torch.Tensor] = []
    tangent_absolute_parts: list[torch.Tensor] = []
    for start in range(0, len(y), CHUNK):
        stop = min(start + CHUNK, len(y))
        yc = y[start:stop]
        hc = base[start:stop]
        hpc = hc.clone()
        hmc = hc.clone()
        hpc[:, -1] = CONTROL_DELTA
        hmc[:, -1] = -CONTROL_DELTA
        if sigma:
            lp = dgp.noisy_log_prob(yc, hpc, sigma)
            lm = dgp.noisy_log_prob(yc, hmc, sigma)
            sp = dgp.noisy_response_score(yc, hpc, sigma)
            sm = dgp.noisy_response_score(yc, hmc, sigma)
            tangent = dgp.noisy_history_tangent(yc, hc, sigma)[:, -1]
        else:
            lp = dgp.log_prob(yc, hpc)
            lm = dgp.log_prob(yc, hmc)
            sp = dgp.response_score(yc, hpc)
            sm = dgp.response_score(yc, hmc)
            tangent = dgp.history_tangent(yc, hc)[:, -1]
        log_plus_parts.append(lp)
        log_minus_parts.append(lm)
        score_gap_squared_parts.append((sp - sm).square().sum(dim=-1))
        score_scale_squared_parts.append(0.5 * (sp.square().sum(dim=-1) + sm.square().sum(dim=-1)))
        tangent_squared_parts.append(tangent.square())
        tangent_absolute_parts.append(tangent.abs())

    log_plus = torch.cat(log_plus_parts)
    log_minus = torch.cat(log_minus_parts)
    score_gap_squared = torch.cat(score_gap_squared_parts)
    score_scale_squared = torch.cat(score_scale_squared_parts)
    tangent_squared = torch.cat(tangent_squared_parts)
    tangent_absolute = torch.cat(tangent_absolute_parts)
    log_mixture = torch.logaddexp(log_plus, log_minus) - math.log(2.0)
    js_terms = torch.where(labels.bool(), log_plus - log_mixture, log_minus - log_mixture)
    posterior_plus = torch.sigmoid(log_plus - log_minus)
    witness = (2.0 * posterior_plus - 1.0) / CONTROL_DELTA
    auc = roc_auc_score(labels.numpy(), posterior_plus.numpy())
    return {
        "generator_seed": seed,
        "sigma": sigma,
        "n_per_side": N_PER_SIDE,
        "jensen_shannon_nats": float(js_terms.mean()),
        "bayes_auc": float(auc),
        "witness_rms": float(torch.sqrt(witness.square().mean())),
        "response_score_gap_rms": float(torch.sqrt(score_gap_squared.mean())),
        "response_score_rms": float(torch.sqrt(score_scale_squared.mean())),
        "relative_response_score_gap": float(
            torch.sqrt(score_gap_squared.mean() / score_scale_squared.mean().clamp_min(1e-24))
        ),
        "history_tangent_rms": float(torch.sqrt(tangent_squared.mean())),
        "history_tangent_abs_mean": float(tangent_absolute.mean()),
    }


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case_dir = OUTPUT / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    expected = len(GENERATOR_SEEDS) * len(SIGMAS)
    index = 0
    for seed in GENERATOR_SEEDS:
        for sigma in SIGMAS:
            index += 1
            case_id = f"g{seed}_s{sigma:.3f}".replace(".", "p")
            path = case_dir / f"{case_id}.json"
            if path.exists():
                print(f"[{index}/{expected}] skip seed={seed} sigma={sigma}", flush=True)
                continue
            print(f"[{index}/{expected}] seed={seed} sigma={sigma}", flush=True)
            try:
                result: dict[str, object] = {"status": "ok", "case_id": case_id, **_evaluate(seed, sigma)}
            except Exception as error:
                result = {
                    "status": "failed",
                    "case_id": case_id,
                    "generator_seed": seed,
                    "sigma": sigma,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                }
            atomic_json(path, result)

    rows = [json.loads(path.read_text()) for path in sorted(case_dir.glob("*.json"))]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    ok = frame.loc[frame.status.eq("ok")].copy()
    summary = (
        ok.groupby("sigma", as_index=False)
        .agg(
            cases=("case_id", "size"),
            jensen_shannon_nats_median=("jensen_shannon_nats", "median"),
            bayes_auc_median=("bayes_auc", "median"),
            witness_rms_median=("witness_rms", "median"),
            response_score_gap_rms_median=("response_score_gap_rms", "median"),
            relative_response_score_gap_median=("relative_response_score_gap", "median"),
            history_tangent_rms_median=("history_tangent_rms", "median"),
        )
        .sort_values("sigma")
    )
    summary.to_csv(OUTPUT / "noise_summary.csv", index=False)
    manifest = {
        "schema_version": "1",
        "status": "complete" if frame.status.eq("ok").all() else "complete_with_failures",
        "expected": expected,
        "observed": len(frame),
        "successful": int(frame.status.eq("ok").sum()),
        "generator_seeds": list(GENERATOR_SEEDS),
        "sigmas": list(SIGMAS),
        "control_delta": CONTROL_DELTA,
        "n_per_side": N_PER_SIDE,
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
