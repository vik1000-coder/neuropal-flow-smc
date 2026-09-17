"""Repair frozen G8 generators with a contrast-aligned classifier mass head.

For the symmetric controls ``-1`` and ``+1``, G8 at control zero is exactly
M=(P_-+P_+)/2.  If eta(y,h0)=P(Z=+|y,h0), then

    dP_+/dM = 2 eta,       dP_-/dM = 2 (1-eta).

This experiment freezes each existing generator, draws from its control-zero
law, and uses a separately trained finite classifier to estimate these bounded
weights.  Self-normalized typed moments test whether the classifier restores
the history-controlled component masses that full-law training ignored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import time

import numpy as np
import pandas as pd
import torch

from history_tangent_benchmark.dgps import G8FunctionalShapeMixture, make_dgp
from history_tangent_benchmark.finite_contrast_runner import _load_model
from history_tangent_benchmark.reporting import load_case_records
from history_tangent_benchmark.serialization import atomic_json


ROOT = Path("history_tangent_benchmark/results/stable_sid_core_20260713")
OUTPUT = Path("analysis/g8_classifier_guided_generator_2026-07-13")
CLASSIFIER_CODE = runpy.run_path("analysis/g8_typed_classifier_benchmark_2026-07-13.py")
READOUT_CODE = runpy.run_path("analysis/g8_corrected_readout_benchmark_2026-07-13.py")

paired_dataset = CLASSIFIER_CODE["_paired_dataset"]
fit_mlp = CLASSIFIER_CODE["_fit_mlp"]
mlp_probability = CLASSIFIER_CODE["_mlp_probability"]
quartic_feature = READOUT_CODE["_quartic_feature"]

DELTA = 1.0
N_TRAIN_PER_SIDE = 4_000
N_VALIDATION_PER_SIDE = 1_000
ANCHORS = 8
PROPOSALS = 1_024
RAW_DRAWS_PER_SIDE = 512
MODELS = {
    "autoregressive_mdn",
    "autoregressive_transformer",
    "affine_flow",
    "diffusion_edm",
    "flow_matching",
}


def _seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _truth(dgp: G8FunctionalShapeMixture) -> float:
    rank = dgp.motif_rank
    constant = rank**2 - rank * float(dgp.motif_rotation.pow(4).sum())
    h = torch.zeros((1, dgp.q), dtype=torch.float64)
    hp, hm = h.clone(), h.clone()
    hp[:, -1], hm[:, -1] = DELTA, -DELTA
    return constant * float(dgp.shape_probability(hp) - dgp.shape_probability(hm)) / DELTA


def _classifier_features(history: torch.Tensor, response: torch.Tensor) -> np.ndarray:
    if response.ndim != 3 or len(history) != len(response):
        raise ValueError("need history [a,q] and response [a,n,dy]")
    repeated = history[:, None, :-1].expand(len(history), response.shape[1], history.shape[1] - 1)
    return torch.cat([repeated, response], dim=-1).reshape(-1, repeated.shape[-1] + response.shape[-1]).numpy()


def _typed_effects(
    dgp: G8FunctionalShapeMixture,
    anchors: torch.Tensor,
    proposal: torch.Tensor,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability_t = torch.as_tensor(probability, dtype=torch.float64).reshape(
        len(anchors), proposal.shape[1]
    ).clamp(1e-5, 1 - 1e-5)
    effects = []
    ess_plus = []
    ess_minus = []
    for index, h0 in enumerate(anchors):
        h = h0[None, :].expand(proposal.shape[1], -1)
        phi = quartic_feature(dgp, proposal[index], h)
        wp = probability_t[index]
        wm = 1.0 - probability_t[index]
        mean_plus = torch.sum(wp * phi) / wp.sum().clamp_min(1e-12)
        mean_minus = torch.sum(wm * phi) / wm.sum().clamp_min(1e-12)
        effects.append(float((mean_plus - mean_minus) / (2 * DELTA)))
        ess_plus.append(float(wp.sum().square() / wp.square().sum().clamp_min(1e-12)))
        ess_minus.append(float(wm.sum().square() / wm.square().sum().clamp_min(1e-12)))
    return np.asarray(effects), np.asarray(ess_plus), np.asarray(ess_minus)


def _raw_effects(
    dgp: G8FunctionalShapeMixture,
    anchors: torch.Tensor,
    sampler,
    seed: int,
) -> np.ndarray:
    hp, hm = anchors.clone(), anchors.clone()
    hp[:, -1], hm[:, -1] = DELTA, -DELTA
    yp = sampler(hp, RAW_DRAWS_PER_SIDE, seed)
    ym = sampler(hm, RAW_DRAWS_PER_SIDE, seed + 1)
    effects = []
    for index in range(len(anchors)):
        hpi = hp[index][None, :].expand(RAW_DRAWS_PER_SIDE, -1)
        hmi = hm[index][None, :].expand(RAW_DRAWS_PER_SIDE, -1)
        pp = quartic_feature(dgp, yp[index], hpi)
        pm = quartic_feature(dgp, ym[index], hmi)
        effects.append(float((pp.mean() - pm.mean()) / (2 * DELTA)))
    return np.asarray(effects)


def _metrics(estimate: np.ndarray, truth: float, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_quartic_nrmse": float(
            np.sqrt(np.mean(np.square(estimate - truth)) / max(truth**2, 1e-12))
        ),
        f"{prefix}_quartic_slope": float(np.mean(estimate) / max(truth, 1e-12)),
        f"{prefix}_quartic_bias": float(np.mean(estimate - truth)),
        f"{prefix}_quartic_sd_across_anchors": float(np.std(estimate, ddof=1)),
    }


def _train_classifier(dgp: G8FunctionalShapeMixture, generator_seed: int, data_seed: int):
    train = paired_dataset(
        dgp,
        N_TRAIN_PER_SIDE,
        _seed(generator_seed, data_seed, "guided_classifier_train"),
    )
    validation = paired_dataset(
        dgp,
        N_VALIDATION_PER_SIDE,
        _seed(generator_seed, data_seed, "guided_classifier_validation"),
    )
    return fit_mlp(
        train["full"],
        train["labels"],
        validation["full"],
        validation["labels"],
        _seed(generator_seed, data_seed, "guided_classifier_fit"),
    )


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case_dir = OUTPUT / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    records = [
        record
        for record in load_case_records(ROOT)
        if record["status"] == "ok"
        and record["generator"]["id"] == "g8_functional_shape"
        and record["model"]["name"] in MODELS
    ]
    records.sort(
        key=lambda record: (
            record["generator"]["seed"],
            record["data_seed"],
            record["model"]["name"],
            record["model"]["seed"],
        )
    )
    classifiers: dict[tuple[int, int], tuple[object, object, dict]] = {}
    oracle_done: set[tuple[int, int]] = set()
    for index, record in enumerate(records, start=1):
        generator_seed = int(record["generator"]["seed"])
        data_seed = int(record["data_seed"])
        key = (generator_seed, data_seed)
        dgp = make_dgp(
            record["generator"]["kind"],
            seed=generator_seed,
            **dict(record["generator"]["params"]),
        )
        if not isinstance(dgp, G8FunctionalShapeMixture):
            raise TypeError("expected G8")
        if key not in classifiers:
            classifiers[key] = _train_classifier(dgp, *key)
        classifier_scaler, classifier, classifier_fit = classifiers[key]
        anchors = dgp.sample_history(ANCHORS, _seed(*key, "guided_anchors"))
        anchors[:, -1] = 0.0
        truth = _truth(dgp)

        if key not in oracle_done:
            oracle_id = f"oracle_g{generator_seed}_d{data_seed}"
            oracle_path = case_dir / f"{oracle_id}.json"
            if not oracle_path.exists():
                proposal = dgp.sample_response(
                    anchors, PROPOSALS, _seed(*key, "oracle_proposal")
                )
                learned_probability = mlp_probability(
                    classifier_scaler,
                    classifier,
                    _classifier_features(anchors, proposal),
                )
                hp = anchors[:, None, :].expand(-1, PROPOSALS, -1).clone()
                hm = hp.clone()
                hp[..., -1], hm[..., -1] = DELTA, -DELTA
                log_plus = dgp.log_prob(proposal, hp)
                log_minus = dgp.log_prob(proposal, hm)
                bayes_probability = torch.sigmoid(log_plus - log_minus).numpy().reshape(-1)
                learned_effect, lep, lem = _typed_effects(
                    dgp, anchors, proposal, learned_probability
                )
                bayes_effect, bep, bem = _typed_effects(
                    dgp, anchors, proposal, bayes_probability
                )
                atomic_json(
                    oracle_path,
                    {
                        "status": "ok",
                        "case_id": oracle_id,
                        "model_name": "oracle_midpoint_law",
                        "generator_seed": generator_seed,
                        "data_seed": data_seed,
                        "truth": truth,
                        **_metrics(learned_effect, truth, "guided"),
                        **_metrics(bayes_effect, truth, "bayes_guided"),
                        "guided_ess_fraction_median": float(
                            np.median(np.concatenate([lep, lem])) / PROPOSALS
                        ),
                        "bayes_guided_ess_fraction_median": float(
                            np.median(np.concatenate([bep, bem])) / PROPOSALS
                        ),
                        "classifier_fit": classifier_fit,
                    },
                )
            oracle_done.add(key)

        output_path = case_dir / f"{record['case_id']}.json"
        if output_path.exists():
            print(f"[{index}/{len(records)}] skip {record['model']['name']}", flush=True)
            continue
        print(f"[{index}/{len(records)}] {key} {record['model']['name']}", flush=True)
        started = time.perf_counter()
        try:
            model, scaler = _load_model(ROOT, record, dgp)

            def sampler(h: torch.Tensor, n: int, sample_seed: int) -> torch.Tensor:
                hs = torch.as_tensor(
                    np.asarray(scaler.transform_history(h), dtype=np.float32)
                )
                with torch.no_grad():
                    ys = model.sample(hs, n, seed=sample_seed)
                return torch.as_tensor(
                    np.asarray(scaler.inverse_response(ys), dtype=np.float64)
                )

            proposal = sampler(
                anchors,
                PROPOSALS,
                _seed(record["case_id"], "model_midpoint_proposal"),
            )
            probability = mlp_probability(
                classifier_scaler,
                classifier,
                _classifier_features(anchors, proposal),
            )
            guided_effect, ess_plus, ess_minus = _typed_effects(
                dgp, anchors, proposal, probability
            )
            raw_effect = _raw_effects(
                dgp,
                anchors,
                sampler,
                _seed(record["case_id"], "model_raw_samples"),
            )
            result = {
                "status": "ok",
                **_metrics(raw_effect, truth, "raw"),
                **_metrics(guided_effect, truth, "guided"),
                "guided_ess_fraction_median": float(
                    np.median(np.concatenate([ess_plus, ess_minus])) / PROPOSALS
                ),
                "classifier_probability_mean": float(np.mean(probability)),
                "classifier_probability_saturation": float(
                    np.mean((probability < 0.01) | (probability > 0.99))
                ),
            }
        except Exception as error:
            result = {
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        atomic_json(
            output_path,
            {
                "case_id": record["case_id"],
                "model_name": record["model"]["name"],
                "generator_seed": generator_seed,
                "data_seed": data_seed,
                "model_seed": record["model"]["seed"],
                "truth": truth,
                "wall_seconds": time.perf_counter() - started,
                **result,
            },
        )

    rows = [json.loads(path.read_text()) for path in sorted(case_dir.glob("*.json"))]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    ok = frame.loc[frame.status.eq("ok")].copy()
    summary = (
        ok.groupby("model_name", as_index=False)
        .agg(
            cases=("case_id", "size"),
            raw_nrmse_median=("raw_quartic_nrmse", "median"),
            raw_slope_median=("raw_quartic_slope", "median"),
            guided_nrmse_median=("guided_quartic_nrmse", "median"),
            guided_nrmse_q25=("guided_quartic_nrmse", lambda x: x.quantile(0.25)),
            guided_nrmse_q75=("guided_quartic_nrmse", lambda x: x.quantile(0.75)),
            guided_slope_median=("guided_quartic_slope", "median"),
            guided_ess_fraction_median=("guided_ess_fraction_median", "median"),
            wall_seconds_median=("wall_seconds", "median"),
        )
        .sort_values("guided_nrmse_median")
    )
    summary.to_csv(OUTPUT / "model_summary.csv", index=False)
    manifest = {
        "schema_version": "1",
        "status": "complete" if frame.status.eq("ok").all() else "complete_with_failures",
        "expected_model_records": len(records),
        "observed_records": len(frame),
        "successful": int(frame.status.eq("ok").sum()),
        "delta": DELTA,
        "anchors": ANCHORS,
        "proposals_per_anchor": PROPOSALS,
        "raw_draws_per_side": RAW_DRAWS_PER_SIDE,
        "classifier_train_per_side": N_TRAIN_PER_SIDE,
        "classifier_validation_per_side": N_VALIDATION_PER_SIDE,
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
