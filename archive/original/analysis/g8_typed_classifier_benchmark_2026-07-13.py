"""Direct finite-contrast density-ratio tests on the audited G8 law.

The conditional generators are trained to fit the entire predictive law.  This
developmental experiment instead gives the estimator the exact inferential
task: discriminate samples generated at control ``+delta`` from samples at
``-delta``.  With equal class priors, the fitted class probability estimates

    W_delta(y,h0) = (2 P(Z=+ | y,h0) - 1) / delta,

the bounded Radon--Nikodym witness of the central signed contrast relative to
the absolute-mixture reference law.  We evaluate both witness error and the
induced typed fourth-order effect E_M[phi W_delta].

Linear and quadratic feature maps are exact population negative controls for
G8 because the two conditional laws have identical means and covariances.
The quartic map is the analytically derived positive control.  Generic MLPs
test whether a flexible contrast-aligned learner can discover the distinction
without fitting the rest of the predictive law.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from history_tangent_benchmark.dgps import G8FunctionalShapeMixture
from history_tangent_benchmark.serialization import atomic_json


OUTPUT = Path("analysis/g8_typed_classifier_benchmark_2026-07-13")
READOUT = runpy.run_path("analysis/g8_corrected_readout_benchmark_2026-07-13.py")
quartic_feature = READOUT["_quartic_feature"]
motif_coordinates = READOUT["_motif_coordinates"]

GENERATOR_SEEDS = (73, 79, 83, 89)
DATA_SEEDS = (9101, 9109)
DELTA = 1.0
N_TRAIN_PER_SIDE = 4_000
N_VALIDATION_PER_SIDE = 1_000
N_TEST_PER_SIDE = 10_000
LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


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


def _paired_dataset(
    dgp: G8FunctionalShapeMixture, n_per_side: int, seed: int
) -> dict[str, np.ndarray]:
    h0 = dgp.sample_history(n_per_side, seed)
    h0[:, -1] = 0.0
    hp = h0.clone()
    hm = h0.clone()
    hp[:, -1] = DELTA
    hm[:, -1] = -DELTA
    yp = dgp.sample_response(hp, 1, seed + 1)[:, 0, :]
    ym = dgp.sample_response(hm, 1, seed + 2)[:, 0, :]
    y = torch.cat([yp, ym], dim=0)
    base_h = torch.cat([h0, h0], dim=0)
    labels = torch.cat(
        [
            torch.ones(n_per_side, dtype=torch.float64),
            torch.zeros(n_per_side, dtype=torch.float64),
        ]
    )
    # Deterministic shuffling prevents optimizers from seeing class blocks.
    generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    permutation = torch.randperm(2 * n_per_side, generator=generator)
    y = y[permutation]
    base_h = base_h[permutation]
    labels = labels[permutation]

    hp_eval = base_h.clone()
    hm_eval = base_h.clone()
    hp_eval[:, -1] = DELTA
    hm_eval[:, -1] = -DELTA
    log_plus = dgp.log_prob(y, hp_eval)
    log_minus = dgp.log_prob(y, hm_eval)
    oracle_probability = torch.sigmoid(log_plus - log_minus)
    u = motif_coordinates(dgp, y, base_h)
    phi = quartic_feature(dgp, y, base_h)
    full = torch.cat([base_h[:, :-1], y], dim=-1)
    return {
        "labels": labels.numpy(),
        "motif": u.numpy(),
        "quartic": phi[:, None].numpy(),
        "full": full.numpy(),
        "oracle_probability": oracle_probability.numpy(),
        "phi": phi.numpy(),
    }


def _fit_logistic(
    degree: int | None,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> Pipeline:
    best_model: Pipeline | None = None
    best_loss = math.inf
    for c_value in LOGISTIC_C_GRID:
        steps: list[tuple[str, object]] = []
        if degree is not None and degree > 1:
            steps.append(
                (
                    "polynomial",
                    PolynomialFeatures(degree=degree, include_bias=False),
                )
            )
        steps.extend(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=2_000,
                        solver="lbfgs",
                        random_state=0,
                    ),
                ),
            ]
        )
        model = Pipeline(steps)
        model.fit(train_x, train_y)
        probability = model.predict_proba(validation_x)[:, 1]
        loss = float(log_loss(validation_y, probability, labels=[0.0, 1.0]))
        if loss < best_loss:
            best_loss = loss
            best_model = model
    assert best_model is not None
    return best_model


class _MLP(nn.Module):
    def __init__(self, input_dim: int, width: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def _fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
) -> tuple[StandardScaler, _MLP, dict[str, float | int]]:
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train_x)
    x_train = torch.as_tensor(scaler.transform(train_x), dtype=torch.float32)
    y_train = torch.as_tensor(train_y, dtype=torch.float32)
    x_validation = torch.as_tensor(
        scaler.transform(validation_x), dtype=torch.float32
    )
    y_validation = torch.as_tensor(validation_y, dtype=torch.float32)
    model = _MLP(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 15
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, 121):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(batch_x), batch_y
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    model(x_validation), y_validation
                )
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return scaler, model, {
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "validation_log_loss": best_loss,
        "fit_seconds": time.perf_counter() - started,
    }


def _mlp_probability(scaler: StandardScaler, model: _MLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.as_tensor(scaler.transform(x), dtype=torch.float32)
        )
    return torch.sigmoid(logits).numpy()


def _truth(dgp: G8FunctionalShapeMixture) -> float:
    rank = dgp.motif_rank
    contrast_constant = rank**2 - rank * float(dgp.motif_rotation.pow(4).sum())
    hp = torch.zeros((1, dgp.q), dtype=torch.float64)
    hm = hp.clone()
    hp[:, -1] = DELTA
    hm[:, -1] = -DELTA
    probability_difference = float(
        dgp.shape_probability(hp) - dgp.shape_probability(hm)
    )
    return contrast_constant * probability_difference / DELTA


def _metrics(
    name: str,
    probability: np.ndarray,
    test: dict[str, np.ndarray],
    truth: float,
    fit: dict[str, object] | None = None,
) -> dict[str, object]:
    labels = test["labels"]
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    oracle_probability = np.asarray(test["oracle_probability"], dtype=float)
    witness = (2.0 * probability - 1.0) / DELTA
    oracle_witness = (2.0 * oracle_probability - 1.0) / DELTA
    values = np.asarray(test["phi"], dtype=float) * witness
    effect = float(np.mean(values))
    plus = labels == 1
    minus = ~plus
    effect_mc_se = float(
        0.5
        * np.sqrt(
            np.var(values[plus], ddof=1) / plus.sum()
            + np.var(values[minus], ddof=1) / minus.sum()
        )
    )
    oracle_rms = max(float(np.sqrt(np.mean(np.square(oracle_witness)))), 1e-12)
    result: dict[str, object] = {
        "estimator": name,
        "auc": float(roc_auc_score(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0.0, 1.0])),
        "brier": float(np.mean(np.square(probability - labels))),
        "witness_nrmse": float(
            np.sqrt(np.mean(np.square(witness - oracle_witness))) / oracle_rms
        ),
        "typed_effect": effect,
        "typed_effect_truth": truth,
        "typed_effect_relative_error": abs(effect - truth) / max(abs(truth), 1e-12),
        "typed_effect_mc_se": effect_mc_se,
        "typed_effect_mc_se_512_per_side": effect_mc_se
        * math.sqrt(float(plus.sum()) / 512.0),
        "typed_effect_error_in_mc_se": (effect - truth) / max(effect_mc_se, 1e-12),
        "witness_rms": float(np.sqrt(np.mean(np.square(witness)))),
    }
    if fit:
        result.update(fit)
    return result


def _run_case(generator_seed: int, data_seed: int) -> list[dict[str, object]]:
    dgp = _dgp(generator_seed)
    train = _paired_dataset(
        dgp, N_TRAIN_PER_SIDE, _stable_seed(generator_seed, data_seed, "train")
    )
    validation = _paired_dataset(
        dgp,
        N_VALIDATION_PER_SIDE,
        _stable_seed(generator_seed, data_seed, "validation"),
    )
    test = _paired_dataset(
        dgp, N_TEST_PER_SIDE, _stable_seed(generator_seed, data_seed, "test")
    )
    truth = _truth(dgp)
    rows = [_metrics("oracle_bayes_witness", test["oracle_probability"], test, truth)]

    specifications = (
        ("linear_motif", "motif", 1),
        ("quadratic_motif", "motif", 2),
        ("typed_quartic", "quartic", 1),
        ("polynomial4_motif", "motif", 4),
    )
    for name, feature, degree in specifications:
        started = time.perf_counter()
        model = _fit_logistic(
            degree,
            train[feature],
            train["labels"],
            validation[feature],
            validation["labels"],
        )
        probability = model.predict_proba(test[feature])[:, 1]
        rows.append(
            _metrics(
                name,
                probability,
                test,
                truth,
                {
                    "fit_seconds": time.perf_counter() - started,
                    "selected_C": float(model.named_steps["classifier"].C),
                },
            )
        )

    for name, feature in (
        ("mlp_motif", "motif"),
        ("mlp_full_path_history", "full"),
    ):
        scaler, model, fit = _fit_mlp(
            train[feature],
            train["labels"],
            validation[feature],
            validation["labels"],
            _stable_seed(generator_seed, data_seed, name),
        )
        rows.append(
            _metrics(
                name,
                _mlp_probability(scaler, model, test[feature]),
                test,
                truth,
                fit,
            )
        )

    # A transparent nonparametric finite-contrast baseline for the typed moment.
    plus = test["labels"] == 1
    minus = ~plus
    empirical_effect = float(
        (test["phi"][plus].mean() - test["phi"][minus].mean()) / (2 * DELTA)
    )
    empirical_mc_se = float(
        np.sqrt(
            np.var(test["phi"][plus], ddof=1) / plus.sum()
            + np.var(test["phi"][minus], ddof=1) / minus.sum()
        )
        / (2 * DELTA)
    )
    rows.append(
        {
            "estimator": "empirical_typed_difference",
            "typed_effect": empirical_effect,
            "typed_effect_truth": truth,
            "typed_effect_relative_error": abs(empirical_effect - truth)
            / max(abs(truth), 1e-12),
            "typed_effect_mc_se": empirical_mc_se,
            "typed_effect_mc_se_512_per_side": empirical_mc_se
            * math.sqrt(float(plus.sum()) / 512.0),
            "typed_effect_error_in_mc_se": (empirical_effect - truth)
            / max(empirical_mc_se, 1e-12),
        }
    )
    for row in rows:
        row.update({"generator_seed": generator_seed, "data_seed": data_seed})
    return rows


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case_dir = OUTPUT / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        (generator_seed, data_seed)
        for generator_seed in GENERATOR_SEEDS
        for data_seed in DATA_SEEDS
    ]
    for index, (generator_seed, data_seed) in enumerate(specifications, start=1):
        case_id = f"g{generator_seed}_d{data_seed}"
        path = case_dir / f"{case_id}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("schema_version") == "2":
                print(f"[{index}/{len(specifications)}] skip {case_id}", flush=True)
                continue
        print(f"[{index}/{len(specifications)}] {case_id}", flush=True)
        started = time.perf_counter()
        try:
            rows = _run_case(generator_seed, data_seed)
            payload = {
                "schema_version": "2",
                "status": "ok",
                "case_id": case_id,
                "wall_seconds": time.perf_counter() - started,
                "rows": rows,
            }
        except Exception as error:
            payload = {
                "schema_version": "2",
                "status": "failed",
                "case_id": case_id,
                "generator_seed": generator_seed,
                "data_seed": data_seed,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "wall_seconds": time.perf_counter() - started,
            }
        atomic_json(path, payload)

    payloads = [json.loads(path.read_text()) for path in sorted(case_dir.glob("*.json"))]
    rows = [row for payload in payloads if payload["status"] == "ok" for row in payload["rows"]]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "case_metrics.csv", index=False)
    summary = (
        frame.groupby("estimator", as_index=False, dropna=False)
        .agg(
            cases=("generator_seed", "size"),
            auc_median=("auc", "median"),
            log_loss_median=("log_loss", "median"),
            witness_nrmse_median=("witness_nrmse", "median"),
            typed_effect_relative_error_median=("typed_effect_relative_error", "median"),
            typed_effect_relative_error_max=("typed_effect_relative_error", "max"),
            typed_effect_mc_se_median=("typed_effect_mc_se", "median"),
            typed_effect_mc_se_512_per_side_median=("typed_effect_mc_se_512_per_side", "median"),
            fit_seconds_median=("fit_seconds", "median"),
        )
        .sort_values("typed_effect_relative_error_median")
    )
    summary.to_csv(OUTPUT / "estimator_summary.csv", index=False)
    manifest = {
        "schema_version": "2",
        "status": "complete" if all(p["status"] == "ok" for p in payloads) else "complete_with_failures",
        "expected_cases": len(specifications),
        "observed_cases": len(payloads),
        "successful_cases": sum(p["status"] == "ok" for p in payloads),
        "generator_seeds": list(GENERATOR_SEEDS),
        "data_seeds": list(DATA_SEEDS),
        "delta": DELTA,
        "n_train_per_side": N_TRAIN_PER_SIDE,
        "n_validation_per_side": N_VALIDATION_PER_SIDE,
        "n_test_per_side": N_TEST_PER_SIDE,
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
