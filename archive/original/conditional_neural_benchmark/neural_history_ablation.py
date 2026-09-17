from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditional_neural_benchmark.data import FoldScaler, make_windows
from conditional_neural_benchmark.distributional_lag_tournament import cohort_folds, load_cohorts
from conditional_neural_benchmark.neural_functional_effects import _choose_parametric, _load_model
from conditional_neural_benchmark.runner import _split_indices


def _log_prob(model, history: np.ndarray, target: np.ndarray, *, device: str, chunk: int = 512) -> np.ndarray:
    values: list[np.ndarray] = []
    target_device = torch.device(device)
    with torch.no_grad():
        for start in range(0, len(history), chunk):
            h = torch.as_tensor(
                history[start : start + chunk].reshape(len(history[start : start + chunk]), -1),
                dtype=torch.float32,
                device=target_device,
            )
            y = torch.as_tensor(
                target[start : start + chunk], dtype=torch.float32, device=target_device,
            )
            values.append(model.log_prob(y, h).detach().cpu().numpy())
    return np.concatenate(values)


def _source_roll(
    history: np.ndarray,
    worm: np.ndarray,
    source: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = history.copy()
    for value in np.unique(worm):
        index = np.flatnonzero(worm == value)
        if len(index) < 2:
            continue
        low = max(1, len(index) // 5)
        high = max(low + 1, len(index) - low)
        shift = int(rng.integers(low, high))
        result[index, :, source] = np.roll(history[index, :, source], shift, axis=0)
    return result


def run(
    neural_root: Path,
    evidence: Path,
    output: Path,
    *,
    device: str,
    rows: int,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    for cohort_name, cohort in load_cohorts().items():
        run_dir = neural_root / cohort_name
        model_id = _choose_parametric(run_dir)
        metrics = pd.read_csv(run_dir / "trial_metrics.csv")
        folds = cohort_folds(cohort_name, cohort, evidence)
        for fold in (3, 4):
            hit = metrics[
                (metrics.phase == "confirmation")
                & (metrics.model_id == model_id)
                & (metrics.fold == fold)
                & (metrics.seed == 1701)
                & (metrics.status == "ok")
            ]
            if len(hit) != 1:
                raise RuntimeError(f"missing confirmation checkpoint {cohort_name}/{model_id}/f{fold}")
            payload, model = _load_model(run_dir / str(hit.iloc[0].checkpoint), device)
            scaler = FoldScaler(
                mean=np.asarray(payload["scaler"]["mean"], dtype=np.float32),
                scale=np.asarray(payload["scaler"]["scale"], dtype=np.float32),
            )
            _, _, test_index = _split_indices(folds, fold)
            windows = make_windows(cohort, test_index, int(payload["lag"]), scaler)
            rng = np.random.default_rng(33000 + fold)
            index = np.arange(len(windows.target))
            if len(index) > rows:
                onset = np.flatnonzero(windows.stratum == "onset")
                other = np.setdiff1d(index, onset)
                n_other = max(0, rows - min(len(onset), rows // 4))
                index = np.sort(np.concatenate([
                    onset[: rows - n_other],
                    rng.choice(other, size=min(n_other, len(other)), replace=False),
                ]))
            history = windows.history[index]
            worm = windows.worm[index]
            target = (
                windows.target[index]
                - history[:, -1, :cohort.n_neurons]
            )
            factual = _log_prob(model, history, target, device=device)
            for source, neuron in enumerate(cohort.neurons):
                rolled = _source_roll(history, worm, source, rng)
                ablated = _log_prob(model, rolled, target, device=device)
                delta = factual - ablated
                for phase in ("all", "onset", "quiet"):
                    use = np.ones(len(delta), dtype=bool)
                    if phase == "onset":
                        use = windows.stratum[index] == "onset"
                    elif phase == "quiet":
                        use = windows.stratum[index] == "off"
                    result_rows.append({
                        "cohort": cohort_name,
                        "model_id": model_id,
                        "fold": fold,
                        "source": neuron,
                        "phase": phase,
                        "n_rows": int(use.sum()),
                        "ablation_nll_increase": float(np.mean(delta[use])) if use.any() else np.nan,
                    })
            del model
            if device == "mps":
                torch.mps.empty_cache()
    frame = pd.DataFrame(result_rows)
    frame.to_csv(output / "source_history_circular_ablation.csv", index=False)
    summary = (
        frame[frame.phase == "all"]
        .groupby("cohort", as_index=False)
        .agg(
            mean_nll_increase=("ablation_nll_increase", "mean"),
            median_nll_increase=("ablation_nll_increase", "median"),
            positive_source_fraction=("ablation_nll_increase", lambda value: float(np.mean(value > 0))),
            rows=("n_rows", "sum"),
        )
    )
    summary.to_csv(output / "ablation_summary.csv", index=False)
    (output / "validation.json").write_text(json.dumps({
        "status": "pass",
        "rows": int(len(frame)),
        "cohorts": sorted(frame.cohort.unique()),
        "external_references_consulted": False,
        "null": "within-worm circular shift of one complete source-history window",
        "metric": "factual log probability minus source-shifted log probability; positive means source history helps",
    }, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural", type=Path, default=Path("results/higher_order_neural_20260828"))
    parser.add_argument(
        "--effects", type=Path,
        default=Path("results/higher_order_neural_effects_20260828"),
    )
    parser.add_argument(
        "--evidence", type=Path,
        default=Path("results/distributed_lag_dynamics_20260828"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/higher_order_neural_ablation_20260828"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--wait-hours", type=float, default=8.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_hours * 3600
    while not (args.effects / "protocol.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("neural effects did not complete before the ablation deadline")
        time.sleep(20)
    result = run(
        args.neural.resolve(), args.evidence.resolve(), args.output.resolve(),
        device=args.device, rows=args.rows,
    )
    print(f"NEURAL_HISTORY_ABLATION_COMPLETE {result}", flush=True)


if __name__ == "__main__":
    main()
