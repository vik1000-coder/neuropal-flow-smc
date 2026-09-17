"""Independent dense-density and saved-score replay audit of completed fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import multivariate_normal, multivariate_t

from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.inference import load_checkpoint
from conditional_neural_benchmark.runner import _split_windows
from conditional_neural_benchmark.distribution_structure_runner import DEFAULT_RUN, FOLD_RUN, atomic_json
from conditional_neural_benchmark.distribution_structure_evaluate import keyed_seed, verify_checkpoint
from conditional_neural_benchmark.distribution_structure_scoring import score_samples


def validate(run_dir: Path):
    torch.set_num_threads(2)
    if not (run_dir / "analysis/validation.json").exists():
        raise RuntimeError("the complete analysis must validate before numerical replay")
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    records = []
    for fold in range(5):
        scaler, _, _, testing = _split_windows(cohort, folds, fold, 80)
        for path in sorted((run_dir / "evaluation").glob(f"*__f{fold}__*__r731__N128.npz")):
            with np.load(path, allow_pickle=False) as z:
                metadata = json.loads(str(z["metadata"]))
                model, payload, _ = load_checkpoint(metadata["checkpoint"], "cpu")
                verify_checkpoint(payload, metadata["model_id"], fold, metadata["model_seed"], cohort, scaler)
                index = z["history_index"][:32]
                h = testing.history[index]
                last = h[:, -1, :cohort.n_neurons]
                delta = testing.target[index] - last
                tensor = torch.as_tensor(h.reshape(len(h), -1))
                seed = keyed_seed("sample", fold, metadata["model_seed"], 731, 128, 0)
                with torch.no_grad():
                    draws = model.sample(tensor, 128, seed=seed).cpu().numpy()
                scores = score_samples(draws, delta, z["tail_threshold"], variogram_offset=last)
                names = list(z["metric_names"].astype(str))
                actual = np.stack([scores[k] for k in names], axis=1)
                expected = z["scores"][0, :len(index)]
                replay_error = float(np.max(np.abs(actual - expected)))
                if replay_error > 1e-7:
                    raise RuntimeError(f"score replay failed: {path.name}, {replay_error}")
                density_error = None
                if metadata["model_id"] != "flow":
                    with torch.no_grad():
                        context = model.context(tensor[:8])
                        mean, log_sd, factor = model.head.parameters_at(context)
                        nlog = model.log_prob(torch.as_tensor(delta[:8]), tensor[:8]).numpy()
                    mean, log_sd, factor = mean.numpy().astype(float), log_sd.numpy().astype(float), factor.numpy().astype(float)
                    covariance = (np.eye(cohort.n_neurons)[None] * np.exp(2 * log_sd)[:, :, None]
                                  + factor @ factor.transpose(0, 2, 1))
                    dense = []
                    for i in range(len(mean)):
                        if model.head.student:
                            nu = float(model.head.degrees_of_freedom.detach())
                            dense.append(multivariate_t.logpdf(delta[i], loc=mean[i],
                                         shape=covariance[i] * ((nu - 2) / nu), df=nu))
                        else:
                            dense.append(multivariate_normal.logpdf(delta[i], mean=mean[i], cov=covariance[i]))
                    density_error = float(np.max(np.abs(np.asarray(dense) - nlog)))
                    if density_error > 0.01:
                        raise RuntimeError(f"dense log-density mismatch: {path.name}, {density_error}")
                records.append({"archive": path.name, "model": metadata["model_id"],
                                "score_max_absolute_error": replay_error,
                                "dense_logpdf_max_absolute_error": density_error})
                print(f"REPLAY_OK {path.name} error={replay_error:.3g} density={density_error}", flush=True)
    if len(records) != 40 or sum(r["dense_logpdf_max_absolute_error"] is not None for r in records) != 30:
        raise RuntimeError("replay grid is incomplete")
    atomic_json(run_dir / "analysis/numerical_replay.json", {
        "status": "pass", "score_replays": 40, "density_checkpoints": 30,
        "score_tolerance": 1e-7, "logpdf_tolerance_per_vector": 0.01,
        "max_score_error": max(r["score_max_absolute_error"] for r in records),
        "max_dense_logpdf_error": max(r["dense_logpdf_max_absolute_error"] for r in records if r["dense_logpdf_max_absolute_error"] is not None),
        "records": records})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    validate(args.run_dir.resolve())


if __name__ == "__main__":
    main()
