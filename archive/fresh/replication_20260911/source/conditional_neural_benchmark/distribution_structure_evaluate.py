"""Held-out sample-bank audit with exact marginal-preserving permutations."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.inference import load_checkpoint
from conditional_neural_benchmark.runner import _split_windows
from conditional_neural_benchmark.chemical_encoding_runner import sha256
from conditional_neural_benchmark.distribution_structure_runner import (
    DEFAULT_RUN, FLOW_ID, FOLD_RUN, atomic_json, candidate_configs, flow_checkpoint,
)
from conditional_neural_benchmark.distribution_structure_scoring import (
    marginal_invariance, score_samples, shuffle_coordinates,
)


def keyed_seed(*parts) -> int:
    digest = hashlib.sha256(json.dumps(parts).encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2 ** 31 - 1)


def sensitivity_indices(windows, count: int) -> np.ndarray:
    selected = []
    for worm in np.unique(windows.worm):
        for stratum in sorted(set(windows.stratum[windows.worm == worm])):
            rows = np.flatnonzero((windows.worm == worm) & (windows.stratum == stratum))
            rng = np.random.default_rng(keyed_seed("sensitivity", int(worm), str(stratum)))
            selected.extend(rng.choice(rows, size=min(count, len(rows)), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def fixed_residual_sd(model, training, device: str, chunk: int = 128):
    total, count = np.zeros(training.target.shape[1], dtype=np.float64), 0
    with torch.no_grad():
        for start in range(0, len(training.target), chunk):
            history = training.history[start:start + chunk]
            h = torch.as_tensor(history.reshape(len(history), -1), device=device)
            mean, _, _ = model.head.parameters_at(model.context(h))
            target = training.target[start:start + chunk] - history[:, -1, :training.target.shape[1]]
            total += ((target.astype(float) - mean.cpu().numpy()) ** 2).sum(axis=0)
            count += len(history)
    return np.sqrt(np.maximum(total / count, 1e-6))


def verify_checkpoint(payload, model_id, fold, seed, cohort, scaler):
    expected = FLOW_ID if model_id == "flow" else model_id
    if payload["model_config"]["model_id"] != expected or payload["fold"] != fold or payload["seed"] != seed:
        raise RuntimeError("wrong model/fold/seed checkpoint")
    if payload["neurons"] != list(cohort.neurons) or payload["stimulus_schema_fingerprint"] != cohort.stimulus_schema_fingerprint:
        raise RuntimeError("checkpoint cohort mismatch")
    if payload["lag"] != 80 or payload["input_channels"] != cohort.n_neurons + 1 or not payload["model_config"]["residual_target"]:
        raise RuntimeError("wrong conditioning or residual specification")
    for key in ("mean", "scale"):
        np.testing.assert_array_equal(payload["scaler"][key], getattr(scaler, key))


def evaluate(run_dir: Path, models: list[str], *, device: str, sensitivity: bool = False,
             smoke: bool = False, chunk: int = 32, fold_filter: list[int] | None = None):
    protocol = json.loads((run_dir / "protocol.json").read_text())
    for source, digest in protocol["training_source_sha256"].items():
        if sha256(Path(__file__).resolve().parents[1] / source) != digest:
            raise RuntimeError(f"training source changed: {source}")
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    if cohort.stimulus_schema_fingerprint != protocol["stimulus_schema_fingerprint"]:
        raise RuntimeError("stimulus provenance changed")
    stage = "evaluation_smoke" if smoke else ("particle_sensitivity" if sensitivity else "evaluation")
    output = run_dir / stage
    output.mkdir(exist_ok=True)
    source_hash = {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("distribution_structure_scoring.py")]}
    settings = {"source": source_hash, "protocol": protocol["fingerprint"], "device": device,
                "chunk": chunk, "smoke": smoke, "sensitivity": sensitivity}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("evaluation resume rejected: source/settings changed")
    else:
        atomic_json(manifest, dict(settings, fingerprint=fingerprint))
    model_seeds = protocol["seeds"][:1] if smoke else protocol["seeds"]
    particle_counts = [32] if smoke else ([128, 256] if sensitivity else [128])
    sampling_seeds = [731] if smoke or sensitivity else protocol["evaluation"]["sample_seeds"]
    repeats = 2 if smoke else (4 if sensitivity else protocol["evaluation"]["shuffle_repeats"])
    selected_folds = protocol["folds"][:1] if smoke else protocol["folds"]
    if fold_filter is not None:
        selected_folds = [f for f in selected_folds if f in fold_filter]
    for fold in selected_folds:
        scaler, training, _, testing = _split_windows(cohort, folds, fold, 80)
        index = np.arange(len(testing.target))
        if sensitivity or smoke:
            index = sensitivity_indices(testing, 2 if smoke else 16)
        selected = testing.take(index)
        innovation = training.target - training.history[:, -1, :cohort.n_neurons]
        threshold = 2 * np.maximum(np.sqrt(np.mean(innovation.astype(float) ** 2, axis=0)), 1e-3)
        del innovation
        for model_id in models:
            if sensitivity and model_id != "flow":
                continue
            for model_seed in model_seeds:
                checkpoint = flow_checkpoint(fold, model_seed) if model_id == "flow" else (
                    run_dir / "checkpoints" / ("matched_smoke" if protocol["smoke"] else "matched_full_cv") /
                    f"{model_id}__L80__f{fold}__s{model_seed}.pt")
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                checkpoint_hash = sha256(checkpoint)
                if model_id == "flow":
                    declared = {item["path"]: item["sha256"] for item in protocol["frozen_flow_checkpoints"]}
                    if declared.get(str(checkpoint)) != checkpoint_hash:
                        raise RuntimeError("frozen flow checksum mismatch")
                pending = []
                for n in particle_counts:
                    for sample_seed in sampling_seeds:
                        path = output / f"{model_id}__f{fold}__s{model_seed}__r{sample_seed}__N{n}.npz"
                        if path.exists():
                            with np.load(path, allow_pickle=False) as saved:
                                old = json.loads(str(saved["metadata"]))
                                if old["fingerprint"] != fingerprint or old["checkpoint_sha256"] != checkpoint_hash:
                                    raise RuntimeError("saved evaluation differs from frozen inputs")
                        else:
                            pending.append((n, sample_seed, path))
                if not pending:
                    continue
                model, payload, resolved = load_checkpoint(checkpoint, device)
                verify_checkpoint(payload, model_id, fold, model_seed, cohort, scaler)
                fixed_sd = fixed_residual_sd(model, training, str(resolved)) if model_id == "matched_gaussian_diag" else None
                for n, sample_seed, path in pending:
                    started = time.perf_counter()
                    print(f"EVAL_START model={model_id} fold={fold} seed={model_seed} sample_seed={sample_seed} N={n} histories={len(index)}", flush=True)
                    variants = ["original"] + ([f"shuffle_{r}" for r in range(repeats)] if model_id == "flow" else [])
                    if fixed_sd is not None:
                        variants.append("fixed_variance")
                    buckets = {variant: [] for variant in variants}
                    nll_values, constant_nll = [], []
                    errors = []
                    metric_names = None
                    for start in range(0, len(index), chunk):
                        history = selected.history[start:start + chunk]
                        last = history[:, -1, :cohort.n_neurons]
                        delta = selected.target[start:start + chunk] - last
                        seed = keyed_seed("sample", fold, model_seed, sample_seed, n, start)
                        tensor = torch.as_tensor(history.reshape(len(history), -1), device=resolved)
                        with torch.no_grad():
                            draws = model.sample(tensor, n, seed=seed).cpu().numpy()
                            if model.capabilities.normalized_density:
                                y_tensor = torch.as_tensor(delta, device=resolved)
                                nll_values.extend((-model.log_prob(y_tensor, tensor) / cohort.n_neurons).cpu().numpy())
                            if fixed_sd is not None:
                                mu, log_sd, _ = model.head.parameters_at(model.context(tensor))
                                mu, sd = mu.cpu().numpy(), np.exp(log_sd.cpu().numpy())
                                z = (draws.astype(float) - mu[:, None]) / sd[:, None]
                                fixed_draws = mu[:, None] + z * fixed_sd[None, None]
                                constant_nll.extend((0.5 * ((delta - mu) / fixed_sd) ** 2 + np.log(fixed_sd) + 0.5 * np.log(2 * np.pi)).mean(axis=1))
                        scores = score_samples(draws, delta, threshold, variogram_offset=last)
                        if metric_names is None:
                            metric_names = list(scores)
                        buckets["original"].append(np.stack([scores[k] for k in metric_names], axis=1))
                        if model_id == "flow":
                            # A common draw permutation must preserve every joint score too.
                            common = draws[:, np.random.default_rng(seed + 1).permutation(n)]
                            common_scores = score_samples(common, delta, threshold, variogram_offset=last)
                            for key in metric_names:
                                np.testing.assert_allclose(scores[key], common_scores[key], rtol=0, atol=1e-10)
                            for repeat in range(repeats):
                                shuffled = shuffle_coordinates(draws, keyed_seed("shuffle", seed, repeat))
                                altered = score_samples(shuffled, delta, threshold, variogram_offset=last)
                                errors.append(marginal_invariance(scores, altered))
                                if repeat == 0:
                                    np.testing.assert_array_equal(np.sort(draws, axis=1), np.sort(shuffled, axis=1))
                                buckets[f"shuffle_{repeat}"].append(np.stack([altered[k] for k in metric_names], axis=1))
                        if fixed_sd is not None:
                            control = score_samples(fixed_draws, delta, threshold, variogram_offset=last)
                            buckets["fixed_variance"].append(np.stack([control[k] for k in metric_names], axis=1))
                    scores = np.stack([np.concatenate(buckets[v], axis=0) for v in variants])
                    if not np.isfinite(scores).all():
                        raise RuntimeError("nonfinite predictive score")
                    metadata = {"fingerprint": fingerprint, "checkpoint": str(checkpoint),
                                "checkpoint_sha256": checkpoint_hash, "model_id": model_id,
                                "fold": fold, "model_seed": model_seed, "sample_seed": sample_seed,
                                "samples": n, "histories": len(index), "shuffle_repeats": repeats if model_id == "flow" else 0,
                                "max_marginal_invariance_error": max(errors, default=0.0),
                                "wall_seconds": time.perf_counter() - started,
                                "parameter_count": model.parameter_count,
                                "best_epoch": payload["fit_trace"]["best_epoch"],
                                "stopped_epoch": payload["fit_trace"]["stopped_epoch"],
                                "student_df": float(model.head.degrees_of_freedom.detach().cpu()) if model_id == "matched_student_t_rank8" else None}
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, scores=scores, metric_names=np.asarray(metric_names),
                                            variants=np.asarray(variants), history_index=index,
                                            worm=selected.worm, worm_ids=np.asarray(cohort.worm_ids),
                                            time=selected.time, stratum=selected.stratum,
                                            nll=np.asarray(nll_values), fixed_nll=np.asarray(constant_nll),
                                            tail_threshold=threshold, fixed_sd=np.asarray([] if fixed_sd is None else fixed_sd),
                                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
                    temporary.replace(path)
                    print(f"EVAL_DONE {path.name} seconds={metadata['wall_seconds']:.1f} energy={scores[0,:,metric_names.index('energy')].mean():.6f}", flush=True)
                del model
                if str(resolved) == "mps":
                    torch.mps.empty_cache()
    print(f"EVALUATION_STAGE_COMPLETE stage={stage} models={models}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--models", nargs="+", default=["flow"] + [c.model_id for c in candidate_configs()])
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    allowed = {"flow"} | {c.model_id for c in candidate_configs()}
    if not set(args.models) <= allowed:
        parser.error("unknown model")
    torch.set_num_threads(args.threads)
    evaluate(args.run_dir.resolve(), args.models, device=args.device, sensitivity=args.sensitivity,
             smoke=args.smoke, chunk=args.chunk, fold_filter=args.folds)


if __name__ == "__main__":
    main()
