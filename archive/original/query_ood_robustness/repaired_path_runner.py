from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from compatibility_neural_benchmark.core import (
    GeneratorAdapter,
    RepairedResponseConfig,
    causal_fill,
    fit_anchor_projection,
    standardize_for_checkpoint,
    stimulus_for_trace,
    training_source_quantiles,
)
from compatibility_neural_benchmark.progressive_smc import progressive_smc_repaired_responses
from conditional_neural_benchmark.data import load_cohort
from conditional_neural_benchmark.focused_world_model_runner import _load_folds
from conditional_neural_benchmark.runner import _split_indices, _split_windows

from .protocol import FOLD_RUN, RUN_ROOT, atomic_csv, freeze_protocol, update_status
from .study import FAMILY_ORDER, _checkpoint_path, _verify_checkpoint, keyed_seed


def _phase(value: str) -> str:
    return {
        "off": "baseline",
        "onset": "onset",
        "active": "active",
        "offset": "offset",
        "mixed": "active",
    }.get(str(value), "baseline")


def _sensitivity_queries(manifest: pd.DataFrame, count: int) -> pd.DataFrame:
    factual = manifest[manifest.query_class == "supported_history_common_event"].copy()
    factual = factual.sort_values(["history_support_value", "fold", "worm_id"])
    positions = np.unique(np.linspace(0, len(factual) - 1, min(count, len(factual)), dtype=int))
    return factual.iloc[positions].copy()


def run_repaired_path(run_root: Path, *, smoke: bool = False) -> None:
    """Run the repository's original progressive repaired-path estimand.

    This sensitivity is deliberately separate from static one-step soft-event
    controls. It uses the exact existing sampler-only repair law and reports a
    one-frame population-mean high-minus-low response for seven preselected
    factual cuts spanning empirical support.
    """
    protocol = freeze_protocol(run_root)
    update_status(run_root, "repaired_path_smc", "running")
    torch.set_num_threads(2)
    cohort = load_cohort(0.90, cohort_mode="oh16230_head")
    folds = _load_folds(cohort, FOLD_RUN)
    manifest = pd.read_csv(run_root / "query_manifest.csv")
    queries = _sensitivity_queries(manifest, 1 if smoke else 7)
    families = FAMILY_ORDER[:1] if smoke else FAMILY_ORDER
    particle_counts = [128] if smoke else protocol["smc"]["particle_counts"]
    sampling_seeds = protocol["smc"]["sampling_seeds"][:1] if smoke else protocol["smc"]["sampling_seeds"]
    effect_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    context: dict[int, tuple] = {}
    for query in queries.itertuples():
        fold = int(query.fold)
        if fold not in context:
            scaler, _, _, _ = _split_windows(cohort, folds, fold, 80)
            train_index, _, _ = _split_indices(folds, fold)
            standardized_training = [
                causal_fill(scaler.transform(cohort.traces[int(index)]))
                for index in train_index
            ]
            projection = fit_anchor_projection(standardized_training, rank=12)
            quantiles = training_source_quantiles(
                standardized_training,
                [cohort.stimulus_schedules[int(index)] for index in train_index],
                source_window_frames=4,
            )
            pooled = np.concatenate(standardized_training)
            thresholds = np.quantile(pooled, 0.90, axis=0).astype(np.float32)
            context[fold] = (scaler, projection, quantiles, thresholds)
        scaler, projection, quantiles, thresholds = context[fold]
        worm_index = int(query.worm_index)
        source = int(query.source_index)
        trace = causal_fill(scaler.transform(cohort.traces[worm_index]))
        phase = _phase(str(query.stimulus_context))
        cut_time = int(query.anchor_time)
        cut_time = min(cut_time, len(trace) - 2)
        if cut_time - 4 - 80 + 1 < 0:
            raise RuntimeError(f"repaired-path cut lacks history: {query.query_id}")
        for family in families:
            checkpoint = _checkpoint_path(run_root, family, fold, 1701)
            if smoke and family in {"autoregressive_mdn4", "conditional_edm_diffusion"}:
                checkpoint = Path(str(checkpoint).replace("/matched_primary/", "/matched_smoke/"))
            adapter = GeneratorAdapter.load(str(checkpoint), device=protocol["resources"]["device"])
            _verify_checkpoint(adapter.checkpoint, cohort, scaler, fold, 1701)
            stimulus = stimulus_for_trace(trace, cohort, worm_index, adapter.checkpoint)
            for n_particles in particle_counts:
                config = RepairedResponseConfig(
                    history_frames=80,
                    repair_frames=4,
                    source_window_frames=4,
                    source_lag_frames=0,
                    horizon_frames=(1,),
                    n_particles=n_particles,
                    anchor_lambda=0.25,
                    epsilon_iqr_fraction=0.25,
                    anchor_rank=12,
                    min_ess=max(20.0, 0.10 * n_particles),
                    max_normalized_weight=0.20,
                    min_achieved_fraction=0.25,
                    sampling_chunk_size=128 if family == "autoregressive_mdn4" else 512,
                )
                for sampling_seed in sampling_seeds:
                    started = time.perf_counter()
                    result = progressive_smc_repaired_responses(
                        adapter,
                        trace,
                        stimulus,
                        cut_time=cut_time,
                        projection=projection,
                        source_low=quantiles[phase]["low"],
                        source_high=quantiles[phase]["high"],
                        source_iqr=quantiles[phase]["iqr"],
                        thresholds=thresholds,
                        config=config,
                        seed=keyed_seed("repaired_path", family, query.query_id, n_particles, sampling_seed),
                        branch_factor=2,
                        future_branch_factor=2,
                        tempering_ess_fraction=0.70,
                        source_indices=np.asarray([source]),
                    )
                    elapsed = time.perf_counter() - started
                    keep = np.arange(cohort.n_neurons) != source
                    effect = float(result["response_endpoint_mean"][0, 0, keep].mean())
                    low_ess_fraction = np.asarray(result["diagnostic_step_ess_fraction_low"])[0]
                    high_ess_fraction = np.asarray(result["diagnostic_step_ess_fraction_high"])[0]
                    minimum_ess_fraction = float(min(low_ess_fraction.min(), high_ess_fraction.min()))
                    ancestor_fraction = float(min(
                        result["diagnostic_distinct_ancestors_low"][0],
                        result["diagnostic_distinct_ancestors_high"][0],
                    ) / n_particles)
                    log10_mass = float(min(
                        result["diagnostic_log10_total_low"][0],
                        result["diagnostic_log10_total_high"][0],
                    ))
                    healthy = bool(result["diagnostic_valid"][0] > 0.5 and minimum_ess_fraction >= 0.20 and ancestor_fraction >= 0.10)
                    base = {
                        "run_id": run_root.name,
                        "dataset": "neuropal_oh16230",
                        "system": "repaired_path_predictive_query",
                        "generator_seed": -1,
                        "worm_id": query.worm_id,
                        "fold": fold,
                        "model_family": family,
                        "model_seed": 1701,
                        "sampling_seed": sampling_seed,
                        "query_id": f"{query.query_id}__repaired_path",
                        "query_class": f"repaired_path_{phase}",
                        "source_neuron": query.source_neuron,
                        "target_functional": "one_frame_population_mean_excluding_source",
                        "amplitude": float(result["diagnostic_target_high"][0]),
                        "history_support_value": float(query.history_support_value),
                        "event_rarity": math.pow(10.0, max(-300.0, log10_mass)),
                        "particle_count": n_particles,
                        "smc_method": "progressive_repaired_path",
                        "effect_estimate": effect,
                        "monte_carlo_error": np.nan,
                        "minimum_ess_fraction": minimum_ess_fraction,
                        "final_ess_fraction": float(min(low_ess_fraction[-1], high_ess_fraction[-1])),
                        "max_normalized_weight": float(max(
                            result["diagnostic_max_weight_low"][0],
                            result["diagnostic_max_weight_high"][0],
                        )),
                        "unique_ancestor_fraction": ancestor_fraction,
                        "endpoint_constraint_satisfied": bool(result["diagnostic_valid"][0] > 0.5),
                        "healthy_smc": healthy,
                        "wall_seconds": elapsed,
                        "model_evaluations": int(2 * 4 * 2 * n_particles + 2 * 2 * n_particles),
                    }
                    effect_rows.append(base)
                    for step in range(4):
                        diagnostic_rows.append({
                            **base,
                            "arm": "paired_low_high",
                            "stage": step,
                            "beta_end_low": float(result["diagnostic_step_beta_low"][0, step]),
                            "beta_end_high": float(result["diagnostic_step_beta_high"][0, step]),
                            "ess_fraction_low": float(low_ess_fraction[step]),
                            "ess_fraction_high": float(high_ess_fraction[step]),
                            "max_weight_low": float(result["diagnostic_step_max_weight_low"][0, step]),
                            "max_weight_high": float(result["diagnostic_step_max_weight_high"][0, step]),
                            "tempering_resamples_low": int(result["diagnostic_step_tempering_resamples_low"][0, step]),
                            "tempering_resamples_high": int(result["diagnostic_step_tempering_resamples_high"][0, step]),
                            "forced_tempering_low": int(result["diagnostic_step_forced_tempering_low"][0, step]),
                            "forced_tempering_high": int(result["diagnostic_step_forced_tempering_high"][0, step]),
                        })
            print(f"REPAIRED_PATH_DONE family={family} query={query.query_id}", flush=True)
            del adapter
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    atomic_csv(run_root / "repaired_path_effects.csv", pd.DataFrame(effect_rows))
    atomic_csv(run_root / "smc_diagnostics_repaired_path.csv", pd.DataFrame(diagnostic_rows))
    update_status(run_root, "repaired_path_smc", "complete", rows=len(effect_rows), queries=len(queries))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_repaired_path(args.run_root.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
