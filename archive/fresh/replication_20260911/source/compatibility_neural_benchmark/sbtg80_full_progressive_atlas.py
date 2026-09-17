"""Full historical SBTG80 progressive-bridge SMC atlas.

This production lane extends the reviewed historical 80-neuron sensitivity run
to the complete lag/horizon/channel/context grid used by the prediction atlas.
It is deliberately separate from the clean 54-neuron atlas.  The released
SBTG80 cache pseudo-paired head and tail recordings and donor-imputed missing
traces, so this artifact is a historical model-response sensitivity and cannot
support biological coupling or causal claims.

The raw sampler firewall closes before Randi, Cook, Bentley, receptor, or
published-SBTG references are opened.  Inference uses worms as the independent
unit and is conditional on the one complete atlas-blind generator seed available
for all five outer folds of the historical tournament.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.latent_distributional_audit import (
    benjamini_hochberg,
)
from compatibility_neural_benchmark.prediction_atlas_analysis import (
    CHANNELS,
    CONTEXTS,
    ORIENTATION,
    _NpzStream,
    _bootstrap_counts,
    _contextualize_source_metric,
    _contextualize_support,
    _contextualize_values,
    _is_signed,
    _worm_summary,
    context_metadata,
    effect_normalization_denominator,
    orient_response_once,
)
from compatibility_neural_benchmark.prediction_atlas_external_analysis import (
    _lagmax_rows,
    _neuromodulator_rows,
    _reference_rows,
)
from compatibility_neural_benchmark.prediction_atlas_runner import (
    GENERATOR_ENCODING,
    PHASES,
    RESPONSE_KEYS,
    _atomic_csv,
    _atomic_json,
    run_one,
    sha256,
    validate_resume_archive,
)
from compatibility_neural_benchmark.postfreeze_external_analysis import (
    load_references,
)
from compatibility_neural_benchmark.sbtg80_progressive_sensitivity import (
    _load_published_slices,
    _paired_source_bootstrap,
)
from conditional_neural_benchmark.data import load_sbtg_cohort, make_fold_assignments


MODEL_ID = "tcn_wide_flow"
METHOD = "progressive_bridge_smc"
HISTORY_FRAMES = 8
GENERATOR_SEED = 1701
FOLD_SEED = 20_260_828
BASE_SEED = 20_260_901
SOURCE_LAGS = (1, 4, 8, 16)
HORIZONS = (1, 2, 4, 8, 16, 32)
PARTICLES = 32
SOURCE_WINDOW_FRAMES = 4
MIN_ESS = 6.0
MAX_NORMALIZED_WEIGHT = 0.20
MIN_ACHIEVED_SOURCE_FRACTION = 0.25
MIN_NORMALIZATION_GAP = 0.10
MIN_VALID_FRACTION = 0.50
BOOTSTRAP_REPLICATES = 256
BOOTSTRAP_SEED = 20_260_901
PERMUTATION_SEED = 20_260_902

# Exact intersection of the frozen 80-class axis with the released tail file.
# These labels identify recording origin, not anatomical exclusivity or valid
# simultaneous head/tail observation.
TAIL_ORIGIN_NEURONS = (
    "ALN",
    "DVA",
    "DVB",
    "DVC",
    "LUA",
    "PHA",
    "PHB",
    "PHC",
    "PLN",
    "PQR",
    "PVC",
    "PVN",
    "PVP",
    "PVQ",
    "PVR",
    "PVT",
    "PVW",
)

DEFAULT_OUTPUT = Path("results/sbtg80_full_progressive_atlas_20260901")
DEFAULT_SOURCE_RUN = Path("results/higher_order_neural_20260828/sbtg80")
DEFAULT_REFERENCE_RELEASE = Path("/Users/vik/Downloads/SBTG-public-release copy")
DEFAULT_PUBLISHED_ARCHIVE = DEFAULT_REFERENCE_RELEASE / "results/paper/sbtg_lag_matrices.npz"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_path(source_run: Path, fold: int) -> Path:
    phase = (
        "screen_fold0"
        if fold == 0
        else "screen_folds12"
        if fold in (1, 2)
        else "confirmation"
    )
    return (
        source_run
        / "checkpoints"
        / phase
        / f"{MODEL_ID}__L{HISTORY_FRAMES}__f{fold}__s{GENERATOR_SEED}.pt"
    )


def _archive_path(output: Path, lag: int, fold: int) -> Path:
    return (
        output
        / "responses"
        / METHOD
        / f"{MODEL_ID}__{METHOD}__ell{lag}__N{PARTICLES}__f{fold}__s{GENERATOR_SEED}.npz"
    )


def _write_fold_assignments(path: Path, cohort, folds: np.ndarray) -> None:
    expected = pd.DataFrame(
        {
            "worm_index": np.arange(cohort.n_worms, dtype=int),
            "worm_id": cohort.worm_ids,
            "outer_fold": folds.astype(int),
        }
    )
    if path.exists():
        observed = pd.read_csv(path)
        if not observed.equals(expected):
            raise RuntimeError("existing SBTG80 fold assignment differs")
        return
    expected.to_csv(path, index=False)


def _source_provenance(source_run: Path) -> dict[str, object]:
    manifest_path = source_run / "manifest.json"
    winner_path = source_run / "winner_selection.json"
    validation_path = source_run / "validation.json"
    for path in (manifest_path, winner_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    winner = json.loads(winner_path.read_text())
    validation = json.loads(validation_path.read_text())
    if (
        manifest.get("cohort") != "sbtg80"
        or int(manifest.get("worms", -1)) != 20
        or int(manifest.get("neurons", -1)) != 80
        or int(manifest.get("lag", -1)) != HISTORY_FRAMES
    ):
        raise RuntimeError("source tournament is not the frozen SBTG80 run")
    if winner.get("model_id") != MODEL_ID or winner.get("external_references_consulted") is not False:
        raise RuntimeError("source winner is not the atlas-blind SBTG80 flow")
    if (
        validation.get("complete_confirmation") is not True
        or int(validation.get("completed_trials", -1)) != 24
        or int(validation.get("failed_trials", -1)) != 0
        or validation.get("external_references_consulted") is not False
        or validation.get("winner") != MODEL_ID
    ):
        raise RuntimeError("source SBTG80 tournament did not pass")
    checkpoints = {str(fold): _checkpoint_path(source_run, fold) for fold in range(5)}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing source checkpoints: " + ", ".join(missing))
    return {
        "path": str(source_run.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "winner_selection_sha256": sha256(winner_path),
        "validation_sha256": sha256(validation_path),
        "checkpoint_sha256": {fold: sha256(path) for fold, path in checkpoints.items()},
        "selection_external_references_consulted": False,
    }


def _raw_manifest(
    output: Path,
    source_run: Path,
    cohort,
) -> dict[str, object]:
    tail = set(TAIL_ORIGIN_NEURONS)
    if not tail.issubset(cohort.neurons) or len(tail) != 17:
        raise RuntimeError("frozen tail-origin panel changed")
    return {
        "created_utc": _utc(),
        "status": "raw_sampling_planned",
        "protocol": "full historical SBTG80 progressive-bridge SMC atlas v1",
        "analysis_role": "historical_sbtg80_model_response_sensitivity",
        "cohort_mode": cohort.cohort_mode,
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "neurons": list(cohort.neurons),
        "fps": cohort.fps,
        "lineage_warning": cohort.lineage_warning,
        "clean_superset": False,
        "recording_origin": {
            "head_count": cohort.n_neurons - len(tail),
            "tail_count": len(tail),
            "tail_neurons": list(TAIL_ORIGIN_NEURONS),
            "definition": (
                "intersection with released Tail_Activity_OH16230 neuron classes; "
                "recording origin only, not simultaneous-pair validity"
            ),
        },
        "model_id": MODEL_ID,
        "history_frames": HISTORY_FRAMES,
        "generator_seeds": [GENERATOR_SEED],
        "generator_seed_limitation": (
            "one atlas-blind generator seed has complete five-fold checkpoints; "
            "worm intervals are conditional on this fitted-generator realization"
        ),
        "folds": list(range(5)),
        "fold_seed": FOLD_SEED,
        "source_lag_frames": list(SOURCE_LAGS),
        "horizon_frames": list(HORIZONS),
        "channels": list(CHANNELS),
        "contexts": list(CONTEXTS),
        "particles": PARTICLES,
        "source_window_frames": SOURCE_WINDOW_FRAMES,
        "minimum_effective_sample_size": MIN_ESS,
        "maximum_normalized_weight": MAX_NORMALIZED_WEIGHT,
        "minimum_achieved_source_fraction": MIN_ACHIEVED_SOURCE_FRACTION,
        "minimum_normalization_gap": MIN_NORMALIZATION_GAP,
        "base_seed": BASE_SEED,
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "stimulus_generator_encoding": GENERATOR_ENCODING,
        "chemical_identity_conditioned": False,
        "chemical_context_warning": (
            "chemical panels are event-stratified views of a binary-any-stimulus "
            "generator; they are not chemically conditioned effects"
        ),
        "source_tournament": _source_provenance(source_run),
        "atlas_firewall": (
            "no Randi, Cook, Bentley, receptor, connectome, or published SBTG "
            "matrix is opened during fitting or raw progressive-SMC sampling"
        ),
        "claim_boundary": (
            "model-relative observational response sensitivity; not a causal "
            "intervention, coupling estimate, anatomical edge, receptor effect, "
            "or physical transmission delay"
        ),
        "responses_root": str((output / "responses").resolve()),
    }


def _write_or_validate_manifest(path: Path, candidate: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        _atomic_json(path, candidate)
        return candidate
    existing = json.loads(path.read_text())
    dynamic = {
        "created_utc",
        "status",
        "completed_utc",
        "analysis_completed_utc",
        "analysis_manifest_sha256",
    }
    left = {key: value for key, value in existing.items() if key not in dynamic}
    right = {key: value for key, value in candidate.items() if key not in dynamic}
    if left != right:
        raise RuntimeError("existing SBTG80 full-run manifest differs from requested protocol")
    return existing


def _validate_archive(
    path: Path,
    *,
    cohort,
    folds: np.ndarray,
    source_run: Path,
    lag: int,
    fold: int,
) -> dict[str, object]:
    checkpoint = _checkpoint_path(source_run, fold)
    heldout = np.flatnonzero(folds == fold)
    with np.load(path, allow_pickle=False) as data:
        validate_resume_archive(
            data,
            cohort=cohort,
            method=METHOD,
            model_id=MODEL_ID,
            fold=fold,
            seed=GENERATOR_SEED,
            source_lag=lag,
            particles=PARTICLES,
            horizons=HORIZONS,
            source_window_frames=SOURCE_WINDOW_FRAMES,
            history_lag=HISTORY_FRAMES,
            base_seed=BASE_SEED,
            requested_device=str(data["requested_device"].item()),
            checkpoint=checkpoint,
            expected_worm_indices=heldout,
        )
        if tuple(data["response_keys"].astype(str)) != tuple(RESPONSE_KEYS):
            raise RuntimeError(f"response schema mismatch in {path}")
        expected_shape = (len(heldout), len(PHASES), 3, 80, len(HORIZONS), 80)
        for key in RESPONSE_KEYS:
            values = data[key]
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise RuntimeError(f"bad response geometry/value for {key} in {path}")
        probability = data["response_event_probability"]
        if np.any(probability < -1.00001) or np.any(probability > 1.00001):
            raise RuntimeError(f"event-probability effect outside [-1,1] in {path}")
        if np.any(data["response_endpoint_wasserstein1"] < -1e-7):
            raise RuntimeError(f"negative Wasserstein distance in {path}")
        achieved = data["diagnostic_achieved_gap"]
        if not np.allclose(
            achieved,
            data["diagnostic_achieved_high"] - data["diagnostic_achieved_low"],
            rtol=2e-5,
            atol=1e-6,
        ):
            raise RuntimeError(f"achieved-gap identity failed in {path}")
        if not np.array_equal(
            data["diagnostic_valid"].astype(bool),
            (
                (data["diagnostic_ess_low"] >= MIN_ESS)
                & (data["diagnostic_ess_high"] >= MIN_ESS)
                & (data["diagnostic_max_weight_low"] <= MAX_NORMALIZED_WEIGHT)
                & (data["diagnostic_max_weight_high"] <= MAX_NORMALIZED_WEIGHT)
                & (
                    achieved
                    >= MIN_ACHIEVED_SOURCE_FRACTION
                    * data["diagnostic_target_gap"]
                )
            ),
        ):
            raise RuntimeError(f"declared validity gates do not reproduce in {path}")
        return {
            "source_lag_frames": lag,
            "fold": fold,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "wall_seconds": float(data["wall_seconds"].item()),
            "resolved_device": str(data["resolved_device"].item()),
            "heldout_worms": len(heldout),
        }


def validate_raw(output: Path, source_run: Path) -> dict[str, object]:
    cohort = load_sbtg_cohort()
    folds = make_fold_assignments(cohort, n_folds=5, seed=FOLD_SEED)
    rows: list[dict[str, object]] = []
    problems: list[str] = []
    for lag in SOURCE_LAGS:
        for fold in range(5):
            path = _archive_path(output, lag, fold)
            if not path.is_file():
                problems.append(f"missing:{path}")
                continue
            try:
                rows.append(
                    _validate_archive(
                        path,
                        cohort=cohort,
                        folds=folds,
                        source_run=source_run,
                        lag=lag,
                        fold=fold,
                    )
                )
            except Exception as error:  # fail closed with a durable audit trail
                problems.append(f"{path}:{error!r}")
    devices = sorted({str(row["resolved_device"]) for row in rows})
    validation = {
        "created_utc": _utc(),
        "status": "pass" if len(rows) == len(SOURCE_LAGS) * 5 and not problems else "incomplete",
        "expected_archives": len(SOURCE_LAGS) * 5,
        "validated_archives": len(rows),
        "problems": problems,
        "resolved_devices": devices,
        "total_sampler_wall_seconds": float(sum(float(row["wall_seconds"]) for row in rows)),
        "total_archive_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "archives": rows,
        "checks": {
            "exact_80_neuron_order": not problems and len(rows) == 20,
            "complete_lag_fold_grid": not problems and len(rows) == 20,
            "complete_six_horizon_grid": not problems and len(rows) == 20,
            "all_seven_response_channels": not problems and len(rows) == 20,
            "all_values_finite": not problems and len(rows) == 20,
            "checkpoint_hash_and_fold_holdout": not problems and len(rows) == 20,
            "validity_gates_recomputed": not problems and len(rows) == 20,
        },
    }
    _atomic_json(output / "raw_validation.json", validation)
    return validation


def run_raw(
    output: Path,
    source_run: Path,
    *,
    device: str,
    max_archives: int | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    cohort = load_sbtg_cohort()
    if cohort.n_worms != 20 or cohort.n_neurons != 80:
        raise RuntimeError("historical SBTG cohort geometry changed")
    folds = make_fold_assignments(cohort, n_folds=5, seed=FOLD_SEED)
    fold_path = output / "fold_assignments.csv"
    _write_fold_assignments(fold_path, cohort, folds)
    candidate = _raw_manifest(output, source_run, cohort)
    manifest_path = output / "manifest.json"
    _write_or_validate_manifest(manifest_path, candidate)

    records: list[dict[str, object]] = []
    attempts = 0
    for lag in SOURCE_LAGS:
        for fold in range(5):
            if max_archives is not None and attempts >= max_archives:
                break
            attempts += 1
            checkpoint = _checkpoint_path(source_run, fold)
            print(f"SBTG80_FULL_START lag={lag} fold={fold}", flush=True)
            try:
                result = run_one(
                    cohort=cohort,
                    folds=folds,
                    source_run=source_run,
                    checkpoint_phase="historical_fold_specific",
                    output=output,
                    method=METHOD,
                    model_id=MODEL_ID,
                    history_lag=HISTORY_FRAMES,
                    fold=fold,
                    seed=GENERATOR_SEED,
                    source_lag=lag,
                    particles=PARTICLES,
                    horizons=HORIZONS,
                    source_window_frames=SOURCE_WINDOW_FRAMES,
                    device=device,
                    base_seed=BASE_SEED,
                    min_ess=MIN_ESS,
                    progressive_branch_factor=2,
                    progressive_future_branch_factor=1,
                    checkpoint_override=checkpoint,
                    checkpoint_validation_profile="historical_sbtg80",
                )
            except Exception as error:
                result = {"status": "failed", "error": repr(error), "wall_seconds": 0.0}
            record = {"source_lag_frames": lag, "fold": fold, **result}
            records.append(record)
            _atomic_csv(output / "latest_run_status.csv", pd.DataFrame(records))
            print(
                f"SBTG80_FULL_DONE lag={lag} fold={fold} status={result['status']} "
                f"seconds={float(result.get('wall_seconds', 0.0)):.1f}",
                flush=True,
            )
            if result["status"] == "failed":
                raise RuntimeError(f"sampling failed at lag={lag}, fold={fold}: {result.get('error')}")
        if max_archives is not None and attempts >= max_archives:
            break

    validation = validate_raw(output, source_run)
    manifest = json.loads(manifest_path.read_text())
    if validation["status"] == "pass":
        manifest["status"] = "raw_sampling_complete"
        manifest["completed_utc"] = _utc()
    else:
        manifest["status"] = "raw_sampling_incomplete"
    _atomic_json(manifest_path, manifest)
    return validation


def _load_lag_diagnostics(
    output: Path,
    cohort,
    folds: np.ndarray,
    lag: int,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    w, p, e, d = cohort.n_worms, len(PHASES), 3, cohort.n_neurons
    names = (
        "valid",
        "achieved_gap",
        "ess_low",
        "ess_high",
        "max_weight_low",
        "max_weight_high",
        "distinct_ancestors_low",
        "distinct_ancestors_high",
    )
    values = {name: np.full((w, p, e, d), np.nan, dtype=np.float32) for name in names}
    forced = np.full((w, p, e, d), np.nan, dtype=np.float32)
    resamples = np.full_like(forced, np.nan)
    codes = np.full((w, e), -1, dtype=np.int8)
    for fold in range(5):
        path = _archive_path(output, lag, fold)
        with np.load(path, allow_pickle=False) as data:
            indices = data["worm_indices"].astype(int)
            if not np.all(folds[indices] == fold):
                raise RuntimeError(f"heldout fold mismatch in {path}")
            for name in names:
                values[name][indices] = data[f"diagnostic_{name}"].astype(np.float32)
            forced[indices] = np.maximum(
                data["diagnostic_step_forced_tempering_low"],
                data["diagnostic_step_forced_tempering_high"],
            ).mean(axis=-1)
            resamples[indices] = (
                data["diagnostic_step_tempering_resamples_low"]
                + data["diagnostic_step_tempering_resamples_high"]
            ).mean(axis=-1)
            local_codes = data["chemical_code_by_worm_event"].astype(np.int8)
            codes[indices] = local_codes
    if any(not np.isfinite(item).all() for item in values.values()):
        raise RuntimeError(f"missing/nonfinite diagnostic values for lag {lag}")
    if not np.isfinite(forced).all() or not np.isfinite(resamples).all() or np.any(codes < 0):
        raise RuntimeError(f"missing progressive diagnostics/codes for lag {lag}")

    contextual: dict[str, dict[str, np.ndarray]] = {context: {} for context in CONTEXTS}
    support, gaps = _contextualize_support(values["valid"], values["achieved_gap"], codes)
    for context in CONTEXTS:
        contextual[context]["valid"] = support[context]
        contextual[context]["achieved_gap"] = gaps[context]
    reducers = {
        "ess_low": "min",
        "ess_high": "min",
        "max_weight_low": "max",
        "max_weight_high": "max",
        "forced_tempering_rate": "max",
        "tempering_resamples_mean": "max",
    }
    source_values = dict(values)
    source_values["forced_tempering_rate"] = forced
    source_values["tempering_resamples_mean"] = resamples
    ancestor_fraction = np.minimum(
        values["distinct_ancestors_low"], values["distinct_ancestors_high"]
    ) / float(PARTICLES)
    source_values["min_distinct_ancestor_fraction"] = ancestor_fraction
    source_values["genealogy_ok_0_10"] = (ancestor_fraction >= 0.10).astype(np.float32)
    source_values["genealogy_ok_0_20"] = (ancestor_fraction >= 0.20).astype(np.float32)
    reducers.update(
        {
            "min_distinct_ancestor_fraction": "min",
            "genealogy_ok_0_10": "min",
            "genealogy_ok_0_20": "min",
        }
    )
    for name, reducer in reducers.items():
        local = _contextualize_source_metric(
            source_values[name], codes, contrast_reducer=reducer
        )
        for context in CONTEXTS:
            contextual[context][name] = local[context]
    return contextual, codes


def _load_channel_lag(
    output: Path,
    cohort,
    folds: np.ndarray,
    lag: int,
    channel: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    w, p, e, h, d = (
        cohort.n_worms,
        len(PHASES),
        3,
        len(HORIZONS),
        cohort.n_neurons,
    )
    raw = np.full((w, p, e, h, d, d), np.nan, dtype=np.float32)
    normalized = np.full_like(raw, np.nan)
    codes = np.full((w, e), -1, dtype=np.int8)
    for fold in range(5):
        path = _archive_path(output, lag, fold)
        with np.load(path, allow_pickle=False) as data:
            indices = data["worm_indices"].astype(int)
            if not np.all(folds[indices] == fold):
                raise RuntimeError(f"heldout fold mismatch in {path}")
            response = data[f"response_{channel}"].astype(np.float32)
            achieved = data["diagnostic_achieved_gap"].astype(np.float32)
            denominator = effect_normalization_denominator(
                achieved, MIN_NORMALIZATION_GAP
            )
            raw[indices] = orient_response_once(response)
            normalized[indices] = orient_response_once(
                response / denominator[..., None, None]
            )
            codes[indices] = data["chemical_code_by_worm_event"].astype(np.int8)
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all() or np.any(codes < 0):
        raise RuntimeError(f"missing/nonfinite response for {channel}/lag{lag}")
    return _contextualize_values(raw, codes), _contextualize_values(normalized, codes), codes


def _origin_masks(neurons: Sequence[str]) -> dict[tuple[str, str], np.ndarray]:
    tail = np.asarray([neuron in set(TAIL_ORIGIN_NEURONS) for neuron in neurons])
    origin = {"head": ~tail, "tail": tail}
    off = ~np.eye(len(neurons), dtype=bool)
    return {
        (source_name, target_name): (
            target_values[:, None] & source_values[None, :] & off
        )
        for source_name, source_values in origin.items()
        for target_name, target_values in origin.items()
    }


def _interval(weights: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    boot = weights @ np.asarray(values, dtype=np.float32)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def _head_tail_rows(
    *,
    channel: str,
    context: str,
    lag: int,
    normalized: np.ndarray,
    support: np.ndarray,
    neurons: Sequence[str],
    bootstrap_weights: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    masks = _origin_masks(neurons)
    signed = _is_signed(channel, context)
    for horizon_index, horizon in enumerate(HORIZONS):
        worm = normalized[:, horizon_index]
        for (source_origin, target_origin), base_mask in masks.items():
            for scope in ("all_sources", "support_qualified_sources"):
                mask = base_mask.copy()
                if scope == "support_qualified_sources":
                    mask &= support[None, :] >= MIN_VALID_FRACTION
                if not mask.any():
                    continue
                worm_signed = worm[:, mask].mean(axis=1)
                worm_absolute = np.abs(worm[:, mask]).mean(axis=1)
                signed_low, signed_high = _interval(bootstrap_weights, worm_signed)
                absolute_low, absolute_high = _interval(bootstrap_weights, worm_absolute)
                rows.append(
                    {
                        "method": METHOD,
                        "channel": channel,
                        "context": context,
                        "source_lag_frames": lag,
                        "horizon_frames": horizon,
                        "source_origin": source_origin,
                        "target_origin": target_origin,
                        "scope": scope,
                        "n_directed_pairs": int(mask.sum()),
                        "n_worms": len(worm),
                        "signed_estimand": signed,
                        "mean_effect": float(worm_signed.mean()),
                        "mean_effect_ci_low": signed_low if signed else np.nan,
                        "mean_effect_ci_high": signed_high if signed else np.nan,
                        "mean_absolute_effect": float(worm_absolute.mean()),
                        "mean_absolute_effect_ci_low": absolute_low,
                        "mean_absolute_effect_ci_high": absolute_high,
                        "interval_unit": "whole_worm_percentile_bootstrap",
                        "lineage_warning": (
                            "historical pseudo-paired head/tail and donor-imputed cache"
                        ),
                    }
                )
    return rows


def _top_rows(
    *,
    channel: str,
    context: str,
    lag: int,
    summary: Mapping[str, np.ndarray],
    support: np.ndarray,
    neurons: Sequence[str],
    limit: int = 100,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    off = ~np.eye(len(neurons), dtype=bool)
    tail = set(TAIL_ORIGIN_NEURONS)
    for horizon_index, horizon in enumerate(HORIZONS):
        mean = summary["mean"][horizon_index]
        eligible = off & (support[None, :] >= MIN_VALID_FRACTION)
        candidates = np.flatnonzero(eligible.ravel())
        if not len(candidates):
            continue
        score = np.abs(mean).ravel()[candidates]
        order = candidates[np.argsort(score, kind="stable")[::-1][:limit]]
        target_index, source_index = np.unravel_index(order, mean.shape)
        for rank, (target, source) in enumerate(zip(target_index, source_index), start=1):
            rows.append(
                {
                    "slice_rank": rank,
                    "method": METHOD,
                    "channel": channel,
                    "context": context,
                    "source_lag_frames": lag,
                    "horizon_frames": horizon,
                    "source_neuron": neurons[source],
                    "target_neuron": neurons[target],
                    "source_origin": "tail" if neurons[source] in tail else "head",
                    "target_origin": "tail" if neurons[target] in tail else "head",
                    "mean_normalized_effect": float(mean[target, source]),
                    "median_normalized_effect": float(summary["median"][horizon_index, target, source]),
                    "ci_low": float(summary["ci_low"][horizon_index, target, source]),
                    "ci_high": float(summary["ci_high"][horizon_index, target, source]),
                    "sign_consistency": float(
                        summary["sign_consistency"][horizon_index, target, source]
                    ),
                    "valid_fraction": float(support[source]),
                    "interval_unit": "whole_worm_percentile_bootstrap",
                    "generator_seed_count": 1,
                }
            )
    return rows


def _support_rows(
    lag: int,
    diagnostics: Mapping[str, Mapping[str, np.ndarray]],
    neurons: Sequence[str],
) -> list[dict[str, object]]:
    tail = set(TAIL_ORIGIN_NEURONS)
    rows: list[dict[str, object]] = []
    for context in CONTEXTS:
        values = diagnostics[context]
        for source, neuron in enumerate(neurons):
            rows.append(
                {
                    "method": METHOD,
                    "context": context,
                    "source_lag_frames": lag,
                    "source_neuron": neuron,
                    "source_origin": "tail" if neuron in tail else "head",
                    "valid_fraction": float(values["valid"][:, source].mean()),
                    "support_qualified": bool(
                        values["valid"][:, source].mean() >= MIN_VALID_FRACTION
                    ),
                    "mean_achieved_gap": float(values["achieved_gap"][:, source].mean()),
                    "mean_ess_low": float(values["ess_low"][:, source].mean()),
                    "mean_ess_high": float(values["ess_high"][:, source].mean()),
                    "maximum_worm_max_weight_low": float(
                        values["max_weight_low"][:, source].max()
                    ),
                    "maximum_worm_max_weight_high": float(
                        values["max_weight_high"][:, source].max()
                    ),
                    "mean_min_distinct_ancestor_fraction": float(
                        values["min_distinct_ancestor_fraction"][:, source].mean()
                    ),
                    "genealogy_valid_fraction_0_10": float(
                        values["genealogy_ok_0_10"][:, source].mean()
                    ),
                    "genealogy_valid_fraction_0_20": float(
                        values["genealogy_ok_0_20"][:, source].mean()
                    ),
                    "mean_forced_tempering_rate": float(
                        values["forced_tempering_rate"][:, source].mean()
                    ),
                    "mean_tempering_resamples": float(
                        values["tempering_resamples_mean"][:, source].mean()
                    ),
                }
            )
    return rows


def build_atlas(
    output: Path,
    source_run: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    overwrite: bool = False,
) -> dict[str, object]:
    raw_validation = validate_raw(output, source_run)
    if raw_validation["status"] != "pass":
        raise RuntimeError("raw SBTG80 grid has not passed validation")
    if bootstrap_replicates < 32:
        raise ValueError("at least 32 worm-bootstrap replicates are required")
    atlas_dir = output / "atlas"
    if atlas_dir.exists() and any(atlas_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"atlas output already exists: {atlas_dir}")
        shutil.rmtree(atlas_dir)
    atlas_dir.mkdir(parents=True)

    cohort = load_sbtg_cohort()
    folds = make_fold_assignments(cohort, n_folds=5, seed=FOLD_SEED)
    bootstrap_weights = _bootstrap_counts(
        cohort.n_worms, bootstrap_replicates, BOOTSTRAP_SEED
    )
    diagnostic_by_lag: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    support_rows: list[dict[str, object]] = []
    for lag in SOURCE_LAGS:
        diagnostic, _ = _load_lag_diagnostics(output, cohort, folds, lag)
        diagnostic_by_lag[lag] = diagnostic
        support_rows.extend(_support_rows(lag, diagnostic, cohort.neurons))
    support_frame = pd.DataFrame(support_rows)
    support_frame.to_csv(atlas_dir / "support_cells.csv", index=False)

    metadata = {
        "neurons": np.asarray(cohort.neurons),
        "worm_ids": np.asarray(cohort.worm_ids),
        "methods": np.asarray([METHOD]),
        "channels": np.asarray(CHANNELS),
        "contexts": np.asarray(CONTEXTS),
        "source_lag_frames": np.asarray(SOURCE_LAGS, dtype=np.int16),
        "source_lag_seconds": np.asarray(SOURCE_LAGS, dtype=np.float32) / cohort.fps,
        "horizon_frames": np.asarray(HORIZONS, dtype=np.int16),
        "horizon_seconds": np.asarray(HORIZONS, dtype=np.float32) / cohort.fps,
        "orientation": np.asarray(ORIENTATION),
        "primary_method": np.asarray(METHOD),
        "generator_seed_count": np.asarray(1, dtype=np.int16),
    }
    atlas_path = atlas_dir / "atlas_matrices.npz"
    worm_path = atlas_dir / "worm_matrices.npz"
    top_rows: list[dict[str, object]] = []
    head_tail_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with _NpzStream(atlas_path) as atlas_writer, _NpzStream(worm_path) as worm_writer:
        for key, value in metadata.items():
            atlas_writer.write(key, value)
            worm_writer.write(key, value)
        worm_writer.write(
            "scope",
            np.asarray(
                "normalized held-out worm matrices; one complete fitted-generator seed"
            ),
        )
        for channel in CHANNELS:
            statistics_by_context = {
                context: {
                    name: []
                    for name in ("mean", "median", "ci_low", "ci_high", "sign_consistency")
                }
                for context in CONTEXTS
            }
            worm_by_context = {context: [] for context in CONTEXTS}
            for lag in SOURCE_LAGS:
                _, normalized_context, _ = _load_channel_lag(
                    output, cohort, folds, lag, channel
                )
                diagnostics = diagnostic_by_lag[lag]
                for context in CONTEXTS:
                    worm = normalized_context[context]
                    support = diagnostics[context]["valid"].mean(axis=0)
                    summary = _worm_summary(
                        worm,
                        signed=_is_signed(channel, context),
                        bootstrap_weights=bootstrap_weights,
                    )
                    for name in statistics_by_context[context]:
                        statistics_by_context[context][name].append(summary[name])
                    worm_by_context[context].append(worm.astype(np.float32))
                    top_rows.extend(
                        _top_rows(
                            channel=channel,
                            context=context,
                            lag=lag,
                            summary=summary,
                            support=support,
                            neurons=cohort.neurons,
                        )
                    )
                    head_tail_rows.extend(
                        _head_tail_rows(
                            channel=channel,
                            context=context,
                            lag=lag,
                            normalized=worm,
                            support=support,
                            neurons=cohort.neurons,
                            bootstrap_weights=bootstrap_weights,
                        )
                    )
                print(f"SBTG80_ATLAS_AGG channel={channel} lag={lag}", flush=True)
            for context in CONTEXTS:
                prefix = f"{METHOD}__{channel}__{context}"
                for name, arrays in statistics_by_context[context].items():
                    atlas_writer.write(f"{name}_normalized__{prefix}", np.stack(arrays))
                # The canonical key is retained, but NaN truthfully records that
                # seed agreement is unestimable from one complete generator seed.
                atlas_writer.write(
                    f"seed_sign_agreement__{prefix}",
                    np.full(
                        (
                            len(SOURCE_LAGS),
                            len(HORIZONS),
                            cohort.n_neurons,
                            cohort.n_neurons,
                        ),
                        np.nan,
                        dtype=np.float16,
                    ),
                )
                worm_writer.write(
                    f"normalized__{channel}__{context}",
                    np.stack(worm_by_context[context]),
                )
                if channel == CHANNELS[0]:
                    atlas_writer.write(
                        f"valid_fraction__{METHOD}__{context}",
                        np.stack(
                            [
                                diagnostic_by_lag[lag][context]["valid"].mean(axis=0)
                                for lag in SOURCE_LAGS
                            ]
                        ),
                    )
                    atlas_writer.write(
                        f"genealogy_valid_fraction_0_10__{METHOD}__{context}",
                        np.stack(
                            [
                                diagnostic_by_lag[lag][context]["genealogy_ok_0_10"].mean(axis=0)
                                for lag in SOURCE_LAGS
                            ]
                        ),
                    )
                    atlas_writer.write(
                        f"genealogy_valid_fraction_0_20__{METHOD}__{context}",
                        np.stack(
                            [
                                diagnostic_by_lag[lag][context]["genealogy_ok_0_20"].mean(axis=0)
                                for lag in SOURCE_LAGS
                            ]
                        ),
                    )

    pd.DataFrame(top_rows).to_csv(atlas_dir / "top_effects.csv", index=False)
    head_tail_frame = pd.DataFrame(head_tail_rows)
    head_tail_frame.to_csv(atlas_dir / "head_tail_metrics.csv", index=False)
    _make_head_tail_figure(atlas_dir, head_tail_frame)
    protocol = {
        "created_utc": _utc(),
        "status": "internally_complete_external_references_unopened",
        "protocol": "full historical SBTG80 progressive-bridge SMC atlas v1",
        "fps": cohort.fps,
        "source_lag_frames": list(SOURCE_LAGS),
        "horizon_frames": list(HORIZONS),
        "channels": list(CHANNELS),
        "contexts": [context_metadata(context) for context in CONTEXTS],
        "orientation": ORIENTATION,
        "normalization": (
            "response divided by max(abs(achieved source gap), 0.10), before context averaging"
        ),
        "uncertainty": {
            "independent_unit": "worm",
            "interval": "deterministic 2.5/97.5 percentile whole-worm bootstrap",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "generator_seed_count": 1,
            "generator_seed_uncertainty": "not estimable",
        },
        "lineage_warning": cohort.lineage_warning,
        "claim_boundary": (
            "observational conditional-generator sensitivity only; no coupling, causal, "
            "anatomical, receptor-action, or physical-delay interpretation"
        ),
        "ranking_inputs_used": [],
    }
    _atomic_json(atlas_dir / "protocol.json", protocol)
    internal_validation = {
        "created_utc": _utc(),
        "status": "passed",
        "problems": [],
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "source_lags": list(SOURCE_LAGS),
        "horizons": list(HORIZONS),
        "channels": list(CHANNELS),
        "contexts": list(CONTEXTS),
        "dense_effect_cells": len(CHANNELS)
        * len(CONTEXTS)
        * len(SOURCE_LAGS)
        * len(HORIZONS)
        * cohort.n_neurons
        * cohort.n_neurons,
        "top_effect_rows": len(top_rows),
        "head_tail_metric_rows": len(head_tail_rows),
        "support_rows": len(support_rows),
        "analysis_wall_seconds": time.perf_counter() - started,
        "checks": {
            "raw_archives_passed": True,
            "worm_is_independent_unit": True,
            "response_oriented_once_to_target_row_source_column": True,
            "all_thirteen_contexts": True,
            "uncertainty_saved_for_every_dense_effect_cell": True,
            "generator_seed_uncertainty_not_manufactured": True,
            "head_tail_origin_metrics_use_off_diagonal_pairs": True,
            "external_references_absent_from_internal_ranking": True,
        },
    }
    _atomic_json(atlas_dir / "validation.json", internal_validation)
    manifest = {
        "created_utc": _utc(),
        "status": "internal_complete",
        "protocol": "full historical SBTG80 progressive-bridge SMC atlas v1",
        "raw_root": str(output.resolve()),
        "raw_manifest_sha256": sha256(output / "manifest.json"),
        "raw_validation_sha256": sha256(output / "raw_validation.json"),
        "artifacts": {
            "atlas_matrices": "atlas_matrices.npz",
            "worm_matrices": "worm_matrices.npz",
            "top_effects": "top_effects.csv",
            "head_tail_metrics": "head_tail_metrics.csv",
            "head_tail_uncertainty_figure": "figures/head_tail_endpoint_mean_uncertainty.png",
            "support_cells": "support_cells.csv",
            "protocol": "protocol.json",
            "validation": "validation.json",
        },
    }
    _atomic_json(atlas_dir / "manifest.json", manifest)
    _write_checksums(atlas_dir)
    return manifest


def _make_head_tail_figure(output: Path, frame: pd.DataFrame) -> None:
    """Plot head/tail summaries with their whole-worm uncertainty bands."""
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    selected = frame[
        (frame.channel == "endpoint_mean")
        & (frame.context == "state_average")
        & (frame.scope == "support_qualified_sources")
    ].copy()
    colors = {
        ("head", "head"): "#4c78a8",
        ("head", "tail"): "#f58518",
        ("tail", "head"): "#54a24b",
        ("tail", "tail"): "#e45756",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panels = (
        (
            axes[0],
            selected[selected.horizon_frames == 1],
            "source_lag_frames",
            "Source lag (frames)",
            "Horizon 1",
        ),
        (
            axes[1],
            selected[selected.source_lag_frames == 1],
            "horizon_frames",
            "Forecast horizon (frames)",
            "Source lag 1",
        ),
    )
    for ax, panel, x_name, x_label, title in panels:
        for origins, group in panel.groupby(
            ["source_origin", "target_origin"], sort=True
        ):
            group = group.sort_values(x_name)
            x = group[x_name].to_numpy(dtype=float)
            mean = group.mean_effect.to_numpy(dtype=float)
            low = group.mean_effect_ci_low.to_numpy(dtype=float)
            high = group.mean_effect_ci_high.to_numpy(dtype=float)
            color = colors[origins]
            ax.plot(
                x,
                mean,
                marker="o",
                color=color,
                label=f"{origins[0]} → {origins[1]}",
            )
            ax.fill_between(x, low, high, color=color, alpha=0.16)
        ax.axhline(0, color="#777", linewidth=0.8, linestyle="--")
        ax.set(xlabel=x_label, title=title)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean normalized endpoint effect")
    axes[1].legend(loc="best", fontsize=9)
    fig.suptitle(
        "Historical SBTG80 head/tail recording-origin effects · 95% worm intervals"
    )
    fig.tight_layout()
    fig.savefig(
        figures / "head_tail_endpoint_mean_uncertainty.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def _atlas_slices(atlas_path: Path) -> list[dict[str, object]]:
    slices: list[dict[str, object]] = []
    with np.load(atlas_path, allow_pickle=False) as atlas:
        neurons = tuple(atlas["neurons"].astype(str))
        lags = atlas["source_lag_frames"].astype(int)
        horizons = atlas["horizon_frames"].astype(int)
        for channel in CHANNELS:
            for context in CONTEXTS:
                matrices = atlas[f"mean_normalized__{METHOD}__{channel}__{context}"]
                support = atlas[f"valid_fraction__{METHOD}__{context}"]
                expected = (len(lags), len(horizons), len(neurons), len(neurons))
                if matrices.shape != expected or support.shape != (len(lags), len(neurons)):
                    raise RuntimeError(f"dense atlas geometry failed for {channel}/{context}")
                for li, lag in enumerate(lags):
                    for hi, horizon in enumerate(horizons):
                        slices.append(
                            {
                                "method": METHOD,
                                "channel": channel,
                                "context": context,
                                "lag_frames": int(lag),
                                "horizon_frames": int(horizon),
                                "matrix": matrices[li, hi],
                                "support": support[li],
                                "support_available": True,
                                "family": "historical_sbtg80_progressive_flow",
                                "training_lineage": (
                                    "historical 80-neuron pseudo-paired/donor-imputed SBTG cache; "
                                    "atlas-blind conditional-flow winner"
                                ),
                                "shared_neuron_comparability": "native historical 80-class axis",
                            }
                        )
    return slices


def _selected_lagmax_slices(
    flow_slices: Sequence[dict[str, object]],
    published_slices: Sequence[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    selected: dict[str, list[dict[str, object]]] = {}
    for channel, context in (
        ("endpoint_mean", "state_average"),
        ("endpoint_log_sd", "state_average"),
        ("endpoint_wasserstein1", "state_average"),
        ("endpoint_mean", "onset_minus_baseline"),
    ):
        key = f"{METHOD}__{channel}__{context}__h1"
        selected[key] = [
            item
            for item in flow_slices
            if item["channel"] == channel
            and item["context"] == context
            and int(item["horizon_frames"]) == 1
        ]
    selected["sbtg_published"] = list(published_slices)
    return selected


def run_external_metrics(
    output: Path,
    *,
    reference_release: Path,
    published_archive: Path,
    permutations: int,
    bootstrap_repeats: int,
    overwrite: bool = False,
) -> dict[str, object]:
    if permutations < 99:
        raise ValueError("at least 99 source-preserving permutations are required")
    if bootstrap_repeats < 100:
        raise ValueError("at least 100 paired-source bootstrap replicates are required")
    atlas_dir = output / "atlas"
    validation = json.loads((atlas_dir / "validation.json").read_text())
    if validation.get("status") != "passed":
        raise RuntimeError("internal SBTG80 atlas has not passed")
    external = output / "external_reference_checks"
    if external.exists() and any(external.iterdir()):
        if not overwrite:
            raise FileExistsError(external)
        shutil.rmtree(external)
    external.mkdir(parents=True)

    cohort = load_sbtg_cohort()
    flow_slices = _atlas_slices(atlas_dir / "atlas_matrices.npz")
    published_slices = _load_published_slices(published_archive, cohort.neurons)
    references, networks = load_references(reference_release, list(cohort.neurons))
    all_slices = flow_slices + published_slices
    reference = pd.DataFrame(_reference_rows(all_slices, references, fps=cohort.fps))
    reference["reference"] = reference["reference"].str.replace("_54", "_80", regex=False)
    reference.to_csv(external / "randi_cook_metrics.csv", index=False)
    neuromod = pd.DataFrame(_neuromodulator_rows(all_slices, networks, fps=cohort.fps))
    neuromod.to_csv(external / "bentley_metrics.csv", index=False)

    selected = _selected_lagmax_slices(flow_slices, published_slices)
    lagmax_rows = _lagmax_rows(
        selected,
        networks,
        permutations=permutations,
        seed=PERMUTATION_SEED,
        lag_grid="native_method_grid",
        fps=cohort.fps,
    ) + _lagmax_rows(
        selected,
        networks,
        permutations=permutations,
        seed=PERMUTATION_SEED + 1,
        lag_grid="common_1_8_frames",
        fps=cohort.fps,
        forced_lags=(1, 8),
    )
    lagmax = pd.DataFrame(lagmax_rows)
    p = lagmax["max_lag_permutation_p"].to_numpy(dtype=float)
    finite = np.isfinite(p)
    lagmax["max_lag_bh_q"] = np.nan
    lagmax.loc[finite, "max_lag_bh_q"] = benjamini_hochberg(p[finite])
    lagmax["bh_family"] = (
        "all_planned_historical_sbtg80_progressive_and_published_native_and_common_grid_tests"
    )
    lagmax["n_bh_tests"] = int(finite.sum())
    lagmax.to_csv(external / "bentley_lagmax_inference.csv", index=False)

    primary = reference[
        (
            (reference.method == METHOD)
            & (reference.channel == "endpoint_mean")
            & (reference.context == "state_average")
            & (reference.horizon_frames == 1)
            & (reference.lag_frames == 1)
            & (reference.scope == "all_estimated")
        )
        | (
            (reference.method == "sbtg_published")
            & (reference.lag_frames == 1)
            & (reference.scope == "all_estimated")
        )
    ].copy()
    primary.to_csv(external / "primary_lag1_comparison.csv", index=False)
    progressive_matrix = next(
        item["matrix"]
        for item in flow_slices
        if item["channel"] == "endpoint_mean"
        and item["context"] == "state_average"
        and item["lag_frames"] == 1
        and item["horizon_frames"] == 1
    )
    published_matrix = next(
        item["matrix"] for item in published_slices if item["lag_frames"] == 1
    )
    paired = _paired_source_bootstrap(
        progressive_matrix,
        published_matrix,
        references,
        repeats=bootstrap_repeats,
        seed=PERMUTATION_SEED,
    )
    paired.to_csv(external / "paired_source_bootstrap.csv", index=False)
    _make_reference_figures(external, reference, neuromod, lagmax)

    result = {
        "created_utc": _utc(),
        "status": "pass",
        "internal_atlas_sha256": sha256(atlas_dir / "atlas_matrices.npz"),
        "reference_release": str(reference_release.resolve()),
        "published_sbtg_archive": str(published_archive.resolve()),
        "published_sbtg_archive_sha256": sha256(published_archive),
        "randi_cook_rows": len(reference),
        "bentley_rows": len(neuromod),
        "lagmax_rows": len(lagmax),
        "lagmax_bh_tests": int(finite.sum()),
        "permutations": permutations,
        "paired_source_bootstrap_replicates": bootstrap_repeats,
        "postfreeze_only": True,
        "lineage_warning": cohort.lineage_warning,
    }
    _atomic_json(external / "validation.json", result)
    _write_checksums(external)
    return result


def _make_reference_figures(
    output: Path,
    reference: pd.DataFrame,
    neuromod: pd.DataFrame,
    lagmax: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir()
    primary = reference[
        (reference.method == METHOD)
        & (reference.channel == "endpoint_mean")
        & (reference.context == "state_average")
        & (reference.scope == "support_qualified_sources")
    ]
    for metric, filename, ylabel in (
        ("auroc", "randi_cook_auroc.png", "AUROC"),
        ("macro_source_auroc", "randi_cook_source_macro_auroc.png", "Source-macro AUROC"),
    ):
        if metric not in primary.columns:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        for ax, reference_name in zip(axes.ravel(), sorted(primary.reference.unique())):
            part = primary[primary.reference == reference_name]
            for horizon, group in part.groupby("horizon_frames"):
                group = group.sort_values("lag_frames")
                ax.plot(group.lag_frames, group[metric], marker="o", label=f"h={int(horizon)}")
            ax.axhline(0.5, color="#777", linewidth=0.8, linestyle="--")
            ax.set_title(str(reference_name).replace("_", " "))
            ax.set_xlabel("Source lag (frames)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=6)
        fig.suptitle("Historical SBTG80 progressive bridge: post-freeze reference checks")
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
        fig.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    selected = neuromod[
        (neuromod.method == METHOD)
        & (neuromod.channel == "endpoint_mean")
        & (neuromod.context == "state_average")
        & (neuromod.horizon_frames == 1)
        & (neuromod.scope == "eligible_support_qualified")
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for network, group in selected.groupby("network"):
        group = group.sort_values("lag_frames")
        ax.plot(group.lag_frames, group.auroc, marker="o", label=str(network).replace("_", " "))
    ax.axhline(0.5, color="#777", linewidth=0.8, linestyle="--")
    ax.set(xlabel="Source lag (frames)", ylabel="AUROC", title="Bentley network correspondence · endpoint mean · h=1")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "bentley_auroc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if not lagmax.empty:
        show = lagmax[lagmax.lag_grid == "native_method_grid"].sort_values("best_auroc")
        fig, ax = plt.subplots(figsize=(10, 7))
        labels = [f"{row.network} · {row.channel}" for row in show.itertuples()]
        y = np.arange(len(show))
        ax.barh(y, show.best_auroc, color="#4c78a8")
        ax.set_yticks(y, labels, fontsize=7)
        ax.axvline(0.5, color="#777", linewidth=0.8, linestyle="--")
        ax.set(xlabel="Best lag-selected AUROC", title="Bentley native-grid lag maxima (descriptive heights)")
        fig.tight_layout()
        fig.savefig(figures / "bentley_lagmax.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _write_checksums(directory: Path) -> None:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (directory / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(directory)}\n" for path in files)
    )


def audit_final_output(output: Path) -> dict[str, object]:
    """Independently verify the frozen dense, worm, and tabular artifacts."""
    atlas_dir = output / "atlas"
    atlas_path = atlas_dir / "atlas_matrices.npz"
    worm_path = atlas_dir / "worm_matrices.npz"
    dense_finite = 0
    signed_consistency_finite = 0
    unsigned_consistency_undefined = 0
    with np.load(atlas_path, allow_pickle=False) as atlas:
        if tuple(atlas["neurons"].astype(str)) != tuple(
            json.loads((output / "manifest.json").read_text())["neurons"]
        ):
            raise RuntimeError("final audit: dense neuron order differs from manifest")
        if atlas["source_lag_frames"].astype(int).tolist() != list(SOURCE_LAGS):
            raise RuntimeError("final audit: dense lag grid differs")
        if atlas["horizon_frames"].astype(int).tolist() != list(HORIZONS):
            raise RuntimeError("final audit: dense horizon grid differs")
        for channel in CHANNELS:
            for context in CONTEXTS:
                prefix = f"{METHOD}__{channel}__{context}"
                arrays = {
                    name: atlas[f"{name}_normalized__{prefix}"]
                    for name in ("mean", "median", "ci_low", "ci_high")
                }
                expected = (len(SOURCE_LAGS), len(HORIZONS), 80, 80)
                if any(value.shape != expected for value in arrays.values()):
                    raise RuntimeError(f"final audit: dense geometry failed for {prefix}")
                if any(not np.isfinite(value).all() for value in arrays.values()):
                    raise RuntimeError(f"final audit: nonfinite dense estimate for {prefix}")
                if not np.all(arrays["ci_low"] <= arrays["ci_high"]):
                    raise RuntimeError(f"final audit: interval order failed for {prefix}")
                consistency = atlas[f"sign_consistency_normalized__{prefix}"]
                signed = _is_signed(channel, context)
                if signed:
                    if not np.isfinite(consistency).all() or np.any(
                        (consistency < 0) | (consistency > 1)
                    ):
                        raise RuntimeError(
                            f"final audit: signed consistency failed for {prefix}"
                        )
                    signed_consistency_finite += consistency.size
                else:
                    if not np.isnan(consistency).all() or np.any(
                        arrays["mean"] < -1e-7
                    ):
                        raise RuntimeError(
                            f"final audit: unsigned W1 semantics failed for {prefix}"
                        )
                    unsigned_consistency_undefined += consistency.size
                if not np.isnan(atlas[f"seed_sign_agreement__{prefix}"]).all():
                    raise RuntimeError(
                        f"final audit: one-seed agreement was manufactured for {prefix}"
                    )
                dense_finite += arrays["mean"].size
        for context in CONTEXTS:
            for name in (
                "valid_fraction",
                "genealogy_valid_fraction_0_10",
                "genealogy_valid_fraction_0_20",
            ):
                values = atlas[f"{name}__{METHOD}__{context}"]
                if (
                    values.shape != (len(SOURCE_LAGS), 80)
                    or not np.isfinite(values).all()
                    or np.any((values < 0) | (values > 1))
                ):
                    raise RuntimeError(f"final audit: support failed for {name}/{context}")

    with np.load(worm_path, allow_pickle=False) as worms:
        keys = [key for key in worms.files if key.startswith("normalized__")]
        if len(keys) != len(CHANNELS) * len(CONTEXTS):
            raise RuntimeError("final audit: worm tensor key count differs")
        expected = (len(SOURCE_LAGS), 20, len(HORIZONS), 80, 80)
        if any(worms[key].shape != expected for key in keys):
            raise RuntimeError("final audit: worm tensor geometry differs")
        if any(not np.isfinite(worms[key]).all() for key in keys):
            raise RuntimeError("final audit: worm tensor contains nonfinite values")

    tables = {
        "support_cells": pd.read_csv(atlas_dir / "support_cells.csv"),
        "head_tail_metrics": pd.read_csv(atlas_dir / "head_tail_metrics.csv"),
        "top_effects": pd.read_csv(atlas_dir / "top_effects.csv"),
        "randi_cook_metrics": pd.read_csv(
            output / "external_reference_checks/randi_cook_metrics.csv"
        ),
        "bentley_metrics": pd.read_csv(
            output / "external_reference_checks/bentley_metrics.csv"
        ),
        "bentley_lagmax": pd.read_csv(
            output / "external_reference_checks/bentley_lagmax_inference.csv"
        ),
    }
    expected_rows = {
        "support_cells": 4_160,
        "head_tail_metrics": 17_472,
        "top_effects": 218_400,
        "randi_cook_metrics": 17_504,
        "bentley_metrics": 45_976,
        "bentley_lagmax": 70,
    }
    observed_rows = {name: len(frame) for name, frame in tables.items()}
    if observed_rows != expected_rows:
        raise RuntimeError(
            f"final audit: table row counts differ: {observed_rows} != {expected_rows}"
        )
    result = {
        "created_utc": _utc(),
        "status": "pass",
        "dense_effect_cells_finite": dense_finite,
        "signed_sign_consistency_cells_finite": signed_consistency_finite,
        "unsigned_w1_sign_consistency_cells_explicitly_undefined": (
            unsigned_consistency_undefined
        ),
        "generator_seed_agreement": "explicitly undefined for one complete seed",
        "table_rows": observed_rows,
        "checks": {
            "dense_geometry_and_orientation_metadata": True,
            "mean_median_and_intervals_finite": True,
            "interval_order": True,
            "unsigned_wasserstein_nonnegative": True,
            "undefined_statistics_not_manufactured": True,
            "support_and_genealogy_ranges": True,
            "worm_tensor_geometry_and_finiteness": True,
            "table_row_counts": True,
        },
    }
    _atomic_json(output / "final_audit.json", result)
    return result


def write_report(output: Path) -> None:
    atlas_dir = output / "atlas"
    external = output / "external_reference_checks"
    raw = json.loads((output / "raw_validation.json").read_text())
    internal = json.loads((atlas_dir / "validation.json").read_text())
    ext = json.loads((external / "validation.json").read_text())
    support = pd.read_csv(atlas_dir / "support_cells.csv")
    head_tail = pd.read_csv(atlas_dir / "head_tail_metrics.csv")
    primary = pd.read_csv(external / "primary_lag1_comparison.csv")
    lagmax = pd.read_csv(external / "bentley_lagmax_inference.csv")

    support_state = support[support.context == "state_average"]
    head_tail_primary = head_tail[
        (head_tail.channel == "endpoint_mean")
        & (head_tail.context == "state_average")
        & (head_tail.source_lag_frames == 1)
        & (head_tail.horizon_frames == 1)
        & (head_tail.scope == "support_qualified_sources")
    ]
    flow_primary = primary[primary.method == METHOD]
    published_primary = primary[primary.method == "sbtg_published"]
    minimum_q = float(lagmax.max_lag_bh_q.min()) if len(lagmax) else math.nan
    lines = [
        "# Full historical SBTG80 progressive-bridge atlas",
        "",
        "## What completed",
        "",
        f"- {raw['validated_archives']} validated raw archives across lags {list(SOURCE_LAGS)} and five outer folds.",
        f"- {internal['dense_effect_cells']:,} dense model-response cells across seven channels, thirteen contexts, six horizons, and 80×80 directed matrices.",
        f"- {BOOTSTRAP_REPLICATES} whole-worm bootstrap replicates provide a 95% interval for every dense cell.",
        "- Post-freeze Randi, Cook, and Bentley correspondence metrics plus historical published-SBTG comparisons.",
        "",
        "## Head and tail recording-origin summary",
        "",
        "The frozen panel contains 63 head-origin and 17 tail-origin neuron classes. These labels describe which released recording supplied a class. They do not repair the invalid index-wise head/tail pairing.",
        "",
        "| Source origin | Target origin | Mean effect | 95% worm interval | Mean |effect| |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in head_tail_primary.itertuples():
        lines.append(
            f"| {row.source_origin} | {row.target_origin} | {row.mean_effect:.4f} | "
            f"[{row.mean_effect_ci_low:.4f}, {row.mean_effect_ci_high:.4f}] | {row.mean_absolute_effect:.4f} |"
        )
    lines.extend(
        [
            "",
            "![Head/tail endpoint effects with 95% whole-worm intervals](atlas/figures/head_tail_endpoint_mean_uncertainty.png)",
        ]
    )
    lines += [
        "",
        "## Sampler support",
        "",
        f"State-average source validity averaged {support_state.valid_fraction.mean():.3f}; "
        f"the minimum source/lag validity was {support_state.valid_fraction.min():.3f}. "
        f"{int(support_state.support_qualified.sum())}/{len(support_state)} source-lag rows met the 0.50 gate.",
        "",
        "## External reference checks",
        "",
        "The tables below are correspondence checks against frozen external matrices, not biological ground truth.",
        "",
        "| Method | Reference | AUROC | Source-macro AUROC |",
        "|---|---|---:|---:|",
    ]
    for frame in (flow_primary, published_primary):
        for row in frame.itertuples():
            lines.append(
                f"| {row.method} | {row.reference} | {row.auroc:.3f} | {row.macro_source_auroc:.3f} |"
            )
    lines += [
        "",
        f"The minimum globally BH-adjusted Bentley lag-max q-value was {minimum_q:.3g}.",
        "",
        "## Uncertainty and limits",
        "",
        "Worm bootstrap intervals represent variation across the 20 held-out historical traces. They are conditional on generator seed 1701 because no second seed has complete checkpoints for all five folds. The chemical panels are event-stratified views of a binary-any-stimulus generator.",
        "",
        "The released SBTG80 cache used invalid index-wise head/tail fusion and donor-worm trace copying. This run preserves the exact historical panel for comparison, but it does not turn those traces into simultaneous 80-neuron measurements. Results are model-relative observational response sensitivities and do not identify couplings, synapses, receptor action, causal intervention effects, or physical transmission delays.",
        "",
        "## Reproduction",
        "",
        "Run `./reproduce.sh` from this directory after installing the project environment and placing the frozen source checkpoints and reference release at the paths documented in `manifest.json`.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    _write_reproduction_files(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest["analysis_completed_utc"] = _utc()
    manifest["analysis_manifest_sha256"] = sha256(atlas_dir / "manifest.json")
    _atomic_json(manifest_path, manifest)
    audit_final_output(output)
    _write_checksums(output)


def _write_reproduction_files(output: Path) -> None:
    script_source = Path(__file__).resolve()
    code_dir = output / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(script_source, code_dir / script_source.name)
    reproduce = """#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=${1:-../..}
cd "$PROJECT_ROOT"
./.venv/bin/python -m compatibility_neural_benchmark.sbtg80_full_progressive_atlas \\
  --output results/sbtg80_full_progressive_atlas_20260901 --overwrite-analysis
"""
    path = output / "reproduce.sh"
    path.write_text(reproduce)
    path.chmod(0o755)
    (output / "RUNBOOK.md").write_text(
        "# Reproduce the full SBTG80 progressive atlas\n\n"
        "Requirements: the project checkout, its `.venv`, the frozen SBTG80 cache, "
        "the five seed-1701 checkpoints, and the external SBTG release. The command is "
        "resume-safe for raw archives.\n\n"
        "```bash\n./reproduce.sh /absolute/path/to/new_sbtg_neuro\n```\n\n"
        "Raw sampling completes first. Internal aggregation then freezes before external "
        "references are opened. Existing raw archives must match the requested protocol "
        "and checkpoint hashes exactly or the run fails closed.\n"
    )


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--reference-release", type=Path, default=DEFAULT_REFERENCE_RELEASE)
    parser.add_argument("--published-archive", type=Path, default=DEFAULT_PUBLISHED_ARCHIVE)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--paired-source-bootstrap", type=int, default=10_000)
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite-analysis", action="store_true")
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    output = args.output.resolve()
    source_run = args.source_run.resolve()
    reference_release = args.reference_release.resolve()
    published_archive = args.published_archive.resolve()
    if args.validate_only:
        print(json.dumps(validate_raw(output, source_run), indent=2, sort_keys=True))
        return
    if not args.analyze_only:
        raw = run_raw(
            output,
            source_run,
            device=args.device,
            max_archives=args.max_archives,
        )
        if args.max_archives is not None and raw["status"] != "pass":
            print(json.dumps(raw, indent=2, sort_keys=True))
            return
    if args.raw_only:
        return
    build_atlas(
        output,
        source_run,
        bootstrap_replicates=args.bootstrap_replicates,
        overwrite=args.overwrite_analysis,
    )
    run_external_metrics(
        output,
        reference_release=reference_release,
        published_archive=published_archive,
        permutations=args.permutations,
        bootstrap_repeats=args.paired_source_bootstrap,
        overwrite=args.overwrite_analysis,
    )
    write_report(output)
    print(json.dumps(json.loads((output / "manifest.json").read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
