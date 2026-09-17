"""Historical 80-neuron progressive-bridge SMC sensitivity analysis.

This is deliberately separate from the canonical 54-neuron prediction atlas.
It reuses the atlas-blind winner from the 80-neuron higher-order tournament,
runs the reviewed progressive bridge at N=32, and opens Randi/Cook/Bentley only
after the response archives are frozen.  The historical SBTG cache carries the
released donor-imputation and pseudo-pairing lineage and is not promoted as a
cleaner biological cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from compatibility_neural_benchmark.latent_distributional_audit import (
    benjamini_hochberg,
)
from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
)
from compatibility_neural_benchmark.prediction_atlas_analysis import (
    effect_normalization_denominator,
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
)
from compatibility_neural_benchmark.postfreeze_external_analysis import align_square
from conditional_neural_benchmark.data import load_sbtg_cohort, make_fold_assignments


MODEL_ID = "tcn_wide_flow"
HISTORY_FRAMES = 8
GENERATOR_SEED = 1701
SOURCE_LAGS = (1, 4, 8)
HORIZONS = (1,)
CHANNELS = ("endpoint_mean", "endpoint_log_sd", "endpoint_wasserstein1")
CONTEXTS = ("state_average", "onset_minus_baseline")
MIN_NORMALIZATION_GAP = 0.10
METHOD = "progressive_bridge_smc"
PARTICLES = 32
BASE_SEED = 20_260_830
FOLD_SEED = 20_260_828
REFERENCE_KEYS = (
    "randi_wild_type",
    "cook_struct_54",
    "cook_chem_54",
    "cook_gap_54",
)


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


def _write_folds(path: Path, cohort, folds: np.ndarray) -> None:
    frame = pd.DataFrame(
        {
            "worm_index": np.arange(cohort.n_worms, dtype=int),
            "worm_id": cohort.worm_ids,
            "outer_fold": folds.astype(int),
        }
    )
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.equals(frame):
            raise RuntimeError("existing historical fold assignment differs")
        return
    frame.to_csv(path, index=False)


def _raw_manifest(output: Path, source_run: Path, reference_release: Path, cohort) -> dict:
    winner_path = source_run / "winner_selection.json"
    source_manifest_path = source_run / "manifest.json"
    winner = json.loads(winner_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    if winner.get("model_id") != MODEL_ID or winner.get("external_references_consulted") is not False:
        raise RuntimeError("80-neuron winner selection is not the expected atlas-blind flow")
    if source_manifest.get("cohort") != "sbtg80" or int(source_manifest.get("neurons", -1)) != 80:
        raise RuntimeError("source tournament is not the frozen 80-neuron cohort")
    checkpoints = {str(fold): _checkpoint_path(source_run, fold) for fold in range(5)}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing historical checkpoints: " + ", ".join(missing))
    return {
        "created_utc": _utc(),
        "status": "raw_sampling_planned",
        "analysis_role": "historical_sbtg80_sensitivity_only",
        "protocol": (
            "bounded progressive bridge SMC N32; horizon 1; source lags 1/4/8; "
            "post-freeze external comparison"
        ),
        "cohort_mode": cohort.cohort_mode,
        "n_worms": cohort.n_worms,
        "n_neurons": cohort.n_neurons,
        "neurons": list(cohort.neurons),
        "fps": cohort.fps,
        "lineage_warning": cohort.lineage_warning,
        "clean_superset": False,
        "model_id": MODEL_ID,
        "history_frames": HISTORY_FRAMES,
        "generator_seed": GENERATOR_SEED,
        "fold_seed": FOLD_SEED,
        "source_lag_frames": list(SOURCE_LAGS),
        "horizon_frames": list(HORIZONS),
        "channels_analyzed": list(CHANNELS),
        "contexts_analyzed": list(CONTEXTS),
        "particles": PARTICLES,
        "base_seed": BASE_SEED,
        "stimulus_generator_encoding": GENERATOR_ENCODING,
        "chemical_identity_conditioned": False,
        "source_tournament": str(source_run.resolve()),
        "source_tournament_manifest_sha256": sha256(source_manifest_path),
        "winner_selection_sha256": sha256(winner_path),
        "checkpoint_sha256": {
            fold: sha256(path) for fold, path in checkpoints.items()
        },
        "reference_release": str(reference_release.resolve()),
        "atlas_firewall": (
            "Randi, Cook, Bentley, receptor, connectome, and published SBTG matrices "
            "were absent from model fitting and progressive-SMC sampling"
        ),
        "claim_boundary": (
            "historical model-relative observed-law sensitivity; not a causal intervention, "
            "anatomical edge recovery, receptor effect, or physical transmission delay"
        ),
        "scope_decision": (
            "lag 16 omitted because it is absent from the equal-grid 1/8-frame Bentley "
            "comparison and cannot change the requested progressive-versus-published result"
        ),
    }


def run_raw(output: Path, source_run: Path, reference_release: Path, device: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cohort = load_sbtg_cohort()
    if cohort.n_worms != 20 or cohort.n_neurons != 80:
        raise RuntimeError("historical SBTG cohort geometry changed")
    folds = make_fold_assignments(cohort, n_folds=5, seed=FOLD_SEED)
    _write_folds(output / "fold_assignments.csv", cohort, folds)
    candidate = _raw_manifest(output, source_run, reference_release, cohort)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("source_lag_frames") == [1, 4, 8, 16]
            and candidate["source_lag_frames"] == [1, 4, 8]
            and not tuple((output / "responses").glob("**/*__ell16__*.npz"))
        ):
            # The original execution plan included a native-only lag-16 panel.
            # It was intentionally stopped before any lag-16 archive was
            # written once the equal-grid 1/8 comparison was complete.
            candidate["created_utc"] = existing.get("created_utc", candidate["created_utc"])
        dynamic = {
            "created_utc",
            "status",
            "completed_utc",
            "analysis_completed_utc",
            "analysis_validation_sha256",
        }
        comparable = {k: v for k, v in existing.items() if k not in dynamic}
        expected = {k: v for k, v in candidate.items() if k not in dynamic}
        legacy_scope = not set(comparable).difference(expected) and all(
            comparable.get(key) == value
            for key, value in expected.items()
            if key not in {"protocol", "source_lag_frames", "scope_decision"}
        )
        if comparable != expected and not legacy_scope:
            raise RuntimeError("existing 80-neuron sensitivity manifest differs")
        if comparable != expected:
            _atomic_json(manifest_path, candidate)
    else:
        _atomic_json(manifest_path, candidate)

    records: list[dict[str, object]] = []
    for lag in SOURCE_LAGS:
        for fold in range(5):
            checkpoint = _checkpoint_path(source_run, fold)
            print(f"SBTG80_PROGRESSIVE_START lag={lag} fold={fold}", flush=True)
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
                    source_window_frames=4,
                    device=device,
                    base_seed=BASE_SEED,
                    min_ess=6.0,
                    progressive_branch_factor=2,
                    progressive_future_branch_factor=1,
                    checkpoint_override=checkpoint,
                    checkpoint_validation_profile="historical_sbtg80",
                )
            except Exception as error:
                result = {"status": "failed", "error": repr(error), "wall_seconds": 0.0}
            record = {"source_lag_frames": lag, "fold": fold, **result}
            records.append(record)
            _atomic_csv(output / "run_status.csv", pd.DataFrame(records))
            print(
                f"SBTG80_PROGRESSIVE_DONE lag={lag} fold={fold} "
                f"status={result['status']} seconds={float(result.get('wall_seconds', 0.0)):.1f}",
                flush=True,
            )
            if result["status"] == "failed":
                break
        if records and records[-1]["status"] == "failed":
            break

    failures = [row for row in records if row["status"] == "failed"]
    validation = {
        "created_utc": _utc(),
        "status": (
            "pass"
            if len(records) == len(SOURCE_LAGS) * 5 and not failures
            else "failed"
        ),
        "expected_archives": len(SOURCE_LAGS) * 5,
        "completed_or_skipped": sum(row["status"] in {"ok", "skipped"} for row in records),
        "failures": failures,
    }
    _atomic_json(output / "raw_validation.json", validation)
    if validation["status"] != "pass":
        raise RuntimeError("80-neuron progressive sampling did not complete")
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "raw_sampling_complete"
    manifest["completed_utc"] = _utc()
    _atomic_json(manifest_path, manifest)


def _load_flow_slices(output: Path, cohort, folds: np.ndarray) -> tuple[list[dict], dict]:
    slices: list[dict] = []
    matrix_payload: dict[str, np.ndarray] = {
        "neurons": np.asarray(cohort.neurons),
        "source_lag_frames": np.asarray(SOURCE_LAGS, dtype=np.int16),
        "horizon_frames": np.asarray(HORIZONS, dtype=np.int16),
        "channels": np.asarray(CHANNELS),
        "contexts": np.asarray(CONTEXTS),
    }
    for channel in CHANNELS:
        context_matrices = {context: [] for context in CONTEXTS}
        context_support = {context: [] for context in CONTEXTS}
        for lag in SOURCE_LAGS:
            normalized = np.full(
                (cohort.n_worms, len(PHASES), 3, cohort.n_neurons, cohort.n_neurons),
                np.nan,
                dtype=np.float32,
            )
            valid = np.full(
                (cohort.n_worms, len(PHASES), 3, cohort.n_neurons),
                np.nan,
                dtype=np.float32,
            )
            for fold in range(5):
                path = (
                    output
                    / "responses"
                    / METHOD
                    / f"{MODEL_ID}__{METHOD}__ell{lag}__N{PARTICLES}__f{fold}__s{GENERATOR_SEED}.npz"
                )
                with np.load(path, allow_pickle=False) as data:
                    if data["status"].item() != "complete":
                        raise RuntimeError(f"incomplete response archive {path}")
                    worm_indices = data["worm_indices"].astype(int)
                    if not np.all(folds[worm_indices] == fold):
                        raise RuntimeError(f"held-out fold mismatch in {path}")
                    response = data[f"response_{channel}"].astype(np.float32)
                    achieved = data["diagnostic_achieved_gap"].astype(np.float32)
                    denominator = effect_normalization_denominator(
                        achieved, MIN_NORMALIZATION_GAP
                    )
                    local = response / denominator[..., None, None]
                    # heldout,phase,event,source,horizon,target ->
                    # heldout,phase,event,target,source (the only horizon is 1).
                    local = local.transpose(0, 1, 2, 4, 5, 3)[:, :, :, 0]
                    normalized[worm_indices] = local
                    valid[worm_indices] = data["diagnostic_valid"].astype(np.float32)
            if not np.isfinite(normalized).all() or not np.isfinite(valid).all():
                raise RuntimeError(f"nonfinite or missing 80-neuron values for {channel}/lag{lag}")
            contexts = {
                "state_average": normalized.mean(axis=(1, 2)),
                "onset_minus_baseline": (normalized[:, 1] - normalized[:, 0]).mean(axis=1),
            }
            supports = {
                "state_average": valid.mean(axis=(1, 2)),
                "onset_minus_baseline": np.minimum(valid[:, 1], valid[:, 0]).mean(axis=1),
            }
            for context in CONTEXTS:
                matrix = contexts[context].mean(axis=0).astype(np.float32)
                support = supports[context].mean(axis=0).astype(np.float32)
                context_matrices[context].append(matrix)
                context_support[context].append(support)
                slices.append(
                    {
                        "method": METHOD,
                        "channel": channel,
                        "context": context,
                        "lag_frames": lag,
                        "horizon_frames": 1,
                        "matrix": matrix,
                        "support": support,
                        "support_available": True,
                        "family": "historical_sbtg80_progressive_flow",
                        "training_lineage": (
                            "historical 80-neuron donor-imputed/pseudo-paired SBTG cache; "
                            "atlas-blind 80-neuron conditional-flow winner"
                        ),
                        "shared_neuron_comparability": "native historical 80-neuron axis",
                    }
                )
        for context in CONTEXTS:
            matrix_payload[f"mean_normalized__{channel}__{context}"] = np.stack(
                context_matrices[context]
            )
            matrix_payload[f"valid_fraction__{channel}__{context}"] = np.stack(
                context_support[context]
            )
    return slices, matrix_payload


def _load_published_slices(path: Path, neurons: tuple[str, ...]) -> list[dict]:
    with np.load(path, allow_pickle=False) as data:
        published_neurons = data["neuron_names"].astype(str).tolist()
        lags = data["lags"].astype(int)
        matrices = [
            align_square(data[f"mu_hat_lag{lag}"], published_neurons, list(neurons))
            for lag in lags
        ]
    support = np.ones(len(neurons), dtype=np.float32)
    return [
        {
            "method": "sbtg_published",
            "channel": "score_product",
            "context": "all_windows",
            "lag_frames": int(lag),
            "horizon_frames": None,
            "matrix": np.asarray(matrix, dtype=np.float32),
            "support": support,
            "support_available": False,
            "family": "sbtg_historical",
            "training_lineage": "released 80-neuron SBTG artifact",
            "shared_neuron_comparability": "exact native historical 80-neuron axis",
        }
        for lag, matrix in zip(lags, matrices)
    ]


def _paired_source_bootstrap(
    progressive: np.ndarray,
    published: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = progressive.shape[1]
    rows: list[dict[str, object]] = []
    for reference_name in REFERENCE_KEYS:
        reference = references[reference_name]
        observed_left = binary_metrics(progressive, reference["labels"], reference["mask"])
        observed_right = binary_metrics(published, reference["labels"], reference["mask"])
        draws = {"auroc": [], "auprc": []}
        for _ in range(repeats):
            columns = rng.integers(0, d, size=d)
            left = binary_metrics(
                progressive[:, columns],
                reference["labels"][:, columns],
                reference["mask"][:, columns],
            )
            right = binary_metrics(
                published[:, columns],
                reference["labels"][:, columns],
                reference["mask"][:, columns],
            )
            for metric in draws:
                draws[metric].append(float(left[metric] - right[metric]))
        for metric, values in draws.items():
            values = np.asarray(values, dtype=float)
            rows.append(
                {
                    "reference": reference_name.replace("_54", "_80"),
                    "metric": metric,
                    "progressive_minus_sbtg_published": float(
                        observed_left[metric] - observed_right[metric]
                    ),
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                    "bootstrap_replicates": repeats,
                    "bootstrap_unit": "source_column",
                    "inference_limit": "frozen-matrix source sensitivity; not animal or generator-refit uncertainty",
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    output: Path,
    primary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    lagmax: pd.DataFrame,
    support_summary: pd.DataFrame,
) -> None:
    refs = ["randi_wild_type", "cook_struct_80", "cook_chem_80", "cook_gap_80"]
    labels = {
        "randi_wild_type": "Randi WT",
        "cook_struct_80": "Cook structural",
        "cook_chem_80": "Cook chemical",
        "cook_gap_80": "Cook gap",
    }
    lines = [
        "# Historical 80-neuron progressive-bridge sensitivity",
        "",
        "This is a post-freeze sensitivity on the released 20-worm/80-neuron SBTG cache. "
        "That cache is not a clean superset: it carries the historical donor-imputation and "
        "index-wise pseudo-pairing lineage. Results are contextual and cannot supersede the "
        "canonical clean 54-neuron atlas.",
        "",
        "## Lag-1 Randi/Cook comparison",
        "",
        "All-estimated matrices use the exact same 80-node denominator for progressive SMC "
        "and SBTG-published. AUROC is primary; AUPRC is prevalence-sensitive.",
        "",
        "| Reference | Progressive AUROC / AUPRC | SBTG-published AUROC / AUPRC | AUROC delta | Source-bootstrap 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for reference in refs:
        group = primary[primary.reference == reference].set_index("method")
        left = group.loc[METHOD]
        right = group.loc["sbtg_published"]
        ci = bootstrap[(bootstrap.reference == reference) & (bootstrap.metric == "auroc")].iloc[0]
        lines.append(
            f"| {labels[reference]} | {left.auroc:.3f} / {left.auprc:.3f} | "
            f"{right.auroc:.3f} / {right.auprc:.3f} | "
            f"{left.auroc - right.auroc:+.3f} | [{ci.ci_low:+.3f}, {ci.ci_high:+.3f}] |"
        )
    lines += [
        "",
        "The paired source bootstrap resamples source columns of two already-frozen matrices. "
        "It is a source-sensitivity interval, not animal, imputation, model-refit, or generator-seed uncertainty.",
        "",
        "## Progressive support",
        "",
        "| Lag (frames) | Mean valid-source fraction | Sources with ≥50% validity |",
        "| ---: | ---: | ---: |",
    ]
    for row in support_summary.itertuples():
        lines.append(
            f"| {int(row.source_lag_frames)} | {row.mean_valid_fraction:.3f} | "
            f"{int(row.sources_valid_ge_050)}/80 |"
        )
    lines += [
        "",
        "## Bentley comparison",
        "",
        "Bentley panels are binary directed receptor/pathway possibilities. The table below "
        "uses the common 1/8-frame grid so progressive SMC and SBTG-published are compared on "
        "the same candidate lags. BH correction spans every evaluable common-grid and native-grid "
        "test in this 80-neuron sensitivity family.",
        "",
        "| Network | Method/channel | Best lag | Best AUROC | permutation p | BH q |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    common = lagmax[
        (lagmax.lag_grid == "common_1_8_frames")
        & lagmax.evaluable.astype(bool)
        & lagmax.network.isin(["monoamine_all", "neuropeptide_all", "neuromodulator_union"])
    ].copy()
    for network in ("monoamine_all", "neuropeptide_all", "neuromodulator_union"):
        candidates = common[common.network == network].sort_values(
            ["max_lag_bh_q", "best_auroc"], ascending=[True, False]
        )
        for method in (METHOD, "sbtg_published"):
            subset = candidates[candidates.method == method]
            if subset.empty:
                continue
            row = subset.iloc[0]
            method_label = (
                f"progressive/{row.channel}" if method == METHOD else "SBTG-published/score_product"
            )
            lines.append(
                f"| {network} | {method_label} | {int(row.best_lag_frames)} | "
                f"{row.best_auroc:.3f} | {row.max_lag_permutation_p:.3f} | {row.max_lag_bh_q:.3f} |"
            )
    lines += [
        "",
        "## Claim boundary",
        "",
        "These are frozen-matrix reference correspondences from an observational conditional "
        "generator. They do not identify synapses, receptor action, anatomical rewiring, causal "
        "interventions, or physical transmission delays.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


def analyze(
    output: Path,
    source_run: Path,
    reference_release: Path,
    published_archive: Path,
    *,
    permutations: int,
    bootstrap_repeats: int,
) -> None:
    raw_validation = json.loads((output / "raw_validation.json").read_text())
    if raw_validation.get("status") != "pass":
        raise RuntimeError("raw sensitivity run has not passed")
    cohort = load_sbtg_cohort()
    folds = make_fold_assignments(cohort, n_folds=5, seed=FOLD_SEED)
    flow_slices, matrix_payload = _load_flow_slices(output, cohort, folds)
    published_slices = _load_published_slices(published_archive, cohort.neurons)
    np.savez_compressed(output / "flow_matrices.npz", **matrix_payload)

    references, networks = load_references(reference_release, list(cohort.neurons))
    all_slices = flow_slices + published_slices
    reference_frame = pd.DataFrame(_reference_rows(all_slices, references, fps=cohort.fps))
    reference_frame["reference"] = reference_frame.reference.str.replace("_54", "_80", regex=False)
    reference_frame.to_csv(output / "randi_cook_metrics.csv", index=False)
    neuromod_frame = pd.DataFrame(_neuromodulator_rows(all_slices, networks, fps=cohort.fps))
    neuromod_frame.to_csv(output / "bentley_metrics.csv", index=False)

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
            if item["channel"] == channel and item["context"] == context
        ]
    selected["sbtg_published"] = published_slices
    native = _lagmax_rows(
        selected,
        networks,
        permutations=permutations,
        seed=20_260_830,
        lag_grid="native_method_grid",
        fps=cohort.fps,
    )
    common = _lagmax_rows(
        selected,
        networks,
        permutations=permutations,
        seed=20_260_831,
        lag_grid="common_1_8_frames",
        fps=cohort.fps,
        forced_lags=(1, 8),
    )
    lagmax = pd.DataFrame(native + common)
    p = lagmax.max_lag_permutation_p.to_numpy(dtype=float)
    finite = np.isfinite(p)
    lagmax["max_lag_bh_q"] = np.nan
    lagmax.loc[finite, "max_lag_bh_q"] = benjamini_hochberg(p[finite])
    lagmax["bh_family"] = "all_80neuron_progressive_and_sbtg_published_native_and_common_grid_tests"
    lagmax["n_bh_tests"] = int(finite.sum())
    lagmax.to_csv(output / "bentley_lagmax_inference.csv", index=False)

    primary = reference_frame[
        (
            (reference_frame.method == METHOD)
            & (reference_frame.channel == "endpoint_mean")
            & (reference_frame.context == "state_average")
            & (reference_frame.horizon_frames == 1)
            & (reference_frame.lag_frames == 1)
            & (reference_frame.scope == "all_estimated")
        )
        |
        (
            (reference_frame.method == "sbtg_published")
            & (reference_frame.lag_frames == 1)
            & (reference_frame.scope == "all_estimated")
        )
    ].copy()
    primary.to_csv(output / "primary_lag1_comparison.csv", index=False)
    progressive_matrix = next(
        item["matrix"]
        for item in flow_slices
        if item["channel"] == "endpoint_mean"
        and item["context"] == "state_average"
        and item["lag_frames"] == 1
    )
    published_matrix = next(
        item["matrix"] for item in published_slices if item["lag_frames"] == 1
    )
    bootstrap = _paired_source_bootstrap(
        progressive_matrix,
        published_matrix,
        references,
        repeats=bootstrap_repeats,
        seed=20_260_830,
    )
    bootstrap.to_csv(output / "paired_source_bootstrap.csv", index=False)

    support_rows = []
    for lag_index, lag in enumerate(SOURCE_LAGS):
        support = matrix_payload["valid_fraction__endpoint_mean__state_average"][lag_index]
        support_rows.append(
            {
                "source_lag_frames": lag,
                "mean_valid_fraction": float(support.mean()),
                "sources_valid_ge_050": int(np.sum(support >= 0.50)),
                "minimum_valid_fraction": float(support.min()),
            }
        )
    support_summary = pd.DataFrame(support_rows)
    support_summary.to_csv(output / "support_summary.csv", index=False)
    _write_report(output, primary, bootstrap, lagmax, support_summary)

    result_files = (
        "flow_matrices.npz",
        "randi_cook_metrics.csv",
        "bentley_metrics.csv",
        "bentley_lagmax_inference.csv",
        "primary_lag1_comparison.csv",
        "paired_source_bootstrap.csv",
        "support_summary.csv",
        "REPORT.md",
    )
    checksums = "".join(
        f"{sha256(output / name)}  {name}\n" for name in result_files
    )
    (output / "analysis_checksums.sha256").write_text(checksums)
    analysis_validation = {
        "created_utc": _utc(),
        "status": "pass",
        "raw_archives": len(SOURCE_LAGS) * 5,
        "randi_cook_rows": len(reference_frame),
        "bentley_metric_rows": len(neuromod_frame),
        "bentley_lagmax_rows": len(lagmax),
        "bentley_bh_tests": int(finite.sum()),
        "bootstrap_replicates": bootstrap_repeats,
        "permutations": permutations,
        "reference_release_sha256": {
            "functional_wild_type": sha256(
                reference_release
                / "reference_data/functional_atlas/aligned_atlas_wild_type.npz"
            ),
            "connectome_nodes": sha256(
                reference_release / "reference_data/connectome/nodes.json"
            ),
        },
        "published_sbtg_archive": str(published_archive.resolve()),
        "published_sbtg_archive_sha256": sha256(published_archive),
        "source_tournament_manifest_sha256": sha256(source_run / "manifest.json"),
        "lineage_warning": cohort.lineage_warning,
        "dashboard_created": False,
    }
    _atomic_json(output / "analysis_validation.json", analysis_validation)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest["analysis_completed_utc"] = _utc()
    manifest["analysis_validation_sha256"] = sha256(output / "analysis_validation.json")
    _atomic_json(manifest_path, manifest)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sbtg80_progressive_sensitivity_20260830"),
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("results/higher_order_neural_20260828/sbtg80"),
    )
    parser.add_argument(
        "--reference-release",
        type=Path,
        default=Path("/Users/vik/Downloads/SBTG-public-release copy"),
    )
    parser.add_argument(
        "--published-archive",
        type=Path,
        default=Path(
            "/Users/vik/Downloads/SBTG-public-release copy/results/paper/sbtg_lag_matrices.npz"
        ),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    output = args.output.resolve()
    source_run = args.source_run.resolve()
    reference_release = args.reference_release.resolve()
    published_archive = args.published_archive.resolve()
    if not args.analyze_only:
        run_raw(output, source_run, reference_release, args.device)
    analyze(
        output,
        source_run,
        reference_release,
        published_archive,
        permutations=args.permutations,
        bootstrap_repeats=args.bootstrap,
    )


if __name__ == "__main__":
    main()
