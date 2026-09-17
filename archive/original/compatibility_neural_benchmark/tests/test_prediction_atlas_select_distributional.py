from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_select_distributional import (
    CANONICAL_PREDICTION_COLUMNS,
    CHANNELS,
    CHEMICAL_CONTRAST_CONTEXTS,
    DESIGN_STRATA,
    DistributionalSelectionError,
    PHASE_CONTEXTS,
    freeze_distributional_selection,
    select_balanced_candidates,
)
from compatibility_neural_benchmark.prediction_atlas_select_targeted import (
    ADDED_COLUMNS,
    CANONICAL_QUEUE_COLUMNS,
    SUPPORTED_CHANNELS,
    SUPPORTED_CONTEXTS,
    sha256,
)
from compatibility_neural_benchmark.targeted_confirmation import (
    load_candidate_groups,
)
from compatibility_neural_benchmark.targeted_confirmation_analysis import (
    GroupSpec,
    _load_candidates,
)


NEURONS = tuple(f"N{index:02d}" for index in range(54))
LAGS = (1, 4, 8, 16)
HORIZONS = (1, 2, 4, 8, 16, 32)


def _context_for_stratum(name: str) -> str:
    if name.endswith("__onset_minus_baseline"):
        return "onset_minus_baseline"
    if name.endswith("__chemical_onset_minus_baseline"):
        return "butanone_onset_minus_baseline"
    if name.endswith("__phase_specific"):
        return "baseline"
    raise AssertionError(name)


def _row(
    *,
    channel: str,
    context: str,
    source: int,
    target: int,
    lag: int,
    horizon: int,
    score: float,
    mean: float,
    tier: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in CANONICAL_PREDICTION_COLUMNS}
    signed = channel != "endpoint_wasserstein1" or context.endswith(
        "minus_baseline"
    )
    chemical = context.split("_onset_minus_baseline")[0] if context in CHEMICAL_CONTRAST_CONTEXTS else ""
    row.update(
        {
            "method": "progressive_bridge_smc",
            "model_id": "synthetic_binary_flow",
            "channel": channel,
            "context": context,
            "chemical": chemical,
            "conditioning_status": (
                "exploratory_event_stratified_under_binary_any_stimulus_generator"
                if chemical
                else "binary_any_stimulus_conditioned"
            ),
            "source_neuron": NEURONS[source],
            "target_neuron": NEURONS[target],
            "source_index": source,
            "target_index": target,
            "source_lag_frames": lag,
            "source_to_cut_seconds": lag / 4,
            "horizon_frames": horizon,
            "forecast_horizon_seconds": horizon / 4,
            "source_to_readout_seconds": (lag + horizon) / 4,
            "mean_raw": mean * 0.4,
            "mean_normalized": mean,
            "median_normalized": mean * 0.95,
            "ci_2_5": mean * 0.5,
            "ci_97_5": mean * 1.5,
            "screen_t_p_value": 0.01 if signed else np.nan,
            "sign_flip_p_value": 0.01 if signed else np.nan,
            "sign_flip_q_value": 0.02 if signed else np.nan,
            "bh_family": f"progressive_bridge_smc|{channel}|{context}|ell={lag}|h={horizon}",
            "inference_scope": "post_screen_exploratory_no_full_family_fdr_control",
            "test_sidedness": (
                "two_sided_candidate_sign_flip"
                if signed
                else "not_tested_unsigned_distance"
            ),
            "sign_consistency": 1.0 if signed else np.nan,
            "valid_fraction": 0.9,
            "genealogy_gate_applicable": True,
            "genealogy_min_distinct_ancestor_fraction_strong": 0.1,
            "genealogy_min_distinct_ancestor_fraction_sensitivity": 0.2,
            "genealogy_valid_fraction_0_10": 0.85,
            "genealogy_valid_fraction_0_20": 0.7,
            "genealogy_strong_gate_pass": True,
            "genealogy_sensitivity_gate_pass": False,
            "seed_spearman": 0.7,
            "seed_sign_agreement": 1.0 if signed else np.nan,
            "n_worms": 17,
            "effect_direction": "positive" if mean > 0 else "negative",
            "support_tier": tier,
            "evidence_score": score,
            "interpretation_limit": "model_based_lag_association_not_causal_or_physical_delay",
        }
    )
    return row


def _specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for index, stratum in enumerate(DESIGN_STRATA):
        context = _context_for_stratum(stratum.name)
        tier = (
            "model_only"
            if stratum.channel == "endpoint_wasserstein1" and context in PHASE_CONTEXTS
            else "supported_exploratory"
        )
        # Every stratum's highest row uses source zero.  The global assignment
        # must sacrifice five of those local maxima to obtain six unique sources.
        specs.append(
            {
                "row": _row(
                    channel=stratum.channel,
                    context=context,
                    source=0,
                    target=10 + index,
                    lag=LAGS[index % len(LAGS)],
                    horizon=HORIZONS[index % len(HORIZONS)],
                    score=100 - index,
                    mean=0.20 + index * 0.01,
                    tier=tier,
                ),
                "counterpart": 0.18 + index * 0.01,
                "counterpart_valid": 0.8,
            }
        )
        specs.append(
            {
                "row": _row(
                    channel=stratum.channel,
                    context=context,
                    source=index + 1,
                    target=20 + index,
                    lag=LAGS[index % len(LAGS)],
                    horizon=HORIZONS[index % len(HORIZONS)],
                    score=90 - index,
                    mean=0.15 + index * 0.01,
                    tier=tier,
                ),
                "counterpart": 0.14 + index * 0.01,
                "counterpart_valid": 0.8,
            }
        )
    return specs


def _checksum_bundle(atlas: Path) -> None:
    names = sorted(
        path.name
        for path in atlas.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (atlas / "checksums.sha256").write_text(
        "".join(f"{sha256(atlas / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _atlas(tmp_path: Path, specs: list[dict[str, object]] | None = None) -> Path:
    specs = list(_specs() if specs is None else specs)
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    prediction = pd.DataFrame(
        [value["row"] for value in specs], columns=CANONICAL_PREDICTION_COLUMNS
    )
    prediction.to_parquet(atlas / "prediction_cells.parquet", index=False)
    support_rows: list[dict[str, object]] = []
    seen_support: set[tuple[object, ...]] = set()
    for value in specs:
        row = value["row"]
        key = (row["context"], row["source_lag_frames"], row["source_index"])
        if key in seen_support:
            continue
        seen_support.add(key)
        support_rows.append(
            {
                "method": "progressive_bridge_smc",
                "context": row["context"],
                "source_neuron": row["source_neuron"],
                "source_index": row["source_index"],
                "source_lag_frames": row["source_lag_frames"],
                "valid_fraction": row["valid_fraction"],
                "genealogy_gate_applicable": True,
                "genealogy_valid_fraction_0_10": row[
                    "genealogy_valid_fraction_0_10"
                ],
                "genealogy_valid_fraction_0_20": row[
                    "genealogy_valid_fraction_0_20"
                ],
                "genealogy_strong_gate_pass": row["genealogy_strong_gate_pass"],
                "genealogy_sensitivity_gate_pass": row[
                    "genealogy_sensitivity_gate_pass"
                ],
            }
        )
    pd.DataFrame(support_rows).to_parquet(
        atlas / "support_cells.parquet", index=False
    )

    arrays: dict[str, np.ndarray] = {
        "neurons": np.asarray(NEURONS),
        "methods": np.asarray(["direct_importance", "progressive_bridge_smc"]),
        "channels": np.asarray(SUPPORTED_CHANNELS),
        "contexts": np.asarray(SUPPORTED_CONTEXTS),
        "source_lag_frames": np.asarray(LAGS, dtype=np.int16),
        "horizon_frames": np.asarray(HORIZONS, dtype=np.int16),
        "orientation": np.asarray("target_row_source_column"),
        "primary_method": np.asarray("progressive_bridge_smc"),
    }
    used = sorted(
        {(str(value["row"]["channel"]), str(value["row"]["context"])) for value in specs}
    )
    base = (
        np.arange(54 * 54, dtype=np.float32).reshape(54, 54) / 10000 + 0.01
    )
    for channel, context in used:
        primary = np.stack(
            [
                np.stack([base + li * 0.001 + hi * 0.0001 for hi in range(6)])
                for li in range(4)
            ]
        )
        counterpart = primary * 0.9
        primary_valid = np.full((4, 54), 0.9, dtype=np.float32)
        direct_valid = np.full((4, 54), 0.8, dtype=np.float32)
        for value in specs:
            row = value["row"]
            if row["channel"] != channel or row["context"] != context:
                continue
            li = LAGS.index(int(row["source_lag_frames"]))
            hi = HORIZONS.index(int(row["horizon_frames"]))
            target, source = int(row["target_index"]), int(row["source_index"])
            primary[li, hi, target, source] = float(row["mean_normalized"])
            counterpart[li, hi, target, source] = float(value["counterpart"])
            direct_valid[li, source] = float(value["counterpart_valid"])
        arrays[
            f"mean_normalized__progressive_bridge_smc__{channel}__{context}"
        ] = primary
        arrays[
            f"mean_normalized__direct_importance__{channel}__{context}"
        ] = counterpart
        arrays[f"valid_fraction__progressive_bridge_smc__{context}"] = primary_valid
        arrays[f"valid_fraction__direct_importance__{context}"] = direct_valid
    np.savez_compressed(atlas / "atlas_matrices.npz", **arrays)

    pd.DataFrame({"queue_rank": [1]}).to_csv(
        atlas / "hypothesis_queue.csv", index=False
    )
    (atlas / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "created_utc": "2026-08-30T00:00:00+00:00",
                "primary_method": "progressive_bridge_smc",
                "dense_matrix_orientation": "target_row_source_column",
                "artifacts": {
                    "prediction_cells": "prediction_cells.parquet",
                    "support_cells": "support_cells.parquet",
                    "atlas_matrices": "atlas_matrices.npz",
                    "hypothesis_queue": "hypothesis_queue.csv",
                    "protocol": "protocol.json",
                    "validation": "validation.json",
                    "models": "models.json",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (atlas / "protocol.json").write_text(
        json.dumps(
            {
                "ranking": "no connectome or external reference data used",
                "orientation": "target_row_source_column",
                "n_worms": 17,
                "n_neurons": 54,
                "source_lag_frames": list(LAGS),
                "horizon_frames": list(HORIZONS),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (atlas / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "problems": [],
                "archive_checks_completed": {"orientation_self_test": True},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (atlas / "models.json").write_text(
        json.dumps(
            {
                "generator_stimulus_encoding": "binary_any_stimulus",
                "chemical_conditioning": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _checksum_bundle(atlas)
    return atlas


def _context_class(value: str) -> str:
    if value == "onset_minus_baseline":
        return "global_contrast"
    if value in CHEMICAL_CONTRAST_CONTEXTS:
        return "chemical_contrast"
    if value in PHASE_CONTEXTS:
        return "phase"
    raise AssertionError(value)


def test_freeze_is_deterministic_balanced_unique_and_downstream_compatible(
    tmp_path: Path,
) -> None:
    atlas = _atlas(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = freeze_distributional_selection(atlas, first)
    freeze_distributional_selection(atlas, second)

    for name in (
        "hypothesis_queue.csv",
        "atlas_matrices.npz",
        "selection_rationale.md",
        "manifest.json",
        "checksums.sha256",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    selected = pd.read_csv(first / "hypothesis_queue.csv")
    assert list(selected.columns) == list(CANONICAL_QUEUE_COLUMNS) + list(
        ADDED_COLUMNS
    )
    assert len(selected) == 6
    assert selected.run_confirmation.all()
    assert selected.source_neuron.nunique() == 6
    assert selected.channel.value_counts().to_dict() == {
        "endpoint_sd": 2,
        "endpoint_log_sd": 2,
        "endpoint_wasserstein1": 2,
    }
    assert selected.context.map(_context_class).value_counts().to_dict() == {
        "global_contrast": 2,
        "chemical_contrast": 2,
        "phase": 2,
    }
    unsigned = selected[
        (selected.channel == "endpoint_wasserstein1")
        & selected.context.isin(PHASE_CONTEXTS)
    ]
    assert len(unsigned) == 1
    assert unsigned.iloc[0].support_tier == "model_only"
    assert not bool(unsigned.iloc[0].promotion_eligible)
    assert manifest["source_hypothesis_queue_used"] is False
    assert manifest["external_reference_inputs_used"] == []
    assert manifest["unique_selected_sources"] == 6
    assert manifest["downstream_schema_extension_required"] is False
    assert sha256(first / "atlas_matrices.npz") == sha256(
        atlas / "atlas_matrices.npz"
    )
    assert (first / "atlas_matrices.npz").stat().st_ino != (
        atlas / "atlas_matrices.npz"
    ).stat().st_ino
    for line in (first / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha256(first / name) == digest

    runner_groups = load_candidate_groups(
        first / "hypothesis_queue.csv", NEURONS
    )
    analysis_groups = tuple(
        GroupSpec(
            lag=value.source_lag_frames,
            sources=value.source_indices,
            source_names=tuple(NEURONS[index] for index in value.source_indices),
            horizons=value.horizon_frames,
            phases=value.phases,
            selection_fingerprint=value.selection_fingerprint,
        )
        for value in runner_groups
    )
    cells = _load_candidates(
        first / "hypothesis_queue.csv",
        groups=analysis_groups,
        neurons=NEURONS,
    )
    assert len(cells) == 6
    assert {value.screen_channel for value in cells} == set(CHANNELS)


def test_signed_gates_exclude_wrong_sign_low_direct_support_and_model_only(
    tmp_path: Path,
) -> None:
    specs = _specs()
    template = specs[0]["row"]
    invalid = [
        {
            "row": _row(
                channel=str(template["channel"]),
                context=str(template["context"]),
                source=30,
                target=31,
                lag=1,
                horizon=32,
                score=10000,
                mean=0.9,
                tier="supported_exploratory",
            ),
            "counterpart": -0.8,
            "counterpart_valid": 0.9,
        },
        {
            "row": _row(
                channel=str(template["channel"]),
                context=str(template["context"]),
                source=32,
                target=33,
                lag=4,
                horizon=32,
                score=9000,
                mean=0.8,
                tier="supported_exploratory",
            ),
            "counterpart": 0.7,
            "counterpart_valid": 0.4,
        },
        {
            "row": _row(
                channel=str(template["channel"]),
                context=str(template["context"]),
                source=34,
                target=35,
                lag=8,
                horizon=32,
                score=8000,
                mean=0.7,
                tier="model_only",
            ),
            "counterpart": 0.6,
            "counterpart_valid": 0.9,
        },
    ]
    atlas = _atlas(tmp_path, specs + invalid)
    output = tmp_path / "selection"
    freeze_distributional_selection(atlas, output)
    selected = pd.read_csv(output / "hypothesis_queue.csv")
    assert not {"N30", "N32", "N34"}.intersection(selected.source_neuron)


def test_external_support_field_and_checksum_tamper_are_rejected(
    tmp_path: Path,
) -> None:
    atlas = _atlas(tmp_path)
    support = pd.read_parquet(atlas / "support_cells.parquet")
    support["cook_auroc"] = 0.9
    support.to_parquet(atlas / "support_cells.parquet", index=False)
    _checksum_bundle(atlas)
    with pytest.raises(DistributionalSelectionError, match="external-reference"):
        freeze_distributional_selection(atlas, tmp_path / "external")

    support = support.drop(columns="cook_auroc")
    support.to_parquet(atlas / "support_cells.parquet", index=False)
    _checksum_bundle(atlas)
    with (atlas / "hypothesis_queue.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(DistributionalSelectionError, match="checksum failed"):
        freeze_distributional_selection(atlas, tmp_path / "tampered")


def test_reuse_fallback_optimizes_the_completed_six_row_objective() -> None:
    source_index = {"A": 0, "B": 1, "C": 2}
    compact = [
        (0, "B", 20.0, "T0"),
        (0, "C", 11.0, "T1"),
        (1, "A", 10.0, "T0"),
        (1, "C", 3.0, "T1"),
        (2, "B", 18.0, "T0"),
        (2, "C", 10.0, "T1"),
        (3, "A", 7.0, "T0"),
        (3, "A", 8.0, "T1"),
        (4, "A", 7.0, "T0"),
        (4, "B", 5.0, "T1"),
        (5, "B", 13.0, "T0"),
        (5, "A", 3.0, "T1"),
    ]
    candidates = pd.DataFrame(
        [
            {
                "design_stratum": DESIGN_STRATA[stratum].name,
                "context": f"c{stratum}",
                "source_neuron": source,
                "target_neuron": target,
                "source_lag_frames": 1,
                "horizon_frames": 1,
                "source_index": source_index[source],
                "target_index": int(target[-1]),
                "evidence_score": score,
            }
            for stratum, source, score, target in compact
        ]
    )

    selected = select_balanced_candidates(candidates)
    shuffled = select_balanced_candidates(
        candidates.sample(frac=1.0, random_state=20260830).reset_index(drop=True)
    )
    by_stratum = selected.set_index("design_stratum")
    shuffled_by_stratum = shuffled.set_index("design_stratum")
    expected_sources = ["B", "C", "B", "A", "A", "B"]

    assert selected.source_neuron.nunique() == 3
    assert selected.evidence_score.sum() == pytest.approx(69.0)
    assert [
        by_stratum.loc[value.name, "source_neuron"] for value in DESIGN_STRATA
    ] == expected_sources
    assert by_stratum.loc[DESIGN_STRATA[3].name, "target_neuron"] == "T1"
    pd.testing.assert_frame_equal(
        by_stratum.sort_index(), shuffled_by_stratum.sort_index()
    )


@pytest.mark.parametrize("table_name", ["prediction_cells", "support_cells"])
def test_string_encoded_genealogy_booleans_are_rejected(
    tmp_path: Path,
    table_name: str,
) -> None:
    atlas = _atlas(tmp_path)
    path = atlas / f"{table_name}.parquet"
    frame = pd.read_parquet(path)
    frame["genealogy_gate_applicable"] = frame[
        "genealogy_gate_applicable"
    ].map({True: "True", False: "False"})
    frame.to_parquet(path, index=False)
    _checksum_bundle(atlas)

    with pytest.raises(
        DistributionalSelectionError,
        match=r"genealogy_gate_applicable.*non-null boolean dtype",
    ):
        freeze_distributional_selection(atlas, tmp_path / "selection")


def test_sensitivity_gate_disagreement_is_rejected(tmp_path: Path) -> None:
    atlas = _atlas(tmp_path)
    support = pd.read_parquet(atlas / "support_cells.parquet")
    support.loc[0, "genealogy_sensitivity_gate_pass"] = not bool(
        support.loc[0, "genealogy_sensitivity_gate_pass"]
    )
    support.to_parquet(atlas / "support_cells.parquet", index=False)
    _checksum_bundle(atlas)

    with pytest.raises(
        DistributionalSelectionError,
        match="genealogy sensitivity gates disagree",
    ):
        freeze_distributional_selection(atlas, tmp_path / "selection")


@pytest.mark.parametrize(
    "extra_column",
    ["synapse_count", "receptor_expression_score", "mystery_score"],
)
def test_support_allowlist_rejects_every_unreviewed_column(
    tmp_path: Path,
    extra_column: str,
) -> None:
    atlas = _atlas(tmp_path)
    support = pd.read_parquet(atlas / "support_cells.parquet")
    support[extra_column] = 1.0
    support.to_parquet(atlas / "support_cells.parquet", index=False)
    _checksum_bundle(atlas)

    with pytest.raises(
        DistributionalSelectionError,
        match="support table contains unreviewed columns",
    ):
        freeze_distributional_selection(atlas, tmp_path / "selection")


@pytest.mark.parametrize("case", ["top_level", "nested", "inside_list"])
def test_recursive_json_provenance_firewall_rejects_external_fields(
    tmp_path: Path,
    case: str,
) -> None:
    atlas = _atlas(tmp_path)
    if case == "inside_list":
        path = atlas / "models.json"
        value = json.loads(path.read_text())
        value["source_run_provenance"] = [
            {"metadata": {"receptor_expression_score": 0.9}}
        ]
    else:
        path = atlas / "manifest.json"
        value = json.loads(path.read_text())
        if case == "top_level":
            value["cook_auroc"] = 0.9
        else:
            value["artifacts"]["metadata"] = {"cook_auroc": 0.9}
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    _checksum_bundle(atlas)

    with pytest.raises(
        DistributionalSelectionError,
        match="prohibited external-reference fields",
    ):
        freeze_distributional_selection(atlas, tmp_path / "selection")


def test_json_top_level_allowlist_rejects_neutral_ignored_field(
    tmp_path: Path,
) -> None:
    atlas = _atlas(tmp_path)
    path = atlas / "manifest.json"
    value = json.loads(path.read_text())
    value["mystery_score"] = 0.9
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    _checksum_bundle(atlas)

    with pytest.raises(
        DistributionalSelectionError,
        match="unreviewed top-level fields",
    ):
        freeze_distributional_selection(atlas, tmp_path / "selection")
