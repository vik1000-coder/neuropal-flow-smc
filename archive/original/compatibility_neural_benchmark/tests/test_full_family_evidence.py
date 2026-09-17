from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.full_family_evidence import (
    EXPECTED_FAMILIES,
    PRACTICAL_STATUS,
    build_joint_evidence,
    exact_upper_tail_p,
    load_and_validate_families,
    sha256,
    verify_checksums,
)


def test_exact_upper_tail_p_counts_ties_and_nans():
    null = np.asarray([1.0, 2.0, 2.0, 4.0])
    observed = np.asarray([0.5, 2.0, 3.0, np.nan])
    result = exact_upper_tail_p(null, observed)
    np.testing.assert_allclose(result[:3], [1.0, 0.75, 0.25])
    assert np.isnan(result[3])


def _write_checksums(directory: Path) -> None:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (directory / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )


def _write_family(
    root: Path,
    family_id: str,
    signs: np.ndarray,
    *,
    strong: bool,
    sensitivity: bool = False,
) -> None:
    channel, context = EXPECTED_FAMILIES[family_id]
    directory = root / family_id
    directory.mkdir(parents=True, exist_ok=True)
    neurons = [f"N{i:02d}" for i in range(54)]
    neurons[8] = "SMD"
    neurons[9] = "RID"
    neurons[10] = "RIB"
    edge_rows = []
    cell_rows = []
    for target in range(54):
        for source in range(54):
            if source == target:
                continue
            eligible = source < (12 if sensitivity else 8)
            edge_t = 7.0 if strong and source < 6 and target == (source + 1) % 54 else 0.2
            lag_signal = bool(
                sensitivity
                and family_id == "baseline_endpoint_mean"
                and source == 8
                and target in {9, 10}
            )
            edge_rows.append(
                {
                    "edge_rank": len(edge_rows) + 1,
                    "target_neuron": neurons[target],
                    "source_neuron": neurons[source],
                    "target_index": target,
                    "source_index": source,
                    "channel": channel,
                    "context": context,
                    "primary_method": "progressive_bridge_smc",
                    "support_eligible": eligible,
                    "edge_max_abs_t": edge_t if eligible else np.nan,
                    "edge_omnibus_p_value": 0.001 if edge_t > 1 else (0.8 if eligible else np.nan),
                    "edge_bh_q_value": 0.01 if edge_t > 1 else 1.0,
                    "flat_lag_max_abs_t": (7.0 if target == 9 else 4.0) if lag_signal else (0.1 if eligible else np.nan),
                    "flat_lag_omnibus_p_value": 0.001 if lag_signal else (0.9 if eligible else np.nan),
                    "flat_lag_bh_q_value": 0.01 if lag_signal else 1.0,
                }
            )
            for lag in (1, 4, 16):
                for horizon in (1, 2):
                    t_value = edge_t if lag == 1 and horizon == 1 else 0.1
                    lag_t = (
                        (7.0 if target == 9 else 4.0)
                        if lag_signal and lag == 16 and horizon == 1
                        else 0.1
                    )
                    cell_rows.append(
                        {
                            "target_neuron": neurons[target],
                            "source_neuron": neurons[source],
                            "target_index": target,
                            "source_index": source,
                            "channel": channel,
                            "context": context,
                            "support_eligible": eligible,
                            "source_lag_frames": lag,
                            "source_lag_seconds": lag / 4,
                            "horizon_frames": horizon,
                            "horizon_seconds": horizon / 4,
                            "mean_normalized": t_value / 10 if eligible else np.nan,
                            "standard_error": 0.1 if eligible else np.nan,
                            "student_t": t_value if eligible else np.nan,
                            "pointwise_sign_flip_p_value": 0.001 if t_value > 1 else 0.8,
                            "global_max_t_p_value": 0.01 if t_value > 1 else 1.0,
                            "lag_contrast_mean": lag_t / 10 if eligible else np.nan,
                            "lag_contrast_student_t": lag_t if eligible else np.nan,
                            "lag_contrast_global_max_t_p_value": 1.0,
                        }
                    )
    pd.DataFrame(edge_rows).to_csv(directory / "edge_inference.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(directory / "cell_inference.csv", index=False)
    null = np.linspace(0.5, 5.0, len(signs))
    lag_null = np.linspace(0.5, 4.0, len(signs))
    with (directory / "inference_arrays.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            channel=np.asarray(channel),
            context=np.asarray(context),
            sign_patterns=signs,
            global_max_abs_t=null,
            global_max_abs_lag_contrast_t=lag_null,
        )
    sign_hash = hashlib.sha256(signs.tobytes()).hexdigest()
    protocol = {
        "schema_version": "full-family-worm-signflip-v1",
        "family": {"directed_edges": 54 * 53},
        "randomization": {
            "resolved_mode": "exact",
            "sign_patterns_sha256": sign_hash,
        },
        "input": {"sha256": "shared-input-hash"},
    }
    summary = {
        "status": "complete",
        "channel": channel,
        "context": context,
        "n_worms": signs.shape[1],
    }
    manifest = {"status": "complete"}
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    _write_checksums(directory)


def _write_root(root: Path, *, sensitivity: bool = False) -> np.ndarray:
    signs = np.asarray(
        [[1, -1, -1, -1], [1, 1, -1, -1], [1, -1, 1, -1], [1, 1, 1, 1]],
        dtype=np.int8,
    )
    for index, family_id in enumerate(EXPECTED_FAMILIES):
        _write_family(
            root,
            family_id,
            signs,
            strong=index == 0,
            sensitivity=sensitivity,
        )
    return signs


def test_checksum_verification_detects_tampering(tmp_path: Path):
    root = tmp_path / "families"
    _write_root(root)
    directory = root / "baseline_endpoint_mean"
    assert verify_checksums(directory)["verified_count"] >= 6
    with (directory / "summary.json").open("a") as handle:
        handle.write("tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(directory)


def test_family_validation_rejects_sign_order_mismatch(tmp_path: Path):
    root = tmp_path / "families"
    signs = _write_root(root)
    family_id = "baseline_endpoint_log_sd"
    directory = root / family_id
    _write_family(directory.parent, family_id, signs[::-1], strong=False)
    with pytest.raises(ValueError, match="sign-pattern ordering"):
        load_and_validate_families(root)


def test_joint_bundle_labels_and_shortlist_are_deterministic(tmp_path: Path):
    root = tmp_path / "families"
    output = tmp_path / "bundle"
    _write_root(root)
    queue = pd.DataFrame(
        [
            {
                "channel": "endpoint_mean",
                "context": "baseline",
                "source_index": 0,
                "target_index": 1,
                "source_lag_frames": 1,
                "horizon_frames": 1,
                "queue_rank": 1,
                "sign_flip_q_value": 0.001,
                "inference_scope": "post_screen_exploratory_no_full_family_fdr_control",
            }
        ]
    )
    queue_path = tmp_path / "queue.csv"
    queue.to_csv(queue_path, index=False)
    queue_sha256 = sha256(queue_path)
    parent_ledger = tmp_path / "checksums.sha256"
    parent_ledger.write_text(f"{queue_sha256}  queue.csv\n")
    manifest = build_joint_evidence(
        root,
        output,
        queue_path=queue_path,
    )
    assert manifest["status"] == "complete"
    edge = pd.read_csv(output / "edge_evidence.csv")
    cell = pd.read_parquet(output / "cell_evidence.parquet")
    shortlist = pd.read_csv(output / "sampling_null_shortlist.csv")
    assert len(edge) == 4 * 54 * 53
    assert not edge.experiment_ready.any()
    assert set(edge.practical_status) == {PRACTICAL_STATUS}
    assert "strong_edge_lag_unresolved" in set(edge.evidence_label)
    assert len(shortlist) == 6
    assert shortlist.source_index.nunique() == 6
    assert not shortlist.experiment_ready.any()
    match = cell.loc[
        (cell.family_id == "baseline_endpoint_mean")
        & (cell.source_index == 0)
        & (cell.target_index == 1)
        & (cell.source_lag_frames == 1)
        & (cell.horizon_frames == 1)
    ].iloc[0]
    assert match.legacy_queue_match
    assert match.legacy_q_label == "post_screen_legacy_not_full_family_fdr"
    explorer = json.loads((output / "explorer_evidence.json").read_text())
    assert explorer["headline"]["experiment_ready"] == 0
    assert len(explorer["sampling_null_shortlist"]) == 6
    strict_keys = {
        (str(row.family_id), int(row.source_index), int(row.target_index))
        for row in edge.itertuples(index=False)
        if row.joint_primary_edge_max_t_p_value <= 0.05
    }
    explorer_keys = {
        (str(row["family_id"]), str(row["source_neuron"]), str(row["target_neuron"]))
        for row in explorer["top_edges"]
    }
    index_to_name = {
        int(row.source_index): str(row.source_neuron)
        for row in edge.itertuples(index=False)
    }
    target_to_name = {
        int(row.target_index): str(row.target_neuron)
        for row in edge.itertuples(index=False)
    }
    assert {
        (family_id, index_to_name[source_index], target_to_name[target_index])
        for family_id, source_index, target_index in strict_keys
    }.issubset(explorer_keys)
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["legacy_queue_sha256_pinned"]
    assert validation["legacy_queue_provenance"]["sha256"] == queue_sha256
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["legacy_queue"]["input"] == validation[
        "legacy_queue_provenance"
    ]
    assert manifest["inputs"]["legacy_queue"] == validation[
        "legacy_queue_provenance"
    ]
    assert validation["legacy_queue_provenance"]["parent_checksum_ledger"][
        "sha256"
    ] == sha256(parent_ledger)
    assert verify_checksums(output)["verified_count"] >= 9


def test_sensitivity_supplement_never_promotes_primary_claims(tmp_path: Path):
    root = tmp_path / "families"
    sensitivity_root = tmp_path / "sensitivity"
    output = tmp_path / "bundle"
    _write_root(root)
    _write_root(sensitivity_root, sensitivity=True)
    build_joint_evidence(
        root,
        output,
        sensitivity_family_root=sensitivity_root,
    )
    sensitivity_edge = pd.read_csv(output / "sensitivity_edge_evidence.csv")
    sensitivity_only = sensitivity_edge.loc[sensitivity_edge.sensitivity_only]
    assert len(sensitivity_only) > 0
    assert set(sensitivity_only.evidence_label) == {"sampling_limited"}
    assert not sensitivity_edge.experiment_ready.any()
    lag_shortlist = pd.read_csv(
        output / "sampling_null_lag_sensitivity_shortlist.csv"
    )
    assert len(lag_shortlist) == 2
    assert set(lag_shortlist.source_neuron) == {"SMD"}
    assert set(lag_shortlist.target_neuron) == {"RID", "RIB"}
    assert set(lag_shortlist.source_lag_frames) == {16}
    assert set(lag_shortlist.horizon_frames) == {1}
    assert set(lag_shortlist.evidence_label) == {"sampling_limited"}
    controls = lag_shortlist.planned_sham_controls.iloc[0]
    assert "within-stimulus block shift" not in controls
    assert "independent A/B same-arm" in controls
    explorer = json.loads((output / "explorer_evidence.json").read_text())
    assert explorer["sensitivity_support05"]["available"]
    assert explorer["headline"]["experiment_ready"] == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert (
        manifest["artifacts"]["sampling_null_lag_sensitivity_shortlist"]
        == "sampling_null_lag_sensitivity_shortlist.csv"
    )
    assert (
        manifest["artifacts"]["sampling_null_combined"]
        == "sampling_null_combined_8.csv"
    )
    combined_path = output / "sampling_null_combined_8.csv"
    combined = pd.read_csv(combined_path)
    assert list(combined.queue_rank) == list(range(1, 9))
    assert list(combined.selection_origin) == ["strong_primary"] * 6 + [
        "lag_sensitivity"
    ] * 2
    assert combined.run_sampling_nulls.all()
    assert not combined.duplicated(
        [
            "channel",
            "context",
            "source_index",
            "target_index",
            "source_lag_frames",
            "horizon_frames",
        ]
    ).any()
    assert combined_path.read_text().splitlines()[0] == (
        "queue_rank,channel,context,source_neuron,target_neuron,source_index,"
        "target_index,source_lag_frames,horizon_frames,selection_origin,"
        "run_sampling_nulls"
    )
    assert all(line.endswith(",true") for line in combined_path.read_text().splitlines()[1:])
    validation = json.loads((output / "validation.json").read_text())
    assert validation["sensitivity_only_labels_sampling_limited"]
    assert validation["sampling_null_combined_count_eight"]
    assert validation["sampling_null_combined_keys_unique"]
    assert validation["sampling_null_combined_composition_valid"]
    assert verify_checksums(output)["verified_count"] >= 14
