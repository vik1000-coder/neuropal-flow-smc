from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compatibility_neural_benchmark.prediction_atlas_external_analysis import (
    EXPECTED_NEUROMODULATOR_SOURCES,
    REFERENCE_RELEASE_FILES,
)
from compatibility_neural_benchmark.prediction_atlas_report import (
    ATLAS_REQUIRED,
    EXTERNAL_REQUIRED,
    ROOT_DOCUMENTS,
    TARGETED_REQUIRED,
    ReportInputError,
    build_prediction_atlas_report,
    sha256,
)
from compatibility_neural_benchmark.targeted_confirmation_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    INTERPRETATION_LIMIT,
    ORIENTATION,
)
from compatibility_neural_benchmark.tests.test_prediction_atlas_dashboard import (
    _write_fixture as _write_dashboard_fixture,
)


CREATED = "2026-08-29T20:00:00+00:00"
SIGNED_METRICS = (
    "endpoint_mean",
    "time_average_mean",
    "pathwise_peak_mean",
    "crossing_probability",
    "endpoint_log_sd",
    "endpoint_sd",
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_ledger(directory: Path, names: tuple[str, ...]) -> None:
    (directory / "checksums.sha256").write_text(
        "".join(f"{sha256(directory / name)}  {name}\n" for name in names)
    )


def _write_atlas(root: Path) -> tuple[Path, tuple[str, ...]]:
    atlas = _write_dashboard_fixture(root)
    neurons = tuple(f"N{index:02d}" for index in range(54))
    queue = pd.read_csv(atlas / "hypothesis_queue.csv").iloc[:24].copy()
    queue["source_neuron"] = [neurons[int(value)] for value in queue["source_index"]]
    queue["target_neuron"] = [neurons[int(value)] for value in queue["target_index"]]
    # Exercise two distinct candidate cells that share the same directed pair.
    queue.loc[1, ["source_index", "target_index"]] = queue.loc[
        0, ["source_index", "target_index"]
    ].to_numpy()
    queue.loc[1, ["source_neuron", "target_neuron"]] = queue.loc[
        0, ["source_neuron", "target_neuron"]
    ].to_numpy()
    queue.to_csv(atlas / "hypothesis_queue.csv", index=False)
    dashboard = json.loads((atlas / "dashboard_snapshot.json").read_text())
    dashboard["top_candidates"] = json.loads(queue.iloc[:10].to_json(orient="records"))
    dashboard["stimulus_composition_summary"] = [
        {
            "phase": "onset",
            "source_lag_frames": 1,
            "horizon_frames": 4,
            "n_worm_events": 51,
            "source_window_stimulus_fraction_min": 0.5,
            "source_window_stimulus_fraction_mean": 0.75,
            "source_window_stimulus_fraction_max": 1.0,
            "forecast_window_stimulus_fraction_min": 1.0,
            "forecast_window_stimulus_fraction_mean": 1.0,
            "forecast_window_stimulus_fraction_max": 1.0,
            "cut_stimulus_fraction_min": 1.0,
            "cut_stimulus_fraction_mean": 1.0,
            "cut_stimulus_fraction_max": 1.0,
            "forecast_endpoint_stimulus_fraction_min": 1.0,
            "forecast_endpoint_stimulus_fraction_mean": 1.0,
            "forecast_endpoint_stimulus_fraction_max": 1.0,
            "source_window_crosses_stimulus_boundary_any": True,
            "forecast_window_crosses_stimulus_boundary_any": False,
            "cut_to_endpoint_stimulus_transition_any": False,
        }
    ]
    _write_json(atlas / "dashboard_snapshot.json", dashboard)
    protocol = json.loads((atlas / "protocol.json").read_text())
    protocol["stimulus_composition"] = {
        "artifact": "stimulus_composition.csv",
        "grain": "worm x phase x event-position x source-lag x forecast-horizon",
        "row_count": 1,
    }
    _write_json(atlas / "protocol.json", protocol)
    pd.DataFrame(
        [
            {
                "worm_id": "worm-0",
                "phase": "onset",
                "event_index": 0,
                "source_lag_frames": 1,
                "horizon_frames": 4,
                "source_window_stimulus_fraction": 0.75,
                "cut_stimulus_indicator": True,
                "forecast_window_stimulus_fraction": 1.0,
                "forecast_endpoint_stimulus_indicator": True,
            }
        ]
    ).to_csv(atlas / "stimulus_composition.csv", index=False)
    _write_json(
        atlas / "manifest.json",
        {
            "status": "complete",
            "protocol": "reviewed neural prediction atlas v1",
            "primary_method": "progressive_bridge_smc",
            "dense_matrix_orientation": ORIENTATION,
            "created_utc": CREATED,
        },
    )
    np.savez_compressed(
        atlas / "atlas_matrices.npz",
        orientation=np.asarray(ORIENTATION),
        neurons=np.asarray(neurons),
        methods=np.asarray(["direct_importance", "progressive_bridge_smc"]),
        channels=np.asarray(dashboard["channels"]),
        contexts=np.asarray([item["context"] for item in dashboard["contexts"]]),
    )
    _write_ledger(
        atlas,
        tuple(name for name in ATLAS_REQUIRED if name != "checksums.sha256"),
    )
    return atlas, neurons


def _write_targeted(root: Path, atlas: Path, neurons: tuple[str, ...]) -> Path:
    selection = root / "targeted_selection"
    selection.mkdir()
    canonical_text = pd.read_csv(
        atlas / "hypothesis_queue.csv", dtype=str, keep_default_na=False
    )
    staged = canonical_text.iloc[:2].copy()
    staged["run_confirmation"] = True
    staged.to_csv(selection / "hypothesis_queue.csv", index=False)
    (selection / "atlas_matrices.npz").write_bytes(
        (atlas / "atlas_matrices.npz").read_bytes()
    )
    (selection / "selection_rationale.md").write_text(
        "# Synthetic frozen selection\n\nInternal queue rows only.\n"
    )
    _write_json(
        selection / "manifest.json",
        {
            "schema_version": "prediction_atlas_targeted_selection_v1",
            "status": "complete",
            "source_atlas": str(atlas.resolve()),
            "source_manifest_sha256": sha256(atlas / "manifest.json"),
            "source_protocol_sha256": sha256(atlas / "protocol.json"),
            "source_validation_sha256": sha256(atlas / "validation.json"),
            "source_atlas_matrices_sha256": sha256(
                atlas / "atlas_matrices.npz"
            ),
            "frozen_hypothesis_queue_sha256": sha256(
                atlas / "hypothesis_queue.csv"
            ),
            "selection_sha256": sha256(selection / "hypothesis_queue.csv"),
            "maximum_rows": 6,
            "particles": 128,
            "confirmation_method": "progressive_bridge_smc_targeted_three_arm",
            "screen_dir_compatible": True,
            "selected_rows": len(staged),
            "external_reference_inputs_used": [],
            "external_fields_detected": [],
        },
    )
    _write_ledger(
        selection,
        (
            "manifest.json",
            "hypothesis_queue.csv",
            "atlas_matrices.npz",
            "selection_rationale.md",
        ),
    )
    targeted = root / "targeted_analysis"
    targeted.mkdir()
    queue = pd.read_csv(atlas / "hypothesis_queue.csv").iloc[:2]
    cell_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    screen_rows: list[dict[str, object]] = []
    contrasts = ("high_low", "high_factual", "low_factual")
    for candidate_number, row in enumerate(queue.itertuples(index=False), start=1):
        candidate = f"targeted_{candidate_number:04d}"
        base = {
            "candidate_id": candidate,
            "queue_rank": int(row.queue_rank),
            "source_neuron": str(row.source_neuron),
            "target_neuron": str(row.target_neuron),
            "source_index": int(row.source_index),
            "target_index": int(row.target_index),
            "source_lag_frames": int(row.source_lag_frames),
            "horizon_frames": int(row.horizon_frames),
            "context": str(row.context),
            "valid_fraction": 0.88,
            "n_worms": 17,
            "n_model_seeds": 2,
            "n_particles": 128,
            "matrix_orientation": ORIENTATION,
            "row_axis": "target_neuron",
            "column_axis": "source_neuron",
            "interpretation_limit": INTERPRETATION_LIMIT,
        }
        for contrast_number, contrast in enumerate(contrasts, start=1):
            for metric_number, metric in enumerate((*SIGNED_METRICS, "endpoint_wasserstein1"), start=1):
                value = 0.03 * candidate_number * contrast_number * metric_number
                cell_rows.append(
                    {
                        **base,
                        "contrast": contrast,
                        "metric_family": (
                            "pairwise_wasserstein1"
                            if metric == "endpoint_wasserstein1"
                            else "signed_contrast"
                        ),
                        "metric": metric,
                        "signed": (
                            metric != "endpoint_wasserstein1"
                            or str(base["context"]).endswith("minus_baseline")
                        ),
                        "mean_normalized": value,
                        "mean_raw": value * 0.5,
                        "ci_2_5": max(0.0, value - 0.02) if metric == "endpoint_wasserstein1" else value - 0.02,
                        "ci_97_5": value + 0.02,
                        "evidence_label": (
                            "model_relative_descriptive"
                            if metric == "endpoint_wasserstein1"
                            else "model_relative_consistent"
                        ),
                    }
                )
            for quantile in (0.25, 0.5, 0.75):
                quantile_rows.append(
                    {
                        **base,
                        "contrast": contrast,
                        "endpoint_quantile": quantile,
                        "signed": True,
                        "mean_normalized": 0.1 * candidate_number * contrast_number,
                        "ci_2_5": 0.02,
                        "ci_97_5": 0.18,
                    }
                )
        support_rows.append(
            {
                "source_neuron": str(row.source_neuron),
                "source_index": int(row.source_index),
                "source_lag_frames": int(row.source_lag_frames),
                "context": str(row.context),
                "valid_fraction": 0.88,
                "mean_achieved_gap_magnitude": 1.25,
                "mean_ess_low": 96.0,
                "mean_ess_high": 91.0,
                "n_worms": 17,
                "n_particles": 128,
                "matrix_orientation": ORIENTATION,
            }
        )
        screen_rows.append(
            {
                "candidate_id": candidate,
                "source_neuron": str(row.source_neuron),
                "target_neuron": str(row.target_neuron),
                "source_lag_frames": int(row.source_lag_frames),
                "horizon_frames": int(row.horizon_frames),
                "context": str(row.context),
                "screen_method": "progressive_bridge_smc",
                "targeted_method": "progressive_bridge_smc_targeted_three_arm",
                "same_sampler_family": True,
                "comparison_type": "particle_escalation_within_progressive_bridge",
                "screen_channel": "endpoint_mean",
                "targeted_metric": "endpoint_mean",
                "screen_mean_normalized": 0.025 * candidate_number,
                "targeted_n128_mean_normalized": 0.03 * candidate_number,
                "targeted_minus_screen": 0.005 * candidate_number,
                "direction_agreement": 1.0,
                "magnitude_agreement": 0.9,
                "screen_valid_fraction": 0.84,
                "targeted_valid_fraction": 0.88,
                "consistency_label": "direction_consistent",
                "interpretation_limit": (
                    "post-screen N128 consistency; descriptive and selection-conditioned"
                ),
            }
        )
    pd.DataFrame(cell_rows).to_csv(targeted / "targeted_cells.csv", index=False)
    pd.DataFrame(quantile_rows).to_csv(
        targeted / "targeted_quantile_shifts.csv", index=False
    )
    pd.DataFrame(support_rows).to_csv(
        targeted / "support_diagnostics.csv", index=False
    )
    pd.DataFrame(screen_rows).to_csv(
        targeted / "screen_consistency.csv", index=False
    )
    raw = root / "targeted_raw_input.bin"
    raw.write_bytes(b"immutable synthetic targeted input")
    targeted_run = root / "targeted_run"
    targeted_run.mkdir()
    _write_json(
        targeted_run / "manifest.json",
        {
            "method": "progressive_bridge_smc_targeted_three_arm",
            "hypothesis_queue": str(
                (selection / "hypothesis_queue.csv").resolve()
            ),
            "hypothesis_queue_sha256": sha256(
                selection / "hypothesis_queue.csv"
            ),
            "particles": 128,
            "matrix_orientation": ORIENTATION,
            "stimulus_generator_encoding": "binary_any_stimulus",
            "chemical_identity_conditioned": False,
        },
    )
    pd.DataFrame(
        [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (
                raw,
                targeted_run / "manifest.json",
                selection / "hypothesis_queue.csv",
                selection / "atlas_matrices.npz",
            )
        ]
    ).to_csv(targeted / "input_checksums.csv", index=False)
    _write_json(
        targeted / "protocol.json",
        {
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "particle_count": 128,
            "independent_unit": "worm",
            "matrix_orientation": ORIENTATION,
            "signed_metrics": list(SIGNED_METRICS),
            "endpoint_quantiles": [0.25, 0.5, 0.75],
            "external_reference_firewall": "no external reference was loaded",
            "chemical_warning": (
                "chemical contexts are event-stratified under a binary-any-stimulus "
                "generator, not chemical-conditioned effects"
            ),
            "claim_boundary": INTERPRETATION_LIMIT,
        },
    )
    _write_json(
        targeted / "summary.json",
        {
            "status": "reviewed_model_relative",
            "n_worms": 17,
            "n_candidates": 2,
            "n_cell_rows": len(cell_rows),
            "n_quantile_rows": len(quantile_rows),
            "evidence_label_counts": {
                "model_relative_consistent": 36,
                "model_relative_descriptive": 6,
            },
            "screen_vs_n128": {
                "status": "computed",
                "rows": len(screen_rows),
                "direction_agreement_fraction": 1.0,
                "median_magnitude_agreement": 0.9,
            },
            "experimental_confirmation_label_assigned": False,
            "external_reference_data_used": False,
        },
    )
    _write_json(
        targeted / "validation.json",
        {
            "status": "passed",
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "created_utc": CREATED,
            "archive_count": 4,
            "expected_archive_count": 4,
            "n_worms": 17,
            "checks": {
                "full_requested_grid": True,
                "worm_level_inference": True,
                "target_row_source_column_orientation": True,
                "raw_input_checksums": True,
            },
        },
    )
    _write_json(
        targeted / "manifest.json",
        {
            "status": "complete",
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "matrix_orientation": ORIENTATION,
            "input_run_dir": str(targeted_run.resolve()),
            "input_run_manifest_sha256": sha256(
                targeted_run / "manifest.json"
            ),
            "input_queue_sha256": sha256(selection / "hypothesis_queue.csv"),
            "screen_dir": str(selection.resolve()),
            "created_utc": CREATED,
        },
    )
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    return targeted


def _write_external(root: Path, atlas: Path) -> Path:
    release = root / "reference_release"
    release_hashes: dict[str, str] = {}
    for index, relative in enumerate(REFERENCE_RELEASE_FILES):
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-reference-{index}".encode())
        release_hashes[relative] = sha256(path)

    sbtg_root = root / "sbtg_archive"
    sbtg_root.mkdir()
    archive = sbtg_root / "aligned_sbtg.npz"
    archive.write_bytes(b"synthetic aligned SBTG archive")
    _write_json(
        sbtg_root / "protocol.json",
        {"orientation": "all saved comparison matrices are [target, source]"},
    )
    _write_json(
        sbtg_root / "validation.json",
        {
            "status": "pass",
            "neuron_order_exact_match": True,
            "paired_orientation_transposed_once": True,
        },
    )
    (sbtg_root / "checksums.sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n"
    )
    sbtg_hashes = {
        name: sha256(sbtg_root / name)
        for name in (archive.name, "protocol.json", "validation.json", "checksums.sha256")
    }

    external = root / "postfreeze_external"
    external.mkdir()
    comparison_rows: list[dict[str, object]] = []
    for index, (panel, method, reference) in enumerate(
        (
            ("randi_cook_prespecified_flow_h1", "progressive_bridge_smc", "randi_wild_type"),
            ("randi_cook_prespecified_flow_h1", "direct_importance", "cook_struct_54"),
            ("randi_cook_contextual_sbtg_shared54_lineage_mismatch", "sbtg_current", "randi_wild_type"),
            ("randi_cook_contextual_sbtg_shared54_lineage_mismatch", "sbtg_published", "cook_struct_54"),
        )
    ):
        comparison_rows.append(
            {
                "panel": panel,
                "method": method,
                "channel": "endpoint_mean" if "sbtg" not in method else "score_product",
                "context": "state_average" if "sbtg" not in method else "historical",
                "reference_or_network": reference,
                "timing_label": "flow source-to-cut + H1" if "sbtg" not in method else "historical SBTG lag bin",
                "scope_or_grid": (
                    "all_estimated"
                    if method.startswith("sbtg")
                    else "support_qualified_sources"
                ),
                "auroc": 0.52 + 0.02 * index,
                "auprc": 0.18 + 0.01 * index,
                "continuous_spearman": 0.05 + 0.01 * index,
                "permutation_p": 0.2,
                "bh_q": 0.4,
                "comparison_note": "post-freeze contextual comparison",
            }
        )
    comparison_rows.append(
        {
            **comparison_rows[0],
            "panel": "neuromodulator_native_grid_lagmax",
            "reference_or_network": "monoamine_all",
        }
    )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(external / "dashboard_external_summary.csv", index=False)

    lag_rows: list[dict[str, object]] = []
    for network_index, (network, source_count) in enumerate(
        EXPECTED_NEUROMODULATOR_SOURCES.items()
    ):
        for method_index, method in enumerate(
            ("progressive_bridge_smc", "direct_importance")
        ):
            lag_rows.append(
                {
                    "method": method,
                    "channel": "endpoint_mean",
                    "context": "state_average",
                    "network": network,
                    "lag_grid": "native_method_grid",
                    "best_lag_frames": (1, 4, 8, 16)[network_index % 4],
                    "horizon_frames": 1,
                    "best_auroc": 0.5 + 0.01 * network_index + 0.005 * method_index,
                    "max_lag_permutation_p": 0.03 + 0.01 * network_index,
                    "max_lag_bh_q": 0.04 if network_index == 0 else 0.3,
                    "n_eligible_sources": source_count,
                    "n_positive": source_count * 8,
                    "lag_semantics": "repaired_source_window_end_to_prediction_cut",
                    "timing_comparability": "not a physical delay",
                }
            )
    lagmax = pd.DataFrame(lag_rows)
    lagmax.to_csv(external / "neuromodulator_lagmax_inference.csv", index=False)
    comparisons.iloc[:4].to_csv(external / "randi_cook_metrics.csv", index=False)
    comparisons.iloc[4:].to_csv(external / "neuromodulator_metrics.csv", index=False)
    comparisons.iloc[:2].to_csv(external / "primary_randi_cook_profile.csv", index=False)
    lagmax.iloc[:4].to_csv(external / "primary_neuromodulator_profile.csv", index=False)

    firewall = {
        "internal_manifest_sha256": sha256(atlas / "manifest.json"),
        "internal_protocol_sha256": sha256(atlas / "protocol.json"),
        "internal_validation_sha256": sha256(atlas / "validation.json"),
        "internal_atlas_matrices_sha256": sha256(
            atlas / "atlas_matrices.npz"
        ),
        "internal_checksums_sha256": sha256(atlas / "checksums.sha256"),
        "frozen_hypothesis_queue_sha256": sha256(atlas / "hypothesis_queue.csv"),
    }
    _write_json(
        external / "manifest.json",
        {
            "status": "passed",
            "created_utc": CREATED,
            "analysis_role": "post_freeze_external_convergence_only",
            "ranking_effect": "none; frozen internal ranking",
            "internal_firewall": firewall,
            "internal_firewall_reverified_after_analysis": True,
            "internal_prediction_ranking_inputs": [],
            "reference_release": str(release.resolve()),
            "reference_release_file_sha256": release_hashes,
            "sbtg_archive": str(archive.resolve()),
            "sbtg_archive_sha256": sha256(archive),
            "sbtg_archive_provenance_sha256": sbtg_hashes,
            "rows": {
                "dashboard_external_summary": len(comparisons),
                "neuromodulator_lagmax_inference": len(lagmax),
            },
        },
    )
    input_lines = [
        f"{digest}  reference_release/{relative}\n"
        for relative, digest in sorted(release_hashes.items())
    ]
    input_lines.append(
        f"{firewall['frozen_hypothesis_queue_sha256']}  internal_atlas/hypothesis_queue.csv\n"
    )
    input_lines.append(
        f"{firewall['internal_atlas_matrices_sha256']}  internal_atlas/atlas_matrices.npz\n"
    )
    input_lines.append(
        f"{firewall['internal_checksums_sha256']}  internal_atlas/checksums.sha256\n"
    )
    input_lines.extend(
        f"{digest}  sbtg_archive/{name}\n"
        for name, digest in sorted(sbtg_hashes.items())
    )
    (external / "input_checksums.sha256").write_text("".join(input_lines))
    _write_ledger(
        external,
        tuple(name for name in EXTERNAL_REQUIRED if name != "checksums.sha256"),
    )
    return external


def _write_docs(root: Path) -> Path:
    atlas_root = root / "atlas_root"
    atlas_root.mkdir()
    text = (
        "This technical record is not causal. A selected lag is not a physical delay. "
        "The generator uses a binary stimulus indicator and chemical panels are not "
        "chemically conditioned. Matrices use target rows and source columns. Worms are "
        "the independent unit, and external references are contextual only. "
    )
    for index, name in enumerate(ROOT_DOCUMENTS):
        (atlas_root / name).write_text(f"# Document {index}\n\n{text}{text}\n")
    return atlas_root


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path, tuple[str, ...]]:
    atlas, neurons = _write_atlas(tmp_path)
    targeted = _write_targeted(tmp_path, atlas, neurons)
    external = _write_external(tmp_path, atlas)
    atlas_root = _write_docs(tmp_path)
    return atlas, targeted, external, atlas_root, neurons


def _build(tmp_path: Path) -> tuple[dict[str, object], Path]:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    output = tmp_path / "report"
    artifact = build_prediction_atlas_report(
        atlas,
        targeted,
        external,
        atlas_root,
        output,
        provenance_root=tmp_path,
    )
    return artifact, output


def test_report_accepts_hyphenated_chemical_conditioning_boundary(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    for name in ROOT_DOCUMENTS:
        path = atlas_root / name
        path.write_text(
            path.read_text().replace(
                "not chemically conditioned",
                "not chemical-conditioned",
            )
        )

    artifact = build_prediction_atlas_report(
        atlas,
        targeted,
        external,
        atlas_root,
        tmp_path / "hyphenated_report",
        provenance_root=tmp_path,
    )
    assert artifact["surface"] == "report"


def test_report_is_bounded_source_backed_and_markdown_preserved(tmp_path: Path) -> None:
    artifact, output = _build(tmp_path)
    assert artifact["surface"] == "report"
    manifest = artifact["manifest"]
    assert manifest["surface"] == "report"
    assert manifest["blocks"][0]["body"].startswith("# Neural lag-effect atlas")
    assert manifest["blocks"][1]["body"].startswith("## Technical summary")
    block_ids = [block["id"] for block in manifest["blocks"]]
    assert block_ids.index("definitions_section") < block_ids.index("internal_section")
    assert block_ids.index("methodology_section") < block_ids.index("internal_section")
    chart_ids = {chart["id"] for chart in manifest["charts"]}
    assert {
        "internal_stability_chart",
        "targeted_three_arm_chart",
        "external_comparison_chart",
        "neuromod_lagmax_chart",
    } == chart_ids
    targeted_labels = {
        row["candidate_label"]
        for row in artifact["snapshot"]["datasets"]["targeted_primary"]
    }
    assert len(targeted_labels) == 2
    for index, block in enumerate(manifest["blocks"]):
        if block["type"] == "chart":
            assert index > 0 and manifest["blocks"][index - 1]["type"] == "markdown"
    source_ids = {source["id"] for source in artifact["sources"]}
    source_map = {source["id"]: source for source in artifact["sources"]}
    for card in manifest["cards"]:
        assert card["sourceId"] in source_ids
        assert source_map[card["sourceId"]]["query"]["sql"].startswith(
            "SELECT UNNEST(snapshot.datasets."
        )
    for item in (*manifest["charts"], *manifest["tables"]):
        assert item["sourceId"] in source_ids
        assert source_map[item["sourceId"]]["query"]["sql"].startswith(
            "SELECT UNNEST(snapshot.datasets."
        )
    lag_chart = next(
        chart for chart in manifest["charts"] if chart["id"] == "neuromod_lagmax_chart"
    )
    assert lag_chart["referenceLines"] == [
        {"axis": "x", "value": 0.5, "label": "Chance"}
    ]
    assert "n_positive" in {
        item["field"] for item in lag_chart["encodings"]["tooltip"]
    }
    external_table = next(
        table for table in manifest["tables"] if table["id"] == "external_comparison_table"
    )
    assert "scope_or_grid" in {column["field"] for column in external_table["columns"]}
    lag_table = next(
        table for table in manifest["tables"] if table["id"] == "neuromod_lagmax_table"
    )
    assert "n_positive" in {column["field"] for column in lag_table["columns"]}
    quantile_table = next(
        table for table in manifest["tables"] if table["id"] == "targeted_quantile_table"
    )
    assert {"pair_label", "context"}.issubset(
        {column["field"] for column in quantile_table["columns"]}
    )
    primary_table = next(
        table for table in manifest["tables"] if table["id"] == "targeted_primary_table"
    )
    assert {"metric", "contrast", "mean_normalized"}.issubset(
        {column["field"] for column in primary_table["columns"]}
    )
    for rows in artifact["snapshot"]["datasets"].values():
        assert len(rows) <= 2_000
        assert all(
            not isinstance(value, (dict, list, tuple))
            for row in rows
            for value in row.values()
        )
    assert len((output / "artifact.json").read_bytes()) < 3_000_000
    markdown = (output / "TECHNICAL_REPORT.md").read_text()
    for heading in (
        "## Technical summary",
        "## Internal support",
        "## Selected N128",
        "## Randi, Cook, and SBTG",
        "## Bentley neuromodulator",
        "## Metric, cohort, and method definitions",
        "## How repaired histories become directed lag matrices",
        "## Limitations and robustness boundaries",
        "## Recommended next steps",
        "## Further questions",
        "## Reproducibility inventory",
    ):
        assert heading in markdown
    assert "### Selected N128 endpoint quantile shifts" in markdown
    ledger = {
        name: digest
        for digest, name in (
            line.split("  ", 1)
            for line in (output / "checksums.sha256").read_text().splitlines()
        )
    }
    assert ledger == {
        "artifact.json": sha256(output / "artifact.json"),
        "TECHNICAL_REPORT.md": sha256(output / "TECHNICAL_REPORT.md"),
    }


def test_report_suppresses_mixed_estimand_three_arm_chart(tmp_path: Path) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    cells = pd.read_csv(targeted / "targeted_cells.csv")
    screen_path = targeted / "screen_consistency.csv"
    screen = pd.read_csv(screen_path)
    candidate = str(screen.iloc[1]["candidate_id"])
    matching = cells[
        (cells["candidate_id"].astype(str) == candidate)
        & (cells["contrast"].astype(str) == "high_low")
        & (cells["metric"].astype(str) == "pathwise_peak_mean")
    ]
    assert len(matching) == 1
    targeted_value = float(matching.iloc[0]["mean_normalized"])
    row_index = screen.index[screen["candidate_id"].astype(str) == candidate][0]
    screen_value = float(screen.loc[row_index, "screen_mean_normalized"])
    screen.loc[row_index, "screen_channel"] = "peak_mean"
    screen.loc[row_index, "targeted_metric"] = "pathwise_peak_mean"
    screen.loc[row_index, "targeted_n128_mean_normalized"] = targeted_value
    screen.loc[row_index, "targeted_minus_screen"] = targeted_value - screen_value
    screen.loc[row_index, "direction_agreement"] = float(
        np.sign(targeted_value) == np.sign(screen_value)
    )
    screen.loc[row_index, "magnitude_agreement"] = 1.0 - abs(
        targeted_value - screen_value
    ) / (abs(targeted_value) + abs(screen_value) + 1e-12)
    screen.to_csv(screen_path, index=False)
    summary_path = targeted / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["screen_vs_n128"]["direction_agreement_fraction"] = float(
        screen["direction_agreement"].mean()
    )
    summary["screen_vs_n128"]["median_magnitude_agreement"] = float(
        screen["magnitude_agreement"].median()
    )
    _write_json(summary_path, summary)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )

    artifact = build_prediction_atlas_report(
        atlas,
        targeted,
        external,
        atlas_root,
        tmp_path / "mixed_report",
        provenance_root=tmp_path,
    )
    chart_ids = {chart["id"] for chart in artifact["manifest"]["charts"]}
    assert "targeted_three_arm_chart" not in chart_ids
    primary_table = next(
        table
        for table in artifact["manifest"]["tables"]
        if table["id"] == "targeted_primary_table"
    )
    assert "unlike metrics are not pooled" in primary_table["subtitle"]
    primary_metrics = {
        row["metric"]
        for row in artifact["snapshot"]["datasets"]["targeted_primary"]
    }
    assert primary_metrics == {"endpoint_mean", "pathwise_peak_mean"}
    markdown = (tmp_path / "mixed_report" / "TECHNICAL_REPORT.md").read_text()
    assert "magnitudes from unlike metrics are not pooled" in markdown


def test_report_fails_closed_on_each_input_provenance_layer(tmp_path: Path) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    (atlas / "hypothesis_queue.csv").write_text("tampered\n")
    with pytest.raises(ReportInputError, match="checksum verification failed"):
        build_prediction_atlas_report(
            atlas, targeted, external, atlas_root, tmp_path / "out_atlas", provenance_root=tmp_path
        )


def test_report_validates_output_disjointness_before_overwrite_cleanup(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    sentinel = atlas / "artifact.json"
    sentinel.write_text("do not delete\n")

    with pytest.raises(ReportInputError, match="disjoint"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            atlas,
            provenance_root=tmp_path,
            overwrite=True,
        )

    assert sentinel.read_text() == "do not delete\n"

    nested_report = atlas_root / "technical_report"
    artifact = build_prediction_atlas_report(
        atlas,
        targeted,
        external,
        atlas_root,
        nested_report,
        provenance_root=tmp_path,
    )
    assert artifact["surface"] == "report"
    assert (nested_report / "artifact.json").is_file()

    collision = tmp_path / "collision_report"
    collision.mkdir()
    preserved = collision / "artifact.json"
    preserved.write_text("preserve this exact file\n")
    (collision / "report.html").mkdir()
    with pytest.raises(ReportInputError, match="not a regular file"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            collision,
            provenance_root=tmp_path,
            overwrite=True,
        )
    assert preserved.read_text() == "preserve this exact file\n"

    tmp_two = tmp_path / "second"
    tmp_two.mkdir()
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_two)
    (targeted / "targeted_cells.csv").write_text("tampered\n")
    with pytest.raises(ReportInputError, match="checksum verification failed"):
        build_prediction_atlas_report(
            atlas, targeted, external, atlas_root, tmp_two / "out_targeted", provenance_root=tmp_two
        )

    tmp_three = tmp_path / "third"
    tmp_three.mkdir()
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_three)
    reference = tmp_three / "reference_release" / REFERENCE_RELEASE_FILES[0]
    reference.write_bytes(b"changed after external analysis")
    with pytest.raises(ReportInputError, match="reference-release input changed"):
        build_prediction_atlas_report(
            atlas, targeted, external, atlas_root, tmp_three / "out_external", provenance_root=tmp_three
        )


def test_report_rejects_external_bundle_without_dense_atlas_lineage(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    manifest_path = external / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["internal_firewall"].pop("internal_atlas_matrices_sha256")
    _write_json(manifest_path, manifest)
    _write_ledger(
        external,
        tuple(name for name in EXTERNAL_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="not this frozen atlas"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            tmp_path / "out",
            provenance_root=tmp_path,
        )


def test_report_rejects_target_name_index_mismatch_even_with_fresh_ledger(tmp_path: Path) -> None:
    atlas, targeted, external, atlas_root, neurons = _bundle(tmp_path)
    cells = pd.read_csv(targeted / "targeted_cells.csv")
    cells.loc[0, "target_neuron"] = neurons[(int(cells.loc[0, "target_index"]) + 1) % 54]
    cells.to_csv(targeted / "targeted_cells.csv", index=False)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="frozen selected queue"):
        build_prediction_atlas_report(
            atlas, targeted, external, atlas_root, tmp_path / "out", provenance_root=tmp_path
        )


def test_report_rejects_unlinked_targeted_queue_and_count_drift(tmp_path: Path) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    manifest_path = targeted / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_queue_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="frozen atlas selection"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            tmp_path / "bad_link",
            provenance_root=tmp_path,
        )

    count_root = tmp_path / "count_case"
    count_root.mkdir()
    atlas, targeted, external, atlas_root, _ = _bundle(count_root)
    summary_path = targeted / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["evidence_label_counts"]["model_relative_consistent"] -= 1
    _write_json(summary_path, summary)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="evidence-label counts"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            count_root / "bad_counts",
            provenance_root=count_root,
        )


def test_report_rejects_self_consistent_raw_run_with_wrong_staged_queue(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    targeted_manifest_path = targeted / "manifest.json"
    targeted_manifest = json.loads(targeted_manifest_path.read_text())
    run_manifest_path = (
        Path(str(targeted_manifest["input_run_dir"])) / "manifest.json"
    )
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["hypothesis_queue"] = str(
        (atlas / "hypothesis_queue.csv").resolve()
    )
    run_manifest["hypothesis_queue_sha256"] = sha256(
        atlas / "hypothesis_queue.csv"
    )
    _write_json(run_manifest_path, run_manifest)
    targeted_manifest["input_run_manifest_sha256"] = sha256(run_manifest_path)
    _write_json(targeted_manifest_path, targeted_manifest)

    inventory_path = targeted / "input_checksums.csv"
    inventory = pd.read_csv(inventory_path)
    run_row = inventory["path"].astype(str) == str(run_manifest_path.resolve())
    assert int(run_row.sum()) == 1
    inventory.loc[run_row, "sha256"] = sha256(run_manifest_path)
    inventory.loc[run_row, "size_bytes"] = run_manifest_path.stat().st_size
    inventory.to_csv(inventory_path, index=False)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )

    with pytest.raises(ReportInputError, match="staged N128 queue"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            tmp_path / "out",
            provenance_root=tmp_path,
        )


def test_report_rejects_selection_maximum_above_canonical_six(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    targeted_manifest = json.loads((targeted / "manifest.json").read_text())
    selection = Path(str(targeted_manifest["screen_dir"]))
    selection_manifest_path = selection / "manifest.json"
    selection_manifest = json.loads(selection_manifest_path.read_text())
    selection_manifest["maximum_rows"] = 7
    _write_json(selection_manifest_path, selection_manifest)
    _write_ledger(
        selection,
        (
            "manifest.json",
            "hypothesis_queue.csv",
            "atlas_matrices.npz",
            "selection_rationale.md",
        ),
    )

    with pytest.raises(ReportInputError, match="frozen atlas selection"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            tmp_path / "out",
            provenance_root=tmp_path,
        )


def test_report_rejects_quantile_misattribution_and_invalid_diagnostic_ranges(
    tmp_path: Path,
) -> None:
    atlas, targeted, external, atlas_root, neurons = _bundle(tmp_path)
    quantile_path = targeted / "targeted_quantile_shifts.csv"
    quantiles = pd.read_csv(quantile_path)
    cells = pd.read_csv(targeted / "targeted_cells.csv")
    candidate = str(quantiles.loc[0, "candidate_id"])
    expected = cells[cells["candidate_id"].astype(str) == candidate].iloc[0]
    replacement_index = (int(expected.source_index) + 1) % len(neurons)
    quantiles.loc[0, "source_index"] = replacement_index
    quantiles.loc[0, "source_neuron"] = neurons[replacement_index]
    quantiles.to_csv(quantile_path, index=False)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="quantile metadata disagrees"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            tmp_path / "bad_quantile_metadata",
            provenance_root=tmp_path,
        )

    quantile_range_root = tmp_path / "quantile_range_case"
    quantile_range_root.mkdir()
    atlas, targeted, external, atlas_root, _ = _bundle(quantile_range_root)
    quantile_path = targeted / "targeted_quantile_shifts.csv"
    quantiles = pd.read_csv(quantile_path)
    quantiles.loc[0, "valid_fraction"] = 1.01
    quantiles.to_csv(quantile_path, index=False)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="quantile shifts fail range/orientation"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            quantile_range_root / "bad_quantile_range",
            provenance_root=quantile_range_root,
        )

    range_root = tmp_path / "support_range_case"
    range_root.mkdir()
    atlas, targeted, external, atlas_root, _ = _bundle(range_root)
    support_path = targeted / "support_diagnostics.csv"
    support = pd.read_csv(support_path)
    support.loc[0, "mean_ess_high"] = 129.0
    support.to_csv(support_path, index=False)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="support diagnostics fail grain/range"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            range_root / "bad_support_range",
            provenance_root=range_root,
        )


def test_report_rejects_targeted_signed_semantic_mismatch(tmp_path: Path) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    cells_path = targeted / "targeted_cells.csv"
    cells = pd.read_csv(cells_path)
    non_w1 = cells["metric"].astype(str) != "endpoint_wasserstein1"
    cells.loc[non_w1.idxmax(), "signed"] = False
    cells.to_csv(cells_path, index=False)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )

    with pytest.raises(ReportInputError, match="signed flags"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            tmp_path / "bad_signed",
            provenance_root=tmp_path,
        )


def test_report_fails_closed_on_validation_and_claim_ledgers(tmp_path: Path) -> None:
    atlas, targeted, external, atlas_root, _ = _bundle(tmp_path)
    atlas_validation_path = atlas / "validation.json"
    atlas_validation = json.loads(atlas_validation_path.read_text())
    check_name = next(iter(atlas_validation["archive_checks_completed"]))
    atlas_validation["archive_checks_completed"][check_name] = False
    _write_json(atlas_validation_path, atlas_validation)
    _write_ledger(
        atlas, tuple(name for name in ATLAS_REQUIRED if name != "checksums.sha256")
    )
    with pytest.raises(ReportInputError, match="archive checks"):
        build_prediction_atlas_report(
            atlas, targeted, external, atlas_root, tmp_path / "bad_atlas", provenance_root=tmp_path
        )

    target_root = tmp_path / "target_case"
    target_root.mkdir()
    atlas, targeted, external, atlas_root, _ = _bundle(target_root)
    targeted_validation_path = targeted / "validation.json"
    targeted_validation = json.loads(targeted_validation_path.read_text())
    targeted_validation["checks"]["worm_level_inference"] = False
    _write_json(targeted_validation_path, targeted_validation)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="validation did not pass completely"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            target_root / "bad_targeted",
            provenance_root=target_root,
        )

    targeted_validation["checks"]["worm_level_inference"] = True
    _write_json(targeted_validation_path, targeted_validation)
    _write_ledger(
        targeted,
        tuple(name for name in TARGETED_REQUIRED if name != "checksums.sha256"),
    )
    external_manifest_path = external / "manifest.json"
    external_manifest = json.loads(external_manifest_path.read_text())
    external_manifest["internal_firewall_reverified_after_analysis"] = False
    _write_json(external_manifest_path, external_manifest)
    _write_ledger(
        external,
        tuple(name for name in EXTERNAL_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="external firewall contract"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            target_root / "bad_external",
            provenance_root=target_root,
        )

    lag_root = tmp_path / "lag_case"
    lag_root.mkdir()
    atlas, targeted, external, atlas_root, _ = _bundle(lag_root)
    lag_path = external / "neuromodulator_lagmax_inference.csv"
    lag = pd.read_csv(lag_path)
    lag.loc[0, "max_lag_permutation_p"] = 1.5
    lag.to_csv(lag_path, index=False)
    _write_ledger(
        external,
        tuple(name for name in EXTERNAL_REQUIRED if name != "checksums.sha256"),
    )
    with pytest.raises(ReportInputError, match="AUROC/p/q"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            lag_root / "bad_lagmax",
            provenance_root=lag_root,
        )

    docs_root = tmp_path / "docs_case"
    docs_root.mkdir()
    atlas, targeted, external, atlas_root, _ = _bundle(docs_root)
    (atlas_root / "README.md").unlink()
    with pytest.raises(ReportInputError, match="root document is missing"):
        build_prediction_atlas_report(
            atlas,
            targeted,
            external,
            atlas_root,
            docs_root / "bad_docs",
            provenance_root=docs_root,
        )


def test_report_artifact_builds_with_installed_portable_builder(tmp_path: Path) -> None:
    _, output = _build(tmp_path)
    node = Path(
        "/Users/vik/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    )
    builder = Path(
        "/Users/vik/.codex/plugins/cache/openai-curated-remote/data-analytics/"
        "0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs"
    )
    if not node.is_file() or not builder.is_file():
        pytest.skip("installed portable report builder is unavailable")
    html = output / "report.html"
    result = subprocess.run(
        [
            str(node),
            str(builder),
            "--input",
            str(output / "artifact.json"),
            "--output",
            str(html),
            "--timeout-ms",
            "20000",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert html.is_file() and html.stat().st_size > 10_000
