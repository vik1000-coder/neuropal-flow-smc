"""Render frozen baseline-mean edge decisions; no new inference or sampling."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from common import ROOT, OUT, BLUE, INK, GREY, style, save_figure, sha256

ATLAS = ROOT / "results/neural_prediction_atlas_20260829"
STEM = "f12_baseline_mean_evidence_map"
SOURCES = {
    "axis": ATLAS / "canonical/atlas_matrices.npz",
    "edge_evidence": ATLAS / "complete_family_evidence_20260830/edge_evidence.csv",
    "joint_protocol": ATLAS / "complete_family_evidence_20260830/protocol.json",
    "baseline_protocol": ATLAS / "complete_family_inference_20260830/baseline_endpoint_mean/protocol.json",
    "active_protocol": ATLAS / "complete_family_inference_20260830/active_minus_baseline_endpoint_mean/protocol.json",
}
CATEGORIES = {
    0: "self_pair_not_tested",
    1: "not_support_eligible",
    2: "support_eligible_no_joint_discovery",
    3: "support_eligible_joint_p_le_0_05",
}
COLORS = ["#FFFFFF", "#E3E7EB", "#BFD2E2", BLUE]


def main():
    style()
    plt.rcParams["axes.titleweight"] = "bold"
    hashes = {key: sha256(path) for key, path in SOURCES.items()}
    with np.load(SOURCES["axis"]) as archive:
        neurons = archive["neurons"].astype(str).tolist()
    assert len(neurons) == 54 and len(set(neurons)) == 54
    all_edges = pd.read_csv(SOURCES["edge_evidence"])
    edges = all_edges[all_edges.family_id.eq("baseline_endpoint_mean")].copy()
    assert len(edges) == 2862
    assert not edges.duplicated(["target_index", "source_index"]).any()
    assert edges.target_index.ne(edges.source_index).all()
    assert edges.n_lag_horizon_cells.eq(24).all()
    assert edges.n_worms.eq(17).all()
    assert not (edges.support_eligible & edges.joint_primary_flat_lag_max_t_p_value.le(.05)).any()
    for row in edges.itertuples():
        assert neurons[row.target_index] == row.target_neuron
        assert neurons[row.source_index] == row.source_neuron
    lookup = edges.set_index(["target_index", "source_index"]).to_dict("index")
    matrix = np.full((54, 54), -1, dtype=np.int8)
    records = []
    for target_index, target in enumerate(neurons):
        for source_index, source in enumerate(neurons):
            if target_index == source_index:
                row = {"family_id": "baseline_endpoint_mean", "source_neuron": source,
                       "target_neuron": target, "support_eligible": False,
                       "joint_primary_edge_max_t_p_value": np.nan}
                category = 0
            else:
                row = lookup[(target_index, source_index)].copy()
                category = 1 if not row["support_eligible"] else (3 if row["joint_primary_edge_max_t_p_value"] <= .05 else 2)
            matrix[target_index, source_index] = category
            row.update({"target_index": target_index, "source_index": source_index,
                        "display_row_index": target_index, "display_column_index": source_index,
                        "self_pair": target_index == source_index,
                        "category_code": category, "evidence_category": CATEGORIES[category],
                        "axis_order": "unchanged canonical atlas neuron order on both axes",
                        "orientation": "target_rows_source_columns"})
            records.append(row)
    frame = pd.DataFrame(records)
    counts = {CATEGORIES[k]: int((matrix == k).sum()) for k in CATEGORIES}
    assert counts == {CATEGORIES[0]: 54, CATEGORIES[1]: 1378, CATEGORIES[2]: 1382, CATEGORIES[3]: 102}
    assert len(frame) == 2916 and int(edges.support_eligible.sum()) == 1484
    assert set(lookup) == {(t, s) for t in range(54) for s in range(54) if t != s}
    assert np.array_equal(frame.category_code.to_numpy().reshape(54, 54), matrix)
    assert (matrix.diagonal() == 0).all()
    for row in edges.itertuples():
        assert (matrix[row.target_index, row.source_index] == 3) == bool(row.support_eligible and row.joint_primary_edge_max_t_p_value <= .05)
    sentinel = next((t, s) for t in range(54) for s in range(54) if matrix[t, s] == 3 and matrix[s, t] != 3)
    frame["source_files"] = ";".join(str(path.relative_to(ROOT)) for path in SOURCES.values())
    frame["source_sha256"] = ";".join(hashes.values())
    data_path = OUT / "data" / f"{STEM}.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path, index=False)

    fig = plt.figure(figsize=(18, 19))
    ax = fig.add_axes([.085, .19, .88, .71])
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-.5, 4.5), 4)
    ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", origin="upper", aspect="equal")
    ax.set_xticks(np.arange(54), neurons, rotation=60, ha="right", rotation_mode="anchor", fontsize=11)
    ax.set_yticks(np.arange(54), neurons, fontsize=11)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_xlabel("Source neuron class →  (columns)", fontsize=13, labelpad=15)
    ax.set_ylabel("Target neuron class  (rows)", fontsize=13, labelpad=18)
    ax.set_xticks(np.arange(-.5, 54, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 54, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=.35, alpha=.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GREY)
        spine.set_linewidth(.7)
    fig.text(.055, .97, "F12  Baseline mean-response edge evidence", fontsize=22, fontweight="bold", ha="left", color=INK)
    fig.text(.055, .94, "54 class-level channels · progressive bridge SMC · existing joint four-family max-T, α = 0.05", fontsize=13, color=GREY)
    legend_labels = [
        "Self-pair: not tested  (54)",
        "Not support-eligible  (1,378)",
        "Eligible, no joint discovery  (1,382)",
        "Eligible, joint p ≤ 0.05  (102)",
    ]
    handles = [Patch(facecolor=color, edgecolor=GREY, linewidth=.6, label=label) for color, label in zip(COLORS, legend_labels)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .085), ncol=2,
               frameon=False, fontsize=12, columnspacing=3.2, labelspacing=1.2, handlelength=1.6)
    fig.text(.09, .055, "1,484 of 2,862 off-diagonal pairs meet the primary support gate. White diagonal cells are excluded, not zero effects.", fontsize=11, color=GREY)
    fig.text(.09, .031, "Edge decisions scan four source lags × six forecast horizons; colors are evidence categories, not effect strength.\nNo strong-support edge has resolved lag structure. Unsupported or non-discovered pairs are not biological negatives.", fontsize=11, color=GREY, linespacing=1.5)
    outputs = [str(data_path.relative_to(ROOT)), *save_figure(fig, STEM)]
    caption_path = OUT / "EVIDENCE_MAP_CAPTION.md"
    if caption_path.exists():
        outputs.append(str(caption_path.relative_to(ROOT)))
    manifest = {
        "status": "complete",
        "scope": "existing baseline-mean edge decisions on all canonical54 class pairs; no new tests, sampling, model fit, or neuron-category comparison",
        "script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve()),
                   "analysis/figure_atlas_20260831/common.py": sha256(Path(__file__).resolve().parent / "common.py")},
        "source_hashes": {str(path.relative_to(ROOT)): hashes[key] for key, path in SOURCES.items()},
        "outputs": {path: sha256(ROOT/path) for path in outputs},
        "checks": {"canonical_labels": 54, "matrix_shape": [54, 54], "data_rows": 2916,
                   "directed_off_diagonal_pairs": 2862, "support_eligible_pairs": 1484,
                   "resolved_strong_support_lag_edges": 0,
                   "category_counts": counts, "all_edge_decisions_match_source": True,
                   "axis_order": "unchanged canonical order on both axes",
                   "orientation": "target rows, source columns",
                   "asymmetric_orientation_sentinel": {"target": neurons[sentinel[0]], "source": neurons[sentinel[1]],
                                                       "forward_category": int(matrix[sentinel]), "reverse_category": int(matrix[sentinel[::-1]])}},
        "validation_rating": "share with caveats: fixed-model conditional evidence, incomplete source support, overlapping cross-validation training, no lag-localization or biological-causality claim",
        "visual_qa": {"status": "manual inspection required after regeneration"},
    }
    (OUT / "evidence_map_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": manifest["status"], "outputs": outputs, "checks": manifest["checks"]}, indent=2))


if __name__ == "__main__":
    main()
