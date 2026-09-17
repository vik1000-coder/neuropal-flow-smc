"""One-question Cook/Randi prototype using frozen saved results only.

No fitting, sampling, resampling, or inference occurs in this renderer. The four
point differences are copied from the existing F03 export; the interval bounds
are the original audit's reported three-decimal source-bootstrap bounds.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/figure_redesign_prototypes_20260831"
OLD = ROOT / "results/figure_atlas_20260831"
EXTERNAL = ROOT / "results/neural_prediction_atlas_20260829/postfreeze_external"
AUDIT = EXTERNAL.parent / "POSTFREEZE_EXTERNAL_AUDIT.md"
SCORES = OLD / "data/f03_fixed_lag_reference_rows.csv"
INTERVALS = OLD / "data/f03_reported_source_bootstrap_rows.csv"
OLD_MANIFEST = OLD / "reference_manifest.json"
BLUE = "#326B9B"
INK = "#242A2F"
GREY = "#737C84"
REFS = [
    ("randi_wild_type", "Randi (functional)", "Randi WT", 1349, 257),
    ("cook_struct_54", "Cook (all anatomical)", "Cook structural", 2862, 758),
    ("cook_chem_54", "Cook (chemical)", "Cook chemical", 2862, 699),
    ("cook_gap_54", "Cook (gap junctions)", "Cook gap", 2862, 169),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_ledger(path: Path, ledger: Path) -> dict:
    entries = {}
    for line in ledger.read_text().splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            entries[(ledger.parent / name).resolve()] = digest
    actual = sha256(path)
    assert entries[path.resolve()] == actual, f"Original ledger mismatch: {path}"
    return dict(path=relative(path), sha256=actual,
                verified_against=relative(ledger), ledger_sha256=sha256(ledger))


def verify_inputs() -> list[dict]:
    old_manifest = json.loads(OLD_MANIFEST.read_text())
    inventory = []
    for path in (SCORES, INTERVALS):
        actual = sha256(path)
        assert old_manifest["output_sha256"][relative(path)] == actual
        inventory.append(dict(path=relative(path), sha256=actual,
                              verified_against=relative(OLD_MANIFEST),
                              ledger_sha256=sha256(OLD_MANIFEST)))
    inventory.append(check_ledger(EXTERNAL / "randi_cook_metrics.csv",
                                  EXTERNAL / "checksums.sha256"))
    inventory.append(check_ledger(AUDIT, AUDIT.parent / "checksums.sha256"))
    return inventory


def extract_rows() -> list[dict]:
    scores = read_rows(SCORES)
    intervals = read_rows(INTERVALS)
    raw_scores = read_rows(EXTERNAL / "randi_cook_metrics.csv")
    assert len(intervals) == 4
    assert len({r["reference"] for r in intervals}) == 4
    table_start = AUDIT.read_text().split(
        "| Reference | Observed difference | Exploratory 95% interval |\n", 1
    )[1].splitlines()[1:5]
    audit_rows = {}
    for line in table_start:
        label, delta, bounds = [x.strip() for x in line.strip("|").split("|")]
        low, high = [float(x) for x in bounds.strip("[]").split(",")]
        audit_rows[label] = (float(delta), low, high)

    exported = []
    for ref, label, audit_label, count, positive in REFS:
        iv = next(r for r in intervals if r["reference"] == ref)
        methods = {}
        for method in ("progressive_bridge_smc", "sbtg_published"):
            selected = [r for r in scores if r["reference"] == ref and r["method"] == method]
            assert len(selected) == 1
            row = selected[0]
            assert row["scope"] == "all_estimated" and row["lag_frames"] == "1"
            assert int(row["n_edges"]) == count and int(row["n_positive"]) == positive
            if method == "progressive_bridge_smc":
                assert row["channel"] == "endpoint_mean"
                assert row["context"] == "state_average" and float(row["horizon_frames"]) == 1
            else:
                assert row["channel"] == "score_product" and row["context"] == "all_windows"
            original = [r for r in raw_scores if all(r[k] == row[k] for k in
                        ("method", "reference", "scope", "channel", "context", "lag_frames"))
                        and r["horizon_frames"] == row["horizon_frames"]]
            assert len(original) == 1
            assert math.isclose(float(original[0]["auroc"]), float(row["auroc"]), abs_tol=1e-14)
            methods[method] = row

        delta = float(iv["progressive_minus_published"])
        recomputed_difference = (float(methods["progressive_bridge_smc"]["auroc"])
                                 - float(methods["sbtg_published"]["auroc"]))
        assert math.isclose(delta, recomputed_difference, abs_tol=1e-14)
        low, high = float(iv["reported_ci_low"]), float(iv["reported_ci_high"])
        audit_delta, audit_low, audit_high = audit_rows[audit_label]
        assert low == audit_low and high == audit_high
        assert float(iv["reported_delta"]) == audit_delta
        assert abs(delta - audit_delta) < 0.0006
        assert low <= delta <= high
        assert iv["bootstrap_replicates"] == "500" and iv["bootstrap_unit"] == "source column"
        exported.append(dict(
            reference=ref, label=label,
            auroc_difference=iv["progressive_minus_published"],
            reported_ci_low=iv["reported_ci_low"], reported_ci_high=iv["reported_ci_high"],
            auroc_difference_units="raw AUROC units; progressive_bridge_smc minus sbtg_published",
            n_matched_pairs=count, n_positive_pairs=positive,
            progressive_auroc=methods["progressive_bridge_smc"]["auroc"],
            sbtg_published_auroc=methods["sbtg_published"]["auroc"],
            flow_selection="state_average endpoint_mean; source lag 1; forecast horizon 1",
            sbtg_selection="all_windows score_product; lag 1",
            mask="all_estimated; matched pairs on 54-class axis",
            interval_type="existing exploratory pointwise 95% source-column bootstrap",
            bootstrap_replicates=500, interval_precision=iv["interval_precision"],
            source_difference=relative(INTERVALS), source_interval=relative(AUDIT),
        ))
    return exported


def render(rows: list[dict]) -> dict:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 18,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "none", "svg.hashsalt": "reference-comparison-prototype",
    })
    fig, ax = plt.subplots(figsize=(12, 7), dpi=220)
    fig.subplots_adjust(left=0.305, right=0.95, bottom=0.29, top=0.765)
    fig.text(0.045, 0.95, "Cook and Randi: progressive SMC vs published SBTG",
             fontsize=20, weight="semibold", va="top")
    fig.text(0.045, 0.882, "Matching neuron pairs from the 54-class set",
             fontsize=15, va="top", color=GREY)

    ax.set_xlim(-0.06, 0.12)
    ax.set_ylim(3.5, -0.5)
    ax.set_yticks(range(4), [r["label"] for r in rows], fontsize=18)
    ax.tick_params(axis="y", length=0, pad=16)
    ax.set_xticks([-0.05, 0, 0.05, 0.10])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x:+.2f}"))
    ax.tick_params(axis="x", labelsize=16, length=4, color=GREY, pad=10)
    ax.set_xlabel("Difference in connection-ranking score (AUROC)", fontsize=17, labelpad=18)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#C5CBD0")
    ax.axvline(0, color=GREY, linewidth=1.4, linestyle=(0, (3, 3)), zorder=1)
    for value in (-0.05, 0.05, 0.10):
        ax.axvline(value, color="#E8EBED", linewidth=0.8, zorder=0)

    transform = ax.get_xaxis_transform()
    ax.text(-0.06, 1.06, "← SBTG higher", fontsize=13, color=GREY,
            ha="left", va="bottom", transform=transform)
    ax.text(0, 1.06, "Same AUROC", fontsize=12, color=GREY,
            ha="center", va="bottom", transform=transform)
    ax.text(0.12, 1.06, "Progressive higher →", fontsize=13, color=GREY,
            ha="right", va="bottom", transform=transform)
    for y, row in enumerate(rows):
        center = float(row["auroc_difference"])
        low, high = float(row["reported_ci_low"]), float(row["reported_ci_high"])
        ax.errorbar(center, y, xerr=[[center - low], [high - center]],
                    fmt="o", color=BLUE, markersize=9, elinewidth=2.4,
                    capsize=5, capthick=2.0, zorder=3)

    fig.text(0.045, 0.101, "Lines: existing 95% source-bootstrap intervals.",
             fontsize=12, va="bottom")
    fig.text(0.045, 0.061, "Exploratory and pointwise; not corrected tests of superiority.",
             fontsize=12, va="bottom")

    # Detect text clipping in the exported canvas, without changing the data.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    checked = 0
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= bounds.width + 1 and box.y1 <= bounds.height + 1, artist.get_text()
        checked += 1
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"reference_comparison.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)
    svg = ET.parse(OUT / "reference_comparison.svg")
    assert len(svg.findall(".//{http://www.w3.org/2000/svg}text")) >= 15
    return dict(figure_inches=[12, 7], png_pixels=[2640, 1540], dpi=220,
                title_points=20, row_label_points=18, caption_points=12,
                text_bounds_checked=checked, text_within_canvas=True,
                svg_xml_parsed=True, svg_editable_text=True)


def main() -> None:
    inventory = verify_inputs()
    rows = extract_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "reference_values.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checks = render(rows)
    outputs = [OUT / "reference_comparison.png", OUT / "reference_comparison.svg",
               csv_path, OUT / "REFERENCE_CAPTION.md"]
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="generated_pending_visual_review",
        question="How does progressive bridge SMC compare with published SBTG on Cook and Randi?",
        no_new_training_sampling_resampling_or_inference=True,
        plotted_metric="AUROC(progressive_bridge_smc) - AUROC(sbtg_published)",
        metric_definition="AUROC ranks the absolute estimated effect for reference-positive versus reference-negative neuron pairs. The displayed difference is in raw 0-to-1 AUROC units, not percent or percent accuracy.",
        direction="positive: progressive has the higher AUROC; zero: same AUROC",
        input_inventory=inventory,
        code=dict(path=relative(Path(__file__)), sha256=sha256(Path(__file__))),
        precision=dict(points="exact saved old F03 CSV differences", intervals="existing audit bounds reported to three decimals; not re-estimated"),
        fair_comparison=dict(axis="54 pooled head neuron classes", matched_pairs=True,
                             scope="all_estimated; no method-specific support filter",
                             randi="1349 classifiable pairs: 257 significant responses, 1092 confident negatives; ambiguous pairs excluded",
                             cook="2862 off-diagonal pairs; positives: all anatomical 758, chemical 699, gap junctions 169",
                             training_history="Flow: corrected 54-class atlas. Published SBTG: historical 80-class training lineage, restricted to the shared comparison axis; not identically trained models.",
                             timing="Flow endpoint mean, state-average, source lag 1, horizon 1 (0.50 s source-window-end to readout); published SBTG saved lag-1 score product is a different estimand."),
        uncertainty=dict(unit="paired source columns", replicates=500, confidence_level=0.95,
                         qualification="Exploratory post-hoc pointwise source sensitivity; not multiplicity-adjusted superiority testing. No animal, model-refit, or reference uncertainty is captured.",
                         result="Randi interval includes zero; all three Cook intervals do not. This is not a corrected or independently replicated superiority claim."),
        omitted_context=dict(companion="results/figure_atlas_20260831/REFERENCE_CAPTIONS.md",
                             other_metrics="Published SBTG retains slightly higher Cook-gap AUPRC despite lower AUROC. This chart concerns only the specified AUROC comparison, not overall model quality.",
                             historical80="The distinct historical80 sensitivity favors published SBTG; it is not a clean neuron-count comparison.",
                             other_methods="Direct importance and SBTG-current remain in the original F03 score export; neither is displayed here."),
        deterministic_checks=checks,
        output_sha256={relative(path): sha256(path) for path in outputs},
    )
    (OUT / "reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(rows=len(rows), outputs=[relative(p) for p in outputs], checks=checks), indent=2))


if __name__ == "__main__":
    main()
