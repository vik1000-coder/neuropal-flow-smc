"""Plot the five saved primary E30 tests and descriptive tail calibration.

This script selects and validates frozen rows; it performs no new inference,
sampling, fitting, or interval estimation. Display-unit scaling is linear only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import BLUE, GREY, INK, ORANGE, OUT, ROOT, save_figure, sha256, style


SOURCE = ROOT / "results/distribution_structure_20260831"
TESTS = [
    ("flow_shuffle_energy", "A  Keep flow draws joint", "Control: marginal-preserving shuffle",
     1e3, "Energy loss Δ (×10⁻³)", (-1.0, 0.45), [-1, -0.5, 0, 0.4]),
    ("flow_shuffle_variogram", "B  Keep flow draws joint", "Control: marginal-preserving shuffle",
     1e5, "Next-state variogram loss Δ (×10⁻⁵)", (-0.4, 2.9), [0, 1, 2]),
    ("changing_variance_crps", "C  History-dependent variance", "Control: fixed variance; same Gaussian mean",
     1e4, "Marginal CRPS loss Δ (×10⁻⁴)", (-0.5, 4.8), [0, 2, 4]),
    ("student_vs_gaussian_crps", "D  Student-t versus Gaussian", "Control: rank-8 Gaussian; independently fitted",
     1e4, "Marginal CRPS loss Δ (×10⁻⁴)", (-0.8, 9.3), [0, 3, 6, 9]),
    ("flow_vs_student_energy", "E  Flow versus Student-t", "Control: rank-8 Student-t; independently fitted",
     1e2, "Energy loss Δ (×10⁻²)", (-0.5, 6), [0, 2, 4, 6]),
]


def verify_sources(paths):
    hashes = dict((relative, digest) for digest, relative in
                  (line.split("  ", 1) for line in (SOURCE / "SHA256SUMS").read_text().splitlines()))
    entries = []
    for path in paths:
        relative = str(path.relative_to(SOURCE))
        actual = sha256(path)
        if hashes.get(relative) != actual:
            raise ValueError(f"Source ledger mismatch: {path}")
        entries.append(dict(path=str(path.relative_to(ROOT)), sha256=actual,
                            existing_ledger=str((SOURCE / "SHA256SUMS").relative_to(ROOT)), verified=True))
    return entries


def main():
    style()
    sources = [SOURCE / "README.md", SOURCE / "protocol.json", SOURCE / "analysis/REPORT.md",
               SOURCE / "analysis/paired_comparisons.csv", SOURCE / "analysis/model_scoreboard.csv"]
    inventory = verify_sources(sources)
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    frame = pd.read_csv(SOURCE / "analysis/paired_comparisons.csv")
    primary_flag = frame.primary.astype(str).str.lower().eq("true")
    primary = frame[primary_flag & frame.context.eq("all")].copy()
    expected = protocol["evaluation"]["primary_test_family"]
    assert set(primary.test) == set(expected) == {x[0] for x in TESTS}
    assert len(primary) == 5 and primary.n_worms.eq(17).all()
    assert primary.holm_p.notna().all() and primary.holm_p.between(0, 1).all()
    scoreboard = pd.read_csv(SOURCE / "analysis/model_scoreboard.csv").set_index("model")
    for row in primary.itertuples():
        control_mean = float(scoreboard.loc[row.control, row.metric])
        candidate_mean = float(scoreboard.loc[row.candidate, row.metric])
        assert np.isclose(control_mean - candidate_mean, row.delta, atol=1e-14)
        assert np.isclose(100 * row.delta / control_mean, row.percent, atol=1e-12)
    primary["control_score_mean"] = [float(scoreboard.loc[r.control, r.metric]) for r in primary.itertuples()]
    primary["candidate_score_mean"] = [float(scoreboard.loc[r.candidate, r.metric]) for r in primary.itertuples()]
    primary["display_scale_multiplier"] = primary.test.map({t[0]: t[3] for t in TESTS})
    primary["delta_definition"] = "control loss minus candidate loss; positive favors candidate"
    primary["ci_scope"] = "saved pointwise whole-worm bootstrap; not multiplicity adjusted"
    primary["p_scope"] = "saved exact sign-flip p with Holm correction across all five primary tests"
    tail = scoreboard.loc[["flow", "matched_student_t_rank8", "matched_gaussian_rank8"]].reset_index()
    assert tail.tail_frequency.nunique() == 1
    tail["comparison_scope"] = "descriptive secondary tail calibration; no new significance test"
    tail["tail_event"] = protocol["evaluation"]["tail_event"]
    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    primary_path = data_dir / "f11_five_primary_saved_comparisons.csv"
    tail_path = data_dir / "f11_descriptive_tail_calibration.csv"
    primary.to_csv(primary_path, index=False)
    tail.to_csv(tail_path, index=False)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.8))
    fig.subplots_adjust(left=0.065, right=0.97, top=0.78, bottom=0.23, hspace=0.80, wspace=0.30)
    fig.text(0.045, 0.97, "F11  Conditional-distribution prediction audit", va="top", fontsize=19, weight="bold")
    fig.text(0.045, 0.925,
             "Five saved primary tests · 17 held-out worms · 54 head classes · 20 s history → next 0.25 s · all contexts",
             va="top", fontsize=11)
    fig.text(0.045, 0.888,
             "Lower loss is better. Each point is control − candidate; positive favors the candidate. Panels use different score units and scales.",
             va="top", fontsize=11)
    legend = [Line2D([], [], marker="o", markersize=7, linestyle="none", color=BLUE,
                     markerfacecolor=BLUE, label="Holm p < 0.05"),
              Line2D([], [], marker="o", markersize=7, linestyle="none", color=BLUE,
                     markerfacecolor="white", label="Holm p ≥ 0.05")]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.97, 0.975),
               frameon=False, ncol=2, fontsize=10)
    indexed = primary.set_index("test")
    for ax, config in zip(axes.flat, TESTS):
        name, heading, control_label, multiplier, xlabel, xlim, ticks = config
        row = indexed.loc[name]
        delta, lo, hi = [float(row[x]) * multiplier for x in ["delta", "ci_low", "ci_high"]]
        ax.set_title(heading, loc="left", pad=31, fontsize=12, weight="bold")
        ax.text(0, 1.08, control_label, transform=ax.transAxes, fontsize=10, va="bottom")
        ax.axvline(0, color=INK, linestyle=":", linewidth=1.2)
        ax.errorbar(delta, 0.0, xerr=[[delta - lo], [hi - delta]], fmt="o", color=BLUE,
                    markersize=8, markerfacecolor=BLUE if row.holm_p < 0.05 else "white",
                    capsize=5, elinewidth=2, markeredgewidth=1.8)
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks)
        ax.set_ylim(-0.8, 0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        ax.set_xlabel(xlabel, fontsize=10.5)
        p = f"{row.holm_p:.3f}" if row.holm_p >= .001 else f"{row.holm_p:.7f}"
        ax.text(0.02, 0.86, f"Holm p = {p}  ·  {int(row.positive_worms)}/17 worms improve",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.text(0.02, 0.12, f"Δ {delta:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax = axes[1, 2]
    names = ["Flow", "Student-t, rank 8", "Gaussian, rank 8"]
    colors = [BLUE, ORANGE, GREY]
    for i, (row, color) in enumerate(zip(tail.itertuples(), colors)):
        ax.plot(row.tail_probability * 100, i, marker="o", color=color, markersize=7)
        dx = -0.25 if row.model == "matched_gaussian_rank8" else 0.22
        ax.text(row.tail_probability * 100 + dx, i, f"{row.tail_probability * 100:.2f}%",
                fontsize=10, va="center", ha="right" if dx < 0 else "left",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.7))
    observed = float(tail.tail_frequency.iloc[0]) * 100
    ax.axvline(observed, color=INK, linestyle=":", linewidth=1.4)
    ax.set_title("F  Large-innovation probability", loc="left", pad=31, fontsize=12, weight="bold")
    ax.text(0, 1.08, "Descriptive secondary check · no displayed intervals", transform=ax.transAxes, fontsize=10, va="bottom")
    ax.set_yticks(range(3), names)
    ax.set_ylim(2.7, -0.6)
    ax.set_xlim(0, 12.4)
    ax.set_xticks([0, 3, 6, 9, 12], ["0%", "3%", "6%", "9%", "12%"])
    ax.set_xlabel(f"Mean predicted probability · observed = {observed:.2f}%", fontsize=10.5)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0, labelsize=10)
    notes = [
        "Intervals: saved pointwise worm-bootstrap CIs. P-values: saved sign-flip tests, Holm-adjusted across all five primary comparisons.",
        "A positive pointwise CI need not pass the adjusted test (panel B). Different losses are not commensurate; do not compare bar/interval lengths across panels.",
        "Fitted CV models overlap; previously studied worms are not independent biological confirmation. Means are shared only in the variance control (C).",
        "Tail event: |next − current| > 2 × training-fold RMS innovation per neuron. No claim of neuronal independence, identified shape mechanism, or lag recovery.",
    ]
    for i, note in enumerate(notes):
        fig.text(0.045, 0.135 - i * 0.026, note, fontsize=10, va="top")
    files = save_figure(fig, "f11_predictive_distribution_audit")
    files += [str(primary_path.relative_to(ROOT)), str(tail_path.relative_to(ROOT))]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="generated_pending_visual_review",
                    no_new_inference_sampling_training=True, input_inventory=inventory,
                    protocol_fingerprint=protocol["fingerprint"], plotted_primary_rows=5, descriptive_tail_rows=3,
                    minimum_font_points=10,
                    primary_scope="all-context, all five frozen primary tests; equal worm weights after nuisance averaging",
                    transformations="Only exact power-of-ten axis scaling and percent display for probabilities; original deltas/CIs/p-values retained in CSV",
                    meaning="Positive raw loss difference favors candidate; losses have different units; no commensurability claim",
                    uncertainty="Pointwise saved bootstrap intervals; Holm p over all five tests; fixed-model/CV-overlap caveats",
                    code_sha256={str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
                                 str(Path(__file__).with_name("common.py").relative_to(ROOT)): sha256(Path(__file__).with_name("common.py"))},
                    output_sha256={f: sha256(ROOT / f) for f in files})
    (OUT / "density_audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": files, "primary_rows": 5, "source_checksums": "all five verified"}, indent=2))


if __name__ == "__main__":
    main()
