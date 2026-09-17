"""Static figures from frozen evidence only; never create new tests or samples."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import ROOT, OUT, BLUE, ORANGE, INK, GREY, style, save_figure, sha256

ATLAS = ROOT / "results/neural_prediction_atlas_20260829"
EVIDENCE = ATLAS / "complete_family_evidence_20260830"
CONTROL = ATLAS / "sampling_null_controls_combined8_n128_20260830/analysis"
SOURCES = {
    "families": EVIDENCE / "family_summary.csv",
    "edges": EVIDENCE / "edge_evidence.csv",
    "cells": EVIDENCE / "cell_evidence.parquet",
    "atlas": ATLAS / "canonical/atlas_matrices.npz",
    "protocol": ATLAS / "canonical/protocol.json",
    "evidence_protocol": EVIDENCE / "protocol.json",
    "mean_family_protocol": ATLAS / "complete_family_inference_20260830/baseline_endpoint_mean/protocol.json",
    "active_mean_family_protocol": ATLAS / "complete_family_inference_20260830/active_minus_baseline_endpoint_mean/protocol.json",
    "controls": CONTROL / "sham_calibration.csv",
    "control_protocol": CONTROL / "protocol.json",
    "primary_screen": ATLAS / "targeted_confirmation/analysis/screen_consistency.csv",
    "distribution_screen": ATLAS / "targeted_confirmation_distributional/analysis/screen_consistency.csv",
}
HASHES = {key: sha256(path) for key, path in SOURCES.items()}
FILES = []


def export_table(frame, stem, sources):
    frame = frame.copy()
    frame["source_files"] = ";".join(str(SOURCES[key].relative_to(ROOT)) for key in sources)
    frame["source_sha256"] = ";".join(HASHES[key] for key in sources)
    path = OUT / "data" / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    FILES.append(str(path.relative_to(ROOT)))


def save(fig, stem):
    FILES.extend(save_figure(fig, stem))


def title(fig, heading, subtitle):
    fig.suptitle(heading, x=0.03, y=0.985, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.03, 0.94, subtitle, ha="left", va="top", fontsize=11, color=GREY)


def ptext(value):
    return f"{value:.2g}" if value < 0.001 else f"{value:.3f}"


def family_figure(families, edges):
    assert families.joint_primary_edge_max_t_discoveries.tolist() == [102, 0, 0, 0]
    assert families.joint_primary_cell_max_t_discoveries.tolist() == [492, 0, 0, 0]
    assert families.joint_primary_flat_lag_edge_discoveries.sum() == 0
    qualifying = edges[(edges.support_eligible) & (edges.joint_primary_edge_max_t_p_value <= .05)].copy()
    assert len(qualifying) == 102 and qualifying.family_id.eq("baseline_endpoint_mean").all()
    qualifying = qualifying.sort_values("joint_edge_rank")
    top = qualifying.head(6).copy()
    top["selection_rule"] = "first six saved joint_edge_rank among support-eligible joint-primary edge max-T p<=0.05"
    export_table(families, "f06_four_family_decisions", ["families", "evidence_protocol", "mean_family_protocol"])
    export_table(qualifying, "f06_full_qualifying_mean_edges", ["edges"])
    export_table(top, "f06_displayed_mean_candidates", ["edges"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.04, 1]})
    fig.subplots_adjust(top=.77, bottom=.18, left=.20, right=.97, wspace=.53)
    title(fig, "F06  Existing significance decisions", "Corrected 54-class atlas · progressive bridge SMC · 17 held-out worms · no new tests")
    labels = ["Baseline: mean", "Baseline: log-SD", "Active − baseline: mean", "Active − baseline: log-SD"]
    y = np.arange(4)
    ax = axes[0]
    ax.barh(y, families.joint_primary_edge_max_t_discoveries, color=BLUE, height=.55)
    for i, row in families.iterrows():
        n = int(row.joint_primary_edge_max_t_discoveries)
        ax.text(n+2, i, f"{n} / {int(row.support_eligible_edges):,} eligible", va="center", fontsize=10)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 155)
    ax.set_xlabel("Edges passing joint four-family max-T")
    ax.set_title("A  Effect different from zero", loc="left", pad=12)
    ax.grid(axis="x")
    ax.text(0, -.23, "492 mean-effect cells pass; other families: 0.\nResolved lag structure: 0 edges in every family.", transform=ax.transAxes, fontsize=10, va="top")

    ax = axes[1]
    yy = np.arange(len(top))
    effect = top.joint_primary_edge_max_t_p_value.to_numpy()
    lag = top.joint_primary_flat_lag_max_t_p_value.to_numpy()
    ax.scatter(effect, yy-.11, color=BLUE, s=45, marker="o", label="Effect vs zero", zorder=3)
    ax.scatter(lag, yy+.11, facecolors="white", edgecolors=ORANGE, s=50, marker="D", label="Difference across lags", zorder=3)
    ax.axvline(.05, color=GREY, ls="--", lw=1)
    ax.text(.048, 5.25, "α = 0.05", ha="right", fontsize=10, color=GREY)
    ax.set_yticks(yy, [f"{r.source_neuron} → {r.target_neuron}" for _, r in top.iterrows()])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(8e-6, 1.45)
    ax.set_xlabel("Saved joint-adjusted p-value (log scale)")
    ax.set_title("B  Highest-ranked mean candidates", loc="left", pad=12)
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(.48, -.20), frameon=False, ncol=1)
    save(fig, "f06_existing_family_and_candidate_evidence")
    return qualifying


def mean_profiles(qualifying, cells, controls):
    selected = qualifying.head(4)
    result = []
    with np.load(SOURCES["atlas"]) as atlas:
        neurons = list(atlas["neurons"].astype(str))
        lags = atlas["source_lag_frames"].astype(int)
        horizons = atlas["horizon_frames"].astype(int)
        key = "__progressive_bridge_smc__endpoint_mean__baseline"
        means = atlas["mean_normalized"+key]
        lo = atlas["ci_low_normalized"+key]
        hi = atlas["ci_high_normalized"+key]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
        fig.subplots_adjust(top=.81, bottom=.16, left=.10, right=.97, hspace=.55, wspace=.30)
        title(fig, "F07  Mean-response profiles for the top adjusted-statistic candidates", "Baseline · first four saved edge ranks · each panel uses that edge's saved peak forecast horizon")
        for panel, ((_, edge), ax) in enumerate(zip(selected.iterrows(), axes.flat)):
            si, ti = neurons.index(edge.source_neuron), neurons.index(edge.target_neuron)
            horizon = int(edge.peak_horizon_frames)
            h = list(horizons).index(horizon)
            mu, low, high = means[:, h, ti, si], lo[:, h, ti, si], hi[:, h, ti, si]
            cell = cells[(cells.family_id == "baseline_endpoint_mean") & (cells.source_neuron == edge.source_neuron) &
                         (cells.target_neuron == edge.target_neuron) & (cells.horizon_frames == horizon)].sort_values("source_lag_frames")
            assert np.allclose(cell.mean_normalized, mu, atol=1e-6) and len(cell) == 4
            row = cell.copy()
            row["archived_ci_low"] = low
            row["archived_ci_high"] = high
            row["interval_scope"] = "archived pointwise worm interval; frozen-model and selected-horizon conditional"
            row["selection_rule"] = "first four saved significant edge ranks; saved peak horizon; all source lags shown"
            result.append(row)
            ax.axhline(0, color=GREY, lw=1)
            ax.errorbar(lags, mu, yerr=np.vstack([mu-low, high-mu]), fmt="o", color=BLUE,
                        capsize=4, ms=6, lw=1.6)
            ax.set_title(f"{chr(65+panel)}  {edge.source_neuron} → {edge.target_neuron}  |  H{horizon} = {horizon/4:g} s", loc="left")
            ax.set_xticks(lags, ["1\n0.25 s", "4\n1 s", "8\n2 s", "16\n4 s"])
            ax.set_xlim(-.3, 17.3)
            ax.set_ylabel("Normalized mean response")
            ax.grid(axis="y")
            ax.text(.02, .95, f"effect p={ptext(edge.joint_primary_edge_max_t_p_value)}\nlag-structure p={ptext(edge.joint_primary_flat_lag_max_t_p_value)}",
                    transform=ax.transAxes, va="top", fontsize=10)
            lower = min(float(low.min()), 0)
            upper = max(float(high.max()), 0)
            ax.set_ylim(lower - (upper-lower)*.08, upper + (upper-lower)*.42)
        for ax in axes[1]:
            ax.set_xlabel("Source lag: frames / window-end-to-cut seconds")
        fig.text(.10, .035, "Bars are pointwise intervals, not simultaneous lag comparisons. Selected horizons and y-scales differ.\nNo panel establishes a unique lag, physical delay, or experimental causal effect.", fontsize=10, color=GREY)
        save(fig, "f07_top_mean_candidate_lag_profiles")
    export_table(pd.concat(result), "f07_plotted_lag_profiles", ["edges", "cells", "atlas", "protocol"])


def followup_figures(distribution, primary, controls):
    assert len(distribution) == 6 and len(primary) == 6 and len(controls) == 24
    distribution = distribution.copy()
    for _, row in distribution.iterrows():
        matched = controls[(controls.source_neuron == row.source_neuron) & (controls.target_neuron == row.target_neuron) &
                           (controls.context == row.context) & (controls.source_lag_frames == row.source_lag_frames) &
                           (controls.horizon_frames == row.horizon_frames) & (controls.metric == row.targeted_metric)]
        assert len(matched) == 0, "A matched control exists: update the distributional panel and caveat"
    distribution["sampling_controls_status"] = "not available for this exact selected channel/context/lag/horizon; do not borrow from other panels"
    distribution["evidence_scope"] = "selection-conditioned particle escalation; no full-grid W1 or SD adjusted significance supplied"
    export_table(distribution, "f08_distributional_screen_inventory", ["distribution_screen"])
    export_table(primary, "f08_primary_screen_inventory", ["primary_screen"])
    export_table(controls, "f08_all_sampler_control_evidence", ["controls"])
    metrics = {"endpoint_sd": "SD change", "endpoint_log_sd": "log-SD change", "endpoint_wasserstein1": "W1 distance"}
    contexts = {"baseline": "baseline", "butanone_onset_minus_baseline": "butanone onset − baseline", "pentanedione_onset_minus_baseline": "pentanedione onset − baseline", "onset_minus_baseline": "pooled onset − baseline"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8))
    fig.subplots_adjust(top=.82, bottom=.16, left=.08, right=.98, hspace=.62, wspace=.38)
    title(fig, "F08a  Six selected distributional predictions: screen versus N128", "All six saved distributional follow-ups · selected/exploratory · matching sampler family, not independent replication")
    for k, ((_, row), ax) in enumerate(zip(distribution.iterrows(), axes.flat)):
        metric = metrics[row.targeted_metric]
        if row.context.endswith("_minus_baseline") and row.targeted_metric == "endpoint_wasserstein1":
            metric = "Change in W1 distance"
        ax.axhline(0, color=GREY, lw=1)
        ax.plot([0, 1], [row.screen_mean_normalized, row.targeted_n128_mean_normalized], color=GREY, lw=1)
        ax.scatter(0, row.screen_mean_normalized, marker="o", facecolors="white", edgecolors=GREY, s=60, zorder=3)
        ax.scatter(1, row.targeted_n128_mean_normalized, marker="D", color=BLUE, s=50, zorder=3)
        ax.set_xticks([0, 1], ["Screen N32", "Rerun N128"])
        ax.set_xlim(-.4, 1.4)
        ax.set_title(f"{chr(65+k)}  {row.source_neuron} → {row.target_neuron}\n{contexts[row.context]}\nL{int(row.source_lag_frames)} / H{int(row.horizon_frames)}",
                     loc="left", fontsize=11, pad=10)
        ax.set_ylabel(metric+" (normalized)")
        vals = [0, row.screen_mean_normalized, row.targeted_n128_mean_normalized]
        span = max(max(vals)-min(vals), .01)
        ax.set_ylim(min(vals)-span*.24, max(vals)+span*.25)
        for x, y in [(0, row.screen_mean_normalized), (1, row.targeted_n128_mean_normalized)]:
            ax.annotate(f"{y:+.4f}" if metric != "W1 distance" else f"{y:.4f}", (x, y), xytext=(0, 9 if y>=0 else -17), textcoords="offset points", ha="center", fontsize=10)
        ax.grid(axis="y")
    fig.text(.08, .04, "Panel A is unsigned W1; panel C is a signed difference of W1 distances. Panel scales differ by outcome.\nNo matched sampler-null controls are available for these six exact selections. Chemical names are event strata of a binary-stimulus model.", fontsize=10, color=GREY)
    save(fig, "f08a_distributional_screen_to_n128")

    selected_flp = controls[(controls.source_neuron == "FLP") & (controls.target_neuron == "ADE") &
                           (controls.context == "baseline") & (controls.source_lag_frames == 1) & (controls.horizon_frames == 8)]
    selected_rip = controls[(controls.source_neuron == "RIP") & (controls.target_neuron == "URB") &
                           (controls.context == "baseline") & (controls.source_lag_frames == 4) & (controls.horizon_frames == 2)]
    selected = pd.concat([selected_flp, selected_rip]).copy()
    assert len(selected) == 6 and len(selected_flp) == len(selected_rip) == 3
    assert selected.support_pass.all() and selected.selection_eligible.all()
    selected["selection_rule"] = "all three calibrated metrics for the first two saved strong-primary candidates: FLP-to-ADE L1 H8 and RIP-to-URB L4 H2, baseline"
    selected["positive_sampling_excess_gate_pass"] = ((selected.sampling_excess_mean > 0) &
        (selected.sampling_excess_ci_2_5 > 0) & (selected.sampling_excess_joint_max_t_p <= .05))
    selected["positive_quiet_excess_gate_pass"] = ((selected.temporal_specificity_excess_mean > 0) &
        (selected.temporal_specificity_ci_2_5 > 0) & (selected.temporal_specificity_joint_max_t_p <= .05))
    assert selected.positive_sampling_excess_gate_pass.tolist() == [True, False, False, True, False, True]
    assert not selected.positive_quiet_excess_gate_pass.any()
    export_table(selected, "f08b_plotted_sampler_control_evidence", ["controls", "control_protocol"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.7), sharey="col")
    fig.subplots_adjust(top=.81, bottom=.20, left=.08, right=.97, wspace=.38, hspace=.46)
    title(fig, "F08b  Matched controls for two baseline predictions", "N128 · all three outcomes per pair · existing pointwise intervals and joint 64-test max-T p-values")
    for k, metric in enumerate(["endpoint_mean", "endpoint_log_sd", "endpoint_wasserstein1"]):
        current = selected[selected.metric == metric]
        lower = min(current.sampling_excess_ci_2_5.min(), current.temporal_specificity_ci_2_5.min(), 0)
        upper = max(current.sampling_excess_ci_97_5.max(), current.temporal_specificity_ci_97_5.max(), 0)
        span = max(upper-lower, .01)
        for panel_row, source in enumerate(["FLP", "RIP"]):
            row = current[current.source_neuron == source].iloc[0]
            ax = axes[panel_row, k]
            xs = np.arange(2)
            means = np.array([row.sampling_excess_mean, row.temporal_specificity_excess_mean])
            low = np.array([row.sampling_excess_ci_2_5, row.temporal_specificity_ci_2_5])
            high = np.array([row.sampling_excess_ci_97_5, row.temporal_specificity_ci_97_5])
            ps = [row.sampling_excess_joint_max_t_p, row.temporal_specificity_joint_max_t_p]
            ax.axhline(0, color=GREY, ls="--", lw=1)
            for x, mu, ll, hh, color in zip(xs, means, low, high, [BLUE, GREY]):
                ax.errorbar(x, mu, yerr=[[mu-ll], [hh-mu]], fmt="o" if x==0 else "D", color=color, capsize=4, ms=6)
            ax.set_xticks(xs, ["Above sampler\ncontrol", "Above quiet-time\ncontrol"])
            ax.set_xlim(-.5, 1.5)
            metric_label = {"endpoint_mean":"Mean", "endpoint_log_sd":"log-SD", "endpoint_wasserstein1":"W1"}[metric]
            ax.set_title(f"{chr(65+panel_row*3+k)}  {metric_label}\n{row.source_neuron} → {row.target_neuron}, baseline L{int(row.source_lag_frames)} / H{int(row.horizon_frames)}",
                         loc="left", fontsize=12, pad=12)
            ax.set_ylabel("Excess response magnitude (normalized)")
            ax.set_ylim(lower-span*.15, upper+span*.48)
            for x, p in zip(xs, ps):
                ax.text(x, upper+span*.25, f"p={ptext(p)}", ha="center", fontsize=10)
            ax.grid(axis="y")
    fig.text(.08, .065, "Mean passes the positive sampler-excess gate for both pairs; W1 also passes for RIP → URB. None passes the quiet-time gate.\nShared y-scales within columns. Negative log-SD excess is not stronger-than-control evidence.\nW1 can reflect a mean shift: it does not establish mean-independent shape or gain modulation, or a causal effect.", fontsize=10, color=GREY, linespacing=1.6)
    save(fig, "f08b_matched_sampling_and_quiet_controls")

    questions = []
    biological_questions = {
        "targeted_0001": "Does a feasible AWC source perturbation shift the full AVA endpoint distribution 8 seconds after the cut in baseline conditions?",
        "targeted_0002": "Does a feasible ASH source perturbation reduce OLQ endpoint spread 8 seconds after the cut in baseline conditions?",
        "targeted_0003": "Is the ASE-to-ASK distributional response smaller during observed butanone onset than baseline at the 0.25-second readout?",
        "targeted_0004": "Does observed pentanedione onset modify the AFD-to-URY log-SD response at the 0.5-second readout?",
        "targeted_0005": "Does stimulus onset modify the AWA-to-RIB SD response at the 2-second readout?",
        "targeted_0006": "Does stimulus onset modify the AIM-to-AIB log-SD response at the 0.25-second readout?",
    }
    rerun_interpretations = {
        "targeted_0001": "nonnegative W1 remains but shrinks from 0.606745 to 0.290554; exact matched null still needed",
        "targeted_0002": "SD reduction direction and much of its magnitude persist; exact matched null still needed",
        "targeted_0003": "signed context contrast shrinks toward zero; sign agreement alone is weak evidence",
        "targeted_0004": "N128 context contrast is essentially zero despite preserved negative sign",
        "targeted_0005": "N128 context contrast is small relative to its screen value",
        "targeted_0006": "N128 context contrast reverses sign and is near zero; direction is not stable",
    }
    for _, row in distribution.iterrows():
        questions.append({"source": row.source_neuron, "target": row.target_neuron, "context": row.context,
                          "lag_frames": row.source_lag_frames, "horizon_frames": row.horizon_frames,
                          "source_window_end_before_cut_seconds": row.source_lag_frames / 4,
                          "forecast_endpoint_after_cut_seconds": row.horizon_frames / 4,
                          "outcome": row.targeted_metric, "question": biological_questions[row.candidate_id],
                          "existing_rerun_result": rerun_interpretations[row.candidate_id],
                          "status": "exploratory selected prediction; particle-rerun agreement is not biological confirmation",
                          "missing": "exact-cell sampler and quiet controls; independent biological validation; predeclared meaningful effect threshold",
                          "gain_caveat": "variance or SD change alone does not identify gain modulation",
                          "stimulus_caveat": "chemical strata label observed events; generator uses binary-any-stimulus input"})
    export_table(pd.DataFrame(questions), "f08_biological_question_inventory", ["distribution_screen"])


def main():
    style()
    plt.rcParams["axes.titleweight"] = "bold"
    families = pd.read_csv(SOURCES["families"])
    edges = pd.read_csv(SOURCES["edges"])
    cells = pd.read_parquet(SOURCES["cells"])
    qualifying_cells = cells[cells.support_eligible & (cells.joint_primary_cell_max_t_p_value <= .05)].copy()
    assert len(qualifying_cells) == 492 and qualifying_cells.family_id.eq("baseline_endpoint_mean").all()
    export_table(qualifying_cells, "f06_full_qualifying_mean_cells", ["cells", "evidence_protocol"])
    controls = pd.read_csv(SOURCES["controls"])
    primary = pd.read_csv(SOURCES["primary_screen"])
    distribution = pd.read_csv(SOURCES["distribution_screen"])
    qualifying = family_figure(families, edges)
    mean_profiles(qualifying, cells, controls)
    followup_figures(distribution, primary, controls)
    caption_path = OUT / "PREDICTION_CAPTIONS.md"
    if caption_path.exists():
        FILES.append(str(caption_path.relative_to(ROOT)))
    manifest = {"status": "complete", "scope": "frozen evidence and selected candidate figure exports; no new tests, fitting, or samples",
                "build_provenance": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve()),
                                     "analysis/figure_atlas_20260831/common.py": sha256(Path(__file__).resolve().parent / "common.py")},
                "source_hashes": {str(path.relative_to(ROOT)): HASHES[key] for key, path in SOURCES.items()},
                "outputs": {path: sha256(ROOT/path) for path in FILES},
                "checks": {"joint_mean_edges": 102, "joint_mean_cells": 492, "resolved_strong_lag_edges": 0,
                           "distributional_followups": 6, "primary_followups_in_inventory": 6, "calibration_rows": 24,
                           "distributional_exact_cell_control_matches": 0,
                           "plotted_matched_control_rows": 6,
                           "plotted_positive_sampling_excess_gate_passes": 3,
                           "plotted_positive_quiet_excess_gate_passes": 0,
                           "mean_profile_reconstruction": "matched saved cell evidence within 1e-6"},
                "validation_rating": "share with caveats: conditional fitted-model evidence; selection and support restrictions retained",
                "visual_qa": {"status": "manual inspection required after regeneration"},
                "caption_qa_corrections": {
                    "date": "2026-08-31",
                    "changes": ["Baseline uses four lags by six horizons; active-minus-baseline uses four by five, excluding H32 for offset crossing",
                                "Worm is the inference/resampling unit; overlapping cross-validation training can couple fitted-model estimates",
                                "F08b includes all three outcomes for both FLP-to-ADE and RIP-to-URB; RIP-to-URB W1 passes positive sampler-excess but not quiet-excess, and does not establish mean-independent shape or gain modulation"]}}
    (OUT / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": "complete", "files": FILES, "checks": manifest["checks"]}, indent=2))


if __name__ == "__main__":
    main()
