"""Clarity-first prediction views from sealed results; no new model inference."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (ROOT, OUT, OLD, BLUE, GREY, INK, new_figure, finish,
                    verify_old_inputs, write_manifest, sha256, relative)

ATLAS = ROOT / "results/neural_prediction_atlas_20260829"
OLD_PATHS = {
    "controls": OLD / "data/f08b_plotted_sampler_control_evidence.csv",
    "all_controls": OLD / "data/f08_all_sampler_control_evidence.csv",
    "lag_profiles": OLD / "data/f07_plotted_lag_profiles.csv",
    "screen": OLD / "data/f08_distributional_screen_inventory.csv",
    "captions": OLD / "PREDICTION_CAPTIONS.md",
    "old_code": ROOT / "analysis/figure_atlas_20260831/prediction_figures.py",
}
NATIVE_PATHS = {
    "atlas": ATLAS / "canonical/atlas_matrices.npz",
    "atlas_protocol": ATLAS / "canonical/protocol.json",
    "targeted_cells": ATLAS / "targeted_confirmation_distributional/analysis/targeted_cells.csv",
    "targeted_protocol": ATLAS / "targeted_confirmation_distributional/analysis/protocol.json",
}
EXTRA_OUTPUTS = []


def verify_native(paths):
    verified = {}
    for path in paths:
        ledger = path.parent / "checksums.sha256"
        entries = {}
        for line in ledger.read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            entries[(ledger.parent / name.lstrip("*")).resolve()] = expected
        actual = sha256(path)
        assert entries[path.resolve()] == actual, f"Native checksum mismatch: {path}"
        verified[relative(path)] = actual
        verified[relative(ledger)] = sha256(ledger)
    return verified


def export(frame, stem, sources, illustrated=False):
    frame = frame.copy()
    frame["source_kind"] = "explicit illustration, not empirical results" if illustrated else "frozen empirical fitted-model summary"
    frame["source_files"] = ";".join(relative(path) for path in sources)
    frame["source_sha256"] = ";".join(sha256(path) for path in sources)
    path = OUT / "data" / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    EXTRA_OUTPUTS.append(path)
    caption = OUT / "captions" / f"{stem}.md"
    assert caption.exists(), f"Technical caption missing: {caption}"
    EXTRA_OUTPUTS.append(caption)


def sampler_case(controls, metric):
    selected = controls[(controls.source_neuron == "RIP") & (controls.target_neuron == "URB") &
                        (controls.context == "baseline") & (controls.source_lag_frames == 4) &
                        (controls.horizon_frames == 2) & (controls.metric == metric)].copy()
    assert len(selected) == 1
    row = selected.iloc[0]
    assert row.support_pass and row.selection_eligible and row.positive_sampling_excess_gate_pass
    assert not row.positive_quiet_excess_gate_pass and row.temporal_specificity_joint_max_t_p == 1
    is_mean = metric == "endpoint_mean"
    stem = "prediction_mean_sampler_excess" if is_mean else "prediction_w1_sampler_excess"
    question = "Does the mean response exceed sampling noise?" if is_mean else "Does the distributional change exceed sampling noise?"
    fig, ax = new_figure(question, "RIP → URB · selected baseline case · 128 particles\nSource window ends 1 s before cut; target readout 0.5 s after cut")
    plt.rcParams["axes.titleweight"] = "bold"
    estimate = row.sampling_excess_mean
    low, high = row.sampling_excess_ci_2_5, row.sampling_excess_ci_97_5
    ax.axvline(0, color=GREY, lw=1.7, ls="--", ymax=.82)
    ax.errorbar(estimate, 0, xerr=[[estimate-low], [high-estimate]], fmt="o", color=BLUE, markersize=11,
                capsize=8, elinewidth=3, capthick=2)
    ax.set_xlim((-.02, .17) if is_mean else (-.01, .105))
    ax.set_ylim(-.48, .62)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=.5)
    ax.set_xticks([0, .05, .10, .15] if is_mean else [0, .025, .05, .075, .10])
    ax.set_xlabel(("Mean-response magnitude" if is_mean else "W1 distance") + " beyond sampler control\n(source-gap normalized)", labelpad=14)
    ax.text(estimate, .105, f"{estimate:.3f}", ha="center", va="bottom", fontsize=22, color=BLUE)
    numerator = int(round(row.sampling_excess_joint_max_t_p * 65536))
    assert np.isclose(row.sampling_excess_joint_max_t_p, numerator/65536, atol=1e-15)
    ax.text(.98, .92, f"Corrected p = {row.sampling_excess_joint_max_t_p:.6f}", transform=ax.transAxes, ha="right", fontsize=17)
    ax.text(0, .55, "No excess", ha="center", va="top", fontsize=14, color=GREY)
    if is_mean:
        lines = ["Estimate and saved 95% worm interval; the positive sampler-excess test passes.",
                 "No evidence of excess over quiet-time responses (corrected p = 1); not causal evidence."]
    else:
        lines = ["W1 can include a mean shift; it does not establish mean-independent shape change.",
                 "Saved 95% worm interval. No evidence of excess over quiet-time responses (p = 1); no gain claim."]
    plotted = selected.copy()
    plotted["plotted_estimate"] = estimate
    plotted["plotted_ci_low"] = low
    plotted["plotted_ci_high"] = high
    plotted["exact_max_t_numerator"] = numerator
    plotted["exact_max_t_denominator"] = 65536
    export(plotted, stem, [OLD_PATHS["controls"], OLD_PATHS["captions"]])
    result = finish(fig, stem, lines)
    result["question"] = question
    result["scientific_scope"] = "selected fitted-model sampling-control contrast, not independent biological confirmation"
    return result


def lag_profile(profiles):
    selected = profiles[(profiles.source_neuron == "RIP") & (profiles.target_neuron == "URB") &
                        (profiles.family_id == "baseline_endpoint_mean") & (profiles.horizon_frames == 2)].sort_values("source_lag_frames").copy()
    assert selected.source_lag_frames.tolist() == [1, 4, 8, 16]
    assert selected.joint_primary_flat_lag_max_t_p_value.eq(1).all()
    with np.load(NATIVE_PATHS["atlas"]) as archive:
        ns = archive["neurons"].astype(str).tolist()
        h = archive["horizon_frames"].tolist().index(2)
        index = (slice(None), h, ns.index("URB"), ns.index("RIP"))
        key = "__progressive_bridge_smc__endpoint_mean__baseline"
        for field, prefix in [("mean_normalized", "mean_normalized"), ("archived_ci_low", "ci_low_normalized"), ("archived_ci_high", "ci_high_normalized")]:
            assert np.allclose(selected[field], archive[prefix+key][index], atol=1e-7)
    selected["display_source_lag_seconds"] = selected.source_lag_frames / 4
    fig, ax = new_figure("Does RIP → URB have a resolved preferred lag?", "Selected baseline profile from the original 32-particle screen · target readout 0.5 s after cut")
    x, y = selected.display_source_lag_seconds.to_numpy(), selected.mean_normalized.to_numpy()
    low, high = selected.archived_ci_low.to_numpy(), selected.archived_ci_high.to_numpy()
    ax.axhline(0, color=GREY, lw=1.5)
    ax.errorbar(x, y, yerr=[y-low, high-y], fmt="o", color=BLUE, ms=9, capsize=6, elinewidth=2.4)
    ax.plot(x, y, color=BLUE, lw=1.6, alpha=.5)
    ax.set_xlim(0, 4.3)
    ax.set_ylim(-.015, .33)
    ax.set_xticks(x, ["0.25", "1", "2", "4"])
    ax.set_yticks([0, .1, .2, .3])
    ax.set_xlabel("Source window ends this many seconds before the cut", labelpad=14)
    ax.set_ylabel("Mean response\n(source-gap normalized)")
    ax.grid(axis="y", alpha=.5)
    ax.text(.98, .94, "No resolved lag difference\nCorrected p = 1.000", ha="right", va="top", transform=ax.transAxes, fontsize=17)
    stem = "prediction_mean_lag_profile"
    export(selected, stem, [OLD_PATHS["lag_profiles"], NATIVE_PATHS["atlas"], NATIVE_PATHS["atlas_protocol"]])
    result = finish(fig, stem, ["Bars are pointwise 95% worm intervals, not simultaneous comparisons between lags.",
                                "Selected forecast horizon; the fitted model does not establish a physical delay."])
    result["question"] = "Is the selected mean-response profile detectably non-flat across source lags?"
    return result


def spread_case(screen):
    saved = screen[(screen.source_neuron == "ASH") & (screen.target_neuron == "OLQ") &
                   (screen.context == "baseline") & (screen.source_lag_frames == 4) &
                   (screen.horizon_frames == 32) & (screen.targeted_metric == "endpoint_sd")]
    assert len(saved) == 1
    saved = saved.iloc[0]
    targeted = pd.read_csv(NATIVE_PATHS["targeted_cells"])
    targeted = targeted[(targeted.source_neuron == "ASH") & (targeted.target_neuron == "OLQ") &
                        (targeted.context == "baseline") & (targeted.source_lag_frames == 4) &
                        (targeted.horizon_frames == 32) & (targeted.metric == "endpoint_sd") &
                        (targeted.contrast == "high_low")]
    assert len(targeted) == 1
    row = targeted.iloc[0]
    assert row.n_worms == 17 and row.n_particles == 128
    assert np.isclose(row.mean_normalized, saved.targeted_n128_mean_normalized, atol=1e-12)
    with np.load(NATIVE_PATHS["atlas"]) as archive:
        ns = archive["neurons"].astype(str).tolist()
        index = (archive["source_lag_frames"].tolist().index(4), archive["horizon_frames"].tolist().index(32), ns.index("OLQ"), ns.index("ASH"))
        key = "__progressive_bridge_smc__endpoint_sd__baseline"
        screen_values = [float(archive[p+key][index]) for p in ["mean_normalized", "ci_low_normalized", "ci_high_normalized"]]
    assert np.isclose(screen_values[0], saved.screen_mean_normalized, atol=1e-7)
    all_controls = pd.read_csv(OLD_PATHS["all_controls"])
    matching = all_controls[(all_controls.source_neuron == "ASH") & (all_controls.target_neuron == "OLQ") &
                            (all_controls.context == "baseline") & (all_controls.source_lag_frames == 4) &
                            (all_controls.horizon_frames == 32) & (all_controls.metric == "endpoint_sd")]
    assert len(matching) == 0
    rows = pd.DataFrame([
        {"stage": "Screen", "n_particles": 32, "estimate": screen_values[0], "ci_low": screen_values[1], "ci_high": screen_values[2], "bootstrap_replicates": 256},
        {"stage": "Rerun", "n_particles": 128, "estimate": row.mean_normalized, "ci_low": row.ci_2_5, "ci_high": row.ci_97_5, "bootstrap_replicates": 2000},
    ])
    for field, value in {"source_neuron":"ASH", "target_neuron":"OLQ", "context":"baseline", "source_lag_frames":4, "horizon_frames":32,
                         "source_window_end_before_cut_seconds":1, "forecast_endpoint_after_cut_seconds":8, "metric":"endpoint_sd", "contrast":"high_low",
                         "n_worms":17, "interval_scope":"saved pointwise percentile worm bootstrap, conditional fitted model and selection", "exact_matched_controls_available":False,
                         "claim_status":"exploratory; no full-family SD inference or between-run significance test"}.items():
        rows[field] = value
    fig, ax = new_figure("ASH → OLQ: a possible change in spread", "Selected baseline case · source window ends 1 s before cut · target readout 8 s after cut")
    for index, record in rows.iterrows():
        y = 1-index
        color = GREY if index == 0 else BLUE
        ax.errorbar(record.estimate, y, xerr=[[record.estimate-record.ci_low], [record.ci_high-record.estimate]],
                    fmt="o" if index == 0 else "D", color=color, mfc="white" if index == 0 else BLUE,
                    ms=10, capsize=7, elinewidth=2.5)
        ax.text(record.estimate, y+.16, f"{record.estimate:.3f}", ha="center", color=color, fontsize=19)
    ax.axvline(0, color=GREY, lw=1.5, ls="--", ymax=.85)
    ax.text(0, 1.46, "No spread change", ha="center", color=GREY, fontsize=14)
    ax.set_xlim(-.11, .035)
    ax.set_ylim(-.45, 1.6)
    ax.set_yticks([1, 0], ["Screen\n32 particles", "Rerun\n128 particles"])
    ax.set_xticks([-.10, -.05, 0])
    ax.set_xlabel("High-source minus low-source target SD\n(source-gap normalized)", labelpad=14)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.grid(axis="x", alpha=.5)
    stem = "prediction_spread_consistency"
    export(rows, stem, [OLD_PATHS["screen"], OLD_PATHS["all_controls"], *NATIVE_PATHS.values()])
    result = finish(fig, stem, ["Both selected estimates suggest lower spread in the higher-source arm; intervals are pointwise 95%.",
                                "No exact matched sampler control for this case; exploratory, not a gain-modulation result."])
    result["question"] = "Does the selected negative spread estimate persist with more particles?"
    return result


def illustration():
    fig, unused = new_figure("Mean and spread describe different changes", "Illustration only — these curves are not fitted or sampled neural results")
    unused.remove()
    axes = [fig.add_axes([.10, .31, .36, .42]), fig.add_axes([.59, .31, .36, .42])]
    x = np.linspace(-4.5, 4.5, 501)
    parameters = [("Mean changes", "Reference", -1.2, .9, GREY, "--"),
                  ("Mean changes", "Changed", 1.2, .9, BLUE, "-"),
                  ("Spread changes", "Reference", 0, .9, GREY, "--"),
                  ("Spread changes", "Changed", 0, 1.65, BLUE, "-")]
    records = []
    for panel_index, (name, ax) in enumerate(zip(["Mean changes", "Spread changes"], axes)):
        for panel, role, mean, sd, color, linestyle in parameters:
            if panel != name:
                continue
            density = np.exp(-.5*((x-mean)/sd)**2)/(sd*np.sqrt(2*np.pi))
            ax.plot(x, density, color=color, ls=linestyle, lw=2.8)
            records.extend({"panel":panel, "role":role, "response_arbitrary_units":float(xx), "illustrative_density":float(yy),
                            "illustrative_mean":mean, "illustrative_sd":sd, "empirical":False} for xx, yy in zip(x, density))
        ax.set_title(name, fontsize=19, fontweight="bold", pad=15)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(0, .59)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([0, .2, .4])
        ax.set_xlabel("Response (arbitrary units)", fontsize=14, labelpad=12)
        ax.set_ylabel("Illustrative density", fontsize=14)
        if panel_index == 0:
            ax.text(-1.35, .51, "Reference", color=GREY, ha="center", fontsize=14)
            ax.text(1.35, .51, "Changed", color=BLUE, ha="center", fontsize=14)
        else:
            ax.text(-.5, .51, "Reference", color=GREY, ha="center", fontsize=14)
            target_y = np.exp(-.5*(2/1.65)**2)/(1.65*np.sqrt(2*np.pi))
            ax.annotate("Changed", xy=(2, target_y), xytext=(1.25, .30), color=BLUE, fontsize=14,
                        arrowprops={"arrowstyle":"-", "color":BLUE, "lw":1.2})
    stem = "prediction_mean_versus_spread_illustration"
    export(pd.DataFrame(records), stem, [Path(__file__).resolve()], illustrated=True)
    result = finish(fig, stem, ["Left: the center moves without changing spread. Right: spread changes without moving the center.",
                                "Teaching sketch only. A W1 distance can respond to either change; neither alone identifies gain."])
    result["question"] = "What is the difference between a mean change and a spread change?"
    result["illustrative_not_empirical"] = True
    result["parameters"] = [{"panel":p, "role":r, "mean":m, "sd":s} for p,r,m,s,_,_ in parameters]
    return result


def main():
    inputs = verify_old_inputs(OLD_PATHS.values())
    inputs.update(verify_native(NATIVE_PATHS.values()))
    controls = pd.read_csv(OLD_PATHS["controls"])
    profiles = pd.read_csv(OLD_PATHS["lag_profiles"])
    screen = pd.read_csv(OLD_PATHS["screen"])
    figures = [sampler_case(controls, "endpoint_mean"), lag_profile(profiles),
               sampler_case(controls, "endpoint_wasserstein1"), spread_case(screen), illustration()]
    manifest = write_manifest("prediction", inputs, figures,
        {"output_files":EXTRA_OUTPUTS, "generator":{"path":relative(__file__), "sha256":sha256(__file__)},
         "scope":"five clarity-first views: four sealed-evidence views and one explicitly illustrative teaching chart",
         "checks":{"empirical_figures":4, "illustrative_figures":1, "rip_controls_exact_match":True,
                   "rip_lag_profile_matches_canonical":True, "ash_screen_and_n128_means_match_previous_exports":True,
                   "ash_cis_exact_saved_worm_bootstraps":True, "ash_exact_sampler_control_matches":0,
                   "old_and_native_ledgers_verified":True},
         "limitations":["selected fitted-model effects, not independent biological validation", "overlapping cross-validation training can couple held-out-worm estimates",
                        "pointwise intervals are not simultaneous lag comparisons", "W1 does not isolate mean-independent shape or gain modulation",
                        "ASH-to-OLQ SD remains exploratory despite selected interval consistency"]})
    print(json.dumps({"manifest":relative(manifest), "figures":[fig["stem"] for fig in figures]}, indent=2))


if __name__ == "__main__":
    main()
