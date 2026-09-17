"""Standalone data/method/observed-response figures; no new model inference."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import ROOT, OUT, BLUE, ORANGE, INK, GREY, LIGHT, style, sha256, save_figure
from conditional_neural_benchmark.data import load_cohort

ATLAS = ROOT / "results/neural_prediction_atlas_20260829"
CLASSES = ROOT / "results/neuron_class_effects_20260831"
OBSERVED = ROOT / "results/biological_lag_analysis_20260828/chemical_corrected_observed_20260828"
CHEMICAL_COLORS = {"butanone": BLUE, "pentanedione": "#A68638", "nacl": "#776287"}
CHEMICAL_LABELS = {"butanone": "Butanone", "pentanedione": "Pentanedione", "nacl": "NaCl"}


def box(ax, xy, width, height, title, body, color=INK):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.01",
                              facecolor="white", edgecolor="#BAC3CA", linewidth=1))
    ax.text(x + .025, y + height - .075, title, color=color, weight="semibold", fontsize=12, va="top")
    ax.text(x + .025, y + height - .17, body, fontsize=11, va="top", linespacing=1.5)


def data_overview(cohort):
    current = pd.read_csv(CLASSES / "classification/neuron_classification.csv")
    historical = pd.read_csv(CLASSES / "classification/historical80_classification.csv")
    with np.load(ATLAS / "canonical/atlas_matrices.npz", allow_pickle=False) as a:
        assert tuple(a["neurons"]) == cohort.neurons
        assert tuple(a["worm_ids"]) == cohort.worm_ids
    assert cohort.n_worms == 17 and cohort.n_neurons == 54 and cohort.fps == 4
    schedules = pd.DataFrame([{
        "worm_id": s.worm_id, "event_position": e + 1, "chemical": chem,
        "chemical_code": code, "onset_seconds": interval[0], "offset_seconds": interval[1],
    } for s in cohort.stimulus_schedules for e, (chem, code, interval) in enumerate(zip(
        s.chemical_name_by_event, s.chemical_code_by_event, s.event_intervals_seconds))])
    schedules.to_csv(OUT / "data/f01_stimulus_schedules.csv", index=False)
    rows = [{"roster": "Current 54", "head": 54, "tail": 0},
            {"roster": "Historical 80", "head": int(historical.recording_region.eq("head").sum()),
             "tail": int(historical.recording_region.eq("tail").sum())}]
    pd.DataFrame(rows).to_csv(OUT / "data/f01_roster_counts.csv", index=False)
    assert rows[1]["head"] == 63 and rows[1]["tail"] == 17
    wi = min(range(cohort.n_worms), key=lambda i: cohort.worm_ids[i])
    trace = cohort.traces[wi].astype(float)
    trace_z = (trace - np.nanmean(trace, axis=0)) / np.maximum(np.nanstd(trace, axis=0), 1e-8)
    saved = pd.DataFrame(trace_z, columns=cohort.neurons)
    saved.insert(0, "time_seconds", np.arange(len(trace)) / cohort.fps)
    saved.to_csv(OUT / "data/f01_example_recording_display_z.csv", index=False)
    fig = plt.figure(figsize=(13.8, 9.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.55], hspace=.6, wspace=.34)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A  Recording coverage", loc="left", pad=16)
    ax.barh([1, 0], [54, 63], color=BLUE, label="Head classes", height=.45)
    ax.barh([1, 0], [0, 17], left=[54, 63], color=ORANGE, label="Tail classes", height=.45)
    ax.text(27, 1, "54", color="white", ha="center", va="center", fontsize=12)
    ax.text(31.5, 0, "63", color="white", ha="center", va="center", fontsize=12)
    ax.text(71.5, 0, "17", color="white", ha="center", va="center", fontsize=12)
    ax.set_yticks([1, 0], ["Current atlas", "Historical SBTG"])
    ax.set_xlim(0, 85); ax.set_xlabel("Pooled neuron classes (not individual cells)")
    ax.set_ylim(-.65, 1.75); ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.02))
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B  Recorded stimulus order", loc="left", pad=16)
    order_counts = pd.Series([s.chemical_code_by_event for s in cohort.stimulus_schedules]).value_counts().sort_index()
    order_array = np.asarray(order_counts.index.tolist())
    cmap = ListedColormap([CHEMICAL_COLORS[c] for c in ["butanone", "pentanedione", "nacl"]])
    ax.imshow(order_array, cmap=cmap, vmin=1, vmax=3, aspect="auto")
    for ri in range(3):
        for ci in range(3):
            code = order_array[ri, ci]
            ax.text(ci, ri, ["Butanone", "Pentanedione", "NaCl"][code - 1], color="white", ha="center", va="center", fontsize=10)
    ax.set_xticks([0, 1, 2], ["First event", "Second event", "Third event"])
    ax.set_yticks([0, 1, 2], [f"{n} worms" for n in order_counts.values])
    ax.tick_params(length=0)
    ax = fig.add_subplot(gs[1, :])
    cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("#DDE2E6")
    im = ax.imshow(trace_z.T, aspect="auto", origin="lower", cmap=cm, vmin=-3, vmax=3,
                   extent=[0, len(trace) / cohort.fps, .5, cohort.n_neurons + .5], interpolation="nearest")
    s = cohort.stimulus_schedules[wi]
    for (start, end), chem in zip(s.event_intervals_seconds, s.chemical_name_by_event):
        ax.axvline(start, color="black", lw=.8, alpha=.65)
        ax.plot([start, end], [57, 57], color=CHEMICAL_COLORS[chem], lw=7, clip_on=False, solid_capstyle="butt")
        ax.text((start + end) / 2, 59.5, CHEMICAL_LABELS[chem], fontsize=10, ha="center", clip_on=False)
    ax.set_title(f"C  Example head recording: {cohort.worm_ids[wi]} (first ID; not selected for response)",
                 loc="left", pad=46)
    ax.set_yticks([1, 10, 20, 30, 40, 54]); ax.set_ylabel("Pooled class coordinate\n(canonical atlas order)")
    ax.set_xlabel("Time from recording start (seconds)")
    bar = fig.colorbar(im, ax=ax, pad=.018, fraction=.025, extend="both")
    bar.set_label("Within-class display z-score")
    fig.suptitle("NeuroPAL data and stimulus provenance", x=.075, y=1.005, ha="left", fontsize=17)
    fig.text(.075, -.045, "Current: 17 head recordings at 4 Hz. Historical: head/tail index pairing and donor filling, not simultaneous whole-body data.\n"
             "Each chemical occurs once per worm. The model uses binary any-stimulus history; true chemicals remain event metadata.\n"
             "Display normalization only; color clips at ±3 SD but the CSV is unclipped. Missing boundary frames remain missing (gray).", fontsize=10)
    return save_figure(fig, "f01_data_and_stimuli")


def method_figure():
    fig = plt.figure(figsize=(13.6, 10.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.12, 1.15, .95], hspace=.64, wspace=.38)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("A  Learn a conditional distribution, then compare two repaired laws", loc="left", pad=14)
    box(ax, (.005, .08), .275, .80, "1. Fit on training worms", "20 s of neural history\n+ binary stimulus history\n→ next-state distribution")
    box(ax, (.35, .08), .285, .80, "2. Sample the frozen model", "Low / high repaired histories\nProgressive bridge SMC\nCheck weights, ancestry, support", BLUE)
    box(ax, (.70, .08), .285, .80, "3. Measure the effect", "Mean · spread · event probability\nWhole-distribution distance\nRepeat over sources and time")
    for a, b in [(.282, .347), (.638, .697)]:
        ax.annotate("", (b, .47), (a, .47), arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.5})
    ax = fig.add_subplot(gs[1, :])
    ax.set_title("B  Source lag and forecast horizon are different time axes", loc="left", pad=32)
    ax.set_xlim(-4.15, 1.8); ax.set_ylim(-1, 1.75)
    ax.axhline(0, color=GREY, lw=1)
    frames = np.arange(-12, 5)
    times = frames / 4
    ax.scatter(times, np.zeros(len(times)), color="#A6AFB7", s=20, zorder=2)
    source = np.arange(-11, -7)
    ax.scatter(source / 4, np.zeros(4), color=BLUE, s=75, zorder=3)
    ax.axvline(0, color=INK, lw=1.5)
    ax.scatter([1], [0], color=ORANGE, s=90, zorder=4)
    ax.text(0, 1.52, "Prediction cut", ha="center", weight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=5)
    ax.text(-2.375, -.38, "Four-frame source summary\n(1-second window)", ha="center", color=BLUE, va="top")
    ax.annotate("", (0, .55), (-2, .55), arrowprops={"arrowstyle": "<->", "color": BLUE})
    ax.text(-1, .67, "Source lag = 2 s", ha="center", color=BLUE)
    ax.annotate("", (1, 1.02), (0, 1.02), arrowprops={"arrowstyle": "<->", "color": ORANGE})
    ax.text(.5, 1.15, "Horizon = 1 s", ha="center", color=ORANGE)
    ax.text(-3.16, .3, "Observed history\nbefore repair", ha="right", fontsize=10)
    ax.annotate("", (-3, -.09), (-4, -.09), arrowprops={"arrowstyle": "->", "color": GREY})
    ax.set_xticks([-4, -3, -2, -1, 0, 1]); ax.set_xlabel("Seconds relative to prediction cut")
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    example = pd.DataFrame({"frame_relative_to_cut": frames, "seconds_relative_to_cut": times,
                            "source_window": np.isin(frames, source), "target_readout": frames == 4})
    example.to_csv(OUT / "data/f02_timing_example.csv", index=False)
    x = np.linspace(-3, 3, 401)
    normal = lambda mu, sd: np.exp(-.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    illustrations = []
    for i, (title, low, high, xlabel) in enumerate([
        ("C  Mean shift", normal(-.4, .75), normal(.4, .75), "Illustrative target activity"),
        ("D  Spread change", normal(0, .55), normal(0, 1), "Illustrative target activity"),
    ]):
        ax = fig.add_subplot(gs[2, i])
        ax.plot(x, low, color=BLUE, label="Low-source law")
        ax.plot(x, high, color=ORANGE, ls="--", label="High-source law")
        ax.set_title(title, loc="left"); ax.set_xlabel(xlabel); ax.set_ylabel("Density"); ax.set_yticks([])
        if i == 0: ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(-.05, -.30))
        illustrations.append(pd.DataFrame({"illustration": title, "x": x, "low": low, "high": high}))
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    ax.set_title("E  Interpreting the differences", loc="left")
    ax.text(0, .91, "Mean: signed activity shift\n\nSD / log-SD: response spread\n(not proof of gain)\n\nW1: any distributional change\n(unsigned; needs sampling controls)", va="top", fontsize=11)
    pd.concat(illustrations, ignore_index=True).to_csv(OUT / "data/f02_illustrative_densities.csv", index=False)
    fig.suptitle("From conditional dynamics to lagged response matrices", x=.125, y=1.01, ha="left", fontsize=17)
    fig.text(.125, -.045, "Density curves are schematic illustrations, not model outputs. The timing example matches the 4-Hz frame convention.\n"
             "Effects are observational, model-relative high-minus-low contrasts; neither physical interventions nor identified delays.", fontsize=10)
    return save_figure(fig, "f02_sampling_and_effect_definitions")


def observed_onsets():
    source = pd.read_csv(OBSERVED / "neuron_response_statistics.csv")
    panel = source[source.cohort.eq("oh16230_head") & source.neuron.eq("AWC")].copy()
    assert len(panel) == 18 and panel.n_worms.eq(17).all()
    panel.to_csv(OUT / "data/f09_awc_observed_onset_statistics.csv", index=False)
    worms = pd.read_csv(OBSERVED / "worm_event_neuron_responses.csv")
    worms = worms[worms.cohort.eq("oh16230_head") & worms.neuron.eq("AWC")].copy()
    worms.drop(columns=["source_recording"]).to_csv(OUT / "data/f09_awc_observed_worm_responses.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5), sharex=True, sharey=True)
    for ax, chem in zip(axes, ["butanone", "pentanedione", "nacl"]):
        p = panel[panel.chemical.eq(chem)].sort_values("horizon_seconds")
        x = p.horizon_seconds.to_numpy(); y = p.mean_onset_minus_quiet_z.to_numpy()
        ax.plot(x, y, color=CHEMICAL_COLORS[chem], lw=1.5)
        ax.errorbar(x, y, yerr=[y-p.ci95_low, p.ci95_high-y], fmt="none", color=CHEMICAL_COLORS[chem], capsize=3)
        for row in p.itertuples():
            ax.scatter(row.horizon_seconds, row.mean_onset_minus_quiet_z, s=55,
                       edgecolor=CHEMICAL_COLORS[chem], facecolor=CHEMICAL_COLORS[chem] if row.bh_q_value < .05 else "white", zorder=4)
        ax.axhline(0, color=GREY, lw=1, ls="--"); ax.set_xscale("log", base=2)
        ax.set_xticks([.25, .5, 1, 2, 4, 10], ["0.25", "0.5", "1", "2", "4", "10"])
        ax.set_title(CHEMICAL_LABELS[chem], loc="left"); ax.set_xlabel("Post-onset averaging window (s)")
        ax.set_ylim(-2.45, .8); ax.grid(axis="y")
        selected = p[p.horizon_seconds.isin([4, 10])]
        text = " · ".join(f"{r.horizon_seconds:g} s: q={r.bh_q_value:.3g}" for r in selected.itertuples())
        ax.text(.02, .97, text, transform=ax.transAxes, fontsize=10, va="top")
    axes[0].set_ylabel("Observed onset-minus-quiet activity change\n(within-worm non-stimulus SD units)")
    fig.suptitle("Observed AWC stimulus responses—not inferred inter-neuron effects", x=.125, y=1.035, ha="left", fontsize=16)
    fig.text(.125, -.10, "Points: means across 17 worms; bars: pointwise 95% worm-bootstrap intervals. Filled markers: existing BH q<0.05\n"
             "within the 54-neuron family for that chemical and averaging window—not correction across all chemicals/windows.\n"
             "AWC was an existing named case, not selected here by largest response. These tests do not compare chemicals with one another.", fontsize=10)
    return save_figure(fig, "f09_observed_onset_vs_quiet")


def class_summary():
    frame = pd.read_parquet(CLASSES / "analysis/class_effects.parquet")
    common = frame[frame.method.eq("progressive_bridge_smc") & frame.roster.eq("cook_primary") &
                   frame.channel.eq("endpoint_mean") & frame.context.eq("baseline") &
                   frame.source_lag_frames.eq(1) & frame.horizon_frames.eq(1)].copy()
    common.to_csv(OUT / "data/f10_class_descriptive_blocks.csv", index=False)
    order = ["sensory", "interneuron", "motor"]
    labels = ["Sensory", "Interneuron", "Motor"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    for ax, policy, title in zip(axes, ["all_estimated", "strong_common_lags"],
                                  ["All estimated sources", "Sources supported at every lag"]):
        p = common[common.support_policy.eq(policy)]
        matrix = p.pivot(index="source_class", columns="target_class", values="mean_absolute_pooled_edge").loc[order, order]
        counts = p.pivot(index="source_class", columns="target_class", values="n_pairs").loc[order, order]
        ns = p.groupby("source_class").n_sources.first().reindex(order)
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=.085)
        for i in range(3):
            for j in range(3):
                color = "white" if matrix.iloc[i, j] > .05 else INK
                ax.text(j, i-.09, f"{matrix.iloc[i,j]:.4f}", ha="center", va="center", fontsize=12, color=color)
                ax.text(j, i+.18, f"{counts.iloc[i,j]} pairs", ha="center", va="center", fontsize=10, color=color)
        ax.set_xticks(range(3), labels); ax.set_yticks(range(3), [f"{name}\n{n} sources" for name, n in zip(labels, ns)])
        ax.set_xlabel("Target class"); ax.set_ylabel("Source class"); ax.set_title(title, loc="left", pad=14)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=.7, label="Mean absolute pooled-edge effect\n(source-gap normalized units)", pad=.04)
    fig.suptitle("Class-level estimates: descriptive, with no between-class significance test", fontsize=15)
    fig.supxlabel("Baseline · source lag 0.25 s · horizon 0.25 s · progressive bridge SMC\n"
                  "Strict support retains only SIA and SMB as motor sources. Color is magnitude, not statistical evidence.", fontsize=10)
    return save_figure(fig, "f10_class_estimates_and_support")


def main():
    style(); (OUT / "data").mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(cohort_mode="oh16230_head")
    files = data_overview(cohort) + method_figure() + observed_onsets() + class_summary()
    sources = [ROOT / "SBTG/data/Head_Activity_OH16230.mat", ROOT / "SBTG/data/Head_Activity_OH15500.mat",
               ROOT / "conditional_neural_benchmark/data.py", ROOT / "sid_elegans/combined_data.py",
               ATLAS / "canonical/atlas_matrices.npz", ATLAS / "canonical/protocol.json",
               CLASSES / "classification/neuron_classification.csv", CLASSES / "classification/historical80_classification.csv",
               CLASSES / "analysis/class_effects.parquet", OBSERVED / "neuron_response_statistics.csv",
               OBSERVED / "worm_event_neuron_responses.csv", ROOT / "compatibility_neural_benchmark/chemical_observed_analysis.py",
               ROOT / "compatibility_neural_benchmark/prediction_atlas_runner.py", Path(__file__), Path(__file__).with_name("common.py")]
    metadata = {"status": "complete", "files": files,
                "cohort_fingerprint": cohort.stimulus_schema_fingerprint,
                "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                "new_inferential_tests": False, "new_models_or_paths": False,
                "observed_case_selection": "AWC: previously named biological case; all three chemicals and all six saved windows",
                "observed_multiplicity": "Existing BH across 54 neurons within each cohort/chemical/window, not global across windows",
                "class_inference": "descriptive; no between-class significance claims or intervals for plotted absolute pooled-edge values"}
    (OUT / "data_method_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
