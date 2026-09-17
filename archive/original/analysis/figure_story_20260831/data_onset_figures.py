"""Simple observed recordings and onset figures from frozen saved exports only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from common import (ROOT, OUT, OLD, BLUE, ORANGE, INK, GREY, LIGHT,
                    new_figure, finish, verify_old_inputs, write_manifest, sha256, relative)

DATA = OLD / "data"
CHEMICALS = {"butanone": "Butanone", "pentanedione": "Pentanedione", "nacl": "NaCl"}
EXPORTS = []


def save_csv(frame, stem):
    path = OUT / "data" / f"{stem}.csv"
    frame.to_csv(path, index=False)
    EXPORTS.append(path)
    return path


def caption(stem, text):
    path = OUT / "captions" / f"{stem}.md"
    path.write_text(text.strip() + "\n")
    EXPORTS.append(path)


def recordings():
    stem = "data_recording"
    original = pd.read_csv(DATA / "f01_example_recording_display_z.csv")
    schedule = pd.read_csv(DATA / "f01_stimulus_schedules.csv")
    first_id = sorted(schedule.worm_id.unique())[0]
    events = schedule[schedule.worm_id.eq(first_id)].sort_values("event_position")
    assert first_id == "OH16230:0924_01" and len(events) == 3
    channels = ["AWC", "RIP", "URB"]
    trace = original[["time_seconds", *channels]].copy()
    assert len(trace) == 960 and np.allclose(np.diff(trace.time_seconds), .25)
    assert trace[channels].iloc[:4].isna().all().all()
    fig, unused = new_figure("NeuroPAL recordings and stimulus timing",
                            "17 worms · 54 pooled head-neuron classes · 4 samples per second")
    unused.remove()
    fig.text(.045, .806, "Observed activity (within-recording SD units)", fontsize=13, color=GREY)
    for name, bottom in zip(channels, [.65, .48, .31]):
        ax = fig.add_axes([.17, bottom, .78, .12])
        y = trace[name].to_numpy()
        ax.plot(trace.time_seconds, y, color=BLUE, linewidth=1.25)
        for event in events.itertuples():
            ax.axvspan(event.onset_seconds, event.offset_seconds, color=LIGHT, zorder=0)
        ax.axhline(0, color=GREY, linewidth=.65, zorder=0)
        ax.set_xlim(0, 240)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=23, fontsize=16)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", labelsize=12, length=3)
        ax.spines["bottom"].set_visible(False)
        ax.set_yticks([0, 4] if name != "RIP" else [0, 10])
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        ax.set_ylim(ymin - .12 * (ymax - ymin), ymax + .12 * (ymax - ymin))
    stim = fig.add_axes([.17, .226, .78, .028])
    stim.set_xlim(0, 240)
    stim.set_ylim(-.1, 1.1)
    binary = np.zeros(len(trace), dtype=int)
    for event in events.itertuples():
        on = (trace.time_seconds >= event.onset_seconds) & (trace.time_seconds < event.offset_seconds)
        binary[on] = 1
        stim.text((event.onset_seconds + event.offset_seconds) / 2, 1.55,
                  CHEMICALS[event.chemical], ha="center", va="bottom", fontsize=12)
    stim.step(trace.time_seconds, binary, where="post", color=INK, linewidth=1.3)
    stim.set_yticks([0, 1], ["0", "1"], fontsize=10)
    stim.set_ylabel("Stimulus", fontsize=13, rotation=0, ha="right", va="center", labelpad=18)
    stim.set_xticks([0, 60, 120, 180, 240])
    stim.tick_params(axis="x", labelsize=12, pad=5)
    stim.set_xlabel("Time from recording start (s)", fontsize=14, labelpad=7)
    result = finish(fig, stem, [
        "Example worm OH16230:0924_01 (first ID). Trace scales differ; peaks are not clipped.",
        "Chemical labels come from event metadata. The model's stimulus input is binary (0/1).",
    ])
    save_csv(trace, stem)
    save_csv(events, "data_recording_events")
    caption(stem, f"""# NeuroPAL recordings and stimulus timing

The three traces are observed recordings from **{first_id}**, the first lexicographic worm ID, not the worm with the largest response. AWC is the existing observed-response case; RIP and URB are the existing modeled mean-effect pair. Channel choice is therefore illustrative and case-linked, not a representative population sample or a new response ranking. Axes preserve all recorded peaks and missing values; their vertical ranges differ. Values are per-channel, within-recording z-scores from the saved display export, not raw fluorescence or model effect units.

The shaded periods and binary strip use the actual first-worm schedule: pentanedione 60.5–70.5 s, butanone 120.5–130.5 s, NaCl 180.5–190.5 s. Recording order differs across worms; event position must not substitute for chemical identity. The current conditional model uses binary any-stimulus input. This single-worm view is descriptive, with no significance or inter-neuron causality claim.

The cohort is 17 OH16230 head recordings at 4 Hz, with 54 pooled class channels rather than 54 individual cells. The historical 80-class data include 63 head and 17 tail classes with different donor-filling/pairing provenance; they are not simply a larger simultaneous recording.

[Plotted traces](../data/{stem}.csv) · [Actual events](../data/data_recording_events.csv) · [Full original data provenance](../../figure_atlas_20260831/README.md#f01--data-and-provenance)
""")
    return result


def observed_onsets():
    table = pd.read_csv(DATA / "f09_awc_observed_onset_statistics.csv")
    worm_rows = pd.read_csv(DATA / "f09_awc_observed_worm_responses.csv")
    assert len(table) == 18 and table.n_worms.eq(17).all()
    results = []
    for chemical, label in CHEMICALS.items():
        stem = f"observed_awc_{chemical}"
        rows = table[table.chemical.eq(chemical)].sort_values("horizon_seconds").copy()
        assert len(rows) == 6
        for row in rows.itertuples():
            values = worm_rows[worm_rows.chemical.eq(chemical) &
                               worm_rows.horizon_frames.eq(row.horizon_frames)].onset_minus_quiet_z
            assert len(values) == 17
            assert np.isclose(values.mean(), row.mean_onset_minus_quiet_z, atol=1e-12)
        fig, ax = new_figure(f"Observed AWC response: {label}",
                            "Onset change minus matched quiet-time change · 17 worms")
        ax.set_position([.18, .29, .76, .47])
        x = rows.horizon_seconds.to_numpy()
        y = rows.mean_onset_minus_quiet_z.to_numpy()
        ax.set_xscale("log", base=2)
        ax.set_xlim(.20, 12.5)
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.set_ylim(-2.4, .9)
        ax.set_yticks([-2, -1, 0])
        ax.axhline(0, color=GREY, linestyle=(0, (3, 3)), linewidth=1.2)
        ax.errorbar(x, y, yerr=[y-rows.ci95_low, rows.ci95_high-y],
                    fmt="none", color=BLUE, linewidth=2, capsize=5)
        for row in rows.itertuples():
            ax.plot(row.horizon_seconds, row.mean_onset_minus_quiet_z, "o", markersize=9,
                    markeredgecolor=BLUE, markeredgewidth=1.8,
                    markerfacecolor=BLUE if row.bh_q_value < .05 else "white")
        # Separate tested nested-window averages; no interpolated time-course line.
        ax.set_xlabel("Post-onset averaging window (s)", labelpad=12)
        ax.set_ylabel("Observed activity contrast\n(non-stimulus SD units)", labelpad=14)
        result = finish(fig, stem, [
            "Bars: saved pointwise 95% worm-bootstrap intervals. Filled dots: saved BH q < 0.05.",
            "BH covers 54 neurons within each chemical/window. These are observed responses, not edges.",
        ])
        save_csv(rows, stem)
        caption(stem, f"""# Observed AWC response: {label}

Each point is the average across 17 worms of **(post-onset mean − immediately preceding 1-s mean) − (matched quiet-window mean − its preceding 1-s mean)**, divided by the worm/channel non-stimulus SD (floor 0.05). The quiet pseudo-onset is 15 s before the actual onset. The six post-onset windows are nested averages of 0.25, 0.5, 1, 2, 4, and 10 s, not instantaneous samples or estimated transmission delays. No interpolated time-course curve is drawn.

Bars are the archived 95% percentile worm-bootstrap intervals (5,000 resamples). Filled dots identify the saved BH q < 0.05 decisions; correction is across 54 neurons separately within each cohort/chemical/window, not jointly across chemicals or windows. The underlying saved tests are two-sided one-sample worm-level t tests. The exact p/q values are retained in the CSV. No tests or intervals were recomputed for this figure.

AWC is the prior named biological case, not newly selected by the largest effect. All three chemicals and all six windows are retained in separate, same-scale plots. There is one exposure to each chemical per worm. These are observed-activity associations, not inferred inter-neuron effects, adaptation tests, or between-chemical difference tests. An open dot does not establish absence of response.

[Exact values and q values](../data/{stem}.csv) · [Saved full observed analysis](../../figure_atlas_20260831/data/f09_awc_observed_onset_statistics.csv) · [Estimand implementation](../../../compatibility_neural_benchmark/chemical_observed_analysis.py)
""")
        results.append(result)
    return results


def modeled_onsets():
    table = pd.read_csv(DATA / "f08_distributional_screen_inventory.csv")
    rows = table[table.context.str.contains("onset_minus_baseline")].copy()
    assert len(rows) == 4
    labels = {
        "endpoint_wasserstein1": ("W1-distance effect", "Change in W1 distance\n(source-gap normalized)"),
        "endpoint_sd": ("SD effect", "Change in SD effect\n(source-gap normalized)"),
        "endpoint_log_sd": ("log-SD effect", "Change in log-SD effect\n(source-gap normalized)"),
    }
    results = []
    for row in rows.itertuples():
        stem = f"modeled_onset_{row.source_neuron.lower()}_{row.target_neuron.lower()}"
        pair = f"{row.source_neuron} → {row.target_neuron}"
        context = {"butanone_onset_minus_baseline": "Butanone events",
                   "pentanedione_onset_minus_baseline": "Pentanedione events",
                   "onset_minus_baseline": "Pooled stimulus events"}[row.context]
        short, ylabel = labels[row.screen_channel]
        fig, ax = new_figure(f"{pair}: onset-minus-baseline {short}",
                            f"{context} · source lag {row.source_lag_frames/4:g} s · readout {row.horizon_frames/4:g} s after cut")
        values = np.array([row.screen_mean_normalized, row.targeted_n128_mean_normalized])
        ax.set_position([.2, .29, .65, .44])
        ax.set_xlim(-.4, 1.4)
        low, high = min(0, values.min()), max(0, values.max())
        span = max(high - low, .02)
        ax.set_ylim(low - .43 * span, high + .40 * span)
        ax.axhline(0, color=GREY, linewidth=1.2, linestyle=(0, (3, 3)))
        ax.text(-.35, .06 * span, "No onset–baseline difference", fontsize=12,
                color=GREY, ha="left", va="bottom")
        ax.plot([0, 1], values, color=GREY, linewidth=1.2, zorder=1)
        for x, value in enumerate(values):
            ax.plot(x, value, "o", color=BLUE, markersize=11,
                    markerfacecolor="white" if x == 0 else BLUE, markeredgewidth=2, zorder=3)
            label = f"{value:+.4f}" if abs(value) >= .001 else f"{value:+.6f}"
            ax.annotate(label, (x, value), xytext=(0, -24), textcoords="offset points",
                        ha="center", va="top", fontsize=14, color=INK)
        ax.set_xticks([0, 1], ["Screen", "128-particle rerun"], fontsize=15)
        ax.tick_params(axis="x", pad=12, length=0)
        ax.set_ylabel(ylabel, labelpad=15, fontsize=15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        result = finish(fig, stem, [
            "Selected exploratory contrast; no exact matched sampler controls or run-difference test.",
            "Points are saved estimates. Chemical labels describe events; the model input remains binary.",
        ])
        save_csv(table[table.candidate_id.eq(row.candidate_id)], stem)
        caption(stem, f"""# {pair}: onset-minus-baseline {short}

This plot retains the archived screen and 128-particle follow-up estimates for {pair}, context `{row.context}`, source-window-end lag **{row.source_lag_frames/4:g} s before the cut**, and target endpoint **{row.horizon_frames/4:g} s after the cut**. Each state effect compares high-source and low-source repaired target predictions; the plotted context contrast subtracts the baseline effect from the onset effect. Event-wise quantities are normalized by `max(abs(achieved_source_gap), 0.1)` before aggregation. The numerical screen-to-rerun difference is descriptive, not a significance test or an independent biological replication.

The two saved estimates are {row.screen_mean_normalized:+.9f} and {row.targeted_n128_mean_normalized:+.9f}. The plotted screen-consistency table contains point estimates, not run-difference intervals. No uncertainty was invented. All four selected onset follow-ups are included as separate plots to preserve their distinct outcomes and units. Their N128 point estimates are closer to zero; this is an arithmetic description, not evidence that the true effect vanishes.

None of the four exact source/target/context/lag/horizon/channel selections has matched sampler controls in the saved combined-eight control panel. Full-family active-minus-baseline tests cannot be borrowed for these onset selections. These plots are exploratory, not statistically established onset-specific interactions.

For W1, the within-state distance is unsigned, but onset minus baseline is signed; a negative contrast is a smaller distance at onset, not target inhibition. SD is spread, not variance; log-SD is a log spread measure, not a raw activity change. Neither alone establishes gain modulation. The generator conditions on binary any-stimulus input; chemical strata are observed event contexts, not chemically specified interventions. Phase-specific source targets and windows may differ across contexts.

[Exact selected row](../data/{stem}.csv) · [All six distributional follow-ups](../../figure_atlas_20260831/data/f08_distributional_screen_inventory.csv) · [Full prediction definitions](../../figure_atlas_20260831/PREDICTION_CAPTIONS.md)
""")
        results.append(result)
    return results


def main():
    for folder in ("figures", "data", "captions", "qa"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    sources = [DATA / name for name in [
        "f01_example_recording_display_z.csv", "f01_stimulus_schedules.csv",
        "f09_awc_observed_onset_statistics.csv", "f09_awc_observed_worm_responses.csv",
        "f08_distributional_screen_inventory.csv",
    ]] + [OLD / "PREDICTION_CAPTIONS.md", OLD / "README.md", OLD / "data_method_manifest.json"]
    hashes = verify_old_inputs(sources)
    original_manifest = json.loads((OLD / "data_method_manifest.json").read_text())
    for name in ["results/neural_prediction_atlas_20260829/canonical/protocol.json",
                 "compatibility_neural_benchmark/chemical_observed_analysis.py"]:
        assert sha256(ROOT / name) == original_manifest["source_hashes"][name]
        hashes[name] = sha256(ROOT / name)
    figures = [recordings()] + observed_onsets() + modeled_onsets()
    write_manifest("data_onset", hashes, figures, {
        "output_files": EXPORTS,
        "code": {"path": relative(__file__), "sha256": sha256(__file__)},
        "n_new_figures": len(figures),
        "selection": "First worm ID; AWC/RIP/URB are previously named cases. All three AWC chemical panels and all four selected onset reruns retained.",
        "observed_bh_scope": "54 neurons within each chemical/window; no global chemical/window correction",
        "modeled_onset_inference": "Selected point estimates only; no exact matched controls or significance claim",
    })
    print(json.dumps({"created": [figure["stem"] for figure in figures]}, indent=2))


if __name__ == "__main__":
    main()
