"""Clarity-first schematic of the frozen repaired-history sampling estimand.

Every drawn trace/density is explicitly illustrative, not a model/data sample.
No training, path sampling, or significance tests are performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/figure_redesign_prototypes_20260831"
BLUE = "#326B9B"
ORANGE = "#BA682D"
INK = "#252B30"
GREY = "#89929A"
LIGHT = "#E9EDF0"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(path, ledger):
    entries = {}
    for line in ledger.read_text().splitlines():
        h, name = line.split(maxsplit=1)
        entries[(ledger.parent / name).resolve()] = h
    assert digest(path) == entries[path.resolve()], path


def arrow(fig, start, end, color=GREY, style="->", linewidth=1.5):
    artist = FancyArrowPatch(start, end, transform=fig.transFigure,
                             arrowstyle=style, mutation_scale=14,
                             linewidth=linewidth, color=color)
    fig.add_artist(artist)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "results/neural_prediction_atlas_20260829/canonical/protocol.json"
    verify_source(protocol_path, protocol_path.parent / "checksums.sha256")
    protocol = json.loads(protocol_path.read_text())
    assert protocol["history_frames"] == 80 and protocol["fps"] == 4
    assert protocol["source_window_frames"] == 4
    assert protocol["effect_definition"] == "high-source repaired response minus low-source repaired response"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": GREY, "ytick.color": GREY,
        "svg.fonttype": "none", "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#B4BDC4",
    })
    fig = plt.figure(figsize=(12.5, 7.0))
    fig.text(.045, .94, "How sampling defines a neural effect", fontsize=25, weight="bold")
    fig.text(.045, .885, "Same trained model · same stimulus schedule", fontsize=15, color="#58646D")
    for x, label in [(.045, "1. Start from one history"),
                     (.365, "2. Sample compatible histories"),
                     (.725, "3. Compare predictions")]:
        fig.text(x, .775, label, fontsize=16, weight="bold")

    # Shared starting-history illustration: these are NOT observed waveforms.
    t = np.linspace(-20, 0, 161)
    ax = fig.add_axes([.045, .445, .20, .23])
    rows = []
    for i in range(3):
        y = .16 * np.sin((t + 1.7 * i) * .62) + .07 * np.sin(t * 1.9 + i) + (2 - i) * .62
        ax.plot(t, y, color="#667680", lw=1.5)
        rows.extend({"illustration": "starting_history", "channel": i + 1,
                     "time_before_repair_s": float(a), "activity": float(b)} for a, b in zip(t, y))
    ax.set_xlim(-20, 0); ax.set_ylim(-.4, 1.75)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_title("Neural activity", loc="left", fontsize=14, pad=8)
    stim_ax = fig.add_axes([.045, .32, .20, .07])
    stim = ((t >= -7) & (t < -3)).astype(int)
    stim_ax.step(t, stim, where="post", color=INK, lw=1.7)
    stim_ax.set_title("Stimulus history", loc="left", fontsize=14, pad=7)
    stim_ax.set_ylim(-.12, 1.25); stim_ax.set_xlim(-20, 0)
    stim_ax.set_yticks([0, 1], ["0", "1"]); stim_ax.tick_params(labelsize=11, length=0)
    stim_ax.set_xticks([])
    stim_ax.spines[["left", "bottom"]].set_visible(False)
    fig.text(.145, .285, "20 s of starting history", ha="center", fontsize=12, color="#58646D")
    rows.extend({"illustration": "stimulus_history", "channel": 0,
                 "time_before_repair_s": float(a), "activity": float(b)} for a, b in zip(t, stim))

    # The four source frames close two seconds before the prediction cut.
    rt = np.arange(-12, 1) / 4
    mean_path = np.array([0, .70, .82, .77, .76, .58, .44, .27, .31, .21, .10, .14, .08])
    source_indices = np.array([1, 2, 3, 4])
    assert np.array_equal(rt[source_indices], np.array([-2.75, -2.5, -2.25, -2]))
    for upper, color, sign, bottom, label in [
        (True, ORANGE, 1, .545, "Higher source history"),
        (False, BLUE, -1, .34, "Lower source history"),
    ]:
        ax = fig.add_axes([.365, bottom, .265, .12])
        ax.axvspan(-3, -2, color=LIGHT, zorder=0)
        for j, perturb in enumerate([-.12, 0, .12]):
            wave = sign * mean_path + perturb * np.sin(np.linspace(0, 2.5, len(rt)))
            ax.plot(rt, wave, color=color, alpha=.4 if perturb else 1,
                    lw=1.2 if perturb else 2.0)
            rows.extend({"illustration": label, "channel": j + 1,
                         "time_before_cut_s": float(a), "activity": float(b)} for a, b in zip(rt, wave))
        ax.axvline(0, color=GREY, lw=1.2, clip_on=False)
        ax.set_title(label, color=color, fontsize=14, loc="left", pad=6)
        ax.set_xlim(-3, 0); ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["left", "bottom"]].set_visible(False)
    fig.text(.365 + .265 / 6, .305, "Source\nwindow", ha="center", va="top", fontsize=12)
    fig.text(.63, .305, "Prediction\ncut", ha="center", va="top", fontsize=12)
    arrow(fig, (.365 + .265 / 3, .318), (.63, .318), style="<->", linewidth=1.2)
    fig.text(.365 + .265 * 2 / 3, .288, "Lag", ha="center", fontsize=12)
    fig.text(.365, .215, "Compatible population histories", fontsize=13, color="#58646D")
    arrow(fig, (.25, .525), (.35, .607))
    arrow(fig, (.25, .525), (.35, .40))

    ax = fig.add_axes([.755, .36, .20, .31])
    x = np.linspace(-2.7, 2.7, 401)
    sd = .66
    for name, mu, color in [("Lower", -.65, BLUE), ("Higher", .65, ORANGE)]:
        y = np.exp(-.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax.plot(x, y, color=color, lw=2.3)
        ax.plot([mu, mu], [0, .76], color=color, lw=1, ls=(0, (3, 3)))
        rows.extend({"illustration": "target_density_" + name.lower(),
                     "target_activity": float(a), "density": float(b)} for a, b in zip(x, y))
    ax.text(-1.65, .64, "Lower\nsource", color=BLUE, ha="center", fontsize=12)
    ax.text(1.65, .64, "Higher\nsource", color=ORANGE, ha="center", fontsize=12)
    ax.annotate("", xy=(.65, .8), xytext=(-.65, .8),
                arrowprops={"arrowstyle": "<->", "lw": 1.2, "color": INK})
    ax.text(0, .87, "Mean change", fontsize=13, ha="center")
    ax.set_ylim(0, 1.08); ax.set_xlim(-2.7, 2.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Target activity", fontsize=14, labelpad=8)
    ax.set_ylabel("Probability density", fontsize=12, labelpad=7)
    ax.spines["left"].set_visible(False)
    fig.text(.755, .703, "At one time after the cut", fontsize=13, color="#58646D")
    arrow(fig, (.647, .607), (.72, .535), color=ORANGE)
    arrow(fig, (.647, .40), (.72, .465), color=BLUE)
    fig.text(.755, .235, "Higher-source mean\n− lower-source mean", fontsize=14)

    caption = (
        "Progressive SMC resamples compatible low/high source histories, then predicts the target.\n"
        "The contrast is model-based; other neurons may also change during repair. All curves are schematic."
    )
    fig.text(.045, .085, caption, fontsize=12, linespacing=1.6, color="#45515B", va="top")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    checked_text = 0
    for artist in fig.findobj(match=Text):
        if artist.get_visible() and artist.get_text().strip():
            b = artist.get_window_extent(renderer)
            assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width + 1 and b.y1 <= fig.bbox.height + 1, artist.get_text()
            checked_text += 1
    outputs = []
    for ext in ["png", "svg"]:
        path = OUT / f"sampling_diagram.{ext}"
        fig.savefig(path, dpi=220, bbox_inches=None)
        outputs.append(path)
    review_path = OUT / "qa/sampling_1100px.png"
    review_path.parent.mkdir(exist_ok=True)
    fig.savefig(review_path, dpi=88, bbox_inches=None)
    outputs.append(review_path)
    plt.close(fig)
    data_path = OUT / "sampling_illustrations.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    outputs.append(data_path)
    caption_path = OUT / "SAMPLING_CAPTION.md"
    caption_path.write_text(
        "# Sampling prototype\n\n"
        "Intended points: (1) the two branches share their starting history, trained model, and stimulus schedule; "
        "(2) an effect is a contrast between target predictions under compatible lower- and higher-source histories.\n\n"
        "## Short caption\n\n" + caption.replace("\n", " ") + "\n\n"
        "## Definitions retained outside the figure\n\n"
        "The left traces and binary pulse are teaching illustrations, not observed recordings. "
        "The model uses 80 history frames (20 s at 4 Hz). The repair illustration has four source frames "
        "at −2.75, −2.5, −2.25, and −2 s relative to the prediction cut. Thus the source-window end is "
        "2 s before the cut; the target distribution is evaluated at a separately chosen time after the cut. "
        "The two example target densities illustrate a positive mean contrast only; they do not assert "
        "that higher source activity must increase a target. Other effect channels require their own definitions.\n\n"
        "Progressive bridge SMC gradually applies a soft source-window constraint while maintaining population "
        "compatibility under the conditional model. It does not hold every other neuron fixed or perform a physical intervention. "
        "The same observed episode anchors both repair branches, but their simulated histories may differ. "
        "Atlas effects divide the high-minus-low contrast by the achieved source gap (floor 0.1), then aggregate within worms. "
        "That numerical normalization, Monte Carlo support checks, and inferential tests are intentionally not drawn as additional stages.\n\n"
        "Sources: [canonical protocol](../neural_prediction_atlas_20260829/canonical/protocol.json) and "
        "[repaired-path method specification](../../FLOW_REPAIRED_LAG_METHODS_20260828.md). "
        "Only the generic sampler definitions are taken from the earlier methods record; current cohort/encoding semantics use the canonical protocol.\n"
    )
    outputs.append(caption_path)
    source_paths = [protocol_path, ROOT / "FLOW_REPAIRED_LAG_METHODS_20260828.md",
                    ROOT / "compatibility_neural_benchmark/prediction_atlas_runner.py", Path(__file__)]
    manifest = {
        "status": "rendered_pending_visual_review",
        "main_points": ["Shared starting conditions", "Compare target predictions after source-history repair"],
        "all_curves_are_schematic": True, "new_fits_samples_or_inference": False,
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "output_hashes": {str(p.relative_to(ROOT)): digest(p) for p in outputs},
        "semantic_checks": {"history_frames": 80, "fps": 4, "source_window_frames": 4,
                            "illustrative_source_window_end_lag_seconds": 2,
                            "source_lag_is_not_forecast_horizon": True,
                            "density_mean_difference_is_illustrative_not_measured": True},
        "layout_checks": {"text_bounds_checked": checked_text, "all_text_within_canvas": True,
                          "review_width_pixels": 1100, "visual_review": "pending"},
    }
    (OUT / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"outputs": [str(p.relative_to(ROOT)) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
