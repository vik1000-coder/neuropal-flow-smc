"""Shared styling and provenance for the clarity-first static figure set."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/figure_story_20260831"
OLD = ROOT / "results/figure_atlas_20260831"
BLUE = "#326B9B"
ORANGE = "#BA682D"
INK = "#252B30"
GREY = "#75808A"
LIGHT = "#E6EBEF"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "axes.titlesize": 17, "axes.titleweight": "semibold",
        "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 13, "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B5BEC5", "grid.color": LIGHT, "axes.axisbelow": True,
        "lines.linewidth": 2.3, "svg.fonttype": "none", "savefig.facecolor": "white",
        "svg.hashsalt": "neuropal-clarity-first-20260831",
    })


def new_figure(title, subtitle=""):
    style()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=220)
    fig.subplots_adjust(left=0.17, right=0.95, bottom=0.29, top=0.76)
    fig.text(0.045, 0.95, title, fontsize=20, weight="semibold", va="top")
    if subtitle:
        fig.text(0.045, 0.879, subtitle, fontsize=14, color=GREY, va="top")
    return fig, ax


def verify_old_inputs(paths):
    ledger = OLD / "SHA256SUMS"
    entries = {}
    for line in ledger.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        entries[(ledger.parent / name.lstrip("*")).resolve()] = expected
    verified = {}
    for path in paths:
        path = Path(path).resolve()
        actual = sha256(path)
        assert path in entries, f"Input is absent from old figure ledger: {path}"
        assert actual == entries[path], f"Frozen input changed: {path}"
        verified[relative(path)] = actual
    verified[relative(ledger)] = sha256(ledger)
    return verified


def finish(fig, stem, caption_lines):
    assert len(caption_lines) <= 2, "Move longer explanations to the technical caption."
    for index, line in enumerate(caption_lines):
        fig.text(0.045, 0.116 - index * 0.04, line, fontsize=12, va="top")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    checked = 0
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        # Matplotlib retains invisible out-of-range tick labels as artists.
        if getattr(text, "axes", None) is not None and not text.axes.get_visible():
            continue
        box = text.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width + 1 and box.y1 <= height + 1, (
            f"Text outside {stem} canvas: {text.get_text()!r}, {box.bounds}")
        checked += 1
    paths = []
    for directory in ("figures", "qa", "data", "captions"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = OUT / "figures" / f"{stem}.{ext}"
        kwargs = {"metadata": {"Date": None}} if ext == "svg" else {}
        fig.savefig(path, dpi=220, **kwargs)
        paths.append(path)
    review = OUT / "qa" / f"{stem}_1100px.png"
    fig.savefig(review, dpi=1100 / fig.get_figwidth())
    paths.append(review)
    plt.close(fig)
    svg = ET.parse(OUT / "figures" / f"{stem}.svg")
    assert svg.findall(".//{http://www.w3.org/2000/svg}text")
    return {
        "stem": stem, "outputs": {relative(path): sha256(path) for path in paths},
        "caption_lines": caption_lines,
        "layout": {"text_bounds_checked": checked, "all_text_inside_canvas": True,
                   "review_width_pixels": 1100, "editable_svg_text": True},
        "visual_review": "pending",
    }


def write_manifest(name, inputs, figures, extra=None):
    extra = dict(extra or {})
    output_files = extra.pop("output_files", [])
    outputs = {key: value for figure in figures for key, value in figure["outputs"].items()}
    for path in output_files:
        outputs[relative(path)] = sha256(path)
    metadata = {
        "status": "rendered_pending_visual_review",
        "no_new_fits_model_samples_or_inference": True,
        "source_hashes": inputs,
        "common_renderer": {"path": relative(__file__), "sha256": sha256(__file__)},
        "figures": figures,
        "output_hashes": outputs,
        **extra,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_manifest.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path
