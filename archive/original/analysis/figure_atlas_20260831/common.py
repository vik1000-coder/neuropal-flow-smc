"""Shared export conventions for the standalone scientific figure set."""
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/figure_atlas_20260831"
BLUE = "#326B9B"
ORANGE = "#BA682D"
INK = "#252B30"
GREY = "#75808A"
LIGHT = "#E6EBEF"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "semibold",
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.facecolor": "white",
        "axes.facecolor": "white", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#A4ABB2", "grid.color": "#E4E8EC",
        "axes.axisbelow": True, "lines.linewidth": 1.8,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_figure(fig, stem):
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("png", "svg"):
        path = folder / f"{stem}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.18)
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return files
