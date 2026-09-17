"""Figure generation (matplotlib, Agg backend). Sections 20.1-20.3."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _ensure(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def kernel_recovery_plot(timescales_or_lags, estimate, oracle, path, title="Kernel recovery",
                         ci_low=None, ci_high=None):
    _ensure(path)
    x = np.asarray(timescales_or_lags, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, oracle, "k--o", label="oracle", markersize=4)
    ax.plot(x, estimate, "C0-s", label="estimate", markersize=4)
    if ci_low is not None and ci_high is not None:
        ax.fill_between(x, ci_low, ci_high, color="C0", alpha=0.2, label="sup-t band")
    ax.set_xlabel("lag / timescale")
    ax.set_ylabel("gain (d log v)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pit_histogram(u, path, title="PIT histogram"):
    _ensure(path)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(u, bins=20, range=(0, 1), density=True, color="C0", alpha=0.8)
    ax.axhline(1.0, color="k", ls="--")
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def eprocess_trace(trace, threshold, path, t_change=None, alarm=None, title="e-process"):
    _ensure(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trace, "C0-", lw=1)
    ax.axhline(threshold, color="r", ls="--", label=f"threshold 1/α={threshold:g}")
    if t_change is not None:
        ax.axvline(t_change, color="g", ls=":", label="true change")
    if alarm is not None:
        ax.axvline(alarm, color="k", ls="-.", label="alarm")
    ax.set_yscale("log")
    ax.set_xlabel("stream index")
    ax.set_ylabel("e-value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def heatmap(M, path, xlabel="source", ylabel="target", title="readout heatmap"):
    _ensure(path)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(M)) if np.any(M) else -1,
                   vmax=np.nanmax(np.abs(M)) if np.any(M) else 1)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
