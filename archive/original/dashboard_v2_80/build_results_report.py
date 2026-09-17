"""Build the frozen LaTeX/PDF report for the optimized historical-80 atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))
from compatibility_neural_benchmark.postfreeze_external_analysis import load_references
from compatibility_neural_benchmark.sbtg80_progressive_sensitivity import (
    _load_published_slices,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/sbtg80_optimized_full_atlas_20260902"
EXTERNAL = RESULT / "external_reference_checks"
DEFAULT_OUTPUT = RESULT / "latex_report"
REFERENCE_RELEASE = Path("/Users/vik/Downloads/SBTG-public-release copy")
PUBLISHED_ARCHIVE = REFERENCE_RELEASE / "results/paper/sbtg_lag_matrices.npz"

NETWORK_LABELS = {
    "monoamine_all": "All monoamines",
    "monoamine_dopamine": "Dopamine",
    "monoamine_serotonin": "Serotonin",
    "monoamine_tyramine": "Tyramine",
    "monoamine_octopamine": "Octopamine",
    "neuropeptide_all": "All neuropeptides",
    "neuromodulator_union": "Neuromodulator union",
}


def value(number: float, digits: int = 3) -> str:
    return "--" if pd.isna(number) else f"{number:.{digits}f}"


def lag(number: float) -> str:
    return "--" if pd.isna(number) else f"{number / 4:g} s"


def percentile(matrix: np.ndarray, mask: np.ndarray, target: int, source: int) -> float:
    values = np.abs(matrix[mask])
    selected = abs(float(matrix[target, source]))
    return float((np.sum(values < selected) + 0.5 * np.sum(values == selected)) / len(values))


def write_bentley_edge_table(output: Path) -> Path:
    """Write one inspectable row per positive Bentley neuron relationship."""

    atlas_path = RESULT / "atlas/atlas_matrices.npz"
    with np.load(atlas_path, allow_pickle=False) as atlas:
        neurons = tuple(atlas["neurons"].astype(str))
        endpoint = atlas[
            "mean_normalized__progressive_bridge_smc__endpoint_mean__state_average"
        ][0, 0]
        endpoint_low = atlas[
            "ci_low_normalized__progressive_bridge_smc__endpoint_mean__state_average"
        ][0, 0]
        endpoint_high = atlas[
            "ci_high_normalized__progressive_bridge_smc__endpoint_mean__state_average"
        ][0, 0]
        wasserstein = atlas[
            "mean_normalized__progressive_bridge_smc__endpoint_wasserstein1__state_average"
        ][0, 0]
        support = atlas[
            "valid_fraction__progressive_bridge_smc__state_average"
        ][0]
    _, networks = load_references(REFERENCE_RELEASE, list(neurons))
    published = next(
        item["matrix"]
        for item in _load_published_slices(PUBLISHED_ARCHIVE, neurons)
        if item["lag_frames"] == 1
    )
    tail = set(json.loads((RESULT / "manifest.json").read_text())["recording_origin"]["tail_neurons"])
    off = ~np.eye(len(neurons), dtype=bool)
    rows = []
    for network in NETWORK_LABELS:
        labels = networks[network].astype(bool)
        eligible = labels.any(axis=0)
        scope = off & eligible[None, :]
        for target, source in np.argwhere(labels & off):
            rows.append(
                {
                    "network": network,
                    "source_neuron": neurons[source],
                    "target_neuron": neurons[target],
                    "source_recording_origin": "tail" if neurons[source] in tail else "head",
                    "target_recording_origin": "tail" if neurons[target] in tail else "head",
                    "source_support_fraction_lag1": float(support[source]),
                    "source_support_qualified_lag1": bool(support[source] >= 0.5),
                    "progressive_endpoint_mean_lag1_h1": float(endpoint[target, source]),
                    "progressive_endpoint_mean_ci_low": float(endpoint_low[target, source]),
                    "progressive_endpoint_mean_ci_high": float(endpoint_high[target, source]),
                    "progressive_endpoint_wasserstein1_lag1_h1": float(wasserstein[target, source]),
                    "published_sbtg_score_lag1": float(published[target, source]),
                    "progressive_endpoint_abs_percentile_among_eligible_pairs": percentile(endpoint, scope, target, source),
                    "progressive_wasserstein_percentile_among_eligible_pairs": percentile(wasserstein, scope, target, source),
                    "published_sbtg_abs_percentile_among_eligible_pairs": percentile(published, scope, target, source),
                    "orientation": "source_neuron_to_target_neuron; matrices are target-row/source-column",
                }
            )
    path = output / "bentley_positive_neuron_relationships.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build(output: Path) -> Path:
    output = output.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (RESULT / "atlas/figures/head_tail_endpoint_mean_uncertainty.png", "head_tail.png"),
        (EXTERNAL / "figures/randi_cook_auroc.png", "randi_cook_auroc.png"),
        (EXTERNAL / "figures/bentley_auroc.png", "bentley_auroc.png"),
        (EXTERNAL / "figures/bentley_lagmax.png", "bentley_lagmax.png"),
    ):
        shutil.copy2(source, figures / target)
    edge_table_path = write_bentley_edge_table(output)

    primary = pd.read_csv(EXTERNAL / "primary_lag1_comparison.csv")
    primary_rows = []
    for reference, subset in primary.groupby("reference"):
        progressive = subset[subset.method == "progressive_bridge_smc"].iloc[0]
        published = subset[subset.method == "sbtg_published"].iloc[0]
        primary_rows.append(
            f"{reference.replace('_', r'\_')} & {progressive.auroc:.3f} & "
            f"{published.auroc:.3f} & {progressive.auroc - published.auroc:+.3f} \\\\"
        )

    metrics = pd.read_csv(EXTERNAL / "bentley_metrics.csv")
    mask = (
        (metrics.method == "progressive_bridge_smc")
        & (metrics.channel == "endpoint_mean")
        & (metrics.context == "state_average")
        & (metrics.lag_frames == 1)
        & (metrics.horizon_frames == 1)
        & (metrics.scope == "eligible_support_qualified")
    )
    progressive = metrics[mask].set_index("network")
    progressive_w1 = metrics[
        (metrics.method == "progressive_bridge_smc")
        & (metrics.channel == "endpoint_wasserstein1")
        & (metrics.context == "state_average")
        & (metrics.lag_frames == 1)
        & (metrics.horizon_frames == 1)
        & (metrics.scope == "eligible_support_qualified")
    ].set_index("network")
    published = metrics[
        (metrics.method == "sbtg_published")
        & (metrics.lag_frames == 1)
        & (metrics.scope == "eligible_sources")
    ].set_index("network")
    order = list(NETWORK_LABELS)
    y = np.arange(len(order), dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    p_values = np.asarray([progressive.loc[network].auroc for network in order], dtype=float)
    s_values = np.asarray([published.loc[network].auroc for network in order], dtype=float)
    ax.axvline(0.5, color="#728087", linestyle="--", linewidth=1, label="Chance AUROC")
    ax.scatter(p_values, y - 0.13, s=48, color="#287d75", label="Seed 1701 progressive")
    ax.scatter(s_values, y + 0.13, s=48, color="#d98b4b", label="Published SBTG")
    for index, number in enumerate(p_values):
        if not np.isfinite(number):
            ax.text(0.505, y[index] - 0.13, "not evaluable", fontsize=8, va="center")
    ax.set_yticks(y, [NETWORK_LABELS[network] for network in order])
    ax.invert_yaxis()
    ax.set_xlim(0.46, 0.69)
    ax.set_xlabel("Lag-1 AUROC")
    ax.set_title("Bentley correspondence by neuromodulator network", pad=34)
    ax.grid(axis="x", color="#d9e0e2", linewidth=0.7)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    fig.savefig(figures / "bentley_specific_lag1.png", dpi=220)
    plt.close(fig)
    bentley_lag1_rows = []
    for network, label in NETWORK_LABELS.items():
        p, w, s = progressive.loc[network], progressive_w1.loc[network], published.loc[network]
        bentley_lag1_rows.append(
            f"{label} & {int(p.n_eligible_sources)} & {int(p.n_supported_eligible_sources)} & "
            f"{value(p.auroc)} & {value(w.auroc)} & {value(s.auroc)} & "
            f"{value(p.auprc)} & {value(w.auprc)} & {value(s.auprc)} \\\\"
        )

    all_inference = pd.read_csv(EXTERNAL / "bentley_lagmax_inference.csv")
    inference = all_inference[
        (all_inference.lag_grid == "native_method_grid")
        & (all_inference.method == "progressive_bridge_smc")
        & (all_inference.channel == "endpoint_mean")
        & (all_inference.context == "state_average")
        & (all_inference.horizon_frames == 1)
    ].set_index("network")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    best = np.asarray([inference.loc[network].best_auroc for network in order], dtype=float)
    ax.axvline(0.5, color="#728087", linestyle="--", linewidth=1)
    ax.scatter(best, y, s=52, color="#287d75")
    for index, network in enumerate(order):
        row = inference.loc[network]
        if np.isfinite(row.best_auroc):
            ax.text(
                row.best_auroc + 0.004,
                y[index],
                f"{row.best_lag_frames / 4:g} s · q={row.max_lag_bh_q:.3f}",
                fontsize=8,
                va="center",
            )
        else:
            ax.text(0.505, y[index], "not evaluable", fontsize=8, va="center")
    ax.set_yticks(y, [NETWORK_LABELS[network] for network in order])
    ax.invert_yaxis()
    ax.set_xlim(0.49, 0.72)
    ax.set_xlabel("Best-lag AUROC; label gives lag and global BH q")
    ax.set_title("Endpoint-mean lag-max correspondence with permutation calibration")
    ax.grid(axis="x", color="#d9e0e2", linewidth=0.7)
    fig.savefig(figures / "bentley_specific_lagmax.png", dpi=220)
    plt.close(fig)
    bentley_lagmax_rows = []
    for network, label in NETWORK_LABELS.items():
        row = inference.loc[network]
        bentley_lagmax_rows.append(
            f"{label} & {int(row.n_common_supported_eligible_sources)} & {int(row.n_positive)} & "
            f"{lag(row.best_lag_frames)} & {value(row.best_auroc)} & "
            f"{value(row.max_lag_permutation_p, 4)} & {value(row.max_lag_bh_q, 4)} \\\\"
        )

    profile_labels = {
        ("progressive_bridge_smc", "endpoint_mean", "state_average"): "Progressive mean",
        ("progressive_bridge_smc", "endpoint_log_sd", "state_average"): "Progressive log-SD",
        ("progressive_bridge_smc", "endpoint_wasserstein1", "state_average"): "Progressive W1",
        ("progressive_bridge_smc", "endpoint_mean", "onset_minus_baseline"): "Progressive onset-baseline",
        ("sbtg_published", "score_product", "all_windows"): "Published SBTG",
    }
    full_lagmax_rows = []
    for row in all_inference.sort_values(
        ["lag_grid", "network", "method", "channel", "context"]
    ).itertuples():
        profile = profile_labels[(row.method, row.channel, row.context)]
        grid = "native" if row.lag_grid == "native_method_grid" else "common 1/8"
        full_lagmax_rows.append(
            f"{grid} & {NETWORK_LABELS[row.network]} & {profile} & "
            f"{int(row.n_common_supported_eligible_sources)} & {int(row.n_positive)} & "
            f"{lag(row.best_lag_frames)} & {value(row.best_auroc)} & "
            f"{value(row.max_lag_permutation_p, 4)} & {value(row.max_lag_bh_q, 4)} \\\\"
        )

    edges = pd.read_csv(edge_table_path)
    top_pair_sections = []
    for network, label in NETWORK_LABELS.items():
        group = edges[edges.network == network].copy()
        supported = group[group.source_support_qualified_lag1]
        display = supported if len(supported) else group
        display = display.sort_values(
            "progressive_endpoint_abs_percentile_among_eligible_pairs",
            ascending=False,
        ).head(10)
        rows = []
        for row in display.itertuples():
            rows.append(
                f"{row.source_neuron} $\\to$ {row.target_neuron} & "
                f"{100 * row.source_support_fraction_lag1:.0f}\\% & "
                f"{row.progressive_endpoint_mean_lag1_h1:+.3f} & "
                f"[{row.progressive_endpoint_mean_ci_low:+.3f}, {row.progressive_endpoint_mean_ci_high:+.3f}] & "
                f"{row.progressive_endpoint_wasserstein1_lag1_h1:.3f} & "
                f"{row.published_sbtg_score_lag1:+.3f} & "
                f"{100 * row.progressive_endpoint_abs_percentile_among_eligible_pairs:.1f}\\% \\\\"
            )
        qualifier = (
            "Top support-qualified Bentley-positive relationships by absolute progressive endpoint-mean rank."
            if len(supported)
            else "No relationship has a support-qualified source; rows are shown only to document the unsupported source."
        )
        top_pair_sections.append(
            rf"""\Needspace{{0.38\textheight}}
\subsection*{{{label}}}
{qualifier}
\begin{{center}}\scriptsize
\begin{{tabular}}{{llrrrrr}}
\toprule
Pair & Support & Mean & 95\% interval & W1 & SBTG & Mean rank \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{center}
"""
        )

    strongest = all_inference.sort_values("max_lag_bh_q").iloc[0]
    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,xcolor,hyperref,longtable,pdflscape,float,needspace}
\definecolor{ink}{HTML}{172126}
\definecolor{teal}{HTML}{287D75}
\hypersetup{hypertexnames=false,colorlinks=true,linkcolor=teal,urlcolor=teal}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\renewcommand{\arraystretch}{1.15}
\begin{document}
\begin{center}
{\LARGE\bfseries Historical 80-Neuron Response Atlas}\par
\vspace{0.35em}
{\large Optimized progressive-bridge SMC results and external correspondence checks}\par
\vspace{0.4em}
{\small Frozen analysis: 2 September 2026 \quad Report build: 3 September 2026}
\end{center}

\section*{Result at a glance}
The finalized single-generator atlas uses seed 1701, selected only by five-fold held-out predictive score. It contains 80 pooled neuron classes, 20 historical traces, four history lags, six forecast horizons, seven response channels, and thirteen contexts. All 20 raw sampling archives passed validation, and the lag-1/horizon-1 arrays match the frozen optimization run bit-for-bit.

The atlas estimates model-relative observational response sensitivity. It does not identify a causal intervention effect, a synapse, receptor action, or physical transmission delay. The historical SBTG80 construction pseudo-pairs head and tail recordings and donor-imputes missing traces; this limitation applies to every result in this report.

\subsection*{Bentley answer first}
The most defensible Bentley results are the broad network summaries. At the prespecified lag-1/horizon-1 cell, progressive endpoint-Wasserstein AUROC is 0.565 for neuropeptides and 0.547 for the neuromodulator union, compared with published-SBTG AUROCs of 0.523 and 0.525. After selecting over lags and adjusting the complete planned family, the progressive endpoint-Wasserstein lag-1 results for neuropeptides (AUROC 0.567) and the union (AUROC 0.550) both have within-source permutation $p=0.001$ and global BH $q=0.0155$.

For individual monoamines, octopamine has the strongest prespecified progressive endpoint-mean value (AUROC 0.630 versus published SBTG 0.495), but it is based on one eligible source neuron. Dopamine has two eligible sources; serotonin, tyramine, and octopamine each have one. On the common two-lag grid, serotonin W1 reaches AUROC 0.700 with global $q=0.0413$, but this is driven by a single eligible source and is not independent biological replication. Tyramine is not evaluable for progressive SMC because its sole eligible source fails the saved support gate. These are rank-correspondence findings against binary receptor/pathway edges, not evidence of neuromodulator transmission or causal influence.

\section*{Lag-1 correspondence with Randi and Cook}
At the prespecified 0.25-s lag and 0.25-s forecast, the single-seed progressive result exceeds the published SBTG point estimate for all four references.

\begin{center}
\begin{tabular}{lrrr}
\toprule
Reference & Progressive & Published SBTG & Difference \\
\midrule
""" + "\n".join(primary_rows) + r"""
\bottomrule
\end{tabular}
\end{center}

The paired source-bootstrap difference interval excludes zero for Cook structural, Cook chemical, and Cook gap. It crosses zero for Randi. A separate two-seed sensitivity improves every point estimate but is not averaged into the primary atlas or its uncertainty bands.

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{figures/randi_cook_auroc.png}
\caption{Lag-indexed AUROC correspondence with the Randi functional and Cook anatomical references.}
\end{figure}

\section*{Bentley neuromodulator correspondence}
Bentley is reduced to directed binary receptor/pathway-edge existence after duplicate receptor rows are collapsed. AUROC is primary because row multiplicity is not a calibrated biological effect size. The table below reports the prespecified progressive endpoint-mean, state-average, lag-1/horizon-1 result on support-qualified eligible sources, alongside published SBTG on its eligible-source scope. Different source counts make close numerical comparisons descriptive.

\begin{center}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Network & Eligible & Supported & Mean AUROC & W1 AUROC & SBTG AUROC & Mean AUPRC & W1 AUPRC & SBTG AUPRC \\
\midrule
""" + "\n".join(bentley_lag1_rows) + r"""
\bottomrule
\end{tabular}
}
\end{center}

Tyramine is not evaluable for the progressive support-qualified lag-1 comparison because its sole Bentley-eligible source fails the saved support gate. Dopamine has two eligible sources; serotonin, tyramine, and octopamine each have one. These transmitter-specific results therefore have very limited source-level replication.

\subsection*{Interpretation by network}
\textbf{All monoamines.} Progressive lag-1 AUROC is 0.530 for endpoint mean and 0.515 for W1, below published SBTG at 0.577. The progressive endpoint-mean lag maximum is 0.591 at 2 s with native-grid global $q=0.0558$; on the common two-lag grid it has $q=0.0413$. This relies on four support-qualified eligible sources.

\textbf{Dopamine.} Progressive lag-1 AUROC is 0.531 for mean and 0.492 for W1 versus 0.567 for published SBTG. The strongest progressive dopamine profile is onset-minus-baseline mean, AUROC 0.621 at 1 s with global $q=0.0979$. With only two eligible sources, this is descriptive.

\textbf{Serotonin.} Progressive lag-1 AUROC is 0.534 for mean and 0.624 for W1 versus 0.663 for published SBTG. Progressive W1 reaches 0.700 at 2 s with native-grid global $q=0.0551$ and common-grid $q=0.0413$; published SBTG reaches 0.692 at its 5-s lag index with $q=0.0979$. Both rely on one eligible source.

\textbf{Tyramine.} The sole eligible source fails the progressive support gate, so progressive lag-1 and lag-max values are deliberately unavailable. Published SBTG has lag-1 AUROC 0.490 and best-lag AUROC 0.538 with global $q=0.9554$.

\textbf{Octopamine.} Progressive lag-1 endpoint-mean AUROC is 0.630 versus 0.495 for published SBTG. Progressive log-SD reaches 0.738 at 1 s, but global $q=0.0845$; endpoint mean reaches 0.642 at 2 s with $q=0.2717$. This network has one eligible source.

\textbf{Neuropeptides.} Progressive lag-1 AUROC is 0.538 for mean and 0.565 for W1 versus 0.523 for published SBTG. W1 is strongest at lag 1 (AUROC 0.567 after the common lag-support intersection), with permutation $p=0.001$ and global $q=0.0155$ across 40 support-qualified eligible sources.

\textbf{Neuromodulator union.} Progressive lag-1 AUROC is 0.534 for mean and 0.547 for W1 versus 0.525 for published SBTG. W1 is strongest at lag 1 (AUROC 0.550 after the native-grid support intersection), with $p=0.001$ and global $q=0.0155$ across 41 support-qualified eligible sources.

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{figures/bentley_specific_lag1.png}
\caption{Prespecified lag-1 Bentley correspondence by neuromodulator network.}
\end{figure}

\subsection*{Lag-max calibration}
Selecting the best lag inflates apparent correspondence. Each reported maximum was therefore compared with the maximum from 999 within-source target-label permutations; the resulting tests were adjusted across the full planned Bentley family by Benjamini--Hochberg. These lag indices are correspondence locations, not estimates of biological delay.

\begin{center}
\scriptsize
\begin{tabular}{lrrrrrr}
\toprule
Network & Sources & Positive & Best lag & AUROC & Raw $p$ & Global $q$ \\
\midrule
""" + "\n".join(bentley_lagmax_rows) + r"""
\bottomrule
\end{tabular}
\end{center}

The strongest globally adjusted planned Bentley result is the state-average endpoint-Wasserstein channel for the """ + NETWORK_LABELS[str(strongest.network)] + f" network: AUROC {strongest.best_auroc:.3f} at {strongest.best_lag_frames / 4:g} s, permutation $p={strongest.max_lag_permutation_p:.4f}$, and global $q={strongest.max_lag_bh_q:.4f}$." + r""" This is evidence of rank correspondence with the binary molecular reference under the frozen analysis, subject to the data-lineage and model limitations above.

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{figures/bentley_specific_lagmax.png}
\caption{Endpoint-mean best-lag Bentley correspondence with within-source permutation calibration and global adjustment.}
\end{figure}

\clearpage
\begin{landscape}
\section*{Complete calibrated Bentley lag-max family}
The following table contains every one of the 70 saved planned lag-max rows: five model/outcome profiles, seven networks, and two lag grids. The native grid uses each method's saved lag set; the common grid restricts both methods to lag indices 1 and 8. All finite permutation values were adjusted together as one 62-test family. ``Sources'' is the intersection of support-qualified eligible sources across the candidate lags for progressive rows; published SBTG has no comparable sampler-support gate. A dash means the profile was not evaluable.

\footnotesize
\renewcommand{\arraystretch}{0.92}
\begin{longtable}{lllrrrrrr}
\toprule
Grid & Network & Profile & Sources & Positive & Best lag & AUROC & Raw $p$ & Global $q$ \\
\midrule
\endfirsthead
\toprule
Grid & Network & Profile & Sources & Positive & Best lag & AUROC & Raw $p$ & Global $q$ \\
\midrule
\endhead
""" + "\n".join(full_lagmax_rows) + r"""
\bottomrule
\end{longtable}
\end{landscape}

\clearpage
\section*{Specific Bentley-positive neuron relationships}
The tables below show the ten highest-ranked Bentley-positive source-to-target relationships in each network using the absolute progressive endpoint-mean lag-1 score. Support is the valid-episode fraction for the source at lag 1. Mean intervals are the saved pointwise whole-trace bootstrap intervals. W1 is unsigned. ``Mean rank'' is the percentile of the pair's absolute progressive endpoint-mean score among all off-diagonal pairs from Bentley-eligible sources in that network. The complete 1,820-row inventory, including all lower-ranked relationships and recording-origin labels, is included as \texttt{bentley\_positive\_neuron\_relationships.csv}.

""" + "\n".join(top_pair_sections) + r"""

\clearpage
\section*{Effect uncertainty and head/tail origin}
Every dense effect cell has a pointwise 95\% interval from 256 whole-trace bootstrap resamples of the 20 historical traces. The bootstrap holds generator seed 1701 fixed and excludes generator-refit, head/tail pairing, donor-imputation, and multiplicity uncertainty.

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figures/head_tail.png}
\caption{Endpoint-mean effects summarized by recording origin with whole-trace bootstrap intervals. Origin labels do not repair the historical pseudo-pairing.}
\end{figure}

State-average source validity averaged 0.870; 292 of 320 source-lag rows met the 0.50 support gate. Sampler validity, effective sample size, and ancestry diagnostics characterize numerical support, not biological truth.

\section*{Reproducibility and included artifacts}
The portable directory includes this TeX source and compiled PDF, exact frozen atlas matrices, the observed cohort used by the viewer, the full Bentley metric and lag-max CSV files, an inspectable positive-edge table named \texttt{bentley\_positive\_neuron\_relationships.csv}, reference figures, checksum manifests, validation receipts, and the analysis/dashboard source snapshot. Each edge-table row gives the Bentley source and target neurons, recording origins, progressive lag-1 values and intervals, published SBTG score, source support, and within-network rank percentiles. It reproduces the viewer and report from frozen outputs. Training requires the separately held model checkpoints, original MAT inputs, and the complete research environment.

\end{document}
"""
    path = output / "main.tex"
    path.write_text(tex)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
