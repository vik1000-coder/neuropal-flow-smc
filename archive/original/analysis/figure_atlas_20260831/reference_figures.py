"""Static reference figures from frozen, audited E27/E28 results.

No new fitting, sampling, bootstrap, or inferential tests. Rounded values from
the saved audit are explicitly identified separately from exact saved CSV rows.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import BLUE, GREY, INK, ORANGE, OUT, ROOT, save_figure, sha256, style


EXTERNAL = ROOT / "results/neural_prediction_atlas_20260829/postfreeze_external"
AUDIT = EXTERNAL.parent / "POSTFREEZE_EXTERNAL_AUDIT.md"
HISTORICAL = ROOT / "results/sbtg80_progressive_sensitivity_20260830"
DATA = OUT / "data"
METHODS = ["progressive_bridge_smc", "direct_importance", "sbtg_published", "sbtg_current"]
METHOD_LABELS = {
    "progressive_bridge_smc": "Progressive bridge",
    "direct_importance": "Direct importance",
    "sbtg_published": "SBTG-published",
    "sbtg_current": "SBTG-current",
}
COLORS = dict(zip(METHODS, [BLUE, ORANGE, GREY, "#B0B6BC"]))
MARKERS = dict(zip(METHODS, ["o", "^", "s", "D"]))
REFS = ["randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54"]
REF_LABELS = dict(zip(REFS, ["Randi WT", "Cook structural", "Cook chemical", "Cook gap"]))
NETWORKS = ["monoamine_all", "monoamine_dopamine", "monoamine_serotonin",
            "monoamine_tyramine", "monoamine_octopamine", "neuropeptide_all",
            "neuromodulator_union"]
NETWORK_LABELS = dict(zip(NETWORKS, ["All monoamines", "Dopamine", "Serotonin",
                                      "Tyramine", "Octopamine", "Neuropeptides", "Union"]))


def verified_input(path: Path) -> dict:
    """Verify a particular input against its existing adjacent frozen ledger."""
    ledger = None
    expected = None
    for candidate in [path.parent / "checksums.sha256", path.parent / "analysis_checksums.sha256"]:
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            if (candidate.parent / relative).resolve() == path.resolve():
                expected = digest
                ledger = candidate
                break
        if expected is not None:
            break
    actual = sha256(path)
    if expected is not None and expected != actual:
        raise ValueError(f"Frozen checksum mismatch: {path}")
    return {"path": str(path.relative_to(ROOT)), "sha256": actual,
            "existing_ledger": str(ledger.relative_to(ROOT)) if expected else None,
            "verified_against_existing_ledger": expected is not None}


def markdown_table(text: str, header: str) -> list[list[str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line == header)
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip().replace("**", "") for cell in line.strip("|").split("|")])
    return rows


def save_rows(frame: pd.DataFrame, name: str) -> str:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    frame.to_csv(path, index=False)
    return str(path.relative_to(ROOT))


def title(fig, number: str, heading: str, subtitle: str):
    fig.text(0.055, 0.97, f"{number}  {heading}", fontsize=18, weight="semibold", va="top")
    fig.text(0.055, 0.924, subtitle, fontsize=11, va="top", color=INK)


def footer(fig, lines: list[str]):
    for i, line in enumerate(reversed(lines)):
        fig.text(0.055, 0.035 + i * 0.026, line, fontsize=10, va="bottom", color=INK)


def method_legend(fig, methods=METHODS, y=0.855):
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle="none",
                      markersize=7, label=METHOD_LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, y),
               ncol=len(methods), frameon=False, columnspacing=2.0)


def compare_points(ax, data, metric, references, labels, methods=METHODS):
    offsets = np.linspace(-0.24, 0.24, len(methods))
    for mi, method in enumerate(methods):
        frame = data[data.method == method].set_index("reference").loc[references]
        ax.plot(frame[metric], np.arange(len(references)) + offsets[mi], linestyle="none",
                marker=MARKERS[method], color=COLORS[method], markersize=7,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_yticks(np.arange(len(references)), labels)
    ax.set_ylim(len(references) - 0.45, -0.6)
    ax.grid(axis="x", alpha=0.9)
    ax.tick_params(axis="y", length=0, pad=8)


def select_references() -> pd.DataFrame:
    frame = pd.read_csv(EXTERNAL / "randi_cook_metrics.csv")
    flow = frame.channel.eq("endpoint_mean") & frame.context.eq("state_average") & frame.horizon_frames.eq(1)
    sbtg = frame.method.isin(["sbtg_published", "sbtg_current"]) & frame.channel.eq("score_product")
    selected = frame[frame.scope.eq("all_estimated") & frame.lag_frames.eq(1) & (flow | sbtg)].copy()
    assert len(selected) == 16
    assert not selected.duplicated(["method", "reference"]).any()
    assert selected.groupby("reference").n_edges.nunique().eq(1).all()
    assert selected.groupby("reference").n_positive.nunique().eq(1).all()
    assert selected[selected.reference.eq("randi_wild_type")].n_edges.eq(1349).all()
    assert selected[~selected.reference.eq("randi_wild_type")].n_edges.eq(2862).all()
    selected["figure_selection"] = "all_estimated; L1; flow endpoint_mean/state_average/H1; SBTG score_product"
    return selected


def audit_bootstrap(reference: pd.DataFrame) -> pd.DataFrame:
    rows = markdown_table(AUDIT.read_text(), "| Reference | Observed difference | Exploratory 95% interval |")
    values = []
    reverse = {v: k for k, v in REF_LABELS.items()}
    for label, delta, interval in rows:
        low, high = [float(x.strip()) for x in interval.strip("[]").split(",")]
        ref = reverse[label]
        x = reference[reference.reference.eq(ref)].set_index("method")
        actual = x.loc["progressive_bridge_smc", "auroc"] - x.loc["sbtg_published", "auroc"]
        assert abs(actual - float(delta)) < 0.0006
        values.append(dict(reference=ref, progressive_minus_published=actual,
                           reported_delta=float(delta), reported_ci_low=low, reported_ci_high=high,
                           interval_precision="reported audit bounds rounded to 3 decimals",
                           bootstrap_replicates=500, bootstrap_unit="source column",
                           source=str(AUDIT.relative_to(ROOT)),
                           inference_limit="post-hoc pointwise source sensitivity; not adjusted; no animal/model-refit uncertainty"))
    assert len(values) == 4
    return pd.DataFrame(values)


def figure03(reference: pd.DataFrame) -> dict:
    interval = audit_bootstrap(reference)
    files = [save_rows(reference, "f03_fixed_lag_reference_rows.csv"),
             save_rows(interval, "f03_reported_source_bootstrap_rows.csv")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.18, 1.1, 1.14]})
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.22, top=0.76, wspace=0.28)
    title(fig, "F03", "Cook and Randi reference correspondence",
          "Frozen 54-class matrices · source lag 1 · flow state-average mean at horizon 1 · same reference masks")
    method_legend(fig)
    denom = reference[reference.method.eq("progressive_bridge_smc")].set_index("reference")
    labels = [f"{REF_LABELS[r]}\n{int(denom.loc[r, 'n_positive']):,} / {int(denom.loc[r, 'n_edges']):,} positive" for r in REFS]
    compare_points(axes[0], reference, "auroc", REFS, labels)
    axes[0].set_title("Edge-presence ranking", loc="left", pad=14)
    axes[0].set_xlim(0.45, 0.72)
    axes[0].set_xticks([0.45, 0.5, 0.6, 0.7])
    axes[0].axvline(0.5, color=INK, linestyle=":", linewidth=1.2)
    axes[0].set_xlabel("AUROC · chance = 0.5")
    compare_points(axes[1], reference, "auprc", REFS, [""] * 4)
    axes[1].set_title("Positive-pair retrieval", loc="left", pad=14)
    axes[1].set_xlim(0, 0.43)
    axes[1].set_xlabel("AUPRC · baseline = prevalence")
    for i, r in enumerate(REFS):
        prev = float(denom.loc[r, "prevalence"])
        axes[1].plot([prev, prev], [i - 0.38, i + 0.38], color=INK, linestyle=":", linewidth=1.2)
    iv = interval.set_index("reference").loc[REFS]
    delta = iv.progressive_minus_published.to_numpy()
    axes[2].errorbar(delta, np.arange(4),
                     xerr=np.array([delta - iv.reported_ci_low, iv.reported_ci_high - delta]),
                     fmt="o", color=BLUE, capsize=4, markersize=6)
    axes[2].axvline(0, color=INK, linestyle=":", linewidth=1.2)
    axes[2].set_yticks(np.arange(4), [""] * 4)
    axes[2].set_ylim(3.55, -0.6)
    axes[2].set_xlim(-0.06, 0.12)
    axes[2].set_xticks([-0.05, 0, 0.05, 0.10])
    axes[2].set_title("Progressive − published SBTG", loc="left", pad=14)
    axes[2].set_xlabel("AUROC difference · reported 95% CI")
    axes[2].grid(axis="x")
    for i, r in enumerate(REFS):
        rr = iv.loc[r]
        axes[2].text(-0.055, i + 0.37,
                      f"{rr.reported_delta:+.3f}  [{rr.reported_ci_low:+.3f}, {rr.reported_ci_high:+.3f}]",
                      fontsize=10, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    footer(fig, [
        "Dots are descriptive scores, not corrected superiority tests. CI: 500 source-column bootstrap draws; audit-rounded bounds.",
        "Randi excludes ambiguous pairs. Cook uses all off-diagonal pairs. Published SBTG keeps its historical 80-class training lineage.",
        "Flow lag 1 + horizon 1 is 0.50 s source-window-end to readout; SBTG lag 1 is a different estimand. No physical-delay claim.",
    ])
    files += save_figure(fig, "f03_cook_randi_correspondence")
    return dict(id="f03", files=files, plotted_rows=16, interval_rows=4,
                question="How do the frozen methods rank functional and anatomical pairs on equal masks?",
                limits="Descriptive ranking; source-bootstrap intervals are post-hoc, pointwise, audit-rounded, not refit uncertainty.")


def audit_profiles(lagmax: pd.DataFrame) -> pd.DataFrame:
    table = markdown_table(AUDIT.read_text(), "| Method/network | L1 | L4 | L8 | L16 |")
    rows = []
    for label, *values in table:
        method = "progressive_bridge_smc" if label.startswith("Progressive") else "direct_importance"
        network = "neuropeptide_all" if label.endswith("neuropeptide") else "neuromodulator_union"
        q = lagmax[(lagmax.method == method) & (lagmax.channel == "endpoint_wasserstein1") &
                   (lagmax.context == "state_average") & (lagmax.network == network) &
                   (lagmax.lag_grid == "native_method_grid")].iloc[0]
        n = int(q.n_common_supported_eligible_sources)
        for lag, val in zip([1, 4, 8, 16], values):
            rows.append(dict(method=method, channel="endpoint_wasserstein1", network=network,
                             lag_frames=lag, source_to_cut_seconds=lag / 4, horizon_frames=1,
                             context="state_average", auroc=float(val), n_sources=n,
                             n_edges=n * 53, n_positive=int(q.n_positive),
                             support_scope="same source intersection across all four lags",
                             value_precision="saved audit curve, rounded to 3 decimals",
                             source=str(AUDIT.relative_to(ROOT))))
    out = pd.DataFrame(rows)
    assert len(out) == 16
    assert out.groupby(["method", "network"]).n_edges.nunique().eq(1).all()
    return out


def figure04() -> dict:
    lagmax = pd.read_csv(EXTERNAL / "neuromodulator_lagmax_inference.csv")
    profiles = audit_profiles(lagmax)
    neuromod = pd.read_csv(EXTERNAL / "neuromodulator_metrics.csv")
    sbtg = neuromod[(neuromod.method == "sbtg_published") & (neuromod.scope == "eligible_sources") &
                    neuromod.network.isin(["neuropeptide_all", "neuromodulator_union"]) &
                    neuromod.lag_frames.isin([1, 8])].copy()
    assert len(sbtg) == 4
    selected = lagmax[(lagmax.method == "progressive_bridge_smc") &
                      (lagmax.context == "state_average") &
                      lagmax.channel.isin(["endpoint_mean", "endpoint_wasserstein1"]) &
                      (lagmax.lag_grid == "native_method_grid")].copy()
    selected["n_tested_pairs"] = selected.n_common_supported_eligible_sources * 53
    assert len(selected) == 14
    files = [save_rows(profiles, "f04_common_source_w1_profiles_audit_rounded.csv"),
             save_rows(sbtg, "f04_sbtg_common_grid_profile_rows.csv"),
             save_rows(selected, "f04_progressive_network_lagmax_rows.csv")]
    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], left=0.08, right=0.96,
                           top=0.835, bottom=0.145, hspace=0.46, wspace=0.20)
    title(fig, "F04", "Bentley reference alignment across source lags",
          "State-average responses · one-frame forecast · four source lags · receptor/pathway existence, not active signaling")
    for ni, network in enumerate(["neuropeptide_all", "neuromodulator_union"]):
        ax = fig.add_subplot(grid[0, ni])
        for method in ["progressive_bridge_smc", "direct_importance"]:
            data = profiles[(profiles.method == method) & (profiles.network == network)].sort_values("lag_frames")
            ns = int(data.n_sources.iloc[0])
            ax.plot(data.source_to_cut_seconds, data.auroc, color=COLORS[method], marker=MARKERS[method],
                    label=f"{METHOD_LABELS[method]} W1 · {ns} sources")
        data = sbtg[sbtg.network.eq(network)].sort_values("lag_frames")
        ns = int(data.n_eligible_sources.iloc[0])
        ax.plot(data.lag_index_seconds, data.auroc, color=GREY, marker="s", linestyle="--",
                label=f"SBTG-published score · {ns} sources · L1/L8")
        ax.axhline(0.5, color=INK, linestyle=":", linewidth=1.2)
        ax.set_xticks([0.25, 1, 2, 4], ["0.25\nL1", "1\nL4", "2\nL8", "4\nL16"])
        ax.set_xlim(0.1, 4.15)
        ax.set_ylim(0.35, 0.62)
        ax.set_yticks([0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
        ax.set_ylabel("AUROC · chance = 0.5")
        ax.set_xlabel("Nominal lag seconds / index\n(flow source-to-cut; SBTG score lag)")
        ax.set_title(NETWORK_LABELS[network], loc="left", pad=12)
        ax.grid(axis="y")
        ax.legend(loc="lower right", fontsize=10, frameon=False)
    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis("off")
    table_ax.set_title("Progressive mean and W1: saved maximum-over-lags tests", loc="left", pad=18)
    cell_rows = []
    for network in NETWORKS:
        pair = selected[selected.network.eq(network)].set_index("channel")
        n = int(pair.loc["endpoint_mean", "n_common_supported_eligible_sources"])
        positive = int(pair.loc["endpoint_mean", "n_positive"])
        values = [NETWORK_LABELS[network], f"{n} / {n * 53:,} / {positive}"]
        for channel in ["endpoint_mean", "endpoint_wasserstein1"]:
            row = pair.loc[channel]
            if not bool(row.evaluable):
                values += ["Not evaluable", "—", "—"]
            else:
                values += [f"{row.best_auroc:.3f} @ L{int(row.best_lag_frames)}",
                           f"{row.max_lag_permutation_p:.3f}", f"{row.max_lag_bh_q:.3f}"]
        cell_rows.append(values)
    headers = ["Reference", "Supported sources\n/ pairs / positives", "Mean: max AUROC", "p", "q",
               "W1: max AUROC", "p", "q"]
    table = table_ax.table(cellText=cell_rows, colLabels=headers, cellLoc="center", loc="upper left",
                           colWidths=[0.18, 0.17, 0.16, 0.07, 0.07, 0.16, 0.07, 0.07], bbox=[0, 0.02, 1, 0.95])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E0E5E9")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#F1F4F6")
            cell.set_text_props(weight="semibold", color=INK)
        elif col == 0:
            cell.set_text_props(ha="left")
        if row in [6, 7] and col in [5, 6, 7]:
            cell.set_text_props(color=BLUE, weight="semibold")
    footer(fig, [
        "Curves: fixed source masks within each method; masks differ between methods. W1 profiles are audit-rounded; SBTG points are exact saved scores.",
        "Table: 999 source-preserving target-label permutations; global BH across 110 evaluable panels. Tyramine has no common-supported source.",
        "Post-hoc progressive-neuropeptide L1–L4 interval includes zero. No named monoamine passes BH; a max-lag test does not identify a unique delay.",
    ])
    files += save_figure(fig, "f04_bentley_lag_alignment")
    return dict(id="f04", files=files, curve_rows=20, inferential_rows=14,
                question="Do distributional distances align at particular source-history lags, and which network tests survive correction?",
                limits="Flow curves are rounded audit values on fixed source intersections; support masks differ by method; no curve CI exists here; no unique delay inference.")


def figure05(reference: pd.DataFrame) -> dict:
    selected = reference[reference.method.isin(METHODS[:3])].copy()
    historical = pd.read_csv(HISTORICAL / "paired_source_bootstrap.csv")
    historical = historical[historical.metric.eq("auroc")].copy()
    assert len(historical) == 4
    hist_refs = pd.read_csv(HISTORICAL / "randi_cook_metrics.csv")
    flow = hist_refs.method.eq("progressive_bridge_smc") & hist_refs.channel.eq("endpoint_mean") & hist_refs.context.eq("state_average") & hist_refs.horizon_frames.eq(1)
    sbtg = hist_refs.method.eq("sbtg_published") & hist_refs.channel.eq("score_product")
    hist_refs = hist_refs[hist_refs.scope.eq("all_estimated") & hist_refs.lag_frames.eq(1) & (flow | sbtg)].copy()
    assert len(hist_refs) == 8
    assert hist_refs.groupby("reference").n_edges.nunique().eq(1).all()
    files = [save_rows(selected, "f05_continuous_reference_rows.csv"),
             save_rows(historical, "f05_historical80_saved_bootstrap_rows.csv"),
             save_rows(hist_refs, "f05_historical80_reference_rows.csv")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.08, 0.9, 1.3]})
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.76, wspace=0.45)
    title(fig, "F05", "Continuous correspondence and historical-cohort sensitivity",
          "Left / middle: 54-class L1 flow mean effects and SBTG scores · right: historical80, a different data lineage")
    method_legend(fig, METHODS[:3])
    compare_points(axes[0], selected, "absolute_spearman", REFS,
                   [REF_LABELS[x] for x in REFS], METHODS[:3])
    axes[0].set_title("All evaluated pairs", loc="left", pad=14)
    axes[0].set_xlim(-0.02, 0.30)
    axes[0].set_xticks([0, 0.1, 0.2, 0.3])
    axes[0].set_xlabel("Absolute-effect Spearman ρ")
    axes[0].axvline(0, color=INK, linestyle=":", linewidth=1.2)
    compare_points(axes[1], selected, "positive_edge_absolute_spearman", REFS[1:],
                   ["Structural", "Chemical", "Gap"], METHODS[:3])
    axes[1].set_title("Cook: positive edges only", loc="left", pad=14)
    axes[1].set_xlim(-0.02, 0.30)
    axes[1].set_xticks([0, 0.1, 0.2, 0.3])
    axes[1].set_xlabel("Absolute-effect Spearman ρ")
    axes[1].axvline(0, color=INK, linestyle=":", linewidth=1.2)
    order = ["randi_wild_type", "cook_struct_80", "cook_chem_80", "cook_gap_80"]
    data = historical.set_index("reference").loc[order]
    hist_points = hist_refs[hist_refs.method.eq("progressive_bridge_smc")].set_index("reference")
    published = hist_refs[hist_refs.method.eq("sbtg_published")].set_index("reference")
    delta = data.progressive_minus_sbtg_published.to_numpy()
    assert np.allclose(delta, (hist_points.auroc - published.auroc).loc[order], atol=1e-12)
    axes[2].errorbar(delta, np.arange(4), xerr=[delta - data.ci_low, data.ci_high - delta],
                     fmt="o", color=BLUE, capsize=4, markersize=6)
    axes[2].axvline(0, color=INK, linestyle=":", linewidth=1.2)
    axes[2].set_yticks(np.arange(4), [f"{label}\n{int(hist_points.loc[r, 'n_positive']):,} / {int(hist_points.loc[r, 'n_edges']):,} positive"
                                      for r, label in zip(order, ["Randi WT", "Cook structural", "Cook chemical", "Cook gap"])])
    axes[2].set_ylim(3.6, -0.6)
    axes[2].set_xlim(-0.18, 0.02)
    axes[2].set_xticks([-0.15, -0.10, -0.05, 0])
    axes[2].grid(axis="x")
    axes[2].set_title("Historical80: progressive − published", loc="left", pad=14)
    axes[2].set_xlabel("AUROC difference · saved 95% source CI")
    for i, (_, row) in enumerate(data.iterrows()):
        axes[2].text(-0.174, i + 0.37,
                     f"{row.progressive_minus_sbtg_published:+.3f}  [{row.ci_low:+.3f}, {row.ci_high:+.3f}]",
                     fontsize=10, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    footer(fig, [
        "Continuous correlations are descriptive, without displayed CIs. All-edge Cook correlations partly capture zero-versus-positive edge existence.",
        "Historical80 intervals: 10,000 saved source-column bootstrap draws, not animal/model-refit uncertainty and not multiplicity-adjusted tests.",
        "The 80-class cache uses donor traces and index-wise head/tail pseudo-pairing. Differences cannot be attributed to neuron count alone.",
    ])
    files += save_figure(fig, "f05_weight_and_cohort_sensitivity")
    return dict(id="f05", files=files, correlation_rows=12, interval_rows=4,
                question="Does coarse edge ranking extend to continuous weights, and does the 54-class advantage persist historically?",
                limits="Point correlations lack intervals; 80-class sensitivity has a distinct imputed/pseudo-paired lineage; saved source CIs omit refitting and are unadjusted.")


def main():
    style()
    inputs = [EXTERNAL / "randi_cook_metrics.csv", EXTERNAL / "neuromodulator_metrics.csv",
              EXTERNAL / "neuromodulator_lagmax_inference.csv", AUDIT,
              HISTORICAL / "paired_source_bootstrap.csv", HISTORICAL / "randi_cook_metrics.csv",
              Path(__file__), Path(__file__).with_name("common.py")]
    inventory = [verified_input(path) for path in inputs]
    reference = select_references()
    figures = [figure03(reference), figure04(), figure05(reference)]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="generated_pending_visual_review",
                    no_new_training_sampling_or_inference=True, input_inventory=inventory,
                    palette={"progressive": BLUE, "direct": ORANGE, "sbtg": "neutral greys"},
                    minimum_font_points=10, figures=figures,
                    accuracy_checks=["16 unique fixed-lag method/reference rows on identical denominators",
                                     "4 audit-rounded 54-class source-bootstrap intervals agree with exact score differences",
                                     "16 W1 audit profile points use fixed per-method network support across lags",
                                     "4 exact saved SBTG common-grid reference points",
                                     "14 progressive network/channel lag-max rows, including explicit non-evaluable tyramine",
                                     "4 historical80 AUROC intervals taken verbatim from saved CSV"])
    manifest["output_sha256"] = {f: sha256(ROOT / f) for item in figures for f in item["files"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": [x["id"] for x in figures], "files": sum(len(x["files"]) for x in figures)}, indent=2))


if __name__ == "__main__":
    main()
