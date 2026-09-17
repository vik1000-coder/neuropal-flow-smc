"""Clarity-first per-pathway figures from saved, frozen lag correspondence.

Only extraction, deterministic checks, and rendering occur here. No model
samples, scores, resampling distributions, p-values, or q-values are estimated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ROOT, OUT, OLD, BLUE, GREY, new_figure, finish,
                    relative, sha256, verify_old_inputs, write_manifest)


EXTERNAL = ROOT / "results/neural_prediction_atlas_20260829/postfreeze_external"
AUDIT = EXTERNAL.parent / "POSTFREEZE_EXTERNAL_AUDIT.md"
ORIGINAL_CODE = ROOT / "compatibility_neural_benchmark/prediction_atlas_external_analysis.py"
PROFILE = OLD / "data/f04_common_source_w1_profiles_audit_rounded.csv"
STATS = OLD / "data/f04_progressive_network_lagmax_rows.csv"
METHOD = "progressive_bridge_smc"
CHANNEL = "endpoint_wasserstein1"
LAGS = [1, 4, 8, 16]
NETWORKS = [
    ("neuropeptide_all", "neuropeptides", "Neuropeptides"),
    ("monoamine_all", "monoamines", "All monoamines"),
    ("monoamine_dopamine", "dopamine", "Dopamine"),
    ("monoamine_serotonin", "serotonin", "Serotonin"),
    ("monoamine_tyramine", "tyramine", "Tyramine"),
    ("monoamine_octopamine", "octopamine", "Octopamine"),
    ("neuromodulator_union", "union", "Monoamine / neuropeptide union"),
]


def verified_native(path: Path, ledger: Path) -> dict[str, str]:
    entries = {}
    for line in ledger.read_text().splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            entries[(ledger.parent / name).resolve()] = digest
    actual = sha256(path)
    assert entries[path.resolve()] == actual, f"Native ledger mismatch: {path}"
    return {relative(path): actual, relative(ledger): sha256(ledger)}


def inputs() -> dict[str, str]:
    values = verify_old_inputs([PROFILE, STATS, OLD / "REFERENCE_CAPTIONS.md",
                                OLD / "reference_manifest.json"])
    for path in [EXTERNAL / "neuromodulator_metrics.csv",
                 EXTERNAL / "neuromodulator_lagmax_inference.csv"]:
        values.update(verified_native(path, EXTERNAL / "checksums.sha256"))
    for path in [AUDIT, ORIGINAL_CODE]:
        values.update(verified_native(path, AUDIT.parent / "checksums.sha256"))
    return values


def saved_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    stats = pd.read_csv(STATS)
    stats = stats[(stats.method == METHOD) & (stats.context == "state_average") &
                  (stats.horizon_frames == 1) & (stats.lag_grid == "native_method_grid")]
    mean_stats = stats[stats.channel == "endpoint_mean"].set_index("network")
    stats = stats[stats.channel == CHANNEL].set_index("network")
    assert len(stats) == 7 and stats.index.is_unique
    assert stats.candidate_lags_frames.eq("1,4,8,16").all()
    assert stats.support_scope.eq("intersection_across_all_candidate_lags").all()
    assert stats.n_bh_tests.eq(110).all() and stats.support_threshold.eq(0.5).all()
    original_stats = pd.read_csv(EXTERNAL / "neuromodulator_lagmax_inference.csv")
    original_stats = original_stats[(original_stats.method == METHOD) &
                                    (original_stats.context == "state_average") &
                                    (original_stats.horizon_frames == 1) &
                                    (original_stats.channel == CHANNEL) &
                                    (original_stats.lag_grid == "native_method_grid")].set_index("network")
    for field in ["best_auroc", "best_lag_frames", "max_lag_permutation_p", "max_lag_bh_q",
                  "n_common_supported_eligible_sources", "n_positive"]:
        np.testing.assert_allclose(stats[field], original_stats.loc[stats.index, field],
                                   rtol=0, atol=1e-14, equal_nan=True)

    rounded = pd.read_csv(PROFILE)
    rounded = rounded[(rounded.method == METHOD) & (rounded.channel == CHANNEL) &
                      (rounded.context == "state_average") & (rounded.horizon_frames == 1)]
    exact = pd.read_csv(EXTERNAL / "neuromodulator_metrics.csv")
    exact = exact[(exact.method == METHOD) & (exact.channel == CHANNEL) &
                  (exact.context == "state_average") & (exact.horizon_frames == 1) &
                  (exact.scope == "eligible_support_qualified")]
    assert len(exact) == 28
    groups = {}
    for network, _, label in NETWORKS:
        stat = stats.loc[network]
        n_sources = int(stat.n_common_supported_eligible_sources)
        n_pairs = n_sources * 53
        n_positive = int(stat.n_positive)
        if network in ["neuropeptide_all", "neuromodulator_union"]:
            rows = rounded[rounded.network == network].sort_values("lag_frames").copy()
            assert rows.n_sources.eq(n_sources).all()
            assert rows.n_edges.eq(n_pairs).all() and rows.n_positive.eq(n_positive).all()
            precision = "saved audit common-source curve, rounded to three decimals"
            source = relative(PROFILE)
            mask_proof = "explicit saved all-lag source intersection in frozen F04 export"
        else:
            rows = exact[exact.network == network].sort_values("lag_frames").copy()
            # The original code's common set is a subset of each per-lag set.
            # Equal cardinality at every lag therefore proves the per-lag sets
            # equal that intersection, not merely that row counts coincide.
            assert rows.n_supported_eligible_sources.eq(n_sources).all()
            assert rows.n_edges.eq(n_pairs).all() and rows.n_positive.eq(n_positive).all()
            precision = "exact saved CSV value"
            source = relative(EXTERNAL / "neuromodulator_metrics.csv")
            mask_proof = "each per-lag supported eligible set has the same size as its saved all-lag intersection; the code guarantees subset inclusion"
        assert rows.lag_frames.tolist() == LAGS
        np.testing.assert_allclose(rows.source_to_cut_seconds, np.array(LAGS) / 4, rtol=0, atol=0)
        evaluable = bool(stat.evaluable)
        if evaluable:
            assert np.isfinite(rows.auroc).all()
            tolerance = 0.00051 if network in ["neuropeptide_all", "neuromodulator_union"] else 1e-14
            assert abs(float(rows.auroc.max()) - float(stat.best_auroc)) < tolerance
            assert int(rows.loc[rows.auroc.idxmax(), "lag_frames"]) == int(stat.best_lag_frames)
        else:
            assert network == "monoamine_tyramine" and n_sources == n_pairs == n_positive == 0
            assert rows.auroc.isna().all() and pd.isna(stat.max_lag_bh_q)
        out = pd.DataFrame({
            "network": network, "network_label": label, "method": METHOD,
            "channel": CHANNEL, "context": "state_average", "lag_frames": LAGS,
            "source_to_cut_seconds": np.array(LAGS) / 4,
            "horizon_frames": 1, "forecast_horizon_seconds": 0.25,
            "source_to_readout_seconds": np.array(LAGS) / 4 + 0.25,
            "auroc": rows.auroc.to_numpy(), "evaluable": evaluable,
            "n_reference_eligible_sources": int(stat.n_reference_eligible_sources),
            "n_common_supported_sources": n_sources, "n_tested_pairs": n_pairs,
            "n_positive_pairs": n_positive, "support_threshold": 0.5,
            "support_scope": "same eligible-source intersection across all four candidate lags",
            "fixed_mask_verification": mask_proof, "curve_value_precision": precision,
            "max_lag_permutation_p": stat.max_lag_permutation_p,
            "global_bh_q": stat.max_lag_bh_q, "n_global_bh_tests": 110,
            "inference_scope": "alignment at any searched lag; not a between-lag or physical-delay test",
            "curve_source": source, "inference_source": relative(STATS),
        })
        groups[network] = out
    return groups, stats, mean_stats


def notes(network: str, stat: pd.Series) -> list[str]:
    if network == "monoamine_tyramine":
        return ["No score or q-value is plotted: missing support is not a zero effect.",
                "The reference contains 1 eligible source and 27 positive pairs."]
    q = float(stat.max_lag_bh_q)
    if network == "neuropeptide_all":
        return [f"Lag-search q = {q:.3f}: corrected evidence of alignment.",
                "Selected pathway; early lags are not clearly separated. This is not a transmission delay."]
    if network == "neuromodulator_union":
        return [f"Lag-search q = {q:.3f}: corrected evidence of alignment.",
                "Overlaps the peptide network; not an independent confirmation or an identified delay."]
    n = int(stat.n_common_supported_eligible_sources)
    qualifier = (f"Only {n} source{'s are' if n != 1 else ' is'} represented; lack of evidence does not rule out signaling."
                 if n <= 2 else "A missing correspondence signal does not establish absence of monoamine signaling.")
    return [f"No corrected alignment evidence (lag-search q = {q:.3f}).", qualifier]


def plot_one(network: str, short: str, label: str, rows: pd.DataFrame, stat: pd.Series) -> dict:
    n, pairs = int(stat.n_common_supported_eligible_sources), int(stat.n_common_supported_eligible_sources) * 53
    title = f"{label}: reference alignment across source lags"
    if network == "neuromodulator_union":
        title = "Combined neuromodulators: alignment across source lags"
    subtitle = f"Progressive SMC · W1 · readout 0.25 s after cut · {n} supported source{'s' if n != 1 else ''} / {pairs:,} pairs"
    fig, ax = new_figure(title, subtitle)
    if bool(stat.evaluable):
        ax.plot(rows.source_to_cut_seconds, rows.auroc, color=BLUE, marker="o", markersize=8)
        ax.set_xlim(0.05, 4.2)
        ax.set_ylim(0.35, 0.65)
        ax.set_xticks([0.25, 1, 2, 4], ["0.25", "1", "2", "4"])
        ax.set_yticks([0.4, 0.5, 0.6], ["0.40", "0.50", "0.60"])
        ax.set_xlabel("Source lag before the prediction cut (seconds)", labelpad=14)
        ax.set_ylabel("Reference alignment (AUROC)", labelpad=14)
        ax.grid(axis="y", alpha=0.7)
        ax.axhline(0.5, color=GREY, linestyle=(0, (3, 3)), linewidth=1.5)
        above_chance = bool(rows.auroc.gt(0.5).all())
        ax.text(4.15, 0.495 if above_chance else 0.505, "Chance = 0.50",
                fontsize=12, color=GREY, ha="right", va="top" if above_chance else "bottom")
    else:
        ax.axis("off")
        ax.text(0.5, 0.66, "Not evaluable", transform=ax.transAxes,
                ha="center", va="center", fontsize=26, color=GREY)
        ax.text(0.5, 0.40, "Tyramine is present in the reference.\nIts source is unsupported at all four lags.",
                transform=ax.transAxes, ha="center", va="center", fontsize=18, linespacing=1.7)
    result = finish(fig, f"lag_{short}", notes(network, stat))
    result.update(network=network, question="How does this pathway's reference correspondence change with source lag?",
                  n_supported_sources=n, n_tested_pairs=pairs,
                  row_count=len(rows), evaluable=bool(stat.evaluable))
    return result


def caption(network: str, short: str, label: str, rows: pd.DataFrame, stat: pd.Series) -> Path:
    n = int(stat.n_common_supported_eligible_sources)
    p = int(stat.n_positive)
    if bool(stat.evaluable):
        q = float(stat.max_lag_bh_q)
        result = (f"The saved lag-search test {'passes' if q < 0.05 else 'does not pass'} global correction (q={q:.6f}). "
                  "This tests alignment at any searched lag, not differences between lags or a transmission delay.")
    else:
        result = "No curve or test statistic is estimable: the sole reference source fails support at every tested lag. Tyramine is not absent from the reference, and this is not a zero-effect result."
    specific = {
        "neuropeptide_all": "This is the selected main example from the full network search, not an independently selected validation pathway. The existing post-hoc progressive-neuropeptide L1-minus-L4 interval crosses zero; an early band is more defensible than a unique peak.",
        "neuromodulator_union": "This union overlaps the peptide network and is not an independent replication. Its existing source-bootstrap interval for alignment crosses chance ([0.468, 0.597]), although the different target-label permutation test passes. Neither summarizes animal/model-refit uncertainty completely.",
        "monoamine_dopamine": "Only two supported source classes contribute. A lack of corrected correspondence evidence does not rule out dopamine signaling or establish an inverted biological effect.",
        "monoamine_serotonin": "Only one supported source class contributes. The rising descriptive curve does not establish a preferred lag or rule out serotonin signaling when its alignment test fails correction.",
        "monoamine_octopamine": "Only one supported source class contributes. A below-chance descriptive AUROC is not evidence for a reversed biological mechanism; there is no corrected positive-alignment result.",
        "monoamine_all": "This aggregate overlaps the named monoamine networks; these are not independent replications. No named monoamine has corrected alignment evidence in this W1 analysis.",
        "monoamine_tyramine": "The full reference has one eligible source, 53 off-diagonal potential pairs, and 27 positive pairs. The support-qualified intersection has zero pairs; the CSV retains missing scores as missing, not as zeros.",
    }[network]
    content = f"""# {label}: lag correspondence

Progressive-bridge SMC, state-average endpoint distributional effects (W1), forecast horizon 1 frame (0.25 seconds). The curve asks whether larger predicted distribution shifts occur on Bentley receptor/pathway-compatible neuron pairs. {result}

## Reading this figure

- The vertical axis is the AUROC connection-ranking score, not the W1 distance itself. W1 is an unsigned distance between endpoint distributions and is not specific to variance or mean-independent shape.
- The same {n} supported source class{'es' if n != 1 else ''} {'contribute' if n != 1 else 'contributes'} at every lag: {n * 53:,} tested off-diagonal pairs, {p} reference positives. Support requires valid fraction ≥0.5; source sets are intersected across the full lag grid.
- Source lag is source-window-end to prediction cut: 0.25, 1, 2 and 4 seconds (1, 4, 8 and 16 frames). The target readout is a further 0.25 seconds after the cut. Line segments join tested settings; intermediate dynamics are not estimated.

{specific}

## Statistical and provenance limits

The original analysis used 999 within-source target-label permutations, shared across lags, to calibrate the maximum AUROC, followed by global BH correction across 110 evaluable planned tests. These are frozen-matrix reference-label tests: source degree is preserved, target degree is not. They do not quantify model-refit, reference, or animal-sampling uncertainty, and no pointwise curve intervals are available here. Bentley describes molecular compatibility, not activity, receptor occupancy, or active transmission.

Curve precision: {rows.curve_value_precision.iloc[0]}. No new scores, samples, p-values or q-values were computed for this figure. The monoamine exact rows are provably on the all-lag intersection because every per-lag supported set has the same cardinality as the saved common subset; peptide/union curves instead use the explicitly common-source audit export.

[PNG](../figures/lag_{short}.png) · [SVG](../figures/lag_{short}.svg) · [Plotted rows](../data/lag_{short}.csv) · [All pathway evidence](lag_evidence_summary.md) · [Manifest](../lag_manifest.json)

Sources: [saved lag-max tests](../../neural_prediction_atlas_20260829/postfreeze_external/neuromodulator_lagmax_inference.csv), [exact per-lag metrics](../../neural_prediction_atlas_20260829/postfreeze_external/neuromodulator_metrics.csv), [common-source audit](../../neural_prediction_atlas_20260829/POSTFREEZE_EXTERNAL_AUDIT.md), [earlier figure evidence](../../figure_atlas_20260831/REFERENCE_CAPTIONS.md).
"""
    path = OUT / "captions" / f"lag_{short}.md"
    path.write_text(content)
    return path


def evidence_summary(stats: pd.DataFrame, mean_stats: pd.DataFrame) -> list[Path]:
    values, table = [], []
    for network, short, label in NETWORKS:
        row = stats.loc[network]
        n = int(row.n_common_supported_eligible_sources)
        q = float(row.max_lag_bh_q)
        outcome = ("Not evaluable" if not bool(row.evaluable) else
                   "Corrected alignment evidence" if q < 0.05 else "No corrected alignment evidence")
        values.append(dict(network=network, label=label, n_reference_sources=int(row.n_reference_eligible_sources),
                           n_common_supported_sources=n, n_pairs=n * 53, n_positive=int(row.n_positive),
                           w1_lagmax_p=row.max_lag_permutation_p, w1_global_bh_q=q,
                           w1_best_auroc=row.best_auroc, w1_largest_score_lag_frames=row.best_lag_frames,
                           mean_global_bh_q=mean_stats.loc[network, "max_lag_bh_q"],
                           conclusion=outcome, n_bh_tests=110, source=relative(STATS)))
        qlabel = "—" if pd.isna(q) else f"{q:.3f}"
        table.append(f"| [{label}](lag_{short}.md) | {n} | {n * 53:,} | {qlabel} | {outcome} |")
    csv_path = OUT / "data/lag_evidence_summary.csv"
    pd.DataFrame(values).to_csv(csv_path, index=False)
    md_path = OUT / "captions/lag_evidence_summary.md"
    md_path.write_text("""# Which pathways show lag correspondence?

These are the seven planned progressive-bridge SMC **W1** pathway tests: state-average endpoint effects, one-frame forecast, source lags 1/4/8/16. Neuropeptides are the selected main example; the full set below prevents that example from standing in for all neuromodulators.

| Reference pathway | Supported sources | Tested pairs | Corrected lag-search q | Evidence under the saved reference-label test |
| --- | ---: | ---: | ---: | --- |
""" + "\n".join(table) + """

The peptide and union results pass global correction, but the union overlaps peptides and is not independent confirmation. None of the named monoamines passes. Tyramine is **not evaluable**, not a negative finding: its only reference source is unsupported. Small source counts make the named-pathway results especially limited.

Passing this test means correspondence at some tested lag under the saved target-label null. It does not establish a significant difference between lags, a unique delay, or active signaling. For progressive peptides, the existing L1-minus-L4 interval includes zero. The source sets are fixed within each curve, but differ across pathways; compare shapes cautiously, not as equally powered biological competitions.

The same saved endpoint-mean tests also have no globally corrected pathway alignment; their q-values remain in the linked CSV and original companion record. W1 can reflect mean, spread or other distributional changes and is not a variance-only or mean-independent measure.

The original test used 999 within-source target-label permutations shared across the lag grid, then global BH across 110 evaluable tests. No inference was rerun. The reference is Bentley molecular compatibility, not measured activity. Pointwise curve intervals are not provided. Peptide/union plotted values are audit-rounded common-source profiles; the monoamine values are exact saved scores on source sets verified to equal the all-lag intersection.

[Exact saved evidence fields](../data/lag_evidence_summary.csv) · [Manifest](../lag_manifest.json) · [Full earlier technical record](../../figure_atlas_20260831/REFERENCE_CAPTIONS.md)
""")
    return [csv_path, md_path]


def main() -> None:
    source_hashes = inputs()
    groups, stats, mean_stats = saved_data()
    for directory in ("figures", "data", "captions"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    figures, files = [], []
    for network, short, label in NETWORKS:
        rows, stat = groups[network], stats.loc[network]
        path = OUT / "data" / f"lag_{short}.csv"
        rows.to_csv(path, index=False)
        files.append(path)
        figures.append(plot_one(network, short, label, rows, stat))
        files.append(caption(network, short, label, rows, stat))
    files.extend(evidence_summary(stats, mean_stats))
    path = write_manifest("lag", source_hashes, figures, dict(
        renderer={"path": relative(__file__), "sha256": sha256(__file__)},
        output_files=files, figure_count=7, evaluable_curves=6, explicitly_non_evaluable_figures=1,
        row_count=28, reference="Bentley receptor/pathway compatibility", method=METHOD,
        channel=CHANNEL, context="state_average", forecast_horizon_frames=1,
        source_lag_frames=LAGS, source_lag_seconds=[0.25, 1, 2, 4],
        inference="Existing 999-permutation maximum-over-lags reference-label tests; BH over 110 evaluable tests; no new inference",
        mask_verification="For monoamines, the saved intersection is a subset of each per-lag supported set and has equal size at every lag, proving set equality. Peptide/union use explicitly common-source audit profiles.",
        uncertainty="No pointwise curve intervals are added; reference-label null does not identify a unique lag or include animal/model-refit uncertainty.",
        precision="Peptide/union curves: saved audit values rounded to 3 decimals; all monoamine curves and all q-values: exact saved CSV values.",
        independent_replication="Overlapping peptide/union and named/aggregate monoamine networks are not independent replications.",
    ))
    print(json.dumps({"manifest": relative(path), "figures": [f["stem"] for f in figures]}, indent=2))


if __name__ == "__main__":
    main()
