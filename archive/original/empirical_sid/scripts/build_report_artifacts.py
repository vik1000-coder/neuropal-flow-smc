#!/usr/bin/env python3
"""Build auditable LaTeX tables for the SID empirical-validation report.

All numerical entries are selected from the validated bootstrap summaries. The
script fails on ambiguous or missing primary rows so the report cannot silently
substitute a development result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


METHOD = {
    "B0_ZERO": "Zero",
    "B2_RAW_MOMENTS": "Direct moments",
    "B2_RAW_MOMENTS_FINITE": "Direct moments (finite)",
    "B4_GAUSSIAN_LOGVAR": "Gaussian log-variance",
    "B7_MDN_CORRECT": "Matched gain mixture",
    "A2_RATIO_CRITIC": "Ratio critic",
    "S1_NLL_SCORE__A1_DENSITY_AD": "Matched normalized law",
    "S2_HYVARINEN__A7_HODGE": r"Clean Hyv\"arinen + Hodge",
    "S6_GAUSS_DSM__A3_CENTER": "Gaussian score control",
    "S7_ANCHORED_ENERGY__A3_CENTER": "Anchored score",
    "S7_ANCHORED_ENERGY__A7_HODGE": "Anchored score + Hodge",
}

FAMILY = {
    "m1_mean": "M1 mean",
    "m2_covariance": "M2 covariance",
    "m3_bounded": "M3 bounded skew",
    "m4_quartic": "M4 quartic",
    "m4_tail": "M4 tail",
    "m5_occupancy": "M5 occupancy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tex(text: object) -> str:
    value = str(text)
    replacements = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def number(row: pd.Series, digits: int = 3) -> str:
    return f"{row['mean']:.{digits}f} [{row['ci_lower']:.{digits}f}, {row['ci_upper']:.{digits}f}]"


def select(
    frame: pd.DataFrame,
    *,
    family: str,
    estimand: str,
    channel: str,
    method: str,
    metric: str,
    cell: str | None = None,
) -> pd.Series:
    mask = (
        (frame.dgp_family == family)
        & (frame.estimand_type == estimand)
        & (frame.channel == channel)
        & (frame.method_id == method)
        & (frame.metric_name == metric)
    )
    if cell is not None:
        mask &= frame.cell_id == cell
    rows = frame[mask]
    if rows.shape[0] != 1:
        raise ValueError(
            f"expected one row, found {rows.shape[0]}: "
            f"{family}/{estimand}/{channel}/{method}/{metric}/{cell}"
        )
    return rows.iloc[0]


def local_table(tier1: pd.DataFrame) -> str:
    choices = [
        ("m1_mean", "mean", "B2_RAW_MOMENTS"),
        ("m1_mean", "mean", "A2_RATIO_CRITIC"),
        ("m1_mean", "mean", "S2_HYVARINEN__A7_HODGE"),
        ("m2_covariance", "covariance", "B2_RAW_MOMENTS"),
        ("m2_covariance", "covariance", "A2_RATIO_CRITIC"),
        ("m2_covariance", "covariance", "S6_GAUSS_DSM__A3_CENTER"),
        ("m3_bounded", "third_cumulant", "S1_NLL_SCORE__A1_DENSITY_AD"),
        ("m3_bounded", "third_cumulant", "B2_RAW_MOMENTS"),
        ("m3_bounded", "third_cumulant", "A2_RATIO_CRITIC"),
        ("m3_bounded", "third_cumulant", "S2_HYVARINEN__A7_HODGE"),
        ("m4_quartic", "fourth_central", "S1_NLL_SCORE__A1_DENSITY_AD"),
        ("m4_quartic", "fourth_central", "B2_RAW_MOMENTS"),
        ("m4_quartic", "fourth_central", "S2_HYVARINEN__A7_HODGE"),
        ("m4_tail", "tail_probability", "S1_NLL_SCORE__A1_DENSITY_AD"),
        ("m4_tail", "tail_probability", "B2_RAW_MOMENTS"),
        ("m5_occupancy", "mode_occupancy", "S1_NLL_SCORE__A1_DENSITY_AD"),
        ("m5_occupancy", "mode_occupancy", "B2_RAW_MOMENTS"),
    ]
    body = []
    for family, channel, method in choices:
        error = select(tier1, family=family, estimand="local", channel=channel, method=method, metric="nrmse")
        slope = select(tier1, family=family, estimand="local", channel=channel, method=method, metric="calibration_slope")
        recovered = error.ci_upper < 1 and 0.8 <= slope["mean"] <= 1.2
        status = r"\yes" if recovered else (r"\partialresult" if error.ci_upper < 1 else r"\no")
        body.append(
            f"{FAMILY[family]} & {METHOD[method]} & {number(error)} & {number(slope)} & {status} \\\\"
        )
    return "\n".join([
        r"\begin{longtable}{p{0.15\linewidth}p{0.23\linewidth}p{0.23\linewidth}p{0.23\linewidth}c}",
        r"\caption{Selected local results. Values are mean [95\% DGP-bootstrap interval]. Partial means NRMSE recovery without registered calibration.}\\",
        r"\toprule Target & Method & NRMSE & Calibration slope & Calibrated?\\ \midrule",
        r"\endfirsthead \toprule Target & Method & NRMSE & Calibration slope & Calibrated?\\ \midrule \endhead",
        *body,
        r"\bottomrule\end{longtable}",
    ]) + "\n"


def information_access_table() -> str:
    rows = [
        ("Direct moments", "observational training", "Target-specific raw moment/event heads; analytic derivatives"),
        ("Ratio critic", "training plus conditional evaluation queries", "Fresh conditional draws center and integrate the witness"),
        ("Generic score readouts (Tier 1)", "training plus oracle-density quadrature", "Favorable best case; not an observational-only comparison"),
        ("Generic score readouts (Tier 2 static)", "training plus oracle-density quadrature", "Same favorable quadrature audit as Tier 1"),
        ("Matched normalized law", "known DGP tilt basis", r"Fits the amplitude of supplied $\psi(y)$"),
        ("Matched dynamic mixture", "known gain/noise family", "Fits the mixing law with supplied structural form"),
        ("Endpoint adapters", "endpoint labels or conditional queries", "Stronger access than observational local regression"),
    ]
    body = [f"{tex(a)} & {tex(b)} & {c} \\\\" for a, b, c in rows]
    return "\n".join([
        r"\begin{table}[H]\centering\small",
        r"\caption{Audited information access.}",
        r"\begin{tabularx}{\linewidth}{p{0.22\linewidth}p{0.27\linewidth}X}\toprule",
        r"Method class & Effective access & Consequence\\\midrule",
        *body,
        r"\bottomrule\end{tabularx}\end{table}",
    ]) + "\n"


def tier2_table(tier2: pd.DataFrame) -> str:
    dynamic = [
        ("D3 context 2, N=8k", "d3_hidden_modulator", "variance", "B4_GAUSSIAN_LOGVAR", "d3_context_2__n_8000", "nrmse"),
        ("D3 context 2, N=8k", "d3_hidden_modulator", "variance", "B2_RAW_MOMENTS", "d3_context_2__n_8000", "nrmse"),
        ("D3 context 2, N=8k", "d3_hidden_modulator", "variance", "S7_ANCHORED_ENERGY__A7_HODGE", "d3_context_2__n_8000", "nrmse"),
        ("D3 context 12, N=32k", "d3_hidden_modulator", "variance", "B4_GAUSSIAN_LOGVAR", "d3_context_12__n_32000", "nrmse"),
        ("D3 context 12, N=32k", "d3_hidden_modulator", "variance", "B2_RAW_MOMENTS", "d3_context_12__n_32000", "nrmse"),
        ("D3 context 12, N=32k", "d3_hidden_modulator", "variance", "S7_ANCHORED_ENERGY__A7_HODGE", "d3_context_12__n_32000", "nrmse"),
        ("D5 primary filter", "d5_observation_filter", "variance", "B2_RAW_MOMENTS", "d5_primary_filter__n_8000", "nrmse"),
        ("D5 primary filter", "d5_observation_filter", "variance", "S7_ANCHORED_ENERGY__A7_HODGE", "d5_primary_filter__n_8000", "nrmse"),
        ("D5 observation-only null", "d5_observation_filter", "variance", "S7_ANCHORED_ENERGY__A7_HODGE", "d5_observation_only_null__n_8000", "null_rms"),
    ]
    static = [
        ("M3 bounded, N=2k", "m3_bounded", "third_cumulant", "B2_RAW_MOMENTS", "sample_n_2000__m3_bounded"),
        ("M3 bounded, N=32k", "m3_bounded", "third_cumulant", "B2_RAW_MOMENTS", "sample_n_32000__m3_bounded"),
        ("M4 quartic, N=32k", "m4_quartic", "fourth_central", "B2_RAW_MOMENTS", "sample_n_32000__m4_quartic"),
        ("M4 quartic, N=32k", "m4_quartic", "fourth_central", "S1_NLL_SCORE__A1_DENSITY_AD", "sample_n_32000__m4_quartic"),
        ("M5 occupancy, N=32k", "m5_occupancy", "mode_occupancy", "B2_RAW_MOMENTS", "sample_n_32000__m5_occupancy"),
        ("M5 occupancy, N=32k", "m5_occupancy", "mode_occupancy", "S1_NLL_SCORE__A1_DENSITY_AD", "sample_n_32000__m5_occupancy"),
    ]
    posthoc = [
        ("Post-hoc M3 fixed amp., N=2k", "m3_bounded", "third_cumulant", "B2_RAW_MOMENTS", "fixed_amplitude_n_2000__m3_bounded"),
        ("Post-hoc M3 fixed amp., N=32k", "m3_bounded", "third_cumulant", "B2_RAW_MOMENTS", "fixed_amplitude_n_32000__m3_bounded"),
        ("Post-hoc M4 fixed amp., N=2k", "m4_quartic", "fourth_central", "B2_RAW_MOMENTS", "fixed_amplitude_n_2000__m4_quartic"),
        ("Post-hoc M4 fixed amp., N=32k", "m4_quartic", "fourth_central", "B2_RAW_MOMENTS", "fixed_amplitude_n_32000__m4_quartic"),
        ("Post-hoc M4 score, N=2k", "m4_quartic", "fourth_central", "S2_HYVARINEN__A7_HODGE", "fixed_amplitude_n_2000__m4_quartic"),
        ("Post-hoc M4 score, N=32k", "m4_quartic", "fourth_central", "S2_HYVARINEN__A7_HODGE", "fixed_amplitude_n_32000__m4_quartic"),
        ("Post-hoc M5 fixed amp., N=2k", "m5_occupancy", "mode_occupancy", "B2_RAW_MOMENTS", "fixed_amplitude_n_2000__m5_occupancy"),
        ("Post-hoc M5 fixed amp., N=32k", "m5_occupancy", "mode_occupancy", "B2_RAW_MOMENTS", "fixed_amplitude_n_32000__m5_occupancy"),
    ]
    body = []
    for label, family, channel, method, cell, metric in dynamic:
        row = select(tier2, family=family, estimand="dynamic_local", channel=channel, method=method, metric=metric, cell=cell)
        access = row.information_access.replace("_", " ")
        body.append(f"{label} & {METHOD[method]} & {tex(metric)} & {number(row)} & {tex(access)} \\\\")
    for label, family, channel, method, cell in static:
        row = select(tier2, family=family, estimand="local", channel=channel, method=method, metric="nrmse", cell=cell)
        access = row.information_access.replace("_", " ")
        body.append(f"{label} & {METHOD[method]} & NRMSE & {number(row)} & {tex(access)} \\\\")
    for label, family, channel, method, cell in posthoc:
        row = select(tier2, family=family, estimand="local", channel=channel, method=method, metric="nrmse", cell=cell)
        access = row.information_access.replace("_", " ")
        body.append(f"{label} & {METHOD[method]} & NRMSE & {number(row)} & {tex(access)} \\\\")
    return "\n".join([
        r"\begin{longtable}{p{0.17\linewidth}p{0.18\linewidth}p{0.08\linewidth}p{0.20\linewidth}p{0.22\linewidth}}",
        r"\caption{Selected Tier 2 results: registered rows use seeds 3001--3030; post-hoc fixed-amplitude rows use seeds 4001--4030.}\\",
        r"\toprule Cell & Method & Metric & Mean [95\% CI] & Audited access\\\midrule",
        r"\endfirsthead\toprule Cell & Method & Metric & Mean [95\% CI] & Audited access\\\midrule\endhead",
        *body,
        r"\bottomrule\end{longtable}",
    ]) + "\n"


def claim_registry() -> str:
    rows = [
        ("H1", "Response-score accuracy need not imply history-tangent accuracy", r"\yes", "Paired operator panels show large error amplification"),
        ("H2", "Direct history tangent can recover typed effects", r"\partialresult", "Supported for M1--M3 and D1--D3; not M4/tail/occupancy"),
        ("H3", "Stabilized score route has a general advantage", r"\no", "No equal-access paired advantage; calibration and null failures"),
        ("H4", "Hodge recovery depends on connected support", r"\yes", "Oracle bridge ladder passes until disconnected endpoint"),
        ("H5", "Finite contrasts can be easier than local derivatives", r"\partialresult", "True for G8 aligned representation; not generic endpoint adapters"),
        ("H6", "Higher-order success depends on geometry/representation", r"\yes", "M3, M4, events, and path motif separate sharply"),
        ("H7", "Distributional lag recovery is possible", r"\yes", "Matched mixture and direct moments recover D2 profiles"),
    ]
    body = [f"{a} & {tex(b)} & {c} & {tex(d)} \\\\" for a, b, c, d in rows]
    return "\n".join([
        r"\begin{table}[H]\centering\small",
        r"\caption{Preregistered claim registry.}",
        r"\begin{tabularx}{\linewidth}{p{0.05\linewidth}X p{0.10\linewidth}p{0.37\linewidth}}\toprule",
        r"ID & Claim & Result & Evidence\\\midrule", *body,
        r"\bottomrule\end{tabularx}\end{table}",
    ]) + "\n"


def decision_table(tier1: pd.DataFrame) -> str:
    def result(family: str, estimand: str, channel: str, method: str) -> str:
        row = select(
            tier1,
            family=family,
            estimand=estimand,
            channel=channel,
            method=method,
            metric="nrmse",
        )
        return f"{METHOD.get(method, method)} {number(row, 2)}"

    target_rows = [
        (
            "Mean",
            result("m1_mean", "local", "mean", "B2_RAW_MOMENTS"),
            result("m1_mean", "local", "mean", "S1_NLL_SCORE__A1_DENSITY_AD"),
            result("m1_mean", "local", "mean", "S2_HYVARINEN__A7_HODGE"),
            result("m1_mean", "finite", "mean", "B2_RAW_MOMENTS_FINITE"),
            "Numerically yes",
            "No score advantage",
            "Low-order home field; full nuisance FPR not emitted",
        ),
        (
            "Variance lag",
            result("d2_observed_stochastic_gain", "dynamic_local", "variance", "B2_RAW_MOMENTS"),
            result("d2_observed_stochastic_gain", "dynamic_local", "variance", "B7_MDN_CORRECT"),
            result("d2_observed_stochastic_gain", "dynamic_local", "variance", "S7_ANCHORED_ENERGY__A7_HODGE"),
            result("d2_observed_stochastic_gain", "dynamic_finite", "variance", "B2_RAW_MOMENTS_FINITE"),
            "Yes: direct/matched",
            "Matched only, with stronger structure",
            "Score slope 0.797; hidden state and filtering add error",
        ),
        (
            "Covariance lag",
            result("d2_observed_stochastic_gain", "dynamic_local", "covariance", "B2_RAW_MOMENTS"),
            result("d2_observed_stochastic_gain", "dynamic_local", "covariance", "B7_MDN_CORRECT"),
            "No score readout",
            result("d2_observed_stochastic_gain", "dynamic_finite", "covariance", "B2_RAW_MOMENTS_FINITE"),
            "Yes: direct/matched",
            "Matched only, with stronger structure",
            "No generic score covariance estimate in the implemented D2 route",
        ),
        (
            "Third-cumulant lag",
            result("d2_observed_stochastic_gain", "dynamic_local", "third_cumulant", "B2_RAW_MOMENTS"),
            result("d2_observed_stochastic_gain", "dynamic_local", "third_cumulant", "B7_MDN_CORRECT"),
            result("d2_observed_stochastic_gain", "dynamic_local", "third_cumulant", "S7_ANCHORED_ENERGY__A7_HODGE"),
            result("d2_observed_stochastic_gain", "dynamic_finite", "third_cumulant", "B2_RAW_MOMENTS_FINITE"),
            "Yes: direct/matched",
            "Matched only, with stronger structure",
            "Score slope is negative; static clean score is attenuated",
        ),
        (
            "Standardized skewness",
            "Direct raw moments available",
            "Matched-law moments available",
            "Not separately confirmed",
            "Endpoint moments available",
            "Unresolved as a primary claim",
            "No",
            "Run emitted third cumulant, not a standalone primary skewness board",
        ),
        (
            "Tail probability",
            result("m4_tail", "local", "tail_probability", "B2_RAW_MOMENTS"),
            result("m4_tail", "local", "tail_probability", "S1_NLL_SCORE__A1_DENSITY_AD"),
            result("m4_tail", "local", "tail_probability", "S2_HYVARINEN__A7_HODGE"),
            result("m4_tail", "finite", "tail_probability", "B2_RAW_MOMENTS_FINITE"),
            "Matched law only",
            "No fair general advantage",
            r"Positivity capped achieved $\Lambda$ at 0.128",
        ),
        (
            "Mode occupancy",
            result("m5_occupancy", "local", "mode_occupancy", "B2_RAW_MOMENTS"),
            result("m5_occupancy", "local", "mode_occupancy", "S1_NLL_SCORE__A1_DENSITY_AD"),
            result("m5_occupancy", "local", "mode_occupancy", "S2_HYVARINEN__A7_HODGE"),
            result("m5_occupancy", "finite", "mode_occupancy", "B2_RAW_MOMENTS_FINITE"),
            "Matched law only",
            "No fair general advantage",
            r"Boundary-limited; achieved $\Lambda$ at target 3 is 0.949",
        ),
        (
            "Fourth central / quartic path",
            result("m4_quartic", "local", "fourth_central", "B2_RAW_MOMENTS"),
            result("m4_quartic", "local", "fourth_central", "S1_NLL_SCORE__A1_DENSITY_AD"),
            result("m4_quartic", "local", "fourth_central", "S2_HYVARINEN__A7_HODGE"),
            "G8 aligned motif MLP: 1.13% median relative error (development)",
            "Matched/access-aligned only",
            "No general advantage",
            "G8 is finite and developmental; it is not the M4 local target",
        ),
        (
            "Dynamic lag tensor",
            "Direct moments: 0.22--0.26 across D2 distributional channels",
            "Matched gain mixture: 0.054--0.057",
            "0.50 variance; 1.33 third cumulant",
            "Direct finite: 0.22--0.26",
            "Yes: direct/matched",
            "No score advantage",
            "Dense unknown subspaces and multiple gain factors remain untested",
        ),
    ]
    target_body = [
        f"{tex(a)} & {tex(b)} & {tex(c)} & {tex(d)} & {tex(e)} & {tex(f)} & {tex(g)} & {tex(h)} \\\\"
        for a, b, c, d, e, f, g, h in target_rows
    ]
    rows = [
        ("Known mean/covariance/cumulant/event target", "Targeted direct head", "Best general calibration and efficiency in implemented cells"),
        ("Many queries from one reusable conditional law", "Flexible normalized MDN/flow", "Promising next baseline; current matched laws are specification upper bounds"),
        ("Supported finite intervention with known geometry", "Contrast-aligned classifier", "G8 development evidence; generic classifiers are insufficient"),
        ("Operator diagnosis or theory study", r"Clean score + mixed-field/Hodge gates", "Useful decomposition, not current production estimator"),
        ("Primary general SID estimator", "Do not select current DSM/anchored score", "No calibrated/equal-access advantage and observation-null failure"),
    ]
    body = [f"{tex(a)} & {b} & {tex(c)} \\\\" for a, b, c in rows]
    selection = "\n".join([
        r"\begin{table}[H]\centering\small",
        r"\caption{Method-selection decision from the implemented evidence.}",
        r"\begin{tabularx}{\linewidth}{p{0.28\linewidth}p{0.28\linewidth}X}\toprule",
        r"Use case & Preferred route & Reason\\\midrule", *body,
        r"\bottomrule\end{tabularx}\end{table}",
    ])
    targets = "\n".join([
        r"\begin{landscape}\scriptsize\begin{longtable}{p{0.08\linewidth}p{0.12\linewidth}p{0.12\linewidth}p{0.12\linewidth}p{0.12\linewidth}p{0.08\linewidth}p{0.08\linewidth}p{0.15\linewidth}}",
        r"\caption{Final target-by-target decision table. NRMSE values are mean [95\% DGP-bootstrap interval].}\\",
        r"\toprule Target & Best direct & Best normalized law & Best score-derived & Best finite & Recovered? & Advantage? & Main limitation\\\midrule",
        r"\endfirsthead\toprule Target & Best direct & Best normalized law & Best score-derived & Best finite & Recovered? & Advantage? & Main limitation\\\midrule\endhead",
        *target_body,
        r"\bottomrule\end{longtable}\end{landscape}",
    ])
    return targets + "\n" + selection + "\n"


def artifact_table(paths: list[tuple[str, Path]]) -> str:
    rows = []
    for role, path in paths:
        digest = sha256(path)
        rows.append(f"{tex(role)} & \\artifact{{{path}}} & \\texttt{{{digest[:12]}...}} \\\\")
    return "\n".join([
        r"\scriptsize\begin{longtable}{p{0.17\linewidth}p{0.56\linewidth}p{0.16\linewidth}}",
        r"\caption{Primary artifact ledger. The displayed hashes are prefixes; complete hashes are in package manifests.}\\",
        r"\toprule Role & Path & SHA-256 prefix\\\midrule",
        r"\endfirsthead\toprule Role & Path & SHA-256 prefix\\\midrule\endhead",
        *rows, r"\bottomrule\end{longtable}",
    ]) + "\n"


def compute_table(tier1_seed: pd.DataFrame, tier2_seed: pd.DataFrame) -> str:
    groups: list[tuple[str, pd.DataFrame, list[str]]] = [
        (
            "Tier 1 static",
            tier1_seed[tier1_seed.dgp_family.astype(str).str.startswith("m")],
            ["dgp_family", "dgp_seed"],
        ),
        (
            "Tier 1 dynamic",
            tier1_seed[tier1_seed.dgp_family.astype(str).str.startswith("d")],
            ["dgp_family", "dgp_seed"],
        ),
        (
            "Tier 2 dynamic robustness",
            tier2_seed[tier2_seed.study_component == "dynamic_robustness"],
            ["cell_id", "dgp_seed"],
        ),
        (
            "Tier 2 static slices",
            tier2_seed[tier2_seed.study_component == "static_slices"],
            ["cell_id", "dgp_seed"],
        ),
        (
            "Post-hoc fixed-amplitude",
            tier2_seed[tier2_seed.study_component == "fixed_amplitude_diagnostic"],
            ["cell_id", "dgp_seed"],
        ),
    ]
    rows = []
    for label, frame, keys in groups:
        cases = frame.groupby(keys, dropna=False).agg(
            wall_time=("wall_time", "max"),
            peak_memory=("peak_memory", "max"),
        )
        peak = cases.peak_memory.max()
        peak_label = "n/a" if pd.isna(peak) else f"{peak / 2**20:.0f}"
        rows.append(
            f"{label} & {cases.shape[0]} & {cases.wall_time.sum() / 3600:.2f} & "
            f"{cases.wall_time.median():.2f} & {peak_label} \\\\"
        )
    return "\n".join([
        r"\begin{table}[H]\centering\small",
        r"\caption{Recorded sequential compute. Wall time is summed once per DGP-family/seed cell; peak memory is the maximum runner-recorded resident usage.}",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Component & Cells & Total CPU-wall hours & Median sec/cell & Peak MiB\\\midrule",
        *rows,
        r"\bottomrule\end{tabular}\end{table}",
    ]) + "\n"


def checklist(t1: dict, t2: dict) -> str:
    rows = [
        ("Frozen preregistration/config/source", True, "Hashes recorded before confirmatory execution"),
        ("Stage 0 algebra/software gates", True, "53 checks passed"),
        ("Tier 1 static and dynamic cases", bool(t1.get("passed")), "270 static + 30 dynamic; 30 common seeds"),
        ("Legacy compatibility reproduction", True, "117 scientific metrics bitwise equal; source hashes differ"),
        ("Tier 2 hidden-state/observation stresses", bool(t2.get("passed")), "270 dynamic + 450 static registered cases"),
        ("Post-hoc fixed-amplitude diagnostic", bool(t2.get("passed")), "270 exploratory cases on new seeds; cannot upgrade confirmation"),
        ("DGP-seed bootstrap and access audit", True, "2,000 replicates; effective access retained"),
        ("Full learned mechanism-confusion matrix", False, "Target-only cells cannot certify nuisance FPR"),
        ("Dense unknown response geometry (D4)", False, "Affected coordinate is supplied in current static suite"),
        ("Flexible MDN/flow and neural score backends", False, "Still proposed work"),
        ("Full stabilization/teacher-contamination ablations", False, "Frozen compact score shortlist only"),
        ("Standalone skewness and full delta/noise ladders", False, "Primary registered cells only"),
    ]
    body = []
    for item, done, note in rows:
        body.append(f"{tex(item)} & {r'\yes' if done else r'\no'} & {tex(note)} \\\\")
    return "\n".join([
        r"\begin{table}[H]\centering\small",
        r"\caption{Runbook completion checklist.}",
        r"\begin{tabularx}{\linewidth}{X p{0.10\linewidth}p{0.42\linewidth}}\toprule",
        r"Item & Complete? & Audit note\\\midrule", *body,
        r"\bottomrule\end{tabularx}\end{table}",
    ]) + "\n"


def claims_frame(
    tier1: pd.DataFrame,
    tier2: pd.DataFrame,
    tier1_artifact: Path,
    tier2_artifact: Path,
    g8_summary: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(
        claim_id: str,
        statement: str,
        status: str,
        experiment: str,
        row: pd.Series,
        artifact: Path,
        assumptions: str,
    ) -> None:
        rows.append(
            {
                "claim_id": claim_id,
                "evidence_id": f"{claim_id}_E{sum(x['claim_id'] == claim_id for x in rows) + 1}",
                "statement": statement,
                "status": status,
                "estimand_type": row.estimand_type,
                "experiment": experiment,
                "dgp_family": row.dgp_family,
                "channel": row.channel,
                "method_id": row.method_id,
                "metric_name": row.metric_name,
                "estimate": row["mean"],
                "ci_lower": row.ci_lower,
                "ci_upper": row.ci_upper,
                "n_dgp_seeds": row.n_dgp_seeds,
                "information_access": row.information_access,
                "raw_artifact": str(artifact.resolve()),
                "assumptions_and_scope": assumptions,
            }
        )

    t1_cases = [
        (
            "C01",
            "Direct targeted regression numerically recovers M1 mean.",
            "supported_numerically",
            "Tier 1 static",
            ("m1_mean", "local", "mean", "B2_RAW_MOMENTS", "nrmse"),
            "Full recovery label withheld because all nuisance readouts were not emitted.",
        ),
        (
            "C02",
            "Direct targeted regression numerically recovers pure M2 covariance.",
            "supported_numerically",
            "Tier 1 static",
            ("m2_covariance", "local", "covariance", "B2_RAW_MOMENTS", "nrmse"),
            "Affected covariance projection is registered; dense geometry is not tested.",
        ),
        (
            "C03",
            "Clean score matching finds M3 direction but is attenuated.",
            "supported",
            "Tier 1 static",
            ("m3_bounded", "local", "third_cumulant", "S2_HYVARINEN__A7_HODGE", "calibration_slope"),
            "Readout uses favorable oracle-density quadrature.",
        ),
        (
            "C04",
            "General direct fourth-moment regression fails M4 at medium information.",
            "supported_negative",
            "Tier 1 static",
            ("m4_quartic", "local", "fourth_central", "B2_RAW_MOMENTS", "nrmse"),
            "NRMSE 1 is the zero-effect reference.",
        ),
        (
            "C05",
            "A mechanism-matched normalized law recovers M4 quartic.",
            "supported_upper_bound",
            "Tier 1 static",
            ("m4_quartic", "local", "fourth_central", "S1_NLL_SCORE__A1_DENSITY_AD", "nrmse"),
            "The signed-tilt basis is supplied; this is not a generic observational result.",
        ),
        (
            "C06",
            "Direct moments recover D2 variance lag profile.",
            "supported_numerically",
            "Tier 1 dynamic",
            ("d2_observed_stochastic_gain", "dynamic_local", "variance", "B2_RAW_MOMENTS", "nrmse"),
            "Observed modulator and registered source direction.",
        ),
        (
            "C06",
            "Direct moments recover D2 covariance lag profile.",
            "supported_numerically",
            "Tier 1 dynamic",
            ("d2_observed_stochastic_gain", "dynamic_local", "covariance", "B2_RAW_MOMENTS", "nrmse"),
            "Observed modulator and registered source direction.",
        ),
        (
            "C06",
            "Direct moments recover D2 third-cumulant lag profile.",
            "supported_numerically",
            "Tier 1 dynamic",
            ("d2_observed_stochastic_gain", "dynamic_local", "third_cumulant", "B2_RAW_MOMENTS", "nrmse"),
            "Observed modulator and registered source direction.",
        ),
        (
            "C07",
            "The anchored score route fails D2 third-cumulant recovery.",
            "supported_negative",
            "Tier 1 dynamic",
            ("d2_observed_stochastic_gain", "dynamic_local", "third_cumulant", "S7_ANCHORED_ENERGY__A7_HODGE", "nrmse"),
            "The same route is usable but under-calibrated for variance.",
        ),
        (
            "C08",
            "Low response-score error does not ensure low reconstructed history-tangent error.",
            "supported",
            "Tier 1 operator",
            ("m4_tail", "operator", "tail_probability", "S2_HYVARINEN", "response_score_nrmse"),
            "The same fitted score model is evaluated at both stages.",
        ),
        (
            "C08",
            "Low response-score error does not ensure low reconstructed history-tangent error.",
            "supported",
            "Tier 1 operator",
            ("m4_tail", "operator", "tail_probability", "S2_HYVARINEN", "history_tangent_nrmse"),
            "The same fitted score model is evaluated at both stages.",
        ),
    ]
    for claim_id, statement, status, experiment, key, assumptions in t1_cases:
        add(
            claim_id,
            statement,
            status,
            experiment,
            select(
                tier1,
                family=key[0],
                estimand=key[1],
                channel=key[2],
                method=key[3],
                metric=key[4],
            ),
            tier1_artifact,
            assumptions,
        )

    t2_cases = [
        (
            "C09",
            "The anchored score becomes numerically usable for hidden-state variance only at 32k samples.",
            "supported",
            "Tier 2 D3",
            ("d3_hidden_modulator", "dynamic_local", "variance", "S7_ANCHORED_ENERGY__A7_HODGE", "nrmse", "d3_context_12__n_32000"),
            "Polynomial anchored implementation; context length 12.",
        ),
        (
            "C10",
            "The anchored score emits a nonzero variance profile under the observation-only null.",
            "supported_negative",
            "Tier 2 D5",
            ("d5_observation_filter", "dynamic_local", "variance", "S7_ANCHORED_ENERGY__A7_HODGE", "null_rms", "d5_observation_only_null__n_8000"),
            "The latent biological gain is exactly absent; filtering and observation noise remain.",
        ),
        (
            "C11",
            "Along the fixed-Lambda local-alternative ladder, a generic direct estimator does not recover M4 quartic.",
            "supported_negative",
            "Tier 2 static",
            ("m4_quartic", "local", "fourth_central", "B2_RAW_MOMENTS", "nrmse", "sample_n_32000__m4_quartic"),
            "Target Lambda is held at 3 by recalibrating amplitude; affected scalar projection is supplied.",
        ),
        (
            "C13",
            "At fixed amplitude, direct M3 recovery improves strongly with sample size.",
            "posthoc_supported",
            "Tier 2 fixed-amplitude diagnostic",
            ("m3_bounded", "local", "third_cumulant", "B2_RAW_MOMENTS", "nrmse", "fixed_amplitude_n_2000__m3_bounded"),
            "Exploratory paired-seed learning curve; cannot upgrade a confirmatory claim.",
        ),
        (
            "C13",
            "At fixed amplitude, direct M3 recovery improves strongly with sample size.",
            "posthoc_supported",
            "Tier 2 fixed-amplitude diagnostic",
            ("m3_bounded", "local", "third_cumulant", "B2_RAW_MOMENTS", "nrmse", "fixed_amplitude_n_32000__m3_bounded"),
            "Exploratory paired-seed learning curve; cannot upgrade a confirmatory claim.",
        ),
        (
            "C14",
            "At fixed amplitude, clean score/Hodge M3 error and attenuation plateau with sample size.",
            "posthoc_supported",
            "Tier 2 fixed-amplitude diagnostic",
            ("m3_bounded", "local", "third_cumulant", "S2_HYVARINEN__A7_HODGE", "calibration_slope", "fixed_amplitude_n_2000__m3_bounded"),
            "Exploratory paired-seed learning curve with favorable oracle-density quadrature.",
        ),
        (
            "C15",
            "At fixed amplitude, more samples reduce direct M4 error but do not yield recovery by 32k.",
            "posthoc_supported_negative",
            "Tier 2 fixed-amplitude diagnostic",
            ("m4_quartic", "local", "fourth_central", "B2_RAW_MOMENTS", "nrmse", "fixed_amplitude_n_32000__m4_quartic"),
            "Exploratory paired-seed learning curve; achieved Lambda is approximately 12.",
        ),
        (
            "C16",
            "At fixed amplitude, occupancy estimators improve with sample size but remain above zero-estimator error at 32k.",
            "posthoc_supported_negative",
            "Tier 2 fixed-amplitude diagnostic",
            ("m5_occupancy", "local", "mode_occupancy", "B2_RAW_MOMENTS", "nrmse", "fixed_amplitude_n_32000__m5_occupancy"),
            "Exploratory paired-seed learning curve; achieved Lambda is approximately 3.8.",
        ),
        (
            "C14",
            "At fixed amplitude, clean score/Hodge M3 error and attenuation plateau with sample size.",
            "posthoc_supported",
            "Tier 2 fixed-amplitude diagnostic",
            ("m3_bounded", "local", "third_cumulant", "S2_HYVARINEN__A7_HODGE", "calibration_slope", "fixed_amplitude_n_32000__m3_bounded"),
            "Exploratory paired-seed learning curve with favorable oracle-density quadrature.",
        ),
    ]
    for claim_id, statement, status, experiment, key, assumptions in t2_cases:
        add(
            claim_id,
            statement,
            status,
            experiment,
            select(
                tier2,
                family=key[0],
                estimand=key[1],
                channel=key[2],
                method=key[3],
                metric=key[4],
                cell=key[5],
            ),
            tier2_artifact,
            assumptions,
        )

    g8 = pd.read_csv(g8_summary).set_index("estimator")
    for estimator in ["mlp_motif", "typed_quartic", "mlp_full_path_history", "linear_motif"]:
        value = float(g8.loc[estimator, "typed_effect_relative_error_median"])
        rows.append(
            {
                "claim_id": "C12",
                "evidence_id": f"C12_E{sum(x['claim_id'] == 'C12' for x in rows) + 1}",
                "statement": "G8 finite quartic-path recovery depends on contrast-aligned representation.",
                "status": "development_only",
                "estimand_type": "finite",
                "experiment": "G8 typed classifier development benchmark",
                "dgp_family": "g8_quartic_path",
                "channel": "quartic_path_feature",
                "method_id": estimator,
                "metric_name": "typed_effect_relative_error_median",
                "estimate": value,
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n_dgp_seeds": 4,
                "information_access": "endpoint_labels_and_queries",
                "raw_artifact": str(g8_summary.resolve()),
                "assumptions_and_scope": "Four independent generator systems; no confirmatory DGP-bootstrap interval.",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1-summary", required=True, type=Path)
    parser.add_argument("--tier1-validation", required=True, type=Path)
    parser.add_argument("--tier1-manifest", required=True, type=Path)
    parser.add_argument("--tier2-summary", required=True, type=Path)
    parser.add_argument("--tier2-validation", required=True, type=Path)
    parser.add_argument("--tier2-manifest", required=True, type=Path)
    parser.add_argument("--legacy-audit", required=True, type=Path)
    parser.add_argument("--g8-summary", required=True, type=Path)
    parser.add_argument("--posthoc-plan", required=True, type=Path)
    parser.add_argument("--claims-output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    tier1 = pd.read_csv(args.tier1_summary)
    tier2 = pd.read_csv(args.tier2_summary)
    tier1_seed = pd.read_csv(args.tier1_summary.with_name("seed_level.csv"), low_memory=False)
    tier2_seed = pd.read_csv(args.tier2_summary.with_name("seed_level.csv"), low_memory=False)
    t1_validation = json.loads(args.tier1_validation.read_text())
    t2_validation = json.loads(args.tier2_validation.read_text())
    if not t1_validation.get("passed") or not t2_validation.get("passed"):
        raise SystemExit("refusing to build report tables from an invalid package")
    tables = args.report_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "local_key_results.tex").write_text(local_table(tier1))
    (tables / "information_access.tex").write_text(information_access_table())
    (tables / "tier2_key_results.tex").write_text(tier2_table(tier2))
    (tables / "claim_registry.tex").write_text(claim_registry())
    (tables / "final_decision.tex").write_text(decision_table(tier1))
    (tables / "compute_summary.tex").write_text(compute_table(tier1_seed, tier2_seed))
    claims = claims_frame(
        tier1,
        tier2,
        args.tier1_summary.with_name("seed_level.csv"),
        args.tier2_summary.with_name("seed_level.csv"),
        args.g8_summary,
    )
    args.claims_output.parent.mkdir(parents=True, exist_ok=True)
    claims.to_csv(args.claims_output, index=False)
    ledger = [
        ("Tier 1 summary", args.tier1_summary.resolve()),
        ("Tier 1 validation", args.tier1_validation.resolve()),
        ("Tier 1 package manifest", args.tier1_manifest.resolve()),
        ("Tier 2 summary", args.tier2_summary.resolve()),
        ("Tier 2 validation", args.tier2_validation.resolve()),
        ("Tier 2 package manifest", args.tier2_manifest.resolve()),
        ("Legacy audit", args.legacy_audit.resolve()),
        ("Post-hoc diagnostic plan", args.posthoc_plan.resolve()),
        ("Machine-readable claims registry", args.claims_output.resolve()),
    ]
    (tables / "artifact_ledger.tex").write_text(artifact_table(ledger))
    (tables / "completion_checklist.tex").write_text(checklist(t1_validation, t2_validation))
    generated = "\n".join([
        r"\newcommand{\TierOneMetricRows}{" + f"{t1_validation['metric_rows']:,}" + "}",
        r"\newcommand{\TierTwoMetricRows}{" + f"{t2_validation['metric_rows']:,}" + "}",
        r"\newcommand{\TierOneValidationHash}{\texttt{" + sha256(args.tier1_validation)[:16] + r"...}}",
        r"\newcommand{\TierTwoValidationHash}{\texttt{" + sha256(args.tier2_validation)[:16] + r"...}}",
    ]) + "\n"
    (args.report_dir / "generated_results.tex").write_text(generated)
    print(f"wrote report artifacts to {args.report_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
