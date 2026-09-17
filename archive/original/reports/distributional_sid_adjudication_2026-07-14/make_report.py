from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "distributional_sid" / "runs"
HISTORY = ROOT / "history_tangent_benchmark" / "results"
PREVIOUS = ROOT / "reports" / "distributional_sid_decisive_2026-07-14"
FIG = HERE / "figures"
TAB = HERE / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

BLUE = "#356AA0"
ORANGE = "#D9822B"
PURPLE = "#7561A8"
GREEN = "#3B7D67"
GRAY = "#697386"
LIGHT = "#D8E2EC"
RED = "#B24C3D"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)


def read_seed(run: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / run / "seed_level.csv")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def mean(frame: pd.DataFrame, metric: str, method: str, **filters: object) -> float:
    selected = frame[frame.method == method]
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if selected.empty:
        raise ValueError(f"empty selection: {metric=} {method=} {filters=}")
    return float(selected[metric].mean())


def e6_figures() -> None:
    frame = read_seed("e6_adjudication_posthoc_20260714")
    families = [
        ("student_t_nu3", r"Student-$t_3$"),
        ("gap_mu3", "Gap 3"),
        ("gap_mu4", "Gap 4"),
    ]
    methods = ["ORTH_TARGETED", "ORTH_DENSITY_SCORE", "OR_R"]
    labels = ["Targeted sieve", "Density-score", "Oracle Riesz"]
    colors = [ORANGE, BLUE, GRAY]
    sizes = [8000, 32000, 128000]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.25), sharey=True)
    for axis, (prefix, title) in zip(axes, families):
        selected = frame[frame.cell_id.str.startswith(prefix)]
        for method, label, color in zip(methods, labels, colors):
            values = [mean(selected, "target_abs_error", method, n_train=n) for n in sizes]
            axis.plot(sizes, values, "o-", label=label, color=color)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(sizes, ["8k", "32k", "128k"])
        axis.set_title(title)
        axis.set_xlabel("Training histories")
    axes[0].set_ylabel("Absolute target error")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E6 adjudication: direct density-score learning removes the fixed-sieve bias", y=1.03, fontsize=11)
    fig.tight_layout()
    save(fig, "e6_sample_scaling")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.45))
    selected = frame[frame.cell_id.str.startswith("gap_mu4")]
    for method, label, color in zip(methods[:2], labels[:2], colors[:2]):
        alpha = [mean(selected, "alpha_nrmse", method, n_train=n) for n in sizes]
        coverage = [mean(selected, "coverage_fraction", method, n_train=n) for n in sizes]
        axes[0].plot(sizes, alpha, "o-", label=label, color=color)
        axes[1].plot(sizes, coverage, "o-", label=label, color=color)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes, ["8k", "32k", "128k"])
        axis.set_xlabel("Training histories")
    axes[0].set_ylabel(r"Representer NRMSE")
    axes[0].set_title("Nuisance approximation")
    axes[1].set_ylabel("95% target coverage")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].axhline(0.95, color=GRAY, linestyle="--", linewidth=1.1)
    axes[1].set_title("Inference exposes the bias")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E6 severe overlap: more data helps only after changing the nuisance model", y=1.03, fontsize=11)
    fig.tight_layout()
    save(fig, "e6_gap4_diagnosis")


def e8_figures() -> None:
    base = read_seed("e8_adjudication_posthoc_20260714")
    mle = read_seed("e8_mixture_mle_posthoc_20260714")
    base = base[(base.rho == 0.98) & (base.frequency_band == "wide")]
    mle = mle[(mle.rho == 0.98) & (mle.frequency_band == "wide")]
    horizons = [4, 16, 64]
    series = [
        (base, "LATENT_ORACLE_ORTH", "Latent oracle", GRAY, "--"),
        (base, "POINT_KNOWN_ORTH", "Point projection", PURPLE, ":"),
        (base, "CF_DECONV_KNOWN_ORTH", "Exact CF, known filter", BLUE, "-"),
        (mle, "CF_DECONV_MIXTURE_MLE_ORTH", "Exact CF, fitted filter", ORANGE, "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), sharey=True)
    for axis, n in zip(axes, [8000, 32000]):
        for source, method, label, color, style in series:
            values = [mean(source, "nrmse", method, n_train=n, horizon=h) for h in horizons]
            axis.plot(horizons, values, marker="o", linestyle=style, label=label, color=color)
        axis.axhline(1.0, color=RED, linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_xticks(horizons)
        axis.set_xlabel("Observation horizon")
        axis.set_title(f"n = {n // 1000}k")
    axes[0].set_ylabel("Tangent NRMSE")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(r"E8 at $\rho=.98$: horizon, not just sample size, controls recoverability", y=1.03, fontsize=11)
    fig.tight_layout()
    save(fig, "e8_horizon_deconvolution")

    fig, axis = plt.subplots(figsize=(7.2, 3.45))
    known = base[base.method == "CF_DECONV_KNOWN_ORTH"].groupby(["horizon", "n_train"], as_index=False).agg(
        nrmse=("nrmse", "mean"), amplification=("max_inverse_attenuation", "mean")
    )
    markers = {8000: "o", 32000: "s"}
    for n, group in known.groupby("n_train"):
        axis.scatter(group.amplification, group.nrmse, marker=markers[int(n)], s=55, label=f"n={int(n)//1000}k")
        for row in group.itertuples():
            axis.annotate(f"T={int(row.horizon)}", (row.amplification, row.nrmse), xytext=(4, 4), textcoords="offset points")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.axhline(1.0, color=RED, linestyle="--", linewidth=1.0)
    axis.set_xlabel("Maximum deconvolution amplification")
    axis.set_ylabel("Known-filter ORTH NRMSE")
    axis.set_title("Conditioning predicts the measurement-recovery gate")
    axis.legend(frameon=False)
    fig.tight_layout()
    save(fig, "e8_conditioning")


def e9_figures() -> None:
    ident = read_seed("e9_identifiability_posthoc_20260714")
    fixed = read_seed("e9_inversion_ridge_posthoc_20260714")
    oracle = read_seed("e9_oracle_inversion_tuning_20260714")
    dy32 = pd.concat(
        [
            read_seed("e9_response32_direct_posthoc_20260714"),
            read_seed("e9_response32_direct_replication_20260714"),
        ],
        ignore_index=True,
    )
    sizes = [8000, 32000, 128000]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.8))

    rank = ident.groupby("intercept_scheme", as_index=False).agg(
        minimum_singular_value=("coefficient_min_singular_value", "mean"),
        effective_rank=("response_effective_rank", "mean"),
    ).set_index("intercept_scheme")
    rank_order = ["symmetric", "identified"]
    axes[0, 0].bar(
        ["Symmetric\nintercepts", "Identified\nintercepts"],
        [rank.loc[name, "minimum_singular_value"] for name in rank_order],
        color=[ORANGE, BLUE],
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Minimum population singular value")
    axes[0, 0].set_title("Original nominal rank 3 was actually rank 2")
    for index, name in enumerate(rank_order):
        value = rank.loc[name, "minimum_singular_value"]
        axes[0, 0].text(index, value * 1.4, f"rank {rank.loc[name, 'effective_rank']:.0f}", ha="center")

    identified = ident[ident.intercept_scheme == "identified"]
    methods = ["ORTH_DIRECT_COV", "ORTH_DIRECT_COV_LOW_RANK_EFFECTIVE"]
    labels = ["Direct covariance", "Low-rank direct"]
    colors = [BLUE, GREEN]
    for method, label, color in zip(methods, labels, colors):
        values = [mean(identified, "covariance_tensor_nrmse", method, n_train=n) for n in sizes]
        axes[0, 1].plot(sizes, values, "o-", label=label, color=color)
    old_values = [mean(identified, "covariance_tensor_nrmse", "ORTH_CHARACTERISTIC_INVERSION", n_train=n) for n in sizes]
    new_values = [mean(fixed, "covariance_tensor_nrmse", "ORTH_CHARACTERISTIC_INVERSION", n_train=n) for n in sizes]
    axes[0, 1].plot(sizes, old_values, "o--", label=r"CF inversion, ridge $10^{-4}$", color=ORANGE)
    axes[0, 1].plot(sizes, new_values, "o-", label=r"CF inversion, ridge $10^{-8}$", color=PURPLE)
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(sizes, ["8k", "32k", "128k"])
    axes[0, 1].set_ylabel("Covariance-tensor NRMSE")
    axes[0, 1].set_title("The inversion failure was regularization bias")
    axes[0, 1].legend(frameon=False, fontsize=7)

    for metric, label, color in [
        ("response_subspace_angle_degrees", "Response", BLUE),
        ("source_subspace_angle_degrees", "Source", ORANGE),
    ]:
        values = [mean(identified, metric, "ORTH_DIRECT_COV", n_train=n) for n in sizes]
        axes[1, 0].plot(sizes, values, "o-", label=label, color=color)
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xticks(sizes, ["8k", "32k", "128k"])
    axes[1, 0].set_ylabel("Subspace angle (degrees)")
    axes[1, 0].set_title("Identified geometry improves with data")
    axes[1, 0].legend(frameon=False)

    selected = dy32[dy32.method.isin(["ORTH_DIRECT_COV", "ORTH_DIRECT_COV_LOW_RANK_EFFECTIVE", "ORACLE_RESPONSE_SUBSPACE_PROJECTED"])]
    summary = selected.groupby("method", as_index=False).agg(nrmse=("covariance_tensor_nrmse", "mean"))
    order = ["ORTH_DIRECT_COV", "ORTH_DIRECT_COV_LOW_RANK_EFFECTIVE", "ORACLE_RESPONSE_SUBSPACE_PROJECTED"]
    lookup = summary.set_index("method")
    values = [lookup.loc[item, "nrmse"] for item in order]
    axes[1, 1].bar(["Direct", "Low-rank", "Oracle response\nsubspace"], values, color=[BLUE, GREEN, GRAY])
    axes[1, 1].axhline(1.0, color=RED, linestyle="--", linewidth=1.0)
    axes[1, 1].set_ylabel("Covariance-tensor NRMSE")
    axes[1, 1].set_title(r"Response dimension 32, $n=32$k (30 seeds)")
    for index, value in enumerate(values):
        axes[1, 1].text(index, value + 0.015, f"{value:.3f}", ha="center")
    fig.suptitle("E9 adjudication: corrected DGP and inversion reveal a typed-geometry success", y=1.01, fontsize=11)
    fig.tight_layout()
    save(fig, "e9_identification_scaling")

    pivot = oracle.groupby(["frequency_band", "query_count", "inversion_ridge"]).covariance_tensor_nrmse.mean().reset_index()
    bands = ["very_low", "low", "current"]
    ridges = [1e-8, 1e-6, 1e-4]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for axis, q in zip(axes, [64, 256]):
        matrix = np.array(
            [
                [float(pivot[(pivot.frequency_band == band) & (pivot.query_count == q) & (pivot.inversion_ridge == ridge)].covariance_tensor_nrmse.iloc[0]) for ridge in ridges]
                for band in bands
            ]
        )
        image = axis.imshow(np.log10(matrix), cmap="viridis", aspect="auto", vmin=-2.35, vmax=0)
        axis.set_xticks(range(3), [r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$"])
        axis.set_yticks(range(3), ["Very low", "Low", "Current"])
        axis.set_xlabel("Inversion ridge")
        axis.set_title(f"{q} characteristic queries")
        for row in range(3):
            for column in range(3):
                color = "white" if matrix[row, column] < 0.12 else "black"
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color=color, fontsize=8)
    axes[0].set_ylabel("Frequency band")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.03)
    colorbar.set_label(r"$\log_{10}$ oracle covariance NRMSE")
    fig.suptitle("E9 oracle inversion audit: ridge choice dominated query count", y=1.02, fontsize=11)
    save(fig, "e9_oracle_inversion_heatmap")


def legacy_baseline_figure() -> None:
    metrics = pd.read_csv(HISTORY / "stable_sid_core_20260713" / "metrics.csv")
    selected = metrics[(metrics.metric_id == "tangent_nrmse") & (metrics.metric_status == "ok")]
    summary = selected.groupby("model_name", as_index=False).value.agg(["count", "median", "mean"]).reset_index()
    summary = summary.sort_values("median")
    names = {
        "ratio_critic": "Ratio critic",
        "autoregressive_transformer": "AR transformer",
        "affine_flow": "Affine flow",
        "autoregressive_mdn": "AR MDN",
        "diffusion_legacy": "Legacy diffusion",
        "bounded_energy": "Energy model",
        "gaussian_nll": "Gaussian NLL",
        "gaussian_dsm": "Gaussian DSM",
        "diffusion_edm": "EDM diffusion",
        "diffusion_gaussian_anchored": "Anchored diffusion",
    }
    fig, axis = plt.subplots(figsize=(8.6, 4.25))
    labels = [names[name] for name in summary.model_name]
    colors = [PURPLE if "diffusion" in name or "dsm" in name else BLUE for name in summary.model_name]
    axis.barh(labels, summary["median"], color=colors)
    axis.axvline(1.0, color=RED, linestyle="--", linewidth=1.1)
    axis.invert_yaxis()
    axis.set_xlabel("Median history-tangent NRMSE (heterogeneous legacy grid)")
    axis.set_title("Earlier score/generative benchmark: stable training did not imply a reliable tangent")
    for index, value in enumerate(summary["median"]):
        axis.text(value + 0.04, index, f"{value:.2f}", va="center")
    fig.tight_layout()
    save(fig, "legacy_score_model_summary")
    summary.rename(columns={"model_name": "method"}).to_csv(TAB / "legacy_score_model_summary.csv", index=False)


def write_tables_and_snapshot() -> None:
    e6 = read_seed("e6_adjudication_posthoc_20260714")
    e6_summary = e6.groupby(["dgp_family", "cell_id", "n_train", "method"], as_index=False).agg(
        target_abs_error=("target_abs_error", "mean"),
        coverage=("coverage_fraction", "mean"),
        alpha_nrmse=("alpha_nrmse", "mean"),
        oracle_representer_norm=("oracle_representer_norm", "mean"),
        seeds=("dgp_seed", "nunique"),
    )
    e6_summary.to_csv(TAB / "e6_adjudication_summary.csv", index=False)

    e8 = read_seed("e8_adjudication_posthoc_20260714")
    e8_mle = read_seed("e8_mixture_mle_posthoc_20260714")
    e8_all = pd.concat([e8, e8_mle], ignore_index=True)
    e8_summary = e8_all.groupby(["rho", "horizon", "n_train", "frequency_band", "method"], as_index=False).agg(
        nrmse=("nrmse", "mean"),
        coverage=("coverage_fraction", "mean"),
        rho_abs_error=("rho_abs_error", "mean"),
        max_inverse_attenuation=("max_inverse_attenuation", "mean"),
        seeds=("dgp_seed", "nunique"),
    )
    e8_summary.to_csv(TAB / "e8_adjudication_summary.csv", index=False)

    e9_ident = read_seed("e9_identifiability_posthoc_20260714")
    e9_fixed = read_seed("e9_inversion_ridge_posthoc_20260714")
    e9_dy32 = pd.concat(
        [read_seed("e9_response32_direct_posthoc_20260714"), read_seed("e9_response32_direct_replication_20260714")],
        ignore_index=True,
    )
    e9_summary = pd.concat(
        [
            e9_ident.assign(audit="identifiability_ridge_1e-4"),
            e9_fixed.assign(audit="inversion_ridge_1e-8"),
            e9_dy32.assign(audit="response_dim_32_replication"),
        ],
        ignore_index=True,
    ).groupby(["audit", "intercept_scheme", "response_dim", "n_train", "method"], as_index=False).agg(
        covariance_nrmse=("covariance_tensor_nrmse", "mean"),
        feature_nrmse=("feature_tensor_nrmse", "mean"),
        response_angle=("response_subspace_angle_degrees", "mean"),
        source_angle=("source_subspace_angle_degrees", "mean"),
        coverage=("coverage_fraction", "mean"),
        null_fwer=("null_edge_familywise_call", "mean"),
        selected_rank=("selected_rank", "mean"),
        seeds=("dgp_seed", "nunique"),
    )
    e9_summary.to_csv(TAB / "e9_adjudication_summary.csv", index=False)

    gate_matrix = pd.DataFrame(
        [
            ("E0", "Algebra, leakage, reproducibility", "PASS", "20/20 package tests; all new run validations pass", "Core implementation sound"),
            ("E1", "Regular equal-access recovery", "PASS / NO POINT WIN", "ORTH NRMSE 0.104; coverage 0.940; PLUG ratio 1.002", "Inference benefit, not point-error benefit"),
            ("E2", "Orthogonal remainder", "PASS", "Product-bias R2 > .99999997", "Theory realized under controlled nuisances"),
            ("E3", "One future/dependence", "PASS WITH HAC", "rho=.9 NRMSE .145/.093; HAC coverage .949/.951", "Blocked splits and HAC required"),
            ("E4", "Moment-blind law change", "PASS AT RESOLVING BANK", "Power 1.00/.867; moment FWER .067/.067", "Finite bank resolution is part of the claim"),
            ("E5", "Support motion", "PASS", "ORTH NRMSE .016 at zero noise; likelihood score undefined", "Distinctive weak-feature success"),
            ("E6", "Heavy tails/weak overlap", "CONDITIONAL PASS", "Targeted sieve stays biased; density-score ORTH reaches .0036 error and 1.00 coverage in gap4 at n=128k", "Replace fixed sieve; retain overlap warnings"),
            ("E7", "Heterogeneous cancellation", "PASS AT R=2", "Energy 3.982 vs 4.000; sign reversal recovered", "Declare history resolution"),
            ("E8", "Filtered latent recovery", "CONDITIONAL PASS", "rho=.98, T=64, n=32k: matched-model CF ORTH NRMSE .141; T=4 NRMSE 2.88", "Horizon/model correctness are hard gates"),
            ("E9", "Dense unknown geometry", "TYPED PASS / FULL-FEATURE FAIL", "Identified dy8 direct NRMSE .106 at 128k; dy32 low-rank .177 at 32k; full feature NRMSE 1.01", "Use structured covariance readout"),
            ("E10", "Flexible reusable-law baselines", "PASS POOLED / LOCAL", "Five MDN/NSF baselines, seven families, 30 fits per cell", "No universal family winner or scale frontier"),
            ("E11", "Path ordering", "PASS", "Full-path power 1.00; endpoint/unordered FWER .056", "Distinctive path-feature success"),
        ],
        columns=["experiment", "question", "revised_decision", "evidence", "consequence"],
    )
    gate_matrix.to_csv(TAB / "revised_gate_matrix.csv", index=False)

    baseline = pd.DataFrame(
        [
            ("Plug-in derivative", "Same bounded-feature estimand", "E1,E3,E4,E5,E7,E8,E9,E11", "Run", "Family-specific confirmations"),
            ("Riesz-only", "Same bounded-feature estimand", "E1,E3,E4,E5,E7,E8,E9,E11", "Run", "Family-specific confirmations"),
            ("ORTH", "Primary bounded-feature estimand", "E1-E11 where applicable", "Run", "All confirmations"),
            ("Oracle m / oracle alpha / oracle Riesz", "Component upper bounds", "E1,E4,E5,E6", "Run", "OR_M, OR_A, OR_R, OR_PW"),
            ("Zero estimator", "NRMSE reference", "E1,E4,E5,E6,E8,E9", "Run", "Seed-level files"),
            ("Direct mean/moment heads", "Known typed target", "E1,E4", "Run", "DIRECT and MOMENT_ORTH"),
            ("Direct covariance / low-rank", "Known typed covariance target", "E1,E9", "Run", "E9 includes identified dy8 and dy32"),
            ("Known/estimated deconvolution", "Latent characteristic target", "E8", "Run", "Point, exact CF, oracle and fitted measurement models"),
            ("Full-covariance MDN K=5/10/20", "Reusable normalized law", "E10 all seven families", "Run", "Analytic and sampled readouts; 3 inits x 10 seeds"),
            ("Spline flow NSF-6/10", "Reusable normalized law", "E10 all seven families", "Run", "Common sampling budgets 64/256/1024"),
            ("Diffusion/DSM/flow matching", "Response-score/history-tangent models", "Earlier eight-family benchmark", "Run, different estimand", "stable_sid_core_20260713"),
            ("Ratio critic / signed classifier", "Density-ratio or finite contrast", "Earlier tangent/finite grids", "Run, different estimand", "stable_sid_core and stable_sid_finite"),
            ("Finite contrast", "h+delta v versus h-delta v", "Earlier eight-family benchmark", "Run, not a derivative baseline", "stable_sid_finite_20260713"),
        ],
        columns=["baseline_family", "target", "coverage", "status", "source"],
    )
    baseline.to_csv(TAB / "baseline_coverage.csv", index=False)

    validation_runs = [
        "e6_adjudication_tuning_20260714",
        "e6_adjudication_posthoc_20260714",
        "e8_adjudication_posthoc_20260714",
        "e8_mixture_mle_posthoc_20260714",
        "e9_identifiability_posthoc_20260714",
        "e9_oracle_inversion_tuning_20260714",
        "e9_inversion_ridge_posthoc_20260714",
        "e9_query256_posthoc_20260714",
        "e9_response32_direct_posthoc_20260714",
        "e9_response32_direct_replication_20260714",
    ]
    checks: list[dict[str, object]] = []
    for run in validation_runs:
        validation_path = RUNS / run / "validation.json"
        validation = json.loads(validation_path.read_text())
        seed = read_seed(run)
        checks.append(
            {
                "run": run,
                "passed": bool(validation.get("passed", False)),
                "rows": len(seed),
                "failed_rows": int((seed.fit_status != "ok").sum()) if "fit_status" in seed else 0,
                "duplicate_full_rows": int(seed.duplicated().sum()),
                "sha256": hashlib.sha256((RUNS / run / "seed_level.csv").read_bytes()).hexdigest(),
            }
        )
    validation_table = pd.DataFrame(checks)
    validation_table.to_csv(TAB / "validation_audit.csv", index=False)

    dy32_orth = e9_dy32[e9_dy32.method == "ORTH_DIRECT_COV"]
    dy32_low = e9_dy32[e9_dy32.method == "ORTH_DIRECT_COV_LOW_RANK_EFFECTIVE"]
    snapshot = {
        "decision": "go for bounded predictive-feature and typed covariance claims; no-go for unrestricted latent/full-law deployment",
        "new_rows": int(sum(item["rows"] for item in checks)),
        "all_new_validations_passed": bool(validation_table.passed.all()),
        "all_new_failed_rows": int(validation_table.failed_rows.sum()),
        "all_new_duplicate_full_rows": int(validation_table.duplicate_full_rows.sum()),
        "e6_gap4_n128k_density_score_orth": {
            "target_abs_error": mean(e6[e6.cell_id.str.startswith("gap_mu4")], "target_abs_error", "ORTH_DENSITY_SCORE", n_train=128000),
            "alpha_nrmse": mean(e6[e6.cell_id.str.startswith("gap_mu4")], "alpha_nrmse", "ORTH_DENSITY_SCORE", n_train=128000),
            "coverage": mean(e6[e6.cell_id.str.startswith("gap_mu4")], "coverage_fraction", "ORTH_DENSITY_SCORE", n_train=128000),
        },
        "e8_rho098_h64_n32k_mle_orth": {
            "nrmse": mean(e8_mle, "nrmse", "CF_DECONV_MIXTURE_MLE_ORTH", rho=0.98, horizon=64, n_train=32000, frequency_band="wide"),
            "coverage": mean(e8_mle, "coverage_fraction", "CF_DECONV_MIXTURE_MLE_ORTH", rho=0.98, horizon=64, n_train=32000, frequency_band="wide"),
            "rho_abs_error": mean(e8_mle, "rho_abs_error", "CF_DECONV_MIXTURE_MLE_ORTH", rho=0.98, horizon=64, n_train=32000, frequency_band="wide"),
        },
        "e9_identified_dy8_n128k_orth_direct": {
            "covariance_nrmse": mean(e9_ident, "covariance_tensor_nrmse", "ORTH_DIRECT_COV", intercept_scheme="identified", n_train=128000),
            "response_angle": mean(e9_ident, "response_subspace_angle_degrees", "ORTH_DIRECT_COV", intercept_scheme="identified", n_train=128000),
            "source_angle": mean(e9_ident, "source_subspace_angle_degrees", "ORTH_DIRECT_COV", intercept_scheme="identified", n_train=128000),
        },
        "e9_identified_dy32_n32k_30_seed": {
            "orth_direct_nrmse": float(dy32_orth.covariance_tensor_nrmse.mean()),
            "low_rank_nrmse": float(dy32_low.covariance_tensor_nrmse.mean()),
            "orth_coverage": float(dy32_orth.coverage_fraction.mean()),
            "orth_null_fwer": float(dy32_orth.null_edge_familywise_call.mean()),
            "seeds": int(dy32_orth.dgp_seed.nunique()),
        },
        "gate_matrix": gate_matrix.to_dict(orient="records"),
    }
    (HERE / "results_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")


def main() -> None:
    e6_figures()
    e8_figures()
    e9_figures()
    legacy_baseline_figure()
    write_tables_and_snapshot()


if __name__ == "__main__":
    main()
