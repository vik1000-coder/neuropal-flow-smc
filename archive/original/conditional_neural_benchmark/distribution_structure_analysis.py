"""Worm-paired inference and reproducible reporting for the structure audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conditional_neural_benchmark.chemical_encoding_runner import sha256
from conditional_neural_benchmark.distribution_structure_runner import DEFAULT_RUN, atomic_json

LABELS = {"flow": "Flow", "flow_independent": "Flow, shuffled",
          "matched_gaussian_diag": "Gaussian, diagonal",
          "matched_gaussian_diag_fixed": "Gaussian, fixed variance",
          "matched_gaussian_rank8": "Gaussian, rank 8",
          "matched_student_t_rank8": "Student-t, rank 8"}
ORDER = list(LABELS)
PRIMARY = [
    ("flow_shuffle_energy", "flow_independent", "flow", "energy"),
    ("flow_shuffle_variogram", "flow_independent", "flow", "variogram"),
    ("changing_variance_crps", "matched_gaussian_diag_fixed", "matched_gaussian_diag", "crps"),
    ("student_vs_gaussian_crps", "matched_gaussian_rank8", "matched_student_t_rank8", "crps"),
    ("flow_vs_student_energy", "matched_student_t_rank8", "flow", "energy"),
]


def paired_interval(values, seed: int = 91073, replicates: int = 20000):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("expected finite values from at least two worms")
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), size=(replicates, len(x)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(x.mean()), float(low), float(high)


def exact_sign_flip_p(values):
    """Two-sided mean test over one representative of each global sign orbit.

    Studentizing yields the same ordering for a fixed sum of squared effects.
    Exact conditional on sign symmetry, not a biological randomization test.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not 2 <= len(x) <= 20 or not np.isfinite(x).all():
        raise ValueError("exact sign-flip enumeration supports 2--20 finite worms")
    numbers = np.arange(2 ** (len(x) - 1), dtype=np.uint32)
    signs = 2 * ((numbers[:, None] >> np.arange(len(x) - 1)) & 1).astype(float) - 1
    null = (x[0] + signs @ x[1:]) / len(x)
    return float(np.mean(np.abs(null) >= abs(x.mean()) - 1e-14))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1)
    return result


def read_archives(directory: Path):
    records, shuffle_records, inventory, metadata = [], [], [], []
    declared = json.loads((directory / "manifest.json").read_text())["fingerprint"]
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta["fingerprint"] != declared:
                raise RuntimeError("evaluation archive/manifest fingerprint mismatch")
            scores = z["scores"]
            metrics = list(z["metric_names"].astype(str))
            variants = list(z["variants"].astype(str))
            worm, times, strata = z["worm"], z["time"], z["stratum"].astype(str)
            ids = z["worm_ids"].astype(str)
            meta["history_key_sha256"] = hashlib.sha256(
                np.stack([worm, times], axis=1).astype(np.int64).tobytes()
            ).hexdigest()
            if scores.shape != (len(variants), len(worm), len(metrics)) or not np.isfinite(scores).all():
                raise RuntimeError(f"invalid score tensor: {path}")
            expected_nll = len(worm) if meta["model_id"] != "flow" else 0
            expected_fixed = len(worm) if meta["model_id"] == "matched_gaussian_diag" else 0
            for key, size in (("nll", expected_nll), ("fixed_nll", expected_fixed)):
                if z[key].shape != (size,) or not np.isfinite(z[key]).all():
                    raise RuntimeError(f"invalid normalized-density values: {path}, {key}")
            if len([v for v in variants if v.startswith("shuffle_")]) != meta["shuffle_repeats"]:
                raise RuntimeError("archive shuffle count differs from metadata")
            if len(set(zip(worm.tolist(), times.tolist()))) != len(worm):
                raise RuntimeError("duplicate heldout worm/frame")
            if meta["max_marginal_invariance_error"] > 1e-10:
                raise RuntimeError("failed marginal invariance")
            if sha256(Path(meta["checkpoint"])) != meta["checkpoint_sha256"]:
                raise RuntimeError("checkpoint changed after scoring")
            variants_to_use = [(meta["model_id"], scores[variants.index("original")], z["nll"])]
            shuffle = [i for i, v in enumerate(variants) if v.startswith("shuffle_")]
            if shuffle:
                variants_to_use.append(("flow_independent", scores[shuffle].mean(axis=0), np.asarray([])))
            if "fixed_variance" in variants:
                variants_to_use.append((meta["model_id"] + "_fixed", scores[variants.index("fixed_variance")], z["fixed_nll"]))
            for w in np.unique(worm):
                for context in ["all"] + sorted(set(strata[worm == w])):
                    use = (worm == w) & ((strata == context) if context != "all" else True)
                    base = {"fold": meta["fold"], "model_seed": meta["model_seed"],
                            "sample_seed": meta["sample_seed"], "samples": meta["samples"],
                            "worm_id": ids[w], "context": context, "histories": int(use.sum())}
                    for model_id, values, nll in variants_to_use:
                        row = dict(base, model=model_id)
                        row.update({name: float(values[use, i].mean()) for i, name in enumerate(metrics)})
                        row["nll_per_neuron"] = float(nll[use].mean()) if len(nll) else np.nan
                        records.append(row)
                    for repeat, v in enumerate(shuffle):
                        row = dict(base, shuffle_repeat=repeat)
                        row.update({name: float((scores[v, use, i] - scores[0, use, i]).mean())
                                    for i, name in enumerate(metrics)})
                        shuffle_records.append(row)
            inventory.append({"path": str(path), "sha256": sha256(path), "size": path.stat().st_size})
            metadata.append(meta)
    if not records:
        raise RuntimeError(f"no evaluated archives: {directory}")
    return pd.DataFrame(records), pd.DataFrame(shuffle_records), inventory, metadata


def collapse_repetitions(frame):
    id_columns = ["samples", "worm_id", "context", "model"]
    metrics = [c for c in frame if c not in id_columns + ["fold", "model_seed", "sample_seed", "histories"]]
    counts = frame.groupby(id_columns).size().rename("repetitions")
    result = frame.groupby(id_columns, as_index=False)[metrics].mean()
    result = result.merge(counts.reset_index(), on=id_columns, validate="one_to_one")
    # Secondary equal-history-stratum score: equalize strata within each worm.
    balanced = result[result.context != "all"].groupby(["samples", "worm_id", "model"], as_index=False)[metrics].mean()
    balanced["context"], balanced["repetitions"] = "balanced", result.repetitions.max()
    return pd.concat([result, balanced], ignore_index=True)


def make_comparisons(worm):
    rows = []
    extras = [
        ("flow_shuffle_innovation_variogram", "flow_independent", "flow", "innovation_variogram"),
        ("flow_shuffle_crps", "flow_independent", "flow", "crps"),
        ("changing_variance_nll", "matched_gaussian_diag_fixed", "matched_gaussian_diag", "nll_per_neuron"),
        ("gaussian_correlation_energy", "matched_gaussian_diag", "matched_gaussian_rank8", "energy"),
        ("gaussian_correlation_variogram", "matched_gaussian_diag", "matched_gaussian_rank8", "variogram"),
        ("student_vs_gaussian_energy", "matched_gaussian_rank8", "matched_student_t_rank8", "energy"),
        ("student_vs_gaussian_nll", "matched_gaussian_rank8", "matched_student_t_rank8", "nll_per_neuron"),
        ("flow_vs_student_crps", "matched_student_t_rank8", "flow", "crps"),
        ("flow_vs_gaussian_energy", "matched_gaussian_rank8", "flow", "energy"),
        ("flow_vs_gaussian_crps", "matched_gaussian_rank8", "flow", "crps"),
        # Tail Brier was a declared secondary metric; this highlighted pair was
        # added during result review and is not part of the primary test family.
        ("exploratory_student_vs_flow_tail_brier", "flow", "matched_student_t_rank8", "tail_brier"),
    ]
    for context in sorted(set(worm.context)):
        subset = worm[(worm.context == context) & (worm.samples == 128)]
        for name, control, candidate, metric in PRIMARY + extras:
            a = subset[subset.model == control].set_index("worm_id")[metric]
            b = subset[subset.model == candidate].set_index("worm_id")[metric]
            if len(a) == 0 or len(b) == 0:
                continue
            if set(a.index) != set(b.index):
                raise RuntimeError("comparison does not have identical worms")
            delta = (a - b).sort_index()
            if delta.isna().any():
                continue
            mean, low, high = paired_interval(delta.to_numpy())
            rows.append(dict(test=name, context=context, control=control, candidate=candidate,
                             metric=metric, n_worms=len(delta), delta=mean, ci_low=low, ci_high=high,
                             percent=100 * mean / a.mean() if a.mean() != 0 else np.nan,
                             positive_worms=int((delta > 0).sum()),
                             exact_sign_flip_p=exact_sign_flip_p(delta.to_numpy()),
                             primary=context == "all" and name in {p[0] for p in PRIMARY}))
    result = pd.DataFrame(rows)
    result["holm_p"] = np.nan
    mask = result.primary
    if mask.sum() != 5:
        raise RuntimeError("primary five-test family incomplete")
    result.loc[mask, "holm_p"] = holm(result.loc[mask, "exact_sign_flip_p"])
    return result


def validate_coverage(frame, metadata, protocol):
    expected = {(model, fold, seed, sample_seed, 128)
                for model in ["flow"] + [m["model_id"] for m in protocol["models"]]
                for fold in protocol["folds"] for seed in protocol["seeds"]
                for sample_seed in protocol["evaluation"]["sample_seeds"]}
    actual = [(m["model_id"], m["fold"], m["model_seed"], m["sample_seed"], m["samples"]) for m in metadata]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError(f"incomplete evaluation grid; expected {len(expected)}, found {len(actual)}; missing={expected-set(actual)}")
    if pd.DataFrame(metadata).groupby("fold").history_key_sha256.nunique().max() != 1:
        raise RuntimeError("models were scored on different heldout histories")
    original = frame[(frame.context == "all") & ~frame.model.isin(["flow_independent", "matched_gaussian_diag_fixed"])]
    for _, group in original.groupby(["model", "model_seed", "sample_seed"]):
        if len(group) != 17 or set(group.worm_id) != set(protocol["worm_ids"]):
            raise RuntimeError("heldout cohort is not exactly 17 worms")
    if original.groupby("worm_id").histories.nunique().max() != 1:
        raise RuntimeError("models used different numbers of test histories")


def validate_sensitivity(frame, metadata, protocol):
    expected = {(f, s, n) for f in protocol["folds"] for s in protocol["seeds"] for n in (128, 256)}
    actual = [(m["fold"], m["model_seed"], m["samples"]) for m in metadata]
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError("particle sensitivity incomplete")
    if any(m["model_id"] != "flow" or m["sample_seed"] != 731 or m["shuffle_repeats"] != 4 for m in metadata):
        raise RuntimeError("particle sensitivity settings differ from the protocol")
    if pd.DataFrame(metadata).groupby("fold").history_key_sha256.nunique().max() != 1:
        raise RuntimeError("particle counts were scored on different histories")
    original = frame[(frame.context == "all") & (frame.model == "flow")]
    for _, group in original.groupby(["samples", "model_seed"]):
        if len(group) != 17 or set(group.worm_id) != set(protocol["worm_ids"]):
            raise RuntimeError("particle sensitivity cohort is not exactly 17 worms")
    if original.groupby("worm_id").histories.nunique().max() != 1:
        raise RuntimeError("particle sensitivity history counts differ")


def plots(worm, comparisons, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    selected = worm[(worm.context == "all") & (worm.samples == 128)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    colors = ["#175b79", "#b77930", "#5c7284", "#969ea3", "#4c927a", "#976593"]
    for ax, metric, title in zip(axes, ["energy", "crps", "variogram"],
                                  ["Joint energy score", "Marginal CRPS", "Next-state variogram"]):
        reference = selected[selected.model == "flow"].set_index("worm_id")[metric]
        for position, (model, color) in enumerate(zip(ORDER, colors)):
            values = selected[selected.model == model].set_index("worm_id")[metric]
            x = (values - reference).sort_index().to_numpy()
            mean, low, high = paired_interval(x)
            ax.errorbar(mean, position, xerr=[[mean-low], [high-mean]], fmt="o", color=color, capsize=3)
        ax.set_yticks(range(len(ORDER)), [LABELS[m] for m in ORDER])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Score difference from flow")
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Held-out predictive distributions · 17 worms\nPaired 95% worm intervals; positive means worse than flow", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"model_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    names = ["flow_shuffle_energy", "flow_shuffle_variogram", "flow_shuffle_innovation_variogram", "flow_shuffle_crps"]
    labels = ["Joint energy", "Next-state variogram", "Innovation variogram", "Marginal CRPS"]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    for position, (name, label) in enumerate(zip(names, labels)):
        row = comparisons[(comparisons.test == name) & (comparisons.context == "all")].iloc[0]
        denominator = selected[selected.model == row.control][row.metric].mean()
        factor = 100 / denominator
        ax.errorbar(row.delta * factor, position,
                    xerr=[[(row.delta-row.ci_low)*factor], [(row.ci_high-row.delta)*factor]],
                    fmt="o", capsize=4, color="#175b79")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(names)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Score increase after shuffling (% of shuffled score); positive favors original")
    ax.set_title("What joint dependence contributes, with empirical marginals fixed")
    ax.grid(axis="x", alpha=0.15)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"dependence_ablation.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(output, worm, comparisons, metadata, protocol, sensitivity):
    selected = worm[(worm.context == "all") & (worm.samples == 128)]
    metrics = ["energy", "crps", "variogram", "innovation_variogram", "rmse", "coverage90", "sharpness90", "tail_brier", "tail_probability", "tail_frequency", "nll_per_neuron"]
    board = selected.groupby("model")[metrics].mean().reindex(ORDER)
    board.to_csv(output / "model_scoreboard.csv")
    unique_fits = pd.DataFrame(metadata).drop_duplicates("checkpoint")
    fit_summary = unique_fits.groupby("model_id").agg(
        checkpoints=("checkpoint", "size"), parameters=("parameter_count", "first"),
        best_epoch_min=("best_epoch", "min"), best_epoch_max=("best_epoch", "max"),
        stopped_epoch_min=("stopped_epoch", "min"), stopped_epoch_max=("stopped_epoch", "max"))
    fit_summary.to_csv(output / "training_diagnostics.csv")
    primary = comparisons[comparisons.primary]
    df_values = [m["student_df"] for m in metadata if m["student_df"] is not None]
    snapshot = {"primary_tests": primary.replace({np.nan: None}).to_dict("records"),
                "scoreboard": board.reset_index().replace({np.nan: None}).to_dict("records"),
                "student_df_min": min(df_values), "student_df_max": max(df_values),
                "max_marginal_invariance_error": max(m["max_marginal_invariance_error"] for m in metadata),
                "independent_worms": 17, "model_seeds": protocol["seeds"],
                "sampling_seeds": protocol["evaluation"]["sample_seeds"],
                "particle_sensitivity_complete": sensitivity is not None}
    atomic_json(output / "results_summary.json", snapshot)
    natural = comparisons[comparisons.context == "all"].set_index("test")
    dep = natural.loc["flow_shuffle_energy"]
    variance = natural.loc["changing_variance_crps"]
    tails = natural.loc["student_vs_gaussian_crps"]
    full_gain = board.loc["matched_gaussian_rank8", "energy"] - board.loc["flow", "energy"]
    shuffled_gain = board.loc["matched_gaussian_rank8", "energy"] - board.loc["flow_independent", "energy"]
    lines = ["# Marginal uncertainty, joint dependence, and distribution shape", "",
             "This is a frozen-configuration follow-up on the corrected 17-worm, 54-neuron OH16230 head cohort at 4 Hz. Every score is computed on a worm excluded from that model's training and validation sets. It is not independent confirmation on new animals.", "",
             f"Removing generated dependence changes energy by **{dep.delta:+.6g}** (shuffled minus original; 95% worm interval **[{dep.ci_low:+.6g}, {dep.ci_high:+.6g}]**), while marginal scores are unchanged by construction and numerical validation. The correlated Gaussian's energy gap to flow is **{full_gain:+.6g}** before shuffling and **{shuffled_gain:+.6g}** afterward. This is a score comparison, not an additive decomposition of information.", "",
             f"Holding the fitted Gaussian mean fixed, allowing changing variance changes marginal CRPS by **{variance.delta:+.6g}** in improvement units (constant minus conditional). Replacing the correlated Gaussian with Student-t changes marginal CRPS by **{tails.delta:+.6g}** in improvement units (Gaussian minus Student-t). The paired intervals and multiplicity-adjusted evidence are below; independently fitted means prevent unique attribution of the full-model ranking to shape.", "",
             "## Main comparisons", "",
             "Positive differences favor the candidate. Intervals are paired percentile bootstrap intervals over 17 worms after averaging both model seeds, both sample seeds, and (where applicable) all eight shuffles within worm. Five declared primary tests receive Holm correction. Exact sign-flip p-values assume symmetric worm-level differences; overlapping cross-validation training sets limit their interpretation to fitted-model sensitivity. Evidence declarations follow the adjusted primary test, not whether a percentile interval alone crosses zero; those are different inferential procedures.", "",
             "| Question / metric | Control minus candidate | 95% worm interval | Positive worms | Holm p |", "| --- | ---: | --- | ---: | ---: |"]
    for row in primary.itertuples():
        lines.append(f"| {row.test} | {row.delta:+.6g} | [{row.ci_low:+.6g}, {row.ci_high:+.6g}] | {row.positive_worms}/17 | {row.holm_p:.5g} |")
    lines += ["", "## Model scoreboard", "", "Lower scores are better; coverage should approach 0.90, with interval width considered alongside it. Coverage uses finite-sample quantiles, so proximity to 0.90 is not proof of calibration. RMSE averages the per-history root mean squared error of the sample mean. All rows have the same held-out histories and equal animal weights. No flow likelihood is fabricated.", "",
              "| Model | Energy | Marginal CRPS | Variogram | RMSE | Coverage90 | Width90 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, row in board.iterrows():
        lines.append(f"| {LABELS[model]} | {row.energy:.6f} | {row.crps:.6f} | {row.variogram:.6f} | {row.rmse:.6f} | {row.coverage90:.4f} | {row.sharpness90:.4f} |")
    lines += ["", "## Secondary tail-event check", "",
              "The flow is not uniformly best calibrated. Tail Brier was a predeclared secondary metric; the highlighted Student-t versus flow contrast was added during result review and is exploratory, outside the primary five-test family. The event is |next-current| > 2 × the training-fold RMS innovation for that neuron.", "",
              "| Model | Mean predicted event probability | Observed event frequency | Tail Brier (lower better) |",
              "| --- | ---: | ---: | ---: |"]
    for model in ("flow", "matched_gaussian_rank8", "matched_student_t_rank8"):
        row = board.loc[model]
        lines.append(f"| {LABELS[model]} | {100*row.tail_probability:.2f}% | {100*row.tail_frequency:.2f}% | {row.tail_brier:.6f} |")
    lines += ["", "These are averages over histories and neurons, with equal worm weights. They do not by themselves assess calibration separately in every context. A good aggregate energy score is not sufficient validation for tail-based downstream claims.", "",
              "![Model scores](model_comparison.png)", "", "![Dependence ablation](dependence_ablation.png)", "",
              "## What the design isolates", "",
              "- **Dependence:** independent sample-index permutations within each history and neuron preserve every univariate empirical distribution. No observed history, target, animal, or marginal value is exchanged. Marginal CRPS, coverage, interval width, tail probability, and sample means remain invariant. A common permutation of whole vectors is an additional exact joint-score control.",
              "- **Changing uncertainty:** the fixed-variance Gaussian uses exactly the same fitted conditional mean as the diagonal Gaussian. Constant per-neuron SDs are fitted only to training-fold residuals. This tests predictable residual uncertainty around that fitted mean, not intrinsic biological noise.",
              "- **Correlation:** the rank-8 Gaussian is an explicit correlated competitor, so a flow advantage cannot be attributed solely to comparing a joint model against independent marginals.",
              "- **Heavy tails:** Gaussian and Student-t rank-8 models have the same TCN, head depth/width, mean and covariance parameterization, and initialization for shared parameters. Student-t adds one global fitted degrees-of-freedom parameter constrained to (2.1,50); its scale is normalized to hold the declared covariance fixed.",
              "- **Remaining confounding:** the independently trained density heads can learn different means and covariance functions. A flow win does not uniquely demonstrate multimodality, non-elliptical shape, or intrinsic neuronal stochasticity. Rank 8 is a fixed capacity restriction, not an inferred biological dimension.", "",
              "## Training and scoring", "",
              "All fitted alternatives use the existing residual target, 80-frame neural/binary-stimulus history, width-128 TCN, 15% dropout, 0.01 neural-history jitter (stimulus untouched), learning rate 3e-4, weight decay 7.5e-4, batch size 256, 50-epoch maximum, and patience 9. Native validation losses select checkpoints on the same whole-worm validation partitions. The encoders share architecture, not fitted weights. Frozen flow checkpoints are reused byte-for-byte.", "",
              "The training objective remains family-specific: flow matching for flow and exact negative log-likelihood for Gaussian/Student-t. Identical epoch ceilings and patience do not imply identical compute or fitted mean quality. Parameter counts and selected/stopping epochs are retained in [training diagnostics](training_diagnostics.csv); this is a matched protocol comparison, not an exhaustive retuning of every family.", "",
              "Energy and marginal CRPS use all distinct sample pairs with the fair finite-ensemble correction. Variogram uses every neuron pair, power 0.5, and subtracts the usual finite-ensemble mean-estimation variance; the naive score and innovation-only variogram are retained as diagnostics. Brier scores use the training-fixed event |next-current| > 2 × training RMS innovation. Every primary sample bank contains 128 draws. Shuffling approximates independence at finite N and does not enforce zero sample covariance.", "",
              f"Fitted Student-t degrees of freedom range from {min(df_values):.3f} to {max(df_values):.3f}. Maximum marginal-score invariance error is {snapshot['max_marginal_invariance_error']:.3g}.", "",
              "The full five-fold results are primary; equal-history-stratum and individual context results are secondary. Different score estimators, animal weighting, and evaluation draws mean these absolute numbers must not be interpreted as improvements over earlier leaderboard numbers.", "",
              "## Numerical sensitivity", ""]
    if sensitivity is not None:
        lines += ["The predeclared N=128/256 sensitivity uses the same data-only selected histories (up to 16 per worm × history stratum), both generator seeds, one sampling seed, and four shuffles. These are Monte Carlo controls, not additional animals.", "",
                  "| N | Metric | Shuffled minus original | 95% worm interval |", "| ---: | --- | ---: | --- |"]
        for n in sorted(sensitivity.samples.unique()):
            sub = sensitivity[(sensitivity.samples == n) & (sensitivity.context == "all")]
            for metric in ("energy", "variogram", "innovation_variogram", "crps"):
                a = sub[sub.model == "flow_independent"].set_index("worm_id")[metric]
                b = sub[sub.model == "flow"].set_index("worm_id")[metric]
                mean, low, high = paired_interval((a-b).to_numpy())
                lines.append(f"| {n} | {metric} | {mean:+.6g} | [{low:+.6g}, {high:+.6g}] |")
    else:
        lines.append("Particle sensitivity has not completed; do not treat the primary finite-particle effect as numerically confirmed.")
    lines += ["", "## Scope", "",
              "These one-step tests diagnose predictive distributions of observed calcium. They do not establish physical noise sources, latent state dimension, long-rollout adequacy, causal coupling, anatomy, or receptor action. The existing dataset has been used for earlier model development; these intervals do not remove that history of selection. No atlas edge weights or correspondence outcomes enter fitting, stopping, scoring, or selection. The inherited cohort loader uses Cook node names for its neuron vocabulary; the fixed 54-neuron roster is unchanged.", "",
              "## Known-law controls and validation", "",
              "Four additional synthetic controls use known distributions, with no fitted models: independent Gaussian, correlated Gaussian, history-dependent Gaussian variance, and covariance-matched Student-t tails. They confirm that shuffling preserves marginal scores, detects a known joint advantage, and does not create a material energy effect for independent samples; the other controls detect predictable variance and heavy-tail improvements. These are diagnostic checks, not additional biological evidence. Details are in [synthetic control scores](../synthetic_controls/scores.csv) and [their validation](../synthetic_controls/validation.json).", "",
              "The completed run is sealed only after all 40 checkpoints replay their saved scores, all 30 new fitted densities agree with independent dense SciPy calculations, the regression suite passes, and frozen source/checkpoint hashes validate. See [final validation](../VALIDATION.json).", "",
              "## Reproduce", "", "```bash",
              ".venv/bin/python -m conditional_neural_benchmark.distribution_structure_runner",
              ".venv/bin/python -m conditional_neural_benchmark.distribution_structure_evaluate --device cpu",
              ".venv/bin/python -m conditional_neural_benchmark.distribution_structure_evaluate --device cpu --models flow --sensitivity",
              ".venv/bin/python -m conditional_neural_benchmark.distribution_structure_controls",
              ".venv/bin/python -m conditional_neural_benchmark.distribution_structure_analysis",
              ".venv/bin/python -m conditional_neural_benchmark.distribution_structure_validate",
              ".venv/bin/python -m pytest conditional_neural_benchmark/tests compatibility_neural_benchmark/tests -q > results/distribution_structure_20260831/analysis/unit_test_results.txt 2>&1",
              ".venv/bin/python -m conditional_neural_benchmark.distribution_structure_finalize", "```", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def run(run_dir):
    protocol = json.loads((run_dir / "protocol.json").read_text())
    if not (run_dir / "training_validation.json").exists():
        raise RuntimeError("training is incomplete")
    frame, shuffles, inventory, metadata = read_archives(run_dir / "evaluation")
    validate_coverage(frame, metadata, protocol)
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)
    frame.to_csv(output / "worm_replicate_metrics.csv", index=False)
    shuffles.to_csv(output / "shuffle_replicate_metrics.csv", index=False)
    worm = collapse_repetitions(frame)
    if worm.repetitions.min() != 4 or worm.repetitions.max() != 4:
        raise RuntimeError("model/sample replication count differs")
    worm.to_csv(output / "worm_metrics.csv", index=False)
    comparisons = make_comparisons(worm)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    sensitivity = None
    if (run_dir / "particle_sensitivity").exists():
        sf, ss, si, sm = read_archives(run_dir / "particle_sensitivity")
        validate_sensitivity(sf, sm, protocol)
        sensitivity = collapse_repetitions(sf)
        if sensitivity.repetitions.min() != 2 or sensitivity.repetitions.max() != 2:
            raise RuntimeError("particle sensitivity replication count differs")
        sensitivity.to_csv(output / "particle_sensitivity_worm_metrics.csv", index=False)
        inventory += si
    pd.DataFrame(metadata).to_csv(output / "checkpoint_diagnostics.csv", index=False)
    pd.DataFrame(inventory).to_csv(output / "raw_archive_inventory.csv", index=False)
    plots(worm, comparisons, output)
    write_report(output, worm, comparisons, metadata, protocol, sensitivity)
    atomic_json(output / "validation.json", {"status": "pass", "archives": len(metadata),
                "expected_archives": 80, "worms": 17, "primary_tests": 5,
                "max_marginal_invariance_error": max(m["max_marginal_invariance_error"] for m in metadata),
                "particle_sensitivity_complete": sensitivity is not None})
    print(comparisons[comparisons.primary].to_string(index=False))
    print(f"REPORT_READY {output / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run(args.run_dir.resolve())


if __name__ == "__main__":
    main()
