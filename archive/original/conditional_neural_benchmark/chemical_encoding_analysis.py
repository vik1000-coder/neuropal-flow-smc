from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp


PRIMARY_METRIC = "energy__worm_chemical_balanced"
PAIR_KEYS = ("fold", "seed")
CONTROLS = (
    "binary_any_stimulus",
    "position_onehot",
    "chemical_onehot_subject_shuffle",
)
CANDIDATES = ("chemical_onehot", "chemical_plus_position_onehot")
EXPECTED_ENCODINGS = (*CONTROLS[:2], "chemical_scalar", *CANDIDATES, CONTROLS[2])
CHEMICAL_METRICS = (
    "energy__chemical_butanone",
    "energy__chemical_pentanedione",
    "energy__chemical_nacl",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_complete_run(run_dir: Path) -> pd.DataFrame:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"tournament is not complete: {run_dir}")
    frame = pd.read_csv(run_dir / "trial_metrics.csv")
    frame = frame[(frame.status == "ok") & (frame.phase == "chemical_full_cv")].copy()
    if frame.empty:
        raise RuntimeError(f"no completed full-CV trials in {run_dir}")
    required = {*PAIR_KEYS, "stimulus_encoding", PRIMARY_METRIC, "energy", *CHEMICAL_METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"missing tournament columns: {sorted(missing)}")
    encodings = set(frame.stimulus_encoding.astype(str))
    missing_encodings = set(EXPECTED_ENCODINGS).difference(encodings)
    if missing_encodings:
        raise RuntimeError(f"missing encodings: {sorted(missing_encodings)}")
    duplicated = frame.duplicated(["stimulus_encoding", *PAIR_KEYS])
    if duplicated.any():
        raise RuntimeError("duplicate encoding/fold/seed rows")
    reference_pairs: set[tuple[int, int]] | None = None
    for encoding in EXPECTED_ENCODINGS:
        local = frame[frame.stimulus_encoding == encoding]
        pairs = set(zip(local.fold.astype(int), local.seed.astype(int)))
        if reference_pairs is None:
            reference_pairs = pairs
        elif pairs != reference_pairs:
            raise RuntimeError(f"unpaired fold/seed set for {encoding}")
    if reference_pairs is None or len(reference_pairs) != 10:
        raise RuntimeError(f"expected 10 paired fold/seed trials, found {len(reference_pairs or [])}")
    frame.attrs["manifest"] = manifest
    return frame


def paired_comparisons(
    frame: pd.DataFrame,
    *,
    cohort_label: str,
    bootstrap_draws: int = 20_000,
    seed: int = 20_260_828,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for candidate in CANDIDATES:
        candidate_rows = frame[frame.stimulus_encoding == candidate][
            [*PAIR_KEYS, PRIMARY_METRIC]
        ].rename(columns={PRIMARY_METRIC: "candidate_score"})
        for control in CONTROLS:
            control_rows = frame[frame.stimulus_encoding == control][
                [*PAIR_KEYS, PRIMARY_METRIC]
            ].rename(columns={PRIMARY_METRIC: "control_score"})
            paired = candidate_rows.merge(control_rows, on=list(PAIR_KEYS), validate="one_to_one")
            delta = (paired.candidate_score - paired.control_score).to_numpy(float)
            bootstrap = delta[
                rng.integers(0, len(delta), size=(bootstrap_draws, len(delta)))
            ].mean(axis=1)
            wins = int(np.sum(delta < 0))
            ties = int(np.sum(delta == 0))
            non_ties = len(delta) - ties
            if np.std(delta) == 0:
                paired_t_p = 1.0 if delta.mean() == 0 else 0.0
            else:
                paired_t_p = float(ttest_1samp(delta, 0.0).pvalue)
            records.append(
                {
                    "cohort": cohort_label,
                    "candidate": candidate,
                    "control": control,
                    "metric": PRIMARY_METRIC,
                    "delta_definition": "candidate_minus_control; lower_is_better",
                    "n_pairs": len(delta),
                    "mean_candidate": float(paired.candidate_score.mean()),
                    "mean_control": float(paired.control_score.mean()),
                    "mean_paired_delta": float(delta.mean()),
                    "median_paired_delta": float(np.median(delta)),
                    "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "paired_wins": wins,
                    "paired_losses": int(np.sum(delta > 0)),
                    "paired_ties": ties,
                    "sign_test_p_two_sided": (
                        float(binomtest(wins, non_ties, 0.5).pvalue) if non_ties else 1.0
                    ),
                    "paired_ttest_p_two_sided": paired_t_p,
                    "gate_mean_improves": bool(delta.mean() < 0),
                    "gate_majority_wins": bool(wins >= 6),
                    "gate_pair_pass": bool(delta.mean() < 0 and wins >= 6),
                }
            )
    return pd.DataFrame(records)


def gate_summary(comparisons: pd.DataFrame, cohort_label: str) -> pd.DataFrame:
    local = comparisons[comparisons.cohort == cohort_label]
    records = []
    for candidate in CANDIDATES:
        rows = local[local.candidate == candidate]
        if set(rows.control) != set(CONTROLS):
            raise RuntimeError(f"incomplete controls for {candidate}")
        records.append(
            {
                "cohort": cohort_label,
                "candidate": candidate,
                "required_controls": ";".join(CONTROLS),
                "comparisons_passed": int(rows.gate_pair_pass.sum()),
                "comparisons_required": len(CONTROLS),
                "eligible_for_lag_sampling": bool(rows.gate_pair_pass.all()),
            }
        )
    return pd.DataFrame(records)


def score_summary(frame: pd.DataFrame, cohort_label: str) -> pd.DataFrame:
    metrics = (PRIMARY_METRIC, "energy", *CHEMICAL_METRICS)
    summary = frame.groupby("stimulus_encoding")[list(metrics)].agg(["mean", "std", "count"])
    summary.columns = ["__".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.insert(0, "cohort", cohort_label)
    summary["sensitivity_only"] = summary.stimulus_encoding == "chemical_scalar"
    return summary.sort_values(f"{PRIMARY_METRIC}__mean")


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def write_report(
    output_dir: Path,
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    gates: pd.DataFrame,
    primary_label: str,
) -> None:
    primary_gate = gates[gates.cohort == primary_label]
    eligible = primary_gate[primary_gate.eligible_for_lag_sampling].candidate.tolist()
    primary_scores = summaries[summaries.cohort == primary_label]
    selected = None
    if eligible:
        selected = (
            primary_scores[primary_scores.stimulus_encoding.isin(eligible)]
            .sort_values(f"{PRIMARY_METRIC}__mean")
            .iloc[0]
            .stimulus_encoding
        )
    lines = [
        "# Corrected chemical-encoding paired gate",
        "",
        "## Decision",
        "",
    ]
    if selected is None:
        lines.append(
            "**No chemical-aware encoding passed the frozen predictive gate.** Corrected lag sampling must not be launched from this experiment."
        )
    else:
        lines.append(
            f"**`{selected}` passed the frozen predictive gate and is selected atlas-blind for corrected lag sampling.**"
        )
    lines.extend(
        [
            "",
            "The primary score is held-out energy distance averaged equally over worm×chemical cells (lower is better). A candidate passes a control only when its mean paired fold/seed delta is negative and it wins at least 6 of 10 pairs. It must pass binary, position-only, and independently subject-shuffled chemical controls. This rule was frozen while the first full tournament trial was still running.",
            "",
        ]
    )
    for cohort in summaries.cohort.unique():
        lines.extend(
            [
                f"## {cohort}",
                "",
                "| Encoding | Balanced energy | Overall energy | Butanone | Pentanedione | NaCl |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summaries[summaries.cohort == cohort].itertuples():
            lines.append(
                f"| `{row.stimulus_encoding}` | {_fmt(getattr(row, PRIMARY_METRIC + '__mean'))} | "
                f"{_fmt(row.energy__mean)} | {_fmt(row.energy__chemical_butanone__mean)} | "
                f"{_fmt(row.energy__chemical_pentanedione__mean)} | {_fmt(row.energy__chemical_nacl__mean)} |"
            )
        lines.extend(
            [
                "",
                "| Candidate | Control | Mean paired Δ | 95% bootstrap CI | Wins/10 | Pass |",
                "| --- | --- | ---: | ---: | ---: | :---: |",
            ]
        )
        for row in comparisons[comparisons.cohort == cohort].itertuples():
            lines.append(
                f"| `{row.candidate}` | `{row.control}` | {_fmt(row.mean_paired_delta)} | "
                f"[{_fmt(row.bootstrap_ci95_low)}, {_fmt(row.bootstrap_ci95_high)}] | "
                f"{row.paired_wins}/{row.n_pairs} | {'yes' if row.gate_pair_pass else 'no'} |"
            )
        lines.append("")
    lines.extend(
        [
            "Chemical scalar is reported only as a sensitivity arm and is never gate-eligible. The pooled cohort is a resampling sensitivity; the clean OH16230-head cohort controls the launch decision. No Cook, Randi, SBTG, receptor, or neuromodulator map entered training, ranking, or this gate.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    decision = {
        "created_utc": _utc_now(),
        "primary_cohort": primary_label,
        "primary_metric": PRIMARY_METRIC,
        "lower_is_better": True,
        "pair_keys": list(PAIR_KEYS),
        "minimum_pair_wins": 6,
        "required_controls": list(CONTROLS),
        "eligible_candidates": eligible,
        "selected_candidate": selected,
        "lag_sampling_authorized_by_predictive_gate": selected is not None,
        "selection_firewall": "atlas blind; no anatomical, receptor, Cook, Randi, or SBTG target",
    }
    (output_dir / "gate_decision.json").write_text(json.dumps(decision, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--pooled-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_specs = [("OH16230 head, native 4.0 Hz (primary)", args.primary_run.resolve())]
    if args.pooled_run is not None:
        run_specs.append(("Pooled head, OH15500 resampled 4.1→4.0 Hz (sensitivity)", args.pooled_run.resolve()))
    summaries = []
    comparisons = []
    gates = []
    provenance = {}
    for index, (label, run_dir) in enumerate(run_specs):
        frame = load_complete_run(run_dir)
        summaries.append(score_summary(frame, label))
        comparisons.append(paired_comparisons(frame, cohort_label=label, seed=20_260_828 + index))
        provenance[label] = {
            "run_dir": str(run_dir),
            "stimulus_schema_version": frame.attrs["manifest"].get("stimulus_schema_version"),
            "stimulus_schema_fingerprint": frame.attrs["manifest"].get("stimulus_schema_fingerprint"),
        }
    comparison_frame = pd.concat(comparisons, ignore_index=True)
    for label, _ in run_specs:
        gates.append(gate_summary(comparison_frame, label))
    summary_frame = pd.concat(summaries, ignore_index=True)
    gate_frame = pd.concat(gates, ignore_index=True)
    summary_frame.to_csv(output_dir / "score_summary.csv", index=False)
    comparison_frame.to_csv(output_dir / "paired_comparisons.csv", index=False)
    gate_frame.to_csv(output_dir / "gate_summary.csv", index=False)
    (output_dir / "provenance.json").write_text(json.dumps({
        "created_utc": _utc_now(),
        "runs": provenance,
        "analysis_module": __name__,
    }, indent=2) + "\n")
    write_report(
        output_dir, summary_frame, comparison_frame, gate_frame, run_specs[0][0]
    )


if __name__ == "__main__":
    main()
