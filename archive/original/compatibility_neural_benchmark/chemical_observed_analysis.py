from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from conditional_neural_benchmark.data import load_cohort


HORIZON_FRAMES = (1, 2, 4, 8, 16, 40)


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    ordered = finite[np.argsort(values[finite])]
    adjusted = values[ordered] * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[ordered] = np.minimum(adjusted, 1.0)
    return result


def _window_change(trace: np.ndarray, cut: int, horizon: int, baseline: int) -> np.ndarray:
    if cut - baseline < 0 or cut + horizon > len(trace):
        return np.full(trace.shape[1], np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(trace[cut : cut + horizon], axis=0) - np.nanmean(
            trace[cut - baseline : cut], axis=0
        )


def event_responses(cohort) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline = int(round(cohort.fps))
    quiet_offset = int(round(15.0 * cohort.fps))
    for worm, (worm_id, trace, schedule) in enumerate(
        zip(cohort.worm_ids, cohort.traces, cohort.stimulus_schedules)
    ):
        active = np.zeros(len(trace), dtype=bool)
        for start, end in schedule.event_intervals_seconds:
            active[
                int(round(start * cohort.fps)) : int(round(end * cohort.fps))
            ] = True
        background_scale = np.nanstd(np.asarray(trace)[~active], axis=0)
        background_scale = np.maximum(background_scale, 0.05)
        for event, (interval, code, chemical) in enumerate(
            zip(
                schedule.event_intervals_seconds,
                schedule.chemical_code_by_event,
                schedule.chemical_name_by_event,
            )
        ):
            onset = int(round(interval[0] * cohort.fps))
            quiet = onset - quiet_offset
            for horizon in HORIZON_FRAMES:
                onset_change = _window_change(trace, onset, horizon, baseline)
                quiet_change = _window_change(trace, quiet, horizon, baseline)
                contrast = onset_change - quiet_change
                for neuron, value, scale, onset_value, quiet_value in zip(
                    cohort.neurons, contrast, background_scale, onset_change, quiet_change
                ):
                    rows.append(
                        {
                            "cohort": cohort.cohort_mode,
                            "worm_index": worm,
                            "worm_id": worm_id,
                            "strain": schedule.strain,
                            "source_recording": schedule.source_recording,
                            "event_position": event + 1,
                            "chemical_code": code,
                            "chemical": chemical,
                            "neuron": str(neuron),
                            "horizon_frames": horizon,
                            "horizon_seconds": horizon / cohort.fps,
                            "onset_change_raw": float(onset_value),
                            "matched_quiet_change_raw": float(quiet_value),
                            "onset_minus_quiet_raw": float(value),
                            "background_sd": float(scale),
                            "onset_minus_quiet_z": float(value / scale),
                        }
                    )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, seed: int = 20_260_828) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    group_keys = ["cohort", "chemical", "horizon_frames", "horizon_seconds", "neuron"]
    for keys, group in frame.groupby(group_keys, sort=False):
        values = group.onset_minus_quiet_z.to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        draws = values[
            rng.integers(0, len(values), size=(5000, len(values)))
        ].mean(axis=1)
        if len(values) > 1 and np.std(values) > 0:
            p_value = float(ttest_1samp(values, 0.0).pvalue)
        else:
            p_value = 1.0 if values.mean() == 0 else np.nan
        rows.append(
            {
                **dict(zip(group_keys, keys)),
                "n_worms": len(values),
                "mean_onset_minus_quiet_z": float(values.mean()),
                "median_onset_minus_quiet_z": float(np.median(values)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "worm_sign_agreement": float(np.mean(np.sign(values) == np.sign(values.mean()))),
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["bh_q_value"] = np.nan
    for _, index in result.groupby(
        ["cohort", "chemical", "horizon_frames"]
    ).groups.items():
        result.loc[index, "bh_q_value"] = bh_adjust(result.loc[index, "p_value"])
    return result


def write_report(output: Path, responses: pd.DataFrame, summary: pd.DataFrame) -> None:
    designs = (
        responses[["cohort", "worm_id", "chemical", "event_position"]]
        .drop_duplicates()
        .groupby(["cohort", "chemical", "event_position"])
        .size()
        .rename("n_worms")
        .reset_index()
    )
    designs.to_csv(output / "chemical_by_position_counts.csv", index=False)
    awc = summary[summary.neuron == "AWC"].copy()
    awc.to_csv(output / "awc_actual_chemical_events.csv", index=False)
    discoveries = summary[summary.bh_q_value < 0.05].sort_values(
        ["cohort", "chemical", "horizon_seconds", "bh_q_value"]
    )
    discoveries.to_csv(output / "fdr_discoveries.csv", index=False)
    lines = [
        "# Observed chemical-specific NeuroPAL response analysis",
        "",
        "## What this fixes",
        "",
        "Every event is selected by its raw per-worm chemical code. Butanone, pentanedione, and NaCl are analyzed separately; there is one exposure to each chemical per animal, so no repetition or adaptation claim is tested.",
        "",
        "The descriptive estimand is the activity change after the true onset minus the same change at a pseudo-onset 15 seconds earlier in that worm/event. Each change is relative to its immediately preceding one-second baseline and divided by that worm/neuron's non-stimulus standard deviation. This is observed activity, not an inferred inter-neuron edge.",
        "",
        "## AWC by actual chemical",
        "",
        "| Cohort | Chemical | Horizon (s) | Mean z contrast | 95% bootstrap CI | BH q | n |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in awc.sort_values(["cohort", "chemical", "horizon_seconds"]).itertuples():
        lines.append(
            f"| {row.cohort} | {row.chemical} | {row.horizon_seconds:g} | "
            f"{row.mean_onset_minus_quiet_z:.3f} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] | "
            f"{row.bh_q_value:.3g} | {row.n_worms} |"
        )
    lines.extend(
        [
            "",
            f"Across all neuron×chemical×horizon summaries, {len(discoveries)} rows pass BH q<0.05 when correction is performed separately within each cohort×chemical×horizon family of 54 neurons.",
            "",
            "The OH16230-head cohort at native 4.0 Hz is primary. The pooled-head cohort is a sensitivity analysis in which OH15500 is explicitly linearly resampled from 4.1 to 4.0 Hz and its true 20-second third epoch is retained.",
            "",
            "These data can establish stimulus-associated activity changes. They cannot, by themselves, identify a synapse, receptor-mediated effect, causal inter-neuron influence, or physical transmission delay.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    responses = []
    manifests = {}
    for mode in ("oh16230_head", "pooled_resampled"):
        cohort = load_cohort(cohort_mode=mode)
        responses.append(event_responses(cohort))
        manifests[mode] = {
            "n_worms": cohort.n_worms,
            "n_neurons": cohort.n_neurons,
            "analysis_fps": cohort.fps,
            "stimulus_schema_version": cohort.stimulus_schema_version,
            "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        }
    response_frame = pd.concat(responses, ignore_index=True)
    summary = summarize(response_frame)
    response_frame.to_csv(output / "worm_event_neuron_responses.csv", index=False)
    summary.to_csv(output / "neuron_response_statistics.csv", index=False)
    write_report(output, response_frame, summary)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                "cohorts": manifests,
                "event_selection": "raw per-worm 1-based chemical code",
                "estimand": "true-onset change minus matched quiet pseudo-onset change",
                "quiet_offset_seconds": 15.0,
                "baseline_seconds": 1.0,
                "horizon_frames": list(HORIZON_FRAMES),
                "claim_boundary": "observed activity response; not inter-neuron causality, anatomy, receptor action, or physical delay",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
