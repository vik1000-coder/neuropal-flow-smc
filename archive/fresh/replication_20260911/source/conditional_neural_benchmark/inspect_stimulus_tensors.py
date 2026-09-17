from __future__ import annotations

import argparse
import json
from pathlib import Path

from conditional_neural_benchmark.data import _stimulus_features, load_cohort


def inspect() -> dict:
    cohort = load_cohort(cohort_mode="pooled_resampled")
    requested = (
        "OH16230:0924_01",
        "OH16230:0928_01",
        "OH16230:0929_03",
        "OH15500:22_head_run101",
        "OH15500:66_head_run101",
    )
    rows = []
    for worm_id in requested:
        worm = cohort.worm_ids.index(worm_id)
        schedule = cohort.stimulus_schedules[worm]
        chemical = _stimulus_features(
            len(cohort.traces[worm]), schedule, "chemical_onehot"
        )
        position = _stimulus_features(
            len(cohort.traces[worm]), schedule, "position_onehot"
        )
        for event, (start, end) in enumerate(schedule.event_intervals_seconds):
            onset = round(start * schedule.analysis_fps)
            offset = round(end * schedule.analysis_fps)
            expected = [0.0, 0.0, 0.0]
            expected[schedule.chemical_code_by_event[event] - 1] = 1.0
            assert chemical[onset].tolist() == expected
            assert chemical[onset - 1].sum() == 0
            assert chemical[offset].sum() == 0
            rows.append({
                "worm_id": worm_id,
                "strain": schedule.strain,
                "native_fps": schedule.native_fps,
                "analysis_fps": schedule.analysis_fps,
                "event_position": event + 1,
                "interval_seconds": [start, end],
                "chemical_code": schedule.chemical_code_by_event[event],
                "chemical_name": schedule.chemical_name_by_event[event],
                "chemical_vector_at_onset": chemical[onset].tolist(),
                "position_vector_at_onset": position[onset].tolist(),
                "vector_before_onset": chemical[onset - 1].tolist(),
                "vector_at_half_open_offset": chemical[offset].tolist(),
            })
    return {
        "status": "pass",
        "stimulus_schema_version": cohort.stimulus_schema_version,
        "stimulus_schema_fingerprint": cohort.stimulus_schema_fingerprint,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
