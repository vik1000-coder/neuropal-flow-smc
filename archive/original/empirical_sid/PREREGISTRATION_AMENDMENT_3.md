# Preregistration amendment 3: raw-array persistence

**Frozen:** 2026-07-14, after 9 of 270 v3 confirmatory cells had executed and before their results were inspected or summarized.

The runbook requires raw evaluation arrays in addition to immutable seed-level metrics. The v3 runner persisted only the seed-level metrics. v3 was therefore stopped and marked `aborted_missing_raw_arrays`; none of its values enter inference.

Version v4 preserves the identical hypotheses, seeds, information calibration, methods, selected S7 setting, metrics, and thresholds. It additionally writes one compressed NPZ per mechanism-seed case, records its relative URI and SHA-256 hash on every metric row, and requires both artifacts before a case can be considered complete or resumable.
