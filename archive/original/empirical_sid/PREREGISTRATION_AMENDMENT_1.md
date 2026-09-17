# Preregistration amendment 1: positive-noise readout correction

**Frozen:** 2026-07-14, after the eight-seed v1 development screen and before any confirmatory seed was evaluated.

The v1 development screen centered positive-noise DSM and anchored-energy mixed fields against the clean response law and integrated their implied clean-score extrapolation. That is not the declared positive-noise object. The screen therefore served as an evaluation-pipeline falsification and is excluded from all confirmatory intervals.

For v2:

1. S5, S6, and S7 at \(\sigma=0.12\) are centered against the matching Gaussian-convolved conditional law.
2. Mean, covariance, third-cumulant, and nuisance-matched fourth-order derivatives are translated using the preregistered Gaussian corruption ledger.
3. Tail and occupancy readouts at positive noise are labeled `local_smoothed` or `finite_smoothed`; they are not placed on the clean leaderboard.
4. Polynomial score coordinates are scaled before fitting to remove a numerical tail-instability found in development.
5. No confirmatory outcomes, seeds, or test labels were observed. Primary seeds, mechanisms, methods, metrics, information calibration, and recovery/advantage thresholds are unchanged.

The original v1 outputs remain immutable at `runs/sid_tier1_primary_20260714/`.
