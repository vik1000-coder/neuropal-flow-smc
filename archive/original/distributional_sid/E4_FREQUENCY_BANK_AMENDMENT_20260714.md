# E4 development amendment: spectral coverage

Recorded before any E4 confirmation seed was run.

The runbook's default characteristic frequency amplitudes
`(0.25, 0.5, 1.0, 2.0)` failed the two-family E4 development gate.  ORTH NRMSE
was 3.46--5.88 at the preregistered medium-amplitude, `n=8000`, `Q=128`
cells; oracle-Riesz weighting was also worse than the zero estimator.  The
first four raw-moment derivatives remained exactly zero.

The preregistered frequency-bandwidth audit was then executed on development
seeds 3 and 4.  It compared:

- default: `(0.25, 0.5, 1.0, 2.0)`;
- wide: `(0.5, 1.0, 2.0, 4.0)`;
- high: `(1.0, 2.0, 4.0, 8.0)`.

The selection rule was frozen in the audit configuration: minimize mean ORTH
NRMSE plus 0.25 times mean absolute calibration error across the two
medium-amplitude, `n=8000` families.  The high bank was selected.  Its mean
NRMSE was 0.300 versus 0.361 for wide and 4.394 for default.

E4 confirmation therefore uses the high bank, the unchanged medium amplitude,
`n=8000`, `Q=128`, three initializations, and seeds 1001--1030.  The default-bank
failure remains a primary reported result.  This amendment changes the tested
feature resolution; it does not alter the DGP amplitude, sample size, estimator,
or success thresholds.

