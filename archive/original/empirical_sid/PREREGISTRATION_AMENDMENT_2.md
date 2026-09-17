# Preregistration amendment 2: frozen anchored-score setting

**Frozen:** 2026-07-14, after the eight-seed development-only tuning tournament and before any confirmatory seed was evaluated.

The tuning grid compared ridge values \(\{0.001,0.01,0.1,1.0\}\) and three response-noise schedules. One setting was selected globally across M1, M3 cubic, M3 local, and M4 quartic using minimum mean typed NRMSE over all eight development seeds. No per-mechanism setting was selected.

The frozen setting is:

- ridge penalty: 1.0;
- response-noise schedule: \(\{0.03,0.06,0.12,0.25\}\);
- primary evaluation scale: \(\sigma=0.12\).

Its development mean typed NRMSE was 1.322, median 0.971, and 90th percentile 2.752. Thus the screen did not establish recovery; the setting is carried forward to measure the failure precisely rather than dropping or retuning the method. Confirmatory seeds 2001-2030 remained unobserved.
