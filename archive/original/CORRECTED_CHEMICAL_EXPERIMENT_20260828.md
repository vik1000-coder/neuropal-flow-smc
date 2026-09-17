# Corrected NeuroPAL chemical-identity experiment — 2026-08-28

Status: corrected primary and pooled-resampled predictive tournaments complete; frozen gate failed; no corrected lag-matrix launch. This file is the operational index for the correction; `AUDIT_STIMULUS_PROVENANCE_20260828.md` is the authoritative provenance record.

Post-gate note: a separate binary-stimulus, calcium-aware paired-distributional
experiment was subsequently run without claiming chemical identity. Its full
direct → rollout → ESS-SMC confirmation chain is negative and is indexed in
`DISTRIBUTIONAL_LAG_EXPERIMENT_20260828.md` (E25).

## Scientific question

Learn the held-out conditional activity law from lagged neural history and the causal stimulus history, then ask whether correctly encoded chemical identity improves proper predictive scores. Only if it does, use the learned law with direct importance sampling and progressive ESS-SMC to estimate chemical-specific onset-minus-matched-quiet lag matrices.

## Frozen decision order

1. Fit models without Cook, Randi, SBTG, receptor, or neuromodulator information.
2. Rank held-out predictions by energy score averaged equally across worm×chemical cells.
3. A chemical-aware candidate must beat binary, position-only, and independently subject-shuffled chemical controls in mean paired fold×seed score and win at least 6/10 pairs against each.
4. `chemical_scalar` is an ordinal sensitivity only and cannot pass the gate.
5. If no candidate passes, stop: do not generate corrected lag matrices.
6. If a candidate passes, confirm free-running rollout stability, then run direct N=256 and progressive ESS-SMC N=64.
7. Analyze butanone, pentanedione, and NaCl separately by reindexing each worm's events with the raw chemical code.
8. Only after matrices and internal reliability are frozen, compare descriptively with Cook, Randi, a corrected raw-head/no-donor SBTG lag-1 model, and receptor/transmitter edge maps.

## Cohorts

- Primary: 17 OH16230 head recordings, native 4.0 Hz, 54 shared head neurons.
- Sensitivity: the same 17 plus three OH15500 head recordings explicitly linearly resampled from native 4.1 to 4.0 Hz. The OH15500 third event remains 20 seconds.
- Excluded from corrected coupling inference: released SBTG 80-neuron pseudo-paired head/tail lineage and all donor-copied traces.

## Implemented encodings

- `binary_any_stimulus`
- `position_onehot`
- `chemical_scalar` (sensitivity only)
- `chemical_onehot` (primary)
- `chemical_plus_position_onehot`
- `chemical_onehot_subject_shuffle`

Every encoding is active-only. Baseline is all zero; future chemical identity is not exposed before onset. The shuffled control preserves event times and exactly one occurrence of each chemical per worm.

## Saved artifacts

| Artifact | Purpose |
| --- | --- |
| `AUDIT_STIMULUS_PROVENANCE_20260828.md` | Raw facts, hashes, affected/unaffected conclusions, quarantine ledger |
| `results/stimulus_provenance_audit_20260828/tensor_inspection.json` | Named-worm boundary and chemical-channel inspection |
| `results/conditional_distribution_benchmark/chemical_encoding_smoke_20260828_oh16230/` | Six-encoding training smoke and rollout smoke |
| `results/aligned_lag_response_20260828/chemical_onehot_direct_smoke_N8/` | Corrected direct-sampling smoke |
| `results/aligned_lag_response_20260828/chemical_onehot_progressive_smoke_N8/` | Corrected progressive-SMC smoke |
| `results/biological_lag_analysis_20260828/chemical_corrected_observed_20260828/` | Actual-event observed chemical response analysis |
| `results/compatibility_path_response/sbtg_head_corrected_oh16230_20260828/` | Corrected five-fold raw-head/no-donor SBTG lag-1 comparator |
| `results/conditional_distribution_benchmark/chemical_encoding_primary_20260828_oh16230/` | Complete 60-trial primary tournament |
| `results/conditional_distribution_benchmark/chemical_encoding_gate_20260828/` | Frozen paired gate and decision |
| `results/conditional_distribution_benchmark/chemical_encoding_sensitivity_20260828_pooled/` | Complete 60-trial pooled OH15500-resampled sensitivity |
| `cluster_specs/dandiset_000981/` | Cluster-ready prospective-data specification |
| `results/stimulus_provenance_audit_20260828/checksums.sha256` | SHA-256 inventory for the corrected reports and result tables |

## Corrected observed-data result

The observed analysis is not a lag-edge analysis. It selects each actual chemical event and compares its post-onset activity change with a pseudo-onset 15 seconds earlier in the same worm/event.

In primary OH16230 head data, AWC shows butanone-associated suppression at 4 seconds (mean matched contrast -0.873 background SD, BH q=0.00183) and 10 seconds (-1.672, q=0.000079). Pentanedione also shows sustained AWC suppression; NaCl does not yield an AWC FDR result. Because every animal has one event per chemical, no repetition or adaptation analysis is possible in this dataset.

## Primary predictive gate result

No chemical-aware encoding passed. Binary-any-stimulus has the best worm×chemical-balanced energy (1.077671). Chemical+position is second (1.078166) and beats position-only and shuffled chemical under the frozen mean-plus-majority rule, but is worse than binary by mean paired delta +0.000495 despite 6/10 pair wins. Chemical one-hot is worse than binary by +0.001538 with only 3/10 wins. Consequently, corrected direct N=256 and progressive N=64 lag sampling is not authorized on this dataset.

## Pooled-resampled sensitivity result

The complete 60-trial pooled-head sensitivity includes the three OH15500 animals after explicit 4.1-to-4.0 Hz resampling. Chemical one-hot ranks first by worm×chemical-balanced energy (1.310290), ahead of position-only (1.311527), chemical+position (1.314850), shuffled chemical (1.316678), and binary (1.323429). Its paired comparison passes binary (-0.013140, 7/10 wins) and shuffled chemical (-0.006389, 7/10), but fails position-only (-0.001238, only 5/10). Chemical+position passes only shuffled chemical. Thus neither candidate passes all three controls in the sensitivity cohort either. This does not alter the primary-cohort launch decision.

The combined result is informative but limited: true chemical identity contains some predictive signal relative to deliberately wrong chemical labels and, in the pooled sensitivity, relative to a binary active-stimulus flag. It is not reliably separable from stimulus epoch position in this one-event-per-chemical design. That confounding and the small number of animals make chemical-specific lag matrices an unsupported downstream estimand here.

## Verification

- Known OH16230 worm `0924_01` reconstructs order `[2,1,3]`; its first onset activates pentanedione channel `[0,1,0]`, not butanone.
- The half-open offset is all-zero.
- OH15500 native/resampled timing and 20-second last event are regression-tested.
- Training, rollout, direct, and progressive-SMC smokes are finite.
- Both full predictive tournaments contain 60/60 unique successful fold×seed×encoding trials.
- Complete corrected conditional/compatibility suite: 67 passed, one non-failing PyTorch Transformer warning.
- After adding the separate E25 paired-distributional and post-freeze
  correspondence modules and regression tests, the complete suite is 77
  passed with the same non-failing warning.

## Claim boundary

Predictive improvement would establish useful stimulus-conditioned activity modeling. A sampled matrix is a finite contrast of a learned observational law. Cook is structural, Randi is perturbational activity, and neuromodulator maps are transmitter/receptor existence maps. Agreement with any of them is descriptive correspondence—not proof of synapses, receptor action, causal inter-neuron influence, or physical transmission delay.
