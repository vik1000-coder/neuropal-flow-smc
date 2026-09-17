# Stimulus provenance correction — 2026-08-28

Status: critical correction; this document supersedes every local statement that the three NeuroPAL epochs were repeated butanone presentations.

## Frozen raw facts

- `stim_names = [butanone, pentanedione, nacl]` in both head recordings.
- `stims[worm,event]` is the 1-based chemical code indexing that name vector. Each animal receives one occurrence of every chemical in an animal-specific order.
- In the retained 20-worm/54-head-neuron cohort, epoch position equals chemical code in only **24/60** animal-events.
- Known check: OH16230 worm `0924_01` has order `[2,1,3]`; its first epoch is pentanedione, not butanone.
- OH16230 head is sampled at 4.0 Hz with half-open event intervals `[60.5,70.5)`, `[120.5,130.5)`, and `[180.5,190.5)` seconds.
- OH15500 head is sampled at 4.1 Hz. Its final event is `[180.5,200.5)`, not a 10-second event. The corrected pooled sensitivity resamples traces to an explicit zero-origin 4.0-Hz grid before windowing and retains the true 20-second interval.
- Raw SHA-256: OH16230 head `3d89eebac57ac95833662d00c977993a8e45bc10270c24dbcbce3a7eb401d930`; OH15500 head `34d71610b763510b7566047253c3db626f50fc280752df7a54e240daaf30d7c2`; OH16230 tail `7b7aa612a9499ebd1828b389d219d5881116d4ccc2fb780aa130c9f5968bdb93`.

## Separate lineage defect

The released 80-neuron SBTG lineage is historical/contextual only. It merged head and tail arrays by index even though the recording IDs, clocks, and stimulus orders do not define simultaneous animal pairs. It also used donor-worm trace copying for missing neurons (`18` neurons and `34` worm×neuron imputations in the released cache). It must not be described as an 80-neuron simultaneous-recording cohort or used for coupling inference.

The current 54-node set consists entirely of head neurons. Removing the tail file preserves its node names. Corrected analyses nevertheless use `include_tail=False` and prohibit fusion unless an explicit simultaneous-recording pairing table plus matching IDs, clocks, and stimulus orders are supplied.

## Quarantined artifacts and affected conclusions

- `results/conditional_distribution_benchmark/presentation_encoding_20260828/`: the scalar/one-hot labels are epoch-position labels. Its numerical predictive scores remain a record of position conditioning, but none is a chemical-identity result.
- `results/aligned_lag_response_20260828/presentation_scalar_direct_N256/`, `presentation_onehot_direct_N256/`, and the stopped `presentation_onehot_progressive_N64/`: position-conditioned only. The progressive run was terminated after six complete fold/seed archives; all files are preserved.
- `results/biological_lag_analysis_20260828/`: superseded for chemical-specific AWC/butanone and repeated-exposure/adaptation claims. Events were grouped by position or pooled without selecting the actual chemical.
- Any claim that the three local epochs are three butanone repeats, or that response changes across them measure butanone repetition attenuation, is invalid.
- Old pooled OH16230/OH15500 predictive and lag numbers also carry a clock error: OH15500 was treated as 4.0 Hz without explicit resampling and its third event was truncated to the OH16230 10-second schedule.
- SBTG-published 80-neuron results remain historical comparisons only; they are not a corrected simultaneous-neuron benchmark.

## Conclusions not erased by this correction

- Synthetic-method results that do not use these recordings are unaffected.
- Cook, Randi, Bentley, receptor, and neuromodulator reference files themselves are unchanged; their prior use as post-hoc descriptive references remains conceptually separate from model selection.
- The completed position-conditioned predictive scores and finite sampler diagnostics remain reproducibility records for exactly that misspecified estimand. They cannot support chemical identity, chemical adaptation, or chemical-specific lag claims.
- The current 54 neuron names are retained by a head-only load. This does not validate the old chemical grouping or the pooled clock handling.
- Generic statements about observed activity or any-stimulus onset must be recomputed on the corrected head-only schedule before being promoted; old chemical-specific statements are not salvaged by relabeling figures.

## Corrected analysis contract

1. Primary: OH16230 head only, native 4.0 Hz, frozen per-worm chemical order.
2. Sensitivity: pooled head-only cohort with explicit OH15500 4.1→4.0 resampling and its 20-second final event.
3. Encodings: binary-any-stimulus, position one-hot, categorical chemical one-hot, chemical+position one-hot, categorical scalar sensitivity, and subject-wise shuffled chemical assignment.
4. Evaluation: whole-worm folds and equal worm×chemical active-event proper-score weighting; ±1-frame boundary sensitivity.
5. Lag archives: store worm×event chemical code/name and reindex each animal's event matrix by chemical before aggregation.
6. Selection remains atlas-blind. Cook/Randi/SBTG/receptor maps are post-freeze correspondence checks only.

## Correction implementation status

- The active position-one-hot progressive process was terminated without deleting its six complete archives. Its manifest is preserved and a separate `quarantine.json` records that the partial run is invalid for chemical-identity claims.
- Schedule-aware head-only loading, OH15500 4.1→4.0 explicit resampling, half-open event masks, six corrected encodings, frozen subject shuffles, explicit neural/stimulus channel splitting, archive chemical metadata, and schema-fingerprint resume rejection are implemented.
- Focused training, rollout, direct N=8, and progressive ESS-SMC N=8 smokes completed with finite outputs. Manual tensor inspection covers named worms with different chemical orders and both strains.
- The complete conditional/compatibility test suite passes: **67 tests passed** (one non-failing PyTorch Transformer warning).
- Corrected observed-event biology is saved in `results/biological_lag_analysis_20260828/chemical_corrected_observed_20260828/`. It selects actual chemical events and contains no repeat/adaptation analysis.
- A fair SBTG-current lag-1 comparator is saved in `results/compatibility_path_response/sbtg_head_corrected_oh16230_20260828/`: five outer folds, 17 raw OH16230 head recordings, 54 shared head neurons, no tail pseudo-pairing, and no donor copying. The released SBTG 80 artifact remains historical only.
- Both corrected 60-trial predictive tournaments are complete (six encodings × five whole-worm folds × two seeds; zero failed trials in either cohort). In primary OH16230 head data, binary-any-stimulus ranks first (worm×chemical energy 1.077671). Chemical+position is second (1.078166) and passes position-only and shuffled controls, but fails binary by mean paired delta +0.000495 despite 6/10 wins. Chemical one-hot also fails binary (+0.001538; 3/10 wins).
- In the pooled-resampled sensitivity, chemical one-hot ranks first (1.310290) and passes binary (-0.013140; 7/10 wins) and shuffled chemical (-0.006389; 7/10), but fails position-only (-0.001238; 5/10). Chemical+position passes only shuffled chemical. No candidate passes all controls in either cohort. The clean primary cohort controls the decision, so corrected direct N=256 and progressive N=64 lag sampling is prohibited and was not launched.
- The supported predictive conclusion is narrower than a chemical-specific dynamics claim: true chemistry carries signal relative to shuffled labels, but its incremental value cannot be reliably separated from binary onset and/or epoch position in this one-event-per-chemical dataset.

## Post-correction E25 addendum

`DISTRIBUTIONAL_LAG_EXPERIMENT_20260828.md` records a later
binary-any-stimulus paired distributional audit on the corrected primary
17-worm/54-head-neuron cohort. It does not reinterpret position as chemistry.
The atlas-blind calcium-aware model passes a small predictive gate, but the
only direct boundary candidate fails the final proposal-valid N=256 ESS-SMC
proper-score confirmation. No E25 matrix or edge is promoted, and no Cook,
Randi, SBTG, receptor, transmitter, or neuromodulator target entered selection.
The later post-freeze correspondence now includes separate dopamine,
serotonin, tyramine, and octopamine Bentley panels. Their descriptive peaks do
not repair the confirmation failure: no transmitter-specific profile survives
lag-max permutation plus global multiplicity correction, and all but dopamine
have only one eligible source neuron in the shared 54-neuron subset.
