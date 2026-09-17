# Historical 80-neuron Neural Atlas v2 plan

## Goal

Provide the same clear interaction model as Neural Atlas v2 over the complete historical 80-class head/tail sensitivity dataset, while showing the uncertainty in the effect and the numerical support behind each source/lag.

## Frozen production protocol

- Generator: `flow_lr6e4`, conditional flow matching, 20 integration steps.
- Generator realization: seed 1701 only. It was selected by five-fold held-out predictive energy without consulting Randi, Cook, Bentley, or published SBTG correspondence.
- Sampling: progressive-bridge SMC, 64 particles, repair branch factor 4, future branch factor 2, minimum ESS 12.
- Grid: 80 sources × 80 targets, four source lags, six forecast horizons, seven outcomes, and 13 contexts.
- Uncertainty: individual values across 20 historical traces and pointwise 95% whole-trace bootstrap intervals for their mean, with 256 resamples.
- Sensitivity: show the previously completed two-seed high-resolution lag-1 result in reference checks, without mixing it into the primary atlas.

## Data and claim boundary

The historical cache pseudo-pairs head and tail recordings and donor-imputes missing traces. It is useful as a compatibility sensitivity on the same 80-class axis used in the SBTG analysis, but it is not a clean simultaneous 80-neuron cohort. Dashboard text and downloadable data must call the rows historical traces and describe effects as model-relative response sensitivities.

## Build and validation sequence

1. Freeze checkpoint hashes, model configuration, seed, fold assignments, particle settings, branch settings, and random seed in a resume-safe run manifest.
2. Run one fold and compare its lag-1/horizon-1 response arrays bit-for-bit against the already frozen optimization run.
3. Complete the remaining lag/fold archives and validate every archive before aggregation.
4. Build dense target-row/source-column atlas and trace matrices; run Randi/Cook reference metrics only after sampling.
5. Serve small selection-specific JSON payloads so charts load on demand rather than shipping the full atlas to the browser.
6. Test orientation, exact values, trace means, quantiles, sampler diagnostics, reference comparisons, API validation, checksums, and browser interaction.
7. Build a portable directory containing the viewer, frozen data, provenance, and generating source snapshot.
