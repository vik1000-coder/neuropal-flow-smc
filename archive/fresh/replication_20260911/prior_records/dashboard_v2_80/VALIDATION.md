# Validation record

Optimized full-run results were finalized on 2026-09-03. Dashboard and portable-bundle checks are recorded below as they complete.

## Required gates

- [x] 20/20 raw archives validate with exact checkpoint hashes and fold holdouts.
- [x] Every archive records N64, repair branch 4, future branch 2, six horizons, seven response channels, finite values, and reproducible validity gates.
- [x] Lag-1/horizon-1 arrays match the frozen optimization sampling run exactly for all five folds.
- [x] Dense atlas orientation is target row / source column and all 80 × 80 slices have the expected geometry.
- [x] Trace means reproduce dense means and saved pointwise intervals load exactly.
- [x] Reference view contains the single-seed result, published SBTG comparator, two-seed sensitivity, and source-bootstrap intervals.
- [x] Signal quantiles are recomputed from the 20 historical traces without hiding missing values.
- [x] Local API rejects unknown/repeated parameters, emits strict JSON, and serves no filesystem paths.
- [x] Dashboard input and result checksum ledgers verify at startup.
- [x] Browser checks pass for loading, selection changes, target list, recorded signals, reference figures, resize, and console errors.
- [x] Portable directory verifies from its own manifest and loads without the research repository.

Browser QA passed in Chrome at 1440 × 1000 and 760 × 1000: all dashboard views loaded, selectors and the 80-target table updated correctly, figure-data download succeeded, responsive layout held, and the console reported zero errors and zero warnings. A second QA pass verified all seven Bentley network choices, the dopamine AUROC/AUPRC plots, exact lag-1 rows, and the permutation-calibrated lag-max table with zero console errors or warnings.

Portable QA passed from `dist/neural_atlas_v2_80_portable`: all 64 bundle checksums verified; 80 neuron classes, 20 traces, uncertainty, support, Randi/Cook and Bentley reference data, the compiled PDF report, and the 1,820-row positive Bentley relationship table loaded. The metadata and reference APIs returned HTTP 200 while running from the bundle directory.

The 11-page standalone PDF was rendered and visually checked page by page. It contains an executive Bentley interpretation, prespecified lag-1 AUROC/AUPRC comparisons for all seven networks, written network-by-network conclusions, all 70 planned permutation-calibrated lag-max rows, the ten highest-ranked support-qualified Bentley-positive neuron relationships for each network, the Randi/Cook comparisons, and the uncertainty/head-tail caveats. The complete 1,820-row Bentley-positive relationship inventory remains alongside the report as CSV so no pair-level evidence is lost.

## Scientific results

- Raw validation: pass, 20/20 archives, 30,615.5 aggregate sampler-seconds.
- Frozen lag-1 regression: pass bit-for-bit for five folds, seven response channels, and all sampler diagnostics.
- Dense atlas validation: pass, 80 neurons, 20 historical traces, 13,977,600 effect cells, 218,400 ranked-effect rows.
- External-reference validation and final audit: pass.
- Single-seed lag-1 AUROC: Randi 0.657009, Cook structural 0.613661, Cook chemical 0.605225, Cook gap 0.653557.
- Published SBTG lag-1 AUROC: Randi 0.630237, Cook structural 0.580651, Cook chemical 0.569780, Cook gap 0.613297.
- Two-seed sensitivity AUROC: Randi 0.674289, Cook structural 0.629006, Cook chemical 0.617380, Cook gap 0.666226.
- The source-bootstrap AUROC improvement interval excludes zero for Cook structural, chemical, and gap; the Randi interval crosses zero.
- Bentley analysis covers dopamine, serotonin, tyramine, octopamine, all monoamines, all neuropeptides, and the union. The strongest globally adjusted planned result is state-average endpoint-Wasserstein correspondence for the union (AUROC 0.549945 at lag 1, permutation p = 0.001, global BH q = 0.0155); neuropeptides have the same p and q with AUROC 0.566771.
- Specific monoamine estimates have only one or two eligible sources, and the sole tyramine source fails the progressive lag-1 support gate. These network-specific estimates are therefore descriptive and have limited source-level replication.

The uncertainty bands resample the 20 historical traces conditional on generator seed 1701. They do not include generator refitting, imputation uncertainty, or multiplicity correction. The historical 80-class lineage pseudo-pairs head and tail recordings and donor-imputes missing traces, so results are model-relative response sensitivities rather than causal or anatomical effects.
