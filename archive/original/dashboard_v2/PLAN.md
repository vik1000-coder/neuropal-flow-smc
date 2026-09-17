# Neural atlas v2 — implementation plan

31 August 2026. A separate local dashboard; the frozen atlas and original explorer stay unchanged.

## Reading path

1. Choose a source, target, and outcome. Default to FLP → ADE, baseline endpoint mean, progressive bridge SMC.
2. Read a dynamically loaded effect curve with its saved pointwise 95% worm-bootstrap interval. Switch between forecast horizon and source-history lag. Selecting a point updates the estimate, individual-worm plot, and evidence together.
3. Inspect the recorded source/target activity, with an explicit between-worm spread band and optional individual traces. These are observed signals, not repaired trajectories or measurement-error intervals.
4. Inspect exact matched N128 sampling/quiet-time controls where they exist. Keep these separate from canonical screen estimates. Offer archived low/high predictive ranges only at actually saved coordinates.
5. Expand all-target and reference comparisons when needed; keep dense tables and methodology out of the first viewport.

## Presentation

- One compact working surface; no introductory hero or repeated evidence prose.
- Restrained dark neutral theme, readable axis labels, clear zero lines, consistent uncertainty legends.
- Shared source/target/outcome/context controls, four explicit history lags and six horizons; advanced options progressively disclosed.
- Linked SVG figures with hover/focus values and downloadable displayed data. No interpolated statistical claims between sampled times.
- Loading/error/empty states; stale requests must never overwrite the latest selection. Cache bounded requests, not the entire atlas in HTML.
- Preserve all 54 neuron classes, seven outcome channels, 13 saved contexts, and optional direct-importance comparison.

## Data and uncertainty contract

- Exact values and saved CIs from `canonical/atlas_matrices.npz`; target rows/source columns.
- Individual-worm effects from `canonical/worm_matrices.npz`, which contains progressive SMC only. Never invent direct-method worm values.
- Saved atlas intervals use 256 whole-worm percentile bootstrap replicates. They are descriptive, pointwise, conditional on fitted models, and do not include model-refit uncertainty or exploratory multiplicity.
- Existing complete-family p-values remain separate from CIs. Never relabel zero-null reference bands as confidence intervals or map active-minus-baseline evidence onto onset-minus-baseline.
- Exact N128 support and corrected tests come from the sealed combined-eight calibration. Magnitude/excess summaries use the saved per-worm definitions, not absolute pooled means.
- W1 is unsigned within a state and needs sampling controls. A positive CI is not a detection test.
- Observed activity has no supplied measurement-error model. Display real observations and empirical between-worm spread, preserve missing values, and label display standardization.
- Archived predictive quantiles describe generated outcomes for an individual saved worm/event/model seed; they are not an effect CI or independent biological replication.

## Implementation and verification

Use the existing Python/NumPy/Pandas and HTML/JS/SVG stack. A read-only loopback server loads only the selected data, with a small initial page and bounded array caches. No external services, training, sampling, or changes to frozen results.

Validate exact asymmetric cell values, CI endpoints, worm means, evidence joins, missing-data behavior, unsupported controls, request validation and stale-response handling. Verify the original explorer remains unchanged. Document launch, data provenance, uncertainty limitations, and the completed checks.
