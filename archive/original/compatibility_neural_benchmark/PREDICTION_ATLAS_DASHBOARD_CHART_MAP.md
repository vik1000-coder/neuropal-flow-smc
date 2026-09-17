# Prediction atlas dashboard chart map

Use this template when regenerating or reviewing the portable prediction-atlas dashboard. The canonical implementation is an `artifact.json` packaged by the installed Data Analytics portable-artifact builder; do not author a separate HTML implementation.

## Dashboard brief

- Audience: experimentalists and model developers reviewing lag-associated neural predictions.
- Decision: choose source neuron, target neuron, effect channel, stimulus context, source-to-cut lag, and forecast horizon for experimental follow-up.
- Surface: read-only portable dashboard with the unchanged `hypothesis_queue.csv` as its writable-workflow companion.
- Statistical unit: worm (`n = 17`), not event, particle, horizon, fold, or model seed.
- Orientation: target neurons are matrix rows; perturbed source neurons are matrix columns.
- Generator: binary-any-stimulus conditional flow.
- Chemistry boundary: chemical panels are exploratory event strata and are not chemically conditioned effects.
- Claim boundary: model-relative lag association, not a causal intervention or physical transmission delay.

## Metric-card map

| Card | Definition | Source | Reconciliation |
|---|---|---|---|
| Held-out worms | Independent worms in the reviewed cohort | `dashboard_snapshot.json` | Must equal 17 in snapshot, protocol, validation, and queue |
| Neuron classes | Directed matrix dimension | `dashboard_snapshot.json` | Must equal 54 and match target-row/source-column protocol |
| Mean valid fraction | Mean sampler-support validity across reviewed source/context/lag rows | `dashboard_snapshot.json` | Must lie in `[0,1]` |
| Support-qualified rows | Source/context/lag rows meeting the protocol minimum | `dashboard_snapshot.json` | Must not exceed all support rows |
| Retained candidates | Rows in the canonical atlas-only queue | `hypothesis_queue.csv` | Must reconcile to contiguous queue ranks |
| Audit checks passed | Fraction of required archive checks equal to true | `validation.json` | Must equal 100% before artifact generation |

## Chart map

| Block | Reader question | Dataset and grain | Native form | Encodings | Filters | Interpretation boundary |
|---|---|---|---|---|---|---|
| Primary matrix slice, when supplied | Which directed pairs have the largest normalized response in one reviewed slice? | Exactly 54 wide rows after validating a complete 54×54 primary-method/channel/context/lag/horizon slice; each row is one target and the 54 numeric fields are sources | Heatmap | x field = `target_neuron` (target rows); y fields = the 54 source-neuron names (source columns) | Fixed reviewed slice | A model-relative directed effect, not causal connectivity; the wide representation avoids both renderer transposition and the 2,000-row dataset cap |
| Lag-horizon profile fallback | Where do direct importance and progressive bridge SMC give similar directed matrices? | One row per source-lag × forecast-horizon cell for endpoint-mean onset-minus-baseline | Heatmap | Portable fields: x = source-to-cut lag, y = matrix Spearman, color group = forecast horizon; rendered grid = lag columns × horizon rows | Fixed primary channel/context | Agreement is estimator concordance, not biological truth or delay evidence |
| Candidate ranking | Which retained candidates have the highest atlas-only evidence score? | One row per bounded queue candidate | Sorted horizontal bar | Portable fields: x = unique candidate label, y = evidence score; horizontal orientation renders candidates vertically | Method, channel, context, lag, horizon, tier | External references never enter ranking or tier assignment |
| Candidate lag profile | How does the top-ranked candidate's normalized effect vary over the prespecified source-to-cut lag grid? | Four rows for queue rank 1; the companion table retains at most 12 candidates | Line profile | x = source-to-cut lag frames; y = primary normalized effect | Fixed queue rank 1 | Peak lag, selectivity, and worm-bootstrap peak rate are descriptive model-localization summaries, not causal or physical delays |
| Stimulus context | How does direct-versus-progressive agreement vary by context? | One row per stimulus context, aggregated over finite lag/horizon cells | Sorted horizontal bar | Portable fields: x = context, y = mean sampler Spearman; horizontal orientation renders contexts vertically | Fixed endpoint-mean channel | Chemical contexts are event-stratified and not chemically conditioned |
| Targeted screen versus N128, when supplied | How do frozen selected-cell screen effects compare with the N128 high-low sensitivity estimates? | At most six internally selected source-target cells | Scatter | x = frozen screen normalized effect; y = targeted N128 high-low normalized effect; color = descriptive consistency label | Fixed frozen selection; no dashboard filter changes the selection | Post-screen and selection-conditioned sensitivity only; it never re-ranks candidates or changes evidence tiers |

## Table map

| Table | Purpose | Default sort | Boundedness |
|---|---|---|---|
| Candidate evidence and support | Exact source/target, timing, effect, interval, validity, tier, and chemistry-scope lookup | Queue rank ascending | At most 500 embedded queue rows; default 150 |
| Candidate lag-profile audit | Per-lag primary/counterpart effects, worm-bootstrap peak rate, selectivity, and profile agreement | Queue rank ascending | At most 12 candidates × four source lags |
| Audit and provenance inventory | Canonical local paths, roles, SHA-256 checks, and post-freeze boundary | Artifact ascending | Required atlas files plus one optional-reference status row |
| Post-freeze external reference | Optional comparison-only rows | First displayed field ascending | At most 250 rows × 12 columns; default 100 rows |
| Selected-cell screen and N128 sensitivity | Exact frozen screen estimate, N128 high-low estimate, delta, validity, and descriptive consistency | Atlas queue rank ascending | At most six rows |
| Targeted N128 three-arm sensitivity | Selected metric for high-low, high-factual, and low-factual arms | Atlas queue rank ascending | At most six candidates × three contrasts = 18 rows |

## Filter contract

- Method, effect channel, and stimulus context default to progressive bridge SMC, endpoint mean, and onset-minus-baseline when that combination exists in the bounded queue.
- Source-to-cut lag and forecast horizon are separate filters and separate labels.
- Source-to-readout time is shown in tooltips and the evidence table as `(lag + horizon) / 4 Hz`.
- Evidence tier is filterable, but `confirmed` remains reserved for independent experimental confirmation.
- Filters affect the candidate ranking and evidence table; they do not silently redefine the fixed audit/profile charts.

## Provenance contract

- Require `dashboard_snapshot.json`, `hypothesis_queue.csv`, `validation.json`, `protocol.json`, `models.json`, and `checksums.sha256`.
- Verify every required file against `checksums.sha256` before writing `artifact.json`.
- When `--targeted-reference` is supplied, require a validated targeted-analysis directory, verify its checksum inventory and passed checks, verify its selection-bundle checksum inventory, and reconcile the frozen selected queue byte/hash/value linkage back to the canonical atlas queue and the exact N128 run manifest.
- Treat targeted results as a comparison-only layer. The factual arm is a learned-flow rollout from observed history, not an observed response, experiment, or causal intervention.
- Store only provenance-root-relative paths with no parent traversal, credentials, or credential-bearing URLs.
- Preserve an exact reproducible command using `python -m compatibility_neural_benchmark.prediction_atlas_dashboard`.
- Never embed dense NPZ arrays or the full prediction Parquet table in the portable snapshot.

## Release QA

- [ ] Canonical validation status is `passed`; every archive check is true.
- [ ] Worm/neuron counts reconcile to 17/54 across inputs.
- [ ] Queue ranks, timings, evidence tiers, and event-stratified chemistry labels validate.
- [ ] Embedded payload is below the portable three-megabyte ceiling.
- [ ] If a primary matrix is embedded, it has exactly 54 rows, x is `target_neuron`, y fields are the 54 source-neuron names, and the target-row/source-column orientation self-test passes.
- [ ] If targeted N128 data are embedded, all checksum, validation, staged-queue, canonical-queue, run-manifest, and input-checksum linkages pass; the layer has at most six screen rows and 18 contrast rows.
- [ ] Every source-backed card, chart, and table resolves to canonical source metadata.
- [ ] Chart titles are neutral; subtitles state grain, timing, cohort, or claim boundaries.
- [ ] Candidate filters produce a nonempty default view.
- [ ] Builder receipt reports validation/package passed and verification passed or disclosed structural-only status.
- [ ] Generated HTML is treated as builder output and is never hand-edited.
