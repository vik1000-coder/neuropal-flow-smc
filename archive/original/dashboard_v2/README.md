# Neural atlas v2

A simpler, read-only companion to the original explorer. Start with one source–target pair, see its estimated effect and uncertainty, then open the evidence needed to interpret it. No training, inference, or frozen-result edits occur.

## Run

From the repository root:

```sh
./dashboard_v2/run.sh
```

Open <http://127.0.0.1:18781>. The launcher uses the existing repository `.venv`; NumPy, Pandas and the installed Parquet reader are required. Pass `--port 18782` to use another local port. Stop with Ctrl-C. There is no public deployment or background auto-start.

The original explorer remains at <http://127.0.0.1:18780/gui/atlas_explorer.html> when its server is running. To start that separately:

```sh
./.venv/bin/python -m http.server 18780 --bind 127.0.0.1 --directory results/neural_prediction_atlas_20260829
```

## Use

1. Choose source, target, measure and context. The starting example is FLP → ADE before stimulus, at a 0.25 s history lag and 2 s forecast.
2. Switch the effect curve between forecast time and history lag. Click a point to update the estimate, individual worms and matched control results together. Time axes use true seconds.
3. Open **Recorded activity** for source/target observations around a real chemical onset. Show all worms or select one.
4. Expand **Predicted activity distributions** for saved low/high generated outcomes, when available. Select the worm, event history, model seed and sampling repeat.
5. Expand **All targets** for an alphabetical overview with intervals. **Options** includes the estimator comparison. **Reference checks** shows AUROC and continuous-agreement dot plots within one declared panel, followed by the exact saved table.

The current effect selection is reflected in the URL. Download figure data exports the loaded values and labels as JSON; the activity view exports the observed traces and quantiles. Expanded all-target values are available through `/api/targets`.

## What the uncertainty means

| Display | Meaning | Does not include |
| --- | --- | --- |
| Effect curve, shaded 95% interval | Saved pointwise whole-worm bootstrap interval for the mean; 17 worms, 256 resamples | Model-refit uncertainty, a simultaneous band, or correction for dashboard exploration |
| Individual effect lines and dots | Worm-to-worm variation after averaging seeds and events within each worm | Extra biological replication from seeds, events or particles |
| Recorded activity, 10th–90th percentile band | Empirical between-worm spread, standardized within each worm/neuron over its recorded trace | Measurement-error uncertainty, an effect CI, or a chemical comparison test |
| Low/high prediction histograms | Archived generated endpoints for one worm/event/seed/repeat; 256 descendants per arm | Independent samples of animals or a CI on the effect |
| Paired control excess, dots and intervals | Separate selected N128 rerun; per-worm absolute effect minus the matched control magnitude | An onset-modulation test or a replacement for the canonical screen |

Corrected evidence is joined only to its exact saved method, outcome, context, pair and timing. Baseline W1 is an unsigned distance: a positive interval alone is not a detection test. Onset-minus-baseline W1 is a difference of distances. Source support and particle ancestry remain separate checks. Associations are model-relative; no causal connection or physical delay is established.

No direct-estimator individual-worm bank was archived, so that view explicitly omits worm dots. Saved endpoint distributions and N128 control tests cover eight selected baseline coordinates; other selections show an honest unavailable state. Recorded observations preserve missing values rather than filling or smoothing them.

## Sources and implementation

- `results/neural_prediction_atlas_20260829/canonical`: exact atlas means, bootstrap endpoints, progressive worm arrays and support gates. Matrices are target-row/source-column.
- `complete_family_evidence_20260830` under that atlas: corrected cell and edge tests.
- `sampling_null_controls_combined8_n128_20260830`: final control summaries, per-worm inference arrays and saved endpoint draws.
- `SBTG/data/Head_Activity_OH16230.mat`, through the current cohort loader: recorded activity from the 17 worms and 54 pooled head-neuron classes, sampled at 4 Hz.
- The existing fold assignments and reconstructed training-only scaler: inverse transformation of archived endpoint samples. The factual observation is read at `cut + horizon`; the cut is the final observed frame.
- The original `gui/explorer_data.json`: reviewed external-reference comparisons.

Startup verifies 114 entries in the existing frozen ledgers plus six current input hashes in `input_manifest.json`. The additional manifest records current presentation inputs, not retrospective training provenance. A mismatch stops startup; investigate it instead of automatically rewriting hashes.

`server.py` serves a small HTML/JS/SVG app and selected JSON through a loopback-only, read-only API. It checks the Host header, validates coordinates, restricts static files and serializes missing values as JSON null. Selected arrays and requests have bounded caches. Activity and distribution data load only when their views are opened. Superseded requests cannot overwrite the latest selection.

## Verify

```sh
./dashboard_v2/run.sh --validate
./.venv/bin/python -m pytest dashboard_v2/test_dashboard.py -q
node --check dashboard_v2/app.js
```

See `PLAN.md` for the design plan and `VALIDATION.md` for completed checks and limits.

## Portable bundle

Build the sendable presentation-only directory and replace an older build with:

```sh
./.venv/bin/python dashboard_v2/build_portable.py --force
```

The output is `dist/neural_atlas_v2_portable`. It contains the complete viewer, slimmed exact inputs, all 70 saved response banks, observed cohort, fold scalers, reviewed reference data, provenance manifests, and a source snapshot. It runs independently with Python 3.11+, NumPy and Pandas. Its own README provides macOS/Linux and Windows launch instructions.

The portable data extraction removes unused fields but does not round or recompute displayed atlas, worm, control, signal, prediction, or reference values. It reproduces this frozen-results viewer; it does not include training checkpoints or reproduce model fitting. Review its `REDISTRIBUTION_NOTE.md` before forwarding it outside the intended collaboration.
