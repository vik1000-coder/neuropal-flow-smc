# Neural prediction atlas dashboard

The current dashboard is the **full atlas explorer**: a local, self-contained HTML application for the frozen 54-class NeuroPAL analysis. It lets you compare source–target effects, source-history lags, forecast horizons, stimulus-event strata, and the available statistical checks. Changing a filter does not train a model or run a sampler.

- Open URL: [Neural prediction atlas](http://127.0.0.1:18780/gui/atlas_explorer.html).
- Application file: [atlas_explorer.html](results/neural_prediction_atlas_20260829/gui/atlas_explorer.html).
- User and maintenance guide: [DASHBOARD_GUIDE.md](results/neural_prediction_atlas_20260829/DASHBOARD_GUIDE.md).
- Scientific record: [atlas README](results/neural_prediction_atlas_20260829/README.md).

## Open or restart locally

The HTTP serving directory is the atlas result directory, **not** the repository root or its `gui` subdirectory:

```text
/Users/vik/Developer/new_sbtg_neuro/results/neural_prediction_atlas_20260829
```

First check whether the saved address already works:

```bash
curl --noproxy '*' --max-time 10 --head http://127.0.0.1:18780/gui/atlas_explorer.html
```

If it does not, check for an existing listener:

```bash
lsof -nP -iTCP:18780 -sTCP:LISTEN
```

If that port is free, run this in a terminal and leave the terminal running:

```bash
/Users/vik/Developer/new_sbtg_neuro/.venv/bin/python -m http.server 18780 \
  --bind 127.0.0.1 \
  --directory /Users/vik/Developer/new_sbtg_neuro/results/neural_prediction_atlas_20260829
```

Then open the URL above in the Codex browser, or use the default macOS browser:

```bash
open "http://127.0.0.1:18780/gui/atlas_explorer.html"
```

Do not stop an unidentified process if the port is occupied. The [guide](results/neural_prediction_atlas_20260829/DASHBOARD_GUIDE.md#launch-and-reopen) explains alternate ports and direct-file opening. Keep the server bound to `127.0.0.1`; this simple file server has no authentication.

## What to read first

In **Explore**, select a source, target, and timing. The top estimate is the canonical atlas result. Expand the five checks under **Evidence for this selection** before interpreting its strength. A large or brightly colored estimate is not a significance result.

The current frozen analysis contains 102 support-qualified directed-edge discoveries, all in baseline endpoint mean, but no primary non-flat lag discoveries. Among 24 outcome rows from eight selected N128 recalibrations, four clear sampler noise only and none clear both sampler and quiet-time controls. These are model-relative results, not causal effects or experimental confirmation. [Definitions and results](results/neural_prediction_atlas_20260829/DASHBOARD_GUIDE.md#evidence-how-to-read-the-five-checks).

`History lag` locates a repaired source window **before** the prediction cut. `Forecast horizon` locates the readout **after** the cut. Neither is a physical transmission delay. The 54 entries are pooled head-neuron classes; this is not an 80-neuron dashboard. [Timing and cohort details](results/neural_prediction_atlas_20260829/DASHBOARD_GUIDE.md#cohort-model-and-sampling-scope).

## Which page is current?

| File | Purpose |
| --- | --- |
| `gui/atlas_explorer.html` | Current full explorer, schema v3; all five input layers. |
| `gui/dashboard.html` | Older bounded summary dashboard; not the full explorer or its newer evidence layer. |
| `technical_report/report.html` | Earlier portable technical report; a separate narrative artifact. |

These paths are relative to `results/neural_prediction_atlas_20260829/`. The full application path is `/Users/vik/Developer/new_sbtg_neuro/results/neural_prediction_atlas_20260829/gui/atlas_explorer.html`.

## Documentation verification

On 31 August 2026, the saved explorer and all five declared input ledgers passed read-only checksum verification: **59 entries across six ledgers, no failures**. At `2026-08-31T16:26:09Z`, the saved HTTP address did **not** have a listening server; this documentation pass did not start one. The existing [browser QA](results/neural_prediction_atlas_20260829/GUI_BROWSER_QA.md) concerns the earlier successful 30 August browser run, not current server availability.

No UI, scientific data, sampler output, or model was changed for this documentation update.
