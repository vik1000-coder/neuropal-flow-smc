# Neural Atlas v2 · historical 80-neuron dashboard

This read-only dashboard explores the optimized progressive-bridge SMC atlas over the historical 80-class SBTG compatibility dataset. It loads only the selected edge or target slice, so figures update without placing the full matrices in the browser.

## Run

From the repository root:

```bash
./dashboard_v2_80/run.sh
```

Open http://127.0.0.1:18783. Use `./dashboard_v2_80/run.sh --port 18784` if needed. The server binds to localhost, verifies the frozen result ledgers and dashboard input manifest, and never writes through the API.

## What the figures mean

- The main curve is the source-gap-normalized high-source minus low-source model response over forecast horizons or source-history lags.
- The band is the saved pointwise 95% interval for the mean from 256 whole-trace bootstrap resamples. It conditions on fitted generator seed 1701.
- Individual curves and dots are the 20 historical trace-level model effects. They reveal heterogeneity but are not 20 independent simultaneous whole-animal recordings.
- The support figure shows valid-episode fraction, repaired-root ancestry thresholds, ESS, and achieved source gap. These diagnose the finite-particle calculation; they are not biological evidence or a null test.
- Recorded-activity bands are 10th–90th percentiles across historical traces. They are neither confidence intervals nor measurement-error estimates.
- Reference checks compare the single-seed atlas and the separate two-seed lag-1 sensitivity against published SBTG values on Randi and Cook networks.

## Primary model choice

The complete dashboard atlas uses `flow_lr6e4`, seed 1701, 64 particles, repair branch factor 4, future branch factor 2, and 20 flow integration steps. Seed 1701 had the better held-out predictive score. Two-seed averaging improved the four final lag-1 AUROC point estimates, so the dashboard retains that result as a clearly labeled sensitivity rather than requiring two full atlas runs.

## Scientific limitation

The 80-class cache pseudo-pairs head and tail recordings and donor-imputes missing traces. The results are model-relative observational response sensitivities. They are not causal interventions, anatomical edges, receptor effects, or physical transmission delays. The intervals omit generator-refit, imputation, and atlas-wide multiplicity uncertainty.

See `PLAN.md` for the protocol and `VALIDATION.md` for the final checks. `build_portable.py` creates a standalone directory for sharing after the result build passes.
