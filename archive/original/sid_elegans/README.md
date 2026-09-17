# sid_elegans — score-identified distributional dynamics on *C. elegans*

Adapter that runs the `sid_neuromod` score-identified estimator (and baselines) on real
NeuroPAL whole-brain calcium data, reusing the SBTG project's data and evaluation with a
**corrected connectome orientation**. Read `output/FULL_REPORT.md` for the complete
scientific narrative; this README is the code map + reproduction guide.

## Dependencies (two environments)

| env | path | packages | used by |
|---|---|---|---|
| **main** | `../.venv` | numpy, scipy, scikit-learn, statsmodels, matplotlib, torch, `sid_neuromod` | everything except the causal baselines |
| **scratch** | `/tmp/baseline_feas` | lingam (VAR-LiNGAM), pysindy (SINDy), tigramite (PCMCI+) | `lagres/scratch_effects.py`, `lagres/run_pcmci.py` |

Run main-venv code with `PYTHONPATH=..:../SBTG ../.venv/bin/python -m sid_elegans.<module>`.
The scratch venv holds the causal-discovery baselines that don't co-install cleanly with the
main stack on Python 3.14; they exchange data with the main pipeline via `.npz` files.

## Module map

**Data & ground truth**
- `data.py` — load the repo's prepared per-worm standardized traces.
- `combined_data.py` — pool **all** recordings (OH16230 head+tail + OH15500 head, 28 worms);
  complete-case or NaN-imputed; optional calcium deconvolution; **scale-preserving**
  standardization (never per-neuron z-score — see FULL_REPORT §5).
- `deconv.py` — native OASIS AR(1) calcium deconvolution → continuous activity.
- `ground_truth.py` — orientation-**corrected** Cook connectome (chem/gap/struct), Bentley
  monoamine layers, neuropeptide network, optional Randi/Leifer atlas; all `[post,pre]`.

**Estimators (channels)**
- `estimator.py` — SID closed-form quadratic-score distributional connectome
  (mean / gain / tail per lag). The mean-channel is VAR-comparable; gain/tail are the new axis.
- `multiscale.py` — multi-timescale exponential-filter-bank connectome + σ-ledger.
- `twohead.py` — two-head denoising-score-matching neural model (Gaussian conditional).
- `mdn.py` — **mixture-density** (non-Gaussian) conditional model; the expressive estimator.
- `baselines.py` — Pearson / lagged cross-correlation / ridge-VAR reference baselines.

**Evaluation & statistics**
- `evaluate.py` — AUROC/AUPRC/Spearman/F1 of `|coupling|` vs a reference (off-diagonal,
  reproduces the SBTG metric exactly).
- `significance.py` — permutation p-values, bootstrap AUROC-difference CIs, source-variance
  partial-out control.
- `stability.py` — self-supervised split-half reproducibility objective (HP tuning without
  touching the labels).

**Analysis drivers** (each writes to `output/`)
- `run_eval.py` — corrected structural + monoamine comparison vs old SBTG & baselines.
- `run_scaling.py` — data-scaling curves (6→20 worms).
- `run_mdn_verify.py` — rigorous MDN serotonin verification.
- `run_expressive.py` — #3 expressive estimators + the source-variance partial-out control.
- `run_deconv.py` — #1 deconvolution, controls-first, raw vs deconvolved.
- `run_holistic.py` — the all-methods × all-targets overall assessment.

**`lagres/` — the lag-resolved relative analysis** (see `output/lagres/LAGRES_RESULT.md`)
- `effects.py` — main-venv per-lag `[post,pre]` effect matrices (cross-corr, ridge, VAR,
  DYNOTEARS, SBTG mu_hat, SID mean/gain/tail, MDN); orientations synthetic-driver-verified.
- `dynotears.py` — native DYNOTEARS-analog (lagged sparse regression).
- `scratch_effects.py` — VAR-LiNGAM + SINDy (scratch venv; reads `worms.npz`).
- `run_pcmci.py` — PCMCI+ (scratch venv; N=80, τ configurable).
- `metric.py` — the correspondence-vs-lag curves, relative normalizations, per-lag nulls,
  variance controls, shape statistics, worm-bootstrap CIs.
- `run_phase1.py` / `run_metric.py` / `run_contrast.py` — the phase drivers.

## Reproduction (phase by phase)

```bash
cd sid_elegans; MAIN="PYTHONPATH=..:../SBTG ../.venv/bin/python"; SCR="/tmp/baseline_feas/bin/python"

# 0. (once) regenerate the corrected connectome + prepared data
#    ../.venv/bin/python ../SBTG/pipeline/01_prepare_data.py --impute-missing --full-traces

# corrected structural/monoamine + advanced-method reports
eval "$MAIN -m sid_elegans.run_eval"
eval "$MAIN -m sid_elegans.run_deconv"
eval "$MAIN -m sid_elegans.run_holistic"

# lag-resolved analysis
eval "$MAIN -m sid_elegans.lagres.run_phase1"   # main effects + writes /tmp/worms.npz
$SCR sid_elegans/lagres/scratch_effects.py       # VAR-LiNGAM + SINDy  (~2 min)
$SCR sid_elegans/lagres/run_pcmci.py 20 0.05     # PCMCI+ N=80 tau20   (~3.5 h)
eval "$MAIN -m sid_elegans.lagres.run_metric"    # curves + figures + SYNTHESIS.md
eval "$MAIN -m sid_elegans.lagres.run_contrast"  # the paired within-SID bootstrap (headline)
```

## Tests

```bash
PYTHONPATH=..:../SBTG ../.venv/bin/python -m pytest sid_elegans/tests -q     # 13 adapter tests
```

## Reports & artifacts (all in `output/`)

| file | contents |
|---|---|
| `FULL_REPORT.md` | **the complete narrative** (start here) |
| `REPORT.md` | corrected structural + monoamine comparison |
| `ADVANCED_METHODS.md` | multi-timescale bank, σ-ledger, stability tuning, two-head |
| `CORRECTION_full_data.md` | the tyramine retraction + source-variance control |
| `FINAL_deconv_mdn.md` | deconvolution (#1) + MDN (#3): the serotonin result |
| `HOLISTIC_ASSESSMENT.md` | all-methods × all-targets overall verdict |
| `lagres/LAGRES_RESULT.md` | **the lag-resolved payoff** (neuropeptide channel×lag) |
| `PLAN_lag_resolved.md` | the pre-registered lag-analysis design |
| `*.csv`, `*.json`, `*.png`, `lagres/*` | data tables and figures referenced above |
