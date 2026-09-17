# Score-Identified Distributional Dynamics on *C. elegans* — biological analysis index

**For a theory-first explanation of the experiments, read `THEORY_EXPERIMENT_MAP.md` first.** It
connects the consolidated SID manuscript's estimand hierarchy and eight decisive experiments to the
results we actually ran. Use `EXPERIMENT_INDEX.md` as the repository-wide evidence and
supersession ledger; its machine-readable companion is `EXPERIMENT_REGISTRY.csv`.

This file remains the detailed map of the real *C. elegans* biological analysis arc. Where it
disagrees with `EXPERIMENT_INDEX.md` or a later explicitly adjudicating report, defer to the later
artifact.

---

## 1. What this is

We develop **score-identified dynamics (SID)**: estimate the full one-step conditional predictive law
`ρ(y | history)` of neural activity — not just the conditional mean — and read out directed
**mean / gain (∂logVar) / tail** interaction channels. The hypothesis: the *distributional* (gain/tail)
channels should expose slow **neuromodulation** (monoamines, neuropeptides) that conditional-mean
methods are blind to. We test this on whole-brain NeuroPAL calcium imaging of *C. elegans*.

## 2. Bottom line (current conclusions)

| | status |
|---|---|
| **The method (SID)** — closed-form score-matching estimator + distributional readouts | ✅ **Solid.** Decisively validated on synthetics: the gain channel recovers a conditional-variance kernel invisible to every mean method (corr **0.998**). |
| **Pipeline correction** — fixed the prior SBTG connectome-orientation bug (+ 28 others) | ✅ **Solid.** Verified empirically; all methods re-scored on the corrected truth. |
| **Real-data competitiveness** (holistic, across all targets) | ✅ **Solid.** SID's mean channel is the equal-best single method; MDN wins metabotropic serotonin (0.78). |
| **ACMMA** — a new, validated, no-imputation all-data estimator | ✅ **Solid.** Reproduces the complete-case fit exactly (corr 1.000); doubles gain-connectome stability (0.29→0.56). |
| **The neuromodulator LAG signature** (the biology we chased) | ❌ **Does not robustly replicate.** See below. |

**The lag-resolved neuromodulator result is a comprehensive negative.** Across **all transmitters**
(dopamine, serotonin, tyramine, octopamine, pooled amines, and neuropeptides), **both score-matching
modes** (Hyvärinen + denoising), and **all 28 worms** via the validated ACMMA estimator, there is
**no robust lag-resolved distributional neuromodulator signal** against these (anatomical /
receptor-expression) targets. An early "significant" peptidergic result (dBC +3.35) turned out to be
fragile to arbitrary estimation choices (an 80% held-out-rows setting; ridge; data amount) and
collapses under a global-brain-state control. **The methods are the contribution; the biology is an
honest null** (the one untried avenue with upside — *functional* targets via the Randi/Leifer atlas —
needs the `wormneuroatlas` package, not installed).

## 3. Start here (reading order)

1. **`THEORY_EXPERIMENT_MAP.md`** — theory-to-experiment map and current claim ladder.
2. **`EXPERIMENT_INDEX.md`** — repository-wide evidence hierarchy and current-vs-superseded map.
3. **This file** — the detailed biological analysis map.
4. **`PROGRESS.md`** — chronological log of the biological investigation; it predates later suites.
5. **`sid_elegans/output/biolag/ALLDATA_RESULT.md`** — the **final** biological result doc (ACMMA, all-data,
   amines, DSM → the comprehensive negative).
6. **`paper/main.pdf`** — the full written paper. ⚠️ It captures the arc *through* the two-head
   neural check and reports the lag signature as a "global-state-entangled lead"; it **predates** the
   ACMMA / all-data / amines work (steps 11–13 below), so its lag-resolved conclusion is now
   **superseded by ALLDATA_RESULT.md** and awaits revision.
7. **`sid_elegans/output/biolag/INDEX.md`** — index of the lag-resolved (biolag) results in order.

*Background / source materials (not part of the arc):* `score_identified_dynamics.pdf` +
`score_dynamics_empirics.pdf` are the **theory working papers**; `sid_neuro_plans.md` is the original
developer handoff spec; `sid_elegans/README.md` + `sid_elegans/PLAN_lag_resolved.md` are older
module-level notes (predate this index — defer to this file where they disagree).

## 4. Code map

### Package — `sid_neuromod/` (installed, `pip install -e sid_neuromod`)
The SID theory package: closed-form quadratic-score estimator, filter bank, mean/gain/tail readouts,
HAC/sup-t inference, PIT/e-process monitoring, synthetic systems + oracles. **68 tests, 90% coverage.**
Key: `src/sid_neuromod/models/quadratic_score.py` (the closed-form estimator, Algorithm 1).

### Adapter — `sid_elegans/` (real-data analysis)
**Core estimators**
| file | what |
|---|---|
| `estimator.py` | **SID** distributional connectome — closed-form Hyvärinen (σ=0) / denoising (σ>0) score matching; mean/gain/tail readouts. Supports `sigma_frac`. |
| `acmma.py` | **ACMMA** — available-case multivariate moment assembly (the validated all-data estimator, no imputation). Supports `sigma_frac` (denoising). |
| `pairwise.py` | pairwise/marginal distributional connectome (per-edge worm support). |
| `twohead.py` | two-head **neural DSM** model (the SBTG lineage; MLP trunk + heads, Adam). |
| `mdn.py` | mixture-density network (non-Gaussian, maximum-likelihood comparator). |
| `impute.py` | random-donor imputation (shown to fail — reference/foil). |

**Data & ground truth**
| file | what |
|---|---|
| `combined_data.py` | pools all 28 worms (OH16230 + OH15500); complete-case vs imputed loaders; deconv + scale-preserving standardization. |
| `ground_truth.py` | corrected Cook connectome + Bentley monoamine/neuropeptide layers (`[post,pre]`); `load_leifer` (functional atlas, needs `wormneuroatlas`). |
| `deconv.py` | OASIS AR(1) calcium deconvolution. |

**Eval / stats:** `evaluate.py`, `significance.py`, `stability.py` (split-half tuning), `baselines.py`.

**Other methods / earlier-phase run scripts (exploratory or superseded — not in the main arc):**
`multiscale.py` (multi-timescale filter-bank connectome + denoising σ-ledger), `data.py` (lower-level
SBTG trace loader), and the run scripts `run_eval.py`→`REPORT.md`/`eval_results.csv`,
`run_expressive.py`, `run_scaling.py`→`scaling_results.json`, `run_mdn_verify.py`→`mdn_verify.json`.

**Lag-resolved v1 — `lagres/` (log-slope metric, ⚠️ SUPERSEDED):** the first lag analysis
(`effects.py`, `metric.py`, `run_*`). Produced the cached per-lag effect matrices
(`output/lagres/effects_main.pkl`, reused later) and the log-slope contrast. Results archived (§7).

**Lag-resolved v2 — `biolag/` (band-concordance metric, ✅ CURRENT):** the biologically-principled
redesign.
| file | what |
|---|---|
| `config.py` | **EDITABLE** expected-timescale table + fast/slow bands + caveats (for neurobiologist review). |
| `references.py` | reference registry: structural + neuromodulator networks, class + specific transmitters/peptides. |
| `metric.py` | band-concordance metric + circular-shift null + global-mode helpers. |
| `run_curves.py` | Phase 1: auditable correspondence curves (all estimators × all references). |
| `run_phase2.py`, `plot_phase2.py` | Phase 2: confirmatory band-concordance + controls (SID). |
| `tune_twohead.py`, `run_twohead_phase2.py` | neural-DSM null-contrast tuning + its biolag run. |
| `run_comprehensive.py` | SID Hyvärinen vs denoising vs baselines, 28w-imputed vs 6w-clean. |
| `run_phase2_worms.py` | worm-count robustness (14w/20w clean). |
| `tune_ridge_alldata.py` | ridge/stability sweep on imputed data (shows imputation is unfixable). |
| `run_acmma_phase2.py`, `acmma_signmap.py` | ACMMA band-concordance + the data-vs-ridge sign map. |
| `run_amines_dsm.py` | the monoamines + DSM σ-sweep under ACMMA (final follow-up). |

### Paper — `paper/`
`main.tex` / `main.pdf` (the writeup); `make_figures.py` + `figures/` (Figs 1–12);
`compare_approaches.py`, `robustness_worms.py`, `adjudicate.py` (+ JSONs — the robustness battery);
`main_v1_logslope_snapshot.tex` (archived earlier draft).

## 5. The investigation, in order (the narrative arc)

Each step lists the question, the code, the result artifact, and the verdict.

| # | step | code | result artifact | verdict |
|---|---|---|---|---|
| 1 | Method + synthetic proof | `sid_neuromod/` | `sid_neuromod/output/synthetic/` | ✅ gain recovers Kalman kernel (0.998) |
| 2 | Data + pipeline audit/fix | `combined_data.py`, `ground_truth.py` | (bug fix in loaders) | ✅ orientation bug fixed |
| 3 | Distributional connectome + holistic eval | `estimator.py`, `run_holistic.py` | `output/HOLISTIC_ASSESSMENT.md`, `holistic_auroc.csv` | ✅ SID competitive; MDN serotonin 0.78 |
| 4 | Tyramine over-claim retracted | `run_holistic.py` | `output/CORRECTION_full_data.md` | ✅ artifact (source-variance) retracted |
| 5 | Deconvolution + MDN | `deconv.py`, `mdn.py`, `run_deconv.py` | `output/FINAL_deconv_mdn.md` | serotonin lead (holistic, not lag) |
| 6 | Lag-resolved v1 (log-slope) | `lagres/` | `output/archive/logslope_metric/` | ⚠️ SUPERSEDED (weak metric) |
| 7 | Robustness battery | `pairwise.py`, `impute.py`, `paper/{compare_approaches,robustness_worms,adjudicate}.py` | those `.json` | partial-specific, power-limited, surrogate-real (p=0.008), single-instrument |
| 8 | Metric redesign (band concordance) | `biolag/{config,references,metric,run_curves}.py` | `output/biolag/README.md`, `overview_grid.png` | biologically-principled, coarse fast/slow |
| 9 | Phase 2 (SID + controls) | `biolag/run_phase2.py` | `output/biolag/PHASE2_RESULT.md` | passes trivial/structural controls, **collapses under global-mode** |
| 10 | Neural-DSM corroboration | `biolag/{tune_twohead,run_twohead_phase2}.py` | `output/biolag/TWOHEAD_RESULT.md` | ❌ neural DSM does not reproduce |
| 11 | Use-all-data: imputation vs clean; SID SM vs DSM vs baselines | `biolag/run_comprehensive.py` | `output/biolag/COMPREHENSIVE_RESULT.md` | imputation confounds; clean-6w is the ceiling |
| 12 | ACMMA (validated all-data estimator) + definitive test | `acmma.py`, `biolag/run_acmma_phase2.py`, `acmma_signmap.py` | `output/biolag/ALLDATA_RESULT.md` | ✅ ACMMA works; ❌ effect fragile / does not replicate |
| 13 | Amines + DSM (comprehensive completion) | `biolag/run_amines_dsm.py` | `output/biolag/ALLDATA_RESULT.md` (addendum) | ❌ all transmitters + all σ → null |

## 6. Results index (artifact → what it shows → status)

**Current / load-bearing**
- `output/biolag/ALLDATA_RESULT.md` — **the final result**: ACMMA + all-data + amines + DSM → comprehensive negative.
- `output/biolag/{PHASE2,TWOHEAD,COMPREHENSIVE}_RESULT.md` — the steps that built to it.
- `output/HOLISTIC_ASSESSMENT.md` (+ `holistic_auroc.csv`) — SID competitive across targets. Stands.
- `output/CORRECTION_full_data.md` — the tyramine retraction. Stands.
- `output/biolag/*.json` — raw numbers for every biolag result (see `biolag/INDEX.md`).
- `paper/{compare_approaches,robustness_worms,adjudicate}.json` — the robustness battery numbers.
- earlier-phase raw numbers: `output/{eval_results,deconv_results}.csv`,
  `output/{scaling_results,mdn_verify}.json` (see `REPORT.md`, `FINAL_deconv_mdn.md`).

> **Note:** the interpretive `*_RESULT.md` / `*.md` docs are **hand-authored narratives**, not
> script-emitted. The **scripts regenerate the CSV/JSON/PNG** artifacts only. So "reproduce" (§7)
> means re-deriving the *numbers*; the prose docs are the human record of what those numbers mean.

**Superseded / archived (kept for provenance — do NOT cite as current)**
- `output/FULL_REPORT.md` — the older full narrative; its **lag-resolved conclusion predates the
  ACMMA/all-data negative** and is superseded by `ALLDATA_RESULT.md`. (Its Tiers 1–2 still hold.)
- `output/{REPORT,ADVANCED_METHODS,FINAL_deconv_mdn}.md` — earlier-phase records; valid for their
  specific results, predate the final arc.
- `output/archive/logslope_metric/` — the log-slope-metric results (superseded by biolag band concordance).
- `output/lagres/` — v1 lag results (`lagres_summary.csv`, `contrast_logslope.json`, `fig_lag_*.png` —
  superseded, duplicated in `archive/logslope_metric/`) **plus** the still-used **cached effect
  matrices** (`effects_main.pkl`, `scratch_effects.npz`, `pcmci_effects.npz` — data, reused by biolag).
- `paper/main_v1_logslope_snapshot.tex` — earlier paper draft.

## 7. Reproduce / verify

```bash
python -m venv .venv && ./.venv/bin/pip install -e sid_neuromod
./.venv/bin/python -m pytest sid_neuromod/tests -q                    # 68 tests
PYTHONPATH=.:SBTG ./.venv/bin/python -m pytest sid_elegans/tests -q   # 13 adapter tests
```
Every result's **numbers** regenerate from its script (§5 table); the prose `*.md` docs are
hand-authored (see the note in §6). Heavy runs are compute-gentle by env vars
(`OPENBLAS_NUM_THREADS=3 nice -n 19 …`; several accept `*_NSURR/*_NBOOT/*_THROTTLE`). The paper's
figures: `PYTHONPATH=.:SBTG ./.venv/bin/python paper/make_figures.py`.

⚠️ **`biolag/run_acmma_phase2.py` now defaults to `ridge=0.01`** (the standard value that reproduces
the definitive result). The earlier `ridge=3.0` is an over-regularization artifact — do not use it.

**Key validation checks a reviewer can rerun quickly:**
- **ACMMA reproduces the complete-case SID fit at corr 1.000** →
  `PYTHONPATH=.:SBTG ./.venv/bin/python sid_elegans/biolag/validate_acmma.py` (committed; prints 1.000
  for the Hyvärinen case).
- the **synthetic gain-kernel recovery (0.998)** — `sid_neuromod/output/synthetic/hidden_thermostat/`.
- the **held-out-split fragility** of the headline (+3.35 at 80% rows → +1.16 at 100%) — the
  disentangling check in the transcript / `ALLDATA_RESULT.md`.

## 8. Environment notes
Main venv `.venv/` (numpy/scipy/sklearn/statsmodels/torch + `sid_neuromod`). Torch uses **MPS** for the
neural models. Causal baselines (VAR-LiNGAM/SINDy/PCMCI+) used a separate scratch venv; their outputs
are cached in `output/lagres/{scratch,pcmci}_effects.npz`. Prior project data + connectomes live in
`SBTG/` (audited, orientation-bug fixed here).
