# Project progress & current state

Living index of where the analysis stands, so we don't confuse current with superseded work.
Updated 2026-07-10. **For the repo-wide map + code/results index, see the root `README.md`.**

## Current metric: biologically-principled lag-correspondence (`biolag`)
The lag-resolved analysis is being rebuilt around a **biologically-principled, per-reference,
peakedness-aware** metric (BATC — Band-Aligned Timescale Concordance), replacing the earlier
AUROC-vs-lag **log-slope** contrast (which assumed a universal "positive slope = good", ignored
peakedness, and was not contextualised per connection type).

- **Code:** `sid_elegans/biolag/` — `config.py` (EDITABLE expected-timescale table + bands +
  caveats), `references.py` (registry: structural + neuromodulator classes + specific
  transmitters/peptides), `run_curves.py` (Phase-1 descriptive curves).
- **Results:** `sid_elegans/output/biolag/` — `overview_grid.png`, `curves/<reference>.png`,
  `correspondence_curves.csv`, `band_peak_summary.csv`, `README.md`.
- **Config is provisional** (pending neurobiologist review); the active band scheme is **coarse
  fast/slow** (the 4-band is documented but not resolvable at the current calcium-kernel floor).

### Status
- [x] Phase 1 — descriptive correspondence curves per (channel × reference), auditable graphs.
- [x] Phase 2 — null-referenced band concordance + controls (`PHASE2_RESULT.md`). Closed-form SID:
  peptidergic gain-slow concordance real above the circular-shift null (dBC +3.4, p=0.010) and
  dissociates from structure, BUT **collapses when the global brain-state mode is removed**
  (p=0.69). Source-variance control null by construction.
- [x] Neural-score corroboration (`TWOHEAD_RESULT.md`) — the null-contrast-tuned two-head DSM
  (SBTG-style neural score + SID readout) does **not** reproduce the effect (primary p=0.89).
- **Conclusion:** the lag signature is **closed-form-SID-specific and global-state-entangled** — a
  well-characterised lead, not a robust directed-coupling result. Remaining bar is external: more
  animals, a global-state-preserving design, and/or functional targets.
- [x] Neural-score corroboration (two-head, null-contrast tuned) — does not reproduce the effect.
- [x] **Paper fully revised** (`paper/main.tex` → `main.pdf`, 17 pp) — complete detailed writeup of the
  entire arc through the triangulated, global-state-entangled conclusion. Old log-slope-era writeup
  archived as `paper/main_v1_logslope_snapshot.tex`.
- [x] **Adversarial review of the paper** (5 reviewers + adjudicator) — verdict: *factually sound and
  honestly pitched*; ~16 headline numbers re-verified against source files, all matched. Fixed: two
  must-fixes (two-head source filter 5→20 s; abstract ΔBC −0.8→−0.5 observed), p 0.89→0.88 (×4), MDN
  attribution (not a "SID variant"), and disclosures (de-circularisation controls not run; neural arm
  25 vs 100 surrogates).
- [x] **Use-all-the-data investigation** (`sid_elegans/output/biolag/ALLDATA_RESULT.md`). Built
  **ACMMA** (`sid_elegans/acmma.py`) — available-case multivariate moment assembly, a validated
  (exact, Spearman/Pearson 1.000), no-imputation all-data estimator that doubles gain-connectome
  stability (0.29→0.56). Definitive tests: the peptidergic dBC headline (+3.35) is specific to the
  80% held-out split; at 100% rows it halves to +1.16 (ns), and with all 28 worms via ACMMA it is
  −0.41 (CI includes 0). **Conclusion: the peptidergic gain-slow signature does NOT robustly
  replicate** — every principled lever (held-out split, all-data ACMMA, global-mode, DSM, neural DSM,
  variance-VAR, imputation) agrees. The methods (SID, ACMMA, the control battery) are the durable
  contribution.
- [x] **Amines + denoising SM (comprehensive completion)** (`ALLDATA_RESULT.md` addendum,
  `run_amines_dsm.py`, `amines_dsm.json`). Ran ACMMA on all 28 worms across every monoamine
  (dopamine/serotonin/tyramine/octopamine + pooled) and a DSM σ-sweep. **Every transmitter and every
  σ → dBC CI includes 0.** So the negative is now comprehensive: no neuromodulator (aminergic or
  peptidergic), no score-matching mode (Hyvärinen or denoising), all the data → no robust lag-resolved
  distributional signal against these anatomical/receptor-expression targets. Only untried avenue with
  upside = **functional** targets (Randi/Leifer atlas; needs `wormneuroatlas`, not installed).
- [x] **Docs indexed & organized** — root `README.md` rewritten as the authoritative map (conclusions,
  code map, narrative arc, current-vs-superseded, reproduce); `output/biolag/INDEX.md` added;
  `FULL_REPORT.md` banner-marked partially-superseded.
- [ ] **Paper revision** to report the neuromodulator signature as a non-replicating lead and add
  ACMMA + the all-data/amines/DSM result as the capstone control (pending user).
- [ ] (optional) SBTG per-lag volatility channel — heavier pipeline re-run; deferred pending user.
- [ ] (optional) functional-target test (Tier 3) — needs `pip install wormneuroatlas`.

### Key caveats (from the adversarial metric review) — see `output/biolag/README.md`
Calcium/OASIS kernel floor (~1.5–12 s) ⇒ only **coarse fast/slow** + **relative/within-fit**
readouts are defensible; serotonin is biophysically **mixed** (MOD-1); tail channel caveated;
global-brain-state confound needs its own control.

## Superseded / archived
- `sid_elegans/output/archive/logslope_metric/` — the old log-slope contrast results
  (`LAGRES_RESULT.md`, `SYNTHESIS.md`, `contrast_logslope.json`, figure). **Deprecated.**
- Cached per-lag effect matrices live in `sid_elegans/output/lagres/` (`effects_main.pkl`, …)
  and are **reused** by `biolag` — data, not results.
- `paper/main.tex` + `paper/main.pdf` (17 pp) — the **current, biolag-era** paper: it covers the whole
  arc through Phase 2 + the two-head neural check and lands on the "global-state-entangled lead"
  conclusion. It **predates the ACMMA / all-data / amines-DSM work** (steps above), so its
  lag-resolved conclusion is now superseded by `biolag/ALLDATA_RESULT.md` and awaits a revision.
  (The genuinely old log-slope-era draft is the separate `paper/main_v1_logslope_snapshot.tex`.)

## Robustness work that still stands (metric-independent)
`paper/{compare_approaches,robustness_worms,adjudicate}.py` + JSONs: the partial-vs-marginal
finding, the worm-count sweep, and the circular-shift surrogate null (p=0.008) — these are about
the *coupling*, not the specific lag summary, and carry over.
