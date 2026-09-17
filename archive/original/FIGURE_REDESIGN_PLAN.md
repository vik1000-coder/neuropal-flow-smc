# Figure redesign: clarity first

Status: the user requested expansion after the two prototypes. The [clarity-first figure set](results/figure_story_20260831/README.md) now adds 20 figures, with a [single visual gallery](results/figure_story_20260831/GALLERY.md) containing all 22 views. The [two prototypes](results/figure_redesign_prototypes_20260831/README.md), existing technical figures, dashboard files, models, and scientific results remain unchanged. No new inference was run for this redesign.

The existing [figure collection](results/figure_atlas_20260831/README.md) is a technical record. It is not the presentation template for the next version. Keep it available; do not delete or overwrite it.

## Why the current figures are difficult

They ask the reader to learn the method, decode unfamiliar metrics, compare results, and understand the audit at the same time. F03 combines several reference metrics and methods. F04 mixes lag curves with a statistical table. F08b presents six control comparisons before the reader has an intuitive picture of the underlying response.

Larger fonts alone will not solve that. Each new figure needs one question, one main visual, and a short explanation of what can be concluded.

## Proposed reading sequence

These are separate views, not seven panels on one page. The count is provisional: split a view if doing so makes it easier to understand.

| View | One question | What the reader sees |
| --- | --- | --- |
| 1. The recordings | What data do we have? | A few labeled, real activity traces from one recording, aligned with its actual stimulus strip. A short cohort line: 17 worms, 54 pooled head classes. |
| 2. The sampling idea | How does changing source history create an estimated effect? | One shared observed history branching into lower-source and higher-source compatible histories, followed by the target's predicted responses. Label what changes and what is held fixed. |
| 3. Cook and Randi | How does progressive SMC compare with published SBTG? | One horizontal plot of AUROC differences and existing intervals, with a clear zero line. Four reference rows; no second metric or extra model family on this view. |
| 4. Neuromodulator lags | How does correspondence change with source lag? | One progressive-SMC neuropeptide curve, four tested lags in seconds, and the chance line. Other pathways receive a separate short evidence summary and individual companion plots. |
| 5. Mean predictions | Does a selected mean effect exceed sampling noise? | One neuron pair, one outcome, one timing, and the matched-control contrast with uncertainty. Additional pairs repeat the same format on separate views. |
| 6. Distributional predictions | Is the prediction about a mean shift, a spread change, or another distributional change? | Separate case views for these distinct questions. Never put mean, log-SD, and W1 on a shared numerical scale. Teach spread with an explicitly illustrative sketch before showing a real estimate. |
| 7. Stimulus onset | Does activity—or the inferred effect—change around onset? | Separate observed-activity and model-effect views. Each displays one quantity with a clearly named comparison, rather than conflating stimulus response with a source→target interaction. |

## Design rules

- One sentence states the question. One dominant chart answers it.
- Normally one panel; at most two when they are the same comparison repeated.
- One outcome and one unit per chart. Use seconds, plain-language labels, and directly labeled curves.
- At most two focal colors plus neutral references. No legend lookup when a direct label will do.
- Do not shrink text to fit more content. Check readability at normal display size, not only when zoomed in.
- Keep the uncertainty, comparison baseline, and essential limitation visible. Put derivations, full test definitions, and provenance in the caption or linked technical record.
- No dense p-value tables inside plots. Show the relevant corrected test where available; say explicitly when the evidence is only exploratory or descriptive.
- A selected example must be labeled as selected. Retain the complete result inventory in supporting material.
- Do not turn unavailable support into a zero, non-significance into absence, or a displayed maximum into an identified lag.

## Scientific constraints the simpler design must retain

**Data:** use a documented, non-response-selected recording and name the displayed channels. Actual chemical identity is event metadata; the current generator conditions on binary any-stimulus history. The pooled head classes are not individually resolved cells.

**Method:** retain the distinction between the source window before the prediction cut and the target readout afterward. The repair can alter population history; the diagram must not imply an isolated physical intervention. Use actual saved predictive distributions only if they can be recovered correctly; otherwise mark teaching sketches as illustrations, never results.

**Cook/Randi:** the proposed chart has one metric: AUROC difference from published SBTG. It uses the existing paired-source intervals, which are exploratory and pointwise, not adjusted superiority tests. Compared pairs are matched; training histories differ. Absolute scores, AUPRC (including the Cook-gap exception), direct sampling, and SBTG-current remain in the companion results. Do not title the figure with an unqualified “our method is better.”

**Neuromodulators:** the main peptide curve is an identified example from the full network search. Keep a separate, simple summary of the named pathways with supported source counts: corrected evidence, no corrected evidence, or not evaluable. Preserve all individual pathway plots in the companion set. Alignment at any searched lag is different from a significant difference between lags; the reference represents molecular compatibility, not active signaling or transmission time.

**Predictions:** use exact source/target/context/timing controls. Keep the supported RIP→URB mean and W1 examples, but do not call W1 a mean-independent shape effect. Treat ASH→OLQ spread as an exploratory case with missing matched controls. Missing evidence should be stated briefly, not hidden by a smooth-looking distribution plot.

**Stimuli:** observed onset-versus-quiet activity is not a source→target effect. The modeled question is a difference between high-minus-low contrasts across contexts. Chemical-event labels do not create a chemical-conditioned generator. Baseline quiet-time calibration cannot answer onset specificity.

## Supporting material, not the first reading path

Keep full matrices, all methods and metrics, per-pathway plots, class summaries and support denominators, historical80 sensitivity, predictive calibration, all candidate reruns, exact p/q tables, and provenance. These remain accessible and are referenced whenever they limit the interpretation. Moving them does not remove inconvenient outcomes or justify a broader claim.

## Execution: prototypes followed by the expanded set

The first step redrew **view 2 (sampling)** and **view 3 (Cook/Randi)**. The user then asked for more figures. The expanded set adds the recordings, seven separate pathway views, five prediction/concept views, three observed chemical-response views, and four selected modeled-onset views. Splitting the companion cases keeps outcomes and units separate. All old exports are preserved.

At normal viewing size, the reader should be able to answer:

1. What is being changed or compared?
2. What is being measured, and in which units?
3. What does the uncertainty permit us to conclude?

Use that as a readability criterion, not a claim that a user study has been completed. The new images receive normal-size visual review and saved-value checks; further refinements can follow user feedback. No new training, sampling, or significance testing is part of the redesign unless separately agreed.
