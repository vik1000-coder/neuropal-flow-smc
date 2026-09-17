"""Final scientific interpretation; reads saved results, never samples or fits."""
from common import *
import pandas as pd

B=R/'analysis_stopped_20260916';F=R/'final_review_20260916'
LABEL={'randi_wild_type':'Randi functional','cook_struct':'Cook structural','cook_chem':'Cook chemical','cook_gap':'Cook gap'}
def table(x):return x.to_markdown(index=False,floatfmt='.4f')
def link(path,label):return f'[{label}]({path})'

def run():
    metric=pd.read_csv(F/'primary_metric_audit.csv')
    rows=[];deltas=[];stabilities=[];predictions=[]
    for c in SET['cohorts']:
        d=pd.read_csv(B/c/'reference_deltas.csv');d['cohort']=c;deltas.append(d)
        for ref,g in metric[metric.cohort==c].groupby('reference',sort=False):
            a=g[g.method=='flow_ensemble'].iloc[0];s=g[g.method=='published_sbtg'].iloc[0]
            delta=d[(d.scope=='all_common')&(d.reference==ref)].iloc[0]
            rows.append({'cohort':c,'reference':LABEL[ref],'edges':int(a.n_edges),
                         'flow AUROC':a.auroc,'SBTG AUROC':s.auroc,'flow AP':a.auprc,'SBTG AP':s.auprc,
                         'AUROC difference':delta.delta_auroc,'95% interval':f'[{delta.delta_ci_low:.4f}, {delta.delta_ci_high:.4f}]',
                         'simultaneous lower':delta.simultaneous_one_sided_lower})
        s=pd.read_csv(B/c/'stability.csv')
        s=s[(s.channel=='endpoint_mean')&(s.context=='state_average')&(s.scope=='all')&(s.horizon==1)&(s.lag==1)].copy()
        stabilities.append(s[['cohort','comparison','spearman','sign_agreement']])
        p=pd.read_csv(B/c/'predictive_summary.csv');p.insert(0,'cohort',c)
        predictions.append(p[p.horizon.isin([1,4,32])][['cohort','method','horizon','energy','coverage90']])
    d=pd.concat(deltas,ignore_index=True)
    sens=d[d.scope.isin(['strong_fixed_support','endpoint_timing_matched_1frames','endpoint_timing_matched_8frames'])][['cohort','scope','reference','delta_auroc','simultaneous_one_sided_lower']]
    lag=pd.read_csv(F/'lag_counts.csv');lag=lag[(lag.channel=='endpoint_mean')&(lag.context=='baseline')]
    lag['minimum_adjusted_p']=lag.minimum_adjusted_p.map(lambda p:f'p={p:.6g}')
    conv=pd.read_csv(F/'particle_convergence.csv');cost=pd.read_csv(F/'computation.csv')
    conv=conv.merge(cost,on=['cohort','method','particles'])
    conv=conv[['cohort','method','particles','resolved_reference_fraction','pooled_mae','mean_wall_seconds','mean_velocity_sample_evaluations']]
    numerical=pd.read_csv(F/'numerical_adjudication.csv')
    main=numerical[(numerical.normalization=='normalized_effect')&(numerical.scope=='valid_gap_ge_0.1')&(numerical.horizon.astype(str)=='1')]
    deriv=main[main.panel=='derivative'][['cohort','first','second','coverage','mean_abs_difference','median_mc_sd_first','median_mc_sd_second','spearman','sign_agreement']]
    solver=main[main.panel=='solver'][['cohort','coverage','mean_abs_difference','mean_abs_first','median_paired_se','median_change_over_se','spearman']]
    strict=pd.read_csv(F/'strict_reference_sensitivity.csv')
    strict=strict[strict.method!='primary_on_strict_mask'][['cohort','reference','method','status','n_edges','n_positive','auroc','auprc']]
    images=[]
    for name in ['reference_comparison','reference_lag_profiles','source_support','lag_effect_matrices','replication_stability','predictive_rollout','particle_convergence']:
        images.append({'figure':name,**{c:link(F/c/'figures'/f'{name}.png',c) for c in SET['cohorts']}})
    text=f'''**Flow / progressive bridge SMC replication — final stopped-study interpretation**

Prepared {now()}. This report supersedes the scientific interpretation in REPORT_STOPPED.md; that original report, its figures and every frozen worker remain preserved. Supplementary descriptive analyses are documented in {link(R/'amendments/003_final_interpretation_20260916/AMENDMENT.md','amendment 003')}. No model fitting or path sampling was performed after the user-directed stop.

**Verdict.** The fresh computational results support improved atlas ranking in a limited sense: Cook structural and chemical correspondence improves in both cohorts and survives the tested strong-support and endpoint-timing sensitivities. The broader claim that the complete derivative-based lag matrix is reliably better is not established. Smaller contrasts become substantially noisier; the clean cohort has no multiplicity-corrected lag-versus-lag-1 effects; the strict complete-case clean matrix has no surviving entries; all clean primary models reached their training cap. Historical results cannot establish independent-animal replication because their prepared inputs retain head/tail pseudo-pairing and donor copying.

**What was rerun and what was reused.** There are 30 new primary flow fits (two cohorts × five recording folds × three generator seeds), 120 new primary progressive N64 archives, 40 independent-Monte-Carlo repeats, 40 direct N4096 comparisons, 60 endpoint-timing comparisons, 1,440 diagnostic archives and 30 factual predictive evaluations. The source implementation and recordings were copied and hash-pinned. Published SBTG manuscript matrices were reused as the frozen comparator; SBTG was not refitted. The released lag-1 comparator is the hybrid manuscript matrix; other published lags follow that release's production lineage. The longer-training extension stopped at 9/15 fits and 34/60 sampling archives and contributes nothing to the primary ensemble. This is computational reproduction on existing data, not prospective biological replication or a fully independent rewrite of the flow/SMC method.

**Primary atlas correspondence.** The fixed primary comparison is nominal source lag 1, forecast horizon 1, state-average endpoint mean, averaged over three fitted seeds. Matrices are target-by-source; scoring uses absolute magnitudes, common eligible off-diagonal pairs and the same masks for flow and SBTG. AP denotes average precision, the implementation's AUPRC summary. Cook positives have adjacency greater than zero. Randi positives have q < .05; equivalently negative pairs have q_eq < .05 unless already positive; ambiguous pairs are excluded. These scores do not test effect signs or prove causal edges.

{table(pd.DataFrame(rows))}

Intervals are 10,000 paired source-column bootstrap intervals with matrices held fixed. The simultaneous lower bound uses the .05/4 lower quantile for the four references within a panel. The historical all-common comparison is the prespecified primary family; the clean cohort and other masks are sensitivities. These are conditional bootstrap summaries, not full refit/animal uncertainty or family protection across every secondary analysis. All 32 released reference-by-lag metric reproductions match the saved release to numerical precision; the final audit independently recomputed both cohorts' primary AUROC/AP/counts from their saved matrices.

Historical point differences are positive for all four panels, with positive simultaneous lower bounds. Clean structural and chemical differences have positive simultaneous lower bounds; clean Randi and gap differences remain unresolved under that rule. Clean gap AP is slightly lower for flow despite higher AUROC, so superiority is not uniform across metrics.

**Support, timing and origin sensitivities.** Strong fixed support requires validity and genealogy qualification on at least 80% of episodes at every lag, retaining 57/80 historical sources and 33/54 clean sources. Structural and chemical gains persist under this mask in both cohorts. Randi and gap gains are less robust. The timing-1 comparison uses flow source lag 0 plus horizon 1 versus SBTG lag 1; timing-8 uses flow lag 7 plus horizon 1 versus SBTG lag 8. Even matching endpoints does not equate a four-frame soft repaired-history contrast with the SBTG estimand.

{table(sens)}

The secondary timing-8 panels show positive lower bounds for all references in both cohorts, but do not replace the fixed primary comparison or establish lag identification. Historical origin-stratified results are also heterogeneous: the head/head comparisons carry the gains, whereas tail/tail Cook structural AUROC differs by −0.0822 (ordinary 95% interval −0.1576 to −0.0050); cross-origin intervals span zero. These inherited origin labels and dependencies prevent a clean biological interpretation of the historical strata.

**Strict gap sensitivity, newly scored in this supplement.** The primary normalization divides by max(abs(achieved source gap), .10), retaining invalid episodes. The strict arrays instead require validity and a positive achieved gap of at least .10. Ordinary event, phase, recording and seed means propagate any invalid value. This deliberately demanding complete-case rule retains 12 historical source columns / 948 off-diagonal entries and zero clean entries. Zero clean coverage is not proof that every individual clean episode fails; it means no source passes every contributing episode required by this aggregation.

{table(strict)}

On the historical strict mask the primary and strict scores coincide, because every surviving denominator already exceeds the floor. Historical Randi has only 196 labeled pairs and ten positives; its strict AUROC is lower than SBTG. Cook point estimates remain higher on that restricted subset. These post-hoc point scores have no additional inferential claim. The clean strict sensitivity cannot be scored, so clean primary results cannot be presented as robust to this strict exclusion rule. No available-episode averaging was substituted to manufacture coverage.

**Reproducibility of the averaged matrices.** At lag 1 / horizon 1:

{table(pd.concat(stabilities,ignore_index=True))}

Independent sampling repeats correlate around .90, while separate training seeds correlate only .55–.67. Thus training variability is material even when the averaged matrix is relatively stable to Monte Carlo repetition. Agreement with direct N4096 is around .87; correlation alone does not establish small per-cell error. The full saved table retains all seed pairs, lags, horizons and support scopes.

**Particle accuracy and computation.** A direct N16384 reference is considered resolved only where all three independent reference runs pass validity, minimum ESS is at least 12 and normalized-response SD is at most .05. Only 48.5% of historical and 41.8% of clean diagnostic cells meet that rule. The table below pools absolute error using qualified cell counts. Original figures average the eight lag/phase strata instead; both versions are retained and lead to the same qualitative interpretation. Direct N16384's own error uses a leave-one-run-out reference rather than comparing each run to a mean containing itself.

{table(conv)}

Progressive error falls as N rises from 64 to 512 in both cohorts. This supports particle convergence over the tested range, but N64 is not already numerically settled: its qualified-reference error is larger than direct N4096. N64 is faster in these diagnostic runs, so this is a speed/accuracy tradeoff. N128 uses more velocity evaluations than direct N4096 yet has larger qualified-reference error; no broad efficiency advantage is established. A direct bank is shared across sources, whereas these progressive diagnostics sample eight selected sources, so these timings must not be extrapolated into a universal full-matrix speed claim. No exact equal-budget performance frontier was run. Unresolved direct-reference cells cannot settle which sampler is accurate. High terminal ESS after tempering/resampling is not a substitute for ancestry or accuracy checks.

**Derivative approximation: numerical adjudication.** The diagnostic uses 512 progressive particles, three MC seeds, folds 0/1, baseline/onset for the first event, lags 1/4, eight sources and all targets/horizons. Source targets move symmetrically around the quartile midpoint, while clamp bandwidth remains fixed. The following table uses horizon 1 and requires all six runs in each paired comparison to pass validity and achieved gap ≥ .10. Coverage is relative to that diagnostic panel, not the entire production matrix. `first`/`second` are fractions of the full requested source contrast. MC SD is across the three runs, not biological uncertainty.

{table(deriv)}

Full-to-quarter rank agreement is only .499 historical / .501 clean, with sign agreement .682 / .691. On those same eligible cells, median MC SD grows from .0387 to .1216 historical and .0530 to .1414 clean. Mean absolute changes (.0722 / .0830) exceed mean absolute full-contrast effects on those cells (.0552 / .0663). Therefore the available calculations do not resolve a stable per-cell derivative as contrast shrinks.

This is chiefly a failure to demonstrate precision/convergence, not proof that the true softened response is nonlinear: the median paired change is only about .81–.83 estimated standard errors in the full-to-quarter comparison. Three repetitions provide a weak noise estimate. Requested-gap normalization gives the same rank-instability pattern, so it is not solely an achieved-gap denominator artifact. Across all six horizons, full-to-quarter rank agreement is .573 / .586, still insufficient to claim an invariant derivative matrix. Broader positive-gap and conservative .10-gap results, both normalizations and paired cell-level evidence are preserved in the supplement. The primary estimator remains a regularized finite contrast; fixed bandwidth and factual anchoring also distinguish it from a causal intervention derivative.

**Flow integration: numerical adjudication.** Original versus doubled Heun steps were compared with matched MC seeds, N128, folds 0/1, baseline first event, lags 1/16 and eight sources. This assesses the combined solver-plus-SMC estimator; small integration changes can alter discrete resampling decisions and later particle paths.

{table(solver)}

The horizon-1 estimates correlate .832 historical / .877 clean, with mean absolute differences .0377 / .0358. Typical changes are within the estimated paired MC uncertainty, so the saved data do not isolate a dominant deterministic ODE error. Conversely, this is not an equivalence demonstration: differences remain substantial relative to mean absolute effects, and only three MC runs were available. Larger horizons and requested-gap results are in the tables. The reported fraction exceeding two estimated standard errors is descriptive only; with three runs, two SE is not a 95% threshold, and no new cellwise significance claims are made.

All identical-clamp controls have exactly zero raw MSE in both cohorts because low/high arms use coupled randomness. This is a symmetry check, not an independent-arm noise floor. It corrects the contradictory sentence in the preserved original report that described a nonzero zero-query MSE.

**Lag evidence after joint multiplicity correction.** The exact maximum-T calculation jointly includes endpoint mean and log-SD, baseline and onset-minus-baseline, effects and lag-minus-1 contrasts, all selected off-diagonal pairs, lags and horizons. The family contains 756,504 historical and 293,832 clean cells. Enumerating all two-sided recording sign orbits gives 2^19 and 2^16 null realizations respectively. The table distinguishes significant cells (pair × lag × horizon) from unique directed pairs:

{table(lag)}

The clean data support baseline mean effects (2,242 cells across 272 pairs) but no lag-minus-1 changes after correction; the smallest corrected clean baseline-mean lag p is .1476. Historical data have 92 lag-change cells across 29 pairs, but the pseudo-pairing and donor-copying lineage makes their recording-row inference descriptive rather than independent-animal evidence. Neither cohort has corrected onset-minus-baseline effects/lag changes or log-SD effects/lag changes. Absence of significance is not proof of identical lags. These conditional sign-flip results assume joint symmetry and do not account for shared fitted models, refitting or preprocessing selection uncertainty.

The separate lag-maximum atlas permutation analysis detects reference correspondence across the lag scan in several panels, including state-average mean for all four references in both cohorts. That is not a test that one lag differs from another. Changing lag also changes the repair boundary and anchor interval. No physical transmission-delay matrix is established.

**Factual predictive quality.** Each method uses the same deterministic held-out history bank, 32 free draws and horizons 1/2/4/8/16/32. Energy is lower-is-better. Selected horizons are shown; the complete summaries remain linked below.

{table(pd.concat(predictions,ignore_index=True))}

Historical flow beats persistence in energy at all six horizons, but ridge is better after horizon 1. Clean flow beats ridge throughout, but persistence is better from horizon 4 onward. Flow 90% interval coverage falls to .785 historical / .754 clean at horizon 32. The flow is therefore not uniformly the best or well-calibrated long-horizon predictor. These aggregate predictive comparisons are descriptive; no new significance test is implied.

**Implementation and provenance qualifications.** The code review establishes the following boundaries. MAT `norm_traces` are already normalized calcium signals, not raw camera fluorescence; classes average left/right and designated dorsal/ventral traces. Clean class selection uses pooled availability across both head strains before retaining OH16230, and is not fold-nested. Pooled affine normalization cancels under later training-fold scaling to a maximum checked error of 3.815e-6, but availability selection and historical imputation do not disappear. Training histories forward-fill internal missing values and exclude missing targets; the sampler also backfills leading missing values, but no registered episode overlaps those leading frames. The legacy L80 TCN has a 31-frame local convolutional receptive field with older-history influence through temporal GroupNorm. Model optimization/scaling is training-only; sampler nuisance quantiles/projection/thresholds use all non-test recordings including validation. Source constraints and the rank-12 anchor are soft; the source is exempt from anchoring only during its four-frame window. There is no observed neural target beyond the forecast cut in the sampled future.

Every clean primary fit reached the 50-epoch cap and triggered the planned extension. The incomplete extension cannot settle whether further training would change these findings. No result here validates physical causal effects, receptor mechanisms or prospective biological generalization.

**Reviewed figures and evidence.** All fourteen original plots were inspected. Separate copies correct overlapping subplot labels and add scope annotations; source values and original plots remain preserved. Particle plots use the original equal-stratum error averages; the table above additionally reports pooled error. Matrix plots show diagonal responses for context, but all atlas/lag scores exclude them. Heatmap indices follow the neuron order stored in each primary archive.

{table(pd.DataFrame(images))}

{link(R/'code_review_20260916/REVIEW.md','Code-path review')} · {link(R/'PROTOCOL.md','Original protocol')} · {link(R/'ANALYSIS_IMPLEMENTATION.md','Analysis specification')} · {link(F/'numerical_adjudication.csv','Derivative/solver summaries')} · {link(F/'strict_reference_sensitivity.csv','Strict atlas scores')} · {link(F/'lag_counts.csv','Complete lag counts')} · {link(F/'primary_metric_audit.csv','Recomputed primary metrics')}

The detailed original analysis tables are in {link(B/'historical80','historical80')} and {link(B/'clean54','clean54')}; these include all masks, lag scans, prediction horizons and generator comparisons. Supplementary paired cells and reviewed figures are in {link(F,'final_review_20260916')}. The preserved original report is {link(R/'REPORT_STOPPED.md','REPORT_STOPPED.md')}; use this final interpretation for scientific conclusions. Snapshot, worker, checkpoint and artifact hashes preserve provenance. The original copied suite recorded 309 passing tests/two skips, the original replication suite passed eleven tests including its direct-sampling oracle, and the final deterministic rerun passed ten tests with the sampling oracle deliberately excluded. Analytic-oracle success validates a controlled calculation, not the fitted biological model.

**Supported claim for this stopped study:** newly trained flow-based finite contrasts improve structural/chemical atlas ranking under the reported comparisons, while derivative convergence, clean lag specificity, universal sampler efficiency and complete training-convergence robustness remain unestablished.
'''
    (R/'REPORT_FINAL.md').write_text(text)
    print('FINAL_REPORT_WRITTEN',R/'REPORT_FINAL.md')

if __name__=='__main__':run()
