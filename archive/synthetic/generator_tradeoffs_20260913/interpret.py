"""Summarize complete, audited results without changing any fitted model."""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
R=Path(__file__).resolve().parent;A=R/'analysis'

def main():
    assert (A/'independent_audit.json').exists()
    p=pd.read_csv(A/'predictive_dataset_means.csv');e=pd.read_csv(A/'effects_dataset_means.csv');d=pd.read_csv(A/'derivative_dataset_means.csv');f=pd.read_csv(A/'fit_diagnostics.csv');raw=pd.read_csv(A/'effects_raw.csv');solver=pd.read_csv(A/'solver_raw.csv')
    winners=[]
    for (law,region),group in p.groupby(['law','region']):
        s=group.groupby('family').energy.mean().sort_values()
        flow=float(s['flow'])
        winner=s.index[0];win=group[group.family==winner][['data_seed','energy']].merge(group[group.family=='flow'][['data_seed','energy']],on='data_seed',suffixes=('','_flow'))
        delta=win.energy-win.energy_flow
        winners.append({'law':law,'region':region,'lowest_mean_energy_family':winner,'energy':s.iloc[0],'runner_up':s.index[1],'runner_up_energy':s.iloc[1],'flow_energy':flow,'paired_delta_min':delta.min(),'paired_delta_max':delta.max()})
    winners=pd.DataFrame(winners);winners.to_csv(A/'predictive_point_leaders.csv',index=False)
    effects=[]
    for (law,region,event),group in e.groupby(['law','region','event']):
        s=group.groupby('family').agg(error=('absolute_error','mean'),qualified_datasets=('all_initializations_qualified','sum'),datasets=('data_seed','nunique'))
        good=s[(s.qualified_datasets==3)&(s.datasets==3)].sort_values('error')
        effects.append({'law':law,'region':region,'event':event,'fully_qualified_families':len(good),'lowest_mean_error_family':good.index[0] if len(good) else 'unresolved','error':good.error.iloc[0] if len(good) else np.nan,'runner_up':good.index[1] if len(good)>1 else 'none'})
    effects=pd.DataFrame(effects);effects.to_csv(A/'effect_point_leaders_complete_support.csv',index=False)
    costs=f.groupby('family').agg(mean_training_seconds=('training_seconds','mean'),max_training_seconds=('training_seconds','max'),fits=('family','size'))
    speed=p.groupby('family').sampling_seconds_per_history.mean();costs=costs.join(speed);costs.to_csv(A/'cost_summary.csv')
    calibration=p.groupby(['family','region'])[['coverage90','width90','tail_probability_abs_error']].mean().reset_index()
    derivative=d.groupby(['family','region','step'])[['absolute_error','mc_se']].mean().reset_index();derivative.to_csv(A/'derivative_summary.csv',index=False)
    stability=solver.groupby(['family','region'])[['mean_change_rmse','quantile_change_rmse','seconds24','seconds48']].mean().reset_index();stability.to_csv(A/'solver_summary.csv',index=False)
    # All-family disagreement is descriptive, not an OOD detector performance guarantee.
    per=raw.groupby(['law','data_seed','region','event','family']).agg(model_estimate=('model_estimate','mean'),oracle_estimate=('oracle_estimate','mean'),qualified=('qualified','all')).reset_index()
    disagreement=[]
    for key,g in per.groupby(['law','data_seed','region','event']):
        if len(g)==9 and g.qualified.all():
            disagreement.append(dict(zip(['law','data_seed','region','event'],key),between_family_sd=g.model_estimate.std(),mean_absolute_error=np.mean(abs(g.model_estimate-g.oracle_estimate))))
    spread=pd.DataFrame(disagreement);spread.to_csv(A/'qualified_disagreement.csv',index=False)
    corr=spearmanr(spread.between_family_sd,spread.mean_absolute_error).statistic if len(spread)>3 else np.nan
    text=f'''# What changed when we changed the generator?

All192 planned fit/evaluation tasks have completed; independent algebra/data/receipt audits passed. This is a new fixed-configuration synthetic comparison. It is not the missing original NeuroPAL study. It uses4 laws,3 independent data realizations/law and2 neural initializations/dataset. Seed ranges and means below are descriptive, not simultaneous confidence statements or optimized-family comparisons.

## Average predictive accuracy by setting

Lowest mean energy configurations and runner-up configurations are shown for every law/region. All candidates see the same data and evaluation histories. Paired differences compare the listed leader with flow over the same three datasets. Selection of the lowest observed mean is descriptive and will be optimistic; no new model selection or fitting follows these rankings.

{winners.to_markdown(index=False)}

## Conditional-effect accuracy: common and rare queries

This comparison holds the direct reference estimator fixed. Only families whose reference qualifies on all3 datasets and both neural initializations enter each row's ranking. The number of eligible families is shown; rows with few eligible models cannot adjudicate the excluded models. See paired_effect_differences.csv and effects_dataset_means.csv for all estimates, exclusions and shared masks. 'Rare' denotes the predeclared tail-directed soft query; actual support varies with history and is retained in effects_raw.csv.

{effects.to_markdown(index=False)}

## Calibration and derivative tradeoffs

Coverage targets90%, but width must be considered. Tail error uses true-law finite-bank95th-percentile thresholds, so finite-oracle quantile error remains. These are model-output probabilities, not a certification of epistemic uncertainty.

{calibration.to_markdown(index=False)}

Derivative summaries at step.1 (all steps remain in derivative_summary.csv) concern the conditional mean with respect to one history coordinate. They are not repaired-path effects or physical delays. The model-reference soft effects and these derivatives are distinct estimands.

{derivative[derivative.step==.1].to_markdown(index=False)}

## Numerical cost and solver sensitivity

Times are local CPU single-thread observations while another GPU experiment ran; they are implementation-specific, not hardware-independent speed ratios. Each sample-time entry generates256 three-dimensional draws at one history. GP training includes its9-candidate validation grid; ridge includes3 candidates; neural families use one fixed optimizer configuration and validation stopping. This is matched data, not matched training compute or exhaustive tuning.

{costs.reset_index().to_markdown(index=False)}

24-versus48-step mean and quantile changes:

{stability.to_markdown(index=False)}

Cross-family disagreement versus average error has descriptive Spearman correlation {corr:.4g} over {len(spread)} fully qualified query groups. Groups share laws, datasets and generators; no independent-row p-value is justified. This is not a trained or externally validated OOD detector.

## What is still missing

Full-cohort NeuroPAL Transformer/GP and other-family refits, deliberate support exclusions for every generator, and generator-specific repaired lag matrices remain missing. The original per-generator OOD tables/checkpoints were not recovered. This supplement answers whether these fitted generators differ on the specified known laws, not which is biologically correct. Kernel choice, capacity, hyperparameter tuning, sample size, output dimensionality, calibration and optimization can change rankings. Consult fit_diagnostics.csv for convergence flags and the source-pinned PROTOCOL.md for exact settings.
'''
    (R/'FINDINGS.md').write_text(text)

if __name__=='__main__':main()
