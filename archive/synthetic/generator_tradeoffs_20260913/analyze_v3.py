from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
    receipts=list((ROOT/'runs').glob('*/receipt.json'))
    if len(receipts)!=192:raise RuntimeError(f'Expected all192 completed tasks; found {len(receipts)}')
    for receipt in receipts:
        for name,digest in json.loads(receipt.read_text())['files'].items():
            if hashlib.sha256((receipt.parent/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed result')
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    keys=['law','data_seed','family','region']
    def load(name):
        frames=[]
        for r in receipts:
            p=r.parent/(name+'.csv')
            if p.stat().st_size>2:frames.append(pd.read_csv(p))
        return pd.concat(frames,ignore_index=True)
    pred=load('predictive');effect=load('effects');deriv=load('derivatives');sample=load('sampling');solver=load('solver')
    fit=[]
    for r in receipts:
        p=pd.read_csv(r.parent/'predictive.csv',nrows=1).iloc[0]
        f=json.loads((r.parent/'fit.json').read_text());fit.append({**p[['law','data_seed','family','init']].to_dict(),**{k:v for k,v in f.items() if k!='traces'}})
    fit=pd.DataFrame(fit);fit.to_csv(out/'fit_diagnostics.csv',index=False)
    for name,frame in [('predictive_raw',pred),('effects_raw',effect),('derivatives_raw',deriv),('sampling_raw',sample),('solver_raw',solver)]:frame.to_csv(out/f'{name}.csv',index=False)
    metrics=['energy','mean_squared_error','coverage90','width90','tail_brier','tail_probability_abs_error','sampling_seconds_per_history']
    ps=pred.groupby(keys)[metrics].mean().reset_index();ps.to_csv(out/'predictive_dataset_means.csv',index=False)
    summary=ps.groupby(['law','region','family'])[metrics].agg(['mean','min','max']);summary.to_csv(out/'predictive_summary.csv')
    flow=ps[ps.family=='flow'].drop(columns='family');paired=ps.merge(flow,on=['law','data_seed','region'],suffixes=('','_flow'))
    for m in metrics:paired[m+'_delta_vs_flow']=paired[m]-paired[m+'_flow']
    paired.to_csv(out/'paired_predictive_differences.csv',index=False)
    effect['absolute_error']=effect.error.abs();effect['qualified_absolute_error']=effect.absolute_error.where(effect.qualified)
    # Require both neural seeds qualified within each dataset for primary accuracy summaries.
    ek=keys+['event'];qualified=effect.groupby(ek).qualified.all().rename('all_initializations_qualified')
    es=effect.groupby(ek).agg(absolute_error=('absolute_error','mean'),qualified_fraction=('qualified','mean'),reference_seconds=('reference_seconds','mean')).join(qualified).reset_index()
    es['qualified_absolute_error']=es.absolute_error.where(es.all_initializations_qualified)
    es.to_csv(out/'effects_dataset_means.csv',index=False)
    # Paired comparisons require qualification for both families on the same dataset/query.
    ef=es[es.family=='flow'].drop(columns='family');ep=es.merge(ef,on=['law','data_seed','region','event'],suffixes=('','_flow'))
    ep['paired_qualified']=ep.all_initializations_qualified & ep.all_initializations_qualified_flow
    ep['absolute_error_delta_vs_flow']=(ep.absolute_error-ep.absolute_error_flow).where(ep.paired_qualified)
    ep.to_csv(out/'paired_effect_differences.csv',index=False)
    deriv['absolute_error']=deriv.error.abs();ds=deriv.groupby(keys+['step']).agg(absolute_error=('absolute_error','mean'),mc_se=('mc_se','mean')).reset_index();ds.to_csv(out/'derivative_dataset_means.csv',index=False)
    plt.rcParams.update({'font.size':9})
    figs=[]
    for label,frame,value,extra in [('predictive_energy',ps,'energy',None),('tail_probability_error',ps,'tail_probability_abs_error',None),('common_effect_error',es[es.event=='common'],'qualified_absolute_error',None),('rare_effect_error',es[es.event=='rare'],'qualified_absolute_error',None),('reference_qualification',es,'qualified_fraction',None),('mean_derivative_error',ds[ds.step==.1],'absolute_error',None)]:
        fig,axes=plt.subplots(1,3,figsize=(16,6));fig.subplots_adjust(bottom=.30,top=.78,wspace=.28)
        for ax,region in zip(axes,['central','boundary','extrapolated']):
            table=frame[frame.region==region].pivot_table(index='family',columns='law',values=value,aggfunc='mean',dropna=False).reindex(index=sorted(ps.family.unique()),columns=sorted(ps.law.unique()))
            if table.notna().any().any():
                table.plot.bar(ax=ax,legend=False)
                for i in range(len(table)):
                    if table.iloc[i].isna().all():ax.text(i,.02,'N/A',rotation=90,ha='center',transform=ax.get_xaxis_transform())
            else:
                ax.text(.5,.5,'No qualified estimates\nUnresolved; not zero error',ha='center',va='center',transform=ax.transAxes);ax.set_xticks([])
            ax.set_title(region);ax.set_ylabel(value.replace('_',' '));ax.tick_params(axis='x',rotation=60)
        handles,labels=next((ax.get_legend_handles_labels() for ax in axes if ax.get_legend_handles_labels()[0]),([],[]));fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4)
        fig.suptitle(label.replace('_',' ')+' — synthetic benchmark; missing qualified cells are not zero',y=.995)
        path=out/(label+'.png');fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig);figs.append(str(path))
    aggregate=ps.groupby(['family','region'])[metrics].mean().reset_index()
    er=es.groupby(['family','region','event']).agg(qualified_error=('qualified_absolute_error','mean'),qualified_datasets=('all_initializations_qualified','sum'),total_datasets=('all_initializations_qualified','size')).reset_index()
    unresolved=int((fit.convergence=='boundary_best_unresolved').sum())
    text=f'''# Completed synthetic generator comparison

All192 planned fit/evaluation tasks completed. This is a supplementary low-dimensional fixed-configuration experiment, not a reanalysis of missing NeuroPAL tables and not biological replication. {unresolved} neural fits remain boundary-best/nonconverged under the declared limit. Review fit_diagnostics.csv before interpreting architecture differences.

## Predictive comparison

Each entry averages histories, then training initializations within a dataset, then equally across three data seeds and four specified laws. Read law-specific and paired tables; an aggregate winner need not win each mechanism. Error intervals are seed ranges, not confidence intervals. Three data seeds/law do not justify universal superiority claims.

{aggregate[['family','region','energy','mean_squared_error','coverage90','tail_probability_abs_error','sampling_seconds_per_history']].to_markdown(index=False)}

## Generator error in identical soft-event queries

These are independent direct-reference comparisons, not different sampler competitions. Only cells meeting both oracle/model repeat-SE and ESS gates contribute to qualified error. Both initializations must qualify. Coverage differs; use paired_effect_differences.csv for shared qualified comparisons. Unresolved cells are not successes or zero errors.

{er.to_markdown(index=False)}

## Missing biological evidence

The original six-family per-generator OOD tables and original neural checkpoints were absent. This run does not recover their ranking. Full NeuroPAL Transformer/GP and all-family support-exclusion refits, long-history encoder comparisons, full repaired-path lag matrices across families, and broader training-budget/hyperparameter sensitivity remain unperformed. Model sampling entropy is not calibrated epistemic uncertainty. This GP assumes independent Gaussian outputs; its behavior does not generalize to correlated or non-Gaussian GPs.

## Auditing

All raw rows, paired differences, checkpoint receipts and qualified denominators are retained. Six figures require human/agent visual inspection before verified delivery. See PROTOCOL.md for the exact estimands and design, TRADEOFFS.md for prior evidence and architecture tradeoffs.
'''
    (ROOT/'RESULTS.md').write_text(text)
    (out/'validation.json').write_text(json.dumps({'complete_tasks':len(receipts),'figures':figs,'visual_review':'pending','number_audit':'pending'},indent=2))

if __name__=='__main__':main()
