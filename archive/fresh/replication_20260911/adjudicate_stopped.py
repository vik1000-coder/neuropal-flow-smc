"""Descriptive adjudication from saved results only; see amendment 003."""
from common import *
import pandas as pd
from scipy.stats import spearmanr
from analyze import references
from statistical_analysis import binary

BASE=R/'analysis_stopped_20260916'
OUT=R/'final_review_20260916'
KEY=['fold','phase','lag','source','target','horizon']

def paired_panel(frame,panel):
    f=frame[frame.panel==panel].copy()
    factor='contrast_fraction' if panel=='derivative' else 'steps_factor'
    pairs=[(1.,.5),(.5,.25),(1.,.25)] if panel=='derivative' else [(1,2)]
    summaries=[];cell_tables=[]
    for normalization in ['normalized_effect','requested_normalized_effect']:
        for first,second in pairs:
            cols=KEY+['mc_seed',normalization,'valid','achieved_gap']
            a=f[f[factor]==first][cols]
            b=f[f[factor]==second][cols]
            pair=a.merge(b,on=KEY+['mc_seed'],suffixes=('_a','_b'),validate='one_to_one')
            pair['difference']=pair[normalization+'_b']-pair[normalization+'_a']
            pair['valid_pair']=pair.valid_a & pair.valid_b
            pair['finite_pair']=np.isfinite(pair[normalization+'_a']) & np.isfinite(pair[normalization+'_b'])
            pair['gap_min']=pair[['achieved_gap_a','achieved_gap_b']].min(axis=1)
            g=pair.groupby(KEY,observed=True).agg(
                n=('mc_seed','size'),finite_fraction=('finite_pair','mean'),
                valid_fraction=('valid_pair','mean'),gap_min=('gap_min','min'),
                mean_a=(normalization+'_a','mean'),mean_b=(normalization+'_b','mean'),
                sd_a=(normalization+'_a','std'),sd_b=(normalization+'_b','std'),
                difference=('difference','mean'),difference_sd=('difference','std')).reset_index()
            assert (g.n==3).all()
            g['paired_se']=g.difference_sd/np.sqrt(3)
            g['panel']=panel;g['normalization']=normalization;g['first']=first;g['second']=second
            cell_tables.append(g)
            for scope in ['valid_positive_gap','valid_gap_ge_0.1']:
                threshold=0 if scope=='valid_positive_gap' else .1
                use=(g.valid_fraction==1)&(g.finite_fraction==1)&(g.gap_min>0)
                if threshold:use &=g.gap_min>=threshold
                for horizon in ['all','1']:
                    total=g if horizon=='all' else g[g.horizon==1]
                    selected=g[use] if horizon=='all' else g[use&(g.horizon==1)]
                    if not len(selected):continue
                    se=selected.paired_se.to_numpy();delta=selected.difference.to_numpy()
                    ratio=np.divide(abs(delta),se,out=np.full(len(se),np.inf),where=se>0)
                    ratio[(se==0)&(delta==0)]=0
                    summaries.append(dict(panel=panel,normalization=normalization,first=first,second=second,
                        scope=scope,horizon=horizon,eligible_cells=len(selected),total_cells=len(total),
                        coverage=len(selected)/len(total),mean_abs_difference=float(abs(delta).mean()),
                        rmse=float(np.sqrt(np.mean(delta**2))),median_abs_difference=float(np.median(abs(delta))),
                        mean_abs_first=float(abs(selected.mean_a).mean()),mean_abs_second=float(abs(selected.mean_b).mean()),
                        median_mc_sd_first=float(selected.sd_a.median()),median_mc_sd_second=float(selected.sd_b.median()),
                        median_paired_se=float(np.median(se)),median_change_over_se=float(np.median(ratio)),
                        fraction_change_over_2se=float((abs(delta)>2*se).mean()),
                        spearman=float(spearmanr(selected.mean_a,selected.mean_b).statistic),
                        sign_agreement=float((np.sign(selected.mean_a)==np.sign(selected.mean_b)).mean())))
    return pd.DataFrame(summaries),pd.concat(cell_tables,ignore_index=True)

def run():
    OUT.mkdir(exist_ok=True)
    check_snapshot()
    for manifest in ['execution_manifest.json','analysis_manifest.json','stopped_analysis_manifest.json']:
        for f,h in json.loads((R/manifest).read_text())['workers'].items():assert sha(R/f)==h
    inventory={};primary_tables=[];strict_tables=[];lag_tables=[];convergence=[];costs=[];numerical=[]
    for name in SET['cohorts']:
        folder=OUT/name;folder.mkdir(exist_ok=True)
        chunks=[]
        for chunk in pd.read_csv(BASE/name/'diagnostic_cells.csv.gz',chunksize=200000):
            chunks.append(chunk[chunk.panel.isin(['derivative','solver'])])
        frame=pd.concat(chunks,ignore_index=True)
        for panel in ['derivative','solver']:
            summary,cells=paired_panel(frame,panel)
            summary.insert(0,'cohort',name);summary.to_csv(folder/f'{panel}_adjudication.csv',index=False)
            cells.to_csv(folder/f'{panel}_paired_cells.csv.gz',index=False)
            numerical.append(summary)
        with np.load(BASE/name/'primary.npz') as z:
            names=z['neurons'].astype(str).tolist();refs,published=references(names)
            primary=z['endpoint_mean__state_average'][0,:,0].mean(0)
            strict=z['strict_seed__state_average'][:,0,:,0].mean(axis=(0,1))
            finite=np.isfinite(strict);off=~np.eye(len(names),dtype=bool)
            inventory[name]={'strict_finite_offdiagonal_edges':int((finite&off).sum()),
                'strict_sources_with_any_finite_edge':int((finite&off).any(axis=0).sum()),
                'strong_sources':int(z['source_strong'].sum()),'total_sources':len(names)}
            stored=pd.read_csv(BASE/name/'reference_metrics.csv')
            for ref,(labels,mask) in refs.items():
                common=mask&np.isfinite(primary)&np.isfinite(published[1])
                for method,a in [('flow_ensemble',primary),('published_sbtg',published[1])]:
                    metric=binary(a,labels,common)
                    original=stored[(stored.scope=='all_common')&(stored.reference==ref)&(stored.method==method)].iloc[0]
                    for key in ['auroc','auprc','n_edges','n_positive','prevalence']:
                        assert np.isclose(metric[key],original[key],atol=1e-12,rtol=1e-10),(name,ref,method,key)
                    primary_tables.append({'cohort':name,'reference':ref,'method':method,**metric})
                use=common&finite
                for method,a in [('strict',strict),('primary_on_strict_mask',primary),('published_on_strict_mask',published[1])]:
                    strict_tables.append({'cohort':name,'reference':ref,'method':method,
                        'common_edges_before_strict':int(common.sum()),'status':'evaluated' if use.any() else 'no_complete_case_edges',**binary(a,labels,use)})
        sig=pd.read_csv(BASE/name/'lag_inference.csv')
        for keys,g in sig.groupby(['channel','context','test']):
            passed=g[g.joint_maxT_p<=.05]
            lag_tables.append(dict(cohort=name,channel=keys[0],context=keys[1],test=keys[2],
                tested_cells=len(g),significant_cells=len(passed),distinct_directed_pairs=len(passed[['source','target']].drop_duplicates()),
                minimum_adjusted_p=float(g.joint_maxT_p.min())))
        sig[(sig.test=='lag_minus_1')&(sig.joint_maxT_p<=.05)].sort_values('joint_maxT_p').to_csv(folder/'significant_lag_cells.csv',index=False)
        conv=pd.read_csv(BASE/name/'particle_convergence.csv')
        for (method,n),g in conv.groupby(['method','particles']):
            convergence.append(dict(cohort=name,method=method,particles=n,
                resolved_reference_fraction=float(g.reference_resolved_fraction.mean()),
                stratified_mae=float(g.resolved_reference_mae.mean()),
                pooled_mae=float(np.average(g.resolved_reference_mae,weights=g.resolved_reference_cells)),
                resolved_run_cells=int(g.resolved_reference_cells.sum())))
        runs=pd.read_csv(BASE/name/'diagnostic_runs.csv');runs=runs[runs.panel=='convergence']
        for (method,n),g in runs.groupby(['method','particles']):
            costs.append(dict(cohort=name,method=method,particles=n,runs=len(g),
                mean_wall_seconds=float(g.wall_seconds.mean()),median_wall_seconds=float(g.wall_seconds.median()),
                mean_velocity_sample_evaluations=float(g.velocity_network_sample_evaluations.mean()),
                mean_transition_samples=float(g.transition_samples.mean())))
        zero=pd.read_csv(BASE/name/'zero_query_controls.csv')
        assert (zero.zero_query_raw_mse==0).all()
        inventory[name]['identical_arm_mse_max']=float(zero.zero_query_raw_mse.max())
        print('ADJUDICATED',name,flush=True)
    for file,rows in [('primary_metric_audit',primary_tables),('strict_reference_sensitivity',strict_tables),
                      ('lag_counts',lag_tables),('particle_convergence',convergence),('computation',costs)]:
        pd.DataFrame(rows).to_csv(OUT/f'{file}.csv',index=False)
    pd.concat(numerical,ignore_index=True).to_csv(OUT/'numerical_adjudication.csv',index=False)
    atomic_json(OUT/'coverage_and_controls.json',inventory)
    atomic_json(OUT/'adjudication_complete.json',{'status':'complete','created_utc':now(),
        'script_sha256':sha(Path(__file__)),'no_new_training_or_sampling':True,
        'scope':'post-hoc descriptive adjudication of existing saved results; amendment 003'})

if __name__=='__main__':run()
