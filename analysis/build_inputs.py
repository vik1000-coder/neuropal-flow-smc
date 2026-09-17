"""Derive compact publication inputs from frozen artifacts, without fitting/sampling.

Reference metrics are independently recomputed using explicit name alignment and
scikit-learn, rather than calling the original scoring helper. Neuron selections
are made only after retaining the complete corrected test family.
"""
from pathlib import Path
import json,hashlib,shutil,re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
R=Path(__file__).resolve().parents[1]
F=R/'archive/fresh/replication_20260911';S=R/'archive/synthetic/generator_tradeoffs_20260913'
OUT=R/'data/publication'
COHORTS=['clean54','historical80']
def align(a,old,new):
    lookup={s:i for i,s in enumerate(old)};out=np.full((len(new),len(new)),np.nan)
    for i,t in enumerate(new):
        for j,s in enumerate(new):
            if t in lookup and s in lookup:out[i,j]=a[lookup[t],lookup[s]]
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True);checks=[];cost=[];sources=[]
    training=pd.read_csv(F/'analysis_stopped_20260916/training_summary.csv')
    training.to_csv(OUT/'training.csv',index=False)
    for cohort in COHORTS:
        folder=F/'analysis_stopped_20260916'/cohort;dest=OUT/cohort;dest.mkdir(exist_ok=True)
        for n in ['reference_deltas','reference_metrics','reference_lag_profiles','stability','predictive_summary']:
            shutil.copyfile(folder/f'{n}.csv',dest/f'{n}.csv')
        inference=pd.read_csv(folder/'lag_inference.csv')
        inference.to_csv(dest/'all_corrected_tests.csv.gz',index=False)
        with np.load(folder/'primary.npz') as z:
            names=z['neurons'].astype(str);strong=z['source_strong'];lags=z['lags'];horizons=z['horizons']
            # Each seed profile is averaged over recordings; seed range is not a CI.
            save={k:z[k] for k in ['neurons','lags','horizons','source_strong','valid_fraction','genealogy_fraction']}
            for context in ['state_average','baseline','onset_minus_baseline']:
                for channel in ['endpoint_mean','endpoint_log_sd']:
                    save[channel+'__'+context]=z[channel+'__'+context].mean(1)
                    save['seed__'+channel+'__'+context]=z['seed__'+channel+'__'+context].mean(2)
            matrix=save['endpoint_mean__state_average'][0,0]
            np.savez_compressed(dest/'matrices.npz',**save)
            # Selection: fixed baseline mean, lag1/h1, corrected p<=.05, descending |effect|.
            # All tests were computed on a fixed strong-source family upstream.
            choose=inference[(inference.channel=='endpoint_mean')&(inference.context=='baseline')&(inference.test=='effect')&(inference.lag==1)&(inference.horizon==1)&(inference.joint_maxT_p<=.05)].copy()
            choose['abs_effect']=choose.estimate.abs();choose=choose.sort_values(['abs_effect','source','target'],ascending=[False,True,True]).head(6)
            choose.to_csv(dest/'selected_baseline_edges.csv',index=False)
            rows=[]
            for row in choose.itertuples():
                si=list(names).index(row.source);ti=list(names).index(row.target)
                values=z['endpoint_mean__baseline'][:,:,:,ti,si]
                for li,lag in enumerate(lags):
                    for wi,worm in enumerate(z['worm_ids']):
                        for hi,h in enumerate(horizons):rows.append(dict(source=row.source,target=row.target,lag=int(lag),horizon=int(h),recording=str(worm),effect=float(values[li,wi,hi])))
            pd.DataFrame(rows).to_csv(dest/'selected_edge_recordings.csv',index=False)
            strict=z['strict_seed__state_average'][:,0,:,0].mean((0,1))
            sources.append(dict(cohort=cohort,all_sources=len(names),strong_sources=int(strong.sum()),strict_complete_sources=int(np.isfinite(strict).any(0).sum())))
        # Independent external-score calculation: target rows, source columns.
        ref=F/'inputs/published_sbtg/reference_data';off=~np.eye(len(names),dtype=bool)
        with np.load(ref/'functional_atlas/aligned_atlas_wild_type.npz') as z:
            old=z['neuron_order'].astype(str);q=align(z['q'],old,names);qe=align(z['q_eq'],old,names)
        positive=q<.05;refs={'randi_wild_type':(positive,(positive|((qe<.05)&~positive))&off)}
        old=json.loads((ref/'connectome/nodes.json').read_text())
        for kind in ['struct','chem','gap']:
            a=align(np.load(ref/f'connectome/A_{kind}.npy'),old,names);refs['cook_'+kind]=(a>0,np.isfinite(a)&off)
        with np.load(F/'inputs/published_sbtg/results/paper/sbtg_lag_matrices.npz') as z:sb=align(z['mu_hat_lag1'],z['neuron_names'].astype(str),names)
        np.savez_compressed(dest/'reference_matrices.npz',published_sbtg=sb,**{k+'_labels':v[0] for k,v in refs.items()},**{k+'_mask':v[1] for k,v in refs.items()})
        saved=pd.read_csv(dest/'reference_metrics.csv')
        for ref,(labels,mask) in refs.items():
            mask=mask&np.isfinite(matrix)&np.isfinite(sb)
            for method,a in [('flow_ensemble',matrix),('published_sbtg',sb)]:
                y=labels[mask];score=np.abs(a[mask]);au=roc_auc_score(y,score);ap=average_precision_score(y,score)
                row=saved[(saved.scope=='all_common')&(saved.reference==ref)&(saved.method==method)].iloc[0]
                assert np.isclose(au,row.auroc,atol=1e-12) and np.isclose(ap,row.auprc,atol=1e-12)
                assert len(y)==row.n_edges and y.sum()==row.n_positive
                checks.append(dict(cohort=cohort,reference=ref,method=method,auroc=au,average_precision=ap,n_edges=len(y)))
        for lane in ['primary','mc_repeat','direct','matched_time']:
            records=[json.loads(p.read_text()) for p in (F/'runs'/cohort/lane).rglob('*.receipt.json')]
            seconds=sum(x['result']['wall_seconds'] for x in records)
            cost.append(dict(cohort=cohort,stage=lane,archives=len(records),worker_hours=seconds/3600))
        cost.append(dict(cohort=cohort,stage='training',archives=15,worker_hours=training[training.cohort==cohort].train_seconds.sum()/3600))
    pd.DataFrame(cost).to_csv(OUT/'worker_cost.csv',index=False);pd.DataFrame(sources).to_csv(OUT/'support.csv',index=False)
    for name in ['particle_convergence','computation','numerical_adjudication','lag_counts','strict_reference_sensitivity']:
        shutil.copyfile(F/'final_review_20260916'/f'{name}.csv',OUT/f'{name}.csv')
    synthetic=OUT/'synthetic';synthetic.mkdir(exist_ok=True)
    for name in ['predictive_dataset_means','analytic_effects_dataset_means','derivative_dataset_means','cost_summary','solver_summary']:
        shutil.copyfile(S/'analysis'/f'{name}.csv',synthetic/f'{name}.csv')
    # Historical screen: mechanically extract the surviving table, no invented raw uncertainty.
    tex=(R/'archive/original/reports/conditional_flow_model_report_20260901/main.tex').read_text()
    body=tex[tex.index('CFM, width 128, dropout 0.10 &'):];body=body[:body.index('\\bottomrule')]
    rows=[]
    for line in body.splitlines():
        cols=line.split('&')
        if len(cols)==6:rows.append(dict(candidate=cols[0].strip().replace('$',''),energy=float(cols[1]),balanced_energy=float(cols[2]),variogram=float(cols[3]),rmse=float(cols[4]),coverage90=float(cols[5].strip().rstrip('\\').strip()),evidence='historical report table; original raw runs unavailable',cohort='early pooled54'))
    assert len(rows)==16;pd.DataFrame(rows).to_csv(OUT/'original_generator_screen.csv',index=False)
    # Exact historical table transcription, with provenance and no fabricated intervals.
    historical=[('clean54','published SBTG',.622,.565,.558,.621),('clean54','original atlas',.647,.611,.600,.672),('historical80','published SBTG',.630,.581,.570,.613),('historical80','initial attempt',.523,.535,.534,.543),('historical80','optimized two-seed',.674,.629,.617,.666)]
    pd.DataFrame(historical,columns=['cohort','analysis','Randi','Cook structural','Cook chemical','Cook gap']).to_csv(OUT/'historical_reported_auroc.csv',index=False)
    (R/'audit/independent_metrics.json').write_text(json.dumps({'status':'pass','comparisons':checks,'scope':'independent name alignment, masks, AUROC/AP and edge counts; no retraining'},indent=2)+'\n')
    # Every new plot input is identified by bytes; archival provenance lives in its own manifest.
    manifest=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            manifest.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=digest))
    (R/'data/publication_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verified',len(checks),'primary comparisons; compact inputs ready.')
if __name__=='__main__':main()
