from common import *
import pandas as pd,itertools,gzip
from scipy.stats import spearmanr
from statistical_analysis import orient_raw,normalize,binary,paired_source_bootstrap,holm,exact_max_t
from compatibility_neural_benchmark.prediction_atlas_runner import output_path

CHANNELS=['endpoint_mean','cumulative_mean','peak_mean','event_probability','endpoint_sd','endpoint_log_sd','endpoint_wasserstein1']
CONTEXTS=['state_average','baseline','onset_minus_baseline','active_minus_baseline']
OUT=R/'analysis'

def clean_json(value):
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,list):return [clean_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value

def freeze():
    entries={}
    for name in SET['cohorts']:
        cfg,*_=config_for(name)
        for lane in ['primary','mc_repeat','direct','matched_time']:
            n=4096 if lane=='direct' else 64;method='direct_importance' if lane=='direct' else 'progressive_bridge_smc'
            for seed in (SET['seeds'] if lane in ['primary','matched_time'] else [1701]):
                for lag in ([0,7] if lane=='matched_time' else SET['lags']):
                    for fold in SET['folds']:
                        p=output_path(R/'runs'/name/lane,method,cfg.model_id,lag,fold,seed,n)
                        receipt=json.loads(p.with_suffix('.receipt.json').read_text())
                        if receipt['spec_id']!=spec_id() or receipt['archive_sha256']!=sha(p):raise RuntimeError(f'raw archive receipt mismatch: {p}')
                        entries[str(p.relative_to(R))]=sha(p)
    for p in (R/'runs').glob('*/extended_primary/responses/**/*.npz'):
        r=json.loads(p.with_suffix('.receipt.json').read_text())
        if r['archive_sha256']!=sha(p):raise RuntimeError('extension freeze mismatch')
        entries[str(p.relative_to(R))]=sha(p)
    atomic_json(OUT/'frozen_sampling.json',{'spec_id':spec_id(),'completed_utc':now(),'archives':entries,'external_scores_opened_after_this_freeze':True})

def contexts(v):
    # v [recording,phase,event,horizon,target,source]. Equal events within phase.
    phase=np.mean(v,axis=2)
    return {'state_average':phase.mean(1),'baseline':phase[:,0],'onset_minus_baseline':phase[:,1]-phase[:,0],'active_minus_baseline':phase[:,2]-phase[:,0]}

def aggregate(name,lane):
    cfg,*_=config_for(name);meta=json.loads((R/'runs'/name/'cohort.json').read_text());d=len(meta['neurons']);w=len(meta['worm_ids'])
    seeds=SET['seeds'] if lane in ['primary','matched_time','extended_primary'] else [1701];lags=[0,7] if lane=='matched_time' else SET['lags'];h=[1] if lane=='matched_time' else SET['horizons']
    target=OUT/name/f'{lane}.npz';receipt=target.with_suffix('.json')
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['spec_id']!=spec_id() or r['analysis_sha256']!=sha(Path(__file__)) or r['sha256']!=sha(target):raise RuntimeError('aggregate receipt differs')
        return target
    n=4096 if lane=='direct' else 64;method='direct_importance' if lane=='direct' else 'progressive_bridge_smc'
    arrays={k+'__'+ctx:np.zeros((len(lags),w,len(h),d,d),np.float32) for k in CHANNELS for ctx in CONTEXTS}
    seedarrays={k+'__'+ctx:np.zeros((len(seeds),len(lags),w,len(h),d,d),np.float32) for k in ['endpoint_mean','endpoint_log_sd'] for ctx in CONTEXTS}
    strict={ctx:np.full((len(seeds),len(lags),w,len(h),d,d),np.nan,np.float32) for ctx in CONTEXTS}
    valid=np.zeros((len(seeds),len(lags),w,d),np.float32);genealogy=valid.copy();ess=valid.copy();gaps=valid.copy()
    for si,seed in enumerate(seeds):
        for li,lag in enumerate(lags):
            seen=set()
            for fold in SET['folds']:
                p=output_path(R/'runs'/name/lane,method,cfg.model_id,lag,fold,seed,n)
                if lane=='extended_primary' and not p.exists():
                    required=json.loads(receipt_path(name,fold,seed).read_text())['convergence']['extension_required']
                    if required:raise RuntimeError('required extension archive missing')
                    p=output_path(R/'runs'/name/'primary',method,cfg.model_id,lag,fold,seed,n)
                with np.load(p,allow_pickle=False) as z:
                    wi=np.array([meta['worm_ids'].index(str(s)) for s in z['worm_ids']]);assert not seen.intersection(wi);seen.update(wi)
                    gap=z['diagnostic_achieved_gap'];good=z['diagnostic_valid']>0
                    valid[si,li,wi]=good.mean((1,2));gaps[si,li,wi]=gap.mean((1,2))
                    ess[si,li,wi]=np.minimum(z['diagnostic_ess_low'],z['diagnostic_ess_high']).mean((1,2))
                    if 'diagnostic_distinct_ancestors_low' in z:
                        ancestor=np.minimum(z['diagnostic_distinct_ancestors_low'],z['diagnostic_distinct_ancestors_high'])/n
                        genealogy[si,li,wi]=(ancestor>=.2).mean((1,2))
                    else:genealogy[si,li,wi]=good.mean((1,2))
                    for channel in CHANNELS:
                        raw=z['response_'+channel];v=orient_raw(normalize(raw,gap));cv=contexts(v)
                        for ctx,value in cv.items():
                            arrays[channel+'__'+ctx][li,wi]+=value/len(seeds)
                            if channel in ['endpoint_mean','endpoint_log_sd']:seedarrays[channel+'__'+ctx][si,li,wi]=value
                        if channel=='endpoint_mean':
                            # Strict sensitivity excludes an episode when its source fails any gate.
                            sv=orient_raw(normalize(raw,np.where(good,gap,np.nan),strict=True))
                            for ctx,value in contexts(sv).items():strict[ctx][si,li,wi]=value
                print('AGGREGATE',name,lane,seed,lag,fold,flush=True)
            assert seen==set(range(w))
    support_valid=valid.mean((0,2));support_genealogy=genealogy.mean((0,2))
    arrays.update({'seed__'+k:v for k,v in seedarrays.items()})
    arrays.update({'strict_seed__'+k:v for k,v in strict.items()})
    arrays.update(valid_fraction=support_valid,genealogy_fraction=support_genealogy,recording_valid_fraction=valid,recording_genealogy_fraction=genealogy,recording_ess=ess,recording_gap=gaps,source_strong=np.all((support_valid>=.8)&(support_genealogy>=.8),axis=0),source_sensitivity=np.all((support_valid>=.5)&(support_genealogy>=.5),axis=0),neurons=np.array(meta['neurons']),worm_ids=np.array(meta['worm_ids']),lags=np.array(lags),horizons=np.array(h),seeds=np.array(seeds))
    safe_npz(target,**arrays);atomic_json(receipt,{'spec_id':spec_id(),'analysis_sha256':sha(Path(__file__)),'sha256':sha(target),'completed_utc':now()});return target

def align(a,names,wanted,fill=np.nan):
    result=np.full((len(wanted),len(wanted)),fill,dtype=float);lookup={n:i for i,n in enumerate(names)}
    present=[i for i,n in enumerate(wanted) if n in lookup];original=[lookup[wanted[i]] for i in present]
    result[np.ix_(present,present)]=a[np.ix_(original,original)];return result

def references(names):
    base=R/'inputs/published_sbtg/reference_data';off=~np.eye(len(names),dtype=bool)
    with np.load(base/'functional_atlas/aligned_atlas_wild_type.npz') as z:
        n=z['neuron_order'].astype(str).tolist();q=align(z['q'],n,names);eq=align(z['q_eq'],n,names)
    positive=q<.05;negative=(eq<.05)&~positive
    refs={'randi_wild_type':((positive).astype(int),(positive|negative)&off)}
    nodes=json.loads((base/'connectome/nodes.json').read_text())
    for kind in ['struct','chem','gap']:
        a=align(np.load(base/f'connectome/A_{kind}.npy'),nodes,names)
        refs['cook_'+kind]=((a>0).astype(int),off&np.isfinite(a))
    with np.load(R/'inputs/published_sbtg/results/paper/sbtg_lag_matrices.npz') as z:
        original=z['neuron_names'].astype(str).tolist();published={int(l):align(z[f'mu_hat_lag{l}'],original,names) for l in z['lags']}
    return refs,published

def external(name):
    primary=OUT/name/'primary.npz';rows=[];contrasts=[];lagrows=[]
    with np.load(primary) as z:
        names=z['neurons'].astype(str).tolist();refs,published=references(names);off=~np.eye(len(names),dtype=bool)
        tail=set(['ALN','DVA','DVB','DVC','LUA','PHA','PHB','PHC','PLN','PQR','PVC','PVN','PVP','PVQ','PVR','PVT','PVW']);is_tail=np.array([n in tail for n in names])
        masks={'all_common':off,'strong_fixed_support':off&z['source_strong'][None,:],'support50_fixed':off&z['source_sensitivity'][None,:]}
        if name=='historical80':masks.update(head_head=off&~is_tail[:,None]&~is_tail[None,:],tail_tail=off&is_tail[:,None]&is_tail[None,:],cross_origin=off&(is_tail[:,None]!=is_tail[None,:]))
        matrix=z['endpoint_mean__state_average'][0,:,0].mean(0)
        for scope,mask in masks.items():
            for ref,(labels,eligible) in refs.items():
                use=mask&eligible&np.isfinite(matrix)&np.isfinite(published[1])
                for method,a in [('flow_ensemble',matrix),('published_sbtg',published[1])]:rows.append({'cohort':name,'scope':scope,'reference':ref,'method':method,**binary(a,labels,use)})
                b=paired_source_bootstrap(matrix,published[1],labels,use,reps=SET['bootstrap_replicates'])
                contrasts.append({'cohort':name,'scope':scope,'reference':ref,**b})
        for si,seed in enumerate(SET['seeds']):
            a=z['seed__endpoint_mean__state_average'][si,0,:,0].mean(0)
            for ref,(labels,eligible) in refs.items():rows.append({'cohort':name,'scope':'all_common','reference':ref,'method':f'flow_seed_{seed}',**binary(a,labels,eligible&np.isfinite(published[1]))})
        for channel in CHANNELS:
            for ctx in CONTEXTS:
                for li,lag in enumerate(SET['lags']):
                    for hi,h in enumerate(SET['horizons']):
                        a=z[channel+'__'+ctx][li,:,hi].mean(0)
                        for scope in ['all_common','strong_fixed_support']:
                            for ref,(labels,eligible) in refs.items():lagrows.append({'cohort':name,'channel':channel,'context':ctx,'source_lag_frames':lag,'horizon_frames':h,'source_to_endpoint_frames':lag+h,'scope':scope,'reference':ref,'method':'flow',**binary(a,labels,masks[scope]&eligible&np.isfinite(published[1]))})
        for lag,a in published.items():
            for ref,(labels,eligible) in refs.items():lagrows.append({'cohort':name,'channel':'published_sbtg','context':'published_hybrid','source_lag_frames':lag,'horizon_frames':0,'source_to_endpoint_frames':lag,'scope':'all_common','reference':ref,'method':'published_sbtg',**binary(a,labels,eligible)})
    with np.load(OUT/name/'matched_time.npz') as z:
        for li,ell in enumerate([0,7]):
            a=z['endpoint_mean__state_average'][li,:,0].mean(0)
            for ref,(labels,eligible) in refs.items():
                b=paired_source_bootstrap(a,published[ell+1],labels,eligible,reps=SET['bootstrap_replicates'])
                contrasts.append({'cohort':name,'scope':f'endpoint_timing_matched_{ell+1}frames','reference':ref,**b})
    contrast=pd.DataFrame(contrasts);contrast['holm_bootstrap_tail_p']=np.nan
    for scope,frame in contrast.groupby('scope'):
        if 'bootstrap_two_sided_tail_p' in frame and frame.bootstrap_two_sided_tail_p.notna().all():contrast.loc[frame.index,'holm_bootstrap_tail_p']=holm(frame.bootstrap_two_sided_tail_p.to_numpy())
    pd.DataFrame(rows).to_csv(OUT/name/'reference_metrics.csv',index=False);contrast.to_csv(OUT/name/'reference_deltas.csv',index=False);pd.DataFrame(lagrows).to_csv(OUT/name/'reference_lag_profiles.csv',index=False)


def stability(name):
    rows=[]
    with np.load(OUT/name/'primary.npz') as a,np.load(OUT/name/'mc_repeat.npz') as b,np.load(OUT/name/'direct.npz') as d:
        off=~np.eye(len(a['neurons']),dtype=bool);mask=off&a['source_strong'][None,:]
        for ch in ['endpoint_mean','endpoint_log_sd']:
            for ctx in CONTEXTS:
                seed=a['seed__'+ch+'__'+ctx]
                for li,lag in enumerate(SET['lags']):
                    for hi,h in enumerate(SET['horizons']):
                        pairs=[(f'generator_{i}_vs_{j}',seed[i,li,:,hi].mean(0),seed[j,li,:,hi].mean(0)) for i,j in itertools.combinations(range(3),2)]
                        pairs.extend([('independent_MC',seed[0,li,:,hi].mean(0),b[ch+'__'+ctx][li,:,hi].mean(0)),('direct_N4096',seed[0,li,:,hi].mean(0),d[ch+'__'+ctx][li,:,hi].mean(0))])
                        for label,x,y in pairs:
                            for scope,m in [('all',off),('strong',mask)]:
                                valid=m&np.isfinite(x)&np.isfinite(y);xx=x[valid];yy=y[valid]
                                rows.append({'cohort':name,'channel':ch,'context':ctx,'lag':lag,'horizon':h,'comparison':label,'scope':scope,'n_edges':len(xx),'spearman':spearmanr(xx,yy).statistic if len(xx)>2 else np.nan,'rmse':np.sqrt(np.mean((xx-yy)**2)) if len(xx) else np.nan,'sign_agreement':np.mean(np.sign(xx)==np.sign(yy)) if len(xx) else np.nan})
    pd.DataFrame(rows).to_csv(OUT/name/'stability.csv',index=False)


def inference(name):
    # Stream a common exact null across all four prespecified effect families and their lag contrasts.
    path=OUT/name/'lag_inference.csv';nullpath=OUT/name/'lag_joint_null.npz'
    if path.exists() and nullpath.exists():return
    blocks=[];metadata=[]
    with np.load(OUT/name/'primary.npz') as z:
        names=z['neurons'].astype(str);w=len(z['worm_ids']);source=np.flatnonzero(z['source_strong']);targets,sources=np.where((~np.eye(len(names),dtype=bool))&z['source_strong'][None,:])
        for channel in ['endpoint_mean','endpoint_log_sd']:
            for ctx in ['baseline','onset_minus_baseline']:
                v=z[channel+'__'+ctx] # lag,recording,h,target,source
                for typ in ['effect','lag_minus_1']:
                    for li,lag in enumerate(SET['lags']):
                        if typ=='lag_minus_1' and li==0:continue
                        for hi,h in enumerate(SET['horizons']):
                            x=v[li,:,hi][:,targets,sources]
                            if typ=='lag_minus_1':x=x-v[0,:,hi][:,targets,sources]
                            for start in range(0,len(targets),4096):
                                block=x[:,start:start+4096]
                                if not block.size:continue
                                blocks.append(block)
                                metadata.append([{'cohort':name,'channel':channel,'context':ctx,'test':typ,'lag':lag,'horizon':h,'source':str(names[s]),'target':str(names[t]),'estimate':float(block[:,j].mean())} for j,(t,s) in enumerate(zip(targets[start:start+4096],sources[start:start+4096]))])
    if not blocks:
        pd.DataFrame(columns=['cohort','channel','context','test','lag','horizon','source','target','estimate','t','joint_maxT_p']).to_csv(path,index=False);safe_npz(nullpath,null=np.zeros(1),n_recordings=np.array(w));return
    obs,p,null=exact_max_t(blocks,w,progress=lambda i,n:print('MAX_T_BLOCK',name,i+1,n,flush=True))
    output=[]
    for rows,t,pp in zip(metadata,obs,p):
        for row,tt,pv in zip(rows,t,pp):output.append({**row,'t':tt,'joint_maxT_p':pv})
    pd.DataFrame(output).to_csv(path,index=False);safe_npz(nullpath,null=null,n_recordings=np.array(w))


def diagnostics_summary(name):
    index=[]
    cell_path=OUT/name/'diagnostic_cells.csv.gz'
    if cell_path.exists():cell_path.unlink()
    first=True
    files=sorted((R/'runs'/name/'diagnostics').glob('*.npz'))
    if len(files)!=720:raise RuntimeError(f'diagnostic panel incomplete: {name}, {len(files)}/720 files')
    for p in files:
        rows=[]
        meta=json.loads(p.with_suffix('.json').read_text());assert sha(p)==meta['sha256'];m=meta['launch']
        with np.load(p) as z:
            delta=z['response_endpoint_mean'];gap=z['diagnostic_achieved_gap'];req=z['requested_high']-z['requested_low'];valid=z['diagnostic_valid']>0
            for s,src in enumerate(z['source_indices']):
                for hi,h in enumerate(z['horizons']):
                    for t,target in enumerate(z['neuron_names']):
                        if int(src)==t:continue
                        rows.append({**{k:m[k] for k in ['cohort','fold','phase','lag','method','particles','mc_seed','contrast_fraction','steps_factor','panel']},'source':str(z['neuron_names'][src]),'target':str(target),'horizon':int(h),'raw_contrast':float(delta[s,hi,t]),'achieved_gap':float(gap[s]),'requested_gap':float(req[s]),'requested_normalized_effect':float(delta[s,hi,t]/req[s]) if req[s]>0 else np.nan,'normalized_effect':float(delta[s,hi,t]/gap[s]) if gap[s]>0 else np.nan,'valid':bool(valid[s]),'ess':float(min(z['diagnostic_ess_low'][s],z['diagnostic_ess_high'][s])),'wall_seconds':meta['wall_seconds'],'model_evaluations':meta['compute']['velocity_network_sample_evaluations']})
        pd.DataFrame(rows).to_csv(cell_path,mode='w' if first else 'a',header=first,index=False,compression='gzip');first=False
        index.append({**m,'file':str(p.relative_to(R)),'wall_seconds':meta['wall_seconds'],**meta['compute']})
    # This table is potentially large; compress without discarding target-level evidence.
    frame=pd.read_csv(cell_path,dtype={k:'category' for k in ['cohort','phase','method','source','target','panel']});pd.DataFrame(index).to_csv(OUT/name/'diagnostic_runs.csv',index=False)
    keys=['fold','phase','lag','source','target','horizon']
    convergence=frame[frame.panel=='convergence']
    ref=convergence[(convergence.method=='direct')&(convergence.particles==16384)]
    refstats=ref.groupby(keys,observed=True).agg(reference_mean=('normalized_effect','mean'),reference_sd=('normalized_effect','std'),reference_valid_fraction=('valid','mean'),reference_ess=('ess','min')).reset_index()
    # Ref uncertainty is exposed rather than automatically promoting a large bank to truth.
    merged=convergence.merge(refstats,on=keys)
    self_ref=(merged.method=='direct')&(merged.particles==16384)
    merged.loc[self_ref,'reference_mean']=(3*merged.loc[self_ref,'reference_mean']-merged.loc[self_ref,'normalized_effect'])/2
    merged['absolute_difference']=abs(merged.normalized_effect-merged.reference_mean)
    merged['reference_resolved']=(merged.reference_valid_fraction==1)&(merged.reference_ess>=12)&(merged.reference_sd<=.05)
    summary=merged.groupby(['method','particles','lag','phase'],observed=True).agg(mean_absolute_difference=('absolute_difference','mean'),reference_resolved_fraction=('reference_resolved','mean'),valid_fraction=('valid','mean'),mean_ess=('ess','mean')).reset_index();qualified=merged[merged.reference_resolved].groupby(['method','particles','lag','phase'],observed=True).agg(resolved_reference_mae=('absolute_difference','mean'),resolved_reference_cells=('absolute_difference','size')).reset_index()
    summary=summary.merge(qualified,on=['method','particles','lag','phase'],how='left');summary.to_csv(OUT/name/'particle_convergence.csv',index=False)
    zero=frame[frame.panel=='zero_query'].copy()
    zero['squared_raw_contrast']=zero.raw_contrast**2
    zero.groupby(['particles','lag','phase'],observed=True).agg(zero_query_raw_mse=('squared_raw_contrast','mean'),mean_achieved_gap=('achieved_gap','mean')).reset_index().to_csv(OUT/name/'zero_query_controls.csv',index=False)
    for panel in ['derivative','solver']:
        f=frame[frame.panel==panel]
        grouped=f.groupby(keys+['contrast_fraction','steps_factor'],observed=True).agg(effect_mean=('normalized_effect','mean'),effect_sd=('normalized_effect','std'),gap_mean=('achieved_gap','mean'),valid_fraction=('valid','mean')).reset_index()
        grouped.to_csv(OUT/name/f'{panel}_stability.csv',index=False)


def predictions(name):
    frames=[]
    for seed in SET['seeds']:
        for fold in SET['folds']:
            p=R/'runs'/name/'predictive'/f'f{fold}_s{seed}.csv';receipt=json.loads(p.with_suffix('.json').read_text());assert sha(p)==receipt['csv_sha256'];frames.append(pd.read_csv(p))
    for fold in SET['folds']:
        with np.load(R/'runs'/name/'predictive'/f'f{fold}_s1701.bank.npz') as first:
            expected={k:first[k] for k in first.files}
        for seed in SET['seeds'][1:]:
            with np.load(R/'runs'/name/'predictive'/f'f{fold}_s{seed}.bank.npz') as bank:
                if any(not np.array_equal(bank[k],v) for k,v in expected.items()):raise RuntimeError('prediction banks differ across model seeds')
    df=pd.concat(frames,ignore_index=True);metrics=['energy','variogram','rmse','mae','coverage90','sharpness90']
    # Equal strata, then equal recording, with seeds reduced within recording.
    per=df.groupby(['seed','method','worm','horizon','stratum'])[metrics].mean().groupby(['seed','method','worm','horizon']).mean().reset_index()
    per.to_csv(OUT/name/'predictive_per_recording.csv',index=False)
    per.groupby(['method','horizon'])[metrics].mean().reset_index().to_csv(OUT/name/'predictive_summary.csv',index=False)


def lag_correspondence_null(name):
    """Prespecified reference correspondence with a lag-maximum permutation null."""
    from scipy.stats import rankdata
    rows=[];reps=2000
    with np.load(OUT/name/'primary.npz') as z:
        names=z['neurons'].astype(str).tolist();refs,_=references(names)
        for channel in ['endpoint_mean','endpoint_wasserstein1']:
            for context in ['state_average','onset_minus_baseline']:
                matrices=np.abs(z[channel+'__'+context][:,:,0].mean(axis=1))
                for ref,(labels,mask) in refs.items():
                    use=mask&z['source_strong'][None,:];target,source=np.where(use);y=labels[use].astype(float);positive=y.sum();negative=len(y)-positive
                    base={'cohort':name,'channel':channel,'context':context,'reference':ref,'scope':'strong_fixed_support','permutations':reps,'n_edges':len(y)}
                    if positive==0 or negative==0:
                        rows.append({**base,'status':'insufficient_supported_positive_and_negative_edges','permutation_p':1.});continue
                    ranks=np.stack([rankdata(m[use],method='average') for m in matrices])
                    observed=(ranks@y-positive*(positive+1)/2)/(positive*negative)
                    rng=np.random.default_rng(9112026);groups=[np.flatnonzero(source==s) for s in np.unique(source)]
                    null=[]
                    for begin in range(0,reps,100):
                        yy=np.tile(y,(min(100,reps-begin),1))
                        for group in groups:
                            for row in yy:row[group]=row[group][rng.permutation(len(group))]
                        scores=(yy@ranks.T-positive*(positive+1)/2)/(positive*negative);null.extend(scores.max(1))
                    p=(1+np.sum(np.array(null)>=observed.max()-1e-12))/(reps+1)
                    rows.append({**base,'status':'evaluated','best_nominal_lag':SET['lags'][int(observed.argmax())],'max_auroc':float(observed.max()),'permutation_p':float(p),**{f'auroc_lag{l}':float(v) for l,v in zip(SET['lags'],observed)}})
    df=pd.DataFrame(rows);df['holm_family_p']=holm(df.permutation_p.to_numpy());df.to_csv(OUT/name/'lagmax_reference_permutations.csv',index=False)


def main():
    OUT.mkdir(exist_ok=True)
    seal=json.loads((R/'analysis_manifest.json').read_text())
    for name,digest in seal['workers'].items():
        if sha(R/name)!=digest:raise RuntimeError(f'analysis worker changed after freeze: {name}')
    from extensions import run as ensure_extensions
    ensure_extensions();freeze()
    for name in SET['cohorts']:
        (OUT/name).mkdir(exist_ok=True)
        for lane in ['primary','mc_repeat','direct','matched_time']:aggregate(name,lane)
        external(name);stability(name);diagnostics_summary(name);predictions(name);inference(name);lag_correspondence_null(name)
        if (R/'runs'/name/'extended_primary').exists():
            p=aggregate(name,'extended_primary');rows=[]
            with np.load(p) as z:
                refs,published=references(z['neurons'].astype(str).tolist());a=z['endpoint_mean__state_average'][0,:,0].mean(0)
                for ref,(labels,mask) in refs.items():rows.append({'cohort':name,'reference':ref,**binary(a,labels,mask),**paired_source_bootstrap(a,published[1],labels,mask,reps=SET['bootstrap_replicates'])})
            pd.DataFrame(rows).to_csv(OUT/name/'extended_reference_sensitivity.csv',index=False)
    atomic_json(OUT/'analysis_complete.json',{'status':'complete','spec_id':spec_id(),'completed_utc':now(),'analysis_source_sha256':sha(Path(__file__))})
if __name__=='__main__':main()
