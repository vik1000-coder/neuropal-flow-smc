from common import *
import argparse
from dataclasses import replace
from compatibility_neural_benchmark.core import GeneratorAdapter,RepairedResponseConfig,causal_fill,standardize_for_checkpoint,stimulus_for_trace,episode_cuts,generate_path_bank,estimate_repaired_responses
from compatibility_neural_benchmark.prediction_atlas_runner import training_context
from compatibility_neural_benchmark.progressive_smc import progressive_smc_repaired_responses

def one(c,folds,name,fold,adapter,ck,cut,lag,kind,n,mc,contrast=1.,steps_factor=1,panel='convergence'):
    sources=np.linspace(0,c.n_neurons-1,8,dtype=int)
    worm=int(np.flatnonzero(folds==fold)[0]);hist=adapter.lag
    cfg=RepairedResponseConfig(history_frames=hist,repair_frames=lag+4,source_window_frames=4,source_lag_frames=lag,horizon_frames=tuple(SET['horizons']),n_particles=n,min_ess=12,sampling_chunk_size=1024)
    path=R/'runs'/name/'diagnostics'/f'f{fold}_{panel}_{cut.phase}_e{cut.event}_ell{lag}_{kind}_N{n}_mc{mc}_a{contrast}_steps{steps_factor}.npz'
    rec=path.with_suffix('.json')
    sampling_seed=mc+104729 if kind=='direct' and n==16384 else mc
    launch={'sampling_seed':sampling_seed,'spec_id':spec_id(),'worker_sha256':sha(Path(__file__)),'checkpoint_sha256':sha(ck),'cohort':name,'fold':fold,'generator_seed':1701,'worm':worm,'phase':cut.phase,'event':cut.event,'cut':cut.time,'lag':lag,'method':kind,'particles':n,'mc_seed':mc,'contrast_fraction':contrast,'steps_factor':steps_factor,'sources':sources.tolist(),'panel':panel}
    if rec.exists():
        old=json.loads(rec.read_text())
        if old['launch']!=launch or old['sha256']!=sha(path):raise RuntimeError('diagnostic receipt mismatch')
        return
    check_space()
    projection,q,threshold=training_context(c,folds,fold,adapter.checkpoint,cfg)
    quant=q[cut.phase];center=(quant['low']+quant['high'])/2;half=(quant['high']-quant['low'])*contrast/2
    low,high=center-half,center+half
    trace=causal_fill(standardize_for_checkpoint(c.traces[worm],adapter.checkpoint));stim=stimulus_for_trace(c.traces[worm],c,worm,adapter.checkpoint)
    original=adapter.sample_standardized_next;calls={'transition_samples':0,'calls':0}
    steps=int(adapter.model.head.sample_steps)
    def count(h,s,*,seed):
        calls['transition_samples']+=len(h);calls['calls']+=1
        return original(h,s,seed=seed)
    adapter.sample_standardized_next=count;adapter.model.head.sample_steps=steps*steps_factor
    started=time.monotonic()
    try:
        if kind=='progressive':
            z=progressive_smc_repaired_responses(adapter,trace,stim,cut_time=cut.time,projection=projection,source_low=low,source_high=high,source_iqr=quant['iqr'],thresholds=threshold,config=cfg,seed=sampling_seed,branch_factor=4,future_branch_factor=2,source_indices=sources)
        else:
            prefix,future,factual=generate_path_bank(adapter,trace,stim,cut_time=cut.time,config=cfg,seed=sampling_seed)
            full=estimate_repaired_responses(prefix,future,factual,projection,low,high,quant['iqr'],threshold,cfg)
            z={k:np.asarray(v)[sources] for k,v in full.items() if k.startswith(('response_','diagnostic_'))}
        result={k:np.asarray(v) for k,v in z.items() if k.startswith(('response_','diagnostic_'))}
        if any(not np.isfinite(v).all() for v in result.values()):raise RuntimeError('nonfinite diagnostic result')
        result.update(source_indices=sources,requested_low=low[sources],requested_high=high[sources],bandwidth=np.maximum(.25*quant['iqr'][sources],.1),horizons=np.array(cfg.horizon_frames),neuron_names=np.array(c.neurons))
        safe_npz(path,**result)
        elapsed=time.monotonic()-started
        atomic_json(rec,{'launch':launch,'sha256':sha(path),'wall_seconds':elapsed,'compute':{**calls,'velocity_network_sample_evaluations':calls['transition_samples']*2*steps*steps_factor},'completed_utc':now()})
        print('DIAGNOSTIC_COMPLETE',path.name,'seconds',round(elapsed),flush=True)
    finally:
        adapter.sample_standardized_next=original;adapter.model.head.sample_steps=steps
        if adapter.device.type=='mps':torch.mps.empty_cache()

def run(name,fold):
    check_snapshot();c,folds=cohort_for(name);ck=checked_checkpoint(name,fold,1701);adapter=GeneratorAdapter.load(str(ck),device=SET['device'])
    worm=int(np.flatnonzero(folds==fold)[0]);cuts=[x for x in episode_cuts(len(c.traces[worm]),c.stimulus_schedules[worm],4) if x.event==0 and x.phase in ['baseline','onset']]
    for cut in cuts:
        for lag in SET['lags']:
            for kind,n in [('progressive',64),('progressive',128),('progressive',512),('direct',4096),('direct',16384)]:
                for mc in SET['diagnostic_mc_seeds']:one(c,folds,name,fold,adapter,ck,cut,lag,kind,n,mc)
    if fold<2:
        for cut in cuts:
            for lag in [1,4]:
                for a in [1.,.5,.25]:
                    for mc in SET['diagnostic_mc_seeds']:one(c,folds,name,fold,adapter,ck,cut,lag,'progressive',512,mc,contrast=a,panel='derivative')
        for cut in cuts:
            if cut.phase!='baseline':continue
            for lag in [1,16]:
                for factor in [1,2]:
                    for mc in SET['diagnostic_mc_seeds']:one(c,folds,name,fold,adapter,ck,cut,lag,'progressive',128,mc,steps_factor=factor,panel='solver')
    if fold==0:
        for cut in cuts:
            for lag in [1,4]:
                for n in [64,512]:
                    for mc in SET['diagnostic_mc_seeds']:one(c,folds,name,fold,adapter,ck,cut,lag,'progressive',n,mc,contrast=0.,panel='zero_query')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=SET['cohorts']);p.add_argument('fold',type=int);a=p.parse_args();run(a.cohort,a.fold)
