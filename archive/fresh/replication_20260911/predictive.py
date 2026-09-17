"""Equal-recording validation on a common, deterministic factual-history bank."""
from common import *
import argparse,pandas as pd
from conditional_neural_benchmark.runner import _split_windows
from conditional_neural_benchmark.data import multiscale_features
from conditional_neural_benchmark.baselines import GaussianBaseline
from conditional_neural_benchmark.metrics import metric_rows
from compatibility_neural_benchmark.core import GeneratorAdapter,standardize_for_checkpoint,stimulus_for_trace

BANK_PER_RECORDING_STRATUM=16
RIDGE_ALPHA=10.0

def run(name,fold,seed):
    check_snapshot();check_space();ck=checked_checkpoint(name,fold,seed)
    path=R/'runs'/name/'predictive'/f'f{fold}_s{seed}.csv';receipt=path.with_suffix('.json')
    spec={'spec_id':spec_id(),'worker_sha256':sha(Path(__file__)),'checkpoint_sha256':sha(ck),'bank_per_recording_stratum':BANK_PER_RECORDING_STRATUM,'ridge_alpha':RIDGE_ALPHA,'n_samples':32,'cohort':name,'fold':fold,'seed':seed}
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['spec']!=spec or r['csv_sha256']!=sha(path):raise RuntimeError('predictive receipt mismatch')
        return
    c,folds=cohort_for(name);cfg,lag,*_=config_for(name);scaler,tr,va,te=_split_windows(c,folds,fold,lag)
    adapter=GeneratorAdapter.load(str(ck),device=SET['device']);d=c.n_neurons
    baselines={'persistence_gaussian':GaussianBaseline.persistence(tr.history,tr.target),'ridge_gaussian':GaussianBaseline.ridge(multiscale_features(tr.history,d),tr.target,alpha=RIDGE_ALPHA)}
    sampled=[]
    for worm in np.unique(te.worm):
        for stratum in sorted(set(te.stratum[te.worm==worm])):
            candidates=np.flatnonzero((te.worm==worm)&(te.stratum==stratum))
            candidates=np.array([i for i in candidates if te.time[i]+31<len(c.traces[int(worm)]) and np.isfinite(c.traces[int(worm)][te.time[i]:te.time[i]+32]).all()],dtype=int)
            if len(candidates):sampled.extend(candidates[np.unique(np.linspace(0,len(candidates)-1,min(BANK_PER_RECORDING_STRATUM,len(candidates)),dtype=int))])
    ix=np.array(sorted(sampled),dtype=int);assert len(ix)>0
    arrays={'history':te.history[ix],'worm':te.worm[ix],'time':te.time[ix],'stratum':te.stratum[ix]}
    path.parent.mkdir(parents=True,exist_ok=True)
    bank=path.with_suffix('.bank.npz');safe_npz(bank,**{k:v for k,v in arrays.items() if k!='history'})
    traces={int(w):standardize_for_checkpoint(c.traces[int(w)],adapter.checkpoint) for w in np.unique(arrays['worm'])}
    stimuli={int(w):stimulus_for_trace(c.traces[int(w)],c,int(w),adapter.checkpoint) for w in np.unique(arrays['worm'])}
    rows=[];n=32;started=time.monotonic()
    for model_name in ['flow',*baselines]:
        for start in range(0,len(ix),32):
            end=min(start+32,len(ix));b=end-start
            hist=np.repeat(arrays['history'][start:end],n,axis=0)
            worm=arrays['worm'][start:end];times=arrays['time'][start:end];strata=arrays['stratum'][start:end]
            # Same draw seed and conditioning bank for all generator seeds.
            for step in range(1,33):
                rngseed=911001+fold*100000+start*100+step
                if model_name=='flow':
                    with torch.no_grad():
                        y=adapter.sample_standardized_next(torch.as_tensor(hist[:,:,:d],device=adapter.device),torch.as_tensor(hist[:,:,d:],device=adapter.device),seed=rngseed).detach().cpu().numpy()
                else:
                    model=baselines[model_name]
                    y=model.sample(1,rngseed,history=hist,features=multiscale_features(hist,d))[:,0]
                if step in SET['horizons']:
                    draws=y.reshape(b,n,d)
                    target=np.array([traces[int(w)][int(t)+step-1] for w,t in zip(worm,times)])
                    if not np.isfinite(draws).all():raise RuntimeError('nonfinite free rollout')
                    metrics=metric_rows(draws,target,91100+step)
                    for j in range(b):
                        rows.append({'cohort':name,'fold':fold,'seed':seed,'method':model_name,'worm':int(worm[j]),'worm_id':c.worm_ids[int(worm[j])],'time':int(times[j]),'stratum':str(strata[j]),'horizon':step,**{k:float(v[j]) for k,v in metrics.items()}})
                next_stim=np.repeat(np.array([stimuli[int(w)][int(t)+step-1] for w,t in zip(worm,times)]),n,axis=0)
                if next_stim.ndim==1:next_stim=next_stim[:,None]
                nxt=np.concatenate([y,next_stim],axis=1)
                hist=np.concatenate([hist[:,1:],nxt[:,None]],axis=1)
        print('PREDICTION_MODEL_DONE',name,fold,seed,model_name,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(path,index=False)
    atomic_json(receipt,{'spec':spec,'csv_sha256':sha(path),'bank_sha256':sha(bank),'completed_utc':now(),'wall_seconds':time.monotonic()-started,'rows':len(frame)})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=SET['cohorts']);p.add_argument('fold',type=int);p.add_argument('seed',type=int);a=p.parse_args();run(a.cohort,a.fold,a.seed)
