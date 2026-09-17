"""Predeclared convergence sensitivity, triggered only by validation-loss traces."""
from common import *
from conditional_neural_benchmark.runner import RunState,_split_windows,_neural_trial
from compatibility_neural_benchmark.prediction_atlas_runner import run_one,output_path

def run():
    import pandas as pd
    decisions=[]
    for name in SET['cohorts']:
        c,folds=cohort_for(name);cfg,hist,epochs,patience,eval_rows=config_for(name)
        for seed in SET['seeds']:
            for fold in SET['folds']:
                parent=json.loads(receipt_path(name,fold,seed).read_text());required=bool(parent['convergence']['extension_required'])
                decisions.append({'cohort':name,'fold':fold,'seed':seed,'required':required})
                if not required:continue
                root=R/'runs'/name/'extended_training';ck=root/'checkpoints/replication'/f'{cfg.model_id}__L{hist}__f{fold}__s{seed}.pt';rec=root/'receipts'/f'f{fold}_s{seed}.json'
                spec={'spec_id':spec_id(),'worker_sha256':sha(Path(__file__)),'parent_sha256':parent['checkpoint_sha256'],'max_epochs':200,'patience':20,'reason':'primary best epoch near exhausted budget; validation-only trigger'}
                if rec.exists():
                    old=json.loads(rec.read_text())
                    if old['spec']!=spec or old['checkpoint_sha256']!=sha(ck):raise RuntimeError('extension receipt mismatch')
                else:
                    check_space();scaler,tr,va,te=_split_windows(c,folds,fold,hist);state=RunState(root,time.monotonic()+86400*30)
                    _neural_trial(state=state,phase='replication',config=cfg,cohort=c,lag=hist,fold=fold,seed=seed,train=tr,validation=va,test=te,scaler=scaler,device=SET['device'],max_epochs=200,patience=20,eval_rows=eval_rows,n_samples=32,batch_size=256,trial_metadata={'stimulus_encoding':'binary_any_stimulus','cohort_mode':c.cohort_mode,'replication_spec_id':spec_id(),'external_references_consulted':False,'sensitivity':'extended_training'})
                    if not state.records or state.records[-1]['status']!='ok':raise RuntimeError('extended training failed')
                    atomic_json(rec,{'spec':spec,'checkpoint_sha256':sha(ck),'record':state.records[-1],'completed_utc':now()})
                for ell in SET['lags']:
                    output=R/'runs'/name/'extended_primary'
                    result=run_one(cohort=c,folds=folds,source_run=root,checkpoint_phase='replication',output=output,method='progressive_bridge_smc',model_id=cfg.model_id,history_lag=hist,fold=fold,seed=seed,source_lag=ell,particles=64,horizons=tuple(SET['horizons']),source_window_frames=4,device=SET['device'],base_seed=SET['primary_mc_seed'],min_ess=12,progressive_branch_factor=4,progressive_future_branch_factor=2,checkpoint_override=ck)
                    p=output_path(output,'progressive_bridge_smc',cfg.model_id,ell,fold,seed,64)
                    atomic_json(p.with_suffix('.receipt.json'),{'spec_id':spec_id(),'archive_sha256':sha(p),'checkpoint_sha256':sha(ck),'extended_training':True,'completed_utc':now(),'result':result})
    pd.DataFrame(decisions).to_csv(R/'analysis/extension_decisions.csv',index=False)
