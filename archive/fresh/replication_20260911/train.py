from common import *
import argparse,gc
from conditional_neural_benchmark.runner import RunState,_split_windows,_split_indices,_neural_trial

def train(name,fold,seed):
    check_snapshot();check_space()
    done=receipt_path(name,fold,seed)
    if done.exists():checked_checkpoint(name,fold,seed);print('VALIDATED_TRAINING_RECEIPT',done,flush=True);return
    c,folds=cohort_for(name)
    cfg,lag,epochs,patience,eval_rows=config_for(name)
    root=R/'runs'/name/'training';root.mkdir(parents=True,exist_ok=True)
    scaler,tr,va,te=_split_windows(c,folds,fold,lag)
    ti,vi,ei=_split_indices(folds,fold)
    assert not(set(ti)&set(vi) or set(ti)&set(ei) or set(vi)&set(ei))
    spec={'spec_id':spec_id(),'cohort':name,'fold':fold,'seed':seed,'model':cfg.to_dict(),'history':lag,'max_epochs':epochs,'patience':patience,'eval_rows':eval_rows,'samples':32,'batch_size':256,'train_indices':ti.tolist(),'validation_indices':vi.tolist(),'test_indices':ei.tolist(),'neuron_names':list(c.neurons),'worm_ids':list(c.worm_ids),'folds':folds.tolist(),'source_snapshot_sha256':sha(R/'snapshot_manifest.json'),'worker_sha256':sha(Path(__file__)),'scaler':scaler.to_dict(),'stimulus_schema':c.stimulus_schema_dict(),'device':SET['device'],'python':sys.version,'torch':torch.__version__}
    start=root/'launch'/f'f{fold}_s{seed}.json'
    if start.exists() and json.loads(start.read_text())!=spec:raise RuntimeError('resume launch specification differs')
    atomic_json(start,spec)
    state=RunState(root,time.monotonic()+86400*30)
    state.records=[json.loads(p.read_text())['record'] for p in sorted((root/'receipts').glob('*.json'))]
    previous_count=len(state.records)
    # Completion receipts, not a CSV row, determine whether a trial can be reused.
    _neural_trial(state=state,phase='replication',config=cfg,cohort=c,lag=lag,fold=fold,seed=seed,train=tr,validation=va,test=te,scaler=scaler,device=SET['device'],max_epochs=epochs,patience=patience,eval_rows=eval_rows,n_samples=32,batch_size=256,trial_metadata={'stimulus_encoding':'binary_any_stimulus','cohort_mode':c.cohort_mode,'replication_spec_id':spec_id(),'external_references_consulted':False})
    if len(state.records)!=previous_count+1 or state.records[-1]['status']!='ok':raise RuntimeError(f'training failed: {state.records}')
    record=state.records[-1]
    ck=checkpoint_path(name,fold,seed)
    obj=torch.load(ck,map_location='cpu',weights_only=False)
    assert obj['model_config']==cfg.to_dict()
    assert np.array_equal(np.asarray(obj['scaler']['mean']),scaler.mean)
    assert np.array_equal(np.asarray(obj['scaler']['scale']),scaler.scale)
    trace=obj['fit_trace'];loss=np.asarray(trace['validation_loss'])
    assert np.isfinite(loss).all()
    needs_extension=record['stopped_epoch']>=epochs-1 and record['best_epoch']>=epochs-patience
    atomic_json(done,{'spec_id':spec_id(),'checkpoint_sha256':sha(ck),'launch_sha256':sha(start),'completed_utc':now(),'record':record,'convergence':{'best_epoch':record['best_epoch'],'stopped_epoch':record['stopped_epoch'],'epoch_cap_reached':record['stopped_epoch']>=epochs-1,'extension_required':needs_extension},'fit_trace':trace})
    print('TRAINING_COMPLETE',name,fold,seed,'extension_required',needs_extension,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cohort',choices=SET['cohorts']);ap.add_argument('fold',type=int);ap.add_argument('seed',type=int);a=ap.parse_args();train(a.cohort,a.fold,a.seed)
