from common import *
import argparse
from compatibility_neural_benchmark.prediction_atlas_runner import run_one,output_path

def sample(name,fold,seed,lag,lane):
    check_snapshot();check_space();ck=checked_checkpoint(name,fold,seed)
    c,folds=cohort_for(name);cfg,hist,*_=config_for(name)
    direct=lane=='direct';n=4096 if direct else 64
    method='direct_importance' if direct else 'progressive_bridge_smc'
    base=SET['repeat_mc_seed'] if lane=='mc_repeat' else SET['primary_mc_seed']
    horizons=(1,) if lane=='matched_time' else tuple(SET['horizons'])
    output=R/'runs'/name/lane
    path=output_path(output,method,cfg.model_id,lag,fold,seed,n)
    launch={'spec_id':spec_id(),'checkpoint_sha256':sha(ck),'lane':lane,'base_seed':base,'particles':n,'source_lag':lag,'horizons':list(horizons),'branch_factor':4,'future_branch_factor':2,'worker_sha256':sha(Path(__file__))}
    specpath=path.with_suffix('.launch.json')
    if specpath.exists() and json.loads(specpath.read_text())!=launch:raise RuntimeError('sampling launch differs')
    atomic_json(specpath,launch)
    receipt=path.with_suffix('.receipt.json')
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['archive_sha256']!=sha(path) or saved['launch_sha256']!=sha(specpath):raise RuntimeError('sampler receipt differs')
        return
    result=run_one(cohort=c,folds=folds,source_run=R/'runs'/name/'training',checkpoint_phase='replication',output=output,method=method,model_id=cfg.model_id,history_lag=hist,fold=fold,seed=seed,source_lag=lag,particles=n,horizons=horizons,source_window_frames=4,device=SET['device'],base_seed=base,min_ess=12,progressive_branch_factor=4,progressive_future_branch_factor=2,checkpoint_override=ck)
    with np.load(path,allow_pickle=False) as z:
        assert str(z['checkpoint_sha256'])==sha(ck)
        assert np.isfinite(z['response_endpoint_mean']).all()
    atomic_json(receipt,{'spec_id':spec_id(),'archive_sha256':sha(path),'launch_sha256':sha(specpath),'completed_utc':now(),'result':result})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=SET['cohorts']);p.add_argument('fold',type=int);p.add_argument('seed',type=int);p.add_argument('lag',type=int);p.add_argument('lane',choices=['primary','mc_repeat','direct','matched_time']);a=p.parse_args();sample(a.cohort,a.fold,a.seed,a.lag,a.lane)
