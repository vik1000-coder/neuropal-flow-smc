from pathlib import Path
import sys,os,json,hashlib,time,shutil,platform
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source'))
import numpy as np
import torch
from conditional_neural_benchmark.data import load_sbtg_cohort,load_cohort,make_fold_assignments
from conditional_neural_benchmark.runner import ModelConfig
SET=json.loads((R/'settings.json').read_text())

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n');os.replace(tmp,path)
def now():return datetime.now(timezone.utc).isoformat()
def spec_id():
    return hashlib.sha256((sha(R/'settings.json')+sha(R/'PROTOCOL.md')+sha(R/'snapshot_manifest.json')).encode()).hexdigest()
def check_snapshot():
    errors=[p for p,h in json.loads((R/'snapshot_manifest.json').read_text()).items() if not (R/p).is_file() or sha(R/p)!=h]
    if errors:raise RuntimeError(f'snapshot integrity failed: {errors[:10]}')
def check_space():
    free=shutil.disk_usage(R).free/1e9
    if free<SET['min_free_disk_gb']:raise RuntimeError(f'free disk {free:.2f} GB below preserved reserve')
def cohort_for(name):
    c=load_sbtg_cohort() if name=='historical80' else load_cohort(cohort_mode='oh16230_head')
    expected=(20,80) if name=='historical80' else (17,54)
    if (c.n_worms,c.n_neurons)!=expected:raise RuntimeError('unexpected cohort axes')
    return c,make_fold_assignments(c,n_folds=5,seed=SET['fold_seed'])
def config_for(name):
    if name=='historical80':
        return ModelConfig('flow_lr6e4','tcn','conditional_flow_matching',True,width=128,dropout=.10,learning_rate=6e-4,weight_decay=1e-4,head_params={'hidden':128,'layers':4,'sample_steps':20}),8,120,20,5000
    return ModelConfig('stim_binary_any_stimulus_tcn_flow128_dropout15_jitter_p01','tcn','conditional_flow_matching',True,width=128,dropout=.15,learning_rate=3e-4,weight_decay=7.5e-4,head_params={'hidden':128,'layers':4,'sample_steps':24},history_noise_std=.01,history_noise_copies=1),80,50,9,4000

def checkpoint_path(name,fold,seed):
    config,lag,*_=config_for(name)
    return R/'runs'/name/'training/checkpoints/replication'/f'{config.model_id}__L{lag}__f{fold}__s{seed}.pt'
def receipt_path(name,fold,seed):return R/'runs'/name/'training/receipts'/f'f{fold}_s{seed}.json'
def checked_checkpoint(name,fold,seed):
    p=checkpoint_path(name,fold,seed); receipt=json.loads(receipt_path(name,fold,seed).read_text())
    if receipt['spec_id']!=spec_id() or receipt['checkpoint_sha256']!=sha(p):raise RuntimeError('checkpoint receipt mismatch')
    return p

def status(stage,state,**details):
    atomic_json(R/'status.json',{'stage':stage,'state':state,'updated_utc':now(),'pid':os.getpid(),**details})
    print(json.dumps({'time':now(),'stage':stage,'state':state,**details}),flush=True)

def safe_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,path)
