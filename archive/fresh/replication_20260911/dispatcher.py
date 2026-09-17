"""Durable serial dispatcher; one GPU worker at a time, no silent trial deletion."""
from common import *
import subprocess,fcntl,traceback

def tasks():
    # Model fitting is complete and byte-frozen before reference evaluation.
    for name in SET['cohorts']:
        for seed in SET['seeds']:
            for fold in SET['folds']:yield 'training',['train.py',name,fold,seed]
    for lane in ['primary','mc_repeat','direct','matched_time']:
        for name in SET['cohorts']:
            for seed in (SET['seeds'] if lane in ['primary','matched_time'] else [1701]):
                for lag in ([0,7] if lane=='matched_time' else SET['lags']):
                    for fold in SET['folds']:yield lane,['sample.py',name,fold,seed,lag,lane]
    for name in SET['cohorts']:
        for fold in SET['folds']:yield 'diagnostics',['diagnostics.py',name,fold]
    for name in SET['cohorts']:
        for seed in SET['seeds']:
            for fold in SET['folds']:yield 'predictive_validation',['predictive.py',name,fold,seed]
    yield 'analysis',['analyze.py']
    yield 'report',['report.py']

def run():
    lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_snapshot()
    manifest=json.loads((R/'execution_manifest.json').read_text())
    assert spec_id()==manifest['spec_id']
    for f,h in manifest['workers'].items():
        if sha(R/f)!=h:raise RuntimeError(f'frozen worker changed: {f}')
    assert json.loads((R/'validation/preflight.json').read_text())['status']=='pass'
    assert json.loads((R/'validation/source_tests.json').read_text())['status']=='pass'
    (R/'logs').mkdir(exist_ok=True)
    all_tasks=list(tasks());atomic_json(R/'task_plan.json',[{'stage':s,'args':list(map(str,a))} for s,a in all_tasks])
    for index,(stage,args) in enumerate(all_tasks):
        check_space();args=list(map(str,args));label='_'.join(args).replace('.py','')
        logfile=R/'logs'/f'{index:04d}_{label}.log'
        started=time.time()
        with logfile.open('a') as log:
            child=subprocess.Popen([sys.executable,str(R/args[0]),*args[1:]],cwd=R,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'})
            while child.poll() is None:
                atomic_json(R/'status.json',{'stage':stage,'state':'running','updated_utc':now(),'pid':os.getpid(),'worker_pid':child.pid,'task_index':index,'task_count':len(all_tasks),'task':args,'task_elapsed_seconds':round(time.time()-started),'log':str(logfile)})
                time.sleep(20)
        if child.returncode:
            status(stage,'failed',task=args,task_index=index,returncode=child.returncode,log=str(logfile));raise RuntimeError(f'worker failed: {args}')
        print('TASK_COMPLETE',index,stage,args,'seconds',round(time.time()-started),flush=True)
    completion=R/'validation/final.json'
    if not completion.exists() or json.loads(completion.read_text()).get('status')!='pass':raise RuntimeError('analysis delivery validation incomplete')
    status('complete','complete',report=str(R/'REPORT.md'),task_count=len(all_tasks))
if __name__=='__main__':
    try:run()
    except Exception:
        print(traceback.format_exc(),flush=True)
        current=json.loads((R/'status.json').read_text()) if (R/'status.json').exists() else {}
        current.update({'state':'failed','updated_utc':now(),'error':traceback.format_exc()});atomic_json(R/'status.json',current);raise
