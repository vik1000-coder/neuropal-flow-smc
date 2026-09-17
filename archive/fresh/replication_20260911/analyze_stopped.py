"""User-authorized analysis of the complete primary study; no further sampling."""
from common import *
import analyze as a
import fcntl, traceback

OUT = R/'analysis_stopped_20260916'
LOG = R/'logs/stopped_analysis_20260916.log'

def phase(label, name=None):
    status('stopped_study_analysis','running',worker_pid=os.getpid(),phase=label,
           cohort=name,log=str(LOG),output=str(OUT),further_sampling_authorized=False)

def main():
    lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=True);a.OUT=OUT
    for manifest in ['analysis_manifest.json','stopped_analysis_manifest.json']:
        seal=json.loads((R/manifest).read_text())
        for name,digest in seal['workers'].items():
            if sha(R/name)!=digest:raise RuntimeError(f'frozen worker changed: {name}')
    phase('freeze_completed_raw_data');a.freeze()
    inventory=[]
    for name in SET['cohorts']:
        for seed in SET['seeds']:
            for fold in SET['folds']:
                original=json.loads(receipt_path(name,fold,seed).read_text())
                ext=R/'runs'/name/'extended_training/receipts'/f'f{fold}_s{seed}.json'
                record={'cohort':name,'seed':seed,'fold':fold,'extension_required':original['convergence']['extension_required'],'extended_fit_complete':ext.exists()}
                if ext.exists():record['extended_record']=json.loads(ext.read_text())['record']
                inventory.append(record)
    atomic_json(OUT/'stop_scope.json',{'stopped_at_user_request':True,'amendment':'amendments/002_user_stop_20260916T152247Z/AMENDMENT.md','primary_study_complete':True,'extended_sensitivity_complete':False,'extended_sampling_archives':len(list((R/'runs/clean54/extended_primary').rglob('*.receipt.json'))),'training_inventory':inventory})
    for name in SET['cohorts']:
        (OUT/name).mkdir(exist_ok=True)
        for lane in ['primary','mc_repeat','direct','matched_time']:
            phase('aggregate_'+lane,name);a.aggregate(name,lane)
        phase('external_reference_comparison',name);a.external(name)
        phase('matrix_stability',name);a.stability(name)
        phase('numerical_diagnostics',name);a.diagnostics_summary(name)
        phase('predictive_evaluation',name);a.predictions(name)
    for name in SET['cohorts']:
        phase('exact_joint_lag_inference',name);a.inference(name)
        phase('lag_reference_permutations',name);a.lag_correspondence_null(name)
    atomic_json(OUT/'analysis_complete.json',{'status':'complete','scope':'complete primary study; extended sensitivity stopped by user and unresolved','spec_id':spec_id(),'completed_utc':now(),'analysis_source_sha256':sha(R/'analyze.py'),'entrypoint_sha256':sha(Path(__file__))})
    phase('report')
    import report_stopped
    report_stopped.run()
    status('stopped_study_analysis','awaiting_report_review',log=str(LOG),report=str(R/'REPORT_STOPPED.md'),output=str(OUT),further_sampling_authorized=False)

if __name__=='__main__':
    try:main()
    except Exception:
        status('stopped_study_analysis','failed',log=str(LOG),error=traceback.format_exc(),further_sampling_authorized=False)
        raise
