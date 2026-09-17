"""Verify the final supplement against saved evidence, preserving old receipts."""
from common import *
import pandas as pd
import re
from scipy.stats import spearmanr

B=R/'analysis_stopped_20260916';F=R/'final_review_20260916'

def run():
    check_snapshot()
    for name in SET['cohorts']:
        for fold in SET['folds']:
            for seed in SET['seeds']:checked_checkpoint(name,fold,seed)
    for m in ['execution_manifest.json','analysis_manifest.json','stopped_analysis_manifest.json']:
        for f,h in json.loads((R/m).read_text())['workers'].items():assert sha(R/f)==h
    original=json.loads((R/'validation/final_stopped.json').read_text())
    assert sha(R/'REPORT_STOPPED.md')==original['report_sha256']
    for f,h in original['analysis_files'].items():assert sha(R/f)==h,f
    frozen=json.loads((B/'frozen_sampling.json').read_text())
    for f,h in frozen['archives'].items():assert sha(R/f)==h,f
    metric=json.loads((B/'published_metric_reproduction_audit.json').read_text())
    assert metric['status']=='pass' and len(metric['comparisons'])==32
    assert all(x['counts_match'] and x['max_numeric_error']<1e-12 for x in metric['comparisons'])
    numerical=pd.read_csv(F/'numerical_adjudication.csv')
    checks=[]
    for name in SET['cohorts']:
        for panel in ['derivative','solver']:
            cells=pd.read_csv(F/name/f'{panel}_paired_cells.csv.gz')
            groups=['normalization','first','second']
            for key,g in cells.groupby(groups):
                for scope in ['valid_positive_gap','valid_gap_ge_0.1']:
                    good=(g.valid_fraction==1)&(g.finite_fraction==1)&(g.gap_min>0)
                    if scope.endswith('0.1'):good &=g.gap_min>=.1
                    for horizon in ['all','1']:
                        selected=g[good] if horizon=='all' else g[good&(g.horizon==1)]
                        saved=numerical[(numerical.cohort==name)&(numerical.panel==panel)&
                            (numerical.normalization==key[0])&(numerical['first']==key[1])&
                            (numerical['second']==key[2])&(numerical.scope==scope)&
                            (numerical.horizon.astype(str)==horizon)].iloc[0]
                        diff=selected.mean_b.to_numpy()-selected.mean_a.to_numpy()
                        assert len(selected)==saved.eligible_cells
                        assert np.isclose(np.mean(abs(diff)),saved.mean_abs_difference,atol=1e-12)
                        assert np.isclose(spearmanr(selected.mean_a,selected.mean_b).statistic,saved.spearman,atol=1e-12)
            # Independent recomputation of one paired cell directly from raw saved MC rows.
            sample=cells[(cells.normalization=='normalized_effect')&(cells.finite_fraction==1)].iloc[0]
            raw=[]
            keys=['fold','phase','lag','source','target','horizon']
            for chunk in pd.read_csv(B/name/'diagnostic_cells.csv.gz',chunksize=200000):
                use=chunk.panel==panel
                for key in keys:use &=chunk[key]==sample[key]
                raw.append(chunk[use])
            raw=pd.concat(raw)
            factor='contrast_fraction' if panel=='derivative' else 'steps_factor'
            a=raw[raw[factor]==sample['first']].set_index('mc_seed').sort_index()
            b=raw[raw[factor]==sample['second']].set_index('mc_seed').sort_index()
            aa=a.raw_contrast/a.achieved_gap;bb=b.raw_contrast/b.achieved_gap
            assert len(aa)==len(bb)==3
            assert np.isclose((bb-aa).mean(),sample.difference,atol=1e-12)
            assert np.isclose((bb-aa).std(ddof=1)/np.sqrt(3),sample.paired_se,atol=1e-12)
            checks.append({'cohort':name,'panel':panel,'raw_cell_checked':{key:sample[key] for key in keys}})
        lag=pd.read_csv(B/name/'lag_inference.csv')
        expected=756504 if name=='historical80' else 293832
        assert len(lag)==expected
        assert (lag.joint_maxT_p.between(0,1)).all()
        with np.load(B/name/'lag_joint_null.npz') as z:
            assert len(z['null'])==2**(int(z['n_recordings'])-1)
    report=(R/'REPORT_FINAL.md').read_text()
    for path in re.findall(r'\]\((/[^)]+)\)',report):assert Path(path).exists(),path
    figures=sorted(F.glob('*/figures/*.png'));assert len(figures)==14
    audit={'status':'pass','completed_utc':now(),'meaning':'artifact and numerical validation, not scientific success',
           'snapshot_files':670,'primary_checkpoints':30,'frozen_sampling_archives':len(frozen['archives']),
           'original_analysis_files_unchanged':len(original['analysis_files']),
           'numerical_summary_rows_checked':len(numerical),'raw_paired_cell_checks':checks,
           'published_metric_comparisons':32,'report_sha256':sha(R/'REPORT_FINAL.md'),
           'visual_review':{'status':'pass','reviewer':'Codex visual inspection','figure_count':14,
              'figures':[str(p.relative_to(R)) for p in figures],
              'notes':'All final PNGs viewed. Lag-profile subplot/suptitle collisions and horizontal-panel labels corrected in separate copies; source figures preserved.'},
           'no_new_training_or_sampling':True,'extended_training_sensitivity':'incomplete, excluded'}
    atomic_json(F/'validation.json',audit)
    outputs=[p for p in F.rglob('*') if p.is_file()]+[R/'REPORT_FINAL.md']
    sources=[R/p for p in ['adjudicate_stopped.py','review_figures_stopped.py','report_final_stopped.py','validate_final_stopped.py',
             'amendments/003_final_interpretation_20260916/AMENDMENT.md']]
    atomic_json(R/'final_review_manifest.json',{'created_utc':now(),'spec_id':spec_id(),
        'sources':{str(p.relative_to(R)):sha(p) for p in sources},
        'outputs':{str(p.relative_to(R)):sha(p) for p in outputs},
        'input_validation_sha256':sha(R/'validation/final_stopped.json'),
        'input_freeze_sha256':sha(B/'frozen_sampling.json')})
    print(json.dumps(audit,default=str,indent=2),flush=True)

if __name__=='__main__':run()
