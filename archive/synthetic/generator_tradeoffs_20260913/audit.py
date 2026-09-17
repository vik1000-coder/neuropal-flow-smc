"""Independent algebra/count audit after the completed benchmark; no refitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent

def main():
    receipts=sorted((R/'runs').glob('*/receipt.json'));assert len(receipts)==192
    baseline_data={};counts={};checks=[]
    for receipt in receipts:
        run=receipt.parent;rec=json.loads(receipt.read_text())
        for name,digest in rec['files'].items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest
        pred=pd.read_csv(run/'predictive.csv');effects=pd.read_csv(run/'effects.csv');der=pd.read_csv(run/'derivatives.csv');small=pd.read_csv(run/'sampling.csv')
        assert len(pred)==72 and len(effects)==6 and len(der)==27 and len(small)==36
        first=pred.iloc[0];key=(first.law,int(first.data_seed))
        with np.load(run/'data.npz') as d:
            assert d['h'].shape==(1024,3) and d['vh'].shape==(256,3)
            digest=hashlib.sha256(b''.join(d[k].tobytes() for k in ['h','y','vh','vy'])).hexdigest()
        if key in baseline_data:assert digest==baseline_data[key]
        else:baseline_data[key]=digest
        assert np.isfinite(pred.select_dtypes('number')).all().all()
        assert pred.coverage90.between(0,1).all() and (pred.width90>0).all()
        joined=small.merge(effects[['region','event','error','model_estimate','oracle_estimate']],on=['region','event'])
        np.testing.assert_allclose(joined.total_error,joined.particle_error+joined.error,atol=1e-12)
        np.testing.assert_allclose(effects.error,effects.model_estimate-effects.oracle_estimate,atol=1e-12)
        assert np.array_equal(effects.qualified,effects.model_qualified & effects.oracle_qualified)
        counts[run.name]={'predictive':len(pred),'effect':len(effects),'derivative':len(der),'small_bank':len(small)}
    # Validate independent Gaussian oracle conditional-mean formula for every stored linear reference.
    covariance=np.array([[.25,0,0],[.18,.22,0],[.05,-.08,.2]])
    covariance=covariance@covariance.T
    for path in (R/'oracle_references').glob('linear_gaussian_*.json'):
        d=json.loads(path.read_text());r=d['final'];lo,hi=d['centers'];expected=covariance[1,0]/(covariance[0,0]+d['bandwidth']**2)*(hi-lo)
        tolerance=max(.02,6*r['se'])
        if r['qualified']:assert abs(r['estimate']-expected)<tolerance,(str(path),r,expected)
        checks.append({'path':path.name,'estimate':r['estimate'],'analytic_truth':expected,'tolerance':tolerance,'qualified':r['qualified'],'absolute_discrepancy':abs(r['estimate']-expected),'interpretation':'analytic accuracy checked' if r['qualified'] else 'unresolved numerical reference; discrepancy retained'})
    out={'tasks':len(receipts),'datasets':len(baseline_data),'per_task_counts':counts,'gaussian_reference_checks':checks,'paired_data_identity':'passed','error_decomposition':'passed','finite_predictive_outputs':'passed','scope':'Algebra, counts, receipts and Gaussian oracle. Does not certify model accuracy or establish general superiority.'}
    (R/'analysis'/'independent_audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ['per_task_counts','gaussian_reference_checks']},indent=2))

if __name__=='__main__':main()
