from common import *
import subprocess
from conditional_neural_benchmark.runner import _split_indices

def run():
    check_snapshot();checks={}
    for name in SET['cohorts']:
        c,folds=cohort_for(name)
        folder=R/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'cohort.json',{'name':name,'neurons':list(c.neurons),'worm_ids':list(c.worm_ids),'folds':folds.tolist(),'fps':c.fps,'lineage_warning':c.lineage_warning,'stimulus_schema':c.stimulus_schema_dict(),'trace_shapes':[list(x.shape) for x in c.traces]})
        for fold in SET['folds']:
            a,b,d=map(set,_split_indices(folds,fold));assert not(a&b or a&d or b&d);assert a|b|d==set(range(c.n_worms))
        if name=='historical80':
            with np.load(R/'inputs/published_sbtg/reference_snapshot/prepared_data/full_traces_imputed/traces.npz',allow_pickle=False) as z:
                names=json.loads((R/'inputs/published_sbtg/reference_snapshot/prepared_data/full_traces_imputed/standardization.json').read_text())['node_order']
                assert set(names)==set(c.neurons)
                column_map=[names.index(n) for n in c.neurons]
                v=z['values'].copy();v[z['missing']]=np.nan;v=v[:,column_map];o=z['offsets']
                checks['published_trace_equivalence']=all(np.array_equal(x,v[o[i]:o[i+1]].astype(np.float32),equal_nan=True) for i,x in enumerate(c.traces))
                assert checks['published_trace_equivalence']
            names=json.loads((R/'inputs/published_sbtg/reference_snapshot/prepared_data/full_traces_imputed/standardization.json').read_text())['node_order']
            checks['published_column_alignment']={name:names.index(name) for name in c.neurons}
    oracle=json.loads((R/'validation/gaussian_oracle.json').read_text());assert oracle['status']=='pass' and oracle['spec_id']==spec_id()
    check_space();checks['folds_disjoint']=True;checks['source_and_inputs_verified']=True;checks['oracle_passed']=True
    files=['PROTOCOL.md','settings.json','common.py','train.py','sample.py','oracle.py','preflight.py']
    manifest={'spec_id':spec_id(),'created_utc':now(),'workers':{f:sha(R/f) for f in files},'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'mps_available':torch.backends.mps.is_available()},'checks':checks}
    p=R/'execution_manifest.json'
    if p.exists():
        old=json.loads(p.read_text());assert old['spec_id']==manifest['spec_id'] and old['workers']==manifest['workers']
    else:atomic_json(p,manifest)
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (R/'environment.freeze.txt').write_text(freeze)
    atomic_json(R/'validation/preflight.json',{'status':'pass','checks':checks,'spec_id':spec_id(),'completed_utc':now()})
    print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':run()
