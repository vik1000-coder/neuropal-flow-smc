from common import *
from conditional_neural_benchmark.data import FoldScaler
from conditional_neural_benchmark.runner import _split_indices
from compatibility_neural_benchmark.core import episode_cuts, GeneratorAdapter
from sid_elegans import combined_data as combined
import copy

torch.set_num_threads(1)
result={"review_utc":now(),"scope":"read-only existing-data and deterministic code-path checks; no training or path sampling"}
check_snapshot()
result['snapshot_files_verified']=len(json.loads((R/'snapshot_manifest.json').read_text()))
for manifest in ['execution_manifest.json','analysis_manifest.json','stopped_analysis_manifest.json']:
    m=json.loads((R/manifest).read_text())
    for f,h in m['workers'].items():assert sha(R/f)==h,(manifest,f)
result['frozen_worker_manifests_verified']=True
cohorts={}
for name in SET['cohorts']:
    c,folds=cohort_for(name);cfg,L,*_=config_for(name)
    leading=[];affected=[]
    for wi,x in enumerate(c.traces):
        first=np.array([np.flatnonzero(np.isfinite(x[:,j]))[0] for j in range(x.shape[1])])
        for j in np.flatnonzero(first>0):leading.append({'worm':c.worm_ids[wi],'neuron':c.neurons[j],'first_finite_frame':int(first[j])})
        for cut in episode_cuts(len(x),c.stimulus_schedules[wi],4):
            for ell in [0,1,4,7,8,16]:
                lo=cut.time-(ell+4)-L+1
                if (first>lo).any():affected.append({'worm':c.worm_ids[wi],'phase':cut.phase,'event':cut.event,'ell':ell,'earliest_required_frame':lo})
    cohorts[name]={'shape':[c.n_worms,c.n_neurons],'quality_dropped_neurons':list(c.quality_dropped_neurons),'fold_counts':np.bincount(folds).tolist(),'leading_missing':leading,'episodes_with_potential_leading_backfill':affected}
    for seed in SET['seeds']:
        for fold in SET['folds']:checked_checkpoint(name,fold,seed)
result['cohorts']=cohorts
# Reconstruct the same retained clean neuron classes in original MAT norm_traces units.
c,folds=cohort_for('clean54')
d=combined._prep.load_neuropal_data(combined.REPO/'data',include_tail=False,collapse_dv=True)
lookup={str(w):i for i,w in enumerate(d['worm_ids'])}
raw=[combined._worm_matrix(d,lookup[w.split(':',1)[1]],list(c.neurons),True).astype(np.float32) for w in c.worm_ids]
errors=[]
for fold in SET['folds']:
    tr,va,te=_split_indices(folds,fold)
    s0=FoldScaler.fit(raw[i] for i in tr);s1=FoldScaler.fit(c.traces[i] for i in tr)
    error=max(float(np.nanmax(abs(s0.transform(x)-s1.transform(y)))) for x,y in zip(raw,c.traces))
    errors.append(error)
result['clean_global_affine_cancellation_max_abs_error_by_fold']=errors
assert max(errors)<1e-5
# Test whether the legacy L80 TCN depends on older history through GroupNorm.
a=GeneratorAdapter.load(str(checkpoint_path('clean54',0,1701)),device='cpu')
enc=a.model.encoder;torch.manual_seed(91)
x=torch.randn(1,80*55,requires_grad=True);weights=torch.randn(128)
y=(enc(x)[0]*weights).sum();g=torch.autograd.grad(y,x)[0].reshape(80,55)
result['legacy_tcn_early49_frame_gradient_l1']=float(g[:49].abs().sum())
result['legacy_tcn_recent31_frame_gradient_l1']=float(g[49:].abs().sum())
without=copy.deepcopy(enc)
for block in without.blocks:block.norm=torch.nn.Identity()
z=x.detach().clone().requires_grad_(True);grad=torch.autograd.grad((without(z)[0]*weights).sum(),z)[0].reshape(80,55)
result['without_temporal_GroupNorm_early49_gradient_l1']=float(grad[:49].abs().sum())
atomic_json(R/'code_review_20260916/audit_checks.json',result)
print(json.dumps(result,indent=2))
