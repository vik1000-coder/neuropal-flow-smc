from common import *
import itertools
from statistical_analysis import orient_raw,normalize,weighted_auc_samples,paired_source_bootstrap,exact_max_t,holm
from sklearn.metrics import roc_auc_score

def test_source_snapshot_imports():
    import conditional_neural_benchmark.models as m
    import history_tangent_benchmark.models as h
    assert str(R/'source') in m.__file__ and str(R/'source') in h.__file__

def test_orientation_asymmetric_source_sentinel():
    raw=np.zeros((2,5,3,4,6,4));raw[1,2,1,0,3,2]=7
    out=orient_raw(raw)
    assert out.shape==(2,5,3,6,4,4)
    assert out[1,2,1,3,2,0]==7 and out[1,2,1,3,0,2]==0

def test_normalization_does_not_hide_small_gap_in_strict_result():
    a=np.ones((1,2,1,2,3,2));g=np.array([[[[.02,-.5]],[[.2,.4]]]])
    ordinary=normalize(a,g);strict=normalize(a,g,True)
    assert ordinary[0,0,0,0,0,0]==10
    assert np.isnan(strict[0,0]).all()
    assert np.isclose(strict[0,1,0,0,0,0],5)

def test_weighted_bootstrap_matches_sklearn_with_ties():
    rng=np.random.default_rng(41);scores=rng.integers(0,4,40);y=rng.integers(0,2,40);src=np.tile(np.arange(5),8);counts=rng.multinomial(5,np.ones(5)/5,size=30)
    actual=weighted_auc_samples(scores,y,src,counts)
    expected=np.array([roc_auc_score(y,scores,sample_weight=c[src]) for c in counts])
    np.testing.assert_allclose(actual,expected,atol=1e-12)

def test_paired_bootstrap_identical_models_zero():
    rng=np.random.default_rng(42);a=rng.normal(size=(8,8));labels=rng.integers(0,2,(8,8));mask=~np.eye(8,dtype=bool)
    r=paired_source_bootstrap(a,a,labels,mask,reps=100)
    assert r['delta_auroc']==0 and r['delta_ci_low']==0 and r['delta_ci_high']==0

def test_exact_joint_sign_null_matches_brute_force():
    rng=np.random.default_rng(9);x=rng.normal(size=(6,9))+.3
    obs,p,null=exact_max_t([x[:,:4],x[:,4:]],6)
    exact=[]
    for tail in itertools.product([-1,1],repeat=5):
        y=x*np.array([1,*tail])[:,None]
        exact.append(np.max(abs(y.mean(0)/(y.std(0,ddof=1)/np.sqrt(6)))))
    np.testing.assert_allclose(np.sort(null),np.sort(exact),rtol=1e-10)
    observed=abs(x.mean(0)/(x.std(0,ddof=1)/np.sqrt(6)))
    expected=np.array([(np.array(exact)>=v-1e-12).mean() for v in observed])
    np.testing.assert_allclose(np.concatenate(p),expected)

def test_holm_is_monotone_and_bounded():
    p=np.array([.001,.02,.9,.01]);q=holm(p)
    assert np.all(q>=p) and np.all(q<=1)
    assert np.all(np.diff(q[np.argsort(p)])>=0)

def test_reference_alignment_preserves_direction():
    from analyze import align
    x=np.array([[0,2,3],[4,0,6],[7,8,0]])
    a=align(x,['A','B','C'],['C','A','X'])
    assert a[0,1]==7 and a[1,0]==3 and np.isnan(a[2]).all()

def test_contexts_equal_events_and_paired_phase_contrast():
    from analyze import contexts
    x=np.arange(2*5*3*2*2*2).reshape(2,5,3,2,2,2).astype(float)
    out=contexts(x)
    np.testing.assert_allclose(out['state_average'],x.mean((1,2)))
    np.testing.assert_allclose(out['onset_minus_baseline'],x[:,1].mean(1)-x[:,0].mean(1))

def test_direct_importance_agrees_with_independent_precision_oracle():
    from oracle import exact,LinearAdapter
    from compatibility_neural_benchmark.core import RepairedResponseConfig,generate_path_bank,estimate_repaired_responses
    p=np.array([[1,.25],[.15,1.2]],dtype=np.float32)
    cfg=RepairedResponseConfig(history_frames=1,repair_frames=5,source_window_frames=4,source_lag_frames=1,horizon_frames=(1,4),n_particles=32768,min_ess=12)
    truth,_=exact(cfg,p,-.6,.6)
    prefix,future,factual=generate_path_bank(LinearAdapter(),np.zeros((15,2),np.float32),np.zeros(15,np.float32),cut_time=7,config=cfg,seed=123)
    z=estimate_repaired_responses(prefix,future,factual,p,np.full(2,-.6),np.full(2,.6),np.ones(2),np.zeros(2),cfg)
    assert np.sqrt(np.mean((z['response_endpoint_mean']-truth)**2))<.04

def test_full_archive_aggregation_preserves_source_target_and_recording_units(tmp_path,monkeypatch):
    import analyze
    from compatibility_neural_benchmark.prediction_atlas_runner import output_path
    monkeypatch.setattr(analyze,'R',tmp_path);monkeypatch.setattr(analyze,'OUT',tmp_path/'analysis')
    name='historical80';root=tmp_path/'runs'/name;root.mkdir(parents=True)
    names=['A','B','C','D'];worms=[f'w{i}' for i in range(5)]
    (root/'cohort.json').write_text(json.dumps({'neurons':names,'worm_ids':worms}))
    cfg,*_=config_for(name)
    for seed in SET['seeds']:
        for lag in SET['lags']:
            for fold in SET['folds']:
                raw=np.zeros((1,5,3,4,6,4),np.float32)
                for source in range(4):
                    for target in range(4):raw[:,:,:,source,:,target]=100*source+target+fold
                gap=np.ones((1,5,3,4),np.float32)*2
                arrays={'response_'+k:raw for k in analyze.CHANNELS}
                arrays.update(worm_ids=np.array([worms[fold]]),diagnostic_achieved_gap=gap,diagnostic_valid=np.ones_like(gap),diagnostic_ess_low=np.ones_like(gap)*20,diagnostic_ess_high=np.ones_like(gap)*20,diagnostic_distinct_ancestors_low=np.ones_like(gap)*32,diagnostic_distinct_ancestors_high=np.ones_like(gap)*32)
                p=output_path(root/'primary','progressive_bridge_smc',cfg.model_id,lag,fold,seed,64);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**arrays)
    p=analyze.aggregate(name,'primary')
    with np.load(p) as z:
        assert z['endpoint_mean__state_average'].shape==(4,5,6,4,4)
        assert np.isclose(z['endpoint_mean__state_average'][0,3,0,2,1],52.5)
        assert np.all(z['source_strong'])
        np.testing.assert_allclose(z['endpoint_mean__onset_minus_baseline'],0,atol=1e-6)
        np.testing.assert_allclose(z['seed__endpoint_mean__state_average'].mean(0),z['endpoint_mean__state_average'],rtol=1e-6)
