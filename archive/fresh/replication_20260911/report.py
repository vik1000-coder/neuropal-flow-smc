"""Write a results report only after the complete replication evidence exists."""
from common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import OUT,CHANNELS,CONTEXTS

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':180,'axes.titlelocation':'left'})
COLORS={'flow_ensemble':'#2563eb','published_sbtg':'#374151','flow':'#2563eb','persistence_gaussian':'#b45309','ridge_gaussian':'#059669'}

def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,bbox_inches='tight');fig.savefig(path.with_suffix('.svg'),bbox_inches='tight');plt.close(fig)

def figures(name):
    folder=OUT/name/'figures';refs=pd.read_csv(OUT/name/'reference_metrics.csv');x=refs[(refs.scope=='all_common')&refs.method.isin(['flow_ensemble','published_sbtg'])]
    fig,ax=plt.subplots(figsize=(9,4));names=list(x.reference.unique());positions=np.arange(len(names))
    for j,method in enumerate(['published_sbtg','flow_ensemble']):
        rows=x[x.method==method].set_index('reference').reindex(names);ax.bar(positions+(j-.5)*.32,rows.auroc,width=.32,label=method,color=COLORS[method])
    ax.set_xticks(positions,names,rotation=15);ax.set_ylabel('AUROC');ax.set_ylim(0,1);ax.axhline(.5,color='#9ca3af',lw=.8);ax.legend();ax.set_title(f'{name}: fixed nominal lag 1 / horizon 1 comparison');save(fig,folder/'reference_comparison.png')
    df=pd.read_csv(OUT/name/'reference_lag_profiles.csv');f=df[(df.method=='flow')&(df.channel=='endpoint_mean')&(df.context=='state_average')&(df.horizon_frames==1)]
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for ax,ref in zip(axs.flat,names):
        for scope,g in f[f.reference==ref].groupby('scope'):
            ax.plot(g.source_lag_frames/4,g.auroc,'o-',label=scope)
        ax.set_title(ref);ax.set_xlabel('Source-window end to cut (s)');ax.set_ylabel('AUROC');ax.set_ylim(0,1);ax.axhline(.5,color='#9ca3af',lw=.7)
    axs[0,0].legend(fontsize=8);fig.suptitle(f'{name}: descriptive lag profile; forecast horizon = 0.25 s',y=1.01);save(fig,folder/'reference_lag_profiles.png')
    with np.load(OUT/name/'primary.npz') as z:
        fig,axs=plt.subplots(2,1,figsize=(13,4.5),sharex=True)
        for ax,key in zip(axs,['valid_fraction','genealogy_fraction']):
            im=ax.imshow(z[key],aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.set_yticks(range(4),SET['lags']);ax.set_ylabel('Nominal lag');ax.set_title(key.replace('_',' '));fig.colorbar(im,ax=ax,shrink=.8)
        ticks=np.arange(0,len(z['neurons']),3);axs[-1].set_xticks(ticks,z['neurons'][ticks],rotation=90);save(fig,folder/'source_support.png')
        a=z['endpoint_mean__state_average'][:,:,0].mean(1);limit=float(np.quantile(abs(a),.99));limit=max(limit,1e-6)
        fig,axs=plt.subplots(1,4,figsize=(14,4))
        for i,ax in enumerate(axs):
            im=ax.imshow(a[i],vmin=-limit,vmax=limit,cmap='RdBu_r',origin='upper');ax.set_title(f'Nominal source lag {SET["lags"][i]}');ax.set_xlabel('Source index');ax.set_ylabel('Target index')
        fig.colorbar(im,ax=list(axs),shrink=.6,label='Gap-normalized high−low response');save(fig,folder/'lag_effect_matrices.png')
    stability=pd.read_csv(OUT/name/'stability.csv');f=stability[(stability.channel=='endpoint_mean')&(stability.context=='state_average')&(stability.scope=='all')]
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,comp in zip(axs,['generator_0_vs_1','independent_MC','direct_N4096']):
        grid=f[f.comparison==comp].pivot(index='lag',columns='horizon',values='spearman').reindex(index=SET['lags'],columns=SET['horizons']);im=ax.imshow(grid,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto');ax.set_xticks(range(6),SET['horizons']);ax.set_yticks(range(4),SET['lags']);ax.set_title(comp);ax.set_xlabel('Forecast horizon (frames)');ax.set_ylabel('Nominal source lag')
    fig.colorbar(im,ax=list(axs),shrink=.7,label='Signed matrix Spearman correlation');save(fig,folder/'replication_stability.png')
    df=pd.read_csv(OUT/name/'predictive_summary.csv');fig,ax=plt.subplots(figsize=(7,4))
    for method,g in df.groupby('method'):ax.plot(g.horizon/4,g.energy,'o-',label=method,color=COLORS[method])
    ax.set_xlabel('Forecast horizon (s)');ax.set_ylabel('Energy score (lower is better)');ax.legend();ax.set_title(f'{name}: equal-recording factual prediction');save(fig,folder/'predictive_rollout.png')
    df=pd.read_csv(OUT/name/'particle_convergence.csv');fig,ax=plt.subplots(figsize=(7,4))
    for method,g in df.groupby('method'):
        curve=g.groupby('particles').resolved_reference_mae.mean()
        ax.plot(curve.index,curve.values,'o-',label=method)
    ax.set_xscale('log',base=2);ax.set_xlabel('Particles / natural paths');ax.set_ylabel('MAE vs qualified direct N16384 reference');ax.set_title(f'{name}: subset particle convergence');ax.legend();save(fig,folder/'particle_convergence.png')
    return sorted(folder.glob('*.png'))

def table(frame,cols=None):
    if cols is not None:frame=frame[[c for c in cols if c in frame.columns]]
    return frame.to_markdown(index=False,floatfmt='.4f') if len(frame) else '_No eligible rows._'

def run():
    completed=json.loads((OUT/'analysis_complete.json').read_text());assert completed['status']=='complete' and completed['spec_id']==spec_id()
    sections=[];plots=[];training=[]
    for name in SET['cohorts']:
        for seed in SET['seeds']:
            for fold in SET['folds']:
                r=json.loads(receipt_path(name,fold,seed).read_text());checked_checkpoint(name,fold,seed);training.append({'cohort':name,'seed':seed,'fold':fold,**r['convergence'],'energy':r['record']['energy'],'train_seconds':r['record']['train_seconds']})
    training=pd.DataFrame(training);training.to_csv(OUT/'training_summary.csv',index=False)
    for name in SET['cohorts']:
        plots+=figures(name)
        delta=pd.read_csv(OUT/name/'reference_deltas.csv');primary=delta[delta.scope=='all_common'];sig=pd.read_csv(OUT/name/'lag_inference.csv');stability=pd.read_csv(OUT/name/'stability.csv');conv=pd.read_csv(OUT/name/'particle_convergence.csv');pred=pd.read_csv(OUT/name/'predictive_summary.csv');lagmax=pd.read_csv(OUT/name/'lagmax_reference_permutations.csv')
        effects=sig[(sig.test=='effect')&(sig.joint_maxT_p<=.05)] if len(sig) else sig
        lags=sig[(sig.test=='lag_minus_1')&(sig.joint_maxT_p<=.05)] if len(sig) else sig
        improved=primary[primary.simultaneous_one_sided_lower>0] if 'simultaneous_one_sided_lower' in primary else primary.iloc[:0]
        conditional_scope='Historical copied/pseudo-paired recording-row descriptions; not independent-animal inference.' if name=='historical80' else 'Conditional recording sign-flip inference under joint symmetry; overlapping training fits and model/refit uncertainty remain.'
        with np.load(OUT/name/'primary.npz') as z:strong=int(z['source_strong'].sum());total=len(z['neurons'])
        stable=stability[(stability.channel=='endpoint_mean')&(stability.context=='state_average')&(stability.horizon==1)&(stability.scope=='all')]
        desc=f'''## {name}

The ensemble exceeds published SBTG with a positive source-bootstrap simultaneous one-sided lower bound on **{len(improved)} of four** primary reference panels. This is conditional reference correspondence, not proof of synapses or causal recovery. The complete per-seed and mask-specific comparisons remain in `analysis/{name}/reference_metrics.csv` and `reference_deltas.csv`.

{table(primary,['reference','delta_auroc','delta_ci_low','delta_ci_high','simultaneous_one_sided_lower','holm_bootstrap_tail_p'])}

![Reference comparison](analysis/{name}/figures/reference_comparison.png)

### Lag reliability

**{strong}/{total} sources** meet the fixed all-lag strong support/genealogy rule. Joint max-T yields **{len(effects)} effect cells** and **{len(lags)} lag-contrast cells** at p ≤ .05. These are cells, not distinct edges or additional animals. {conditional_scope} Lag changes also change the repair boundary and anchoring interval. The nominal source lag and forecast horizon are reported separately; no physical transmission delay is inferred.

{table(stable,['comparison','lag','spearman','rmse','sign_agreement'])}

![Matrix stability](analysis/{name}/figures/replication_stability.png)
![Lag profiles](analysis/{name}/figures/reference_lag_profiles.png)
![Lag matrices](analysis/{name}/figures/lag_effect_matrices.png)
![Support](analysis/{name}/figures/source_support.png)

The endpoint-timing-matched comparisons use nominal flow lags 0/7 plus one forecast frame against published SBTG lags 1/8. See the separately labeled rows in `reference_deltas.csv`; equality of endpoint timing does not make a four-frame repaired-history contrast identical to the SBTG estimand.

The separate within-source label-permutation lag-maximum sensitivity corrects for choosing the largest reference AUROC over four lags and then adjusts its 16-panel family:

{table(lagmax,['channel','context','reference','best_nominal_lag','max_auroc','permutation_p','holm_family_p','status'])}

### Factual prediction and sampler convergence

{table(pred,['method','horizon','energy','rmse','coverage90'])}

![Factual rollout](analysis/{name}/figures/predictive_rollout.png)
![Particle convergence](analysis/{name}/figures/particle_convergence.png)

The particle figure uses only queries whose three independent N16384 direct runs all pass validity, have minimum ESS ≥ 12, and have normalized-response SD ≤ .05. The resolved-reference fractions, estimator validity, and unconditional differences are all retained in `particle_convergence.csv`. Unresolved direct references cannot adjudicate progressive SMC accuracy. `diagnostic_runs.csv` records actual transition/velocity evaluations and wall time; equal N is not equal computation.

### Derivative, zero-query and solver checks

`derivative_stability.csv` records means and MC standard deviations at full, half and quarter source separation. `diagnostic_cells.csv.gz` retains raw contrasts, requested gaps, achieved gaps and both normalizations. Small or reversed achieved gaps do not become valid derivatives through the historical .10 denominator floor. Fixed clamp bandwidth means this is at most a local sensitivity of a softened model-conditioned path law.

{table(pd.read_csv(OUT/name/'zero_query_controls.csv'))}

The equal-clamp negative control has exact raw contrast zero. Its nonzero raw MSE quantifies finite-particle arm noise. `solver_stability.csv` compares original and doubled flow integration steps with matched seeds. Neither diagnostic is evidence of biological causality.
'''
        ext=OUT/name/'extended_reference_sensitivity.csv'
        if ext.exists():desc+='\n### Predeclared extended-training sensitivity\n\nOriginal primary estimates remain above. This ensemble replaces only validation-triggered boundary-best fits with separately retrained 200-epoch-budget fits.\n\n'+table(pd.read_csv(ext))+'\n'
        sections.append(desc)
    report=f'''# Fresh flow / progressive bridge SMC replication

Completed {now()}. Protocol identity: `{spec_id()}`.

This report distinguishes fresh computational reproduction, predictive quality, particle accuracy, reference correspondence and lag reliability. All **30 primary flow fits were retrained** from copied inputs; none reused an old flow checkpoint. All three generator seeds contribute equally. The published SBTG hybrid manuscript matrices are the frozen comparator. This study reuses existing recordings and does not establish independent biological replication.

## Training and convergence

{table(training)}

Fits with a best validation epoch near an exhausted primary budget trigger the separate extended-training sensitivity recorded in `analysis/extension_decisions.csv`. Early stopping and finite loss are implementation checks, not evidence that the learned law is correct.

## Independent validation

The copied source suite passed 309 tests with two recorded skips. The independent Gaussian precision oracle passed its prespecified bias and particle-improvement gates at lags 0/1/4/16; the standalone replication tests validate orientation, exact joint sign inference, source bootstrap, normalization and direct sampling against the oracle. The complete logs and receipts are under `validation/`.

The historical training cache matches released prepared traces after exact neuron-name reordering and float32 conversion, including missingness. The corrected head-only cohort is 17 recordings and 54 classes. All training folds are recording-disjoint with training-only scaling. Sampler nuisance summaries use non-test recordings, including validation recordings, as in the replicated method.

'''+ '\n'.join(sections)+'''
## Scope and reproducibility

- Read `PROTOCOL.md` and `ANALYSIS_IMPLEMENTATION.md` for the prespecified decisions and exact estimands.
- `snapshot_manifest.json` binds copied source and inputs. `execution_manifest.json` binds primary fitting/sampling workers. Per-fit and per-archive receipts enforce checkpoint and configuration identity.
- `analysis/frozen_sampling.json` seals raw matrices before external scoring. All numerical results and source/reference masks are inspectable.
- Historical 80-class head/tail pseudo-pairing and donor copying are preserved solely for comparator compatibility; source-block or recording-row intervals do not undo those dependencies.
- Model-seed/Monte Carlo repeats are not independent animals. Conditional sign-flip inference does not incorporate complete refitting/selection uncertainty.
- Source histories span four frames and lagged repairs alter their boundary. A reliable conditional response matrix is still not an anatomical, receptor-specific or physical-delay matrix.
- Strong claims require prospective independent recordings and targeted interventions.

Atlas sources: [Cook et al. (2019)](https://www.nature.com/articles/s41586-019-1352-7), [Randi et al. (2023)](https://www.nature.com/articles/s41586-023-06683-4). Published SBTG inputs and the original source lineage are preserved in `inputs/published_sbtg/`.
'''
    (R/'REPORT.md').write_text(report)
    # Completion means all required evidence was produced; negative scientific results still complete the study.
    required=[]
    for name in SET['cohorts']:
        required.extend(OUT/name/f for f in ['primary.npz','mc_repeat.npz','direct.npz','matched_time.npz','reference_metrics.csv','reference_deltas.csv','reference_lag_profiles.csv','stability.csv','lag_inference.csv','lag_joint_null.npz','lagmax_reference_permutations.csv','particle_convergence.csv','derivative_stability.csv','solver_stability.csv','zero_query_controls.csv','diagnostic_runs.csv','diagnostic_cells.csv.gz','predictive_summary.csv'])
    missing=[str(p) for p in required if not p.is_file() or p.stat().st_size==0]
    assert len(training)==30 and len(plots)==14 and not missing
    receipt={'status':'pass','meaning':'artifact completion and numerical schema validation, not a positive scientific verdict','spec_id':spec_id(),'completed_utc':now(),'primary_fits':30,'figure_count':len(plots),'report_sha256':sha(R/'REPORT.md'),'report_source_sha256':sha(Path(__file__)),'analysis_files':{str(p.relative_to(R)):sha(p) for p in required},'visual_review':'pending human/agent rendered inspection; not asserted by this numerical check'}
    atomic_json(R/'validation/final.json',receipt)
    print('REPORT_COMPLETE',R/'REPORT.md',flush=True)
if __name__=='__main__':run()
