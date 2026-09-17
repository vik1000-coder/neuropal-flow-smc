"""Reproduce publication figures from compact checked inputs (no training required)."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=R/'data/publication';OUT=R/'figures'
REFS=['randi_wild_type','cook_struct','cook_chem','cook_gap'];LABELS=['Randi functional','Cook structural','Cook chemical','Cook gap']
BLUE='#2166ac';ORANGE='#d97722';GREY='#666666';GREEN='#247a58'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':170,'pdf.fonttype':42})
CATALOG=[]
def finish(fig,cohort,name,title,caption,source):
    folder=OUT/cohort;folder.mkdir(parents=True,exist_ok=True)
    fig.suptitle(title,fontsize=15,y=.99)
    fig.text(.02,.015,caption,fontsize=8,va='bottom',wrap=True)
    fig.tight_layout(rect=[0,.085,1,.94])
    for suffix in ['png','pdf']:fig.savefig(folder/f'{name}.{suffix}')
    plt.close(fig)
    paths=[]
    for item in source:
        choices=[R/item,D/cohort/item,D/item]
        found=next((p for p in choices if p.exists()),None)
        if found is None:raise FileNotFoundError(item)
        paths.append(str(found.relative_to(R)))
    CATALOG.append(dict(id=f'{cohort}/{name}',title=title,caption=caption,inputs=paths))
def cohort_label(c):return 'Corrected 54 classes · 17 recordings' if c=='clean54' else 'Historical 80 classes · 20 prepared traces'
def caveat(c):return 'Historical head/tail pseudo-pairing and donor copying; conditional model summaries only.' if c=='historical80' else 'Existing recordings; fitted-model uncertainty is not fully represented.'

def cohort_figures(c):
    p=D/c;metrics=pd.read_csv(p/'reference_metrics.csv');delta=pd.read_csv(p/'reference_deltas.csv');z=np.load(p/'matrices.npz');names=z['neurons'].astype(str);lags=z['lags'];horizons=z['horizons']
    primary=metrics[metrics.scope=='all_common'];fig,ax=plt.subplots(1,3,figsize=(14,5.3));y=np.arange(4)
    for j,metric in enumerate(['auroc','auprc']):
        for method,color,marker,label in [('published_sbtg',GREY,'s','Published SBTG'),('flow_ensemble',BLUE,'o','Fresh flow + SMC')]:
            values=primary[primary.method==method].set_index('reference').loc[REFS,metric]
            ax[j].plot(values,y,marker,ms=7,color=color,label=label)
        ax[j].set(yticks=y,yticklabels=LABELS,xlabel='AUROC' if metric=='auroc' else 'Average precision',xlim=(.45,.76) if metric=='auroc' else (.06,.45));ax[j].invert_yaxis();ax[j].grid(axis='x',alpha=.2)
    ax[0].legend(loc='lower right',fontsize=8)
    for scope,color,offset,marker in [('all_common',BLUE,-.12,'o'),('strong_fixed_support',ORANGE,.12,'s')]:
        a=delta[delta.scope==scope].set_index('reference').loc[REFS]
        ax[2].errorbar(a.delta_auroc,y+offset,xerr=[a.delta_auroc-a.delta_ci_low,a.delta_ci_high-a.delta_auroc],fmt=marker,color=color,capsize=3,label='All common' if scope=='all_common' else 'Strong support')
    ax[2].axvline(0,color=GREY,lw=1);ax[2].set(yticks=y,yticklabels=LABELS,xlabel='Flow − SBTG AUROC');ax[2].invert_yaxis();ax[2].legend(fontsize=8)
    finish(fig,c,'01_atlas_comparison',cohort_label(c)+' | Lag-1 atlas ranking','Fixed nominal lag 1 / horizon 1. Bars: 95% paired source-column bootstrap intervals, fitted matrices fixed.\nMultiplicity-protected lower bounds are in the accompanying tables. '+caveat(c),['reference_metrics.csv','reference_deltas.csv'])

    conv=pd.read_csv(D/'particle_convergence.csv');comp=pd.read_csv(D/'computation.csv');a=conv.merge(comp,on=['cohort','method','particles'],validate='one_to_one');a=a[a.cohort==c]
    cost=pd.read_csv(D/'worker_cost.csv');cost=cost[cost.cohort==c]
    fig,ax=plt.subplots(1,2,figsize=(12,5.5))
    for method,color,marker in [('progressive',BLUE,'o'),('direct',ORANGE,'s')]:
        g=a[a.method==method].sort_values('particles');ax[0].plot(g.mean_wall_seconds,g.pooled_mae,'-',marker=marker,color=color,label=method)
        for row in g.itertuples():ax[0].annotate(f'N={row.particles}',(row.mean_wall_seconds,row.pooled_mae),xytext=(4,6),textcoords='offset points',fontsize=8)
    ax[0].set(xscale='log',xlabel='Mean diagnostic wall seconds (log scale)',ylabel='MAE against qualified direct-N16384 reference');ax[0].legend();ax[0].grid(alpha=.2)
    ax[1].barh(cost.stage,cost.worker_hours,color=[GREY if x=='training' else BLUE for x in cost.stage]);ax[1].set(xlabel='Sum of recorded worker hours',title='Selected recorded study stages')
    for i,row in enumerate(cost.itertuples()):ax[1].text(row.worker_hours,i,f' {row.worker_hours:.1f} h',va='center',fontsize=8)
    ax[1].set_xlim(0,cost.worker_hours.max()*1.2)
    coverage=a.resolved_reference_fraction.iloc[0]
    finish(fig,c,'02_sampling_and_cost',cohort_label(c)+' | Sampling accuracy and feasibility',f'Reference qualified on {coverage:.1%} of diagnostic cells. Direct N16384 uses leave-one-repeat-out comparison.\nSelected worker hours exclude diagnostics, predictive runs and partial extension. No matched fresh SBTG timing.', ['particle_convergence.csv','computation.csv','worker_cost.csv'])

    st=pd.read_csv(p/'stability.csv');st=st[(st.channel=='endpoint_mean')&(st.context=='state_average')&(st.lag==1)&(st.horizon==1)&(st.scope=='all')]
    numerical=pd.read_csv(D/'numerical_adjudication.csv');n=numerical[(numerical.cohort==c)&(numerical.panel=='derivative')&(numerical.normalization=='normalized_effect')&(numerical.scope=='valid_gap_ge_0.1')&(numerical.horizon.astype(str)=='1')]
    support=pd.read_csv(D/'support.csv').set_index('cohort').loc[c];train=pd.read_csv(D/'training.csv');train=train[train.cohort==c]
    fig,ax=plt.subplots(1,3,figsize=(14,5.6));labels={'generator_0_vs_1':'Training seeds 1 vs 2','generator_0_vs_2':'Training seeds 1 vs 3','generator_1_vs_2':'Training seeds 2 vs 3','independent_MC':'Independent MC','direct_N4096':'Direct N4096'}
    ax[0].barh([labels.get(x,x) for x in st.comparison],st.spearman,color=BLUE);ax[0].set(xlim=(0,1),xlabel='Signed matrix Spearman correlation')
    pair=n[(n['first']==1)&(n['second']==.25)].iloc[0]
    ax[1].bar(['Full contrast','Quarter contrast'],[pair.median_mc_sd_first,pair.median_mc_sd_second],color=[BLUE,ORANGE]);ax[1].set(ylabel='Median normalized-effect MC SD',title=f'Paired cells retained: {pair.coverage:.0%}')
    ax[2].bar(['All','Strong support','Strict complete'],[support.all_sources,support.strong_sources,support.strict_complete_sources],color=[GREY,BLUE,ORANGE]);ax[2].set(ylabel='Number of source classes',ylim=(0,len(names)*1.18));ax[2].tick_params(axis='x',rotation=15)
    cap=int(train.epoch_cap_reached.sum());extension=int(train.extension_required.sum())
    finish(fig,c,'03_robustness',cohort_label(c)+' | What replicates and what remains noisy',f'15 primary fits; {cap} reached the epoch cap; {extension} met the predeclared extension criterion. Smaller-contrast MC spread is not a confidence interval.\nZero strict-complete sources does not mean every individual episode fails. '+caveat(c),['stability.csv','numerical_adjudication.csv','support.csv','training.csv'])

    pred=pd.read_csv(p/'predictive_summary.csv');prof=pd.read_csv(p/'reference_lag_profiles.csv');fig,ax=plt.subplots(1,3,figsize=(14,5.3))
    for method,color,marker in [('flow',BLUE,'o'),('persistence_gaussian',GREY,'s'),('ridge_gaussian',ORANGE,'^')]:
        q=pred[pred.method==method].sort_values('horizon')
        ax[0].plot(q.horizon/4,q.energy,marker=marker,color=color,label=method.replace('_gaussian',''))
        ax[1].plot(q.horizon/4,q.coverage90,marker=marker,color=color)
    ax[0].set(xlabel='Forecast horizon (seconds)',ylabel='Energy score (lower is better)');ax[0].legend(fontsize=8)
    ax[1].axhline(.9,color='k',ls='--',lw=1);ax[1].set(xlabel='Forecast horizon (seconds)',ylabel='90% interval coverage',ylim=(.5,1))
    for ref,label in zip(REFS,LABELS):
        q=prof[(prof.channel=='endpoint_mean')&(prof.context=='state_average')&(prof.horizon_frames==1)&(prof.scope=='all_common')&(prof.reference==ref)&(prof.method=='flow')].sort_values('source_lag_frames')
        ax[2].plot(q.source_lag_frames/4,q.auroc,'o-',label=label)
    ax[2].set(xlabel='Source placement lag (seconds)',ylabel='AUROC at forecast horizon 0.25 s');ax[2].legend(fontsize=8)
    finish(fig,c,'04_prediction_and_lag',cohort_label(c)+' | Prediction and two time axes','Predictive curves are equal-recording summaries, averaged over training seeds. No independent refit interval is implied.\nSource placement lag, forecast horizon, and source-to-endpoint separation (their sum) are different quantities.', ['predictive_summary.csv','reference_lag_profiles.csv'])

    inf=pd.read_csv(p/'all_corrected_tests.csv.gz');chosen=pd.read_csv(p/'selected_baseline_edges.csv');fig,axes=plt.subplots(2,3,figsize=(13,8));axes=axes.ravel()
    for ax,row in zip(axes,chosen.itertuples()):
        si=list(names).index(row.source);ti=list(names).index(row.target);seed=z['seed__endpoint_mean__baseline'][:,:,0,ti,si]
        ax.plot(lags/4,seed.mean(0),'o-',color=BLUE,label='3-seed mean');ax.fill_between(lags/4,seed.min(0),seed.max(0),color=BLUE,alpha=.15,label='Training-seed range')
        ax.axhline(0,color=GREY,lw=.8);ax.set(title=f'{row.source} → {row.target}\nlag1/h1 adjusted p={row.joint_maxT_p:.3g}',xlabel='Source placement lag (s)',ylabel='Gap-normalized baseline response')
        changes=inf[(inf.channel=='endpoint_mean')&(inf.context=='baseline')&(inf.test=='lag_minus_1')&(inf.horizon==1)&(inf.source==row.source)&(inf.target==row.target)]
        for rr in changes[changes.joint_maxT_p<=.05].itertuples():
            li=list(lags).index(rr.lag);ax.plot(rr.lag/4,seed.mean(0)[li],marker='*',color=ORANGE,ms=14)
    for ax in axes[len(chosen):]:ax.axis('off');ax.text(.1,.5,'No qualifying example',transform=ax.transAxes)
    if len(chosen):axes[0].legend(fontsize=8)
    finish(fig,c,'05_selected_neuron_effects',cohort_label(c)+' | Selected baseline effects','Selection: six largest |baseline mean| among corrected p≤.05 at fixed lag1/h1. Shading is training-seed range, not CI.\nStars: corrected lag-minus-1 test at the same horizon. Selected on these data; not independent confirmation. '+caveat(c),['selected_baseline_edges.csv','matrices.npz','all_corrected_tests.csv.gz'])

    fig,ax=plt.subplots(1,2,figsize=(12,5.7));m=z['endpoint_mean__baseline'][0,0].copy();m[np.eye(len(names),dtype=bool)]=np.nan
    tests=inf[(inf.channel=='endpoint_mean')&(inf.context=='baseline')&(inf.test=='effect')&(inf.lag==1)&(inf.horizon==1)&(inf.joint_maxT_p<=.05)]
    sig=np.zeros(m.shape,bool)
    for row in tests.itertuples():sig[list(names).index(row.target),list(names).index(row.source)]=True
    shown=np.where(sig,m,np.nan);lim=np.nanmax(abs(m));im=ax[0].imshow(shown,cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest');fig.colorbar(im,ax=ax[0],label='Signed gap-normalized mean')
    ax[0].set(xlabel='Source class index (order in matrices.npz)',ylabel='Target class index',title=f'Corrected baseline effects: {len(tests)} cells')
    counts=pd.read_csv(D/'lag_counts.csv');counts=counts[(counts.cohort==c)&(counts.channel=='endpoint_mean')&(counts.context=='baseline')]
    vals=[int(counts[counts.test==t].significant_cells.iloc[0]) for t in ['effect','lag_minus_1']]
    ax[1].bar(['Baseline nonzero','Different from lag 1'],vals,color=[BLUE,ORANGE]);ax[1].set(ylabel='Corrected significant cells across lag × horizon',ylim=(0,max(vals+[1])*1.2))
    for i,v in enumerate(vals):ax[1].text(i,v,f'{v:,}',ha='center',va='bottom')
    finish(fig,c,'06_significance_and_lag_change',cohort_label(c)+' | Nonzero effect ≠ identified lag','Left: fixed lag1/h1, baseline endpoint mean, joint max-T p≤.05; blank cells are not discoveries.\nRight: all registered lag/horizon cells, not distinct neuron pairs. Full corrected family and names are supplied. '+caveat(c),['all_corrected_tests.csv.gz','lag_counts.csv','matrices.npz'])


    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, row in zip(axes, chosen.head(3).itertuples()):
        si=list(names).index(row.source);ti=list(names).index(row.target)
        values=z['endpoint_mean__baseline'][:,:,ti,si]
        limit=max(abs(values).max(),1e-6)
        im=ax.imshow(values,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set(xticks=range(len(horizons)),xticklabels=[f'{h/4:g}' for h in horizons],
               yticks=range(len(lags)),yticklabels=[f'{l/4:g}' for l in lags],
               xlabel='Forecast horizon (seconds)',ylabel='Source placement lag (seconds)',title=f'{row.source} → {row.target}')
        fig.colorbar(im,ax=ax,label='Normalized mean')
    for ax in axes[len(chosen.head(3)):]:ax.axis('off')
    finish(fig,c,'07_neuron_lag_surfaces',cohort_label(c)+' | Two-dimensional responses of selected pairs',
           'First three examples from the documented baseline selection. Separate color ranges emphasize each pair’s shape.\nSource-to-endpoint separation equals placement lag + horizon; a maximum is not a physical delay. '+caveat(c),
           ['selected_baseline_edges.csv','matrices.npz'])
    ranked=inf[inf.joint_maxT_p<=.05].sort_values(['joint_maxT_p','cohort','source','target','lag','horizon'])
    ranked.to_csv(p/'significant_cells.csv.gz',index=False)
    # Minimum already-family-adjusted p is a descriptive ranking, not a new p-value.
    pair_table=inf.groupby(['channel','context','test','source','target']).agg(
        minimum_family_adjusted_p=('joint_maxT_p','min'),
        maximum_absolute_effect=('estimate',lambda x: float(x.abs().max())),
        significant_cells=('joint_maxT_p',lambda x: int((x<=.05).sum())),
        tested_cells=('joint_maxT_p','size')).reset_index()
    pair_table.sort_values(['minimum_family_adjusted_p','maximum_absolute_effect'],ascending=[True,False]).to_csv(p/'neuron_pair_summary.csv',index=False)

def shared_figures():
    screen=pd.read_csv(D/'original_generator_screen.csv').sort_values('energy');fig,ax=plt.subplots(figsize=(11,8))
    ax.plot(screen.energy,np.arange(len(screen)),'o',color=BLUE);ax.set(yticks=np.arange(len(screen)),yticklabels=screen.candidate,xlabel='One-step energy score (lower is better)');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    finish(fig,'models','01_original_screen','Original pooled54 | Sixteen-candidate generator screen','Historical report table; original raw runs and uncertainty unavailable. This is the early pooled cohort, not corrected clean54.\nThese predictive rankings do not establish repaired-effect or derivative accuracy.', ['original_generator_screen.csv'])
    p=pd.read_csv(D/'synthetic/predictive_dataset_means.csv');e=pd.read_csv(D/'synthetic/analytic_effects_dataset_means.csv');families=['ridge','gp_rbf','gaussian_diag','gaussian_rank2','student_rank2','mdn4','transformer_mdn4','flow','edm'];laws=['linear_gaussian','nonlinear_heteroskedastic','bimodal','heavy_tailed']
    fig,ax=plt.subplots(2,4,figsize=(16,9))
    for j,law in enumerate(laws):
        for i,(frame,metric) in enumerate([(p,'energy'),(e,'absolute_error')]):
            g=frame[(frame.law==law)&(frame.region=='central')]
            if i:g=g[(g.event=='common')&g.qualified]
            for k,family in enumerate(families):
                v=g[g.family==family][metric].to_numpy();ax[i,j].plot(np.full(len(v),k)+np.linspace(-.13,.13,len(v)),v,'o',color=BLUE if family=='flow' else GREY,alpha=.6,ms=4)
                if len(v):ax[i,j].plot(k,np.mean(v),'_',color='k',ms=10)
            ax[i,j].set(xticks=range(9),xticklabels=families,ylabel='Energy score' if i==0 else 'Conditional-effect absolute error',title=law.replace('_',' '));ax[i,j].tick_params(axis='x',rotation=70)
    finish(fig,'models','02_synthetic_prediction_vs_effect','Known-law synthetic comparison | Prediction and effects can disagree','Each dot is one independently generated dataset; neural initializations averaged first. Black tick: descriptive mean.\nEffects: central common query, analytic/quadrature truth and qualified learned references. Three inputs/outputs, not NeuroPAL dimensions.', ['synthetic/predictive_dataset_means.csv','synthetic/analytic_effects_dataset_means.csv'])
    der=pd.read_csv(D/'synthetic/derivative_dataset_means.csv');cost=pd.read_csv(D/'synthetic/cost_summary.csv');fig,ax=plt.subplots(1,2,figsize=(13,6.5))
    for family in families:
        g=der[(der.family==family)&(der.step==.1)].groupby('region').absolute_error.mean().reindex(['central','boundary','extrapolated'])
        ax[0].plot(range(3),g,'o-',label=family,alpha=.9 if family=='flow' else .65,lw=2 if family=='flow' else 1)
    ax[0].set(xticks=range(3),xticklabels=['Central','Boundary','Extrapolated'],ylabel='Mean history-derivative absolute error');ax[0].legend(fontsize=7,ncol=2)
    cost=cost.set_index('family').loc[families];ax[1].barh(families,cost.sampling_seconds_per_history*1000,color=[BLUE if f=='flow' else GREY for f in families]);ax[1].set(xscale='log',xlabel='Sampling milliseconds per history (log scale)')
    finish(fig,'models','03_synthetic_derivatives_and_cost','Generator choices | Extrapolation reliability and sampling cost','Derivative panel averages the four laws and three datasets at finite-difference step 0.1; see source tables for spread.\nSampling: 256 draws for a three-output history on this CPU. These timings do not predict 54/80-dimensional runtime.', ['synthetic/derivative_dataset_means.csv','synthetic/cost_summary.csv'])

def original_sampler_figure():
    # Transcribed from the preserved corrected E26 table, not reconstructed run data.
    rows=[['Direct N32',.160,4.3,.567,.552,.538,.616],
          ['Terminal SMC N32',.161,64.4,.586,.561,.555,.616],
          ['Progressive N32',.802,139.8,.637,.611,.603,.650],
          ['Temporal-cut N32',.161,63.5,.590,.563,.558,.621]]
    table=pd.DataFrame(rows,columns=['method','valid_fraction','reported_minutes',*LABELS])
    table.to_csv(D/'original_four_sampler_reported.csv',index=False)
    fig,ax=plt.subplots(1,3,figsize=(14,5.8))
    ax[0].barh(table.method,table.valid_fraction,color=[GREY,GREY,BLUE,GREY]);ax[0].set(xlim=(0,1),xlabel='Compatibility-valid fraction')
    ax[1].barh(table.method,table.reported_minutes,color=[GREY,GREY,BLUE,GREY]);ax[1].set(xlabel='Reported total minutes')
    for label in LABELS:ax[2].plot(table[label],range(4),'o',label=label)
    ax[2].set(yticks=range(4),yticklabels=table.method,xlabel='Lag-1 AUROC',xlim=(.50,.68));ax[2].legend(fontsize=7)
    finish(fig,'models','04_original_four_sampler_comparison','Corrected 54 classes | Original four-sampler comparison',
           'Historical E26 report table; raw outputs unavailable. Same particle count, unequal computation: progressive branches proposals and futures.\nAll methods target the same declared repaired law. This panel does not establish matched-cost superiority or biological lag identification.',
           ['original_four_sampler_reported.csv','archive/original/FLOW_REPAIRED_LAG_METHODS_20260828.md'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--refresh-data',action='store_true');args=ap.parse_args()
    if args.refresh_data:
        from build_inputs import main as refresh
        refresh()
    for cohort in ['clean54','historical80']:cohort_figures(cohort)
    shared_figures();original_sampler_figure();(OUT/'catalog.json').write_text(json.dumps(CATALOG,indent=2)+'\n');print('Wrote',len(CATALOG),'figures in PNG and PDF.')
if __name__=='__main__':main()
