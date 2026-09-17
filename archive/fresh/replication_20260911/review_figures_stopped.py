"""Render unchanged data into separately preserved, spaced review figures."""
from pathlib import Path
import report_stopped as report

original_subplots=report.plt.subplots
def spaced_subplots(*args,**kwargs):
    kwargs['layout']='constrained'
    if kwargs.get('figsize')==(14,4):kwargs['figsize']=(16,4.5)
    if kwargs.get('figsize')==(12,4):kwargs['figsize']=(14,4.5)
    if kwargs.get('figsize')==(10,7):kwargs['figsize']=(10,8)
    return original_subplots(*args,**kwargs)

def save_reviewed(fig,path):
    cohort=path.parent.parent.name
    target=report.R/'final_review_20260916'/cohort/'figures'/path.name
    target.parent.mkdir(parents=True,exist_ok=True)
    if path.stem=='reference_lag_profiles':
        title=fig._suptitle.get_text()
        fig._suptitle.remove();fig._suptitle=None
        fig.suptitle(title,fontsize=11)
    # Shared subplot labels only need to appear once in horizontal panels.
    if path.stem in {'lag_effect_matrices','replication_stability'}:
        for ax in fig.axes[1:-1]:ax.set_ylabel('')
    if path.stem=='lag_effect_matrices':
        fig.suptitle(f'{cohort}: state-average endpoint mean, horizon 1; diagonal shown, excluded from scores')
    if path.stem=='replication_stability':
        labels={'generator_0_vs_1':'Training seeds 1701 vs 2903','independent_MC':'Independent Monte Carlo','direct_N4096':'Direct importance, N=4096'}
        for ax in fig.axes:
            if ax.get_title() in labels:ax.set_title(labels[ax.get_title()],fontsize=10)
    if path.stem=='particle_convergence':
        fig.suptitle('Qualified-reference subset; particle counts are not equal computational budgets',fontsize=10)
    fig.savefig(target,bbox_inches='tight')
    fig.savefig(target.with_suffix('.svg'),bbox_inches='tight')
    report.plt.close(fig)

if __name__=='__main__':
    report.plt.subplots=spaced_subplots
    report.save=save_reviewed
    for cohort in report.SET['cohorts']:
        report.figures(cohort)
        print('RENDERED',cohort,flush=True)
