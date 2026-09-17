"""Regression checks for the new synthesis, independent of missing historical runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
R=Path(__file__).resolve().parents[1];D=R/'data/publication'

def test_primary_matrix_scores_match_saved_reference_tables():
    for cohort in ['clean54','historical80']:
        with np.load(D/cohort/'matrices.npz') as z:matrix=z['endpoint_mean__state_average'][0,0]
        with np.load(D/cohort/'reference_matrices.npz') as refs:
            saved=pd.read_csv(D/cohort/'reference_metrics.csv');saved=saved[saved.scope=='all_common']
            for ref in ['randi_wild_type','cook_struct','cook_chem','cook_gap']:
                mask=refs[ref+'_mask']&np.isfinite(matrix)&np.isfinite(refs['published_sbtg']);y=refs[ref+'_labels'][mask]
                for method,a in [('flow_ensemble',matrix),('published_sbtg',refs['published_sbtg'])]:
                    row=saved[(saved.reference==ref)&(saved.method==method)].iloc[0]
                    assert np.isclose(roc_auc_score(y,abs(a[mask])),row.auroc,atol=1e-12)
                    assert np.isclose(average_precision_score(y,abs(a[mask])),row.auprc,atol=1e-12)
                    assert mask.sum()==row.n_edges

def test_neuron_selection_is_from_fixed_corrected_family_and_matches_matrices():
    for cohort in ['clean54','historical80']:
        inf=pd.read_csv(D/cohort/'all_corrected_tests.csv.gz');selected=pd.read_csv(D/cohort/'selected_baseline_edges.csv')
        eligible=inf[(inf.channel=='endpoint_mean')&(inf.context=='baseline')&(inf.test=='effect')&(inf.lag==1)&(inf.horizon==1)&(inf.joint_maxT_p<=.05)].copy()
        eligible['abs_effect']=eligible.estimate.abs();expected=eligible.sort_values(['abs_effect','source','target'],ascending=[False,True,True]).head(6)
        assert list(zip(selected.source,selected.target))==list(zip(expected.source,expected.target))
        with np.load(D/cohort/'matrices.npz') as z:
            names=z['neurons'].astype(str).tolist()
            for row in selected.itertuples():
                i,j=names.index(row.target),names.index(row.source)
                assert z['source_strong'][j]
                assert np.isclose(z['endpoint_mean__baseline'][0,0,i,j],row.estimate,atol=1e-6)
                np.testing.assert_allclose(z['seed__endpoint_mean__baseline'].mean(0),z['endpoint_mean__baseline'],atol=1e-6)

def test_no_clean_lag_discoveries_are_created_by_plot_selection():
    f=pd.read_csv(D/'clean54/all_corrected_tests.csv.gz')
    assert not ((f.test=='lag_minus_1')&(f.joint_maxT_p<=.05)).any()
    counts=pd.read_csv(D/'lag_counts.csv')
    for cohort in ['clean54','historical80']:
        f=pd.read_csv(D/cohort/'all_corrected_tests.csv.gz')
        for row in counts[counts.cohort==cohort].itertuples():
            g=f[(f.channel==row.channel)&(f.context==row.context)&(f.test==row.test)]
            assert len(g)==row.tested_cells
            assert (g.joint_maxT_p<=.05).sum()==row.significant_cells

def test_synthetic_plot_grain_is_dataset_after_initialization_aggregation():
    p=pd.read_csv(D/'synthetic/predictive_dataset_means.csv')
    assert not p.duplicated(['law','data_seed','family','region']).any()
    assert p.family.nunique()==9 and p.law.nunique()==4 and p.data_seed.nunique()==3
    e=pd.read_csv(D/'synthetic/analytic_effects_dataset_means.csv')
    assert not e.duplicated(['law','data_seed','family','region','event']).any()

def test_notebook_executed_without_hidden_errors():
    n=json.loads((R/'notebooks/four_neuron_tutorial.ipynb').read_text())
    code=[c for c in n['cells'] if c['cell_type']=='code']
    assert all(c['execution_count'] is not None for c in code)
    assert not any(o.get('output_type')=='error' for c in code for o in c.get('outputs',[]))
    summary=json.loads((R/'analysis/toy_outputs/summary.json').read_text())
    assert summary['analytic_jacobian_check']=='passed for all 48 entries'
    assert summary['heldout_mean_rmse_standardized']<.2
