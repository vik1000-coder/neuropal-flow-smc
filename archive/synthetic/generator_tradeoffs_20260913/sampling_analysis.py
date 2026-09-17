"""Post-run summary using the separately validated analytic truth supplement."""
from pathlib import Path
import pandas as pd

def main():
    out=Path(__file__).resolve().parent/'analysis'
    effects=pd.read_csv(out/'analytic_effects_raw.csv')
    samples=pd.read_csv(out/'sampling_raw.csv')
    keys=['law','data_seed','family','init','region','event']
    samples=samples.merge(effects[keys+['supplement_truth','supplement_model_effect','supplement_model_qualified']],on=keys,validate='many_to_one')
    samples['particle_abs_error']=(samples.estimate-samples.supplement_model_effect).abs().where(samples.supplement_model_qualified)
    samples['total_abs_error']=(samples.estimate-samples.supplement_truth).abs()
    grouped=samples.groupby(['law','data_seed','family','region','event','n']).agg(particle_abs_error=('particle_abs_error','mean'),total_abs_error=('total_abs_error','mean'),qualified_fraction=('supplement_model_qualified','mean'),ess=('ess','mean')).reset_index()
    grouped.to_csv(out/'sampling_analytic_dataset_means.csv',index=False)
    grouped.groupby(['family','n'])[['particle_abs_error','total_abs_error','qualified_fraction','ess']].mean().to_csv(out/'sampling_analytic_summary.csv')

if __name__=='__main__':main()
