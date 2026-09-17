import os
os.nice(15)
import pickle,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import benchmark as b
# Trusted local checkpoints were created by benchmark.py executed as __main__.
Generator=b.Generator
R=Path(__file__).resolve().parent

def main():
    rows=[];paths=sorted((R/'runs').glob('*_transformer_mdn4_*/receipt.json'));assert len(paths)==24
    for receipt in paths:
        run=receipt.parent;record=json.loads(receipt.read_text())
        assert b.sha(run/'model.pkl')==record['files']['model.pkl']
        with open(run/'model.pkl','rb') as f:g=pickle.load(f)
        pred=pd.read_csv(run/'predictive.csv');first=pred.iloc[0];ds,init=int(first.data_seed),int(first.init)
        rng=np.random.default_rng(b.keyed('evaluation',ds));g.model.eval()
        for region in b.CENTERS:
            h=rng.uniform(-.5,.5,(24,3))
            if region=='boundary':h[:,0]=rng.uniform(.8,1.,24)
            if region=='extrapolated':h[:,0]=rng.uniform(2.,3.,24)
            draw=g.sample(h,256,b.keyed('predictions',ds,init,region))
            ht=torch.tensor((h-g.hm)/g.hs,dtype=torch.float32).repeat_interleave(256,dim=0)
            yt=torch.tensor(((draw-g.ym)/g.ys).reshape(-1,3),dtype=torch.float32)
            with torch.no_grad():
                logits,means,scales=g.model._conditional_parameters(ht,g.model._ordered(yt))
                weights=torch.softmax(logits,-1)
                mixture_entropy=-(weights*torch.log_softmax(logits,-1)).sum(-1).mean(-1).reshape(24,256).mean(-1).numpy()
                entropy=(-g.model.log_prob(yt,ht)).reshape(24,256).mean(-1).numpy()+np.log(g.ys).sum()
            saved=pred[pred.region==region].sort_values('history_id')
            for i,(_,old) in enumerate(saved.iterrows()):rows.append({**old.to_dict(),'mixture_label_entropy':mixture_entropy[i],'joint_predictive_entropy':entropy[i]})
    frame=pd.DataFrame(rows);frame.to_csv(R/'analysis/transformer_entropy_raw.csv',index=False)
    metrics=['mixture_label_entropy','joint_predictive_entropy','mean_squared_error','coverage90']
    datasets=frame.groupby(['law','data_seed','region'])[metrics].mean().reset_index();datasets.to_csv(R/'analysis/transformer_entropy_dataset_means.csv',index=False)
    summary=datasets.groupby(['law','region'])[metrics].mean().reset_index();summary.to_csv(R/'analysis/transformer_entropy_summary.csv',index=False)
    correlations=[]
    for law,v in datasets.groupby('law'):
        for entropy in metrics[:2]:correlations.append({'law':law,'entropy':entropy,'descriptive_spearman_with_mean_squared_error':spearmanr(v[entropy],v.mean_squared_error).statistic,'n_dataset_region_groups':len(v)})
    correlations=pd.DataFrame(correlations);correlations.to_csv(R/'analysis/transformer_entropy_correlations.csv',index=False)
    text='''# Does Transformer entropy track uncertainty here?

Exploratory supplement: no weights or primary benchmark settings changed. Mixture-label entropy is a categorical representation statistic, not continuous predictive entropy. Joint predictive entropy estimates -E log p(Y|H) from256 model samples, including output scaling. Either can be poorly associated with error in a misspecified model. Differential entropy is coordinate/unit-dependent; compare regions within each law. Means average neural seeds/history rows within each dataset first. Correlations are descriptive across9 dataset-region groups/law, not independent-row significance tests or validated OOD detection.

'''+summary.to_markdown(index=False)+'\n\n'+correlations.to_markdown(index=False)+'\n'
    (R/'ENTROPY_RESULTS.md').write_text(text)

if __name__=='__main__':main()
