"""Independent matrix orientation, source-block comparison and joint max-T inference."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

def orient_raw(a):
    # raw [..., source, horizon, target] -> [..., horizon, target, source]
    return np.moveaxis(np.asarray(a),-3,-1)

def normalize(raw,gap,strict=False):
    raw=np.asarray(raw,dtype=np.float32);gap=np.asarray(gap,dtype=np.float32)
    if strict:
        denominator=np.where(gap>=.1,gap,np.nan)
    else:denominator=np.maximum(np.abs(gap),.1)
    return raw/denominator[...,None,None]

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);result=np.empty(len(p));result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result

def binary(score,labels,mask):
    use=np.asarray(mask,bool)&np.isfinite(score);y=np.asarray(labels)[use];s=np.abs(np.asarray(score)[use]);n=len(y);pos=int(y.sum())
    out={'n_edges':n,'n_positive':pos,'prevalence':pos/n if n else np.nan,'auroc':np.nan,'auprc':np.nan}
    if 0<pos<n:out.update(auroc=float(roc_auc_score(y,s)),auprc=float(average_precision_score(y,s)))
    macro=[]
    for j in range(score.shape[1]):
        u=use[:,j];t=labels[:,j][u]
        if len(t) and t.min()<t.max():macro.append(roc_auc_score(t,np.abs(score[:,j][u])))
    out['source_macro_auroc']=float(np.mean(macro)) if macro else np.nan;out['n_macro_sources']=len(macro)
    return out

def weighted_auc_samples(score,y,source,counts):
    order=np.argsort(score,kind='stable');s=np.asarray(score)[order];y=np.asarray(y)[order];source=np.asarray(source)[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    result=[]
    for start in range(0,len(counts),128):
        w=counts[start:start+128,source];pos=np.add.reduceat(w*y,starts,axis=1);neg=np.add.reduceat(w*(1-y),starts,axis=1)
        numerator=(pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1);denom=pos.sum(1)*neg.sum(1)
        result.extend(np.divide(numerator,denom,out=np.full(len(w),np.nan),where=denom>0))
    return np.array(result)

def paired_source_bootstrap(a,b,labels,mask,reps=10000,seed=911,family_size=4):
    use=mask&np.isfinite(a)&np.isfinite(b);target,source=np.where(use);y=labels[use].astype(float)
    if not len(y) or not 0<y.sum()<len(y):return {'bootstrap_valid':0}
    n=a.shape[1];counts=np.random.default_rng(seed).multinomial(n,np.ones(n)/n,size=reps)
    aa=weighted_auc_samples(np.abs(a[use]),y,source,counts);bb=weighted_auc_samples(np.abs(b[use]),y,source,counts)
    delta=aa-bb;delta=delta[np.isfinite(delta)]
    lo,hi=np.quantile(delta,[.025,.975]);alpha=.05/family_size
    p=min(1.,2*min((1+np.sum(delta<=0))/(len(delta)+1),(1+np.sum(delta>=0))/(len(delta)+1)))
    return {'delta_auroc':float(roc_auc_score(y,np.abs(a[use]))-roc_auc_score(y,np.abs(b[use]))),'delta_ci_low':lo,'delta_ci_high':hi,'simultaneous_one_sided_lower':float(np.quantile(delta,alpha)),'bootstrap_two_sided_tail_p':p,'bootstrap_valid':len(delta)}

def exact_max_t(blocks,n,progress=None):
    """All two-sided sign orbits, common recording sign across every supplied family.

    blocks is a reiterable sequence of [recording, features] finite arrays.
    Exact conditional randomization requires joint sign symmetry. Refit uncertainty
    is not represented. Feature blocks keep memory bounded independently of family size.
    """
    orbit_count=2**(n-1);null=np.zeros(orbit_count,dtype=np.float64);observed=[]
    for bi,x in enumerate(blocks):
        x=np.asarray(x,dtype=np.float64);assert x.shape[0]==n and np.isfinite(x).all()
        ss=np.square(x).sum(0);obs_sum=x.sum(0)
        obs=np.abs(obs_sum)*np.sqrt(n-1)/np.sqrt(np.maximum(n*ss-obs_sum**2,1e-24));observed.append(obs)
        for begin in range(0,orbit_count,2048):
            ids=np.arange(begin,min(begin+2048,orbit_count),dtype=np.uint64)
            signs=np.ones((len(ids),n));signs[:,1:]=2*((ids[:,None]>>np.arange(n-1,dtype=np.uint64))&1).astype(float)-1
            sums=signs@x
            stat=np.abs(sums)*np.sqrt(n-1)/np.sqrt(np.maximum(n*ss[None]-sums*sums,1e-24))
            null[begin:begin+len(ids)]=np.maximum(null[begin:begin+len(ids)],stat.max(axis=1))
        if progress:progress(bi,len(blocks))
    null[-1]=max(null[-1],max(float(t.max()) for t in observed))
    sorted_null=np.sort(null);p=[np.maximum(1/ orbit_count,(orbit_count-np.searchsorted(sorted_null,t-1e-12*np.maximum(1,abs(t)),side='left'))/orbit_count) for t in observed]
    return observed,p,null
