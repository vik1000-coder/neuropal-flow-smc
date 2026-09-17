from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).resolve().parent/'mpl_cache')
import copy, datetime, fcntl, hashlib, json, math, pickle, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from history_tangent_benchmark.models import build_model
from elliptical import ConditionalElliptical
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

LAWS=['linear_gaussian','nonlinear_heteroskedastic','bimodal','heavy_tailed']
FAMILIES=['ridge','gp_rbf','gaussian_diag','gaussian_rank2','student_rank2','mdn4','transformer_mdn4','flow','edm']
SEEDS=[101,202,303]
INITS=[1701,2903]
CENTERS={'central':np.array([0.,0.,0.]),'boundary':np.array([.9,0.,0.]),'extrapolated':np.array([2.5,0.,0.])}
A=np.array([[.8,.3,0.],[.5,-.4,.2],[-.2,.1,.7]])
L=np.array([[.25,0,0],[.18,.22,0],[.05,-.08,.2]])

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');tmp.replace(path)
def status(**kwargs): atomic(ROOT/'status.json',{'utc':datetime.datetime.now(datetime.UTC).isoformat(),'pid':os.getpid(),**kwargs})
def keyed(*parts): return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4],'little')
def true_mean(h,law):
    h=np.atleast_2d(h);m=h@A.T
    if law!='linear_gaussian':
        m=m+np.column_stack([.4*np.sin(2*h[:,0]),.35*h[:,0]*h[:,1],.3*np.tanh(h[:,2])])
    if law=='bimodal':
        p=1/(1+np.exp(-2*h[:,0]));m=m+(2*p-1)[:,None]*np.array([.65,.55,-.15])
    return m
def true_derivative(h,law):
    h=np.asarray(h);d=A[:,0].copy()
    if law!='linear_gaussian':d+=np.array([.8*np.cos(2*h[0]),.35*h[1],0])
    if law=='bimodal':
        p=1/(1+np.exp(-2*h[0]));d+=4*p*(1-p)*np.array([.65,.55,-.15])
    return d
def oracle(h,n,seed,law):
    h=np.atleast_2d(h);rng=np.random.default_rng(seed)
    e=rng.normal(size=(len(h),n,3))@L.T
    if law=='nonlinear_heteroskedastic':e*= (.5+1/(1+np.exp(-2*h)))[:,None,:]
    if law=='heavy_tailed':e*=np.sqrt(2/rng.chisquare(4,size=(len(h),n,1)))
    m=true_mean(h,law)
    if law=='bimodal':
        p=1/(1+np.exp(-2*h[:,0]));v=np.array([.65,.55,-.15])
        m=m-(2*p-1)[:,None]*v
        e+=np.where(rng.random((len(h),n,1))<p[:,None,None],1.,-1.)*v
    return m[:,None,:]+e
def energy(x,y):
    # Unbiased distinct-pair energy score, averaged over independent outcomes.
    n=len(x)
    return float(cdist(x,y).mean()-.5*cdist(x,x).sum()/(n*(n-1)))

class Generator:
    def __init__(self,family,model,hm,hs,ym,ys): self.family,self.model,self.hm,self.hs,self.ym,self.ys=family,model,hm,hs,ym,ys
    def sample(self,h,n,seed):
        h=np.atleast_2d(h);z=(h-self.hm)/self.hs;rng=np.random.default_rng(seed)
        if self.family=='ridge':
            reg,chol=self.model;out=reg.predict(z)[:,None,:]+rng.normal(size=(len(h),n,3))@chol.T
        elif self.family=='gp_rbf':
            mean,std=self.model.predict(z,return_std=True)
            if std.ndim==1:std=np.broadcast_to(std[:,None],mean.shape)
            out=mean[:,None,:]+std[:,None,:]*rng.normal(size=(len(h),n,3))
        else:
            self.model.eval()
            with torch.no_grad():out=self.model.sample(torch.tensor(z,dtype=torch.float32),n,int(seed)).numpy()
        out=out*self.ys+self.ym
        if not np.isfinite(out).all():raise FloatingPointError('Nonfinite generated sample')
        return out

def neural(family):
    if family in ['gaussian_rank2','student_rank2']:
        return ConditionalElliptical(3,3,hidden=32,layers=2,rank=2,student=family=='student_rank2')
    names={'gaussian_diag':'heteroscedastic_gaussian','mdn4':'autoregressive_mdn','transformer_mdn4':'autoregressive_transformer','flow':'conditional_flow_matching','edm':'conditional_edm_diffusion'}
    params={'hidden':32,'layers':2}
    if family=='mdn4':params['components']=4
    if family=='transformer_mdn4':params={'d_model':32,'nhead':4,'transformer_layers':2,'feedforward':64,'components':4}
    if family in ['flow','edm']:params['sample_steps']=24
    return build_model(names[family],3,3,params)

def fit(family,h,y,vh,vy,seed,directory):
    hm,hs=h.mean(0),h.std(0);ym,ys=y.mean(0),y.std(0)
    x=(h-hm)/hs;t=(y-ym)/ys
    start=time.perf_counter();traces=[];best=float('inf');best_epoch=0;chosen=None
    def validation(g):
        samples=g.sample(vh[:128],64,91827)
        return float(np.mean([energy(a,b[None]) for a,b in zip(samples,vy[:128])]))
    if family in ['ridge','gp_rbf']:
        candidates=[(.1,), (1.,), (10.,)] if family=='ridge' else [(l,n) for l in [.5,1.,2.] for n in [.05,.2,.5]]
        for candidate in candidates:
            if family=='ridge':
                reg=Ridge(alpha=candidate[0]).fit(x,t);cov=LedoitWolf().fit(t-reg.predict(x)).covariance_+np.eye(3)*1e-6;model=(reg,np.linalg.cholesky(cov))
            else:
                kernel=ConstantKernel(1.,constant_value_bounds='fixed')*RBF(candidate[0],length_scale_bounds='fixed')+WhiteKernel(candidate[1],noise_level_bounds='fixed')
                model=GaussianProcessRegressor(kernel=kernel,optimizer=None,alpha=1e-7,normalize_y=False).fit(x,t)
            g=Generator(family,model,hm,hs,ym,ys);score=validation(g);traces.append({'candidate':candidate,'validation_energy':score})
            if score<best:best=score;chosen=g
        info={'convergence':'finite_grid_complete','parameters':None}
    else:
        torch.manual_seed(seed);model=neural(family);g=Generator(family,model,hm,hs,ym,ys)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor(t,dtype=torch.float32)
        saved=None;limit=200;extended=False;epoch=0
        while epoch<limit:
            epoch+=1;model.train();order=torch.randperm(len(x));losses=[]
            for idx in order.split(128):
                opt.zero_grad();loss=model.native_loss(xt[idx],yt[idx])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            if epoch%10==0:
                score=validation(g);traces.append({'epoch':epoch,'loss':np.mean(losses),'validation_energy':score})
                if score<best:best=score;best_epoch=epoch;saved=copy.deepcopy(model.state_dict())
                atomic(directory/'training_progress.json',{'epoch':epoch,'best_epoch':best_epoch,'best_energy':best,'elapsed_s':time.perf_counter()-start})
                if epoch==200 and best_epoch>=160:limit=400;extended=True
                if epoch-best_epoch>=40:break
        model.load_state_dict(saved);model.eval();chosen=g
        info={'convergence':'boundary_best_unresolved' if epoch==limit and best_epoch>=limit-40 else 'validation_early_stop','best_epoch':best_epoch,'last_epoch':epoch,'extended':extended,'parameters':model.parameter_count}
    info.update(training_seconds=time.perf_counter()-start,validation_energy=best,traces=traces)
    with open(directory/'model.pkl','wb') as f:pickle.dump(chosen,f)
    atomic(directory/'fit.json',info)
    return chosen,info

def weighted(draw,centers,bw):
    estimates=[];esses=[];probs=[]
    for center in centers:
        logw=-.5*((draw[:,0]-center)/bw)**2;m=logw.max();w=np.exp(logw-m);p=w.mean()*np.exp(m);w/=w.sum()
        estimates.append(w@draw[:,1]);esses.append(1/(w@w));probs.append(p)
    return float(estimates[1]-estimates[0]),min(esses),min(probs)
def reference(sample,centers,bw,seed,oracle_reference=False):
    records=[];levels=[65536,262144] if oracle_reference else [4096,16384,65536]
    for n in levels:
        values=[];ess=[];prob=[]
        for rep in range(3):
            draw=sample(n,keyed(seed,n,rep));a,b,c=weighted(draw,centers,bw);values.append(a);ess.append(b);prob.append(c)
        se=float(np.std(values,ddof=1)/np.sqrt(3));qualified=min(ess)>=200 and se<=.025
        records.append({'n':n,'estimate':np.mean(values),'se':se,'min_ess':min(ess),'min_soft_probability':min(prob),'qualified':qualified,'repeat_estimates':values})
        if qualified:break
    return records[-1],records

def evaluate(g,law,data_seed,init,htrain,ytrain,directory):
    base={'law':law,'data_seed':data_seed,'init':init,'family':g.family};rows=[];rng=np.random.default_rng(keyed('evaluation',data_seed))
    for region in CENTERS:
        h=rng.uniform(-.5,.5,(24,3))
        if region=='boundary':h[:,0]=rng.uniform(.8,1.,24)
        if region=='extrapolated':h[:,0]=rng.uniform(2.,3.,24)
        truths=oracle(h,256,keyed('outcomes',data_seed,region),law)
        started=time.perf_counter();draw=g.sample(h,256,keyed('predictions',data_seed,init,region));seconds=time.perf_counter()-started
        for i,(x,y) in enumerate(zip(draw,truths)):
            lower,upper=np.quantile(x,[.05,.95],axis=0);threshold=np.quantile(y[:,0],.95);p=float((x[:,0]>threshold).mean());event=y[:,0]>threshold
            rows.append({**base,'region':region,'history_id':i,'energy':energy(x,y),'mean_squared_error':np.mean((x.mean(0)-true_mean(h[i],law)[0])**2),'coverage90':np.mean((y>=lower)&(y<=upper)),'width90':np.mean(upper-lower),'tail_brier':np.mean((p-event)**2),'tail_probability':p,'tail_probability_abs_error':abs(p-.05),'sampling_seconds_per_history':seconds/len(h)})
    pd.DataFrame(rows).to_csv(directory/'predictive.csv',index=False)
    rows=[];small=[];derivatives=[];solver=[];mu=ytrain[:,0].mean();sd=ytrain[:,0].std();bw=.4*sd
    for region,h in CENTERS.items():
        for event,centers in [('common',mu+sd*np.array([-.5,.5])),('rare',mu+sd*np.array([2.5,3.5]))]:
            cache=ROOT/'oracle_references'/f'{law}_{data_seed}_{region}_{event}.json'
            if cache.exists():truth_record=json.loads(cache.read_text());truth=truth_record['final']
            else:
                truth,attempts=reference(lambda n,s:oracle(h,n,s,law)[0],centers,bw,keyed('truth',law,data_seed,region,event),True)
                atomic(cache,{'final':truth,'attempts':attempts,'centers':centers.tolist(),'bandwidth':bw})
            start=time.perf_counter();ref,attempts=reference(lambda n,s:g.sample(h,n,s)[0],centers,bw,keyed('reference',law,data_seed,init,g.family,region,event))
            rows.append({**base,'region':region,'event':event,**{f'model_{k}':v for k,v in ref.items() if k!='repeat_estimates'},**{f'oracle_{k}':v for k,v in truth.items() if k!='repeat_estimates'},'error':ref['estimate']-truth['estimate'],'qualified':ref['qualified'] and truth['qualified'],'reference_seconds':time.perf_counter()-start})
            atomic(directory/f'reference_{region}_{event}.json',{'final':ref,'attempts':attempts,'oracle':truth,'centers':centers.tolist(),'bandwidth':bw})
            for n in [128,512]:
                for rep in range(3):
                    estimate,ess,prob=weighted(g.sample(h,n,keyed('small',data_seed,init,region,event,n,rep))[0],centers,bw)
                    small.append({**base,'region':region,'event':event,'n':n,'rep':rep,'estimate':estimate,'particle_error':estimate-ref['estimate'],'total_error':estimate-truth['estimate'],'ess':ess,'reference_qualified':ref['qualified'],'oracle_qualified':truth['qualified']})
        for step in [.2,.1,.05]:
            estimated=[]
            for rep in range(3):
                offset=np.array([step,0.,0.]);seed=keyed('derivative',data_seed,init,region,rep)
                plus=g.sample(h+offset,2048,seed)[0].mean(0);minus=g.sample(h-offset,2048,seed)[0].mean(0)
                estimated.append((plus-minus)/(2*step))
            estimated=np.asarray(estimated);truth=true_derivative(h,law)
            for j in range(3):derivatives.append({**base,'region':region,'step':step,'target':j,'estimate':estimated[:,j].mean(),'mc_se':estimated[:,j].std(ddof=1)/np.sqrt(3),'truth':truth[j],'error':estimated[:,j].mean()-truth[j]})
        if g.family in ['flow','edm']:
            seed=keyed('solver',data_seed,init,region);start=time.perf_counter();a=g.sample(h,2048,seed)[0];ta=time.perf_counter()-start
            g.model.sample_steps=48;start=time.perf_counter();b=g.sample(h,2048,seed)[0];tb=time.perf_counter()-start;g.model.sample_steps=24
            solver.append({**base,'region':region,'mean_change_rmse':np.sqrt(np.mean((a.mean(0)-b.mean(0))**2)),'quantile_change_rmse':np.sqrt(np.mean((np.quantile(a,[.05,.5,.95],axis=0)-np.quantile(b,[.05,.5,.95],axis=0))**2)),'seconds24':ta,'seconds48':tb})
    for name,data in [('effects',rows),('sampling',small),('derivatives',derivatives),('solver',solver)]:pd.DataFrame(data).to_csv(directory/f'{name}.csv',index=False)

def main():
    lock=open(ROOT/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:os.nice(15)
    except OSError:pass
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
    completed=0;tasks=[]
    for law in LAWS:
        for ds in SEEDS:
            for family in FAMILIES:
                for init in ([0] if family in ['ridge','gp_rbf'] else INITS):tasks.append((law,ds,family,init))
    for law,ds,family,init in tasks:
        directory=ROOT/'runs'/f'{law}_{ds}_{family}_{init}';directory.mkdir(parents=True,exist_ok=True)
        receipt=directory/'receipt.json'
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if record['manifest_sha256']!=sha(ROOT/'manifest.json'):raise RuntimeError('Manifest mismatch')
            for name,digest in record['files'].items():
                if sha(directory/name)!=digest:raise RuntimeError('Receipt mismatch '+str(directory/name))
            completed+=1;continue
        status(state='running',stage='fit_or_evaluate',task=directory.name,completed=completed,total=len(tasks))
        rng=np.random.default_rng(ds);h=rng.uniform(-1,1,(1024,3));vh=rng.uniform(-1,1,(256,3));y=oracle(h,1,keyed('train',ds),law)[:,0];vy=oracle(vh,1,keyed('validation',ds),law)[:,0]
        np.savez_compressed(directory/'data.npz',h=h,y=y,vh=vh,vy=vy)
        start=time.perf_counter()
        if (directory/'model.pkl').exists() and (directory/'fit.json').exists():
            with open(directory/'model.pkl','rb') as f:g=pickle.load(f)
        else:g,_=fit(family,h,y,vh,vy,keyed(ds,init),directory)
        status(state='running',stage='evaluate',task=directory.name,completed=completed,total=len(tasks))
        evaluate(g,law,ds,init,h,y,directory)
        files={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='receipt.json'}
        atomic(receipt,{'manifest_sha256':sha(ROOT/'manifest.json'),'files':files,'elapsed_seconds_this_attempt':time.perf_counter()-start})
        completed+=1;print(f'COMPLETE {completed}/{len(tasks)} {directory.name}',flush=True)
    status(state='analysis_required',completed=completed,total=len(tasks))

if __name__=='__main__':
    try:main()
    except Exception:
        status(state='failed',traceback=traceback.format_exc());raise
