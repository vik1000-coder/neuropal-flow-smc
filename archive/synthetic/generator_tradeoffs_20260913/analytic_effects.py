"""Separately labeled analytic/quadrature supplement; primary estimates unchanged."""
import os
os.nice(15)
import pickle,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import t
import benchmark as b
Generator=b.Generator
R=Path(__file__).resolve().parent

def gaussian_arm(mean,cov,c,bw):
    variance=cov[0,0]+bw*bw
    conditional=mean[1]+cov[1,0]/variance*(c-mean[0])
    log_mass=np.log(bw)-.5*np.log(variance)-.5*(c-mean[0])**2/variance
    return float(conditional),float(log_mass)

def student_arm(mean,cov,nu,c,bw):
    scale=np.sqrt(cov[0,0]*(nu-2)/nu)
    def weight(z):return np.exp(-.5*z*z)*t.pdf((c+bw*z-mean[0])/scale,nu)/scale*bw
    mass,err=quad(weight,-12,12,epsabs=1e-13,epsrel=1e-10,limit=150)
    shift,err2=quad(lambda z:(c+bw*z-mean[0])*weight(z),-12,12,epsabs=1e-13,epsrel=1e-10,limit=150)
    if mass<=0 or err/max(mass,1e-300)>1e-6:raise RuntimeError('Unresolved quadrature')
    return float(mean[1]+cov[1,0]/cov[0,0]*shift/mass),float(np.log(mass))

def teacher(h,law,centers,bw):
    h=np.asarray(h);cov=b.L@b.L.T;mean=b.true_mean(h,law)[0];values=[]
    if law=='nonlinear_heteroskedastic':
        scale=.5+1/(1+np.exp(-2*h));cov*=np.outer(scale,scale)
    for center in centers:
        if law=='heavy_tailed':v,_=student_arm(mean,cov,4,center,bw)
        elif law=='bimodal':
            p=1/(1+np.exp(-2*h[0]));offset=np.array([.65,.55,-.15]);base=mean-(2*p-1)*offset
            components=[gaussian_arm(base-offset,cov,center,bw),gaussian_arm(base+offset,cov,center,bw)]
            logw=np.log([1-p,p])+np.array([z[1] for z in components]);w=np.exp(logw-logsumexp(logw));v=w@np.array([z[0] for z in components])
        else:v,_=gaussian_arm(mean,cov,center,bw)
        values.append(v)
    return float(values[1]-values[0])

def learned(g,h,centers,bw):
    h=(np.atleast_2d(h)-g.hm)/g.hs;nu=None
    if g.family=='ridge':
        reg,chol=g.model;mean=reg.predict(h)[0];cov=chol@chol.T
    elif g.family=='gp_rbf':
        mean,std=g.model.predict(h,return_std=True);mean=mean[0];cov=np.diag(np.broadcast_to(std, (1,3))[0]**2)
    else:
        with torch.no_grad():params=g.model.parameters_at(torch.tensor(h,dtype=torch.float32))
        mean=params[0][0].numpy();scale=np.exp(params[1][0].numpy());cov=np.diag(scale**2)
        if len(params)==3:
            factor=params[2][0].numpy();cov+=factor@factor.T
        if g.family=='student_rank2':nu=float(g.model.degrees_of_freedom.detach())
    mean=mean*g.ys+g.ym;cov=cov*np.outer(g.ys,g.ys)
    vals=[student_arm(mean,cov,nu,c,bw)[0] if nu else gaussian_arm(mean,cov,c,bw)[0] for c in centers]
    return float(vals[1]-vals[0])

def main():
    rows=[];checks=[];analytic_families=['ridge','gp_rbf','gaussian_diag','gaussian_rank2','student_rank2']
    for receipt in sorted((R/'runs').glob('*/receipt.json')):
        run=receipt.parent;original=pd.read_csv(run/'effects.csv');record=json.loads(receipt.read_text())
        g=None
        if original.family.iloc[0] in analytic_families:
            assert b.sha(run/'model.pkl')==record['files']['model.pkl']
            with open(run/'model.pkl','rb') as f:g=pickle.load(f)
        for _,row in original.iterrows():
            query=json.loads((run/f'reference_{row.region}_{row.event}.json').read_text());centers=query['centers'];bw=query['bandwidth'];h=b.CENTERS[row.region]
            truth=teacher(h,row.law,centers,bw)
            estimate=learned(g,h,centers,bw) if g else row.model_estimate
            rows.append({**row.to_dict(),'supplement_truth':truth,'supplement_model_effect':estimate,'supplement_error':estimate-truth,'supplement_model_qualified':True if g else row.model_qualified,'model_reference_method':'analytic_or_scalar_quadrature' if g else 'original_independent_MC','original_oracle_discrepancy':row.oracle_estimate-truth})
            if row.oracle_qualified:checks.append(abs(row.oracle_estimate-truth))
    frame=pd.DataFrame(rows);frame.to_csv(R/'analysis/analytic_effects_raw.csv',index=False)
    frame['absolute_error']=frame.supplement_error.abs()
    keys=['law','data_seed','family','region','event']
    datasets=frame.groupby(keys).agg(absolute_error=('absolute_error','mean'),qualified=('supplement_model_qualified','all')).reset_index();datasets.to_csv(R/'analysis/analytic_effects_dataset_means.csv',index=False)
    paired=datasets.merge(datasets[datasets.family=='flow'].drop(columns='family'),on=['law','data_seed','region','event'],suffixes=('','_flow'))
    paired['both_qualified']=paired.qualified & paired.qualified_flow
    paired['delta_vs_flow']=(paired.absolute_error-paired.absolute_error_flow).where(paired.both_qualified)
    paired.to_csv(R/'analysis/analytic_effects_paired.csv',index=False)
    leaders=[]
    for (law,region,event),group in datasets.groupby(['law','region','event']):
        board=group.groupby('family').agg(error=('absolute_error','mean'),qualified=('qualified','sum'))
        valid=board[board.qualified==3].sort_values('error')
        leaders.append({'law':law,'region':region,'event':event,'qualified_families':len(valid),'lowest_mean_error_family':valid.index[0],'error':valid.error.iloc[0],'flow_error':board.loc['flow','error'],'flow_qualified_datasets':board.loc['flow','qualified']})
    leaders=pd.DataFrame(leaders);leaders.to_csv(R/'analysis/analytic_effects_point_leaders.csv',index=False)
    text='''# Conditional effects with analytic / quadrature references

Exploratory supplement documented in analysis_amendment_002.md. It changes neither trained models nor queries. True effects use closed-form Gaussian/mixture conditioning or one-dimensional Student-t quadrature. Gaussian/Student-t learned effects also use analytic/quadrature expectations; MDN, Transformer, flow and EDM retain their original finite-Monte-Carlo reference and precision gate. This unequal numerical access is explicit: excluded models cannot be declared worse from missing reference precision. Primary Monte Carlo results remain intact.

Below are lowest mean errors among families with qualified learned references for all3 datasets and both neural seeds. Means are descriptive across this fixed design, not proof of optimality. Flow errors are displayed even when its qualified-dataset count is below3; do not treat those unresolved values as reliable comparisons. The full paired table masks differences whenever either family lacks a qualified reference.

'''+leaders.to_markdown(index=False)+'\n\nThe largest absolute discrepancy between a *qualified* original oracle MC reference and the supplementary truth was '+str(max(checks))+' output units. All original discrepancies and gates are retained in analytic_effects_raw.csv.\n'
    (R/'ANALYTIC_EFFECTS.md').write_text(text)

if __name__=='__main__':main()
