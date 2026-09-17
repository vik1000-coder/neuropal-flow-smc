"""Independent Gaussian precision solution versus the production particle sampler."""
from common import *
from compatibility_neural_benchmark.core import RepairedResponseConfig,generate_path_bank,estimate_repaired_responses
from compatibility_neural_benchmark.progressive_smc import progressive_smc_repaired_responses
class LinearAdapter:
    lag=1;device=torch.device('cpu');A=np.array([[.70,.10],[.35,.45]],dtype=float);sigma=.45
    def sample_standardized_next(self,neural_history,stimulus_history,*,seed):
        g=torch.Generator().manual_seed(seed)
        return neural_history[:,-1]@torch.tensor(self.A,dtype=torch.float32).T+self.sigma*torch.randn((len(neural_history),2),generator=g)

def exact(config,projection,lo,hi,bandwidth=.25):
    d=2;r=config.repair_frames
    T=np.eye(d*r)
    for t in range(1,r):T[d*t:d*(t+1),d*(t-1):d*t]=-LinearAdapter.A
    precision=T.T@T/LinearAdapter.sigma**2
    end=r-config.source_lag_frames;start=end-config.source_window_frames
    responses=[];gaps=[]
    for source in range(d):
        Q=precision.copy()
        for t in range(r):
            mask=np.eye(d)
            if start<=t<end:mask[source,source]=0
            Q[d*t:d*(t+1),d*t:d*(t+1)]+=config.anchor_lambda*(mask@projection@projection.T@mask)/(projection.shape[1]*r)
        c=np.zeros(d*r);c[np.arange(start,end)*d+source]=1/config.source_window_frames
        Q+=np.outer(c,c)/bandwidth**2
        delta=np.linalg.solve(Q,c*(hi-lo)/bandwidth**2)
        responses.append([np.linalg.matrix_power(LinearAdapter.A,h)@delta[-d:] for h in config.horizon_frames]);gaps.append(c@delta)
    return np.array(responses),np.array(gaps)

def run():
    torch.set_num_threads(1);rows=[];projection=np.array([[1,.25],[.15,1.2]],dtype=np.float32)
    for lag in [0,1,4,16]:
        for n in [32,128,1024]:
            cfg=RepairedResponseConfig(history_frames=1,repair_frames=lag+4,source_window_frames=4,source_lag_frames=lag,horizon_frames=(1,4),n_particles=n,min_ess=6)
            truth,gap=exact(cfg,projection,-.6,.6)
            halves,half_gap=exact(cfg,projection,-.3,.3)
            assert np.allclose(truth/gap[:,None,None],halves/half_gap[:,None,None],atol=1e-12)
            estimates=[];gaps=[]
            for repeat in range(16):
                z=progressive_smc_repaired_responses(LinearAdapter(),np.zeros((lag+12,2),dtype=np.float32),np.zeros(lag+12,dtype=np.float32),cut_time=lag+6,projection=projection,source_low=np.full(2,-.6),source_high=np.full(2,.6),source_iqr=np.ones(2),thresholds=np.zeros(2),config=cfg,seed=911000+8191*repeat,branch_factor=4,future_branch_factor=2)
                estimates.append(z['response_endpoint_mean']);gaps.append(z['diagnostic_achieved_gap'])
            estimates=np.array(estimates)
            row={'lag':lag,'n':n,'rmse':float(np.sqrt(np.mean((estimates-truth)**2))),'mean_bias_rmse':float(np.sqrt(np.mean((estimates.mean(0)-truth)**2))),'source_gap_rmse':float(np.sqrt(np.mean((np.array(gaps)-gap)**2))),'truth':truth.tolist(),'mean_estimate':estimates.mean(0).tolist()}
            rows.append(row);print('ORACLE',lag,n,row['rmse'],flush=True)
    gates=[]
    for lag in [0,1,4,16]:
        a=[v for v in rows if v['lag']==lag];gates.append({'lag':lag,'decreasing_error':a[-1]['rmse']<a[0]['rmse'],'high_particle_mean_bias_below_003':a[-1]['mean_bias_rmse']<.03})
    passed=all(g['decreasing_error'] and g['high_particle_mean_bias_below_003'] for g in gates)
    atomic_json(R/'validation/gaussian_oracle.json',{'spec_id':spec_id(),'oracle_source_sha256':sha(Path(__file__)),'completed_utc':now(),'status':'pass' if passed else 'fail','rows':rows,'gates':gates})
    if not passed:raise RuntimeError('independent Gaussian oracle failed')
if __name__=='__main__':run()
