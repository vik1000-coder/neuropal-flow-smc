"""Run scientific tests in isolated subprocesses, avoiding same-named module collisions."""
from pathlib import Path
import subprocess,sys,os,json,time
R=Path(__file__).resolve().parents[1]
CASES=[('original','archive/original',['conditional_neural_benchmark/tests','compatibility_neural_benchmark/tests','query_ood_robustness/tests']),
       ('replication','archive/fresh/replication_20260911',['test_replication.py']),
       ('synthetic','archive/synthetic/generator_tradeoffs_20260913',['test_benchmark.py','test_analytic_supplement.py'])]
def main():
    records=[]
    for name,where,tests in CASES:
        cwd=R/where;env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(str(p) for p in [cwd,cwd/'history_tangent_benchmark/src',cwd/'sid_neuromod/src',cwd/'source'])
        cmd=[sys.executable,'-m','pytest','--import-mode=importlib','-q','--disable-warnings',*tests]
        start=time.monotonic();p=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True)
        (R/'audit'/f'packaged_{name}_tests.log').write_text(p.stdout+p.stderr)
        records.append(dict(suite=name,returncode=p.returncode,seconds=time.monotonic()-start,command=cmd[1:]))
        print(name,p.returncode,p.stdout[-400:],flush=True)
    (R/'audit/test_runs.json').write_text(json.dumps(records,indent=2)+'\n')
    if any(x['returncode'] for x in records):raise SystemExit(1)
if __name__=='__main__':main()
