"""Create a byte-pinned, independent workspace; never overwrite old research."""
from pathlib import Path
import hashlib,json,shutil,sys,platform,subprocess
from datetime import datetime,timezone
R=Path(__file__).resolve().parent; P=R.parent
release=Path('/Users/vik/Downloads/SBTG-public-release copy')
def sha(p): return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha(src)!=sha(dst): raise RuntimeError(f'existing snapshot differs: {dst}')
    else: shutil.copy2(src,dst)
for pkg in ['conditional_neural_benchmark','compatibility_neural_benchmark','history_tangent_benchmark/src','sid_elegans','SBTG/pipeline','query_ood_robustness']:
    for f in (P/pkg).rglob('*.py'):
        if any(x in f.parts for x in ['__pycache__','output','results']): continue
        copy(f,R/'source'/f.relative_to(P))
for f in (P/'SBTG/data').glob('*'):
    if f.is_file() and (f.suffix=='.mat' or f.name.startswith('SI ')): copy(f,R/'source/SBTG/data'/f.name)
for sub in ['datasets/full_traces_imputed','connectome']:
    for f in (P/'SBTG/results/intermediate'/sub).glob('*'):
        if f.is_file(): copy(f,R/'source/SBTG/results/intermediate'/sub/f.name)
for f in release.rglob('*'):
    if f.is_file() and not any(x in f.parts for x in ['.git','__pycache__']): copy(f,R/'inputs/published_sbtg'/f.relative_to(release))
for name in ['EXPERIMENT_INDEX.md','FLOW_REPAIRED_LAG_METHODS_20260828.md','AUDIT_STIMULUS_PROVENANCE_20260828.md']:
    copy(P/name,R/'prior_records'/name)
for name in ['README.md','VALIDATION.md','PLAN.md']:
    copy(P/'dashboard_v2_80'/name,R/'prior_records/dashboard_v2_80'/name)
copy(P/'reports/flow_smc_code_audit_20260831/probes.py',R/'prior_records/audit_probes.py')
manifest={str(p.relative_to(R)):sha(p) for folder in ['source','inputs','prior_records'] for p in sorted((R/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
f=R/'snapshot_manifest.json'
if f.exists():
    if json.loads(f.read_text())!=manifest: raise RuntimeError('snapshot changed')
else: f.write_text(json.dumps(manifest,indent=2)+'\n')
print('SNAPSHOT',len(manifest),'files',sha(f),flush=True)
