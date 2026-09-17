"""Collect the authorized scientific record without altering its frozen contents.

Run from this repository. Large files use local hard links to avoid duplicating
many GB while staging; release archives and Git blobs are independent byte copies.
No consumer may modify archive files in place.
"""
from pathlib import Path
import shutil, os, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent
SKIP={'.git','__pycache__','.pytest_cache','.DS_Store','node_modules','mpl_cache','.venv','.causal_venv'}
original=['conditional_neural_benchmark','compatibility_neural_benchmark','query_ood_robustness',
          'history_tangent_benchmark','SBTG','sid_elegans','sid_neuromod','neuromod_benchmark',
          'distributional_sid','empirical_sid','analysis','methods','reports','dashboard_v2','dashboard_v2_80',
          'adjudication_configs','docs','pipeline','paper']
original += [p.name for p in SOURCE.glob('*.md') if not p.name.startswith('DANDI')]
original += ['EXPERIMENT_REGISTRY.csv']
groups=[(p,'archive/original/'+p) for p in original]
groups += [('replication_20260911','archive/fresh/replication_20260911'),
           ('generator_tradeoffs_20260913','archive/synthetic/generator_tradeoffs_20260913')]
rows=[]
for source,dest in groups:
    base=SOURCE/source
    if not base.exists(): continue
    files=[base] if base.is_file() else sorted(base.rglob('*'))
    for p in files:
        if not p.is_file() or any(x in SKIP for x in p.relative_to(SOURCE).parts): continue
        if p.suffix in {'.pyc','.lock'} or p.name in {'pipeline.lock','worker.lock'}: continue
        q=ROOT/dest if base.is_file() else ROOT/dest/p.relative_to(base)
        q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists():
            if p.stat().st_size>5_000_000: os.link(p,q)
            else: shutil.copy2(p,q)
        with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        rel=str(q.relative_to(ROOT)); size=p.stat().st_size
        # Keep all frozen run artifacts in the release. Git retains code, reports,
        # cohort definitions, receipts, and small summary tables for easy inspection.
        bulk=size>5_000_000 or (q.suffix in {'.pt','.pth','.npz','.mat','.pkl','.pickle','.npy','.h5','.hdf5'})
        rows.append(dict(path=rel,original_path=str(p.relative_to(SOURCE)),bytes=size,sha256=digest,storage='release' if bulk else 'git'))
(ROOT/'data/archive_manifest.json').write_text(json.dumps({'schema':1,'files':rows},indent=2)+'\n')
(ROOT/'.gitignore').write_text('__pycache__/\n.pytest_cache/\n.venv/\n.DS_Store\n*.pyc\n_downloads/\n_release/\n'+''.join('/'+r['path']+'\n' for r in rows if r['storage']=='release'))
print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'git_bytes':sum(r['bytes'] for r in rows if r['storage']=='git')},indent=2))
