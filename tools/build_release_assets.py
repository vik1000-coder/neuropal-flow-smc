"""Create size-bounded, independently extractable release tarballs and checksums."""
from pathlib import Path
import json,tarfile,hashlib
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'data/archive_manifest.json';d=json.loads(path.read_text())
for r in d['files']:
    parts=Path(r['path']).parts
    payload=any(k in parts for k in ('runs','results','outputs','output','checkpoints'))
    if payload or r['bytes']>2_000_000 or Path(r['path']).suffix in {'.gz','.pt','.pth','.npz','.mat','.pkl','.pickle','.npy','.h5','.hdf5'}:
        r['storage']='release'
path.write_text(json.dumps(d,indent=2)+'\n')
(ROOT/'.gitignore').write_text('__pycache__/\n.pytest_cache/\n.venv/\n.DS_Store\n*.pyc\n_downloads/\n_release/\n'+''.join('/'+r['path']+'\n' for r in d['files'] if r['storage']=='release'))
# Archive members retain their paths relative to the repository root.
chunks=[];current=[];size=0
for r in d['files']:
    if r['storage']!='release':continue
    if current and size+r['bytes']>900_000_000:
        chunks.append(current);current=[];size=0
    current.append(r);size+=r['bytes']
if current:chunks.append(current)
out=ROOT/'_release';out.mkdir(exist_ok=True);assets=[]
for i,rows in enumerate(chunks):
    name=f'archive-part-{i+1:02d}.tar.gz';p=out/name
    with tarfile.open(p,'w:gz',compresslevel=1) as t:
        for r in rows:t.add(ROOT/r['path'],arcname=r['path'],recursive=False)
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    assets.append(dict(name=name,bytes=p.stat().st_size,sha256=sha,files=[r['path'] for r in rows]))
    print(name,p.stat().st_size,flush=True)
(ROOT/'data/release_assets.json').write_text(json.dumps({'repository':'vik1000-coder/neuropal-flow-smc','tag':'v1.0-data','assets':assets},indent=2)+'\n')
print('Git archive bytes:',sum(r['bytes'] for r in d['files'] if r['storage']=='git'),flush=True)
