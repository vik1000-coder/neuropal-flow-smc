"""Maintainer-only release upload; validates local hashes before transmission."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,json,hashlib,time
R=Path(__file__).resolve().parents[1]
def upload(a):
    p=R/'_release'/a['name']
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==a['sha256']
    start=time.monotonic()
    proc=subprocess.run(['gh','release','upload','v1.0-data',str(p),'--repo','vik1000-coder/neuropal-flow-smc'],capture_output=True,text=True)
    return dict(name=a['name'],returncode=proc.returncode,seconds=time.monotonic()-start,output=proc.stdout+proc.stderr,bytes=a['bytes'])
def main():
    assets=json.loads((R/'data/release_assets.json').read_text())['assets'];results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(upload,a) for a in assets]):
            r=f.result();results.append(r);print(r,flush=True)
            (R/'audit/upload_status.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(x['returncode'] for x in results):raise SystemExit(1)
if __name__=='__main__':main()
