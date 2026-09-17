"""Restore selected immutable release assets with SHA-256 and safe extraction.

Examples: python tools/fetch_data.py --group fresh
          python tools/fetch_data.py --group all --verify
Existing correct files are reused. Downloads are discarded after extraction.
"""
from pathlib import Path
import argparse,hashlib,json,tarfile,urllib.request
R=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--group',choices=['all','fresh','original','synthetic'],default='all');ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    manifest=json.loads((R/'data/archive_manifest.json').read_text());release=json.loads((R/'data/release_assets.json').read_text())
    wanted={x['path']:x for x in manifest['files'] if a.group=='all' or x['path'].startswith('archive/'+a.group+'/')}
    missing={k for k,v in wanted.items() if not (R/k).exists() or (a.verify and sha(R/k)!=v['sha256'])}
    downloads=R/'_downloads';downloads.mkdir(exist_ok=True)
    for asset in release['assets']:
        needed=missing.intersection(asset['files'])
        if not needed:continue
        p=downloads/asset['name'];url=f"https://github.com/{release['repository']}/releases/download/{release['tag']}/{asset['name']}"
        if not p.exists() or sha(p)!=asset['sha256']:
            print('Downloading',asset['name'],flush=True);temp=p.with_suffix('.partial');urllib.request.urlretrieve(url,temp)
            if sha(temp)!=asset['sha256']:raise RuntimeError('Download checksum mismatch: '+str(temp))
            temp.replace(p)
        with tarfile.open(p,'r:gz') as tar:
            for member in tar:
                if member.name not in needed:continue
                dest=R/member.name
                if not dest.resolve().is_relative_to(R) or not member.isfile():raise RuntimeError('Unsafe archive member')
                dest.parent.mkdir(parents=True,exist_ok=True)
                # Atomic replacement avoids modifying any existing hard-linked source.
                temp=dest.with_suffix(dest.suffix+'.restore-tmp')
                with tar.extractfile(member) as source,temp.open('wb') as target:
                    while block:=source.read(1024*1024):target.write(block)
                if sha(temp)!=wanted[member.name]['sha256']:raise RuntimeError('Member checksum mismatch')
                temp.replace(dest);missing.remove(member.name)
        p.unlink()
    if missing:raise RuntimeError('Missing Git-tracked or release files: '+repr(sorted(missing)[:10]))
    print('Selected archive restored and downloaded members verified.')
if __name__=='__main__':main()
