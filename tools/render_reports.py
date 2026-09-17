"""Make explicitly labeled reading copies with portable links; preserve originals."""
from pathlib import Path
import re,os,hashlib,json
R=Path(__file__).resolve().parents[1];OUT=R/'docs/reports';OUT.mkdir(exist_ok=True)
chosen={'fresh_report':R/'archive/fresh/replication_20260911/REPORT_FINAL.md',
        'synthetic_summary':R/'archive/synthetic/generator_tradeoffs_20260913/SUMMARY.md',
        'repaired_path_methods':R/'archive/original/FLOW_REPAIRED_LAG_METHODS_20260828.md'}
OLD='/Users/vik/Developer/new_sbtg_neuro/'
receipts=[]
for name,p in chosen.items():
    missing=[]
    def link(m):
        target=m.group(2)
        if target.startswith(('http:','https:','#','mailto:')):return m.group(0)
        if target.startswith(OLD):
            part=target[len(OLD):]
            if part.startswith('replication_20260911/'):q=R/'archive/fresh'/part
            elif part.startswith('generator_tradeoffs_20260913/'):q=R/'archive/synthetic'/part
            else:q=R/'archive/original'/part
        else:q=p.parent/target
        if q.exists():return '['+m.group(1)+']('+os.path.relpath(q,OUT)+')'
        missing.append(target);return m.group(1)+' *(historical link unavailable; see original)*'
    text=re.sub(r'\[([^\]]*)\]\(([^)]+)\)',link,p.read_text())
    original=os.path.relpath(p,OUT)
    header=f'> Portable reading copy of [{p.name}]({original}). Scientific text is preserved; local links are relocated. The frozen original remains authoritative.\n\n'
    (OUT/f'{name}.md').write_text(header+text)
    receipts.append(dict(output=str((OUT/f'{name}.md').relative_to(R)),source=str(p.relative_to(R)),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),unavailable_links=missing))
(R/'audit/report_reading_copies.json').write_text(json.dumps(receipts,indent=2)+'\n')
