"""Read-only byte, syntax and publication audit of every packaged source file."""
from pathlib import Path
import ast, hashlib, json, re, collections
R=Path(__file__).resolve().parents[1]
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    rows=json.loads((R/'data/archive_manifest.json').read_text())['files']
    errors=[];syntax=[];absolute=[];sensitive=[]
    secret=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')
    for r in rows:
        p=R/r['path']
        if not p.exists() or digest(p)!=r['sha256']:errors.append(r['path'])
        if p.suffix=='.py':
            s=p.read_text()
            try:ast.parse(s,filename=r['path'])
            except SyntaxError as e:syntax.append(dict(path=r['path'],line=e.lineno,error=e.msg))
            if '/Users/' in s:absolute.append(r['path'])
        if p.suffix.lower() in {'.py','.md','.json','.toml','.yaml','.yml','.txt','.log','.sh','.tex','.csv','.js','.jsx','.html'} and p.stat().st_size<10_000_000:
            if secret.search(p.read_text(errors='replace')):sensitive.append(r['path'])
    out=dict(files_checked=len(rows),bytes_checked=sum(x['bytes'] for x in rows),hash_errors=errors,syntax_errors=syntax,
             python_files=sum(x['path'].endswith('.py') for x in rows),absolute_path_source_files=absolute,
             credential_pattern_matches=sensitive,scope='Byte and syntax checks are exhaustive for the archive; scientific logic review is separately scoped in CODE_REVIEW.md.')
    (R/'audit/archive_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='absolute_path_source_files'},indent=2))
    if errors or syntax or sensitive:raise SystemExit(1)
if __name__=='__main__':main()
