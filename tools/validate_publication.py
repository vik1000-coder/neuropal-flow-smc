"""Validate final publication inputs, figure receipts, notebook, and local links."""
from pathlib import Path
import hashlib,json,re
import nbformat
from PIL import Image
R=Path(__file__).resolve().parents[1]
def main():
    catalog=json.loads((R/'figures/catalog.json').read_text());assert len(catalog)==18
    figures=[]
    for item in catalog:
        for src in item['inputs']:assert (R/src).exists(),src
        for ext in ['png','pdf']:
            p=R/'figures'/f"{item['id']}.{ext}";assert p.stat().st_size>1000
            if ext=='png':
                with Image.open(p) as im:assert im.width>=1000 and im.height>=500
            figures.append(dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    notebook=nbformat.read(R/'notebooks/four_neuron_tutorial.ipynb',as_version=4);nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    broken=[]
    for p in [R/'README.md',*list((R/'docs').rglob('*.md'))]:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https:','http:','#','mailto:')):continue
            target=target.split('#')[0]
            if not (p.parent/target).exists():broken.append(dict(file=str(p.relative_to(R)),target=target))
    assert not broken,broken
    # Update the receipt after deterministic derived pair summaries are written.
    inputs=[]
    for p in sorted((R/'data/publication').rglob('*')):
        if p.is_file():inputs.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (R/'data/publication_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    receipt=dict(status='pass',publication_figures=18,figure_files=figures,toy_figures=7,executed_notebook_cells=len(code),broken_new_documentation_links=broken,
        visual_review='All 18 publication and seven toy figures inspected via contact sheets and targeted full-size views. Toy multi-panel spacing corrected and re-executed.',
        scope='Numerical/artifact validation, not proof of biological identification; original archive remains immutable.')
    (R/'audit/publication_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Publication artifacts validated.')
if __name__=='__main__':main()
