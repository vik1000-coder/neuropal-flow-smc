"""Fill the scaffolded notebook from the readable percent-format tutorial source."""
from pathlib import Path
import nbformat
R=Path(__file__).resolve().parents[1];source=R/'notebooks/four_neuron_tutorial.py';dest=source.with_suffix('.ipynb')
nb=nbformat.read(dest,as_version=4);cells=[]
for part in source.read_text().split('# %%')[1:]:
    first,*body=part.splitlines()
    if first.strip()=='[markdown]':
        text='\n'.join(line[2:] if line.startswith('# ') else '' if line=='#' else line for line in body).strip()
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:cells.append(nbformat.v4.new_code_cell('\n'.join(body).strip()))
nb.cells=cells;nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nbformat.write(nb,dest);print('Wrote',len(cells),'cells')
