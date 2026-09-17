"""Execute the tutorial with the active Python environment, then save its outputs."""
from pathlib import Path
import sys,tempfile,subprocess,json
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
R=Path(__file__).resolve().parents[1]
def main():
    p=R/'notebooks/four_neuron_tutorial.ipynb'
    # A temporary explicit kernelspec prevents accidentally selecting another Python.
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);k=root/'kernels'/'publication';k.mkdir(parents=True)
        (k/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Publication environment','language':'python'}))
        from jupyter_client.kernelspec import KernelSpecManager
        km=KernelManager(kernel_name='publication',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(root/'kernels')]))
        nb=nbformat.read(p,as_version=4)
        NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(p.parent)}}).execute()
        nbformat.write(nb,p)
    print('Executed and saved all tutorial cells.')
if __name__=='__main__':main()
