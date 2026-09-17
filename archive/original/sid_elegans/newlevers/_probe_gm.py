import numpy as np, json
from sid_elegans.newlevers import common as cm
X, names, fps = cm.get_data("6w_clean_raw")
print("=== global signal: mean vs PC1 alignment, per worm ===")
for w,x in enumerate(X):
    gm = cm.global_signal(x,"mean"); gp = cm.global_signal(x,"pc1")
    Z = np.nan_to_num(x,nan=0.0); Z=Z-Z.mean(0)
    _,S,_ = np.linalg.svd(Z, full_matrices=False)
    ev = (S**2/ (S**2).sum())[0]
    print(f" worm{w}: |corr(mean,pc1)|={abs(np.corrcoef(gm,gp)[0,1]):.3f}  PC1 var-explained={ev:.2f}  frac-of-neurons-var-in-mean={np.var(gm)/np.mean(np.var(Z,0)):.2f}")
# deconv version too (biolag uses deconv)
Xd,_,_ = cm.get_data("6w_clean_deconv")
gm=cm.global_signal(Xd[0],"mean"); gp=cm.global_signal(Xd[0],"pc1")
print("deconv worm0 |corr(mean,pc1)| =", round(abs(np.corrcoef(gm,gp)[0,1]),3))
print()
print("=== ganglion map available in wormneuroatlas? ===")
import wormneuroatlas as wa, pathlib
p = pathlib.Path(wa.__file__).parent/"data"/"aconnectome_ids_ganglia.json"
d = json.load(open(p))
print("type:", type(d).__name__, "len:", len(d) if hasattr(d,'__len__') else '?')
print("sample:", json.dumps(d, indent=0)[:300] if isinstance(d,(dict,list)) else str(d)[:300])
