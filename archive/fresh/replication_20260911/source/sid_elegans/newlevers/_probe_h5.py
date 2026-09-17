import h5py, numpy as np
f = h5py.File("/Users/vik/Developer/new_sbtg_neuro/.venv/lib/python3.14/site-packages/wormneuroatlas/data/funatlas.h5","r")
def show(g, pre=""):
    for k in g:
        item=g[k]
        if isinstance(item, h5py.Group):
            print(pre+"[G]", k); show(item, pre+"  ")
        else:
            print(pre+"[D]", k, item.shape, item.dtype)
print("=== top-level ==="); show(f)
print("=== attrs ===", dict(f.attrs))
ids = [n.decode() for n in f["neuron_ids"][:]]
print("N ids:", len(ids), "sample:", ids[:12])
dFF = f["wt"]["dFF"][:]; q = f["wt"]["q"][:]; occ = f["wt"]["occ1"][:]
print("dFF", dFF.shape, dFF.dtype, "finite %.3f"%np.isfinite(dFF).mean())
print("q", q.shape, "finite %.3f"%np.isfinite(q).mean(), " occ range", int(np.nanmin(occ)), int(np.nanmax(occ)))
sig = np.isfinite(q)&(q<0.05)
print("signif q<0.05:", int(sig.sum()), "density %.4f"%sig.mean())
# kernels: examine one strong well-sampled pair
kg = f["wt"]["kernels"]
print("kernels type:", type(kg), getattr(kg,'shape',None), getattr(kg,'dtype',None))
fin = np.argwhere(np.isfinite(dFF)&(occ>3))
vals=sorted(((abs(dFF[i,j]),i,j) for i,j in fin if i!=j), reverse=True)[:1]
for _,i,j in vals:
    print("strong post<-pre: %s <- %s dFF=%.3f occ=%d q=%.3g"%(ids[i],ids[j],dFF[i,j],occ[i,j],q[i,j]))
    try:
        k = kg[i,j]
        arr=np.array(k)
        print("  kernel raw len:", arr.size, "reshaped:", arr.reshape(-1,4) if arr.size and arr.size%4==0 else arr)
    except Exception as e:
        print("  kernel read err:", repr(e))
