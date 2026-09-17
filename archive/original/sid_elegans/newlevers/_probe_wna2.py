import signal, traceback, types, numpy as np
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO())); signal.alarm(180)
import wormneuroatlas as wa

class Dummy:
    "responds to any attr/call/index without network"
    def __getattr__(self, n): return Dummy()
    def __call__(self, *a, **k): return Dummy()
    def __getitem__(self, k): return Dummy()
    def __iter__(self): return iter(())
for cls in ["WormBase","Cengen","PeptideGPCR","SynapseSign","CellLineage"]:
    setattr(wa, cls, Dummy)

try:
    na = wa.NeuroAtlas(merge_bilateral=False, load_connectomes=True, verbose=False)
    ids = np.array(na.neuron_ids)
    print("N neuron_ids:", len(ids), "sample:", list(ids[:15]))
    dFF = na.get_signal_propagation_map("wt")
    q   = na.get_signal_propagation_q("wt")
    occ = na.occ1["wt"]
    print("dFF shape:", dFF.shape, "dtype:", dFF.dtype)
    print("q shape:", q.shape, "  occ shape:", occ.shape)
    print("dFF finite frac:", np.isfinite(dFF).mean().round(3), " q finite frac:", np.isfinite(q).mean().round(3))
    sig = (np.isfinite(q) & (q < 0.05))
    print("signif (q<0.05) edges:", int(sig.sum()), " of", q.size, " (density %.3f)" % (sig.mean()))
    # orientation sanity: strong known functional driver. AVA is a command interneuron.
    def ai(name):
        w = np.where(ids==name)[0]; return int(w[0]) if len(w) else -1
    # kinetics for a strong pair
    fin = np.argwhere(np.isfinite(dFF) & (occ>2))
    if len(fin):
        # pick the max-|dFF| well-sampled off-diagonal pair
        vals = [(abs(dFF[i,j]), i, j) for i,j in fin if i!=j]
        vals.sort(reverse=True)
        _,i,j = vals[0]
        print("strong pair post<-pre: %s <- %s  dFF=%.3f occ=%d q=%.3g" % (ids[i], ids[j], dFF[i,j], occ[i,j], q[i,j]))
        try:
            k = na.get_kernel(i,j,"wt")
            print("kernel type:", type(k).__name__, " has timescale:", hasattr(k,'g') or 'exp' in str(type(k)).lower())
        except Exception as e:
            print("get_kernel err:", repr(e))
except TO:
    print("TIMEOUT")
except Exception:
    traceback.print_exc()
