import signal, traceback
class TO(Exception): pass
def _h(s,f): raise TO()
signal.signal(signal.SIGALRM,_h); signal.alarm(200)
import numpy as np
try:
    import wormneuroatlas as wa
    print("WNA attrs:", [a for a in dir(wa) if not a.startswith('_')])
    na = wa.NeuroAtlas()
    meths = [m for m in dir(na) if not m.startswith('_')]
    print("NeuroAtlas members:", meths)
    try:
        labels = na.neuron_ids
        print("n_neuron_ids:", len(labels), "sample:", list(labels[:12]))
    except Exception as e:
        print("neuron_ids err:", repr(e))
    for meth in ["get_signal_propagation_map","get_functional_connectivity","get_anatomical_connectome",
                 "get_dFF","get_kernels_map","get_signal_propagation_kernels"]:
        print(("HAS " if hasattr(na, meth) else "no  ")+meth)
except TO:
    print("TIMEOUT during WNA probe")
except Exception:
    traceback.print_exc()
