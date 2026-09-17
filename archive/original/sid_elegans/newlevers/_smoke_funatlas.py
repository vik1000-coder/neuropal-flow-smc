import numpy as np
from sid_elegans.combined_data import load_combined
from sid_elegans.newlevers.funatlas import load_funatlas

X, names, fps = load_combined(complete_case=True, verbose=True)
print("neuron_names N =", len(names), "e.g.", names[:8])
fa = load_funatlas(names, verbose=True)
assert fa is not None, "atlas load failed"
ts = fa["timescale"]; m = fa["both_present"]
print("dFF finite on-support:", int(np.isfinite(fa['dFF'][m]).sum()), "/", int(m.sum()))
print("timescale sec: min %.2f  p25 %.2f  median %.2f  p75 %.2f  max %.2f" % tuple(
      np.nanpercentile(ts[np.isfinite(ts)], [0,25,50,75,100])))
# orientation sanity: AVA/AVE/AVD command interneurons are strong functional HUBS (high out-degree
# as sources). Under [post,pre], a strong source = high COLUMN mass. Check a few if present.
for hub in ["AVA","AVE","AVD","RIM","AIB"]:
    if hub in names:
        i = names.index(hub)
        col = np.nan_to_num(np.abs(fa['dFF'][:, i]))   # outgoing (as source/pre)
        row = np.nan_to_num(np.abs(fa['dFF'][i, :]))   # incoming (as target/post)
        print(f"  {hub}: out(col)|dFF| sum={col.sum():.2f}  in(row)|dFF| sum={row.sum():.2f}")
print("SMOKE OK")
