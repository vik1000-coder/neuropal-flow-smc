# Reference snapshot

This directory contains the immutable inputs needed to rerun the recording-row
bootstrap and phase-duration sensitivity analyses. The snapshot is deliberately
separate from the main paper artifacts: it preserves the exact production model,
fixed hyperparameters, prepared traces, and phase-analysis anchors used by those
analyses.

The original production archive had SHA-256 digest
`341771375f8bad095baf7cfe0836dc76e1eaf56f3d92f903ad958848079730cb`.
It stored hyperparameter dictionaries as Python object arrays, so it is not
distributed. Its scientific content is preserved as the safe pair
`production/sbtg_lag_matrices.npz` and `production/hyperparameters.json`; the NPZ
contains only numeric, Boolean, and Unicode arrays and loads with
`allow_pickle=False`.

The prepared dataset contains only the three files required by the experiment
loaders. `traces.npz` represents variable-length recordings with concatenated
finite values, an explicit missing-value mask, and integer offsets; it loads
with `allow_pickle=False`. Phase directories use neutral labels:

- `baseline`: no displayed stimulus
- `steady`: displayed stimulus before activation
- `on`: activation interval
- `off`: post-activation interval

All paths recorded by the experiment package are relative to the repository.
`release_manifest.csv` and `checksums.sha256` record the distributed-file
digests. The source-archive digests above and in `experiments/snapshot.py`
describe provenance before conversion to the safe public schema.

| Upstream artifact | SHA-256 |
|---|---|
| production model source | `f987cfd13748adf32106353fa6a13431c9bfb8482d223a4af268d583c15df5a7` |
| production result archive | `341771375f8bad095baf7cfe0836dc76e1eaf56f3d92f903ad958848079730cb` |
| production configuration | `5b4c62aa283d78defb37f5d0971783b3dcde1d02638584ea964698af2d8d2d21` |
| baseline phase archive | `782b8762dc04357a50cb6525bc0eed5f67473423ac93406f49ae21fccb53a82f` |
| steady phase archive | `99cdf38d8475a0e875c587ca98ec69c78e2330c9b21f677f1dcac15d54261d0a` |
| activation phase archive | `8048dfb98fb6249dfa9aafcdebbc60edf3bd7179734ebaa3198b54fae837529a` |
| post-activation phase archive | `31a83f28afe20abb9eda6c496c4495041a61d0114d28fe36e0cce539d7fef5f6` |
