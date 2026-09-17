# Data provenance and preparation

## Calcium imaging

The preprocessing pipeline expects the NeuroPAL calcium-imaging MAT files corresponding to the head and tail recordings described by [Yemini et al. (2021)](https://doi.org/10.1016/j.cell.2020.12.012). Raw recordings are omitted from this repository because of file size and redistribution considerations.

Place the source files in `data/` with these names:

| Source | Expected filename |
|---|---|
| NeuroPAL head recordings | `Head_Activity_OH16230.mat` |
| NeuroPAL tail recordings | `Tail_Activity_OH16230.mat` |
| Cook adjacency matrices | `SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx` |

Then run:

```bash
python pipeline/01_prepare_data.py --impute-missing --full-traces
```

Prepared datasets are written below `results/intermediate/datasets/`. Each
dataset stores variable-length recordings in `traces.npz` as finite numeric
values, a Boolean missing-value mask, and integer segment offsets; no prepared
trace archive requires pickle.

The portable sensitivity snapshot includes only `traces.npz`, `standardization.json`, and `segments.csv`, which are the prepared inputs required by the production-lineage resampling and phase analyses. `traces.npz` stores finite values, an explicit missing-value mask, and segment offsets without Python objects, so it loads with `allow_pickle=False`. Recording identifiers in `segments.csv` are experimental sample identifiers, not machine identifiers.

## Structural reference

[Cook et al. (2019)](https://doi.org/10.1038/s41586-019-1352-7) chemical-synapse and gap-junction data are represented by aligned matrices in `reference_data/connectome/`. The original supplemental workbooks are omitted. `nodes.json` records the aligned neuron order.

## Perturbational-response reference

[Randi et al. (2023)](https://doi.org/10.1038/s41586-023-06683-4) wild-type and unc-31 arrays are in `reference_data/functional_atlas/`. The canonical binary evaluation uses `q < 0.05` for positives and `q_eq < 0.05` for confirmed negatives; positives take priority when both conditions occur. Ambiguous measured pairs are excluded.

## Modulatory reference

[Bentley et al. (2016)](https://doi.org/10.1371/journal.pcbi.1005283) monoamine and neuropeptide tables are in `reference_data/modulatory_atlas/` with their source readmes. The monoamine evaluation uses the class-level `edge_lists/edgelist_MA_classes.csv`, matching the cell-class resolution of the SBTG matrices.

## Interpretation

Cook anatomy and Randi perturbational responses are reference networks. Neither is assumed to be complete ground truth for observational lag-specific effective connectivity. Third-party data remain subject to their original terms and citations; consult each linked source before redistribution.
