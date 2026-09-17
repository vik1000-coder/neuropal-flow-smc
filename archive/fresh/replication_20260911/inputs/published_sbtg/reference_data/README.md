# Reference data

This directory contains the compact, machine-readable reference inputs needed by the included analyses:

- aligned Cook chemical-synapse and gap-junction matrices;
- aligned Randi perturbational-response arrays;
- Bentley monoamine and neuropeptide edge/expression tables.

Cook anatomy and Randi perturbational responses are complementary reference networks, not interchangeable ground truth for lag-specific effective connectivity. The original calcium-imaging MAT files and Cook workbooks are not duplicated in this release. Obtain source datasets from their original publications and run `pipeline/01_prepare_data.py` when rebuilding preprocessing from raw data.

The source datasets retain their original publication terms. Inclusion here does not relicense third-party data. See [the data guide](../docs/DATA.md) for provenance and expected filenames.
