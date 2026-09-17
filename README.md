# NeuroPAL: flow models, repaired-path responses, and progressive bridge SMC

A reproducible research archive comparing conditional generative models and
repaired-path response estimators with published SBTG on NeuroPAL calcium recordings.
The corrected **54-class head cohort** and historical **80-class SBTG cohort** are
always analyzed separately. This repository includes the original development,
the fresh September 2026 replication, and independent synthetic comparisons.

**Scientific conclusion:** the fresh flow/SMC matrices improve Cook structural and
chemical ranking under the tested sensitivities. Reliable identification of the
complete derivative-based lag matrix is not established. Historical80 contains
head/tail pseudo-pairing and donor copying; clean54 has no multiplicity-corrected
lag-minus-lag-1 discoveries. These are model-relative observational responses.

## Start here

- [Collection guide and evidence status](docs/COLLECTIONS.md)
- [Code review and documentation corrections](docs/CODE_REVIEW.md)
- [Reproduction instructions](docs/REPRODUCING.md)
- [Fresh replication final report](archive/fresh/replication_20260911/REPORT_FINAL.md)
- [Nine-generator synthetic comparison](archive/synthetic/generator_tradeoffs_20260913/SUMMARY.md)
- [Original conditional-model report](archive/original/reports/conditional_flow_model_report_20260901/conditional_flow_model_report_20260901.pdf)

## Layout

| Location | Contents |
|---|---|
| `archive/original/` | Preserved original code, available data/results, reports, dashboards, and supporting synthetic programs |
| `archive/fresh/replication_20260911/` | Frozen code, inputs, 30 primary fits, sampling, robustness and final interpretation |
| `archive/synthetic/generator_tradeoffs_20260913/` | Completed 192-task, nine-generator known-law comparison |
| `analysis/`, `figures/`, `data/` | New reproducible synthesis, cohort-separated figures and their plotted values |
| `notebooks/` | Executed four-neuron nonlinear lag tutorial |
| `tools/`, `audit/`, `docs/` | Restore/verification tools, audit receipts and documentation |

## Data access

Code, documentation and compact summaries are in Git. Large arrays, checkpoints,
raw recording files and historical run outputs are **included as public GitHub
release assets**, not omitted or stored as unusable LFS pointers. Restore them with:

```bash
python tools/fetch_data.py --group all --verify
```

Download a single collection with `--group fresh`, `original`, or `synthetic`.
Every archived file has a SHA-256 checksum in `data/archive_manifest.json`;
release assets have separate checksums in `data/release_assets.json`. Allow about
14 GB for the full extracted archive and temporary space for one downloaded part.

## Evidence boundaries

The original workspace's top-level `results/` was already missing. Original
reports and embedded tables that refer to those absent runs are preserved and
labeled **reported, raw artifacts unavailable**. We do not manufacture replacements.
The fresh replication and September synthetic benchmark retain their run artifacts.

Frozen historical documents sometimes contain workstation paths or superseded
claims. The collection guide and code review explain which interpretation controls;
original hashes are retained. Do not launch old dispatchers in place: they are
historical execution records and may resume expensive computations.

## Attribution

See [data provenance and rights](docs/PROVENANCE_AND_RIGHTS.md). Third-party datasets,
software and reports retain their original attribution and terms. No new blanket
license is asserted over third-party material.
