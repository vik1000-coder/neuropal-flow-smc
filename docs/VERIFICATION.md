# Verification record

The archive and new synthesis are ready to share within the documented scope.
Scientific limitations are retained in CODE_REVIEW.md and ANALYSIS.md.

| Check | Outcome |
|---|---|
| Source file hashes | 31,768 match their original source bytes |
| Archived Python syntax | 995 files parse; legacy invalid-escape warning retained |
| Original core methods | 324 tests passed, two skipped |
| Fresh replication | 11 tests passed, including independent Gaussian oracle |
| September synthetic benchmark | Nine tests passed |
| New publication checks | Five tests passed |
| Relocated frozen snapshot | All declared files pass; both cohort loaders work |
| Primary checkpoint receipts | All 30 pass after relocation |
| Independent primary matrix scoring | All 16 AUROC/AP comparisons and edge counts match |
| Publication figures | 18 PNG/PDF pairs generated; inspected with captions and uncertainty |
| Tutorial | 20 cells, 10 executable code cells; executed through active-environment runner; seven figures |
| Clean clone | Publication tests and tutorial execute without large release downloads |
| GitHub Linux checks | Tests and all figure rebuilds pass on the analysis branch |
| Data release | 17 assets; 12,787,714,013 compressed bytes; all server SHA-256 digests match |
| Public download | Anonymous repository access and a download/extract/hash smoke test pass |
| New documentation | Local links resolve; portable reading copies preserve frozen originals |

Test counts are per scientific suite, not inflated by repeats in the original,
packaged and clean-clone locations. Only the central scientific paths were manually
reviewed; parsing all historical files is not proof of every exploratory algorithm.
The complete historical workflows whose input/result directories are absent remain
unverified, explicitly labeled archival evidence. No new NeuroPAL training or
sampling was performed to prepare these publication figures.

Machine-readable receipts and test logs are in `audit/`. Public CI status and the
merge record are attached to pull request #1.
