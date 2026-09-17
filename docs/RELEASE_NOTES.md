# v1.0-data — complete preserved research archive

All 17 independent tar.gz parts have uploaded successfully. Their sizes and GitHub
server-side SHA-256 digests match the local release manifest. Repository metadata
was checked without authentication and confirms public visibility.

Use `python tools/fetch_data.py --group all --verify` to restore original paths.
Selective groups are `fresh`, `original`, and `synthetic`. The downloader verifies
both each compressed asset and each extracted member, and removes downloaded parts
when finished. See `data/archive_manifest.json` for full per-file provenance.

The source archive contains 31,768 files / 14,052,503,450 bytes before compression.
Code, reports and selected small tables live in Git; larger files and historical
run payloads live in this release. Environments, caches, Git internals and transient
locks are excluded. Files missing in the original workspace remain explicitly
missing; in particular this release does not invent the absent original `results/`.
