# Publication work plan

Authorized 2026-09-17: review and organize the existing NeuroPAL 54/80 work,
fresh replication and synthetic comparisons; publish a public GitHub repository;
add comparative/robustness/neuron-level figures and an executed four-neuron tutorial.

1. Inventory and preserve available code, data, checkpoints, reports and receipts.
2. Review scientific code paths and document claim/implementation correspondence.
3. Run applicable tests; independently verify packaged primary metrics.
4. Publish the reviewed archive (large immutable data as checksummed release assets).
5. Add reproducible cohort-separated figures and a worked nonlinear four-neuron notebook.
6. Validate artifacts, push an analysis branch, merge it, and check public access.

Original files and frozen hashes are not rewritten. Missing historical results are
listed explicitly. No expensive NeuroPAL retraining is required for this publication;
the tutorial is a new, separately labeled synthetic demonstration.

## Completed publication work

- Archive assembled: 31,768 files; all source hashes checked; 995 Python files parse.
- Packaged scientific suites: 324 original tests, 11 replication tests, nine synthetic
  tests passed; two original tests skipped. Five new publication tests passed.
- Public repository created; all 17 data-release assets uploaded and server hashes
  verified; anonymous visibility and a clean download/restore checked.
- Eighteen publication figures and seven tutorial figures generated and inspected.
- Notebook executed end to end through the portable active-environment runner.
- Sixteen primary AUROC/AP comparisons independently recomputed from saved arrays.
- Analysis branch prepared for GitHub CI and merge to main.

Scientific limitations and missing historical artifacts remain documented rather
than being hidden by a successful software/test/publication outcome.
