# Legacy compatibility audit amendment

The initial launch of `legacy_reproduction_subset_20260714.yaml` was rejected by
configuration validation before any case was created.  The value
`tier: audit_reproduction` is not in the legacy runner's closed tier enumeration.
It was changed to the valid non-confirmatory value `tier: development`; the audit
purpose and source run remain explicit in metadata.  Seeds, generators, models,
training budgets, evaluation settings, and comparison rules were unchanged.
