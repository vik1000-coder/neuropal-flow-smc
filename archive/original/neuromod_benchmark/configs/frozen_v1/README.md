# Frozen-v1 benchmark configuration

These are the claim-bearing configurations for the neuromodulatory dynamics
recovery benchmark. They were copied from `configs/frozen_v1_draft` only after
the metric audit, response-truth calibration, resource audit, and full test
suite passed. No fitted benchmark outcomes were used to select the response
stratum.

The canonical preflight records a SHA-256 digest for every YAML/JSON file and a
combined digest for the benchmark source, local `sid_neuromod` source, and
execution environment. After a successful preflight, changing any source,
documentation, script, or configuration invalidates the run. A scientific or
implementation change therefore requires a new frozen version rather than an
in-place edit.

Execution is one suite at a time with `nice: 19`, no more than two numerical
threads, a 6 GiB RSS ceiling, and a 4 GiB free-disk floor. The response suites
are a deliberately high-signal, one-modulator identifiability panel; they do
not establish recovery in weak-signal or many-modulator regimes.
