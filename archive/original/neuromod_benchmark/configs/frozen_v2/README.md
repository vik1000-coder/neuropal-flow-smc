# Frozen-v2 benchmark configuration

These are the claim-bearing configurations for the neuromodulatory dynamics
recovery benchmark. They preserve every scientific parameter, metric gate, seed,
and method grid from `frozen_v1`; only the suite names and isolated output root
were changed.

`frozen_v1` stopped fail-closed in its second suite because fitted
heteroskedastic estimators inherited an undeclared reduced-form channel alias.
The same audit found that the constant full-covariance baseline deliberately
emitted a zero conditional-correlation derivative without declaring it. Version
2 removes the redundant inherited alias, declares the intentional zero channel,
and adds a fitted contract sweep covering all 36 configured method aliases. No
fitted scientific outcome informed either implementation correction.

The canonical preflight records a SHA-256 digest for every YAML/JSON file and a
combined digest for the benchmark source, local `sid_neuromod` source, and
execution environment. After a successful preflight, changing any source,
documentation, script, or configuration invalidates the run. A subsequent
scientific or implementation change requires a new frozen version.

Execution is one suite at a time with `nice: 19`, no more than two numerical
threads, a 6 GiB RSS ceiling, and a 4 GiB free-disk floor. The response suites
are a deliberately high-signal, one-modulator identifiability panel; they do
not establish recovery in weak-signal or many-modulator regimes.
