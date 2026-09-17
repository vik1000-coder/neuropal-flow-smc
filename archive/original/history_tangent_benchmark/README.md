# History-tangent benchmark

This is an isolated synthetic benchmark for conditional-law fitting and the
history tangent

```text
grad_h log p(y | h).
```

It is separate from `sid_neuromod`, `neuromod_benchmark`, and SBTG because none
of those runners implements this estimand or the required normalized, ratio,
and diffusion capability distinctions.  Existing packages are not modified.

Read [PREEXECUTION_AUDIT.md](PREEXECUTION_AUDIT.md) before interpreting a run.
The supplied 2,147-line plan is scientifically useful but not confirmatory-ready
as written; this package first runs a deliberately developmental correctness
gate and never turns it into H1--H7 decisions.

## Run

```bash
PYTHONPATH=src ../.venv/bin/python -m pytest -q
PYTHONPATH=src ../.venv/bin/python -m history_tangent_benchmark.cli \
  run --config configs/developmental_smoke.yaml
```

The runner writes a frozen manifest, one atomic record per case, tidy metric
tables, and a Markdown smoke report beneath the configured result directory.
Oracle calculations use CPU float64.  Neural fitting can use Apple MPS, CUDA,
or CPU, while tangent evaluation disables autocast.

## Capability rules

- M1 and M2 provide normalized conditional log density, samples, and direct
  likelihood-autograd history tangents.
- M4 provides an interaction tangent but no conditional sampler or likelihood.
- M5 directly provides a response-space score.  Its history tangent is reported
  only through conditionally centered mixed-derivative reconstruction at the
  same noisy law used by the oracle comparator.
- Missing capabilities are stored with an explicit status and reason; no
  denoising loss is placed in an NLL column.

The developmental output is a plumbing/cost result.  Confirmatory claims need a
repaired and newly frozen matrix, more independent generator seeds, and external
compute/storage.

## Stabilized backend and whole-path extension (July 2026)

The benchmark now also contains positivity-constrained Gaussian DSM, an exact
conditional affine flow, EDM and Gaussian-anchored EDM denoisers, ordinary and
Gaussian-source conditional flow matching, and a bounded conditional energy
tilt learned by data-versus-reference classification.  These are deliberately
treated as predictive-law backends rather than interchangeable estimates of
the history score.

G8 is an exact correlated whole-future-path mixture.  Its declared history
direction changes higher-order path topology while leaving conditional mean
and covariance invariant.  It supplies a matched Gaussian shadow in population
and an exact density/history-tangent oracle, making it a falsification lane for
claims that flexible generative models help only because the synthetic law was
chosen to match them.

## Revised July 2026 finite-contrast gate

The first executable test of the revised finite signed-contrast theory is
documented in [REVISED_NOTE_V1_AUDIT.md](REVISED_NOTE_V1_AUDIT.md). It compares
the existing history-tangent, normalized-density, ratio-critic, and diffusion
routes with a bounded signed-mixture classifier and a signed-Riesz response
projection on the same central `h +/- delta*v` oracle-mixture target.

```bash
PYTHONPATH=src ../.venv/bin/python -m history_tangent_benchmark.cli \
  run --config configs/revised_note_v1_base_smoke.yaml
PYTHONPATH=src ../.venv/bin/python -m history_tangent_benchmark.cli \
  finite-contrast --config configs/revised_note_v1_finite_contrast.yaml
```

Both commands are developmental. Clean and positive-response-noise laws,
finite and infinitesimal targets, and oracle ceilings remain separate strata.
