# sid_neuromod

**Score-identified dynamics for neuromodulator discovery in C. elegans neural activity.**

`sid` = *score-identified dynamics*. This package estimates **timescale-resolved
distributional interactions** in neural activity, detects statistically significant
changes in the conditional predictive law, and reports what changed in biologically
interpretable terms.

The core estimand is not a VAR coefficient. It is the conditional predictive law

    rho_t(y | H_t)

where `y` is future neural activity and `H_t` is recent neural/behavioral history.
The central object is the **lag-influence score**

    ell_u(y; h) = grad_{x_{t+1-u}} log rho(y | h),

and effects on the **mean, variance, tails, or any statistic** are read out as
*covariances against this score* (the Stein-shift / covariance readout), rather than by
differentiating a fitted black-box model.

See `../sid_neuro_plans.md` for the full developer specification and the two working
papers (`score_identified_dynamics.pdf`, `score_dynamics_empirics.pdf`) for the theory.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
# Full synthetic validation suite
bash scripts/run_all_synthetic.sh

# Single synthetic experiment
python -m sid_neuromod.experiments.run_synthetic \
  --config configs/synthetic_hidden_thermostat.yaml

# Fit a (mock) C. elegans dataset
python -m sid_neuromod.experiments.run_elegans_fit \
  --config configs/elegans_fast_debug.yaml
```

## Testing

```bash
pytest tests/unit -q                       # Gate 1
pytest tests/integration -q -m "not slow"  # Gate 2
```

## Scientific scope & caveats

These are **predictive distributional effects, not direct structural synapses**. Slow
gain/tail/covariance effects are *candidate* neuromodulatory signatures, not molecular
identification. Predictive directedness must not be read structurally without additional
assumptions. See the report caveat section for the full list.
