# Current two-stage pipeline

The supported workflow has two independently evaluated stages.

## Stage 1: learn the conditional transition law

**Question:** Given recent neural activity and the causal stimulus history, what distribution of neural activity should occur next?

**Code:** [`../conditional_neural_benchmark/`](../conditional_neural_benchmark/)

**Current default:**

- predict the next-frame residual rather than the raw next state;
- use the local width-128 TCN encoder;
- use a conditional flow-matching head;
- train on the naturally occurring window distribution;
- use dropout, modest weight decay, and small neural-history jitter;
- split, scale, tune, and evaluate by whole worm;
- select using proper one-step scores plus free-running rollout scores.

**Promotion gate:** a checkpoint must predict held-out factual activity well and remain stable over multistep rollout. Atlas agreement is not a training criterion.

## Stage 2: estimate a repaired-path contrast

**Question:** Under the frozen learned law, how does the predicted future change when one source neuron's recent activity is repaired from a compatible low value to a compatible high value?

**Code:** [`../compatibility_neural_benchmark/`](../compatibility_neural_benchmark/)

**Current primary estimator:** progressive bridge SMC.

The bridge gradually introduces the source constraint while particles are propagated. ESS-adaptive tempering and resampling retain paths that remain compatible with the requested condition. At the final bridge step, the target clamp is the same clamp used by the direct estimator; the bridge changes the finite-particle route to the target, not the final estimand.

**Independent sensitivity:** direct importance sampling with a larger natural path bank.

**Promotion gate:** sufficient ESS, controlled maximum weight, adequate achieved source displacement, repaired-root diversity, generator-seed stability, cross-sampler stability, animal stability, and timing stability.

## What the combination does and does not establish

The conditional model supplies a flexible sequential generator. Progressive bridge SMC conditions paths from that generator without requiring a tractable transition density. Together they estimate a model-relative conditional path contrast efficiently.

They do **not** by themselves establish:

- a physical intervention;
- a causal neural edge;
- a direct synapse;
- receptor action;
- anatomical rewiring;
- a physical transmission delay.

## Primary code entry points

| Task | Entry point |
| --- | --- |
| Generic conditional-model training | [`../conditional_neural_benchmark/runner.py`](../conditional_neural_benchmark/runner.py) |
| Focused conditional-flow tournament | [`../conditional_neural_benchmark/focused_world_model_runner.py`](../conditional_neural_benchmark/focused_world_model_runner.py) |
| Regularization and rollout tournament | [`../conditional_neural_benchmark/biological_world_model_runner.py`](../conditional_neural_benchmark/biological_world_model_runner.py) |
| Progressive bridge implementation | [`../compatibility_neural_benchmark/progressive_smc.py`](../compatibility_neural_benchmark/progressive_smc.py) |
| Corrected lag-response runner | [`../compatibility_neural_benchmark/aligned_lag_response_runner.py`](../compatibility_neural_benchmark/aligned_lag_response_runner.py) |
| Current atlas plan | [`../PREDICTION_ATLAS_PLAN_AND_CODE_AUDIT_20260829.md`](../PREDICTION_ATLAS_PLAN_AND_CODE_AUDIT_20260829.md) |

## Primary method evidence

- [`../results/compatibility_path_response/frozen_estimator_benchmark_20260826/analysis/REPORT.md`](../results/compatibility_path_response/frozen_estimator_benchmark_20260826/analysis/REPORT.md)
- [`../results/aligned_lag_response_20260828/analysis/TECHNICAL_REPORT.md`](../results/aligned_lag_response_20260828/analysis/TECHNICAL_REPORT.md)
- [`../results/four_sampler_lag_connectome_20260828/analysis/PROGRESSIVE_BRIDGE_SMC_TECHNICAL_REPORT.md`](../results/four_sampler_lag_connectome_20260828/analysis/PROGRESSIVE_BRIDGE_SMC_TECHNICAL_REPORT.md)
