# Which conditional generator is worth using?

This assessment distinguishes architecture properties, preserved experimental evidence, and results that are still missing. New controlled results will be written to RESULTS.md. Source snapshots and a fixed protocol live in this directory; the ongoing flow/SMC replication is separate.

## The comparison that matters

A generator must get the conditional distribution right where a query asks it to predict. Good average forecasting, good rare-event probabilities, stable numerical sampling, and a correct derivative are different achievements. A flexible generator can win forecasting but lose on tail calibration or derivatives. Giving a sampler more particles addresses approximation to one fitted law, not error in that law.

| Generator | Practical strength | Main limitation for these queries | Sampling and uncertainty |
|---|---|---|---|
| Ridge + full Gaussian residuals | Cheap, reproducible baseline; explicit cross-output covariance; straightforward conditional means | Linear conditional mean and history-independent covariance cannot express nonlinear modes or changing tails | Direct Gaussian samples; fitted residual uncertainty, without a posterior over regression parameters |
| RBF GP + Gaussian observations | Nonlinear regression with kernel-based posterior variance; useful low-dimensional baseline | Kernel/likelihood assumptions determine extrapolation; ordinary Gaussian regression misses multimodality and changing residual shape; full GP fitting is expensive as sample count grows | Predictive variance includes latent-function uncertainty under the fitted kernel plus observation noise; the tested implementation treats outputs independently |
| Neural diagonal Gaussian | Fast conditional location and scale; normalized likelihood; smooth mean derivatives | Conditional output independence erases residual cross-output effects from soft conditioning; cannot represent skewness or multiple modes | One network evaluation and Gaussian noise; variance is predicted distributional spread, not automatic model uncertainty |
| Neural low-rank Gaussian | Adds affordable conditional cross-output covariance | Elliptical, light-tailed family; rank and covariance estimation can dominate query effects | Fast joint samples; exact likelihood in the implemented family |
| Neural low-rank Student-t | Retains covariance and permits heavy-tailed joint excursions | Elliptical shape still excludes multimodal laws; degrees-of-freedom estimation and rare-event calibration need validation | Shared radial scale changes joint tails; fast sampling with a chi-square multiplier |
| Autoregressive MDN | Explicit multimodality, cross-output dependence, normalized likelihood | Component collapse and output-order dependence; sample generation is sequential across coordinates | Softmax selects component probabilities; component means and scales also determine uncertainty |
| Autoregressive Transformer MDN | Attention can condition each output on previously generated coordinates; same normalized mixture construction | More parameters/compute; ordering and component issues remain; this implementation is not a long-history encoder experiment | Sequential continuous-output generation. Logit entropy alone omits within-component spread and does not certify OOD reliability |
| Conditional flow matching | Flexible joint continuous distribution; straightforward regression training objective; useful common-noise coupling for finite differences | ODE sampling cost and step sensitivity; no exact likelihood exposed by this implementation; extrapolation remains learned-model dependent | Many velocity evaluations per sample; smooth transport does not by itself ensure accurate conditioning derivatives |
| EDM diffusion | Flexible denoising-based conditional law; useful independent modeling assumptions | Noise schedule/preconditioning/solver affect results; repeated network evaluations; no exact likelihood exposed here | The implementation uses deterministic Heun sampling given base noise; neither sample spread nor denoising stability certifies OOD accuracy |

These are properties of specified implementations, not universal rankings. Neural models use different output families and parameter counts; a fixed-budget experiment compares these configurations, not optimally tuned method classes. GP kernel choice and output dependence deserve as much scrutiny as neural architecture choice.

## What the preserved NeuroPAL evidence actually says

1. In the older pooled-data first screen, flow energy1.3948, MDN-4 1.4122, neural Gaussian1.4249, ridge Gaussian1.5394, persistence1.6702 (lower better). Neural finalists were close relative to the reported standard-error band. Different earlier screens used different cohorts and evaluation banks.
2. In the corrected17-worm structure comparison, flow energy1.02146 versus Student-t1.06621 and rank8Gaussian1.06717. Flow's4.20% energy improvement was accompanied by3.73% lower RMSE. This was average one-step forecasting, not a rare-query generator ranking.
3. Flow's estimated large-innovation frequency was10.38% versus observed5.91%; rank8Gaussian5.54%, Student-t6.78%. Their tail Brier scores (.05570/.05531) were lower than flow's.06411. The pairwise tail comparison was exploratory. Better average energy did not guarantee better tails.
4. MDN slightly beat flow on10-second energy in an older rollout comparison; flow had a better variogram. Horizon and metric change the ranking.
5. The preserved OOD report summary says six families qualified and supported-query mean cross-family SD was.01077. Disagreement correlated.797 with total synthetic error, and16 rows were numerically stable but model-wrong. The summary is insufficient to recover errors by generator and query class. Its matched-N versus matched-cost comparisons concern samplers, not architecture rankings.
6. A source audit finds that deliberate NeuroPAL support-exclusion refits were only ridge-Gaussian probes. The six-family study did not include an autoregressive Transformer or a GP. Earlier Transformer development tests cannot be substituted for that missing comparison.

Local evidence: ../reports/conditional_flow_model_report_20260901/main.tex, ../reports/sbtg_to_compatibility_report_20260902/src/data.json (ood_results), ../query_ood_robustness/protocol.py, ../query_ood_robustness/pseudo_ood_runner.py. The original result tree is missing, so preserved report values are labeled inherited evidence, not newly audited raw results.

## New analyses being executed

The fixed192-task supplementary benchmark compares all nine families against known conditional laws. It reports generator-specific prediction and tail errors, conditional-effect errors with independently qualified numerical references, small-bank particle errors, predictive-mean derivative errors, solver sensitivity, runtime, and fit convergence. Exact GP and output-autoregressive Transformer are included. All families see identical histories and observations.

Generator error is only reported as resolved when oracle and learned-law reference banks both meet the declared ESS/repeat-variation gates. Failure to resolve a rare-event reference is a finding, not a license to rank its noisy estimate as truth. Pairwise comparisons to flow use common qualified cells and retain coverage counts. No model is declared best from pooled averages alone.

Remaining beyond this supplement: full NeuroPAL retraining of Transformer/GP and the other families on identical folds, held-out support-region exclusions across every family, all-family repaired-path lag matrices, long-history Transformer encoder comparisons, and broader hyperparameter/data-size sensitivity. None is silently replaced by a three-dimensional synthetic test.

## Primary methodological sources

- Gaussian-process regression, kernel and likelihood construction: https://gaussianprocess.org/gpml/code/matlab/doc/ and https://scikit-learn.org/stable/modules/gaussian_process.html
- Flow-matching regression objective: https://arxiv.org/abs/2210.02747
- EDM denoising/preconditioning and numerical sampling choices: https://arxiv.org/abs/2206.00364
- Predictive confidence needs calibration rather than being assumed reliable from softmax: https://arxiv.org/abs/1706.04599 (classification evidence; continuous mixture calibration must be tested separately).
