# Generator choice matters, and the best choice depends on the estimand

All 192 planned supplementary fit/evaluation tasks completed across nine generator families, four known synthetic laws, three independently generated datasets per law and two initializations per neural model. This is a new three-input, three-output benchmark with 1,024 training examples per dataset. It does not recover the missing original NeuroPAL OOD tables or establish a biological lag-matrix replication verdict. The separate main replication was not changed.

## The strongest finding for effect estimation

On the central linear correlated law, the diagonal neural Gaussian and correlated neural Gaussian had predictive energy scores of 0.28493 and 0.28147: a difference of about 1.2%. Their absolute errors for the same common soft conditional-effect query were 0.22315 and 0.01183, respectively: approximately a nineteenfold difference. The independent-output GP had effect error 0.22315 as well.

This is a structural distinction. At fixed history, both independent-output models imply zero effect of conditioning on output 0 on the mean of output 1. A well-fitting marginal distribution does not supply the missing joint dependence. Ridge with a full residual covariance gave error 0.00217; flow gave 0.02097; the Transformer mixture gave 0.06577. These are means over the three datasets, averaging neural initializations first. They concern one specified soft conditional contrast, not causality or a physical delay.

Evidence: [analytic effects](ANALYTIC_EFFECTS.md), [paired effects](analysis/analytic_effects_paired.csv), [predictive dataset means](analysis/predictive_dataset_means.csv).

## Tradeoffs in the configurations actually tested

| Generator | What it did well here | What limits the result |
|---|---|---|
| Ridge + full Gaussian residual | Lowest predictive energy in all three regions of the linear law; very accurate linear conditional effects; cheapest sampling. | Its linear mean extrapolated badly on the nonlinear laws. Correct simple structure can be excellent, but assuming that structure is consequential. |
| RBF Gaussian process, independent outputs | Fast sampling; competitive central predictive scores. | Poor extrapolated predictive scores and no cross-output conditional-mean effect at fixed history. Its low upper-tail probability error outside support did not imply good overall calibration: average 90% interval coverage there was only 59.5%. This is not a test of multi-output or non-Gaussian GPs. |
| Neural diagonal Gaussian | Cheap, competitive ordinary predictive fit. | Cannot express residual cross-output dependence. The approximately nineteenfold effect-error example above exposes a failure that the predictive energy score barely distinguished. |
| Neural correlated Gaussian / Student-t | Correlated Gaussian had the lowest mean extrapolated energy on all three nonlinear laws. Student-t was competitive on heavy tails and led the heavy-tail rare-effect comparisons among fully qualified families in the analytic supplement. | Neither represents a general multimodal law. Rankings are configuration-specific; the correlated Gaussian's advantage over flow on heteroskedastic extrapolation reversed on one of three datasets. |
| Autoregressive mixture-density network | Lowest mean central and boundary predictive energy on the bimodal law, though the central advantage over flow was tiny and reversed across datasets. About twelve times faster than flow for these small CPU sampling calls. | Output order and mixture capacity are fixed here; extrapolation and derivatives can still fail. Some rare conditional references remained unresolved. |
| Output-autoregressive Transformer mixture | Competitive boundary predictions; lowest mean boundary energy on the heteroskedastic law. On the extrapolated bimodal rare-effect query, absolute error was 0.0381 versus flow's 0.1241, with lower error in all three paired datasets and all references qualified. | No universal advantage. Slower than the small MDN and simple Gaussian samplers. This tests a three-output Transformer, not long-history attention. Entropy did not certify reliability. |
| Flow / EDM diffusion | Flow was best by mean central energy on the heavy-tailed law and close to the leaders on other central laws. Both can express joint distributions beyond one Gaussian or Student-t. | Integration adds cost. Neither dominated extrapolation or conditional effects. Difficult direct conditional references remained unresolved even at the declared maximum bank size. |

“Lowest mean” is a descriptive ranking of these fitted configurations, not evidence that an entire architecture family is optimal. Three datasets per law are too few for broad superiority claims. See [full findings and paired seed ranges](FINDINGS.md).

## Predictive performance and derivative reliability are different

Linear extrapolation illustrates how the model assumption matters: ridge's energy was 0.2824 versus flow's 0.5591. On nonlinear extrapolation, the neural correlated Gaussian had lower mean energy than flow on all three laws, but errors remained substantial. A relative winner can still be unreliable outside the observed support.

For flow, mean absolute derivative error across the specified laws, seeds and output targets was 0.0705 centrally and 0.4571 at the extrapolated anchor using step 0.1. The corresponding three-repeat Monte Carlo standard errors averaged 0.00147 and 0.00095. Across steps 0.05–0.2, extrapolated error stayed between 0.4564 and 0.4572. Thus reproducible finite differences were substantially wrong against the known derivative. Small sampling noise or step sensitivity is not sufficient evidence of derivative accuracy.

These derivatives concern the conditional mean with respect to one history coordinate. They do not test the repaired path law or recover a time-lag matrix. All steps and targets are retained in [derivative data](analysis/derivatives_raw.csv).

## Transformer entropy: tested directly

The categorical entropy of the Transformer's mixture weights decreased from central to extrapolated histories in all four laws, while mean prediction error increased. In the linear law, mixture entropy fell from 0.681 to 0.643 while mean-squared error rose from 0.00232 to 0.466. Its nominal 90% interval coverage fell from 87.5% to 49.9%.

The full joint predictive differential entropy did increase outside support. That statistic includes mixture locations and scales; mixture-label entropy alone does not. Nevertheless, the generated intervals still under-covered the true outcomes. This is a concrete example of model uncertainty estimates being miscalibrated under misspecification, not proof that entropy is never informative. Mixture labels are also representation-dependent, and differential entropy depends on coordinate units. See [entropy diagnostic](ENTROPY_RESULTS.md).

## Rare effects and numerical accuracy

The frozen primary analysis uses independent Monte Carlo model and oracle references with both repeat-error and effective-sample-size gates. Several references failed those gates, including every central rare oracle. Those failures were preserved, rather than counted as successes or zero errors.

A separately documented post-run supplement derives exact Gaussian/mixture true effects and uses scalar quadrature for the true Student-t law. Gaussian and Student-t learned effects can also be evaluated analytically or by quadrature. For the other families, qualified independent Monte Carlo references remain necessary. This unequal numerical access means that an excluded flow, MDN, Transformer or EDM cannot be declared worse from a failed precision gate. Paired rankings use common qualified cases.

Using the same fitted checkpoints, increasing small direct sample banks from 128 to 512 reduced average particle error on qualified references for every family. It did not uniformly reduce error against the true effect: flow's aggregate true-effect error was 0.2070 versus 0.2088. Those two numbers average every specified query, while the particle-error averages have family-specific qualification masks; they are not an additive decomposition of absolute errors and do not rank samplers. The signed raw error decomposition is audited. See [sampling summary](analysis/sampling_analytic_summary.csv).

Doubling flow/EDM integration from 24 to 48 steps roughly doubled runtime but changed conditional means very little: mean-change RMSE was below 0.00054 in each family/region average. This check did not explain the much larger observed extrapolation errors. Flow's sampling time was approximately 79 times the correlated Gaussian's and twelve times the MDN's for 256 three-dimensional draws per history on this machine. These local CPU ratios do not establish performance at NeuroPAL dimensions.

## Verification and remaining work

All 168 neural fits stopped under the predeclared validation rule; four used the extension beyond 200 epochs, and none ended with the declared boundary-best unresolved flag. This does not establish optimizer optimality. The 24 non-neural fits completed their declared grids. Hyperparameters and model sizes were fixed, rather than exhaustively tuned or matched by compute.

The six original tests passed, followed by three tests for the analytic supplement, including independent direct simulations of all four known laws. Source and result receipts were checked; shared datasets, row counts, finite predictive outputs, aggregation and error algebra were audited. Six figures were visually inspected and layout defects corrected with documented amendments. Figures show descriptive means; qualification coverage and paired seed ranges must be read alongside them. See [verification receipt](analysis/validation.json).

Still unperformed: full-cohort NeuroPAL comparisons including Transformer and GP, generator-specific training-support exclusions, larger output/history dimensions, broader capacity/training/tuning sensitivity, and repaired lag matrices across generator families with Cook/Randi comparisons. The original per-generator OOD raw outputs and checkpoints were absent. The completed synthetic supplement establishes that generator choice can materially alter the answer; it does not identify the biologically correct generator or finish the main replication.
