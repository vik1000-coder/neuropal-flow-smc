import React from "react";

import {
  DataComponent,
  DataTable,
  EvidenceChart,
  MetricCard,
  ReportSection,
  RichNarrative,
  useDataApp,
} from "../../data-app-public.jsx";

const columns = (items) => items.map(([field, label]) => ({ field, label }));
const fixed = (value, digits = 3) => Number(value).toFixed(digits);

function Prose({ id, title, queryId, queryIds, sourceRows, sourceRowsByQuery, children, className = "" }) {
  const { visible } = useDataApp();
  if (!visible(id)) return null;
  return (
    <ReportSection id={id} title={title} queryId={queryId} queryIds={queryIds}
      sourceRows={sourceRows} sourceRowsByQuery={sourceRowsByQuery} showHeading={false}
      className={className}>
      <RichNarrative id={`${id}:body`} value={`## ${title}\n\n${children}`} label={`Edit ${title}`} />
    </ReportSection>
  );
}

function EvidenceTable({ id, title, queryId, rows, fields, description, searchable = false }) {
  const { visible } = useDataApp();
  if (!visible(id)) return null;
  return (
    <DataComponent id={id} title={title} queryId={queryId} kind="table"
      displayRows={rows} sourceRows={rows} description={description}>
      <DataTable rows={rows} columns={columns(fields)} searchable={searchable}
        compactNumbers={false} label={title} />
    </DataComponent>
  );
}

function Equation({ label, children }) {
  return (
    <figure className="equation-block" role="group" aria-label={label}>
      <div className="equation" role="math">{children}</div>
      <figcaption>{label}</figcaption>
    </figure>
  );
}

function ClaimLadder() {
  return (
    <div className="claim-ladder" aria-label="Claim ladder from strongest to weakest support">
      <div><span>Established</span><strong>Predictive conditional distributions contain real signal beyond the mean.</strong></div>
      <div><span>Established within learned models</span><strong>Compatibility-aware conditioning defines a coherent model-relative path response.</strong></div>
      <div><span>Supported computationally</span><strong>Progressive SMC reduces finite-particle error for the repaired-law estimand.</strong></div>
      <div><span>Not established</span><strong>Observed response matrices are causal synapses, receptor mechanisms, or physical delays.</strong></div>
    </div>
  );
}

function RepairFlow() {
  return (
    <div className="repair-flow" aria-label="Compatibility-aware repaired path workflow">
      <div><b>1</b><strong>Factual prefix</strong><span>Initialize from a held-out episode.</span></div>
      <i aria-hidden="true">→</i>
      <div><b>2</b><strong>Natural generation</strong><span>Sample complete prefixes under the learned transition law.</span></div>
      <i aria-hidden="true">→</i>
      <div><b>3</b><strong>Compatibility bridge</strong><span>Favor source-low or source-high paths while anchoring the rest of the population.</span></div>
      <i aria-hidden="true">→</i>
      <div><b>4</b><strong>Release and forecast</strong><span>Remove weights at the cut and roll forward with unchanged dynamics.</span></div>
      <i aria-hidden="true">→</i>
      <div><b>5</b><strong>Compare future laws</strong><span>Mean, spread, tails, events, peaks, Wasserstein distance, or occupancy.</span></div>
    </div>
  );
}

const sourcePreviews = {
  "https://arxiv.org/abs/2211.00472": {
    title: "Backtracking Counterfactuals",
    summary: "Formalizes counterfactuals that preserve causal laws and alter earlier conditions rather than performing only local interventions.",
    source: "von Kugelgen, Mohamed, and Beckers",
    date: "2022",
    approvedForReport: true,
  },
  "https://arxiv.org/abs/2402.01607": {
    title: "Natural Counterfactuals With Necessary Backtracking",
    summary: "Introduces a naturalness criterion that permits controlled backtracking to keep counterfactual scenarios feasible under the data distribution.",
    source: "Hao et al.",
    date: "2024",
    approvedForReport: true,
  },
  "https://arxiv.org/abs/2108.13025": {
    title: "Transport-based Counterfactual Models",
    summary: "Uses couplings between observed distributions to construct statistically faithful counterfactual models without requiring a known structural causal model.",
    source: "de Lara et al.",
    date: "2021",
    approvedForReport: true,
  },
};

export function ReportContent() {
  const { reviewedRows, visible, appTitle, setAppTitle, canEdit, mode } = useDataApp();
  const methods = reviewedRows("method_map");
  const tail = reviewedRows("matched_tail");
  const fields = reviewedRows("field_recovery");
  const meanFields = fields.filter((row) => row.target === "Mean field");
  const dispersionFields = fields.filter((row) => row.target === "Log-variance field");
  const gating = reviewedRows("gating_recovery");
  const contrasts = reviewedRows("finite_contrasts");
  const repairs = reviewedRows("later_repairs");
  const sbtg = reviewedRows("sbtg_localization");
  const changes = reviewedRows("changepoints");
  const real = reviewedRows("real_data_results");
  const repairedConfig = reviewedRows("repaired_path_config");
  const smc = reviewedRows("smc_benchmark");
  const ood = reviewedRows("ood_results");
  const atlas80 = reviewedRows("atlas80_external");
  const current = reviewedRows("current_program");

  const meanChart = { type: "horizontalBar", x: "method", y: "nise",
    yLabel: "Mean-field nISE (lower is better)", valueDecimals: 3, startAtZero: true };
  const dispersionChart = { type: "horizontalBar", x: "method", y: "nise",
    yLabel: "Log-variance-field nISE (lower is better)", valueDecimals: 3, startAtZero: true };
  const smcChart = { type: "horizontalBar", x: "estimator", y: "response_mse",
    yLabel: "Response MSE against the high-particle learned-law reference", valueDecimals: 3, startAtZero: true };

  return (
    <article className="report-content sbtg-story-report" aria-label="Technical synthesis of the SBTG research program">
      <header className="report-hero story-hero">
        <p className="eyebrow">Technical synthesis · evidence through September 2, 2026</p>
        <h1 data-data-app-title contentEditable={canEdit && mode === "edit"}
          suppressContentEditableWarning
          aria-label={canEdit && mode === "edit" ? "Edit report heading" : undefined}
          onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}
          onKeyDown={canEdit && mode === "edit" ? (event) => {
            if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
          } : undefined}>{appTitle}</h1>
        <RichNarrative id="story:deck" className="report-deck"
          value="From score geometry and repeated mechanism-recovery failures to normalized conditional laws, compatibility-aware path responses, and progressive bridge sequential Monte Carlo." />
      </header>

      <ReportSection id="executive-answer" title="Executive answer" queryId="real_data_results"
        queryIds={["real_data_results", "field_recovery", "smc_benchmark", "ood_results"]}
        sourceRowsByQuery={{ real_data_results: real, field_recovery: fields, smc_benchmark: smc, ood_results: ood }}
        showHeading={false} className="executive-answer">
        <RichNarrative id="executive-answer:body" className="report-summary-lead" label="Edit executive answer"
          value={`## The program advanced by changing the question, not by making SBTG larger

The original program asked whether score geometry in consecutive neural states could reveal directed, lagged and mechanism-specific interactions. Extensive synthetic and neural experiments showed that this was too ambitious for the learned object and the available observational calcium data. SBTG retained value as a reduced-form localization probe, but increasingly flexible score models did not convert predictive fit into physical mean, dispersion, gating, receptor or causal recovery.

The strongest reproducible findings were narrower and more useful. Simple normalized models were difficult to beat for conditional means and matched dispersion. Conditional variance improved held-out neural likelihood even when it did not recover a neuromodulator connectome. Finite, supported contrasts worked for some distributional changes, and later diagnosis recovered restricted covariance geometry after fixing rank, conditioning and regularization problems.

The resulting methodological shift was to stop treating an arbitrary coordinate edit as a scientific intervention. A compatibility-aware response instead reweights complete model-generated histories, releases them at a declared cut, and compares their future laws. Progressive bridge SMC estimates this law more reliably for difficult queries. It solves a sampling problem—not omitted state, off-support model error, or causal identification.`} />
        <ClaimLadder />
      </ReportSection>

      <div className="report-facts" aria-label="Three pivotal results">
        {visible("metric-variance") && <MetricCard id="metric-variance" title="Variance adds predictive value"
          queryId="real_data_results" sourceRows={real} value="+0.364 NLL"
          comparison="28 worms · every target improved" negative={false}
          description="Held-out gain over the same fitted conditional mean with constant variance." />}
        {visible("metric-simple") && <MetricCard id="metric-simple" title="Simple likelihood wins matched fields"
          queryId="field_recovery" sourceRows={fields} value="0.369 / 0.586"
          comparison="mean / log-variance nISE" negative={false}
          description="Penalized Gaussian NLL; lower is better and 1.0 is the zero-field baseline." />}
        {visible("metric-smc") && <MetricCard id="metric-smc" title="Progressive SMC lowers particle error"
          queryId="smc_benchmark" sourceRows={smc} value="0.011 MSE"
          comparison="direct importance: 0.031" negative={false}
          description="Error against a high-particle estimate of the same learned repaired law—not biological truth." />}
      </div>

      <Prose id="starting-question" title="1. The original scientific bet" queryId="method_map" sourceRows={methods}>
        {`Let X_t in R^d denote standardized observed neural activity and let H_t collect a finite history together with the stimulus schedule. The scientifically defensible observational object is the conditional predictive law P(Y | H=h), where Y is a future neural state or path feature.

SBTG began from a stronger intuition: if a score model captures the geometry of consecutive states, mixed score quantities might localize source-target interactions and their timing. This was appealing because a score can represent nonlinear, heteroskedastic and non-Gaussian structure without explicitly normalizing the density. The difficulty is that a learned score, a conditional law, a physical mechanism and a causal intervention are different mathematical objects. The experiments repeatedly found that improving one did not validate the next.`}
      </Prose>
      <Equation label="Predictive object and conditional mean">
        <span>p<sub>h</sub>(y) = p(Y = y | H = h), &nbsp; m(h) = E[Y | H = h]</span>
      </Equation>

      <Prose id="method-families" title="2. What we tried" queryId="method_map" sourceRows={methods}>
        {`The program covered three separable layers: a backend representing the conditional law, an adapter comparing histories, and a readout declaring the target channel. Many apparent contradictions disappear once results are compared within the same estimand. A method that predicts a mean well is not thereby a variance estimator; a model that predicts tails is not thereby an estimator of a physical tail derivative; and an edge-ranking statistic is not thereby a causal graph.`}
      </Prose>
      <EvidenceTable id="method-inventory" title="Method inventory and current interpretation" queryId="method_map"
        rows={methods}
        fields={[["stage", "Stage"], ["method", "Method"], ["implementation", "Implementation"],
          ["learned_quantity", "Learned quantity"], ["key_result", "Key empirical result"], ["verdict", "Current role"]]}
        searchable />

      <Prose id="score-math" title="3. The mathematical mismatch behind SBTG" queryId="sbtg_localization" sourceRows={sbtg}>
        {`For a conditional density p_h(y), a history-direction log-density witness is ell_v(y;h) = D_v log p_h(y). It describes the infinitesimal change in the entire conditional law along a supported history direction v. The corresponding mixed response-score operator is K_v(y;h) = D_v grad_y log p_h(y) = grad_y ell_v(y;h).

SBTG-like statistics estimate or summarize the mixed operator. Recovering ell_v from its response-space gradient is an inverse problem: support, integration constants, boundary conditions and conditional centering must be specified. A local cross-moment of a learned joint score is therefore not automatically a conditional-law contrast, physical susceptibility, or causal edge.

Hyvarinen score matching minimizes a proper score involving the squared score norm and its divergence. Denoising score matching replaces the clean score by the score of a Gaussian-corrupted law. The corruption scale can regularize estimation, but the trained target is still the corrupted response score unless an additional inversion or sampling identity is validated.`}
      </Prose>
      <Equation label="History witness and mixed SBTG operator">
        <span>ℓ<sub>v</sub>(y;h) = D<sub>v</sub> log p<sub>h</sub>(y), &nbsp;&nbsp; K<sub>v</sub>(y;h) = D<sub>v</sub>∇<sub>y</sub> log p<sub>h</sub>(y) = ∇<sub>y</sub>ℓ<sub>v</sub>(y;h)</span>
      </Equation>
      <Equation label="Hyvarinen score-matching objective, up to constants">
        <span>J<sub>SM</sub>(θ) = E[ ½ ||s<sub>θ</sub>(Y,H)||² + div<sub>y</sub> s<sub>θ</sub>(Y,H) ]</span>
      </Equation>
      <EvidenceTable id="sbtg-table" title="What the frozen synthetic SBTG statistics localized" queryId="sbtg_localization"
        rows={sbtg} fields={[["readout", "Readout"], ["observed_range", "Observed result"], ["interpretation", "Interpretation"]]} />

      <Prose id="synthetic-design" title="4. The matched synthetic adjudication" queryId="field_recovery"
        queryIds={["field_recovery", "matched_tail", "gating_recovery"]}
        sourceRowsByQuery={{ field_recovery: fields, matched_tail: tail, gating_recovery: gating }}>
        {`The decisive objective-comparison panel held the data-generating process, train/validation/test splits, random seeds, forecast horizon and evaluation metrics fixed across ten methods. It covered null, additive-mean, synaptic-gating, stochastic-dispersion and matched-tail mechanisms over five fresh DGP seeds (8-12). All 250 planned cases completed.

The evaluation used the true synthetic law. Predictive NLL and tail-probability RMSE scored the fitted conditional distribution. Mechanism recovery used normalized integrated squared error (nISE) against the known field; nISE=1 is the error of emitting the zero field. This made a failed learned derivative directly comparable with abstention.`}
      </Prose>
      <Equation label="Normalized integrated squared error">
        <span>nISE(f̂) = ∫(f̂(u) − f(u))² du / ∫f(u)² du; &nbsp; nISE = 1 for f̂ ≡ 0</span>
      </Equation>

      {visible("mean-field-chart") && <EvidenceChart id="mean-field-chart" queryId="field_recovery"
        title="A simple Gaussian likelihood led matched mean-field recovery" spec={meanChart}
        rows={meanFields} sourceRows={meanFields} height={390}
        description="Five fresh DGP seeds; lower nISE is better; the zero-field baseline is 1.0." />}
      <EvidenceTable id="mean-field-table" title="Physical additive-mean field" queryId="field_recovery"
        rows={meanFields.map((row) => ({ method: row.method, nise: fixed(row.nise) }))}
        fields={[["method", "Method"], ["nise", "Mean-field nISE"]]} />

      {visible("dispersion-field-chart") && <EvidenceChart id="dispersion-field-chart" queryId="field_recovery"
        title="Modeling variance helped, but simple Gaussian NLL still won" spec={dispersionChart}
        rows={dispersionFields} sourceRows={dispersionFields} height={430}
        description="The constant-variance model defines the 1.0 zero-field baseline; lower is better." />}
      <EvidenceTable id="dispersion-field-table" title="Physical stochastic-dispersion field" queryId="field_recovery"
        rows={dispersionFields.map((row) => ({ method: row.method, nise: fixed(row.nise) }))}
        fields={[["method", "Method"], ["nise", "Log-variance-field nISE"]]} />

      <Prose id="mean-versus-law" title="5. Why conditional-mean success was not enough" queryId="matched_tail"
        queryIds={["matched_tail", "gating_recovery"]}
        sourceRowsByQuery={{ matched_tail: tail, gating_recovery: gating }}>
        {`A conditional-mean model does not become invalid merely because the data become stochastic. It fails only as a complete representation of the response. Two histories can have the same E[Y | H] while differing in variance, covariance, tail risk, multimodality, dwell time or event probability.

The expectation is not itself noisy. Its finite-sample estimate can be noisy, and global averaging can cancel effects that reverse sign across history regions. More importantly, a mean-preserving distributional change is mathematically invisible to any mean-only summary.

The matched-tail experiment made this separation concrete. The MDN represented the two-scale mixture and achieved NLL -0.614 and tail RMSE 0.0095. Yet its fitted physical shape-tail derivative had nISE 2.02, worse than the zero-field baseline. Learning a distributional feature and learning how a physical modulator changes that feature are separate tasks.`}
      </Prose>
      <Equation label="Mean-blind distributional change">
        <span>E<sub>P+</sub>[Y] = E<sub>P−</sub>[Y] &nbsp; while &nbsp; Var<sub>P+</sub>(Y) ≠ Var<sub>P−</sub>(Y) &nbsp; or &nbsp; P<sub>+</sub>(Y &gt; c) ≠ P<sub>−</sub>(Y &gt; c)</span>
      </Equation>
      <EvidenceTable id="tail-table" title="Matched-tail predictive law" queryId="matched_tail"
        rows={tail.map((row) => ({ method: row.method, heldout_nll: fixed(row.heldout_nll, row.heldout_nll > 1 ? 2 : 3), tail_rmse: fixed(row.tail_rmse, 4) }))}
        fields={[["method", "Method"], ["heldout_nll", "Held-out NLL"], ["tail_rmse", "Tail-probability RMSE"]]} />
      <EvidenceTable id="gating-table" title="Synaptic-gating readout: abstention beat every learned derivative" queryId="gating_recovery"
        rows={gating.map((row) => ({ method: row.method, nise: fixed(row.nise), interpretation: row.interpretation }))}
        fields={[["method", "Method"], ["nise", "Gating nISE"], ["interpretation", "Interpretation"]]} />

      <Prose id="finite-contrast" title="6. Finite supported contrasts were more promising than point gradients" queryId="finite_contrasts" sourceRows={contrasts}>
        {`The point-history convergence sweep completed all 96 fits, passed its fixed-batch overfit checks and reproduced under exact replay. Nevertheless, the derivative-readiness gates failed: G1 location tangents reached NRMSE 0.589-0.719, while the best G2 covariance and G3 skew tangents were 1.068 and 1.167.

The next attempt replaced an infinitesimal derivative by a bounded central contrast between two nearby histories. At delta=0.1, ten adapter seeds and 76,760 metric rows showed useful recovery for location and low-rank mixture changes, but not for covariance-only or skew-only changes. Crucially, oracle-sample classifiers were also poor on those cases, isolating the adapter and feature geometry rather than simply blaming the predictive model.`}
      </Prose>
      <Equation label="Finite central law contrast">
        <span>Δ<sub>δ,v</sub>(y;h) = [p(y | h+δv) − p(y | h−δv)] / (2δ)</span>
      </Equation>
      <EvidenceTable id="finite-contrast-table" title="Ten-seed finite-contrast results" queryId="finite_contrasts"
        rows={contrasts}
        fields={[["generator", "Synthetic generator"], ["best_route", "Best practical route"], ["reported_range", "Witness NRMSE"], ["verdict", "Verdict"]]} />

      <Prose id="later-diagnosis" title="7. Some failures were repaired—but only after narrowing the claim" queryId="later_repairs" sourceRows={repairs}>
        {`Later adjudication was important because it separated genuine impossibility from fixable design error. The E9 covariance experiment had the wrong effective rank and an inverse ridge four orders of magnitude too large. Correcting those choices restored typed covariance recovery. The E6 fixed Riesz sieve had an approximation bias floor, while a learned history-density score plus cross-fitted ORTH improved with sample size. Filtered latent recovery in E8 required a much longer informative horizon.

These repairs do not resurrect unrestricted full-law Jacobian recovery. They support a modular rule: when the downstream channel is known, use a calibrated direct or structured head; when many bounded features are needed, use supported, cross-fitted contrasts with explicit conditioning diagnostics.`}
      </Prose>
      <EvidenceTable id="repair-table" title="Post-hoc diagnoses and repaired conclusions" queryId="later_repairs"
        rows={repairs}
        fields={[["problem", "Experiment"], ["initial_diagnosis", "Initial failure"], ["repair", "Repair"], ["result", "Result"], ["remaining_limit", "Remaining limit"]]} />

      <Prose id="changepoint-wrapper" title="8. Changepoints did not rescue an invalid channel" queryId="changepoints" sourceRows={changes}>
        {`Changepoint detection is an inference layer around a static statistic; it cannot supply the missing semantics. In matched simulations, mean CUSUM and a residual tail-shape probe both detected and localized every designed high-signal change. The residual variance detector detected all dispersion changes but localized only 41%, and it also fired on 95% of mean changes and 99% of matched-tail changes. It was therefore a broad instability alarm, not a dispersion-specific detector.

This result established the project's mechanism-confusion rule: a readout earns a channel label only if it detects its target and remains null or unlocalized under matched nuisance changes.`}
      </Prose>
      <EvidenceTable id="changepoint-table" title="Matched changepoint calibration" queryId="changepoints"
        rows={changes}
        fields={[["probe", "Probe"], ["target_case", "Case"], ["detected", "Detected / FPR"], ["localized", "Localized"], ["nuisance_result", "Interpretation"]]} />

      <Prose id="real-data" title="9. Neural data forced the same narrowing" queryId="real_data_results" sourceRows={real}>
        {`The original peptidergic gain-slow result was large under an 80% row split, attenuated when rows were grouped by animal, and reversed under the validated 28-worm ACMMA analysis. It also collapsed after global-mode removal and did not reproduce under denoising score matching, neural DSM or variance VAR. This was a comprehensive negative for the original lag-resolved anatomical or receptor-target claim.

The durable real-data result was predictive instead: conditional variance improved held-out likelihood relative to a model with the same fitted mean and constant variance. Reliability levers such as denoising-noise averaging and ganglion aggregation improved stability, but not selectively for the hypothesized distributional mechanism. The corrected 54-neuron atlas ultimately retained 102 baseline mean edges yet produced zero strong-support resolved lags and zero rows passing both sampler and matched quiet-time controls.`}
      </Prose>
      <EvidenceTable id="real-results-table" title="Selected neural-data results across the program" queryId="real_data_results"
        rows={real}
        fields={[["stage", "Analysis"], ["cohort_or_scope", "Cohort or scope"], ["result", "Observed result"], ["interpretation", "What it established"]]} searchable />

      <Prose id="hard-edit" title="10. The deeper problem was the response question itself" queryId="smc_benchmark" sourceRows={smc}>
        {`Suppose genuine histories satisfy U_t=V_t. Every conditional-mean extension m_beta(h)=m_0(h)+beta(U_t-V_t) makes identical predictions on observed histories, whatever beta is. A hard edit that changes U_t while holding V_t fixed leaves that support and produces the arbitrary answer beta(u-u_0). Held-out accuracy cannot choose among those extensions.

This is a nonidentification problem, not a tuning problem. It motivated replacing coordinate surgery by a probability distribution over complete histories that the learned dynamics can actually generate.`}
      </Prose>
      <Equation label="Why an off-support hard edit is not identified by predictive accuracy">
        <span>m<sub>β</sub>(h) = m<sub>0</sub>(h) + β(U<sub>t</sub>−V<sub>t</sub>), &nbsp; U<sub>t</sub>=V<sub>t</sub> on observed support; after editing U only, the answer changes by β(u−u<sub>0</sub>)</span>
      </Equation>

      <ReportSection id="repaired-law" title="11. Compatibility-aware repaired path responses" queryId="smc_benchmark"
        queryIds={["smc_benchmark", "method_map"]}
        sourceRowsByQuery={{ smc_benchmark: smc, method_map: methods }} showHeading={false}>
        <RichNarrative id="repaired-law:body" label="Edit repaired-path method" sourcePreviews={sourcePreviews}
          value={`## 11. Compatibility-aware repaired path responses

Choose a cut c. Let P_theta(d omega) be the learned natural law of a generated repair prefix followed by a future. For source neuron j, let A_j(omega) be its average over a declared source window. Training animals determine low and high targets a_j^- and a_j^+, typically the 25th and 75th percentiles. A Gaussian source potential favors paths near the target, while an anchor penalizes implausible departures of the remaining population from the factual episode.

The repaired law Q_j^q is the normalized product of the natural learned path law and this compatibility potential. After the cut, the weights are removed and all particles evolve under the unchanged transition model. The response is the difference between a declared future feature under the high- and low-compatible laws, divided by the source displacement actually achieved.

This construction is inspired by feasibility-preserving [backtracking counterfactuals](https://arxiv.org/abs/2211.00472), [natural counterfactuals](https://arxiv.org/abs/2402.01607), and [transport-based counterfactual models](https://arxiv.org/abs/2108.13025). The important distinction is that the present response assumes neither a structural causal model nor a cross-world coupling. It is a model-relative observational response.`} />
        <RepairFlow />
      </ReportSection>
      <Equation label="Source statistic and compatibility potential">
        <span>A<sub>j</sub>(ω) = W<sup>−1</sup> Σ<sub>r∈window</sub> X<sub>r,j</sub>(ω), &nbsp; G<sub>j</sub><sup>q</sup>(ω) = exp[−½((A<sub>j</sub>−a<sub>j</sub><sup>q</sup>)/ε<sub>j</sub>)²] · exp[−λC<sub>j</sub>(ω,x*)]</span>
      </Equation>
      <Equation label="Repaired path law">
        <span>Q<sub>j</sub><sup>q</sup>(dω) = G<sub>j</sub><sup>q</sup>(ω) P<sub>θ</sub>(dω) / Z<sub>j</sub><sup>q</sup>, &nbsp; Z<sub>j</sub><sup>q</sup> = E<sub>Pθ</sub>[G<sub>j</sub><sup>q</sup>]</span>
      </Equation>
      <Equation label="Normalized high-versus-low future response">
        <span>M<sub>h</sub>(k,j) = [E<sub>Qj+</sub>[F<sub>h</sub><sup>k</sup>] − E<sub>Qj−</sub>[F<sub>h</sub><sup>k</sup>]] / max(E<sub>Qj+</sub>[A<sub>j</sub>] − E<sub>Qj−</sub>[A<sub>j</sub>], 0.10)</span>
      </Equation>
      <EvidenceTable id="repaired-config-table" title="Corrected repaired-path implementation settings"
        queryId="repaired_path_config" rows={repairedConfig}
        fields={[["setting", "Setting"], ["value", "Declared value"], ["role", "Why it is present"]]} />

      <Prose id="smc-method" title="12. Direct weighting and progressive bridge SMC" queryId="smc_benchmark" sourceRows={smc}>
        {`Direct importance weighting samples complete natural paths once and self-normalizes G_j^q for every source, target and horizon. It is fast and statistically pairs the low and high arms, but rare compatible histories can concentrate almost all weight on a few paths.

Progressive bridge SMC constructs intermediate targets proportional to P_theta times G raised to a tempering exponent gamma that moves from 0 to 1. The next exponent is selected to maintain an ESS target; low-ESS particle clouds are resampled and propagated through the next repair step. Once the full condition is imposed at the cut, the particles are unweighted and released into the original learned dynamics.

In the sealed Stage-A benchmark, progressive SMC had the lowest response MSE, bias and Monte Carlo variance and the highest validity rate. This comparison was against high-particle estimates of the same learned law and used no external atlas for selection.`}
      </Prose>
      <Equation label="Self-normalized direct importance estimator">
        <span>Ê<sub>Q</sub>[F] = Σ<sub>n</sub> w̄<sub>n</sub> F(ω<sub>n</sub>), &nbsp; w̄<sub>n</sub> = G(ω<sub>n</sub>) / Σ<sub>m</sub>G(ω<sub>m</sub>)</span>
      </Equation>
      <Equation label="Progressive tempered targets">
        <span>π<sub>γ</sub>(dω) ∝ P<sub>θ</sub>(dω) G(ω)<sup>γ</sup>, &nbsp; 0 = γ<sub>0</sub> &lt; γ<sub>1</sub> &lt; ... &lt; γ<sub>T</sub> = 1</span>
      </Equation>
      {visible("smc-mse-chart") && <EvidenceChart id="smc-mse-chart" queryId="smc_benchmark"
        title="Progressive bridging reduced finite-particle error" spec={smcChart}
        rows={smc} sourceRows={smc} height={290}
        description="Response MSE is measured against high-particle estimates of the same frozen learned-law estimand." />}
      <EvidenceTable id="smc-table" title="Sealed finite-particle estimator comparison" queryId="smc_benchmark"
        rows={smc.map((row) => ({ estimator: row.estimator, response_mse: fixed(row.response_mse, 6),
          bias_squared: fixed(row.bias_squared, 6), mc_variance: fixed(row.mc_variance, 6),
          signed_rho: fixed(row.signed_rho), valid_rate: `${(100 * row.valid_rate).toFixed(1)}%`,
          runtime_minutes: fixed(row.runtime_minutes, 1) }))}
        fields={[["estimator", "Estimator"], ["response_mse", "Response MSE"], ["bias_squared", "Bias²"],
          ["mc_variance", "MC variance"], ["signed_rho", "Signed rho"], ["valid_rate", "Valid"],
          ["runtime_minutes", "Runtime, min"]]} />

      <Prose id="ood-lesson" title="13. SMC fixes particle error, not learned-model error" queryId="ood_results" sourceRows={ood}>
        {`The final robustness study qualified six different conditional generators and separated particle error from learned-model error across supported queries, rare events, deliberate population mismatches, artificial spikes, extreme mismatches, nonlinear VAR systems and FitzHugh-Nagumo networks.

For rare queries, progressive bridging improved markedly over direct importance sampling at the same particle count, but did not beat direct importance under the study's matched-compute comparison. More importantly, sixteen rows were numerically stable but scientifically wrong: SMC gates passed and particle error was small while learned-model error exceeded 0.25.

Effective sample size diagnoses the approximation to one fitted model. It does not certify that the model is correct for the query. The strongest tested warning signal was disagreement across qualified model families. Every downstream result should therefore report empirical support, event rarity, cross-model spread and SMC convergence together.`}
      </Prose>
      <EvidenceTable id="ood-table" title="Model-error and OOD safeguards" queryId="ood_results"
        rows={ood} fields={[["finding", "Finding"], ["value", "Value"], ["status", "Status"], ["meaning", "Meaning"]]} />

      <Prose id="atlas-status" title="14. What the current neural atlases do—and do not—show" queryId="atlas80_external"
        queryIds={["atlas80_external", "current_program"]}
        sourceRowsByQuery={{ atlas80_external: atlas80, current_program: current }}>
        {`The optimized historical 80-neuron atlas completed 20 raw archives and 13,977,600 dense response cells. Its progressive-bridge lag-1 scores exceeded the published SBTG point estimates on the four prespecified Randi and Cook references. Those comparisons are useful as historical model correspondence, not prospective biological validation.

The limitation is decisive: the released 80-neuron cache used invalid index-wise head/tail fusion and donor-worm trace copying. The new estimator cannot turn those traces into simultaneous recordings. By contrast, the corrected 54-neuron program preserved animal identity and explicit onset/quiet controls, but found no strong-support resolved lags and no response cell that passed both sampler and temporal-specificity gates.

The honest conclusion is that the new response law and sampler are methodologically stronger, while the mechanistic neural claim remains unconfirmed.`}
      </Prose>
      <EvidenceTable id="atlas80-table" title="Historical 80-neuron lag-1 external correspondence" queryId="atlas80_external"
        rows={atlas80.map((row) => ({ method: row.method, reference: row.reference, auroc: fixed(row.auroc), source_macro_auroc: fixed(row.source_macro_auroc) }))}
        fields={[["method", "Method"], ["reference", "Reference"], ["auroc", "AUROC"], ["source_macro_auroc", "Source-macro AUROC"]]} />

      <Prose id="where-now" title="15. Where the program stands now" queryId="current_program" sourceRows={current}>
        {`The project now has a clear separation between the predictive model, the response estimand, the particle algorithm and external biological checks. The next prospective stage is the DANDI 000981 sex-aware campaign: 22 retained hermaphrodites and 22 retained males, one exact-name shared neuron vocabulary expected near 153 neurons, separate combined/hermaphrodite/male models, whole-animal folds and all four declared samplers. Model and sampler selection occur before Cook, Randi, Bentley, SBTG or receptor references are loaded.

This design directly addresses the historical fusion problem and creates a stronger test of reproducibility across sex-specific models. It remains an observational/model-relative program unless a separate intervention design supplies the assumptions needed for a causal claim.`}
      </Prose>
      <EvidenceTable id="current-program-table" title="Current artifact and execution status" queryId="current_program"
        rows={current} fields={[["artifact", "Artifact"], ["state", "State"], ["scope", "Scope"], ["claim_boundary", "Claim boundary"]]} />

      <Prose id="conclusion" title="Conclusion" queryId="method_map"
        queryIds={["method_map", "smc_benchmark", "ood_results", "current_program"]}
        sourceRowsByQuery={{ method_map: methods, smc_benchmark: smc, ood_results: ood, current_program: current }}>
        {`The central lesson is not that SBTG was useless or that conditional means failed. It is that the scientific object must be aligned with the representation, adapter, readout and validation design.

SBTG remains a potentially useful reduced-form localization statistic. Simple likelihood and classical dynamics are the right tools when the target is a low-order conditional mean or matched dispersion field. Flexible normalized generators become valuable when the future distribution itself matters. Compatibility-aware repaired paths then provide a disciplined way to query that learned distribution without assigning meaning to an arbitrary off-support edit, and progressive SMC makes difficult versions of that query estimable.

The strongest current contribution is therefore methodological: a hierarchy from predictive-law qualification, through support-aware distributional responses and particle diagnostics, to explicitly post-freeze external comparison. The unresolved scientific question is whether those model-relative response laws remain stable and informative in a clean prospective cohort.`}
      </Prose>

      <ReportSection id="references" title="Sources and literature" queryId="method_map"
        queryIds={["method_map", "real_data_results", "smc_benchmark", "ood_results", "current_program"]}
        sourceRowsByQuery={{ method_map: methods, real_data_results: real, smc_benchmark: smc, ood_results: ood, current_program: current }}
        showHeading={false} className="reference-section">
        <RichNarrative id="references:body" label="Edit sources" sourcePreviews={sourcePreviews}
          value={`## Sources and literature

**Primary repository syntheses**

- SYNTHETIC_METHODS_REPORT_2026-07-12.md: matched objective panel, finite contrasts, SBTG localization, graph, latent and changepoint results.
- EXPERIMENT_REVIEW_2026-07-12.md: cross-experiment interpretation and stop/go rules.
- reports/sid_stable_synthesis_2026-07-13/main.tex and reports/distributional_sid_adjudication_2026-07-14/main.tex: later stability, conditioning and covariance diagnoses.
- sid_elegans/output/newlevers/RESULTS.md: predictive variance and real-data reliability levers.
- FLOW_REPAIRED_LAG_METHODS_20260828.md and results/compatibility_path_response/EXPERIMENT_INDEX.md: repaired-path estimand, sampler chronology and estimator benchmark.
- results/query_ood_robustness_20260901/REPORT.md: model-family, support and stable-but-wrong analysis.
- results/neural_prediction_atlas_20260829/README.md and results/sbtg80_optimized_full_atlas_20260902/REPORT.md: corrected 54-neuron and historical 80-neuron atlas results.
- dandi000981_lag_cluster_bundle/README_CLUSTER.md: prospective campaign design.

**Related counterfactual literature**

- von Kugelgen, Mohamed, and Beckers. [Backtracking Counterfactuals](https://arxiv.org/abs/2211.00472).
- Hao et al. [Natural Counterfactuals With Necessary Backtracking](https://arxiv.org/abs/2402.01607).
- de Lara et al. [Transport-based Counterfactual Models](https://arxiv.org/abs/2108.13025).

The compatibility-aware construction is closest in spirit to feasibility-preserving backtracking and distribution-respecting counterfactuals. It deliberately stops short of calling the result causal because it assumes neither a structural causal model nor a cross-world coupling.`} />
      </ReportSection>
    </article>
  );
}
