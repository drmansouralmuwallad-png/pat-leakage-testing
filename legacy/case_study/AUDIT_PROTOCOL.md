# Literature audit instrument: indexing of simulation-derived predictors

**Purpose.** Measure how often published forecasting studies that use simulated or
reconstructed operational state as model inputs specify the index at which those
predictors are read. The finding "k of n studies do not state it" is reportable on its
own, and converts the manuscript from a self-correction into evidence about a live
practice.

## Inclusion criteria (all must hold)

1. The study reports a quantitative forecasting experiment with an accuracy metric.
2. At least one predictor is produced by a process driven by the target series — an
   inventory or production simulation, a digital twin, a state-space filter, a re-solved
   optimisation model, or a smoothing/imputation step fitted on the series.
3. The predictor is used as a model **input**, not as the forecast target and not as a
   downstream consumer of the forecast.

**Worked exclusion.** Bayer et al. (SpliTech 2024, arXiv:2503.04757) use a digital twin
to *generate the demand series itself* for future grid scenarios, then forecast that
series univariately. No simulated state enters as a feature, so criterion 3 fails and the
study is out of scope. Verified against the full text; recorded here to show the criteria
discriminate.

## Coding rubric — one row per study

| Field | Values |
|---|---|
| `access` | YES if methods section or code is readable; NO otherwise |
| `states_index` | YES = the paper states that predictors are dated at or before the forecast origin (e.g. "features are lagged", "available at time *t*", an explicit index in a feature table) · NO = no such statement found · PARTIAL = stated for some predictors only |
| `verifiable` | CODE = public code lets the indexing be checked · TEXT = prose only · NEITHER |
| `leak_evident` | YES / NO / UNDETERMINABLE — YES only where text or code shows a predictor read at an index at or after the target |
| `quote` | the sentence relied on, with page or section |

## Rules

- Code from the **methods section or released code only**. Abstracts are not sufficient.
  If the full text is inaccessible, record `access = NO` and exclude from the denominator;
  report how many were excluded this way.
- `states_index = NO` means *not stated*, not *wrong*. Report the two separately. The
  defensible headline is about unverifiability, which is what the evidence supports.
- Two coders on at least a 20% sample; report agreement (Cohen's kappa).
- Target n ≥ 20 included studies. Below about 15 the proportion is too imprecise to carry
  a claim.

## Reporting template

> Of *n* studies meeting the criteria with accessible methods, *k* did not state the index
> at which simulation-derived predictors are read; *j* released code permitting
> verification; of those, *i* showed a predictor read at or after the target index.

## Search strings for screening

- digital twin AND demand forecasting AND (feature OR predictor OR input)
- simulated inventory state AND forecasting accuracy
- supply chain digital twin AND machine learning AND forecast
- state-space features AND demand forecasting AND evaluation
- reconstructed inventory AND censored demand AND forecasting

## Candidate leads noted during screening

These were surfaced but **not coded**, because their methods sections were not accessible
in this environment. They are recorded as starting points, not as findings.

- Ziari et al. (2025), *Int. J. Production Economics* — integrated data-driven and AI
  framework for supply chain digital twins; already cited in the manuscript.
- A claim reported second-hand in that paper's literature review — "Su and Lee (2023)
  showed a 40% improvement in demand forecasting accuracy using hybrid digital twin-AI
  models" — is exactly the class of claim the audit targets. **The primary source must be
  read before it is characterised**; the second-hand sentence is not evidence about that
  study's indexing.
