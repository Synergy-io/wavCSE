# LT-0002 Run Note

## Hypothesis

At least one published Task Relation Learning method in each of two human-selected families — (A) asymmetric/directed relations, (B) better relation estimator or task-parameter representation — has a formal assumption that survives our setting (disjoint datasets, heterogeneous 12 / 1251 / 4-class heads, frozen WavLM embeddings) and can be implemented faithfully as published.

## Motivation / prior evidence

Only classical symmetric MTRL has ever been evaluated here (35 runs; `tsm`/`pmr` never trained; GBC archived). `LT-0001` rejected methods for the *F9 gradient-scale rationale* specifically, not for the families themselves. The human re-scoped the programme to benchmarking published variants (DEC-0013, 2026-09-22), deferring the diagnostic-first sequencing; the published-method, taxonomy and category gates are unchanged.

## Method

Primary-source review per family. For each candidate: verify citation against the publication, extract the relation object and update rule, check the six eligibility gates in `PLAN.md`, and state concretely how the method maps onto a three-head model with disjoint datasets — or why it does not.

## Execution ledger

Literature-only Study: no training, no MLflow run. The paper cards in `research/literature/` and this Study's `analysis.md` are the execution and interpretation record.

## Decision criterion

A candidate advances to implementation only if it passes all six eligibility gates. A candidate requiring an invented combination (e.g. a published relation object glued to an unrelated weighting scheme) is recorded as **not faithfully implementable** and does not advance.

## Results

Pending. Cards and the ranked assessment are produced by two parallel literature passes (one per family) and integrated by the controller.

## Interpretation

Pending.

## Decision

Pending.

## Next step

If one or more candidates pass: register a `TR-xxxx` Study per variant on the shared protocol in `PLAN.md` §Controlled variables, screen at one seed against both controls, then confirm at seeds 0–4. If a family fails: record the negative result and report back to the human — do not substitute an adjacent mechanism.
