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

Both family passes completed and were verified against primary sources by the controller. **Family B:** 6 papers verified; **Gonçalves et al. 2016 p-MSSL** (JMLR 17(33)) passes all six gates — a sparse task *precision* learned by graphical lasso (`−d log|Ω| + ℓ₁`), classification supported by the paper's own GLM extension, `1/n_k` loss scaling. Bonilla 2007 is deviation-class (Gaussian regression outputs); Zhao 2020, Zhang & Schneider 2010, Yu 2007 and Fifty 2021 fail their gates. **Family A:** 9 papers verified; **no clean pass.** Lee et al. 2016 AMTL (ICML, directed graph with loss-scaled rows), Zhou & Yang 2023 AutoTR (KDD, closed-form directed matrix) and Oliveira et al. 2019 GAMTL (IJCAI) all require a declared deviation: our heads (12/1251/4 classes) have no aligned parameter columns, so the directed relation can only act on class-mean head summaries. Six others fail outright, including three papers named "GAMTL" that are grouping/undirected methods and Zhang & Yeung's TKDD 2014 "asymmetric" MTRL, whose relation object is in fact symmetric.

## Interpretation

The estimator family contains a method that targets our measured failure (Ω saturation) while remaining a clean published implementation, and its input is literally the same summary matrix the in-category control already uses — so the comparison isolates the estimator. The directed family is real and its canonical member (AMTL) is designed for exactly our sample-size imbalance, but faithful attachment is impossible for heterogeneous heads; the deviation is shared with the control (which also operates on summaries), so it does not break comparability, though it does forfeit external validity.

## Decision

`COMPLETE — CANDIDATES_FOUND`. Recommended order: MSSL (clean), then AMTL if the human admits deviation-class arms, then AutoTR as an ablation of AMTL's loss-scaling ingredient. The deviation-class admissibility call is escalated to the human; it is not decided here.

## Next step

Register `TR-0007` for the clean-pass candidate (MSSL) and run it under `VARIANT_BENCHMARK_PROTOCOL.md` against both controls: screen at one seed, confirm at seeds 0–4. Hold the family-A arm pending the human's deviation-class decision. No adjacent mechanism is substituted for any failed candidate.
