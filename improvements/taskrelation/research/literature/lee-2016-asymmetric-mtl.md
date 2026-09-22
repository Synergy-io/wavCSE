# Lee, Yang and Hwang (2016) — Asymmetric Multi-task Learning

## Citation

Giwoong Lee, Eunho Yang, and Sung Ju Hwang. “Asymmetric Multi-task Learning Based on Task Relatedness and Loss.” *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:230–238, 2016.

Primary source: https://proceedings.mlr.press/v48/leeb16.html (PDF: https://proceedings.mlr.press/v48/leeb16.pdf)

Verified against the PMLR PDF (abstract; Eqs. (1)–(6); Theorem 1; Algorithms 1–2; Tables 1 and Fig. 1). Attribution note: several bibliographies retitle this paper “...Task Relatedness and *Confidence*” (e.g. the reference list of IJCAI-19 paper 444). The PMLR title is “...Task Relatedness and *Loss*”; the authors are Lee, Yang and Hwang, not Lee/Yang alone.

## Problem

Symmetric transfer is the default assumption in MTRL, so a difficult or unreliable task is allowed to regularise an easy, confident one. The paper argues the transfer direction must follow both relatedness and the per-task training loss, and that a sparse directed graph achieves this.

## Mathematical assumption

Each task parameter is *succinctly reconstructed* from the other tasks’ parameters,

`w_t ≈ Σ_{s≠t} B_st w_s`, with `B ≥ 0`, `B_tt = 0`,

so `B` is a non-negative, directed (in general `B_st ≠ B_ts`) and sparse transfer graph. The sparsity penalty on a task’s *outgoing* row is multiplied by that task’s own loss, `(1 + μ‖b_t^o‖₁) L(w_t; D_t)`, so low-loss (confident) tasks may afford outgoing edges and high-loss tasks may not.

## Relation representation

A learned `T × T` directed transfer graph `B`; `B_st` is the weight of task `s` in reconstructing task `t`, and the row `b_t^o` collects outgoing transfers from `t`. The paper proves (Theorem 1) that at any local optimum, a task with larger outgoing transfer norm has no larger loss than the one it displaces, i.e. edges point from low-loss to high-loss tasks. Relatedness is defined by parameter reconstructability, not by an input-distribution similarity.

**Attachment in our model.** The coupling parameter is `B ∈ R^{3×3}` (`B_tt = 0`, six free directed entries). The method requires one comparable parameter vector per task. Our architecture has a single *shared* trunk and per-task heads of width 12 / 1251 / 4, so no per-task parameter copies and no aligned head columns exist; the only common object is the class-mean head summary already used by the classical MTRL control, `w̃_t = (1/C_t) Σ_c W_t[c,:] ∈ R^{2000}`. With that summary, `W = [w̃_KS, w̃_SI, w̃_ER] ∈ R^{2000×3}` and the published penalty is well defined.

## Optimization method

Biconvex, solved by alternating optimization (Algorithm 1):

1. fix `B`, minimise `Σ_t (1 + μ‖b_t^o‖₁) L(w_t; D_t) + λ Σ_t ‖w_t − Σ_{s≠t} B_st w_s‖₂²` in `W` by gradient descent (or block-coordinate per task);
2. fix `W`, solve for each *incoming* column `b_t^i` a non-negative weighted LASSO, `min_{b ≥ 0} μ‖Λ b_t^i‖₁ + λ‖w_t − W b_t^i‖₂²`, where `Λ = diag(task losses)`; the paper uses an existing weighted-LASSO solver.

`λ` and `μ` are tuned by cross-validation; the paper also proposes re-weighting the loss by `c_t = 1/√n_t` when sample sizes differ. The second algorithm, AMTL-Curriculum (Algorithm 2), learns one task at a time in increasing-loss order with a greedy selection rule and yields an acyclic graph.

## Evidence

Synthetic 12-task regression with two groups and two noise levels: easy tasks acquire outgoing edges to hard tasks and not vice versa (Fig. 1c). Real data: MNIST (10-way, one-vs-all as tasks), USPS, School (regression, 139 schools) and AWA. AMTL beats STL, MTFL, GO-MTL, SC-MTL, a Curriculum-simple baseline, a symmetric MTRL variant (SMTL) and an ablation without loss scaling (AMTL-noLoss). The reported gain is attributed to suppressed negative transfer, with no large per-task degradation; the authors state the improvement is larger on datasets with strong imbalance in per-task training size. Runtime: AMTL-Curriculum is ~20× faster than AMTL on the synthetic set. School is the one dataset where GO-MTL wins, which the authors attribute to near-total task homogeneity.

## Assumptions

One shared data space and model space for all tasks; all tasks positively correlated; parameter vectors of equal dimension that are meaningfully linear-combinations of each other; task loss usable as a reliability proxy (they acknowledge overfitting on small tasks and propose `c_t = 1/√n_t`); sparse transfer suffices.

## Differences from our setting

* **No per-task parameter vectors.** Our heads are 12 / 1251 / 4 wide over disjoint label spaces; the published object “`w_t`” does not exist. Only the class-mean head summary aligns the three tasks (2000 = hidden width), and a summary-level penalty also applies a uniform pull across all classes of a head, which the published method does not contemplate.
* **Loss-as-reliability.** AMTL’s asymmetry is driven by per-task training loss. Under F10 the gradient-scale signal in this project is a *training-mixture* property (ER contributes ≈ 47 of every 2048 sampled examples), so loss-scaled edge sparsity would partly track mixture/sample-size effects. AMTL’s own `1/√n_t` re-weighting targets exactly this confound and is the published lever for it.
* **Task count.** `T = 3` leaves six off-diagonal entries; the SPATS card already records that sparse coupling is weakest at small task counts. AMTL’s reported driver (loss imbalance across tasks) is present in KS/SI/ER.
* **Motivation hygiene.** DEC-0013 forbids citing F8/F9 asymmetry or gradient-scale evidence as the reason for this arm.

## Implementation difficulty

Moderate. Matched-pair update under a single optimizer: on each training step compute the three summaries from the current heads, take one gradient step on (trunk, heads) whose loss includes `λ Σ_t ‖w̃_t − Σ_s B_st w̃_s‖₂²` and the loss-scaled term, and periodically (e.g. once per epoch) re-solve `B` with the weighted non-negative LASSO sub-step on the current summaries — this is Algorithm 1 with the `W` step replaced by standard SGD steps, as the paper itself allows for non-quadratic losses. The gradient of the reconstruction penalty with respect to every row of head `t` is `(2/C_t)(w̃_t − Σ_s B_st w̃_s)`.

**Documented deviation.** The published method regularises the *task models*; here the penalty can act only on a collapsed class-averaged summary of each head, because the trunk is shared and the head widths differ. The deviation is forced by the architecture, is the same device the project’s in-category control already uses for `Ω`, and does not alter the relation object or the update rule — but it is a deviation from the published form, so the verdict below is not a clean PASS.

## Candidate Study ID

`LT-0002` — screened candidate (family A). If adopted, one `TR-xxxx` variant Study is pre-registered per DEC-0013 under the shared matched protocol.

## LT-0002 assessment

* **Gate 1 — explicit relation object: PASS.** `B` is a learned sparse *directed transfer graph*, estimated jointly with the task parameters.
* **Gate 2 — taxonomy: PASS.** Task Relation Learning in the Zhang & Yang §2.4 sense: an explicit parameter-reconstruction relation, not low-rank, clustering, decomposition, loss weighting, uncertainty weighting or gradient surgery. (Loss enters only as a *scale on the sparsity penalty over edges*, not as a per-task loss weight; the relation object is `B`.)
* **Gate 3 — heterogeneous heads: PASS WITH THE STATED CHANGE.** The relation survives disjoint datasets (each task keeps its own loss) but not unequal widths in its published form; the stated change is the class-mean head summary, which supplies the aligned 2000-dim columns the published update needs. This change is what makes the verdict a documented deviation rather than a clean pass.
* **Gate 4 — fixed representation: PASS.** Nothing upstream is retrained; the method touches only the trainable trunk and heads over frozen wavCSE/WavLM embeddings.
* **Gate 5 — faithful implementability: PASS.** Eqs. (1)–(3), Theorem 1 and Algorithms 1–2 are implementable as published, with `λ`, `μ` cross-validated or taken from the paper’s ranges and `c_t = 1/√n_t` as the paper’s natural default for unequal sample sizes. No foreign weighting scheme is attached.
* **Gate 6 — source verified: PASS.** Venue, year, authors, equations and evidence checked against the PMLR PDF.

**Verdict: `PASS WITH DOCUMENTED DEVIATION`** — the relation object (sparse loss-scaled directed transfer graph `B`) and the alternating weighted-LASSO update are implementable as published, but the object being related must be a class-mean head summary rather than a per-task model parameter vector. Under the LT-0002 rule that a documented deviation is not a faithful implementation, this counts as a faithfulness fail unless the human rules the shared summary device (already used by the matched MTRL control, which is the arm this variant must beat) an acceptable instantiation.
