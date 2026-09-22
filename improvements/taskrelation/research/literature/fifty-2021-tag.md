# Fifty et al. (2021) — Task Affinity Grouping (TAG)

## Citation

Christopher Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. “Efficiently Identifying Task Groupings for Multi-Task Learning.” *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*, pp. 27503–27516.

Primary source: https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html

## Problem

Deciding which tasks should train together is left to human intuition or exhaustive search over `2^{|T|} − 1` task combinations. The paper wants a cheap, principled way to decide *when to share*: measure how one task’s gradient update affects another task’s loss in a single joint training run, then group tasks accordingly.

## Mathematical assumption

Hard-parameter sharing: shared parameters `s` plus per-task parameters `{θ_i}`; total loss is the (equally weighted) sum of non-negative task losses. For a batch `X^t` at step `t`, a hypothetical single-task update is `s^{t+1|i} := s^t − η ∇_s L_i(X^t, s^t, θ_i^t)`. The **directed inter-task affinity** is the scale-invariant lookahead ratio

`Z^t_ij = 1 − L_j(X^t, s^{t+1|i}, θ_j^t) / L_j(X^t, s^t, θ_j^t)`,

averaged over training, `Ẑ_ij = (1/T) Σ_t Z^t_ij`. `Z_ij` need not equal `Z_ji`. Theory: in the convex setting, with equal gradient norms and a negative-cosine condition on the lower-affinity task, the grouping maximising per-task affinity provably makes more progress than any other group.

## Relation representation

An explicit learned `|T| × |T|` **directed affinity matrix `Ẑ`**, estimated from per-task **losses** (before/after a hypothetical gradient step on shared parameters) rather than from parameter summaries. This is the canonical published answer to the “estimate task relatedness from gradients/losses rather than parameter summaries” lead.

## Optimization method

Not a training regularizer. Pipeline: (I) train all tasks together once; (II) accumulate `Z^t_ij` during that run (every 10 steps suffices); (III) solve a network-selection problem — choose `k ≤ b` multi-task networks maximising the affinity onto each served task (NP-hard; branch-and-bound or binary integer programming); (IV) retrain each selected group and deploy. Auxiliary tasks may be trained in a group without being served from it.

## Evidence

CelebA (9 attributes; TAG beats HOA, random grouping and gradient-cosine grouping, and runs 22× faster than HOA) and Taskonomy (segmentation, depth, keypoints, edges, surface normals; TAG beats MTL by 10.0%, GradNorm by 7.7%, single-task by 1.5% and random grouping by 9.5%; its 2-split beats HOA’s by 2.5% while HOA needs an extra 2,008 Tesla V100 GPU-hours). Ablations: affinity measured every 10 steps matches every-step; affinity scores computed on train and validation are near-identical (Pearson 0.98); affinities change over training; groupings shift with batch size and learning rate. The authors note one failure case (CelebA attribute `a8`, where the affinity difference between best and worst partner is tiny and TAG picks the worst partner).

## Assumptions

Hard parameter sharing with a shared feature representation and per-task heads; the tasks co-train in a pilot run (so their per-step losses are observable jointly); a grouping/assignment model, where each served task belongs to exactly one network at inference; scalar-valued per-task losses whose ratio is meaningful; equal loss weighting during the pilot run.

## Differences from our setting

TAG answers *which subset of tasks trains together*, a decision our protocol never makes — we always train KS+SI+ER jointly. Its output is a discrete grouping, not a matrix used to regularize the model, and its validation requires retraining every candidate group. Our heads also differ in width (12 / 1251 / 4), though the affinity itself only needs shared-parameter gradients and per-task losses, which our model does supply.

## Implementation difficulty

Moderate to implement the affinity measurement (one extra lookahead loss evaluation per task pair every few steps) but there is nothing to implement as a *relation mechanism*: acting on `Ẑ` means changing the training-set composition, which is a different study design (and the ER per-batch-share control, DG-0005, already showed that composition changes the observed signals).

## Candidate Study ID

`LT-0002` — family-B gradient/loss-based lead, verified and closed.

## LT-0002 assessment

1. **Explicit relation object — PASS.** A learned directed affinity matrix estimated from data (per-step losses under hypothetical single-task updates).
2. **Taxonomy — FAIL.** The relation object is consumed by **task grouping** — the clustering category that Zhang & Yang keep separate from §2.4. TAG is a task-grouping method, not a task-relation regularizer.
3. **Heterogeneous-head compatibility — PASS for the estimator.** `Z_ij` needs only shared-parameter gradients and per-task losses, both well-defined in our heterogeneous-head model.
4. **Fixed-representation compatibility — PASS.** Nothing requires upstream retraining; the pilot run only reads gradients/losses.
5. **Faithful implementability — FAIL.** Its published use is retraining selected task groups, a different protocol from our fixed three-task joint training; there is no published training-time update rule that turns `Ẑ` into a regularizer.
6. **Primary-source verification — PASS.** Authors, venue (NeurIPS 2021, vol. 34, pp. 27503–27516), Eq. (1), Lemma 1/Proposition 1, and the CelebA/Taskonomy numbers checked against the NeurIPS PDF.

**Verdict: FAIL — the gradient/loss-based relation estimator is real and explicit, but the paper’s method is task grouping (clustering), which is out of the §2.4 Task Relation Learning category.** Documented negative for the “estimate task relatedness from gradients or losses” lead: the published endpoint of that lead is grouping, not an eligible relation regularizer.
