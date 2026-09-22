# Graffeuille, Koh, Wicker and Lehmann (2024) — Self-Auxiliary Asymmetric Learning

## Citation

Olivier Graffeuille, Yun Sing Koh, Jörg Wicker, and Moritz Lehmann. “Enabling Asymmetric Knowledge Transfer in Multi-Task Learning with Self-Auxiliaries.” arXiv:2410.15875 [cs.LG], 21 October 2024.

Primary source: https://arxiv.org/abs/2410.15875 (full text: https://arxiv.org/html/2410.15875v1)

Verified against the arXiv HTML (abstract; §1–§3; Eqs. (1)–(12); §4 and Tables 1–3). **Preprint only** — no peer-reviewed venue verified; treat the evidence as non-archival.

## Problem

Standard MTL supports only symmetric transfer, so joint training can help one task and hurt another (asymmetric task relationships). The paper wants directed transfer that keeps the positive component and drops the negative one, without altering the final architecture.

## Mathematical assumption

A model `g` splits into shared parameters `θ_sh` and task-specific parameters `θ_t`. A *self-auxiliary* clone `T_{s→t}` uses task `s`’s data, labels and augmentations but task `t`’s task-specific modules, so `s`’s learning signal reaches `θ_t` and not vice versa. Inclusion coefficients `ω_t ≥ 0` (primary task included) and `ω_{s→t} ≥ 0` (directed transfer edge included) enter the training loss

`L^train(θ, ω) = Σ_t [ ω_t L_t(f(x_t^train | θ_sh, θ_t), y_t^train) + Σ_{s≠t} ω_{s→t} L_s(f(x_s^train | θ_sh, θ_t), y_s^train) ]`,

with `ω` selected by a bi-level problem `min_ω L^val(θ*(ω))` s.t. `θ*(ω) = argmin_θ L^train(θ, ω)` (Eqs. (1)–(4)).

## Relation representation

A set of *directed coefficient pairs* `(ω_t, ω_{s→t})` representing which directed task relations exist and how strongly they matter, learned in one of three ways:

* **enumeration** — train a shared-bottom MTL model for every task pair and set `ω_{s→t} = 1` iff `A_t^{{s,t}} > A_t^{{t}}` (boolean, `O(T²)` full trainings);
* **loss weighting** — the bi-level problem approximated by a virtual step and a finite-difference estimator; `ω` trained by Adam (`lr = 1e-4`) with two extra forward/backward passes;
* **combined** — elementwise product of the two, with an `ω` normalisation (Eq. (11)).

The relation is thus a *training-time inclusion/weighting decision*, not a matrix parameter inside the model’s objective. There is no covariance, precision, similarity matrix or transfer graph stored in the model.

## Optimization method

Ordinary SGD/Adam training of the (unchanged) MTL network with the selected self-auxiliary clones added to the batch; clones are discarded at inference, so they add no parameters or compute at test time. Both the paper and its baselines use a half-shared ResNet (first three blocks shared, last two task-specific).

## Evidence

NYUv2 (13-class segmentation, depth, surface normals), Cityscapes (19-class segmentation, 10-class part segmentation, disparity) and CelebA (9 binary attribute tasks), three seeds, evaluated as relative per-task improvement over single-task learning. The asymmetric pattern is documented: on NYUv2 all methods improve depth, most improve segmentation, and *all* degrade normal estimation. SAAL_ew is best overall on all three datasets; SAAL_w is best on depth but poor on normals. Baselines: equal weighting, uncertainty weighting (Kendall), DWA, Auto-λ, PCGrad, CAGrad, GradDrop. No tuning of baselines’ hyperparameters.

## Assumptions

An encoder–decoder architecture with task-specific encoders *and* decoders; a shared input space; crucially, **task `s`’s labels must be predictable by task `t`’s task-specific module** (`L_s(f(x_s | θ_sh, θ_t), y_s)` in Eq. (4)); enough compute for the enumeration variant (`O(T²)` trainings); tasks are dense-prediction/vision tasks with comparable output structure.

## Differences from our setting

* **Taxonomy.** The paper presents itself as a multi-task *optimisation* strategy, benchmarked against loss-weighting (uncertainty weighting, DWA, Auto-λ) and gradient-manipulation (PCGrad, CAGrad, GradDrop) methods. Its relation is expressed as task-loss *coefficients* `ω`, and the project’s binding taxonomy places loss weighting and optimisation methods outside Zhang & Yang §2.4. It is a legitimate *directed-relation estimator* but not a relation object carried by the model.
* **Heterogeneous heads — structural failure.** The self-auxiliary term requires evaluating task `s`’s labels through task `t`’s task-specific parameters. In our model the only task-specific parameters are the heads (12 / 1251 / 4 classes) and the trunk is *shared by all three tasks*, so `T_{KS→SI}` would have to run Speech-Commands labels through the 1251-way speaker head: undefined. Where the architecture does have task-specific encoders, transfer is expressible; here it is not.
* **No parameter to compare against the control.** The learned object is an inclusion set/coefficient vector over losses, not a `T×T` relation matrix that could be compared with MTRL’s `Ω` under a matched protocol.
* **Evidence transfer.** All datasets are shared-input dense-prediction or multi-attribute vision tasks; nothing in the paper tests disjoint datasets with different label spaces.

## Implementation difficulty

For our model: not implementable. Faithfully reproducing a self-auxiliary requires a task-specific encoder/decoder split that does not exist in a shared-trunk three-head design, and the clone’s loss is undefined across disjoint label spaces. Implementing the coefficients alone would reduce the method to task-loss weighting, which is explicitly outside the project’s taxonomy.

## Candidate Study ID

`LT-0002` — screened candidate (family A), recorded as a taxonomy and architecture failure.

## LT-0002 assessment

* **Gate 1 — explicit relation object: PARTIAL/PASS.** The directed coefficients `ω_{s→t}` *are* estimated from data (enumeration or bi-level loss) and do represent directed task relations. But they are not a relation structure in the model — they select which clones to train.
* **Gate 2 — taxonomy: FAIL.** The method is a training-time optimisation/auxiliary-cloning strategy whose coefficients act as per-task loss weights; the project excludes loss weighting and optimisation methods from Task Relation Learning.
* **Gate 3 — heterogeneous heads: FAIL.** Self-auxiliaries require another task’s labels to be predicted by this task’s task-specific module; with disjoint label spaces (12 / 1251 / 4) and a shared trunk this is undefined, and no published update rule remains.
* **Gate 4 — fixed representation: PASS.** The frozen wavCSE embedding is untouched; the method only changes what is trained.
* **Gate 5 — faithful implementability: FAIL.** Eq. (4) cannot be instantiated; a partially-implemented version would be a different method (task-loss weighting).
* **Gate 6 — source verified: PASS (with the caveat that the source is a non-archival preprint).** Author list, equations and experiments checked against the arXiv full text.

**Verdict: `FAIL — training-time auxiliary-cloning / loss-coefficient optimisation outside the Task Relation Learning taxonomy, and structurally inapplicable to disjoint label spaces with a shared trunk and heterogeneous heads.`**
