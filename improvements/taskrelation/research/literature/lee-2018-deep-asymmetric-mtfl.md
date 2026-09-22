# Lee, Yang and Hwang (2018) — Deep Asymmetric Multi-task Feature Learning

## Citation

Hae Beom Lee, Eunho Yang, and Sung Ju Hwang. “Deep Asymmetric Multi-task Feature Learning.” *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:2956–2964, 2018.

Primary source: https://proceedings.mlr.press/v80/lee18d.html (PDF: https://proceedings.mlr.press/v80/lee18d/lee18d.pdf)

Same work as the 2017 preprint arXiv:1708.00260; TP-AMTL (AAAI-21) cites it as “Lee, Yang, and Hwang (2017)”. Verified against the PMLR PDF (abstract; Eqs. (4)–(7); Tables 1–2; Figs. 1–5).

## Problem

Inter-task parameter transfer (AMTL, Lee et al. 2016) fails when tasks are only *partially* related — a task parameter need not be reconstructible from other task parameters — it grows quadratically in `T`, and it does not produce reusable features. The paper wants asymmetric transfer that is scalable and deep.

## Mathematical assumption

Tasks share a latent basis (`W = LS`, `L ∈ R^{d×k}`, `S ∈ R^{k×T}`) and the shared features `Z = σ(XL)` should be reconstructible from the task outputs through an *asymmetric* task-to-basis transfer `A ∈ R^{T×k}`:

`min Σ_t { (1 + ‖a_t^o‖₁) L(L, s_t; X_t, y_t) + ‖s_t‖₁ } + ‖Z − σ(Z S A)‖²_F + ‖L‖²_F`,

with the deep form (Eq. (7)) placing the auto-encoding term on the penultimate activation. `a_t^o` is the outgoing row for task `t`, so a task’s loss again scales how much it may contribute to the bases.

## Relation representation

An asymmetric `T × k` task-to-basis transfer matrix `A` (how much each task contributes to each shared basis), estimated from data, plus a sparse per-task coefficient `s_t` over bases. There is *no* `T × T` task-task relation object: the “relation” couples tasks to latent bases inside a low-rank decomposition. `A` is by construction non-symmetric, but the sharing mechanism is the shared-basis low-rank model `W = LS`.

## Optimization method

Joint SGD/backprop over `L`, `S` and `A` (no alternating scheme needed once the auto-encoding term is differentiable); ReLU nonlinearity in both the network and the reconstruction; task losses scaled by `√N_t`; hyperparameters (learning rate, `β`, `w`) searched over `{1, 0.1, 10⁻², 10⁻³, 10⁻⁴}`; `β = w` to reduce the search space. Multi-class problems are handled as one-vs-all binary tasks so that all tasks have identical output dimension.

## Evidence

Synthetic basis-recovery study (12 tasks, easy/hard groups, overlap bases): the learned transfer shows transfer from easy to hard and none the reverse, with better basis and parameter recovery than GO-MTL and AMTL. Real data, shallow: AWA-A, MNIST, School, Room — AMTFL best overall (School is a near-tie and GO-MTL wins there). Deep models (multi-head CNN/ResNet): MNIST-Imbalanced, CUB-200, AWA-C, ImageNet-Small(352 classes) — Deep-AMTFL outperforms CNN, MT-CNN and Deep-AMTL, with the largest gains on imbalanced data; an ablation shows errors decrease as the number of tasks grows.

## Assumptions

All tasks live in a shared input space and pass through one (deep) network trunk; the representation `Z` is *trained*, not fixed; the asymmetric autoencoder is defined on penultimate activations; partially related tasks; classification tasks must have a common output dimension (one-vs-all) for the latent-basis formulation to be applied uniformly.

## Differences from our setting

* **Taxonomy.** This is a low-rank / shared-basis feature-learning method with an asymmetric weighting on the basis contribution. The project’s binding taxonomy places low-rank and decomposition models outside Zhang & Yang §2.4, so as published it is an out-of-category near-miss, not an asymmetric task-relation method. Its own related-work section contrasts it with AMTL precisely on this axis.
* **Heads.** The one-vs-all construction is what equalises output widths; our three tasks have disjoint datasets and disjoint label spaces (12 / 1251 / 4 classes), so no shared one-vs-all task set exists. Nothing in the paper transfers across tasks with different output dimensions.
* **Attachment.** `Z` would have to be the 2000-dim trunk activation, which makes `A ∈ R^{3×2000}` dimensionally possible, but then `k = 2000` and `L` is the frozen-upstream projection already fixed by the protocol; the reconstruction target would be a trunk activation produced by a trunk that is itself the object the paper intends to learn. The characteristic feature-learning mechanism has no place to live under a frozen upstream.
* **Temporal / clinical variants.** TP-AMTL (AAAI-21) builds on this line with uncertainty instead of loss; see the separate card.

## Implementation difficulty

High, and not faithful. Implementing the relation object `A` requires also implementing basis learning (`L`, `S`, `k` selection) and the auto-encoding term on hidden activations — i.e. building a different model rather than porting this method. There is no way to attach `A` to a three-head model whose representation is a single shared trunk: the ambiguous factorisation and the missing shared output space are intrinsic.

## Candidate Study ID

`LT-0002` — screened candidate (family A), recorded as a category failure.

## LT-0002 assessment

* **Gate 1 — explicit relation object: PARTIAL.** `A` is explicitly parameterised and learned, but it is a task-to-basis relation, not a task-task (or task-parameter) directed relation; there is no transfer graph over tasks.
* **Gate 2 — taxonomy: FAIL.** Low-rank / shared-basis feature learning with an asymmetric weighting; the project excludes low-rank and decomposition methods and treats the asymmetric *weighting* of a shared basis as out of §2.4.
* **Gate 3 — heterogeneous heads: FAIL.** Requires equal-width task outputs (one-vs-all) over a shared input space; the 12 / 1251 / 4-class heads on disjoint datasets cannot supply this.
* **Gate 4 — fixed representation: PASS/PARTIAL.** The frozen wavCSE embedding is untouched, but the method is defined on a trainable representation, so its characteristic mechanism is vacuous when only the small trunk is trainable.
* **Gate 5 — faithful implementability: FAIL.** A faithful implementation would require re-instating latent-basis learning and a shared output space — a new model, not this paper.
* **Gate 6 — source verified: PASS.** Venue, authors, equations and evidence checked against the PMLR PDF.

**Verdict: `FAIL — low-rank/shared-basis feature learning, not Task Relation Learning, and its equal-output-width, shared-input assumption does not survive our disjoint, heterogeneous-head setting.`**
