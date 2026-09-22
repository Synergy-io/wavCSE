# LT-0002 analysis — published relation-learning variants for two selected families

**Decision: `CANDIDATES_FOUND`.** Both families returned verified candidates, but only family B returned one that passes *every* gate without a declared deviation. The deviation question in family A is a human call and is stated explicitly in §4 below.

Method: two independent primary-source passes (one per family) against the six eligibility gates in `PLAN.md`. Every author, venue, year and equation was checked against the publication; the controller re-verified the two recommended candidates directly against their PDFs (MSSL: JMLR 17(33) Eq. 3, 8, 10a–c; AMTL: ICML 2016 Eq. 1–3 and Theorem 1). Cards are in `../literature/`; the registry is `../literature/INDEX.md`.

## 1. What the two passes found

| Family | Verified | Clean pass | Deviation-class | Failed gates | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| A — asymmetric / directed relations | 9 | 0 | 3 | 6 | candidates found, all requiring a declared deviation |
| B — relation estimator / parameter representation | 6 | 1 | 1 | 4 | one clean candidate |

## 2. Family B — one clean candidate

**Gonçalves, Von Zuben & Banerjee (2016), *Multi-task Sparse Structure Learning*, JMLR 17(33):1–30 — p-MSSL — `PASS`.**

Its objective (Eq. 3) is

$$\min_{W,\Omega\succ0}\ \sum_{k}\tfrac{1}{n_k}\sum_i \big(w_k^\top x_{ik}-y_{ik}\big)^2+\lambda_0\,\mathrm{tr}(W\Omega W^\top)-d\log|\Omega|+\lambda_1\|W\|_1+\lambda_2\|\Omega\|_1,$$

i.e. **the same coupling term shape as classical MTRL** — but with the relation object being a *sparse precision* estimated by an alternating step that is exactly the graphical-lasso problem (`Ω = argmin tr(SΩ) − d log|Ω| + λ₂‖Ω‖₁`, `S = (1/d)WᵀW`) solved by ADMM (SVD update + soft-thresholding). Classification enters through the paper's own GLM/logistic extension (Eq. 12–13), and per-task sample imbalance is handled by the paper's own `1/n_k` loss scaling.

Why it matters here: our classical MTRL Ω comes from the closed-form `(WᵀW)^{1/2}` normalisation and **saturates** near ±1/3 (F7, F5, F9), destroying pair-specific information. MSSL's paper states the contrast explicitly — they learn the inverse covariance directly because it "tends to be more stable than computing covariance and then inverting it". The `−d log|Ω|` barrier with `ℓ₁` shrinkage is a bounded, regularised precision parameterisation rather than a normalised covariance, so it targets precisely our observed failure.

Assumption fit: the estimator consumes a `d × m` parameter matrix. The project's mean-head-summary convention supplies exactly that, and **it is the same adapter the in-category control already requires**, so the comparison isolates the estimator. Caveat recorded in the card: the Gaussian-row premise is an approximation of the same kind as the control's matrix-normal premise; MSSL regularises around rank-degeneracy rather than removing it.

Implementation delta from the existing `01-mtrl` arm is small: same summary matrix, same coupling term, replaced Ω update.

## 3. Family A — three candidates, all deviation-class

| Candidate | Relation object | Why it deviates |
| --- | --- | --- |
| **Lee, Yang & Hwang (2016), AMTL**, ICML, PMLR 48:230–238 | Sparse **non-negative directed** transfer graph `B`, rows `ℓ₁`-scaled by the *task's own loss*; `B⪰0`, `B_tt=0` | The published object is each task's parameter vector `w_t`; our heads (12/1251/4 classes) have no aligned columns, so the directed relation can only act on 2000-d class-mean summaries. |
| **Zhou & Yang (2023), AutoTR**, KDD, 3570–3580 | Signed directed relation matrix `R` (`r_kk=0`), closed-form soft-threshold update | Same summary-only attachment; its temporal semantics are also inapplicable (we use its `AutoTR-0` zero-init variant). |
| **Oliveira, Gonçalves & Von Zuben (2019), GAMTL**, IJCAI, 3202–3208 | One directed non-negative matrix per covariate group | Same summary-only attachment **plus** a covariate-group partition that has no counterpart in our frozen-embedding representation and would have to be invented — the paper's "local transference" claim then becomes untestable here. |

AMTL is the family's canonical member (its relation object recurs in AutoTR, both GAMTLs and Deep-AMTFL) and is the best-motivated for **our** known asymmetry: the paper's mechanism is designed for exactly our situation — "large imbalance in the number of training instances across tasks" — and it re-weights the *relation* by task loss, with an optional `c_t = 1/√n_t` correction for the overfitting of small-sample tasks. ER contributes ≈47 of every 2048 sampled examples, the smallest by an order of magnitude.

Theorem 1 is also a testable prediction for us: at any local optimum the lower-loss task carries the larger outgoing transfer. We can check the learned 3×3 `B` against our measured per-task losses instead of only reading accuracy.

## 4. The decision the human must make

LT-0002's pre-registered falsification rule said a documented deviation **counts as a faithfulness fail**. Applied literally, family A has **no admissible arm** — the whole asymmetric family would be closed for this architecture, and only MSSL would run.

There is a genuine counter-argument, which is why this is not being decided autonomously:

* the deviation is **shared with the in-category control**: classical MTRL in this repository already regularises class-mean head summaries, and its gradient already applies a uniform pull across the rows of each head. The adapter is therefore a property of the benchmark, held constant across arms, not a differential confound;
* the variants' *relation objects and update rules* are implemented as published — nothing is glued to an unrelated weighting scheme;
* what is genuinely lost is **external validity**: we would not be reproducing AMTL's own published setting, and the write-up must say so, including the extra uniform-row effect that the paper does not contemplate.

Options: **(i)** admit deviation-class arms, labelled `published mechanism + project summary adapter (same adapter as the control)` and never claimed as a faithful reproduction; **(ii)** hold the literal rule, close family A, and run only MSSL; **(iii)** hold the rule and instead spend the asymmetric budget on a *relabelled* question (does a directed relation help at all under this adapter), which is a project-original mechanism and therefore requires the Option-3 authorization path.

## 5. Failed candidates worth remembering

Recorded so they are not rediscovered: three different papers are named "GAMTL" and only one is in-family (the other two are grouping- and undirected-object methods); *Zhang & Yeung's TKDD 2014 "asymmetric" MTRL is symmetric* — its asymmetry is the target/source *setting*, not a directed relation object, and a keyword search surfaces it first (a trap this project has fallen into before); Deep-AMTFL and TP-AMTL are low-rank/basis and uncertainty-attention methods respectively, outside the taxonomy; Bonilla's multi-task GP needs a multi-class likelihood replacement to port, which is a hybrid rather than a faithful implementation.

## 6. Recommended arm order

1. **MSSL (family B, clean pass)** — first arm. Smallest implementation delta from the existing control, directly targets the saturation failure we measured, and needs no decision from the human to be admissible.
2. **AMTL (family A, deviation-class)** — second arm, *if* the human admits deviation-class arms. Best-motivated for our sample-size imbalance and yields a testable relation prediction (Theorem 1) beyond accuracy.
3. **AutoTR** — third, only if AMTL's directed mechanism shows something worth separating from its loss-scaled sparsity (AutoTR is the same relation object minus the loss scaling, i.e. a clean ablation of that ingredient).

All arms run under `../VARIANT_BENCHMARK_PROTOCOL.md`, against both controls, screen at one seed then confirm at seeds 0–4.

Provenance: `studies/LT-0002/{PLAN.md,NOTE.md}`; cards under `research/literature/`; primary sources verified 2026-09-22 (JMLR 17(33), PMLR 48:230–238, KDD 2023 3570–3580, IJCAI-19 3202–3208, NeurIPS 2007/2010/2021, UAI 2020, ICML 2007/2018, AAAI-21, arXiv:2009.05618, arXiv:2410.15875).
