# Oliveira, Gonçalves and Von Zuben (2019) — Group LASSO with Asymmetric Structure Estimation

## Citation

Saullo H. G. de Oliveira, André R. Gonçalves, and Fernando J. Von Zuben. “Group LASSO with Asymmetric Structure Estimation for Multi-Task Learning.” *Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI-19)*, pp. 3202–3208, 2019. DOI 10.24963/ijcai.2019/444.

Primary source: https://www.ijcai.org/Proceedings/2019/0444.pdf

Verified against that PDF (Eqs. (1)–(5); Algorithm 1; Fig. 1–4; Tables 1–2). The journal extension is Oliveira, Gonçalves and Von Zuben, “Asymmetric Multi-Task Learning with Local Transference,” *ACM TKDD* 16(5), Article 99, 2022, DOI 10.1145/3514252 — paywalled and **not** verified here; it is listed as an unverified lead. Code: https://github.com/shgo/gamtl

## Problem

Group LASSO in MTL assumes that if a covariate group is irrelevant for one task it is irrelevant for all tasks, which implicitly assumes all tasks are related and permits negative transfer. Existing structure-estimating methods (MTRL, AMTL, MSSL) either force a *symmetric* task relation or relate tasks globally. The paper wants an asymmetric relation estimated separately for each group of covariates.

## Mathematical assumption

For each group `g` of covariates, each task parameter restricted to that group is a sparse non-negative linear combination of the other tasks’ parameters in the same group:

`w_t^g ≈ W^g b_t^g` for all `g ∈ G`, with `b_t^g ≥ 0`, `b_{tt}^g = 0`, `Σ_g w_t^g = w_t`.

`b_{ij}^g` is how much task `i` contributes to task `j` in group `g`; rows encode outgoing, columns incoming transfer. The transference structure is therefore *per-group and directed*.

## Relation representation

One learned directed relationship matrix `B^g ∈ R^{T×T}` per covariate group — a family of directed task graphs, one per feature group. The full objective (Eq. (1)) is

`min_{W, B^g} Σ_{t∈T} Σ_{g∈G} (1 + λ₁‖b_t^g‖₁) L(w_t) / m_t + λ₂ Σ_{t,g} ‖w_t^g − W^g b_t^g‖₂² + λ₃ Σ_g d_g ‖w_t^g‖₂`,

`d_g ≍ √|g|`, plus the constraints above (and a variant `GAMTLnr` without the non-negativity restriction on `B`). The loss term again scales the row-`ℓ₁` penalty: a high-loss task pays more to have outgoing edges. The third term is the latent Group LASSO regulariser.

**Attachment in our model.** The coupling parameters are three `3 × 3` matrices `B^g` (one per feature group), plus the group partition itself. The method requires (i) one parameter vector per task per group and (ii) a *feature-group structure shared by all tasks*. Our heads have widths 12 / 1251 / 4, so a per-task vector again exists only as the class-mean head summary over the shared 2000-dim hidden layer; the group partition would then have to be imposed on those 2000 hidden units (e.g. contiguous blocks), which is an invention — in the paper the groups come from the application (brain ROIs).

## Optimization method

Biconvex, solved by alternating optimization (Algorithm 1): fix `B^g`, solve for each `w_t` with FISTA (smooth loss + reconstruction terms, proximal operator = block soft-thresholding for the group `ℓ₂` term); fix `W`, solve for each `b_t^g` as an adaptive-LASSO-type problem with non-negativity constraints, using ADMM (matrix/z soft-thresholding + projection onto the non-negative orthant, `x`-update in closed form by Cholesky). Complexity `O(T³Gn + T²Gn²)` per iteration. Hyperparameters `λ₁, λ₂` recommended in a similar range, `λ₃` chosen independently by cross-validation.

## Evidence

Synthetic: 8 regression tasks, 2 covariate groups, four tasks easy (`σ = 0.3`) and four hard (`σ = 0.9`), 30–100 samples; GAMTL best NMSE at `m ≤ 70`; the recovered `B^g` shows low-cost→high-cost transfer, and the two groups give different directed matrices. Real: ADNI (816 subjects, 116 ROI groups, 5 cognitive-score regression tasks); NMSE — GAMTL **0.774** < LASSO 0.787 < MTRL 0.798 < MT-SGL 0.809 < MTFL 0.814 < AMTL 0.887 < Group LASSO 1.005 (Mann-Whitney `p < 0.05`); per-task, LASSO wins RAVLT TOTAL and T30. All tasks share one input matrix `X`.

## Assumptions

A single shared covariate space partitioned into meaningful groups given by the application; one real-valued (regression) or binary output per task so that all task parameters have the same length; overlapping groups handled by column duplication; non-negative (or unconstrained, in the variant) transfer weights; sample sizes may differ (`1/m_t` normalisation).

## Differences from our setting

* **No feature-group structure.** Our representation is a fixed 2000-dim wavCSE/WavLM-derived hidden layer with no semantically meaningful partition; imposing an arbitrary block partition would make the paper’s distinctive “local transference” claim untestable.
* **Same parameter-alignment problem as AMTL.** Task parameters must be comparable and of equal length — satisfied only through the class-mean head summary, not by the published objects. Additionally, the paper’s tasks are single-output regression/binary problems, while KS/SI/ER are 12 / 1251 / 4-class.
* **Disjoint datasets** are fine (per-task loss with per-task normalisation), but joint training assumes each task’s parameter vector is reconstructible from the others in *every* group, i.e. relatively strong relatedness.

## Implementation difficulty

Moderate to high: FISTA + ADMM sub-steps inside an alternating loop, plus a choice of group structure and of three hyperparameters (`λ₁, λ₂, λ₃`). Under a single optimizer the pragmatic matched-pair form is the same as AMTL’s — periodic re-solve of `B^g` from the current per-task summaries — with the loss-weighted row-`ℓ₁` handled by the soft-thresholding step. **Documented deviation:** per-task vectors are class-mean head summaries *and* the feature groups must be invented rather than taken from the domain.

## Candidate Study ID

`LT-0002` — screened candidate (family A). If adopted, one `TR-xxxx` variant Study is pre-registered per DEC-0013 under the shared matched protocol.

## LT-0002 assessment

* **Gate 1 — explicit relation object: PASS.** Directed, learned per-group relationship matrices `B^g`; no user-supplied relation.
* **Gate 2 — taxonomy: PASS.** Explicit parameter-reconstruction relation learning; the Group LASSO term is a sparsity prior on the task parameters, not a task-grouping or clustering mechanism (the grouping that exists is over *covariates*, given by the application), and there is no low-rank factorisation of `W`.
* **Gate 3 — heterogeneous heads: PASS ONLY WITH STATED CHANGE.** Needs equal-length per-task vectors and a shared covariate partition; the stated change (class-mean head summary over the 2000 hidden units plus an imposed block partition) restores the form but the partition is not domain-given as in every experiment in the paper.
* **Gate 4 — fixed representation: PASS.** Acts on downstream parameters only; frozen embeddings untouched.
* **Gate 5 — faithful implementability: PASS.** Eqs. (1)–(5) and Algorithm 1 are implementable as published with the paper’s hyperparameter guidance; no foreign weighting is attached (the loss-scaled row `ℓ₁` is the paper’s own mechanism, inherited from AMTL).
* **Gate 6 — source verified: PASS.** Venue, authors, equations and evidence checked against the official IJCAI PDF. The 2022 TKDD extension was not accessible and is not relied on.

**Verdict: `PASS WITH DOCUMENTED DEVIATION`** — implementable and in-category, but it requires two deviations in our setting (summary-based task parameters and an invented feature-group partition). Per LT-0002 a documented deviation is not a faithful implementation. Relative to AMTL it adds a group dimension whose scientific value here is doubtful, since our groups would be arbitrary.
