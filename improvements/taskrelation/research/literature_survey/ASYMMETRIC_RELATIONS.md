# Asymmetric and directed task relations — direction semantics, equations, and what would count as proof

**Type:** literature / theory note. No implementation, no configurations, no runs, no canonical records.
**Scope:** the *asymmetric / directed relation* family only. Binding benchmark conditions live in
`improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md`; binding scope decisions live in
`improvements/taskrelation/research/DECISIONS.md`; numbers live in `improvements/taskrelation/research/FINDINGS.md`.
**Head geometry assumed throughout:** three tasks `ks_si_er` with classifier widths **12 / 1251 / 4**, over a
**frozen** `wavlm_large` representation, frame pooling `mean`, layer pooling `smp` `0.5` over **all 25** layers
(`VARIANT_BENCHMARK_PROTOCOL.md` §1). `m = T = 3`, so any `3×3` relation matrix has **six** off-diagonal
entries; its symmetric counterpart has three.

**Evidence tags used below:** `[OBSERVED]` = read from the cited primary source or repository artifact;
`[INFERRED]` = derived by this note from cited algebra/code; `[HYPOTHESIZED]` = a proposed statement with no
supporting measurement in this repository.

---

## 0. Why this note exists

Three different things are routinely called "asymmetric multi-task learning", and only one of them is a
**directed relation object**:

1. a **directed relation matrix** that reconstructs each task's parameters from the others (a *graph*, in
   general not symmetric, and its directedness is a property of the *operator on parameters*, not of a
   covariance);
2. an **asymmetric covariance** — in practice an ill-named *setting* ("target task arrives after source
   tasks") whose relation object is still a symmetric PSD matrix;
3. **loss balancing** — per-task scalar weights or gradient surgery, with no relation object at all.

Confusing (1) with (2) or (3) is the failure mode this note is meant to prevent. The distinction is
category-defining under the project's binding taxonomy: only (1) is Task Relation Learning in the
Zhang & Yang §2.4 sense; (2) is also §2.4 but symmetric; (3) is optimization, explicitly outside the
family (`improvements/taskrelation/research/literature/zhang-yang-2021-mtl-survey.md`; DEC-0005 §3).

A second purpose: the repository's own directed-transfer evidence has been **withdrawn as a motivation**
(DG-0001 → F8). §9 states precisely what remains and what does not.

---

## 1. The three objects, disambiguated

### 1.1 Directed relation matrix (a reconstruction graph)

The family A objects have the common form

```
w_t  ≈  Σ_{s ≠ t} B_{s t} w_s          (equivalently  W ≈ W B)
```

with `W = [w_1 … w_T] ∈ R^{d×T}`, `B ∈ R^{T×T}`, `B_tt = 0`. `B` is a weighted **adjacency matrix of a
directed graph** over tasks; `B_st ≠ B_ts` in general `[OBSERVED]` (AMTL §3; AutoTR; GAMTL §3).

Properties that matter here:

* **Directedness is real but its *induced regularizer* is symmetric.** The coupling penalty is a squared
  reconstruction error,
  `‖W − WB‖²_F = tr(W (I−B)(I−B)ᵀ Wᵀ)`.
  The operator `(I−B)(I−B)ᵀ` is symmetric PSD **by construction**, whatever `B` is. The Zhang & Yang survey
  states this for AMTL in §2.4: *"Though A is asymmetric, from the perspective of the regularizer, the task
  relations here are symmetric and act as the task precision matrix with a restrictive form."*
  `[OBSERVED — arXiv:1707.08114v3 §2.4, Eq. (25) discussion]`. `[INFERRED]`: the same identity holds for
  AutoTR and GAMTL, whose penalties are the same shape (`‖W − WR‖²_F` verified in AutoTR's own
  `update_W.m`, which computes `grad_W += rho1 * 2 * W * T * (T')` with `T = I − R`; `ρ₁‖W − WR‖²_F`).
* **So a directed method is a *restricted symmetric precision* model, not a non-symmetric regularizer.**
  Writing `Ω⁻¹ := (I−B)(I−B)ᵀ` puts AMTL/AutoTR/GAMTL inside classical MTRL's `tr(WΩ⁻¹Wᵀ)` form, with the
  extra structure that `Ω⁻¹` must factor as a squared sparse (and possibly sign-constrained) directed
  operator `[INFERRED]`. What the directed family buys is therefore **not** a non-symmetric penalty; it is
  (i) a *factorised, sparse, interpretable* precision whose edges have a direction, and (ii) an explicit
  hypothesis about **which direction** carries transfer (`B_st ≠ B_ts`).
* **Any direction claim must therefore be tested on the object and on behaviour separately** (§8). A
  penalty with a symmetric operator can still produce a directed graph object.

### 1.2 Asymmetric *covariance* — a setting, not an object

* **Zhang & Yeung 2014 (TKDD 8(3):12)**, the canonical "asymmetric MTRL" hit, keeps a **symmetric PSD task
  covariance `Ω`** throughout. Its "asymmetric setting" is *target-after-sources*: sources are learnt
  symmetrically, then one target task is incorporated via a covariance **vector** `ω` and an augmented
  `Ω̃ = [[(1−σ)Ω, ω],[ωᵀ, σ]]`, which is symmetric by construction. `[OBSERVED —
  improvements/taskrelation/research/literature/zhang-yeung-2014-mtrl-asymmetric.md, verified against
  https://yuzhanghk.github.io/papers/Zhang_Yeung_TKDD14.pdf §2.3/§4.6, Eq. (16)]`
* Zhang & Yang's survey places this whole line under MTRL (its Eq. (21)), i.e. symmetric
  `tr(WΩ⁻¹Wᵀ)`. `[OBSERVED — arXiv:1707.08114v3 §2.4]`
* Practical consequence: a keyword search for "asymmetric MTL" surfaces this paper first and its
  relation object is **symmetric**; this is a recorded attribution trap in this project
  (`literature/zhang-yeung-2014-mtrl-asymmetric.md`, "Purpose of recording it").

### 1.3 Loss balancing (no relation object)

Per-task scalars — Kendall-style homoscedastic uncertainty weighting, GradNorm, dynamic weight averaging,
gradient surgery (PCGrad) — learn *weights or gradient transformations*, not a relation over tasks.
The survey separates them explicitly: §2.10 lists GradNorm as an **optimization technique**, and §2.4's
Task Relation Learning is defined by a learned quantitative relation object.
`[OBSERVED — arXiv:1707.08114v3 §2.4, §2.10; improvements/taskrelation/research/literature/INDEX.md]`

**The overlap that causes the confusion.** AMTL's loss enters its *relation update* — the row-`ℓ₁` sparsity
of `B` is scaled by the task's own loss (`(1 + μ‖b_t^o‖₁) L(w_t; D_t)`). A superficial reading calls this
"loss weighting". It is not: the loss scales **a penalty over edges**; the object being learnt is still the
directed graph `B`, and no per-task loss weight is created. `[OBSERVED — AMTL Eq. (1); LT-0002 card Gate 2]`
Conversely, adding a per-task scalar weight to a directed arm (a "loss-balancing control") is a diagnostic
comparison, **not** a way to import a relation mechanism; DEC-0009 §4 declined broadening to GradNorm-style
methods. `[OBSERVED — improvements/taskrelation/research/DECISIONS.md DEC-0009]`

### 1.4 Boundary table

| Object | Directed? | Symmetric by construction? | Category (survey) | Example |
|---|---|---|---|---|
| Directed reconstruction graph `B` (`W ≈ WB`) | yes (`B_st ≠ B_ts`) | no (object); **yes** for induced `(I−B)(I−B)ᵀ` | §2.4 Task Relation Learning | AMTL, AutoTR, GAMTL |
| Symmetric task covariance/precision `Ω` | no | yes | §2.4 | MTRL; MSSL (precision); SPATS |
| "Asymmetric setting" covariance `Ω̃`, `ω` | no | yes | §2.4 | Zhang & Yeung 2014 |
| Per-task scalar weights / gradient surgery | n/a | n/a | §2.10 optimization | GradNorm, uncertainty weighting, PCGrad |
| Task-to-basis matrix inside a low-rank model | asymmetric object | basis is shared | §2.2 low-rank / §2.5 decomposition | Deep-AMTFL (Lee et al. 2018) |
| Cross-stitch `α` matrix on activations | in general not symmetric | no constraint | §2.1 feature learning | Misra et al. 2016 (§2.4 of this note) |

---

## 2. Primary sources (family A), verified

All three family-A methods were checked against a **primary** artifact while this note was written — the
papers themselves for AMTL and GAMTL, and the authors' official released implementation for AutoTR (whose
camera-ready PDF could not be retrieved). Where a source could not be read in full, the gap is stated
explicitly in §10.

### 2.1 Lee, Yang & Hwang (2016) — AMTL

**Citation.** Giwoong Lee, Eunho Yang, Sung Ju Hwang. "Asymmetric Multi-task Learning Based on Task
Relatedness and Loss." ICML 2016, PMLR 48:230–238.
PDF: <https://proceedings.mlr.press/v48/leeb16.pdf> (note: **not** `leea16.pdf`, an unrelated Hawkes-process
paper). `[OBSERVED — PDF read in full; title, Eq. (1)–(5), Theorem 1 p. 3, Algorithm 1 p. 3, Algorithm 2
p. 5, Tables 1–2, Fig. 1–3]`

**Model.** `w_t ≈ Σ_{s=1}^{T} B_{st} w_s` for all `t`, with `B` a `T×T` **asymmetric** matrix; `B_st` is
"the positive weight of basis `w_s` in representing `w_t`"; `B_tt = 0` by definition. Row vector
`b_t^o ∈ R^{T−1}` = row `t` of `B` (diagonal removed) = *outgoing* transfers from task `t`. The paper
assumes tasks share one data space and one model space and are positively correlated. `[OBSERVED — §3]`

**Objective (Eq. (1)).**

```
min_{W, B ≥ 0}  Σ_t (1 + μ ‖b_t^o‖₁) · L(w_t ; D_t)
              + λ Σ_t ‖ w_t − Σ_s B_st w_s ‖₂²
```

with element-wise non-negativity `B ≥ 0` as an extra constraint, and `μ, λ` tuning parameters. `λ → 0`
gives `B* = 0` (independent tasks); `λ → ∞` forces dense `B`. `[OBSERVED — Eq. (1)]`

**Optimization (Algorithm 1, alternating / biconvex).**

* **W-step (Eq. (2)):** fix `B`, minimise the same objective over `W` — by (block-)gradient descent, since
  the loss is only assumed convex in `W`.
* **B-step (Eq. (3)):** fix `W`; the problem decomposes over the **columns** of `B`. With
  `b^i_t := (B_1t,…,B_Tt)ᵀ` the incoming-edge vector and `Λ := diag_t L(w_t; D_t)`,

  ```
  min_{b ≥ 0}  μ ‖Λ b^i_t‖₁ + λ ‖ w_t − W b^i_t ‖₂²
  ```

  a **non-negative weighted LASSO**, solved with the weighted-lasso solver of Mairal et al. (2010).
  `[OBSERVED — Eqs. (2)–(3), Algorithm 1]`
* The problem is **not** jointly convex; it is biconvex, and the paper cites Gorski et al. (2007) for the
  partial-optimality property of stationary points. `[OBSERVED — proof sketch, p. 3]`

**Theorem 1 (p. 3) — the direction claim, exactly.** For **any** local optimum,
for any `t,u` with `‖b*^o_t‖₁ > ‖b*^o_u‖₁`, **either**

* **(a)** `L(w*_t; D_t) ≤ L(w*_u; D_u)`, **or**
* **(b)** for **any** vectors `b^o_t, b^o_u` with `‖b^o_t‖₁ = ‖b̌^o_u‖₁` and `‖b^o_u‖₁ = ‖b̌^o_t‖₁`, the
  regularisation term satisfies
  `R(W; … , b̌^o_t, … , b̌^o_u, …) ≤ R(W; … , b^o_t, … , b^o_u, …)`.

The paper's own gloss: *"the task with smaller loss will have larger amounts of transfer **as long as it
does not hurt the structural constraint** in the second term of (1)."* `[OBSERVED — Theorem 1 and the
sentence immediately following]`

Consequences that must not be dropped:

* The claim is about **local optima**, not global minima, and is a *disjunction*. Alternative (b) is a
  reconstruction-penalty condition; **"lower loss ⇒ larger outgoing edges" is not unconditional.**
  `[OBSERVED]`
* The edge direction it predicts is **loss-gap driven**; it says nothing about edges between two tasks with
  equal loss. `[INFERRED]`
* The theorem is stated for the *unconstrained* structural side; the paper separately adds `B ≥ 0`.
  `[OBSERVED]`

**Small-sample caveat and re-weighting.** The paper itself names the risk of transfer *from* an overfitted
small-`n_t` task, and proposes `L'(w_t; D_t) = c_t L(w_t; D_t)` with `c_t = 1/√n_t` (AMTL-imbalanced).
`[OBSERVED — §3 discussion + §4.3]`

**Evidence.** Synthetic 12-task regression, two groups, two noise levels: low-noise tasks acquire outgoing
edges to high-noise tasks and not vice versa (Fig. 1(c)). Real: MNIST, USPS, School (139 schools), AWA.
Beats STL, MTFL, GO-MTL, SC-MTL, Curriculum-simple, a symmetric MTRL variant (SMTL) and an ablation without
loss scaling (AMTL-noLoss); School is the one dataset where GO-MTL wins; runtime AMTL ≈ 167 s vs
AMTL-Curriculum ≈ 7.7 s on the synthetic set. `[OBSERVED — Tables 1–2, Fig. 1–3]` The paper reports larger
gains on datasets with "large imbalance in the number of training instances across tasks".
`[OBSERVED — abstract, §4.3]`

### 2.2 Zhou & Yang (2023) — AutoTR

**Citation.** Menghui Zhou, Po Yang. "Automatic Temporal Relation in Multi-Task Learning." KDD '23,
pp. 3570–3580. DOI 10.1145/3580305.3599261. Author copy:
<https://eprints.whiterose.ac.uk/id/eprint/199291/8/AutoTR_KDD_Camera_Ready_v4.pdf>; code:
<https://github.com/menghui-zhou/AutoTR>. `[OBSERVED — DOI record, repo, and the repo's MATLAB sources;
see §10 for the PDF retrieval gap]`

**Model.** `w_k ≈ Σ_{x≠k} r_{x,k} w_x`, i.e. `W ≈ W R`, `R ∈ R^{m×m}`, `r_{k,k} = 0`; `R` is **free and
signed** (no non-negativity), and the directed structure is *learned* rather than assumed. `[OBSERVED —
LT-0002 card; corroborated by `update_R.m`, whose proximal step is `max(0, |R| − ρ S) .* sign(R)`]`

**Objective** (`AutoTemporal.m`, verbatim comment block):

```
argmin_{W,R}  Σ_i 0.5 ‖Y{i} − X{i}' W(:,i)‖²
            + ρ₁ ‖W − W R‖_F²
            + ρ₂ ‖R .* S‖₁ ,      S = (λ₃ − 1) I_m + 1_{m×m}
```

`[OBSERVED — AutoTemporal.m; the LT-0002 card writes the same objective with the paper's λ₁, λ₂ and
‖R⊙S‖_{1,1}, s = 10⁹]`. The `(λ₃ − 1)` diagonal makes the diagonal of `S` dominate, so the `ℓ₁` term
**suppresses the diagonal** rather than penalising it (the paper calls `s` a pseudo-hyperparameter that
only needs to be "large enough"). `[OBSERVED — card; `S` construction in code]`

**Optimization (alternating; accelerated proximal gradient).**

* **R-step:** closed form, `R ← max(0, |R| − ρ S) ⊙ sign(R)` with `ρ = ρ₂ / (2 ρ₁)`
  (`update_R.m`: `rho = rho2/rho1/2`; `R_Projected_Gradient`). Cost `O(m²)`. The smooth gradient is
  `WᵀW (R − I)`. `[OBSERVED — update_R.m]`
* **W-step:** APG/FISTA-style with backtracking; gradient
  `grad_W += ρ₁ · 2 · W (I − R)(I − R)ᵀ` (`update_W.m`, with `T = eye − R`). This is the identity of
  §1.1 read directly off the authors' code. `[OBSERVED — update_W.m]`
* **Warm start:** Gaussian kernel `exp(−|row − col|)` with column normalisation; the zero-init variant
  (`R_ini = 0`) is the paper's `AutoTR-0`, the only applicable variant for a non-temporal task set.
  `[OBSERVED — AutoTemporal.m; card]`

**Evidence.** Six public datasets, `m = 12` time points, 314 features (Alzheimer's progression, Weather).
Baselines Ridge, Lasso, TaskTS, FeaTS, MeanTR. Learned `R` is non-symmetric in every dataset, with a few
small negative entries. **No experiment uses heterogeneous output widths or disjoint feature spaces.**
`[OBSERVED — LT-0002 card only; the paper's tables were **not** re-read this session (see §10)]`

**Attribution note** (kept from the repo record): `W ≈ WR` with an `ℓ₁`-penalised relation matrix is
formally AMTL's relation object *without* the loss-scaled sparsity and *without* non-negativity; AutoTR's
reference list does not cite Lee et al. (2016). `[OBSERVED — LT-0002 card]`

### 2.3 Oliveira, Gonçalves & Von Zuben (2019) — GAMTL

**Citation.** Saullo H. G. de Oliveira, André R. Gonçalves, Fernando J. Von Zuben. "Group LASSO with
Asymmetric Structure Estimation for Multi-Task Learning." IJCAI-19, pp. 3202–3208.
DOI 10.24963/ijcai.2019/444. PDF: <https://www.ijcai.org/Proceedings/2019/0444.pdf>. `[OBSERVED — PDF read
in full; Eq. (1)–(5), Algorithm 1, Figs. 1–4, Tables 1–2]`

**Model.** One relationship matrix **per covariate group**:
`w_t^g ≈ W^g b_t^g` for all `g ∈ G`, with `W^g` the parameter matrix restricted to group `g`, `b_t^g ≥ 0`
and `b_{tt}^g = 0`; `w_t = Σ_g w_t^g`. `b^g_{ij}` = "how much task `i` contributes to task `j` in group
`g`"; **columns = incoming, rows = outgoing** (the paper states this explicitly for `B^g`). A variant
**GAMTLnr** drops the non-negativity constraint. `[OBSERVED — §3, §4]`

**Objective (Eq. (1)).**

```
min_{W, B^g}  Σ_{t∈T} Σ_{g∈G} (1 + λ₁‖b_t^g‖₁) · L(w_t)/m_t
            + λ₂ Σ_{t,g} ‖ w_t^g − W^g b_t^g ‖₂²
            + λ₃ Σ_{t,g} d_g ‖ w_t^g ‖₂ ,    d_g ≍ √|g|
subject to  w_t = Σ_g w_t^g ,  b_t^g ≥ 0  ∀ g, t
```

`[OBSERVED — Eq. (1); the third term is the latent Group LASSO regulariser (Jacob et al. 2009)]`

**Optimization (Algorithm 1, alternating).**

* **w_t-step (Eq. (2)–(3)):** FISTA (Beck & Teboulle 2009); the non-smooth part is the group-`ℓ₂` penalty,
  whose proximal operator is block soft-thresholding. Lipschitz constant by backtracking.
* **b_t^g-step (Eq. (4)):** an Adaptive-LASSO-shaped problem solved with ADMM: soft-thresholding,
  projection onto the non-negative orthant, and a closed-form `x`-update via Cholesky.
* Complexity per iteration `O(T³Gn + T²Gn²)`. `[OBSERVED — §3.1–§3.3]`

**Evidence.** Synthetic: 8 regression tasks, 2 covariate groups, four easy (`σ = 0.3`) / four hard
(`σ = 0.9`); recovered `B^g` shows low-cost → high-cost transfer, and the two groups give **different**
directed matrices. Real: ADNI (816 subjects, 116 ROI groups, 5 cognitive-score tasks, one shared input
matrix `X`). NMSE (Table 1): GAMTL **0.774** < LASSO 0.787 < MT-SGL 0.809 < MTFL 0.814 < MTRL 0.798 <
AMTL 0.887 < Group LASSO 1.005 (Mann-Whitney `p < 0.05`); GAMTLnr 0.787. Per-task, LASSO wins TOTAL and T30.
`[OBSERVED — Tables 1–2, Fig. 3]`

**Why it matters for dimension alignment.** The group partition is **domain-given** in every experiment
(brain ROIs). The paper's distinctive "local transference" claim is about that partition; with an arbitrary
partition the claim is untestable. `[OBSERVED — §1, §4.2]`

### 2.4 Cross-stitch (Misra et al. 2016) — the feature-level near-miss

**Citation.** Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert. "Cross-stitch Networks for
Multi-task Learning." CVPR 2016.
PDF: <https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf>.
`[OBSERVED — PDF read; Eq. (1)–(3), §3.3, §4, §5, Tables 1–6. The proceedings page range is **not** verified
from the PDF and is omitted on purpose.]`

**Model (Eq. (1)).** At every spatial location `(i,j)` and layer `l`, with `x_A, x_B` the activation maps
of the two tasks:

```
[ x̃_A^ij ]   [ α_AA  α_AB ] [ x_A^ij ]
[ x̃_B^ij ] = [ α_BA  α_BB ] [ x_B^ij ]
```

`α_AB`/`α_BA` are the *different-task* values (`α_D`), `α_AA`/`α_BB` the *same-task* values (`α_S`). The
matrix is **not constrained symmetric**; setting `α_AB` or `α_BA` to zero makes a layer task-specific.
Initialisation as a convex combination (`α_S + α_D = 1`) is recommended but **not enforced**, and the units
are trained with a learning rate 10²–10³ × the base network's. `[OBSERVED — §3.3, §4, Tables 1–2]`

**Category.** The survey places cross-stitch explicitly in **§2.1 Feature Learning Approach** (feature
transformation) as "a representative model" of the third deep-MTL sub-category — *not* §2.4.
`[OBSERVED — arXiv:1707.08114v3 §2.1.1, "The last category is to learn different but related feature
representations for different tasks with the cross-stitch network [22] as a representative model"]`

**Dimension requirement.** The unit combines `x_A^{ij}` and `x_B^{ij}` **element-wise at the same location
of identically shaped maps**; one unit per channel is maintained (`pool1` → 96 units for AlexNet), with a
per-layer unit count. The paper restricts scope to "tasks which take the same single input".
`[OBSERVED — §3.3, §4, §5]`

**Evidence.** NYUv2 (SemSeg + surface normals) and PASCAL VOC08 (detection + 64 attributes); gains
concentrated on **data-starved** categories (4.6 % / 4.3 % mAP on the 10 / 20 least-labelled attributes).
`[OBSERVED — Tables 5–6, Figs. 5–6]` Relevance here: it is the ladder's "same activation shape" rung (§6.1),
and it is a *feature*-space adaptation, so it cannot be attached to a classifier head of a different width.

### 2.5 Named alternatives and the traps around them

Kept as one line each because they are surfaced by any "asymmetric MTL" search; each is already carded.

| Paper | Object | Why it is not family A | Repo record |
|---|---|---|---|
| Zhang & Yeung 2014, TKDD 8(3):12 | symmetric `Ω` / `Ω̃` | "asymmetric" = target/source **setting** | `literature/zhang-yeung-2014-mtrl-asymmetric.md` |
| Liu & Pan 2017, IJCAI | signed directed `C` + trace-Lasso | contribution is **task grouping** (clustering) | `literature/liu-2017-trace-lasso-gamtl.md` |
| Yu et al. 2020 (GAMTL name collision) | graph adjacency | **undirected by construction** | `literature/yu-2020-graph-adjacency-gamtl.md` |
| Lee, Yang & Hwang 2018, Deep-AMTFL | task-to-basis `A` in `W = LS` | **low-rank / decomposition**; needs one-vs-all equal widths | `literature/lee-2018-deep-asymmetric-mtfl.md` |
| Nguyen et al. 2021, TP-AMTL | per-pair uncertainty attention | **loss/uncertainty weighting**, temporal only | `literature/nguyen-2021-tp-amtl.md` |
| Graffeuille et al. 2024, SAAL | coefficients over auxiliary clones | **loss weighting**; its Eq. (4) is undefined here | `literature/graffeuille-2024-self-auxiliaries.md` |
| Gonçalves et al. 2016, p-MSSL (family B) | sparse **precision** `Ω` via graphical lasso | symmetric precision; the clean-pass comparator | `literature/goncalves-2016-mssl.md` |

---

## 3. Direction semantics: the conventions (get these right or the equations lie)

| Paper | Symbol | Definition as printed | Outgoing | Incoming |
|---|---|---|---|---|
| AMTL | `B_st` | weight of basis `w_s` in **representing `w_t`** | row `b_t^o` = `(B_t1,…,B_tT)` ("outgoing transfers from task `t`") | column `b^i_t` |
| AutoTR | `r_{x,k}` | relation **from task `x` to task `k`**; `w_k ≈ Σ_x r_{x,k} w_x` | row `x` of `R` | column `k` of `R` |
| GAMTL | `b^g_{ij}` | "how much task `i` contributes to task `j` in group `g`" | row `i` of `B^g` | column `j` of `B^g` |
| Cross-stitch | `α_AB` | weight of **task B's activation** in task A's new activation | row A | column B |

`[OBSERVED — AMTL §3 Fig. 1 legend ("rows denote outgoing edge weights and columns denote incoming edge
weights"); AutoTR card + `update_R.m`; GAMTL §3; Misra Eq. (1)–(3)]`

**Reading rule for this repository.** Because all three family-A papers index the *target* by the **column**
and the *source* by the **row**, "`B_st > 0`" means *`s` transfers into `t`*: a row-heavy task is a
*donor*. AMTL's Theorem 1 therefore says: the **donor** has the smaller loss. In the notation of the
repository's own transfer matrix, `T(t ← s) = acc(t with s) − acc(t alone)`, so a donor `s` should show
`T(t ← s) > 0` with a weakly negative reverse cell. `[INFERRED]`

**`m = 3` geometry.** With `T = 3` there are six free off-diagonal entries; a directed
3-cycle is representable, and AMTL allows cyclic dependencies while its curriculum variant
orders donors before recipients. `[INFERRED from AMTL §3–4]` Six coefficients
are a small parameter count, **not** evidence that the directed effects are
statistically identifiable: collinear summary vectors and task-loss imbalance
can make several graphs observationally equivalent. Matched interventions are
required to interpret their directions.

---

## 4. Signs and constraints

| Method | Diagonal | Sign | Sparsity device | Effect on transfer direction |
|---|---|---|---|---|
| AMTL | `B_tt = 0` by definition | `B ≥ 0` element-wise (extra constraint) | row-`ℓ₁` **scaled by the row task's loss** | non-negative edges; loss-scaled sparsity is the mechanism that makes low-loss donors affordable |
| AutoTR | `r_kk = 0` via `S` diagonal `10⁹` | **unconstrained** (small negatives observed) | `ℓ₁` on `R ⊙ S` | direction learned, not assumed; sign free |
| GAMTL | `b^g_tt = 0` | `b^g_t ≥ 0`; **GAMTLnr** drops it | row-`ℓ₁` loss-scaled + group-`ℓ₂` | per-group direction; non-negativity optional |
| Cross-stitch | none | free scalars | none (values can be driven to 0) | per-channel, per-layer; no task-level graph |

`[OBSERVED — AMTL Eq. (1); AutoTR `update_R.m` and card; GAMTL Eq. (1) and §4.1; Misra §3.3]`

Two consequences worth stating publicly:

* **Sign freedom changes what "direction" can mean.** With `B ≥ 0`, an edge can only *add* a source's
  parameters; a negative transfer effect must be expressed as *absence* of an edge plus the loss-scaled
  penalty. With signed `R`, a negative coefficient can actively subtract, so `r_{x,k} < 0` is not
  "no relation" but "anti-relation". `[INFERRED]`
* **The diagonal suppression is part of the mechanism.** `R_kk = 0` (or a `10⁹` diagonal) is what makes the
  reconstruction non-trivial; without it the identity solution `R = I` minimises `‖W − WR‖²_F`.
  `[INFERRED; consistent with `S = (λ₃ − 1) I + 1` in `AutoTemporal.m`]`

---

## 5. Optimization shapes, in one view

All three are **biconvex, alternating** methods; none is jointly convex, and all rely on the loss being
convex in the task parameters.

| Step | AMTL | AutoTR | GAMTL |
|---|---|---|---|
| Object step | per-column non-negative **weighted LASSO**; `Λ = diag(losses)`; solver: Mairal et al. (2010) | **closed form** soft-threshold `max(0,|R|−ρS)⊙sgn(R)`, `ρ = ρ₂/(2ρ₁)`, `O(m²)` | per-group **ADMM** (soft-threshold + non-negative projection + Cholesky `x`-update) |
| Parameter step | gradient descent / block-coordinate | accelerated proximal gradient (APG) | FISTA with block soft-thresholding |
| Extra | Algorithm 2 = curriculum (greedy task order, acyclic graph) | Gaussian-kernel warm start `exp(−|i−j|)` | group-`ℓ₂` latent Group LASSO term |
| Hyperparameters | `μ, λ` cross-validated | `λ₁, λ₂` grid; `λ₃` "large enough" | `λ₁, λ₂` similar range; `λ₃` separately |

`[OBSERVED — AMTL Algorithm 1/2; AutoTR `update_W.m`/`update_R.m`; GAMTL §3.1–§3.3]`

**Structural reading for this project.** Each object step consumes a parameter matrix `W ∈ R^{d×T}` and a
per-task loss vector. Every method's *object* update is therefore well-defined on **any** `d`; it is only
the *parameter* step (`W ≈ WB`) that demands that the `w_t` be comparable objects. This is the exact hinge
of §6 and §7. `[INFERRED]`

---

## 6. Dimension alignment and unequal-head incompatibility

### 6.1 The alignment ladder

From most to least demanding:

1. **Identical task-parameter width and shared covariates** — `W ∈ R^{d×T}`, one column per task, and (GAMTL)
   a covariate-group partition shared by all tasks. Required by AMTL, AutoTR, GAMTL, Liu & Pan.
2. **Identical output width via one-vs-all** — Deep-AMTFL converts multi-class tasks to binary one-vs-all so
   all tasks share an output dimension. `[OBSERVED — `literature/lee-2018-deep-asymmetric-mtfl.md`,
   verified against PMLR 80:2956–2964]`
3. **Identical activation-map shape at a shared layer** — cross-stitch, element-wise per channel.
4. **A collapsed per-task summary of a shared layer** — the project's class-mean head summary
   `w̃_t = (1/C_t) Σ_c W_t[c,:] ∈ R^{2000}` over the shared 2000-d hidden layer.
   `[OBSERVED — `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md` §"What classical MTRL assumes", item 4]`
5. **Scalar coupling** — a single shared scalar / loss weight.
6. **No shared object** — independent task heads.

### 6.2 What exactly breaks at 12 / 1251 / 4

* **Rung 1 is unavailable.** Each head's parameter block has a different length —
  `W_KS ∈ R^{2000×12}`, `W_SI ∈ R^{2000×1251}`, `W_ER ∈ R^{2000×4}`, i.e. `2000·C_t` parameters with
  `C_t ∈ {12, 1251, 4}` — so no single vector dimension `d` exists and `W = [w_1 w_2 w_3]` cannot be
  formed. The family-A papers themselves never test unequal widths: AMTL's real experiments are
  binary / one-vs-all (MNIST, USPS, AWA) or single-output regression (School); AutoTR uses one shared
  feature matrix `X` and `m` equal-width columns; GAMTL's tasks are single-score regressions over one
  shared `X`. `[OBSERVED — AMTL §4.2; AutoTR card; GAMTL §4.2]`
* **Rung 2 is unavailable without redefining the task set.** Deep-AMTFL equalises widths by one-vs-all
  over a *shared* label space inside one shared trunk; our three tasks have disjoint datasets and disjoint
  label spaces, and the reconstruction target `Z` is a *learnt* trunk activation, whereas the trunk's
  upstream input is frozen here. `[OBSERVED — Deep-AMTFL card Gates 2–3; INFERRED]`
* **Rung 3 is unavailable.** The tasks share the frozen trunk, so `x_A` and `x_B` would be *the same*
  tensor; a cross-stitch unit between two identical maps is not the published object, and the head widths
  still differ. `[INFERRED from Misra Eq. (1)]`
* **Rung 4 is the only reachable rung**, and it is a **collapse**: `w̃_t` is a `2000`-d mean over the
  classes of head `t`. Everything family A needs is then defined, but the object being related is a
  *statistic of a head*, not the head.

### 6.3 What the class-mean summary preserves and destroys

With `W̃ = [w̃_KS, w̃_SI, w̃_ER] ∈ R^{2000×3}` the published objective is literally well defined.

* **Preserved:** the *form* of the relation object and the update rule (a `3×3` directed matrix, its
  sparsity/positivity constraints, its alternating sub-solvers). `[OBSERVED — LT-0002 analysis §3]`
* **Destroyed:** the per-class geometry of each head. The gradient of `λ‖w̃_t − Σ_s B_st w̃_s‖²` with
  respect to head `t` is `(2/C_t)(w̃_t − Σ_s B_st w̃_s)` broadcast to **every** class row — a *uniform*
  pull across the classes of a head, which the published methods do not contemplate (they act on the task's
  own parameter vector, whose entries are individually reconstructible). `[INFERRED; stated as a documented
  deviation in `literature/lee-2016-asymmetric-mtl.md`]`
* **Not measured:** whether the collapse destroys the relation information. This is an open, explicitly
  listed diagnostic (`FRAMEWORK.md` §7 item 4; "parameter-summary adequacy", TR-0006/DG-0003).
  `[OBSERVED — `improvements/taskrelation/research/FRAMEWORK.md` §7]`

---

## 7. Shared-parameter adaptation vs faithful implementation

The honest classification for our setting, stated once and for all:

* **Faithful (published-form): not available for family A.** Each of AMTL/AutoTR/GAMTL assumes aligned
  per-task parameter columns; at 12 / 1251 / 4 those columns do not exist, and GAMTL additionally requires a
  domain-given covariate partition. LT-0002 records all three as `PASS WITH DOCUMENTED DEVIATION`, and under
  its pre-registered rule *a documented deviation counts as a faithfulness fail*.
  `[OBSERVED — `studies/LT-0002/{analysis.md,PLAN.md}` §"Falsification condition", `literature/INDEX.md`]`
* **Adaptation (project-original): what is actually reachable.** Attaching a directed relation to the
  shared trunk / class-mean summary is a **shared-parameter adaptation** — a project-original mechanism
  "motivated by the asymmetric-transfer literature (Lee et al. 2016 in particular)". DEC-0014 authorises
  exactly this under the Option-3 path and makes the labelling mandatory: it **must not** be presented as an
  implementation of AMTL, AutoTR or GAMTL, and it supports **no** external-validity claim.
  `[OBSERVED — `DECISIONS.md` DEC-0014 consequences 1–3]`
* **What the adaptation is allowed to claim.** Only the scoped question
  *"does a directed, sparsity-controlled relation over class-mean head summaries help under this protocol?"*
  (DEC-0014 §2). `[OBSERVED]`
* **What is not a defect of the adaptation.** The summary device is *shared with the in-category control*
  (classical MTRL also regularises `W̃`), so the adapter is a property of the benchmark held constant across
  arms, not a differential confound; what it costs is external validity and the uniform-row effect.
  `[OBSERVED — LT-0002 analysis §4]`
* **Boundary reminder.** Swapping in a *loss-weighting* control (per-task scalars) does not turn a directed
  arm into a faithful method, and importing such a control as the mechanism itself is out of category
  (DEC-0005 §3; survey §2.10). `[OBSERVED]`

---

## Can KS/SID/ER relations be directional?

**Task-semantic hypotheses, not observations.** SI features may encode speaker
identity useful for ER on speakers seen in training, but speaker identity can
instead become a leakage shortcut, so ER←SI must improve on held-out-speaker
LOSO to count as useful emotion transfer. Speech content/phonetic features
could aid KS; task labels do not establish that ER→KS or KS→ER is weaker.
Conversely, ER's prosodic information might help SI or distract it. The frozen
WavLM representation and shared downstream trunk can carry all these signals
without an explicit directed relation.

**Measured evidence, separately:** DG-0001's raw ER-directed gains were
reproduced by optimizer-exposure controls; the approximate controlled
ER←KS residual includes zero, and SI/ER residuals are negative in both
directions. DG-0002's cosine is symmetric by definition and cannot distinguish
donor from recipient. Thus there is currently **no demonstrated beneficial
directed KS/SID/ER effect**. A directed paper licenses a model *hypothesis*,
not a claim of directional transfer in this dataset. The intervention and
falsification criteria are specified below.

## 8. Directed proof: an experimental design that could actually establish directionality

**Status: design only.** No implementation, no runs, no authorisation. Presented because the family-A
literature *predicts* directional structure (AMTL Theorem 1; GAMTL's recovered `B^g`; AutoTR's non-symmetric
`R`), and the project's current evidence does **not** confirm any such structure (§9). Anything below is
`[HYPOTHESIZED]` as a plan, not as a result.

### 8.1 The quantity to estimate

Define the **directed transfer** of `B` into `A` under a fixed protocol:

```
T(A ← B) = acc(A trained with B) − acc(A trained alone)
```

and the **directional asymmetry**

```
Δ(A,B) = T(A ← B) − T(B ← A).
```

This is exactly DG-0001's definition (`studies/DG-0001/STUDY.md` §"Falsifiable hypothesis";
`task_relations/empirical_transfer.json` field `definition`). `[OBSERVED]`

### 8.2 Matched exposure — the non-negotiable control

DG-0001 established that at equal epochs the raw matrix measures **optimization opportunity**, not task
semantics: adding a large auxiliary changes the number of optimizer updates and the target's effective
minibatch size. Raw LOSO gains of `+0.3411` (`ER ← KS`) and `+0.2794` (`ER ← SI`), each positive in 10/10
folds, were reproduced by **ER-only** controls with matched update counts, without any auxiliary data
(`+0.3358` and `+0.3383` of the respective gains). `[OBSERVED — F8, `FINDINGS.md` lines 244–290;
`task_relations/optimization_control.json`]`

Therefore a valid directional test must hold **simultaneously**:

1. identical `num_samples` → identical optimizer steps (protocol §1);
2. identical **per-task** examples per batch (composition), for every arm;
3. identical loss scaling / `1/num_tasks` handling;
4. identical epoch budget and the **same checkpoint tag** (the protocol's `epoch`);
5. the same evaluation protocol — and LOSO for any ER-involving cell.

The decomposition to report is per fold / per seed:

```
raw pair gain  =  optimizer-exposure gain  +  pair-minus-step-matched residual
```

with the **residual** as the transfer estimate and a paired CI over seeds/folds. `[OBSERVED —
DG-0001 Stage D; F8]` Note the residual can be negative; that is a finding, not a failure
(`SI+ER −` step-matched ER-only was `−0.0589` `[−0.0883, −0.0295]`, 0/10 positive).
`[OBSERVED — F8]`

### 8.3 Multi-seed and LOSO structure

* **Screening (one explicit seed):** can only return `PROMISING`/`REJECTED`; never a promoted claim
  (`VARIANT_BENCHMARK_PROTOCOL.md` §3; F1).
* **Confirmation (seeds 0–4):** both directions of each implicated pair and their matched single-task and
  step-matched single-task references, at the same commit, reported as paired per-seed differences with 95 %
  `t` intervals, sign counts, and per-task results. Sub-1 pp effects require this
  (`VARIANT_BENCHMARK_PROTOCOL.md` §3–§4; F1).
* **ER cells additionally require speaker-independent LOSO** (F3; protocol §3). The ordinary split inflates
  ER by ≈ 15 pp (0.7902 vs LOSO 0.6391 ± 0.0506) and any ER cell from it is screening context only.
  `[OBSERVED — F3, FRAMEWORK.md §1 row 7]`
* **Reverse directions must use their own matched references.** DG-0001's reverse cells compared pairwise
  folds against a *single* seed-42 single-task fit, which is why they are labelled sensitivity evidence
  rather than confirmation (`studies/DG-0001/analysis.md` §"Reverse directions"). A real directional claim
  needs both directions refit under the same seed/fold schedule. `[OBSERVED; INFERRED as a design fix]`

### 8.4 The object-side prediction test (what makes it a *relation* claim, not just a transfer claim)

Accuracy alone cannot distinguish "a directed relation explains the transfer" from "the auxiliary helped".
Two object-level checks should be pre-registered alongside the behavioural one, both taken directly from the
primaries:

* **AMTL Theorem 1 prediction.** At the learned optimum, donors (larger `‖b_t^o‖₁`) should be the
  lower-loss tasks. Test: compare the learned `3×3` `B` against the measured per-task training losses; a
  directional claim needs the inequality `‖b_t^o‖₁ > ‖b_u^o‖₁ ⇒ L_t ≲ L_u` to hold (remembering the
  theorem is a disjunction — alternative (b) must be checked and reported, not assumed away).
  `[OBSERVED — AMTL Theorem 1; INFERRED as a test]`
* **Sign agreement between object and residual transfer.** The signed asymmetry in the learned object
  (`B_st − B_ts`, or `r_{x,k} − r_{k,x}`) should agree in sign and order with the exposure-matched
  `Δ(A,B)`. DG-0001 already reports an Ω/transfer sign-agreement count for the *symmetric* case
  (KS↔ER 6/10, SI↔ER 3/10; `studies/DG-0001/analysis.md` §"Comparison with learned Ω") — the same table,
  computed on the directed object, is the natural target. An object that does not track controlled transfer
  is a negative result about the object, not about the tasks. `[OBSERVED — DG-0001 analysis; INFERRED]`

### 8.5 Loss-scaling alternative (not an authorised control)

AMTL's outgoing penalty depends on its task loss, so an observed gain could
reflect changed task-loss scaling rather than correct directed edges. Report
effective loss factors and compare relation-graph behaviour with task-wise
outcomes under the **mandatory** MTRL and wavCSE controls. A separate scalar
loss-balancing arm could help isolate this confound, but is **not authorised**
under DEC-0009/DEC-0013; it needs a human scope/protocol decision before any
future run. Do not count it as part of an already approved confirmation.

### 8.6 Falsification conditions (pre-register before running)

A directed-relation claim should be **weakened or rejected** if any of the following holds:

1. **No exposure-matched asymmetry:** after step/composition matching, `|Δ(A,B)|` and its 95 % interval lie
   inside the project's materiality band (the protocol's `0.20 pp` regression threshold is the natural unit;
   DG-0001 used `0.010` as a *screening* threshold — neither is a significance test).
   `[OBSERVED — VARIANT_BENCHMARK_PROTOCOL.md §4; DG-0001 screening interpretation]`
2. **Residual CI contains zero** for every cell, or the sign is not consistent across seeds/folds
   (DG-0001's `ER ← KS` residual was 5 positive / 5 negative — that is a null, not a weak positive).
3. **Reverse cells are not measured under matched references** — then no direction claim is admissible
   at all (design invalidity, not a negative result).
4. **Object does not predict behaviour:** the learned directed object's sign pattern disagrees with the
   controlled residuals, or Theorem 1's inequality fails while its alternative (b) also fails.
5. **Loss-scaling alternative remains unresolved:** if a separately authorised
   matched diagnostic later reproduces the gain without the directed relation
   (§8.5), directed structure was not the identified cause.
6. **ER cells** not under LOSO, or pools ER ordinary-split numbers as an ER result (protocol §10
   "what invalidates a comparison outright"). `[OBSERVED — VARIANT_BENCHMARK_PROTOCOL.md §10]`

### 8.7 Known limits of any such design here

* Step/noise matching is *approximate*: matching update counts by batch size also changes gradient noise;
  DG-0001 records this and it cannot be removed, only bounded. `[OBSERVED — F8 "Alternative
  explanations"]`
* `T = 3` gives six cells (three unordered pairs); pair-level evidence is thin and the aggregate is
  dominated by SI. Per-task results are mandatory. `[OBSERVED — protocol §4]`
* Any directed arm at 12 / 1251 / 4 is a **project-original adaptation** and cannot inherit the papers'
  own experimental support (§7). `[OBSERVED — DEC-0014]`
* The summary collapse (§6.3) is an unmeasured confound for any object-side test. `[OBSERVED — FRAMEWORK §7]`

---

## 9. Stringent current-evidence caveat (read before citing "asymmetry" anywhere)

**No controlled, multi-seed, exposure-matched evidence of beneficial directional transfer exists in this
repository.** The three diagnostics that bear on it:

| Study | Result | Effect on an asymmetry motivation |
|---|---|---|
| DG-0001 Stage A | one-seed fixed-epoch matrix: `ER ← KS +0.0579`, `ER ← SI +0.0289`; KS/SI cells within 0.17 pp of zero | **screening only**; ER split speaker-leaky |
| DG-0001 Stage C | raw LOSO: `ER ← KS +0.3411`, `ER ← SI +0.2794`, 10/10 folds positive | confounded — see Stage D |
| DG-0001 Stage D | exposure-matched residuals: `ER ← KS +0.0053` `[−0.0289, +0.0394]` (5+/5−); `ER ← SI −0.0589` `[−0.0883, −0.0295]` (0+/10−); reverse: `KS ← ER −0.0038`, `SI ← ER −0.0231` | **the raw asymmetry was optimizer exposure**; no positive residual resolved |
| DG-0002 | seeds 0–4: no pair met the persistent-conflict threshold; baseline late mean cosines KS↔SI `+0.002`, KS↔ER `+0.001`, SI↔ER `+0.022` | no gradient-conflict signal for a directed/interaction mechanism to act on |
| DG-0005 | ER gradient-norm dominance is a **training-mixture** property (`8.962 ± 0.456 → 2.352 ± 0.270`, 5/5 seeds, only composition changed) | scale is not a task-intrinsic relation property (F10); it cannot motivate a directional or reliability-aware relation mechanism |

`[OBSERVED — F8, F9, F10 in `FINDINGS.md`; `studies/DG-0001/{STUDY.md,analysis.md}`;
`task_relations/{empirical_transfer.json, loso_transfer.json, optimization_control.json}`]`

**What may be said today:**

* The *only* exposure- and LOSO-controlled directional result that resolves at all is **negative**
  (SI→ER residual `−0.0589`, all ten folds) or unresolved (KS→ER). `[OBSERVED]`
* The learned symmetric Ω's sign/magnitude does not track controlled transfer
  (KS↔ER 6/10, SI↔ER 3/10 sign agreement), so Ω magnitude is not a utility proxy (F6, F7). `[OBSERVED]`
* Nothing in the repository establishes that symmetric parameterisation is *why* classical MTRL is
  outcome-neutral. DG-0001's own decision list says so explicitly: *"Classical MTRL's negative outcome
  remains unexplained by symmetry."* `[OBSERVED — `studies/DG-0001/analysis.md` §Decision]`
* Therefore: **a directed arm is not motivated by the raw transfer matrix.** Any such arm proceeds as a
  project-original mechanism under DEC-0014, labelled as such, with its own pre-registered protocol —
  and the exposure-matched, multi-seed, LOSO design of §8 is the minimum that could produce a directional
  result. `[OBSERVED — F8; DECISIONS.md DEC-0014]`

---

## 10. Source-verification status (explicit gaps)

| Source | Verified how | Residual gap |
|---|---|---|
| Lee, Yang & Hwang 2016 (AMTL) | PMLR 48 PDF read in full this session (title, Eqs. (1)–(5), Theorem 1 p. 3, Algorithms 1–2, Tables 1–2, Figs. 1–3) | none identified |
| Oliveira, Gonçalves & Von Zuben 2019 (GAMTL) | IJCAI-19 PDF read in full this session (Eqs. (1)–(5), Algorithm 1, §3.1–§3.3, Tables 1–2, Figs. 1–4) | none identified |
| Zhou & Yang 2023 (AutoTR) | KDD DOI record + the authors' official MATLAB implementation (`AutoTemporal.m`, `update_R.m`, `update_W.m`): objective, `S`, `ρ = ρ₂/2ρ₁`, proximal step and the `(I−R)(I−R)ᵀ` gradient all read from code | **the camera-ready PDF was never obtained.** Retrieval attempts this session: (i) `eprints.whiterose.ac.uk/id/eprint/199291/8/AutoTR_KDD_Camera_Ready_v4.pdf` — connected but the server truncates every transfer (best: 539 418 of 1 245 610 bytes, no `%%EOF`, MD5 mismatch); (ii) the alternate White Rose path `…/208825/1/…` — HTTP 401; (iii) `https://dl.acm.org/doi/pdf/10.1145/3580305.3599261` (the DOI's Semantic Scholar `openAccessPdf`, status GOLD) — returns an HTML challenge page. **Consequently: the paper's own equation numbers, Tables 1–5 and t-test values were not re-read, and the LT-0002 card's numbering (Eqs. (2)–(3), (7)–(10)) is unverified against the PDF.** Every AutoTR mechanism statement in §2.2/§4/§5 above is instead verified from the authors' released code, which is a primary artifact; nothing here depends on AutoTR's accuracy tables. |
| Misra et al. 2016 (cross-stitch) | CVPR open-access PDF read in full this session (Eqs. (1)–(3), §3.3, §4, §5, Tables 1–6) | none identified |
| Zhang & Yang 2021 survey | arXiv:1707.08114v3 full text read this session; §2.1, §2.3, §2.4 and §2.10 quoted above | the survey is a secondary source; every mechanism claim used here is sourced to the primary paper as well |
| Zhang & Yeung 2014 | author copy via the existing card; not re-read in full this session | Eq. (16) number and §2.3/§4.6 mapping rest on `literature/zhang-yeung-2014-mtrl-asymmetric.md` |
| Repository evidence (DG-0001/0002/0005, F8–F10, protocol, DEC-0013/0014) | read from `improvements/taskrelation/research/…` this session | none identified |

---

## 11. References (links)

**Primary papers.**

1. Lee, Yang & Hwang (2016). *Asymmetric Multi-task Learning Based on Task Relatedness and Loss.* ICML,
   PMLR 48:230–238. <https://proceedings.mlr.press/v48/leeb16.pdf>
2. Zhou & Yang (2023). *Automatic Temporal Relation in Multi-Task Learning.* KDD '23, 3570–3580.
   DOI 10.1145/3580305.3599261. Author copy (retrieval unreliable, see §10):
   <https://eprints.whiterose.ac.uk/id/eprint/199291/8/AutoTR_KDD_Camera_Ready_v4.pdf>;
   official code: <https://github.com/menghui-zhou/AutoTR> (`AutoTemporal.m`, `update_R.m`, `update_W.m`).
3. de Oliveira, Gonçalves & Von Zuben (2019). *Group LASSO with Asymmetric Structure Estimation for
   Multi-Task Learning.* IJCAI-19, 3202–3208. <https://www.ijcai.org/Proceedings/2019/0444.pdf>
4. Misra, Shrivastava, Gupta & Hebert (2016). *Cross-stitch Networks for Multi-task Learning.* CVPR.
   <https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf>
5. Zhang & Yang (2021). *A Survey on Multi-Task Learning.* IEEE TKDE; arXiv:1707.08114v3, 29 Mar 2021.
   <https://arxiv.org/pdf/1707.08114> (§2.1 feature learning incl. cross-stitch; §2.3 task clustering;
   §2.4 task relation learning incl. AMTL Eq. (25) and the symmetric-precision remark; §2.10 optimization)
6. Zhang & Yeung (2014). *A Regularization Approach to Learning Task Relationships in Multi-Task Learning.*
   ACM TKDD 8(3):12. <https://yuzhanghk.github.io/papers/Zhang_Yeung_TKDD14.pdf>
7. Liu & Pan (2017). *Adaptive Group Sparse Multi-task Learning via Trace Lasso.* IJCAI-17, 2358–2364.
   <https://www.ijcai.org/Proceedings/2017/0328.pdf>

**Repository artifacts cited above** (all under `improvements/taskrelation/research/`).

* `VARIANT_BENCHMARK_PROTOCOL.md` — binding protocol (§1 fixed conditions, §3 staging, §4 endpoints,
  §8 gotchas, §10 invalidating comparisons).
* `FINDINGS.md` — F1, F3, F4, F6, F7, F8, F9, F10; `DECISIONS.md` — DEC-0001, DEC-0005, DEC-0009, DEC-0010,
  DEC-0013, DEC-0014; `FRAMEWORK.md` — §1 conditioned-quantity table, §2 relational vs optimization, §7
  ranked missing evidence.
* `literature/INDEX.md` and the cards `literature/{lee-2016-asymmetric-mtl, zhou-2023-autotr,
  oliveira-2019-group-lasso-asymmetric, lee-2018-deep-asymmetric-mtfl, liu-2017-trace-lasso-gamtl,
  yu-2020-graph-adjacency-gamtl, zhang-yeung-2014-mtrl-asymmetric, zhang-yang-2021-mtl-survey,
  goncalves-2016-mssl}.md`.
* `studies/LT-0002/{PLAN.md, analysis.md, NOTE.md}`; `studies/DG-0001/{STUDY.md, analysis.md, result.json}`;
  `studies/DG-0002/{PLAN.md, analysis.md, confirmation_result.json}`;
  `studies/DG-0005/{PLAN.md, analysis.md, NOTE.md, confirmation_result.json, noise_shape_result.json,
  analyze_noise_shape.py}`.
* `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`;
  `task_relations/{empirical_transfer.json, loso_transfer.json, optimization_control.json}`.

---

## 12. One-paragraph summary

A **directed relation** here means a learned matrix `B` (or `R`, or `{B^g}`) with `w_t ≈ Σ_{s≠t} B_{st} w_s`,
`B_tt = 0` — a graph whose directedness lives in the parameter-reconstruction operator, while the penalty
it induces, `tr(W(I−B)(I−B)ᵀWᵀ)`, is symmetric PSD (survey §2.4). It is **not** the same object as an
"asymmetric" covariance, which is a symmetric `Ω` under a target-after-sources *setting* (Zhang & Yeung
2014), nor as **loss balancing**, which learns per-task scalars and is categorically outside §2.4.
AMTL (loss-scaled row-`ℓ₁`, `B ≥ 0`), AutoTR (signed `R`, closed-form soft-threshold) and GAMTL (per-group
`B^g`, ADMM) are all implementable in principle, but all assume **one aligned parameter column per task**;
at 12 / 1251 / 4 classes those columns do not exist, so any directed relation here is a **project-original
adaptation over class-mean head summaries** (DEC-0014), never a faithful reproduction. Proving direction
would require an exposure-, composition- and seed-matched transfer design with
LOSO for ER, object-side prediction checks (including AMTL Theorem 1's
alternative), and pre-declared falsification. A loss-balancing control is
unapproved and would need separate authorization.
The repository has not established that asymmetry is needed: its raw directed gains were
optimizer exposure (DG-0001/F8), there is no persistent pairwise gradient conflict (DG-0002), and the ER
gradient-scale signal is training-mixture-controlled (DG-0005/F10).
