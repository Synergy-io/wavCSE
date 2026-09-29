# IDENTIFIABILITY — what an `m = 3` relation study can and cannot identify

Status: **literature/theory only.** No code, no configs, no registered study, no run, no
ranking, no winner. This file answers one question — *given exactly three tasks, which
statements about the learned relation object are identified, which are merely descriptive, and
which are invalid as inferences?* — and it does so with explicit preconditions rather than
verdicts about any candidate method.

Companion documents: [`RELATION_SEMANTICS.md`](./RELATION_SEMANTICS.md) (which object the words
refer to; the worked `3 × 3` example reused here), [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md)
(the `3 × 3` identity, the graphical-lasso step, predictions P1–P5),
[`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) (parameter counts, rank geometry,
nominal vs effective sample size), [`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md)
(D6 direct vs conditional, D8 estimability), [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md),
[`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md), [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md),
[`SURVEY_MAP.md`](./SURVEY_MAP.md), [`CANDIDATES.md`](./CANDIDATES.md),
[`SUCCESSOR_STUDIES.md`](./SUCCESSOR_STUDIES.md).

**Epistemic key.** `OBSERVED` = read in a primary source or a repository artefact (named);
`INFERRED` (= the assignment's *DERIVED*) = derived or computed here (algebra/arithmetic
shown); `HYPOTHESIZED` = falsifiable prediction, unmeasured; `PRIMARY_SOURCE_NOT_VERIFIED` =
a claim a prior pass explicitly declined to verify (never promoted here); `GENERAL-METHOD` = a
standard statistical reference cited as background, not read in this project's passes.

**Source provenance.** No primary PDF was reopened for this note. Paper claims are taken from
the repository's existing verified ledgers
([`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md), [`SURVEY_MAP.md`](./SURVEY_MAP.md) §10,
[`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md) §10) and labelled with the ledger they
come from. Where a prior pass recorded a verification gap, the gap is carried forward (§7).

---

## 1. Two questions that are constantly conflated

### 1.1 Algebraic determinacy (a theorem, no data needed)  `INFERRED`

For a positive-definite `3 × 3` covariance `Sigma` with `Omega = Sigma^{-1}` and `k` the third
task,

```
Omega_ij = ( Sigma_ik Sigma_jk - Sigma_ij Sigma_kk ) / det Sigma        (C)
Omega_ij = 0   <=>   Sigma_ij = Sigma_ik Sigma_jk / Sigma_kk            (D)
```

([`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §2.3, derivation via the Schur
complement `(S)`; verified exactly there and re-verified here on the
[`RELATION_SEMANTICS.md`](./RELATION_SEMANTICS.md) §1 example, where
`Sigma_13 Sigma_23 / Sigma_33 = 0.2 = Sigma_12` gives `Omega_12 = 0` at `det Sigma = 0.63`).

Given `Sigma`, the whole precision is determined. **There is no estimation problem here.**
The forward direction (covariance → precision) is a bijection on the PD cone and loses no
information at all ([`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §1.2).
The estimand is conditionally-independent structure *of a specified covariance model*.

### 1.2 Statistical identifiability (a statement about data, noise and design)

In this project `Sigma` is never given. It is estimated from task-parameter summaries that are
themselves produced by a stochastic training run; "the edge is zero" is a statement about a
*random support set* obtained from one fit under one penalty. The repository already states the
distinction in exactly this form:

> "Numerical invertibility of a 3×3 matrix is **not** statistical identifiability of its
> zeros." — [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) Q1 (`OBSERVED`)

Three separate preconditions must all hold before a fitted zero may be read as a conditional
independence, and none of them is a property of the arithmetic in §1.1:

| Precondition | What it needs | Status in this repository |
|---|---|---|
| **P-i** sampling model | p-MSSL assumes coefficient-coordinate rows are approximately i.i.d. Gaussian across features ([primary ledger](PRIMARY_LITERATURE.md)); the 2001 head-summary coordinates would need to support that reading | **unverified**, not disproved: shared training may induce dependence, but its effective sample size is unknown |
| **P-ii** stable support | selection rule and support stability under independent reruns, with the penalty declared | **not yet measured for the corrected TR-0007 screen**; λ₂=.01 is researcher-fixed (DEC-0016) |
| **P-iii** replicated interpretation | repeat fits across seeds and, for ER outcome claims, held-out speakers | **absent for corrected TR-0007 in this checkout**; historical MTRL seeds do not replicate p-MSSL precision zeros |

`DERIVED`. An exactly zero *fitted* edge establishes an optimizer support at a
specified penalty, not population conditional independence or useful transfer.
The algebra in §1.1 remains exact, but P-i–P-iii are needed to generalize it.

---

## 2. Generic unpenalized precision is dense; penalized zeros are selections

`INFERRED` (from `(D)`; the surface argument is analytic, the example numeric).

`Omega_ij = 0` is **one scalar equation** connecting the three correlations:
`rho_ij = rho_ik rho_jk` (divide `(D)` by `sqrt(Sigma_ii Sigma_jj)`; the variance factors
cancel, so the correlation form holds without any unit-variance assumption).
In the 3-dimensional correlation simplex `{det R > 0}` that equation cuts out a
**2-dimensional** surface. Before sparsity penalization, a continuously
varying generic PD covariance therefore has nonzero precision off-diagonals
almost surely. **After ℓ₁ regularization**, exact fitted zeros occupy
*regions* of input Gram space: they are thresholded selections, not rare
numerical coincidences. A zero's interpretation depends on the penalty and
the row-sampling model, not only this geometric dimension count.

Three consequences:

1. Dense relation support is a legitimate possible screen outcome. The prior
   [MSSL analysis](MSSL_SPARSITY_ANALYSIS.md) §5.2 illustrates 0/3 zeros at
   λ₂=.01 for **representative** Grams; it does not measure this study's Gram.
2. A sparse-support claim needs a predeclared estimand and stability check.
   With three task pairs there are only eight undirected support patterns.
3. A zero covariance entry can likewise coexist with nonzero precision; the
   absence semantics must never be imported across objects.

---

## 3. Six limits that follow from `m = 3` and from the frozen benchmark

The frozen conditions that make these limits concrete: three tasks (`ks_si_er`; head widths
12 / 1251 / 4), `wavlm_large` frozen, mean frame pooling, layer pooling `smp` 0.5 over **all
25** layers, shared `1024→512→2000` trunk, 30 epochs, batch 2048, AdamW lr 0.0025, ≈2,820
steps, epoch checkpoint, screen seed 42
([`VARIANT_BENCHMARK_PROTOCOL.md`](../../../../improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md) §1;
[`studies/TR-0007/PLAN.md`](../../../../improvements/taskrelation/research/studies/TR-0007/PLAN.md)).
`OBSERVED`.

### L1 — 3 edges versus 6: capacity is not identifiability

* Counts `OBSERVED`/re-verified: a symmetric object has `m(m-1)/2 = 3` unique off-diagonals; a
  directed object has `m(m-1) = 6`. The undirected support class has `2^3 = 8` members, the
  simple digraphs `2^6 = 64`, the DAGs `25` (OEIS A003024 at `n = 3`), the partitions `5`
  (Bell(3)); with `m = 3` the only asymmetry content is the three differences `B_ij - B_ji`
  ([`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) §2–§2.3).
* **Representational limit.** A symmetric estimator *cannot* encode `T(A←B) ≠ T(B←A)` — a
  statement about the object, not about the world. (`TASK_RELATION_TAXONOMY.md` D2.)
* **Statistical limit (`INFERRED`).** Six coefficients are a small parameter count, **not**
  evidence that directions are identifiable: the reconstruction `W ≈ W B` is identified only
  up to the conditioning of the summary design, and near-collinear `w_t` make several graphs
  observationally equivalent (`ASYMMETRIC_RELATIONS.md` §3: *"Six coefficients are a small
  parameter count, not evidence that the directed effects are statistically identifiable"*).
* **Regularizer limit (`OBSERVED`, survey §2.4).** `||W - W B||_F^2 = tr(W(I-B)(I-B)^T W^T)`
  is symmetric PSD whatever `B` is; the survey classifies the induced relation as a
  "symmetric … task precision matrix with a restrictive form"
  ([`SURVEY_MAP.md`](./SURVEY_MAP.md) §3.2). So a directed arm is a *restricted symmetric
  precision with a factorised, directed parameterisation* (`RELATION_SEMANTICS.md` §2, object (d)).
* **Consequence.** "The directed arm has more capacity" is not an argument that it can express
  *the truth* here. The honest question is whether the three asymmetry differences are
  estimable and behaviourally consequential — which requires the §8 design of
  [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md), not the parameter count.

### L2 — seeds are not new pairs

* A seed replicate re-estimates the **same three** estimands under a different
  initialization and minibatch order. Paired seeds can quantify training
  variation; they do not create a fourth task pair or independent relation
  edges. Five independently seeded **fits** yield five repeated 3-vectors,
  not 15 independent pair observations. `DERIVED`.
* The derived standard errors (`SD/sqrt(n)`, `INFERRED` from the dispersions recorded in
  `FINDINGS.md` F6 / `MTRL_DIAGNOSTIC_SYNTHESIS.md`) are:

  | Entry (size) | reported SD | n | derived SE | derived 95 % half-width |
  |---|---:|---:|---:|---:|
  | 25 L seeds KS↔SI | `0.00017` | 5 | `0.00008` | `0.00021` (`t₄ = 2.776`) |
  | 25 L seeds KS↔ER | `0.266` | 5 | `0.119` | `0.330` |
  | 25 L seeds SI↔ER | `0.266` | 5 | `0.119` | `0.330` |
  | 25 L LOSO KS↔SI | `0.012` | 10 | `0.0038` | `0.0086` (`t₉ = 2.262`) |
  | 25 L LOSO KS↔ER | `0.072` | 10 | `0.023` | `0.052` |
  | 25 L LOSO SI↔ER | `0.103` | 10 | `0.033` | `0.074` |

  (These are **nominal descriptive intervals**, not claims of independent
  LOSO-fold draws. Dependence can change interval width in either direction.)
* **Consequence.** The wide nominal 25L seed intervals for **these two ER
  covariance entries** show low resolution of their sign; the near-fixed
  KS--SID entry has very different precision. There is no universal
  `0.33` detection floor for all relation entries or for task accuracy.
  Five seeds are the existing confirmation protocol, not a proof that
  every small effect is detectable.

### L3 — folds are correlated

* The ER LOSO harness varies **only** the held-out speaker (test = speakers[i], val =
  speakers[(i+1) % 10], train = the other 8; 10 folds, 5 epochs/fold — `FINDINGS.md` F3).
  Representation, trunk, protocol, sampler composition, step count and the relation-fitting
  procedure are shared across folds. `OBSERVED`.
* Therefore the ten fold-level relation objects are *not* ten independent observations of the
  relation object — the repository states this directly
  ([`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) §3.2: "the ten LOSO folds are not
  independent observations (they vary only the ER speaker)";
  `MTRL_DIAGNOSTIC_SYNTHESIS.md`: "`n=10` non-independent folds"). `OBSERVED`.
* Two further correlations are structural at `m = 3`:
  * **Within a fold**, all three edge values come from **one** `Omega` fit on one summary
    matrix — so the three entries are not independent of each other either;
  * **Across axes**, seed variation (initialisation/minibatch order) and fold variation (ER
    speaker data) have different variance structures and are "not interchangeable"
    (`FINDINGS.md` F6, "Alternative explanations"). Mixing them into one `n` is a modelling
    error. `INFERRED`.
* **Consequence.** A fold-level statement such as "10/10 positive" is evidence of *sign
  consistency of one protocol*, not ten confirmations. It is the right protocol for an ER
  claim (F3 makes it mandatory) and it is simultaneously a *single-task* perturbation, so it
  cannot by itself establish that the other two entries are conditionally stable. Consistent
  with F6: at 25 L LOSO, KS↔SI is fold-stable while both ER-involving entries are
  fold-sensitive (ranges `-0.079…+0.201` and `-0.202…+0.205`).

### L4 — three cross-pair values cannot validate a relation-outcome law

With only three fixed unordered pairs (or six ordered pairs sharing these
three tasks), a Pearson/Spearman coefficient **can be computed** from three
numbers; it would have only one residual degree of freedom and no independent
draw of new task pairs. A small or large coefficient is therefore
descriptive for *this set*, not a generalization claim. Treating five seeds
as 15 independent pair observations repeats the same pairs and is
pseudoreplication.

For a separate proposed *covariance of the three edge estimates* across
replicated fits, `n` independent observations of a `p=3` vector yield
centered-scatter rank `≤min(3,n−1)`; when `n=3` it is singular and its
inverse/edge-level partial correlation is undefined. Even `n≥4` is
algebraic possibility, not a reliable fit given shared parameters and
overlapping LOSO folds. The unit is a complete training realization, not
each scalar entry. Cross-edge dependencies, causal edges and population
task-family laws are **not identified** from the current three fixed tasks.

### L5 — training-trajectory dependence: the nominal `d = 2001` is not a sample size

* The adapter feeds the relation estimator a `[3, 2001]` matrix: one row per task, each row the
  mean classifier-weight row (2000) concatenated with the mean bias
  (`04-mssl/README.md`; `studies/TR-0007/PLAN.md`; [`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) §3.1).
  p-MSSL's nominal sample size for the `3 × 3` precision is the number of rows `d`
  (Gonçalves et al. §3.3, `OBSERVED`). `OBSERVED`.
* In this adapter, `d` indexes coordinates of **one** summary vector per
  task from a shared training trajectory; the paper's i.i.d.-row
  assumption is not established here. Shared training makes dependence
  plausible, but the effective sample size cannot be calculated from the
  nominal `d=2001` without a sampling model.
* **The epoch sequence is one dependent path, not replicates.** `Ω` is refreshed on a schedule
  (`mssl.warmup_epochs: 3`, `mssl.omega_update_frequency: 1` → at most 27 updates in the
  30-epoch screen; the LOSO diagnostics record five per-epoch values per fold). Consecutive
  `Ω_t` are functions of an almost-identical weight matrix, so they are strongly serially
  dependent. The synthesis states the consequence itself: a five-epoch budget "is too short to
  claim a general stabilization time relative to validation convergence". `OBSERVED`.
* **The estimate is endogenous.** Both relation arms update `W` *under* the relation penalty,
  so the summary matrix that supplies the "sample" changes when the object changes. Any
  stability claim is a fixed-point property of an alternating optimisation, not an i.i.d.
  estimation statement. `INFERRED` ([`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) Q7).

### L6 — near-singular head summaries: interior is not identified

* `OBSERVED`: at `smp` 25 L (and `lnp` 16 L) the trace-1 MTRL `Omega` saturates towards the
  uniform `±1/3`; the limiting matrix is `Omega* = (1/3) 11^T` with `rank 1`, `tr = 1`, and
  the observed `0.33308 ± 0.00017` reproduces that PSD-cone boundary
  (F5, F7; [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §3.3).
* The MTRL covariance's exact rank-one boundary has no finite inverse.
  MSSL parameterizes **precision** `P≻0`; its `−log|P|` barrier prevents
  an exactly singular *precision* optimum at finite input. This does
  **not** itself put an upper bound on precision eigenvalues or guarantee
  that the implied covariance `P⁻¹` is well conditioned.
* **But interior ≠ identified.** If an estimated second moment is
  near-equicorrelated, its precision may also be exchangeable; for
  equicorrelation `r`, `ρ_partial=r/(1+r)`. A log-det barrier prevents an
  *exact singular precision* fit but supplies no extra pair-discriminating
  information. Near exchangeability must be **measured on the actual Gram**.
* **Support is data-and-penalty dependent.** The ℓ₁ penalty acts on raw
  precision while interpretable partial correlation divides by fitted
  diagonals ([`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §4.3).
  An exact zero is a support-selection event at the fixed λ₂, not a direct
  measurement of beneficial task transfer.
* **Downstream candidates inherit the constraint.** MTHOL's higher-order object needs a
  full-rank Gram for its `log|W^T W|` term; near-rank-one summaries leave that undefined and
  make the "mediated third-task" term an amplification of the leading eigenvector rather than
  new structure ([`CANDIDATES.md`](./CANDIDATES.md) MTHOL row;
  [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) Q7; [`SUCCESSOR_STUDIES.md`](./SUCCESSOR_STUDIES.md)
  CANDIDATE-B). `OBSERVED` (as a stated gap).

---

## 4. Classification of relation claims at `m = 3`

Vocabulary (mapped to the project's evidence levels, [`FRAMEWORK.md`](../../../../improvements/taskrelation/research/FRAMEWORK.md) §3):

* **DESCRIPTIVE** — a statement about *the fitted object under named conditions*. Valid as
  description; **selects no mechanism** and licenses no generalization (framework Level B/C).
* **INFERENTIAL-WEAK** — survives one axis but not all; may motivate a *diagnostic*; still
  selects no mechanism (framework Level B).
* **INVALID** — the inference's precondition is unmet here (a missing object, a missing
  selection procedure, a missing control, or an algebraic error). Not "weak evidence": no
  admissible evidence for the claim exists in the current record.

| # | Claim | Status | Why (precondition) | What would change the status |
|---|---|---|---|---|
| 1 | "At `smp` 25 L, the KS↔SI `Omega` entry was `+0.286 ± 0.012` across 10 LOSO folds" | **DESCRIPTIVE** | a fitted value under named pooling/layers/axis; repository artefact (`FINDINGS.md` F6) | nothing — it is a description. Inference *from* it is governed by rows 3, 5, 6 |
| 2 | "The learned `Omega` changed qualitatively when pooling changed (`mix`→`smp`/`lnp`)" | **DESCRIPTIVE** | same; F2/F5 (`OBSERVED`) | needs a matched intervention to become causal (F2) |
| 3 | "KS↔SI is a stable relation" | **INFERENTIAL-WEAK** | true for the 25 L LOSO condition (10/10, `SD = 0.012`) but false as a task-intrinsic statement: at 16 L seeds it is `+0.110 ± 0.236`, 4/5 (F6) | stability surviving a *second, orthogonal* axis (seeds **and** folds at both layer counts), with the saturation state named |
| 4 | "SI↔ER is the only fully sign-consistent edge at 16 L" | **INFERENTIAL-WEAK** | one configuration, one axis (F6) | replication under a second axis *and* a matched-composition control |
| 5 | "`Omega_ij=0` means these tasks never interact" | **INVALID** | a covariance zero is marginal in its coefficient model, a precision zero is conditional, and the shared trunk still couples task losses; no fitted zero alone establishes utility | name the object and intervene under matched exposure |
| 6 | "`Omega` magnitude measures transfer / utility" | **INVALID** | contradicted by the record: the most stable, most saturated edge (25 L KS↔SI) coexists with no material KS or SI gain (F4/F6/F7); sign agreement with controlled transfer is 6/10 (KS↔ER) and 3/10 (SI↔ER) (`studies/DG-0001/analysis.md`) | a *prospective*, pre-registered prediction of controlled transfer on held-out seeds/folds ([`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) E7) |
| 7 | "The directed `B`'s asymmetry shows directional transfer" | **INVALID as stated** | raw asymmetry (`+0.3411`/`+0.2794` LOSO) was reproduced by optimizer-exposure-only controls; residuals `+0.0053` `[-0.0289,+0.0394]` (5+/5−) and `-0.0589` `[-0.0883,-0.0295]` (0+/10−) (F8) | the exposure-matched, multi-seed, LOSO design with separate matched references per direction ([`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md) §8.2–8.3) |
| 8 | "Ω changes over epochs, **therefore** a dynamic relation *mechanism* improves outcomes" | **INVALID as causal inference** | existing trajectories establish descriptive change, but not prospective outcome prediction | held-out-seed/fold prediction under a fixed schedule |
| 9 | "Correlating three pair edges with outcomes establishes a population law" | **INVALID as inference** | a Pearson coefficient is numerically computable at n=3 but has one residual df and mutually dependent pairs; a 3×3 covariance of three replicated edge vectors is singular (L4) | new independent task sets or narrowly predeclared within-pair interventions; more seeds improve *same-pair* precision, not pair count |
| 10 | "A single fitted zero proves tasks truly conditionally independent" | **INVALID** | a zero is a legitimate *descriptive solver support* at the stated λ, but not independently replicated population structure | state Gaussian-row assumption/λ and report replicated support across fits |
| 11 | "Because precision ≠ covariance, the measured `Omega` table must be re-read as conditional structure" | **INVALID** | object-identity error: `Omega` here is a trace-1 **covariance**; converting requires the full matrix including diagonals and *changes the estimand* (`RELATION_SEMANTICS.md` §5.2) | invert a *fully reported* matrix, or state the estimand as marginal |
| 12 | "`m = 3` prevents the directed arm from encoding direction, so the directed arm is unmotivated" | **INVALID as stated** | the representational limit is real (L1) but the *advantage* claim it is used against was already refuted by exposure control (F8); the representational argument is not the reason | an exposure-matched directional result, or an explicit statement that direction is being studied as a hypothesis, not as a motivated improvement |

`INFERRED` (classification) / `OBSERVED` (the repository facts each row cites).
Rows 1–2 are *not* defects; they are the only thing the record supports. Rows 3–4 are the
boundary at which more of the same evidence stops helping. Rows 5–12 are claims that must not
appear in a wavCSE write-up in their current form.

---

## 5. Strong mathematical warning: precision and covariance are different objects

> ### `Omega_ij = 0` and `Sigma_ij ≠ 0` are perfectly compatible
>
> For any positive-definite `Sigma`, with `Omega = Sigma^{-1}` and `k` the remaining task,
> identities `(C)`/`(D)` of §1.1 hold exactly:
> `Omega_ij = 0 ⟺ Sigma_ij = Sigma_ik Sigma_jk / Sigma_kk`.
> The repository's worked witness
> ([`RELATION_SEMANTICS.md`](./RELATION_SEMANTICS.md) §1) is
> `Sigma = [[1, .2, .4], [.2, 1, .5], [.4, .5, 1]]`, `det Sigma = 0.63`, `cond = 3.9`:
> `Sigma_12 = 0.20` and `Omega_12 = 0`, with the marginal association fully carried by the
> coefficient association through the third task (an undirected `1--3--2` path,
> **not** a causal graph). `Sigma_{12|3} = diag(0.84, 0.75)`.
>
> Consequences that are **mathematical**, not statistical, and therefore cannot be repaired by
> more data, more seeds, more folds or a better optimiser:
>
> 1. **Different zero sets.** A covariance zero and a precision zero are different statements
>    about the same numbers (`TASK_RELATION_TAXONOMY.md` D6).
> 2. **No conversion by rescaling.** The map is the matrix inverse; there is no per-entry
>    transformation from `Sigma_ij` to `Omega_ij` (the earlier pass's point that a "precision
>    instead of covariance" framing adds no information but changes the penalty geometry —
>    [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §1.2).
> 3. **Sign flips are expected.** `rho_ij.k = -Omega_ij/sqrt(Omega_ii Omega_jj)`, so a negative
>    precision entry is a *positive* conditional correlation; a negative covariance entry is
>    an *anti*-correlation. Neither is negative transfer (`TASK_RELATION_TAXONOMY.md` D3;
>    [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) Q8).
> 4. **Direction and conditioning multiply.** A directed coefficient `B_st` is not
>    `Omega_st`, and the induced penalty is symmetric whatever `B` is
>    ([`RELATION_SEMANTICS.md`](./RELATION_SEMANTICS.md) §2, objects (c)–(d)).
>
> Therefore: **"the relation between tasks `i` and `j`" is not a well-formed claim until the
> object is named**, and any sentence of the form "there is no relation between `i` and `j`"
> is meaningless without saying *marginal* or *conditional*. Reading a measured trace-1
> covariance table as conditional structure — or a fitted precision zero as a marginal
> non-interaction — is an error of object identity that no amount of data can repair.
> `INFERRED` (the algebra is exact; the warning is this document's reason to exist).

---

## 6. Crosslinks

* [`RELATION_SEMANTICS.md`](./RELATION_SEMANTICS.md) — the five relation
  objects, worked positive-definite example, and measured KS/SID/ER entries.
* [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) — `(C)`/`(D)`, the Schur
  complement, `Ω = Σ^{-1}` as a bijection, the log-det barrier, the raw-scale ℓ₁ threshold
  warning, and predictions P1–P5.
* [`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) — counts (6/3/8/64/25/5), rank
  geometry, `d` vs effective sample size, `n = 5`/`n = 10` resolution, "simulation vs
  resolution".
* [`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md) — D6 (direct vs conditional),
  D8 (estimability at `m = 3`), the six-encoding table.
* [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md) §3 (direction semantics and
  `m = 3` geometry), §8 (the exposure-matched design that a direction claim would need), §9
  (current-evidence caveat).
* [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) — Q1, Q4, Q7, Q8; empirical unknowns E1, E5, E7.
* [`CANDIDATES.md`](./CANDIDATES.md), [`SUCCESSOR_STUDIES.md`](./SUCCESSOR_STUDIES.md) — the
  candidate-level consequences of L4/L6 (the MTHOL full-rank gate; the "do not run an
  opportunistic λ sweep" rule).
* Repository: [`VARIANT_BENCHMARK_PROTOCOL.md`](../../../../improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md) §§1–4, §10,
  [`FINDINGS.md`](../../../../improvements/taskrelation/research/FINDINGS.md) F1–F8,
  [`FRAMEWORK.md`](../../../../improvements/taskrelation/research/FRAMEWORK.md) §1–§4,
  [`studies/TR-0007/PLAN.md`](../../../../improvements/taskrelation/research/studies/TR-0007/PLAN.md),
  [`studies/DG-0001/analysis.md`](../../../../improvements/taskrelation/research/studies/DG-0001/analysis.md),
  `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`,
  `improvements/taskrelation/research/tests/test_mssl_omega_solver.py`,
  [`DECISIONS.md`](../../../../improvements/taskrelation/research/DECISIONS.md) DEC-0013/0014/0015/0016.

---

## 7. Source-verification status (explicit)

| Source | How it is known here | Residual gap |
|---|---|---|
| Zhang & Yang, *A Survey on Multi-Task Learning*, arXiv:1707.08114**v3** (2021) | `OBSERVED (survey)` via [`SURVEY_MAP.md`](./SURVEY_MAP.md) §10, which records a direct read of §2, §2.4, §2.8, §2.9, §2.10 including the Eq. (25) symmetrisation caveat | secondary source; every mechanism claim used here is also attributed to its primary via the same ledger |
| Gonçalves, Von Zuben & Banerjee, *MSSL*, *JMLR* 17(33):1–30, 2016 | `OBSERVED` via [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md) (paper read directly: §3.1–§3.3, Eqs. (3), (4b), (8)–(11), Algorithm 1) and [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §1.1–§1.2 | the paper's `λ₂` scale is not transferable and no default is published; DEC-0016 fixes `0.01` as a *researcher-fixed screening* value, not a paper value |
| Zhang & Yeung, *MTRL*, UAI 2010 | `OBSERVED` via [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md) and [`SURVEY_MAP.md`](./SURVEY_MAP.md) §10 (Eqs. (1)–(7), §2.2, §3) | none identified for the claims used here |
| Zhang & Yang, *SPATS*, AAAI 2017 | `OBSERVED` via the same ledgers (Eq. (3), Theorem 1, the many-task caveat) | none identified for the claims used here |
| Lee, Yang & Hwang, *AMTL*, ICML 2016 | `OBSERVED` via [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md) (Eqs. (1)–(3), Theorem 1, both alternatives) and [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md) §2.1 | none identified |
| Zhang & Yeung, *MTHOL*, IJCAI 2013 | `OBSERVED` via [`CANDIDATES.md`](./CANDIDATES.md) and [`SUCCESSOR_STUDIES.md`](./SUCCESSOR_STUDIES.md) (Eqs. (2)–(3), (8)–(10); full-rank Gram requirement) | none identified for the claims used here |
| Friedman, Hastie & Tibshirani 2008 (graphical lasso); Banerjee, El Ghaoui & d'Aspremont 2008; Meinshausen & Bühlmann 2010 (stability selection) | cited by MSSL as its own references for the `Ω` step and the stability-selection procedure | `PRIMARY_SOURCE_NOT_VERIFIED` — [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §6 records explicitly that these were not opened in the project's passes. Statements about selection/support recovery below are attributed to that recorded gap, not to a verified read |
| Hurlbert 1984 (pseudoreplication) | `GENERAL-METHOD` — background for the vocabulary in L4 | not read in this project's passes; no repository artefact depends on it |
| Repository artefacts (protocol, F1–F10, DG-0001, TR-0007 plan, solver test, DEC-0013–0016) | `OBSERVED`, read in this worktree | `TR-0007`/`DG-0007`/`TR-0008` are records owned elsewhere; nothing here alters them |
| The `3 × 3` worked example and every number derived from it (§1.1, §2, L2 SEs, L4 rank demo) | `INFERRED` — computed in this pass in a throwaway exact-arithmetic calculation | none; the arithmetic is exact (fractions) or closed-form |

---

## 8. Non-claims

* No ranking, no winner, no endorsed candidate, no registered or authorised study.
* No primary PDF was reopened for this note (see the provenance paragraph at the top of this
  file); all paper claims carry the ledger they come from and prior recorded gaps are carried
  forward, not closed.
* The classification in §4 is about *claims in their stated form*. It does not assert that the
  underlying mechanisms are wrong, only that the present record cannot license the inference.
* Nothing here re-opens or re-specifies the protocols of `TR-0007`, `DG-0007` or `TR-0008`.
