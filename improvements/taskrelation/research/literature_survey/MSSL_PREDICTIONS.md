# MSSL_PREDICTIONS — falsifiable sparse-precision outcomes for the registered `TR-0007` screen

**Scope.** This is a **prediction** document for the already-registered `TR-0007` p-MSSL arm
(§"Fixed arm facts" below) and for the sparse-vs-dense relation questions that arm can and
cannot answer. It registers **no** study, creates **no** authorization, launches **no** run,
and prescribes **no** `λ` sweep. It does not rank arms and does not declare a winner. Every
statement about an unrun arm is `HYPOTHESIZED`, not a result.

**Siblings / crosslinks (this directory).** [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md)
(the Schur-complement, §3 × 3 and `λ₂`-scale derivation this file builds on),
[`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) (the `m = 3` resolution limits and the
8-element support space), [`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md)
(conditional-vs-direct dependence, the SPATS many-task warning),
[`SUCCESSOR_STUDIES.md`](./SUCCESSOR_STUDIES.md) (proposal placeholders; includes the
explicit rule not to duplicate `TR-0007` and not to run a `λ` sweep),
[`CANDIDATES.md`](./CANDIDATES.md) (`H_dense` / `H_conditional` framing),
[`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) §1 (effective sample size) and §8 (status
reconciliation), [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md) (source ledger).
[`MTRL_PREDICTIONS.md`](./MTRL_PREDICTIONS.md) is the companion for the in-category control.

**Repository evidence (read, not paraphrased).** `improvements/taskrelation/research/studies/TR-0007/{PLAN.md,NOTE.md}`;
`improvements/taskrelation/04-mssl/{README.md,mssl_config.yml,mssl_model.py}`;
`improvements/taskrelation/research/{STUDIES.jsonl,DECISIONS.md,FINDINGS.md,VARIANT_BENCHMARK_PROTOCOL.md}`;
`improvements/taskrelation/research/literature/goncalves-2016-mssl.md`; primary source
Gonçalves, Von Zuben & Banerjee, *JMLR* 17(33):1–30, 2016, Eqs. (3), (4a–b), (8), (9)–(11),
§§3.1, 3.3, 4.1 (verified in the card and the `NOTE.md` transcription correction).

**Epistemic key.** `OBSERVED` = read from a primary source or a repository artefact;
`DERIVED` = algebra performed here, shown (= `INFERRED` in the sibling docs);
`HYPOTHESIZED` = predicted, falsifiable, **unmeasured**. Source-verification gaps are flagged
where they occur.

---

## 0. Status of the arm — nothing here is a result

`OBSERVED` (this checkout, `9955166`, `STUDIES.jsonl` record `TR-0007`):

* `status: pre_registered`, `stage: screen`, `method_status: faithful published-method
  implementation`, `lambda_2: 0.01`, `lambda_0: 1.0`,
  `lambda_2_selection: researcher-fixed`.
* The study folder holds `PLAN.md`, `NOTE.md`, `analyze_screen.py`, compute plans
  and control configs — **no corrected screen result or analysis artifact in this
  checkout**. No assertion is made about other worktrees or remote execution state.
* The local record is internally time-skewed: `blocked_on` retains language
  preceding the newer `screen_authorization: DEC-0016` and NOTE append. Treat
  DEC-0016 as authorization of a **seed-42 screen only**, not evidence it ran;
  reconcile the registry before citing any result. The earlier plan header
  says no authorization while its appended note records the later decision.
* **Historical tracking evidence (do not substitute, do not re-discover as new).**
  `OBSERVED` in the record conflict block and `NOTE.md`: DagsHub already holds `TR-0007` screen
  runs from a code line **absent from this repository's history** (commits `10aaaea3…`,
  `3df542d…`, experiment `taskrelation-variant-benchmark`, seed **0**, a different
  implementation with a `covariance_normalization` knob, no `lambda_2_selection` tag):
  p-mssl `0.9607`, classical-mtrl `0.9744`, wavcse-baseline `0.9737` (`test_epoch_acc_all`),
  plus a `scale-corrected` `p-mssl-correlation` run at `0.9644`; all tagged `rejected`.
* **This document assumes no concurrent worktree has finished.** `TR-0008` and `DG-0007` are
  named in `DECISIONS.md`/`STATE.md` but neither is evaluated here; `TR-0008` is
  project-original (`DEC-0014` §2) and `DG-0007` is the control-side audit
  ([`MTRL_PREDICTIONS.md`](./MTRL_PREDICTIONS.md)).

---

## 1. Fixed arm facts the predictions are conditional on

`OBSERVED` (`PLAN.md`; `mssl_config.yml`; `04-mssl/README.md`; `DEC-0015`; `DEC-0016`):

| Component | Value | Source |
|---|---|---|
| Objective (paper Eq. 3 / 4b) | `L(W) + λ₀ tr(W Ω Wᵀ) − d log|Ω| + λ₁‖W‖₁ + λ₂‖Ω‖₁` | JMLR Eq. (3), (4b); `NOTE.md` correction |
| Ω step (paper Eq. 8) | `min_{Ω≻0} λ₀ tr(SΩ) − log|Ω| + (λ₂/d)‖Ω‖₁`, `S = (1/d)WᵀW` | JMLR Eq. (8), verbatim |
| Solver | paper ADMM Eqs. (9), (10a–c), (11); off-diagonal ℓ₁; float64; `ρ` auto-start `1/mean diag(S)` and residual rebalancing (Boyd et al. 2011 §3.4.1); **self-certifying**; returns the split variable `Z` whose zeros are **exact** | `mssl_model.py::graphical_lasso_admm` |
| Summary matrix | `W ∈ ℝ^{3×2001}`: per task `[mean(head.weight, 0); mean(bias)]`, rows unit-normalised (`normalize_w: true`); `d = 2001` | `04-mssl/README.md`; `mssl_config.yml` |
| `λ₀`, `λ₁` | `λ₀ = 1.0` (paper setting); `λ₁ = 0.0` (paper's exclusive-Gaussian case) | `mssl_config.yml` |
| `λ₂` | **`0.01`, researcher-fixed; no grid run; `lambda_2_selection = researcher-fixed`** | `DEC-0016` |
| Protocol | `smp` 0.5, all 25 layers, `ks_si_er`, 30 epochs, batch 2048, AdamW 2.5e-3, ≈2,820 steps, `epoch` checkpoint, seed 42 screen | `VARIANT_BENCHMARK_PROTOCOL.md` §1 |
| Mandatory controls | classical MTRL (`normalize_w: true`) **and** matched wavCSE baseline | protocol §2 |
| Warmup / cadence | `mssl.warmup_epochs: 3`, `omega_update_frequency: 1` (the control's schedule) | `mssl_config.yml` |

`DERIVED` scale fact (from [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §5.1):
with unit rows (`R̃ = W Wᵀ`, unit diagonal) the implemented Eq. (8) equals

```
λ₀ tr(R̃ M) − log|M| + λ₂ ‖M‖₁ + const(d),      M = Ω/d,                      (B)
```

so `λ₂` acts **in the paper's standardised scale** at the numeric value `0.01` — *conditional
on* `normalize_w: true`. If `normalize_w` were off, the effective penalty is `λ₂/c²` for row
norm² `c²` (an observation, not a proposal). The registered `analyze_screen.py` reads and
reports `model.mssl_lambda_2` and `mssl.lambda_2_selection` per run and carries the scoping note
that `λ₂ = 0.01` is researcher-fixed (`DEC-0016`), not a paper default. **No `λ` sweep is
licensed by this document** (`SUCCESSOR_STUDIES.md` §7; `DEC-0016`).

---

## 2. The `m = 3` precision geometry the predictions rest on  `DERIVED`

1. **Conditional-independence content.** `Ω_ij = 0 ⟺ w_i ⫫ w_j | w_{[3]\{i,j}}` (JMLR §3.1,
   3.3 `OBSERVED`). Partial correlation `ρ_{ij·k} = −Ω_ij/√(Ω_iiΩ_jj)`.
2. **Exact `3×3` identity** (sibling §2.3, numerically verified there):
   `Ω_ij = (Σ_ikΣ_jk − Σ_ijΣ_kk)/det(Σ)`, so `Ω_ij = 0 ⟺ Σ_ij = Σ_ikΣ_jk/Σ_kk`.
3. **Unit-diagonal proxy.** With `R̃ = W Wᵀ` unit-diagonal (`normalize_w: true`), define the
   Schur off-diagonal `g_ij := R̃_ij − R̃_ik R̃_jk` (`k` the third task). `g_ij` is the
   un-normalised conditional-covariance off-diagonal, i.e. the marginal association that
   survives conditioning. The ℓ₁ kills an edge when the *regularised* estimate of this
   quantity falls inside the penalty band; the partial correlation is the interpretable
   normalisation of the same number. `DERIVED`, with the honest caveat that the exact `λ₂`
   boundary also carries the ADMM soft-threshold scale `λ₂/(d·ρ)` and diagonal reweighting, so
   the *boundary* is solver-convention-dependent even though the *object* is not.
4. **Support space.** `2³ = 8` patterns ([`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) §2):
   full triangle (dense), three two-edge chains, three single edges, empty graph. A chain's
   endpoints are conditionally independent *given the centre*; a single edge isolates a task.
5. **Equicorrelation bound.** If `R̃` is equicorrelated with `r`, then `ρ_{ij·k} = r/(1+r) ≤ 1/2`
   (sibling §3.2). So a *uniform* precision and a *uniform* covariance are the same absence of
   pair-selectivity, seen in two scales.
6. **Barrier.** `−log|Ω|` diverges as `Ω` becomes singular, so the p-MSSL optimum is strictly
   interior (`Ω ≻ 0`); it **cannot** equal the rank-1, `±1/3` MTRL limiting covariance
   (sibling §4.1). “Does not saturate by construction” is a statement about the barrier, not
   about information (`Ω = Σ⁻¹` is a bijection).

---

## 3. Minimal evidence bundle — what a usable sparse/dense read requires

None of these has been produced by this checkout (`HYPOTHESIZED` that they would be). Each row
names the decision it feeds; omitting one makes the corresponding gate untestable.

| # | Quantity | Symbol / shape | Why it is required | Feeds |
|---|---|---|---|---|
| 1 | Summary Gram **and row norms** | `R̃ = WWᵀ` (3×3) + `‖w_t‖` | `Ω = Σ⁻¹` is a reparameterisation: pair structure must be shown to exist in `R̃` at all; the `g_ij` proxy is read from `R̃` | G4, P1, P3 |
| 2 | Raw precision per epoch | `Ω` (3×3, and `tr`, `cond`, min-eig) | the ℓ₁ acts on this scale; conditioning/boundary state | G1, G4, P5 |
| 3 | Normalised partial correlations | `−Ω_ij/√(Ω_iiΩ_jj)` | the interpretable object; a raw-zero is not a partial-correlation threshold | G4 |
| 4 | **Exact support** (3 bits) | `1[Ω_ij = 0]` on off-diagonals | the only identifiable sparse event; the solver returns exact zeros | G4, G5 |
| 5 | Solver certificate | primal/dual residual, iterations, final `ρ`, `d` used | proves Eq. (8) was solved, not a rescaled surrogate (the historical failure mode) | G1 |
| 6 | Label block | `λ₀, λ₁, λ₂, lambda_2_selection, normalize_w, d` | `λ₂ = 0.01` is researcher-fixed; the screen speaks to that value only | G0, G3 |
| 7 | Per-task + aggregate accuracy vs **both** controls | protocol §4 | the only outcome object; SI-weighted aggregate can mask ER | G3 |
| 8 | Saturation state | equicorrelated-ness of `R̃` and of the partial-correlation triple | a saturated object carries no pair-specific information (F7) | G4 |

A **zero-edge ablation arm does not exist** in the registered design, and the screen has one
p-MSSL run per seed. Therefore bundle rows 1–4 can show *what the estimator did*; they cannot
by themselves show that a zero *helped* (see G5).

---

## 4. Contrasting hypotheses

The **support** alternatives H-dense and H-sparse are distinguishable in the
registered screen. H-beneficial-pair and H-harming require an additional
matched intervention; no screen accuracy difference attributes cause to one zero.

| ID | Hypothesis | Mathematical signature | Contrasted with |
|---|---|---|---|
| **H-dense** | At `λ₂ = 0.01` all three off-diagonal precisions survive. | `support = {12,13,23}`; `\|g_ij\|` all above the penalty band; entry shrinkage only | H-sparse |
| **H-sparse** | At `λ₂ = 0.01` at least one off-diagonal is exactly zero. | one or more of the 8 support patterns minus the triangle; chain or sparser | H-dense |
| **H-beneficial-pair** | A suppressed edge removes harmful *conditional coefficient coupling*. | H-sparse **and** a separately controlled present-vs-absent edge intervention showing per-task improvement | H-inert / alternative optimization effects |
| **H-inert** | `Ω` is diagonal, so the **cross-task head penalty** is absent; shared-trunk MTL and diagonal regularization remain. | `support=∅`, diagonal `Ω`; no inference of baseline equivalence | H-dense / selective coupling |
| **H-harming** | A zero removes useful *conditional coefficient coupling*. | H-sparse **and** a separately controlled present-vs-absent edge intervention showing harm | H-beneficial-pair / generic tuning effects |
| **(H-singular)** | “The finite-objective optimum is exactly singular precision.” | — | mathematically excluded by the log-det barrier; near-singularity is not excluded |

**Read this before using `H-beneficial-pair`.** The `m = 3` literature warns that sparse
relations are a *many-task* strategy: SPATS (AAAI 2017) reports weaker performance on a
**4-task** dataset and attributes it to insufficient transfer under sparse coupling
(`TASK_RELATION_TAXONOMY.md` §“m = 3 geometry constraints”; card
`literature/zhang-yang-2017-spats.md`). At `m=3`, a wrong zero removes
one of three direct conditional-graph edges, while the shared trunk still
couples tasks even when that summary edge is zero.
The researcher-fixed `λ₂=0.01` cannot be *reinterpreted*
after the outcome; its actual support is an empirical question, and even
all three surviving edges would not prove a general dense-transfer benefit.

---

## 5. Falsifiable predictions

| # | Prediction | Falsified if |
|---|---|---|
| **P1 — support prediction** | For representative unit-diagonal Grams in the prior analysis, λ₂=.01 retained all three edges; **hypothesis**: the actual screen may also be dense. | Any exact off-diagonal zero refutes a universal dense-at-.01 prediction; no conclusion about other λ values. |
| **P2 — scale diagnostic** | Compare conditional Gram residual `g_ij=R̃_ij−R̃_ikR̃_jk` with raw precision and partial correlations; diagonal normalization and joint graphical-lasso fitting can change the ordering of selected edges. | A claim that the smallest raw `g_ij` **must** be the first zero lacks a source-backed theorem; examine actual fit rather than assuming it. |
| **P3 — causal gate** | A zero edge does not by itself establish benefit: the screen has no otherwise-matched present-vs-absent edge arm. | If a separately authorized, paired edge intervention isolates a beneficial effect, upgrade the claim; the registered screen alone cannot. |
| **P4 — estimator contrast** | Dense precision need not behave like MTRL: log-det/barrier, covariance trace constraint, diagonal scale and Ω-update differ. | If a fully matched empirical comparison shows indistinguishable outcomes, only the *outcome* is indistinguishable at its resolution, not the objectives. |
| **P5 — solver validity** | Finite-objective precision is PD; trace is not fixed and an exactly rank-one precision is excluded. | Returned exactly singular precision with finite objective, or failed optimality certificate, invalidates the intended solver claim. |
| **P6 — input convention** | With unit-normalized summary rows and d=2001, inspect the Eq.(8) scaling identity and solver certificate at researcher-fixed λ₂=.01. | Wrong `d`, different normalization/penalty convention or failed certificate invalidates comparison. A matrix numerically close to the unpenalized inverse at small positive λ is **not** by itself a failure. |
| **P7 — outcome reporting** | Report all three per-task effects beside the SID-weighted aggregate and both mandatory controls. | Aggregate-only interpretation cannot identify task-wise utility; ER ordinary split cannot establish speaker-independent benefit. |
| **P8 — historical context, not a prediction** | Incomparable earlier seed-0 code-line runs report p-MSSL .9607 vs MTRL .9744 and baseline .9737. | These numbers cannot classify the registered corrected seed-42 screen. |

**Not evidence** (each a recorded project failure mode): a single-seed accuracy delta (F1);
an ordinary-split ER delta (≈15 pp speaker leak, F3); any comparison with unmatched
pooling/layers (F2: pooling alone moved ER 3.25 pp); a saturated relation object read as
“structure” (F7); a gradient-scale rationale for a relation mechanism (F9 → F10, DEC-0010);
counting zero edges as a benefit without a paired outcome and a matched ablation (P3).

---

## 6. Decision gates (mapped from the registered plan — none invented here)

Order matters: a gate read early makes the later ones meaningless.

* **G0 — eligibility / authorization.** Is the run in scope, at the pinned commit, under the
  frozen protocol, with the `lambda_2_selection: researcher-fixed` label and the two controls?
  `OBSERVED` a status inconsistency in `STUDIES.jsonl` (`blocked_on` vs `screen_authorization`);
  resolve the registry, do not silently prefer one read.
* **G1 — solver / faithfulness gate.** Eq. (8) solved with the explicit `d`, off-diagonal ℓ₁,
  a closing primal–dual certificate, and `Ω` in the paper's units. Evidence: bundle rows 3, 5.
  A failed certificate → the run is an execution failure, not a scientific result.
* **G2 — protocol / exposure gate.** Pooling, layers, epochs, batch, steps (≈2,820), sampler
  identical across arms (protocol §10). Otherwise the comparison is void (F2, F8).
* **G3 — outcome gate (pre-registered).** `PROMISING` iff p-MSSL > **both** controls on
  `test_epoch_acc_all` **and** every per-task accuracy within 0.20 pp of both; else `REJECTED`;
  `INCONCLUSIVE` if an arm cannot run under §1-settings or an exposure check fails. A screen can
  never promote (F1); confirmation is a separate, later, explicitly authorized stage.
* **G4 — support read (informational, not a promotion rule).** Report the exact 3-bit support,
  the raw `Ω`, the partial correlations and the eigenvalues/condition number; label the scale
  the ℓ₁ acted on. This distinguishes H-dense / H-sparse / H-inert and nothing else.
* **G5 — sparse-benefit gate (cannot be satisfied by this screen).** To assert
  `H-beneficial-pair` (or `H-harming`) one needs a matched comparison between *the same arm with
  the edge present and absent*, under the frozen protocol, with paired seeds and a
  pre-registered direction. That arm is not part of `TR-0007`, and constructing one is a new
  registration requiring human scientific review (`SUCCESSOR_STUDIES.md`; `DEC-0013` §3). A
  zero edge plus a favourable `REJECTED`/`PROMISING` read is **not** this gate.
* **G6 — confirmation gate.** Seeds `0–4`, paired per-seed differences against both controls;
  any ER claim additionally needs speaker-independent LOSO (F3). Not authorized by `DEC-0016`.

---

## 7. Alternative explanations for a bad (`REJECTED`) outcome

Each row is an explanation that a `REJECTED` screen cannot by itself exclude; the last column
is the discriminator already inside the registered design or the literature record.

| # | Explanation | Why it stays live | Discriminator |
|---|---|---|---|
| A1 | **`λ₂` was never selected.** `0.01` is researcher-fixed, the paper selects on data and publishes no transferable scale. A `REJECTED` reads “did not help at this fixed `λ₂`”, not “cannot help”. | `DEC-0016`; `STUDIES.jsonl` record | The `lambda_2_selection` label + the support/shrinkage read (G4). **Not** a `λ` sweep. |
| A2 | **The summary adapter binds.** The Gaussian-row premise is an approximation; the summaries are rank-degenerate and a shared trunk is not three aligned coefficient columns. | `04-mssl/README.md` deviation 1; `OPEN_QUESTIONS.md` §1, §6 | Eigenvalues/rank of `R̃` (row 1); whether the harm localises to the widest head |
| A3 | **The ℓ₁ acted on the wrong scale for the question.** Penalising raw `Ω` is not a partial-correlation threshold; `λ₂ = 0.01` may shrink without zeroing, so “no support change” and “no effect” are different statements. | sibling §4.3 | raw `Ω` **and** partial correlations side by side |
| A4 | **SI-weighted aggregate masking.** SI has the largest test set; small SI movement can move the aggregate while ER moves more in points. | protocol §4 | per-task numbers |
| A5 | **ER ordinary split leaks speakers.** Screen ER is context only; no ER claim. | F3 | LOSO (G6), not available at screen |
| A6 | **Seed noise.** Single-seed screens cannot carry a claim; single high scores must not be read as improvements. | F1 | G6 confirmation |
| A7 | **Numerical/implementation residue.** `S` floor, `ρ` schedule, float64, certificate — an unconverged solve would masquerade as `Ω` behaviour. | `mssl_model.py`; `NOTE.md` (old solver returned `63·I` vs optimum `7e4·I`) | G1 certificate |
| A8 | **Historical provenance confusion.** The seed-0 DagsHub runs are a different implementation/seed and must never be merged into this screen. | `DEC-0016` preservation rule | exact git SHA + `lambda_2_selection` tag |
| A9 | **Pre-existing evidence that the *conditional* estimand may be the wrong one for this task set.** SI/ER appears harmful both directions under approximate exposure control; no replicated helpful edge exists. | F8; synthesis “Active hypothesis assessment” | G4 (does a zero line up with a harmful pair?) — hypothesis-generating only |

---

## 8. What a zero would and would not prove

| Observation | Entailed | **Not** entailed |
|---|---|---|
| `Ω_ij = 0` exactly at `λ₂ = 0.01` | at that fixed penalty and this estimator, the conditional association is inside the shrinkage band; the support is an identifiable event | that the pair is truly conditionally independent in the population; that removing it *helped*; that SPATS-style covariance sparsity would behave the same |
| support = full triangle | the estimator found no edge below the penalty band | equivalence to MTRL (P4); that sparse conditional relations are harmful |
| `Ω` dense **and** a `PROMISING` screen | the arm beat both controls under this protocol at this `λ₂` | mechanism attribution to sparsity (no support contrast); external validity beyond `smp` 25 L / this seed |
| `ρ_{ij·k}` large, `g_ij` small | the normalized partial correlation and the un-normalized conditional association are different readouts | which one “caused” any accuracy change |
| empty support / diagonal `Ω` | the coupling degenerated to diagonal per-task weighting (H-inert) | that the model equals the baseline (the trunk and diagonal penalty remain) |

---

## 9. Bounds of this document

* **No new protocol, no new study, no `λ` sweep.** Anything that would require one is stated as
  a gate or a missing arm, never as a plan.
* **Conditional predictions only.** P1–P7 are statements about what the registered design
  *would* show; none is a measurement. P8 is context from an unmergeable historical code line.
* **Unresolved registry inconsistency** (§0) is recorded, not smoothed over.
* **Source-verification gap (stated, not hidden).** The `λ₂`-threshold behaviour rests on the
  paper's own citation of Friedman, Hastie & Tibshirani (2008) for the Ω-step and on the
  representative numerical solves in the sibling doc; that citation has **not** been
  independently opened by this branch (sibling §6.2). The exact `λ` boundary at `m = 3` is
  therefore a solver/derivation statement, not a primary-source-verified closed form.
* **No corrected seed-42 screen result is present in this checkout.** Historical
  seed-0 runs remain a separate, non-comparable record; new or parallel
  execution status must be reconciled from its own evidence before use.
