# Task Relation Learning — emerging method-selection framework

Date: 2026-09-22 · Status: **working synthesis at an evidence boundary**, not a completed framework.

This document serves the programme's larger scientific objective: to understand how measurable task relationships and transfer behaviour predict which parameter-based MTL mechanisms are appropriate. It is derived from `FINDINGS.md` (authoritative for numbers and provenance), `DECISIONS.md`, `FAILURES.md` and the Study folders. The per-quantity MTRL evidence lives in `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`; this file is the cross-study selection view.

It must be able to say *"no relation mechanism is justified"* — that is the current answer for KS/SI/ER under the supported representation, and stating it is a result, not a failure to deliver one.

---

## 1. The chain, and where each link is conditioned

The intended chain is: task/data characteristics → task relationships → transfer behaviour → appropriate mechanism. Every link measured so far turned out to be **conditioned on something the earlier framing treated as fixed**. That conditioning is the framework's main content.

| # | Observable | Measured signal | Evidence | What it does **not** establish |
|---|---|---|---|---|
| 1 | Representation choice | Pooling alone moved `er` by `+3.25pp` (`mix` → `lnp`, no architecture change); Ω changed qualitatively across pooling | F2, F5 | That a mechanism caused any difference seen under unmatched pooling |
| 2 | Perturbation axis | 25L `smp` KS↔SI is fold-stable (10/10 positive, `0.286 ± 0.012`) and seed-stable, while 16L KS↔SI is seed-unstable (`0.110 ± 0.236`) and SI↔ER is the only fully sign-consistent edge at 16L | F6 | Any task-intrinsic stability ranking, or that stability transfers across axes |
| 3 | Saturation state | Ω magnitude saturates to ±`1/3` at `smp` 25L and `lnp` 16L, leaving no pair-specific information | F5, F7, F9 | That saturation **causes** the outcome null (no intervention has moved saturation alone) |
| 4 | Optimizer exposure | Raw ER-directed "transfer" of `+0.34`/`+0.28` LOSO collapsed to `+0.0053` (CI spans zero) and `−0.0589` under step-matched ER-only controls | F8 | Semantic transfer from any same-epoch pair-minus-single comparison |
| 5 | Training mixture | Gradient-norm dominance is produced by per-batch sample counts: raising ER from ≈`47/2048` to ≈`435/2048` took the late ratio `8.962 → 2.352` in 5/5 seeds | F10 | That gradient scale is a property of a task, or that the residual drop is pure estimator variance |
| 6 | Pairwise gradient interaction | Late cosines near zero in every seed; no pair met the persistent-conflict threshold | F9, FL-0002 | That conflict-aware mechanisms have anything to act on here |
| 7 | ER evaluation protocol | The ordinary split inflates ER by ≈`15pp` (`0.7902` vs LOSO `0.6391 ± 0.0506`); per-fold SD ≈`5pp` | F3 | Any ER generalization claim from a single split or fold |
| 8 | Relation strength vs utility | 25L KS↔SI is maximally stable/saturated near `+1/3` while neither KS nor SI gains materially | F4, F6 | That Ω magnitude can serve as a transfer or utility proxy |
| 9 | Downstream outcome | Matched five-seed MTRL deltas: 16L aggregate `−0.00010` `[−0.00183, +0.00163]`; 25L aggregate `−0.00056` `[−0.00243, +0.00130]`; ER LOSO `−0.0011` `[−0.0307, +0.0285]` | F4, FL-0001 | That MTRL is *exactly* equivalent — only that no advantage is demonstrable at this resolution |

Reading the table downward is the framework's first rule: **a signal is only interpretable together with the conditions it was measured under.**

---

## 2. Two classes of quantity, and why they must not be merged

The programme's evidence now separates them cleanly:

* **Relational quantities** — relation magnitude (Ω entries), relation confidence (dispersion across seed/fold), relation structure (symmetry, selectivity). These describe how tasks are coupled; classical MTRL models exactly this class (DEC-0001).
* **Non-relational optimization quantities** — task loss/gradient scale, gradient-estimate variance, per-task weighting, sample composition, optimizer exposure. These describe how the shared parameters are updated.

Two rules follow, both already validated:

1. **A non-relational quantity cannot justify a relation mechanism.** The F9-era rationale ("ER has intrinsically larger gradients, so static covariance is inadequate") was withdrawn once F10 showed the scale is mixture-controlled (DEC-0010, FL-0004).
2. **A relation mechanism cannot be justified by a scale artifact, and a scale method cannot be smuggled in as a relation method.** LT-0001 screened eight primary sources: GradNorm and uncertainty weighting address scale directly but learn per-task scalars with no relation object; Bayesian covariance methods learn explicit relations but assume aligned Gaussian multi-output regression and regulate no gradient scale (FL-0003, `literature/INDEX.md`).

**Framework statement.** Relation magnitude, relation confidence and optimization scale are three distinct measurable quantities. Any mechanism proposal must name which one its assumption acts on, or it is out of category.

---

## 3. Evidence levels, and what each licenses

| Level | Meaning | May it select a mechanism? |
|---|---|---|
| **A** | Survived the strongest protocol available for that claim: matched multi-seed, speaker-independent LOSO, or structural replication across folds/settings | Only together with R8's promotion bar, and only within its stated condition |
| **B** | Moderate: replicated descriptive patterns, or matched single-run comparisons with controls | No — motivates a diagnostic, not a selection |
| **C** | Preliminary/confounded: single seed, unmatched pooling, differing checkpoint policy, leaky split | No — cannot carry a relation or architecture claim (F1, F2, F3) |

Every claim in this framework therefore carries its condition: pooling, layers, seed/fold axis, saturation state, sampling regime, and evaluation protocol.

---

## 4. Selection rules supported by current evidence

* **R1 — Pin the representation first.** If pooling/layers are not fixed and matched, no relation or mechanism comparison is interpretable (F2, F5).
* **R2 — One seed screens; five seeds select.** Sub-1pp effects require ≥5 seeds; single-fold ER results are meaningless (F1, F3).
* **R3 — Do not read a saturated Ω as relation structure.** Saturation removes pair-specific information and is an observed failure mode (F7).
* **R4 — Control exposure before attributing semantics.** Changing task count or composition changes optimizer steps and per-task batch size; match or model them (F8).
* **R5 — Name the sampling regime in any gradient-scale claim.** Gradient scale is a training-mixture property, exactly as relation claims must name pooling and axis (F10).
* **R6 — ER requires LOSO before any performance claim.** The ordinary split is screening evidence only (F3).
* **R7 — Stay in category.** The mechanism must be explicit Task Relation Learning, not low-rank, clustering, decomposition, generic loss weighting or gradient surgery (DEC-0005, FL-0003).
* **R8 — Promotion bar.** A candidate must beat the matched wavCSE baseline across seeds with no material regression (> `0.20pp`) on any task, and ER claims additionally need LOSO (OBJECTIVE.md).

---

## 5. What the framework currently implies for KS/SI/ER

| Question | Current answer | Basis |
|---|---|---|
| Is a dense symmetric relation structure *representationally* inadequate? | Not demonstrated — the asymmetry that suggested it was an exposure artifact | F8, DG-0001 |
| Are useful relations sparse/selective? | Unknown empirically; no pair-selective beneficial transfer has been measured | BACKLOG TR-0001 |
| Are relations directional in a way that matters? | Unresolved; DG-0001's raw asymmetry did not survive exposure control | F8 |
| Do relations need to be dynamic? | Not indicated: late gradients are near-orthogonal with no persistent conflict, and no phase-specific negative transfer was found | F9, FL-0002 |
| Do ER's relations need reliability weighting *because ER is data-poor*? | Not on the evidence gathered: the measurable data-regime effect is gradient-estimate scale, which a relation method does not address; relation-estimate noise under matched composition is still unmeasured | F10, DEC-0010 |
| Are relations layer-specific? | Untested | BACKLOG TR-0005 |
| Is the parameter summary adequate? | Plausible but unproven; Ω disagrees with controlled transfer and normalization moves Ω radically | BACKLOG TR-0006, DG-0003 |
| **Does a relation mechanism beat matched wavCSE today?** | **No.** Champion remains the matched wavCSE baseline | F4, current champion in STATE.md |

---

## 6. Decision points currently open (human)

The programme is gated on a scope decision, not on compute (DEC-0009, DEC-0010).

**(a) Bounded diagnostic first.** Pre-register one `DG-xxxx` Study that separates the two surviving explanations for DG-0005's ER norm drop: reduced gradient-estimate variance versus ER convergence/overfitting under ~9× more ER updates per epoch. It reuses the committed sampler; validation/test stay untouched; no mechanism is implied. This is the only question whose answer would change what the framework can claim about scale.

**(b) Close experimentation and finalize the framework.** The conditioned-quantity table plus the "no mechanism justified" conclusion is already a defensible characterisation result. This option stops GPU work and improves documentation only.

**(c) Revisit Option 3 (project-original relation-plus-reliability hybrid).** Requires an explicit human authorization and a mechanism rationale that does *not* rest on F9's withdrawn task-intrinsic reading; it must be labelled project-original, not literature-derived (DEC-0009 §4).

Doing none of these is also coherent: the framework's current output stands on its own.

---

## 7. Highest-information missing evidence, ranked

1. **Estimator variance versus ER convergence** (option a). Highest value because it decides whether "scale" is an estimator property or a saturation artifact — the one remaining ambiguity in F10. Cost: two arms × one seed to screen, ≈0.7 GPU-hours, using the existing default-off sampler.
2. **Relation-estimate noise under matched composition.** Does ER-edge Ω instability track per-batch sample count? Requires an MTRL arm under DG-0005's A1 composition, which DG-0005 explicitly excluded. Would test whether relation *confidence* is also mixture-conditioned — a directly framework-relevant question, and the only route that would make TR-0003's assumption checkable rather than assumed.
3. **Layer-wise gradient relation.** Whether shared low-level/divergent high-level structure exists (TR-0005); instrumentation only, no architecture change.
4. **Parameter-summary adequacy.** Whether the mean-head summary destroys relation information (TR-0006/DG-0003); would need a matched protocol that couples triple-task Ω to a pairwise transfer target.
5. **Prospective stability prediction.** Whether any stability statistic predicts outcome on new runs (DG-0004's remaining half).

Items 1 and 2 are cheap and directly answer framework questions; items 3–5 are progressively further from a decision the framework must make.

---

## 8. Using this framework

1. Measure the quantities in §1 under a pinned representation, with explicit seeds and recorded provenance.
2. Classify each quantity as relational or optimization (§2) before proposing anything.
3. Apply the rules in §4; if a rule's condition is unmet, the comparison is inadmissible.
4. A mechanism proposal must pass both gates of DEC-0005 and state which framework row it addresses, with its promotion protocol fixed in advance (R8).
5. Report the conditioned table even when the conclusion is negative — the characterisation is the contribution.

---

## Provenance

| Study | Role | Decision | Evidence |
|---|---|---|---|
| `BL`/legacy `wavcse-baseline`, `taskrelation-mtrl` | Baseline and MTRL matched evaluation (16L, 25L, seeds 0–4, ER LOSO) | Reference | F1, F3, F4; `bfb1ad44` |
| `02-lnp` | Diagnostic/control: pooling confound | Retained as control | F2, F5 |
| `DG-0001` | Directed transfer matrix + exposure controls | Weakened by F8 | `studies/DG-0001/` |
| `DG-0002` | Exposure-controlled gradient compatibility | CONFIRMED diagnostic | F9; `studies/DG-0002/confirmation_result.json`; `7f6d5248` |
| `LT-0001` | Scale/reliability literature gate | REJECTED — no eligible mechanism | R3, FL-0003; `literature/INDEX.md` |
| `DG-0005` | ER data-regime gradient-scale control | CONFIRMED diagnostic | F10; `studies/DG-0005/confirmation_result.json`; `8032a937` |

Related: `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md` (per-quantity MTRL evidence), `FINDINGS.md` (authoritative numbers), `DECISIONS.md` (binding scope), `FAILURES.md` (negative evidence).
