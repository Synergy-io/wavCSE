# Decisions — Task Relation Learning

Binding decision record for this branch. An entry here must not be silently
re-litigated: it stands until the human changes it, or until new evidence is
recorded in `FINDINGS.md` and the entry is explicitly superseded.

Each entry: ID, title, status (`ACTIVE` / `SUPERSEDED`), date, decision,
rationale, evidence, consequences.

## Terminology (avoid a real ambiguity)

* **wavCSE baseline** — the matched reproduction of the paper's feature-based
  approach (`improvements/base/`). This is the *reference to beat*, and the
  current champion (F4).
* **MTRL** — classical Multi-Task Relationship Learning (Zhang & Yeung
  2010/2014), `improvements/taskrelation/01-mtrl/`. This is the *formal Task
  Relation Learning baseline method* of the research programme: the method we
  diagnose and extend, not the thing we are trying to beat.

Both meanings of "baseline" appear throughout the older docs. Read them this
way.

---

## DEC-0001 — Classical MTRL is the formal starting method and the primary existing Task Relation Learning baseline

**Status:** ACTIVE — 2026-09-21

**Decision:** The formal research programme starts from classical MTRL.
`improvements/taskrelation/01-mtrl/` stays active: it is the starting point for
all future work in this branch, and its accumulated evidence is retained, not
retired.

**Rationale:** MTRL is the canonical method of the taxonomy branch this project
is built on (Zhang & Yang 2021, §2.4) and the only Task Relation Learning method
this project has implemented and evaluated at full rigor (16L, 25L, LOSO,
5-seed). A literature-grounded contribution has to be anchored to a named
published method, and MTRL is that anchor.

**Evidence:** F4; `01-mtrl/README.md`; the survey material in
`weekly/week-02-mtl-architectures-survey/` and
`weekly/week-05-project-proposal/my/kevin/`.

**Consequences:**

1. MTRL's negative result is the programme's first input → the next stage is
   diagnosis, not replacement.
2. No new mechanism is implemented before a diagnostic explains *which* MTRL
   assumption fails for KS/SI/ER (DEC-0005).
3. Generic MTRL hyperparameter re-tuning remains forbidden (F4; `mtrl_lambda`
   already swept — 0.05 was worse than 0.01 on all three tasks).

---

## DEC-0002 — `02-lnp/` is a diagnostic/control study, not an active method

**Status:** ACTIVE — 2026-09-21

**Decision:** LNP is removed from the active method progression and from
architecture search. `improvements/taskrelation/02-lnp/` is classified
**DIAGNOSTIC / CONTROL**. Its findings are retained and remain binding on
experimental methodology.

**Rationale:** LNP's scientific role was never to be a Task Relation Learning
method — it was the control that revealed the pooling confound. Keeping it on
the method list invites both wasted work (re-running it as a candidate) and
invalid comparisons (treating its runs as architecture evidence).

**Evidence:** F2, F5; `02-lnp/README.md` (baseline+`lnp` moved `er` by +3.25pp
with no architectural change; Ω under `lnp` saturated to uniform +1/3).

**Consequences:**

1. Every architecture comparison must pin the pooling configuration first
   (F2). An unmatched-pooling comparison establishes nothing.
2. `02-lnp/configs/*.yml` remain valid as *control* configs; they are not
   candidate method configs.
3. No new runs are scheduled under the `02-lnp` heading; its files are
   historical.

---

## DEC-0003 — `03-gbc/` is ARCHIVED / out of current formal scope

**Status:** ACTIVE — 2026-09-21

**Decision:** GBC is not part of the current formal research programme.

* It is an original project design, not an implementation of a named published
  Task Relation Learning method; no defensible published match has been
  established (an earlier attribution was retracted — see `03-gbc/README.md`).
* The implementation, configs and historical notes are **preserved** as-is.
* GBC results must not motivate, rank, or gate formal literature-grounded Task
  Relation Learning approaches.
* No autonomous research cycles are to be spent optimising, tuning or extending
  GBC unless the human explicitly reactivates it.

**Rationale:** The programme's contribution is claimed as literature-derived. A
method with no verified published source cannot carry that claim, and allowing
it into the comparison table would create an uninterpretable mixture of
literature-grounded and project-original mechanisms.

**Evidence:** `03-gbc/README.md` (citation correction, 2026-09-01); F4
(MTRL is the method with full evaluation rigor); the archived runs
`taskrelation-gbc` (2 runs, no results).

**Consequences:** GBC may still appear in a *historical* comparison table if
explicitly labelled "original project design, not literature-grounded". It does
not appear in the progression, the backlog, or method-selection logic.

---

## DEC-0004 — TSM and PMR are quarantined pending a validity audit

**Status:** ACTIVE — 2026-09-21

**Decision:** TSM (`models/tsm_model.py`, `trainers/tsm_trainer.py`,
`configs/tsm_config.yml`) and PMR (`models/pmr_model.py`,
`trainers/pmr_trainer.py`, `configs/pmr_config.yml`) remain historical /
exploratory evidence. They are **not** automatically the next research
direction, and they are not backlog items.

They may be relied on only after a written validity audit covering all four:

1. **Mathematical validity** — does the implemented objective actually
   implement the stated method (regularizer, update rule, loss terms)?
2. **Literature source** — is the attribution verified against the actual
   publication? (Precedent: GBC's citation was wrong and had to be retracted,
   `03-gbc/README.md`.)
3. **Implementation correctness** — does it run? PRM is known-broken:
   `get_task_parameter_matrix()` `torch.stack`s per-head rows of different
   lengths (`ks` 12×2000, `si` 1251×2000, `er` 4×2000), which crashes once the
   regularizer activates after warmup (`CONTINUATION.md` §11.5).
4. **Taxonomy placement** — is it genuinely a Task Relation Learning method in
   the project's Zhang & Yang §2.4 framing, and does it stay distinct from the
   low-rank / clustering / decomposition branches (a hard project rule)?

**Rationale:** Both have zero MLflow runs — never trained, never validated.
Nothing this project has measured supports them. Treating unwritten code as
"existing evidence" is how unfounded mechanisms enter a comparison table.

**Evidence:** `CONTINUATION.md` §3 (status table) and §11.5; MLflow (no `tsm`
or `pmr` experiments exist).

**Consequences:** the audit is a document, not a training run. Output: one audit
note per method (four sections above) placed next to the implementation. Pass →
they may be registered as properly gated backlog entries; fail → they are
marked unsupported and excluded from the framework.

---

## DEC-0005 — The formal progression is literature-gated: no arbitrary architecture generation

**Status:** ACTIVE — 2026-09-21

**Decision:** The only sanctioned route to a new mechanism is:

```
reproduced wavCSE baseline
  -> classical MTRL
  -> diagnostic analysis of MTRL behaviour
  -> identify concrete limitations of MTRL
  -> targeted literature search for published Task Relation Learning methods
     addressing those limitations
  -> implement justified methods
  -> screen
  -> multi-seed confirmation
  -> LOSO where ER claims are involved
  -> ablation and behavioural analysis
  -> derive method-selection framework
```

Stages are not skipped. In particular, a mechanism study requires **both**:

* diagnostic evidence (`DG-xxxx`) identifying the specific MTRL assumption that
  fails, and
* a published method (`LT-xxxx`) whose stated assumption addresses it.

**Immediate research question** (the one the programme is currently answering):

> Why does classical MTRL fail to produce a significant reproducible
> improvement over wavCSE for KS, SI and ER, and what measurable task behaviour
> indicates which Task Relation Learning assumption should replace or extend it?

**Rationale:** Arbitrary architecture generation produces mechanisms that can
be neither motivated nor interpreted, and it discards the branch's actual asset
— the already-measured KS/SI/ER behaviour (F2, F5, F6, F7). The contribution
claimed by this project is literature-derived; the mechanism must come from the
literature *after* an observed limitation, not before it.

**Evidence:** F4–F7 (why diagnosis precedes mechanism); the ordering already
implied by `BACKLOG.md`'s `DG-*` before `TR-*` numbering.

**Consequences:**

1. Targeted literature mode is **mandatory** after a diagnostic identifies the
   limitation — it is no longer only a plateau-triggered fallback
   (`BACKLOG.md`, Literature Mode).
2. Pre-committed mechanism orderings are removed from `STATE.md`. The previous
   "sparse → asymmetric → confidence → dynamic → layer" ordering was exactly
   the arbitrary sequencing this decision forbids; the hypotheses themselves
   survive as *gated* backlog entries.
3. Category-boundary rule still applies: the contribution must remain Task
   Relation Learning, not low-rank, clustering or decomposition.

---

## DEC-0006 — Historical evidence is reclassified, never erased

**Status:** ACTIVE — 2026-09-21

**Decision:** Every finding, README log, retraction and MLflow run produced
before this reset is retained. Reclassification changes how evidence may be
*used*; it does not remove it. `STUDIES.jsonl` starts empty, and legacy runs are
referenced by MLflow experiment name (see F1–F7 provenance and R2).

**Rationale:** The retraction history (F1, F3) is itself part of the
contribution — it documents how a single-seed claim and a leaky split produced
two false positives. Deleting or rewriting it would destroy the methodological
argument.

**Consequences:**

1. New runs must record Study ID, stage, seed, task set, pooling, layer config,
   git commit SHA, and a DagsHub run note (`.omp/RULES.md`).
2. Before starting a study, search `STUDIES.jsonl` and `FINDINGS.md`
   (`.omp/RULES.md`).
3. Failed studies are recorded in `FAILURES.md`, never deleted.

---

## DEC-0007 — Target literature at task-relation methods for optimization-scale imbalance

**Status:** ACTIVE — 2026-09-22

**Previous direction:** Continue MTRL behavioural diagnosis until DG-0002 determined whether the seed-42 gradient signal was reproducible. Do not enter literature or mechanism work from a single seed.

**Evidence causing the change:** F9 / DG-0002 matched seeds `0–4` confirmed ER shared-gradient norm dominance in the middle and late thirds. Baseline ratios averaged `7.243` and `8.962`; MTRL ratios were `7.129` and `9.004`, with no consistent paired reduction. No seed supported persistent pairwise conflict. Ω magnitude saturated in every seed, while one seed flipped both ER-edge signs.

**Decision:** The diagnostic gate in DEC-0005 is satisfied for a narrow limitation: classical MTRL's static head-parameter covariance does not regulate the unequal shared-gradient scale observed here. The next programme stage is targeted literature research for published Task Relation Learning methods that explicitly handle unequal task scale, task reliability, sample-size-dependent confidence, or optimization-aware relations.

**New direction:** Build literature queries from F9, verify taxonomy against primary papers, and produce structured paper cards before registering any mechanism Study. Generic loss weighting, gradient surgery, mixtures, low-rank, clustering and decomposition remain outside this gate unless the published method retains explicit learned task relations and belongs to the formal Task Relation Learning category.

**Expected consequence:** The next controller iteration performs literature work rather than GPU training or architecture implementation. A defensible candidate may become an `LT-xxxx` Study only when its stated assumption maps to F9. If no eligible published method exists, record the negative search and enter `NEEDS-HUMAN-REVIEW`. Separately, a data-regime diagnostic is still required before claiming the imbalance is task-intrinsic.

---

## DEC-0008 — No literature-eligible F9 mechanism; human scope decision required

**Status:** SUPERSEDED — 2026-09-22 by DEC-0009 (human scope decision recorded below).

**Previous direction:** DEC-0007 required targeted primary-source research for an explicit Task Relation Learning method that models unequal task scale, task reliability, sample-size-dependent confidence, or optimization-aware relations.

**Evidence causing the change:** LT-0001 screened eight primary sources against pre-registered gates. Explicit relation methods did not directly control heterogeneous deep shared-gradient scale or required aligned Gaussian/mean-estimation assumptions. GradNorm and homoscedastic uncertainty weighting directly address scale but learn per-task loss weights rather than task relations. The 2024 informative-relation method relies on `W = H + P` parameter decomposition. Rakitsch et al.’s separate signal/noise covariances are the closest relation principle, but a KS/SI/ER implementation would be a novel classification hybrid and F9 does not establish residual-noise causation.

**Decision:** No mechanism Study is authorized. The programme enters **NEEDS-HUMAN-REVIEW** rather than relabelling an adjacent MTL category or inventing a literature attribution.

**New direction:** A human must choose among:

1. retain strict Task Relation Learning scope and resume causal diagnostics, beginning with an ER data-regime/gradient-noise control;
2. explicitly broaden scope to optimization-aware MTL, allowing methods such as GradNorm as a separate branch/control; or
3. authorize a project-original relation-plus-reliability hybrid, explicitly abandoning the requirement that the mechanism itself be a faithful published method.

**Expected consequence:** No architecture implementation or GPU training begins until the scope decision is recorded. The conservative default is option 1 because it preserves the formal contribution and existing category boundaries.

---

## DEC-0009 — HUMAN DECISION: retain strict Task Relation Learning scope; Option 3 deferred behind explicit gates

**Status:** ACTIVE — 2026-09-22 (human-authored decision)

**Decision (human):**

1. **Retain strict Task Relation Learning scope now.** No mechanism is implemented under the current literature gate. Option 2 (broadening the research family to optimization-aware MTL) is **declined**.
2. **Option 3 (project-original relation-plus-reliability hybrid) is not rejected in principle** but is deferred and gated: it may proceed only when **both** hold — (a) the environment supports it (compute, disk and implementation prerequisites actually available), and (b) the human explicitly authorizes it at that time.
3. **DG-0005 is the authorized continuation:** the ER data-regime/gradient-noise diagnostic, which stays inside existing scope.
4. The GradNorm-style scale-normalization arm proposed as a diagnostic control is **not authorized** here. It is an optimization-scope change and requires its own explicit approval before any implementation.

**Previous direction:** DEC-0007 sent the programme into targeted literature mode from F9; DEC-0008 recorded that no eligible published mechanism exists and left the scope choice open.

**Evidence:** F9 (DG-0002 confirmation, seeds 0–4: baseline ER-to-smallest-task norm ratios 7.243 ± 0.272 middle and 8.962 ± 0.456 late; MTRL 7.129 ± 0.421 and 9.004 ± 1.059; no persistent conflict; Ω magnitude saturation in every seed); FL-0003 / R3 / LT-0001 (eight primary sources, no method satisfying all gates).

**Consequences:**

1. `STATE.md` phase is DG-0005 execution, not mechanism search. The `TR-xxxx` gate stays closed until a human reopens it.
2. Research-family identity in `AGENTS.md` (Task Relation Learning) is **unchanged** — Option 2's rename or new category folder must not be created.
3. DG-0005's outcome becomes the evidence base for any future Option 3 authorization: if ER norm dominance tracks the data regime, a reliability/scale-aware relation mechanism is causally motivated; if dominance persists under matched per-batch composition, the imbalance is task-intrinsic and the mechanism rationale must change accordingly.
4. Any future Option 3 work must be labelled **project-original, not literature-derived**, and must not repeat the retracted-attribution pattern that archived GBC (DEC-0003).
5. Plateau counter remains `1`; DG-0005 is a diagnostic study and does not increment it.

---

## DEC-0010 — F9's gradient-scale signal is a training-mixture property: the task-intrinsic-scale rationale is withdrawn

**Status:** ACTIVE — 2026-09-22

**Previous direction:** DEC-0007 targeted literature and future mechanism work at "unequal task-gradient scale or relation reliability", treating F9's reproducible ER norm dominance as a candidate task property; DEC-0009 authorized DG-0005 as the diagnostic that would decide whether that scale signal is a data-regime artefact or task-intrinsic.

**Evidence causing the change:** DG-0005 matched seeds `0–4` at commit `8032a937`. Holding pooling, layers, epochs, global batch, splits, optimizer, checkpoint policy and optimizer exposure fixed, and changing only ER's per-task sampling weight in the training split, ER's sampled share rose from ≈`47/2048` to ≈`435/2048` and the late max/min task-norm ratio fell `8.962 ± 0.456 → 2.352 ± 0.270` (paired `−6.610`, 95% `[−7.353, −5.867]`), clearing the pre-registered `< 3.0` threshold in `5/5` seeds with every exposure gate passing. Middle fell `7.243 ± 0.272 → 2.818 ± 0.272` (below `3.0` in `4/5` seeds). The change is ER-localized — late ER norm `6.972 → 1.608` against KS `+0.066` and SI `−0.117` — and pairwise cosines stayed near zero in both arms. The A0 arms reproduce DG-0002's baseline numbers exactly, so the sampling knob is inert when unused.

**Decision:**

1. F9's dominance is **not** a task-intrinsic property of ER at this representation. It is a property of how many examples each task contributes to a batch. Recorded as **F10**; F9 is refined in place, not deleted.
2. The rationale "ER has intrinsically larger task gradients, therefore a scale- or reliability-aware relation mechanism is warranted" is **withdrawn**. A mechanism justified by task-intrinsic scale would now rest on a refuted premise.
3. **No mechanism Study is authorized.** The `TR-xxxx` gate stays closed; DG-0005 was a diagnostic and does not open it. Option 3 remains deferred behind the two gates of DEC-0009.
4. The estimator-variance-versus-convergence confound is recorded as the highest-information open question, not as a result: A1 also multiplies ER's optimizer updates per epoch and saturates its ER head harder (train ≈0.99 vs validation ≈0.82; final train–val gap `0.134 → 0.174` at the screening seed). Separating these requires a **new pre-registered diagnostic**, not a reinterpretation of DG-0005.
5. No ER performance claim is made from DG-0005. Its ordinary-split deltas are speaker-leaky context (F3) and their ER direction is consistent with memorization; any ER claim still requires LOSO.

**New direction:** The programme is **blocked on a human scope decision**, not on compute. The honest options are: (a) open a new `DG-xxxx` diagnostic to separate estimator variance from ER convergence/overfitting under the same composition; (b) authorize nothing further and proceed to framework synthesis using the existing F1–F10 evidence; or (c) revisit Option 3 explicitly, which now needs a mechanism rationale that does not depend on F9's task-intrinsic reading.

**Expected consequence:** No autonomous GPU work follows from DG-0005. Future relation-mechanism proposals must justify themselves from relation structure (F5, F6, F7) or from training-mixture effects, and must name the sampling regime when reporting gradient scale, exactly as relation claims must name pooling, layers and seed/fold axis.

---

## DEC-0011 — Framework synthesis is the active stage; the F9-era literature mandate is closed

**Status:** ACTIVE — 2026-09-22

**Previous direction:** DEC-0010 closed the mechanism question by withdrawing F9's task-intrinsic-scale rationale and left the programme "blocked on a human scope decision", with the MTRL diagnostic synthesis as the only analysis artifact. That synthesis still prescribed the DEC-0007 literature sequence and still listed the ER data-regime cause as unresolved — i.e. the written analysis record lagged `FINDINGS.md` after DG-0005.

**Evidence causing the change:** A synthesis pass over the current ledger found: (a) the per-diagnostic synthesis contradicted the updated findings on three points (it treated the ER gradient-scale cause as unresolved, directed the next action at the already-executed literature search, and listed the data-regime hypothesis as unevidenced); (b) no cross-study artifact existed for the programme's larger objective, the method-selection framework; (c) TR-0002/TR-0003/TR-0004 remained motivated by premises that F8, F9 and F10 have since removed. No new experimental evidence was produced — this is a record-consistency and synthesis decision.

**Decision:**

1. **Active stage is framework synthesis.** `research/FRAMEWORK.md` is created as the maintained cross-study framework: conditioned-quantity table, relational-versus-optimization classification, evidence levels, selection rules R1–R8, the current KS/SI/ER answers, the open decision points and the ranked missing evidence. `FINDINGS.md` remains authoritative for numbers.
2. **The DEC-0007 literature mandate is closed**, not merely unexecuted: it was executed by `LT-0001` and returned a negative result (FL-0003). `MTRL_DIAGNOSTIC_SYNTHESIS.md` is updated in place so no reader follows the superseded sequence; its historical text is retained per DEC-0006.
3. **Mechanism entries are annotated, not deleted:** TR-0002 (asymmetry) cannot be satisfied by DG-0001's rejected evidence; TR-0003 (confidence/reliability) loses its data-regime motivation and would need measured relation-estimate noise instead; TR-0004 (dynamic) is not motivated by DG-0002's near-orthogonal gradients. All stay BLOCKED.
4. **No new Study and no GPU work this stage.** One bounded diagnostic remains available if the human wants it (estimator variance versus ER convergence under DG-0005's composition); it is option (a) of the decision points and is not authorized by this decision.

**New direction:** Maintain the framework as the active artifact; consume zero compute until the human chooses (a) the bounded scale diagnostic, (b) close-out with the framework as the characterisation result, or (c) explicit Option-3 authorization with a rationale that does not rest on F9.

**Expected consequence:** A fresh agent restarting from `STATE.md` treats framework synthesis as the current deliverable, does not re-run the closed literature search, does not re-litigate the annotated mechanism gates, and can identify the single bounded diagnostic that would still change what the framework may claim.

---

## DEC-0012 — The remaining scale question is narrowed to the late-phase mean gradient

**Status:** ACTIVE — 2026-09-22

**Previous direction:** DEC-0010 listed option (a) as "a bounded diagnostic separating gradient-estimate variance from ER convergence/overfitting", with the two explanations treated as equally open and the per-batch gradient-norm instrumentation assumed adequate to separate them.

**Evidence causing the change:** A pre-registered post-hoc analysis of DG-0005's and DG-0002's existing artifacts (`studies/DG-0005/analyze_noise_shape.py`, committed at `2ac7f3d` before any statistic was computed) produced three results:

1. Under `σ ∝ 1/√n`, a noise-dominated norm has `E‖g‖ ∝ 1/√n`, so the arm-to-arm mean-norm ratio is bounded by `√(n_A1/n_A0) ≈ 3.0`. The observed late ratio was `3.99–4.69`, bounding estimator size at `0.758` `[0.715, 0.801]` of the late log-drop — at least ~20% (mean 24%) is not estimator size. No compute was used; the inputs already existed.
2. The same bound is ≈ `1.03` `[0.975, 1.082]` in the middle phase, so estimator scaling is sufficient there. The unexplained component is late-specific, which is itself new information about *when* the ER head state matters.
3. The instrumentation cannot measure gradient-estimate variance at all: anisotropic gradient noise aside, the isotropic ceiling on relative norm dispersion at `d = 1,550,800` is `≈ 0.00057`, while the observed within-phase dispersion is `CV ≈ 0.32`. Per-step dispersion is dominated by variation in the mean gradient, so no shape statistic may be used as a noise proxy.

**Decision:**

1. **Option (a) is narrowed, not closed:** what remains open is the late-phase mean-gradient component, not a generic "variance versus convergence" question.
2. **The discriminating measurement is `‖E g‖`** — the pool-mean gradient over the task's training data (or a large fixed subset), which is free of estimator noise — rather than per-batch `‖g‖` or any dispersion statistic. `FRAMEWORK.md` §7 item 1 now records that design.
3. **No new Study is created and no GPU work is authorized by this decision.** Re-instrumenting the existing arms is a bounded future option for the human, and it stays a diagnostic.
4. **No finding is promoted by this analysis.** It is post-hoc and model-dependent; it is recorded inside F10 as a constraint on an existing claim, explicitly not as a new finding.

**New direction:** The framework's missing-evidence list is re-ranked around a noise-free mean-gradient measurement; everything else in DEC-0010/DEC-0011 stands.

**Expected consequence:** A future diagnostic of this question will measure `‖E g‖` and will test the late phase specifically, instead of re-running a composition sweep or trying to infer variance from norm dispersion. The framework's claim about scale is correspondingly sharper: gradient *scale* is a training-mixture property (F10), and the part of it that persists at matched composition is a late-training property of the task's mean gradient.

---

## DEC-0013 — HUMAN DECISION: re-scope to a benchmark of published MTRL variants; diagnostic-first sequencing deferred

**Status:** ACTIVE — 2026-09-22 (human-authored decision)

**Previous direction:** DEC-0005 / DEC-0009 / DEC-0011 kept mechanism work behind two gates — a diagnostic (`DG-xxxx`) that names the failing MTRL assumption, *and* a published method (`LT-xxxx`) whose assumption addresses it. Under that regime the programme sat in framework synthesis with every `TR-xxxx` entry `BLOCKED`, because the only limitations diagnosed so far were non-relational (gradient scale, F9/F10).

**Evidence causing the change:** an explicit human decision, taken after a status review that established: (a) exactly one relation-learning method — classical MTRL — has ever been implemented and evaluated in this branch (35 runs; `tsm`/`pmr` have zero runs since DEC-0004; GBC is archived since DEC-0003); (b) the F9 literature gate's rejections were specific to the gradient-scale *rationale*, not to the variant families themselves; (c) the relation-method space is therefore unexplored rather than exhausted. The human judged variant exploration the higher priority and deferred the diagnostics.

**Decision (human):**

1. **Adopt a variant-benchmark design.** Several *published* Task Relation Learning variants, each implemented faithfully, evaluated under one shared matched protocol: screen, then multi-seed confirmation. The matched in-category control is classical MTRL; the reference is the matched wavCSE baseline. A variant that does not beat both has not contributed.
2. **Diagnostic-first sequencing is deferred, not deleted.** Mechanism Studies may now start from a verified published method without a prior diagnostic naming the failing assumption. Deferred diagnostics (Ω-saturation causality; parameter-summary adequacy; relation-confidence noise under matched composition) stay on the backlog.
3. **The published-method gate is unchanged.** No project-original mechanism and no invented architecture without a verified literature attribution. Faithfulness to the source paper must be checked before screening — the retracted GBC attribution (DEC-0003) is the precedent.
4. **First two families:** (i) asymmetric / directed relations; (ii) better relation estimator or task-parameter representation.
5. **Category boundaries hold.** Loss weighting, gradient surgery, low-rank, clustering and decomposition remain outside scope; the benchmark stays inside Task Relation Learning (DEC-0005 §3, FL-0003).
6. **The contribution claim changes** from a limitation-driven literature-derived extension to a comparative benchmark of published relation-learning variants. Recorded here so the write-up states it honestly.

**New direction:** Verify sources for the two families (`LT-0002`), then pre-register one `TR-xxxx` Study per variant with a single shared protocol so the arms stay comparable.

**Expected consequence:**

1. `STATE.md`'s stage becomes variant benchmarking; `FRAMEWORK.md` remains the synthesis artifact but its "no mechanism justified" statement is now the *reason for* the benchmark rather than a stopping point.
2. The F9-specific literature prohibition (DEC-0011) continues to bar re-running the *F9* search; it does not bar searching the new premise, which this decision explicitly changes.
3. Every variant arm must pin pooling, layers, epochs, batch composition, seeds, checkpoint policy and evaluation protocol, and must report per-task plus aggregate results — otherwise the pooling confound (F2, ~3pp on ER) swamps the mechanism.
4. Any ER performance claim still requires LOSO (F3); the benchmark's ER numbers from the ordinary split remain screening context only.

---

## DEC-0014 — HUMAN DECISION: activate Option 3 for the directed-relation question; the asymmetric arm is project-original, not a published variant

**Status:** ACTIVE — 2026-09-22 (human-authored decision)

**Previous direction:** DEC-0013 re-scoped the programme to benchmarking *published* relation-learning variants, with faithfulness to the source paper required. `LT-0002` then verified 15 papers across the two selected families and found that family A (asymmetric/directed) has **no clean pass**: AMTL (Lee et al. 2016), AutoTR (Zhou & Yang 2023) and GAMTL (Oliveira et al. 2019) each assume aligned per-task parameter columns, which our 12 / 1251 / 4-class heads do not provide, so a directed relation can only act on class-mean head summaries. DEC-0009 §4 had deferred exactly this situation — project-original work — behind explicit human authorization.

**Evidence causing the change:** `LT-0002`'s gate result (see `studies/LT-0002/analysis.md` §3–§4) plus the human's decision among the three presented options.

**Decision (human):** the directed-relation experiment proceeds as a **project-original mechanism** under the Option-3 path, not as a faithful implementation of any published method.

**Consequences:**

1. **Labelling is mandatory.** The arm is described as *project-original, motivated by the asymmetric-transfer literature* (Lee et al. 2016 in particular), and **must not** be presented as an implementation of AMTL, AutoTR or GAMTL. This is the constraint that archived GBC after its attribution was retracted (DEC-0003).
2. **No faithfulness claim, no external-validity claim.** Because the mechanism is ours, the question it answers is explicitly scoped: *does a directed, sparsity-controlled relation over class-mean head summaries help under this protocol?* It cannot support statements about AMTL as published.
3. **Lineage is still cited honestly.** The literature cards remain in the registry as motivation and as the source of design ingredients (loss-scaled sparsity, non-negativity, directed rows), each attributed to its paper.
4. **MSSL is unaffected and remains a published arm** — it passed cleanly (`LT-0002` family B) and its input is literally the summary matrix, so it keeps its faithful-implementation status.
5. **The benchmark therefore has two mechanism arms of different epistemic status**, and every results table must state which is literature-derived and which is project-original: `TR-0007` (MSSL, published) and `TR-0008` (directed relation, project-original).
6. The deferred `DG-xxxx` diagnostics remain deferred; this decision does not reopen them.
