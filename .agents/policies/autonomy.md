# Autonomy policy

What an autonomous cycle may decide, what needs the researcher, and what must
stop. This file is the classification; the cycle's steps live in
`.agents/commands/wav-cycle.md`, the backend's contract in the compute backend's
own README, and the hard invariants in `AGENTS.md`. Do not restate them here.

A cycle consumes this policy; it never edits, widens, or reinterprets it. A
change to this file is a human decision.

---

## AUTONOMOUS — proceed without asking

- Reconcile research records and runtime state; report a disagreement and, when
  one side is provably stale, resolve the record.
- Mechanically repair superseded or dangling records and references: retired
  paths, citations of decisions that have been superseded, a status that
  contradicts the registry. Mechanical means a fact an accepted decision or the
  registry already settles — never a new scientific conclusion.
- Register a study whose plan the accepted protocol already determines, and
  write its `PLAN.md`, configs and compute plan.
- Repair an implementation defect whose intended behaviour is unambiguous, with
  a regression test; commit it; re-run the same science.
- Run tests, the import smoke, and the repository checks.
- Commit the exact experiment implementation and configuration before a recorded
  run.
- Establish artifact readiness from verified canonical bytes (digest identity,
  not filename or size).
- Choose, provision, reprovision, bootstrap and benchmark compute **inside an
  authorization envelope**, and pass the envelope's hourly ceiling to the
  control plane's own price guard.
- Bounded retry of a transient infrastructure failure, with every attempt
  counted in durable state; reconcile an ambiguous outcome before any retry.
- Submit exact-commit jobs, monitor them, reconcile interrupted ones, collect
  and independently verify their outputs, and cancel a job that is no longer
  wanted.
- Run the protocol's pre-registered screening seeds, confirmation seed set, and
  the LOSO folds the protocol requires for the claim.
- Record negative evidence; update research state and MLflow.
- Stop or destroy this scope's compute when it is no longer needed, and adopt
  an unowned scope worker only as far as stopping it.
- Derive the next action, and continue the cycle when it is autonomous.

## HUMAN_DECISION — state the option set and stop

- Change a hypothesis, or a falsification rule, once evidence exists.
- Change the protocol's fixed conditions: pooling, layers, epoch budget, batch
  size, optimizer exposure, splits, checkpoint policy, seed treatment, task set,
  dataset membership.
- Any other change to scientific semantics, including preprocessing, label
  mapping or precision.
- Introduce a project-original mechanism, or decide a faithfulness/deviation
  question.
- Override an eligibility gate.
- Make, or adjust, a claim outside the pre-registered protocol.
- Choose between scientifically different fixes for the same failure.
- Grant, renew, widen, or lift an authorization envelope; accept a cost beyond
  it.
- Reopen a question a decision closed.

## HARD_STOP — report and do not proceed

- Artifact identity cannot be established from any independent source.
- Two authoritative records imply genuinely different scientific semantics.
- Credential, host-key or secret-handling problem.
- A required input is inaccessible or licensed in a way the protocol does not
  settle.
- The intended scientific semantics are ambiguous.
- The authorization is exhausted or expired with no valid fallback.
- An action would threaten a sole copy of anything.

A HARD_STOP is reported with the blocking fact and what was tried. It never
degrades silently into a cheaper or narrower action.

---

## Rules that hold across every class

- **Out-of-memory is never a scientific change.** A resource failure may move the
  same job spec to a compatible, still-authorized resource. Batch size,
  precision, model, checkpoint policy, pooling, layer set, dataset membership,
  splits, preprocessing and label mapping never change to fit hardware.
- **Scientific failure is evidence, not a defect.** A run that completed and
  wrote its outputs is never retried and never "repaired"; it is analysed and
  recorded, in `FINDINGS.md` / `FAILURES.md` as the evidence requires.
- **Implementation failure is never success.** A non-zero execution is reported
  as a failure; nothing interprets a crashed run as a result.
- **Retries are bounded and counted durably.** A restart does not reset an
  attempt budget any more than it resets spend.
- **Nothing is submitted twice.** An ambiguous create or submit is reconciled
  against provider state by its deterministic identity before anything else
  happens.
- **Spend is bounded by an envelope, and the envelope is the human's.** Unknown
  price or unknown lifetime fails closed for new spend.
- **A paid resource always has an owner and a cleanup policy.** An unowned
  worker is reported, never destroyed.
- **Unknown price, unknown cost, or an unbounded ledger is a stop, not an
  estimate.**
- **Busywork is not a reason to continue.** If the only remaining action needs
  the researcher, stop and say so.

## Where the classes are applied

| Concern | Artifact |
|---|---|
| Cycle steps and loop bounds | `.agents/commands/wav-cycle.md` |
| Study design, evidence tiers, promotion | `.agents/skills/wavcse-experiment-operator/SKILL.md` |
| Backend verbs, envelopes, ledger, jobs | the compute backend's README under `improvements/compute/` |
| Spend authority for one scope | `improvements/taskrelation/research/authorizations/` |
| Infrastructure mechanics | the infrastructure checkout's own skills, reached through its CLI |
| Hard invariants | `AGENTS.md` |
