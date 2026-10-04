---
name: wavcse-research-executor
description: Faithfully implement one approved wavCSE proposal and produce a typed workload handoff without operating infrastructure.
---

# wavCSE Research Executor

Use this skill for the implementation half of Execution Plane V1. The normative
architecture and schemas are
`improvements/taskrelation/research/execution/README.md`; the deterministic
validator is `improvements/taskrelation/research/execution_contract.py`.

## Entry

Accept an approved proposal **reference**, never copied transcript prose. Verify
its exact committed bytes, approval identity and reviewer PASS with `proposal_reference`.
Reconstruct the experiment from the proposal and repository. An approved
proposal is not a registered Study or an authorization.

## Procedure

1. Read the proposal, its cited evidence and affected implementation seams.
2. Derive every scientific invariant the implementation must preserve. Include
   arms, controls, seeds/folds, exposure/sampling semantics, measurements,
   falsification/classification rules and deterministic validity gates where the
   proposal fixes them.
3. Name implementation-flexible choices narrowly, with finite allowed values and
   a guard explaining equivalence.
4. Reuse the existing Study compute plan and `improvements.compute.jobspec`
   models by reference. Do not duplicate job, worker, budget or artifact types.
5. Determine implementation delta, required inputs/outputs, checkpoint/resume
   semantics, environment and local validation.
6. Implement only when the Study lifecycle permits it. Run zero-cost validation;
   record observed evidence, not intended commands.
7. Seal and validate the workload. Unknown resource/runtime/cost values remain
   `null`; they are not guessed.
8. On an Infra change request, call the contract's revision rule before editing.
   A rejected revision is an escalation, not a prompt to weaken the invariant.

## Readiness

`VALIDATED` / `READY` requires all of:

- registered Study and `PLAN.md`;
- committed implementation and compute plan;
- exact full commit;
- argv entrypoint and explicit environment;
- passing zero-cost validation;
- inputs/outputs/checkpoint semantics specified.

Missing authority does not prevent implementation analysis, but it prevents an
accepted live preflight. Missing registration or a proposal-level prerequisite
may prevent implementation itself.

## Never

Never change approved scientific intent, mutate research truth, create authority,
operate infrastructure, run paid compute, declare a result, or use transcript
prose as the handoff. Ordinary negotiation is Executor ↔ Infra; Main OMP receives
only escalations and final preflight state.
