---
name: wavcse-research-runner
description: Use for one-shot autonomous wavCSE research cycles; not for narrow specialist tasks.
---

# wavCSE Research Runner

Orchestration for whole-system requests: run the next research cycle; make the
embeddings ready; run the next experiment; get the system experiment-ready; keep
the pipeline moving.

If the request is one narrow task, load only the specialist skill that owns it.
This skill exists to sequence specialists, not to replace them.

## Stance: reconcile, never assume

Start every cycle by reconciling actual state. Do not trust conversation
memory, an earlier session, or a previous cycle's plan.

Reconcile from:

- repository records: current research state, findings, decisions, failure log,
  backlog, the study registry and study folders, architecture READMEs;
- runtime tooling and the `infra` CLI: infrastructure health, workers, jobs,
  storage, artifact readiness and any compute still running;
- the current git revision and working-tree state.

Compare the persisted research record against what the runtime tools report, and
treat a disagreement as unresolved until you know which is stale. Only then
decide what is missing.

## Order of operations

1. Reconcile current state from repository records and runtime tooling.
2. Derive the prerequisites actually missing for the requested goal.
3. Do all controller-side work first: design, configs, data layout, packaging.
4. Provision paid compute only once the prerequisites are verified and the plan
   is authorized.
5. Execute the work.
6. Validate and publish artifacts.
7. Analyse the research evidence.
8. Update persistent research state.
9. Stop paid compute that is no longer needed.

Do not start a paid step while a cheap controller-side prerequisite for the same
goal is still missing. Do not treat step 5 as the end of the cycle.

## Companion skills

This repository (wavCSE):

- `wavcse-experiment-operator` — study design, execution, analysis, decisions.
- `wavcse-embedding-generation` — embedding readiness and extraction semantics.

The wavcse-infra checkout:

- `wavcse-infra-operator` — controller, workers, jobs, storage, lifecycle.
- `gpu-research-operator` — paid GPU provisioning and cost discipline.
- `wavcse-artifact-pipeline` — artifact validation, publication and caching.

Load the relevant companion before consequential work in its domain; follow it
rather than paraphrasing it from here.

Infrastructure actions happen through the `infra` CLI in the wavcse-infra
checkout. Never reimplement provisioning, transfer, caching or publication in
ad-hoc shell, even when that looks faster to type.

## Guardrails

Never violate these in order to save time or money:

- never invent a hypothesis, and never promote a single-seed result;
- never change scientific semantics — model, checkpoint, pooling, layer set,
  dataset membership, splits, preprocessing, label mapping, precision — to make
  work faster or cheaper;
- never publish an unverified artifact as canonical;
- never destroy anything that holds the only copy;
- never leave paid compute running after useful work ends;
- report the exact commit executed.

Also hold the repository's own invariants: one study tests one primary
hypothesis, comparisons are protocol-matched, and no claim outruns its evidence
tier.

## Escalate instead of guessing

Pause and report when a real decision is needed: cost above current policy, a
destructive operation on canonical data, an unresolved scientific criterion, a
proposed semantic or scope change, or a conflict between records. Do not pause
for choices you can derive from repository state or these companions.

## Cycle is not complete until

The goal's work is executed, artifacts are published or explicitly staged as
unverified, evidence is analysed, persistent state matches reality, and paid
compute for finished work is stopped. A launched job is not a completed cycle.
