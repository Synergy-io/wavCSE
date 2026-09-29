---
description: Ensure the embedding sets the current research programme needs are verified READY
---

# Embedding readiness

Make the embedding inputs the current research programme consumes verified and
usable, generating only what is genuinely missing.

This command reconciles and completes embedding data. It does not change
research scope, and it never regenerates an already-canonical compatible set.

Skills: wavcse-embedding-generation
Boundaries: may-provision-compute, mutates-research-state

## Must not

- Never regenerate a set that already exists canonically and matches the
  required scientific configuration; reuse verified compatible artifacts.
- Never change the scientific configuration for operational convenience:
  dataset membership, split membership, labels, filtering, checkpoint or its
  bytes, model configuration, layer selection, preprocessing, sample rate,
  normalization, pooling, precision, or output tensor contract.
- Never mark a set READY at extraction-process exit; readiness follows
  validation and independent verification.
- Never publish an unverified artifact as canonical, never delete inconvenient
  samples to make a set pass, and never silently reduce workload after OOM.
- Never write worker identifiers, hosts, ports, prices, job identifiers,
  credentials or URLs into repository files.

## Steps

1. Fix the requirement from repository records rather than from a dataset list:
   read `improvements/taskrelation/research/STATE.md`,
   `improvements/taskrelation/research/OBJECTIVE.md` and the relevant folders
   under `improvements/taskrelation/research/studies/` to establish which
   datasets, tasks and splits the current configuration consumes.
2. Reconcile against the canonical inventory: for each required dataset record
   whether a canonical artifact exists, whether it matches the required
   scientific configuration, and whether a warm representation is usable.
3. Derive exactly what is missing; only sets with no compatible canonical
   artifact are candidates for generation.
4. Prove source readiness against the current loader contract before any
   compute spend, so a bad corpus is found before extraction rather than after.
5. Generate the missing sets from an exact repository commit with checkpoint
   identity verified, following `skill://wavcse-embedding-generation`.
6. Validate the complete generated set against an independent membership
   oracle, then package deterministically and publish canonically.
7. Independently verify the published bytes and tensor invariants; only then
   treat the artifact as canonical.
8. Leave a verified warm working set shaped for the next research consumer
   rather than for packaging convenience.
9. Report the readiness inventory for every required dataset: source verified,
   scientific configuration, count, tensor contract, canonical verification,
   warm representation, READY.

## Durable state

- Requirement and readiness context:
  `improvements/taskrelation/research/STATE.md` and
  `improvements/taskrelation/research/OBJECTIVE.md`.
- Canonical artifact membership, shard and byte identity: held by the artifact
  publication and run-tracking system the skill designates; never copied into
  repository files as runtime values.
- Policy is durable; per-run identifiers, paths and hashes are runtime values
  and belong only to the run record.

Defer extraction semantics, dataset-specific pitfalls, validation fields and
readiness criteria to `skill://wavcse-embedding-generation`; this command only
orders the work.
