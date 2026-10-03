---
description: Turn an observed failure into verified literature candidates and reserved backlog entries
---

# Targeted literature research

Act as the literature and methods researcher for the wavCSE Task Relation
Learning programme.

The objective is not to collect papers. Start from a failure we have actually
observed in our own experiments, find published mechanisms whose stated
assumptions address that failure, verify which category those methods really
belong to, and stop with justified candidates.

This command ends before implementation.

Skills: wavcse-experiment-operator
Boundaries: mutates-research-state, no-paid-compute

## Must not

- Never modify model, trainer, mechanism or configuration code.
- Never launch GPU training or any paid experiment.
- Never search before the specific empirical problem is stated and sourced.
- Never present a low-rank, clustering, decomposition, loss-weighting,
  gradient-surgery or architecture-search method as Task Relation Learning;
  record category ambiguity explicitly.
- Never record an unimplemented method as a result or as a preferred direction;
  a candidate is a question, not a chosen architecture.
- Never reproduce a paper's reported numbers as our evidence; attribute them.

## Steps

1. Load existing evidence first: `improvements/taskrelation/research/STATE.md`,
   `improvements/taskrelation/research/FINDINGS.md`,
   `improvements/taskrelation/research/FAILURES.md`,
   `improvements/taskrelation/research/DECISIONS.md`,
   `improvements/taskrelation/research/BACKLOG.md`,
   `improvements/taskrelation/research/STUDIES.jsonl`,
   `improvements/taskrelation/research/literature/catalog.jsonl` and
   `improvements/taskrelation/research/literature/INDEX.md`.
   Resolve known paper metadata and `LT-*` Study relationships first with
   `python -m improvements.taskrelation.research.literature_query`; open only
   the returned cards and Study artifacts needed for the question.
2. Name the observed failure and the evidence that establishes it; do not
   proceed from a remembered or assumed limitation.
3. State the literature question for that failure: the observed problem, the
   current method's assumption, why that assumption may be inadequate, and the
   question to search.
4. Search targeted concepts derived from the failure, covering foundational
   methods, their extensions, recent work that changes the assumptions, and
   speech-domain applications where relevant.
5. Verify each candidate's method category against the programme's taxonomy
   before treating it as relevant, and mark diagnostics, controls and
   conceptual inspiration as such. Deduplicate each candidate first with
   `python -m improvements.taskrelation.research.literature_query identify`
   (`known` / `new` / `ambiguous`); never re-retain a source the catalog already
   holds, and never resolve an `ambiguous` verdict by guessing.
6. Before retaining a paper, resolve its identity through
   `improvements/taskrelation/research/literature/catalog.jsonl`. The existing
   card slug is the immutable `paper_id`; never mint a second identity for the
   same source. Add a verified card under
   `improvements/taskrelation/research/literature/` and record the source's
   identity and checks in that card. There is no `literature/papers/` directory
   — the cards remain the retained record of verified sources. Keep screening
   decisions in the `LT-*` Study, never in the identity catalog.
7. Compare candidates by assumption rather than by reported accuracy, and update
   the comparison table in
   `improvements/taskrelation/research/literature/INDEX.md`.
8. For each justified candidate add a backlog entry under
   `improvements/taskrelation/research/BACKLOG.md` with the next free reserved
   identifier and the gate that keeps it blocked, and register the literature
   Study in `improvements/taskrelation/research/STUDIES.jsonl`.
9. Stop. Implementation, screening and training are separate work.

## Durable state

- Negative evidence that motivates the search:
  `improvements/taskrelation/research/FAILURES.md`.
- Literature identity: `improvements/taskrelation/research/literature/catalog.jsonl`;
  read-only metadata and Study relationships:
  `improvements/taskrelation/research/literature_query.py`; retained-artifact
  storage metadata and retrieval:
  `improvements/taskrelation/research/literature/primary_manifest.jsonl` and
  `improvements/taskrelation/research/literature_primary.py`; cards:
  `improvements/taskrelation/research/literature/`; comparison table:
  `improvements/taskrelation/research/literature/INDEX.md`.
- Candidate studies and reserved identifiers:
  `improvements/taskrelation/research/BACKLOG.md`; study registry:
  `improvements/taskrelation/research/STUDIES.jsonl`.
- Binding scope: `improvements/taskrelation/research/DECISIONS.md`.

## Two modes

This command is the write-capable workflow: it may retain a verified card, update
the comparison table and reserve a backlog identifier.

For a question that only needs *existing* retained evidence answered, delegate it
to the read-only `literature-reviewer` subagent instead (task tool,
`agent: "literature-reviewer"`), which carries
`.agents/skills/wavcse-literature-review/SKILL.md` and can read cards, `LT-*`
Study artifacts and any locally available primary artifact. It returns a
provenance-carrying investigation result and cannot mutate research state.

Defer study semantics, gates and evaluation requirements to
`wavcse-experiment-operator`.
