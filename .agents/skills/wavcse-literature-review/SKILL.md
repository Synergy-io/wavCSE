---
name: wavcse-literature-review
description: Conduct one bounded literature investigation for the wavCSE Task Relation Learning programme — resolve identity, read the cheapest sufficient evidence, and return a provenance-carrying synthesis.
---

# wavCSE Literature Review

Methodology for one bounded literature investigation. Load it when a research
question needs external published evidence, an existing claim needs checking
against the retained sources, or a mechanism's literature standing must be
established.

This skill is read-only with respect to canonical research state. It pairs with
the literature tools and, for the write-capable workflow that retains a new
paper, with `.agents/commands/wav-literature.md`.

## Stance

Reconcile from records, never from memory. The approved literature interfaces and
the artifacts they return are the **only** admissible source of a literature
assertion. Everything else is inadmissible, whatever it sounds like:

- model pretraining or "what the paper is known to say";
- a previous session's answer, including your own;
- recalled conversation, agent, or transcript text from a memory tool;
- the main session's context, prompt, or paraphrase;
- an identifier, locator, number, or quote reconstructed from any of the above.

Recalled historical text is never evidence and never provenance: not weaker
evidence, but not evidence at all. If the approved interfaces cannot support an assertion, the
assertion is not established — it is not "probably right".

Prefer the cheapest sufficient evidence. The literature tree is large; a
question is usually answered by a handful of cards, and only sometimes by a
primary PDF.

## Evidence-surface availability — fail closed

Establish that your evidence interfaces are callable **before** answering. The
required capability set is explicit:

```
literature_resolve
literature_query      (including operation=paper_claims and operation=claim)
literature_read
literature_primary
```

`yield` is the harness primitive for returning the result and is not evidence.

If a required capability is unavailable — not callable, missing from your tool
set, erroring, or returning `STORAGE_NOT_CONFIGURED`-class states you cannot work
around — do **not** answer the evidence question. Return this instead and stop:

```
evidence_status: unavailable
reason: REQUIRED_LITERATURE_TOOL_UNAVAILABLE
missing_capability: literature_query | literature_read | literature_primary | literature_resolve
attempted:            # what you tried, in order, with the structured reason each returned
synthesis: not established from approved evidence
```

Report it as a limitation of the run, not of the research question. A partial
answer assembled from memory is worse than this failure, because it is
indistinguishable from evidence.

Two things this does not license: it does not license answering from recall when
*one* capability is missing but another could cover the same need (use the
capability you have); and it does not license a "best effort" synthesis with a
caveat. An unsupported synthesis is a fabrication with a disclaimer.

## The investigation loop

1. **State the evidence need.** Convert the research question into the specific
   thing literature could settle: which mechanism, assumption, or empirical
   claim is in question, and what would count as relevant evidence. If the
   question is a research decision rather than an evidence need, stop — that
   belongs to the main research session.

2. **Confirm the evidence surface.** Check the required capabilities are
   callable; if not, return the fail-closed result above rather than proceeding.

3. **Query known literature cheaply.** Enumerate and filter retained papers
   before opening anything. Resolve any candidate identity you were given; never
   assume a paper is or is not retained.

4. **Resolve a bounded candidate set.** Work from returned IDs and titles to the
   papers the question actually turns on. Do not widen to the whole corpus to be
   safe.

5. **Query recorded claims for the candidate set, before opening prose.** This
   step applies to every question shape, including comparative ones over many
   methods — the breadth of the question changes how many candidates you query,
   never whether you query. For each candidate in the set:

   ```
   literature_query operation=paper_claims paperId=<id>
   literature_query operation=claim     paperId=<id> claimId=<claim-id>
   ```

   A recorded claim is repository-reviewed and source-bound: it names the
   artifact class and the section it came from, and a quoted claim carries text
   verified verbatim against that section. Use its assertion and its locator, and
   report it at the claim's own `source_level`.

   **Every `claim_ref` you report must come from a tool call in this run.** A
   claim identifier that was remembered, inferred, extrapolated from a similar
   one, or reconstructed from prose is a fabrication, even when it looks
   plausible and even when the claim it describes really exists.

6. **Open cards and surveys only where claim coverage is insufficient.** Cards,
   `literature_survey/` documents and `LT-*` Study artifacts are the fallback for
   what claims do not yet record — not the default path. Name which artifact you
   opened and at which section. A Study's verdict describes that Study's own
   question; it is never a general judgement about the paper.

7. **Use the primary artifact when the question demands it.** A card is a
   derived summary; when the answer turns on an exact equation, a reported
   number, or a stated assumption, the primary source is the authority. Resolve
   retention and identity first (`literature_primary operation=status`, which
   lists every retained version), then read a bounded view
   (`literature_primary operation=read`, with `role` when the paper retains more
   than one artifact, plus `page=N`, or `page`/`pageEnd` for an inclusive range).
   The read names the artifact's `role`, `sha256`, its `source_url` and a
   `locator` (`primary:page:N` or `primary:pages:A-B`). `page` is a **1-based
   physical PDF page index**, not a printed page label, so cite the locator,
   never an inferred printed page number. The read output is a derived *view*:
   the PDF bytes are the evidence, and the read reports its extractor and any
   extraction warnings — equations that extract imperfectly must be reported as
   uncertain and never normalised into an equation the extraction did not show.
   Treat `role` + `source_url` as part of the identity: a preprint and the
   published version are different artifacts with different digests, a claim
   recorded against one is not verified by the other, and the tool never chooses
   a version for you. When the assertion is about a specific version, read that
   version; when the versions make different propositions, report both and say
   which artifact each statement came from rather than merging them. If the
   primary artifact is not available locally (`PRIMARY_NOT_AVAILABLE`,
   `STORAGE_NOT_CONFIGURED`), say so and downgrade the claim's evidence level
   rather than implying you verified it.

8. **Separate four things that must never merge.** For every substantive
   statement, know which of these it is:
   - **paper claim** — what the authors state;
   - **reported evidence** — what the paper actually shows;
   - **your interpretation** — what you infer from that;
   - **research implication** — how it might relate to our work.

9. **Compare across papers by assumption, not by reported performance.** Two
   methods with the same headline result but incompatible assumptions are not
   the same evidence. Record where papers disagree, and why.

10. **Return a bounded synthesis with provenance and uncertainty.** Name the
    papers, the evidence level behind each conclusion, what was not inspected,
    and what would change the answer.

## Evidence levels

Every conclusion carries one, and it travels with the claim:

- **primary** — read from the retained primary artifact this run; carry its
  `sha256`, `source_url` and page locator, and name the version it is.
- **card-derived** — read from the retained per-paper card. The card was itself
  written from a verified primary source, but *you* did not re-verify it now.
- **survey-derived** — read from a derived literature/theory synthesis document
  under `literature_survey/`. These documents mark their own sentences
  `OBSERVED` / `INFERRED` / `HYPOTHESIZED`; carry that marker, because an
  inference in a survey is weaker evidence than a statement in a card.
- **Study-derived** — read from an `LT-*` Study's own record or interpretation.

There is no `recalled`, `remembered`, `session-`, or `inferred-from-context`
level, and you may not invent one. Never present card-derived or survey-derived
knowledge as though you had inspected the PDF. When primary evidence was
unavailable, say so explicitly — that is a finding, not a failure.

## Provenance rules

- Keep `paper_id` on every paper-specific conclusion.
- Include an exact location (section, equation, table, figure, page) only when
  you can read it in evidence you actually have: either the artifact you opened
  this run, or a recorded claim whose locator the deterministic layer validated.
  A recorded claim's locator may be reported at that claim's evidence level and
  never upgraded — citing a claim recorded from a survey does not make it
  card-derived.
- Never invent a page or section number, and never restate a card's citation as
  if you had opened the paper. If the card does not state a number, the card is
  not a source for that number.
- Never cite a `claim_ref` you did not receive from `operation=paper_claims` or
  `operation=claim` in this run.
- Attribute a paper's reported numbers to the paper. They are not our results.

## Authority — what this skill does not do

Report evidence; do not decide what to do with it.

- No research roadmap, experiment, screening or compute authorization.
- No edits to `FINDINGS.md`, `FAILURES.md`, `DECISIONS.md`, `BACKLOG.md`,
  `STUDIES.jsonl`, any card, the catalog, the primary manifest, or any Study.
- No new papers, no PDF uploads. Querying recorded claims is expected and
  read-only; creating a claim record is not yours to do.
- No external discovery: the retained corpus is the input.

Useful output includes "this evidence suggests testing X would distinguish these
mechanisms". It never includes "TR-0014 is authorized" or a chosen architecture.

## Result contract

Return one structured investigation result:

```
evidence_status         # "available", or "unavailable" with the fail-closed block below
missing_capability      # only when evidence_status is unavailable
question
synthesis
evidence[]:
    paper_id
    supports            # what this source establishes for the question
    claim_ref           # recorded claim reference (paper_id#claim_id), when one answers it
    location            # exact locator, only if read from available evidence
    evidence_level      # primary | card-derived | survey-derived | Study-derived
    confidence          # with the limitation that bounds it
agreements[]            # where sources converge
disagreements[]         # where they conflict, and on which assumption
uncertainties[]         # what the evidence cannot settle
missing_primary_evidence[]   # claims that need a PDF we do not have locally
implications_for_current_research[]   # hypotheses to consider, not decisions
suggested_followups[]
```

When `evidence_status: unavailable`, `synthesis` says only that the question is
not established from approved evidence. Keep the result compact: IDs and short
locators, not pasted excerpts or whole cards.

## Escalate instead of guessing

Stop and report when: two authoritative records disagree; a required artifact is
unavailable and the answer depends on it; the question is really a research
decision; an answer would need a paper we do not retain; or the evidence
interfaces themselves are unavailable. State what you tried and which structured
reason you received.

Defer study design, evidence tiers and promotion rules to
`wavcse-experiment-operator`. Defer identity, catalog and manifest semantics to
the literature modules.
