# Research Designer ← completed literature investigation — evaluation (INC-019)

> **Historical evaluation record — not the current specification.** For how the
> subsystem works now, read [`LITERATURE.md`](LITERATURE.md).

Purpose: prove the one missing connection in the Research Computer V1 loop —
that a **completed** `LT-*` investigation is a bounded, reconstructible input to
the read-only `research-designer` agent, from durable state alone, with no
Literature Agent transcript and no copied prose.

This evaluation produced **no proposal**: the designer's answer was ephemeral and
nothing was persisted from it.

## What was built (for reference)

- `literature_query study <LT-id>` — one investigation's registered metadata
  (`question`, `scope`, `status`, `decision`, timestamps, artifact pointers), its
  lifecycle completion record when it has one, and registry-derived
  `assessment_count` / `papers_assessed` / `synthesis_ids`. It is distinct from
  `study-papers`, which additionally returns the paper relationships.
- The model adapter (`.omp/tools/literature.ts`) previously mapped both
  `operation=study` and `operation=study_papers` to the `study-papers` argv, so
  the metadata-only read was unreachable; each now resolves to its own command.
- `research-designer` keeps the read-only literature subset (`literature_query`,
  `literature_read`, `literature_primary`, `literature_resolve`) and is never
  granted `literature_record`.

## Harness

A fresh `research-designer` subagent (agent model role from its own definition),
in a session that never ran a Literature Agent and never saw one. The only
literature input was:

```
literature_investigation_ids: ["LT-0002"]
design_question: "…does the retained literature support replacing the
                  closed-form covariance relation estimator, and what would
                  still have to be measured before a study could be proposed?"
constraints: ["read-only", "ephemeral analysis only"]
```

No transcript, no copied conclusion, no summarised paper list, no remembered
identifiers, and no repository `read`/`grep` of project records were supplied.
The designer had `read`/`grep`/`glob`/`find` granted and used none of them: the
whole reconstruction came through the literature tools.

Run 2026-10-04. Session transcript:
`~/.omp/agent/sessions/-projects-fyp-wavCSE-wavCSE/2026-10-03T22-08-50-609Z_01a103d0-13b1-76bf-bfcf-417523399fcb/DesignerLTSlice.jsonl`.

## Observed call sequence (from the session transcript, not self-report)

| # | Call | Purpose |
| --- | --- | --- |
| 1 | `literature_query operation=study LT-0002` | lifecycle metadata, pointers, counts |
| 2 | `literature_query operation=study_papers LT-0002` | the 15 assessed papers |
| 3–5 | `literature_read source=study` plan / analysis / note | the question, gates, result |
| 6 | `literature_query operation=assessment LT-0002 goncalves-2016-mssl` | one `(investigation, paper)` verdict |
| 7 | `literature_query operation=paper_claims goncalves-2016-mssl` | recorded claims |
| 8–10 | `literature_query operation=claim` ×3 | the primary-verified objective, the graphical-lasso step, the λ-selection claim |
| 11 | `literature_primary operation=status` | retained versions |
| 12 | `literature_primary operation=read role=published page=8 pageEnd=9` | primary evidence |
| 13 | `literature_read source=card goncalves-2016-mssl` | derived card |
| 14 | `literature_query operation=synthesis_list status=active` | synthesis registry metadata only |

No `mcp__deja_deja` call and no generic `read` call. The designer declined to
open any synthesis prose, and said why: those syntheses are broader later work
and must not be folded back into `LT-0002`'s conclusion.

## Independent verification of the answer

Checked mechanically against durable state, not by reading the answer:

- the three cited `claim_ref`s exist in `literature/claims.jsonl` for
  `goncalves-2016-mssl`; none invented. Their levels as reported match the
  registry: `barrier-placement-and-1-over-d-absorbable` is `source_level:
  primary`, `lambda-penalties-selected-on-data` is `source_level: card`;
- `LT-0002#goncalves-2016-mssl` is `verdict: pass` with 6 gates;
- `literature_primary status goncalves-2016-mssl` retains a `published` artifact
  (`sha256 5dcca4cf…3ce2b`) and the same page range reads back as
  `locator primary:pages:8-9` with that digest;
- the "15 papers, 9 directed + 6 estimator" split matches `LT-0002/analysis.md`
  §1; the 4 `synthesis_ids` match the study view;
- the reported question wording is verbatim from the `LT-0002/plan` artifact;
  the designer said so, and correctly noted the registry row's `question` is
  `null` for this pre-INC-018 investigation.

**Scientific misinterpretations found: none.** It kept the classes apart in
practice: it labelled the project's near-±1/3 Ω-saturation premise as appearing
in the *Study's* analysis rather than elevating it to a Finding it had read, it
treated the PaperAssessment as the investigation's judgement rather than a
global paper status, and it treated synthesis metadata as registry state, not
evidence.

## Gaps observed (evidence only, not built)

1. **A pre-INC-018 investigation cannot be read from metadata alone.** `LT-0002`'s
   row carries no `question`/`scope` and no completion record; the question lives
   only in the Study's `PLAN.md`, and that file's header still says
   `Status: RUNNING` while the registry says `complete`. The designer handled
   both explicitly, but this is a real legacy-record limitation, not something
   the read layer can repair without rewriting historical records.
2. **`literature_query` domain failures are a plain stderr line**, not a
   structured `kind`, so the adapter returns `ok:false` with the message but no
   branchable code for a missing/non-literature investigation.
3. **A one-shot subagent's transcript is session-scoped.** The call sequence above
   is read from the persisted session file; the INC-019 build was verified with
   session persistence on (`--no-session` was not used).

## What this does not establish

Nothing about whether p-MSSL, or any estimator replacement, improves KS/SI/ER or
aggregate performance, and nothing about the Family-A admissibility decision.
This is a *read and reconstruction* slice: it proves the agent boundary, not a
research result. Proposal generation and review remain the next increment and are
deliberately not implemented here.
