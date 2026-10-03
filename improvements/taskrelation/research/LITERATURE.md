# Literature subsystem — V1 architecture and operator entry point

**Status: CURRENT NORMATIVE.** This is the one document that describes how the
literature subsystem works *now*. Read it before the increment history. The
per-increment design records (`LITERATURE_AGENT_ROADMAP.md`), the V1 evaluations
(`LITERATURE_AGENT_V1_EVAL.md`, `RESEARCH_COMPUTER_V1_EVAL.md`,
`RESEARCH_DESIGNER_LITERATURE_HANDOFF_EVAL.md`) and the
`weekly/` reports are **historical provenance**, not current specification.

Scope of V1: a deterministic, operator-owned literature corpus with a bounded,
read-only-plus-investigation-scoped Literature Agent. Canonical admission and
acquisition are deliberately *outside* the agent's authority.

---

## 1. Architecture map

One module owns each invariant. "Authority" is the single component that writes
the state; everything else reads it.

| Concept | Authoritative component | Durable state |
| --- | --- | --- |
| Paper identity / catalog | `literature_catalog.LiteratureCatalog` | `literature/catalog.jsonl`, cards `literature/<paper_id>.md` |
| CandidatePaper (external metadata) | `literature_discovery` (`core.StructuredDiscovery`, `model.CandidatePaper`) | none — returned data only |
| Canonical admission (CandidatePaper → Paper) | `literature_admit.LiteratureAdmission` | new catalog row + card; `literature/canonicalizations.jsonl` ledger |
| PrimaryArtifact (retained bytes) | `literature_primary.LiteraturePrimary` | `literature/primary_manifest.jsonl` + disposable local cache |
| User-supplied ingestion | `literature_ingest.LiteratureIngest` | artifact via `literature_primary`; `literature/acquisitions.jsonl` ledger |
| Public acquisition | `literature_acquire.PublicAcquirer` | artifact via `literature_ingest`; same acquisition ledger |
| Claim | `literature_claims.ClaimRegistry` | `literature/claims.jsonl` |
| PaperAssessment | `literature_assessment.AssessmentRegistry` | `literature/assessments.jsonl` |
| PaperCard (derived knowledge) | per-paper Markdown card, validated by `literature_catalog` | `literature/<paper_id>.md` |
| Synthesis | `literature_synthesis.SynthesisRegistry` | `literature_survey/registry.jsonl` + survey docs |
| Study / LT investigation lifecycle | `literature_investigation` | `STUDIES.jsonl` (rows with `type: literature`) + `studies/LT-*/` |
| Bounded scoped recording | `literature_record` | investigation `analysis.md`; assessments/claims/synthesis registries |
| Structured query | `literature_query.LiteratureQuery` | read-only over the above |
| Bounded text read | `literature_read.LiteratureReader` | read-only over cards / survey / Study artifacts / syntheses |
| Atomic registry write | `literature_io` | shared by `literature_investigation` and `literature_record` |
| Model-facing adapter | `.omp/tools/literature.ts` | — |
| Agent definition + methodology | `.omp/agents/literature-reviewer.md`, `.agents/skills/wavcse-literature-review/SKILL.md` | — |

### Transitions and their owners

```
discovery ─────────────► CandidatePaper          StructuredDiscovery (read-only, metadata)
CandidatePaper ────────► canonical Paper          literature_admit       (operator/deterministic)
Paper ─────────────────► PrimaryArtifact          literature_ingest / literature_acquire (operator)
PrimaryArtifact ───────► evidence read            literature_primary (read) / literature_read
evidence ──────────────► Claim                    literature_claims      (operator or agent-scoped)
Paper + LT ────────────► PaperAssessment          literature_assessment  (operator or agent-scoped)
Claims/Papers/LT ──────► Synthesis                literature_synthesis   (operator or agent-scoped)
Main OMP ──────────────► open/delegate LT         literature_investigation (operator)
Literature Agent ──────► scoped records           literature_record      (agent, one delegation only)
LT ────────────────────► complete/abandon         literature_investigation (operator)
```

`discovery ≠ admission ≠ acquisition`: discovery returns *candidate* metadata and
unvalidated artifact locations; admission turns a candidate into a canonical
Paper; acquisition retrieves and validates bytes for a **known** Paper. The
Literature Agent may *select* a candidate, but never performs any of the three
writes.

---

## 2. Authority model

**Main OMP / operator (deterministic modules, reached from the research session).**
Owns: catalog + card authoring via `literature_admit`; bytes via
`literature_ingest` / `literature_acquire`; the LT lifecycle via
`literature_investigation`; and all research state outside literature
(`FINDINGS.md`, `DECISIONS.md`, `FAILURES.md`, `BACKLOG.md`, `STUDIES.jsonl` in
general, `proposals/`, `authorizations/`).

**Literature Agent (`literature-reviewer`).** Read-only over the retained
corpus, plus exactly two write surfaces:

1. `literature_record` — `note` / `assessment` / `claim` / `synthesis`, accepted
   only under the **one currently delegated** `LT-*` investigation
   (`OUTSIDE_DELEGATED_SCOPE` otherwise; re-recording an identity updates in
   place). It cannot create or close a Study, admit a paper, acquire an artifact,
   edit a card or the catalog, or touch findings/decisions/failures/backlog/
   proposals/authorizations.
2. the disposable primary cache managed by `literature_primary`.

The agent's granted tools are pinned to the adapter surface by
`tests/test_literature_agent_assets.py`; runtime reachability is proved by
`scripts/agents/literature_agent_transcript.py` from a run's own transcript.

---

## 3. Durable state

| Path | Owner | Shape |
| --- | --- | --- |
| `literature/catalog.jsonl` | identity catalog | `catalog.schema.json` |
| `literature/<paper_id>.md` | per-paper card | validated against the catalog row |
| `literature/claims.jsonl` | claim records | embedded `EvidenceReference`s, source-bound |
| `literature/assessments.jsonl` | `(investigation_id, paper_id)` verdicts | — |
| `literature/primary_manifest.jsonl` | retained-artifact identity/provenance | `primary_manifest.schema.json` |
| `literature/acquisitions.jsonl` | append-only acquisition attempts | `acquisitions.schema.json` |
| `literature/canonicalizations.jsonl` | append-only admission attempts | `canonicalizations.schema.json` |
| `literature/INDEX.md` | derived human view (card links + generated assessment block) | link targets validated by `literature_catalog`; generated block by `literature_assessment check` |
| `literature_survey/registry.jsonl` | synthesis identity/status/provenance | docs `literature_survey/*.md` |
| `STUDIES.jsonl` | Study registry; `LT-*` rows carry the investigation lifecycle | — |
| `studies/LT-*/` | `PLAN.md`, `NOTE.md`, `analysis.md`, `result.json` | — |

Historical witnesses remain readable but are **not** an authority: an
`LT-*` `result.json`, an `analysis.md` narrative, and old per-card verdict
sections. `literature_assessment check` proves the canonical registry is the sole
active authority (no card verdict section, no `STUDIES.jsonl` `cards` membership,
no INDEX drift).

`literature/catalog.jsonl` has 23 papers; `assessments.jsonl` 23 records;
`claims.jsonl` 18 records; `literature_survey/registry.jsonl` 19 documents (as of
2026-10-04 — count with `literature_record validate`, never from memory).

---

## 4. Operator commands

All run as modules from the repository root, e.g.
`python -m improvements.taskrelation.research.<module> ...`.

| Module | Commands |
| --- | --- |
| `literature_catalog` | (default) validate + summary; `--lookup <identity>` |
| `literature_query` | `list`, `resolve`, `paper-studies`, `study-papers`, `study`, `assessment`, `identify`, `syntheses`, `synthesis` |
| `literature_discovery` | `discover` (`--doi/--arxiv/--title`), `search`, `references`, `citations`, `providers` |
| `literature_admit` | `admit`, `validate` |
| `literature_ingest` | `ingest` (local file + hints), `validate` |
| `literature_acquire` | `acquire`, `policy` (dry-run source policy), `validate` |
| `literature_primary` | `status`, `get`, `read`, `register`, `validate` |
| `literature_claims` | `validate`, `list`, `paper`, `get` |
| `literature_assessment` | `validate`, `check`, `index-assessments`, `list`, `paper`, `investigation`, `get` |
| `literature_synthesis` | `validate`, `check`, `list`, `get` |
| `literature_investigation` | `open`, `delegate`, `complete`, `abandon`, `get`, `list`, `validate` |
| `literature_record` | `note`, `assessment`, `claim`, `synthesis`, `validate` |
| `literature_read` | `card`, `study`, `primary`, `survey`, `synthesis` |

The full write-capable literature workflow (the roadmap from observed failure →
candidate → investigation) is `.agents/commands/wav-literature.md`.

### Verify

```bash
make check                  # agents-check + compute-check + research-check (no GPU/network)
make literature-tools-check # needs bun; exercises the model-facing adapter
```

---

## 5. Model-facing surface

Six tools, all defined in `.omp/tools/literature.ts` and granted to
`literature-reviewer`:

| Tool | Backing | Kind |
| --- | --- | --- |
| `literature_resolve` | `literature_query.identify_candidate` | read |
| `literature_query` | `literature_query` (+ `literature_claims` reads) | read |
| `literature_read` | `literature_read` | read |
| `literature_primary` | `literature_primary` | read + disposable cache |
| `literature_discover` | `literature_discovery` | read (metadata only) |
| `literature_record` | `literature_record` | scoped write |

`literature_discover` is an *optional* discovery capability, not part of the
required evidence set; a missing discovery grant never excuses answering from
memory. Canonical admission and acquisition are **not** model tools.

The read-only subset (`literature_resolve`, `literature_query`, `literature_read`,
`literature_primary`) is also granted to `research-designer`, which consumes a
completed investigation by reference: it is handed `literature_investigation_ids`
and reconstructs each one from `literature_query study` / `study-papers` /
`assessment` / `synthesis` / `paper_claims` and the bounded reads. It is never
granted `literature_record`, so the Literature Agent stays the scoped literature
writer.

Evidence levels and the claim/assessment/synthesis/decision distinction are
defined in `.agents/skills/wavcse-literature-review/SKILL.md`; this document does
not restate them. The consumer-side read contract (`study` metadata, completion
record, provenance classes) is in `.omp/agents/research-designer.md`.

---

## 6. Increment history (why the shape is this)

Each increment is recorded in `LITERATURE_AGENT_ROADMAP.md` and validated by a
focused test module. Summary of what produced V1:

| Increment | Result |
| --- | --- |
| INC-001 | `literature_catalog` — declared paper identity |
| INC-002 | `literature_assessment` — question-scoped assessments |
| INC-003 | `literature_query` — structured read interface |
| INC-004/B.1/C | `literature_primary` — retained primaries, roles, page reads |
| INC-005 | `literature_claims` — source-bound claims + `EvidenceReference` |
| INC-006/007/008 | synthesis registry, review skill, Literature Agent |
| INC-011 | `literature_ingest` — user-supplied ingestion + acquisition ledger |
| INC-012 | `literature_discovery` — metadata-only structured discovery |
| INC-013 | `literature_acquire` — deterministic public acquisition |
| INC-016 | capability-surface repair + drift tests |
| INC-017 | `literature_admit` — canonical admission from discovery |
| INC-018 | `literature_investigation` + `literature_record` — lifecycle + scoped persistence |
| INC-019 | completed investigation → Research Designer read contract; the `study` query operation made reachable and the LT metadata (`question`/`scope`/`outcome`/`completion`) exposed |

**Not yet built** (deliberately out of V1): proposal generation from the designer's
literature reading (INC-020); the INC-014 paper hunter; INC-015 desktop transport;
remote (S3) primary store behind `literature_primary`'s `fetcher` seam.
