# Literature Agent Architecture Reconnaissance and Incremental Roadmap

**Status:** planning only. This document introduces no agent, tool, schema, migration, storage operation, or change to an existing research record.

**Reconnaissance basis:** repository state, agent instructions and workflows, representative research records, the two `LT-*` Studies, literature cards, the integrated literature survey, task-relation outputs, tests, and relevant Git history through 2026-09-30.

# 1. Current Architecture

The repository is already a research information system. It is document-led rather than registry-led: authority is distributed by information kind, and agents reconstruct a current answer by reading several Markdown registries plus selected Study folders.

```text
                                  human scope decisions
                                           |
                                           v
external papers/URLs ---> literature cards ---> LT Study PLAN/NOTE/analysis/result
        |                       |                         |
        |                       +---- literature INDEX --+
        |                                                 |
        +----> literature_survey/                         v
                 theory maps, predictions,       STUDIES.jsonl + BACKLOG.md
                 syntheses, open questions                |
                                                          v
MLflow/DagsHub runs ---> Study result JSON ---> FINDINGS.md / FAILURES.md
  (execution record)       + analysis.md            |          |
             |                    |                  +----+-----+
             +--------------------+                       |
                                                          v
                                                DECISIONS.md
                                                     |
                      task_relations/*.json ----------+----> FRAMEWORK.md
                                 |                    |      and detailed syntheses
                                 +--------------------+
                                                     |
                                                     v
                                                  STATE.md
                                           canonical restart view
```

The normal workflow is write-through across several files:

1. `OBJECTIVE.md`, `DECISIONS.md`, and protocol documents constrain a question.
2. `BACKLOG.md` and `STUDIES.jsonl` identify or register a Study.
3. `studies/<ID>/PLAN.md` pre-registers it; `NOTE.md` records execution context; result JSON and `analysis.md` hold outputs and interpretation.
4. Internal experimental evidence is promoted into `FINDINGS.md` or `FAILURES.md`; literature work is retained as cards, an index, and an `LT-*` Study outcome.
5. `DECISIONS.md` binds scope changes; `STATE.md` restates enough of the current position for restart; `FRAMEWORK.md` and other syntheses derive cross-study conclusions.

## Artifact ownership and lifecycle

| Artifact | Information owned | Canonical or derived | Current writer | Referenced by | Lifecycle / overlap / query limitation |
|---|---|---|---|---|---|
| `OBJECTIVE.md` | Programme identity, active method scope, progression gates, screening/confirmation rules, ER and plateau policy | Canonical policy/configuration | Human decisions recorded by the main research workflow | `AGENTS.md`, skills, plans, framework | Mutable when a binding decision changes it. Compact and machine-like, but not schema-validated. Overlaps decision consequences. |
| `FRAMEWORK.md` | Cross-study method-selection synthesis, evidence levels, selection rules, open questions | Derived maintained synthesis | Analysis workflow/main OMP | `STATE.md`, commands, future plans | Explicitly derived from findings, decisions, failures, Studies. Mutable. Cheap for broad conclusions, poor for exact claim-to-source traversal. |
| `STATE.md` | Canonical restart position, phase, current work, concise summaries, operational caveats | Canonical **view** for restart, not canonical evidence | Main cycle when state changes | Every agent workflow | Mutable and heavily duplicated. It says to remain concise but is now large and contains historical detail. It can lag or conflict; `FINDINGS.md` wins for evidence. |
| `BACKLOG.md` | Scientific questions, priority, gates, reserved/proposed Study identities | Canonical planning state | Main OMP; agents may reprioritize with recorded reason | Study planning and commands | Mutable. Mixes active entries, completed outcomes, policy, and literature-mode procedure. IDs can be allocated before registration, so presence does not imply a Study exists. |
| `FINDINGS.md` | Established scientific findings, evidence tier, exact numbers, provenance; record-level observations | Canonical for findings, numbers, provenance | Analysis/Study-closing workflow | `STATE.md`, framework, decisions, literature questions | Mutable-in-place with explicit refinements; entries are never silently deleted. Stable `F#` and `R#` IDs. Literature outcomes appear only indirectly, e.g. `R3`. |
| `FAILURES.md` | Negative evidence grouped by explanation | Canonical negative-evidence registry | Analysis/Study-closing workflow | literature workflow, framework, decisions | Append-oriented but entries can be refined. Stable `FL-xxxx`. It mixes experimental failures, failed justifications, and negative literature results. |
| `DECISIONS.md` | Binding scope and research decisions, evidence causing change, consequences | Canonical binding decision log | Human decisions recorded by main OMP; engineering conclusions where allowed | All workflows | Append/supersede model. Stable `DEC-xxxx`. Corrections and supersession preserve history. Not a queryable relation table despite rich evidence references. |
| `STUDIES.jsonl` | One current structured record per registered Study: identity, type, lifecycle, hypothesis, outcome, links, execution metadata | Canonical Study registry | Study registration/closing workflow | All commands, `STATE.md`, skills | Mutable one-record-per-Study registry, not an event log. Eight current rows have a highly heterogeneous union of fields and no visible schema/validator. Stable Study IDs. |
| `studies/<ID>/PLAN.md` | Pre-registered question, hypothesis, competing explanation, variables, controls, protocol, decision rule | Canonical pre-registration for that Study | Main OMP before execution; literature workflow for `LT-*` | Study analysis, tests, decisions | Intended as a historical plan. Status text can remain as written at registration: `LT-0002/PLAN.md` says `RUNNING` while the registry and analysis say complete. This is useful history but ambiguous without lifecycle rules. |
| `studies/<ID>/NOTE.md` | Human gate, run note, execution ledger, interpretation summary | Canonical Study context/run note | Study workflow | MLflow/DagsHub note, status workflows | Mutable during Study lifecycle. Shape varies by Study generation. |
| `studies/<ID>/result*.json` | Machine-readable analysis output or Study result | Canonical repository artifact for the recorded analysis; underlying execution remains in MLflow/DagsHub | Deterministic analysis scripts or Study workflow | analyses, findings, tests | Generated or assembled per Study, with Study-specific schemas. Not globally queryable. `LT-0001` has `result.json`; `LT-0002` does not. |
| `studies/<ID>/analysis.md` | Study-specific interpretation and decision against the plan | Canonical Study interpretation | Analysis/Study-closing workflow | findings, failures, decisions, framework | Mutable by explicit correction/addendum. Rich provenance, but relations are prose links. |
| `literature/` cards | Per-paper verified citation, problem, assumptions, relation object, method, evidence, project differences, difficulty, and Study-specific assessment | Canonical retained record of a verified source; content is derived from a primary paper | Current `wav-literature` workflow / literature Study | `literature/INDEX.md`, LT analyses, audits, architecture records | Mutable with explicit corrections. Card filename slug acts as the de facto paper ID. No formal bibliographic identity, lifecycle, claim IDs, checksums, or globally structured screening reason. |
| `literature/INDEX.md` | Human comparison tables grouped by `LT-0001` and `LT-0002`, plus synthesis | Canonical human registry/view for verified cards | Literature workflow | commands, skills, findings, failures | Mutable and duplicated with LT analyses and `STUDIES.jsonl.cards`. “Current literature Study” still names completed `LT-0002`. Good browsing view; poor exact/deduplicated queries. |
| `literature_survey/` | Taxonomy maps, primary-source ledger, predictions, candidate analyses, hypothesis graph, open questions, and post-TR-0007 synthesis | Derived literature/theory knowledge and historical prediction snapshots | Imported independent survey work plus later reconciliation | `STATE.md`, proposals, backlog | Not binding; files explicitly defer to findings/decisions/protocols. Git history shows 18 documents moved from a second root research tree without prose changes; stale checkout-specific statements are preserved and corrected by status notes/addenda. This is valuable history, not a canonical paper catalog. |
| `task_relations/*.json` | Machine-readable directed-transfer, LOSO, and optimizer-control summaries keyed to `DG-0001` | Derived machine outputs from Study analysis | Deterministic analysis scripts | findings, framework, `MTRL_DIAGNOSTIC_SYNTHESIS.md` | Generated/replaceable for the producing Study. Stable `study_id` and `stage`, but no independent finding/relation IDs. |
| `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md` | Per-quantity synthesis of MTRL evidence and failure explanations | Derived maintained synthesis | Analysis workflow | framework, state/status commands | Mutable and explicitly subordinate to `FINDINGS.md`. High-value context but expensive to reread and not cheaply filterable. |
| `proposals/` | Draft pre-registrations using allocated future Study IDs | Non-canonical proposal state | Main OMP/design workflow | backlog, state, decisions | Mutable drafts. Explicitly not registered and not executable. Their `TR-*` names overlap the Study namespace by reservation, which requires reading status text to distinguish proposal from Study. |
| `audits/` | Dated theory-to-implementation and integration audits | Canonical audit record for the audited question; derived from code/papers | Audit workflow/reviewer | decisions, Studies, tests | Historical, corrected by addendum rather than rewrite. Date/title filenames, no audit registry. |
| `authorizations/` | Human authority envelope for paid compute, one YAML per Study scope | Canonical authorization policy | Human only | compute backend | Strict, committed, expiring, frozen while busy. Uses Study ID as filename. Runtime facts and credentials correctly live elsewhere. |
| `tests/` | Executable scientific/protocol/identity invariants | Canonical executable checks | Engineering workflow | `make check` | Current tests cover experiments, mechanisms, run identity, and protocols. There is no literature catalog, literature lifecycle, or research-registry consistency test. |
| `.agents/commands/wav-literature.md` | Current literature procedure and mutation scope | Canonical workflow intent | Agent-asset maintainers | OMP command routing | Already literature-specialized but broad: it reads and mutates cards, index, backlog, and Study registry. It has no narrow deterministic interface. |
| `.agents/skills/*` | Research and experiment operating rules | Canonical agent competence | Agent-asset maintainers | OMP/Codex | The experiment skill defines IDs and closing semantics. There is no dedicated literature skill yet. |
| `.omp/AGENTS.md` | OMP instruction entry point | Symlink/pointer to canonical `AGENTS.md` | Agent-asset maintainers | OMP | No separate OMP configuration or research registry is present. Repository history and instructions say it must remain a symlink, avoiding a second instruction source. |
| weekly reports/slides | Supervisor-facing time-sliced views | Derived view | Weekly workflow/humans | human reporting | Date-keyed snapshots; not authoritative research state. |

## Existing write paths

The repository does not identify individual human authors for every record. It does identify workflow authority:

- Human: binding scientific decisions and compute authorization.
- Main OMP/research cycle: Study registration, state reconciliation, backlog, decisions, closing updates.
- Analysis workflow: findings, failures, framework, task-relation synthesis.
- Literature workflow: verified cards, `literature/INDEX.md`, literature Study records, justified backlog candidates.
- Deterministic scripts: Study-specific result JSON and task-relation JSON.
- MLflow/DagsHub: canonical execution facts; Git: scientific interpretation and policy.

# 2. Existing Entities

No new entity is proposed in this section. These are the concepts already represented.

| Existing concept | Current representation | Existing identifier |
|---|---|---|
| Study | `STUDIES.jsonl` row plus `studies/<ID>/` | `BL-xxxx`, `DG-xxxx`, `TR-xxxx`, `AB-xxxx`, `LT-xxxx`; `FW-xxxx` appears in backlog for final analyses |
| Run | MLflow/DagsHub run plus Study-local run/result metadata | MLflow run ID and structured run name; linked to a Study ID |
| Finding | `FINDINGS.md` | `F1`–`F10` |
| Record-level observation | `FINDINGS.md` | `R1`–`R3` |
| Negative evidence / failure | `FAILURES.md` | `FL-xxxx` |
| Decision | `DECISIONS.md` | `DEC-xxxx` |
| Backlog question | `BACKLOG.md` section | Usually a future Study-shaped ID (`DG-*`, `TR-*`, `AB-*`, `FW-*`); status determines whether it is merely planned |
| Proposal | `proposals/<reserved-study-id>_*.md` | Reserved `TR-*` ID, explicitly `NOT REGISTERED` |
| Authorization | `authorizations/<Study-ID>.yaml` | The Study ID is the scope ID |
| Audit | `audits/<date>-<slug>.md` | Date/slug filename only |
| Paper | `literature/<author-year-slug>.md`; sometimes a key in `LT-0001/result.json` | De facto stable card slug. No formal `paper_id`; external DOI/arXiv identity is not consistently structured |
| Literature assessment | Card sections such as `LT-0001 decision` or `LT-0002 assessment`, index row, and LT analysis matrix | No independent ID; context is inferred from paper slug + `LT-*` Study |
| Literature finding | `LT-*` analysis/outcome; sometimes promoted to `FINDINGS.md` record observation or `FAILURES.md` negative result | No dedicated literature-finding ID; currently `R3` and `FL-0003` carry one such outcome |
| Literature claim | Prose in cards and `literature_survey/PRIMARY_LITERATURE.md` | No stable claim ID |
| Research question | `PLAN.md` section, backlog entry, or locally numbered `OPEN_QUESTIONS.md` item | Usually addressed by a Study ID; open-question numbering is file-local |
| Relation measurement | `task_relations/*.json`, Study result JSON, syntheses | Producing Study ID + stage + local matrix/cell key |
| Synthesis | `FRAMEWORK.md`, `MTRL_DIAGNOSTIC_SYNTHESIS.md`, `literature_survey/*.md` | Filename only |
| Primary experimental evidence | MLflow/DagsHub run artifacts and metrics | Run ID, Study ID, stage, seed, commit |
| Primary literature evidence | External paper URLs cited in cards/ledgers | URL and prose citation; no repository-wide structured external identifier/checksum |

The strongest existing reusable identifiers are Study IDs, finding/failure/decision IDs, MLflow run IDs, and literature card slugs. The roadmap should build on these before inventing replacements.

# 3. Current Sources of Truth

| Question | Current authority | Ambiguity or duplication |
|---|---|---|
| What is the programme trying to do? | `OBJECTIVE.md`, with binding amendments in `DECISIONS.md` | `STATE.md`, `BACKLOG.md`, and framework prose restate the progression. `STATE.md` still contains a paragraph saying both diagnostic and literature gates are required even though `OBJECTIVE.md` and `DEC-0013` defer the diagnostic gate. Binding decision wins. |
| Where should a fresh agent resume? | `STATE.md` | It duplicates findings, decisions, current Studies, infrastructure notes, and history; freshness must be reconciled. |
| What Studies exist and what is their current lifecycle? | `STUDIES.jsonl` | Study PLAN headers may preserve an earlier lifecycle state. Registry schema varies by Study. `BACKLOG.md` and proposals also use Study-shaped IDs without registration. |
| What was pre-registered? | `studies/<ID>/PLAN.md` | Plans are historical snapshots, but this immutability rule is not encoded. |
| What happened in one Study? | Study result JSON + `analysis.md` + `NOTE.md`; MLflow/DagsHub for execution facts | File set and JSON schema differ by Study. Closing facts are duplicated into the registry, findings/failures, state, and backlog. |
| What numbers and scientific findings are established? | `FINDINGS.md` | `STATE.md`, framework, and syntheses duplicate summaries. They explicitly defer to findings when inconsistent. |
| What negative evidence exists? | `FAILURES.md` | Negative Study outcomes also remain in Study analysis and registry; classification spans experimental, literature, and justification failures. |
| What decisions bind? | `DECISIONS.md` | Consequences are repeated in objective, state, backlog, plans, and agent instructions. |
| What papers have been retained as verified? | Cards under `literature/` and the human index | `STUDIES.jsonl.cards` covers LT-0002 only; LT-0001 has a separate `result.json` source list. No single structured complete catalog. |
| What does a paper say? | Ultimately the primary paper at the cited URL; locally, the card is the retained verified interpretation | No managed PDF identity, checksum, S3 locator, or cache status. Cards and survey ledgers may both summarize the same source. |
| Why was a paper included/excluded? | Study-specific assessment in its card plus LT analysis/index | A paper verdict is contextual. The same paper can be useful evidence but rejected for one mechanism gate. There is no structured assessment keyed by paper and question/Study. |
| What literature claims exist? | Card prose and `literature_survey/PRIMARY_LITERATURE.md` | No first-class claim identity, locator, topic link, support/contradiction relation, or confidence field. |
| What internal task-relation measurements exist? | Producing Study artifacts and `task_relations/*.json`; established interpretation in `FINDINGS.md` | Machine records are Study-specific; syntheses repeat them. |
| What is the current execution state? | MLflow/DagsHub and compute/infra runtime state | Git contains interpretation, not live worker/job truth. This separation is explicit and should remain. |
| What is authorized? | `authorizations/<scope>.yaml` plus binding human decisions | Absence means no authority. This is already a strong deterministic seam. |

# 4. Literature Architecture Today

## `literature/`: verified per-paper cards

This directory is the retained source registry in human-readable form. It has 23 cards plus `INDEX.md`. Every card follows a recognizable but unenforced template: citation, problem, mathematical assumption, relation representation, optimization, evidence, assumptions, differences from wavCSE, implementation difficulty, candidate Study ID, and a Study-specific assessment.

The filename slug is already used as identity:

- `STUDIES.jsonl` LT-0002 lists card slugs.
- `literature/INDEX.md` links by slug.
- audits and survey documents cite card paths.

Therefore the repository has a de facto paper identity, but not a declared or validated one. Bibliographic identity, DOI/arXiv IDs, primary URL, verification event, and PDF artifact identity remain embedded in prose.

Cards are not original papers. They are derived, revisable knowledge checked against a primary source. The Git history demonstrates corrections in place with a dated correction section, e.g. the MSSL equation correction.

## `studies/LT-*`: bounded literature investigations

`LT-0001` and `LT-0002` are Studies, not papers or topic folders.

- `LT-0001` starts from internal finding `F9`, pre-registers a literature question and gates, screens eight sources, and records a negative Study outcome. Its `result.json` is the only structured per-paper source list for that Study. The result is promoted into `FINDINGS.md` `R3`, `FAILURES.md` `FL-0003`, `DEC-0008`/later decisions, the literature index, and state/framework prose.
- `LT-0002` starts from human re-scope `DEC-0013`, screens 15 cards across two method families, and produces candidate rankings. It has no `result.json`; its structured card list lives in the `STUDIES.jsonl` row, while gate outcomes live in Markdown analysis and cards.

An `LT-*` ID identifies the **question and screening context**, not the paper. A paper assessment should therefore never be reduced to one global include/exclude status. `LT-0001` exclusions are scoped to the F9 rationale; `LT-0002` asks a different benchmark/faithfulness question.

## `literature_survey/`: integrated theory and synthesis corpus

This directory is neither the paper registry nor an `LT-*` Study. It is an 18-document literature/theory work product originally developed in a separate worktree under a second root research tree. Commit `aefac2d` moved it under the canonical research memory and rewrote 284 relative links without changing prose.

Its roles include:

- taxonomy map;
- primary-source claim ledger;
- candidate and directional analyses;
- pre-registered theoretical predictions;
- hypothesis graph and open questions;
- post-TR-0007 reconciliation/synthesis.

The files explicitly label `OBSERVED`, `DERIVED`, `HYPOTHESIZED`, and `RECORD`, defer to repository authorities, and preserve stale checkout-specific statements with status notes rather than rewriting history. They are derived knowledge and historical prediction records. They overlap cards in citation and paper interpretation, and overlap Study/framework documents in synthesis, but they do not replace either.

## Duplication and missing semantics

Current duplication is purposeful in some places but untyped:

- Paper identity appears in card filename, citation prose, index row, LT result keys, and `STUDIES.jsonl.cards`.
- A screening verdict appears in the card, index, LT analysis, LT result/registry, failure/finding records, and decisions.
- Primary-source claims appear in both cards and `PRIMARY_LITERATURE.md`.
- Topic syntheses repeat internal evidence from findings and task-relation syntheses.

Missing semantics:

1. No declared stable `paper_id`, despite slugs functioning as one.
2. No complete structured paper catalog across both LT Studies.
3. No contextual assessment record keyed by `(paper, LT Study/question)`.
4. No explicit distinction in machine state among discovered, retrieved, verified, screened, retained, excluded-for-this-question, and superseded assessment.
5. No first-class literature claim with a source locator and provenance.
6. No managed primary-artifact reference, checksum, S3 object identity, or local-cache status.
7. No explicit paper-to-topic/research-question links outside prose.
8. No consistency check that every card/index/Study reference resolves.
9. No cheap way to ask whether a DOI/title is already known before searching.

# 5. Existing Query Patterns

| Query | Current traversal |
|---|---|
| What do we know now? | `STATE.md` → `FINDINGS.md` → `FAILURES.md` → `DECISIONS.md` → recent `STUDIES.jsonl` rows and Study analyses → `FRAMEWORK.md`/task-relation synthesis |
| What should we investigate next? | `STATE.md` current phase/pending work → `BACKLOG.md` gates → `DECISIONS.md` → `FRAMEWORK.md` missing evidence → relevant Study plans/proposals |
| What evidence supports finding `F10`? | `FINDINGS.md#F10` → `studies/DG-0005/{PLAN,analysis,result files}` → MLflow experiment/run IDs; then decisions/framework for consequence |
| What happened in `DG-0007`? | `STUDIES.jsonl` → `studies/DG-0007/{PLAN,NOTE,...}` → audit → decisions → `STATE.md`; runtime/tracking reconciliation if execution is disputed |
| What literature motivated `TR-0013`? | `STATE.md`/`BACKLOG.md` → proposal/Study plan → `literature_survey/POST_TR0007_SYNTHESIS.md` → MSSL card/primary ledger → `LT-0002` analysis → `DEC-0015`/`DEC-0016` |
| What papers do we know about sparse precision? | `literature/INDEX.md` → likely cards → `literature_survey/SURVEY_MAP.md`, `MSSL_SPARSITY_ANALYSIS.md`, `PRIMARY_LITERATURE.md` → LT analyses |
| Have we screened paper Y? | Search card filenames/titles and `INDEX.md`, then LT analysis/result/registry. DOI/title variants are not normalized. |
| Why was paper Y excluded? | Card's LT-specific assessment → matching LT PLAN gates → LT analysis matrix; possibly decisions if the premise later changed |
| Which papers support or contradict claim X? | Read cards and primary ledger, reconstruct claim wording and topic manually; no claim IDs or stance relation |
| What unresolved questions concern X? | `BACKLOG.md` + `FRAMEWORK.md` + `literature_survey/OPEN_QUESTIONS.md` + proposal documents; local numbering and stale snapshots must be interpreted |
| Which decision relied on which evidence? | Search `DECISIONS.md` prose for finding/Study IDs, then traverse those records. No reverse index from evidence to decision. |
| Which Study produced a finding? | Finding provenance prose → Study folder/MLflow. This works when the provenance paragraph is current, but is not structured. |

The current workflow is evidence-conscious and traceable, but query cost scales with context volume rather than result count.

# 6. Agent Pain Points

1. **High restart cost.** Current commands instruct an agent to read eight or more top-level records plus recent Study folders before answering status or literature questions.
2. **Manual joins.** Relations among paper slug, LT Study, internal finding, failure, decision, and successor Study are prose references.
3. **Contextual verdict ambiguity.** “Rejected” can mean excluded by taxonomy, rejected for one rationale, not faithfully portable, or experimentally rejected. A global paper status would be wrong; the current files require careful reconstruction.
4. **Duplicate lifecycle text.** Registry, PLAN header, NOTE, analysis, index, state, and backlog can describe different lifecycle moments. `LT-0002` demonstrates this: its PLAN retains `RUNNING`, while registry/index/analysis say complete.
5. **Heterogeneous machine state.** `STUDIES.jsonl` is structured but has no uniform schema beyond a small core. Study result JSON is not uniform.
6. **No cheap literature deduplication.** The agent cannot reliably answer “already known?” from DOI/arXiv/title without reading cards and handling citation variants.
7. **No claim-level provenance.** The smallest addressable literature unit is usually a whole card or Markdown row. Exact support/contradiction and page/equation locators are not queryable.
8. **Primary evidence is unmanaged.** URLs are present, but retrieval status, immutable checksum, canonical S3 key, and local cache are absent. Re-reading can depend on live URLs.
9. **Derived/canonical boundaries are stated in prose, not enforced.** The survey is disciplined about evidence labels, but a future agent must know that its predictions and stale snapshots cannot override findings/decisions.
10. **Broad mutation authority.** The existing literature command can touch cards, index, backlog, and Study registry in one workflow. That is wider than a bounded literature specialist should receive.
11. **No reverse references.** “Which decisions used this paper/finding?” requires full-text search.
12. **No literature consistency gate.** Existing tests validate scientific protocols and run identity, not card completeness, duplicate external identity, or broken literature references.
13. **State bloat.** `STATE.md` is authoritative for restart but now contains substantial historical and operational material. An agent pays this cost because no cheap current-state projection exists.
14. **Historical snapshots need interpretation.** The survey intentionally preserves stale statements and adds reconciliation notes. A naïve agent may quote the old sentence instead of following the addendum.

# 7. Target Architecture

The smallest reasonable long-term direction is a **federated registry/query layer**, not a replacement research store.

```text
question
   |
   v
narrow query interface
   |
   v
Git registries / resolvers
   |---------------------------|
   v                           v
literature identity +       existing Study/finding/
contextual assessments      decision authorities
   |                           |
   +-----------+---------------+
               v
        selected detailed artifacts
        cards / analyses / syntheses
               |
               v
 primary evidence only when required
 papers from S3/cache or MLflow/DagsHub runs
```

## Preserve the current authorities

- `FINDINGS.md` remains authoritative for established internal findings.
- `DECISIONS.md` remains binding.
- `STATE.md` remains the restart view until actual query usage justifies changing it.
- `STUDIES.jsonl` and Study folders remain the Study system.
- Existing cards remain the per-paper derived knowledge artifacts; do **not** create another summary tree.
- `literature_survey/` remains derived/historical synthesis; do **not** normalize or rewrite it.
- MLflow/DagsHub remains execution authority.
- Original PDFs become authoritative primary literature artifacts when deterministic retrieval is introduced.

## Add only the missing seams

1. **Literature identity seam.** Declare existing card slugs as stable paper IDs and add structured external identifiers/paths once, without moving cards.
2. **Contextual assessment seam.** Represent a screening outcome as paper + literature question/Study, not a global paper verdict.
3. **Query seam.** A small deterministic read-only module resolves IDs and returns records plus paths. It hides Markdown traversal from normal callers without hiding evidence.
4. **Primary-artifact seam.** Deterministic retrieval maps paper identity to S3 object/checksum/local cache. Infrastructure implementation remains in the infrastructure repository; credentials never enter model context.
5. **Claim seam.** Add claim records only when claim-level queries are exercised. Claims cite paper IDs and exact locators; no graph database is required.
6. **Agent seam.** The eventual Literature Review Agent receives capability-oriented literature interfaces, not Bash and not authority over decisions, experiments, or compute.

## Authority model

| State | Intended authority |
|---|---|
| Original paper/supplement | Authoritative for what the publication says |
| Paper catalog and literature assessment records | Authoritative for repository processing identity, lifecycle, verification, and question-specific screening |
| Literature claims | Canonical structured extraction with exact source provenance |
| Existing cards | Derived per-paper knowledge and implementation-oriented interpretation |
| Existing survey/syntheses | Derived cross-paper or cross-evidence knowledge |
| `LT-*` Study | Canonical record of a bounded literature investigation and its gate outcome |
| Main OMP decisions | Sole authority for research direction, experiment approval, and promotion into binding programme state |

The registry should return IDs and pointers first. Detailed cards, syntheses, and primary PDFs should be opened only for the small result set.

# 8. Incremental Roadmap

## Execution order actually taken (read this before numbering)

This section is the plan; the order below is what was executed, recorded here so
the increment numbers are not mistaken for the build order.

| Roadmap slot | Content | Status |
| --- | --- | --- |
| INC-001 | Declare/catalog paper identity | done |
| INC-002 | Question-scoped assessment ledger | **not built** — superseded |
| INC-003 | Read-only literature query module | done |
| INC-004 | Primary-artifact manifest and retrieval contract | done (remote seam deferred) |
| — | **Literature Agent vertical slice (read-only V1)** | this increment, inserted before INC-004B |
| INC-004B | Connect primary-artifact storage to `infra/` | **deferred** |
| INC-005 | Literature claim records | **partially built** — provenance, locators and verbatim quotes enforced; topic/stance links and Study links deferred |
| INC-006 | Cards/syntheses provenance | later |
| INC-007 | Literature-review skill | later |
| INC-008 | Bounded Literature Review Agent | later |
| INC-009 | Exercise real questions, measure query behaviour | later |
| INC-010 | Broader research query index | later |

Roadmap INC-002 was not built to restore numbering, and must not be. Query-time
joins in `literature_query` already derive Study-scoped assessments from their
existing authoritative sources — `STUDIES.jsonl.cards` and each Study's
`result.json` — so a second persisted ledger would duplicate state that already
has an owner. The invariant it existed to protect (assessments are
Study-scoped, never a global paper status) is enforced in the query layer and
tested there.

INC-004 was deliberately stopped at the repository boundary because this
repository has no S3 client and infrastructure mechanics belong to `infra/`;
INC-004B records the remaining work. It is deferred so the Literature Agent
workflow can be observed first.

## LIT-AGENT-V1 — Read-only Literature Agent vertical slice (this increment)

**Goal**

Build the smallest useful Literature Agent — an evidence/review specialist the
main research OMP session can delegate a bounded literature question to, which
answers from retained evidence and hands back provenance plus uncertainty.

**Why now (and why this reorders INC-005..INC-008)**

The roadmap's original order put claims (INC-005), card/synthesis provenance
(INC-006), the literature skill (INC-007) and the bounded agent (INC-008) after
primary retrieval. Building a first, read-only vertical slice against *existing*
retained evidence answers the more valuable question earlier: which capability is
actually missing in practice? A read-only slice needs no claim schema and no
durable write semantics, so it cannot get the authority model wrong, and it
produces the usage evidence that INC-004B (and any later increment) should be
justified by. Nothing is skipped: INC-005..INC-010 remain open.

**Exact scope**

- A compact project-scope Literature Agent definition (`.omp/agents/`).
- One Literature Review skill (`.agents/skills/wavcse-literature-review/`)
  carrying the procedural methodology; the agent definition stays thin.
- Semantic model-facing tools over the deterministic operations that already
  exist: `literature_resolve`, `literature_query`, `literature_read`,
  `literature_primary`.
- A bounded artifact reader (`literature_read.py`) that accepts only resolved
  references — a `paper_id` or a registered `LT-*` artifact kind — never an
  arbitrary filesystem path.
- A structured investigation-result contract that separates paper claim,
  reported evidence, agent interpretation and research implication.
- Two real evaluation runs against current Task Relation Learning questions.

**New invariant introduced**

The Literature Agent mutates no canonical research state. Its only write is the
disposable primary cache, reached through `literature_primary`. Every
paper-specific statement it returns carries `paper_id` and an evidence level
(`primary`, `card-derived`, `Study-derived`), and a claim derived from a card is
never presented as if the primary PDF had been inspected.

**Migration required**

None. All existing artifacts, cards, Studies and registries are read as they are.

**Backward compatibility**

Full. Existing commands and skills are unchanged; the current
`wav-literature` command remains the write-capable workflow and is not replaced.

**Tests/verification**

Deterministic adapter tests (identity resolution, dedup verdicts, query,
Study-scoped assessment retrieval, bounded artifact read, invalid reference,
primary unavailable, integrity failure, no canonical mutation, no filesystem
escape, structured errors), agent-asset validation, and two real bounded
investigations with recorded tool use, provenance quality and friction.

**What is deliberately NOT done**

No S3/`infra` integration (INC-004B), no claim registry or durable claim writes,
no external discovery or search, no PDF ingestion, no OCR, no vector/RAG store,
no research-state normalization, no repository restructuring, and no OMP-facing
capability the agent could use to mutate research state.

**Exit criteria**

The main research session can delegate a bounded literature question, the
Literature Agent answers it from retained evidence through the semantic tools
alone with structured provenance and explicit uncertainty, the agent demonstrably
cannot mutate canonical research state, and the friction observed across at least
two real runs is recorded with the next increment recommended from evidence.

## INC-001 — Declare and catalog existing paper identity

**Goal**

Make “Do we already know this paper?” and “Which card is the canonical retained record?” cheap and deterministic.

**Why now**

The 23 existing card slugs already function as IDs in `STUDIES.jsonl`, indexes, audits, and survey links. The repository lacks only a declared, validated identity layer. This is the smallest change that improves the next literature discovery/screening workflow without changing any historical artifact.

**Exact scope**

- Declare each existing card slug as an immutable `paper_id`.
- Add one Git-versioned identity catalog under `research/literature/` with one row per card.
- Record only identity and resolution fields initially: `paper_id`, canonical card path, normalized title, year, authors/citation key, DOI/arXiv/venue identifier when known, and primary-source URL(s).
- Add a schema and deterministic validator.
- Validate references from `STUDIES.jsonl.cards` and `literature/INDEX.md` where mechanically resolvable.

**Files/components affected**

- New `improvements/taskrelation/research/literature/catalog.jsonl`.
- New catalog schema and validator/test in the existing research test area.
- Minimal pointer/invariant additions to `literature/INDEX.md` and the relevant agent instruction; existing card content remains untouched.

**New invariant introduced**

Every `literature/*.md` paper card except `INDEX.md` has exactly one catalog row; every `paper_id` and external identifier is unique; `paper_id` equals the existing immutable card slug; every catalog path exists.

**Migration required**

One identity row for each of the 23 cards. Extract bibliographic identity from existing cards only. Do not change card prose, filenames, LT records, verdicts, or survey documents.

**Backward compatibility**

Complete. Existing paths, links, slugs, Study card lists, and human index remain valid. The catalog points to them.

**Tests/verification**

- JSON schema validation.
- Unique paper IDs and normalized external identifiers.
- Catalog/card one-to-one coverage.
- All catalog paths and Study card references resolve.
- A smoke query by exact slug, DOI/arXiv ID, and normalized title returns the existing card path.

**What is deliberately NOT done**

No lifecycle statuses, screening verdicts, claims, topics, summaries, agent, query server, PDF retrieval, S3, card front matter, or broader research registry.

**Exit criteria**

A deterministic local check can answer whether a candidate paper is already known and return its canonical card in one lookup, while every existing research artifact remains byte-for-byte unchanged except the index/instruction pointer if required.

## INC-002 — Add question-scoped literature assessments

**Goal**

Answer “Was paper Y screened, for which question, and why was it included or excluded?” without treating a contextual verdict as a global paper status.

**Why now**

`LT-0001` and `LT-0002` prove that screening verdicts are Study-specific. Their representations differ: LT-0001 has structured source records; LT-0002 has a registry card list and Markdown gate matrix.

**Exact scope**

- Add a structured assessment ledger keyed by `(study_id, paper_id)`.
- Store assessment lifecycle, gate outcomes, include/exclude classification, concise reason, primary-source verification status/date, and links to the authoritative card/Study analysis.
- Backfill only `LT-0001` and `LT-0002` from existing records.
- Keep Study-level decision in `STUDIES.jsonl`; keep full reasoning in cards/analysis.

**Files/components affected**

- New `literature/assessments.jsonl`, schema, validator, and tests.
- Catalog validator extended to resolve `paper_id`.
- No changes to historical cards or LT Study files.

**New invariant introduced**

A paper may have many assessments, but at most one current assessment per `(study_id, paper_id)`; every assessment resolves to an existing paper and registered `LT-*` Study; a verdict always carries its question/Study context.

**Migration required**

Backfill 23 Study-paper assessment rows from LT-0001/LT-0002. Record conflicts instead of selecting silently.

**Backward compatibility**

Cards, index tables, and LT analyses remain authoritative detailed records and human views. The ledger is a structured locator and lifecycle record alongside them.

**Tests/verification**

- All LT-0001/LT-0002 screened papers represented exactly once for their Study.
- Gate/verdict vocabulary validation.
- References resolve.
- Known cases demonstrate contextual semantics: GradNorm excluded for LT-0001 taxonomy; MSSL passes LT-0002; AMTL is deviation-class rather than globally rejected.

**What is deliberately NOT done**

No claim extraction, topic ontology, automatic prose generation, global paper status, or historical rewrite.

**Exit criteria**

One deterministic query returns every screening context and reason for a paper without opening every LT artifact.

## INC-003 — Introduce a read-only literature query module

**Goal**

Expose cheap literature lookups over identity and assessments before introducing any mutating tool or agent.

**Why now**

INC-001/002 create the smallest structured substrate. Query patterns can now be measured instead of guessed.

**Exact scope**

Implement a small local module/CLI with read-only operations for identity lookup, assessment lookup, list-by-Study, and candidate deduplication. Return compact JSON records with paths, never synthesized scientific conclusions.

**Files/components affected**

A narrow module under the Task Relation Learning research tooling area, command entry point, and behavior tests. Exact placement should follow repository module conventions at implementation time.

**New invariant introduced**

Normal literature lookup reads validated registries first and opens Markdown only after returning a bounded set of IDs/paths.

**Migration required**

None beyond INC-001/002.

**Backward compatibility**

Direct Markdown reading remains supported. Existing commands can adopt the query module incrementally.

**Tests/verification**

Behavioral tests for known-paper lookup, unknown paper, ambiguous title, DOI normalization, per-Study verdicts, and deterministic ordering. Smoke the CLI against the 23-paper corpus.

**What is deliberately NOT done**

No mutation, search provider, LLM synthesis, vector search, database, or generic research-query interface.

**Exit criteria**

The current “already known / screened / why excluded?” queries are answered from compact JSON plus selected paths, and measured query output is substantially smaller than reading the literature tree.

## INC-004 — Add deterministic primary-paper retrieval and cache identity

**Goal**

Make the original PDF reliably retrievable by paper ID while keeping credentials and storage mechanics out of model reasoning.

**Why now**

Only after identity is stable can S3 keys, checksums, and cache entries be durable. Retrieval is needed before an agent can safely re-verify claims when live URLs fail.

**Exact scope**

- Define a paper-artifact manifest relation: paper ID → original artifact checksum/media type/source URL/S3 locator.
- Put object publication, credentials, S3 transfer, cache validation, and eviction behind deterministic infrastructure tooling in `wavcse-infra`, consistent with repository boundaries.
- Keep the disposable local cache outside Git.
- Add a repository-side adapter that receives no credentials and returns a verified local path or explicit absence.

**Files/components affected**

Identity catalog artifact fields or a separate manifest; infrastructure-repository retrieval implementation; narrow repository adapter/tests. No PDFs in Git.

**New invariant introduced**

A cached PDF is trusted only when its checksum matches the manifest; S3 is the canonical stored primary artifact; cache copies are disposable; URLs are provenance, not cache identity.

**Migration required**

Populate manifests and upload existing retained primary papers only under explicit infrastructure/storage authority. Missing papers remain explicit, not silently substituted.

**Backward compatibility**

Cards and URLs continue to work. A paper without an S3 artifact remains queryable and reports retrieval unavailable.

**Tests/verification**

Checksum validation, cache hit/miss/corruption, no credentials in logs/model-facing JSON, deterministic S3 key resolution, and offline retrieval smoke for one authorized test paper.

**What is deliberately NOT done**

No OCR pipeline, bulk corpus ingestion, semantic chunking, web application, or agent.

**Exit criteria**

Given a paper ID, deterministic code returns a checksum-verified local primary artifact or a precise unavailable reason without exposing credentials.

## INC-004B — Connect primary-artifact storage to wavcse-infra

**Status:** `DEFERRED` — deliberately not implemented. Deferred on 2026-10-01 until
the first Literature Agent vertical slice has been exercised on real research
questions and the actual workflow is observed.

**Goal**

Define the clean generic storage boundary between wavCSE research semantics and
`infra/` mechanics for retained primary artifacts.

**Why deferred**

INC-004 established the manifest schema, checksum/cache policy and the
deterministic retrieval contract, with remote transfer as an injected seam and
no transfer backend wired. Whether that seam is the right shape should be decided
from observed Literature Agent usage, not from anticipation. Speculative
infrastructure is exactly what this programme's roadmap philosophy forbids.

**Ownership rules (binding when implemented)**

wavCSE owns:

- `paper_id` (the immutable canonical literature card slug);
- literature semantics — cards, `LT-*` Studies, catalog;
- `primary_manifest` semantics — which artifact is retained, its SHA-256, media
  type, size, provenance URL and derived object key;
- provenance and research state.

`infra/` owns:

- credentials and the provider credential chain;
- S3 (or provider) mechanics;
- upload and download;
- retry and backoff;
- reconciliation of ambiguous or interrupted transfers;
- provider configuration and endpoint/region selection.

`infra/` MUST NOT need to understand `paper_id` as a scientific/domain concept,
literature cards, `LT-*` Studies, claims, or any literature-specific workflow. It
receives an opaque object key and a destination and reports facts.

**Exact scope when implemented**

- A real `fetcher` implementing the seam already declared in
  `literature_primary.LiteraturePrimary(fetcher=...)`, reached through the
  bounded `infra` CLI contract rather than ad-hoc shell.
- Object publication, transfer, retry and eviction behind that boundary.
- The repository-side adapter stays credential-free; the model-facing surface
  must remain `literature_primary(paper_id)`.
- Manifest population and the first upload of retained papers only under
  explicit infrastructure/storage authority, with the manifest row written from
  the verified artifact rather than authored by hand.

**Not this increment**

No repository consolidation or `infra/` merge, no `paper_id` awareness inside
`infra/`, and no widening of the model-facing tool surface.

**Exit criteria when implemented**

Given a `paper_id`, an authorized run obtains the canonical remote artifact
through `infra/` semantics, verifies it against the manifest checksum, populates
the disposable cache atomically, and returns the same structured result INC-004
already defines — with no credential ever entering model context.

## INC-005 — Introduce literature claim records for active questions

**Status (2026-10-03): partially built, and the reason is recorded here.**

Built: `improvements/taskrelation/research/literature_claims.py` plus
`literature/claims.jsonl` (17 records over the 7 papers the two live questions
turn on), the bounded `literature_read source=survey`, and the model-facing
`literature_query operation=paper_claims|claim`. The motivating failure — an
agent attributing a paper's `λ₂` grid to the per-paper card — is now expressed as
a positive record: the grid is recorded against
`MSSL_SPARSITY_ANALYSIS.md` §5.1 at `unverified_primary`, while the card records
only the selection procedure.

Enforced mechanically, which is the increment's real invariant:

- every claim names an artifact class and a **section that exists**;
- a quoted claim's text must be verbatim **inside that section** — not merely
  somewhere in the same file, which is what keeps one section from borrowing
  another's authority;
- `primary_verified` is rejected outright while no primary artifact is retained,
  so the strongest label is unavailable rather than merely discouraged;
- unknown keys are rejected, so a screening verdict, decision or authorization
  cannot be smuggled into a claim record;
- lookups are exact: no fuzzy or text matching, stable `paper_id#claim_id` refs.

Test surface: `tests/test_literature_claims.py` (22 cases) and the extended
`tests/test_literature_agent_assets.py`, both inside `make check`; the model-facing
adapter is exercised by `make literature-tools-check` (19 checks, needs bun). All
pass at this commit, including the negative case that matters most — a quote
lifted from the survey and attributed to the card is rejected as
`QUOTE_NOT_VERBATIM`.

Deliberately deferred, with the reason:

- **topic/research-question references and support/contradiction stance.** A
  stance has no meaning until a claim is bound to a question, and a topic
  registry is INC-010's business. Adding them now would repeat this repository's
  existing problem of untyped prose relations.
- **Study/finding/decision links.** The contextual verdict already has owners
  (`STUDIES.jsonl.cards`, each Study's `result.json`, `DECISIONS.md`); a claim
  record that carried a Study ID would become a second, divergent place to answer
  "was this paper accepted for Study X?".
- **extraction date/method fields.** The Git commit that adds or edits a line is
  the extraction record; a duplicated timestamp in the row would be a second
  provenance that can silently disagree with it.

Divergence from the text below to note: the increment is titled "for active
questions", but the registry binds claims to **papers only** and refuses
`study_id` as an unknown key. That is deliberate — the observed failure was
provenance, not question-scoping.

Reachability: verified 2026-10-03 from two independent directions. In a persisted
in-session run (`LitEvalRun2V3`, transcript-checked) the agent called
`literature_query operation=paper_claims`, read the survey through the new bounded
source, and reported the `λ2` grid at `survey-derived / unverified_primary` instead
of attributing it to the card. The same question re-run in a fresh
`omp -p --no-session` process reproduced every element: seven `claim_ref`s cited,
all seven present in `literature/claims.jsonl`, zero invented, and the code-side
half of the question deferred to the main session.

Provenance authority (hardened 2026-10-03). The Literature Agent previously held
`mcp__deja_deja`, and used recalled session text as evidence when the literature
tools were unavailable. That bypass is closed at the configuration boundary, not by
instruction: `<repo>/.omp/mcp.json` disables the deja MCP server for this project
(the agent frontmatter `tools:` list cannot gate MCP tools at all). The agent's
granted set is now exactly the four literature capabilities plus `yield`, and the
skill requires a *fail-closed* return — `evidence_status: unavailable`,
`REQUIRED_LITERATURE_TOOL_UNAVAILABLE` — instead of a synthesis when the evidence
surface is missing. Because a run's tool grant is a property of the session rather
than the agent, no evaluation is admissible until
`scripts/agents/literature_agent_transcript.py` proves from the run's own
transcript that the four tools were granted and called and that no recall tool was.
Procedure and root causes: *Hardening* and *Evaluation protocol* in
`LITERATURE_AGENT_V1_EVAL.md`.

Proviso, recorded as `INTERMITTENT_TOOL_GRANT_GAP` in `LITERATURE_AGENT_V1_EVAL.md`:
inside one session the literature grant was present at 15:20/15:22 and absent at
15:10, 15:11 and 15:29, with no repository-side difference between those spawns. The
agent cannot tell the difference — it either reports a blocker or answers from
memory. So a fresh isolated process is the reproducible path, and any evaluation
must prove from its own transcript that the literature tools were called before its
result counts.

## INC-006 — Connect existing cards and syntheses to structured provenance

**Goal**

Use the existing per-paper cards and topic syntheses as derived views over stable identities/claims without creating duplicate summaries.

**Why now**

The repository already has high-quality card and survey content. Replacement would destroy history and add no query capability.

**Exact scope**

- Add minimal identity/claim references when an existing card or synthesis is next revised.
- Define view rules: cards are per-paper derived knowledge; `literature_survey/` files are topic/theory syntheses; LT analyses are question-specific Study interpretations.
- Add consistency checks for references and stale/addendum markers.

**Files/components affected**

Only touched cards/syntheses, validators, and documentation. No mass rewrite.

**New invariant introduced**

Derived literature text names the structured evidence it relies on where available; missing structured provenance is explicit and does not invalidate untouched historical material.

**Migration required**

Incremental on touch. No normalization pass.

**Backward compatibility**

All existing paths and historical wording remain.

**Tests/verification**

Reference/link checks and a smoke reconstruction from claim → card/synthesis → paper.

**What is deliberately NOT done**

No new per-paper summary directory, no regeneration of historical Markdown, and no attempt to make every survey sentence structured.

**Exit criteria**

The next revised synthesis can be traced to claim IDs and papers without duplicating its prose in another store.

## INC-007 — Introduce a dedicated literature-review skill over narrow interfaces

**Goal**

Separate literature competence and mutation rules from the general experiment operator before creating an autonomous agent.

**Why now**

The existing `wav-literature` command is a useful procedure but depends on unrestricted file traversal and broad multi-registry mutation.

**Exact scope**

- Add a literature skill that uses identity/query/retrieval/assessment/claim interfaces.
- Define explicit ownership: literature discovery, retrieval, reading, claims, cards, literature syntheses, and literature lifecycle.
- Define explicit non-ownership: final decisions, Study promotion, backlog priority, experiment design/execution, compute/infrastructure policy, and interpretation of internal results as literature evidence.
- Keep discovery in this one skill; no discovery subagent.

**Files/components affected**

Agent skill/command assets and their validation tests. Existing command may become a thin workflow entry point.

**New invariant introduced**

Literature workflow mutation crosses only literature-owned interfaces; proposed research consequences are handed to main OMP with evidence references.

**Migration required**

None for historical records. Update workflow routing after behavioral parity is proven.

**Backward compatibility**

Main OMP can still perform literature work. Existing command behavior remains available during cutover.

**Tests/verification**

Agent-asset checks plus scenario tests for known-paper detection, contextual exclusion, claim recording, and refusal to make a binding research decision.

**What is deliberately NOT done**

No autonomous agent, unrestricted Bash, multiple literature subagents, or experiment authority.

**Exit criteria**

A main OMP session can invoke the literature skill for one bounded question and receive structured, auditable literature output without broad repository rereads.

## INC-008 — Add the bounded Literature Review Agent

**Goal**

Create the first specialized agent only after its information and authority seams have been exercised deterministically.

**Why now**

At this point the agent does not need to invent identity, storage, provenance, or mutation mechanics in context.

**Exact scope**

- Give the agent narrow search, query, retrieve/get, ingest/update, and claim-record capabilities backed by deterministic code.
- Allow mutation only of literature-owned catalog/assessment/claim/card/synthesis state under validation.
- Require handoff to main OMP for Study registration/promotion, backlog changes, decisions, experiment proposals, and any infrastructure action.
- Keep paper discovery as an agent capability, not a separate subagent.

**Files/components affected**

OMP agent definition/configuration, capability adapters, policy tests, and documentation.

**New invariant introduced**

The Literature Review Agent cannot mutate binding programme decisions, execute experiments, provision infrastructure, or reinterpret internal experimental findings as literature evidence.

**Migration required**

None. Historical state remains where it is.

**Backward compatibility**

Main OMP retains read access and final authority. Manual workflows remain possible.

**Tests/verification**

End-to-end sandbox cases for discovery, deduplication, retrieval, screening, claim extraction, synthesis, and forbidden mutations. Verify that model-facing calls contain no credentials.

**What is deliberately NOT done**

No broad research agent hierarchy, no autonomous experiment loop, no graph/vector database, and no separate discovery agent.

**Exit criteria**

The agent completes one bounded literature review, updates only literature-owned state, and produces an evidence-linked handoff that main OMP can accept or reject.

## INC-009 — Exercise real questions and measure query behavior

**Goal**

Validate the architecture against actual research questions before normalizing broader research state.

**Why now**

The correct interfaces and indexes should be chosen from observed query traffic, not hypothetical completeness.

**Exact scope**

Run several bounded questions representing: known-paper lookup, prior exclusion, topic claims, support/contradiction, and literature-to-internal-evidence handoff. Record files opened, registry hits, primary-paper reads, unresolved joins, and human corrections.

**Files/components affected**

Test fixtures/evaluation records and small fixes to existing literature interfaces. Do not add new stores unless a measured failure requires one.

**New invariant introduced**

A new abstraction must name the observed query/failure it solves.

**Migration required**

Only corrections discovered by evidence, with provenance.

**Backward compatibility**

Unchanged.

**Tests/verification**

Scenario replay with expected IDs, sources, and authority boundaries; compare context/file-read count against the current manual traversal.

**What is deliberately NOT done**

No performance benchmark theatre, broad backfill, or expansion into internal research normalization.

**Exit criteria**

The common literature queries are demonstrably answered through registry → selected artifacts → primary evidence, and remaining pain points are ranked by measured frequency/impact.

## INC-010 — Consider a broader research query index, not a canonical rewrite

**Goal**

Only if INC-009 demonstrates need, make study → finding → decision and unresolved-question queries cheap across internal and external evidence.

**Why now**

Current internal IDs and authorities are strong but joins are prose. Literature-first usage will reveal which joins matter enough to encode.

**Exact scope**

Build read-only adapters/indexes over existing Study, finding, failure, decision, proposal, authorization, and task-relation records. Prefer generated indexes and explicit references added on touch. Do not replace canonical Markdown or `STUDIES.jsonl` in the first pass.

**Files/components affected**

A broader research query module, generated/validated reference index, and consistency tests. Exact schema follows observed queries.

**New invariant introduced**

Every indexed edge points to an existing canonical entity and records which artifact asserted it; the index never outranks its source.

**Migration required**

Selective references for high-value current entities. Historical records remain readable without complete indexing.

**Backward compatibility**

Full. Existing authorities and human views remain.

**Tests/verification**

Queries such as “which Study produced F10?”, “which decisions cite LT-0002?”, and “what unresolved questions concern relation-estimate noise?” return canonical IDs and paths, with conflict reporting.

**What is deliberately NOT done**

No relational/graph/vector database, no event bus, no rewrite of `STATE.md`/`FINDINGS.md`/`DECISIONS.md`, and no repository reorganization.

**Exit criteria**

Measured cross-domain queries are cheap and evidence-linked, while canonical historical records stay in place.

# First Implementation Recommendation

Implement **INC-001 — Declare and catalog existing paper identity** first.

It has the highest leverage because every subsequent literature capability—deduplication, screening history, retrieval, claims, summaries, and agent tools—needs a stable paper identity. It has the lowest migration risk because the repository already uses the 23 card slugs as de facto IDs; the increment formalizes and validates that existing convention rather than renaming files, rewriting cards, changing Study semantics, or introducing a new storage system. Its immediate concrete payoff is the next discovery workflow: before spending search/review effort, a deterministic lookup can answer whether a paper is already known and return the canonical card.