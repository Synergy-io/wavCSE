# wavCSE Research Computer V1 — audit and vertical-slice evaluation

**Evaluation date:** 2026-10-03

**Starting revision:** `8a0878231c6cff6acbe22481856549c572d9ad8d`

**Scope:** goal → evidence → proposal → independent review → revision → human gate → stop.

**Scientific fixture:** `tests/fixtures/DG-0008_design_input.md`; input/test data only.

No study was registered, no authorization was created, no scientific finding or
decision was changed, and no compute was provisioned or submitted.

## A. Architecture discovered before implementation

| Responsibility | Existing mechanism | Classification | V1 action |
| --- | --- | --- | --- |
| Main orchestration | Main OMP session; `wavcse-research-runner`; `/wav-cycle` | **ALREADY EXISTS** for full execution, but no proposal-only stopping loop | Add one small proposal-loop skill/command; no orchestrator agent |
| Literature evidence | `.omp/agents/literature-reviewer.md`, bounded literature tools, proven V1/V2 evaluation | **ALREADY EXISTS** | Unchanged; reused only when a question is literature-bound |
| Study-design rules | `wavcse-experiment-operator`, AGENTS invariants, benchmark protocol | **ALREADY EXISTS** | Autoload into a read-only designer |
| Design specialist | No project agent with a read-only design-only grant | **REQUIRED FOR V1** | Add `research-designer` |
| Scientific challenge | Bundled OMP `reviewer` is code-review scoped, not scientific-method review | **REQUIRED FOR V1** | Add `research-reviewer`; reuse OMP task transport |
| Delegation transport | OMP `task` → structured subagent result | **ALREADY EXISTS** | Reuse; no bus, queue, mailbox or RPC |
| Draft representation | `research/proposals/*.md` already stores unregistered pre-registrations | **ALREADY EXISTS**, but implicit and unvalidated | Keep markdown; add schema-1 contract + validator |
| Study registration | `STUDIES.jsonl` + `studies/<ID>/PLAN.md` | **ALREADY EXISTS** | Untouched; deliberately after human approval |
| Compute authority | `authorizations/<ID>.yaml`, strict backend consumption | **ALREADY EXISTS** | Untouched; distinct from proposal and registration |
| Human/autonomy boundary | `.agents/policies/autonomy.md` | **ALREADY EXISTS** | Consumed, not weakened or rewritten |
| Deterministic execution | `improvements/compute` → bounded `infra` CLI | **ALREADY EXISTS** | **DEFER**; V1 never crosses this boundary |
| Monitoring, Discord, recovery | Execution-side concerns | **DEFER** | Not built |
| Generic agent framework, event bus, DB, RAG | No V1 need | **NOT NEEDED** | Not built |

## B. Research Computer V1 architecture

```text
Human goal
    ↓
Main OMP session (orchestrator; wavcse-research-computer)
    ├── literature-reviewer  [only when literature-bound]
    ├── research-designer    [read-only design]
    └── research-reviewer    [read-only independent critique]
    ↓
proposals/<ID>_<slug>.md + proposal_check.py
    ↓
Human approval request
    ↓
STOP

Later, separately: human approval → registration → human authorization →
improvements/compute → infra CLI
```

The main session remains the orchestrator. A dedicated main-agent definition
would duplicate OMP itself; the missing component was an explicit, proposal-only
control-loop skill and entry command.

## C. Tool and authority boundary

| Role | Read authority | Write authority | Tool authority |
| --- | --- | --- | --- |
| Main orchestrator | Repository research state; specialist results | Schema-1 proposal and evaluation/approval report only in this V1 | Delegation; no automatic authorization; the `/wav-propose` contract has `no-paid-compute`, `no-commit` |
| Literature Agent | Retained literature surface only | Disposable primary cache only | `literature_resolve`, `literature_query`, `literature_read`, `literature_primary`; unchanged |
| Research Designer | Repository evidence + retained literature reads | None; returns text by `yield` | Exact allowlist: `read`, `grep`, `glob`, `find`, four literature tools; no `write`, `edit`, `bash`, `task`, web or compute |
| Research Reviewer | Repository evidence + proposal | None; returns critique by `yield` | Exact allowlist: `read`, `grep`, `glob`, `find`; no write, shell, delegation, literature mutation or compute |
| Proposal validator | Proposal files, registry IDs and study-directory presence | None | Deterministic Python; strict schema/status/cross-state checks |
| Future executor | Registered plan + committed authorization | Deterministic runtime state through the backend | `improvements/compute`; outside V1 |

Static tests fail if either specialist's tool allowlist widens. Prompt text is not
the only safety boundary.

## D. Proposal representation and lifecycle

Existing markdown under `proposals/` remains the durable representation. New
schema-1 proposals add strict frontmatter and exactly eleven body sections; old
TR-0012/TR-0013 drafts remain legacy and are not rewritten.

Lifecycle:

```text
DRAFT → REVIEW_REQUIRED → CHANGES_REQUESTED → REVIEW_REQUIRED …
      → READY_FOR_HUMAN → APPROVED | REJECTED
```

Four events remain independent:

1. proposal exists (`proposals/`);
2. human approves a direction;
3. study is registered (`STUDIES.jsonl` + `studies/<ID>/`);
4. compute is authorized (`authorizations/<ID>.yaml`).

`proposal_check.py` rejects a pre-decision proposal whose allocated Study ID is
already registered or has a study directory. `READY_FOR_HUMAN` requires
`review: PASS`; `APPROVED` additionally requires the human identity/time fields.

## E. In-process DG-0008 vertical slice

1. **Goal/evidence:** the main session treated the prior DG-0008 design as a
   fixture, reconciled F1/F3/F8/F9/F10, DEC-0010–0014, DG-0001/DG-0005,
   framework, protocol, registry and worker-environment records.
2. **Literature:** no new literature request. The question was empirical and the
   relevant evidence was already in project records; calling the Literature
   Agent would add no evidence.
3. **Designer:** `research-designer` returned an initial proposal and verified
   that DG-0008 was absent from `STUDIES.jsonl`.
4. **Reviewer:** first usable independent review returned
   `CHANGES_REQUIRED` with eight blockers: causal overreach, undefined H2
   control, directionality mismatch, test-based Stage-2 selection, unsupported
   SESOI/screen threshold, fold pseudo-replication risk, mixed scientific
   questions, and incomplete cost/run matrix.
5. **Revision:** conservative rewrite narrowed the estimand to two preselected
   ER-target behavioral cells, made Stage 1 implementation-only, made seeds the
   independent unit, removed the unrelated 30-epoch question, demoted gradient
   geometry to exploratory evidence and exposed the full 200-job Stage-2 matrix.
6. **Re-review:** `CHANGES_REQUIRED` remained, correctly. Nine scope/staging
   defects were cleared; remaining pre-decision blockers are the declarative
   opportunity schedule, diagnostic SESOI/power, optional estimator decision,
   and a bounded Stage-1 ceiling/lawful corpus prerequisite.
7. **Main reconciliation:** fixed a lifecycle bug found on re-review: corpus
   restoration, implementation proof and measured benchmarking are **post**
   approval/registration/Stage-1 authorization, not pre-registration work.
8. **Validator:** schema-1 proposal passed; historical proposals reported as
   legacy.

The proposal remains `CHANGES_REQUESTED`. This is the correct V1 outcome: the
loop did not launder incomplete test data into a human-ready or executable plan.

## F. Fresh-process evaluation

Harness: fresh `omp -p --no-session` process, explicit parent tool surface
`read,grep,glob,find,task,wait`; no parent write, edit, shell or execution tool.
The process read the fixture/current proposal/state, delegated one read-only
assessment to each specialist, reconciled them and printed an approval request.

Observed sequence:

```text
human-like DG-0008 goal
→ repository evidence reconciled
→ literature reuse/not-needed decision
→ research-designer: conservative revision is structurally correct
→ research-reviewer: CHANGES_REQUIRED
→ main session reconciles pre- vs post-approval prerequisites
→ REQUEST CHANGES approval request
→ STOP
```

Fresh-process checks:

- designer/reviewer remained inside their read-only grants;
- no literature call was made because the question was not literature-bound;
- proposal stayed `CHANGES_REQUESTED` / `CHANGES_REQUIRED`;
- DG-0008 had no registry entry, study directory or authorization;
- no proposal transition, worker, volume, job, benchmark or experiment occurred;
- screening/validation and confirmation stayed separate;
- cost uncertainty was surfaced as a blocker, not guessed;
- scientific success/stop regions existed structurally but remained disabled
  until a justified numeric SESOI/power calculation exists;
- human decisions were explicit.

OMP startup attempted to initialize a configured RunPod MCP connection and
received HTTP 401; no RunPod tools became available and no RunPod operation or
resource mutation occurred. This is startup discovery noise, not execution, but
it is recorded rather than hidden.

## G. Approval request produced by V1

```text
Proposed study: DG-0008 (draft only)
Why: Test whether F8's ER←SI harmful residual, but not ER←KS, survives exact
     within-cell opportunity control across independent seeds and LOSO.
Evidence: F1, F3, F8-F10; FRAMEWORK R4/R6/R7; DEC-0013; DEC-0014;
          studies/DG-0001/; task_relations/{loso_transfer,optimization_control}.json
First stage: After registration and a human Stage-1 envelope, implementation /
             exposure validation only: 5 five-epoch jobs plus 4 optional
             diagnostic suites; no held-out test endpoint, no scientific verdict.
Later stage: Separately authorized Stage 2: 2 cells × 2 arms × 5 seeds × 10 LOSO
             folds = 200 jobs; 40 optional suites only if retained.
Cost: Stage-1 conservative ceiling and Stage-2 measured estimate unresolved;
      unknown cost is a stop, not an estimate.
Stop condition: Stage 1 stops on any validity mismatch. Stage 2 runs the complete
                valid matrix, reports H1/H2/H3 or INCONCLUSIVE under the frozen
                SESOI interval rules, then stops regardless of outcome.
Reviewer: CHANGES_REQUIRED.
Human decisions required: whether to reopen; accept the behavioral-only scope;
                          approve the exact-opportunity intervention; choose the
                          SESOI after seed-level precision work; retain/delete
                          exploratory diagnostics; later grant Stage 1 and Stage 2
                          separately.
Action: REQUEST CHANGES
Proposal: proposals/DG-0008_exact_matched_directed_transfer.md
          (not registered; no compute authorized)
```

## H. Deterministic verification

- New focused suites: 26 tests, pass.
- `make agents-check`: 5 skills, 8 commands, 21 files scanned, no violations.
- `make check`: compute suite 276 tests pass; research suite 452 tests pass.
- `python -m improvements.run_improvements --help`: pass.
- `git diff --check`: pass.
- `proposal_check.py check`: DG-0008 OK; two historical proposals reported legacy.

No GPU test or network compute was required.

## I. Weaknesses observed

1. Model-role routing was not reliable: `@slow` exhausted quota and its fallback
   returned `402 Insufficient Balance`. Both V1 specialists were pinned to the
   available repository model; role independence still comes from isolated
   sessions, different prompts and non-overlapping grants, not model diversity.
2. The proposal contract validates structure and lifecycle, not scientific
   completeness; independent review remains necessary.
3. A fresh OMP process initializes configured MCP servers before the prompt's
   tool-boundary logic; failed RunPod startup discovery appeared even though no
   RunPod tool was enabled for the task.
4. DG-0008 exposed a real missing design-closure step: deterministic opportunity
   schedules, SESOI/power and a bounded Stage-1 ceiling must exist before
   `READY_FOR_HUMAN`, while implementation proof and benchmarking must remain
   after registration/authorization.

## J. Exact next increment (not started)

Add one **pre-registration design-closure pass**, still controller-only and
zero-compute, that can attach deterministic no-training opportunity manifests,
a seed-level SESOI/precision calculation, an explicit keep/delete decision for
optional diagnostics, and a conservative Stage-1 ceiling to a
`CHANGES_REQUESTED` proposal; then re-run the independent reviewer. Only a
reviewer `PASS` may move it to `READY_FOR_HUMAN`.

Do not add execution, monitoring, Discord, repair or GPU control until a real
proposal reaches `READY_FOR_HUMAN` and the human separately approves,
registers and authorizes it.
