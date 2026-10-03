---
name: wavcse-research-computer
description: Run the wavCSE research-computer loop — a human goal to evidence, a designed proposal, independent review and an approval request, stopping at the human gate before any compute.
---

# wavCSE Research Computer V1

One control loop from a human research goal to a **reviewed, human-gated
proposal**, then stop. V1 ends at the human gate: it never authorizes a study,
provisions a worker, submits a job or spends money. Execution, monitoring and
recovery are later increments with their own contracts.

This skill wires existing components; it does not replace them. Study-design
content is `wavcse-experiment-operator`, cycle execution is `wavcse-research-runner`
and `.agents/commands/wav-cycle.md`, literature evidence is the read-only
`literature-reviewer` agent, and spend authority is
`improvements/taskrelation/research/authorizations/` consumed by
`improvements/compute`. The autonomy classification is `.agents/policies/autonomy.md`.

## Roles and authority

| Role | Agent | Reads | Writes | Never |
|---|---|---|---|---|
| Orchestrator | main session | all research state, git, runtime status | a DRAFT proposal file under `proposals/`, the approval request, this cycle's report | grant human authority; register a study; create an authorization; provision compute |
| Literature | `literature-reviewer` | retained literature corpus only | disposable primary cache | touch research state; decide |
| Designer | `research-designer` | repo research evidence + literature tools | nothing (returns proposal text) | write, commit, authorize, provision |
| Reviewer | `research-reviewer` | repo research evidence + the proposal | nothing (returns a verdict) | write, execute, approve |
| Validator | `proposal_check.py` | `proposals/`, `STUDIES.jsonl`, `studies/` | nothing | invent state |

The orchestrator may autonomously inspect evidence, delegate literature work,
generate hypotheses, design a study, critique and revise it, estimate cost and
prepare an approval request. It may **not** approve a research direction, create
an authorization, provision a worker or submit a job — those stay human decisions
per the autonomy policy. Tool allowlists on the two specialist agents are the
deterministic boundary; the prompt is not the only control.

## The loop (one cycle)

0. **Reconcile.** Read `OBJECTIVE.md`, `STATE.md`, `FINDINGS.md`, `FAILURES.md`,
   `DECISIONS.md`, `BACKLOG.md`, `STUDIES.jsonl`, `FRAMEWORK.md`,
   `VARIANT_BENCHMARK_PROTOCOL.md`, the open `proposals/`, the
   `authorizations/` that exist, and `git rev-parse HEAD`. Never trust
   conversation memory.
1. **Frame the question.** State the open question in the programme's own terms,
   the framework row or backlog entry it acts on, and the evidence that leaves it
   open. An already-settled question is not a cycle.
2. **Gather literature evidence** only when the question is literature-bound:
   delegate to `literature-reviewer` and carry its `paper_id#claim_id` references
   forward. Reuse an existing investigation before commissioning a new one.
3. **Design.** Delegate to `research-designer` with the question, the reconciled
   evidence and any literature result. It returns the proposal body.
4. **Persist the draft.** Write `proposals/<STUDY-ID>_<slug>.md` from the
   proposal contract below with `status: DRAFT`.
5. **Review.** Set `status: REVIEW_REQUIRED` and delegate to
   `research-reviewer`. It returns a verdict and findings; record them in the
   proposal's `## Review` section.
6. **Converge.** Apply the bounded rule in *Bounded convergence loop* below.
   `CHANGES_REQUIRED` is not by itself a reason to return control to the human:
   the main session owns ordinary convergence between the designer and the
   reviewer.
7. **Validate** with `python3 improvements/taskrelation/research/proposal_check.py
   check` and fix any violation before continuing.
8. **Approval request** (below) and **stop**. Never proceed to registration or
   compute in this cycle.

## Bounded convergence loop

The main session — not the human — owns ordinary convergence between the designer
and the reviewer. After every reviewer verdict it classifies the result with
`improvements/taskrelation/research/convergence.py` (`classify_review_outcome`,
`MAX_REVISION_CYCLES`) and acts on exactly one of:

- `PASS` → **READY_FOR_HUMAN** (set `review: PASS`, `reviewed_by:
  research-reviewer`); validate; emit the approval request; yield.
- `CHANGES_REQUIRED`, **correctable within the current scope and no human
  decision required** → revise (delegate to `research-designer` with the
  reviewer's `questions_for_designer`), set `status: REVIEW_REQUIRED`, re-review,
  increment the revision-cycle counter, and loop. The human is not involved.
- `CHANGES_REQUIRED`, **a genuine human decision required** (a research-direction
  choice, a human-fixed decision, or a protocol/resource choice classified as
  human by `.agents/policies/autonomy.md`) → set `status: CHANGES_REQUESTED` and
  yield the numbered decisions to the human.
- `CHANGES_REQUIRED`, **unavailable evidence or capability blocks progress**
  (missing record, destroyed corpus, licence-gated data, or a capability outside
  the orchestrator's grants) → set `status: CHANGES_REQUESTED` and yield a blocker.
- **Non-convergence / unresolved disagreement:** after `MAX_REVISION_CYCLES = 3`
  revision cycles without a `PASS`, or when the reviewer's findings and the
  designer's response cannot be reconciled from evidence, set `status:
  CHANGES_REQUESTED` and yield the specialist disagreement. Never start a fourth
  cycle.

The loop may **never**: broaden the scientific question silently; change a
human-fixed decision; register a study; create or widen an authorization;
provision or submit compute; execute an experiment; modify `FINDINGS.md`,
`DECISIONS.md`, `FAILURES.md`, `BACKLOG.md` or `STUDIES.jsonl`; or bypass a
reviewer `PASS` to reach `READY_FOR_HUMAN`.

## Proposal object

The durable representation is one markdown file in
`improvements/taskrelation/research/proposals/`, contract in that directory's
`README.md`, validated by `proposal_check.py`. The proposal describes a study; it
is **not** the study. It must not restate canonical numbers that live in
`FINDINGS.md` — it cites them (`F8`, `F10`, `DEC-0010`, a `paper_id#claim_id`).

Required: `proposal_schema: 1`, a mechanically valid status, and the body
sections named in `proposals/README.md` (question, evidence basis, hypotheses,
proposed study, discriminating measurements, success and stop conditions,
expected compute, risks and confounds, repository effects, human decisions
required, review). Screening and confirmation must be separate, pre-registered
stages with their own cost and success/stop criteria.

## Lifecycle — four distinct events

```
DRAFT → REVIEW_REQUIRED → { CHANGES_REQUESTED → REVIEW_REQUIRED } → READY_FOR_HUMAN → APPROVED | REJECTED
```

`APPROVED` is the human accepting a direction. It is **not** registration
(`STUDIES.jsonl` + `studies/<ID>/`) and **not** authorization
(`authorizations/<ID>.yaml`), and **not** execution. A proposal whose allocated
study ID is already registered is a state error the validator rejects; a
`READY_FOR_HUMAN` or `DRAFT` proposal must have no registration and no envelope.

## Human approval request

The cycle's only external product. Compact, evidence-linked, and explicit about
what the human is deciding:

    Proposed study: <allocated ID>
    Why: <the open question, one line>
    Evidence: <F-ids / claim refs / study paths>
    First stage: <screening, what it measures, cost>
    Later stages: <confirmation / LOSO, cost>
    Stop condition: <pre-registered>
    Reviewer: <PASS | CHANGES_REQUIRED + one-line reason>
    Human decisions required: <numbered>
    Action: APPROVE / REJECT / REQUEST CHANGES
    Proposal: proposals/<file>   (not registered, no compute authorized)

Cost is surfaced before approval; success and stop criteria exist before
execution; screening is never presented as confirmation.

## Never

- register a study, create or widen an authorization, provision or submit
  compute, or spend money;
- write `FINDINGS.md`, `FAILURES.md`, `DECISIONS.md`, `BACKLOG.md` or
  `STUDIES.jsonl`;
- let a proposal claim authorization or a registered state;
- turn a `READY_FOR_HUMAN` proposal into execution, or treat silence as approval;
- yield on a correctable `CHANGES_REQUIRED` that needs no human decision and is
  not blocked by missing evidence or capability;
- let the designer or reviewer write, commit, authorize or execute anything.
