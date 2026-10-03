# Research proposals

A proposal is the research-computer's product: a designed, independently
reviewed study offered to the human for a decision. It is **not** a study and
**not** an authorization. Nothing here is executable.

- A proposal exists here.
- A *registered* study exists in `../STUDIES.jsonl` and `../studies/<ID>/`.
- A *compute authorization* exists in `../authorizations/<ID>.yaml`.

These are three different events. Creating a proposal creates neither of the
other two, and a proposal may be rejected or deferred without any of them. The
deterministic validator `../proposal_check.py` (`python3 proposal_check.py check`)
enforces this boundary: while a proposal is pre-decision, its
`allocated_study_id` must not be registered.

## File contract

One markdown file per proposal: `<ALLOCATED-ID>_<slug>.md`, with a YAML
frontmatter block and the body sections below. Files that predate this contract
(no `proposal_schema`) are reported as `legacy` and are not validated.

### Frontmatter

```yaml
---
proposal_schema: 1
proposal_id: DP-0008              # proposal-local id, not a study id
allocated_study_id: DG-0008       # reserved for drafting; not a registration
status: DRAFT
created_at: 2026-10-03T00:00:00+00:00
review: ""                        # "" | PASS | CHANGES_REQUIRED
reviewed_by: ""                   # research-reviewer | ""
approved_by: ""                   # filled only on APPROVED, by the human
approved_at: ""
superseded_by: ""
---
```

Strict keys: an unknown key is rejected, because a misspelled limit would
silently not bind. A proposal has **no** `authorization`, `budget` or `scope`
key — it cannot claim spend authority.

### Status

`DRAFT` → `REVIEW_REQUIRED` → `CHANGES_REQUESTED` → `REVIEW_REQUIRED` …
→ `READY_FOR_HUMAN` → `APPROVED` | `REJECTED`. `SUPERSEDED` closes a proposal
replaced by another.

- `READY_FOR_HUMAN` and `APPROVED` require `review: PASS`.
- `APPROVED` requires `approved_by` and `approved_at` — the human's act.
- Pre-decision proposals must have no `STUDIES.jsonl` entry and no
  `studies/<allocated_study_id>/` directory.

### Body sections (exact `##` headings)

`## Question`, `## Evidence basis`, `## Hypotheses`, `## Proposed study`,
`## Discriminating measurements`, `## Success and stop conditions`,
`## Expected compute`, `## Risks and confounds`, `## Repository effects`,
`## Human decisions required`, `## Review`.

Keep canonical numbers in `FINDINGS.md`; a proposal cites them (`F8`, `F10`,
`DEC-0010`, a `paper_id#claim_id`), it does not restate them. Do not duplicate
what `STUDIES.jsonl`, `studies/<ID>/PLAN.md` or an authorization already
represent — those exist only once the human approves and the study is registered.

## Template

`python3 ../proposal_check.py template` prints a conforming skeleton.
