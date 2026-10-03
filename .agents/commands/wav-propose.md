---
description: Turn a research goal into a reviewed, human-gated proposal and stop before any compute
---

# Propose a study

Run one Research Computer V1 cycle: reconcile the evidence, gather bounded
literature where the question needs it, have a design produced by the
read-only research designer, challenge it with the independent research
reviewer, persist the proposal, and emit a human approval request. Stop there.

This command never registers a study, creates or renews an authorization,
provisions or submits compute, or spends money. Execution is a separate,
human-gated increment.

Skills: wavcse-research-computer
Boundaries: writes-reports, no-commit, no-paid-compute

## Entry

Take the goal from the invocation. If it is not a research question this
programme can act on (a settled question, a scope change, an execution request),
say so and stop rather than manufacturing work.

## Steps

1. Reconcile the records listed in the skill's step 0, from the repository, not
   from memory. Report any record conflict instead of resolving it silently.
2. Frame the open question and its evidence basis; name the framework row or
   backlog entry.
3. If and only if the question is literature-bound, delegate to the
   `literature-reviewer` agent and carry its claim references forward.
4. Delegate the design to the `research-designer` agent, giving it the question,
   the reconciled evidence and any literature result.
5. Write `improvements/taskrelation/research/proposals/<ID>_<slug>.md` as
   `DRAFT` per the contract in `proposals/README.md`.
6. Set `REVIEW_REQUIRED` and delegate to the `research-reviewer` agent.
7. Reconcile the verdict: revise and re-review on `CHANGES_REQUIRED`, otherwise
   move to `READY_FOR_HUMAN`. Record the review in the proposal.
8. Validate: `python3 improvements/taskrelation/research/proposal_check.py check`.
9. Emit the approval request (skill format) and stop.

## Never

- approve, register, authorize, provision, submit, or spend;
- write findings, decisions, the registry, or the backlog;
- promote screening evidence to a confirmation claim;
- leave the cycle without either an approval request or a stated HARD_STOP.
