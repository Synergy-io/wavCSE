---
name: wavcse-execution-plane
description: Orchestrate bounded approved-proposal preflight between Research Executor and Infrastructure Engineer, stopping before unauthorized compute.
---

# wavCSE Execution Plane V1

This skill begins only after proposal status `APPROVED`. It does not redesign the
proposal lifecycle and does not imply Study registration or compute authority.
The normative architecture, state machine and schema live in
`improvements/taskrelation/research/execution/README.md`.

## Main OMP procedure

1. Reconcile the approved proposal, registration, Study plan, compute plan,
   authorization, git state and deterministic infra status.
2. Give `research-executor` only the approved proposal reference and durable
   execution directory. It reconstructs science and produces a sealed workload.
3. Give a fresh `infrastructure-engineer` only workload/negotiation references.
   Do not paste the Executor transcript.
4. Apply the assessment with `execution_contract.apply_assessment`.
5. For `REQUEST_IMPLEMENTATION_CHANGE`, return the assessment reference directly
   to the Executor. Validate the revised workload with `validate_revision`, then
   send its reference to Infra. At most three assessment rounds.
6. Stop and escalate on scientific change, authority/budget/policy decision,
   blocker, persistent disagreement or convergence limit.
7. `ACCEPT` means preflight accepted, not authorized execution. Live execution
   still requires registered Study, committed plans/code, committed envelope and
   deterministic backend checks.

## Candidate changes

Implementation arrives from a specialist as a candidate, never as a direct edit
to the canonical checkout. Main OMP spawns `research-executor` /
`infrastructure-engineer` with an isolated workspace and `task.isolation.apply`
is false, so the candidate is returned as an `omp/task/<id>` branch or patch.
Before integrating anything,
Main OMP runs the deterministic candidate gate
(`python3 scripts/agents/candidate_gate.py check --repo . --baseline <commit> …`)
and, only after it accepts, the baseline-defined validation (`make check`). A
candidate that edits the gate, its policy, validation, test discovery, an agent
definition or an existing test is rejected by reason code; it never judges
itself. Main OMP remains the trusted integration boundary and the residual
authority this does not sandbox.

## Authority

Main OMP persists contract artifacts and routes only escalation/final state. It
does not approve science, create or widen an authorization, or reinterpret an
invariant. The specialists talk through sealed JSON references, not remembered
conversation.

For EXEC-PLANE-V1 / INC-021, stop after preflight. Do not provision, submit,
collect results, analyse findings or build a Result Analyst.
