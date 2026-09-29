---
description: Advance the research programme autonomously until a decision genuinely needs the researcher
---

# Autonomous research cycle

Advance the Task Relation Learning programme as far as the evidence and the
current authorization allow, then stop at the first thing that genuinely needs
the researcher. One invocation may open one Study, commit it, acquire compute,
run its pre-registered stages, verify the evidence, update the records and stop
paying — or it may do none of that, because the next action is a human decision
or a hard stop.

Skills: wavcse-research-runner, wavcse-experiment-operator
Boundaries: mutates-research-state, may-provision-compute, may-commit

## ARC v1 integration gate

Before a paid worker request, three runtime contracts must hold, and each is now
implemented with its own regression tests. Confirm them rather than assume them:

- **the exact commit is available to workers** — `python -m improvements.compute
  preflight --plan <PLAN>` answers against the remote a disposable worker clones, and
  `worker-ensure` refuses before any billable request. Publication is a developer action:
  when it is required, report it, never push and never substitute another commit.
- **a verified input reaches the loader** — a plan that loads embeddings declares
  `embedding_layout`, the job wrapper builds the loader's tree from the declared,
  digest-verified artifacts and names it explicitly, and no other root is ever used. A
  plan without it, or a job whose layout cannot be prepared, must not run.
- **stored bytes are validated as this run's evidence** — `collect` reads the stored
  manifest and metrics back and validates identity, completeness and structure before an
  entry becomes COLLECTED. Bytes alone are never evidence.
- **cleanup survives the controller** — `reap --execute` ends paid compute whose lease
  deadline passed, with no session, runner or agent alive. Install it once per controller
  with `reap-install --install` (and enable it deliberately, with `--enable`).

If any of these is missing or fails, report a HARD_STOP; a successful unit suite alone
does not establish a runtime contract.

## Objective

Advance the programme's scientific question, not merely a score, and treat the
repository — never the conversation — as the memory.

## When to stop

Exactly three terminations:

1. a **human decision** the policy classifies as HUMAN_DECISION;
2. a **hard stop** the policy classifies as HARD_STOP;
3. the authorized scope is **exhausted**: the Study is closed and no READY next
   action has a protocol-determined path.

Anything else continues. Switching to the infrastructure checkout is not a
termination; neither is finishing one stage, nor a launched job.

## Steps

0. Reconcile. Read `improvements/taskrelation/research/OBJECTIVE.md`,
   `STATE.md`, `FINDINGS.md`, `DECISIONS.md`, `FAILURES.md`, `BACKLOG.md`,
   `STUDIES.jsonl`, `FRAMEWORK.md`, `VARIANT_BENCHMARK_PROTOCOL.md`, the active
   Study folder, and `.agents/policies/autonomy.md`. Then reconcile runtime
   reality with `python -m improvements.compute status --scope <SCOPE>`; that
   picture is authoritative for compute, spend, leases and jobs. Reconcile
   provider state again at the end. A disagreement between a record and runtime
   reality is unresolved until you know which is stale.
1. Resume first. If a stage has jobs in flight, a lease is live, or a run ledger
   entry is unresolved, continue *that* before starting anything else. A new
   cycle is never a reason to open a new Study or to leave compute running.
2. Resolve mechanical inconsistencies the policy classifies as AUTONOMOUS, and
   record what was reconciled. Anything that would decide a question between two
   authoritative records is a human decision, not a repair.
3. Repair an unambiguous implementation defect: reproduce it, fix it with a
   regression test, commit it, and re-run the same science. A failing
   implementation is never evidence.
4. Derive the next pre-registered action. If it needs a hypothesis, protocol,
   semantic or authorization change, write the human gate (below) and stop.
5. Register at most ONE new Study: `STUDIES.jsonl`, `PLAN.md`, configs, the
   compute plan under `studies/<ID>/compute/`, and the envelope the human
   granted for it. No code before the plan exists.
6. Commit the exact implementation, configuration and compute plan. A recorded
   job is generated only from a commit-clean tree.
7. Establish artifact readiness from verified canonical bytes — digest identity,
   never a filename or a size — and declare the required inputs.
8. Check the envelope before spending:
   `python -m improvements.compute envelope-check --scope <SCOPE> --action create-worker`,
   and confirm publication with `preflight --plan <PLAN>`; an unavailable commit is a
   human gate, not a reason to push.
   Within the envelope, ensure the worker (`worker-ensure`), submit the stage
   (`advance --stage screen|confirm`), monitor by bounded transitions, and collect
   (`collect`) so required outputs are verified through the control plane's read-back and
   their content validated as this run's evidence rather than a log line.
9. Analyse every completed run per task (KS, SI, ER, aggregate), including
   negative transfer, validation behaviour, seed variability, and the relation,
   gradient and transfer diagnostics; state which alternative explanation
   survives. A completed run with a negative outcome is evidence: never retry
   it and never repair it.
10. Update only the persistent state the evidence belongs to: `STATE.md`,
    `FINDINGS.md`, `FAILURES.md`, `DECISIONS.md`, `BACKLOG.md`, `STUDIES.jsonl`
    and the Study folder. Keep `STATE.md` a concise restart point; record the
    exact commits and the MLflow run names, never worker or job identifiers.
11. Stop or destroy scope compute that is no longer needed
    (`python -m improvements.compute finish`, then `sweep --scope <SCOPE>`), and
    confirm the sweep is clean. Whatever this cycle leaves behind, the controller's
    `reap` timer must be able to end on its own: check it is installed and its last pass
    is clean.
12. Derive the next action and continue at step 3 unless it is a termination.

## Loop bounds

- at most one new Study per invocation;
- at most the envelope's concurrent workers, and never more spend than it grants;
- bounded retries only, with every attempt counted in durable state;
- no re-entry into a step abandoned for a hard stop, without new evidence;
- never invent work to keep the loop alive.

## Human gate

Write it into the Study's `NOTE.md` as `## Human gate`, add it to `STATE.md`
"Current Pending Work", and set `escalated_to_human` in `STUDIES.jsonl`:

    HUMAN_DECISION_REQUIRED
    Scope / Study / class:
    Blocking fact:
    Why autonomous continuation is forbidden: (the governing clause)
    Option A: scientific consequence / compute consequence
    Option B: scientific consequence / compute consequence
    Current paid resources and estimated spend / remaining authorization:

Never ask a question the policy already answers.

## Never

- promote a single-seed result: screening yields only REJECTED, INCONCLUSIVE or
  PROMISING; CONFIRMED requires the matched multi-seed protocol, and any ER claim
  additionally requires speaker-independent LOSO (F1, F3);
- run generic hyperparameter search, or invent architectures, to keep busy;
- change model, checkpoint, pooling, layer set, dataset membership, splits,
  preprocessing, label mapping or precision for speed or cost — including after
  an out-of-memory failure;
- widen, renew or invent a compute authorization;
- leave this cycle's jobs unfinished for a later cycle, except a documented
  long-running continuation recorded as active in the research state document;
- report an infrastructure failure as falsification, or a falsification as a
  defect to repair.
