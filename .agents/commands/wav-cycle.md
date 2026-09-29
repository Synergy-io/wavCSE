---
description: Run one autonomous Task Relation Learning research cycle end to end
---

# Autonomous Research Cycle

Execute exactly ONE research cycle for the wavCSE Task Relation Learning
programme. This merges the two legacy cycle commands into the same
reconcile-then-act workflow, scoped to at most one Study's worth of work.

Skills: wavcse-research-runner, wavcse-experiment-operator
Boundaries: mutates-research-state, may-provision-compute, may-commit

## Objective

Advance the programme's scientific question, not merely a score: which
relation-learning mechanisms transfer across KS, SI and ER without material
negative transfer, and the repository, not the conversation, is the memory.

## Steps

1. Reconcile state before acting. Read
   `improvements/taskrelation/research/OBJECTIVE.md`, `improvements/taskrelation/research/STATE.md`,
   `improvements/taskrelation/research/FINDINGS.md`, `improvements/taskrelation/research/DECISIONS.md`,
   `improvements/taskrelation/research/FAILURES.md`, `improvements/taskrelation/research/BACKLOG.md`,
   `improvements/taskrelation/research/STUDIES.jsonl`, `improvements/taskrelation/research/FRAMEWORK.md`
   and `improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md`; then reconcile
   runtime state through the project's runtime tooling (the `infra` CLI). Treat a
   disagreement between a record and runtime reality as unresolved until you know
   which is stale.
2. Recover interrupted work first: if a Study's `STUDIES.jsonl` status is not terminal
   (`complete` / `confirmed` / `rejected`), or it is waiting on jobs, analysis or LOSO, continue
   it; a new cycle is not a reason to open a new Study.
3. Otherwise resolve a pending verdict before exploring: a screen verdict of `PROMISING`
   (`VARIANT_BENCHMARK_PROTOCOL.md` §3) must be confirmed or explicitly retired, not left open
   while fresh mechanisms are explored.
4. Otherwise choose the highest-information READY question in
   `improvements/taskrelation/research/BACKLOG.md`, stating the current evidence,
   the uncertainty, one falsifiable hypothesis, a competing explanation, the
   falsification condition and the expected information gain.
5. Create at most ONE new Study. Register it in
   `improvements/taskrelation/research/STUDIES.jsonl` and write
   `improvements/taskrelation/research/studies/<STUDY_ID>/PLAN.md` before touching
   model code, with the question, hypothesis, independent variable, matched control,
   protocol, diagnostics, decision rule, compute estimate and allocation.
6. Prefer analysis, existing checkpoints or a config change over new architecture
   code; take the smallest intervention that can falsify the hypothesis. Never
   invent a mechanism: every variant needs a verified published source and a
   faithfulness check.
7. Commit the implementation and configuration BEFORE confirmation runs and record
   that commit in the Study folder and `improvements/taskrelation/research/STUDIES.jsonl`.
8. Authorization gate: do not provision paid compute until the plan is approved and
   the spend is authorized under current policy. If authorization or the cost policy
   is unclear, stop and escalate to the human instead of provisioning.
9. Run the cycle's runs (candidate plus matched control, multiple seeds, or LOSO
   folds) under one protocol, holding upstream representation, task set, splits,
   pooling, layer set, preprocessing, optimizer, epoch budget, checkpoint policy and
   seed treatment fixed. Never change scientific semantics to save time or money.
10. Analyse every completed run per task (KS, SI, ER, aggregate), including negative
    transfer, validation behaviour, seed variability, and the relation, gradient and
    transfer diagnostics; state which alternative explanation survives.
11. Update only the persistent state the evidence belongs to:
    `improvements/taskrelation/research/STATE.md`, `improvements/taskrelation/research/FINDINGS.md`,
    `improvements/taskrelation/research/FAILURES.md`, `improvements/taskrelation/research/DECISIONS.md`,
    `improvements/taskrelation/research/BACKLOG.md`, `improvements/taskrelation/research/STUDIES.jsonl`
    and the Study folder (`PLAN.md`, `NOTE.md`, `result.json`, `analysis.md`). Keep the
    research state document a concise restart point.
12. Stop paid compute that is no longer needed and leave the next action explicit for
    a fresh session.

## Never

- promote a single-seed result: screening yields only REJECTED, INCONCLUSIVE or
  PROMISING; CONFIRMED requires the matched multi-seed protocol, and any ER claim
  additionally requires speaker-independent LOSO (F1, F3);
- run generic hyperparameter search, or invent architectures, to keep busy;
- change model, checkpoint, pooling, layer set, dataset membership, splits,
  preprocessing, label mapping or precision for speed or cost;
- leave this cycle's jobs unfinished for a later cycle, except a documented
  long-running continuation recorded as ACTIVE in the research state document.
