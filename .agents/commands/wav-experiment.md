---
description: Execute one already-approved Task Relation Learning study from its registered plan
---

# Approved Study Execution

Execute ONE Study that is already registered with an approved plan, under the
variant benchmark contract in
`improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md`. This command
is not the question chooser: the autonomous cycle (`/wav-cycle`) selects and
registers the Study; this command carries it out.

Skills: wavcse-experiment-operator, wavcse-execution-plane
Boundaries: mutates-research-state, may-provision-compute

## Entry gates: refuse to start when unmet

1. `improvements/taskrelation/research/studies/<STUDY_ID>/PLAN.md` exists and the
   Study is registered in `improvements/taskrelation/research/STUDIES.jsonl`.
2. The plan's eligibility gates are met: a verified published source with a
   faithfulness check, a matched control, a stated evaluation protocol, and the
   fixed conditions of the variant benchmark contract.
3. The plan's compute allocation is authorized by a committed envelope under
   `improvements/taskrelation/research/authorizations/`, and no other Study of
   this programme already holds the required compute.
4. A sealed Execution Plane preflight has reached `PREFLIGHT_ACCEPTED` for the
   exact approved proposal and exact workload revision being executed. The
   assessment grants no authority; gates 1–3 still apply independently.

If the plan, authority or accepted preflight is missing, or any gate fails: stop
and report the exact unmet gate. Never substitute a different experiment, choose
a study yourself, or lower a gate to make progress. If the workload needs an
implementation that is not yet committed, hand back to the Research Executor;
this command does not commit.

## Steps

1. Re-read the plan and the note contract; confirm the independent variable is
   isolated and the configs are explicit and reproducible.
2. Confirm the fixed conditions are respected: upstream representation, task
   set, splits, pooling, layer set, epochs, optimizer, sampling and checkpoint
   policy identical to the control arms. Never adjust them.
3. Acquire compute through the backend, never by hand:
   `python -m improvements.compute worker-ensure --scope <STUDY_ID> --plan <plan>`
   provisions or reuses a worker strictly inside the committed envelope, and
   `advance` submits the stage's exact-commit jobs. At most the envelope's
   concurrent workers, never a hand-written `infra` invocation, and never a
   scientific parameter changed to fit the hardware.
4. Run exactly what the plan specifies: screening (one explicit seed) or
   confirmation (matched seeds 0-4), together with the plan's matched control
   arms. Any ER performance claim additionally requires speaker-independent
   LOSO (F3).
5. Record provenance for every run: Study ID, stage, method, task set,
   representation, seed, commit and status, in the execution ledger and in the
   Study's `NOTE.md`. A run without provenance is not a result.
6. Verify each launch started and progressed beyond initialization before
   treating it as running. Recover infrastructure failures by repairing and
   rerunning the same scientific configuration and recording the failure; a
   retry is not a new Study.
7. Analyse per task (KS, SI, ER, aggregate), including negative transfer,
   validation behaviour, seed variability where available, and the relation,
   gradient and transfer diagnostics the plan names. Never reduce the result to
   an aggregate score.
8. Decide the Study outcome as the evidence allows: REJECTED, INCONCLUSIVE,
   PROMISING (screening) or CONFIRMED (confirmation protocol satisfied). Never
   promote a single-seed result.
9. Update the Study directory (`PLAN.md` status, `NOTE.md`, `result.json`,
   `analysis.md`) and only the persistent state the evidence belongs to:
   `improvements/taskrelation/research/STUDIES.jsonl`,
   `improvements/taskrelation/research/STATE.md`, and
   `improvements/taskrelation/research/FINDINGS.md` when a durable finding is
   justified. Keep `improvements/taskrelation/research/STATE.md` a concise
   restart document.
10. Stop paid compute no longer needed
    (`python -m improvements.compute finish --scope <STUDY_ID> --plan <plan>`),
    then report the outcome and next action.

## Never

- change model, checkpoint, pooling, layer set, dataset membership, splits,
  preprocessing, label mapping or precision for speed or cost;
- tune a candidate into comparability after the fact, or select on test data;
- edit another Study's folder, or answer a different hypothesis than the
  registered plan.
