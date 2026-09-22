# Task Relation Research Cycle

Act as the principal investigator and research engineer for **one complete Task Relation Learning research cycle** in the wavCSE project.

This command is allowed to inspect code, modify code, create configs, launch experiments, analyze results, search literature when justified, and update the persistent research state.

Do not treat this as a generic "improve the score" task.

The objective is simultaneously:

1. improve matched wavCSE performance on KS, SI and ER;
2. avoid meaningful negative transfer;
3. understand why each mechanism succeeds or fails;
4. accumulate evidence for a framework connecting observable task behaviour to suitable parameter-based MTL approaches.

---

# 1. Reconstruct Project State

Before proposing or modifying anything, read:

- `.omp/AGENTS.md`
- `.omp/RULES.md`
- `improvements/taskrelation/research/OBJECTIVE.md`
- `improvements/taskrelation/research/STATE.md`
- `improvements/taskrelation/research/FINDINGS.md`
- `improvements/taskrelation/research/BACKLOG.md`
- `improvements/taskrelation/research/STUDIES.jsonl` if it exists
- `improvements/taskrelation/research/FAILURES.md` if it exists
- `improvements/taskrelation/research/DECISIONS.md` if it exists

Then inspect only the architecture READMEs and source files relevant to the current question.

At minimum be aware of existing work in:

- `improvements/taskrelation/01-mtrl/`
- `improvements/taskrelation/02-lnp/`
- `improvements/taskrelation/03-gbc/`
- older TSM/PMR implementations where relevant

Do not rediscover an established result and present it as new.

---

# 2. Determine What Kind of Cycle Is Needed

Classify the next cycle as exactly one primary type:

- `diagnostic`
- `screen`
- `confirmation`
- `ablation`
- `literature-derived`
- `analysis-only`

Prefer continuing an already active/promising Study over creating a new Study.

Create a new Study only when there is a distinct scientific hypothesis.

A new hyperparameter value alone does not automatically deserve a new Study.

---

# 3. Select the Highest-Information Question

Before coding, state:

## Current evidence

What do we already know that constrains this experiment?

## Unknown

What specific uncertainty are we trying to reduce?

## Hypothesis

State one falsifiable hypothesis.

## Alternative explanation

State at least one plausible competing explanation.

## Falsification condition

What result would make us reject or weaken the hypothesis?

## Expected information gain

Explain why this experiment is worth its GPU cost.

Prefer experiments that discriminate between explanations rather than experiments that merely generate another score.

---

# 4. Check for Duplicate Work

Search:

- `STUDIES.jsonl`
- `FINDINGS.md`
- `FAILURES.md`
- architecture READMEs
- MLflow/DagsHub references recorded in those files

Determine whether the same hypothesis, configuration, or equivalent mechanism has already been tested.

If it has:

- do not rerun it merely because the result was disappointing;
- rerun only if this is an explicit confirmation, reproducibility check, protocol correction, or controlled comparison.

Record the reason for any deliberate rerun.

---

# 5. Assign or Reuse a Study ID

Use the existing active Study ID if this is a continuation.

Otherwise assign the next valid ID using:

- `BL-xxxx` — baseline / reproduction
- `DG-xxxx` — diagnostic
- `TR-xxxx` — Task Relation Learning mechanism
- `AB-xxxx` — ablation
- `LT-xxxx` — literature-derived experiment

Never reuse an ID.

Create:

`improvements/taskrelation/research/studies/<STUDY_ID>/`

with at least:

- `PLAN.md`
- `NOTE.md`

Create `result.json` and `analysis.md` when results become available.

Add the Study to `STUDIES.jsonl`.

---

# 6. Write PLAN.md Before Implementation

`PLAN.md` must contain:

- Study ID
- title
- status
- research question
- hypothesis
- competing explanation
- independent variable
- matched control
- controlled variables
- evaluation protocol
- screening protocol
- confirmation protocol if promoted
- metrics
- task-relation diagnostics
- success / rejection criteria
- expected compute
- intended GPU allocation
- parent Study if applicable
- literature references if applicable

Do this before modifying model code.

---

# 7. Preserve Experimental Fairness

Unless a variable is intentionally under study, hold constant:

- upstream embeddings
- task set
- dataset construction
- pooling
- selected transformer layers
- optimizer
- learning-rate schedule
- epoch budget
- checkpoint-selection policy
- evaluation protocol
- seed treatment

Do not compare architectures with different pooling and interpret the difference as an architectural effect.

Do not modify test data.

Do not optimize against test results.

For ER:

- speaker-leaky single-split evaluation may be used for screening;
- serious claims require the project's speaker-independent LOSO protocol.

---

# 8. Decide Whether Code Is Actually Necessary

Before creating a new architecture, ask whether the hypothesis can be answered by:

- analyzing existing runs;
- running an existing model on a new controlled task subset;
- adding instrumentation;
- changing only a config;
- running pairwise/single-task controls.

Prefer the smallest intervention capable of falsifying the hypothesis.

---

# 9. Implement Minimally

If implementation is required:

- change only what is necessary for the Study;
- preserve the existing wavCSE embedding pipeline;
- keep the mechanism inside the Task Relation Learning research scope;
- avoid accidentally moving into low-rank, clustering, or decomposition work;
- add tests or sanity checks where appropriate;
- maintain compatibility with the existing improvement runner and MLflow infrastructure.

Before training:

- inspect the diff;
- verify the intended independent variable is actually isolated;
- ensure configs are explicit and reproducible.

Commit the experiment implementation/config before confirmation runs.

Record the Git commit SHA.

---

# 10. MLflow / DagsHub Metadata

Every training execution must belong to a Study.

Preferred run name:

`{study_id}__{stage}__{method}__{tasks}__{representation}__s{seed}`

Example:

`TR-0012__screen__mtrl-sparse__kser__smp25__s00`

Do not encode every hyperparameter in the run name.

Log detailed values as MLflow parameters.

Every run should carry, where applicable:

- `study_id`
- `stage`
- `family`
- `method`
- `hypothesis_slug`
- `task_set`
- `representation`
- `pooling`
- `layers`
- `seed`
- `parent_study`
- `baseline_study`
- `agent_generated`
- `git_commit`
- `status`

---

# 11. DagsHub Run Note

Before launching training, populate:

`research/studies/<STUDY_ID>/NOTE.md`

with a concise human-readable description containing:

## Hypothesis

## Motivation / prior evidence

## Exact change

## Matched control

## Evaluation protocol

## Promotion/rejection criterion

## Git commit

Attach/use the same content as the MLflow run note when supported by the runner.

After evaluation, append:

## Results

## Interpretation

## Decision

## Next step

Do not fill notes with raw logs.

---

# 12. GPU Scheduling

There are two physical GPUs.

Before launching any GPU job:

```bash
df -h
nvidia-smi
```

Respect the repository disk guard as well.

Do not run more than two training jobs simultaneously.

Prefer scientifically paired GPU use.

## Screening

Preferred:

- GPU 0 — candidate
- GPU 1 — matched control

or:

- GPU 0 — candidate A
- GPU 1 — candidate B

only when A and B answer the same scientific question directly.

## Confirmation

For five seeds prefer:

- GPU 0 — seeds 0, 2, 4
- GPU 1 — seeds 1, 3

Run sequentially per GPU.

Never launch unrelated experiments merely because a GPU is idle.

---

# 13. Long-Running Training

Use durable detached execution, preferably `tmux`, so research reasoning is not blocked by a training process.

Use descriptive tmux session names containing the Study ID and GPU.

Examples:

`TR-0012-g0`

`TR-0012-g1`

Capture logs to paths associated with the Study.

After launch, verify:

- process exists;
- correct GPU is being used;
- memory usage looks plausible;
- MLflow run has appeared;
- training has advanced beyond initialization.

Do not launch the next GPU experiment until current resource allocation is understood.

---

# 14. Use Idle Time Productively

While GPU training is active, useful CPU/reasoning work may include:

- inspecting previous Study results;
- implementing analysis scripts;
- reading a directly relevant paper;
- preparing confirmation configs;
- checking relation diagnostics;
- reviewing run metadata;
- improving documentation.

Do not start an unrelated architecture search while waiting.

---

# 15. Analyze Every Completed Run

Never reduce the result to `test_acc_all`.

Analyze separately:

- KS
- SI
- ER
- aggregate metric
- validation behaviour
- training behaviour
- matched baseline delta
- seed variability when available

Also inspect available task-relation diagnostics:

- learned relation matrix;
- pairwise relation magnitude/sign;
- gradient cosine;
- gradient norm;
- negative-conflict frequency;
- relation stability;
- relation evolution;
- relevant task transfer measurements.

Ask:

1. Did performance actually improve?
2. Which tasks changed?
3. Is there negative transfer?
4. Is the change larger than known stochastic variation?
5. Did the hypothesized mechanism behave as expected?
6. Does the task-relation evidence support the proposed explanation?
7. What alternative explanation remains?

---

# 16. Screening Decision

A one-seed Study can end as:

- `REJECTED`
- `INCONCLUSIVE`
- `PROMISING`

Never `CONFIRMED`.

A result should normally be promoted only when:

- the experiment is protocol-matched;
- the gain is not obviously within ordinary noise;
- no task suffers unacceptable regression;
- the mechanism behaves consistently with the hypothesis;
- the result is worth confirmation compute.

---

# 17. Confirmation

For a promoted candidate:

Use matched seeds:

`0, 1, 2, 3, 4`

Run both candidate and necessary control under equivalent conditions if comparable baseline seed results do not already exist.

Report:

- mean;
- standard deviation;
- per-seed values;
- paired deltas where seeds are matched;
- confidence interval or an appropriate significance analysis when useful.

Do not claim significance merely because the candidate mean is larger.

For any important ER conclusion, run the appropriate LOSO evaluation.

---

# 18. Scientific Decision

After sufficient evidence, classify the Study as:

- `REJECTED`
- `INCONCLUSIVE`
- `CONFIRMED`
- `SUPERSEDED`

Then explicitly state:

## What was learned?

Even rejected models should ideally produce knowledge.

Examples:

- symmetric coupling assumption unsupported;
- relation strength unstable across seeds;
- sparse relation mechanism collapsed to dense coupling;
- apparent gain explained by pooling;
- ER gain disappeared under LOSO;
- directionality predicted transfer successfully.

---

# 19. Update Persistent Research Memory

After the cycle, update as appropriate:

- `STUDIES.jsonl`
- `STATE.md`
- `FINDINGS.md`
- `FAILURES.md`
- `DECISIONS.md`
- `BACKLOG.md`
- Study `analysis.md`
- Study `result.json`
- architecture README if a durable architecture-specific result was established

Do not put every detail in STATE.md.

STATE.md should remain a concise restart document.

---

# 20. Plateau Handling

Read the plateau definition from `OBJECTIVE.md`.

If the plateau threshold has been reached:

Do not respond by random hyperparameter search.

Stop mechanism generation and invoke the reasoning pattern from `/tr-literature`:

1. identify the observed failure mode;
2. formulate targeted literature queries;
3. inspect foundational and recent work;
4. map assumptions to our measured task behaviour;
5. create literature cards;
6. propose only experiments whose assumptions fit observed evidence.

A literature-derived method should still go through the normal Study lifecycle.

---

# 21. End-of-Cycle Handoff

Before finishing, ensure a completely fresh agent session could resume the project.

STATE.md must clearly contain:

- current champion;
- active Study;
- most recent completed Study;
- important new finding;
- unresolved question;
- next recommended experiment;
- GPU jobs still running, if any;
- confirmation work pending, if any;
- current failed-study/plateau count.

Do not rely on this chat session as persistent memory.

The repository is the research memory.

# Unattended / Loop Execution Rules

When this command is invoked from `/loop`, one invocation still represents
ONE COMPLETE research cycle.

Do not finish the cycle merely because training was launched in tmux.

If this cycle launches training jobs:

1. record the tmux session names and Study ID;
2. periodically check whether the jobs are still running;
3. inspect logs and MLflow/DagsHub status;
4. wait for the jobs belonging to this Study to finish;
5. analyze their results;
6. update all persistent research state;
7. only then consider this `/tr-cycle` invocation complete.

Do not leave unfinished jobs for the next `/tr-cycle` unless:
- the job is intentionally a very long validation such as LOSO; and
- STATE.md explicitly records it as ACTIVE and the next cycle is instructed
  to continue that same Study rather than start unrelated work.

When an existing Study has unfinished GPU jobs, continue that Study before
creating a new one.

If a job crashes:
- diagnose it;
- fix infrastructure/code issues when appropriate;
- rerun only if the scientific configuration remains unchanged;
- record the failure;
- do not silently replace it with a different experiment.

If both GPUs are occupied by this project, do useful analysis/research while
waiting rather than launching another training job.

At the end of every unattended cycle, ensure:
- no untracked experiment results exist;
- DagsHub/MLflow metadata is updated;
- Study status is correct;
- STATE.md accurately describes the next action.
-
