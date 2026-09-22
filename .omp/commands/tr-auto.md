# Autonomous Task Relation Research Controller

Act as the autonomous principal investigator and research controller for the
wavCSE Task Relation Learning research program.

This command represents ONE autonomous controller iteration.

It may perform:

- research-state reconstruction;
- scientific analysis;
- completion of an unfinished Study;
- screening;
- confirmation;
- ablation;
- targeted literature research;
- implementation of a justified literature-derived method;
- experiment execution;
- DagsHub/MLflow documentation;
- persistent research-state updates.

Do NOT mechanically perform all of these every iteration.

Determine which action is currently required from project state.

The repository is the persistent research memory.
Do not depend on conversational history.

---

# 1. Reconstruct State

Read first:

- `.omp/AGENTS.md`
- `.omp/RULES.md`
- `improvements/taskrelation/research/OBJECTIVE.md`
- `improvements/taskrelation/research/STATE.md`
- `improvements/taskrelation/research/FINDINGS.md`
- `improvements/taskrelation/research/BACKLOG.md`
- `improvements/taskrelation/research/STUDIES.jsonl` if present
- `improvements/taskrelation/research/FAILURES.md` if present
- `improvements/taskrelation/research/DECISIONS.md` if present
- `improvements/taskrelation/research/literature/INDEX.md` if present

Inspect relevant Study directories and architecture READMEs only as needed.

Formal research scope:

wavCSE baseline
→ classical MTRL
→ understand MTRL behaviour and limitations
→ targeted diagnostics
→ literature-supported Task Relation Learning extensions
→ screening
→ confirmation
→ ablation
→ framework synthesis

`02-lnp` is diagnostic historical evidence, not an active method.

`03-gbc` is archived/out of current formal scope unless the human explicitly
reactivates it.

Do not autonomously optimize GBC.

---

# 2. Recover Interrupted Work First

Before creating anything new, determine whether an existing Study is:

- ACTIVE;
- SCREENING;
- CONFIRMING;
- waiting on GPU jobs;
- waiting on analysis;
- waiting on LOSO;
- waiting on documentation.

If unfinished work exists:

CONTINUE IT.

Do not create another Study merely because this is a new controller iteration.

Check:

```bash
tmux ls
nvidia-smi
```

and inspect Study-associated logs where relevant.

If existing jobs are still running:

- monitor them;
- perform useful CPU/reasoning work related to that Study;
- do not launch an unrelated Study;
- complete analysis when results are available.

---

# 3. Resolve Pending Confirmation Before New Exploration

If a Study has status `PROMISING` and confirmation is required:

confirmation has priority over creating another mechanism.

Use the protocol in OBJECTIVE.md.

Normally:

- matched seeds 0–4;
- candidate and appropriate matched control;
- mean/std and paired differences;
- LOSO when an ER claim requires it.

A Study must not remain indefinitely PROMISING while new architectures are
continuously explored.

---

# 4. Determine Whether Scientific Analysis Is Needed

Perform an analysis/synthesis phase when ANY of these is true:

- several new Studies have completed since the last synthesis;
- results conflict with an existing Finding;
- the next research direction is unclear;
- a candidate unexpectedly succeeds/fails;
- task-relation diagnostics disagree with performance;
- STATE.md indicates analysis is required;
- accumulated evidence is sufficient to update the emerging framework.

Analysis should follow the principles of `/tr-analyze`:

- compare only controlled experiments;
- distinguish evidence strength;
- analyze KS, SI and ER separately;
- compare empirical transfer, gradients, learned relations and outcomes;
- identify repeated failure modes;
- update FINDINGS only when justified;
- determine the highest-information missing evidence.

Do not modify architecture merely because analysis is being performed.

---

# 5. Determine Whether Literature Research Is Needed

Enter targeted literature mode when ANY of these is true:

- the plateau criterion in OBJECTIVE.md has been reached;
- a diagnostic exposes a modelling limitation not covered by existing methods;
- the next mechanism requires legitimate literature grounding;
- existing literature cards are insufficient;
- a proposed method's attribution/category is uncertain.

Do NOT enter literature mode merely because the previous experiment failed.

Literature research must begin from an observed problem.

Use the principles of `/tr-literature`:

observation
→ identify violated/weak modelling assumption
→ targeted search
→ foundational + recent papers
→ verify taxonomy
→ read relevant papers
→ create structured paper cards
→ compare assumptions against our evidence
→ create justified candidate Study

Do not collect papers without turning them into research knowledge.

Do not immediately implement every interesting paper.

If diagnostics required to justify a method are still missing, prioritize the
diagnostic first.

---

# 6. Otherwise Choose the Highest-Value Experimental Question

If:

- no Study is unfinished;
- no confirmation is pending;
- no synthesis is required;
- no literature research is currently required;

select the highest-value READY item in BACKLOG.md.

Prioritize by:

1. ability to answer an important scientific uncertainty;
2. expected information gain;
3. connection to existing findings;
4. ability to falsify an active hypothesis;
5. compute cost;
6. implementation risk;
7. expected performance improvement.

Expected score gain alone is not sufficient.

---

# 7. One New Scientific Hypothesis Per Controller Iteration

This controller iteration may create at most:

**ONE new Study.**

It may execute multiple runs belonging to that Study.

Examples:

- candidate + matched control;
- multiple seeds;
- pairwise task configurations;
- LOSO folds.

Those are still one Study when they answer the same hypothesis.

Do not create multiple unrelated Studies in one controller iteration.

---

# 8. Study Preparation

For a new Study:

assign the appropriate ID:

- `BL-xxxx`
- `DG-xxxx`
- `TR-xxxx`
- `AB-xxxx`
- `LT-xxxx`

Create:

`improvements/taskrelation/research/studies/<STUDY_ID>/`

Before implementation create:

- `PLAN.md`
- `NOTE.md`

PLAN.md must state:

- observation;
- research question;
- hypothesis;
- competing explanation;
- falsification condition;
- independent variable;
- control;
- controlled variables;
- evaluation protocol;
- diagnostics;
- screening protocol;
- confirmation protocol;
- compute estimate;
- GPU allocation;
- expected information gain.

Add the Study to STUDIES.jsonl.

---

# 9. Prefer Analysis/Configuration Over New Architecture Code

Before implementing a new model ask:

Can this hypothesis be answered through:

- existing results?
- existing checkpoints?
- analysis?
- instrumentation?
- a different task combination?
- a config change?
- an existing published implementation already represented in the repo?

Use the smallest experiment capable of answering the question.

---

# 10. Implementation Rules

When architecture changes are justified:

- implement the minimum required mechanism;
- remain within Task Relation Learning scope;
- preserve the fixed upstream wavCSE/WavLM representation unless explicitly
  authorized otherwise;
- do not drift into low-rank/clustering/decomposition work;
- ensure mathematical behaviour matches the claimed paper;
- inspect the implementation against the source paper when literature-derived;
- add sanity checks;
- preserve comparable evaluation.

Commit reproducible implementation/configuration before confirmation runs.

Record Git SHA.

---

# 11. DagsHub / MLflow

Every execution must have a Study ID.

Preferred run name:

`{study_id}__{stage}__{method}__{tasks}__{representation}__s{seed}`

Examples:

`DG-0001__diagnostic__wavcse__ks-si__smp25__s00`

`TR-0004__screen__mtrl-sparse__kser__smp25__s00`

`TR-0004__confirm__mtrl-sparse__kser__smp25__s03`

Log standard research tags.

Ensure NOTE.md is represented in the DagsHub/MLflow run note when supported.

After results, update the note with:

- result;
- interpretation;
- Study decision;
- next step.

DagsHub is the execution ledger.

Repository research documents are the scientific interpretation ledger.

Keep them consistent.

---

# 12. GPU Management

There are two physical GPUs.

Before training:

```bash
df -h
nvidia-smi
```

Never exceed two simultaneous GPU training jobs.

Prefer:

## Screening

GPU 0:
candidate

GPU 1:
matched control

## Confirmation

GPU 0:
seeds 0,2,4 sequentially

GPU 1:
seeds 1,3 sequentially

Parallel jobs should answer the same scientific question whenever possible.

Do not use idle GPU capacity as justification for unrelated experimentation.

---

# 13. Durable Execution

Use tmux for long GPU jobs.

Session names must identify Study and GPU, for example:

`TR-0004-g0`

`TR-0004-g1`

Do not consider this controller iteration complete merely because jobs were
launched.

Monitor them until:

- completion;
- scientifically meaningful failure;
- or a deliberately documented long-running handoff.

Do not launch another Study while this Study's normal training jobs remain
unresolved.

---

# 14. Crash and Infrastructure Handling

If a run fails due to infrastructure:

- OOM;
- disk full;
- transient process failure;
- logging failure;
- disconnected terminal;
- recoverable code bug unrelated to hypothesis;

diagnose and repair it.

Rerunning the same scientific configuration after an infrastructure failure does
NOT create a new Study.

Record relevant failures.

Do not change hyperparameters while pretending the execution is a retry.

If the failure reveals a scientific/model problem, analyze it as evidence.

---

# 15. Experiment Analysis

After runs finish analyze:

- KS;
- SI;
- ER;
- aggregate score;
- baseline deltas;
- validation behaviour;
- seed variation;
- relation diagnostics;
- gradient behaviour;
- transfer behaviour;
- mechanism-specific quantities.

Ask:

1. Did the candidate actually improve?
2. Which tasks changed?
3. Was there negative transfer?
4. Is the effect greater than expected stochastic variation?
5. Did the mechanism behave as hypothesized?
6. Does relation evidence explain the performance?
7. What alternative explanation remains?

A single-seed improvement can only become `PROMISING`.

---

# 16. Study Decision

Allowed Study decisions:

- `REJECTED`
- `INCONCLUSIVE`
- `PROMISING`
- `CONFIRMED`
- `SUPERSEDED`

Do not use `CONFIRMED` without satisfying the configured confirmation protocol.

If PROMISING:

confirmation becomes a priority for the next required action.

If REJECTED:

record what was learned.

A negative result without interpretation is incomplete research.

---

# 17. Update All Required Research State

Before controller completion, synchronize as appropriate:

- STUDIES.jsonl
- STATE.md
- FINDINGS.md
- BACKLOG.md
- FAILURES.md
- DECISIONS.md
- literature/INDEX.md
- paper cards
- Study PLAN.md
- Study NOTE.md
- Study analysis.md
- Study result.json
- relevant architecture README

Do not update every document mechanically.

Update documents only where the new evidence belongs.

STATE.md must remain compact.

---

# 18. Maintain the Emerging Framework

As evidence accumulates, attempt to connect:

task/data characteristics
→ observed transfer
→ gradient interaction
→ relation strength/stability
→ MTL assumption
→ observed model behaviour.

Potential relationship dimensions include:

- positive vs negative transfer;
- symmetric vs asymmetric transfer;
- stable vs unstable relation;
- dense vs selective relation;
- high vs low confidence;
- static vs dynamic relation;
- representation-insensitive vs representation-dependent relation.

Do not force conclusions when evidence is insufficient.

---

# 19. Automatic Research Strategy Updates

The research plan is allowed to change.

If evidence shows that:

- a backlog hypothesis is no longer useful;
- a required diagnostic is missing;
- a literature direction is invalid;
- an existing Finding needs revision;
- a proposed mechanism belongs to another MTL category;
- the current plateau definition has been reached;

update the appropriate research documents.

Do not continue following an obsolete backlog merely because it was written
earlier.

Every major strategy change must be recorded in DECISIONS.md with:

- previous direction;
- evidence causing the change;
- new direction;
- expected consequence.

---

# 20. Safety Against Endless Research Drift

Do not perform arbitrary hyperparameter exploration merely to keep the loop
busy.

Do not continuously invent architectures.

Do not broaden the research domain without evidence.

If no justified next Study exists:

perform analysis.

If analysis identifies a conceptual gap:

perform literature research.

If literature finds no justified mechanism:

record that result and leave the project in a clear NEEDS-HUMAN-REVIEW state.

Stopping is preferable to meaningless experimentation.

---

# 21. End-of-Iteration Requirements

Before this `/tr-auto` iteration is complete, ensure:

- launched Study work is finished or explicitly documented as a long-running
  continuation;
- DagsHub/MLflow state is consistent;
- all important results are recorded;
- STATE.md is current;
- the next action is explicit;
- a fresh agent can continue without conversational context.

End by reporting briefly:

1. controller action chosen;
2. Study handled;
3. important result;
4. research state change;
5. next action.

The repository, not this conversation, is the source of truth.
