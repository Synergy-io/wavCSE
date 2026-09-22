Never modify test data or use test metrics for model selection.

Never treat a single-seed result as a confirmed improvement.

Never claim an ER improvement without appropriate speaker-independent
evaluation.

Never compare architectures using different pooling, embedding, data
split, epoch budget, or evaluation protocol unless that difference is the
explicit independent variable.

Every training run must have a Study ID.

Every MLflow run must include the standard research tags and a DagsHub
run note.

Record failed experiments. Never delete negative evidence.

Before starting a new study, search STUDIES.jsonl and FINDINGS.md to ensure
the hypothesis has not already been tested.

One study should test one primary scientific hypothesis.

Never silently modify the baseline evaluation protocol.

Do not change the fixed upstream wavCSE/WavLM embedding unless the human
explicitly changes project scope.

Task Relation Learning must remain distinct from the project's low-rank,
clustering, and decomposition research branches.

Do not run more than two simultaneous GPU training jobs.

Before GPU training check:
df -h
nvidia-smi

Commit the implementation/config before launching confirmation runs.
The commit SHA used by every run must be recorded.

## Hyperparameter Tuning

Hyperparameter tuning is allowed only when scientifically justified.

Do not perform broad or opportunistic hyperparameter sweeps solely to improve
the reported metric.

For a new method:

1. begin from literature-recommended, theoretically natural, or matched
   baseline settings;
2. change only hyperparameters relevant to the mechanism being studied;
3. use the validation set, never the test set, for selection;
4. use a small predefined search space;
5. record every attempted configuration, including unsuccessful ones;
6. keep tuning budget comparable across competing methods where fairness
   requires it;
7. once a configuration is selected, freeze it before multi-seed confirmation
   and LOSO evaluation;
8. do not tune separately on test seeds/folds.

A tuning experiment may be its own Study or an explicit sub-stage of a Study
when the scientific purpose is clear.

## Research Time Tracking

Every Study record must preserve:

- `created_at`
- `started_at`
- `completed_at` when completed
- status-transition dates where materially useful

Use ISO 8601 timestamps.

These timestamps are required so weekly progress can be reconstructed from
research evidence rather than conversational memory.

Do not rewrite historical timestamps when updating a Study later.
