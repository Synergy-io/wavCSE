---
name: wavcse-experiment-operator
description: Design, run, evaluate and record one wavCSE Task Relation Learning study under the repository's controls, evidence tiers and provenance rules.
---

# wavCSE Experiment Operator
Scope: designing, running, evaluating and recording one Task Relation Learning study so its result
carries a defensible evidence tier and traceable provenance. The programme's objective and binding
scope decisions (`improvements/taskrelation/research/OBJECTIVE.md`, `.../DECISIONS.md`) are not this
skill's to change.

Other domains belong to sibling skills: `wavcse-embedding-generation` (embeddings, dataset layout)
and `wavcse-research-runner` (cycle orchestration). Infrastructure mechanics — workers, jobs,
storage, artifact publication, GPU provisioning and cost — belong to the bounded `infra/` subsystem,
not to a sibling skill: read `infra/AGENTS.md` and the documents under `infra/docs/`, and reach them
through the `infra` CLI (paid actions only via `improvements.compute`).

## Reconcile the record first
The record lives under `improvements/taskrelation/research/`: `OBJECTIVE.md`, `STATE.md`,
`FINDINGS.md`, `DECISIONS.md`, `FAILURES.md`, `BACKLOG.md`, `FRAMEWORK.md`,
`VARIANT_BENCHMARK_PROTOCOL.md`, `STUDIES.jsonl`, `studies/<STUDY_ID>/`, `literature/INDEX.md`.

- `STATE.md` is the canonical restart point, not an experiment log.
- `FINDINGS.md` is authoritative for numbers and provenance; when it and `STATE.md` disagree,
  `FINDINGS.md` wins.
- `DECISIONS.md` entries bind until the human changes them or new evidence explicitly supersedes
  them.
- Before opening a study, search `STUDIES.jsonl` and `FINDINGS.md` so an already-tested hypothesis
  is not rediscovered as new (`DEC-0006`). `AGENTS.md` carries the repository-level invariants.

## Study types and identifier prefixes
Fixed prefixes; never reuse an ID (`AGENTS.md`, `STATE.md`):

- `BL-xxxx` baseline/reproduction; `DG-xxxx` diagnostic; `TR-xxxx` Task Relation Learning mechanism;
  `AB-xxxx` ablation; `LT-xxxx` experiment derived directly from literature investigation
  (`STATE.md`).
- `FW-xxxx` final framework analyses (`BACKLOG.md`, "P2 — Final framework analyses"); `FL-xxxx`
  negative-evidence entries in `FAILURES.md`; `R1`–`R3` record-level observations about the record
  itself, not scientific claims (`FINDINGS.md`).
- Runs predating Study-ID tagging are retained as `LEGACY-PRE-ID` evidence (`FAILURES.md`;
  `FINDINGS.md` R2).

One study tests one primary scientific hypothesis, and a study is not one MLflow run: `TR-0012` may
span a screening seed, confirmation seeds, ablations and LOSO evaluation (`STATE.md`).

## Registering a study
- Every training run carries a Study ID (`AGENTS.md`). `studies/<STUDY_ID>/` holds `PLAN.md` and
  `NOTE.md`, plus `result.json` (or `confirmation_result.json`) and `analysis.md` once results exist
  (`BACKLOG.md`, DG-0001 "Study record"); DG-0001 predates the template and uses `STUDY.md`. Register
  in `STUDIES.jsonl` with `study_id`, `type`,
  `status`, `stage`, `hypothesis` and `outcome` (`DEC-0006`).
- Preserve `created_at`, `started_at`, `completed_at` and status-transition dates in ISO 8601; never
  rewrite historical timestamps (`AGENTS.md`, "Research Time Tracking").

## The plan a study must state before compute
`BACKLOG.md` ("Agent Backlog Rules") requires: the observation motivating it, a falsifiable
hypothesis, the independent variable, the matched control, the cheapest adequate experiment, the
expected information gain, whether literature motivated it, and a category-boundary check. Add the
stage, evaluation protocol, success/rejection criteria, expected compute and GPU allocation
(`studies/DG-0005/PLAN.md` is the pattern), and prefer high information gain per GPU-hour over the
largest number of experiments; a new hyperparameter value alone does not deserve a new study
(`BACKLOG.md`, `STATE.md`).

Category check: the work must stay Task Relation Learning, not loss weighting, gradient surgery,
low-rank, clustering or decomposition (`DEC-0005` §3, `FRAMEWORK.md` R7, `BACKLOG.md` taxonomy
gate).

## Hypothesis, competing explanation, falsification
- State one falsifiable hypothesis and at least one plausible competing explanation, and pre-declare
  a third non-exclusive outcome where one exists (`studies/DG-0005/PLAN.md`, H1 vs H2). Make the
  threshold explicit and numeric where possible: DG-0005 pre-registered a late max/min task-norm
  ratio `< 3.0` plus a passing exposure gate.
- Fix the promotion protocol in advance (`FRAMEWORK.md` R8, §8 step 4); never invent a favourable
  post-hoc threshold, and admit a post-hoc analysis only when it was committed before any statistic
  was computed — it promotes no new finding (`DEC-0012`, `studies/DG-0005/analyze_noise_shape.py`).

## Controls and comparability
- Hold every factor constant unless it is the explicit independent variable (`AGENTS.md`;
  `FINDINGS.md` F2: pooling, layer selection, data, epochs, optimization, evaluation protocol,
  random-seed treatment).
- Terminology: the **wavCSE baseline** (`improvements/base/`) is the reference to beat and the
  current champion; **MTRL** (`improvements/taskrelation/01-mtrl/`) is the formal baseline *method*
  under diagnosis, not the thing to beat (`DECISIONS.md` terminology).
- A variant needs two controls: classical symmetric MTRL as the matched in-category control, and the
  matched wavCSE baseline; a variant that beats neither has not contributed
  (`VARIANT_BENCHMARK_PROTOCOL.md` §2, `DEC-0013`). §1 fixes the conditions identical in every arm
  and §5 lets an arm vary exactly one thing, the relation mechanism. Optimizer exposure is a
  control: changing task count or composition changes update counts and effective per-task batch
  size; match or model it (`FINDINGS.md` F8, F10; `FRAMEWORK.md` R4).
- Never lower `dataset.subset_percentage` (it also subsets validation and test), never "fix" the
  inert `scheduler_patience` key inside one arm, and do not fix unrelated code
  (`VARIANT_BENCHMARK_PROTOCOL.md` §8). Never silently modify the baseline evaluation protocol
  (`AGENTS.md`). §10 lists what invalidates a comparison outright: unmatched pooling or layer
  selection, a different epoch budget, checkpoint tag or seed treatment, optimizer exposure without
  a matching control, selecting on the test split, a screened seed versus a control's confirmed
  seeds, or an ordinary-split ER delta as an ER result.

## Evidence tiers
`FINDINGS.md` status vocabulary: `ESTABLISHED` (survived the strongest protocol the project has for
the claim), `SCREENING` (single-seed/single-split only; cannot carry a claim), `RECORD` (about the
record, not the tasks). `FRAMEWORK.md` §3 levels: **A** matched multi-seed, speaker-independent LOSO
or structural replication; **B** moderate, motivates a diagnostic not a selection; **C** preliminary
or confounded, carrying no relation or architecture claim (F1, F2, F3).

- One seed screens; five seeds select — a single high score is not a confirmed improvement, sub-1pp
  effects need ≥5 seeds, and a screen can produce only `PROMISING` or `REJECTED` (`FINDINGS.md` F1,
  `FRAMEWORK.md` R2, `VARIANT_BENCHMARK_PROTOCOL.md` §3). Confirmation seeds are `0,1,2,3,4`,
  reported as paired per-seed differences (`OBJECTIVE.md`; §3).
- Report mean ± SD, paired per-seed differences against each control with 95% t intervals, sign
  counts and per-task results: the sample-weighted aggregate is dominated by SI and can mask ER
  movement (§4; `FAILURES.md` "Aggregate task-size masking").
- ER: any serious claim requires speaker-independent LOSO (`OBJECTIVE.md`
  `er.serious_claim_requires_loso`; `FINDINGS.md` F3). The ordinary split leaks speakers (~15pp
  inflation, per-fold SD ≈5pp, honest ceiling ~64%) and is screening context only.
- Promotion bar: beats both controls across seeds with no material regression beyond `0.20pp` on any
  task (`OBJECTIVE.md`; `FRAMEWORK.md` R8).

### Speaker-independent ER evaluation — the fold design
Ten folds over IEMOCAP's ten speakers (5 sessions × 2 speakers), sorted deterministically: for fold
`i`, `test = speakers[i]`, `val = speakers[(i + 1) % 10]`, `train` = the other eight. Every speaker is
test exactly once and validation exactly once across the folds; the rotating validation speaker is
never the fold's test speaker, so validation stays unseen for the fold being scored.
Only the `er`/IEMOCAP split is folded: `ks`/`si` keep their normal official splits every fold, and
training stays joint `ks_si_er`, matching the real architecture. Folds are built by
`improvements/base/kfold_iemocap.py::build_loso_fold` and run by `improvements/base/run_base_er_kfold.py`
(`--task_type ks_si_er --config configs/base_kfold_config.yml --num_folds 10`), reusing
`LoadEmbedding._load_iemocap()` unmodified via a `_LOSOLoadEmbedding` subclass. Cite the mean ± SD
over all ten folds, never one fold (`improvements/base/README.md`); folds use a 5-epoch budget, not the single split's 30.

## Diagnostics are not mechanisms
- Under `DEC-0013` the entry gate is a verified published method rather than a prior diagnostic, but
  the epistemic rules stand: a proposal must name which framework row and which class of quantity
  its assumption acts on (`FRAMEWORK.md` §2, §8).
- A non-relational quantity cannot justify a relation mechanism, and a relation mechanism cannot be
  justified by a scale artifact (`FRAMEWORK.md` §2; `FAILURES.md` FL-0004).
- A diagnostic may state which assumption fails; it may not authorize a mechanism by itself, and
  diagnostic studies do not increment the plateau counter (`BACKLOG.md`, DG-0005 "Diagnostic only";
  `STATE.md`).
- A learned relation or diagnostic pattern is not evidence of useful transfer: 25L `smp` KS↔SI is
  maximally stable near `+1/3` while neither KS nor SI gains materially, so Ω magnitude is not a
  transfer or utility proxy, and retrospective relation variance alone cannot select a mechanism
  (`FINDINGS.md` F6, `FRAMEWORK.md` R8; `BACKLOG.md` DG-0004). Report the conditioned result even
  when the conclusion is negative — the characterisation is the contribution (`FRAMEWORK.md` §8
  item 5).

## Failure is not falsification
- Record failed experiments; never delete negative evidence, and reclassify historical evidence
  rather than erasing it: F9 was refined by F10 and its withdrawn task-intrinsic rationale is
  recorded as withdrawn (`AGENTS.md`, `DEC-0006`, `DEC-0010`; `FINDINGS.md` F9/F10).
- `FAILURES.md` statuses in use: `ESTABLISHED NEGATIVE RESULT`, `ESTABLISHED NEGATIVE DIAGNOSTIC`,
  `ESTABLISHED NEGATIVE LITERATURE RESULT`, `ESTABLISHED NEGATIVE JUSTIFICATION`. Distinguish
  infrastructure failure from falsification: a rerun after an infrastructure failure does not create
  a new Study, but the failure is recorded, as is every attempted configuration
  (`VARIANT_BENCHMARK_PROTOCOL.md` §7, §6). Do not rerun a result merely because it was
  disappointing; rerun only as an explicit confirmation, reproducibility check, protocol correction
  or controlled comparison (`DEC-0005`, `DEC-0006`). Never silently retry a failed run into
  different science: fix the infrastructure, keep the scientific configuration unchanged.

## Hyperparameter tuning policy
Tuning is allowed only when scientifically justified (`AGENTS.md`, "Hyperparameter Tuning"):
begin from literature-recommended, theoretically natural or matched baseline settings; change only
mechanism-relevant hyperparameters; select on validation, never test; use a small predefined space;
record every attempted configuration; keep budget comparable across arms; freeze the configuration
before multi-seed confirmation and LOSO; never tune separately on test seeds or folds. Otherwise
take hyperparameters from the paper or its natural defaults (`VARIANT_BENCHMARK_PROTOCOL.md` §6).

Generic MTRL retuning is forbidden: `mtrl_lambda` 0.05 was worse than 0.01 on every task, and 60
epochs underperformed 30 (`DEC-0001`, `FINDINGS.md` F4, `FAILURES.md` FL-0001). Never turn a
diagnostic study into ad-hoc tuning, and never change model, checkpoint, pooling, layer set, dataset
membership, splits, preprocessing, label mapping or precision to make a job faster or cheaper
(`.agents/commands/wav-experiment.md`).

## Provenance: MLflow / DagsHub
- DagsHub/MLflow is the canonical execution record — one experiment per (category, architecture)
  pair, with the `wavcse-` prefix reserved for base and base-derived runs (`improvements/README.md`)
  — while the repository holds the scientific interpretation (`VARIANT_BENCHMARK_PROTOCOL.md` §9,
  `STATE.md`).
- Run name `{study_id}__{stage}__{method}__{ks_si_er}__{smp25}__s{seed}` (§7);
  `improvements/mlflow_utils.py::build_research_run_name` emits
  `{study_id}__{stage}__{method}__{task_type}__{representation}__s{seed:02d}`.
- `improvements/mlflow_utils.py::set_standard_tags` carries the standard cross-experiment tags
  (`category`, `model`, `method`, `study_id`, `stage`, `seed`, `pooling`, `layers`, `git_commit`,
  `status`, plus the rest of the set in that module), and every run also carries a DagsHub run note
  (`NOTE.md`) (`VARIANT_BENCHMARK_PROTOCOL.md` §7, `DEC-0006`).
- Commit the implementation and config before launching and record the SHA in the study and in
  `STUDIES.jsonl` (`VARIANT_BENCHMARK_PROTOCOL.md` §7, `AGENTS.md`). `resolve_git_commit()`
  returns the checked-out HEAD the execution actually used; record only the commit actually executed
  — a later documentation-only commit does not change the commit a run reports (`STATE.md`).

## Reporting is read-only
A status or weekly summary reconstructs state from the record and changes nothing: no code, no
research files, no experiments, no studies, no backlog edits, no commits
(`.agents/commands/wav-status.md`, `.agents/commands/wav-weekly.md`). Never inflate preliminary
evidence, hide negative results, claim significance without evidence, call a run a study, or count a
smoke test as scientific progress (`.agents/commands/wav-weekly.md`, "Must not"). Classify what is
reported: confirmed / strong diagnostic / preliminary-screening / rejected / inconclusive. When
repository records disagree, state the inconsistency explicitly instead of guessing which one is
stale (`.agents/commands/wav-status.md`). Weekly reports declare an evidence cutoff and label
post-cutoff events, excluding them from counts (`.agents/commands/wav-weekly.md` step 1;
`improvements/taskrelation/research/weekly/2026-09-22.md`).

## Closing a study
- Decide from the criterion declared in the plan, using the decision vocabulary in use —
  `CONFIRMED` / `REJECTED` in `STUDIES.jsonl`, `INCONCLUSIVE` for an arm that cannot run under the
  shared protocol (`VARIANT_BENCHMARK_PROTOCOL.md` §5), `SUPERSEDED` for a backlog entry
  (`BACKLOG.md`) —
  and the `BACKLOG.md` study statuses `DONE`, `PARTIALLY ESTABLISHED`, `AUTHORIZED-FOR-BENCHMARK`;
  backlog entry statuses are `READY`, `BLOCKED`, `ACTIVE`, `DONE`, `REJECTED`, `SUPERSEDED`, and a
  programme-level block is `NEEDS-HUMAN-REVIEW` (`DEC-0008`).
- State what was learned even from a rejected study (`FAILURES.md`, `FRAMEWORK.md` §5). Update, as
  applicable: `STUDIES.jsonl`; the study's `result.json`/`confirmation_result.json`, `analysis.md`
  and `NOTE.md`; `STATE.md` "Last State Update"; `FINDINGS.md` when a finding is promoted;
  `FAILURES.md` when the result is negative; `DECISIONS.md` for a binding scope change; `BACKLOG.md`
  when a gate changes.
- Keep the record consistent in one pass: when a study changes a finding, update every file still
  citing the old text (`DEC-0011` corrected a written analysis that lagged `FINDINGS.md`).
- A launched job is not a completed cycle: record the execution session names and the Study ID, wait
  for that study's jobs, analyse them, then update state.
- Plateau: after 5 consecutive failed mechanism studies, literature mode is mandatory
  (`OBJECTIVE.md`, `plateau.consecutive_failed_studies: 5`).

## Never
- Change the fixed upstream wavCSE/WavLM embedding without an explicit human scope change
  (`AGENTS.md`).
- Present a project-original mechanism as a published method, or a mechanism result as a diagnostic
  (`DEC-0014`).
