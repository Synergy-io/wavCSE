# DG-0007 Run Note

Status: READY_FOR_REVIEW — pre-registered, no run, no MLflow run
Type: diagnostic (implementation-faithfulness control for the in-category MTRL arm)
Created: 2026-09-29
Research family: Task Relation Learning
Predecessor: the historical classical-MTRL campaign (`improvements/taskrelation/01-mtrl/`)
Reason: theory-to-implementation audit
Correction: `model.normalize_w: true → false`
Protocol: `../../VARIANT_BENCHMARK_PROTOCOL.md`
Config: `improvements/taskrelation/01-mtrl/mtrl_norm_corrected_25L_config.yml`
Compute: none. No authorization envelope exists for this scope.

## Hypothesis

Removing the row normalization of the task parameter matrix — so that the
published `Ω` closed form (TKDD 2014 Eq. (14)) and the published relation
regularizer apply to `W` itself — changes what the classical-MTRL arm measures,
against both its own historical adapted implementation and the matched wavCSE
baseline.

Competing explanation stated before any run: the change is a repackaging. The
normalization-corrected arm reproduces the historical arm within noise, and F4 transfers to the
published method. Pre-declared third outcome: the declared mean-head-summary
adapter binds, the normalization-corrected arm is dominated by the largest-norm head, and the
result is a boundary of the control rather than an accuracy statement.

## Current state of the arm

Pre-registered only. No code change, no config launch, no MLflow run. The
successor needs no new implementation: `normalize_w` is already a forwarded
model argument (`run_improvements.py:144-145`, `mtrl_er_kfold.py:132`) and every
historical config sets it to `true`, so the correction is a single YAML value.

## Why this is not a re-run of a disappointing result

`DEC-0005`/`DEC-0006` forbid re-running a result merely because it was
disappointing. This is not that. The audit shows the currently registered
in-category control executes a *different objective* from the one it is named
for (measured gap to the published optimum: 50.11 %; penalty scale-invariance
1.0031 instead of 9.0), and the corrected configuration has never been run
under the binding `smp` 25-layer protocol. The comparison is a protocol
correction plus a controlled comparison, which `DEC-0005`/`DEC-0006` permit.

## What is unresolved — and why it needs the orchestrator/human

1. **Authorization coverage.** No envelope covers this scope; the audit created
   none. Whether `DEC-0013`'s published-variant benchmark envelope extends to a
   corrected implementation of the existing control is a human call.
2. **Protocol text.** `VARIANT_BENCHMARK_PROTOCOL.md` §2 names
   `mtrl_poolingwinner_25L_config.yml` as "Classical symmetric MTRL". On this
   audit's evidence that label is inaccurate. Editing the protocol is a `DEC`
   decision (`DEC-0013` is the human's) and was deliberately not done here.
3. **Adopting a corrected control.** If H1 holds, the in-category control for
   every family-A/B variant arm should change; if H2 holds, only the
   documentation changes. Either way it is a `DEC`-level change.

## Deliverables in the audit branch

* `../../audits/2026-09-29-mtrl-theory-to-implementation-audit.md` — the audit.
* `improvements/taskrelation/01-mtrl/mtrl_norm_corrected_25L_config.yml` — the
  corrected config; differs from the historical control in exactly one
  scientific key (`model.normalize_w`) plus its two output-directory names.
* `../tests/test_mtrl_theory_faithfulness.py` — the mathematical regression
  suite. The historical arm's mathematics is now asserted directly (the Omega
  gap and the penalty scale ratio), independently of the configuration value,
  and the same criterion is shown to pass for the corrected arm.
* `configs/` — the six execution configs (`<arm>.yml` for the screen and
  `confirm_<arm>.yml` for confirmation) that carry the `research:` identity
  block, so every arm is unambiguously DG-0007 in the MLflow run name and tags
  and in the `ARC_RUN_IDENTITY` record. Every arm runs through
  `improvements.run_improvements` (`--model mtrl` for both MTRL arms,
  `--model original` for the matched baseline), per DG-0002's convention;
  `../tests/test_dg0007_run_identity.py` asserts it against the real helpers.
* `../tests/test_mtrl_runtime_faithfulness_checker.py` — pins the gate's exit
  contract (0 pass / 1 scientific fail / 2 unusable evidence) and its
  run-identity binding, including the malformed-artifact and wrong-run cases.

The study may not be launched until the authorization question is answered: it
requests no compute, and none is authorized.

## Next step

Orchestrator review: confirm or deny authorization coverage, then schedule the
seed-42 screen (three arms) under the matched protocol. Any ER claim that
emerges requires the speaker-independent LOSO protocol before it is stated.

---

## 2026-09-29 — Protocol decision recorded: `DEC-0017` (RETAIN), §2 unchanged

Appended, not substituted: the unresolved-question list above stays on record as the state the
Study was in when the decision was taken.

The researcher resolved items 2 and 3 above as **`DEC-0017` — RETAIN**:

* **Historical MTRL is retained — the control and the reproducibility anchor.**
  `improvements/taskrelation/01-mtrl/mtrl_poolingwinner_25L_config.yml` (`normalize_w: true`)
  remains `VARIANT_BENCHMARK_PROTOCOL.md` §2.1's in-category control. It is **not** replaced
  and **not** redefined, and §2.1's text is unchanged; a dated decision note now sits beside it.
* **Normalization-corrected MTRL does not become the control.** It remains a distinct `DG-0007`
  successor **experimental** arm. Whether it should later *become* the standing control is a
  separate decision for `DG-0007`'s own evidence and is **not** settled here.
* **Item 3's conditional is therefore answered in the retaining direction for now.** If H1
  holds, every citation of the control must carry its documented `D2` modification — the
  protocol note records that a "beats classical symmetric MTRL" claim has to name it — but
  adopting the corrected arm as the standing control is not authorized by this decision. If H2
  holds, only the documentation changes, exactly as `PLAN.md` already states.

Consequences for this Study, none of them a design change:

* the three arms, the independent variable (`model.normalize_w`), both gates and both decision
  rules are **unchanged** — the control's identity was never an experimental question;
* the Study is now blocked on **authorization coverage alone** (item 1 above remains open);
* no compute was authorized, requested, provisioned, submitted or executed by this decision,
  and no metric of any run was touched.


---

## 2026-09-29 — Screen execution attempted and blocked: controller MLflow credentials absent

Appended, not substituted. The pre-registration above is unchanged; the screen did **not**
run and no scientific result exists.

**What was done.** With the researcher's grant (`DEC-0018`: screen only, `$3.00`, 3-hour
window) and the envelope committed at `8fb395b`, a DG-0007-owned worker was provisioned
from an independent checkout (`~/projects/wavCSE-mtrl-integrate`, never the live
`~/projects/wavCSE`), the existing `200 GB` EU-RO-1 cache volume was attached, and the
three-arm seed-42 submit was attempted.

**The blocker.** `infra job submit` refuses every job whose spec declares
`environment_secrets` when those variables are not set on the controller:

```
these secret environment variables are required by the job specification but are not set
on this controller: MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_USERNAME; export them in
this shell before submitting
```

`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` are **not** in this session's
environment and exist in no sanctioned store on the host: no `.env` in any wavCSE checkout
or `$HOME`; no `Environment=`/`EnvironmentFile=` in the reaper unit; and
`~/.config/wavcse-infra/config.toml` carries only AWS/RunPod/SSH keys. `TR-0007`'s submits
succeeded because the credentials were present in *that* orchestration's environment; they
are not persisted anywhere a fresh session can read.

**Why the plan was not adapted.** Dropping `environment_secrets` would make the submit
succeed and would also remove the runs' MLflow/DagsHub logging, which `DEC-0006`, protocol
§7 and `AGENTS.md` require for every run. That is changing the scientific configuration to
make a failing step pass, which this screen's grant explicitly forbids. The plan, arms,
seeds, layer policy and pooling are untouched.

**Framework reconciliation (recorded for provenance).** The first submit left
`submission_pending: true` on the `mtrl_norm_corrected` entry with `failure_class:
TRANSIENT_INFRA` and no `job_id`, after the framework's own post-failure reconcile found no
matching provider job. Provider truth was then established independently — `infra job list
--worker b0ucfxlwrw3iz7` returned `[]`, and `infra worker exec … -- ps` showed no job
process — proving the submission never landed. The stale flag was cleared through the
backend's own state API with an event appended; nothing was re-submitted blindly.

**Cost and cleanup.** One DG-0007 worker (`b0ucfxlwrw3iz7`, NVIDIA RTX A4500, `$0.25/h`,
envelope digest `5b365dc285621a12e201cc89cbefdb585c43707805566bf35635d3a5e3249302`) was
created and destroyed after ≈`0.07` paid wall-clock hours ≈ **`$0.02`**. The persistent
`200 GB` EU-RO-1 cache volume was **not** destroyed. `TR-0007` was not stopped, restarted,
adopted or otherwise touched, and the live `~/projects/wavCSE` worktree was never modified.

**Remediation (human action).** Export `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` in the environment that runs the orchestrator (or place a
gitignored `.env` in the orchestration checkout), then re-run `worker-ensure` followed by
`advance --scope DG-0007 --plan improvements/taskrelation/research/studies/DG-0007/compute/plan.json
--stage screen`. Provisioning took 68 s, so resuming is cheap; the envelope at `8fb395b`
expires `2026-09-30T00:11:20+00:00`.
