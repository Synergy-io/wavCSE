# TR-0007 screen — analysis

**Classification: `REJECTED`** (pre-registered rule, applied unchanged after the runs).
Stage: `screen`, seed 42, one run per arm, protocol checkpoint `epoch` (the protocol's primary).
lambda_2 = 0.01 **researcher-fixed** (DEC-0016); lambda_0 = 1.0; all 25 wavLM layers; `smp` 0.5;
30 epochs; batch 2048; AdamW lr 0.0025. Machine-readable result: `screen_result.json`.

## Result

| arm | commit | `test_epoch_acc_all` | ks | si | er | evidence |
|---|---|---|---|---|---|---|
| **p-MSSL** (candidate) | `d13e82e` | **0.9662** | 0.9867 | 0.9615 | 0.7848 | collected, verified |
| classical MTRL (in-category control) | `664c572` | 0.9752 | 0.9855 | 0.9794 | 0.7848 | collected, verified |
| matched wavCSE baseline (control) | `664c572` | 0.9748 | 0.9861 | 0.9782 | 0.7848 | collected, verified |

Paired deltas (candidate − control, percentage points) at the protocol checkpoint:

| vs | all | ks | si | er |
|---|---|---|---|---|
| classical MTRL | **−0.90** | +0.12 | **−1.79** | 0.00 |
| matched baseline | **−0.86** | +0.06 | **−1.67** | 0.00 |

The pre-registered rule required the candidate to exceed **both** controls in the aggregate
with no per-task regression beyond `0.20pp`. It exceeds neither, and SI regresses by ~1.7pp, so
the screen is `REJECTED`. Nothing was re-tuned, no threshold was moved.

Reproducibility: the candidate arm was executed twice (see *Execution record*); the two
executions produced identical checkpoint bytes and metrics to five decimal places
(`0.966238` both times), so the outcome is not a run-to-run artefact.

## Why it failed — the mechanism, measured

The Ω step is live but **mis-scaled for this representation at λ₂ = 0.01**, exactly the caveat
`PLAN.md` pre-registered before any run. Evidence from the candidate run's per-epoch relation
artefact (`omega_history.json`, MLflow) and the implied task-row alignment `C ≈ d·Ω⁻¹`
(unit-norm rows give `S = C/d`, so the unpenalised precision is `d·C⁻¹`):

| epoch | Ω trace | implied cosines (ks·si, ks·er, si·er) | mean abs off-diagonal | relative duality gap |
|---|---|---|---|---|
| 3 (before the coupling) | 6 308 | +0.26, +0.04, +0.01 | 211 | −4.3e−05 |
| 4 – 30 (after) | 113 438 (frozen) | **+0.981, +0.981, +0.981** | 18 571 | −0.0535 |

1. As soon as the coupling term engages (warmup ends at epoch 3), the three task summaries
   align to cos ≈ 0.981.
2. Aligned unit-norm rows make `C` nearly singular, so Ω jumps ~20× to a nearly uniform
   precision — the same *uninformative* object classical MTRL saturates into (F7), reached
   differently.
3. The coupling gradient `2·lambda_0·Ω·W` is then `~1e5` against task gradients of `~1e-2`, so
   the regularizer, not the tasks, drives the update: the reported training loss is dominated by
   the coupling term (`≈2015` against `≈0.5` of task losses) and validation plateaus from
   epoch ≈10 (`val_acc_all` 0.9609 for the last five epochs). The learning-rate scheduler, fed
   by the coupling-dominated validation loss, decays to its floor.
4. The Ω snapshots also show the ADMM's own limit: from epoch 4 the relative duality gap stays
   at `−0.054`, i.e. the 2000-iteration cap was reached at that conditioning. The solver reports
   this per snapshot rather than hiding it; it does not change the verdict (the arm was already
   regularizer-dominated), but it is a limit of the current solver settings at this scale.

This is *not* a solver or protocol defect: the Ω step is verified against closed forms and the
problem's own optimality certificate (`test_mssl_omega_solver.py`), and every frozen protocol
field was checked (`test_tr0007_protocol.py`). It is the published estimator behaving as
published: Eq. (8) is not scale free, and the paper selects λ₂ on data precisely because its
magnitude is representation-dependent. With λ₂ = 0.01 fixed by the researcher, the arm is
`REJECTED` **at that fixed λ₂** — the screen cannot say that p-MSSL "cannot work", only that it
does not work here, at this scale, under this protocol.

## What this screen does and does not establish

* It establishes a **screening-level negative result** (single seed, single split): the
  faithful p-MSSL implementation, with the shared mean-head summary adapter and
  λ₂ = 0.01, does not beat either control on this protocol.
* It does **not** establish anything about ER as a result: the ordinary split is speaker-leaky
  (F3), and all three arms are within 0.004pp of each other on ER (0.7848). Any ER claim needs
  LOSO.
* It does **not** promote p-MSSL, and it authorizes no confirmation run (confirmation is a
  separate human grant).
* It leaves the open question it exposed, for a future decision rather than an autonomous one:
  whether a scale-calibrated λ₂ (validation-selected, or expressed relative to `d`) would make
  the mechanism informative, and whether the coupling term needs a magnitude control
  independent of Ω's scale. Both are protocol-level questions, not something to tune after
  seeing this screen.

## Execution record (infrastructure, recorded because it shaped the run)

* Raw-dataset provisioning (a gap ARC does not close): the loader's dataset classes read the
  *raw* voice corpora at `paths.root_data_path` (`~/voice_dataset/<dataset>`), which no study
  plan declares — the plan materialises embeddings only — while the network volume mounts at
  `/workspace/cache`. Before submitting, the worker's raw tree was verified and the loader's
  paths were pointed at the cache with the project's own workstation convention, a symlink per
  dataset (`~/voice_dataset/speechcommand/SpeechCommands/speech_commands_v0.01`,
  `~/voice_dataset/voxceleb/{wav,iden_split.txt}`, `~/voice_dataset/iemocap`), verified by
  file counts (64,727 / 153,516 / 10,039 wavs). No scientific parameter changed; a resumed or
  transferred attempt needs the same step.
* One worker (`8jy5oamc…`, `NVIDIA RTX A4500`, `$0.25/h`, EU-RO-1 with the pre-existing
  network volume), reused for all three arms; **paid wall clock 2.96 h, cost `$0.7404`**, well
  inside the screen envelope (0.80/h, 5.00 total, 6 h).
* Three jobs failed *before* producing science and were re-run under the bounded retry policy:
  two job-preparation failures caused by the worker's 60 GB container disk filling during
  checkout/transfer (the container disk, not the NFS cache volume, holds each job's ~45 GB
  scratch), and one duplicate candidate execution whose `MANIFEST.json` could not be persisted
  because ARC's output keys are per `(study, arm, seed)` and the first attempt's object already
  existed. The duplicate's metrics are retained as corroboration; it is excluded from the
  classification.
* The two control arms therefore ran at commit `664c572` and the candidate's collected run at
  `d13e82e`. The diff between those commits is **control-plane only** (job-preparation failure
  classification in `improvements/compute/failures.py` and its test); no scientific
  configuration or training code differs. `screen_commits.json` records this per arm, and the
  analysis binds each arm to the commit its evidence came from instead of to one SHA.
* The four historical `TR-0007__screen__*` / `scale-corrected` runs in
  `taskrelation-variant-benchmark` (commits `10aaaea3…`, `3df542d…`, seed 0) were **not**
  touched, relabelled or substituted; they remain historical evidence only.
