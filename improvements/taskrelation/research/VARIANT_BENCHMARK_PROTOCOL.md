# Variant benchmark protocol — shared conditions for every relation-learning arm

Authority: DEC-0013 (human re-scope, 2026-09-22). This file is the contract that makes arms comparable. It is fixed **before** any candidate is implemented, so no arm can be tuned into comparability after the fact.

A variant Study may only report a result against this protocol. If an arm cannot run under these conditions, it is `INCONCLUSIVE` by construction, not a result.

---

## 1. Fixed conditions (identical in every arm)

| Component | Setting |
|---|---|
| Upstream representation | `wavlm_large`, frozen; frame pooling `mean`; layer pooling `smp` 0.5 over **all 25** layers |
| Task set | `ks_si_er` (Speech Commands 12-class, VoxCeleb1 1251-class, IEMOCAP 4-class) |
| Shared trunk | 1024→512 projector, 512→2000 hidden, dropout 0.4 / 0.6 — unchanged across arms |
| Epochs | 30 |
| Global batch | 2048, `drop_last_train: true` |
| Optimizer | AdamW, lr 0.0025, weight decay 5e-8 |
| Regularization | l1 1e-7, l2 1e-5 |
| Scheduler | ReduceLROnPlateau, factor 0.5, **effective patience 1** (see §8 — the config key `patience: 5` is inert) |
| Checkpoint policy | as `01-mtrl` / `base`: best / opt / epoch, protocol checkpoint = `epoch` |
| Data splits | unchanged; `subset_percentage: 100` (never lower it — it also subsets validation and test) |
| Optimizer exposure | same `num_samples`, hence the same ≈2,820 steps per run, unless the variant's own mechanism requires otherwise, in which case the control arm is re-run to match |
| Sampling composition | standard concatenated sampler (≈539 KS / ≈1462 SI / ≈47 ER per 2048-batch) unless the variant's mechanism is about composition — and then both arms change together |
| Gradient diagnostics | same instrumentation and schedule as `improvements/gradient_diagnostics.py` (first, final, every 20th step) |

## 2. Controls (two per variant, both mandatory)

1. **Classical symmetric MTRL** — the in-category control, `improvements/taskrelation/01-mtrl/` at `mtrl_poolingwinner_25L_config.yml` (λ=0.01, `omega_epsilon` 1e-4, `normalize_w: true`). The variant must beat **this**, not just the baseline, to be interesting as a relation mechanism.
2. **Matched wavCSE baseline** — the reference to beat (`improvements/base/`, same protocol). No relation method has ever displaced it (F4).

A variant that beats neither has not contributed. A variant that beats MTRL but not the baseline is a mechanism-level finding, not a champion.

## 3. Staging

* **Screen:** one explicit seed, one run per arm (variant, MTRL, baseline), with protocol and exposure checks verified before any comparison is read. A screen can only produce `PROMISING` / `REJECTED`, never a promoted claim (F1).
* **Confirmation:** seeds `0,1,2,3,4`, candidate and both controls at the same commit, reported as paired per-seed differences.
* **ER:** any ER performance claim additionally requires speaker-independent LOSO (F3). Ordinary-split ER numbers are screening context only — never carry them into a claim.

## 4. Endpoints and decision rule

* Primary: per-task and aggregate test accuracy at the protocol checkpoint.
* Required reporting: mean ± SD across seeds, **paired** per-seed differences against each control with 95% t intervals, sign counts, and per-task results (the sample-weighted aggregate is dominated by SI and can mask ER movement).
* Promotion bar (unchanged): beats both controls, no material regression on any task beyond `0.20pp`.
* Mechanism diagnostics: the variant's relation object (Ω or its analogue) dumped per epoch in the same JSON shape family as `01-mtrl`'s `omega_history.json`, so relation behaviour is comparable across arms. Also report saturation state — a saturated relation object carries no pair-specific information (F7).

## 5. What an arm may vary

Exactly one thing: the **relation mechanism** (the relation object, its estimator, and its update rule) plus the hyperparameters that appear in its source paper. Everything in §1 stays fixed. No arm may adjust pooling, layers, epochs, batch size, optimizer, splits or checkpoint policy to improve its number — that converts the comparison into a representation comparison (F2: pooling alone moved ER by 3.25pp, larger than any relation effect measured here).

## 6. Fairness

* Hyperparameters come from the paper or its natural defaults; if a small search is unavoidable, it uses the **validation** split, a pre-declared tiny grid, and the same budget for every arm.
* Freeze hyperparameters before confirmation; never select on test seeds or folds.
* Record every attempted configuration, including failures, in the Study folder and MLflow.

## 7. Provenance

* Every run carries a Study ID, stage, method, seed, task set, pooling, layers, git commit SHA, and a DagsHub run note (`NOTE.md`).
* Run name: `{study_id}__{stage}__{method}__{ks_si_er}__{smp25}__s{seed}`.
* Commit the implementation and config **before** launching; record the SHA in the Study and in `STUDIES.jsonl`.
* Rerunning after an infrastructure failure does not create a new Study, but the failure is recorded.

## 8. Known gotchas that invalidate or embarrass runs

* **`patience: 5` in the configs is inert** — the trainer reads `scheduler_patience` (default 1). Effective patience is 1 everywhere. Do not "fix" it inside one arm; all arms inherit it.
* **`dataset.subset_percentage` subsets validation and test too** — never use it to shrink training data.
* **ER's ordinary split leaks speakers** — ~15pp inflation; LOSO gives the honest ≈64% ceiling.
* **Task count changes optimizer exposure** — adding/removing tasks changes steps and per-task batch size; match or model it (F8).
* **Gradient scale is composition-driven** — do not interpret per-task gradient norms without naming the sampling regime (F10).

## 9. Operations

* Pre-flight `df -h` and `nvidia-smi`; the root disk sits near full on this shared machine and a mid-run checkpoint write failure silently corrupts a run.
* At most two training jobs at once; tmux sessions named `<STUDY-ID>-g<index>`.
* Log to DagsHub/MLflow; the repository keeps interpretation, DagsHub keeps execution evidence.

## 10. What invalidates a comparison outright

Unmatched pooling or layer selection; different epoch budgets; different optimizer exposure without a matching control; different checkpoint-tag choice; different seed treatment; selecting on the test split; comparing a variant's screened seed against a control's confirmed seeds; quoting an ordinary-split ER delta as an ER result.
