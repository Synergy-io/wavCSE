# DG-0005 — ER data-regime gradient-scale control

Status: READY — pre-registered, not yet executed
Type: diagnostic
Created: 2026-09-22T09:40:00+00:00
Research family: Task Relation Learning
Parent Study: DG-0002
Motivating finding: F9
Authorization: DEC-0009 (human decision: strict Task Relation Learning scope retained; this diagnostic authorized)

## Observation

F9: under matched `smp(0.5)` 25-layer `ks_si_er` training, ER's shared-parameter gradient norm is 7.24×/8.96× the smallest task norm in baseline middle/late phases (seeds 0–4), classical MTRL does not reduce it, and no pair shows persistent conflict. F9 is correlational: ER also contributes by far the fewest examples per mixed batch (mean ≈47 valid examples versus ≈539 KS and ≈1462 SI at global batch 2048). A batch gradient built from fewer samples is a noisier estimate, and `E||g|| ≥ ||E g||`, so ER's larger norm may be an artifact of its sampling regime rather than a property of its task relation.

## Research question

Holding optimizer exposure, epochs, global batch size, data splits, pooling, layers, optimizer and checkpoint policy fixed, does ER's shared-gradient norm dominance shrink when ER's per-batch sample contribution is raised to KS/SI scale?

## Hypothesis (H1 — data-regime explanation)

ER norm dominance is primarily a gradient-estimate-noise / data-regime effect. Raising ER's per-batch contribution (sampling with replacement, training only) removes most of the imbalance: the late-phase max/min mean task-norm ratio falls below the pre-registered 3.0 dominance threshold that DG-0002 used.

## Competing explanation (H2 — task-intrinsic scale)

ER's loss surface genuinely produces larger shared gradients at this representation regardless of per-batch count (4-way emotion classification over IEMOCAP embeddings, class overlap, label noise, or cross-entropy calibration on a small head). Prediction: ratios stay ≥3.0 in the middle and late thirds even with matched per-batch composition, with at most a mild reduction attributable to variance.

A third, non-exclusive outcome is that the ER norm *increases* with more ER samples per batch, because the batch gradient becomes a cleaner estimate of a genuinely larger mean gradient. That outcome also supports H2 and must be reported as such, not as a failed run.

## Falsification condition

- **Support H1** if, at the screening seed, arm A1 satisfies the exposure check (identical step count, identical total training samples, realized ER per-batch count ≥ 0.5× KS's realized count) and its late-phase max/min mean-norm ratio is `< 3.0` while A0's is `≥ 3.0`.
- **Reject H1** if A1's middle *and* late ratios remain `≥ 3.0`, or if the reduction is attributable to changed exposure rather than composition.
- **Inconclusive → run A2** (reverse direction: reduce KS/SI training contribution to ER scale) before any conclusion. This branch is pre-registered, not ad hoc.
- Under H1, F9 must be re-stated as a data-regime property and any reliability-aware mechanism rationale rests on this causal result. Under H2, F9 stands as a task-scale property.

## Independent variable

Per-task training-batch composition for ER, controlled by one training-only sampling weight. A1 fixes the ER weight at `11.5` while KS/SI remain `1.0`; this approximately equalizes ER's and KS's sampling mass from the observed 539:47 count ratio without changing `num_samples`.

## Matched control

A0 — fresh plain wavCSE control at the same commit, same seed, standard concatenated sampler, identical protocol to the DG-0002 baseline configuration (ER ≈47 examples per 2048-example batch).

DG-0002's confirmation runs may be cited as corroborating history but are **not** the pre-registered control, because they were produced at a different commit and DG-0005 needs exact paired comparison at its own commit.

## Controlled variables

Held identical across arms:

- fixed upstream `wavlm_large` embeddings; frame-mean pooling; `smp(0.5)`; all 25 layers;
- task set `ks_si_er`; training/validation/test splits unchanged, full data;
- evaluation data never resampled or subsetted (the existing global `subset_percentage` also subsets validation and testing, so it must not be used for this Study);
- 30 epochs; global batch 2048; `drop_last_train`; `num_samples` unchanged so optimizer steps stay ≈2820;
- AdamW LR 0.0025, weight decay 5e-8, l1 1e-7, l2 1e-5, and the inherited effective ReduceLROnPlateau patience 1 / factor 0.5 (the historical `patience`/`factor` config keys remain inert and matched);
- checkpoint selection and the test evaluator;
- seed, and the gradient diagnostic schedule (first, final, every 20th step) with the same shared-parameter definition (all trainable parameters excluding `classifiers.`).

Only arm A1's training-batch composition differs from A0.

## Evaluation protocol

- Primary: seed-level phase summaries of per-task shared-gradient norms, the max/min mean-norm ratio, and the ER-to-smallest-ratio, using the same definition as DG-0002.
- Exposure check is a gate, not a detail: per sampled step the realized valid-example counts per task must be recorded and reported. A composition arm whose realized counts do not match its design target is invalid, not merely noisy (F8).
- Secondary, context only: fixed-final-epoch KS/SI/ER/aggregate accuracy.
- Ordinary-split ER is speaker-leaky (F3): it is screening context only. This Study makes no ER performance claim, so LOSO is not required unless a later Study converts the gradient finding into an ER accuracy claim.

## Diagnostics

Reuse `improvements/gradient_diagnostics.py` (DG-0002's instrumentation). Report per phase, per arm, per seed: task norm means/medians, max/min ratio, ER-to-smallest ratio, all three pairwise cosines, negative-conflict frequencies, and realized per-task valid-example counts. Also report the train/val accuracy gap per task as context for the noise-vs-difficulty question. No Ω history is needed: no MTRL arm belongs to this Study.

## Screening protocol

Seed 42, two arms, concurrent on separate GPUs: A0 (matched baseline, GPU 0) and A1 (ER-weighted composition, GPU 1). ≈17–19 min per arm at this protocol. Promote to confirmation only when the exposure check passes and the ratio moves in a pre-registered direction.

## Confirmation protocol

Seeds `0,1,2,3,4` for the decisive arm plus a matched baseline (10 runs), distributed per project policy (GPU 0: seeds 0,2,4; GPU 1: seeds 1,3). Report per-seed phase summaries, mean/SD, paired arm-minus-baseline differences with 95% intervals, and sign consistency. Treat each seed as the independent unit; within-run samples are correlated.

## Compute estimate

- Screening: 2 runs ≈ 0.7 GPU-hours.
- Optional A2 branch: 1 additional screening run ≈ 0.35 GPU-hours.
- Confirmation: up to 10 runs ≈ 6 GPU-hours.
- Total bounded by ≈ 8 GPU-hours, ≤2 concurrent jobs.

## GPU allocation

Screening: GPU 0 = A0, GPU 1 = A1. Confirmation: GPU 0 = seeds 0,2,4; GPU 1 = seeds 1,3, as project policy requires. Check `df -h` and `nvidia-smi` before launch; root currently has ≈19 GB free and GPU 1 is shared with another user's job, so at most one of our jobs plus that occupancy is realistic at times.

## Verified codebase facts (read 2026-09-22, current HEAD)

These were confirmed by reading the sources, not inferred. They define the
implementation surface and the invariants any change must preserve.

### Data plumbing

1. `downstream/dataset/load_embedding.py::load_embedding()` builds
   `CombinedDataset([...])` in `dataset_id_array` order, which follows
   `task_type`. For `ks_si_er` the order is `[speechcommand, voxceleb, iemocap]`.
   `CombinedDataset.__getitem__` therefore dispatches by index range.
2. `subset_percentage` is applied whenever it is not `None`, and it wraps
   **training, validation and testing**. The DG-0002/DG-0005 protocol sets it to
   `100`, so a `PercentageSubset` wrapper is always present around
   `CombinedDataset` and must be unwrapped (`.dataset`) to reach the per-task
   sub-datasets.
3. A sample's task membership is derivable two ways: from the index ranges of
   `CombinedDataset.datasets` (no embedding reads), or from its label row, where
   absent tasks carry `ignore_index = -1`. No helper exposes either today.
4. Per-task batch composition is currently **determined**, never controlled:
   `downstream/dataset/custom_emb_dataloader.py::CustomEmbDataLoader` extends
   `DataLoader` with a fixed argument list and accepts **no** `sampler`;
   its `defined_collate` stacks `(embedding, label)` into a `[B, T]` long tensor.

### Exposure arithmetic (must be preserved)

5. 193,874 training samples / batch 2048 with `drop_last_train: true` = 94
   batches per epoch; × 30 epochs = **2,820 optimizer steps**, which matches
   DG-0002's logged `total_training_steps=2820` and its 142 sampled steps at
   `sample_interval_steps: 20`.
6. Consequently, a replacement sampler must use `num_samples == len(dataset)`
   to keep 94 batches/epoch. `WeightedRandomSampler` also requires
   `shuffle=False`, so the training loader's shuffle flag must be disabled when
   a sampler is supplied.
7. With `replacement=True` oversampling, ER's *per-batch count* rises while its
   distinct-sample pool stays the same, so the per-epoch repetition of ER
   examples increases. This is the quantity under test (gradient-estimate
   noise), not additional ER information. Record it as a caveat, not as a flaw.

### Trainer and config wiring

8. `improvements/run_improvements.py::build_trainer()` passes
   `training_cfg = cfg["training"]` verbatim with no key rewriting. A new knob
   therefore belongs under the config's `training:` block and is read in the
   trainer as `training_cfg.get(...)`. The existing
   `training.gradient_diagnostics` block is read the same way.
9. `MultiTasksModelTrainer.__init__` receives no seed argument and stores no
   seed attribute; if the sampler needs deterministic seeding, that must be
   plumbed explicitly rather than assumed available.
10. **Scheduler key caveat:** the trainer reads `scheduler_patience`
    (default `1`) and `scheduler_factor` (default `0.5`). The existing protocol
    configs set `patience: 5` / `factor: 0.5`, so `patience` is **inert** and the
    effective scheduler patience is `1`. DG-0002's logged LR path
    (halving repeatedly after epoch 6) is consistent with this. Both arms of
    DG-0005 inherit the same effective setting, so the comparison stays valid —
    but do **not** "fix" the key name in an arm, and do not describe the
    protocol as using patience 5.

### Gradient diagnostics (reused, unchanged)

11. `improvements/gradient_diagnostics.py` samples the first, final and every
    20th step; computes each task's unweighted masked cross-entropy and its
    `torch.autograd.grad` on shared parameters (all trainable parameters
    excluding `classifiers.`, 1,550,800 parameters); and writes
    `results/gradient_diagnostics.json` containing per-record
    `valid_examples`, `phase`, `progress`, and a `summary` with
    `task_norms`, `pairwise_cosines` and `max_min_mean_norm_ratio` for
    `all`/`early`/`middle`/`late`, plus `total_training_steps` and
    `sampled_steps`.

### Invariants any implementation must satisfy

12. Training split only; validation and test composition unchanged; do not
    reuse or alter `dataset.subset_percentage`.
13. New config keys must be additive and default-off so that with the knob
    absent the pipeline is byte-equivalent to the DG-0002 baseline protocol
    (project rule: never silently modify the baseline evaluation protocol).
14. The implementation must be committed before any run and that SHA recorded
    for every run (`.omp/RULES.md`).
15. The per-step realized `valid_examples` counts are the exposure gate. An arm
    whose realized composition misses its design target is invalid, not noisy
    (F8).

## Expected information gain

High, and it converts F9 from correlational to causal. It decides between:

- **H1 →** the dominant shared-gradient signal is a sampling/scarcity property; a future reliability-aware relation mechanism (Option 3, DEC-0009) would be causally motivated, and scientific reporting must stop describing the dominance as a task-intrinsic relation property;
- **H2 →** the dominance is task-intrinsic at this representation; Option 3's rationale changes to scale normalization on task-intrinsic gradients, and the framework statement ("relation magnitude ≠ relation scale") becomes a first-class finding.

Either outcome is prerequisite evidence before any human Option-3 authorization. This Study implements no mechanism and does not reopen the `TR-xxxx` gate.
