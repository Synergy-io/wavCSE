# DG-0002 Run Note

## Hypothesis

At least one KS/SI/ER pair has persistent shared-parameter gradient conflict, norm dominance, or phase-dependent compatibility that classical MTRL's learned Ω does not represent or mitigate.

## Motivation / prior evidence

Classical MTRL is outcome-neutral under matched multi-seed and ER LOSO evaluation (F4). DG-0001 rejected raw ER-directed asymmetry because optimizer exposure dominated pairwise transfer (F8). Shared-parameter gradient compatibility remains unmeasured and is the highest-priority diagnostic in STATE.md and BACKLOG.md.

## Exact change

Instrument the existing plain wavCSE and classical MTRL trainers identically. On the first, final, and every twentieth training step with all tasks present, compute observational per-task gradients of unweighted masked cross-entropy over shared parameters only; record task norms, pairwise cosine, negative-conflict indicators, valid-example counts, normalized progress, and early/middle/late phase. Training losses and optimizer gradients remain unchanged.

## Matched controls

The seed-42 screen compared plain wavCSE with MTRL at `ks_si_er`, fixed WavLM-Large embeddings, frame-mean plus `smp(0.5)` pooling, all 25 layers, full data, 30 epochs, batch 2048, and identical optimizer, scheduler, data, checkpoint, evaluation, and diagnostic schedules. Confirmation repeats both arms at matched seeds `0,1,2,3,4` without changing this protocol.

## Evaluation protocol

Use fixed-final-epoch KS/SI/ER/aggregate outcomes as context, with `opt` and `best` as sensitivity tags. Primary evidence is phase-wise gradient norm, cosine, conflict frequency, batch-exposure agreement, and MTRL Ω correspondence. Ordinary-split ER remains screening-only; this Study makes no ER performance claim.

## Promotion/rejection criterion

Promote to matched seeds 0–4 only for an exposure-matched persistent signal: pairwise mean cosine at most -0.05 with conflict frequency at least 0.60 in at least two thirds, norm ratio at least 3 in at least two thirds, or a material compatibility sign/order change across training. Weaken the hypothesis if gradients are near-neutral and similarly scaled, or if MTRL coherently mitigates the measured conflict. One seed cannot confirm the hypothesis.

## Confirmation execution

Matched baseline and MTRL seeds `0,1,2,3,4` completed under stage `confirm`. Each arm produced 142 exactly exposure-matched diagnostic samples; seed-level phase summaries are the independent observations.

## Git commits

- instrumentation/screen config: `75e31b81e860d54f6125dd4a623d85e93718b7ec`
- confirmation configs: `f38a6dbea81837ac802caf7065144c125a392275`
- commit used by all ten confirmation runs: `7f6d5248f40c0c1cbd30f15b8f7cd1fe2dbb04eb`

## Execution incident

The first queue attempt used the launcher as a file path and exited before MLflow or training with `ModuleNotFoundError: improvements`. Commit `7f6d5248` changed it to the package entry point; the unchanged scientific configurations were then run successfully. The failed command remains in `logs/confirmation/confirm_baseline_s00.log`.

## Results

Across seeds 0–4, baseline ER-to-smallest-task shared-gradient norm ratios were `7.243 ± 0.272` in the middle third and `8.962 ± 0.456` late; every seed exceeded the pre-registered ratio-3 threshold in both phases. MTRL ratios were `7.129 ± 0.421` and `9.004 ± 1.059`, again above threshold in every seed. Paired MTRL-minus-baseline ratio differences were −0.114 (95% CI [−0.717, +0.490]) middle and +0.042 ([−1.473, +1.556]) late: no consistent mitigation.

No task pair met the persistent-conflict threshold in any seed; late mean cosines remained near zero. Ω saturated in magnitude in all five seeds. Four seeds ended uniform positive; seed 4 reproduced the known joint ER-edge sign flip. Fixed-epoch MTRL-minus-baseline outcome deltas were aggregate −0.00031, KS −0.00018, SI −0.00036 and ordinary-split ER −0.00109; every paired 95% interval included zero.

## Interpretation

ER gradient-scale dominance is reproducible under this matched `smp` 25-layer protocol, and classical MTRL does not remove it. Persistent pairwise gradient conflict is rejected as the explanation supported by DG-0002. The result identifies an optimization-scale limitation that Ω does not regulate; it does not prove that the imbalance causes MTRL's performance null. ER's smaller effective batch, task difficulty and label noise remain competing causes. Ordinary-split ER outcomes remain speaker-leaky and support no performance claim.

## Decision

`CONFIRMED` diagnostic finding. This is confirmation of task-gradient scale imbalance, not confirmation of an architecture improvement or an ER accuracy effect.

## Next step

Enter targeted literature mode. Search for published Task Relation Learning methods whose stated assumption addresses unequal task-gradient scale or reliability while retaining explicit learned task relations. Do not implement generic gradient surgery, loss weighting, or a new architecture unless its taxonomy and mechanism are verified against this finding. A data-regime control remains necessary before calling the scale imbalance task-intrinsic.
