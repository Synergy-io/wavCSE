# DG-0002 — Exposure-controlled gradient compatibility baseline

Status: CONFIRMED — matched baseline/MTRL seeds 0–4 completed 2026-09-22
Type: diagnostic
Started: 2026-09-22
Research family: Task Relation Learning
Parent Study: DG-0001
Baseline evidence: F4 and DG-0001

## Current evidence

- Classical MTRL has no reproducible meaningful advantage over the matched wavCSE baseline at `smp` with 16 or 25 layers (F4).
- DG-0001 showed that same-epoch pairwise transfer is dominated by task-count-dependent optimizer exposure. Its raw ER-directed asymmetry therefore cannot justify an asymmetric mechanism (F8).
- Under `smp` with all 25 WavLM layers, learned Ω often saturates toward uniform positive coupling. Stable Ω magnitude has not predicted useful KS or SI transfer (F6–F7).
- No existing run records per-task shared-parameter gradient norms, pairwise cosine, or negative-conflict frequency (BACKLOG DG-0002; diagnostic synthesis).
- Pooling, layer selection, task set, data, epoch budget, optimizer, checkpoint policy, and seed must remain matched because their effects can exceed architecture effects (F1–F3).

## Unknown

During matched three-task training, do KS, SI, and ER exert persistent conflicting or norm-dominant gradients on the shared wavCSE downstream parameters, and does classical MTRL's learned Ω correspond to or mitigate those interactions?

## Research question

Does the null outcome of classical MTRL reflect a mismatch between its static symmetric head-parameter covariance and the optimization interactions acting on the shared backbone?

## Hypothesis

At least one task pair will show persistent shared-parameter gradient conflict or a stable task-gradient norm imbalance under ordinary wavCSE training, while classical MTRL will either assign that pair positive/saturated coupling or fail to materially reduce the conflict. Such a mismatch would support optimization interaction—not raw directed transfer—as a concrete limitation of the current MTRL mechanism.

## Competing explanation

The three task gradients may be mostly compatible and similarly scaled. In that case MTRL's null result is more likely due to weak/degenerate relation estimation, insufficient regularizer leverage, or the mean-head parameter summary rather than shared-backbone gradient conflict.

A second competing explanation is that apparent conflict is transient or seed-specific; one seed can identify a screening signal but cannot establish a task-intrinsic relation.

## Falsification condition

Weaken the primary hypothesis if, in the matched baseline:

- every pair has phase-wise mean cosine within `[-0.05, +0.05]` and negative-conflict frequency between `0.40` and `0.60`, with no direction persistent across at least two training thirds; and
- the phase-wise maximum/minimum mean task-gradient norm ratio remains below `3`.

Also weaken the mismatch claim if MTRL consistently reduces a baseline conflict signal by at least `0.10` absolute negative-conflict frequency in two or more training thirds and its Ω sign/order agrees with the measured compatibility pattern.

These are pre-registered screening thresholds, not significance tests.

## Expected information gain

Two matched one-seed runs distinguish three explanations using the existing baseline and MTRL rather than creating another architecture:

1. persistent conflict/norm dominance plus Ω disagreement → investigate literature on optimization-aware task relations;
2. benign gradients plus saturated Ω → prioritize relation estimation/parameter-summary diagnostics;
3. dynamic sign changes → test time-varying relation assumptions before any mechanism selection.

The result closes the largest explicit gap in the MTRL diagnostic synthesis at the cost of one paired training run.

## Independent variable

Training method:

- matched control: plain wavCSE downstream multi-task model;
- diagnostic arm: classical MTRL (`mtrl_lambda=0.01`, `normalize_w=true`, warmup 3 epochs, Ω update every epoch).

Gradient instrumentation is identical in both arms and does not alter the loss, optimizer, data, or model forward pass.

## Matched control

Plain wavCSE, `ks_si_er`, seed 42, `smp` layer pooling with parameter 0.5, all 25 WavLM-Large layers, full data, 30 epochs, batch size 2048, and the established optimizer/scheduler/checkpoint settings.

## Controlled variables

Held identical across both arms:

- fixed upstream `wavlm_large` embeddings and frame-mean pooling;
- tasks and dataset construction: `ks_si_er`, 100% data, unchanged splits;
- all 25 selected transformer layers;
- `smp` layer pooling, parameter 0.5;
- shared-layer dimensions and dropout;
- seed 42 and deterministic data-loader shuffle sequence;
- AdamW learning rate, weight decay, L1/L2 penalties, scheduler, batch size, workers, and epoch budget;
- aggregate loss weighting, validation behavior, checkpoint selection, and test evaluator;
- gradient diagnostic sampling interval and shared-parameter definition.

The MTRL regularizer and Ω updates are the only intended method difference.

## Evaluation protocol

- One paired diagnostic seed (`42`), run concurrently on separate physical GPUs.
- Both arms use the ordinary three-task split for KS/SI/ER and the existing `opt`, `best`, and final-epoch checkpoint evaluations.
- Test metrics provide outcome context only and are not used to select diagnostic thresholds or tune the method.
- ER accuracy remains speaker-leaky screening evidence; this Study makes no ER performance claim and therefore does not replace LOSO.
- Primary outcome comparison uses the fixed final-epoch checkpoint; validation-selected tags are sensitivity context.
- Verify that sampled per-task valid-example counts match between arms at corresponding diagnostic steps; mismatch invalidates the paired interaction comparison.

## Gradient diagnostic protocol

For the shared parameters (all trainable parameters except `classifiers.*`):

- use each task's unweighted masked cross-entropy from the same forward pass;
- sample the first, final, and every 20th optimizer step when all three tasks are present;
- compute per-task L2 gradient norm without concatenating parameter tensors;
- compute pairwise cosine for KS/SI, KS/ER, and SI/ER;
- record valid examples per task, global step, normalized training progress, and early/middle/late third;
- summarize mean, median, sample standard deviation, range, and negative-conflict frequency per pair and phase;
- summarize task-gradient norms and phase-wise maximum/minimum mean-norm ratio;
- retain raw sampled records as an MLflow/DagsHub artifact;
- compare MTRL gradient summaries against its per-epoch Ω history.

`torch.autograd.grad` is observational: it must not write optimizer `.grad` buffers or change the combined training loss.

## Screening protocol

This is a one-seed diagnostic screen. A pattern is `PROMISING` for confirmation only if:

- a pair's mean cosine is at most `-0.05` with negative-conflict frequency at least `0.60` in at least two training thirds; or
- phase-wise maximum/minimum mean task-gradient norm ratio is at least `3` in at least two thirds; or
- cosine sign/order changes materially across thirds, supporting a dynamic rather than static relation;
- the paired batch-exposure check passes; and
- the interpretation is not based on ER test accuracy.

Otherwise classify the Study `INCONCLUSIVE` or `REJECTED` for the gradient-conflict explanation.

## Confirmation protocol if promoted

Repeat matched baseline and MTRL arms for seeds `0,1,2,3,4`, preserving the same diagnostic schedule and protocol. Report per-seed phase summaries, mean and standard deviation across seeds, paired method deltas, sign consistency, and intervals over seed-level summaries. Do not pool correlated within-run diagnostic samples as independent evidence. LOSO is required only before converting an ER-specific optimization pattern into an ER performance claim.

## Metrics

Outcome context:

- KS, SI, ER, and sample-weighted aggregate train/validation/test accuracy and loss;
- per-seed and paired baseline–MTRL final-epoch deltas.

Primary diagnostics:

- task gradient norms;
- pairwise gradient cosine;
- negative-conflict frequency;
- phase-wise norm-dominance ratio;
- relation evolution over early/middle/late training;
- MTRL Ω entries, saturation state, and qualitative agreement with gradient compatibility.

## Success and rejection criteria

Support the hypothesis at screening level when the exposure check passes and at least one pre-registered persistent conflict, norm-dominance, or dynamic-relation criterion is met while MTRL's Ω/gradient response fails the corresponding agreement or mitigation check.

Reject or weaken the gradient-conflict explanation when the baseline has no pre-registered signal. Classify as inconclusive when a signal is isolated to one training third, exposure pairing fails, diagnostics are numerically unstable, or baseline and MTRL differ without a coherent Ω/optimization interpretation.

A one-seed result cannot be `CONFIRMED`.

## Expected compute

Approximately 0.5–0.8 GPU-hours per arm, including sampled extra backward traversals and evaluation; about 1.0–1.6 total GPU-hours with paired wall time under one hour if both GPUs are available. Diagnostic sampling every 20 steps limits overhead while retaining dozens of observations per training third.

## Intended GPU allocation

- GPU 0: plain wavCSE matched control, seed 42.
- GPU 1: classical MTRL diagnostic arm, seed 42.

No unrelated job will be launched while these arms run.

## Literature references

No new mechanism or literature-derived claim is introduced. Classical MTRL remains Zhang and Yeung, *A Convex Formulation for Learning Task Relationships in Multi-Task Learning* (UAI 2010; TKDD 2014). Targeted next-method literature search is gated on this Study identifying a concrete limitation.
