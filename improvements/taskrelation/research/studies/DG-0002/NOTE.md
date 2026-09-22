# DG-0002 Run Note

## Hypothesis

At least one KS/SI/ER pair has persistent shared-parameter gradient conflict, norm dominance, or phase-dependent compatibility that classical MTRL's learned Ω does not represent or mitigate.

## Motivation / prior evidence

Classical MTRL is outcome-neutral under matched multi-seed and ER LOSO evaluation (F4). DG-0001 rejected raw ER-directed asymmetry because optimizer exposure dominated pairwise transfer (F8). Shared-parameter gradient compatibility remains unmeasured and is the highest-priority diagnostic in STATE.md and BACKLOG.md.

## Exact change

Instrument the existing plain wavCSE and classical MTRL trainers identically. On the first, final, and every twentieth training step with all tasks present, compute observational per-task gradients of unweighted masked cross-entropy over shared parameters only; record task norms, pairwise cosine, negative-conflict indicators, valid-example counts, normalized progress, and early/middle/late phase. Training losses and optimizer gradients remain unchanged.

## Matched control

Plain wavCSE versus MTRL at `ks_si_er`, seed 42, fixed WavLM-Large embeddings, frame-mean plus `smp(0.5)` pooling, all 25 layers, full data, 30 epochs, batch 2048, and identical optimizer, scheduler, data, checkpoint, evaluation, and diagnostic schedules.

## Evaluation protocol

Use fixed-final-epoch KS/SI/ER/aggregate outcomes as context, with `opt` and `best` as sensitivity tags. Primary evidence is phase-wise gradient norm, cosine, conflict frequency, batch-exposure agreement, and MTRL Ω correspondence. Ordinary-split ER remains screening-only; this Study makes no ER performance claim.

## Promotion/rejection criterion

Promote to matched seeds 0–4 only for an exposure-matched persistent signal: pairwise mean cosine at most -0.05 with conflict frequency at least 0.60 in at least two thirds, norm ratio at least 3 in at least two thirds, or a material compatibility sign/order change across training. Weaken the hypothesis if gradients are near-neutral and similarly scaled, or if MTRL coherently mitigates the measured conflict. One seed cannot confirm the hypothesis.

## Git commit

Pending implementation/config commit; populate before launch.
