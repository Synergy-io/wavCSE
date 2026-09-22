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

## Confirmation continuation

The `PROMISING` screen is now being confirmed with baseline and MTRL seeds `0,1,2,3,4`. Each MLflow run uses stage `confirm`; seed-level phase summaries are the independent observations. The primary confirmation question is whether ER gradient-norm dominance reproduces and whether MTRL consistently fails to reduce it while Ω saturates.

## Git commit

Implementation/config commit: `75e31b81e860d54f6125dd4a623d85e93718b7ec`.

## Results

Both seed-42 arms finished at commit `75e31b81e860d54f6125dd4a623d85e93718b7ec` with 142 exactly exposure-matched diagnostic samples. Baseline ER shared-gradient norms were 7.36× and 7.66× the smallest task norm in the middle and late thirds; MTRL ratios were 6.75× and 9.28×. No pair met the persistent-conflict threshold. Final Ω saturated to uniform positive coupling (off-diagonal range `6.95e-5`). Fixed-epoch MTRL deltas were aggregate −0.00090, KS +0.00073, SI −0.00303, and leaky-split ER +0.01085.

## Interpretation

The persistent pairwise-conflict explanation is weakened. A stronger screening signal is ER gradient-norm dominance, which MTRL does not mitigate or represent in its saturated Ω. ER's much smaller effective batch (mean 47 examples versus KS 539 and SI 1462) remains a competing data-regime explanation. One seed cannot establish generality, and ordinary-split ER accuracy is not a valid performance claim.

## Decision

`PROMISING` diagnostic screen; matched-seed confirmation active. No mechanism promotion and no literature search unless the signal confirms.

## Next step

Complete and analyze all ten matched confirmation runs, update each DagsHub note with the Study decision, then either enter targeted literature mode for the confirmed limitation or return to Ω estimation/parameter-summary diagnostics.
