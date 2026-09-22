# DG-0005 Run Note

## Hypothesis

ER's shared-gradient norm dominance is primarily a data-regime / gradient-estimate-noise effect: raising ER's per-batch training contribution to KS/SI scale, with exposure otherwise held fixed, removes most of the imbalance (late max/min task-norm ratio below the pre-registered 3.0 threshold).

## Motivation / prior evidence

F9 (DG-0002 confirmation, seeds 0–4) established ER norm dominance of 7.24×/8.96× with no persistent pairwise conflict and no MTRL mitigation, but could not separate task semantics from ER's much smaller per-batch sample count (≈47 versus 539 KS and 1462 SI). DEC-0009 authorized this diagnostic and kept strict Task Relation Learning scope; no mechanism is implemented here.

## Exact change

Arm A1 changes only one thing relative to the matched control A0: ER's sampling weight in the training loader, so ER contributes approximately KS-scale examples per 2048-example batch. Weighted sampling is training-only; validation and test data are untouched. `num_samples` is unchanged, so optimizer steps remain ≈2820 and epochs remain 30.

## Matched control

A0 — fresh plain wavCSE `ks_si_er` at the same commit and seed, standard concatenated sampler, protocol identical to DG-0002's baseline configuration.

## Evaluation protocol

Primary evidence is the seed-level phase summary of shared-parameter gradient norms: max/min mean-norm ratio, ER-to-smallest ratio, task norm means/medians, pairwise cosines, negative-conflict frequencies, and per-step realized valid-example counts. Fixed-final-epoch accuracy is context only. Ordinary-split ER is speaker-leaky; no ER performance claim is made and LOSO is not required for this diagnostic.

## Promotion/rejection criterion

Promote when the exposure check passes (same steps, same total samples, realized ER per-batch count ≥ 0.5× KS's) and A1's late ratio is `< 3.0` while A0's is `≥ 3.0`. Reject H1 if A1's middle and late ratios both stay `≥ 3.0`. If ambiguous, run the pre-registered reverse-direction arm A2 (reduce KS/SI training contribution to ER scale) before concluding.

## Git commit

Pending: implementation commit to be recorded here and in every MLflow run tag.

## Pre-registered run names

`DG-0005__screen__wavcse__ks_si_er__smp25__s42` (A0/A1 method tag distinguishes arms; confirmation uses `__confirm__` with the standard seed suffix).

## Results

Pending execution.

## Interpretation

Pending execution.

## Decision

Pending execution.

## Next step

Implement the additive, default-off per-task sampling knob, commit it, then run screening (A0 on GPU 0, A1 on GPU 1), monitor to completion, and analyze seed-level phase summaries. Report the outcome to the human before any Option-3 discussion.
