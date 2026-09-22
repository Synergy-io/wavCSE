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

`0162224e63c1f77d3aaa7228bc2a13f3f2d85b14` — additive, default-off task-weighted training sampler, deterministic run-seed plumbing, matched A0/A1 configs, actual-data validation, and support for IEMOCAP's subset-wrapped training dataset.

## Pre-registered run names

`DG-0005__screen__wavcse-standard-composition__ks_si_er__smp25__s42` (A0) and `DG-0005__screen__wavcse-er-weighted-composition__ks_si_er__smp25__s42` (A1).

## Results

The decisive A0/A1 screen completed at commit `0162224`, seed 42. Configuration and exposure checks passed: both arms used 2,820 steps and 142 samples; A1 realized KS/SI/ER counts `439.23 / 1173.99 / 434.79` per sampled batch (ER/KS `0.990`). A0 middle/late max-min norm ratios were `7.357 / 7.661`; A1 reduced them to `2.822 / 2.756`, crossing the pre-registered `< 3.0` late threshold. Fixed-epoch A1-minus-A0 deltas were aggregate `-0.00019`, KS `+0.00015`, SI `-0.00133`, ER `+0.01266`; the ER value is speaker-leaky context only. Screen run IDs: A0 `5cda2a0b01d54470bc5d04fba641d756`, A1 `301215870efc46b9a7688eb8722c1d17`.

## Interpretation

The screening direction supports a data-regime-sensitive explanation for F9: ER's late mean norm fell from `6.363` to `1.906`, while KS/SI did not inflate enough to explain the ratio collapse. One seed cannot establish the effect. Replacement sampling also increases ER repetition and reduces KS/SI counts under the fixed global batch, so confirmation must quantify seed stability before F9 is revised.

## Decision

`PROMISING` screening result; matched multi-seed confirmation is required. No mechanism is authorized.

## Next step

Run A0/A1 confirmation at seeds `0,1,2,3,4`, report mean/SD and paired differences, then decide `CONFIRMED`, `REJECTED`, or `INCONCLUSIVE`. A2 is not launched because the screen was not ambiguous.
