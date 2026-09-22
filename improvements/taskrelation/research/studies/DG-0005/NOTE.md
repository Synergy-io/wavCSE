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

## Screening result

The decisive A0/A1 screen completed at commit `0162224`, seed 42. Configuration and exposure checks passed: both arms used 2,820 steps and 142 samples; A1 realized KS/SI/ER counts `439.23 / 1173.99 / 434.79` per sampled batch (ER/KS `0.990`). A0 middle/late max-min norm ratios were `7.357 / 7.661`; A1 reduced them to `2.822 / 2.756`, crossing the pre-registered `< 3.0` late threshold. Fixed-epoch A1-minus-A0 deltas were aggregate `-0.00019`, KS `+0.00015`, SI `-0.00133`, ER `+0.01266`; the ER value is speaker-leaky context only. Screen run IDs: A0 `5cda2a0b01d54470bc5d04fba641d756`, A1 `301215870efc46b9a7688eb8722c1d17`.

## Screening interpretation

The screening direction supports a data-regime-sensitive explanation for F9: ER's late mean norm fell from `6.363` to `1.906`, while KS/SI did not inflate enough to explain the ratio collapse. One seed cannot establish the effect. Replacement sampling also increases ER repetition and reduces KS/SI counts under the fixed global batch, so confirmation must quantify seed stability before F9 is revised.

## Confirmation result

Ten matched runs completed at commit `8032a937050d8bbd3114b172cb813a8fc7370b37`; all five seed pairs passed the exposure gate (A1 realized ER/KS sampled-batch count ratio `0.990–1.008`, both arms 2,820 optimizer steps and 142 diagnostic samples).

| Phase | A0 max/min norm ratio | A1 max/min norm ratio | Paired A1−A0 | 95% t interval |
| --- | --- | --- | ---: | --- |
| middle | `7.243 ± 0.272` | `2.818 ± 0.272` | `−4.425` | `[−4.980, −3.869]` |
| late | `8.962 ± 0.456` | `2.352 ± 0.270` | `−6.610` | `[−7.353, −5.867]` |

A0 late ratio exceeded 3.0 and A1 fell below it in `5/5` seeds, satisfying the pre-registered support condition. Late mean shared-parameter norms moved ER `6.972 → 1.608`, KS `0.828 → 0.894`, SI `0.804 → 0.687`: the collapse is localized to ER rather than produced by KS/SI. Late pairwise cosines stayed near zero in both arms (A0 `+0.0022 / +0.0013 / +0.0221`; A1 `+0.0020 / +0.0019 / +0.0027`), so no conflict mechanism is involved. The A0 arms reproduce DG-0002's baseline phase summaries exactly, confirming the sampling knob is default-off and byte-equivalent when unused.

One pre-registered nuance: the upper bound is on the late phase only. A1's middle ratio stayed below 3.0 in `4/5` seeds and was `3.054` in one, so "dominance removed" is solid for late and borderline for middle.

## Interpretation

The reproducible F9 ER gradient-norm dominance is **not task-intrinsic at this representation**; it tracks the training composition. Raising ER from ~47 to ~435 examples per 2048-example batch removes ~77% of ER's late norm and the whole 7–9× ratio. Fixed-epoch ordinary-split accuracy moves are small (aggregate `+0.00106 [+0.00026, +0.00186]`, ER `+0.01049 [+0.00249, +0.01849]`) and remain speaker-leaky context with no claim (F3).

Alternative explanation not yet excluded: the same knob removes ER gradient-estimate variance **and** multiplies ER's optimizer updates per epoch, so part of the norm drop may reflect faster ER convergence/overfitting (ER train accuracy rises to ~0.99 while validation plateaus near 0.82) rather than only estimator scale. The single-knob design cannot separate these two.

## Post-hoc bound (no compute, 2026-09-22)

A pre-registered post-hoc analysis of these runs (`analyze_noise_shape.py`, pre-registered at `2ac7f3d`, results in `noise_shape_result.json`) bounded how much of ER's mean-norm drop estimator size can carry. Since a noise-dominated norm satisfies `E‖g‖ ∝ 1/√n`, the arm-to-arm mean ratio is capped at `√(n_A1/n_A0) ≈ 3.0`; the observed late ratio was `3.99–4.69`, so estimator-size scaling explains at most `0.758` `[0.715, 0.801]` of the late log-drop and **≥20% must be something else** (a smaller ER mean gradient, or noise growing faster than `1/√n`). The middle phase is fully estimator-consistent (share `1.028` `[0.975, 1.082]`). Per-step dispersion cannot proxy estimator variance here: observed `CV ≈ 0.32` against an isotropic-noise ceiling of `≈ 0.00057`. DG-0002's matched baseline reproduces the A0 statistics value for value. This is a bound, not an identification, and it carries no `CONFIRMED` claim; it is recorded in F10 and DEC-0012.

## Decision

`CONFIRMED` diagnostic. No mechanism is authorized; the `TR-xxxx` gate stays closed under DEC-0009 (narrowed by DEC-0012).

## Next step

F9's interpretation is updated and the framework consequence is recorded (gradient scale is a training-mixture property). A human decides the Option-3 gate. If a further diagnostic is opened, its target is now the late-phase mean gradient specifically — measured as `‖E g‖` on the existing arms rather than inferred from per-batch dispersion, per DEC-0012 and `FRAMEWORK.md` §7.
