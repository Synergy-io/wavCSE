# DG-0005 screening analysis

## Decision

**PROMISING — confirmation required.** Seed 42 satisfies the pre-registered H1 screening criterion. This is not a confirmed finding and does not authorize a mechanism.

## Protocol validity

Both decisive arms finished at commit `0162224e63c1f77d3aaa7228bc2a13f3f2d85b14`, seed 42, with matched `smp(0.5)` 25-layer configs, 30 epochs, batch 2048, 2,820 optimizer steps, 142 diagnostic samples, and 1,550,800 shared parameters. The configuration and exposure checks passed.

A0 used the standard composition. A1 used replacement sampling with ER weight 11.5 and unchanged `num_samples`. Mean valid examples per sampled batch were:

| Arm | KS | SI | ER | ER / KS |
| --- | ---: | ---: | ---: | ---: |
| A0 standard | 539.10 | 1461.68 | 47.22 | 0.088 |
| A1 ER-weighted | 439.23 | 1173.99 | 434.79 | 0.990 |

A1 therefore met the pre-registered `ER >= 0.5 x KS` gate while retaining 2,048 examples per batch and the same step count. Validation and test datasets were not sampled.

The first A1 launch at commit `8159696` failed before its first optimizer step because IEMOCAP's component dataset is `Subset`-wrapped. Commit `0162224` corrected membership discovery and added an actual-data validation. The pre-fix A0 is retained as execution evidence but excluded from paired inference; both decisive arms were rerun at `0162224`.

## Primary result

| Phase | A0 max/min norm ratio | A1 max/min norm ratio | A1 - A0 |
| --- | ---: | ---: | ---: |
| early | 2.605 | 2.018 | -0.587 |
| middle | 7.357 | 2.822 | -4.536 |
| late | 7.661 | 2.756 | -4.905 |

A1 moved the late ratio below the pre-registered 3.0 threshold while A0 remained above it. The middle ratio also moved below 3.0. The change came primarily from ER: its late mean shared-gradient norm fell from 6.363 to 1.906; KS changed from 0.864 to 0.878 and SI from 0.831 to 0.692.

Pairwise late cosines remained near zero rather than revealing persistent conflict: A1 KS-SI `+0.0019`, KS-ER `+0.0141`, SI-ER `-0.0110`. The result changes gradient scale, not the prior F9 conclusion about conflict.

## Outcome context

Fixed-final-epoch A1-minus-A0 accuracy deltas were aggregate `-0.00019`, KS `+0.00015`, SI `-0.00133`, and ordinary-split ER `+0.01266`. These are single-seed screening values. ER's split is speaker-leaky, so the ER delta supports no performance claim and does not trigger LOSO in this diagnostic.

ER's final train-validation accuracy gap increased from 0.1345 to 0.1738 under replacement oversampling. This is consistent with stronger repetition/overfit risk and prevents interpreting the ordinary-split accuracy movement as a generalization benefit.

## Interpretation

The screening result supports H1's directional prediction: ER's 7-8x middle/late norm dominance is not invariant to its per-batch data regime. Raising ER to KS-scale sampling reduced the dominance ratio below 3 without creating pairwise conflict.

The mechanism is not fully isolated to variance alone. Fixed batch size means A1 simultaneously raises ER from about 47 to 435 examples and reduces KS/SI counts; replacement sampling also repeats the same small ER pool more often. However, KS/SI late norms did not inflate enough to explain the ratio collapse, while ER's norm fell by about 70%. The most defensible screening interpretation is a data-regime-sensitive scale effect, not yet a pure estimator-variance causal estimate.

## Alternative explanations

1. A single seed may overstate the magnitude or threshold crossing (F1).
2. Replacement oversampling changes ER example repetition and the training trajectory, not only instantaneous gradient-estimate variance.
3. Reducing KS/SI batch counts is inseparable from raising ER under fixed global batch size; the pre-registered A2 reverse arm remains reserved for an ambiguous result, which this screen is not.
4. Later learning-rate trajectories can diverge through validation behavior even though scheduler configuration is matched.

## Next action

Run matched A0/A1 confirmation at seeds 0-4. Treat seeds as independent units; report mean/SD, paired differences and sign consistency for middle/late ratios. A2 is not launched because the pre-registered H1 screening criterion was met. No Task Relation Learning mechanism is authorized by this screen.
