# TR-0013 Run Note

Status: REGISTERED — screen approved by the researcher and executed under
`authorizations/TR-0013.yaml`.
Type: mechanism / protocol correction inside a published arm (family B of DEC-0013/LT-0002).
Created: 2026-09-29.
Research family: Task Relation Learning.
Protocol: `../../VARIANT_BENCHMARK_PROTOCOL.md` (frozen; identical to TR-0007's screen).
Pre-registration: `PLAN.md`.

## Why this study exists

TR-0007 closed `REJECTED` on the faithful published p-MSSL arm, but at a `lambda_2` the
researcher fixed (`0.01`, DEC-0016) rather than a value the source paper's own procedure
would have chosen: Algorithm 1 selects `lambda_1`/`lambda_2` on data, and its classification
experiments cross-validate over `{0.01, 0.1, 1, 10, 100}` (JMLR 17(33) §4.1), with no
published default and a numeric scale that is not transferable between representations.
TR-0007's own `REJECTED` is therefore scoped to that fixed value, and its recorded mechanism
(coupling-scale domination: summary cosines ≈ 0.981, `S` near-singular, Ω trace 1.13e5, the
coupling's gradient ≈ 1e5 against task gradients ≈ 1e-2, plateau from epoch ≈ 10) is what
TR-0013 must test against the last faithful explanation left open.

TR-0013 changes **one** thing: the `lambda_2` axis becomes the arm's own pre-registered
procedure — validation-selected over the paper's two smallest grid values `{0.01, 0.1}`.
Nothing else moves: not `lambda_0`, not the loss normalization, not the summary adapter, not
the schedule, not the architecture, not the protocol, not the seed. TR-0012's
scale-commensurate coupling is **not** introduced here; the researcher has not approved it for
paid execution.

## Hypothesis and competing explanation (pre-registered)

**H-λ.** A validation-selected `lambda_2` restores a resolvable run — either by removing the
harmful edge or by shrinking the precision until the coupling stops dominating — so the
outcome becomes attributable to the relation object rather than to its own regularizer.

**H-scale (competing, the synthesis's prediction).** `lambda_2` cannot: the coupling's
magnitude is `lambda_2`-invariant for every value that retains an edge, and the value that
removes all edges still carries `3d` on the diagonal. Under H-scale the arm closes as a
faithful negative result — which is exactly what would make a scale-convention deviation
(`TR-0012`) scientifically necessary rather than convenient.

Full falsifiers, the selection rule, the tie rule and the decision rule: `PLAN.md` §3, §6, §7.

## Registered runs (seed 42, one worker, sequential)

| Arm | Config | Experiment | `lambda_2` |
|---|---|---|---|
| `mssl-l2-0p01` | `configs/mssl-l2-0p01.yml` | `taskrelation-mssl` | 0.01 |
| `mssl-l2-0p1` | `configs/mssl-l2-0p1.yml` | `taskrelation-mssl` | 0.1 |
| `classical-mtrl` | `configs/classical-mtrl.yml` | `taskrelation-mtrl` | n/a |
| `wavcse-baseline` | `configs/wavcse-baseline.yml` | `wavcse-baseline` | n/a |

Controls are TR-0007's own control configs, re-identified for this study and otherwise
untouched (`DEC-0017` retains historical MTRL as the §2 in-category control). Both candidate
values are recorded even though only the validation-selected one enters the classification:
"record every attempted configuration".

## Authorization (human grant, this session)

`authorizations/TR-0013.yaml`, scope `TR-0013` only: 0.80 USD/GPU-hour, 3.00 USD total, 4 paid
wall-clock hours, one worker, replacements allowed, the existing network volume allowed, no new
persistent resources, container disk ≤ 120 GB, destroy on completion. It authorizes this screen
only — **not** TR-0012, TR-0008, DG-0007, confirmation, LOSO, any other λ value, or any
architecture exploration.

## Measurement bundle

Per epoch, in the run's own mechanism artifacts (`omega_history.json`, `coupling_scale.json`,
logged to MLflow and copied into `mechanism/`): `lambda_2` and its selection rule; Ω — raw
matrix, trace, eigenvalues, mean |off-diagonal|, exact off-diagonal zeros, support/edge count,
partial correlations, ADMM dual violation and relative duality gap; summary — Gram, cosines,
Gram eigenvalues, row norms (normalised and raw); coupling value beside the task loss;
relation-gradient norm, task-gradient norm (all parameters and classifier heads) and their
ratio; coupling onset epoch. Post-hoc, from the stored checkpoints and metric curves: the
summary geometry of every arm (including the controls) and the plateau shape.

The in-run instrumentation is proven read-only by
`research/tests/test_mssl_mechanism_probe.py`: a training step with the probe is bit-identical
to the same step without it.

## Result

Recorded after execution — see `analysis.md`, `result.json`, and the appended section below.

---

# 2026-09-29 — Screen executed (result appended, pre-registration untouched)

Appended, never substituted: everything above is the state of the study before paid execution.

_To be completed at collection; the pre-registered classification and the mechanism
measurements are recorded here verbatim from the runs._
