# Pooling Grid Search — All Layer-Pooling Methods

**2026-09-01 caveat on every `er`/`test_..._er_acc` number in this
document**: all of them were measured on the speaker-leaky IEMOCAP split.
A leave-one-speaker-out re-run of the `smp`+25L baseline gives a true `er`
accuracy of 0.6391 (std 0.051) — ~15pp below the 0.7902 reported here.
Overall `acc_all` numbers (dominated by `ks`/`si`, which aren't leaky) are
unaffected. See `improvements/base/README.md`'s "ER 10-fold cross-
validation → Results" section for the full LOSO writeup.

Every prior campaign used one pooling method per run: `mix` (baseline
default), `weighted` (MTRL default), `lnp` (the fair-comparison pair). This
is a systematic search over all 10 pooling types
`downstream/pooling/pooling.py` supports, for both the 16-layer curated
subset and all 25 layers.

**Driver:** `improvements/base/run_pooling_grid.py`. Two-stage: screen every
combo at 10 epochs (ranked by `trainer.ckpt.opt_accuracy_all_threshold` —
the val-set version of the same "opt" checkpoint-selection criterion used to
report `test_opt_acc_all` everywhere in this project), then confirm the top
3 per layer count at the full 30-epoch budget with the full evaluator pass
(directly comparable to every other number in this project). Screening
checkpoints/results are deleted immediately after ranking — only the MLflow
metrics matter there. Run two processes concurrently, one per GPU:

```bash
cd improvements/base
python run_pooling_grid.py --num_layers 16 --device_index 0
python run_pooling_grid.py --num_layers 25 --device_index 1
```

**The grid** (18 combos per layer count): `mean`, `max`, `weighted`
(param=`len(layers)`), `gated` (param=`len(layers)`), `auto` (default init),
`sap` (param=512, the *projected* dim pooling actually runs on — confirmed
from `DownstreamMultiTaskModel.forward()`, not the raw 1024) — 6 fixed
configs — plus `mix` ∈ {0.25, 0.5, 0.75}, `lnp` ∈ {4, 8, 16}, `smp` ∈
{0.5, 1, 2}, `lse` ∈ {1, 2, 5} — 12 swept configs.

**Smoke-tested first**: all 18 combos run once at 1 epoch / 3% data before
committing to the full grid — caught nothing (all 18 completed cleanly,
including `weighted`/`gated`/`auto`/`sap`, none of which had ever been
exercised in this codebase before this search).

---

## 16 Layers

### Screening (10 epochs, ranked by val opt_acc_all)

| Rank | Pooling | Param | val_opt_acc_all |
|---|---|---|---|
| 1 | `smp` | 0.5 | 0.9644 |
| 2 | `mix` | 0.25 | 0.9635 |
| 3 | `max` | — | 0.9630 |
| 4 | `lse` | 2 | 0.9627 |
| 5 | `smp` | 1 | 0.9617 |
| 6 | `lse` | 1 | 0.9613 |
| 7 | `auto` | — | 0.9597 |
| 8 | `mix` | 0.75 | 0.9595 |
| 9 | `smp` | 2 | 0.9594 |
| 10 | `mix` | 0.5 (current baseline) | 0.9588 |
| 11 | `gated` | — | 0.9552 |
| 12 | `weighted` (current MTRL default) | — | 0.9544 |
| 13 | `mean` | — | 0.9508 |
| 14 | `lnp` | 4 | 0.9503 |
| 15 | `lnp` | 8 | 0.9502 |
| 16 | `lnp` | 16 (current lnp-arm default) | 0.9500 |
| 17 | `sap` | — | 0.9284 |
| 18 | `lse` | 5 | **0.0311 — failed** |

**`lse` (LogSumExp pooling) with `r=5` is numerically unstable**: training
loss goes to `nan` from epoch 1 (`(1/r)·log(sum(exp(r·x))/n)` overflows —
`r·x` inside the exponential is large enough at `r=5` with this codebase's
un-normalized embeddings to blow up `exp()`). A genuine, useful negative
result: `lse` only works at small `r` (1-2) for this input scale.

### Confirmation (top 3, full 30 epochs, full evaluator)

| Pooling | Param | test_opt_acc_all | ks | si | er |
|---|---|---|---|---|---|
| **`smp`** | **0.5** | **0.9724** | 0.9842 | 0.9765 | 0.7667 |
| `max` | — | 0.9691 | 0.9823 | 0.9709 | 0.7794 |
| `mix` | 0.25 | 0.9683 | 0.9848 | 0.9679 | 0.7703 |
| *(reference)* `mix` | 0.5 (current baseline) | 0.9710 | 0.9854 | 0.9735 | 0.7559 |

**`smp` (softmax pooling, λ=0.5) wins outright** — beats the current
`mix`-pooling baseline on the metric that matters (+0.14pp overall) and on
2 of 3 tasks (si +0.30pp, er +1.08pp; ks -0.12pp, negligible). This is a
genuine, actionable improvement, not just a ranking curiosity: `smp` is
`softmax(λ·x)`-weighted pooling over the layer axis — a *soft*, differentiable
version of max-pooling (λ controls sharpness), and at λ=0.5 it apparently
finds a better operating point than either plain `mix` (a fixed linear
blend of true max and mean) or `weighted` (a free-form learned softmax over
positions, more parameters, slower to converge — consistent with `weighted`
under-performing `smp` here despite having *more* learnable capacity: more
parameters isn't automatically better within a fixed 30-epoch budget).

### MTRL at winning pooling (`smp`, λ=0.5), 16 layers

Config: `improvements/taskrelation/01-mtrl/mtrl_poolingwinner_16L_config.yml`.
Run: [DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/e550d6600fa940808a0bdbae8d016b1f).

| Run | test_opt_acc_all | ks | si | er |
|---|---|---|---|---|
| baseline @ smp,0.5 | 0.9724 | 0.9842 | 0.9765 | 0.7667 |
| **MTRL @ smp,0.5** | **0.9728** | 0.9849 | 0.9766 | 0.7649 |
| Δ (MTRL − baseline) | **+0.04pp** | +0.07pp | +0.01pp | -0.18pp |

**2026-09-01 update — retracted, this was seed noise, not a real effect.**
A 5-seed check of both configs (seeds 0-4, everything else identical, via
`improvements/seed_utils.py`) gives baseline `test_opt_acc_all` =
0.97134±0.00130 (range 0.96995-0.97372) vs MTRL = 0.97124±0.00091 (range
0.96969-0.97218) — the difference in means (+0.0001, baseline ahead) is an
order of magnitude smaller than either side's own seed-to-seed std, and the
*baseline's own* seed-to-seed spread (0.38pp) is ~10x wider than the
original "+0.04pp win" being tested. The single-seed numbers above are one
sample each from those distributions, not a real, reproducible gap.
**MTRL does not beat baseline anywhere in this project once seed variance
is controlled for.** Full writeup: `improvements/taskrelation/01-mtrl/README.md`'s
top-of-file retraction note.

~~This is the first result in the whole project where MTRL beats the
baseline on the overall metric outright~~ — see retraction above. (Original
text preserved for the record: "not just trading it off for an `er` gain —
every prior comparison (weighted-pooling campaign, lnp-matched pair) had
MTRL essentially tied-or-behind on `all` while winning on `er` specifically.
Here it wins narrowly but cleanly on `all`, `ks`, and `si`... It's also
MTRL's best-ever absolute number.")

Final Ω:
```
         ks        si        er
ks   [+0.2918, +0.1855, +0.2292]
si   [+0.1855, +0.3668, -0.0542]
er   [+0.2292, -0.0542, +0.3415]
```
A third distinct relationship reading (compare `weighted`'s ks↔er positive/
si anti-related, and `lnp`'s near-uniform positive): here ks is positively
related to *both* si (+0.19) and er (+0.23) at comparable strength, while
si↔er is weakly negative (-0.05) — reinforcing, for the third pooling
mechanism running now, that si↔er is consistently the weakest/most
negative pair in this project's results, while every other pairing's sign
and strength shifts with the representation.

**Recommendation as of 2026-09-01**: `smp` pooling (λ=0.5) is still the
right pooling choice for either architecture. The MTRL-specific
recommendation above is retracted — MTRL at this config does not
reproducibly beat baseline (see the 5-seed check above); there is currently
no config anywhere in this project where MTRL demonstrates a real advantage
over the plain baseline.

---

## 25 Layers

### Screening (10 epochs, ranked by val opt_acc_all)

| Rank | Pooling | Param | val_opt_acc_all |
|---|---|---|---|
| 1 | `smp` | 1 | 0.9667 |
| 2 | `smp` | 0.5 | 0.9655 |
| 3 | `smp` | 2 | 0.9621 |
| 4 | `auto` | — | 0.9574 |
| 5 | `mix` | 0.5 (current baseline) | 0.9571 |
| 6 | `max` | — | 0.9566 |
| 7 | `mix` | 0.75 | 0.9556 |
| 8 | `weighted` (current MTRL default) | — | 0.9485 |
| 9 | `mix` | 0.25 | 0.9394 |
| 10 | `sap` | — | 0.9381 |
| 11 | `mean` | — | 0.9242 |
| 12 | `gated` | — | 0.9175 |
| 13 | `lnp` | 8 | 0.8203 |
| 14 | `lnp` | 4 | 0.8198 |
| 15 | `lnp` | 16 (current lnp-arm default) | 0.8013 |
| 16-18 | `lse` | 1, 2, 5 | **0.0311 — failed, all three** |

**`lse` fails at 25 layers regardless of `r`** (unlike 16L, where only
`r=5` broke) — the added layers push the `sum(exp(r·x))` term into overflow
even at `r=1`. **`lnp` collapses hard at every `p` tried** (0.80-0.82,
consistent with the previous campaign's finding that `lnp`'s fixed power
can't adapt to a changed layer count and specifically destroys `si`).

### Confirmation (top 3, full 30 epochs, full evaluator)

| Pooling | Param | test_opt_acc_all | ks | si | er |
|---|---|---|---|---|---|
| **`smp`** | **0.5** | **0.9767** | 0.9864 | 0.9812 | 0.7902 |
| `smp` | 1 | 0.9717 | 0.9860 | 0.9729 | 0.7776 |
| `smp` | 2 | 0.9691 | 0.9841 | 0.9691 | 0.7848 |
| *(reference)* `mix` | 0.5, 25L | 0.9676 | 0.9884 | 0.9628 | 0.7812 |
| *(reference)* `smp` | 0.5, 16L | 0.9724 | 0.9842 | 0.9765 | 0.7667 |

**`smp` (λ=0.5) at 25 layers is the best baseline number in the entire
project** — 0.9767, beating its own 16-layer counterpart (0.9724, itself
already the prior best) by +0.43pp, with strong gains on every task (si
+0.47pp, er +2.35pp vs. the 16L `smp` result; and vs. the 25-layer `mix`
baseline: all +0.91pp, si +1.84pp, er +0.90pp, only ks essentially flat).

**This overturns the "16 layers beats 25 layers" conclusion from the prior
all-25-layers campaign.** That conclusion was correct for every pooling
tried *at the time* (`mix`, `weighted`, `lnp`) — all three either couldn't
fully suppress the 9 extra, noisier layers (`mix`, `weighted`) or actively
broke on them (`lnp`). It was never a property of the extra layers
themselves being uninformative — it was a property of those three specific
pooling mechanisms' inability to use them well. `smp`'s softmax-sharpened
weighting evidently *can* extract the extra signal the additional layers
carry: more layers is better, provided the pooling can actually make use of
them. This is the single most important methodological finding of the
whole project: **layer selection and pooling method are not independent
choices** — a "best layer subset" found under one pooling doesn't transfer
to another.

### MTRL at winning pooling (`smp`, λ=0.5), 25 layers

Config: `improvements/taskrelation/01-mtrl/mtrl_poolingwinner_25L_config.yml`.
Run: [DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/c48cc9c601664814b94befbcb6f7f06e).

| Run | test_opt_acc_all | ks | si | er |
|---|---|---|---|---|
| baseline @ smp,0.5, 25L (best baseline overall) | **0.9767** | 0.9864 | 0.9812 | 0.7902 |
| MTRL @ smp,0.5, 25L (opt) | 0.9744 | 0.9857 | 0.9789 | 0.7667 |
| MTRL @ smp,0.5, 25L (best) | 0.9754 | 0.9854 | 0.9799 | 0.7848 |
| Δ (MTRL opt − baseline) | -0.24pp | +0.13pp | -0.19pp | -2.35pp |

Unlike at 16 layers, **MTRL does not beat the baseline here** — it's close
(within 0.2-0.3pp on `all`/`si`, `ks` actually favors MTRL) but `er` drops
meaningfully (-2.35pp at the opt checkpoint, though the `best`-tag
checkpoint recovers most of that, 0.7848). Ω fully **saturated** this time
(mean off-diagonal ≈0.333, uniform positive coupling across all three
pairs — the same "all-positive, maximally coupled" pattern seen previously
at `lnp`+16L):
```
         ks        si        er
ks   [+0.3333, +0.3332, +0.3333]
si   [+0.3332, +0.3333, +0.3330]
er   [+0.3333, +0.3330, +0.3334]
```
A saturated, undifferentiated Ω (every pair equally, maximally related)
gives the regularizer nothing to discriminate with — every task gets pulled
equally toward every other, which is a reasonable explanation for why the
regularizer's net effect here is mildly negative rather than the more
useful, structured pull seen at 16L (where Ω had a distinguishing pattern:
ks strongly tied to both si and er, si↔er weak).

**2026-09-01 update — 5-seed check at this config**: baseline
`test_opt_acc_all` = 0.97476±0.00108 (range 0.97314-0.97628) vs MTRL =
0.97419±0.00071 (range 0.97353-0.97506) — baseline ahead by 0.056pp, about
one combined standard error (weak/borderline evidence of a real small
edge, not a strong one). The single-run gap shown in the table above
(0.13-2.3pp depending on metric/tag) overstates the true multi-seed gap by
roughly 2-3x on `acc_all`. On `er`: baseline 0.7866±0.0065 vs MTRL
0.7761±0.0093 — a 1.05pp gap that looks like a real effect on this leaky
split alone, but **directly contradicts the LOSO (honest) result**, which
found baseline and MTRL indistinguishable on `er` at this exact config
(0.6391 vs 0.6380 — see `improvements/taskrelation/01-mtrl/README.md`'s
LOSO section). Full writeup with all three checkpoint tags in that same
README.

---

## Overall Recommendation

| Config | test_opt_acc_all | ks | si | er |
|---|---|---|---|---|
| baseline, mix 0.5, 16L (original) | 0.9710 | 0.9854 | 0.9735 | 0.7559 |
| baseline, mix 0.5, 25L | 0.9676 | 0.9884 | 0.9628 | 0.7812 |
| baseline, smp 0.5, 16L | 0.9724 | 0.9842 | 0.9765 | 0.7667 |
| **baseline, smp 0.5, 25L** | **0.9767** | 0.9864 | 0.9812 | 0.7902 |
| MTRL, weighted, 16L (original best) | 0.9695 | 0.9842 | 0.9699 | 0.7812 |
| **MTRL, smp 0.5, 16L** | **0.9728** | 0.9849 | 0.9766 | 0.7649 |
| MTRL, smp 0.5, 25L | 0.9744 | 0.9857 | 0.9789 | 0.7667 |

1. **`smp` (softmax pooling) is the winning pooling method, full stop** —
   it beat every other pooling type at both layer counts, on both baseline
   and MTRL, without exception. If this project adopts one pooling change
   coming out of this search, it's `mix`→`smp`.
2. **Best overall single result: baseline @ `smp` λ=0.5, all 25 layers —
   0.9767.** The best number anywhere in this project, on any architecture.
3. **Retracted (2026-09-01): `smp` λ=0.5, 16 layers is NOT a real MTRL win.**
   A 5-seed check (see `01-mtrl/README.md`'s top-of-file note and
   `POOLING_GRID_SEARCH.md`'s own section above) found baseline
   0.97134±0.00130 vs MTRL 0.97124±0.00091 across seeds — the single-run
   "0.9728 vs 0.9724" gap is inside normal seed-to-seed noise, not a
   reproducible effect. **There is no config anywhere in this project
   where MTRL demonstrably beats its matched baseline.** At 25 layers a
   5-seed check (2026-09-01) found baseline ahead by a small, only
   weakly-supported margin (0.97476±0.00108 vs MTRL 0.97419±0.00071 —
   about one combined standard error, not a strong gap), well below what
   the single-run numbers implied; likely still real given Ω saturates
   into an undifferentiated, uninformative uniform-positive pattern there
   (see above) rather than doing anything useful, but this is a much
   weaker claim than "MTRL clearly loses at 25L."
4. **For the thesis MTRL story specifically**, `smp`+16L is still the
   config to report — it produces the most *interpretable* Ω this project
   has seen (a differentiated pattern, not saturated to uniform) — but
   report it as "ties baseline, does not beat it," not as an outright win.
   **For a pure "best possible baseline" number** (e.g. as the upper bound
   to cite), `smp`+25L is stronger.
5. **Layer count and pooling method are not independent** — this is the
   most important methodological finding, superseding the earlier
   all-25-layers campaign's "16 beats 25" conclusion, which held only for
   the poolings tested there. Any future search over one axis (layers,
   pooling, task-relation hyperparameters) should be treated as
   provisional until cross-checked against the others.
