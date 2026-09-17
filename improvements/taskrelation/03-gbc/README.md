# wavCSE-GBC — Global Bias Coupling

**2026-09-01 — citation correction.** This architecture used to be
attributed to "Zhang et al. (2010)"/"Bayesian Online Multi-label
Classification (BOMC)." That citation does not hold up:

- The real 2010 Zhang & Yeung paper it pointed to ("Bayesian Online
  Learning for Multi-label and Multi-variate Performance Measures,"
  AISTATS 2010) is about online Gaussian-density-filtering multi-label
  classification — no "global bias"/"hub parameter" coupling concept
  appears in it.
- Zhang & Yang's 2021 MTL survey (arXiv:1707.08114) — the taxonomy this
  whole project's four architecture branches are organized around — does
  not mention any such method either.
- No real published paper matching the description ("a shared global bias
  hub parameter couples all task heads") was found.

**Verdict: GBC is an original design for this project, not an
implementation of a named published method.** The closest real analog in
spirit — a shared parameter all tasks couple through, plus a per-task
adjustment — is **Evgeniou & Pontil, "Regularized Multi-Task Learning"
(KDD 2004)**, which constrains each task's weight vector to a shared vector
`w0` plus a task-specific deviation `v_t`, regularized toward `w0`. The
mechanism actually implemented here is different: Evgeniou & Pontil
regularize full linear weight vectors directly (a loss-level constraint);
GBC instead learns a nonlinear bottleneck projection of the shared hidden
representation into a small "global bias space," combines it with a
learned global parameter, and additively projects that combined signal
into every task head's logits — coupling here is architectural (a shared
computation path), not a loss-level regularizer. See
`gbc_model.py`'s module docstring for the full technical writeup.

This folder is the second `0N-<name>/` self-contained architecture folder
(after `01-mtrl/`) — GBC was promoted out of the flat
`taskrelation/models/`/`configs/` layout (where TSM and PMR remain) so it
gets the same evaluation rigor MTRL got: the pooling-grid-search winner
config, and a multi-seed check before any comparison to baseline is
trusted. **GBC had never been trained in this project before this
folder was created** — no prior results exist to compare against.

## What it does

Shared backbone (unchanged from plain wavCSE) → a learned global bias
vector (`nn.Parameter`) combined with a per-example nonlinear projection of
the shared representation (`global_bias_projector`) forms a "global
coupling" signal → that signal is linearly projected into each task head's
own output space and added to that task's logits, alongside a per-task
learned bias and the task's own classifier weights. `gbc_global_dim`
(default 64) controls the width of the shared coupling space. See
`gbc_model.py` for the exact forward pass.

Known implementation detail worth flagging: `self.classifiers[t]` is a
plain `nn.Linear` (bias included by default) and the forward pass adds a
*separate* `self.task_biases[t]` parameter on top of it — so each task head
effectively has two additive bias terms doing the same job
(`gbc_model.py:91-99,164-166`). This isn't broken (the two just get
summed during training, harmless redundancy), but it's not what the
docstring's "no bias from Linear" comment suggests. Left as-is per the
"citation fix, not a rewrite" scope of this change — flagging it here for
whoever next touches this model.

## Config

`gbc_config.yml` — the historical/original config (`weighted` pooling,
16-layer list), kept unchanged for reference, same as how `01-mtrl/
mtrl_config.yml` keeps its own original settings.

`gbc_poolingwinner_16L_config.yml` / `gbc_poolingwinner_25L_config.yml` —
the pooling-grid-search winner (`smp`, λ=0.5), matching
`01-mtrl/mtrl_poolingwinner_{16,25}L_config.yml`'s non-GBC-specific fields
exactly, for a fair comparison against baseline/MTRL at the same settings.

## Running

```bash
cd wavCSE   # repo root
.venv/bin/python -m improvements.run_improvements --model gbc --task_type ks_si_er \
    --config improvements/taskrelation/03-gbc/gbc_poolingwinner_16L_config.yml --seed 42
```

## Experiment log

<!-- Screening + multi-seed results appended below as they're run. -->
