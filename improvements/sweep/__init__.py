"""Layer-combination sweeps: declare many combos, run as many as fit.

This package exists because the layer axis of the wavCSE representation is a
*downstream load-time slice* (``embedding[transformer_layer_array, :]`` in
``downstream/dataset/preprocess_embedding.py``), so one layer-combination
setting costs one training process and no upstream re-extraction. That makes a
sweep of many combos cheap to *declare* and expensive only in wall clock, which
is exactly the case where running several at once is worth having.

It deliberately does not use ``improvements.compute``. That backend is a paid
compute contract: one authorized envelope, one deterministic job per
(arm, seed), one job in flight per scope, evidence staged and digest-verified
by its own validator. A sweep wants the opposite shape -- many small trainings
concurrently on one hired device, admitted and drained by live RAM/GPU
pressure -- so this package is a direct runner instead. The trade is recorded
rather than hidden:

* there is no envelope, so nothing bounds spend except wall clock; the
  supervisor enforces ``policy.max_wall_seconds`` and reports elapsed time.
* there is no ``jobspec``/evidence validator, so each run's outputs are hashed
  into the sweep manifest instead of being re-derived by the control plane.
* there is no ``reap`` timer; a drain flag and the supervisor's own loop are
  what end a sweep.

What it does keep from the backend's design, because those parts are about
correctness rather than about billing: deterministic per-run identity (the same
idea as ``jobspec.job_key``), an append-only ledger so a restart resumes from
proven state, dry-run by default, and an exact-commit binding (a sweep refuses
to generate configs or start from a dirty tree).
"""

from improvements.sweep import (config_gen, ledger, manifest, remote, report,
                                resources, scheduler, supervisor)

__all__ = ["manifest", "config_gen", "resources", "scheduler", "ledger",
           "supervisor", "remote", "report"]
