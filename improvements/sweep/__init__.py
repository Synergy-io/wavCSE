"""Layer-combination sweeps: declare many combos, run as many as fit.

This package exists because the layer axis of the wavCSE representation is a
*downstream load-time slice* (``embedding[transformer_layer_array, :]`` in
``downstream/dataset/preprocess_embedding.py``), so one layer-combination
setting costs one training process and no upstream re-extraction. That makes a
sweep of many combos cheap to *declare* and expensive only in wall clock, which
is exactly the case where running several at once is worth having.

This legacy local runner is not the private
``wavcse_research.execution`` authorized paid-compute backend: it has no
envelope or crash-independent provider cleanup. It may analyze and execute
layer combinations on an operator-controlled machine, but must not be used
as a substitute for the standalone ``infra`` worker and job lifecycle.

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
