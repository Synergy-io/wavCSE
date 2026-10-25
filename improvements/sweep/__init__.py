"""Layer-combination sweeps: declare many combos, run as many as fit.

This package exists because the layer axis of the wavCSE representation is a
*downstream load-time slice* (``embedding[transformer_layer_array, :]`` in
``downstream/dataset/preprocess_embedding.py``), so one layer-combination
setting costs one training process and no upstream re-extraction. That makes a
sweep of many combos cheap to *declare* and expensive only in wall clock, which
is exactly the case where running several at once is worth having.

This local runner is not the private ``wavcse_research.execution`` authorized
paid-compute backend. Remote commands address only a declared, non-billable
static SSH host through standalone ``infra worker`` transport; they refuse
RunPod and Colab. The supervisor enforces ``policy.max_wall_seconds`` and
reports elapsed time, but it is not a spend envelope.

Compared with a recorded infrastructure job:
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
