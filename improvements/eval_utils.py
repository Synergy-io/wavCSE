"""Explicit run identity for evaluation.

`downstream/evaluator/evaluator_model.py` accepts ``results_run_id`` and
``checkpoint_run_id``, but falls back to ``_latest_run_id(root)`` — the
most-recently-modified subdirectory of the root — when they are not supplied.
With more than one run under a root, that can evaluate a *different* run's
checkpoints than the one this process just trained, and nothing in the output
would show it.

`downstream/` is frozen, so the fix is in the callers: every entry point in
`improvements/` passes the identifiers belonging to the trainer it just built.
This module derives those identifiers from the trainer object, so the binding is
to the run this process produced rather than to whatever is newest on disk.
"""

import os


def _run_id_from_dir(directory, attribute):
    if not directory:
        raise RuntimeError(
            "Trainer exposes no {} directory; cannot bind evaluation to this "
            "run's own output.".format(attribute)
        )
    run_id = os.path.basename(os.path.normpath(str(directory)))
    if not run_id or run_id in (".", "..", os.sep):
        raise RuntimeError(
            "Trainer's {} is {!r}, which yields no run identifier; refusing to "
            "fall back to 'latest'.".format(attribute, directory)
        )
    return run_id


def evaluation_run_ids(trainer):
    """Return ``(results_run_id, checkpoint_run_id)`` for the trainer's own run.

    Never returns a 'latest' guess: the values are the basenames of the
    directories the trainer created for this process. Raises ``RuntimeError``
    when the trainer does not expose them.
    """

    results_dir = getattr(trainer, "results_dir", None)
    checkpoint_dir = (
        getattr(trainer, "ckpt_dir", None)
        or getattr(trainer, "checkpoints_dir", None)
    )
    return (
        _run_id_from_dir(results_dir, "results_dir"),
        _run_id_from_dir(checkpoint_dir, "ckpt_dir"),
    )
