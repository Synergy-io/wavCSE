"""Report the run identity a training process just created.

Evaluation, staging and analysis all need to know *which* run this process
produced. Deriving it from "the newest directory under the root" is an
inference that silently breaks when two runs share a root, so the process that
created the run states it explicitly instead:

* it prints one ``ARC_RUN_IDENTITY`` line to stdout, and
* it appends the same record to the file named by ``ARC_RUN_IDENTITY_FILE``
  when that variable is set, which is how the job wrapper reads it without
  parsing stdout.

The record is emitted as soon as the trainer has created its directories, so a
run that dies mid-training still identifies itself.
"""

import json
import os

TOKEN = "ARC_RUN_IDENTITY"
ENV_FILE = "ARC_RUN_IDENTITY_FILE"


def identity_record(trainer, *, model=None, task_type=None, seed=None, extra=None):
    """Build the identity record for one trainer."""

    results_dir = getattr(trainer, "results_dir", None)
    checkpoints_dir = (
        getattr(trainer, "ckpt_dir", None)
        or getattr(trainer, "checkpoints_dir", None)
    )
    if not results_dir or not checkpoints_dir:
        raise RuntimeError(
            "trainer exposes no results/checkpoint directory; cannot report a run "
            "identity"
        )
    record = {
        "results_dir": os.path.abspath(results_dir),
        "checkpoints_dir": os.path.abspath(checkpoints_dir),
        "run_id": os.path.basename(os.path.normpath(results_dir)),
        "checkpoint_run_id": os.path.basename(os.path.normpath(checkpoints_dir)),
    }
    if model is not None:
        record["model"] = model
    if task_type is not None:
        record["task_type"] = task_type
    if seed is not None:
        record["seed"] = seed
    if extra:
        record.update({key: value for key, value in extra.items()})
    return record


def emit_run_identity(trainer, *, model=None, task_type=None, seed=None, extra=None):
    """Print and (when configured) persist the run identity."""

    record = identity_record(
        trainer, model=model, task_type=task_type, seed=seed, extra=extra
    )
    line = TOKEN + " " + json.dumps(record, sort_keys=True, ensure_ascii=False)
    print(line, flush=True)
    destination = os.environ.get(ENV_FILE, "").strip()
    if destination:
        directory = os.path.dirname(os.path.abspath(destination))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(destination, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return record


def read_identity_file(path):
    """Read the identity records a training process wrote, oldest first."""

    records = []
    if not path or not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records
