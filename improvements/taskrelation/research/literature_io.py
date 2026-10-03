"""Atomic sibling-temp writes for the literature durable registries.

``literature_investigation`` and ``literature_record`` both replace a durable
registry (``STUDIES.jsonl``; ``assessments.jsonl`` / ``claims.jsonl`` /
``literature_survey/registry.jsonl``) with the same mechanism: write the whole
new document to a temp file in the target's directory, flush and fsync it, then
``os.replace`` it over the target. Keeping that one mechanism in one place means
the two writers cannot drift in durability or crash-safety behaviour.

Deliberately *not* consolidated here:

* ``literature_admit._atomic_write`` writes **bytes** and removes its staging
  file when the write fails partway; used for a canonical card and the catalog;
* ``literature_primary`` stages the manifest and cache copies with its own
  copy/checksum flow.

Those flows have different failure and integrity semantics (binary payloads,
checksum verification, cleanup-on-error), so they stay with their owners rather
than being bent into this text helper.
"""

import os
import tempfile
from pathlib import Path


def stage_text(path, text):
    """Write ``text`` to a sibling temp file and return its path.

    The temp file is created in the target's directory so ``os.replace`` stays
    on one filesystem. The caller either replaces the target with it or removes
    it on failure; the parent directory is never created here.
    """

    path = Path(path)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    return Path(handle.name)


def atomic_write_text(path, text):
    """Replace ``path`` with ``text`` via a staged sibling temp + ``os.replace``."""

    staged = stage_text(path, text)
    os.replace(str(staged), str(Path(path)))
