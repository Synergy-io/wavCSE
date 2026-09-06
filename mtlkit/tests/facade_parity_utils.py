"""Facade-parity test infrastructure (Next Step 5 / Eng Review cross-model
tension 5): import `downstream/` as it existed at a fixed pre-rewrite git
commit, in isolation from the live (post-rewrite) `downstream/` modules of
the same name, so a test can construct BOTH and assert identical behavior.

Captured once, dynamically, from git history -- not duplicated into the
repo as frozen source, so there is exactly one place drift could hide (the
git ref itself, which never moves) instead of a second hand-maintained copy
that itself needs keeping in sync.
"""

import contextlib
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The commit immediately before mtlkit/ existed and before downstream/ was
# touched by Next Step 5's wrapper rewrite -- "today's pre-refactor code"
# per the design doc's Success Criteria.
PRE_REFACTOR_COMMIT = "a6b9823"

_COLLIDING_TOP_LEVEL_NAMES = ("pooling", "model", "trainer", "evaluator", "dataset", "utils")


@contextlib.contextmanager
def reference_downstream(commit: str = PRE_REFACTOR_COMMIT):
    """Extract `downstream/` as of `commit` into a temp directory, swap it
    onto `sys.path` in place of the live `downstream/` directory, and purge
    any already-imported top-level modules that would collide (`pooling`,
    `model`, `trainer`, `evaluator`, `dataset`, `utils` -- downstream/'s
    package names). Restores everything on exit, including re-priming
    `sys.modules` with whatever was live before, so later tests in the same
    process see the live modules again, not the frozen reference.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "archive.tar")
        with open(archive_path, "wb") as archive_file:
            subprocess.run(
                ["git", "archive", commit, "downstream"],
                cwd=REPO_ROOT,
                stdout=archive_file,
                check=True,
            )
        subprocess.run(["tar", "-xf", archive_path], cwd=tmpdir, check=True)
        reference_downstream_dir = os.path.join(tmpdir, "downstream")

        saved_modules = {}
        for name in list(sys.modules):
            if name.split(".")[0] in _COLLIDING_TOP_LEVEL_NAMES:
                saved_modules[name] = sys.modules.pop(name)

        saved_path = list(sys.path)
        live_downstream_dir = os.path.join(REPO_ROOT, "downstream")
        sys.path = [reference_downstream_dir] + [
            p for p in sys.path if os.path.abspath(p) != live_downstream_dir
        ]

        try:
            yield reference_downstream_dir
        finally:
            for name in list(sys.modules):
                if name.split(".")[0] in _COLLIDING_TOP_LEVEL_NAMES:
                    del sys.modules[name]
            sys.path = saved_path
            sys.modules.update(saved_modules)
