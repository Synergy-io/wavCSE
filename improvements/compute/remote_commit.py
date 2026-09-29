"""Can a disposable worker actually obtain the commit a job names?

A job spec pins an exact commit, and the worker checks that commit out of the anonymous
HTTPS remote the spec carries. Nothing in that chain consults this controller's
repository, so a commit that exists only here — committed locally and never published —
is a commit no worker can check out. Creating a paid Pod for it spends money on a job
that cannot start, which is the timeline this module exists to break:

    local commit exists -> authorization valid -> paid worker created ->
    worker cannot fetch the commit -> paid resource wasted

The check therefore runs *before* any billable request, against **the remote the worker
will actually use**, and it answers one question: can that remote serve this exact
object? Two things it deliberately does not do:

* it never accepts an indirect proof. A branch existing, a tag existing, or the object
  being present in this checkout says nothing about what the remote can serve; only a
  remote read does. A commit is reported available when the remote lists it as a ref, or
  when the remote serves the object itself to a real (shallow, blobless) fetch.
* it never changes anything. It does not push, fetch into the working repository, create
  a ref, or touch the remote. Publication is a developer action; when it is required,
  the honest output is an actionable refusal naming the commit, never a quiet
  substitution of a different one.

Absence and indeterminacy are different failures and are reported differently: an
unavailable commit needs a human to publish, while a remote read that could not complete
proves nothing and fails closed as a retryable infrastructure condition.
"""

import os
import re
import shutil
import subprocess
import tempfile

from improvements.compute.errors import (
    RepositoryConflictError,
    TransientInfraError,
)

FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
DEFAULT_TIMEOUT_SECONDS = 180.0


class CommitAvailability(object):
    """What one remote read established about one commit."""

    def __init__(self, commit, repository, *, available, mechanism, detail=""):
        self.commit = commit
        self.repository = repository
        self.available = available
        self.mechanism = mechanism
        self.detail = detail

    def as_dict(self):
        return {
            "commit": self.commit,
            "repository": self.repository,
            "available": self.available,
            "mechanism": self.mechanism,
            "detail": self.detail,
        }


def _run(argv, *, timeout, cwd=None):
    try:
        return subprocess.run(
            argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, universal_newlines=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TransientInfraError(
            "git {} did not finish within {}s, so the commit's availability on the "
            "remote could not be established".format(" ".join(argv[1:4]), timeout)
        ) from exc
    except OSError as exc:
        raise TransientInfraError(
            "git is not usable on this controller ({}), so the commit's availability on "
            "the remote could not be established".format(exc)
        ) from exc


def verify_available(repository, commit, *, timeout=DEFAULT_TIMEOUT_SECONDS, runner=None):
    """Prove whether ``repository`` can serve ``commit``.

    Three proofs, cheapest first, and each one is a fact about the *remote* rather than
    about this checkout:

    1. the remote advertises the commit as a ref — one round trip, and the case every
       published commit takes;
    2. the remote serves the exact object to a shallow, blobless fetch — the case where
       the commit sits behind a moved branch tip but the server allows a
       `want` for an unadvertised object;
    3. the remote serves the object at all, proven by a treeless fetch of every head
       followed by a local object check — the same fallback the worker job runner uses
       when a direct fetch of the commit fails, so a server that cannot answer a bare
       `want` is not mistaken for a remote that does not have the commit.

    ``runner`` is a seam for tests; production uses :func:`_run`.
    """

    run = runner or _run
    commit = str(commit).strip().lower()
    if not FULL_COMMIT.match(commit):
        raise RepositoryConflictError(
            "an exact full commit is required to prove remote availability, got {!r}; "
            "branches, tags, short prefixes and HEAD are not reproducible".format(commit)
        )
    if not repository:
        raise RepositoryConflictError(
            "the plan declares no source repository, so no worker could check out {}"
            .format(commit)
        )

    listed = run(["git", "ls-remote", repository], timeout=timeout)
    if listed.returncode != 0:
        raise TransientInfraError(
            "could not read {} to check whether it carries commit {} ({}), so its "
            "availability is unproven; refusing to spend on an unproven commit".format(
                repository, commit, (listed.stderr or "").strip()[:400]
            )
        )
    for line in (listed.stdout or "").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].lower() == commit:
            return CommitAvailability(
                commit, repository, available=True, mechanism="ls-remote-ref",
                detail="the remote advertises this commit as {}".format(fields[1]),
            )

    scratch = tempfile.mkdtemp(prefix="arc-commit-check-")
    try:
        bare = os.path.join(scratch, "probe.git")
        init = run(["git", "init", "--bare", "--quiet", bare], timeout=timeout)
        if init.returncode != 0:
            raise TransientInfraError(
                "could not prepare a scratch repository in {}: {}".format(
                    scratch, (init.stderr or "").strip()[:400]
                )
            )
        # Shallow and blobless: enough to prove the remote serves this exact object,
        # small enough that the proof never becomes the expensive part of the plan.
        direct = run(
            ["git", "-C", bare, "fetch", "--quiet", "--no-tags", "--depth=1",
             "--filter=blob:none", repository, commit],
            timeout=timeout,
        )
        if direct.returncode == 0:
            return CommitAvailability(
                commit, repository, available=True, mechanism="shallow-fetch",
                detail="the remote served this exact object to a shallow fetch",
            )

        # The server would not answer a bare object want. Fetch every head without
        # blobs — the same fallback the worker runner performs — and ask the objects
        # directly, so a nervous server is never reported as a missing commit.
        fallback = run(
            ["git", "-C", bare, "fetch", "--quiet", "--no-tags", "--filter=blob:none",
             repository, "+refs/heads/*:refs/probe/heads/*"],
            timeout=timeout,
        )
        if fallback.returncode != 0:
            raise TransientInfraError(
                "the fetch of {} from {} did not complete ({}), so commit {}'s "
                "availability could not be established; refusing to spend on an "
                "unproven commit".format(
                    repository, repository,
                    ((fallback.stderr or "") + (fallback.stdout or "")).strip()[:400],
                    commit,
                )
            )
        present = run(["git", "-C", bare, "cat-file", "-e", commit + "^{commit}"],
                      timeout=timeout)
        if present.returncode == 0:
            return CommitAvailability(
                commit, repository, available=True, mechanism="reachable-fetch",
                detail="the remote served this exact object while fetching its heads",
            )
        return CommitAvailability(
            commit, repository, available=False, mechanism="reachable-fetch",
            detail="the remote served every head and this object was not among them",
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def require_available(repository, commit, *, timeout=DEFAULT_TIMEOUT_SECONDS, runner=None):
    """Verify availability and raise the actionable refusal when it fails."""

    availability = verify_available(repository, commit, timeout=timeout, runner=runner)
    if not availability.available:
        raise RepositoryConflictError(
            "commit {} is not available on the source remote a disposable worker would "
            "clone ({}). The worker checks out that exact commit, so it could not start "
            "this job. Publishing the commit (git push) is a developer action and is "
            "never taken automatically, and no substitute commit is chosen here. Push "
            "{} to {}, then re-run.".format(
                availability.commit, availability.repository,
                availability.commit[:12], availability.repository,
            )
        )
    return availability


def require_plan_commit(plan, commit, *, timeout=DEFAULT_TIMEOUT_SECONDS, runner=None):
    """The preflight a paid worker creation runs, before anything is requested."""

    return require_available(
        plan.get("repository"), commit, timeout=timeout, runner=runner
    )
