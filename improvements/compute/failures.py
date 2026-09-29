"""Failure taxonomy, retry bounds, and what each class permits.

Three rules this module exists to enforce:

* **A scientific failure is evidence.** A run that completed, wrote its outputs
  and produced a negative result is never retried and never "repaired". Only
  execution failures have dispositions.
* **OOM never changes the science.** A resource failure may move the *same* job
  spec to a compatible, still-authorized resource; it may never alter batch
  size, precision, model, checkpoint policy, pooling, layer set, dataset
  membership, splits, preprocessing or label mapping.
* **Retries are bounded and counted in durable state.** Attempt counts live in
  the run ledger, so restarting the agent cannot reset a retry budget any more
  than it can reset spend.
"""

from improvements.compute.errors import (
    ArtifactIntegrityError,
    AuthorizationError,
    CapacityError,
    ComputeError,
    ConfigurationError,
    CostError,
    ImplementationBugError,
    ReconcilableError,
    RepositoryConflictError,
    ScientificFailureError,
    SecurityError,
    TransientInfraError,
)

# The taxonomy. Names are stable and appear in the run ledger and in gate reports.
TRANSIENT_INFRA = "TRANSIENT_INFRA"
CAPACITY = "CAPACITY"
RESOURCE_OOM = "RESOURCE_OOM"
IMPLEMENTATION_BUG = "IMPLEMENTATION_BUG"
SCIENTIFIC_FAILURE = "SCIENTIFIC_FAILURE"
ARTIFACT_INTEGRITY = "ARTIFACT_INTEGRITY"
AUTHORIZATION = "AUTHORIZATION"
COST = "COST"
REPOSITORY_CONFLICT = "REPOSITORY_CONFLICT"
SECURITY = "SECURITY"

CLASSES = (
    TRANSIENT_INFRA, CAPACITY, RESOURCE_OOM, IMPLEMENTATION_BUG,
    SCIENTIFIC_FAILURE, ARTIFACT_INTEGRITY, AUTHORIZATION, COST,
    REPOSITORY_CONFLICT, SECURITY,
)

# Bounded attempts per class. A class absent here is never retried automatically.
MAX_ATTEMPTS = {
    TRANSIENT_INFRA: 3,
    CAPACITY: 2,
    RESOURCE_OOM: 2,
    ARTIFACT_INTEGRITY: 2,
}

# Marker text in a job log for an out-of-memory kill, which is the only
# resource failure that may legitimately change the *resource*.
_OOM_MARKERS = (
    "CUDA out of memory",
    "out of memory",
    "OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
    "Killed",
)

# Exit codes that mean "the OS killed the process", not "the science said no".
_SIGNAL_EXIT_CODES = (137, 143, 139)

_EXCEPTION_CLASSES = (
    (ReconcilableError, TRANSIENT_INFRA, True),
    (TransientInfraError, TRANSIENT_INFRA, True),
    (CapacityError, CAPACITY, True),
    (CostError, COST, False),
    (AuthorizationError, AUTHORIZATION, False),
    (ArtifactIntegrityError, ARTIFACT_INTEGRITY, True),
    (SecurityError, SECURITY, False),
    (RepositoryConflictError, REPOSITORY_CONFLICT, False),
    (ConfigurationError, REPOSITORY_CONFLICT, False),
    (ImplementationBugError, IMPLEMENTATION_BUG, False),
    (ScientificFailureError, SCIENTIFIC_FAILURE, False),
)


class Assessment(object):
    """What the taxonomy says about one failure."""

    def __init__(self, klass, *, retryable, max_attempts, repairable,
                 scientific_change_allowed, reconcile_first=False, reason=""):
        self.klass = klass
        self.retryable = retryable
        self.max_attempts = max_attempts
        self.repairable = repairable
        self.scientific_change_allowed = scientific_change_allowed
        self.reconcile_first = reconcile_first
        self.reason = reason

    def as_dict(self):
        return {
            "class": self.klass,
            "retryable": self.retryable,
            "max_attempts": self.max_attempts,
            "repairable": self.repairable,
            "scientific_change_allowed": self.scientific_change_allowed,
            "reconcile_first": self.reconcile_first,
            "reason": self.reason,
        }


def assess_exception(exc):
    """Classify one exception and state what may be done about it."""

    for klass, name, retryable in _EXCEPTION_CLASSES:
        if isinstance(exc, klass):
            return _build(name, retryable, reason=str(exc)[:400],
                          reconcile_first=isinstance(exc, ReconcilableError))
    if isinstance(exc, ComputeError):
        return Assessment(
            TRANSIENT_INFRA, retryable=False, max_attempts=0, repairable=False,
            scientific_change_allowed=False,
            reason=("unclassified control-plane failure, treated as non-retryable: "
                    "{}".format(str(exc)[:400])),
        )
    return Assessment(
        IMPLEMENTATION_BUG, retryable=False, max_attempts=0, repairable=None,
        scientific_change_allowed=False,
        reason="unclassified exception: {}: {}".format(
            type(exc).__name__, str(exc)[:400]
        ),
    )


def _build(name, retryable, *, reason, reconcile_first=False):
    return Assessment(
        name,
        retryable=retryable and name in MAX_ATTEMPTS,
        max_attempts=MAX_ATTEMPTS.get(name, 0),
        repairable=name == IMPLEMENTATION_BUG,
        scientific_change_allowed=False,
        reconcile_first=reconcile_first,
        reason=reason,
    )


def looks_like_oom(exit_code, log_text):
    """True when the evidence names a memory kill rather than a code failure."""

    if exit_code in _SIGNAL_EXIT_CODES:
        return True
    text = log_text or ""
    return any(marker in text for marker in _OOM_MARKERS)


def assess_job_outcome(record, log_text=None, outputs_verified=True):
    """Classify a finished job record.

    A job that exited zero with its required outputs verified is a *completed
    execution*: whether its metrics are good or bad is the science's business
    and is classified as evidence, never as a defect. Anything else is an
    execution failure and is classified for disposition.
    """

    state = str(record.get("state") or "").upper()
    exit_code = record.get("exit_code")

    if state == "SUCCEEDED" and outputs_verified:
        return Assessment(
            SCIENTIFIC_FAILURE, retryable=False, max_attempts=0, repairable=False,
            scientific_change_allowed=False,
            reason=("the run completed and its outputs verified; a negative "
                    "outcome is evidence, not a defect"),
        )

    if state == "SUCCEEDED" and not outputs_verified:
        return _build(
            ARTIFACT_INTEGRITY, True,
            reason="the job reported success but its required outputs are not "
                   "verified; the artifact cannot be trusted as-is",
        )

    if state == "CANCELLED":
        return Assessment(
            IMPLEMENTATION_BUG, retryable=False, max_attempts=0, repairable=None,
            scientific_change_allowed=False,
            reason="the job was cancelled deliberately; decide explicitly whether "
                   "to re-run before spending again",
        )

    if record.get("worker_absent") or record.get("remote_status") in (
            "worker_absent", "worker_destroyed", "worker_terminating",
            "workspace_absent"):
        return _build(
            TRANSIENT_INFRA, True,
            reason="the worker or its workspace disappeared before the job outcome "
                   "could be verified; this is infrastructure failure, not scientific evidence",
        )

    if looks_like_oom(exit_code, log_text):
        return _build(
            RESOURCE_OOM, True,
            reason=("the process was killed by a resource limit (exit {}); the "
                    "same job spec may move to a compatible, still-authorized "
                    "resource, but no scientific setting may change".format(exit_code)),
        )

    return _build(
        IMPLEMENTATION_BUG, False,
        reason="the job failed with exit {}; a non-zero execution is never a "
               "scientific result".format(exit_code),
    )


def decide_retry(assessment, attempts_so_far, *, reconciled=False):
    """Whether another attempt is permitted, and why not when it is not."""

    if not assessment.retryable:
        return False, "{} is not retryable: {}".format(
            assessment.klass, assessment.reason
        )
    if assessment.reconcile_first and not reconciled:
        return False, ("{} must be reconciled against provider state before any "
                       "retry; a blind retry can duplicate a paid resource or a "
                       "running job".format(assessment.klass))
    if attempts_so_far >= assessment.max_attempts:
        return False, ("{} exhausted its retry budget ({} of {} attempts "
                       "already used)".format(
                           assessment.klass, attempts_so_far,
                           assessment.max_attempts))
    return True, "retry permitted ({} of {} attempts used)".format(
        attempts_so_far, assessment.max_attempts
    )
