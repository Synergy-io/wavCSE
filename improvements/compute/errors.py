"""Typed failures for the compute backend.

The classes mirror the failure taxonomy in `improvements/compute/failures.py`,
which decides retry bounds and whether a repair is permitted. Keeping the
exception type and the taxonomy class aligned means a caller can classify a
failure without re-parsing provider text.

Nothing here imports `wavcse_infra`: the backend talks to the control plane
through its CLI only.
"""


class ComputeError(Exception):
    """Base class for every compute-backend failure."""


class UsageError(ComputeError):
    """The caller asked for something the backend cannot interpret."""


class ConfigurationError(ComputeError):
    """Local configuration (envelope, request file, state directory) is unusable."""


class AuthorizationError(ComputeError):
    """An authorization envelope is missing, expired, widened, or exhausted."""


class CostError(ComputeError):
    """The proposed action does not fit the authorized cost."""


class CapacityError(ComputeError):
    """No compatible resource satisfies the request inside the envelope."""


class BusyError(ComputeError):
    """Another orchestrator instance holds this scope. Nothing is done twice."""


class TransientInfraError(ComputeError):
    """A bounded, retryable infrastructure failure (transport, throttling, readiness)."""


class ReconcilableError(TransientInfraError):
    """An operation whose outcome is unknown: reconcile against provider state.

    Subclasses :class:`TransientInfraError` so it stays inside the taxonomy, but
    a caller must never blind-retry it. For a billable create the only correct
    response is to reconcile by the generated identity, because a retry is how a
    duplicate paid resource is born.
    """

    def __init__(self, message, *, action=None):
        super(ReconcilableError, self).__init__(message)
        self.action = action


class ArtifactIntegrityError(ComputeError):
    """An artifact's identity cannot be established or does not match."""


class EvidenceError(ComputeError):
    """A stored result's bytes are present but its content is not this run's evidence.

    Distinct from :class:`ArtifactIntegrityError` on purpose: a wrong or missing digest is
    a transfer problem, while bytes that verify and still describe the wrong run, the
    wrong tasks or an impossible value are a *result-identity* problem. Neither is ever
    retried as a scientific failure.
    """


class RepositoryConflictError(ComputeError):
    """Repository state cannot be safely reconciled or is not commit-clean."""


class SecurityError(ComputeError):
    """Credential, host-key, or secret-handling problem. Never retried."""


class ImplementationBugError(ComputeError):
    """A failure inside our own code with an unambiguous intended behaviour."""


class ScientificFailureError(ComputeError):
    """A completed run whose outcome is negative. Evidence, never a defect."""
