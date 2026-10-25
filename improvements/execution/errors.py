"""Worker-local failures; no controller or infrastructure import is required."""


class WorkerError(Exception):
    """A public worker cannot establish the declared run or its evidence."""


class ConfigurationError(WorkerError):
    """A plan, input declaration or configuration is invalid."""


class ArtifactIntegrityError(WorkerError):
    """Materialized input bytes are inconsistent with their declared digest."""


class EvidenceError(WorkerError):
    """The run identity or resulting evidence cannot be established."""


class ImplementationBugError(WorkerError):
    """An internal invariant in the worker adapter failed."""
