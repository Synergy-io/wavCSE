"""Execution-scope identity and kind.

An authorized execution scope is a short identity plus an explicit kind, and
the kind decides what the scope is allowed to mean:

* ``study`` — a research Study. The scope identity *is* the Study id, its jobs
  are research jobs, and the evidence it produces carries the registered
  protocol's semantics. This is the default: a plan or an authorization written
  before kinds existed carries none and is deterministically a Study, so every
  legacy artifact keeps its exact meaning.
* ``infrastructure_validation`` — a first-class non-Study scope. It may own
  compute, jobs, leases and storage under its own authorization, and it never
  borrows, implies or confers a Study's scientific authority. Its jobs carry no
  research evidence.

Kind and identity are always explicit where a plan meets its authorization, so
a mismatch fails closed instead of being resolved by preference. Nothing here
is a general workflow or type system: two kinds, one identity shape, and the
one comparison that keeps them apart.
"""

import re

from improvements.compute.errors import ConfigurationError

STUDY = "study"
INFRASTRUCTURE_VALIDATION = "infrastructure_validation"
KINDS = (STUDY, INFRASTRUCTURE_VALIDATION)

# The identity every authorization, plan and lease is named by.
SCOPE_ID = re.compile(r"^[A-Z]{2}-[0-9]{4}$")


def normalize_kind(value, *, field="scope_kind"):
    """The declared kind; a missing value is the legacy Study kind.

    An unrecognised kind is refused rather than defaulted, so a plan that names
    a kind this backend does not implement can never run as a Study.
    """

    if value is None or (isinstance(value, str) and not value.strip()):
        return STUDY
    if not isinstance(value, str) or value not in KINDS:
        raise ConfigurationError(
            "{} must be one of {}, got {!r}".format(
                field, ", ".join(KINDS), value
            )
        )
    return value


def require_scope_id(value, *, field="scope"):
    """Validate one scope identity. The shape is shared by every kind."""

    if not isinstance(value, str) or not SCOPE_ID.match(value):
        raise ConfigurationError(
            "{} must be a scope identifier of the form IN-0001, got {!r}".format(
                field, value
            )
        )
    return value
