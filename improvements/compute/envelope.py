"""Authorization envelopes: the only authority for spending money.

An envelope is the human's written grant for one scope. It lives in Git at
``improvements/taskrelation/research/authorizations/<SCOPE>.yaml`` so that a
change to the spend authority is a reviewable diff, and it is evaluated against
provider-derived facts so that the accounting is never self-reported.

The backend can only *consume* an envelope:

* it never writes, renews, widens, or invents one;
* it refuses an envelope that differs from the committed revision while the
  scope is busy, so an authorization cannot drift underneath a running job;
* it records the digest it acted under, so every action is traceable to the
  grant that permitted it.

Evidence, not memory, decides: the model is validated strictly (unknown keys are
rejected, because a misspelled limit would silently not bind), and the totals
come from ``infra worker list --json`` rather than from a counter the backend
keeps for itself.
"""

import os
import re
from decimal import Decimal, InvalidOperation

from improvements.compute import state as state_module
from improvements.compute.errors import (
    AuthorizationError,
    CapacityError,
    ConfigurationError,
    CostError,
)

SCHEMA_VERSION = 1
AUTHORIZATIONS_DIR = os.path.join(
    "improvements", "taskrelation", "research", "authorizations"
)

SCOPE_PATTERN = r"^[A-Z]{2}-[0-9]{4}$"

# Actions the orchestrator may ask for.
ACTION_CREATE_WORKER = "create-worker"
ACTION_SUBMIT_JOB = "submit-job"
ACTION_CREATE_VOLUME = "create-volume"
ACTION_ATTACH_VOLUME = "attach-volume"
ACTION_STOP_WORKER = "stop-worker"
ACTION_DESTROY_WORKER = "destroy-worker"

_COSTLY_ACTIONS = (ACTION_CREATE_WORKER, ACTION_SUBMIT_JOB, ACTION_ATTACH_VOLUME,
                   ACTION_CREATE_VOLUME)

_TOP_LEVEL_KEYS = {
    "schema_version", "scope", "granted_by", "granted_at", "expires_at",
    "budget", "concurrency", "resources", "stop_policy",
}
_BUDGET_KEYS = {"max_gpu_hourly_usd", "max_total_gpu_usd", "max_wall_clock_hours"}
_CONCURRENCY_KEYS = {"max_simultaneous_workers", "replacement_workers_allowed"}
_RESOURCES_KEYS = {
    "existing_network_volume_allowed", "network_volume_selector",
    "new_persistent_resources", "container_disk_gb_max",
}
# The selector is optional: a scope that never mounts a volume need not name one.
_RESOURCES_REQUIRED = {
    "existing_network_volume_allowed", "new_persistent_resources",
    "container_disk_gb_max",
}
_SELECTOR_KEYS = {"min_size_gb", "datacenter"}
_STOP_KEYS = {"destroy_on_completion", "retain_for_reuse_hours"}

_SECRET_MARKERS = (
    "X-Amz-", "BEGIN", "Bearer ", "bearer ", "Authorization:", "AKIA",
    "?Signature=", "&Signature=", "api_key", "apikey", "password", "secret",
    "token", "sk-",
)


def _decimal(value, field):
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ConfigurationError(
            "envelope field {} must be a number, got {!r}".format(field, value)
        )
    if not parsed.is_finite():
        raise ConfigurationError(
            "envelope field {} must be finite, got {!r}".format(field, value)
        )
    return parsed


def _require_mapping(value, field):
    if not isinstance(value, dict):
        raise ConfigurationError(
            "envelope field {} must be a mapping, got {}".format(
                field, type(value).__name__
            )
        )
    return value


def _require_keys(mapping, allowed, field):
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigurationError(
            "envelope field {} has unknown key(s) {}; a misspelled limit would "
            "silently not bind, so unknown keys are rejected".format(
                field, ", ".join(unknown)
            )
        )
    missing = sorted(set(allowed) - set(mapping))
    return missing


def _scan_for_secrets(payload, path=""):
    if isinstance(payload, dict):
        for key, value in payload.items():
            _scan_for_secrets(value, "{}.{}".format(path, key))
        return
    if isinstance(payload, list):
        for index, value in enumerate(payload):
            _scan_for_secrets(value, "{}[{}]".format(path, index))
        return
    if isinstance(payload, str):
        for marker in _SECRET_MARKERS:
            if marker in payload:
                raise ConfigurationError(
                    "envelope field {} contains what looks like credential or "
                    "runtime-state material ({!r}); envelopes carry policy only".format(
                        path.lstrip("."), marker
                    )
                )


def validate(envelope):
    """Validate one parsed envelope. Raises on the first problem."""

    _require_mapping(envelope, "envelope")
    _scan_for_secrets(envelope)
    missing = _require_keys(envelope, _TOP_LEVEL_KEYS, "envelope")
    if missing:
        raise ConfigurationError(
            "envelope is missing required key(s): {}".format(", ".join(missing))
        )
    if envelope["schema_version"] != SCHEMA_VERSION:
        raise ConfigurationError(
            "envelope schema_version must be {}, got {!r}".format(
                SCHEMA_VERSION, envelope["schema_version"]
            )
        )

    scope = envelope["scope"]
    if not isinstance(scope, str) or not re.match(SCOPE_PATTERN, scope):
        raise ConfigurationError(
            "envelope scope must be a study identifier of the form TR-0007, got "
            "{!r}".format(scope)
        )
    for field in ("granted_by", "granted_at", "expires_at"):
        if not isinstance(envelope[field], str) or not envelope[field].strip():
            raise ConfigurationError(
                "envelope field {} must be a non-empty string".format(field)
            )

    budget = _require_mapping(envelope["budget"], "budget")
    missing = _require_keys(budget, _BUDGET_KEYS, "budget")
    if missing:
        raise ConfigurationError(
            "envelope budget is missing key(s): {}".format(", ".join(missing))
        )
    hourly = _decimal(budget["max_gpu_hourly_usd"], "budget.max_gpu_hourly_usd")
    total = _decimal(budget["max_total_gpu_usd"], "budget.max_total_gpu_usd")
    wall_clock = _decimal(budget["max_wall_clock_hours"], "budget.max_wall_clock_hours")
    if hourly <= 0 or total <= 0 or wall_clock <= 0:
        raise ConfigurationError(
            "envelope budget limits must all be positive (hourly, total, wall clock)"
        )

    concurrency = _require_mapping(envelope["concurrency"], "concurrency")
    missing = _require_keys(concurrency, _CONCURRENCY_KEYS, "concurrency")
    if missing:
        raise ConfigurationError(
            "envelope concurrency is missing key(s): {}".format(", ".join(missing))
        )
    maximum = concurrency["max_simultaneous_workers"]
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ConfigurationError(
            "concurrency.max_simultaneous_workers must be an integer >= 1"
        )
    if not isinstance(concurrency["replacement_workers_allowed"], bool):
        raise ConfigurationError(
            "concurrency.replacement_workers_allowed must be a boolean"
        )

    resources = _require_mapping(envelope["resources"], "resources")
    unknown = sorted(set(resources) - _RESOURCES_KEYS)
    if unknown:
        raise ConfigurationError(
            "envelope resources has unknown key(s): {}".format(", ".join(unknown))
        )
    missing = sorted(_RESOURCES_REQUIRED - set(resources))
    if missing:
        raise ConfigurationError(
            "envelope resources is missing key(s): {}".format(", ".join(missing))
        )
    for field in ("existing_network_volume_allowed", "new_persistent_resources"):
        if not isinstance(resources[field], bool):
            raise ConfigurationError("resources.{} must be a boolean".format(field))
    selector = resources.get("network_volume_selector")
    if selector is not None:
        _require_mapping(selector, "resources.network_volume_selector")
        unknown = sorted(set(selector) - _SELECTOR_KEYS)
        if unknown:
            raise ConfigurationError(
                "resources.network_volume_selector has unknown key(s): {}".format(
                    ", ".join(unknown)
                )
            )
    disk_max = resources["container_disk_gb_max"]
    if not isinstance(disk_max, int) or isinstance(disk_max, bool) or disk_max < 1:
        raise ConfigurationError("resources.container_disk_gb_max must be an integer >= 1")

    stop_policy = _require_mapping(envelope["stop_policy"], "stop_policy")
    missing = _require_keys(stop_policy, _STOP_KEYS, "stop_policy")
    if missing:
        raise ConfigurationError(
            "envelope stop_policy is missing key(s): {}".format(", ".join(missing))
        )
    if not isinstance(stop_policy["destroy_on_completion"], bool):
        raise ConfigurationError("stop_policy.destroy_on_completion must be a boolean")
    retain = stop_policy["retain_for_reuse_hours"]
    if not isinstance(retain, int) or isinstance(retain, bool) or retain < 0:
        raise ConfigurationError(
            "stop_policy.retain_for_reuse_hours must be an integer >= 0"
        )

    _expiry(envelope)
    return envelope


def _expiry(envelope):
    try:
        expires = state_module.parse_timestamp(envelope["expires_at"])
    except ValueError as exc:
        raise ConfigurationError(
            "envelope expires_at is not a valid ISO 8601 timestamp: {}".format(exc)
        ) from exc
    if expires is None:
        raise ConfigurationError("envelope expires_at is required")
    return expires


def expiry(envelope):
    return _expiry(envelope)


def is_expired(envelope, now=None):
    moment = now or state_module.utc_now()
    return moment >= expiry(envelope)


def envelope_path(scope, repo_root=None):
    root = repo_root or _repo_root()
    return os.path.join(root, AUTHORIZATIONS_DIR, "{}.yaml".format(scope))


def _repo_root():
    from improvements.compute import resolve as resolve_module

    return resolve_module.repo_root()


def _read_committed(path, repo_root):
    """Read the committed revision of a path, for tamper comparison."""

    import subprocess

    relative = os.path.relpath(path, repo_root)
    completed = subprocess.run(
        ["git", "-C", repo_root, "show", "HEAD:{}".format(relative)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        universal_newlines=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


class EnvelopeView(object):
    """A validated envelope plus the provenance needed to act on it."""

    def __init__(self, scope, envelope, digest, committed_digest, source,
                 problems=()):
        self.scope = scope
        self.envelope = envelope
        self.digest = digest
        self.committed_digest = committed_digest
        self.source = source
        self.problems = list(problems)

    @property
    def modified_in_tree(self):
        return (
            self.committed_digest is not None
            and self.committed_digest != self.digest
        )

    @property
    def uncommitted(self):
        return self.committed_digest is None

    def as_dict(self):
        return {
            "scope": self.scope,
            "digest": self.digest,
            "committed_digest": self.committed_digest,
            "source": self.source,
            "modified_in_tree": self.modified_in_tree,
            "uncommitted": self.uncommitted,
            "expires_at": self.envelope["expires_at"],
            "budget": self.envelope["budget"],
            "concurrency": self.envelope["concurrency"],
            "resources": self.envelope["resources"],
            "stop_policy": self.envelope["stop_policy"],
        }


def load(scope, repo_root=None):
    """Load and validate a scope's envelope without touching runtime state."""

    import yaml

    root = repo_root or _repo_root()
    path = envelope_path(scope, root)
    if not os.path.exists(path):
        raise AuthorizationError(
            "no compute authorization exists for scope {} (expected {}). Paid "
            "work stays blocked until a human grants one; see {}.".format(
                scope,
                os.path.relpath(path, root) if root else path,
                os.path.join(AUTHORIZATIONS_DIR, "README.md"),
            )
        )
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    try:
        envelope = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            "envelope {} is not valid YAML: {}".format(path, exc)
        ) from exc
    validate(envelope)
    if envelope["scope"] != scope:
        raise ConfigurationError(
            "envelope {} declares scope {} but was requested for {}".format(
                path, envelope["scope"], scope
            )
        )
    digest = state_module.digest_of_text(text)
    committed = _read_committed(path, root)
    committed_digest = state_module.digest_of_text(committed) if committed is not None else None
    return EnvelopeView(scope, envelope, digest, committed_digest, "repository")


def snapshot_path(scope):
    return state_module.path_for("envelopes", scope)


def active_snapshot(scope):
    document = state_module.read_json(snapshot_path(scope), default=None) or {}
    return document.get("active")


def record_snapshot(view, *, busy):
    """Record the digest the backend acted under.

    Rotation is refused while the scope is busy, which is what makes "the
    authorization cannot change underneath a running job" enforceable rather
    than aspirational.
    """

    path = snapshot_path(scope=view.scope)
    document = state_module.read_json(path, default=None) or {
        "schema_version": SCHEMA_VERSION,
        "scope": view.scope,
        "active": None,
        "history": [],
    }
    active = document.get("active")
    if active and active.get("digest") == view.digest:
        return active
    if active and busy:
        raise AuthorizationError(
            "the authorization for {} changed ({}) while the scope still has "
            "active compute or jobs. A human must re-grant it explicitly; the "
            "orchestrator never widens or refreshes its own authorization.".format(
                view.scope, view.digest[:12]
            )
        )
    history = list(document.get("history") or [])
    if active:
        history.append(active)
    entry = {
        "digest": view.digest,
        "committed_digest": view.committed_digest,
        "recorded_at": state_module.isoformat(state_module.utc_now()),
        "expires_at": view.envelope["expires_at"],
        "envelope": view.envelope,
    }
    document.update({"active": entry, "history": history})
    state_module.write_json_atomically(path, document)
    return entry


class Decision(object):
    """The verdict for one proposed action."""

    def __init__(self, allowed, klass, reason, *, action=None, details=None):
        self.allowed = allowed
        self.klass = klass
        self.reason = reason
        self.action = action
        self.details = details or {}

    def as_dict(self):
        return {
            "allowed": self.allowed,
            "class": self.klass,
            "reason": self.reason,
            "action": self.action,
            "details": self.details,
        }

    def require(self):
        """Raise the typed error matching the reported class."""

        if self.allowed:
            return
        if self.klass == "COST":
            raise CostError(self.reason)
        if self.klass == "CAPACITY":
            raise CapacityError(self.reason)
        raise AuthorizationError(self.reason)


def check(view, action, facts, requested=None):
    """Decide whether ``action`` fits the envelope.

    ``facts`` is derived from provider state by :mod:`improvements.compute.ledger`:
    ``live_workers`` (scope workers that still bill), ``estimated_spend_usd``,
    ``estimated_wall_clock_hours`` and ``estimated_hourly_exposure_usd``.
    ``requested`` carries the proposed action's own numbers (for example a
    ``hourly_usd`` for a create).
    """

    requested = requested or {}
    envelope = view.envelope
    budget = envelope["budget"]
    concurrency = envelope["concurrency"]
    resources = envelope["resources"]

    if view.modified_in_tree:
        return Decision(False, "AUTHORIZATION",
                        "the envelope for {} differs from the committed revision; "
                        "uncommitted authorization changes are never honoured".format(
                            view.scope), action=action)

    if view.uncommitted:
        return Decision(False, "AUTHORIZATION",
                        "the envelope for {} is not committed; commit the grant "
                        "before any paid action".format(view.scope), action=action)

    if is_expired(envelope):
        return Decision(False, "AUTHORIZATION",
                        "the authorization for {} expired at {}".format(
                            view.scope, envelope["expires_at"]), action=action)

    if action in _COSTLY_ACTIONS and facts.get("accounting_bounded") is False:
        # If the scope's spend cannot be bounded from provider facts, no paid
        # action is authorizable — whatever else about the request looks fine.
        return Decision(False, "COST",
                        "the scope's spend cannot be bounded from provider facts "
                        "({}), so a new paid action cannot be authorized".format(
                            "; ".join(facts.get("unknowns") or ["unknown cost"])),
                        action=action)

    maximum = int(concurrency["max_simultaneous_workers"])
    live = list(facts.get("live_workers") or [])
    live_count = len(live)
    if action in _COSTLY_ACTIONS and live_count > maximum:
        return Decision(False, "CAPACITY",
                        "{} workers are already billable for {}, above the authorized "
                        "concurrent limit of {}; reconcile and stop the excess before "
                        "further paid work".format(live_count, view.scope, maximum),
                        action=action)

    if action == ACTION_CREATE_WORKER:
        # Concurrency is about workers that still bill, not about how many were
        # ever created: a replacement must stop or destroy its predecessor first.
        if live_count + 1 > maximum:
            if concurrency["replacement_workers_allowed"] and live_count >= maximum:
                return Decision(False, "CAPACITY",
                                "{} worker(s) still bill for {} and the envelope "
                                "allows {} at once. Replacement is permitted, but "
                                "the finished worker must be stopped or destroyed "
                                "before a replacement is created.".format(
                                    live_count, view.scope, maximum), action=action)
            return Decision(False, "CAPACITY",
                            "creating another worker would exceed the envelope's "
                            "concurrent-worker limit of {} for {}".format(
                                maximum, view.scope), action=action)

        disk = requested.get("container_disk_gb")
        if disk is not None and int(disk) > int(resources["container_disk_gb_max"]):
            return Decision(False, "COST",
                            "requested container disk {} GB exceeds the envelope's "
                            "limit of {} GB".format(
                                disk, resources["container_disk_gb_max"]),
                            action=action)

        hourly = requested.get("hourly_usd")
        if hourly is None:
            return Decision(False, "COST",
                            "the provider did not report a price for the requested "
                            "resource, so the envelope cannot be enforced; refusing "
                            "to spend unverifiable money", action=action)
        hourly = _decimal(hourly, "requested.hourly_usd")
        ceiling = _decimal(budget["max_gpu_hourly_usd"], "budget.max_gpu_hourly_usd")
        if hourly > ceiling:
            return Decision(False, "COST",
                            "provider price ${}/hour exceeds the authorized hourly "
                            "ceiling ${}/hour".format(hourly, ceiling), action=action)

    if action == ACTION_ATTACH_VOLUME:
        if not resources["existing_network_volume_allowed"]:
            return Decision(False, "AUTHORIZATION",
                            "the envelope does not allow using an existing network "
                            "volume", action=action)

    if action == ACTION_CREATE_VOLUME:
        if not resources["new_persistent_resources"]:
            return Decision(False, "AUTHORIZATION",
                            "the envelope forbids creating new persistent "
                            "(storage-billing) resources", action=action)

    if action in _COSTLY_ACTIONS:
        spend = _decimal(facts.get("estimated_spend_usd", 0), "facts.estimated_spend_usd")
        total = _decimal(budget["max_total_gpu_usd"], "budget.max_total_gpu_usd")
        projected = spend
        hourly = requested.get("hourly_usd")
        horizon = requested.get("projected_hours")
        if hourly is not None and horizon is not None:
            projected = spend + (_decimal(hourly, "requested.hourly_usd")
                                 * _decimal(horizon, "requested.projected_hours"))
        if projected > total:
            return Decision(False, "COST",
                            "projected spend ${} would exceed the authorized total "
                            "of ${} for {}".format(projected, total, view.scope),
                            action=action)

        wall = _decimal(facts.get("estimated_wall_clock_hours", 0),
                        "facts.estimated_wall_clock_hours")
        wall_limit = _decimal(budget["max_wall_clock_hours"], "budget.max_wall_clock_hours")
        if wall >= wall_limit:
            return Decision(False, "COST",
                            "{} paid wall-clock hours already consumed of the "
                            "authorized {}".format(wall, wall_limit), action=action)

    remaining = _remaining_usd(view, facts)
    return Decision(True, "AUTONOMOUS",
                    "within the authorization for {}".format(view.scope),
                    action=action,
                    details={
                        "estimated_spend_usd": str(facts.get("estimated_spend_usd", "0")),
                        "remaining_usd": str(remaining),
                        "max_simultaneous_workers": maximum,
                        "live_workers": live_count,
                        "digest": view.digest,
                    })


def _remaining_usd(view, facts):
    total = _decimal(view.envelope["budget"]["max_total_gpu_usd"], "budget.max_total_gpu_usd")
    spent = _decimal(facts.get("estimated_spend_usd", 0), "facts.estimated_spend_usd")
    return total - spent
