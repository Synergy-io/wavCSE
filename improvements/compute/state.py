"""Controller-local runtime state for the compute backend.

Runtime facts never live in Git: worker identifiers, hosts, ports, prices,
active job identifiers and provider runtime state are all controller-local, and
so is the ledger that counts them. Git keeps the science (envelopes, study
definitions, commit SHAs); this directory keeps what only exists at runtime.

Layout, rooted at :func:`state_root`::

    envelopes/<SCOPE>.json      authorized envelope snapshot + digest history
    runs/<SCOPE>.json           per-scope run ledger (jobs, attempts, spend)
    leases.json                 worker ownership: pending creates + leases
    compute-events.jsonl        append-only audit trail

Every write is atomic (temp file in the same directory, fsync, os.replace) and
serialized by an advisory ``flock`` on a per-document ``.lock`` file, the same
approach the control plane uses for its own state. The lock serializes
processes on one controller; it is deliberately not distributed consensus.
"""

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone

from improvements.compute.errors import BusyError, ConfigurationError

SCHEMA_VERSION = 1
STATE_ROOT_ENV = "WAVCSE_RESEARCH_STATE"
DEFAULT_STATE_ROOT = os.path.join("~", ".local", "state", "wavcse-research")


def state_root():
    """Absolute path of the controller-local compute state directory."""

    configured = os.environ.get(STATE_ROOT_ENV, "").strip()
    root = configured if configured else DEFAULT_STATE_ROOT
    return os.path.abspath(os.path.expanduser(root))


def utc_now():
    """Timezone-aware UTC timestamp; the only clock the ledger trusts."""

    return datetime.now(timezone.utc)


def isoformat(moment):
    """ISO 8601 with an explicit offset, matching the research-record convention."""

    return moment.astimezone(timezone.utc).isoformat()


def parse_timestamp(value):
    """Parse an ISO 8601 timestamp, tolerating a trailing ``Z``."""

    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def canonical_json(payload):
    """Deterministic JSON text: sorted keys, fixed indent, trailing newline."""

    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def digest_of_bytes(payload):
    """SHA-256 of raw bytes, as 64 lowercase hex characters."""

    return hashlib.sha256(payload).hexdigest()


def digest_of_text(text):
    return digest_of_bytes(text.encode("utf-8"))


class MutationLock(object):
    """Advisory exclusive lock over one state document or one scope.

    Used as a context manager. The lock file is separate from the document so a
    reader never has to hold the lock to read a consistent snapshot (writes are
    atomic replacements).

    Acquisition is non-blocking and refuses rather than waiting: two
    orchestrator instances must never spend for the same scope, and a silent
    wait would hide that from the operator. The refusal names the lock file, so
    a stale lock from a killed process is diagnosable.
    """

    def __init__(self, name):
        safe = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in str(name)
        )
        self.path = os.path.join(state_root(), "locks", safe + ".lock")
        self._handle = None

    def __enter__(self):
        directory = os.path.dirname(self.path)
        _ensure_directory(directory)
        self._handle = open(self.path, "a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._handle.close()
            self._handle = None
            raise BusyError(
                "another process holds the lock for this scope ({}); refusing to "
                "act so two orchestrators cannot spend or run the same work. Wait "
                "for it to finish, or remove the lock file if its owner is gone."
                .format(self.path)
            ) from exc
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None
        return False


def _ensure_directory(path):
    if not path:
        return
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem failure
        raise ConfigurationError(
            "cannot create state directory {}: {}".format(path, exc)
        ) from exc


def _fsync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(descriptor)


def write_json_atomically(path, payload):
    """Replace one JSON document atomically.

    The staging file is a fixed sibling of the destination, so the rename is
    within one directory and therefore atomic on POSIX. A crashed writer leaves
    only a stale staging file, never a half-written document.
    """

    directory = os.path.dirname(path)
    _ensure_directory(directory)
    text = canonical_json(payload)
    staging = temp_for(path)
    try:
        with open(staging, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
        _fsync_directory(directory)
    finally:
        if os.path.exists(staging):  # pragma: no cover - only on failure
            try:
                os.unlink(staging)
            except OSError:
                pass


def temp_for(path):
    """The staging path used by :func:`write_json_atomically`."""

    return path + ".staging"


def read_json(path, default=None):
    """Read one JSON document, returning ``default`` when it does not exist."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except OSError as exc:
        raise ConfigurationError(
            "cannot read state document {}: {}".format(path, exc)
        ) from exc
    except ValueError as exc:
        raise ConfigurationError(
            "state document {} is not valid JSON: {}".format(path, exc)
        ) from exc


def append_event(event, *, kind):
    """Append one audit record to the compute event log.

    The audit trail is append-only so a restart cannot lose the fact that an
    action was taken. Only non-secret, non-credential facts are recorded.
    """

    path = os.path.join(state_root(), "compute-events.jsonl")
    _ensure_directory(os.path.dirname(path))
    record = dict(event)
    record["kind"] = kind
    record["recorded_at"] = isoformat(utc_now())
    record["schema_version"] = SCHEMA_VERSION
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def read_events():
    """Every audit record, oldest first."""

    path = os.path.join(state_root(), "compute-events.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:  # pragma: no cover - a torn line is skipped, not fatal
                continue
    return records


def path_for(kind, name):
    """Absolute path of one state document."""

    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(name)
    )
    return os.path.join(state_root(), kind, safe)


def exists(path):
    return os.path.exists(path)


def scope_lock(scope):
    """Exclusive lock covering every mutation for one scope."""

    return MutationLock("scope-" + str(scope))
