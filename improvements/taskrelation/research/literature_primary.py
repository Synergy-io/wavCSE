"""Paper-id-driven retrieval of retained primary literature artifacts.

Authority split (roadmap INC-004, and `AGENTS.md`'s infrastructure boundary):

* **Git** owns the manifest: which ``paper_id`` has a retained primary
  artifact, its SHA-256, media type, byte size, provenance URL and object key.
* **S3** is the canonical durable store for the artifact bytes; publication,
  credentials, transfer and eviction live in the ``wavcse-infra`` checkout.
* **The local cache** is disposable: it is never authoritative, and deleting it
  must not destroy research knowledge or the canonical artifact.

This module is the repository-side adapter. It receives no credentials, resolves
everything from ``paper_id``, and returns either a checksum-verified local path
or a precise, structured unavailable reason. Remote transfer is an injected
seam so the adapter stays credential-free and testable offline.

Usage::

    python -m improvements.taskrelation.research.literature_primary validate
    python -m improvements.taskrelation.research.literature_primary status <paper_id>
    python -m improvements.taskrelation.research.literature_primary get <paper_id>
"""

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from improvements.taskrelation.research import literature_catalog


# Deterministic failure taxonomy: a caller branches on `kind`, never on prose.
UNKNOWN_PAPER = "UNKNOWN_PAPER"
PRIMARY_NOT_AVAILABLE = "PRIMARY_NOT_AVAILABLE"
STORAGE_NOT_CONFIGURED = "STORAGE_NOT_CONFIGURED"
CREDENTIALS_UNAVAILABLE = "CREDENTIALS_UNAVAILABLE"
REMOTE_RETRIEVAL_FAILED = "REMOTE_RETRIEVAL_FAILED"
INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"

# Role -> (file suffix, canonical media type). The object key depends on the
# paper and its role only, so metadata changes never move stored artifacts.
_ROLE_SPEC = {"source": ("pdf", "application/pdf")}
_DEFAULT_ROLE = "source"
_DEFAULT_CACHE_DIRNAME = "literature-primary"
_CACHE_ROOT_ENV = "WAVCSE_PRIMARY_CACHE"
_BUCKET_ENV = "WAVCSE_PRIMARY_BUCKET"
_PREFIX_ENV = "WAVCSE_PRIMARY_PREFIX"

_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PAPER_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ALLOWED_KEYS = {
    "schema_version",
    "paper_id",
    "role",
    "object_key",
    "sha256",
    "size_bytes",
    "media_type",
    "source_url",
    "retained_at",
}
_REQUIRED_KEYS = _ALLOWED_KEYS

_RESEARCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RESEARCH_DIR.parents[2]
_DEFAULT_LITERATURE_DIR = _RESEARCH_DIR / "literature"
_CATALOG_FILENAME = "catalog.jsonl"
_MANIFEST_FILENAME = "primary_manifest.jsonl"
_DEFAULT_MANIFEST = _DEFAULT_LITERATURE_DIR / _MANIFEST_FILENAME

Fetcher = Callable[..., None]


class PrimaryError(Exception):
    """A primary-artifact operation failed with a deterministic, branchable kind."""

    def __init__(self, message, kind, detail=None):
        super(PrimaryError, self).__init__(message)
        self.kind = kind
        self.detail = dict(detail or {})

    def as_dict(self):
        return {"error": str(self), "kind": self.kind, "detail": self.detail}


class PrimaryManifestError(ValueError):
    """The retained-artifact manifest is unusable or disagrees with the catalog."""


@dataclass(frozen=True)
class StoragePolicy:
    """Central path and object-key policy; no absolute path lives in Git."""

    research_dir: Path
    cache_root: Path
    bucket: Optional[str] = None
    prefix: str = ""
    repo_root: Path = _REPO_ROOT

    def __post_init__(self):
        cache_root = Path(self.cache_root).resolve()
        repo_root = Path(self.repo_root).resolve()
        if cache_root == repo_root or repo_root in cache_root.parents:
            raise PrimaryManifestError(
                "cache_root must be outside the repository (got {}); the primary "
                "cache is disposable state and is never committed".format(cache_root)
            )

    @property
    def literature_dir(self):
        return Path(self.research_dir) / "literature"

    @property
    def configured(self):
        return bool(self.bucket)

    def object_key(self, paper_id, role=_DEFAULT_ROLE):
        """Deterministic storage key: ``[prefix]papers/<paper_id>/<role>.<ext>``."""

        suffix = _role_spec(role)[0]
        _require_paper_id(paper_id)
        return "{}papers/{}/{}.{}".format(self.prefix, paper_id, role, suffix)

    def cache_path(self, paper_id, role=_DEFAULT_ROLE):
        """Deterministic disposable cache path, mirroring the object key."""

        suffix = _role_spec(role)[0]
        _require_paper_id(paper_id)
        return Path(self.cache_root) / "papers" / paper_id / "{}.{}".format(role, suffix)

    @classmethod
    def from_environment(cls, *, research_dir=None, repo_root=None, env=None):
        """Resolve the policy once, centrally, from the environment."""

        env = os.environ if env is None else env
        research_dir = Path(research_dir) if research_dir is not None else _RESEARCH_DIR
        repo_root = Path(repo_root) if repo_root is not None else _REPO_ROOT
        prefix = env.get(_PREFIX_ENV, "")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return cls(
            research_dir=research_dir,
            cache_root=_resolve_cache_root(env),
            bucket=env.get(_BUCKET_ENV) or None,
            prefix=prefix,
            repo_root=repo_root,
        )


def _resolve_cache_root(env):
    override = env.get(_CACHE_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    xdg_cache = env.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "wavcse" / _DEFAULT_CACHE_DIRNAME


def _role_spec(role):
    try:
        return _ROLE_SPEC[role]
    except KeyError as exc:
        raise PrimaryManifestError(
            "role must be one of: {} (got {!r})".format(
                ", ".join(sorted(_ROLE_SPEC)), role
            )
        ) from exc


def _require_paper_id(paper_id):
    if not isinstance(paper_id, str) or not _PAPER_ID.match(paper_id):
        raise PrimaryManifestError(
            "not a valid paper_id (expected the canonical card slug): {!r}".format(
                paper_id
            )
        )
    return paper_id


@dataclass(frozen=True)
class PrimaryManifestEntry:
    """Storage metadata for one retained artifact; identity stays in the catalog."""

    paper_id: str
    role: str
    object_key: str
    sha256: str
    size_bytes: int
    media_type: str
    source_url: str
    retained_at: str


@dataclass(frozen=True)
class PrimaryStatus:
    """Read-only description of retention and cache state for one paper."""

    paper_id: str
    retained: bool
    cache_state: str
    retrieval: str
    object_key: Optional[str] = None
    sha256: Optional[str] = None
    media_type: Optional[str] = None
    source_url: Optional[str] = None
    cache_path: Optional[str] = None

    def as_dict(self):
        return {
            "paper_id": self.paper_id,
            "retained": self.retained,
            "cache_state": self.cache_state,
            "retrieval": self.retrieval,
            "object_key": self.object_key,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "source_url": self.source_url,
            "cache_path": self.cache_path,
        }


@dataclass(frozen=True)
class PrimaryArtifact:
    """A checksum-verified local copy of a retained primary artifact."""

    paper_id: str
    role: str
    sha256: str
    size_bytes: int
    media_type: str
    source_url: str
    object_key: str
    path: Path
    source: str

    def as_dict(self):
        return {
            "paper_id": self.paper_id,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "source_url": self.source_url,
            "object_key": self.object_key,
            "path": str(self.path),
            "source": self.source,
        }


class LiteraturePrimary:
    """Resolve ``paper_id`` to verified primary bytes, or to a precise reason."""

    def __init__(self, policy=None, *, repo_root=None, research_dir=None,
                 catalog_path=None, manifest_path=None, fetcher=None):
        if policy is None:
            policy = StoragePolicy.from_environment(
                research_dir=research_dir,
                repo_root=repo_root,
            )
        self.policy = policy
        self._fetcher = fetcher
        self._catalog_path = Path(catalog_path) if catalog_path is not None else (
            Path(policy.literature_dir) / _CATALOG_FILENAME
        )
        self._manifest_path = Path(manifest_path) if manifest_path is not None else (
            Path(policy.literature_dir) / _MANIFEST_FILENAME
        )
        self._catalog = literature_catalog.load_catalog(
            self._catalog_path, repo_root=Path(policy.repo_root),
            validate_references=False,
        )
        self._entries = self._load_manifest()

    def object_key(self, paper_id, role=_DEFAULT_ROLE):
        return self.policy.object_key(paper_id, role)

    @property
    def manifest_path(self):
        return self._manifest_path

    def cache_path(self, paper_id, role=_DEFAULT_ROLE):
        return self.policy.cache_path(paper_id, role)

    def manifest_rows(self):
        return tuple(self._entries[key] for key in sorted(self._entries))

    def validate(self):
        """Return the number of validated manifest rows (0 is a valid manifest)."""

        return len(self._entries)

    def status(self, paper_id):
        """Describe retention and cache state without mutating anything."""

        paper_id = self._known_paper_id(paper_id)
        entry = self._entries.get((paper_id, _DEFAULT_ROLE))
        cache = self.cache_path(paper_id)
        return PrimaryStatus(
            paper_id=paper_id,
            retained=entry is not None,
            cache_state=self._cache_state(cache, entry),
            retrieval="configured" if self._fetcher is not None else "not_configured",
            object_key=entry.object_key if entry else self.policy.object_key(paper_id),
            sha256=entry.sha256 if entry else None,
            media_type=entry.media_type if entry else None,
            source_url=entry.source_url if entry else None,
            cache_path=str(cache),
        )

    def get(self, paper_id, role=_DEFAULT_ROLE):
        """Return verified primary bytes for ``paper_id``.

        Raises :class:`PrimaryError` with a deterministic ``kind`` when the
        artifact is unavailable, storage is unconfigured, the transfer fails, or
        integrity cannot be established. Never returns unverified bytes.
        """

        paper_id = self._known_paper_id(paper_id)
        entry = self._entries.get((paper_id, role))
        if entry is None:
            raise PrimaryError(
                "no retained primary artifact is declared for {!r}".format(paper_id),
                kind=PRIMARY_NOT_AVAILABLE,
                detail={"paper_id": paper_id, "role": role},
            )
        cache = self.cache_path(paper_id, role)
        state = self._cache_state(cache, entry)
        if state == "valid":
            return self._artifact(entry, cache, "cache")
        if state == "corrupt":
            removed = self._discard_cache_entry(cache)
            raise PrimaryError(
                "cached artifact for {!r} failed integrity verification".format(paper_id),
                kind=INTEGRITY_MISMATCH,
                detail={
                    "paper_id": paper_id,
                    "cache_path": str(cache),
                    "expected_sha256": entry.sha256,
                    "cache_entry_removed": removed,
                },
            )
        if self._fetcher is None:
            raise PrimaryError(
                "no primary-artifact transfer backend is configured; S3 retrieval "
                "is owned by the wavcse-infra checkout",
                kind=STORAGE_NOT_CONFIGURED,
                detail={
                    "paper_id": paper_id,
                    "object_key": entry.object_key,
                    "bucket_configured": self.policy.configured,
                },
            )
        self._fetch_to_cache(self._fetcher, entry, cache)
        return self._artifact(entry, cache, "remote")

    def _known_paper_id(self, paper_id):
        _require_paper_id(paper_id)
        try:
            entry = self._catalog.get(paper_id)
        except literature_catalog.CatalogError as exc:
            raise PrimaryError(
                "unknown paper_id {!r}".format(paper_id),
                kind=UNKNOWN_PAPER,
                detail={"paper_id": paper_id},
            ) from exc
        return entry.paper_id

    def _cache_state(self, cache, entry):
        if entry is None or not cache.is_file():
            return "absent"
        return "valid" if _matches(cache, entry) else "corrupt"

    def _discard_cache_entry(self, cache):
        """Remove a provably invalid disposable cache copy; never touch S3."""

        try:
            cache.unlink()
        except FileNotFoundError:
            return False
        return True

    def _fetch_to_cache(self, fetcher, entry, cache):
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.parent / ".{}.{}.partial".format(entry.paper_id, entry.role)
        try:
            fetcher(
                object_key=entry.object_key,
                destination=partial,
                policy=self.policy,
            )
            if not partial.is_file():
                raise PrimaryError(
                    "transfer backend returned without writing {!r}".format(
                        entry.object_key
                    ),
                    kind=REMOTE_RETRIEVAL_FAILED,
                    detail={"paper_id": entry.paper_id, "object_key": entry.object_key},
                )
            if not _matches(partial, entry):
                raise PrimaryError(
                    "downloaded artifact for {!r} does not match its manifest "
                    "checksum".format(entry.paper_id),
                    kind=INTEGRITY_MISMATCH,
                    detail={
                        "paper_id": entry.paper_id,
                        "object_key": entry.object_key,
                        "expected_sha256": entry.sha256,
                        "actual_sha256": _sha256(partial),
                    },
                )
            os.replace(str(partial), str(cache))
        except PrimaryError:
            raise
        except Exception as exc:  # transfer-backend failure, not our contract
            raise PrimaryError(
                "primary-artifact retrieval failed for {!r}".format(entry.paper_id),
                kind=REMOTE_RETRIEVAL_FAILED,
                detail={
                    "paper_id": entry.paper_id,
                    "object_key": entry.object_key,
                    "error_type": type(exc).__name__,
                },
            ) from exc
        finally:
            if partial.exists():
                partial.unlink()

    @staticmethod
    def _artifact(entry, cache, source):
        return PrimaryArtifact(
            paper_id=entry.paper_id,
            role=entry.role,
            sha256=entry.sha256,
            size_bytes=entry.size_bytes,
            media_type=entry.media_type,
            source_url=entry.source_url,
            object_key=entry.object_key,
            path=cache,
            source=source,
        )

    def _load_manifest(self):
        try:
            lines = self._manifest_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise PrimaryManifestError(
                "cannot read primary manifest {}: {}".format(self._manifest_path, exc)
            ) from exc
        entries = {}
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise PrimaryManifestError(
                    "manifest line {} is blank".format(line_number)
                )
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise PrimaryManifestError(
                    "manifest line {} is not valid JSON: {}".format(line_number, exc)
                ) from exc
            entry = self._validate_row(record, line_number)
            key = (entry.paper_id, entry.role)
            if key in entries:
                raise PrimaryManifestError(
                    "duplicate manifest entry for {!r} role {!r}".format(
                        entry.paper_id, entry.role
                    )
                )
            entries[key] = entry
        return entries

    def _validate_row(self, record, line_number):
        where = "manifest line {}".format(line_number)
        if not isinstance(record, dict):
            raise PrimaryManifestError("{} must be a JSON object".format(where))
        unknown = sorted(set(record) - _ALLOWED_KEYS)
        missing = sorted(_REQUIRED_KEYS - set(record))
        if unknown:
            raise PrimaryManifestError(
                "{} has unknown key(s): {}".format(where, ", ".join(unknown))
            )
        if missing:
            raise PrimaryManifestError(
                "{} is missing key(s): {}".format(where, ", ".join(missing))
            )
        if record["schema_version"] != _SCHEMA_VERSION:
            raise PrimaryManifestError(
                "{}.schema_version must be {}".format(where, _SCHEMA_VERSION)
            )
        paper_id = _require_paper_id(record["paper_id"])
        try:
            paper = self._catalog.get(paper_id)
        except literature_catalog.CatalogError as exc:
            raise PrimaryManifestError(
                "{} references unknown paper_id {!r}".format(where, paper_id)
            ) from exc
        role = record["role"]
        suffix, media_type = _role_spec(role)
        object_key = record["object_key"]
        expected_key = self.policy.object_key(paper_id, role)
        if object_key != expected_key:
            raise PrimaryManifestError(
                "{}.object_key must be exactly {!r} (got {!r}); keys are derived "
                "from paper_id, never authored".format(where, expected_key, object_key)
            )
        sha256 = record["sha256"]
        if not isinstance(sha256, str) or not _SHA256.match(sha256):
            raise PrimaryManifestError(
                "{}.sha256 must be 64 lowercase hex characters; a size or a "
                "filename is not an identity".format(where)
            )
        size = record["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise PrimaryManifestError(
                "{}.size_bytes must be a non-negative integer".format(where)
            )
        if record["media_type"] != media_type:
            raise PrimaryManifestError(
                "{}.media_type must be {!r} for role {!r}".format(
                    where, media_type, role
                )
            )
        source_url = record["source_url"]
        if not isinstance(source_url, str) or source_url not in paper.source_urls:
            raise PrimaryManifestError(
                "{}.source_url is not recorded in the catalog for {!r}".format(
                    where, paper_id
                )
            )
        retained_at = record["retained_at"]
        if not isinstance(retained_at, str) or not retained_at.strip():
            raise PrimaryManifestError(
                "{}.retained_at must be a non-empty ISO 8601 string".format(where)
            )
        if not suffix:  # pragma: no cover - guarded by _role_spec
            raise PrimaryManifestError("{}.role has no file suffix".format(where))
        return PrimaryManifestEntry(
            paper_id=paper_id,
            role=role,
            object_key=object_key,
            sha256=sha256,
            size_bytes=size,
            media_type=media_type,
            source_url=source_url,
            retained_at=retained_at,
        )


def _sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path, entry):
    """Checksum is the identity; size is only a cheap pre-filter."""

    try:
        if path.stat().st_size != entry.size_bytes:
            return False
    except OSError:
        return False
    return _sha256(path) == entry.sha256


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("validate", help="validate the retained-artifact manifest")
    status = commands.add_parser("status", help="describe retention and cache state")
    status.add_argument("paper_id")
    get = commands.add_parser("get", help="resolve a verified local primary artifact")
    get.add_argument("paper_id")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    command = args.command or "validate"
    manifest_path = _DEFAULT_MANIFEST
    try:
        primary = LiteraturePrimary()
        manifest_path = primary.manifest_path
        if command == "validate":
            print(
                "primary manifest: OK ({} retained artifact(s))".format(primary.validate())
            )
        elif command == "status":
            print(json.dumps(primary.status(args.paper_id).as_dict(), sort_keys=True))
        else:
            print(
                json.dumps(primary.get(args.paper_id).as_dict(), sort_keys=True)
            )
        return 0
    except PrimaryError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 1
    except PrimaryManifestError as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "kind": "MANIFEST_INVALID",
                    "detail": {"manifest": str(manifest_path)},
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
