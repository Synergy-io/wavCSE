"""Deterministic admission of a user-supplied primary artifact.

This is the first entry path of the acquisition architecture (roadmap INC-011):
a researcher hands the system one local file and, optionally, identity hints.
Every acquisition path — user-supplied bytes, structured scholarly discovery and
a future paper hunter — must converge on the *same* admission pipeline, so the
plumbing lives here and never in an LLM or a browsing agent:

    candidate bytes  ->  format validation  ->  scholarly identity/version
    validation  ->  exact-byte SHA-256  ->  duplicate/version handling  ->
    immutable local retention  ->  a ``PrimaryArtifact`` manifest row

The artifact store, its checksum identity, the manifest and the retrieval
contract already exist in :mod:`literature_primary`; this module adds only what
that operator-side ``register`` primitive cannot express on its own:

* **Identity resolution from hints.** ``register`` requires an exact canonical
  ``paper_id``. A researcher instead has a DOI, an arXiv id, a title or a URL,
  and the artifact must be attached to the *right* Paper or refused. Hints are
  resolved through the existing catalog vocabulary
  (:meth:`literature_catalog.LiteratureCatalog.lookup`) — no fuzzy matching and
  no second identity system. A supplied canonical ``paper_id`` that disagrees
  with another hint is an explicit ``IDENTITY_MISMATCH``; hints that resolve to
  more than one Paper without an explicit anchor fail closed as
  ``AMBIGUOUS_PAPER``.
* **Acquisition-attempt provenance.** ``register`` records the retained artifact,
  not the attempt and not *how* the bytes arrived. Every attempt is appended to
  ``literature/acquisitions.jsonl`` — an append-only, Git-tracked ledger with the
  provenance channel (``USER_SUPPLIED`` today), the resolved identity, the
  artifact digest and a machine-readable failure ``kind`` — so a failed or
  repeated attempt is auditable rather than lost in a traceback.

Authority boundary (binding, mirrors ``AGENTS.md``):

* Raw bytes are **untrusted data**. Nothing here parses PDF *content*, executes
  a shell, touches the network, or reads a credential; the only facts taken from
  the file are its byte length, its SHA-256 and its ``%PDF-`` header.
* An attempt may write exactly three things: the disposable primary cache and
  the primary manifest (both through ``literature_primary.register``), and the
  acquisition ledger. It can never create a Claim, PaperAssessment, PaperCard,
  Synthesis, finding, decision, Study or proposal, and it never edits project
  research state.
* Ingestion is **operator-side**. Like ``register`` it is deliberately absent
  from the model-facing ``literature_primary`` surface (``status|get|read``), so
  a browsing agent or an LLM cannot turn a URL or a file into an admitted
  artifact.

Usage::

    python -m improvements.taskrelation.research.literature_ingest ingest \
        --paper-id goncalves-2016-mssl --role published \
        --source-url https://jmlr.org/papers/volume17/15-215/15-215.pdf \
        ~/Downloads/paper.pdf
    python -m improvements.taskrelation.research.literature_ingest validate
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from improvements.taskrelation.research import literature_catalog
from improvements.taskrelation.research import literature_primary as lp


# Admission outcomes: a caller branches on `status`, never on prose.
ADMITTED = "ADMITTED"
DUPLICATE = "DUPLICATE"
REJECTED = "REJECTED"
_STATUSES = (ADMITTED, DUPLICATE, REJECTED)

# Provenance channels. Only user-supplied ingestion exists today; structured
# discovery and the paper-hunter fallback will add channels, never a new pipeline.
USER_SUPPLIED = "USER_SUPPLIED"
_PROVENANCES = (USER_SUPPLIED,)

# Identity-resolution failures (this module); artifact/registration failures are
# reused verbatim from `literature_primary` so the taxonomy stays one vocabulary.
NO_IDENTITY_INPUT = "NO_IDENTITY_INPUT"
UNKNOWN_PAPER = lp.UNKNOWN_PAPER
AMBIGUOUS_PAPER = "AMBIGUOUS_PAPER"
IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
ROLE_REQUIRED = "ROLE_REQUIRED"
SOURCE_URL_REQUIRED = "SOURCE_URL_REQUIRED"
INVALID_REQUEST = "INVALID_REQUEST"

_KNOWN_ROLES = tuple(sorted(lp._ROLE_SPEC))
_DEFAULT_ROLE = lp._DEFAULT_ROLE

_SCHEMA_VERSION = 1
_LEDGER_FILENAME = "acquisitions.jsonl"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_LEDGER_ALLOWED_KEYS = {
    "schema_version",
    "acquisition_id",
    "recorded_at",
    "provenance",
    "status",
    "paper_id",
    "role",
    "sha256",
    "size_bytes",
    "media_type",
    "source_url",
    "source_label",
    "failure_kind",
    "detail",
}
_LEDGER_REQUIRED_KEYS = _LEDGER_ALLOWED_KEYS


class IngestLedgerError(ValueError):
    """The acquisition ledger is unusable or disagrees with the manifest."""


@dataclass(frozen=True)
class AcquisitionResult:
    """The structured outcome of one acquisition attempt.

    A rejected attempt is a *returned* result, not an exception: identity
    ambiguity, a wrong-Paper hint and a corrupt file are real outcomes a caller
    must branch on. ``status`` is ``ADMITTED``, ``DUPLICATE`` or ``REJECTED``;
    ``kind`` carries the deterministic failure taxonomy only when rejected.
    """

    status: str
    provenance: str
    paper_id: Optional[str] = None
    role: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    media_type: Optional[str] = None
    source_url: Optional[str] = None
    source_label: Optional[str] = None
    acquisition_id: Optional[str] = None
    retained_role: Optional[str] = None
    kind: Optional[str] = None
    detail: Mapping = field(default_factory=dict)
    message: Optional[str] = None

    @property
    def admitted(self):
        return self.status in (ADMITTED, DUPLICATE)

    def as_dict(self):
        return {
            "status": self.status,
            "provenance": self.provenance,
            "admitted": self.admitted,
            "paper_id": self.paper_id,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "source_url": self.source_url,
            "source_label": self.source_label,
            "acquisition_id": self.acquisition_id,
            "retained_role": self.retained_role,
            "kind": self.kind,
            "detail": dict(self.detail),
            "message": self.message,
        }


class LiteratureIngest:
    """Resolve one local file to a Paper, admit it, and record the attempt."""

    def __init__(self, policy=None, *, repo_root=None, research_dir=None,
                 catalog_path=None, manifest_path=None, ledger_path=None,
                 primary=None):
        if primary is None:
            primary = lp.LiteraturePrimary(
                policy=policy,
                repo_root=repo_root,
                research_dir=research_dir,
                catalog_path=catalog_path,
                manifest_path=manifest_path,
            )
        self._primary = primary
        literature_dir = Path(primary.manifest_path).parent
        self._catalog_path = (
            Path(catalog_path)
            if catalog_path is not None
            else Path(primary.catalog_path)
        )
        self._catalog = literature_catalog.load_catalog(
            self._catalog_path,
            repo_root=Path(primary.policy.repo_root),
            validate_references=False,
        )
        self._ledger_path = (
            Path(ledger_path)
            if ledger_path is not None
            else literature_dir / _LEDGER_FILENAME
        )
        self._ledger = self._load_ledger()

    @property
    def ledger_path(self):
        return self._ledger_path

    @property
    def catalog_path(self):
        return self._catalog_path

    def attempts(self):
        """Return every recorded acquisition attempt, in ledger order."""

        return tuple(row for row in self._ledger)

    def validate(self):
        """Validate the ledger and its binding to the manifest; return row count.

        Every admitted (or duplicate) attempt must still be backed by a retained
        manifest artifact with the same ``paper_id``, ``role``, ``sha256`` and
        ``size_bytes``. A ledger that claims an artifact the manifest does not
        declare is corrupt, not merely stale.
        """

        manifest = {
            (row.paper_id, row.role, row.sha256, row.size_bytes)
            for row in self._primary.manifest_rows()
        }
        for index, row in enumerate(self._ledger):
            if row["status"] in (ADMITTED, DUPLICATE):
                key = (
                    row["paper_id"],
                    row["role"],
                    row["sha256"],
                    row["size_bytes"],
                )
                if key not in manifest:
                    raise IngestLedgerError(
                        "ledger line {} claims {} artifact {}/{} not declared in "
                        "the primary manifest".format(
                            index + 1, row["status"], row["paper_id"], row["role"]
                        )
                    )
        return len(self._ledger)

    def ingest(self, source_path, *, paper_id=None, doi=None, arxiv=None,
               title=None, source_url=None, role=None, source_label=None):
        """Admit one user-supplied local file as a primary artifact.

        Returns an :class:`AcquisitionResult`; never raises for a domain failure.
        The returned artifact (when admitted) is immediately reachable through
        the existing ``literature_primary`` / ``literature_read`` path, because
        the manifest row and the local copy are written by
        ``literature_primary.register`` before this method returns.
        """

        hints = self._hints(paper_id, doi, arxiv, title, source_url)
        if not hints:
            return self._record(
                self._reject(
                    NO_IDENTITY_INPUT,
                    "at least one of paper_id, doi, arxiv, title or source_url "
                    "is required to attach the artifact to a canonical Paper",
                    {},
                )
            )

        label = self._source_label(source_path, source_label)
        source, sha256, size_bytes, read_error = self._read_source(source_path)
        if read_error is not None:
            return self._record(
                self._reject(
                    read_error["kind"], read_error["message"], read_error["detail"],
                    hints=hints, source_label=label,
                )
            )
        if not lp._is_pdf(source):
            return self._record(
                self._reject(
                    lp.SOURCE_NOT_PDF,
                    "the supplied file is not a PDF (missing a %PDF- header)",
                    {"source_path": str(source_path)},
                    hints=hints, sha256=sha256, size_bytes=size_bytes,
                    source_label=label,
                )
            )

        paper, failure = self._resolve_paper(hints)
        if failure is not None:
            return self._record(
                self._reject(
                    failure["kind"], failure["message"], failure["detail"],
                    hints=hints, sha256=sha256, size_bytes=size_bytes,
                    source_label=label,
                )
            )

        existing = self._retained_for_digest(paper.paper_id, sha256, size_bytes)
        if existing is not None:
            return self._record(
                self._result(
                    DUPLICATE,
                    paper_id=paper.paper_id,
                    role=existing.role,
                    retained_role=existing.role,
                    sha256=existing.sha256,
                    size_bytes=existing.size_bytes,
                    media_type=existing.media_type,
                    source_url=existing.source_url,
                    source_label=label,
                    hints=hints,
                    message=(
                        "byte-identical to the artifact already retained for "
                        "{!r} as role {!r}".format(paper.paper_id, existing.role)
                    ),
                )
            )

        resolved_role, failure = self._resolve_role(paper.paper_id, role)
        if failure is not None:
            return self._record(
                self._reject(
                    failure["kind"], failure["message"], failure["detail"],
                    hints=hints, paper_id=paper.paper_id, role=role,
                    sha256=sha256, size_bytes=size_bytes, source_label=label,
                )
            )

        resolved_url, failure = self._resolve_source_url(paper, source_url)
        if failure is not None:
            return self._record(
                self._reject(
                    failure["kind"], failure["message"], failure["detail"],
                    hints=hints, paper_id=paper.paper_id, role=resolved_role,
                    sha256=sha256, size_bytes=size_bytes, source_label=label,
                )
            )

        try:
            entry = self._primary.register(
                paper.paper_id, source, source_url=resolved_url,
                role=resolved_role, replace=False,
            )
        except lp.PrimaryError as exc:
            return self._record(
                self._reject(
                    exc.kind, str(exc), exc.detail,
                    hints=hints, paper_id=paper.paper_id, role=resolved_role,
                    sha256=sha256, size_bytes=size_bytes, source_label=label,
                )
            )
        return self._record(
            self._result(
                ADMITTED,
                paper_id=entry.paper_id,
                role=entry.role,
                sha256=entry.sha256,
                size_bytes=entry.size_bytes,
                media_type=entry.media_type,
                source_url=entry.source_url,
                source_label=label,
                hints=hints,
                message="admitted as {!r} for {!r}".format(entry.role, entry.paper_id),
            )
        )

    # -- identity resolution ------------------------------------------------

    def _hints(self, paper_id, doi, arxiv, title, source_url):
        values = {
            "paper_id": paper_id,
            "doi": doi,
            "arxiv": arxiv,
            "title": title,
            "source_url": source_url,
        }
        hints = {}
        for name, value in values.items():
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                hints[name] = None
            else:
                hints[name] = value.strip()
        return hints

    def _resolve_paper(self, hints):
        anchor = hints.get("paper_id")
        if anchor is not None:
            try:
                paper = self._catalog.get(anchor)
            except literature_catalog.CatalogError:
                return None, {
                    "kind": UNKNOWN_PAPER,
                    "message": "unknown paper_id {!r}".format(anchor),
                    "detail": {"paper_id": anchor},
                }
            for field_name in ("doi", "arxiv", "title", "source_url"):
                value = hints.get(field_name)
                if value is None:
                    continue
                entry = self._lookup(value)
                if entry is None and field_name == "source_url":
                    # A provenance URL that is in no catalog entry at all is a
                    # missing recorded source, not a contradiction of the anchor.
                    return None, {
                        "kind": lp.SOURCE_URL_NOT_RECORDED,
                        "message": (
                            "source_url must be one recorded in the catalog for "
                            "{!r}".format(paper.paper_id)
                        ),
                        "detail": {
                            "paper_id": paper.paper_id,
                            "source_url": value,
                            "recorded": list(paper.source_urls),
                        },
                    }
                if entry is None or entry.paper_id != paper.paper_id:
                    return None, {
                        "kind": IDENTITY_MISMATCH,
                        "message": (
                            "{} {!r} does not identify the requested Paper {!r}"
                            .format(field_name, value, paper.paper_id)
                        ),
                        "detail": {
                            "requested_paper_id": paper.paper_id,
                            "field": field_name,
                            "value": value,
                            "resolved_paper_id": entry.paper_id if entry else None,
                        },
                    }
            return paper, None

        resolved = {}
        unresolved = []
        for field_name in ("doi", "arxiv", "title", "source_url"):
            value = hints.get(field_name)
            if value is None:
                continue
            entry = self._lookup(value)
            if entry is None:
                unresolved.append({"field": field_name, "value": value})
            else:
                resolved[entry.paper_id] = entry
        if not resolved:
            return None, {
                "kind": UNKNOWN_PAPER,
                "message": "no catalog Paper matches the supplied identity hints",
                "detail": {"unresolved": unresolved},
            }
        if len(resolved) == 1 and not unresolved:
            return next(iter(resolved.values())), None
        return None, {
            "kind": AMBIGUOUS_PAPER,
            "message": (
                "the identity hints do not identify exactly one Paper; supply an "
                "explicit paper_id"
            ),
            "detail": {
                "candidate_paper_ids": sorted(resolved),
                "unresolved": unresolved,
            },
        }

    def _lookup(self, value):
        try:
            return self._catalog.lookup(value)
        except literature_catalog.CatalogError:
            return None

    # -- version and provenance resolution ----------------------------------

    def _resolve_role(self, paper_id, role):
        if role is None:
            retained = sorted(
                entry.role
                for entry in self._primary.manifest_rows()
                if entry.paper_id == paper_id
            )
            if not retained:
                return _DEFAULT_ROLE, None
            return None, {
                "kind": ROLE_REQUIRED,
                "message": (
                    "{!r} already retains {}; name the artifact role so the new "
                    "bytes are never conflated with an existing version"
                    .format(paper_id, ", ".join(retained))
                ),
                "detail": {"paper_id": paper_id, "retained_roles": retained},
            }
        if not isinstance(role, str) or role not in _KNOWN_ROLES:
            return None, {
                "kind": lp.UNKNOWN_ROLE,
                "message": "role must be one of: {}".format(", ".join(_KNOWN_ROLES)),
                "detail": {"role": role, "known_roles": list(_KNOWN_ROLES)},
            }
        return role, None

    def _resolve_source_url(self, paper, source_url):
        if source_url is not None:
            for recorded in paper.source_urls:
                if recorded == source_url or recorded.rstrip("/") == source_url.rstrip("/"):
                    return recorded, None
            return None, {
                "kind": lp.SOURCE_URL_NOT_RECORDED,
                "message": (
                    "source_url must be one recorded in the catalog for {!r}"
                    .format(paper.paper_id)
                ),
                "detail": {
                    "paper_id": paper.paper_id,
                    "source_url": source_url,
                    "recorded": list(paper.source_urls),
                },
            }
        if len(paper.source_urls) == 1:
            return paper.source_urls[0], None
        return None, {
            "kind": SOURCE_URL_REQUIRED,
            "message": (
                "{!r} records {} source URLs; name the one this artifact "
                "corresponds to".format(paper.paper_id, len(paper.source_urls))
            ),
            "detail": {"paper_id": paper.paper_id, "recorded": list(paper.source_urls)},
        }

    # -- byte handling ------------------------------------------------------

    def _read_source(self, source_path):
        try:
            source = Path(source_path)
        except TypeError:
            return None, None, None, {
                "kind": lp.SOURCE_NOT_FOUND,
                "message": "source_path must be a filesystem path",
                "detail": {"source_path": repr(source_path)},
            }
        if not source.is_file():
            return None, None, None, {
                "kind": lp.SOURCE_NOT_FOUND,
                "message": "the supplied source is not a readable file",
                "detail": {"source_path": str(source)},
            }
        try:
            sha256 = lp._sha256(source)
            size_bytes = source.stat().st_size
        except OSError as exc:
            return None, None, None, {
                "kind": lp.SOURCE_NOT_FOUND,
                "message": "the supplied source could not be read",
                "detail": {"source_path": str(source), "error_type": type(exc).__name__},
            }
        return source, sha256, size_bytes, None

    def _retained_for_digest(self, paper_id, sha256, size_bytes):
        for entry in self._primary.manifest_rows():
            if (
                entry.paper_id == paper_id
                and entry.sha256 == sha256
                and entry.size_bytes == size_bytes
            ):
                return entry
        return None

    @staticmethod
    def _source_label(source_path, source_label):
        if source_label is not None:
            return str(source_label)
        try:
            return Path(source_path).name
        except TypeError:
            return None

    # -- ledger -------------------------------------------------------------

    def _result(self, status, *, hints, **fields):
        provenance = USER_SUPPLIED
        acquisition_id = _acquisition_id(provenance, hints, fields)
        return AcquisitionResult(
            status=status,
            provenance=provenance,
            acquisition_id=acquisition_id,
            **fields
        )

    @staticmethod
    def _reject(kind, message, detail, *, hints=None, **fields):
        provenance = USER_SUPPLIED
        acquisition_id = _acquisition_id(provenance, hints or {}, fields)
        return AcquisitionResult(
            status=REJECTED,
            provenance=provenance,
            kind=kind,
            detail=dict(detail or {}),
            message=message,
            acquisition_id=acquisition_id,
            **fields
        )

    def _record(self, result):
        row = _ledger_row(result)
        # Idempotent per attempt: an identical attempt (same channel, hints and
        # resolved artifact facts) is never recorded twice, so repeating an
        # ingestion grows neither the manifest nor the ledger.
        if not any(
            existing["acquisition_id"] == row["acquisition_id"]
            for existing in self._ledger
        ):
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(self._ledger_path), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            self._ledger.append(row)
        return result

    def _load_ledger(self):
        try:
            lines = self._ledger_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise IngestLedgerError(
                "cannot read acquisition ledger {}: {}".format(self._ledger_path, exc)
            ) from exc
        rows = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise IngestLedgerError(
                    "ledger line {} is blank".format(line_number)
                )
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise IngestLedgerError(
                    "ledger line {} is not valid JSON: {}".format(line_number, exc)
                ) from exc
            rows.append(self._validate_ledger_row(record, line_number))
        return rows

    def _validate_ledger_row(self, record, line_number):
        where = "ledger line {}".format(line_number)
        if not isinstance(record, dict):
            raise IngestLedgerError("{} must be a JSON object".format(where))
        unknown = sorted(set(record) - _LEDGER_ALLOWED_KEYS)
        missing = sorted(_LEDGER_REQUIRED_KEYS - set(record))
        if unknown:
            raise IngestLedgerError(
                "{} has unknown key(s): {}".format(where, ", ".join(unknown))
            )
        if missing:
            raise IngestLedgerError(
                "{} is missing key(s): {}".format(where, ", ".join(missing))
            )
        if record["schema_version"] != _SCHEMA_VERSION:
            raise IngestLedgerError(
                "{}.schema_version must be {}".format(where, _SCHEMA_VERSION)
            )
        for field_name in ("acquisition_id", "recorded_at"):
            value = record[field_name]
            if not isinstance(value, str) or not value.strip():
                raise IngestLedgerError(
                    "{}.{} must be a non-empty string".format(where, field_name)
                )
        if record["provenance"] not in _PROVENANCES:
            raise IngestLedgerError(
                "{}.provenance must be one of: {}".format(
                    where, ", ".join(_PROVENANCES)
                )
            )
        if record["status"] not in _STATUSES:
            raise IngestLedgerError(
                "{}.status must be one of: {}".format(where, ", ".join(_STATUSES))
            )
        if not isinstance(record["detail"], dict):
            raise IngestLedgerError("{}.detail must be an object".format(where))
        sha256 = record["sha256"]
        if sha256 is not None and (
            not isinstance(sha256, str) or not _SHA256.match(sha256)
        ):
            raise IngestLedgerError(
                "{}.sha256 must be null or 64 lowercase hex characters".format(where)
            )
        size_bytes = record["size_bytes"]
        if size_bytes is not None and (
            not isinstance(size_bytes, int) or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise IngestLedgerError(
                "{}.size_bytes must be null or a non-negative integer".format(where)
            )
        return record


def _acquisition_id(provenance, hints, fields):
    """Deterministic identity of an attempt, from its inputs only.

    Two attempts with the same channel, hints and resolved artifact facts share
    an id, so a repeated identical attempt is not recorded twice; a changed file
    or a changed identity hint yields a different id and a new record.
    """

    document = {
        "provenance": provenance,
        "hints": {key: hints[key] for key in sorted(hints)},
        "paper_id": fields.get("paper_id"),
        "role": fields.get("role"),
        "sha256": fields.get("sha256"),
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _ledger_row(result):
    return {
        "schema_version": _SCHEMA_VERSION,
        "acquisition_id": result.acquisition_id,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "provenance": result.provenance,
        "status": result.status,
        "paper_id": result.paper_id,
        "role": result.role,
        "sha256": result.sha256,
        "size_bytes": result.size_bytes,
        "media_type": result.media_type,
        "source_url": result.source_url,
        "source_label": result.source_label,
        "failure_kind": result.kind,
        "detail": dict(result.detail),
    }


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("validate", help="validate the acquisition ledger")
    ingest = commands.add_parser(
        "ingest", help="admit one user-supplied local PDF as a primary artifact"
    )
    ingest.add_argument("source_path", help="path to the local file to admit")
    ingest.add_argument("--paper-id", default=None, help="canonical paper_id (card slug)")
    ingest.add_argument("--doi", default=None)
    ingest.add_argument("--arxiv", default=None)
    ingest.add_argument("--title", default=None)
    ingest.add_argument("--source-url", default=None)
    ingest.add_argument("--role", default=None, choices=sorted(_KNOWN_ROLES))
    ingest.add_argument(
        "--source-label", default=None,
        help="opaque provenance label recorded in the ledger (never a path)",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    command = args.command or "validate"
    try:
        ingestor = LiteratureIngest()
        if command == "validate":
            ingestor.validate()
            print(
                "acquisition ledger: OK ({} attempt(s))".format(len(ingestor.attempts()))
            )
            return 0
        result = ingestor.ingest(
            args.source_path,
            paper_id=args.paper_id,
            doi=args.doi,
            arxiv=args.arxiv,
            title=args.title,
            source_url=args.source_url,
            role=args.role,
            source_label=args.source_label,
        )
        print(json.dumps(result.as_dict(), sort_keys=True))
        return 0 if result.admitted else 1
    except IngestLedgerError as exc:
        print(
            json.dumps(
                {"error": str(exc), "kind": "LEDGER_INVALID", "detail": {}},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    except (lp.PrimaryManifestError, literature_catalog.CatalogError) as exc:
        print(
            json.dumps({"error": str(exc), "kind": "STATE_INVALID", "detail": {}},
                       sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
