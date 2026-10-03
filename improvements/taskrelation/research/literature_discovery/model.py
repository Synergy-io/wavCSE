"""Normalized, provider-independent model for structured scholarly discovery.

Discovery answers *what might this paper be and where might its artifact live*.
It never admits, downloads or retains an artifact, and never creates a `Paper`
identity. Provider-shaped records from Crossref, arXiv, Semantic Scholar and
OpenReview are normalized into the two small records below; a provider's own
field names never reach the Literature Agent.

A `CandidatePaper` is a claim made by a provider, carrying provenance. A
`CandidateArtifactLocation` is an *unvalidated* pointer: it is discovery
evidence, never a `PrimaryArtifact`. Identity classification against the
canonical catalog uses the existing vocabulary in `literature_query`
(`known` / `new` / `ambiguous`) and is attached separately, because it is a
fact about our catalog, not about the provider.

Provider-returned strings are untrusted data. Nothing here evaluates them.
"""

import json
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


SCHEMA_VERSION = 1
RESULT_KIND = "DISCOVERY"

# -- failure taxonomy -------------------------------------------------------
# A discovery failure is a returned, structured value, never a raw exception.
PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
RATE_LIMITED = "RATE_LIMITED"
NOT_FOUND = "NOT_FOUND"
INVALID_QUERY = "INVALID_QUERY"
MALFORMED_PROVIDER_RESPONSE = "MALFORMED_PROVIDER_RESPONSE"
PROVIDER_AUTH_REQUIRED = "PROVIDER_AUTH_REQUIRED"
PROVIDER_ERROR = "PROVIDER_ERROR"
IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
DISCOVERY_EXHAUSTED = "DISCOVERY_EXHAUSTED"
# A bounded response that exceeded its byte cap. Shared with artifact
# acquisition, which sets a larger explicit cap.
RESPONSE_TOO_LARGE = "RESPONSE_TOO_LARGE"

FAILURE_KINDS = frozenset(
    {
        PROVIDER_UNAVAILABLE,
        RATE_LIMITED,
        NOT_FOUND,
        INVALID_QUERY,
        MALFORMED_PROVIDER_RESPONSE,
        PROVIDER_AUTH_REQUIRED,
        PROVIDER_ERROR,
        IDENTITY_CONFLICT,
        DISCOVERY_EXHAUSTED,
        RESPONSE_TOO_LARGE,
    }
)

# -- identity verdicts (the catalog's own vocabulary) ----------------------
KNOWN = "known"
NEW = "new"
AMBIGUOUS = "ambiguous"

# -- relation kinds --------------------------------------------------------
REFERENCE = "reference"
CITATION = "citation"

# -- artifact location classes ---------------------------------------------
ARTIFACT_PDF = "pdf"
ARTIFACT_HTML = "html"
ARTIFACT_LANDING = "landing"
ARTIFACT_ABSTRACT = "abstract"
ARTIFACT_OTHER = "other"

ACCESS_OPEN = "open"
ACCESS_UNKNOWN = "unknown"
ACCESS_RESTRICTED = "restricted"

# A provider-supplied abstract is bounded before it can reach model context.
ABSTRACT_MAX_CHARS = 1200


def bounded_text(text, limit=ABSTRACT_MAX_CHARS):
    if text is None:
        return None
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


@dataclass(frozen=True)
class CandidateArtifactLocation:
    """An unvalidated pointer to where an artifact *may* live.

    Never a `PrimaryArtifact`. The URL may 404, redirect, require a paywall, or
    serve HTML; nothing here has fetched or validated it.
    """

    url: str
    provider: str
    artifact_type: str = ARTIFACT_OTHER
    media_type: Optional[str] = None
    access: str = ACCESS_UNKNOWN
    evidence: str = ""
    version: Optional[str] = None

    def as_dict(self):
        return {
            "url": self.url,
            "provider": self.provider,
            "artifact_type": self.artifact_type,
            "media_type": self.media_type,
            "access": self.access,
            "evidence": self.evidence,
            "version": self.version,
            "validated": False,
        }


@dataclass(frozen=True)
class CandidatePaper:
    """One provider's normalized record of a scholarly work.

    Providers are never merged: two providers returning the same work produce
    two `CandidatePaper` records with distinct `provider`/`provider_record_id`.
    """

    provider: str
    provider_record_id: str
    title: Optional[str] = None
    authors: Tuple[str, ...] = ()
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    openreview_id: Optional[str] = None
    other_ids: Tuple[Tuple[str, str], ...] = ()
    abstract: Optional[str] = None
    url: Optional[str] = None
    artifact_locations: Tuple[CandidateArtifactLocation, ...] = ()
    provenance: Mapping = field(default_factory=dict)

    def identifiers(self):
        """Return the exact identifiers this candidate carries, most specific first."""

        ids = []
        if self.doi:
            ids.append(("doi", self.doi))
        if self.arxiv_id:
            ids.append(("arxiv", self.arxiv_id))
        if self.openreview_id:
            ids.append(("openreview", self.openreview_id))
        return ids

    def as_dict(self):
        return {
            "provider": self.provider,
            "provider_record_id": self.provider_record_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "openreview_id": self.openreview_id,
            "other_ids": [{"type": kind, "value": value} for kind, value in self.other_ids],
            "abstract": self.abstract,
            "url": self.url,
            "artifact_locations": [item.as_dict() for item in self.artifact_locations],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class IdentityVerdict:
    """The catalog's view of a set of identity fields; never a screening status."""

    status: str
    matched_paper_id: Optional[str] = None
    matched_field: Optional[str] = None
    conflicts: Tuple[str, ...] = ()

    def as_dict(self):
        return {
            "status": self.status,
            "matched_paper_id": self.matched_paper_id,
            "matched_field": self.matched_field,
            "conflicts": list(self.conflicts),
        }


@dataclass(frozen=True)
class DiscoveredCandidate:
    """A normalized candidate plus its catalog identity verdict and relation."""

    paper: CandidatePaper
    identity: IdentityVerdict
    relation: Optional[str] = None

    def as_dict(self):
        payload = self.paper.as_dict()
        payload["identity"] = self.identity.as_dict()
        payload["relation"] = self.relation
        return payload


@dataclass(frozen=True)
class DiscoveryFailure:
    """A structured provider or orchestration failure."""

    kind: str
    provider: Optional[str] = None
    message: str = ""
    detail: Mapping = field(default_factory=dict)

    def as_dict(self):
        return {
            "kind": self.kind,
            "provider": self.provider,
            "message": self.message,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class DiscoveryResult:
    """The normalized outcome of one discovery request."""

    mode: str
    query: Mapping
    candidates: Tuple[DiscoveredCandidate, ...] = ()
    failures: Tuple[DiscoveryFailure, ...] = ()
    identity: Optional[IdentityVerdict] = None
    providers_queried: Tuple[str, ...] = ()
    providers_failed: Tuple[str, ...] = ()
    limit: int = 0
    truncated: bool = False
    notes: Tuple[str, ...] = ()

    @property
    def returned(self):
        return len(self.candidates)

    @property
    def ok(self):
        """A well-formed request is `ok` even when some providers failed."""

        return all(failure.kind != INVALID_QUERY for failure in self.failures)

    def as_dict(self):
        return {
            "kind": RESULT_KIND,
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "query": dict(self.query),
            "identity": self.identity.as_dict() if self.identity else None,
            "returned": self.returned,
            "limit": self.limit,
            "truncated": self.truncated,
            "providers_queried": list(self.providers_queried),
            "providers_failed": list(self.providers_failed),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "failures": [failure.as_dict() for failure in self.failures],
            "notes": list(self.notes),
        }

    def to_json(self):
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


def other_ids(identifiers):
    """Normalize a provider identifier map into a sorted tuple of (type, value)."""

    pairs = set()
    for kind, value in (identifiers or {}).items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if item:
                    pairs.add((str(kind), str(item)))
        else:
            pairs.add((str(kind), str(value)))
    return tuple(sorted(pairs))
