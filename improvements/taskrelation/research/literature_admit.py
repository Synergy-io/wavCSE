"""Deterministic canonical-Paper admission from structured discovery (INC-017).

Structured discovery (INC-012) produces a *candidate* paper: external provider
metadata that is not canonical state. A `Paper` is a durable canonical
bibliographic identity, owned by ``literature/catalog.jsonl``. This module is the
one deterministic transition between them:

    CandidatePaper (normalized, provider-carried)
        -> identity/provenance validation against the catalog
        -> canonical Paper (new catalog row + canonical card)
        -> the existing LiteratureIngest / literature_acquire path
        -> PrimaryArtifact

Discovery is never admission, and model judgment never mutates a canonical
identity. The Literature Agent may *select* a paper it needs; this operator-side
primitive owns normalization, duplicate/conflict detection, identifier
validation, serialization and the idempotent catalog mutation.

Identity evidence, strongest first:

* DOI;
* arXiv identifier;
* OpenReview identifier;
* exact normalized bibliographic identity (title + authors + year).

Fuzzy title similarity is never proof of identity: only exact normalized
equality counts, and it is the weakest admissible evidence. A candidate whose
evidence points at two different existing Papers, or whose only match is a title
while a strong unrecorded identifier is present, fails closed.

Authority boundary (binding, mirrors ``AGENTS.md``):

* The only authority is a validated `CandidatePaper` (or a bounded set of them
  that share an identifier): it yields ADMITTED, KNOWN or a structured rejection.
  Arbitrary free-form Paper JSON is refused.
* An admission writes exactly three things: the canonical card, one
  ``catalog.jsonl`` row (inserted in sorted position, atomically) and one
  append-only ``canonicalizations.jsonl`` provenance row. It can create no Claim,
  PaperAssessment, card-synthesis, finding, decision, Study or proposal, and it
  never edits any other project research state.
* No credentials, headers or raw provider payloads are persisted: only provider
  names, provider record ids, normalized identifiers and a bounded detail.
* Admission is operator-side. It is deliberately absent from the model-facing
  ``.omp/tools/literature.ts`` surface, exactly as ingestion (INC-011) and public
  acquisition (INC-013) are, so the Literature Agent gains no catalog-write
  authority.

Usage::

    python -m improvements.taskrelation.research.literature_admit admit \
        --discovery /tmp/discovery.json --index 0
    python -m improvements.taskrelation.research.literature_admit admit \
        --provider arxiv --provider-record-id 2001.06782 \
        --arxiv 2001.06782 --title "Gradient Surgery for Multi-Task Learning" \
        --author "Tianhe Yu" --year 2020 --venue "NeurIPS 2020"
    python -m improvements.taskrelation.research.literature_admit validate
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Tuple
from urllib.parse import urlsplit

from improvements.taskrelation.research import literature_acquire as la
from improvements.taskrelation.research import literature_catalog
from improvements.taskrelation.research.literature_discovery import model as discovery_model


SCHEMA_VERSION = 1

# Admission outcomes: a caller branches on `status`, never on prose.
ADMITTED = "ADMITTED"
KNOWN = "KNOWN"
REJECTED = "REJECTED"
_STATUSES = (ADMITTED, KNOWN, REJECTED)

# Failure taxonomy. IDENTITY_CONFLICT is reused verbatim from discovery so the
# vocabulary stays one; the rest are admission-local, deterministic and
# branchable.
IDENTITY_CONFLICT = discovery_model.IDENTITY_CONFLICT
IDENTITY_INSUFFICIENT = "IDENTITY_INSUFFICIENT"
AMBIGUOUS_PAPER = "AMBIGUOUS_PAPER"
INVALID_CANDIDATE = "INVALID_CANDIDATE"
CATALOG_CONFLICT = "CATALOG_CONFLICT"
SOURCE_CORRESPONDENCE_INVALID = "SOURCE_CORRESPONDENCE_INVALID"

FAILURE_KINDS = frozenset(
    {
        IDENTITY_CONFLICT, IDENTITY_INSUFFICIENT, AMBIGUOUS_PAPER,
        INVALID_CANDIDATE, CATALOG_CONFLICT, SOURCE_CORRESPONDENCE_INVALID,
    }
)

# Identity fields, strongest first. Title is a *tier*, not a strong identifier.
_IDENTITY_FIELDS = ("doi", "arxiv", "openreview", "title")
_FIELD_LABEL = {
    "doi": "DOI", "arxiv": "arXiv", "openreview": "OpenReview", "title": "title",
}

_DOI = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9]+$", re.IGNORECASE)
_ARXIV = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)

# A title contributes at most this many words to a generated paper_id, and a
# generated id is refused rather than silently suffixed when it collides.
_TITLE_SLUG_WORDS = 4
_SOURCE_URL_LIMIT = 5

# Words dropped from the title slug so ``... Using Uncertainty to ...`` yields a
# readable id. Purely mechanical: the identity is the full title, never the slug.
_TITLE_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "for", "and", "on", "in", "to", "with", "using",
        "via", "is", "are", "at", "as", "by", "from", "its", "into", "over",
        "under", "we", "our", "toward", "towards", "based", "through",
    }
)

# Candidate URL hosts that are provider *metadata/API* pages, never a canonical
# bibliographic source; such a URL is not admitted as a Paper source URL.
_METADATA_HOSTS = frozenset(
    {
        "api.crossref.org", "api.semanticscholar.org", "export.arxiv.org",
        "api.openreview.net", "api2.openreview.net",
    }
)

_LEDGER_FILENAME = "canonicalizations.jsonl"
_LEDGER_BASE_KEYS = {
    "schema_version", "admission_id", "recorded_at", "status", "paper_id",
    "kind", "matched_field", "identifiers", "provenance", "detail",
}
_LEDGER_IDENTIFIER_KEYS = {"doi", "arxiv", "openreview", "title"}
_LEDGER_PROVENANCE_KEYS = {"provider", "provider_record_id"}

_RESEARCH_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = _RESEARCH_DIR.parents[2]
_DEFAULT_LITERATURE_DIR = _RESEARCH_DIR / "literature"
_DEFAULT_CATALOG = _DEFAULT_LITERATURE_DIR / "catalog.jsonl"

# Catalog row key order mirrors the existing corpus so a new row is
# indistinguishable in shape from the hand-established ones.
_CATALOG_KEY_ORDER = (
    "schema_version", "paper_id", "card_path", "title", "year", "authors",
    "venue", "external_ids", "source_urls", "aliases",
)


class AdmissionError(ValueError):
    """An admission request is malformed or its evidence disagrees."""

    def __init__(self, message, kind=None, detail=None):
        super(AdmissionError, self).__init__(message)
        self.kind = kind
        self.detail = dict(detail or {})


class AdmissionLedgerError(ValueError):
    """The canonicalization ledger is unusable."""


@dataclass(frozen=True)
class AdmissionResult:
    """The structured outcome of one candidate-admission attempt."""

    status: str
    paper_id: Optional[str] = None
    matched_field: Optional[str] = None
    matched_identifiers: Tuple[Tuple[str, str], ...] = ()
    source_urls: Tuple[str, ...] = ()
    provenance: Tuple[Mapping, ...] = ()
    kind: Optional[str] = None
    detail: Mapping = field(default_factory=dict)
    message: Optional[str] = None
    admission_id: Optional[str] = None

    @property
    def admitted(self):
        return self.status == ADMITTED

    def as_dict(self):
        return {
            "kind": "ADMISSION",
            "status": self.status,
            "admitted": self.admitted,
            "paper_id": self.paper_id,
            "matched_field": self.matched_field,
            "matched_identifiers": [
                {"field": name, "value": value}
                for name, value in self.matched_identifiers
            ],
            "source_urls": list(self.source_urls),
            "provenance": [dict(item) for item in self.provenance],
            "failure_kind": self.kind,
            "detail": dict(self.detail),
            "message": self.message,
            "admission_id": self.admission_id,
        }

    def to_json(self):
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class _Identity:
    """The normalized, merged identity of a candidate set."""

    title: str
    title_key: str
    authors: Tuple[str, ...]
    year: int
    venue: str
    doi: Optional[str] = None
    arxiv: Optional[str] = None
    openreview: Optional[str] = None

    def lookup_items(self):
        """Return (field, lookup value, canonical value) strongest first."""

        items = []
        if self.doi:
            items.append(("doi", self.doi, self.doi))
        if self.arxiv:
            items.append(("arxiv", self.arxiv, self.arxiv))
        if self.openreview:
            items.append(
                ("openreview", _openreview_url(self.openreview), self.openreview)
            )
        if self.title:
            items.append(("title", self.title, self.title))
        return items

    def has_strong(self):
        return bool(self.doi or self.arxiv or self.openreview)


class LiteratureAdmission:
    """Deterministic CandidatePaper -> canonical Paper admission."""

    def __init__(self, *, catalog_path=None, repo_root=None, literature_dir=None,
                 ledger_path=None, clock=None):
        self._catalog_path = Path(catalog_path) if catalog_path is not None else (
            _DEFAULT_CATALOG
        )
        self._repo_root = Path(repo_root) if repo_root is not None else (
            _DEFAULT_REPO_ROOT
        )
        self._literature_dir = (
            Path(literature_dir) if literature_dir is not None
            else self._catalog_path.parent
        )
        self._ledger_path = Path(ledger_path) if ledger_path is not None else (
            self._literature_dir / _LEDGER_FILENAME
        )
        self._clock = clock
        self._catalog = self._load_catalog()

    # -- introspection -----------------------------------------------------

    @property
    def catalog_path(self):
        return self._catalog_path

    @property
    def catalog(self):
        return self._catalog

    @property
    def ledger_path(self):
        return self._ledger_path

    def attempts(self):
        """Return every recorded admission attempt, in ledger order."""

        return tuple(self._load_ledger())

    # -- public interface --------------------------------------------------

    def admit(self, candidates):
        """Validate and admit one candidate paper, or a set of provider records.

        ``candidates`` is one :class:`~literature_discovery.model.CandidatePaper`
        or a non-empty sequence of them that must describe the same work. Never
        raises for a domain failure: returns an :class:`AdmissionResult` a caller
        branches on.
        """

        try:
            papers = _normalize_input(candidates)
        except AdmissionError as exc:
            return self._record(self._reject(INVALID_CANDIDATE, str(exc), {}))

        try:
            identity = _merge_identity(papers)
        except AdmissionError as exc:
            kind = exc.kind or INVALID_CANDIDATE
            return self._record(self._reject(kind, str(exc), exc.detail))

        provenance = tuple(
            {"provider": paper.provider, "provider_record_id": paper.provider_record_id}
            for paper in papers
        )

        matches, matched_field = self._match_catalog(identity)
        distinct = set(matches.values())
        if len(distinct) > 1:
            detail = {
                "matched": [
                    {"field": name, "paper_id": paper_id}
                    for name, paper_id in sorted(matches.items())
                ]
            }
            return self._record(
                self._reject(
                    IDENTITY_CONFLICT,
                    "candidate evidence resolves to more than one existing Paper",
                    detail,
                    provenance=provenance,
                    source_urls=self._source_urls(papers, identity),
                )
            )
        if len(distinct) == 1:
            paper_id = next(iter(distinct))
            if set(matches) == {"title"} and identity.has_strong():
                return self._record(
                    self._reject(
                        AMBIGUOUS_PAPER,
                        "the title matches {!r} but a supplied strong identifier is "
                        "not recorded; refusing to merge".format(paper_id),
                        {"title_paper_id": paper_id},
                        provenance=provenance,
                    )
                )
            return self._record(
                self._known(paper_id, matched_field, matches, provenance)
            )

        return self._record(self._admit_new(papers, identity, provenance))

    # -- identity resolution ----------------------------------------------

    def _match_catalog(self, identity):
        """Resolve every present identity field; return (field -> paper_id, best)."""

        matches = {}
        for name, lookup_value, _canonical in identity.lookup_items():
            try:
                entry = self._catalog.lookup(lookup_value)
            except literature_catalog.CatalogError:
                continue
            matches[name] = entry.paper_id
        best = None
        for name in _IDENTITY_FIELDS:
            if name in matches:
                best = name
                break
        return matches, best

    # -- admission ---------------------------------------------------------

    def _admit_new(self, papers, identity, provenance):
        failure = _sufficiency_failure(identity)
        if failure is not None:
            kind, message, detail = failure
            return self._reject(kind, message, detail, provenance=provenance)

        external_ids = {}
        if identity.doi:
            external_ids["doi"] = [identity.doi]
        if identity.arxiv:
            external_ids["arxiv"] = [identity.arxiv]

        source_urls = self._source_urls(papers, identity)
        if not source_urls:
            return self._reject(
                SOURCE_CORRESPONDENCE_INVALID,
                "no canonical source URL could be established for the candidate",
                {},
                provenance=provenance,
            )

        paper_id = _paper_id(identity)
        if paper_id in self._catalog._paper_ids:
            return self._reject(
                CATALOG_CONFLICT,
                "generated paper_id {!r} already identifies another Paper; "
                "refusing to reuse it".format(paper_id),
                {"paper_id": paper_id},
                provenance=provenance,
                source_urls=source_urls,
            )

        card_path = self._card_path(paper_id)
        card_text = _render_card(identity, source_urls, external_ids)
        record = {
            "schema_version": literature_catalog.SCHEMA_VERSION,
            "paper_id": paper_id,
            "card_path": card_path,
            "title": identity.title,
            "year": identity.year,
            "authors": list(identity.authors),
            "venue": identity.venue,
            "external_ids": external_ids,
            "source_urls": list(source_urls),
            "aliases": [],
        }

        # Validate the row against the catalog schema before touching the disk.
        try:
            entry = literature_catalog._validate_record(record, 0)
        except literature_catalog.CatalogError as exc:
            return self._reject(
                INVALID_CANDIDATE, "candidate does not form a canonical Paper",
                {"reason": str(exc)},
            )

        self._write_card(card_path, card_text)
        self._insert_catalog_row(record)

        return AdmissionResult(
            status=ADMITTED,
            paper_id=entry.paper_id,
            matched_field=None,
            matched_identifiers=tuple(
                (name, value) for name, value, _c in identity.lookup_items()
            ),
            source_urls=tuple(source_urls),
            provenance=provenance,
            message="admitted {!r} as a canonical Paper".format(entry.paper_id),
        )

    def _known(self, paper_id, matched_field, matches, provenance):
        entry = self._catalog.get(paper_id)
        return AdmissionResult(
            status=KNOWN,
            paper_id=paper_id,
            matched_field=matched_field,
            matched_identifiers=tuple(
                (name, self._matched_value(matches, entry, name))
                for name in _IDENTITY_FIELDS
                if name in matches
            ),
            source_urls=entry.source_urls,
            provenance=provenance,
            message="already a canonical Paper; no new catalog row",
        )

    @staticmethod
    def _matched_value(matches, entry, field):
        if field == "title":
            return entry.title
        if field == "doi":
            return entry.external_ids["doi"][0]
        if field == "arxiv":
            return entry.external_ids["arxiv"][0]
        return entry.paper_id

    # -- source URLs -------------------------------------------------------

    def _source_urls(self, papers, identity):
        """Return the canonical, deterministic source URLs for an admission."""

        urls = []
        seen = set()

        def add(url):
            if not isinstance(url, str) or not url.strip():
                return
            if not literature_catalog._looks_like_url(url):
                return
            key = literature_catalog._normalize_url(url)
            if key in seen:
                return
            seen.add(key)
            urls.append(url.strip())

        if identity.doi:
            add("https://doi.org/{}".format(identity.doi))
        if identity.arxiv:
            add("https://arxiv.org/abs/{}".format(identity.arxiv))
        if identity.openreview:
            add(_openreview_url(identity.openreview))
        for paper in sorted(papers, key=lambda item: (item.provider, item.provider_record_id)):
            candidate_url = paper.url
            if not candidate_url:
                continue
            host = (urlsplit(candidate_url).hostname or "").lower()
            if host in _METADATA_HOSTS:
                continue
            if la.AcquisitionPolicy.classify(host) == la.UNKNOWN:
                continue
            add(candidate_url)
        return tuple(urls[:_SOURCE_URL_LIMIT])

    # -- filesystem --------------------------------------------------------

    def _load_catalog(self):
        return literature_catalog.load_catalog(
            self._catalog_path,
            repo_root=self._repo_root,
            validate_references=False,
        )

    def _card_path(self, paper_id):
        literature_dir = self._literature_dir
        try:
            relative_dir = literature_dir.resolve().relative_to(self._repo_root.resolve())
        except ValueError:
            relative_dir = literature_dir
        return (Path(relative_dir) / (paper_id + ".md")).as_posix()

    def _write_card(self, card_path, card_text):
        target = self._repo_root / card_path
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, card_text.encode("utf-8"))

    def _insert_catalog_row(self, record):
        try:
            lines = self._catalog_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AdmissionError(
                "cannot read catalog {}: {}".format(self._catalog_path, exc)
            ) from exc
        new_line = json.dumps(record, separators=(",", ":"))
        entry_lines = []
        for line in lines:
            if not line.strip():
                continue
            parsed = json.loads(line)
            entry_lines.append((parsed["paper_id"], line))
        entry_lines.append((record["paper_id"], new_line))
        entry_lines.sort(key=lambda item: item[0])
        serialized = "".join(line + "\n" for _pid, line in entry_lines)
        self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self._catalog_path, serialized.encode("utf-8"))
        self._catalog = self._load_catalog()

    # -- ledger ------------------------------------------------------------

    def _reject(self, kind, message, detail, *, provenance=(), source_urls=()):
        return AdmissionResult(
            status=REJECTED,
            kind=kind,
            detail=dict(detail or {}),
            message=message,
            provenance=tuple(provenance),
            source_urls=tuple(source_urls),
        )

    def _record(self, result):
        row = _ledger_row(result, self._now())
        if not any(
            existing["admission_id"] == row["admission_id"]
            for existing in self._load_ledger()
        ):
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(self._ledger_path), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        result = AdmissionResult(
            status=result.status,
            paper_id=result.paper_id,
            matched_field=result.matched_field,
            matched_identifiers=result.matched_identifiers,
            source_urls=result.source_urls,
            provenance=result.provenance,
            kind=result.kind,
            detail=result.detail,
            message=result.message,
            admission_id=row["admission_id"],
        )
        return result

    def _now(self):
        if self._clock is not None:
            value = self._clock()
            if isinstance(value, datetime):
                return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()
            return str(value)
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _load_ledger(self):
        try:
            lines = self._ledger_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise AdmissionLedgerError(
                "cannot read canonicalization ledger {}: {}".format(
                    self._ledger_path, exc
                )
            ) from exc
        rows = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise AdmissionLedgerError(
                    "ledger line {} is blank".format(line_number)
                )
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise AdmissionLedgerError(
                    "ledger line {} is not valid JSON: {}".format(line_number, exc)
                ) from exc
            rows.append(_validate_ledger_row(record, line_number))
        return rows

    def validate(self):
        """Validate the canonicalization ledger; return the row count."""

        return len(self._load_ledger())


# -- candidate normalization -------------------------------------------------


def _normalize_input(candidates):
    if candidates is None:
        raise AdmissionError("a CandidatePaper (or a non-empty sequence) is required")
    if isinstance(candidates, discovery_model.CandidatePaper):
        papers = [candidates]
    elif isinstance(candidates, (list, tuple)):
        papers = list(candidates)
    else:
        raise AdmissionError(
            "expected a CandidatePaper or a sequence of them, got {}".format(
                type(candidates).__name__
            )
        )
    if not papers:
        raise AdmissionError("at least one CandidatePaper is required")
    for index, paper in enumerate(papers):
        if not isinstance(paper, discovery_model.CandidatePaper):
            raise AdmissionError(
                "candidate[{}] must be a CandidatePaper, got {}".format(
                    index, type(paper).__name__
                )
            )
        if not paper.provider or not paper.provider.strip():
            raise AdmissionError("candidate[{}] has no provider".format(index))
        if not paper.provider_record_id or not paper.provider_record_id.strip():
            raise AdmissionError(
                "candidate[{}] has no provider_record_id".format(index)
            )
    return tuple(papers)


def _candidate_from_dict(document):
    """Reconstruct a strict CandidatePaper from a discovery-result candidate map.

    Provider payloads never reach this function: only the normalized fields
    ``literature_discovery`` emits are accepted, and everything else — including
    a provider's own raw fields — is refused.
    """

    if not isinstance(document, Mapping):
        raise AdmissionError("a discovery candidate must be a JSON object")
    known = {
        "provider", "provider_record_id", "title", "authors", "year", "venue",
        "doi", "arxiv_id", "openreview_id", "other_ids", "abstract", "url",
        "artifact_locations", "provenance", "identity", "relation",
    }
    unknown = sorted(set(document) - known)
    if unknown:
        raise AdmissionError(
            "discovery candidate has unexpected key(s): {}".format(", ".join(unknown))
        )
    provider = document.get("provider")
    record_id = document.get("provider_record_id")
    if not isinstance(provider, str) or not provider.strip():
        raise AdmissionError("discovery candidate has no provider")
    if not isinstance(record_id, str) or not record_id.strip():
        raise AdmissionError("discovery candidate has no provider_record_id")
    authors = document.get("authors") or ()
    if not isinstance(authors, (list, tuple)):
        raise AdmissionError("discovery candidate authors must be a list")
    year = document.get("year")
    if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
        raise AdmissionError("discovery candidate year must be an integer or null")
    other_ids = document.get("other_ids") or ()
    pairs = []
    for item in other_ids:
        if not isinstance(item, Mapping) or "type" not in item or "value" not in item:
            raise AdmissionError("discovery candidate other_ids entries are malformed")
        pairs.append((str(item["type"]), str(item["value"])))
    locations = []
    for item in document.get("artifact_locations") or ():
        if not isinstance(item, Mapping) or not item.get("url"):
            raise AdmissionError("discovery candidate artifact location is malformed")
        locations.append(
            discovery_model.CandidateArtifactLocation(
                url=str(item["url"]),
                provider=str(item.get("provider") or provider),
                artifact_type=str(item.get("artifact_type") or discovery_model.ARTIFACT_OTHER),
                media_type=item.get("media_type"),
                access=str(item.get("access") or discovery_model.ACCESS_UNKNOWN),
                evidence=str(item.get("evidence") or ""),
                version=item.get("version"),
            )
        )
    return discovery_model.CandidatePaper(
        provider=provider.strip(),
        provider_record_id=record_id.strip(),
        title=_optional_str(document.get("title")),
        authors=tuple(str(author) for author in authors),
        year=year,
        venue=_optional_str(document.get("venue")),
        doi=_optional_str(document.get("doi")),
        arxiv_id=_optional_str(document.get("arxiv_id")),
        openreview_id=_optional_str(document.get("openreview_id")),
        other_ids=tuple(pairs),
        abstract=_optional_str(document.get("abstract")),
        url=_optional_str(document.get("url")),
        artifact_locations=tuple(locations),
        provenance={},
    )


def _optional_str(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdmissionError("expected a string, got {}".format(type(value).__name__))
    value = value.strip()
    return value or None


# -- identity merge ----------------------------------------------------------


def _merge_identity(papers):
    """Merge compatible provider records of the same work into one identity.

    Providers disagreeing on a strong identifier, or on the title, fail closed
    rather than being merged; nothing is chosen because a provider ran first.
    """

    dois = _distinct(papers, "doi", _normalize_doi)
    arxivs = _distinct(papers, "arxiv_id", _normalize_arxiv)
    openreviews = _distinct(papers, "openreview_id", _clean_id)
    titles = _distinct(papers, "title", _normalize_title)

    for field, values in (("doi", dois), ("arxiv", arxivs), ("openreview", openreviews)):
        if len(values) > 1:
            raise AdmissionError(
                "providers disagree on the {} identifier".format(_FIELD_LABEL[field]),
                IDENTITY_CONFLICT,
                {"field": field, "values": list(values)},
            )
    if len(titles) > 1:
        raise AdmissionError(
            "providers disagree on the work's title; refusing to merge",
            AMBIGUOUS_PAPER,
            {"titles": list(titles)},
        )

    title = _select_title(papers)
    authors = _select_authors(papers)
    year = _select_year(papers)
    venue = _select_venue(papers)
    return _Identity(
        title=title,
        title_key=_normalize_title(title),
        authors=authors,
        year=year,
        venue=venue,
        doi=dois[0] if dois else None,
        arxiv=arxivs[0] if arxivs else None,
        openreview=openreviews[0] if openreviews else None,
    )


def _distinct(papers, attribute, normalize):
    values = set()
    for paper in papers:
        raw = getattr(paper, attribute)
        if raw is None or not str(raw).strip():
            continue
        values.add(normalize(str(raw)))
    return tuple(sorted(values))


def _select_title(papers):
    titles = [paper.title for paper in papers if paper.title and paper.title.strip()]
    if not titles:
        return ""
    key = _normalize_title(titles[0])
    variants = {title.strip() for title in titles if _normalize_title(title) == key}
    return sorted(variants, key=lambda value: (-len(value), value))[0]


def _select_authors(papers):
    candidates = [
        tuple(author.strip() for author in paper.authors if author and author.strip())
        for paper in papers
        if paper.authors
    ]
    candidates = [authors for authors in candidates if authors]
    if not candidates:
        return ()
    return sorted(candidates, key=lambda value: (-len(value), value))[0]


def _select_year(papers):
    years = [paper.year for paper in papers if isinstance(paper.year, int)
             and not isinstance(paper.year, bool)]
    return max(years) if years else None


def _select_venue(papers):
    venues = [paper.venue.strip() for paper in papers if paper.venue and paper.venue.strip()]
    if not venues:
        return None
    return sorted(venues, key=lambda value: (-len(value), value))[0]


def _sufficiency_failure(identity):
    if not identity.title:
        return (IDENTITY_INSUFFICIENT, "the candidate has no title", {})
    if not identity.authors:
        return (IDENTITY_INSUFFICIENT, "the candidate has no authors", {})
    if identity.year is None:
        return (IDENTITY_INSUFFICIENT, "the candidate has no year", {})
    if not (1800 <= identity.year <= 2100):
        return (
            IDENTITY_INSUFFICIENT,
            "the candidate year is outside the supported range",
            {"year": identity.year},
        )
    return None


# -- normalization helpers ---------------------------------------------------


def _normalize_title(value):
    return literature_catalog._normalize_title(str(value))


def _normalize_doi(value):
    normalized = literature_catalog._normalize_doi(str(value))
    if not _DOI.match(normalized):
        raise AdmissionError("DOI {!r} is not a valid DOI".format(value),
                             INVALID_CANDIDATE, {"doi": str(value)})
    return normalized


def _normalize_arxiv(value):
    normalized = literature_catalog._normalize_arxiv(str(value))
    if not _ARXIV.match(normalized):
        raise AdmissionError("arXiv identifier {!r} is not valid".format(value),
                             INVALID_CANDIDATE, {"arxiv": str(value)})
    return normalized


def _clean_id(value):
    return str(value).strip().casefold()


def _openreview_url(identifier):
    return "https://openreview.net/forum?id={}".format(identifier)


def _paper_id(identity):
    family = _slug_token(identity.authors[0].split()[-1] if identity.authors else "")
    if not family:
        family = "paper"
    words = [
        token for token in _normalize_title(identity.title).split()
        if token not in _TITLE_STOPWORDS
    ]
    if not words:
        words = _normalize_title(identity.title).split()
    slug = "-".join(_slug_token(token) for token in words[:_TITLE_SLUG_WORDS])
    slug = slug.strip("-") or "paper"
    return "{}-{}-{}".format(family, identity.year, slug)


def _slug_token(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _render_card(identity, source_urls, external_ids):
    citation = "{} “{}.” {}, {}.".format(
        ", ".join(identity.authors), identity.title, identity.venue,
        identity.year,
    )
    lines = [
        "# {}".format(identity.title),
        "",
        "## Citation",
        "",
        citation,
        "",
        "Primary source: {}".format(source_urls[0]),
    ]
    for url in source_urls[1:]:
        lines.append("Also: {}".format(url))
    lines.append("")
    for id_type in ("doi", "arxiv"):
        for value in external_ids.get(id_type, ()):
            lines.append("{}: {}".format(_FIELD_LABEL[id_type], value))
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path, data):
    descriptor, partial = tempfile.mkstemp(
        prefix=".{}.".format(path.name), dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, str(path))
    except BaseException:
        try:
            os.unlink(partial)
        except OSError:
            pass
        raise


# -- ledger ------------------------------------------------------------------


def _ledger_row(result, recorded_at):
    identifiers = {
        name: value for name, value in result.matched_identifiers
    }
    row = {
        "schema_version": SCHEMA_VERSION,
        "admission_id": _admission_id(result),
        "recorded_at": recorded_at,
        "status": result.status,
        "paper_id": result.paper_id,
        "kind": result.kind,
        "matched_field": result.matched_field,
        "identifiers": _bounded_identifiers(identifiers),
        "provenance": [
            {key: item[key] for key in sorted(_LEDGER_PROVENANCE_KEYS) if key in item}
            for item in result.provenance
        ][:16],
        "detail": _bounded_detail(result.detail),
    }
    return row


def _bounded_identifiers(identifiers):
    bounded = {}
    for key in _LEDGER_IDENTIFIER_KEYS:
        value = identifiers.get(key)
        if value is None:
            continue
        bounded[key] = str(value)[:512]
    return bounded


def _bounded_detail(detail):
    try:
        text = json.dumps(detail, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        text = json.dumps({"unserializable": True})
    if len(text) > 2048:
        return {"truncated": True, "preview": text[:2048]}
    return json.loads(text)


def _admission_id(result):
    document = {
        "status": result.status,
        "paper_id": result.paper_id,
        "kind": result.kind,
        "matched_field": result.matched_field,
        "identifiers": {name: value for name, value in result.matched_identifiers},
        "provenance": sorted(
            (item.get("provider", ""), item.get("provider_record_id", ""))
            for item in result.provenance
        ),
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _validate_ledger_row(record, line_number):
    where = "canonicalization ledger line {}".format(line_number)
    if not isinstance(record, dict):
        raise AdmissionLedgerError("{} must be a JSON object".format(where))
    unknown = sorted(set(record) - _LEDGER_BASE_KEYS)
    missing = sorted(_LEDGER_BASE_KEYS - set(record))
    if unknown:
        raise AdmissionLedgerError(
            "{} has unknown key(s): {}".format(where, ", ".join(unknown))
        )
    if missing:
        raise AdmissionLedgerError(
            "{} is missing key(s): {}".format(where, ", ".join(missing))
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise AdmissionLedgerError(
            "{}.schema_version must be {}".format(where, SCHEMA_VERSION)
        )
    if record["status"] not in _STATUSES:
        raise AdmissionLedgerError("{}.status is not an admission status".format(where))
    for key in ("admission_id", "recorded_at"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise AdmissionLedgerError(
                "{}.{} must be a non-empty string".format(where, key)
            )
    if record["kind"] is not None and record["kind"] not in FAILURE_KINDS:
        raise AdmissionLedgerError("{}.kind is not a known failure kind".format(where))
    if not isinstance(record["identifiers"], dict):
        raise AdmissionLedgerError("{}.identifiers must be an object".format(where))
    if not isinstance(record["provenance"], list):
        raise AdmissionLedgerError("{}.provenance must be a list".format(where))
    if not isinstance(record["detail"], dict):
        raise AdmissionLedgerError("{}.detail must be an object".format(where))
    return record


# -- CLI ---------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("validate", help="validate the canonicalization ledger")

    admit = commands.add_parser(
        "admit", help="admit a discovered CandidatePaper as a canonical Paper"
    )
    admit.add_argument(
        "--discovery",
        help="path to a `literature_discovery` result JSON file",
    )
    admit.add_argument(
        "--index", type=int, default=0,
        help="which candidate in the discovery result to admit (default 0)",
    )
    admit.add_argument("--provider", default=None)
    admit.add_argument("--provider-record-id", default=None)
    admit.add_argument("--doi", default=None)
    admit.add_argument("--arxiv", default=None)
    admit.add_argument("--openreview", default=None)
    admit.add_argument("--title", default=None)
    admit.add_argument("--author", action="append", default=[])
    admit.add_argument("--year", type=int, default=None)
    admit.add_argument("--venue", default=None)
    admit.add_argument("--url", default=None)
    return parser


def _candidates_from_args(args):
    if args.discovery:
        document = json.loads(Path(args.discovery).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or document.get("kind") != discovery_model.RESULT_KIND:
            raise AdmissionError(
                "--discovery must be a structured discovery result "
                "(kind == {!r})".format(discovery_model.RESULT_KIND)
            )
        candidates = document.get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            raise AdmissionError("the discovery result has no candidates to admit")
        if args.index < 0 or args.index >= len(candidates):
            raise AdmissionError("--index is outside the discovery result")
        return [_candidate_from_dict(candidates[args.index])]
    if not args.provider or not args.provider_record_id:
        raise AdmissionError(
            "--provider and --provider-record-id are required when --discovery "
            "is not supplied"
        )
    return [
        discovery_model.CandidatePaper(
            provider=args.provider,
            provider_record_id=args.provider_record_id,
            title=args.title,
            authors=tuple(args.author),
            year=args.year,
            venue=args.venue,
            doi=args.doi,
            arxiv_id=args.arxiv,
            openreview_id=args.openreview,
            url=args.url,
        )
    ]


def main(argv=None):
    args = _build_parser().parse_args(argv)
    admission = LiteratureAdmission()
    if args.command == "validate":
        try:
            count = admission.validate()
        except AdmissionLedgerError as exc:
            print("canonicalization ledger: ERROR: {}".format(exc), file=sys.stderr)
            return 1
        print("canonicalization ledger: OK ({} attempts)".format(count))
        return 0
    if args.command != "admit":
        _build_parser().print_help()
        return 2
    try:
        candidates = _candidates_from_args(args)
    except (AdmissionError, OSError, ValueError) as exc:
        payload = {
            "kind": "ADMISSION",
            "status": REJECTED,
            "admitted": False,
            "failure_kind": INVALID_CANDIDATE,
            "message": str(exc),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1
    result = admission.admit(candidates)
    print(result.to_json())
    return 0 if result.status in (ADMITTED, KNOWN) else 1


if __name__ == "__main__":
    sys.exit(main())
