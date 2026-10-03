"""Deterministic public primary-artifact acquisition (INC-013).

This is the second acquisition entry path (roadmap INC-013): a candidate artifact
location reported by structured scholarly discovery is turned into validated,
admitted bytes for a *known* canonical Paper. It is tooling, not a browsing
agent: every step is deterministic, bounded and fail-closed, and the model never
performs any of them.

    CandidateArtifactLocation
        -> source-policy check (scheme, host class, SSRF)
        -> bounded HTTPS retrieval (reusing INC-012's ``HttpFetcher``)
        -> artifact/content validation (size, media type, %PDF- header, readability)
        -> scholarly identity/version validation (DOI / arXiv / title evidence)
        -> exact-byte SHA-256 (computed once, by LiteratureIngest)
        -> LiteratureIngest(provenance=PUBLIC_ACQUIRED)
        -> LiteraturePrimary.register -> PrimaryArtifact + acquisition ledger row

The admission pipeline is **not duplicated**: validated bytes are written to a
disposable temporary file and handed to the existing
:class:`literature_ingest.LiteratureIngest`, which owns byte validation, SHA-256,
duplicate/version handling, immutable local retention and the manifest row. This
module adds only the network and identity steps the operator-side ``ingest``
primitive cannot express, and records how the bytes arrived in the same
append-only ledger.

Authority boundary (binding, mirrors ``AGENTS.md``):

* Retrieved bytes and every response header are **untrusted data**. Nothing here
  evaluates provider or PDF text; PDF text is read only to look for a DOI, an
  arXiv id or the paper's title, is bounded, and never becomes an instruction.
* URLs are **hostile**, because a model can influence which candidate location is
  attempted. The retriever refuses non-HTTPS schemes, embedded credentials,
  non-standard ports, restricted hosts, unknown hosts, and any host that resolves
  into loopback / link-local / private / metadata address space; every redirect
  target is re-checked against the same policy before it is followed.
* An attempt writes exactly the same three things as user-supplied ingestion: the
  disposable primary cache and the primary manifest (both through
  ``literature_primary.register``) and the acquisition ledger. It can create no
  Claim, PaperAssessment, card, synthesis, finding, decision, Study or proposal,
  never edits the catalog, and never edits project research state.
* No model-facing tool is added: the Literature Agent's evidence surface stays
  read-only. The agent may *select* a candidate location that
  ``literature_discover`` returned; acquisition itself is operator-side, exactly
  as INC-011 ingestion is.

Usage::

    python -m improvements.taskrelation.research.literature_acquire acquire \
        --paper-id zhang-yang-2021-mtl-survey \
        --url https://arxiv.org/pdf/1707.08114v3 \
        --provider arxiv --arxiv 1707.08114v3
    python -m improvements.taskrelation.research.literature_acquire policy \
        --url https://arxiv.org/pdf/1707.08114v3
    python -m improvements.taskrelation.research.literature_acquire validate
"""

import argparse
import ipaddress
import json
import re
import socket
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Tuple
from urllib.parse import urlsplit

from improvements.taskrelation.research import literature_catalog
from improvements.taskrelation.research import literature_ingest as li
from improvements.taskrelation.research import literature_primary as lp
from improvements.taskrelation.research import literature_primary_text as lpt
from improvements.taskrelation.research.literature_discovery import http as http_layer
from improvements.taskrelation.research.literature_discovery import model as discovery_model


# Admission outcomes: a caller branches on `status`, never on prose.
ACQUIRED = "ACQUIRED"
DUPLICATE = "DUPLICATE"
FAILED = "FAILED"
_STATUSES = (ACQUIRED, DUPLICATE, FAILED)

# The deterministic acquisition failure taxonomy. Every reason is a returned,
# structured value; no raw HTTP or socket exception crosses this interface.
POLICY_BLOCKED = "POLICY_BLOCKED"
ACCESS_RESTRICTED = "ACCESS_RESTRICTED"
NOT_FOUND = "NOT_FOUND"
RATE_LIMITED = "RATE_LIMITED"
NETWORK_FAILURE = "NETWORK_FAILURE"
RESPONSE_TOO_LARGE = "RESPONSE_TOO_LARGE"
INVALID_CONTENT_TYPE = "INVALID_CONTENT_TYPE"
INVALID_ARTIFACT = "INVALID_ARTIFACT"
IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
IDENTITY_INSUFFICIENT = "IDENTITY_INSUFFICIENT"
VERSION_REQUIREMENT_UNSATISFIED = "VERSION_REQUIREMENT_UNSATISFIED"
INVALID_REQUEST = "INVALID_REQUEST"
# Reused verbatim from the admission layer so the vocabulary stays one.
REGISTRATION_CONFLICT = lp.REGISTRATION_CONFLICT
SOURCE_URL_NOT_RECORDED = lp.SOURCE_URL_NOT_RECORDED
UNKNOWN_PAPER = lp.UNKNOWN_PAPER

FAILURE_KINDS = frozenset(
    {
        POLICY_BLOCKED, ACCESS_RESTRICTED, NOT_FOUND, RATE_LIMITED,
        NETWORK_FAILURE, RESPONSE_TOO_LARGE, INVALID_CONTENT_TYPE,
        INVALID_ARTIFACT, IDENTITY_MISMATCH, IDENTITY_INSUFFICIENT,
        VERSION_REQUIREMENT_UNSATISFIED, INVALID_REQUEST, REGISTRATION_CONFLICT,
        SOURCE_URL_NOT_RECORDED, UNKNOWN_PAPER,
    }
)

# -- deterministic network limits ------------------------------------------
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_ARTIFACT_BYTES = 40 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 1.0
DEFAULT_HOST_INTERVAL = 1.0
HOST_INTERVALS = {
    "arxiv.org": 3.0,
    "www.arxiv.org": 3.0,
    "export.arxiv.org": 3.0,
    "openreview.net": 0.5,
    "api.openreview.net": 0.5,
    "api2.openreview.net": 0.5,
}
_IDENTITY_TEXT_CHARS = 4000

# -- source policy classes --------------------------------------------------
RECOGNIZED_SCHOLARLY = "recognized_scholarly"
PUBLISHER = "publisher"
REPOSITORY = "repository"
RESTRICTED = "restricted"
UNKNOWN = "unknown"

# Recognized scholarly/public artifact sources: preprint servers, proceedings,
# open-access hosts, DOI resolvers and aggregator landing pages. Domain alone
# never establishes identity; it only decides whether retrieval *may* be tried.
_RECOGNIZED_DOMAINS = (
    "arxiv.org", "export.arxiv.org", "openreview.net", "aclanthology.org",
    "proceedings.mlr.press", "mlr.press", "proceedings.neurips.cc",
    "papers.nips.cc", "jmlr.org", "jmlr.csail.mit.edu", "ijcai.org",
    "aaai.org", "ojs.aaai.org", "openaccess.thecvf.com", "eprint.iacr.org",
    "doi.org", "dx.doi.org", "zenodo.org", "osf.io", "hal.science",
    "hal.archives-ouvertes.fr", "biorxiv.org", "medrxiv.org",
    "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "semanticscholar.org",
)
# Open-access publisher hosts (paywalled publishers are RESTRICTED below).
_PUBLISHER_DOMAINS = (
    "springeropen.com", "mdpi.com", "frontiersin.org", "plos.org", "peerj.com",
    "f1000research.com", "wellcomeopenresearch.org", "nature.com",
)
# Known paywalled / authenticated publishers and aggregators: never attempted.
_RESTRICTED_DOMAINS = (
    "sciencedirect.com", "dl.acm.org", "ieeexplore.ieee.org", "link.springer.com",
    "onlinelibrary.wiley.com", "tandfonline.com", "jstor.org", "cambridge.org",
    "academic.oup.com", "science.org", "cell.com", "journals.aps.org",
    "pubs.acs.org", "pubs.rsc.org", "iopscience.iop.org", "spie.org",
    "sagepub.com", "emerald.com", "degruyter.com", "bmj.com", "jamanetwork.com",
    "royalsocietypublishing.org",
)
# Institutional-repository patterns (suffix match on the registrable host).
_REPOSITORY_MARKERS = ("eprints.", "repository.", "dspace", "digital.library.")

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_ARXIV_TEXT_RE = re.compile(r"arxiv[:\s/]*?(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE
)
_HTML_TYPES = ("text/html", "application/xhtml+xml")


class AcquisitionError(Exception):
    """A deterministic acquisition failure a caller branches on by ``kind``."""

    def __init__(self, kind, message, detail=None, *, status=None):
        super(AcquisitionError, self).__init__(message)
        self.kind = kind
        self.status = status
        self.detail = dict(detail or {})

    def as_dict(self):
        return {"kind": self.kind, "message": str(self), "detail": dict(self.detail)}


@dataclass(frozen=True)
class Candidate:
    """A candidate artifact location plus the provider identity it came with.

    This mirrors the fields of ``literature_discovery.model``'s
    ``CandidateArtifactLocation`` and ``CandidatePaper`` that acquisition needs,
    without importing the discovery orchestration. The URL is unvalidated
    discovery evidence until the policy and content checks accept it.
    """

    url: str
    provider: Optional[str] = None
    artifact_type: str = "other"
    media_type: Optional[str] = None
    access: str = "unknown"
    version: Optional[str] = None
    doi: Optional[str] = None
    arxiv: Optional[str] = None
    openreview: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    authors: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Retrieved:
    """A bounded artifact response; the body is untrusted data until validated."""

    url: str
    body: bytes
    headers: Mapping
    content_type: Optional[str]


@dataclass(frozen=True)
class AcquisitionOutcome:
    """The structured outcome of one public-acquisition attempt."""

    status: str
    kind: Optional[str] = None
    paper_id: Optional[str] = None
    role: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    media_type: Optional[str] = None
    retrieved_url: Optional[str] = None
    recorded_source_url: Optional[str] = None
    discovery_provider: Optional[str] = None
    identity_evidence: Tuple[str, ...] = ()
    acquisition_id: Optional[str] = None
    message: Optional[str] = None
    detail: Mapping = field(default_factory=dict)

    @property
    def admitted(self):
        return self.status in (ACQUIRED, DUPLICATE)

    def as_dict(self):
        return {
            "status": self.status,
            "kind": self.kind,
            "admitted": self.admitted,
            "paper_id": self.paper_id,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "retrieved_url": self.retrieved_url,
            "recorded_source_url": self.recorded_source_url,
            "discovery_provider": self.discovery_provider,
            "identity_evidence": list(self.identity_evidence),
            "acquisition_id": self.acquisition_id,
            "message": self.message,
            "detail": dict(self.detail),
        }


# -- SSRF --------------------------------------------------------------------


def default_resolver(host):
    """Resolve a hostname to its address set (the seam tests inject)."""

    infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


_METADATA_ADDRESSES = frozenset(
    {"169.254.169.254", "100.100.100.200", "192.0.0.192", "fd00:ec2::254"}
)


def is_forbidden_address(address):
    """True when an address is loopback, private, link-local, metadata or reserved."""

    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return True
    if str(parsed) in _METADATA_ADDRESSES:
        return True
    return bool(
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


class AcquisitionPolicy:
    """Deterministic source policy: which locations retrieval may even attempt.

    The policy is *not* identity validation. A permitted host only means a
    retrieval may be tried; the retrieved bytes are admitted only if the identity
    checks pass. Restricted and unknown hosts fail closed.
    """

    def __init__(self, *, resolver=default_resolver):
        self._resolver = resolver

    @staticmethod
    def classify(host):
        host = (host or "").lower()
        if _matches(host, _RESTRICTED_DOMAINS):
            return RESTRICTED
        if _matches(host, _RECOGNIZED_DOMAINS):
            return RECOGNIZED_SCHOLARLY
        if _matches(host, _PUBLISHER_DOMAINS):
            return PUBLISHER
        if any(marker in host for marker in _REPOSITORY_MARKERS):
            return REPOSITORY
        return UNKNOWN

    def check(self, url, *, phase="request"):
        """Return the host class, or raise :class:`AcquisitionError` to refuse."""

        if not isinstance(url, str) or not url.strip():
            raise AcquisitionError(INVALID_REQUEST, "a candidate URL is required")
        parts = urlsplit(url.strip())
        if parts.scheme.lower() != "https":
            raise AcquisitionError(
                POLICY_BLOCKED,
                "only https locations may be retrieved (got {!r})".format(parts.scheme),
                {"url": _redact(url), "phase": phase},
            )
        if parts.username or parts.password:
            raise AcquisitionError(
                POLICY_BLOCKED,
                "a URL with embedded credentials is refused",
                {"url": _redact(url), "phase": phase},
            )
        host = (parts.hostname or "").lower()
        if not host:
            raise AcquisitionError(
                POLICY_BLOCKED, "the URL has no host", {"url": _redact(url)}
            )
        if parts.port not in (None, 443):
            raise AcquisitionError(
                POLICY_BLOCKED,
                "only the default https port is allowed (got {!r})".format(parts.port),
                {"url": _redact(url), "phase": phase},
            )
        host_class = self.classify(host)
        if host_class == RESTRICTED:
            raise AcquisitionError(
                ACCESS_RESTRICTED,
                "{!r} is a restricted/authenticated source; acquisition is "
                "refused".format(host),
                {"url": _redact(url), "host": host, "source_class": host_class},
            )
        if host_class == UNKNOWN:
            raise AcquisitionError(
                POLICY_BLOCKED,
                "{!r} is not a recognized scholarly, publisher or repository "
                "source".format(host),
                {"url": _redact(url), "host": host, "source_class": host_class},
            )
        self._check_addresses(host, url, phase)
        return host_class

    def _check_addresses(self, host, url, phase):
        addresses = self._addresses(host, url)
        if not addresses:
            raise AcquisitionError(
                NETWORK_FAILURE,
                "{!r} did not resolve to any address".format(host),
                {"url": _redact(url), "host": host, "phase": phase},
            )
        forbidden = [address for address in addresses if is_forbidden_address(address)]
        if forbidden:
            raise AcquisitionError(
                POLICY_BLOCKED,
                "{!r} resolves into forbidden address space".format(host),
                {
                    "url": _redact(url), "host": host, "phase": phase,
                    "addresses": sorted(forbidden),
                },
            )

    def _addresses(self, host, url):
        try:
            ipaddress.ip_address(host)
            return [host]
        except ValueError:
            pass
        try:
            return list(self._resolver(host))
        except AcquisitionError:
            raise
        except Exception as exc:  # resolution failure is a structured outcome
            raise AcquisitionError(
                NETWORK_FAILURE,
                "{!r} could not be resolved".format(host),
                {"url": _redact(url), "host": host, "reason": type(exc).__name__},
            ) from exc

    def guard_redirect(self, url):
        """A redirect target is subject to the full policy again."""

        return self.check(url, phase="redirect")


def _matches(host, domains):
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _redact(url):
    """Keep diagnostics useful without echoing query values as if they were trusted."""

    parsed = urlsplit(url)
    return "{}://{}{}".format(parsed.scheme, parsed.netloc, parsed.path)


# -- retrieval ---------------------------------------------------------------


class HostThrottle:
    """A per-host minimum interval, reusing the discovery rate limiter."""

    def __init__(self, *, intervals=None, default=DEFAULT_HOST_INTERVAL,
                 clock=time.monotonic, sleep=time.sleep):
        self._intervals = dict(HOST_INTERVALS if intervals is None else intervals)
        self._default = default
        self._clock = clock
        self._sleep = sleep
        self._limiters = {}

    def acquire(self, provider):
        interval = self._intervals.get(provider, self._default)
        limiter = self._limiters.get(provider)
        if limiter is None:
            limiter = http_layer.RateLimiter(
                intervals={provider: interval}, clock=self._clock, sleep=self._sleep
            )
            self._limiters[provider] = limiter
        limiter.acquire(provider)


class ArtifactRetriever:
    """Bounded, redirect-guarded artifact HTTP over INC-012's fetcher."""

    def __init__(self, *, transport=None, policy=None, timeout=DEFAULT_TIMEOUT,
                 max_bytes=DEFAULT_MAX_ARTIFACT_BYTES,
                 max_redirects=DEFAULT_MAX_REDIRECTS, retries=DEFAULT_RETRIES,
                 backoff=DEFAULT_BACKOFF, user_agent=None, throttle=None,
                 sleep=time.sleep):
        self.policy = policy if policy is not None else AcquisitionPolicy()
        self.max_bytes = max_bytes
        self._fetcher = http_layer.HttpFetcher(
            transport=transport,
            timeout=timeout,
            max_response_bytes=max_bytes,
            max_redirects=max_redirects,
            retries=retries,
            backoff=backoff,
            user_agent=user_agent or http_layer.default_user_agent(),
            rate_limiter=throttle if throttle is not None else HostThrottle(sleep=sleep),
            cache=None,
            sleep=sleep,
        )

    def retrieve(self, candidate):
        url = candidate.url
        host = (urlsplit(url).hostname or "").lower()
        self.policy.check(url)
        try:
            response = self._fetcher.fetch_response(
                host or "artifact",
                url,
                accept="application/pdf",
                max_bytes=self.max_bytes,
                redirect_guard=self.policy.guard_redirect,
            )
        except AcquisitionError:
            raise
        except http_layer.HttpError as exc:
            raise _map_http_error(exc) from exc
        headers = response.headers or {}
        return Retrieved(
            url=response.url or url,
            body=response.body,
            headers=headers,
            content_type=_content_type(headers),
        )


def _map_http_error(exc):
    kind = exc.kind
    detail = dict(exc.detail or {})
    if exc.status is not None:
        detail.setdefault("status", exc.status)
    if kind == discovery_model.RATE_LIMITED:
        return AcquisitionError(RATE_LIMITED, str(exc), detail)
    if kind == discovery_model.NOT_FOUND:
        return AcquisitionError(NOT_FOUND, str(exc), detail)
    if kind == discovery_model.PROVIDER_AUTH_REQUIRED:
        return AcquisitionError(ACCESS_RESTRICTED, str(exc), detail)
    if kind == http_layer.RESPONSE_TOO_LARGE:
        return AcquisitionError(RESPONSE_TOO_LARGE, str(exc), detail)
    if kind == discovery_model.PROVIDER_UNAVAILABLE:
        return AcquisitionError(NETWORK_FAILURE, str(exc), detail)
    return AcquisitionError(NETWORK_FAILURE, str(exc), detail)


def _content_type(headers):
    value = (headers or {}).get("content-type") or ""
    return value.split(";", 1)[0].strip().lower() or None


# -- content validation ------------------------------------------------------


def validate_artifact(retrieved, path, *, extractor=None):
    """Validate the retrieved bytes as a readable PDF body, or raise.

    A 200 response is not sufficient. The media type, the byte length and the
    ``%PDF-`` magic are checked, then (when a bounded text extractor is
    available) the PDF is opened well enough to yield its first page. ``path`` is
    the already-written temporary copy of the bytes.
    """

    body = retrieved.body
    content_type = retrieved.content_type
    if not body:
        raise AcquisitionError(
            INVALID_ARTIFACT, "the response body is empty",
            {"url": _redact(retrieved.url), "content_type": content_type},
        )
    if not body.startswith(b"%PDF-"):
        looks_html = (
            content_type in _HTML_TYPES
            or content_type is not None and content_type.startswith("text/")
            or body.lstrip()[:1] == b"<"
        )
        raise AcquisitionError(
            INVALID_CONTENT_TYPE if looks_html else INVALID_ARTIFACT,
            "the response is not a PDF (missing a %PDF- header)",
            {
                "url": _redact(retrieved.url),
                "content_type": content_type,
                "first_bytes": body[:16].decode("latin-1", "replace"),
            },
        )
    extract = extractor if extractor is not None else lpt.extract_document
    try:
        document = extract(path)
    except lpt.PrimaryTextError as exc:
        if exc.kind == lpt.EXTRACTOR_UNAVAILABLE:
            return None  # no extractor: magic + size stand as the readable check
        raise AcquisitionError(
            INVALID_ARTIFACT,
            "the PDF could not be opened as a readable document",
            {"url": _redact(retrieved.url), "reason": exc.kind},
        ) from exc
    if getattr(document, "page_count", 0) < 1:
        raise AcquisitionError(
            INVALID_ARTIFACT,
            "the PDF yielded no readable pages",
            {"url": _redact(retrieved.url)},
        )
    return document


def _write_temp(body):
    handle = tempfile.NamedTemporaryFile(
        prefix="wavcse-acquire-", suffix=".pdf", delete=False
    )
    try:
        handle.write(body)
        return Path(handle.name)
    finally:
        handle.close()


def _discard_temp(path):
    try:
        Path(path).unlink()
    except OSError:
        pass


# -- identity validation -----------------------------------------------------


def _norm_id(kind, value):
    if kind == "doi":
        return literature_catalog._normalize_doi(value)
    if kind == "arxiv":
        return literature_catalog._normalize_arxiv(value)
    return str(value).strip().casefold()


def _paper_ids(paper, kind):
    values = (paper.external_ids or {}).get(kind, ())
    return {_norm_id(kind, value) for value in values}


def _doi_from_url(url):
    match = _DOI_RE.search(url or "")
    return match.group(0) if match else None


def _arxiv_from_url(url):
    if "arxiv" not in (url or "").lower():
        return None
    match = _ARXIV_URL_RE.search(url)
    return match.group(1) if match else None


def _doi_from_text(text):
    match = _DOI_RE.search(text or "")
    return match.group(0).rstrip(".") if match else None


def _arxiv_from_text(text):
    match = _ARXIV_TEXT_RE.search(text or "")
    return match.group(1) if match else None


def _normalize_title(value):
    return literature_catalog._normalize_title(value or "")


def _surnames(authors):
    result = set()
    for author in authors or ():
        parts = [part for part in re.split(r"\s+", str(author).strip()) if part]
        if parts:
            result.add(parts[-1].casefold())
    return result


def _title_evidence(paper, candidate, pdf_text):
    """Whether the paper's title is established, with corroborating agreement."""

    title = _normalize_title(paper.title)
    if not title:
        return False
    candidate_title = _normalize_title(candidate.title)
    if candidate_title and candidate_title == title:
        corroborated = (
            (candidate.year is not None and candidate.year == paper.year)
            or bool(_surnames(candidate.authors) & _surnames(paper.authors))
            or len(title) >= 25
        )
        if corroborated:
            return True
    if pdf_text and len(title) >= 12 and title in _normalize_title(pdf_text):
        return True
    return False


class Identities:
    """Validate that retrieved bytes belong to the requested Paper.

    Evidence hierarchy, strongest first: a candidate DOI / arXiv id, a DOI / arXiv
    id extracted from the PDF, then the paper's exact title corroborated by year or
    author. A strong identifier that conflicts with the Paper — or with a
    different catalog Paper — fails closed. Title similarity is never fuzzy:
    only exact normalized equality counts, and it is the weakest admissible
    evidence.
    """

    def __init__(self, catalog):
        self._catalog = catalog

    def validate(self, paper, candidate, pdf_text=None):
        evidence = []
        self._check_identifier(paper, "doi", candidate.doi, evidence)
        self._check_identifier(paper, "arxiv", candidate.arxiv, evidence)
        self._check_identifier(paper, "openreview", candidate.openreview, evidence)
        if pdf_text:
            self._check_identifier(paper, "doi", _doi_from_text(pdf_text), evidence,
                                   source="pdf")
            self._check_identifier(paper, "arxiv", _arxiv_from_text(pdf_text), evidence,
                                   source="pdf")
        if _title_evidence(paper, candidate, pdf_text):
            evidence.append("title")
        if not evidence:
            raise AcquisitionError(
                IDENTITY_INSUFFICIENT,
                "no DOI, arXiv id or exact title ties the retrieved artifact to "
                "{!r}".format(paper.paper_id),
                {
                    "paper_id": paper.paper_id,
                    "candidate_doi": candidate.doi,
                    "candidate_arxiv": candidate.arxiv,
                    "candidate_title": candidate.title,
                },
            )
        return tuple(sorted(set(evidence)))

    def _check_identifier(self, paper, kind, value, evidence, source="candidate"):
        if not value:
            return
        norm = _norm_id(kind, value)
        recorded = _paper_ids(paper, kind)
        if norm in recorded:
            evidence.append("{}:{}".format(source, kind))
            return
        other = self._lookup(value)
        if other is not None and other != paper.paper_id:
            raise AcquisitionError(
                IDENTITY_MISMATCH,
                "{} {!r} identifies {!r}, not the requested {!r}".format(
                    kind, value, other, paper.paper_id
                ),
                {
                    "requested_paper_id": paper.paper_id,
                    "field": kind,
                    "value": value,
                    "resolved_paper_id": other,
                },
            )
        if recorded:
            # The Paper records a *different* value of this strong identifier.
            raise AcquisitionError(
                IDENTITY_MISMATCH,
                "{} {!r} contradicts the identifier recorded for {!r}".format(
                    kind, value, paper.paper_id
                ),
                {
                    "requested_paper_id": paper.paper_id,
                    "field": kind,
                    "value": value,
                    "recorded": sorted(recorded),
                },
            )
        # The Paper records no identifier of this kind: inconclusive, not a match.

    def _lookup(self, identity):
        try:
            return self._catalog.lookup(identity).paper_id
        except literature_catalog.CatalogError:
            return None


# -- source-URL correspondence -----------------------------------------------


def recorded_source_url(paper, candidate, retrieved):
    """Map the candidate location onto one of the Paper's recorded source URLs.

    The manifest's ``source_url`` is a catalog URL by design; a public
    acquisition must therefore associate the retrieved bytes with the recorded
    representation of the *same source* — exact URL, a shared DOI, or a shared
    arXiv base id, with same-host agreement as the weakest admissible link. When
    no recorded URL corresponds, admission would create an unrecorded provenance,
    so the attempt fails closed.
    """

    recorded = list(paper.source_urls)
    if not recorded:
        return None
    candidate_ids = {
        "doi": {_norm_id("doi", candidate.doi)} if candidate.doi else set(),
        "arxiv": {_norm_id("arxiv", candidate.arxiv)} if candidate.arxiv else set(),
    }
    candidate_urls = [url for url in (retrieved.url, candidate.url) if url]
    for url in candidate_urls:
        doi = _doi_from_url(url)
        if doi:
            candidate_ids["doi"].add(_norm_id("doi", doi))
        arxiv = _arxiv_from_url(url)
        if arxiv:
            candidate_ids["arxiv"].add(_norm_id("arxiv", arxiv))

    best_url = None
    best_score = 0
    for url in sorted(recorded):
        score = _source_score(url, candidate_urls, candidate_ids)
        if score > best_score:
            best_score, best_url = score, url
    return best_url if best_score >= 1 else None


def _source_score(recorded_url, candidate_urls, candidate_ids):
    score = 0
    recorded_norm = literature_catalog._normalize_url(recorded_url)
    recorded_host = (urlsplit(recorded_url).hostname or "").lower()
    recorded_doi = _norm_id("doi", _doi_from_url(recorded_url)) if _doi_from_url(
        recorded_url
    ) else None
    recorded_arxiv = _arxiv_from_url(recorded_url)
    for url in candidate_urls:
        if literature_catalog._normalize_url(url) == recorded_norm:
            return 2
        host = (urlsplit(url).hostname or "").lower()
        if recorded_host and host == recorded_host:
            score = max(score, 1)
        doi = _doi_from_url(url)
        if doi and recorded_doi and _norm_id("doi", doi) == recorded_doi:
            score = 2
        arxiv = _arxiv_from_url(url)
        if arxiv and recorded_arxiv and _norm_id("arxiv", arxiv) == _norm_id(
            "arxiv", recorded_arxiv
        ):
            score = 2
    if recorded_doi and recorded_doi in candidate_ids["doi"]:
        score = 2
    if recorded_arxiv and _norm_id("arxiv", recorded_arxiv) in candidate_ids["arxiv"]:
        score = 2
    return score


# -- orchestration -----------------------------------------------------------


class PublicAcquirer:
    """Turn one candidate artifact location into a validated, admitted artifact."""

    def __init__(self, *, ingest=None, catalog=None, retriever=None, policy=None,
                 extractor=None, clock=None):
        self._ingest = ingest if ingest is not None else li.LiteratureIngest()
        catalog_path = Path(self._ingest.catalog_path)
        self._catalog = catalog if catalog is not None else literature_catalog.load_catalog(
            catalog_path,
            repo_root=Path(self._ingest._primary.policy.repo_root),
            validate_references=False,
        )
        self._retriever = retriever if retriever is not None else ArtifactRetriever(
            policy=policy
        )
        self._identities = Identities(self._catalog)
        self._extractor = extractor
        self._clock = clock if clock is not None else _utc_now

    def acquire(self, paper_id, candidate, *, role=None):
        """Retrieve, validate and admit one candidate; never raises for a domain
        failure. Returns an :class:`AcquisitionOutcome`.
        """

        if not isinstance(paper_id, str) or not paper_id.strip():
            return self._fail(
                INVALID_REQUEST, "paper_id must be a non-empty canonical paper_id"
            )
        if not isinstance(candidate, Candidate) or not candidate.url:
            return self._fail(
                INVALID_REQUEST, "a Candidate with a URL is required"
            )
        try:
            paper = self._catalog.get(paper_id.strip())
        except literature_catalog.CatalogError:
            return self._fail(
                UNKNOWN_PAPER, "unknown paper_id {!r}".format(paper_id),
                detail={"paper_id": paper_id},
            )
        try:
            retrieved = self._retriever.retrieve(candidate)
        except AcquisitionError as exc:
            return self._fail(exc.kind, str(exc), detail=exc.detail, paper_id=paper.paper_id)
        try:
            path = _write_temp(retrieved.body)
        except OSError as exc:
            return self._fail(
                NETWORK_FAILURE,
                "the retrieved artifact could not be staged locally",
                detail={"reason": type(exc).__name__},
                paper_id=paper.paper_id, retrieved_url=retrieved.url,
            )
        try:
            try:
                document = validate_artifact(
                    retrieved, path, extractor=self._extractor
                )
            except AcquisitionError as exc:
                return self._fail(
                    exc.kind, str(exc), detail=exc.detail, paper_id=paper.paper_id,
                    retrieved_url=retrieved.url,
                )
            pdf_text = _bounded_text(document)
            try:
                evidence = self._identities.validate(paper, candidate, pdf_text)
            except AcquisitionError as exc:
                return self._fail(
                    exc.kind, str(exc), detail=exc.detail, paper_id=paper.paper_id,
                    retrieved_url=retrieved.url,
                )
            source_url = recorded_source_url(paper, candidate, retrieved)
            if source_url is None:
                return self._fail(
                    SOURCE_URL_NOT_RECORDED,
                    "no source URL recorded for {!r} corresponds to the retrieved "
                    "location".format(paper.paper_id),
                    detail={
                        "paper_id": paper.paper_id,
                        "retrieved_url": _redact(retrieved.url),
                        "recorded": list(paper.source_urls),
                    },
                    paper_id=paper.paper_id, retrieved_url=retrieved.url,
                )
            acquisition = {
                "retrieved_url": retrieved.url,
                "discovery_provider": candidate.provider,
                "retrieved_at": self._clock(),
                "candidate_identifiers": _candidate_identifiers(candidate),
                "identity_evidence": list(evidence),
            }
            result = self._ingest.ingest(
                path,
                paper_id=paper.paper_id,
                source_url=source_url,
                role=role,
                source_label=_host_label(retrieved.url),
                provenance=li.PUBLIC_ACQUIRED,
                acquisition=acquisition,
            )
        finally:
            _discard_temp(path)
        return self._from_ingest(
            result, candidate, retrieved, source_url, evidence
        )

    # -- helpers ----------------------------------------------------------

    def _from_ingest(self, result, candidate, retrieved, source_url, evidence):
        if result.status == li.ADMITTED:
            status = ACQUIRED
        elif result.status == li.DUPLICATE:
            status = DUPLICATE
        else:
            status = FAILED
        return AcquisitionOutcome(
            status=status,
            kind=None if status != FAILED else _map_ingest_kind(result.kind),
            paper_id=result.paper_id or None,
            role=result.retained_role or result.role or None,
            sha256=result.sha256,
            size_bytes=result.size_bytes,
            media_type=result.media_type,
            retrieved_url=retrieved.url,
            recorded_source_url=source_url,
            discovery_provider=candidate.provider,
            identity_evidence=evidence,
            acquisition_id=result.acquisition_id,
            message=result.message,
            detail=dict(result.detail or {}),
        )

    def _fail(self, kind, message, *, detail=None, paper_id=None, retrieved_url=None):
        return AcquisitionOutcome(
            status=FAILED,
            kind=kind,
            paper_id=paper_id,
            retrieved_url=retrieved_url,
            message=message,
            detail=dict(detail or {}),
        )


def _map_ingest_kind(kind):
    if kind in (li.ROLE_REQUIRED, lp.UNKNOWN_ROLE):
        return VERSION_REQUIREMENT_UNSATISFIED
    if kind is None:
        return INVALID_REQUEST
    return kind


def _candidate_identifiers(candidate):
    identifiers = {}
    if candidate.doi:
        identifiers["doi"] = candidate.doi
    if candidate.arxiv:
        identifiers["arxiv"] = candidate.arxiv
    if candidate.openreview:
        identifiers["openreview"] = candidate.openreview
    return identifiers


def _bounded_text(document):
    if document is None:
        return None
    pages = getattr(document, "pages", ()) or ()
    text = "\n".join(pages[:2])
    return text[:_IDENTITY_TEXT_CHARS]


def _host_label(url):
    return (urlsplit(url).hostname or "").lower() or "public"


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# -- CLI ---------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("validate", help="validate the acquisition ledger")

    policy = commands.add_parser(
        "policy", help="dry-run the source policy for a candidate URL (no fetch)"
    )
    policy.add_argument("--url", required=True)

    acquire = commands.add_parser(
        "acquire", help="retrieve, validate and admit one candidate artifact"
    )
    acquire.add_argument("--paper-id", required=True)
    acquire.add_argument("--url", required=True)
    acquire.add_argument("--provider", default=None)
    acquire.add_argument("--artifact-type", default="other")
    acquire.add_argument("--media-type", default=None)
    acquire.add_argument("--access", default="unknown")
    acquire.add_argument("--version", default=None)
    acquire.add_argument("--doi", default=None)
    acquire.add_argument("--arxiv", default=None)
    acquire.add_argument("--openreview", default=None)
    acquire.add_argument("--title", default=None)
    acquire.add_argument("--year", type=int, default=None)
    acquire.add_argument("--author", action="append", default=[])
    acquire.add_argument("--role", default=None, choices=sorted(li._KNOWN_ROLES))
    return parser


def _emit(payload, ok):
    stream = sys.stdout if ok else sys.stderr
    stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return 0 if ok else 1


def main(argv=None):
    args = _build_parser().parse_args(argv)
    command = args.command or "validate"
    try:
        if command == "validate":
            ingestor = li.LiteratureIngest()
            ingestor.validate()
            return _emit(
                {"kind": "VALIDATED", "attempts": len(ingestor.attempts())}, True
            )
        if command == "policy":
            policy = AcquisitionPolicy()
            try:
                host_class = policy.check(args.url)
            except AcquisitionError as exc:
                return _emit(
                    {
                        "kind": "POLICY",
                        "url": _redact(args.url),
                        "allowed": False,
                        "blocked_kind": exc.kind,
                        "reason": str(exc),
                    },
                    True,
                )
            return _emit(
                {
                    "kind": "POLICY",
                    "url": _redact(args.url),
                    "allowed": True,
                    "source_class": host_class,
                },
                True,
            )
        candidate = Candidate(
            url=args.url, provider=args.provider, artifact_type=args.artifact_type,
            media_type=args.media_type, access=args.access, version=args.version,
            doi=args.doi, arxiv=args.arxiv, openreview=args.openreview,
            title=args.title, year=args.year, authors=tuple(args.author),
        )
        outcome = PublicAcquirer().acquire(args.paper_id, candidate, role=args.role)
        return _emit(outcome.as_dict(), outcome.admitted)
    except AcquisitionError as exc:
        return _emit(exc.as_dict(), False)
    except (li.IngestLedgerError, lp.PrimaryManifestError,
            literature_catalog.CatalogError) as exc:
        return _emit(
            {"kind": "STATE_INVALID", "message": str(exc), "detail": {}}, False
        )


if __name__ == "__main__":
    sys.exit(main())
