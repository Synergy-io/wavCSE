"""Provider adapters: one normalized discovery contract, provider shapes hidden.

Four providers are implemented, each for a distinct scholarly function:

* **Crossref** — DOI registration metadata and reference lists, no key required.
* **arXiv** — exact arXiv identity and full-text metadata search, no key required.
* **Semantic Scholar** — cross-source search and the reference/citation graph;
  works unauthenticated at a low rate and accepts an optional API key.
* **OpenReview** — venue-structured ML metadata (ICLR/NeurIPS and friends).

They are not the only sources that exist, and adding one is a small, isolated
change behind this contract. PMLR and ACL Anthology are deliberately *not*
adapters: neither exposes a stable metadata query API (PMLR has none; ACL
Anthology has only a bulk XML dump), so their pages remain candidate artifact
*locations* reported by the providers above and are handled by the acquisition
source policy in INC-013, not by a discovery adapter here.

Every adapter converts a provider response into `model.CandidatePaper` and
`model.CandidateArtifactLocation` records and converts every failure into a
structured `model.DiscoveryFailure`. No provider field name escapes this module.
Provider-returned strings are inert data: nothing here evaluates them.
"""

import datetime
import json
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple

from . import http as http_layer
from . import model


class UnsupportedOperation(Exception):
    """The provider does not offer this capability; callers must ask one that does."""


@dataclass
class ProviderResult:
    candidates: Tuple = ()
    failure: Optional[model.DiscoveryFailure] = None
    truncated: bool = False


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _plain(value):
    """Strip provider markup (Crossref abstracts are JATS) — normalization, not interpretation."""

    import re

    if value is None:
        return None
    return re.sub(r"<[^>]+>", " ", str(value))


def _clean(value):
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _year_from_date_parts(parts):
    try:
        return int(parts[0][0])
    except (TypeError, IndexError, ValueError):
        return None


def _candidate(provider, record_id, **fields):
    return model.CandidatePaper(provider=provider, provider_record_id=str(record_id), **fields)


class ProviderAdapter:
    name = ""
    capabilities = frozenset()
    api_key_env = None

    def __init__(self, fetcher, *, api_key=None, contact=None):
        self.fetcher = fetcher
        self.api_key = api_key
        self.contact = contact

    def availability(self):
        """Return ``(available, reason)``; a missing optional key degrades, never crashes."""

        return True, None

    def describe(self):
        available, reason = self.availability()
        return {
            "provider": self.name,
            "available": available,
            "reason": reason,
            "capabilities": sorted(self.capabilities),
            "credentialed": bool(self.api_key),
        }

    # Each method either returns a ProviderResult or raises UnsupportedOperation.
    def resolve(self, kind, value) -> ProviderResult:
        raise UnsupportedOperation(self.name, kind)

    def search(self, query, *, limit, year=None) -> ProviderResult:
        raise UnsupportedOperation(self.name, "search")

    def references(self, target, *, limit) -> ProviderResult:
        raise UnsupportedOperation(self.name, "references")

    def citations(self, target, *, limit) -> ProviderResult:
        raise UnsupportedOperation(self.name, "citations")

    def _guarded(self, call):
        """Convert a networking or parsing failure into a structured result."""

        try:
            return call()
        except http_layer.HttpError as exc:
            return ProviderResult(failure=exc.failure())
        except (ValueError, KeyError, TypeError, IndexError, ET.ParseError) as exc:
            return ProviderResult(
                failure=model.DiscoveryFailure(
                    kind=model.MALFORMED_PROVIDER_RESPONSE,
                    provider=self.name,
                    message="{} returned an unexpected structure".format(self.name),
                    detail={"reason": str(exc)},
                )
            )


class CrossrefProvider(ProviderAdapter):
    name = "crossref"
    capabilities = frozenset({"doi", "search", "references"})
    BASE = "https://api.crossref.org"
    _SELECT = "DOI,title,author,issued,published,container-title,type,URL,link,abstract,reference"

    def resolve(self, kind, value):
        if kind != "doi":
            raise UnsupportedOperation(self.name, kind)

        def call():
            payload = self.fetcher.get_json(
                self.name,
                "{}/works/{}".format(self.BASE, urllib.parse.quote(value, safe="")),
            )
            message = payload["message"]
            actual = (message.get("DOI") or "").strip().lower()
            if actual and actual != value.strip().lower():
                return ProviderResult(
                    failure=model.DiscoveryFailure(
                        kind=model.IDENTITY_CONFLICT,
                        provider=self.name,
                        message="the DOI resolved to a different record",
                        detail={"requested": value, "returned": message.get("DOI")},
                    )
                )
            return ProviderResult(candidates=(self._candidate_from_item(message),))

        return self._guarded(call)

    def search(self, query, *, limit, year=None):
        def call():
            params = {"query.bibliographic": query, "rows": limit, "select": self._SELECT}
            payload = self.fetcher.get_json(self.name, "{}/works".format(self.BASE), params=params)
            message = payload["message"]
            items = message.get("items") or []
            total = message.get("total-results")
            candidates = tuple(self._candidate_from_item(item) for item in items)
            return ProviderResult(
                candidates=candidates,
                truncated=bool(total is not None and total > len(candidates)),
            )

        return self._guarded(call)

    def references(self, target, *, limit):
        doi = target.get("doi")
        if not doi:
            raise UnsupportedOperation(self.name, "references:doi")
        return self._guarded(lambda: self._references_for(doi, limit))

    def _references_for(self, doi, limit):
        payload = self.fetcher.get_json(
            self.name, "{}/works/{}".format(self.BASE, urllib.parse.quote(doi, safe=""))
        )
        references = payload["message"].get("reference") or []
        candidates = tuple(
            self._candidate_from_reference(reference) for reference in references[:limit]
        )
        return ProviderResult(
            candidates=candidates, truncated=len(references) > len(candidates)
        )

    def _candidate_from_item(self, item):
        doi = _clean(item.get("DOI"))
        title = _clean(_first(item.get("title")))
        authors = tuple(
            _clean(" ".join(part for part in (author.get("given"), author.get("family")) if part))
            or _clean(author.get("name"))
            for author in (item.get("author") or ())
        )
        authors = tuple(author for author in authors if author)
        year = _year_from_date_parts((item.get("issued") or {}).get("date-parts"))
        venue = _clean(_first(item.get("container-title"))) or _clean(item.get("publisher"))
        locations = []
        for link in item.get("link") or ():
            url = _clean(link.get("URL"))
            if not url:
                continue
            locations.append(
                model.CandidateArtifactLocation(
                    url=url,
                    provider=self.name,
                    artifact_type=_artifact_type(link.get("content-type"), url),
                    media_type=_clean(link.get("content-type")),
                    access=model.ACCESS_UNKNOWN,
                    evidence="crossref.link",
                    version=_clean(link.get("content-version")),
                )
            )
        source_url = _clean(item.get("URL")) or (
            "https://doi.org/{}".format(doi) if doi else None
        )
        if source_url and all(location.url != source_url for location in locations):
            locations.append(
                model.CandidateArtifactLocation(
                    url=source_url,
                    provider=self.name,
                    artifact_type=_artifact_type(None, source_url),
                    access=model.ACCESS_UNKNOWN,
                    evidence="crossref.URL",
                )
            )
        return _candidate(
            self.name,
            doi or _clean(item.get("URL")) or "unknown",
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            other_ids=model.other_ids({"type": item.get("type")}),
            abstract=model.bounded_text(_plain(item.get("abstract"))),
            url=source_url,
            artifact_locations=tuple(locations),
            provenance={"endpoint": "works", "title": title},
        )

    def _candidate_from_reference(self, reference):
        doi = _clean(reference.get("DOI"))
        title = _clean(reference.get("article-title"))
        year = None
        if reference.get("year"):
            try:
                year = int(reference["year"])
            except (TypeError, ValueError):
                year = None
        identifier = doi or _clean(reference.get("unstructured")) or title or "unknown"
        return _candidate(
            self.name,
            identifier,
            title=title,
            year=year,
            venue=_clean(reference.get("journal-title")),
            doi=doi,
            url="https://doi.org/{}".format(doi) if doi else None,
            other_ids=model.other_ids({"unstructured": _clean(reference.get("unstructured"))})
            if not doi
            else (),
            provenance={"endpoint": "reference", "key": _clean(reference.get("key"))},
        )


class ArxivProvider(ProviderAdapter):
    name = "arxiv"
    capabilities = frozenset({"arxiv", "search"})
    BASE = "https://export.arxiv.org/api/query"
    _ATOM = "{http://www.w3.org/2005/Atom}"
    _ARXIV = "{http://arxiv.org/schemas/atom}"

    def resolve(self, kind, value):
        if kind != "arxiv":
            raise UnsupportedOperation(self.name, kind)

        def call():
            text = self.fetcher.get_text(
                self.name, self.BASE,
                params={"id_list": value, "max_results": 1},
                accept="application/atom+xml",
            )
            entries = _atom_entries(text)
            if not entries:
                return ProviderResult(
                    failure=model.DiscoveryFailure(
                        kind=model.NOT_FOUND, provider=self.name,
                        message="arXiv has no record for this identifier",
                        detail={"arxiv": value},
                    )
                )
            candidate = self._candidate_from_entry(entries[0])
            if candidate.arxiv_id and _normalize_arxiv(candidate.arxiv_id) != _normalize_arxiv(value):
                return ProviderResult(
                    failure=model.DiscoveryFailure(
                        kind=model.IDENTITY_CONFLICT, provider=self.name,
                        message="the arXiv identifier resolved to a different record",
                        detail={"requested": value, "returned": candidate.arxiv_id},
                    )
                )
            return ProviderResult(candidates=(candidate,))

        return self._guarded(call)

    def search(self, query, *, limit, year=None):
        def call():
            text = self.fetcher.get_text(
                self.name, self.BASE,
                params={"search_query": "all:{}".format(query), "max_results": limit},
                accept="application/atom+xml",
            )
            entries = _atom_entries(text)
            total = _atom_total_results(text)
            candidates = tuple(self._candidate_from_entry(entry) for entry in entries)
            return ProviderResult(
                candidates=candidates,
                truncated=bool(total is not None and total > len(candidates)),
            )

        return self._guarded(call)

    def _candidate_from_entry(self, entry):
        raw_id = _text(entry, self._ATOM + "id")
        arxiv_id = _arxiv_from_url(raw_id)
        title = _clean(_text(entry, self._ATOM + "title"))
        abstract = model.bounded_text(_clean(_text(entry, self._ATOM + "summary")))
        published = _clean(_text(entry, self._ATOM + "published"))
        year = None
        if published and len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])
        authors = tuple(
            _clean(_text(author, self._ATOM + "name"))
            for author in entry.findall(self._ATOM + "author")
        )
        authors = tuple(author for author in authors if author)
        doi = _clean(_text(entry, self._ARXIV + "doi"))
        journal_ref = _clean(_text(entry, self._ARXIV + "journal_ref"))
        locations = []
        landing = None
        for link in entry.findall(self._ATOM + "link"):
            href = _clean(link.get("href"))
            if not href:
                continue
            link_type = (link.get("type") or "").lower()
            rel = (link.get("rel") or "").lower()
            if link_type == "application/pdf" or href.lower().endswith(".pdf"):
                locations.append(
                    model.CandidateArtifactLocation(
                        url=href, provider=self.name, artifact_type=model.ARTIFACT_PDF,
                        media_type="application/pdf", access=model.ACCESS_OPEN,
                        evidence="arxiv.eprint", version=_clean(link.get("title")),
                    )
                )
            elif rel == "alternate":
                landing = href
        if landing:
            locations.append(
                model.CandidateArtifactLocation(
                    url=landing, provider=self.name, artifact_type=model.ARTIFACT_LANDING,
                    access=model.ACCESS_OPEN, evidence="arxiv.abs",
                )
            )
        return _candidate(
            self.name,
            arxiv_id or raw_id or "unknown",
            title=title,
            authors=authors,
            year=year,
            venue=journal_ref or "arXiv",
            doi=doi,
            arxiv_id=arxiv_id,
            abstract=abstract,
            url=landing,
            artifact_locations=tuple(locations),
            provenance={"endpoint": "atom", "updated": _clean(_text(entry, self._ATOM + "updated"))},
        )


class SemanticScholarProvider(ProviderAdapter):
    name = "semanticscholar"
    capabilities = frozenset({"doi", "arxiv", "search", "references", "citations"})
    BASE = "https://api.semanticscholar.org/graph/v1"
    api_key_env = "WAVCSE_S2_API_KEY"
    _FIELDS = "paperId,title,authors,year,venue,externalIds,abstract,url,openAccessPdf,publicationTypes,journal"

    def _get(self, url, params=None):
        # The optional key rides in a request header, never in a URL; the fetcher
        # records no header on a failure and caches only the response body.
        headers = {"x-api-key": self.api_key} if self.api_key else None
        body = self.fetcher.fetch(
            self.name, url, params=params, accept="application/json", headers=headers
        )
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise http_layer.HttpError(
                model.MALFORMED_PROVIDER_RESPONSE,
                "{} returned a body that is not JSON".format(self.name),
                provider=self.name, detail={"reason": str(exc)},
            ) from exc

    def resolve(self, kind, value):
        if kind == "doi":
            prefix = "DOI:"
        elif kind == "arxiv":
            prefix = "ARXIV:"
        else:
            raise UnsupportedOperation(self.name, kind)

        def call():
            payload = self._get(
                "{}/paper/{}".format(self.BASE, urllib.parse.quote(prefix + value, safe="")),
                params={"fields": self._FIELDS},
            )
            candidate = self._candidate_from_paper(payload)
            returned = {"doi": candidate.doi, "arxiv": candidate.arxiv_id}[kind]
            if returned and _normalize_identifier(kind, returned) != _normalize_identifier(kind, value):
                return ProviderResult(
                    failure=model.DiscoveryFailure(
                        kind=model.IDENTITY_CONFLICT, provider=self.name,
                        message="the identifier resolved to a different record",
                        detail={"requested": value, "returned": returned},
                    )
                )
            return ProviderResult(candidates=(candidate,))

        return self._guarded(call)

    def search(self, query, *, limit, year=None):
        def call():
            params = {"query": query, "limit": limit, "fields": self._FIELDS}
            if year is not None:
                params["year"] = str(year)
            payload = self._get("{}/paper/search".format(self.BASE), params=params)
            items = payload.get("data") or []
            candidates = tuple(self._candidate_from_paper(item) for item in items)
            total = payload.get("total")
            return ProviderResult(
                candidates=candidates,
                truncated=bool(total is not None and total > len(candidates)),
            )

        return self._guarded(call)

    def references(self, target, *, limit):
        identifier = _s2_identifier(target)
        if not identifier:
            raise UnsupportedOperation(self.name, "references:identifier")
        return self._expansion(identifier, "references", limit)

    def citations(self, target, *, limit):
        identifier = _s2_identifier(target)
        if not identifier:
            raise UnsupportedOperation(self.name, "citations:identifier")
        return self._expansion(identifier, "citations", limit)

    def _expansion(self, identifier, relation, limit):
        def call():
            payload = self._get(
                "{}/paper/{}/{}".format(
                    self.BASE, urllib.parse.quote(identifier, safe=""), relation
                ),
                params={"limit": limit, "fields": self._FIELDS},
            )
            key = "citedPaper" if relation == "references" else "citingPaper"
            items = [entry.get(key) for entry in (payload.get("data") or [])]
            candidates = tuple(
                self._candidate_from_paper(item) for item in items if item
            )
            return ProviderResult(candidates=candidates, truncated=len(items) >= limit)

        return self._guarded(call)

    def _candidate_from_paper(self, item):
        external = item.get("externalIds") or {}
        doi = _clean(external.get("DOI"))
        arxiv_id = _clean(external.get("ArXiv"))
        others = {
            key: value for key, value in external.items() if key not in ("DOI", "ArXiv")
        }
        authors = tuple(
            _clean(author.get("name")) for author in (item.get("authors") or ())
        )
        authors = tuple(author for author in authors if author)
        locations = []
        open_access = item.get("openAccessPdf") or {}
        if open_access.get("url"):
            status = (open_access.get("status") or "").upper()
            locations.append(
                model.CandidateArtifactLocation(
                    url=open_access["url"], provider=self.name,
                    artifact_type=_artifact_type(None, open_access["url"]),
                    access=model.ACCESS_OPEN if status in ("GOLD", "GREEN", "BRONZE", "HYBRID")
                    else model.ACCESS_UNKNOWN,
                    evidence="s2.openAccessPdf:{}".format(status or "UNKNOWN"),
                    version=status or None,
                )
            )
        url = _clean(item.get("url"))
        if url:
            locations.append(
                model.CandidateArtifactLocation(
                    url=url, provider=self.name, artifact_type=model.ARTIFACT_LANDING,
                    access=model.ACCESS_UNKNOWN, evidence="s2.url",
                )
            )
        venue = _clean(item.get("venue")) or _clean((item.get("journal") or {}).get("name"))
        return _candidate(
            self.name,
            item.get("paperId") or "unknown",
            title=_clean(item.get("title")),
            authors=authors,
            year=item.get("year"),
            venue=venue,
            doi=doi,
            arxiv_id=arxiv_id,
            other_ids=model.other_ids(others),
            abstract=model.bounded_text(item.get("abstract")),
            url=url,
            artifact_locations=tuple(locations),
            provenance={
                "endpoint": "paper",
                "publication_types": list(item.get("publicationTypes") or ()),
            },
        )


class OpenReviewProvider(ProviderAdapter):
    name = "openreview"
    capabilities = frozenset({"openreview", "search"})
    BASE = "https://api2.openreview.net"

    def resolve(self, kind, value):
        if kind != "openreview":
            raise UnsupportedOperation(self.name, kind)

        def call():
            payload = self.fetcher.get_json(
                self.name, "{}/notes".format(self.BASE),
                params={"forum": value, "limit": 1},
            )
            notes = [note for note in (payload.get("notes") or ()) if _or_value(
                (note.get("content") or {}).get("title")
            )]
            if not notes:
                return ProviderResult(
                    failure=model.DiscoveryFailure(
                        kind=model.NOT_FOUND, provider=self.name,
                        message="OpenReview has no submission note for this identifier",
                        detail={"openreview": value},
                    )
                )
            return ProviderResult(candidates=(self._candidate_from_note(notes[0]),))

        return self._guarded(call)

    def search(self, query, *, limit, year=None):
        def call():
            payload = self.fetcher.get_json(
                self.name, "{}/notes/search".format(self.BASE),
                params={
                    "term": query, "limit": limit, "content": "all",
                    "group": "all", "source": "forum",
                },
            )
            notes = [note for note in (payload.get("notes") or ()) if _or_value(
                (note.get("content") or {}).get("title")
            )]
            candidates = tuple(self._candidate_from_note(note) for note in notes)
            count = payload.get("count")
            return ProviderResult(
                candidates=candidates,
                truncated=bool(count is not None and count > len(candidates)),
            )

        return self._guarded(call)

    def _candidate_from_note(self, note):
        content = note.get("content") or {}
        identifier = _clean(note.get("forum")) or _clean(note.get("id"))
        title = _clean(_or_value(content.get("title")))
        authors = _or_value(content.get("authors"))
        authors = tuple(_clean(author) for author in (authors or ())) if isinstance(authors, list) else ()
        authors = tuple(author for author in authors if author)
        abstract = model.bounded_text(_clean(_or_value(content.get("abstract"))))
        year = None
        cdate = note.get("cdate") or note.get("tcdate")
        if cdate:
            try:
                year = datetime.datetime.fromtimestamp(
                    int(cdate) / 1000.0, datetime.timezone.utc
                ).year
            except (TypeError, ValueError, OverflowError, OSError):
                year = None
        venue = _clean(_or_value(content.get("venue"))) or _clean(
            _or_value(content.get("venueid"))
        )
        locations = []
        pdf = _clean(_or_value(content.get("pdf")))
        if pdf:
            locations.append(
                model.CandidateArtifactLocation(
                    url=pdf, provider=self.name, artifact_type=model.ARTIFACT_PDF,
                    media_type="application/pdf", access=model.ACCESS_UNKNOWN,
                    evidence="openreview.content.pdf",
                )
            )
        url = "https://openreview.net/forum?id={}".format(identifier) if identifier else None
        return _candidate(
            self.name,
            identifier or "unknown",
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            openreview_id=identifier,
            abstract=abstract,
            url=url,
            artifact_locations=tuple(locations),
            provenance={"endpoint": "notes/search", "invitation": _clean(note.get("invitation"))},
        )


def default_providers(fetcher, *, env=None, api_keys=None):
    """Build the default provider set in a stable discovery order."""

    import os

    environment = os.environ if env is None else env
    keys = dict(api_keys or {})

    def key_for(name, env_name):
        if name in keys:
            return keys[name]
        if env_name:
            return (environment.get(env_name) or "").strip() or None
        return None

    providers = [
        CrossrefProvider(fetcher),
        ArxivProvider(fetcher),
        SemanticScholarProvider(
            fetcher, api_key=key_for("semanticscholar", SemanticScholarProvider.api_key_env)
        ),
        OpenReviewProvider(fetcher),
    ]
    return {provider.name: provider for provider in providers}


# -- shared parsing helpers -------------------------------------------------


def _or_value(field):
    """OpenReview wraps every content field as ``{"value": ...}`` (or v1 plain)."""

    if isinstance(field, dict) and "value" in field:
        return field.get("value")
    return field


def _atom_entries(text):
    root = ET.fromstring(text)
    return root.findall("{http://www.w3.org/2005/Atom}entry")


def _atom_total_results(text):
    root = ET.fromstring(text)
    node = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    if node is None or node.text is None:
        return None
    try:
        return int(node.text.strip())
    except ValueError:
        return None


def _text(element, path):
    node = element.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _arxiv_from_url(value):
    if not value:
        return None
    tail = value.rstrip("/").rsplit("/", 1)[-1]
    return _clean(tail) or None


def _normalize_arxiv(value):
    value = (value or "").strip().lower()
    if value.startswith("arxiv:"):
        value = value[len("arxiv:"):]
    return value.split("v")[0] if "v" in value else value


def _normalize_identifier(kind, value):
    if kind == "arxiv":
        return _normalize_arxiv(value)
    return (value or "").strip().lower()


def _s2_identifier(target):
    if target.get("arxiv"):
        return "ARXIV:{}".format(target["arxiv"])
    if target.get("doi"):
        return "DOI:{}".format(target["doi"])
    if target.get("provider_record_id"):
        return target["provider_record_id"]
    return None


def _artifact_type(content_type, url):
    declared = (content_type or "").lower()
    lower = (url or "").lower()
    if "pdf" in declared or lower.endswith(".pdf"):
        return model.ARTIFACT_PDF
    if "html" in declared or lower.endswith(".html") or lower.endswith(".htm"):
        return model.ARTIFACT_HTML
    if "unspecified" in declared or not declared:
        return model.ARTIFACT_LANDING
    return model.ARTIFACT_OTHER
