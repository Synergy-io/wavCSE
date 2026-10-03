"""Structured scholarly discovery: identity and search inputs → normalized candidates.

`StructuredDiscovery` is the one interface the Literature Agent reasons about. It
orchestrates provider adapters behind the normalized model in :mod:`.model` and
classifies every candidate against the existing catalog through
`literature_query.LiteratureQuery.identify_candidate` — the same `known` / `new` /
`ambiguous` vocabulary and exact-identity rules the rest of the repository uses.
Discovery never creates a Paper identity: it is a *view* over the catalog plus
provider metadata.

Hard boundary (INC-012): this module performs metadata discovery only. It never
downloads, retains, hashes or admits an artifact, never writes `primary_manifest`
or the acquisition ledger, and never calls `LiteraturePrimary.register`. A
`CandidateArtifactLocation` is unvalidated discovery evidence, never a
`PrimaryArtifact`.

Authority: read-only with respect to research state. ``search`` and ``discover``
have no side effects beyond an in-memory request cache.

Multi-provider aggregation policy (INC-016). Every queryable provider is invoked
with the same global ``limit``; their result streams are then merged by
deterministic round-robin interleave in provider order, deduplicated by the
candidate's strongest shared identifier (DOI > arXiv id > OpenReview id >
provider-local record id), and truncated to the global bound. Because the merge
interleaves rather than concatenates, one provider cannot consume the whole
candidate budget merely because it is consulted first, while a lone healthy
provider still fills the whole budget. Provider order, per-provider result order,
and the dedup key are all deterministic, so identical inputs yield an identical
result.
"""

import re

from . import http as http_layer
from . import model
from . import providers as provider_module


_DOI = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9]+$", re.IGNORECASE)
_ARXIV = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)


class DiscoveryError(ValueError):
    """A discovery request is malformed before any provider is consulted."""


class StructuredDiscovery:
    """Deterministic structured discovery over a small set of provider adapters."""

    DEFAULT_LIMIT = 10
    MAX_LIMIT = 25

    def __init__(
        self,
        *,
        providers=None,
        fetcher=None,
        query=None,
        env=None,
        catalog_path=None,
        repo_root=None,
        cache=None,
        sleep=None,
    ):
        if fetcher is None:
            fetcher = http_layer.HttpFetcher(
                cache=cache if cache is not None else http_layer.BoundedCache(),
                **({"sleep": sleep} if sleep is not None else {}),
            )
        self._fetcher = fetcher
        self._providers = (
            providers
            if providers is not None
            else provider_module.default_providers(fetcher, env=env)
        )
        self._order = tuple(self._providers)
        self._query = query
        self._env = env
        self._catalog_path = catalog_path
        self._repo_root = repo_root

    # -- introspection -----------------------------------------------------

    @property
    def order(self):
        return self._order

    def provider_status(self):
        statuses = []
        for name in self._order:
            try:
                statuses.append(self._providers[name].describe())
            except AttributeError:
                statuses.append({"provider": name, "available": True})
        return tuple(statuses)

    # -- public discovery interface ---------------------------------------

    def discover(self, *, doi=None, arxiv=None, title=None, authors=None,
                 venue=None, year=None, limit=None):
        """Resolve an identifier or a near-exact title into candidate papers."""

        request = {"doi": doi, "arxiv": arxiv, "title": title}
        authors = tuple(authors or ())
        try:
            bound = self._limit(limit)
            if doi is not None:
                doi = self._validate_doi(doi)
                request["doi"] = doi
            if arxiv is not None:
                arxiv = self._validate_arxiv(arxiv)
                request["arxiv"] = arxiv
            if title is not None:
                title = self._validate_text(title, "title")
                request["title"] = title
            if venue is not None:
                request["venue"] = self._validate_text(venue, "venue")
            if year is not None:
                request["year"] = year
            if authors:
                request["authors"] = list(authors)
        except DiscoveryError as exc:
            return self._failure_result(
                mode="identity", query=request,
                failures=[model.DiscoveryFailure(kind=model.INVALID_QUERY, message=str(exc))],
                limit=self.DEFAULT_LIMIT,
            )

        if not any((doi, arxiv, title)):
            return self._failure_result(
                mode="identity",
                query=request,
                failures=[
                    model.DiscoveryFailure(
                        kind=model.INVALID_QUERY,
                        message="one of doi, arxiv or title is required",
                    )
                ],
                limit=bound,
            )

        if doi or arxiv:
            return self._discover_identity(
                request=request, doi=doi, arxiv=arxiv, limit=bound
            )
        return self._discover_lookup(
            request=request, title=title, authors=authors, venue=venue,
            year=year, limit=bound,
        )

    def search(self, query, *, limit=None, year=None):
        """Search providers by a scholarly query and return bounded candidates."""

        request = {"query": query}
        try:
            bound = self._limit(limit)
            query = self._validate_text(query, "query")
            request["query"] = query
            if year is not None:
                request["year"] = year
        except DiscoveryError as exc:
            return self._failure_result(
                mode="search", query=request,
                failures=[model.DiscoveryFailure(kind=model.INVALID_QUERY, message=str(exc))],
                limit=self.DEFAULT_LIMIT,
            )

        groups = []
        failures = []
        truncated = False
        for name in self._order:
            if "search" not in self._providers[name].capabilities:
                continue
            result = self._invoke(name, "search", query=query, limit=bound, year=year)
            if result is None:
                continue
            failures.extend([result.failure] if result.failure else [])
            truncated = truncated or result.truncated
            groups.append((name, self._classify_all(result.candidates)))

        queried = [name for name, _ in groups]
        candidates, merged_truncated = self._merge_providers(groups, bound)
        return self._assemble(
            mode="search", request=request, candidates=candidates,
            failures=failures, queried=queried, limit=bound,
            truncated=truncated or merged_truncated,
        )

    def references(self, paper_id, *, limit=None):
        """Expand a retained paper's reference list where a provider supports it."""

        return self._expand(paper_id, mode="references", relation=model.REFERENCE,
                            limit=self._limit(limit))

    def citations(self, paper_id, *, limit=None):
        """Expand a retained paper's citing works where a provider supports it."""

        return self._expand(paper_id, mode="citations", relation=model.CITATION,
                            limit=self._limit(limit))

    # -- modes -------------------------------------------------------------

    def _discover_identity(self, *, request, doi, arxiv, limit):
        target = {"doi": doi, "arxiv": arxiv}
        request_identity = self._classify(doi=doi, arxiv=arxiv)
        if request_identity is not None and request_identity.status == model.AMBIGUOUS:
            return self._failure_result(
                mode="identity", query=request, limit=limit,
                identity=request_identity,
                failures=[
                    model.DiscoveryFailure(
                        kind=model.IDENTITY_CONFLICT,
                        message="the supplied identifiers disagree; refusing to merge",
                        detail={"conflicts": list(request_identity.conflicts)},
                    )
                ],
            )

        collected = []
        failures = []
        queried = []
        for name in self._order:
            provider = self._providers[name]
            kind = "doi" if doi else "arxiv"
            if kind not in provider.capabilities:
                continue
            result = self._invoke(name, "resolve", kind=kind, value=(doi or arxiv))
            if result is None:
                continue
            queried.append(name)
            if result.failure:
                failures.append(result.failure)
                continue
            collected.extend(self._classify_all(result.candidates))

        candidates, dropped = self._truncate(collected, limit)
        return self._assemble(
            mode="identity", request=request, candidates=candidates, failures=failures,
            queried=queried, limit=limit, truncated=dropped, identity=request_identity,
        )

    def _discover_lookup(self, *, request, title, authors, venue, year, limit):
        composed = title
        if authors:
            composed = "{} {}".format(title, " ".join(authors))
        groups = []
        failures = []
        truncated = False
        for name in self._order:
            if "search" not in self._providers[name].capabilities:
                continue
            result = self._invoke(name, "search", query=composed, limit=limit, year=year)
            if result is None:
                continue
            failures.extend([result.failure] if result.failure else [])
            truncated = truncated or result.truncated
            groups.append((name, self._classify_all(result.candidates)))
        queried = [name for name, _ in groups]
        ranked, merged_truncated = self._merge_providers(
            groups, limit, rank=lambda items: self._rank_by_title(items, title)
        )
        return self._assemble(
            mode="lookup", request=request, candidates=ranked, failures=failures,
            queried=queried, limit=limit, truncated=truncated or merged_truncated,
        )

    def _expand(self, paper_id, *, mode, relation, limit):
        if not isinstance(paper_id, str) or not paper_id.strip():
            return self._failure_result(
                mode=mode, query={"paper_id": paper_id}, limit=limit,
                failures=[model.DiscoveryFailure(
                    kind=model.INVALID_QUERY, message="paper_id must be a non-empty string")],
            )
        paper_id = paper_id.strip()
        try:
            paper = self._query_or_default().resolve_paper(paper_id)
        except Exception as exc:  # CatalogError wrapped by LiteratureQueryError
            return self._failure_result(
                mode=mode, query={"paper_id": paper_id}, limit=limit,
                failures=[model.DiscoveryFailure(
                    kind=model.INVALID_QUERY,
                    message="unknown paper_id {!r}".format(paper_id),
                    detail={"reason": str(exc)},
                )],
            )

        target, source = self._expand_target(paper)
        capability = "references" if relation == model.REFERENCE else "citations"
        groups = []
        failures = []
        for name in self._order:
            provider = self._providers[name]
            if capability not in provider.capabilities or not self._can_target(provider, target):
                continue
            result = self._invoke(name, capability, target=target, limit=limit)
            if result is None:
                continue
            failures.extend([result.failure] if result.failure else [])
            groups.append((name, self._classify_all(result.candidates, relation=relation)))

        queried = [name for name, _ in groups]
        candidates, dropped = self._merge_providers(groups, limit)
        if not candidates and not queried:
            failures.append(
                model.DiscoveryFailure(
                    kind=model.DISCOVERY_EXHAUSTED,
                    message="no available provider supports {} for this paper".format(mode),
                    detail={"paper_id": paper_id, "identifiers": sorted(target)},
                )
            )
        return self._assemble(
            mode=mode, request={"paper_id": paper_id, "identifiers": target},
            candidates=candidates, failures=failures, queried=queried, limit=limit,
            truncated=dropped, identity=None,
        )

    def _expand_target(self, paper):
        external = getattr(paper, "external_ids", None) or {}
        doi = _first_id(external, "doi")
        arxiv = _first_id(external, "arxiv")
        target = {"paper_id": getattr(paper, "paper_id", None), "doi": doi, "arxiv": arxiv}
        return {key: value for key, value in target.items() if value}, paper

    @staticmethod
    def _can_target(provider, target):
        return bool(
            target.get("doi") or target.get("arxiv") or target.get("provider_record_id")
        )

    # -- helpers -----------------------------------------------------------

    def _invoke(self, name, method, **kwargs):
        provider = self._providers[name]
        available, reason = provider.availability()
        if not available:
            return provider_module.ProviderResult(
                failure=model.DiscoveryFailure(
                    kind=model.PROVIDER_UNAVAILABLE, provider=name,
                    message=reason or "{} is unavailable".format(name),
                )
            )
        try:
            return getattr(provider, method)(**kwargs)
        except provider_module.UnsupportedOperation:
            return None

    def _classify_all(self, candidates, relation=None):
        classified = []
        for candidate in candidates:
            verdict = self._classify(
                title=candidate.title, doi=candidate.doi, arxiv=candidate.arxiv_id
            )
            classified.append(
                model.DiscoveredCandidate(
                    paper=candidate,
                    identity=verdict or model.IdentityVerdict(status=model.NEW),
                    relation=relation,
                )
            )
        return classified

    def _classify(self, *, paper_id=None, title=None, doi=None, arxiv=None):
        fields = {}
        if paper_id:
            fields["paper_id"] = paper_id
        if doi:
            fields["doi"] = doi
        if arxiv:
            fields["arxiv"] = arxiv
        # Only fall back to the title when no exact identifier is available, so a
        # provider's title variant can never weaken an exact identity match.
        if not fields and title:
            fields["title"] = title
        if not fields:
            return model.IdentityVerdict(status=model.NEW)
        try:
            match = self._query_or_default().identify_candidate(**fields)
        except Exception:
            return model.IdentityVerdict(status=model.NEW)
        conflicts = tuple(
            sorted(getattr(paper, "paper_id", "?") for paper in match.conflicts)
        )
        matched = getattr(match.paper, "paper_id", None) if match.paper else None
        return model.IdentityVerdict(
            status=match.status,
            matched_paper_id=matched,
            matched_field=match.matched_field,
            conflicts=conflicts,
        )

    def _query_or_default(self):
        if self._query is None:
            from improvements.taskrelation.research import literature_query

            kwargs = {}
            if self._catalog_path is not None:
                kwargs["catalog_path"] = self._catalog_path
            if self._repo_root is not None:
                kwargs["repo_root"] = self._repo_root
            self._query = literature_query.LiteratureQuery(**kwargs)
        return self._query

    def _assemble(self, *, mode, request, candidates, failures, queried, limit,
                  truncated, identity=None):
        providers_failed = tuple(
            sorted({failure.provider for failure in failures if failure.provider})
        )
        failures = list(failures)
        if not candidates and (failures or not queried):
            if not any(failure.kind == model.DISCOVERY_EXHAUSTED for failure in failures):
                failures.append(
                    model.DiscoveryFailure(
                        kind=model.DISCOVERY_EXHAUSTED,
                        message="no provider returned a candidate for this request",
                        detail={"providers_queried": list(queried)},
                    )
                )
        return model.DiscoveryResult(
            mode=mode,
            query=request,
            candidates=tuple(candidates),
            failures=tuple(failures),
            identity=identity,
            providers_queried=tuple(queried),
            providers_failed=providers_failed,
            limit=limit,
            truncated=bool(truncated),
        )

    def _failure_result(self, *, mode, query, failures, limit, identity=None):
        return model.DiscoveryResult(
            mode=mode, query=query, candidates=(), failures=tuple(failures),
            identity=identity, providers_queried=(), providers_failed=(),
            limit=limit, truncated=False,
        )

    def _truncate(self, candidates, limit):
        if len(candidates) <= limit:
            return candidates, False
        return candidates[:limit], True

    def _merge_providers(self, groups, limit, *, rank=None):
        """Merge per-provider candidate streams deterministically and fairly.

        Aggregation policy: every provider is queried with the same global bound,
        and its results are interleaved *round-robin* in provider order rather
        than concatenated. A single provider therefore cannot consume the whole
        budget merely because it executes first — any provider that returned a
        candidate is represented before any provider contributes a second one,
        while a lone healthy provider still fills the whole budget. The merged
        stream is deduplicated by its strongest shared identifier (DOI, then
        arXiv id, then OpenReview id, otherwise the provider-local record id), so
        the same work returned by two providers is not counted twice; the first
        occurrence in provider order is kept together with its own identity
        verdict. Ordering is fully deterministic: provider order, then each
        provider's own result order. An optional ``rank`` reorders the bounded
        result (title lookup promotes exact matches). Returns
        ``(candidates, truncated)``.
        """
        seen = set()
        merged = []
        cursors = [0] * len(groups)
        total = sum(len(items) for _, items in groups)
        taken = 0
        while len(merged) < limit and taken < total:
            for index, (_, items) in enumerate(groups):
                if len(merged) >= limit or taken >= total:
                    break
                cursor = cursors[index]
                if cursor >= len(items):
                    continue
                candidate = items[cursor]
                cursors[index] = cursor + 1
                taken += 1
                key = _candidate_identity_key(candidate.paper)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)
        if rank is not None:
            merged = rank(merged)
        return merged, taken < total

    def _rank_by_title(self, candidates, title):
        normalized = _normalize_title(title)
        exact = [candidate for candidate in candidates
                 if _normalize_title(candidate.paper.title) == normalized]
        rest = [candidate for candidate in candidates if candidate not in exact]
        return exact + rest

    @staticmethod
    def _limit(value):
        if value is None:
            return StructuredDiscovery.DEFAULT_LIMIT
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise DiscoveryError("limit must be a positive integer")
        return min(value, StructuredDiscovery.MAX_LIMIT)

    @staticmethod
    def _validate_text(value, field):
        if not isinstance(value, str) or not value.strip():
            raise DiscoveryError("{} must be a non-empty string".format(field))
        return value.strip()

    @staticmethod
    def _validate_doi(value):
        value = StructuredDiscovery._validate_text(value, "doi")
        if not _DOI.match(value):
            raise DiscoveryError("doi {!r} is not a valid DOI".format(value))
        return value

    @staticmethod
    def _validate_arxiv(value):
        value = StructuredDiscovery._validate_text(value, "arxiv")
        if not _ARXIV.match(value):
            raise DiscoveryError("arxiv {!r} is not a valid arXiv identifier".format(value))
        return value


def _candidate_identity_key(paper):
    """A deterministic cross-provider dedup key for one normalized candidate.

    The strongest shared identifier wins, so the same work returned by two
    providers collapses to a single candidate in an aggregated result set; a
    record that carries no shared identifier is keyed by its provider-local
    record id and is never merged across providers.
    """

    if paper.doi:
        return ("doi", paper.doi.strip().lower())
    if paper.arxiv_id:
        return ("arxiv", paper.arxiv_id.strip().lower())
    if paper.openreview_id:
        return ("openreview", paper.openreview_id.strip())
    return ("record", paper.provider, paper.provider_record_id)


def _first_id(external, kind):
    values = external.get(kind) if external else None
    if isinstance(values, (list, tuple)):
        return values[0] if values else None
    return values


def _normalize_title(value):
    import unicodedata

    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


__all__ = ["StructuredDiscovery", "DiscoveryError"]
