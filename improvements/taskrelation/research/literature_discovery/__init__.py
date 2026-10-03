"""Structured scholarly discovery (INC-012).

Metadata-only discovery: identity and search inputs in, normalized candidate
papers and *unvalidated* candidate artifact locations out. No artifact bytes are
fetched, and no canonical Paper identity or `PrimaryArtifact` is created here.

    from improvements.taskrelation.research.literature_discovery import discover

    result = discover(doi="10.1145/3580305.3599261")
    result.as_dict()
"""

from .core import DiscoveryError, StructuredDiscovery
from .http import BoundedCache, HttpError, HttpFetcher, RateLimiter, UrllibTransport
from .model import (
    AMBIGUOUS,
    CITATION,
    KNOWN,
    NEW,
    REFERENCE,
    CandidateArtifactLocation,
    CandidatePaper,
    DiscoveredCandidate,
    DiscoveryFailure,
    DiscoveryResult,
    IdentityVerdict,
)
from .providers import default_providers

__all__ = [
    "StructuredDiscovery",
    "DiscoveryError",
    "discover",
    "search",
    "references",
    "citations",
    "HttpFetcher",
    "HttpError",
    "UrllibTransport",
    "RateLimiter",
    "BoundedCache",
    "default_providers",
    "CandidatePaper",
    "CandidateArtifactLocation",
    "DiscoveredCandidate",
    "DiscoveryResult",
    "DiscoveryFailure",
    "IdentityVerdict",
    "KNOWN",
    "NEW",
    "AMBIGUOUS",
    "REFERENCE",
    "CITATION",
]


def discover(**kwargs):
    """Convenience one-shot discovery over the default provider set."""

    return StructuredDiscovery().discover(**kwargs)


def search(query, **kwargs):
    return StructuredDiscovery().search(query, **kwargs)


def references(paper_id, **kwargs):
    return StructuredDiscovery().references(paper_id, **kwargs)


def citations(paper_id, **kwargs):
    return StructuredDiscovery().citations(paper_id, **kwargs)
