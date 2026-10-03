"""Deterministic, bounded HTTP for metadata-only scholarly discovery.

This layer exists so provider adapters never see redirects, retries, timeouts,
rate limits or raw transport exceptions. It is deliberately small:

* HTTPS only; a redirect that would downgrade the scheme is refused;
* a hard response-size bound and a per-request timeout;
* a small bounded retry/backoff policy for 429 and 5xx, honoring ``Retry-After``;
* a per-provider minimum request interval, so a search cannot burst a provider;
* a bounded in-memory TTL cache keyed by the exact request;
* every failure is a structured :class:`HttpError`, never a raw exception.

Nothing here writes to disk, and no credential is ever logged or returned. The
transport is injectable, so tests exercise the whole policy with fixtures and no
network.

Only metadata endpoints are reachable through this class. It is never given an
artifact URL: downloading artifact bytes is acquisition (INC-013), not discovery.
"""

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Mapping

from . import model


DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 1.0
DEFAULT_CACHE_TTL = 300.0
DEFAULT_MAX_CACHE_ENTRIES = 128
MAX_BACKOFF = 30.0

# Conservative per-provider minimum interval between requests. arXiv asks for a
# gap between calls; Crossref and Semantic Scholar ask for identification and
# steady traffic; OpenReview is lightly loaded by these queries.
DEFAULT_INTERVALS = {
    "crossref": 0.5,
    "arxiv": 3.0,
    "semanticscholar": 1.0,
    "openreview": 0.5,
}

DEFAULT_CONTACT_ENV = "WAVCSE_DISCOVERY_CONTACT"
_TOOL_UA = "wavcse-structured-discovery/0.1"


def default_user_agent(env=None):
    """Build the User-Agent from the optional configured contact address.

    A contact email is recommended by Crossref and Semantic Scholar so the
    caller is placed in their polite pool. The value is configuration, never a
    credential; it is read at runtime and never persisted.
    """

    import os

    environment = os.environ if env is None else env
    contact = (environment.get(DEFAULT_CONTACT_ENV) or "").strip()
    if contact:
        return "{} (mailto:{})".format(_TOOL_UA, contact)
    return _TOOL_UA


class HttpError(Exception):
    """A structured networking failure; `kind` is a discovery failure taxonomy value."""

    def __init__(self, kind, message, *, provider=None, status=None, detail=None):
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        self.status = status
        self.detail = dict(detail or {})

    def failure(self):
        payload = dict(self.detail)
        if self.status is not None:
            payload.setdefault("status", self.status)
        return model.DiscoveryFailure(
            kind=self.kind, provider=self.provider, message=str(self), detail=payload
        )


class HttpTransportError(Exception):
    """A transport-level failure (DNS, connection, timeout)."""


@dataclass
class HttpResponse:
    status: int
    headers: Mapping
    body: bytes
    url: str


class Transport:
    """The minimal transport contract the fetcher depends on."""

    def request(self, url, *, headers, timeout, max_bytes):
        raise NotImplementedError


class UrllibTransport(Transport):
    """The default transport. One hop only; redirects are handled by the fetcher."""

    def request(self, url, *, headers, timeout, max_bytes):
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read(max_bytes + 1)
                return HttpResponse(
                    status=response.status,
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=body,
                    url=response.geturl(),
                )
        except urllib.error.HTTPError as exc:
            body = exc.read(max_bytes + 1) if hasattr(exc, "read") else b""
            return HttpResponse(
                status=exc.code,
                headers={key.lower(): value for key, value in (exc.headers or {}).items()},
                body=body,
                url=url,
            )
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise HttpTransportError(str(exc)) from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass
class RateLimiter:
    """A per-provider minimum-interval limiter with an injectable clock/sleep."""

    intervals: Mapping = field(default_factory=lambda: dict(DEFAULT_INTERVALS))
    clock: Callable = time.monotonic
    sleep: Callable = time.sleep

    def __post_init__(self):
        self._last = {}

    def acquire(self, provider):
        interval = float(self.intervals.get(provider, 0.0))
        if interval <= 0:
            return
        now = self.clock()
        previous = self._last.get(provider)
        if previous is not None:
            wait = interval - (now - previous)
            if wait > 0:
                self.sleep(wait)
                now = self.clock()
        self._last[provider] = now


@dataclass
class BoundedCache:
    """A bounded LRU cache with a TTL. In-memory only; never persisted."""

    ttl: float = DEFAULT_CACHE_TTL
    max_entries: int = DEFAULT_MAX_CACHE_ENTRIES
    clock: Callable = time.monotonic

    def __post_init__(self):
        self._entries = OrderedDict()

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and self.clock() >= expires_at:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    def set(self, key, value, ttl=None):
        lifetime = self.ttl if ttl is None else ttl
        expires_at = None if lifetime is None else self.clock() + float(lifetime)
        self._entries[key] = (expires_at, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


class HttpFetcher:
    """Bounded, rate-limited, cached metadata HTTP with a structured failure model."""

    def __init__(
        self,
        *,
        transport=None,
        timeout=DEFAULT_TIMEOUT,
        max_response_bytes=DEFAULT_MAX_RESPONSE_BYTES,
        max_redirects=DEFAULT_MAX_REDIRECTS,
        retries=DEFAULT_RETRIES,
        backoff=DEFAULT_BACKOFF,
        user_agent=None,
        rate_limiter=None,
        cache=None,
        sleep=time.sleep,
    ):
        self.transport = transport if transport is not None else UrllibTransport()
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.max_redirects = max_redirects
        self.retries = retries
        self.backoff = backoff
        self.user_agent = user_agent or default_user_agent()
        self.rate_limiter = rate_limiter
        self.cache = cache
        self._sleep = sleep

    def build_url(self, url, params):
        if not params:
            return url
        filtered = {key: value for key, value in params.items() if value is not None}
        separator = "&" if urllib.parse.urlparse(url).query else "?"
        return url + separator + urllib.parse.urlencode(filtered, doseq=True)

    def fetch(self, provider, url, *, params=None, accept=None, headers=None,
              cache_ttl=None):
        """Return the response body bytes or raise a structured :class:`HttpError`.

        ``headers`` carries per-request credentials (e.g. an API key). Headers are
        never cached, logged, or attached to an error; only the response body is
        cached, under a key that does not include them.
        """

        target = self.build_url(url, params)
        cache_key = (provider, target, accept)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        request_headers = {"User-Agent": self.user_agent, "Accept": accept or "application/json"}
        if headers:
            request_headers.update(headers)
        redirects = 0
        attempt = 0
        last_error = None
        while attempt <= self.retries:
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(provider)
            try:
                response = self.transport.request(
                    target, headers=request_headers, timeout=self.timeout,
                    max_bytes=self.max_response_bytes,
                )
            except HttpTransportError as exc:
                last_error = HttpError(
                    model.PROVIDER_UNAVAILABLE,
                    "{} could not be reached: {}".format(provider, exc),
                    provider=provider,
                    detail={"url": _redact(target)},
                )
                attempt += 1
                if attempt <= self.retries:
                    self._pause(self._backoff_for(attempt))
                    continue
                raise last_error from exc

            status = response.status
            if 300 <= status < 400:
                location = (response.headers or {}).get("location")
                if not location or redirects >= self.max_redirects:
                    raise HttpError(
                        model.PROVIDER_ERROR,
                        "{} returned an unusable redirect".format(provider),
                        provider=provider, status=status, detail={"url": _redact(target)},
                    )
                target = urllib.parse.urljoin(target, location)
                if not target.lower().startswith("https://"):
                    raise HttpError(
                        model.PROVIDER_ERROR,
                        "{} redirected to a non-HTTPS URL; refused".format(provider),
                        provider=provider, status=status, detail={"url": _redact(target)},
                    )
                redirects += 1
                continue

            if status == 200:
                if len(response.body) > self.max_response_bytes:
                    raise HttpError(
                        model.MALFORMED_PROVIDER_RESPONSE,
                        "{} response exceeded the size bound".format(provider),
                        provider=provider, status=status,
                        detail={"bound": self.max_response_bytes,
                                "url": _redact(target)},
                    )
                if self.cache is not None:
                    self.cache.set(cache_key, response.body, ttl=cache_ttl)
                return response.body

            if status == 429:
                retry_after = _retry_after(response.headers)
                last_error = HttpError(
                    model.RATE_LIMITED,
                    "{} rate-limited the request".format(provider),
                    provider=provider, status=status,
                    detail={"retry_after": retry_after, "url": _redact(target)},
                )
                attempt += 1
                if attempt <= self.retries:
                    self._pause(retry_after if retry_after else self._backoff_for(attempt))
                    continue
                raise last_error

            if 500 <= status < 600:
                last_error = HttpError(
                    model.PROVIDER_ERROR,
                    "{} returned server error {}".format(provider, status),
                    provider=provider, status=status, detail={"url": _redact(target)},
                )
                attempt += 1
                if attempt <= self.retries:
                    self._pause(self._backoff_for(attempt))
                    continue
                raise last_error

            if status in (401, 403):
                raise HttpError(
                    model.PROVIDER_AUTH_REQUIRED,
                    "{} requires credentials for this request".format(provider),
                    provider=provider, status=status, detail={"url": _redact(target)},
                )
            if status == 404:
                raise HttpError(
                    model.NOT_FOUND,
                    "{} has no record for this request".format(provider),
                    provider=provider, status=status, detail={"url": _redact(target)},
                )
            raise HttpError(
                model.PROVIDER_ERROR,
                "{} returned unexpected status {}".format(provider, status),
                provider=provider, status=status, detail={"url": _redact(target)},
            )

        raise last_error or HttpError(
            model.PROVIDER_ERROR, "{} request failed".format(provider), provider=provider
        )

    def get_json(self, provider, url, *, params=None, cache_ttl=None):
        body = self.fetch(provider, url, params=params, accept="application/json",
                          cache_ttl=cache_ttl)
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise HttpError(
                model.MALFORMED_PROVIDER_RESPONSE,
                "{} returned a body that is not JSON".format(provider),
                provider=provider, detail={"reason": str(exc)},
            ) from exc

    def get_text(self, provider, url, *, params=None, accept=None, cache_ttl=None):
        body = self.fetch(provider, url, params=params, accept=accept, cache_ttl=cache_ttl)
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HttpError(
                model.MALFORMED_PROVIDER_RESPONSE,
                "{} returned an undecodable body".format(provider),
                provider=provider, detail={"reason": str(exc)},
            ) from exc

    def _backoff_for(self, attempt):
        return min(self.backoff * (2 ** (attempt - 1)), MAX_BACKOFF)

    def _pause(self, seconds):
        if seconds and seconds > 0:
            self._sleep(min(float(seconds), MAX_BACKOFF))


def _retry_after(headers):
    value = (headers or {}).get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _redact(url):
    """Keep diagnostics useful without echoing query values as if they were trusted."""

    parsed = urllib.parse.urlparse(url)
    return parsed._replace(query="", fragment="").geturl()
