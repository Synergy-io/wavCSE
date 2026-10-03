"""Deterministic, page-indexed text extraction from a verified primary PDF.

The retained PDF bytes are the evidence; this module only derives a *view* of
them. Extraction is performed on demand from an already checksum-verified local
copy, and every view carries the artifact's SHA-256, so a derived view can never
be attributed to a different artifact version. Nothing here mutates state, and
no PDF is ever committed: the extractor reads only the disposable local copy that
``literature_primary`` resolved.

The backend is poppler's ``pdftotext``. It is deterministic for a given binary
version, preserves page boundaries (pages are separated by a form feed), and
exposes a text layer when the PDF has one. No OCR is attempted: a page with no
usable text layer is reported as empty rather than invented.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple

EXTRACTOR_UNAVAILABLE = "EXTRACTOR_UNAVAILABLE"
EXTRACTION_FAILED = "EXTRACTION_FAILED"
INVALID_LOCATOR = "INVALID_LOCATOR"

DEFAULT_MAX_CHARS = 20000
MAX_CHARS_CEILING = 100000

_BACKEND = "pdftotext"
_PAGE_SEPARATOR = "\f"
_MAX_WARNINGS = 20

# Injectable for tests: takes the argv and returns (returncode, stdout, stderr).
Runner = Callable[[Sequence[str]], Tuple[int, bytes, bytes]]


class PrimaryTextError(Exception):
    """A primary-text derivation failed with a deterministic, branchable kind."""

    def __init__(self, message, kind, detail=None):
        super(PrimaryTextError, self).__init__(message)
        self.kind = kind
        self.detail = dict(detail or {})

    def as_dict(self):
        return {"error": str(self), "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class ExtractedDocument:
    """A page-indexed text view of one PDF, plus how it was produced."""

    page_count: int
    pages: Tuple[str, ...]
    extractor: str
    extractor_version: str
    warnings: Tuple[str, ...]

    def text(self):
        return "\n\n".join(self.pages)


def extract_document(pdf_path, *, binary=None, runner=None):
    """Extract a page-indexed text view from ``pdf_path``.

    Raises :class:`PrimaryTextError` with kind ``EXTRACTOR_UNAVAILABLE`` when the
    backend is missing and ``EXTRACTION_FAILED`` when it exits non-zero. Never
    returns text for a PDF it could not read.
    """

    path = Path(pdf_path)
    executable = _resolve_binary(binary)
    version = _backend_version(executable, runner)
    argv = [executable, "-enc", "UTF-8", str(path), "-"]
    returncode, stdout, stderr = _run(argv, runner)
    if returncode != 0:
        raise PrimaryTextError(
            "primary-text extraction failed for the retained artifact",
            kind=EXTRACTION_FAILED,
            detail={"extractor": _BACKEND, "returncode": returncode},
        )
    pages = split_pages(stdout.decode("utf-8", "replace"))
    return ExtractedDocument(
        page_count=len(pages),
        pages=pages,
        extractor=_BACKEND,
        extractor_version=version,
        warnings=_warnings(stderr),
    )


def extract_text(pdf_path, **kwargs):
    """Return the whole extracted document as one string (pages joined)."""

    return extract_document(pdf_path, **kwargs).text()


def split_pages(raw):
    """Split ``pdftotext`` output into pages, preserving empty pages.

    poppler terminates every page with a form feed, including the last, so a
    single trailing separator is the document terminator, not an extra page.
    """

    if raw == "":
        return ()
    parts = raw.split(_PAGE_SEPARATOR)
    if parts and parts[-1] == "":
        parts.pop()
    return tuple(parts)


def validate_max_chars(value):
    if value is None:
        return DEFAULT_MAX_CHARS
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value > MAX_CHARS_CEILING
    ):
        raise PrimaryTextError(
            "max_chars must be an integer in 1..{}".format(MAX_CHARS_CEILING),
            kind=INVALID_LOCATOR,
        )
    return value


def validate_page(value, page_count, field="page"):
    """Validate a 1-based physical PDF page index against the document."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > page_count
    ):
        raise PrimaryTextError(
            "{} must be an integer in 1..{} (1-based physical PDF page); "
            "printed page labels are not used as identity".format(field, page_count),
            kind=INVALID_LOCATOR,
            detail={"field": field, "value": value, "page_count": page_count},
        )
    return value


def page_locator(page, page_end=None):
    """Deterministic locator string for a page or an inclusive page range."""

    if page_end is None or page_end == page:
        return "primary:page:{}".format(page)
    return "primary:pages:{}-{}".format(page, page_end)


def _resolve_binary(binary):
    if binary is None:
        binary = _BACKEND
    found = shutil.which(binary)
    if not found:
        raise PrimaryTextError(
            "no PDF text extractor is available ({} not found on PATH)".format(binary),
            kind=EXTRACTOR_UNAVAILABLE,
            detail={"extractor": binary},
        )
    return found


def _backend_version(executable, runner):
    try:
        returncode, _stdout, stderr = _run([executable, "-v"], runner)
    except PrimaryTextError:
        return "unknown"
    if returncode != 0:
        return "unknown"
    first = stderr.decode("utf-8", "replace").strip().splitlines()
    if not first:
        return "unknown"
    # e.g. "pdftotext version 26.01.0"
    return first[0].rsplit(" ", 1)[-1].strip() or "unknown"


def _warnings(stderr):
    if not stderr:
        return ()
    lines = [line.strip() for line in stderr.decode("utf-8", "replace").splitlines()]
    return tuple(line for line in lines if line)[:_MAX_WARNINGS]


def _run(argv, runner):
    run = runner if runner is not None else _default_runner
    try:
        return run(argv)
    except FileNotFoundError as exc:
        raise PrimaryTextError(
            "PDF text extractor could not be executed",
            kind=EXTRACTOR_UNAVAILABLE,
            detail={"extractor": argv[0]},
        ) from exc
    except OSError as exc:
        raise PrimaryTextError(
            "PDF text extractor failed to start",
            kind=EXTRACTION_FAILED,
            detail={"error_type": type(exc).__name__},
        ) from exc


def _default_runner(argv):
    completed = subprocess.run(
        list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return completed.returncode, completed.stdout, completed.stderr
